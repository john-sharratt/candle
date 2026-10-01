//! Front-end B — compile a tool catalog into a tool-call stencil tree.
//!
//! The catalog (the same JSON that feeds the prompt's tool list) becomes a tree
//! that guarantees: the tool name is one of the catalog's, the JSON parses,
//! every required parameter is present in order, optionals appear in declared
//! order in any subset, enum values are exactly the allowed strings, and no
//! leading/trailing comma is ever produced.
//!
//! Value handling by type:
//! - `string` — a free-text span (`Terminator::JsonStringValue`) that includes
//!   its own opening quote and ends at the unescaped closing one.
//! - `boolean` — a `true`/`false` branch.
//! - string `enum` — a branch over the allowed strings, each arm quoted.
//! - `object` with `properties` — written by the grammar: `{`, the keys and
//!   separators under the same required/optional rules as the arguments
//!   themselves, then `}`. The model decodes only the field values.
//! - `array` whose `items` open on a delimiter (strings, enums, objects with
//!   properties) — written by the grammar too: `[`, then per element a branch
//!   between another element (`, {`) and `]`, unrolled to
//!   [`MAX_ARRAY_ELEMENTS`]. An array inside such an array decodes free.
//! - `integer` — digit by digit under the mask, unrolled to
//!   [`MAX_INTEGER_DIGITS`], with `-` only where the schema's `minimum` allows
//!   a negative; the last arm of each digit carries what follows the value. A
//!   string, a `null` or a commit id cannot be decoded there.
//! - `number`, and any array or object the above does not cover (no schema,
//!   nullable, scalar elements) — emitted as any structurally-valid JSON value
//!   (`Terminator::JsonValue`), lookahead-terminated at the enclosing
//!   `,`/`}`/`]`. The session pushes that delimiter back to the next node when
//!   the next node continues with it, and drops it when it does not, so the
//!   grammar writes the structure the model got wrong. This guarantees valid
//!   JSON structure without strictly enforcing the scalar type.
//!
//! **A key stops where the model's own token begins.** A guided array or
//! object is keyed through its lead-in — ` [` and ` {` are the grammar's, since
//! it writes the container — and every other value is keyed up to `":` and no
//! further, so the model writes ` "`, ` ""`, ` [`, ` ["`, ` 5` or ` true` as the
//! one token it was trained on.
//!
//! A string is keyed only to its colon because which token opens it depends on
//! the value: Qwen spells an empty string ` ""` as one token and a non-empty one
//! ` "` then content, so a grammar that prefilled ` "` would choose "non-empty"
//! before the model chose anything — asked for an empty `prefix`, the model
//! writes `}}` and the string swallows the call's own close. A key ending in a
//! bare space fails the same way from the other side: grammar-written calls
//! came out `"commands":  [` with the space doubled, and on one Cline turn the
//! first token for the value was a space and a closer — `{"commands":  }}`, not
//! a call.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use candle_transformers::models::dialect::{CallStyle, Dialect};

use super::error::BuildError;
use super::spec::{NodeSpec, SpecId, TreeSpec};
use super::terminator::Terminator;
use super::tree::FreeTextLimits;

/// A parameter's value type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ParamType {
    String,
    Integer,
    Number,
    Boolean,
    Array,
    Object,
}

/// One tool parameter — or, nested, one field of an object value or the schema
/// every element of an array value follows.
#[derive(Debug, Clone, Deserialize)]
pub struct Param {
    /// The key. An array's element schema has none and leaves it empty.
    #[serde(default)]
    pub name: String,
    #[serde(rename = "type")]
    pub ty: ParamType,
    #[serde(default)]
    pub required: bool,
    /// When present, the value is constrained to one of these strings.
    #[serde(default, rename = "enum")]
    pub enum_values: Option<Vec<String>>,
    /// `array`: the schema every element follows. `None` leaves the array a
    /// free JSON value.
    #[serde(default)]
    pub items: Option<Box<Param>>,
    /// `array`: the schema's `minItems`. A guided array offers no `]` before
    /// this many elements, so an array that must not be empty cannot be.
    #[serde(default)]
    pub min_items: usize,
    /// `object`: its fields, ordered as a tool's own parameters are. `None`
    /// leaves the object a free JSON value; `Some` of an empty list is `{}`.
    /// `ParamType::Object` refined into a typed sub-object rather than left as
    /// any structurally-valid JSON — what an effector resource's
    /// schema-driven invoke body needs (§11) as much as a general tool schema.
    #[serde(default)]
    pub properties: Option<Vec<Param>>,
    /// `null` is also a value: the grammar offers it beside the typed value.
    #[serde(default)]
    pub nullable: bool,
    /// `integer`: the schema's `minimum`. At or above zero, the grammar offers
    /// no `-`; absent or negative, the value may be signed.
    #[serde(default)]
    pub minimum: Option<f64>,
    /// A required enum field: the fields of the same object each value
    /// requires beyond the object's own `required` list, as `(value, fields)`
    /// — the schema's `if`/`then` ([`value_requirements`]). The grammar writes
    /// them as required after that value, so a call cannot close without
    /// what its choice depends on.
    #[serde(default)]
    pub requires: Vec<(String, Vec<String>)>,
    /// A required enum field: the fields each value **brings with it**, as
    /// `(value, fields)`. After that value the object carries exactly those
    /// fields, required and typed as given, in place of any sibling of the same
    /// name — so what a value means decides what may be written beside it. An
    /// `invoke` call's `url` brings the `body` its endpoint accepts. Values
    /// with equal field lists share one sub-tree, so the tree grows with the
    /// number of distinct shapes and not the number of values.
    #[serde(default)]
    pub shapes: Vec<(String, Vec<Param>)>,
}

/// One tool: a name and an ordered parameter list.
#[derive(Debug, Clone, Deserialize)]
pub struct ToolSpec {
    pub name: String,
    #[serde(default)]
    pub params: Vec<Param>,
}

/// Parse a JSON array of tool descriptions (the flat `{name, params}` form).
pub fn parse_tools(json: &str) -> Result<Vec<ToolSpec>, BuildError> {
    serde_json::from_str(json).map_err(|e| BuildError::ToolSchema(e.to_string()))
}

impl ToolSpec {
    /// Build a [`ToolSpec`] from a tool name and a `schemars`-style JSON Schema
    /// (draft-07 `{type:object, properties:{…}, required:[…]}`).  Property
    /// `type` may be a string (`"integer"`) or a nullable array
    /// (`["integer","null"]`); the non-`null` member is used.  An `enum` of
    /// strings becomes a constrained branch.  Unknown/compound types fall back
    /// to "any JSON value" (still structurally validated).
    ///
    /// **An array's `items` and an object's `properties` are kept, recursively**,
    /// so the tree can write their brackets, keys and separators rather than
    /// leave the whole structure to the model.
    ///
    /// **A value that may be `null` says so** — a `null` in its `type` list or
    /// its `enum`, `nullable: true`, or an `anyOf`/`oneOf` of one schema and
    /// `{"type": "null"}` (how zod and pydantic write an optional value) — and
    /// the grammar offers `null` beside it rather than forcing the typed value.
    ///
    /// **Required parameters come in the order the `required` list names them,
    /// then optionals in the order the properties object declares them**, and
    /// the tree emits them in that order. The workspace builds `serde_json` with
    /// `preserve_order`, so the properties iterate as the schema's author wrote
    /// them — two optionals meant to be read together (a range's start before
    /// its end) stay offered in that order, where a sorted map would put the
    /// second one first and a call written in reading order could never reach
    /// it. A nested object's fields follow the same rule.
    pub fn from_json_schema(name: &str, schema: &Value) -> ToolSpec {
        ToolSpec {
            name: name.to_string(),
            params: fields_of(schema, schema),
        }
    }
}

/// How far a `$ref` chain is followed before the schema is treated as opaque.
/// A schema that refers to itself — a tree node whose children are the same
/// node — is legal JSON Schema and has no finite parameter tree, so the
/// recursion has to stop somewhere rather than hang the builder.
const MAX_REF_DEPTH: usize = 8;

/// A schema with its `$ref` and single-member `allOf` indirection followed.
///
/// `schemars` writes a named type as `{"$ref": "#/definitions/Name"}`, and a
/// named type that also carries a description as
/// `{"allOf": [{"$ref": …}], "description": …}`. Neither carries a `type`, so
/// without resolving them every named type in the catalog — a credential kind,
/// a revision, a push action — reaches [`parse_param_type`] as an unrecognized
/// schema and becomes "any JSON value", losing exactly the `enum` the grammar
/// exists to enforce.
fn resolve<'a>(schema: &'a Value, root: &'a Value, depth: usize) -> &'a Value {
    if depth >= MAX_REF_DEPTH {
        return schema;
    }
    // `allOf` of a single member is that member; siblings are metadata. A
    // conditional member (`if`/`then`) is a rule over the siblings, not the
    // schema itself ([`value_requirements`]).
    if let Some([only]) = schema
        .get("allOf")
        .and_then(Value::as_array)
        .map(|a| &a[..])
    {
        if only.get("if").is_none() {
            return resolve(only, root, depth + 1);
        }
    }
    let Some(pointer) = schema.get("$ref").and_then(Value::as_str) else {
        return schema;
    };
    // Only local pointers; a remote `$ref` is not fetched.
    let Some(rest) = pointer.strip_prefix("#/") else {
        return schema;
    };
    let mut at = root;
    for segment in rest.split('/') {
        // JSON Pointer escapes, in the order the spec requires.
        let key = segment.replace("~1", "/").replace("~0", "~");
        match at.get(&key) {
            Some(next) => at = next,
            None => return schema,
        }
    }
    resolve(at, root, depth + 1)
}

/// Merge a discriminated `oneOf` over objects into one object schema.
///
/// A tagged union — `{"kind":"branch","name":…}` or `{"kind":"commit","id":…}`
/// — has no representation in [`Param`], which carries one shape per value.
/// Leaving it unresolved costs the whole structure: the model gets "any JSON
/// value" and can write any key at all. Merging keeps most of it. The result
/// offers the union of every arm's fields, requires only what every arm
/// requires (the discriminator), and gives the discriminator the union of the
/// arms' constant values — so the tag is constrained to the tags that exist
/// and the keys to the keys that exist.
///
/// What it does not keep is the dependency between them: the grammar will let
/// `{"kind":"head","id":…}` through, a combination no arm allows. That is
/// caught one layer later, when the call deserializes into the tool's request
/// type and comes back `invalid_arguments` naming the field — a far better
/// outcome than an unconstrained object, where the model can invent key names
/// that match nothing at all.
fn merge_variants(arms: &[&Value], root: &Value, depth: usize) -> Option<Param> {
    let objects: Vec<&Value> = arms.iter().map(|a| resolve(a, root, depth)).collect();
    // Every arm must be an object schema with properties, or there is nothing
    // structural to merge and the value stays opaque.
    if !objects
        .iter()
        .all(|o| o.get("properties").is_some_and(Value::is_object))
    {
        return None;
    }
    let mut merged: Vec<Param> = Vec::new();
    for object in &objects {
        for field in fields_of(object, root) {
            match merged.iter_mut().find(|m| m.name == field.name) {
                None => merged.push(field),
                Some(existing) => {
                    // A field two arms share keeps the union of their allowed
                    // values, so a discriminator ends up listing every tag.
                    match (&mut existing.enum_values, field.enum_values) {
                        (Some(have), Some(more)) => {
                            for v in more {
                                if !have.contains(&v) {
                                    have.push(v);
                                }
                            }
                        }
                        // One arm leaves it open, so the merged field is open.
                        (slot, _) => *slot = None,
                    }
                    existing.nullable |= field.nullable;
                }
            }
        }
    }
    // Required only where every arm requires it.
    for field in &mut merged {
        field.required = objects.iter().all(|o| {
            o.get("required")
                .and_then(Value::as_array)
                .is_some_and(|r| r.iter().any(|v| v.as_str() == Some(field.name.as_str())))
        });
    }
    // The discriminator first, then the rest in first-seen order.
    merged.sort_by_key(|f| !f.required);
    Some(Param {
        name: String::new(),
        ty: ParamType::Object,
        required: false,
        enum_values: None,
        items: None,
        min_items: 0,
        properties: Some(merged),
        nullable: false,
        minimum: None,
        requires: Vec::new(),
        shapes: Vec::new(),
    })
}

/// An object schema's properties as parameters, required first in the
/// `required` list's order, then optionals in declared order.
fn fields_of(schema: &Value, root: &Value) -> Vec<Param> {
    let required: Vec<&str> = schema
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
        .unwrap_or_default();
    let mut params = Vec::new();
    if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
        // Iterating the object yields a deterministic field order.
        for (pname, pschema) in props {
            params.push(Param {
                name: pname.clone(),
                required: required.contains(&pname.as_str()),
                ..param_of(pschema, root)
            });
        }
    }
    // Stable, so optionals keep their property order behind the required.
    params.sort_by_key(|p| {
        required
            .iter()
            .position(|r| *r == p.name)
            .unwrap_or(usize::MAX)
    });
    for (field, value, fields) in value_requirements(schema) {
        if let Some(p) = params.iter_mut().find(|p| p.name == field) {
            p.requires.push((value, fields));
        }
    }
    params
}

/// What each value of a field requires, from an object schema's `allOf` of
/// `{"if": {"properties": {F: {"const": V}}}, "then": {"required": [...]}}`
/// — `enum` in place of `const` names several values at once — as
/// `(F, V, fields)`.
///
/// A request whose fields depend on a mode — `git_commit`'s `from: files`
/// needs `changes`, `from: patch` needs `patch` — can list none of them in
/// `required`, since no one of them is needed by every mode. Left optional,
/// the grammar offers the close beside each, and a model takes it: measured
/// live, `git_commit` was sent `{"from": "files", "message": …}` and then
/// `{"from": "files", "changes": []}` while the model's reasoning named both
/// files it meant to `take`. Each `if` that matches neither shape is skipped.
fn value_requirements(schema: &Value) -> Vec<(String, String, Vec<String>)> {
    let mut out = Vec::new();
    let Some(rules) = schema.get("allOf").and_then(Value::as_array) else {
        return out;
    };
    for rule in rules {
        let (Some(when), Some(then)) = (
            rule.pointer("/if/properties").and_then(Value::as_object),
            rule.pointer("/then/required").and_then(Value::as_array),
        ) else {
            continue;
        };
        // One field tested, or the rule is not about one field's value.
        let mut tests = when.iter();
        let (Some((field, test)), None) = (tests.next(), tests.next()) else {
            continue;
        };
        let values: Vec<&str> = match (test.get("const"), test.get("enum")) {
            (Some(one), _) => one.as_str().into_iter().collect(),
            (None, Some(Value::Array(many))) => many.iter().filter_map(Value::as_str).collect(),
            _ => continue,
        };
        let fields: Vec<String> = then
            .iter()
            .filter_map(|f| f.as_str().map(str::to_string))
            .collect();
        for value in values {
            out.push((field.clone(), value.to_string(), fields.clone()));
        }
    }
    out
}

/// One value schema as an unnamed, optional [`Param`].
fn param_of(schema: &Value, root: &Value) -> Param {
    let schema = resolve(schema, root, 0);
    // One schema or `null`: the schema, nullable.
    let alternatives = schema
        .get("anyOf")
        .or_else(|| schema.get("oneOf"))
        .and_then(Value::as_array);
    if let Some(alternatives) = alternatives {
        let is_null =
            |s: &&Value| resolve(s, root, 0).get("type").and_then(Value::as_str) == Some("null");
        let others: Vec<&Value> = alternatives.iter().filter(|s| !is_null(s)).collect();
        let nullable = others.len() < alternatives.len();
        if let [only] = others.as_slice() {
            let inner = param_of(only, root);
            return Param {
                nullable: inner.nullable || nullable,
                ..inner
            };
        }
        // Several real alternatives: a tagged union, if they are all objects.
        if let Some(merged) = merge_variants(&others, root, 0) {
            return Param { nullable, ..merged };
        }
    }
    let enum_members = schema.get("enum").and_then(Value::as_array);
    let enum_values = enum_members
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty());
    let (ty, type_nullable) = parse_param_type(schema);
    let nullable = type_nullable
        || enum_members.is_some_and(|a| a.iter().any(Value::is_null))
        || schema.get("nullable").and_then(Value::as_bool) == Some(true);
    // A minimum past the unrolled bound has no level to require it at, and the
    // grammar would write exactly the bound — a call the schema refuses. Such
    // an array is left a free value.
    let min_items = schema
        .get("minItems")
        .and_then(Value::as_u64)
        .map_or(0, |n| n as usize);
    let items = match (ty, schema.get("items")) {
        (ParamType::Array, Some(items @ Value::Object(_))) if min_items <= MAX_ARRAY_ELEMENTS => {
            Some(Box::new(param_of(items, root)))
        }
        _ => None,
    };
    let guided_min = if items.is_some() { min_items } else { 0 };
    let properties = match (ty, schema.get("properties")) {
        (ParamType::Object, Some(Value::Object(_))) => Some(fields_of(schema, root)),
        _ => None,
    };
    Param {
        name: String::new(),
        ty,
        required: false,
        enum_values,
        items,
        min_items: guided_min,
        properties,
        nullable,
        minimum: schema.get("minimum").and_then(Value::as_f64),
        requires: Vec::new(),
        shapes: Vec::new(),
    }
}

/// The schema's type, and whether `null` is also allowed.
fn parse_param_type(pschema: &Value) -> (ParamType, bool) {
    let (type_str, nullable) = match pschema.get("type") {
        Some(Value::String(s)) => (Some(s.as_str()), false),
        Some(Value::Array(arr)) => (
            arr.iter().filter_map(|v| v.as_str()).find(|s| *s != "null"),
            arr.iter().any(|v| v.as_str() == Some("null")),
        ),
        _ => (None, false),
    };
    let ty = match type_str {
        Some("string") => ParamType::String,
        Some("integer") => ParamType::Integer,
        Some("number") => ParamType::Number,
        Some("boolean") => ParamType::Boolean,
        Some("array") => ParamType::Array,
        // "object" or anything unrecognized → any structurally-valid JSON value.
        _ => ParamType::Object,
    };
    (ty, nullable)
}

/// The dialect-specific tool-call envelope strings.
///
/// # Two shapes, not one shape with different strings
///
/// This began as three strings around a JSON object, which is all the ChatML
/// families need. Qwen3.5 does not put JSON in a tool call at all — it writes a
/// nested element per argument, with unescaped values — so the difference is
/// structural and the builder has to branch. [`ToolCallEnvelope::style`] is what
/// it branches on, and every field below says which shapes it belongs to, so a
/// reader can tell at a glance which half of the file a string reaches.
#[derive(Debug, Clone)]
pub struct ToolCallEnvelope {
    /// Which shape the fields below describe. See [`CallStyle`].
    pub style: CallStyle,
    /// What ends the act's name, riding on each branch arm.
    ///
    /// On the arm rather than in the following static so a name that is a
    /// prefix of another (`read` / `read_back`) stays distinguishable in the
    /// trie — the arms differ at the terminator even when one name runs out
    /// first. `"` closes a JSON string; `>\n` closes `<function=…`.
    pub name_close: String,
    /// [`CallStyle::FunctionBlock`]: what opens one argument, before its name.
    pub param_open: String,
    /// [`CallStyle::FunctionBlock`]: what follows the argument's name, before
    /// its value.
    pub param_name_close: String,
    /// [`CallStyle::FunctionBlock`]: what ends one argument's value.
    ///
    /// **Consumed by the value span rather than emitted after it** — see
    /// [`Terminator::Until`]. A raw value has no delimiter of its own, so the
    /// marker is what ends it, and a static node emitting the marker as well
    /// would write it twice.
    ///
    /// `&'static str` rather than `String` because it becomes a
    /// [`Terminator`], which is `Copy` and therefore cannot own one. Every
    /// envelope here is built from literals, so nothing is lost.
    pub param_close: &'static str,
    /// Up to and including the opening quote of the name. Default:
    /// `"<tool_call>\n{\"name\": \""`.
    pub open: String,
    /// From the name's closing quote to the arguments object's `{`. Default:
    /// `"\", \"arguments\": {"`.
    pub args_open: String,
    /// Closes the arguments object, the outer object, and the call. Default:
    /// `"}}\n</tool_call>"`.
    pub close: String,
    /// The marker that *starts* a call — `"<tool_call>"`.
    ///
    /// Absent from [`Self::open`] because the model emits the marker itself and
    /// the tree resumes after it, so a single-call grammar never needs to name
    /// it. [`compile_tool_call_loop`] does: a second call inside one turn has no
    /// trigger to fire, so the grammar has to offer the marker as the arm that
    /// means "act again".
    pub marker: String,
    /// What the chat template writes between one call's `close` and the next
    /// call's `marker`, when a message carries several — `"\n"` for both Qwen
    /// shapes, whose templates emit a newline before every call after the
    /// first.
    ///
    /// [`compile_tool_call_loop`]'s "act again" arm is this followed by the
    /// marker. The marker glued straight onto the previous close would mask
    /// the one token a model trained on the template writes to continue, and
    /// offered only that or the turn terminator, the model ends the turn — a
    /// grammar permitting four calls would produce one.
    pub between_calls: String,
}

impl ToolCallEnvelope {
    /// The Qwen3 ChatML tool-call envelope.  `args_open` deliberately does NOT
    /// start with the name's closing `"` — that quote is appended to each name
    /// branch arm so a name that is a prefix of another (e.g. `ssh_session_exec`
    /// vs `ssh_session_exec_async`) is disambiguated by the quote.
    pub fn qwen3() -> Self {
        ToolCallEnvelope {
            style: CallStyle::JsonBlock,
            open: "<tool_call>\n{\"name\": \"".to_string(),
            args_open: ", \"arguments\": {".to_string(),
            close: "}}\n</tool_call>".to_string(),
            marker: "<tool_call>".to_string(),
            between_calls: "\n".to_string(),
            name_close: "\"".to_string(),
            param_open: String::new(),
            param_name_close: String::new(),
            param_close: "",
        }
    }

    /// The Qwen3.5 / Qwen3.8 envelope: a nested function element, one child per
    /// argument, values raw.
    ///
    /// ```text
    /// <tool_call>
    /// <function=say>
    /// <parameter=intent>
    /// what I mean
    /// </parameter>
    /// </function>
    /// </tool_call>
    /// ```
    ///
    /// `open` runs up to and including `<function=`, so the name branch's arms
    /// carry `name + ">\n"` — the closing `>` rides on the arm for the reason
    /// the quote does in [`Self::qwen3`]: a name that prefixes another
    /// (`read` / `read_back`) stays distinguishable in the trie.
    ///
    /// `args_open` is empty because nothing separates the name from the first
    /// argument, and there are no separators *between* arguments either — each
    /// is a self-delimiting element, which is what removes the whole
    /// `emitted_any` comma problem the JSON shape has to carry.
    pub fn qwen35() -> Self {
        ToolCallEnvelope {
            style: CallStyle::FunctionBlock,
            open: "<tool_call>\n<function=".to_string(),
            args_open: String::new(),
            // **The layout newlines belong to the grammar, the tags to the
            // model.** Each piece leads with the newline that separates it from
            // whatever came before, so the tree emits every line break in the
            // call and the model is never required to produce one.
            close: "\n</function>\n</tool_call>".to_string(),
            marker: "<tool_call>".to_string(),
            between_calls: "\n".to_string(),
            name_close: ">".to_string(),
            param_open: "\n<parameter=".to_string(),
            param_name_close: ">\n".to_string(),
            // **The tag alone, with no surrounding whitespace.**
            //
            // This was `"\n</parameter>\n"`, which made a line break part of the
            // delimiter the model had to produce. It does not reliably produce
            // one: taken from the persisted substrate, it wrote
            // `…to pass again.</parameter>` — tag correct, newline absent — so
            // the terminator never fired, the span stayed open, and every
            // element after it was swallowed into the value. The parser then
            // read a later call's arguments as this one's and reported the
            // missing ones, which is how an `ask` came to be missing `to` while
            // carrying `reflect`'s three arguments.
            //
            // A delimiter has to be something the model either wrote or did
            // not. Layout is not that.
            param_close: "</parameter>",
        }
    }

    /// One call, written out — what a worked example in a system prompt has to
    /// look like.
    ///
    /// **Here, beside the grammar, and nowhere else.** A prompt that teaches one
    /// syntax while the stencil forces another is not a prompt that is merely
    /// wrong: the grammar wins, so the model is held to a shape its instructions
    /// never described, and the instructions become noise it has to work around.
    /// That was live — the prompt documented one JSON object per line for a
    /// checkpoint whose grammar emitted `<tool_call>` blocks — and it survived
    /// because the only test in the area compared the prompt against the
    /// *parser*, which accepted both.
    ///
    /// Built from the same fields [`compile_tool_call_tree`] compiles, so the
    /// two cannot drift without a test noticing.
    pub fn render(&self, name: &str, args: &[(&str, &str)]) -> String {
        let mut s = String::with_capacity(96);
        s.push_str(&self.open);
        s.push_str(name);
        s.push_str(&self.name_close);
        s.push_str(&self.args_open);
        match self.style {
            CallStyle::FunctionBlock => {
                for (k, v) in args {
                    s.push_str(&self.param_open);
                    s.push_str(k);
                    s.push_str(&self.param_name_close);
                    s.push_str(v);
                    s.push_str(self.param_close);
                }
            }
            CallStyle::JsonBlock | CallStyle::Lines => {
                for (i, (k, v)) in args.iter().enumerate() {
                    if i > 0 {
                        s.push_str(", ");
                    }
                    // **A JSON scalar is written as one; everything else is a
                    // string.** `build_value` emits a `Number` unquoted and a
                    // `String` quoted, and this function's whole claim is that
                    // it cannot drift from the grammar — so quoting every value
                    // was a drift waiting to be noticed. It went unnoticed
                    // because the only callers were worked examples whose
                    // arguments are prose; the ingest chains prefill numeric
                    // line ranges, and `"start_line": "1"` teaches a shape the
                    // grammar would never emit.
                    //
                    // A quote inside a string value would end it early, so the
                    // string arm goes through `Value::String`, which escapes.
                    // The one hazard is a genuinely string-typed argument whose
                    // text is all digits — it renders unquoted. Nothing here
                    // has one (paths and prefixes carry `/` or `.`), and the
                    // typed grammar is what actually constrains the decode.
                    s.push_str(&format!("\"{k}\": {}", json_scalar(v)));
                }
            }
        }
        s.push_str(&self.close);
        s
    }

    /// The envelope a dialect calls for.
    ///
    /// **The one place a call style becomes a grammar.** Asked of the dialect
    /// rather than chosen by the caller, so a checkpoint's own template decides
    /// the shape its decode is constrained to — which is the property that was
    /// missing when the prompt described one syntax and the grammar forced
    /// another.
    pub fn for_dialect(d: &Dialect) -> Self {
        match d.call_style {
            CallStyle::FunctionBlock => Self::qwen35(),
            // `Lines` has no envelope of its own. It gets the JSON block's,
            // which is what this engine has always compiled — the markers are
            // prefilled into the assistant turn and stripped again by the
            // parser, so the shape on the wire is the same either way.
            CallStyle::JsonBlock | CallStyle::Lines => Self::qwen3(),
        }
    }

    /// [`Self::for_dialect`] adapted for the **assistant turn**, where one call
    /// IS the whole reply: the marker the model has already emitted comes off
    /// the front, and the turn terminator goes on the close so the turn ends
    /// with the call.
    ///
    /// **The close must END on that terminator, with nothing after it.** A
    /// static run's last token is held back from the forward and rides the next
    /// decode step, which commits it; a token in any earlier slot goes through
    /// `push_forwarded`, whose flag is dropped. ChatML's `assistant_end` is
    /// `"<|im_end|>\n"`, so appending it verbatim buries the EOS one slot from
    /// the end where nothing sees it — the decode does not stop and the model
    /// writes its own next turn into this one. Measured from the persisted
    /// substrate: one assistant turn holding three `<|im_start|>`/`<|im_end|>`
    /// pairs — the real call, the turn end, then a hallucinated `user` header
    /// with a second think block and a second call.
    ///
    /// Hence `trim_end`, and hence this being a constructor rather than two
    /// lines at the call site: the rule is a property of the envelope, and
    /// `the_assistant_close_ends_on_the_turn_terminator` holds it for every
    /// dialect. `stencil::think` carries the identical rule for the reasoning
    /// block and learned it the same way.
    pub fn for_assistant_turn(d: &Dialect) -> Self {
        let base = Self::for_dialect(d);
        Self {
            open: base
                .open
                .strip_prefix(&base.marker)
                .unwrap_or(&base.open)
                .to_string(),
            close: format!("{}{}", base.close, d.assistant_end.trim_end()),
            ..base
        }
    }

    /// [`Self::for_assistant_turn`]'s sibling for a turn that may make **several**
    /// calls ([`compile_tool_call_loop`]): the marker comes off the front for the
    /// same reason, and the turn terminator stays **off** the close.
    ///
    /// That is the whole difference, and it is structural rather than stylistic.
    /// A single-call tree ends the turn unconditionally, so its close may carry
    /// the terminator. A loop must decide *after* each call whether to open
    /// another or stop, so the terminator is the loop's other arm — baked into
    /// `close` it would end the turn before that choice exists, and the tree
    /// would be the single-call tree with extra nodes.
    ///
    /// [`Self::turn_close`] is the terminator to pass alongside it, so the two
    /// halves of the split come from one place.
    pub fn for_assistant_calls(d: &Dialect) -> Self {
        let base = Self::for_dialect(d);
        Self {
            open: base
                .open
                .strip_prefix(&base.marker)
                .unwrap_or(&base.open)
                .to_string(),
            ..base
        }
    }

    /// The assistant-turn terminator, trimmed to end exactly ON the EOS — the
    /// `close_turn` argument [`compile_tool_call_loop`] finishes a turn with.
    /// Trimmed for the reason [`Self::for_assistant_turn`] records: a dialect's
    /// `assistant_end` carries a trailing newline, and an EOS one slot from the
    /// end is an EOS nothing sees.
    pub fn turn_close(d: &Dialect) -> String {
        d.assistant_end.trim_end().to_string()
    }
}

/// One argument value as JSON: a number or boolean verbatim, anything else as a
/// quoted, escaped string. Mirrors [`ToolTreeBuilder::build_value`]'s split
/// between `ParamType::Number`/`Boolean` and `ParamType::String`.
fn json_scalar(v: &str) -> String {
    if v == "true" || v == "false" || v.parse::<f64>().is_ok() {
        v.to_string()
    } else {
        Value::String(v.to_string()).to_string()
    }
}

/// The tool-call stencil tree's label. The scheduler keys tool-call-specific
/// behavior off the active stencil's tree label (repetition-penalty
/// suppression, the in-call reprojection freeze at first-token promotion), so
/// the label is a shared constant rather than a string literal in each place.
pub const TOOL_CALL_TREE_LABEL: &str = "tool_call";

/// How many tool calls one assistant turn may make.
///
/// **A turn that can only call once pays a full round-trip per call.** Reading
/// four files it already knows it wants costs four reasoning blocks, four
/// prefills of a growing context, four reprojections and four belief scans — and
/// measured on a codebase tour, fifteen such rounds were the bulk of the wall
/// clock while the calls themselves ran in milliseconds. Batching what the model
/// already knows it needs collapses those round-trips into one.
///
/// Five covers the common fan-out — list a directory, then read the handful of
/// files it names — without letting one turn commit to a long speculative run
/// whose later calls are all chosen before any result has come back. The
/// ceiling is a grammar bound, not a target: a turn making one call remains
/// perfectly ordinary, because the loop's other arm is always the turn
/// terminator.
pub const MAX_TOOL_CALLS_PER_TURN: usize = 5;

/// Compile a tool catalog into a [`TreeSpec`].  Errors on an empty catalog or a
/// name/enum collision the trie rejects.
pub fn compile_tool_call_tree(
    tools: &[ToolSpec],
    env: &ToolCallEnvelope,
) -> Result<TreeSpec, BuildError> {
    if tools.is_empty() {
        return Err(BuildError::ToolSchema("empty tool catalog".into()));
    }
    let mut b = ToolTreeBuilder::new(env);
    let end = b.spec.push(NodeSpec::End);

    // Each tool: name arm -> args_open static -> its argument object -> close.
    let mut arms: Vec<(String, SpecId)> = Vec::with_capacity(tools.len());
    for tool in tools {
        let args_entry = b.build_fields(&tool.params, &env.close, end)?;
        let arm_target = b.spec.push(NodeSpec::Static {
            text: env.args_open.clone(),
            next: args_entry,
        });
        // The arm carries the name's own terminator so prefix-related names
        // stay distinguishable in the trie — see `ToolCallEnvelope::name_close`.
        arms.push((format!("{}{}", tool.name, env.name_close), arm_target));
    }
    let name_branch = b.spec.push(NodeSpec::Branch { arms });
    let open = b.spec.push(NodeSpec::Static {
        text: env.open.clone(),
        next: name_branch,
    });
    b.spec.root = open;
    // Failsafe: if a token ever escapes the mask, close the JSON + the tool-call
    // block so the partial output is at least terminated for the extractor.
    b.spec.bail = env.close.clone();
    Ok(b.spec)
}

/// Compile a catalog into a tree that admits **up to `max_calls` calls in one
/// turn**, then ends.
///
/// [`compile_tool_call_tree`] is the assistant shape: one call *is* the whole
/// reply, so its close carries the turn's EOS and the turn is over. An **action
/// loop** is a different thing. A character turning as it speaks or moving as it
/// signals is doing two things in one moment, and a grammar that admits exactly
/// one act cannot express that — it would silently make every turn a single act
/// and nothing would report the amputation.
///
/// So after each call the grammar offers a choice, and the model makes it:
///
/// ```text
///   …}}</tool_call>  ─┬─ "<tool_call>{"name": "  → another act (levels remain)
///                     └─ close_turn              → done
/// ```
///
/// `max_calls` is a **bound, not a target**: every level offers the closing arm,
/// so a character that has said what it means stops at one. The bound exists
/// because the alternative is an unbounded loop the model can sit in — the same
/// runaway the reasoning block had, wearing a different hat.
///
/// The levels are unrolled rather than cycled: a cycle would make "how many so
/// far" a property the tree cannot see, and the count is the whole point.
///
/// `env.close` must **not** carry the turn terminator here — `close_turn` does,
/// on the finishing arm only. An `env.close` ending in EOS would end the turn on
/// the first call and the loop would be unreachable.
/// Copy `sub`'s nodes into `host`, redirecting its `End` to `tail`, and return
/// its root's id in `host`.
///
/// **One tree, not two joined by a trigger.** A trigger fires on a *decoded*
/// token, so a marker the grammar injects as static text does not necessarily
/// fire one — which would leave the model free at exactly the join the grammar
/// exists to close. Splicing makes the transition a node edge instead: the tree
/// walks from the end of one part into the start of the next with no moment in
/// between where control returns to the decoder.
fn splice(host: &mut TreeSpec, sub: &TreeSpec, tail: SpecId) -> SpecId {
    let base = host.nodes.len();
    // `End` becomes `tail`; every other id shifts by the host's current length.
    let remap = |id: SpecId| -> SpecId {
        match sub.nodes.get(id.0) {
            Some(NodeSpec::End) => tail,
            _ => SpecId(id.0 + base),
        }
    };
    for node in &sub.nodes {
        let moved = match node {
            // Kept as a node so ids stay dense and `remap` above stays a pure
            // index shift; nothing ever reaches it, because every edge that
            // pointed at it now points at `tail`.
            NodeSpec::End => NodeSpec::End,
            NodeSpec::Static { text, next } => NodeSpec::Static {
                text: text.clone(),
                next: remap(*next),
            },
            NodeSpec::Branch { arms } => NodeSpec::Branch {
                arms: arms.iter().map(|(t, n)| (t.clone(), remap(*n))).collect(),
            },
            NodeSpec::FreeText {
                term,
                eos_ends,
                limits,
                close_token,
                suppress_close,
                next,
            } => NodeSpec::FreeText {
                term: *term,
                eos_ends: *eos_ends,
                limits: *limits,
                close_token: *close_token,
                suppress_close: *suppress_close,
                next: remap(*next),
            },
        };
        host.nodes.push(moved);
    }
    remap(sub.root)
}

pub fn compile_tool_call_loop(
    tools: &[ToolSpec],
    env: &ToolCallEnvelope,
    max_calls: usize,
    close_turn: &str,
) -> Result<TreeSpec, BuildError> {
    compile_action_loop(tools, env, max_calls, close_turn, None)
}

/// The action loop, optionally preceded by a reasoning block.
///
/// **The whole turn in one grammar.** With `think` set the tree is: the block's
/// steered spans, its closing tag, then straight into the first call — no moment
/// between them where the decoder is free. Without it the tree starts at the
/// call.
///
/// This is what a mission-specific thinking dial needs. Some work is a thinking
/// problem — drafting a story into an eleven-year silence — and some is not, and
/// the same character must be able to do both. What must not vary is that the
/// turn ends in acts: a character that thinks and then writes prose has done
/// nothing, and the world cannot tell that from a character that chose to wait.
pub fn compile_action_loop(
    tools: &[ToolSpec],
    env: &ToolCallEnvelope,
    max_calls: usize,
    close_turn: &str,
    think: Option<&TreeSpec>,
) -> Result<TreeSpec, BuildError> {
    if tools.is_empty() {
        return Err(BuildError::ToolSchema("empty tool catalog".into()));
    }
    if max_calls == 0 {
        return Err(BuildError::ToolSchema(
            "a turn that may make no calls is a turn that cannot act".into(),
        ));
    }
    let mut b = ToolTreeBuilder::new(env);
    let end = b.spec.push(NodeSpec::End);
    // The arm that finishes the turn: emit the terminator and stop.
    let finish = b.spec.push(NodeSpec::Static {
        text: close_turn.to_string(),
        next: end,
    });

    // **Built back to front.** Each level's continuation points at the level
    // above it, so the deepest is constructed first and the shallowest ends up
    // as the root. Forwards it would need a node id before the node exists.
    let mut after_call = finish;
    let mut root_open = None;
    for level in (0..max_calls).rev() {
        // One call: choose a name, fill its arguments, close the block.
        let mut arms: Vec<(String, SpecId)> = Vec::with_capacity(tools.len());
        for tool in tools {
            let args_entry = b.build_fields(&tool.params, &env.close, after_call)?;
            let arm_target = b.spec.push(NodeSpec::Static {
                text: env.args_open.clone(),
                next: args_entry,
            });
            // The name's own terminator, from the envelope — a hardcoded `"`
            // here spliced a JSON quote into a function block and produced act
            // names like `reflect"<parameter=inner_thoughts"`, rejected on
            // every turn. See `ToolCallEnvelope::name_close`.
            arms.push((format!("{}{}", tool.name, env.name_close), arm_target));
        }
        let name_branch = b.spec.push(NodeSpec::Branch { arms });
        let open = b.spec.push(NodeSpec::Static {
            text: env.open.clone(),
            next: name_branch,
        });

        // What the *previous* level's close leads to: go again, or finish. The
        // continuation arm carries what the template writes between two calls
        // and then the marker — exactly the text the model would have emitted
        // to start another call — so choosing it is choosing to act again.
        after_call = b.spec.push(NodeSpec::Branch {
            arms: vec![
                (format!("{}{}", env.between_calls, env.marker), open),
                (close_turn.to_string(), end),
            ],
        });
        if level == 0 {
            root_open = Some(open);
        }
    }

    let first_call = root_open.expect("max_calls > 0 builds at least one level");
    // With a reasoning block, the turn starts inside it and flows into the first
    // call as a node edge — see [`splice`] for why that is not a trigger.
    b.spec.root = match think {
        Some(prelude) => splice(&mut b.spec, prelude, first_call),
        None => first_call,
    };
    b.spec.bail = format!("{}{}", env.close, close_turn);
    Ok(b.spec)
}

/// How many elements a guided array admits. The grammar is unrolled one level
/// per element — the compiler rejects cycles, and a cycle would hide the count
/// the bound needs — so the last level offers only the close.
pub const MAX_ARRAY_ELEMENTS: usize = 64;

/// How many digits an integer value may have — enough for any `i64`
/// (`9223372036854775807` is nineteen). Unrolled one level per digit, as an
/// array is per element, for the same reason: the compiler rejects cycles.
pub const MAX_INTEGER_DIGITS: usize = 19;

struct ToolTreeBuilder<'a> {
    spec: TreeSpec,
    env: &'a ToolCallEnvelope,
    /// How many guided arrays enclose the value being built. An array inside
    /// one decodes free: unrolled, it would repeat per enclosing element and
    /// the tree would grow as the product of the bounds.
    array_depth: u32,
}

/// One object's gate and value memo, so the gate graph stays linear instead of
/// exploding over subsets — and never leaks between objects, which have
/// different optional lists.
#[derive(Default)]
struct FieldMemo {
    /// `(optional index, emitted_any) -> gate entry`.
    gates: HashMap<(usize, bool), SpecId>,
    /// `optional index -> (lead-in, value entry)`. Every gate that offers an
    /// optional leads its value to the same successor gate, so the value is
    /// the same sub-tree whichever gate it was reached from.
    values: HashMap<usize, (String, SpecId)>,
}

/// Whether the grammar can begin a value of this schema — offer every way it
/// can start as an arm. Only such elements can be guided inside an array: the
/// arms carry the separator and the opening together (`, {`, `, true`), and an
/// element the model opens itself (a number) has nothing for an arm to hold.
fn guided_element(p: &Param) -> bool {
    p.enum_values.is_some()
        || matches!(p.ty, ParamType::String | ParamType::Boolean)
        || (p.ty == ParamType::Object && p.properties.is_some())
}

impl<'a> ToolTreeBuilder<'a> {
    fn new(env: &'a ToolCallEnvelope) -> Self {
        ToolTreeBuilder {
            spec: TreeSpec::new(TOOL_CALL_TREE_LABEL),
            env,
            array_depth: 0,
        }
    }

    /// An object's field sequence, written up to and including `close` — the
    /// envelope close for a call's arguments, `}` for an object value — and
    /// ending at `end`.  Returns the entry node.
    fn build_fields(
        &mut self,
        params: &[Param],
        close: &str,
        end: SpecId,
    ) -> Result<SpecId, BuildError> {
        let required: Vec<&Param> = params.iter().filter(|p| p.required).collect();
        let optional: Vec<&Param> = params.iter().filter(|p| !p.required).collect();
        self.build_sequence(&required, &optional, true, close, end)
    }

    /// `required` in order, then the gates over `optional`, through `close` to
    /// `end`. `first` says no field precedes them.
    fn build_sequence(
        &mut self,
        required: &[&Param],
        optional: &[&Param],
        first: bool,
        close: &str,
        end: SpecId,
    ) -> Result<SpecId, BuildError> {
        let Some((&p, rest)) = required.split_first() else {
            // Optional gates start with emitted_any = (a required field precedes them).
            let mut memo = FieldMemo::default();
            return self.opt_gates(optional, 0, !first, close, end, &mut memo);
        };
        let decides = !p.requires.is_empty() || !p.shapes.is_empty();
        if let (true, Some(values), false) = (decides, &p.enum_values, p.nullable) {
            return self.build_discriminated(p, values, rest, optional, first, close, end);
        }
        let tail = self.build_sequence(rest, optional, false, close, end)?;
        let (leadin, value) = self.build_value(p, tail)?;
        Ok(self.spec.push(NodeSpec::Static {
            text: self.key(p, first, &leadin),
            next: value,
        }))
    }

    /// A required enum field `p` whose values decide the rest of the object
    /// ([`Param::requires`], [`Param::shapes`]): each value goes on to the rest
    /// of the object with the fields it requires moved from `optional` to
    /// required, and the fields it brings added, after `rest`.
    ///
    /// **One tail per distinct continuation.** A value's tail is fixed by the
    /// fields it promotes and the fields it brings, so values that agree on
    /// both point at the same sub-tree: a thousand endpoints that accept the
    /// same body cost that body once.
    #[allow(clippy::too_many_arguments)]
    fn build_discriminated(
        &mut self,
        p: &Param,
        values: &[String],
        rest: &[&Param],
        optional: &[&Param],
        first: bool,
        close: &str,
        end: SpecId,
    ) -> Result<SpecId, BuildError> {
        let mut tails: HashMap<String, SpecId> = HashMap::new();
        let mut arms: Vec<(String, SpecId)> = Vec::with_capacity(values.len());
        for value in values {
            // Every rule naming the value counts.
            let needs: Vec<&String> = p
                .requires
                .iter()
                .filter(|(v, _)| v == value)
                .flat_map(|(_, fields)| fields)
                .collect();
            let brought: Vec<Param> = p
                .shapes
                .iter()
                .filter(|(v, _)| v == value)
                .flat_map(|(_, fields)| fields)
                .map(|f| Param {
                    required: true,
                    ..f.clone()
                })
                .collect();
            let brings = |o: &&Param| brought.iter().any(|b| b.name == o.name);
            let wanted = |o: &&Param| needs.contains(&&o.name);
            let promoted: Vec<Param> = optional
                .iter()
                .copied()
                .filter(|o| wanted(o) && !brings(o))
                .map(|o| Param {
                    required: true,
                    ..o.clone()
                })
                .collect();
            let key = format!("{promoted:?}{brought:?}");
            let tail = match tails.get(&key) {
                Some(&tail) => tail,
                None => {
                    let mut now_required: Vec<&Param> =
                        rest.iter().copied().filter(|r| !brings(r)).collect();
                    now_required.extend(promoted.iter());
                    now_required.extend(brought.iter());
                    let still_optional: Vec<&Param> = optional
                        .iter()
                        .copied()
                        .filter(|o| !wanted(o) && !brings(o))
                        .collect();
                    let tail =
                        self.build_sequence(&now_required, &still_optional, false, close, end)?;
                    tails.insert(key, tail);
                    tail
                }
            };
            arms.push((value.clone(), tail));
        }
        let branch = match self.env.style {
            // The end marker rides on each arm, as in [`Self::build_value`].
            CallStyle::FunctionBlock => {
                let marker = self.env.param_close;
                let arms = arms
                    .into_iter()
                    .map(|(v, next)| (format!("{v}{marker}"), next))
                    .collect();
                self.spec.push(NodeSpec::Branch { arms })
            }
            // Quoted, each carrying the first step of its own tail — so the
            // value and the key it requires next are one choice — and its
            // space, the key ending at the colon.
            CallStyle::JsonBlock | CallStyle::Lines => {
                let quoted = arms
                    .into_iter()
                    .map(|(v, next)| (Value::String(v).to_string(), next))
                    .collect();
                let arms = self
                    .with_continuations(quoted)
                    .into_iter()
                    .map(|(t, next)| (format!(" {t}"), next))
                    .collect();
                self.spec.push(NodeSpec::Branch { arms })
            }
        };
        Ok(self.spec.push(NodeSpec::Static {
            text: self.key(p, first, ""),
            next: branch,
        }))
    }

    /// What opens one argument, up to the point its value begins.
    ///
    /// `first` says whether a separator is needed before it — which only the
    /// JSON shapes have. A function block's arguments are self-delimiting
    /// elements, so there is nothing between them and the flag is ignored.
    fn key(&self, p: &Param, first: bool, leadin: &str) -> String {
        match self.env.style {
            CallStyle::FunctionBlock => format!(
                "{}{}{}",
                self.env.param_open, p.name, self.env.param_name_close
            ),
            // Through the lead-in when there is one, else up to the colon: the
            // space before a value is the first byte of the model's own token
            // (` [`, ` 5`, ` true`), and a key ending on it leaves the model
            // mid-token. See the module docs.
            CallStyle::JsonBlock | CallStyle::Lines => {
                let sep = if first { "" } else { ", " };
                match leadin {
                    "" => format!("{sep}\"{}\":", p.name),
                    _ => format!("{sep}\"{}\": {leadin}", p.name),
                }
            }
        }
    }

    /// The gate over optionals `idx..`, given whether a field was already emitted.
    fn opt_gates(
        &mut self,
        opts: &[&Param],
        idx: usize,
        emitted_any: bool,
        close: &str,
        end: SpecId,
        memo: &mut FieldMemo,
    ) -> Result<SpecId, BuildError> {
        if let Some(&id) = memo.gates.get(&(idx, emitted_any)) {
            return Ok(id);
        }
        // No more optionals: emit the close and finish.
        if idx == opts.len() {
            let id = self.spec.push(NodeSpec::Static {
                text: close.to_string(),
                next: end,
            });
            memo.gates.insert((idx, emitted_any), id);
            return Ok(id);
        }
        let mut arms: Vec<(String, SpecId)> = Vec::with_capacity(opts.len() - idx + 1);
        for (j, p) in opts.iter().enumerate().skip(idx) {
            let (leadin, value) = match memo.values.get(&j) {
                Some(built) => built.clone(),
                None => {
                    // Include optional j: a field is emitted, so everything
                    // after has emitted_any = true.
                    let after = self.opt_gates(opts, j + 1, true, close, end, memo)?;
                    let built = self.build_value(p, after)?;
                    memo.values.insert(j, built.clone());
                    built
                }
            };
            arms.push((self.key(p, !emitted_any, &leadin), value));
        }
        // The "stop" arm: close the object.
        arms.push((close.to_string(), end));
        let id = self.spec.push(NodeSpec::Branch { arms });
        memo.gates.insert((idx, emitted_any), id);
        Ok(id)
    }

    /// A value sub-tree for `p`, transitioning to `next` after the value.
    /// Returns a `lead-in` string that must be appended to the preceding key/arm
    /// static, plus the value's entry node.  Folding the lead-in (a string/enum
    /// value's opening `"`) into the key keeps structural merges like ` "`
    /// internal to one static, rather than leaving a lone `"` after a branch arm
    /// that merges backward into the committed arm (an unrepresentable retract).
    fn build_value(&mut self, p: &Param, next: SpecId) -> Result<(String, SpecId), BuildError> {
        // A nested JSON object refined into typed fields (`object | recurse`).
        // Function blocks keep the raw free value below. The opening brace
        // rides on the lead-in, same as `value_arms`'s object arm. A
        // *nullable* object still needs the null-vs-object choice `value_arms`
        // builds below — forcing the object here would deny the model `null`.
        if self.env.style != CallStyle::FunctionBlock && !p.nullable {
            if let Some(fields) = &p.properties {
                let entry = self.build_fields(fields, "}", next)?;
                return Ok(("{".to_string(), entry));
            }
        }
        // **A function block's values are raw**, so the closing delimiter is the
        // element's own end marker rather than a quote, and nothing is escaped.
        // Handled before the JSON cases because both the enum branch and the
        // free span differ, not just one of them.
        if self.env.style == CallStyle::FunctionBlock {
            let close = self.env.param_close;
            if let Some(values) = &p.enum_values {
                // The end marker rides on each arm for the reason the quote
                // does below: a value that prefixes another stays
                // distinguishable in the trie.
                let branch = self.spec.push(NodeSpec::Branch {
                    arms: values
                        .iter()
                        .map(|v| (format!("{v}{close}"), next))
                        .collect(),
                });
                return Ok((String::new(), branch));
            }
            // Every type is raw text here — there is no JSON to be structurally
            // valid against, and the act that receives the call parses its own
            // scalars exactly as it does on the other styles.
            let span = self.spec.push(NodeSpec::FreeText {
                term: Terminator::Until {
                    marker: self.env.param_close,
                },
                eos_ends: false,
                limits: FreeTextLimits::json_string(),
                close_token: None,
                suppress_close: false,
                next,
            });
            return Ok((String::new(), span));
        }
        // **A plain string's opening quote is the model's.** The key stops at
        // the colon, because which token opens a string depends on the value:
        // Qwen writes an empty string as the single token ` ""` and a non-empty
        // one as ` "` then content, so a prefilled ` "` has chosen "non-empty"
        // before the model has chosen anything. See `Terminator::JsonStringValue`.
        //
        // A nullable string cannot be a choice between ` "` and ` null` for the
        // same reason — the ` "` arm is that prefill — so it is a free JSON
        // value instead, where ` ""`, ` "src"` and ` null` are each the model's
        // own tokens and a skipped value is written `null`. That is the common
        // case, not an edge: an `Option<String>` argument such as `file_list`'s
        // `prefix` is nullable in its schema. An array element's quote stays
        // the separator arm's (`, "`) — see [`Self::value_arms`].
        if p.ty == ParamType::String && p.enum_values.is_none() {
            if p.nullable {
                return Ok((String::new(), self.free_value(next)));
            }
            let span = self.spec.push(NodeSpec::FreeText {
                term: Terminator::JsonStringValue,
                eos_ends: false,
                limits: FreeTextLimits::json_string(),
                close_token: None,
                suppress_close: false,
                next,
            });
            return Ok((String::new(), span));
        }
        // An enum field's arms carry the first step of what follows the field
        // — the next key, or the close — so the value's closing quote and the
        // separator after it are one arm's text. See [`Self::continuations`].
        let arms = self.value_arms(p, next)?.map(|arms| match p.enum_values {
            Some(_) => self.with_continuations(arms),
            None => arms,
        });
        match arms {
            // One way to begin — an object's `{`, an array's `[`, a one-value
            // enum: it is the lead-in, folded into the key (`"filter": {`).
            Some(mut arms) if arms.len() == 1 => Ok(arms.remove(0)),
            // A choice — `true`/`false`, an enum, a value or `null`. The key
            // ends at the colon, so each arm carries its space.
            Some(arms) => Ok((
                String::new(),
                self.spec.push(NodeSpec::Branch {
                    arms: arms
                        .into_iter()
                        .map(|(t, n)| (format!(" {t}"), n))
                        .collect(),
                }),
            )),
            None => Ok((String::new(), self.free_value(next))),
        }
    }

    /// Every way a value of `p` can begin, as arm text with no leading space
    /// and where each leads — or `None` when the grammar cannot begin it and
    /// the model writes the value whole ([`Self::free_value`]).
    ///
    /// **A closed set is a choice, never free text.** An enum, `true`/`false`
    /// and `null` are each a complete value the grammar can write, so the model
    /// chooses among them under the mask: it cannot spell `True`, and a
    /// nullable field can actually be `null`. A string, a guided object and a
    /// guided array begin with their opening byte, and `null` beside it when
    /// the schema allows it.
    fn value_arms(
        &mut self,
        p: &Param,
        next: SpecId,
    ) -> Result<Option<Vec<(String, SpecId)>>, BuildError> {
        let mut arms: Vec<(String, SpecId)> = match (&p.enum_values, p.ty) {
            // The closing quote rides on each arm, so a value that prefixes
            // another stays distinguishable in the trie.
            (Some(values), _) => values
                .iter()
                .map(|v| (Value::String(v.clone()).to_string(), next))
                .collect(),
            (None, ParamType::Boolean) => vec![("true".into(), next), ("false".into(), next)],
            (None, ParamType::String) => {
                let span = self.spec.push(NodeSpec::FreeText {
                    term: Terminator::JsonString,
                    eos_ends: false,
                    limits: FreeTextLimits::json_string(),
                    close_token: None,
                    suppress_close: false,
                    next,
                });
                vec![("\"".into(), span)]
            }
            // **An object with a schema is written by the grammar**: its brace,
            // its keys, its separators — the model decodes only the values.
            (None, ParamType::Object) => match &p.properties {
                Some(fields) => vec![("{".into(), self.build_fields(fields, "}", next)?)],
                None => return Ok(None),
            },
            // **So is an array whose elements the grammar can begin** — see
            // [`Self::build_array`]. Anything else stays a free value.
            (None, ParamType::Array) => match p.items.as_deref() {
                Some(item) if self.array_depth == 0 && guided_element(item) => {
                    self.array_depth += 1;
                    let built = self.build_array(item, p.min_items, next);
                    self.array_depth -= 1;
                    vec![("[".into(), built?)]
                }
                _ => return Ok(None),
            },
            (None, ParamType::Integer) => match self.integer_arms(p, next) {
                Some(arms) => arms,
                None => return Ok(None),
            },
            (None, ParamType::Number) => return Ok(None),
        };
        if p.nullable {
            arms.push(("null".into(), next));
        }
        Ok(Some(arms))
    }

    /// **An integer is written digit by digit under the mask**, so nothing but
    /// an integer can be decoded there.
    ///
    /// ```text
    ///   ─┬─ "0" ──────────────── <end>
    ///    ├─ "1".."9" ─ D₁₈ ─┬─ "0".."9" ─ D₁₇ ─ … ─ D₀ ─ <end>
    ///    └─ "-" ─ "1".."9" ─┘  └─ <end>
    /// ```
    ///
    /// `<end>` is what follows the value — the next key, the close — taken as
    /// each arm's text ([`Self::continuations`]), so the digit the model does
    /// not write and the separator it does are one choice. `-` is offered only
    /// when the schema's `minimum` allows a negative, and a leading `0` ends the
    /// number, as JSON requires.
    ///
    /// It was a free JSON value, which guarantees structure and not type: live,
    /// a `git_show` call decoded `"page": "7eb876659c1b…"` — a commit id in an
    /// integer field — and was refused a layer later. Under the mask the `"`
    /// that opens it is never a candidate.
    ///
    /// `None` when what follows the value cannot be written as arm text — its
    /// first step is free text, not a key or a close — and the value stays a
    /// free JSON value.
    fn integer_arms(&mut self, p: &Param, next: SpecId) -> Option<Vec<(String, SpecId)>> {
        const DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
        let ends = self.continuations(next);
        if ends.iter().any(|(text, _)| text.is_empty()) {
            return None;
        }
        let finished = self.choice(ends.clone());
        // Built back to front: after the last admitted digit only the end
        // remains, and each level before it offers one more digit or the end.
        let mut more = finished;
        for _ in 1..MAX_INTEGER_DIGITS {
            let mut arms: Vec<(String, SpecId)> =
                DIGITS.iter().map(|d| (d.to_string(), more)).collect();
            arms.extend(ends.iter().cloned());
            more = self.spec.push(NodeSpec::Branch { arms });
        }
        let leading: Vec<(String, SpecId)> =
            DIGITS[1..].iter().map(|d| (d.to_string(), more)).collect();
        let mut first = vec![("0".to_string(), finished)];
        first.extend(leading.iter().cloned());
        if p.minimum.is_none_or(|m| m < 0.0) {
            let negative = self.spec.push(NodeSpec::Branch { arms: leading });
            first.push(("-".to_string(), negative));
        }
        Some(first)
    }

    /// A node offering `arms`: a static when there is only one way on, which is
    /// no choice at all.
    fn choice(&mut self, mut arms: Vec<(String, SpecId)>) -> SpecId {
        match arms.len() {
            1 => {
                let (text, next) = arms.remove(0);
                self.spec.push(NodeSpec::Static { text, next })
            }
            _ => self.spec.push(NodeSpec::Branch { arms }),
        }
    }

    /// The first step of `next`, as `(text, successor)` arms an enum value's
    /// arm can carry on its end: a static's text, each arm of a branch, or —
    /// for anything the grammar does not write itself — nothing.
    ///
    /// **Why an enum value takes its successor's first step with it.** Its
    /// closing quote is the grammar's, and the next thing written is a
    /// separator or a close the model chooses (`, "path":` against `}`). A
    /// tokenizer writes those two together — Qwen's `",` and `"}` are single
    /// tokens — so a quote committed on its own leaves the model to choose from
    /// a boundary its encoding never produces, holding a lone `,` it is never
    /// trained to write after `"`. Measured live: a `file_list` whose `repo` is
    /// an enum closed its arguments after the repository three times running,
    /// the model's reasoning each time saying it meant to pass `path`. With
    /// the separator on the arm, the value and what follows it are one choice
    /// the model writes in its own tokens.
    ///
    /// Object fields only. An array element's successor is the next element's
    /// own separator branch, so folding there compounds across every element
    /// of the array; a field's successor is a key or the close, one step deep.
    fn continuations(&self, next: SpecId) -> Vec<(String, SpecId)> {
        match self.spec.node(next) {
            NodeSpec::Static { text, next } => vec![(text.clone(), *next)],
            NodeSpec::Branch { arms } => arms.clone(),
            NodeSpec::FreeText { .. } | NodeSpec::End => vec![(String::new(), next)],
        }
    }

    /// `arms` with each one's successor's first step appended
    /// ([`Self::continuations`]).
    fn with_continuations(&self, arms: Vec<(String, SpecId)>) -> Vec<(String, SpecId)> {
        arms.into_iter()
            .flat_map(|(text, next)| {
                self.continuations(next)
                    .into_iter()
                    .map(move |(tail, after)| (format!("{text}{tail}"), after))
            })
            .collect()
    }

    /// Any structurally-valid JSON value, lookahead-terminated at the enclosing
    /// `,`, `}` or `]` (the session pushes that delimiter back, or drops it when
    /// the successor does not continue with it).  This guarantees valid JSON
    /// structure; it does not strictly enforce the scalar type. The value's
    /// leading space is the model's — see [`Self::key`].
    fn free_value(&mut self, next: SpecId) -> SpecId {
        self.spec.push(NodeSpec::FreeText {
            term: Terminator::JsonValue,
            eos_ends: false,
            limits: FreeTextLimits::json_value(),
            close_token: None,
            suppress_close: false,
            next,
        })
    }

    /// The elements of an array value, after its `[` (the caller's lead-in),
    /// through its `]`, ending at `next`.
    ///
    /// ```text
    ///   [ ─┬─ <open> element₁ ─┬─ ", <open>" element₂ ─ … ─ element₆₄ ─ "]"
    ///      └─ "]"              └─ "]"
    /// ```
    ///
    /// `<open>` is every way the element can begin ([`Self::value_arms`]): one
    /// arm for a string or object, one per value for a closed set — an array
    /// of booleans offers `true`, `false` and `]` at each element.
    ///
    /// **The grammar writes every bracket and separator**, so the model cannot
    /// close the array with the wrong bracket or leave an element open. Live,
    /// with the array a free value, a Cline `read_files` call came out
    /// `{"path": "…", "start_line": 3380, "end_line": 3420]}}}` — the element's
    /// `}` never written, a `]` in its place, not JSON, so no call. A free span
    /// counts one depth for `[` and `{` alike and could not see it.
    ///
    /// **The first `min_items` levels offer no `]`**, so the schema's
    /// `minItems` holds under the mask. Live, `git_push` — whose `pushes`
    /// needs at least one — was decoded `"pushes": []` three turns running
    /// while the model's reasoning named the branch it meant to push: offered
    /// the close beside the first element, it took the close.
    ///
    /// Unrolled to [`MAX_ARRAY_ELEMENTS`] and built back to front, as
    /// [`compile_action_loop`] builds its levels: each level's continuation
    /// points at the next, so the deepest exists first.
    fn build_array(
        &mut self,
        item: &Param,
        min_items: usize,
        next: SpecId,
    ) -> Result<SpecId, BuildError> {
        // After the last admitted element, only the close remains.
        let mut after = self.spec.push(NodeSpec::Static {
            text: "]".to_string(),
            next,
        });
        for level in (0..MAX_ARRAY_ELEMENTS).rev() {
            let opens = self
                .value_arms(item, after)?
                .expect("`build_array` is reached only for a guided element");
            let sep = if level == 0 { "" } else { ", " };
            let mut arms: Vec<(String, SpecId)> = opens
                .into_iter()
                .map(|(open, element)| (format!("{sep}{open}"), element))
                .collect();
            if level >= min_items {
                arms.push(("]".to_string(), next));
            }
            after = self.choice(arms);
        }
        Ok(after)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::stencil::compile::compile;
    use crate::stencil::vocab::TestVocab;

    /// `render` writes the shape the grammar compiles — including the scalar
    /// split the grammar makes.
    ///
    /// The JSON arm used to quote every value, so a numeric argument rendered
    /// `"page": "1"` while `build_value`'s `Number` arm emits `1`. Only
    /// prose-valued callers existed, so nothing caught it until an ingest chain
    /// needed to prefill a page number.
    #[test]
    fn render_writes_numbers_bare_and_strings_quoted() {
        let json = ToolCallEnvelope::qwen3();
        let out = json.render("file_read", &[("path", "a/mod.rs"), ("page", "1")]);
        assert!(
            out.contains(r#""path": "a/mod.rs""#),
            "a string argument stays quoted and escaped: {out}"
        );
        assert!(
            out.contains(r#""page": 1"#),
            "a numeric argument is bare, as `build_value` emits it: {out}"
        );

        // A function block's values are raw whatever their type, so neither
        // gains quotes and the scalar question does not arise.
        let fb = ToolCallEnvelope::qwen35();
        let out = fb.render("file_read", &[("path", "a/mod.rs"), ("page", "1")]);
        assert!(
            out.contains("<parameter=path>\na/mod.rs</parameter>"),
            "{out}"
        );
        assert!(out.contains("<parameter=page>\n1</parameter>"), "{out}");
        assert!(!out.contains('"'), "raw values carry no quotes: {out}");
    }

    /// A quote inside a string value must not end the string early.
    #[test]
    fn render_escapes_a_quote_in_a_string_value() {
        let json = ToolCallEnvelope::qwen3();
        let out = json.render("say", &[("intent", r#"that "it" is mine"#)]);
        assert!(
            out.contains(r#""intent": "that \"it\" is mine""#),
            "the quote is escaped rather than terminating: {out}"
        );
    }

    /// **The assistant close must END on the turn terminator.** The rule and the
    /// measured consequence are on
    /// [`ToolCallEnvelope::for_assistant_turn`]; this is the guard.
    ///
    /// It exists because the regression that motivated it replaced a correct
    /// literal (`"}}\n</tool_call><|im_end|>"`) with `assistant_end` appended
    /// verbatim — right in intent, since the shape belongs to the dialect and
    /// not to a literal, but ChatML's `assistant_end` trails a newline. Nothing
    /// in the area asserted the terminator's POSITION, so every Qwen tool call
    /// ran past its own turn end until the substrate was read by hand.
    ///
    /// Asserted over every dialect, not just the broken one: the three that
    /// happen not to trail whitespace today are one edit away from doing so.
    #[test]
    fn the_assistant_close_ends_on_the_turn_terminator() {
        for d in [
            Dialect::chat_ml(),
            Dialect::qwen35(),
            Dialect::llama2(),
            Dialect::llama3(),
            Dialect::deepseek(),
        ] {
            let env = ToolCallEnvelope::for_assistant_turn(&d);
            let term = d.assistant_end.trim_end();
            assert!(
                !term.is_empty(),
                "{:?}: no assistant terminator to end on",
                d.dialect_type
            );
            assert!(
                env.close.ends_with(term),
                "{:?}: close {:?} must end on {:?} — a terminator in any earlier \
                 slot has its flag dropped and the decode never stops",
                d.dialect_type,
                env.close,
                term
            );
            // The marker the model already emitted must not be re-emitted.
            assert!(
                !env.open.starts_with(&env.marker),
                "{:?}: open {:?} still leads with the marker {:?}",
                d.dialect_type,
                env.open,
                env.marker
            );
        }
    }

    fn catalog() -> Vec<ToolSpec> {
        parse_tools(
            r#"[
              {"name":"read_file","params":[{"name":"path","type":"string","required":true}]},
              {"name":"write_file","params":[
                  {"name":"path","type":"string","required":true},
                  {"name":"append","type":"boolean","required":false}
              ]},
              {"name":"set_mode","params":[
                  {"name":"mode","type":"string","required":true,
                   "enum":["read","write","exec"]}
              ]}
            ]"#,
        )
        .unwrap()
    }

    #[test]
    fn parses_catalog() {
        let c = catalog();
        assert_eq!(c.len(), 3);
        assert_eq!(c[2].params[0].enum_values.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn compiles_to_a_valid_tree() {
        let tree = compile(
            &compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen3()).unwrap(),
            &TestVocab::new(),
        )
        .unwrap();
        assert!(tree.len() > 5);
        assert_eq!(tree.label(), "tool_call");
    }

    // ── the function-block shape ────────────────────────────────────────────

    /// Every static string the tree can emit, in id order — enough to see what
    /// shape a decode is being held to without walking edges.
    fn statics(spec: &TreeSpec) -> Vec<String> {
        spec.nodes
            .iter()
            .filter_map(|n| match n {
                NodeSpec::Static { text, .. } => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    /// Every branch arm the tree offers.
    fn arms(spec: &TreeSpec) -> Vec<String> {
        spec.nodes
            .iter()
            .flat_map(|n| match n {
                NodeSpec::Branch { arms } => arms.iter().map(|(t, _)| t.clone()).collect(),
                _ => Vec::new(),
            })
            .collect()
    }

    /// **The grammar emits Qwen3.5's own syntax**, not JSON wearing its
    /// markers. Nothing checked this before: the prompt described one shape,
    /// the grammar forced another, and the only test in the area compared the
    /// prompt against the *parser*.
    #[test]
    fn the_function_block_tree_emits_elements_not_json() {
        let spec = compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen35()).unwrap();
        let st = statics(&spec);
        let all = st.join("");

        assert!(all.contains("<tool_call>\n<function="), "{st:?}");
        assert!(all.contains("<parameter=path>\n"), "{st:?}");
        assert!(all.contains("</function>\n</tool_call>"), "{st:?}");

        // None of the JSON scaffolding survives — no quoted keys, no argument
        // object, no comma separators between arguments.
        assert!(!all.contains("\"arguments\""), "{st:?}");
        assert!(!all.contains("\"path\""), "{st:?}");
        assert!(!all.contains(", "), "a JSON separator leaked in: {st:?}");
    }

    /// The act name's closing `>` rides on each arm, so a name that prefixes
    /// another stays distinguishable in the trie — the same rule the JSON
    /// shape's closing quote follows.
    #[test]
    fn a_function_name_arm_carries_its_own_terminator() {
        let spec = compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen35()).unwrap();
        let a = arms(&spec);
        assert!(a.iter().any(|s| s == "read_file>"), "{a:?}");
        assert!(a.iter().any(|s| s == "write_file>"), "{a:?}");
    }

    /// An enumerated value is raw text with the element's end marker on it —
    /// no quotes, because there is no JSON string to close.
    #[test]
    fn an_enum_value_is_unquoted_and_carries_the_end_marker() {
        let spec = compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen35()).unwrap();
        let a = arms(&spec);
        assert!(a.iter().any(|s| s == "read</parameter>"), "{a:?}");
        assert!(
            !a.iter().any(|s| s.contains("\"")),
            "an enum arm was quoted: {a:?}"
        );
    }

    /// A free value ends at the element marker rather than at a quote, which
    /// is what lets a character's prose hold quotes and newlines untouched.
    ///
    /// The marker is the **bare tag**, with no layout newlines on either side.
    /// A marker of `"\n</parameter>\n"` only fires when the model happens to put
    /// the tag on its own line, and it does not always: measured live, an `ask`
    /// ended a value `...pass again.</parameter>` with the tag hard against the
    /// prose. The span stayed open, swallowed the two elements that followed —
    /// so the `ask` appeared to carry `reflect`'s arguments — and the injected
    /// close then wrote a second tag on top. The tag alone is the model's
    /// close; the newlines around it belong to the grammar.
    #[test]
    fn a_free_value_is_terminated_by_the_element_and_not_by_a_quote() {
        let spec = compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen35()).unwrap();
        let terms: Vec<Terminator> = spec
            .nodes
            .iter()
            .filter_map(|n| match n {
                NodeSpec::FreeText { term, .. } => Some(*term),
                _ => None,
            })
            .collect();
        assert!(!terms.is_empty(), "no free span at all");
        assert!(
            terms.iter().all(|t| matches!(
                t,
                Terminator::Until {
                    marker: "</parameter>"
                }
            )),
            "{terms:?}"
        );
    }

    /// It still compiles against a real vocabulary — the shape being right is
    /// not the same as the trie accepting it.
    #[test]
    fn the_function_block_tree_compiles() {
        let tree = compile(
            &compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen35()).unwrap(),
            &TestVocab::new(),
        )
        .unwrap();
        assert!(tree.len() > 5);
        assert_eq!(tree.label(), "tool_call");
    }

    /// **The action loop is the tree npcd actually compiles**, and it builds its
    /// own name arms.
    ///
    /// It had a hardcoded `"` where the single-call tree had one too — fixing
    /// only the latter left the loop splicing a JSON quote into a function
    /// block, so every live act arrived named `reflect"<parameter=inner_thoughts"`
    /// and was rejected as an unknown tool. Twenty-two of twenty-two ticks.
    /// The single-call tree is not what runs; this is.
    #[test]
    fn the_action_loop_writes_the_same_shape_as_a_single_call() {
        let spec = compile_action_loop(
            &catalog(),
            &ToolCallEnvelope::qwen35(),
            2,
            "<|im_end|>",
            None,
        )
        .unwrap();
        let a = arms(&spec);
        assert!(
            a.iter().any(|s| s == "read_file>"),
            "the loop's name arms are not the envelope's: {a:?}"
        );
        assert!(
            !a.iter().any(|s| s.starts_with("read_file\"")),
            "a JSON quote leaked into a function block: {a:?}"
        );
        // And it still compiles for both styles at the bound npcd uses.
        for env in [ToolCallEnvelope::qwen3(), ToolCallEnvelope::qwen35()] {
            let spec = compile_action_loop(&catalog(), &env, 2, "<|im_end|>", None).unwrap();
            compile(&spec, &TestVocab::new())
                .unwrap_or_else(|e| panic!("{:?} will not compile: {e}", env.style));
        }
    }

    /// The JSON families are untouched — the whole point of the style axis is
    /// that adding one changes nothing for the others.
    #[test]
    fn the_json_tree_is_exactly_what_it_always_was() {
        let spec = compile_tool_call_tree(&catalog(), &ToolCallEnvelope::qwen3()).unwrap();
        let all = statics(&spec).join("");
        assert!(all.contains("<tool_call>\n{\"name\": \""));
        assert!(all.contains(", \"arguments\": {"));
        assert!(all.contains("}}\n</tool_call>"));
        assert!(!all.contains("<parameter="), "{all}");
    }

    /// The envelope the action loop uses: `close` carries no turn terminator,
    /// because the finishing arm does.
    fn loop_env() -> ToolCallEnvelope {
        ToolCallEnvelope {
            close: "}}\n</tool_call>".to_string(),
            ..ToolCallEnvelope::qwen3()
        }
    }

    /// **The bound binds, and it is a bound rather than a target.**
    ///
    /// Every level offers the closing arm, so a character that has said what it
    /// means stops at one call. What the bound removes is the unbounded loop a
    /// model can sit in — the same runaway the reasoning block had.
    #[test]
    fn the_loop_admits_up_to_its_bound_and_can_always_stop() {
        for max in 1..=4 {
            let spec = compile_tool_call_loop(&catalog(), &loop_env(), max, "<|im_end|>").unwrap();

            // One name branch per level. Counted by an arm naming a tool rather
            // than by arm *count* — an enum parameter also compiles to a branch
            // with as many arms as the catalog has tools, which is a coincidence
            // that made the first version of this test pass for the wrong reason.
            let name_branches = spec
                .nodes
                .iter()
                .filter(|n| {
                    matches!(n, NodeSpec::Branch { arms }
                        if arms.iter().any(|(t, _)| t == "read_file\""))
                })
                .count();
            assert_eq!(name_branches, max, "expected {max} levels");

            // And a continuation branch after every call, each of which offers
            // the closing arm — so stopping is always reachable.
            let continuations: Vec<&NodeSpec> = spec
                .nodes
                .iter()
                .filter(|n| {
                    matches!(n, NodeSpec::Branch { arms }
                        if arms.iter().any(|(t, _)| t == "<|im_end|>"))
                })
                .collect();
            assert_eq!(continuations.len(), max);
            for c in continuations {
                let NodeSpec::Branch { arms } = c else {
                    unreachable!()
                };
                assert!(
                    arms.iter().any(|(text, _)| text == "<|im_end|>"),
                    "a level with no way to stop is an unbounded loop"
                );
            }

            compile(&spec, &TestVocab::new()).expect("the loop must compile");
        }
    }

    /// **Thinking flows into acting as a node edge, with no free join.**
    ///
    /// The reason it is one tree rather than two composed by a trigger: a
    /// trigger fires on a *decoded* token, so a marker the grammar injects as
    /// static text may not fire one — and the model would be free at exactly the
    /// join the grammar exists to close. Spliced, the block's closing tag walks
    /// straight into the first call.
    #[test]
    fn a_thinking_turn_walks_from_the_block_into_the_calls() {
        use crate::stencil::think::{compile_think_tree, ThinkMode, ThinkSteerEnvelope};

        let env = ThinkSteerEnvelope {
            think_open: 1,
            think_close: 2,
            eos: 3,
            // This test asserts the SPLICE closes the join, so the injected
            // marker must stay out of it.
            after_close: "",
        };
        let prelude = compile_think_tree(ThinkMode::Balanced, &env);
        let spec =
            compile_action_loop(&catalog(), &loop_env(), 2, "<|im_end|>", Some(&prelude)).unwrap();

        // The spliced prelude is present…
        assert!(
            spec.nodes
                .iter()
                .any(|n| matches!(n, NodeSpec::FreeText { .. })),
            "the reasoning span did not survive the splice"
        );
        // …and nothing still points at an `End` that would have handed control
        // back between the block and the calls. Every edge out of the prelude
        // leads onward.
        let ends: Vec<usize> = spec
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| matches!(n, NodeSpec::End))
            .map(|(i, _)| i)
            .collect();
        assert_eq!(ends.len(), 2, "one live End, one spliced-over remnant");

        // The turn still starts inside the block, not at a call.
        assert!(
            !matches!(&spec.nodes[spec.root.0], NodeSpec::Static { text, .. } if text.contains("name")),
            "a thinking turn must begin in the block"
        );
        compile(&spec, &TestVocab::new()).expect("the spliced tree must compile");
    }

    /// A turn that may make no calls is a turn that cannot act, which for an
    /// action loop is not a configuration — it is a mistake with no symptom.
    #[test]
    fn a_loop_of_zero_calls_is_refused() {
        assert!(compile_tool_call_loop(&catalog(), &loop_env(), 0, "<|im_end|>").is_err());
        assert!(compile_tool_call_loop(&[], &loop_env(), 3, "<|im_end|>").is_err());
    }

    #[test]
    fn empty_catalog_errors() {
        assert!(matches!(
            compile_tool_call_tree(&[], &ToolCallEnvelope::qwen3()),
            Err(BuildError::ToolSchema(_))
        ));
    }

    #[test]
    fn optional_only_tool_heals_brace_quote_merge() {
        // Reproduces the real-tokenizer failure: a `datetime`-style tool with no
        // required params and one optional string.  Its args object opens `{`
        // immediately followed by the optional-gate branch (`"timezone": …` or
        // the close `}…`).  With a tokenizer that merges `{"` and `{}` into one
        // token (as Qwen3 does), the `{`→branch boundary must HEAL rather than
        // error.
        let v = TestVocab::new()
            .with_special("{\"", 300)
            .with_special("{}", 301);
        let tools = parse_tools(
            r#"[{"name":"datetime","params":[
                  {"name":"timezone","type":"string","required":false}]}]"#,
        )
        .unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
        // Must compile (no BoundaryMerge) and be walkable.
        let tree = compile(&spec, &v).unwrap();
        assert!(tree.len() > 3);
    }

    #[test]
    fn gate_after_string_value_compiles_with_quote_comma_merge() {
        // A tool with a required string field and an optional field: the optional
        // gate sits *after* the string value.  With a tokenizer that merges
        // `",` (closing quote + comma) into one token, the old in-context
        // lowering tried to retract the committed opening quote and errored.  The
        // clean-boundary lowering of free-text successors fixes it.
        let v = TestVocab::new().with_special("\",", 300);
        let tools = parse_tools(
            r#"[{"name":"write_file","params":[
                  {"name":"path","type":"string","required":true},
                  {"name":"create","type":"boolean","required":false}]}]"#,
        )
        .unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
        let tree = compile(&spec, &v).unwrap();
        assert!(tree.len() > 5);
    }

    #[test]
    fn number_field_gate_compiles_with_digit_delimiter_merge() {
        // A number (lookahead) field with an optional → a gate follows the value.
        // A tokenizer that merges the value's last digit with the gate's
        // delimiter (`0,` / `0}`) must not break compilation: the gate is lowered
        // from a fresh boundary, not in the value's `…0` context.
        let v = TestVocab::new()
            .with_special("0,", 300)
            .with_special("0}", 301);
        let tools = parse_tools(
            r#"[{"name":"wait","params":[
                  {"name":"secs","type":"integer","required":true},
                  {"name":"unit","type":"string","required":false}]}]"#,
        )
        .unwrap();
        let spec = compile_tool_call_tree(&tools, &ToolCallEnvelope::qwen3()).unwrap();
        let tree = compile(&spec, &v).unwrap();
        assert!(tree.len() > 5);
    }

    /// An array's element schema and an object's fields survive parsing,
    /// recursively and in the same field order a tool's parameters take — a
    /// nullable container included, which is guided with `null` beside it.
    #[test]
    fn items_and_properties_are_kept_recursively() {
        let schema: Value = serde_json::from_str(
            r#"{
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "end_line": {"type": ["number", "null"]},
                                "path": {"type": "string"}
                            },
                            "required": ["path"]
                        }
                    },
                    "maybe": {"type": ["array", "null"], "items": {"type": "string"}},
                    "loose": {"type": "object"},
                    "any_items": {"type": "array", "items": true}
                },
                "required": ["files"]
            }"#,
        )
        .unwrap();
        let spec = ToolSpec::from_json_schema("read_files", &schema);
        let by_name = |n: &str| spec.params.iter().find(|p| p.name == n).unwrap();

        let item = by_name("files")
            .items
            .as_deref()
            .expect("files keeps its items");
        assert_eq!(item.ty, ParamType::Object);
        assert!(item.name.is_empty());
        let fields = item
            .properties
            .as_ref()
            .expect("the element keeps its fields");
        let names: Vec<(&str, bool)> = fields
            .iter()
            .map(|p| (p.name.as_str(), p.required))
            .collect();
        assert_eq!(names, [("path", true), ("end_line", false)]);
        assert_eq!(fields[1].ty, ParamType::Number);
        assert!(fields[1].nullable);
        assert!(!fields[0].nullable);

        let maybe = by_name("maybe");
        assert_eq!(maybe.ty, ParamType::Array);
        assert!(maybe.nullable);
        assert!(maybe.items.is_some(), "a nullable array keeps its items");
        assert!(by_name("loose").properties.is_none(), "no properties, free");
        assert!(
            by_name("any_items").items.is_none(),
            "`items: true` is free"
        );
    }

    /// Every way a schema can say "or null" is read as nullable, and the type
    /// it qualifies is kept.
    #[test]
    fn every_spelling_of_nullable_is_read() {
        let param = |schema: Value| param_of(&schema, &schema);
        for (schema, ty) in [
            (json!({"type": ["boolean", "null"]}), ParamType::Boolean),
            (json!({"type": ["null", "string"]}), ParamType::String),
            (
                json!({"type": "integer", "nullable": true}),
                ParamType::Integer,
            ),
            (
                json!({"anyOf": [{"type": "boolean"}, {"type": "null"}]}),
                ParamType::Boolean,
            ),
            (
                json!({"oneOf": [{"type": "null"}, {"type": "array", "items": {"type": "string"}}]}),
                ParamType::Array,
            ),
            (json!({"enum": ["a", "b", null]}), ParamType::Object),
        ] {
            let p = param(schema.clone());
            assert!(p.nullable, "{schema}");
            assert_eq!(p.ty, ty, "{schema}");
        }
        let e = param(json!({"enum": ["a", null]}));
        assert_eq!(e.enum_values.as_deref(), Some(&["a".to_string()][..]));
        for schema in [
            json!({"type": "boolean"}),
            json!({"anyOf": [{"type": "string"}, {"type": "integer"}]}),
            json!({"enum": ["a"]}),
        ] {
            assert!(!param(schema.clone()).nullable, "{schema}");
        }
        // `anyOf` of one schema and null keeps that schema's structure.
        let object = param(json!({"anyOf": [
            {"type": "object", "properties": {"k": {"type": "string"}}, "required": ["k"]},
            {"type": "null"}
        ]}));
        assert!(object.nullable);
        assert_eq!(object.properties.map(|f| f.len()), Some(1));
    }

    /// **A named type is followed to its definition.** `schemars` writes one as
    /// a bare `$ref`, or as an `allOf` of a single `$ref` when it also carries
    /// a description. Both must reach the definition, or the `enum` the
    /// grammar exists to enforce is lost and the value becomes any JSON at all.
    #[test]
    fn a_named_type_is_resolved_to_its_definition() {
        let schema = json!({
            "type": "object",
            "definitions": {
                "CredType": {"type": "string", "enum": ["ssh_key", "http_bearer"]}
            },
            "properties": {
                "bare": {"$ref": "#/definitions/CredType"},
                "described": {
                    "allOf": [{"$ref": "#/definitions/CredType"}],
                    "description": "Credential type."
                },
                "optional": {
                    "anyOf": [{"$ref": "#/definitions/CredType"}, {"type": "null"}]
                }
            },
            "required": ["bare", "described"]
        });
        let spec = ToolSpec::from_json_schema("credential_save", &schema);
        let by_name = |n: &str| spec.params.iter().find(|p| p.name == n).unwrap();
        for name in ["bare", "described", "optional"] {
            let p = by_name(name);
            assert_eq!(p.ty, ParamType::String, "{name}");
            assert_eq!(
                p.enum_values.as_deref(),
                Some(&["ssh_key".to_string(), "http_bearer".to_string()][..]),
                "{name} lost its enum",
            );
        }
        assert!(by_name("optional").nullable);
        assert!(!by_name("bare").nullable);
    }

    /// A `$ref` that leads nowhere, or in a circle, leaves the value opaque
    /// rather than hanging the builder or panicking.
    #[test]
    fn an_unresolvable_or_circular_ref_is_left_opaque() {
        let dangling = json!({
            "type": "object",
            "properties": {"x": {"$ref": "#/definitions/Missing"}}
        });
        assert_eq!(
            ToolSpec::from_json_schema("t", &dangling).params[0].ty,
            ParamType::Object
        );

        let circular = json!({
            "type": "object",
            "definitions": {"Loop": {"$ref": "#/definitions/Loop"}},
            "properties": {"x": {"$ref": "#/definitions/Loop"}}
        });
        assert_eq!(
            ToolSpec::from_json_schema("t", &circular).params[0].ty,
            ParamType::Object
        );

        // A remote pointer is not fetched.
        let remote = json!({
            "type": "object",
            "properties": {"x": {"$ref": "https://example.com/s.json#/A"}}
        });
        assert_eq!(
            ToolSpec::from_json_schema("t", &remote).params[0].ty,
            ParamType::Object
        );
    }

    /// **A tagged union is merged into one object whose discriminator lists
    /// every tag.** Without this a revision argument reaches the grammar as
    /// "any JSON value" and the model may write any key at all; with it the
    /// tag is one of the five that exist and the keys are the keys that exist.
    ///
    /// The merge is deliberately permissive about *combinations* — `kind` and
    /// every arm's field are offered together — because `Param` carries one
    /// shape per value. A combination no arm allows is refused when the call
    /// deserializes, which names the offending field.
    #[test]
    fn a_tagged_union_merges_into_one_object_listing_every_tag() {
        let schema = json!({
            "type": "object",
            "definitions": {
                "RevArg": {
                    "oneOf": [
                        {"type": "object", "properties": {
                            "kind": {"type": "string", "enum": ["head"]}},
                         "required": ["kind"]},
                        {"type": "object", "properties": {
                            "kind": {"type": "string", "enum": ["commit"]},
                            "id": {"type": "string"}},
                         "required": ["kind", "id"]},
                        {"type": "object", "properties": {
                            "kind": {"type": "string", "enum": ["branch"]},
                            "name": {"type": "string"}},
                         "required": ["kind", "name"]}
                    ]
                }
            },
            "properties": {
                "rev": {"anyOf": [{"$ref": "#/definitions/RevArg"}, {"type": "null"}]}
            }
        });
        let spec = ToolSpec::from_json_schema("git_log", &schema);
        let rev = &spec.params[0];
        assert_eq!(rev.ty, ParamType::Object);
        assert!(rev.nullable, "the field itself is optional");
        let fields = rev.properties.as_ref().expect("the union keeps its fields");

        let kind = fields.iter().find(|f| f.name == "kind").unwrap();
        assert_eq!(
            kind.enum_values.as_deref(),
            Some(
                &[
                    "head".to_string(),
                    "commit".to_string(),
                    "branch".to_string()
                ][..]
            ),
            "the discriminator must list every arm's tag",
        );
        assert!(kind.required, "every arm requires the discriminator");

        // The arms' own fields are offered, and none is required, because no
        // single one is required by every arm.
        let names: Vec<&str> = fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(names, ["kind", "id", "name"], "discriminator first");
        for f in fields.iter().filter(|f| f.name != "kind") {
            assert!(!f.required, "{} is required by only one arm", f.name);
            assert_eq!(f.ty, ParamType::String);
        }
    }

    /// A union whose arms are not all objects stays opaque — there is no
    /// single shape to merge them into.
    #[test]
    fn a_union_of_unlike_arms_stays_opaque() {
        let schema = json!({
            "type": "object",
            "properties": {
                "x": {"oneOf": [
                    {"type": "object", "properties": {"a": {"type": "string"}}},
                    {"type": "string"}
                ]}
            }
        });
        let x = &ToolSpec::from_json_schema("t", &schema).params[0];
        assert_eq!(x.ty, ParamType::Object);
        assert!(x.properties.is_none(), "no single shape, so no fields");
    }

    /// A union nested inside an array's elements resolves too — `git_push`
    /// takes an array of tagged push items.
    #[test]
    fn a_union_inside_an_array_resolves() {
        let schema = json!({
            "type": "object",
            "definitions": {
                "PushItem": {
                    "oneOf": [
                        {"type": "object", "properties": {
                            "action": {"type": "string", "enum": ["update_branch"]},
                            "branch": {"type": "string"}},
                         "required": ["action", "branch"]},
                        {"type": "object", "properties": {
                            "action": {"type": "string", "enum": ["delete_tag"]},
                            "tag": {"type": "string"}},
                         "required": ["action", "tag"]}
                    ]
                }
            },
            "properties": {
                "pushes": {"type": "array", "items": {"$ref": "#/definitions/PushItem"}}
            },
            "required": ["pushes"]
        });
        let spec = ToolSpec::from_json_schema("git_push", &schema);
        let element = spec.params[0]
            .items
            .as_ref()
            .expect("elements keep a schema");
        let fields = element
            .properties
            .as_ref()
            .expect("the element is an object");
        let action = fields.iter().find(|f| f.name == "action").unwrap();
        assert_eq!(
            action.enum_values.as_deref(),
            Some(&["update_branch".to_string(), "delete_tag".to_string()][..])
        );
        assert!(fields.iter().any(|f| f.name == "branch"));
        assert!(fields.iter().any(|f| f.name == "tag"));
    }

    /// **An enum field's arm spells the separator after it, so the model writes
    /// the quote and the separator as its own token.** `repo` is an optional
    /// enum ahead of the optional `path`: after the value, the arms go on to
    /// the next key or to the close, never stopping at a lone closing quote —
    /// the boundary that left a live model unable to write `",` and closing its
    /// arguments instead of naming the path it meant to.
    #[test]
    fn an_enum_field_arm_carries_the_separator_after_it() {
        let schema = json!({
            "type": "object",
            "properties": {
                "repo": {"type": "string", "enum": ["a", "b"]},
                "path": {"type": "string"}
            }
        });
        let spec = compile_tool_call_tree(
            &[ToolSpec::from_json_schema("t", &schema)],
            &ToolCallEnvelope::qwen3(),
        )
        .unwrap();
        let all = arms(&spec);
        for want in [
            " \"a\", \"path\":",
            " \"b\", \"path\":",
            " \"a\"}}\n</tool_call>",
            " \"b\"}}\n</tool_call>",
        ] {
            assert!(
                all.iter().any(|a| a == want),
                "{want:?} missing from {all:?}"
            );
        }
        assert!(
            !all.iter().any(|a| a == " \"a\"" || a == " \"b\""),
            "an enum arm must not end on its bare closing quote: {all:?}"
        );
    }

    /// A closed set compiles to a choice, with `null` among the arms exactly
    /// when the schema allows it. A string is not a closed set.
    #[test]
    fn a_closed_set_compiles_to_a_choice() {
        let arms_for = |value: Value| {
            let schema = json!({"type": "object", "properties": {"v": value}, "required": ["v"]});
            let spec = compile_tool_call_tree(
                &[ToolSpec::from_json_schema("t", &schema)],
                &ToolCallEnvelope::qwen3(),
            )
            .unwrap();
            // The one-tool name branch is not the value's.
            let mut a: Vec<String> = arms(&spec).into_iter().filter(|a| a != "t\"").collect();
            a.sort();
            a
        };
        assert_eq!(arms_for(json!({"type": "boolean"})), [" false", " true"]);
        assert_eq!(
            arms_for(json!({"type": ["boolean", "null"]})),
            [" false", " null", " true"]
        );
        // An enum arm carries what follows the value — here the call's close —
        // so the value's closing quote and the close are one arm's text.
        let close = "}}\n</tool_call>";
        assert_eq!(
            arms_for(json!({"enum": ["x", "y", null]})),
            [
                format!(" \"x\"{close}"),
                format!(" \"y\"{close}"),
                format!(" null{close}")
            ]
        );
        assert_eq!(
            arms_for(json!({"enum": ["x", "y"]})),
            [format!(" \"x\"{close}"), format!(" \"y\"{close}")]
        );
        // A nullable string is not a closed set: it is a free value, so the
        // model's own ` ""` and ` null` tokens are both reachable and no arm
        // prefills its opening quote.
        assert!(arms_for(json!({"anyOf": [{"type": "string"}, {"type": "null"}]})).is_empty());
        // An enum value is written as JSON, escapes and all.
        assert_eq!(
            arms_for(json!({"enum": ["say \"x\"", "b"]})),
            [
                format!(" \"b\"{close}"),
                format!(" \"say \\\"x\\\"\"{close}")
            ]
        );
        // An array of booleans chooses at every element.
        let a = arms_for(json!({"type": "array", "items": {"type": "boolean"}}));
        for arm in ["true", "false", "]", ", true", ", false"] {
            assert!(a.iter().any(|x| x == arm), "{arm:?} missing from {a:?}");
        }
    }

    /// **A guided array honours `minItems`**: the levels before the minimum
    /// offer no `]`, so `"pushes": []` cannot be decoded for a `git_push` that
    /// needs one push. Each unrolled level offers one `]` arm otherwise, so
    /// the count of them is the count of levels the array may close at.
    #[test]
    fn a_guided_array_offers_no_close_before_its_minimum() {
        let closes = |min: Option<u64>| {
            let mut array = json!({"type": "array", "items": {"type": "boolean"}});
            if let Some(min) = min {
                array["minItems"] = json!(min);
            }
            let schema = json!({"type": "object", "properties": {"v": array}, "required": ["v"]});
            let tool = ToolSpec::from_json_schema("t", &schema);
            assert_eq!(tool.params[0].min_items, min.unwrap_or(0) as usize);
            let spec = compile_tool_call_tree(&[tool], &ToolCallEnvelope::qwen3()).unwrap();
            arms(&spec).into_iter().filter(|a| a == "]").count()
        };
        assert_eq!(closes(None), MAX_ARRAY_ELEMENTS);
        assert_eq!(closes(Some(1)), MAX_ARRAY_ELEMENTS - 1);
        assert_eq!(closes(Some(3)), MAX_ARRAY_ELEMENTS - 3);
        // A minimum past the unrolled bound could only be written as exactly
        // the bound, which the schema refuses: the array is left free.
        let schema = json!({"type": "object", "properties": {"v": {
            "type": "array", "items": {"type": "boolean"}, "minItems": 1000}}});
        let past = &ToolSpec::from_json_schema("t", &schema).params[0];
        assert!(past.items.is_none(), "{past:?}");
        assert_eq!(past.min_items, 0);
    }

    /// `git_push`'s real schema — `pushes` an array of `$ref`'d objects with
    /// `minItems: 1` — compiles with its first element required.
    #[test]
    fn a_push_needs_a_push() {
        let schema = json!({
            "type": "object",
            "definitions": {"PushItem": {"type": "object", "properties": {
                "action": {"type": "string", "enum": ["update_branch", "delete_branch"]},
                "name": {"type": "string"}},
                "required": ["action", "name"]}},
            "properties": {
                "remote": {"type": "string"},
                "pushes": {"type": "array", "minItems": 1, "maxItems": 50,
                           "items": {"$ref": "#/definitions/PushItem"}}},
            "required": ["remote", "pushes"]
        });
        let tool = ToolSpec::from_json_schema("git_push", &schema);
        let pushes = tool.params.iter().find(|p| p.name == "pushes").unwrap();
        assert_eq!(pushes.min_items, 1);
        let spec = compile_tool_call_tree(&[tool], &ToolCallEnvelope::qwen3()).unwrap();
        // Every level but the first may close: the first element is written.
        let closes = arms(&spec).into_iter().filter(|a| a == "]").count();
        assert_eq!(
            closes,
            MAX_ARRAY_ELEMENTS - 1,
            "an empty push list is offered"
        );
    }

    /// A mode-switched request, `git_commit`'s shape: `from` decides which of
    /// the optional fields a call needs, through `if`/`then`.
    fn moded_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "repo": {"type": "string"},
                "from": {"type": "string", "enum": ["changes", "files", "patch", "revert", "noop"]},
                "message": {"type": "string"},
                "changes": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                "patch": {"type": "string"},
                "commit": {"type": "string"}
            },
            "required": ["repo", "from"],
            "allOf": [
                {"if": {"properties": {"from": {"const": "files"}}},
                 "then": {"required": ["message", "changes"]}},
                {"if": {"properties": {"from": {"enum": ["changes", "patch"]}}},
                 "then": {"required": ["message"]}},
                {"if": {"properties": {"from": {"const": "patch"}}},
                 "then": {"required": ["patch"]}},
                {"if": {"properties": {"from": {"const": "revert"}}},
                 "then": {"required": ["commit"]}},
                {"if": {"properties": {"from": {"const": "files"}, "repo": {"const": "x"}}},
                 "then": {"required": ["patch"]}}
            ]
        })
    }

    /// **Each `if`/`then` lands on the field it tests**, value by value — an
    /// `enum` naming several — and a rule testing two fields is not one field's.
    #[test]
    fn value_requirements_are_read_onto_the_field_they_test() {
        let tool = ToolSpec::from_json_schema("t", &moded_schema());
        let from = tool.params.iter().find(|p| p.name == "from").unwrap();
        let s = |v: &[&str]| v.iter().map(|f| f.to_string()).collect::<Vec<_>>();
        assert_eq!(
            from.requires,
            vec![
                ("files".to_string(), s(&["message", "changes"])),
                ("changes".to_string(), s(&["message"])),
                ("patch".to_string(), s(&["message"])),
                ("patch".to_string(), s(&["patch"])),
                ("revert".to_string(), s(&["commit"])),
            ]
        );
        assert!(tool
            .params
            .iter()
            .all(|p| p.name == "from" || p.requires.is_empty()));
    }

    /// **A value cannot close the call before the fields it requires**: each
    /// arm of `from` goes on into its own first required key, and only a value
    /// that requires nothing may close.
    #[test]
    fn a_mode_value_goes_on_into_the_fields_it_requires() {
        let spec = compile_tool_call_tree(
            &[ToolSpec::from_json_schema("t", &moded_schema())],
            &ToolCallEnvelope::qwen3(),
        )
        .unwrap();
        let all = arms(&spec);
        let has = |want: &str| all.iter().any(|a| a == want);
        for want in [
            " \"files\", \"message\":",
            " \"changes\", \"message\":",
            " \"patch\", \"message\":",
            " \"revert\", \"commit\":",
            " \"noop\"}}\n</tool_call>",
            " \"noop\", \"message\":",
        ] {
            assert!(has(want), "{want:?} missing from {all:?}");
        }
        for closed in ["files", "changes", "patch", "revert"] {
            let early = format!(" \"{closed}\"}}}}\n</tool_call>");
            assert!(!has(&early), "{closed} may close the call: {all:?}");
        }
        // Required after their value, the keys are the grammar's to write.
        let fixed = statics(&spec);
        for key in [", \"changes\": [", ", \"patch\":"] {
            assert!(
                fixed.iter().any(|s| s == key),
                "{key:?} missing from {fixed:?}"
            );
        }
    }

    /// The function-block shape branches the same way, each value ending on
    /// the element's own marker.
    #[test]
    fn a_mode_value_branches_in_the_function_block_shape_too() {
        let env = ToolCallEnvelope::qwen35();
        let spec =
            compile_tool_call_tree(&[ToolSpec::from_json_schema("t", &moded_schema())], &env)
                .unwrap();
        let all = arms(&spec);
        for value in ["changes", "files", "patch", "revert", "noop"] {
            let want = format!("{value}{}", env.param_close);
            assert!(all.contains(&want), "{want:?} missing from {all:?}");
        }
    }

    /// Which arrays and objects the grammar writes, read off the spec: a free
    /// `JsonValue` span per value the model writes whole.
    #[test]
    fn only_elements_the_grammar_can_begin_are_guided() {
        let free_spans = |schema: &str| {
            let schema: Value = serde_json::from_str(schema).unwrap();
            let tool = ToolSpec::from_json_schema("t", &schema);
            let spec = compile_tool_call_tree(&[tool], &ToolCallEnvelope::qwen3()).unwrap();
            spec.nodes
                .iter()
                .filter(|n| {
                    matches!(
                        n,
                        NodeSpec::FreeText {
                            term: Terminator::JsonValue,
                            ..
                        }
                    )
                })
                .count()
        };
        let array_of = |items: &str| {
            format!(
                r#"{{"type":"object","properties":{{"a":{{"type":"array","items":{items}}}}},
                    "required":["a"]}}"#
            )
        };
        // Strings, closed sets and objects with fields are written by the
        // grammar.
        assert_eq!(free_spans(&array_of(r#"{"type":"string"}"#)), 0);
        assert_eq!(free_spans(&array_of(r#"{"type":"boolean"}"#)), 0);
        assert_eq!(free_spans(&array_of(r#"{"type":["boolean","null"]}"#)), 0);
        assert_eq!(free_spans(&array_of(r#"{"enum":["a","b"]}"#)), 0);
        assert_eq!(
            free_spans(&array_of(
                r#"{"type":"object","properties":{"p":{"type":"string"}},"required":["p"]}"#
            )),
            0
        );
        // Numbers open on no delimiter: the whole array is one free value.
        assert_eq!(free_spans(&array_of(r#"{"type":"integer"}"#)), 1);
        // An array inside a guided array is free — once per unrolled element.
        assert_eq!(
            free_spans(&array_of(
                r#"{"type":"object","properties":{
                     "tags":{"type":"array","items":{"type":"string"}}},"required":["tags"]}"#
            )),
            MAX_ARRAY_ELEMENTS
        );
    }

    /// The guided shapes compile against a vocabulary that merges across their
    /// boundaries the way a BPE tokenizer does — `[{`, `[]`, `}]`, `}, {`, `["`.
    #[test]
    fn a_guided_array_of_objects_compiles_with_bracket_merges() {
        let v = TestVocab::new()
            .with_special("[{", 300)
            .with_special("[]", 301)
            .with_special("}]", 302)
            .with_special("}, {", 303)
            .with_special("{\"", 304)
            .with_special("\"]", 305)
            .with_special("[\"", 306)
            .with_special("\": [", 307)
            .with_special(" [", 308);
        let schema: Value = serde_json::from_str(
            r#"{"type":"object","properties":{
                 "files":{"type":"array","items":{"type":"object","properties":{
                   "path":{"type":"string"},"start_line":{"type":["number","null"]}},
                   "required":["path"]}},
                 "commands":{"type":"array","items":{"type":"string"}}},
               "required":["files"]}"#,
        )
        .unwrap();
        let tool = ToolSpec::from_json_schema("t", &schema);
        let spec = compile_tool_call_tree(&[tool], &ToolCallEnvelope::qwen3()).unwrap();
        compile(&spec, &v).expect("guided arrays compile across bracket merges");
    }

    /// Required parameters take the `required` list's order; optionals follow
    /// in the order the schema text declares them, unsorted. Parsed from text
    /// rather than built with `json!`, because a client's schema arrives as
    /// text.
    #[test]
    fn params_follow_the_required_list_then_declared_order() {
        let schema: Value = serde_json::from_str(
            r#"{
                "type": "object",
                "properties": {
                    "width": {"type": "integer"},
                    "path": {"type": "string"},
                    "height": {"type": "integer"},
                    "note": {"type": "string"},
                    "after": {"type": "string"}
                },
                "required": ["path", "note"]
            }"#,
        )
        .unwrap();
        let spec = ToolSpec::from_json_schema("probe", &schema);
        let names: Vec<&str> = spec.params.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["path", "note", "width", "height", "after"]);
        assert!(spec.params[..2].iter().all(|p| p.required));
        assert!(spec.params[2..].iter().all(|p| !p.required));
    }
}
