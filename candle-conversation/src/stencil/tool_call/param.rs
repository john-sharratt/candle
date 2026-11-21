//! Describes a tool's parameters and the schema facts that bound them, as the
//! grammar needs them.
//!
//! A `Param` carries a parameter's type and the constraints the schema places on
//! it: an enum's values, an integer's range, an array's minimum and maximum, a
//! string's pattern, and any required choice among fields. Schemas are read
//! through their named references and single-member `allOf`. A tagged union is
//! merged into one object that lists every tag, so the grammar can enforce the
//! keys and the tags, though not every combination of them. A shape the grammar
//! cannot express stays a free JSON value, which guarantees structure but not
//! type.

use super::*;

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
    /// `array`: the schema's `maxItems`. A guided array offers only its close
    /// after this many elements; `None` bounds it at [`MAX_ARRAY_ELEMENTS`].
    #[serde(default)]
    pub max_items: Option<usize>,
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
    /// `integer`: the schema's `maximum`. Below 10 the value is one digit, and
    /// the grammar offers only the digits up to it — `context` 0–5 cannot be
    /// written 6. Above, it bounds the digit count ([`Self::max_tokens`]), and
    /// a value of the right length past it is the tool's to refuse.
    #[serde(default)]
    pub maximum: Option<f64>,
    /// `string`: the most tokens the value may run to before the grammar closes
    /// it — the schema's `maxLength`, counted in tokens. Bounds the value below
    /// the envelope's [`ToolCallEnvelope::max_string_tokens`], never above it.
    /// `integer`: the digits of the schema's `maximum`.
    #[serde(default)]
    pub max_tokens: Option<u32>,
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
    /// An optional field that is one of a choice the object must make: the
    /// schema's `anyOf` or `oneOf` of `required`. The object cannot close until
    /// a member of the choice is written.
    #[serde(default)]
    pub pick: Pick,
    /// `string`: the schema's `pattern`, a whole-value regular expression the
    /// value is held to token by token ([`super::value_rule::ValueRule`]).
    #[serde(default)]
    pub pattern: Option<String>,
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

/// A schema with its `$ref` and single-member `allOf` indirection followed.
///
/// `schemars` writes a named type as `{"$ref": "#/definitions/Name"}`, and a
/// named type that also carries a description as
/// `{"allOf": [{"$ref": …}], "description": …}`. Neither carries a `type`, so
/// without resolving them every named type in the catalog — a credential kind,
/// a revision, a push action — reaches [`parse_param_type`] as an unrecognized
/// schema and becomes "any JSON value", losing exactly the `enum` the grammar
/// exists to enforce.
pub(super) fn resolve<'a>(schema: &'a Value, root: &'a Value, depth: usize) -> &'a Value {
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
pub(super) fn merge_variants(arms: &[&Value], root: &Value, depth: usize) -> Option<Param> {
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
        max_items: None,
        properties: Some(merged),
        nullable: false,
        minimum: None,
        maximum: None,
        max_tokens: None,
        requires: Vec::new(),
        shapes: Vec::new(),
        pick: Pick::None,
        pattern: None,
    })
}

/// An object schema's properties as parameters, required first in the
/// `required` list's order, then optionals in declared order.
pub(super) fn fields_of(schema: &Value, root: &Value) -> Vec<Param> {
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
    for (field, value, brought) in value_shapes(schema, root) {
        if let Some(p) = params.iter_mut().find(|p| p.name == field) {
            p.shapes.push((value, brought));
        }
    }
    for (rule, pick) in [("anyOf", Pick::AtLeastOne), ("oneOf", Pick::ExactlyOne)] {
        for member in required_choice(schema, rule) {
            if let Some(p) = params.iter_mut().find(|p| p.name == member && !p.required) {
                p.pick = pick;
                p.nullable = false;
            }
        }
    }
    params
}

/// The fields an object's `anyOf` or `oneOf` of `required` chooses among: the
/// field each single-`required` member names. Empty when it is any other kind
/// of union, which [`param_of`] reads.
pub(super) fn required_choice<'a>(schema: &'a Value, rule: &str) -> Vec<&'a str> {
    let Some(members) = schema.get(rule).and_then(Value::as_array) else {
        return Vec::new();
    };
    let named: Option<Vec<&str>> = members
        .iter()
        .map(|m| match m.get("required")?.as_array()?.as_slice() {
            [only] => only.as_str(),
            _ => None,
        })
        .collect();
    named.unwrap_or_default()
}

/// The field one `if` tests, and the values it names: the `if` of
/// [`value_requirements`] and [`value_shapes`]. `None` when it tests more than
/// one field, or anything but values.
pub(super) fn tested_values(rule: &Value) -> Option<(&String, Vec<&str>)> {
    let when = rule.pointer("/if/properties")?.as_object()?;
    let mut tests = when.iter();
    let (Some((field, test)), None) = (tests.next(), tests.next()) else {
        return None;
    };
    let values = match (test.get("const"), test.get("enum")) {
        (Some(one), _) => one.as_str().into_iter().collect(),
        (None, Some(Value::Array(many))) => many.iter().filter_map(Value::as_str).collect(),
        _ => return None,
    };
    Some((field, values))
}

/// What each value of a field makes of a sibling, from an `allOf` of
/// `{"if": {…}, "then": {"properties": {…}}}`, as `(F, V, fields)`: each field
/// the sibling's own schema with the rule laid over it, so a `pattern` for one
/// value leaves the type, and the other values, alone.
///
/// `bytes_transcode`'s `data` is hex or base64 by what `from` says. Its field
/// comes after the value that decides it, or the rule cannot be applied.
pub(super) fn value_shapes(schema: &Value, root: &Value) -> Vec<(String, String, Vec<Param>)> {
    let mut out = Vec::new();
    let Some(rules) = schema.get("allOf").and_then(Value::as_array) else {
        return out;
    };
    for rule in rules {
        let Some(then) = rule.pointer("/then/properties").and_then(Value::as_object) else {
            continue;
        };
        let Some((field, values)) = tested_values(rule) else {
            continue;
        };
        let brought: Vec<Param> = then
            .iter()
            .map(|(name, over)| {
                let mut merged = schema
                    .pointer(&format!("/properties/{name}"))
                    .map_or_else(|| Value::Object(Default::default()), |s| s.clone());
                if let (Value::Object(base), Value::Object(over)) = (&mut merged, over) {
                    base.extend(over.clone());
                }
                Param {
                    name: name.clone(),
                    required: true,
                    ..param_of(&merged, root)
                }
            })
            .collect();
        for value in values {
            out.push((field.clone(), value.to_string(), brought.clone()));
        }
    }
    out
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
pub(super) fn value_requirements(schema: &Value) -> Vec<(String, String, Vec<String>)> {
    let mut out = Vec::new();
    let Some(rules) = schema.get("allOf").and_then(Value::as_array) else {
        return out;
    };
    for rule in rules {
        let Some(then) = rule.pointer("/then/required").and_then(Value::as_array) else {
            continue;
        };
        // One field tested, or the rule is not about one field's value.
        let Some((field, values)) = tested_values(rule) else {
            continue;
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
pub(super) fn param_of(schema: &Value, root: &Value) -> Param {
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
    let pattern = schema
        .get("pattern")
        .and_then(Value::as_str)
        .filter(|_| ty == ParamType::String && enum_values.is_none())
        .map(str::to_string);
    // A pattern takes the place of `null`: a nullable string is a free JSON
    // value, which no rule reaches. Leaving the key out is how such a value is
    // absent, and an `Option` field reads that the same way.
    let nullable = pattern.is_none()
        && (type_nullable
            || enum_members.is_some_and(|a| a.iter().any(Value::is_null))
            || schema.get("nullable").and_then(Value::as_bool) == Some(true));
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
    let guided_max = items.as_ref().and(
        schema
            .get("maxItems")
            .and_then(Value::as_u64)
            .map(|n| n as usize),
    );
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
        max_items: guided_max,
        properties,
        nullable,
        minimum: schema.get("minimum").and_then(Value::as_f64),
        maximum: match ty {
            ParamType::Integer => schema.get("maximum").and_then(Value::as_f64),
            _ => None,
        },
        // Characters bound tokens from above: no token is shorter than one. An
        // integer's bound is its `maximum`'s digits: each is one token.
        max_tokens: match ty {
            ParamType::Integer => schema
                .get("maximum")
                .and_then(Value::as_f64)
                .filter(|m| *m >= 1.0)
                .map(|m| (m.log10().floor() as u32) + 1),
            _ => schema
                .get("maxLength")
                .and_then(Value::as_u64)
                .map(|n| n.min(u32::MAX as u64) as u32),
        },
        requires: Vec::new(),
        shapes: Vec::new(),
        pick: Pick::None,
        pattern,
    }
}

/// The schema's type, and whether `null` is also allowed.
pub(super) fn parse_param_type(pschema: &Value) -> (ParamType, bool) {
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
