//! Front-end B — compile a tool catalog into a tool-call stencil tree.
//!
//! The catalog (the same JSON that feeds the prompt's tool list) becomes a tree
//! that guarantees: the tool name is one of the catalog's, the JSON parses,
//! every required parameter is present in order, optionals appear in declared
//! order in any subset, enum values are exactly the allowed strings, and no
//! leading/trailing comma is ever produced.
//!
//! Value handling by type:
//! - `string` — a free-text span closed at the unescaped closing quote.
//! - `boolean` — a `true`/`false` branch.
//! - string `enum` — a branch over the allowed strings.
//! - `integer`/`number`/`array`/`object` — emitted as any structurally-valid
//!   JSON value (`Terminator::JsonValue`), lookahead-terminated at the enclosing
//!   `,`/`}`, which the session pushes back to the next node.  This guarantees
//!   valid JSON structure without strictly enforcing the scalar type.

use std::collections::{BTreeSet, HashMap};

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

/// One tool parameter.
#[derive(Debug, Clone, Deserialize)]
pub struct Param {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: ParamType,
    #[serde(default)]
    pub required: bool,
    /// When present, the value is constrained to one of these strings.
    #[serde(default, rename = "enum")]
    pub enum_values: Option<Vec<String>>,
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
    pub fn from_json_schema(name: &str, schema: &Value) -> ToolSpec {
        let required: BTreeSet<&str> = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
            .unwrap_or_default();
        let mut params = Vec::new();
        if let Some(props) = schema.get("properties").and_then(|p| p.as_object()) {
            // Iterating the object yields a deterministic field order.
            for (pname, pschema) in props {
                let enum_values = pschema
                    .get("enum")
                    .and_then(|e| e.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect::<Vec<_>>()
                    })
                    .filter(|v| !v.is_empty());
                params.push(Param {
                    name: pname.clone(),
                    ty: parse_param_type(pschema),
                    required: required.contains(pname.as_str()),
                    enum_values,
                });
            }
        }
        ToolSpec {
            name: name.to_string(),
            params,
        }
    }
}

fn parse_param_type(pschema: &Value) -> ParamType {
    let type_str = match pschema.get("type") {
        Some(Value::String(s)) => Some(s.as_str()),
        Some(Value::Array(arr)) => arr.iter().filter_map(|v| v.as_str()).find(|s| *s != "null"),
        _ => None,
    };
    match type_str {
        Some("string") => ParamType::String,
        Some("integer") => ParamType::Integer,
        Some("number") => ParamType::Number,
        Some("boolean") => ParamType::Boolean,
        Some("array") => ParamType::Array,
        // "object" or anything unrecognized → any structurally-valid JSON value.
        _ => ParamType::Object,
    }
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
                    // The value is a JSON string here, so a quote in it would
                    // end the string early. Examples are authored prose and do
                    // contain them.
                    s.push_str(&format!("\"{k}\": {}", Value::String((*v).to_string())));
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
}

/// The tool-call stencil tree's label. The scheduler keys tool-call-specific
/// behavior off the active stencil's tree label (repetition-penalty
/// suppression, the in-call reprojection freeze at first-token promotion), so
/// the label is a shared constant rather than a string literal in each place.
pub const TOOL_CALL_TREE_LABEL: &str = "tool_call";

/// Compile a tool catalog into a [`TreeSpec`].  Errors on an empty catalog or a
/// name/enum collision the trie rejects.
pub fn compile_tool_call_tree(
    tools: &[ToolSpec],
    env: &ToolCallEnvelope,
) -> Result<TreeSpec, BuildError> {
    if tools.is_empty() {
        return Err(BuildError::ToolSchema("empty tool catalog".into()));
    }
    let mut b = ToolTreeBuilder {
        spec: TreeSpec::new(TOOL_CALL_TREE_LABEL),
        env,
    };
    let end = b.spec.push(NodeSpec::End);

    // Each tool: name arm -> args_open static -> its argument object -> close.
    let mut arms: Vec<(String, SpecId)> = Vec::with_capacity(tools.len());
    for tool in tools {
        let args_entry = b.build_args(&tool.params, end)?;
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
    let mut b = ToolTreeBuilder {
        spec: TreeSpec::new(TOOL_CALL_TREE_LABEL),
        env,
    };
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
            let args_entry = b.build_args(&tool.params, after_call)?;
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
        // continuation arm carries the marker the model would have emitted to
        // start another call, so choosing it is choosing to act again.
        after_call = b.spec.push(NodeSpec::Branch {
            arms: vec![(env.marker.clone(), open), (close_turn.to_string(), end)],
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

struct ToolTreeBuilder<'a> {
    spec: TreeSpec,
    env: &'a ToolCallEnvelope,
}

/// `(optional index, emitted_any) -> gate entry`, per-tool, so the gate graph
/// stays linear instead of exploding over subsets — and never leaks between
/// tools, which have different optional lists.
type GateMemo = HashMap<(usize, bool), SpecId>;

impl ToolTreeBuilder<'_> {
    /// The argument object's field sequence, ending at `end` (via the envelope
    /// close).  Returns the entry node.
    fn build_args(&mut self, params: &[Param], end: SpecId) -> Result<SpecId, BuildError> {
        let required: Vec<&Param> = params.iter().filter(|p| p.required).collect();
        let optional: Vec<&Param> = params.iter().filter(|p| !p.required).collect();
        let mut memo: GateMemo = HashMap::new();

        // Optional gates start with emitted_any = (a required field precedes them).
        let mut opt_entry = self.opt_gates(&optional, 0, !required.is_empty(), end, &mut memo)?;

        // Prepend the required fields, in order, building backwards.
        for (i, p) in required.iter().enumerate().rev() {
            let (leadin, value) = self.build_value(p, opt_entry)?;
            opt_entry = self.spec.push(NodeSpec::Static {
                text: self.key(p, i == 0, leadin),
                next: value,
            });
        }
        Ok(opt_entry)
    }

    /// What opens one argument, up to the point its value begins.
    ///
    /// `first` says whether a separator is needed before it — which only the
    /// JSON shapes have. A function block's arguments are self-delimiting
    /// elements, so there is nothing between them and the flag is ignored.
    fn key(&self, p: &Param, first: bool, leadin: &'static str) -> String {
        match self.env.style {
            CallStyle::FunctionBlock => format!(
                "{}{}{}",
                self.env.param_open, p.name, self.env.param_name_close
            ),
            CallStyle::JsonBlock | CallStyle::Lines => {
                let sep = if first { "" } else { ", " };
                format!("{sep}\"{}\": {leadin}", p.name)
            }
        }
    }

    /// The gate over optionals `idx..`, given whether a field was already emitted.
    fn opt_gates(
        &mut self,
        opts: &[&Param],
        idx: usize,
        emitted_any: bool,
        end: SpecId,
        memo: &mut GateMemo,
    ) -> Result<SpecId, BuildError> {
        if let Some(&id) = memo.get(&(idx, emitted_any)) {
            return Ok(id);
        }
        // No more optionals: emit the envelope close and finish.
        if idx == opts.len() {
            let id = self.spec.push(NodeSpec::Static {
                text: self.env.close.clone(),
                next: end,
            });
            memo.insert((idx, emitted_any), id);
            return Ok(id);
        }
        let mut arms: Vec<(String, SpecId)> = Vec::with_capacity(opts.len() - idx + 1);
        for (j, p) in opts.iter().enumerate().skip(idx) {
            // Include optional j: a field is emitted, so everything after has
            // emitted_any = true.
            let after = self.opt_gates(opts, j + 1, true, end, memo)?;
            let (leadin, value) = self.build_value(p, after)?;
            arms.push((self.key(p, !emitted_any, leadin), value));
        }
        // The "stop" arm: close the object.
        arms.push((self.env.close.clone(), end));
        let id = self.spec.push(NodeSpec::Branch { arms });
        memo.insert((idx, emitted_any), id);
        Ok(id)
    }

    /// A value sub-tree for `p`, transitioning to `next` after the value.
    /// Returns a `lead-in` string that must be appended to the preceding key/arm
    /// static, plus the value's entry node.  Folding the lead-in (a string/enum
    /// value's opening `"`) into the key keeps structural merges like ` "`
    /// internal to one static, rather than leaving a lone `"` after a branch arm
    /// that merges backward into the committed arm (an unrepresentable retract).
    fn build_value(
        &mut self,
        p: &Param,
        next: SpecId,
    ) -> Result<(&'static str, SpecId), BuildError> {
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
                return Ok(("", branch));
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
            return Ok(("", span));
        }
        if let Some(values) = &p.enum_values {
            // `"` <branch over `value"`> — the closing quote rides on each arm
            // so a value that prefixes another stays distinguishable.  The
            // opening `"` is the lead-in (folded into the key).
            let branch = self.spec.push(NodeSpec::Branch {
                arms: values.iter().map(|v| (format!("{v}\""), next)).collect(),
            });
            return Ok(("\"", branch));
        }
        match p.ty {
            ParamType::String => {
                let span = self.spec.push(NodeSpec::FreeText {
                    term: Terminator::JsonString,
                    eos_ends: false,
                    limits: FreeTextLimits::json_string(),
                    close_token: None,
                    suppress_close: false,
                    next,
                });
                Ok(("\"", span))
            }
            ParamType::Boolean => Ok((
                "",
                self.spec.push(NodeSpec::Branch {
                    arms: vec![("true".into(), next), ("false".into(), next)],
                }),
            )),
            // Numbers, arrays, and objects are emitted as any structurally-valid
            // JSON value, lookahead-terminated at the enclosing `,`/`}` (the
            // session pushes that delimiter back).  This guarantees valid JSON
            // structure; it does not strictly enforce the scalar type.
            ParamType::Integer | ParamType::Number | ParamType::Array | ParamType::Object => Ok((
                "",
                self.spec.push(NodeSpec::FreeText {
                    term: Terminator::JsonValue,
                    eos_ends: false,
                    limits: FreeTextLimits::json_value(),
                    close_token: None,
                    suppress_close: false,
                    next,
                }),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stencil::compile::compile;
    use crate::stencil::vocab::TestVocab;

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
            after_close: "",
        };
        let prelude = compile_think_tree(ThinkMode::Balanced, &env).unwrap();
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
}
