//! Holds the dialect-specific strings that wrap a tool call, so the grammar and
//! the prompt's worked example are built from the same shape.
//!
//! A dialect's call is either a JSON object inside a tool-call tag or a nested
//! element per argument. The envelope names which, and the grammar is built from
//! it, so a prompt cannot teach a syntax the grammar will not accept. Each name's
//! terminator rides on its arm, so a name that prefixes another stays distinct in
//! the trie. The assistant-turn close ends exactly on the turn terminator, since
//! the decode only ends the turn on a terminator that is the final token of the
//! close.

use super::*;

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
    /// The most tokens one free string value may run to before the grammar
    /// closes it as written — [`MAX_STRING_VALUE_TOKENS`] unless a caller asks
    /// for less.
    ///
    /// **A field that has said what it holds is closed, not left to wander.**
    /// The default is sized for a whole file a call writes. An answer whose
    /// fields are sentences wants far less: a table's `checked`, its finding
    /// stated, went on "I am iron. I am steel. I am copper…" to the turn's cap
    /// and was lost with the call around it. Bounded, the value is closed where
    /// it stands and the call completes.
    pub max_string_tokens: u32,
}

impl ToolCallEnvelope {
    /// The same envelope with its free string values bounded at `tokens` — see
    /// [`Self::max_string_tokens`].
    pub fn with_max_string_tokens(mut self, tokens: u32) -> Self {
        self.max_string_tokens = tokens;
        self
    }

    /// The span limits a free string value of `p` is decoded under: the
    /// envelope's bound, or the field's own where it is tighter.
    ///
    /// **A field sized for a sentence is closed at a sentence's length.** One
    /// bound for every value of a call is sized for its longest: a table's
    /// one-sentence `event` went "datethe is datethe" for 1,800 tokens under a
    /// bound sized for its whole reading, and the call was refused for it.
    ///
    /// **A field's bound is its value's, beside its delimiters.** The span
    /// counts every token it emits, and the model writes the value's opening
    /// quote and its close inside it ([`Terminator::JsonStringValue`]), so a
    /// bound of `maxLength` closed the value one character short: `file_edit`'s
    /// four-character `at` came out `8Qy` for `8QyN` on every call and was
    /// refused each time for its length. The field's bound is lifted by
    /// [`VALUE_DELIMITER_TOKENS`], still never above the envelope's.
    pub(super) fn string_limits(&self, p: &Param) -> FreeTextLimits {
        FreeTextLimits {
            forced_after: p.max_tokens.map_or(self.max_string_tokens, |m| {
                m.saturating_add(VALUE_DELIMITER_TOKENS)
                    .min(self.max_string_tokens)
            }),
            ..FreeTextLimits::json_string()
        }
    }

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
            max_string_tokens: MAX_STRING_VALUE_TOKENS,
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
            max_string_tokens: MAX_STRING_VALUE_TOKENS,
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
pub(super) fn json_scalar(v: &str) -> String {
    if v == "true" || v == "false" || v.parse::<f64>().is_ok() {
        v.to_string()
    } else {
        Value::String(v.to_string()).to_string()
    }
}
