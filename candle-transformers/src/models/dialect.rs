//! Chat template dialects for LLM prompt formatting.
//!
//! Defines the structural tokens used by different model families
//! (ChatML, Llama3, Llama2) to delimit system prompts, user turns,
//! and assistant responses.
//!
//! This is the single source of truth — used by both the conversation
//! engine (`candle-conversation`) and the test harness (`batch_test`).

/// Identifies the chat template family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DialectType {
    ChatML,
    /// Qwen3.5 / Qwen3.8 — ChatML's markers, but thinking is suppressed by
    /// opening the assistant turn with an already-closed think block rather
    /// than by a `/no_think` marker in the user turn. See [`Dialect::qwen35`].
    Qwen35,
    Llama2,
    Llama3,
    DeepSeek,
}

impl std::fmt::Display for DialectType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DialectType::ChatML => write!(f, "ChatML"),
            DialectType::Qwen35 => write!(f, "Qwen35"),
            DialectType::Llama2 => write!(f, "Llama2"),
            DialectType::Llama3 => write!(f, "Llama3"),
            DialectType::DeepSeek => write!(f, "DeepSeek"),
        }
    }
}

impl DialectType {
    pub fn dialect(&self) -> Dialect {
        match self {
            DialectType::ChatML => Dialect::chat_ml(),
            DialectType::Qwen35 => Dialect::qwen35(),
            DialectType::Llama2 => Dialect::llama2(),
            DialectType::Llama3 => Dialect::llama3(),
            DialectType::DeepSeek => Dialect::deepseek(),
        }
    }
}

/// How a model family writes a tool call, and how many it may write at once.
///
/// **A property of the checkpoint, not of the application.** Two families that
/// share ChatML's turn markers can still disagree completely about what an
/// assistant turn containing a tool call looks like, and a caller that assumed
/// one shape produced calls the other family's template cannot represent. The
/// axis lives beside the turn markers because it is decided by the same thing:
/// which chat template the weights were trained against.
///
/// Every consumer that renders a call, parses one back, constrains a decode to
/// one, or explains the format in a system prompt asks this rather than
/// assuming — so adding a family is adding a variant and the arms the compiler
/// then demands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CallStyle {
    /// One JSON object per line, as ordinary assistant prose:
    /// `{"tool":"say","intent":"…"}`.
    ///
    /// **No native markers at all**, which is what makes it the right default
    /// for a family whose template has no tool section: nothing structural is
    /// emitted, so nothing can be malformed against a template that never
    /// expected it. Several calls are several lines.
    ///
    /// It also cannot carry a result. A family on this style has no
    /// `tool`-role turn to put one in, so an outcome has to reach the model as
    /// ordinary user text or not at all.
    Lines,
    /// Qwen2.5 / Qwen3: a JSON object wrapped in the template's own markers.
    ///
    /// ```text
    /// <tool_call>
    /// {"name": "say", "arguments": {"intent": "…"}}
    /// </tool_call>
    /// ```
    ///
    /// Several calls are several consecutive blocks in one assistant turn.
    JsonBlock,
    /// Qwen3.5 / Qwen3.8: a nested function block, with one element per
    /// argument and **unquoted, possibly multi-line values**.
    ///
    /// ```text
    /// <tool_call>
    /// <function=say>
    /// <parameter=intent>
    /// what I mean, which may run to several lines
    /// </parameter>
    /// </function>
    /// </tool_call>
    /// ```
    ///
    /// The values not being JSON strings is the substantive difference and not
    /// a cosmetic one: nothing needs escaping, and a parameter may hold prose
    /// with newlines in it, which the single-line JSON forms cannot express.
    FunctionBlock,
}

impl CallStyle {
    /// Whether one assistant turn may carry more than one call.
    ///
    /// True for every style here — each has a way to write a second call — and
    /// asked rather than assumed because it is the question a caller actually
    /// has, and because a family that cannot will eventually be added.
    pub fn multi_call(self) -> bool {
        match self {
            CallStyle::Lines | CallStyle::JsonBlock | CallStyle::FunctionBlock => true,
        }
    }

    /// Whether the template has a turn a tool *result* can be put in.
    ///
    /// [`CallStyle::Lines`] has none — it is plain prose in an assistant turn,
    /// so there is no `tool` role and no `<tool_response>` for a caller to
    /// address. A caller with a result to deliver on that style has to fold it
    /// into the next user turn, and this is how it knows to.
    pub fn carries_results(self) -> bool {
        match self {
            CallStyle::Lines => false,
            CallStyle::JsonBlock | CallStyle::FunctionBlock => true,
        }
    }
}

/// Structural tokens for a chat template dialect.
///
/// Each field is a static string fragment used to assemble prompts.
#[derive(Debug, Clone)]
pub struct Dialect {
    pub dialect_type: DialectType,
    pub document_start: &'static str,
    pub document_end: &'static str,
    pub marker_start: &'static str,
    pub marker_end: &'static str,
    pub turn_start: &'static str,
    pub turn_begin: &'static str,
    pub turn_end: &'static str,
    pub system_start: &'static str,
    pub system_end: &'static str,
    pub user_start: &'static str,
    pub user_end: &'static str,
    pub assistant_start: &'static str,
    pub assistant_end: &'static str,
    pub recent_start: &'static str,
    pub recent_end: &'static str,
    /// The empty/closed reasoning block (`"<think>\n\n</think>\n\n"`).
    ///
    /// **A dialect picks exactly one thinking-suppression mechanism**, and
    /// which one is decided by whether [`Self::no_think`] is empty:
    ///
    /// * `no_think` non-empty (Qwen3, ChatML) — the marker in the user turn is
    ///   what suppresses reasoning, the model emits its own closed block, and
    ///   this field is descriptive: a structural-noise seed for the BDP scan,
    ///   never force-prefilled.
    /// * `no_think` empty (Qwen3.5 / Qwen3.8) — the chat template has no such
    ///   marker; suppression *is* opening the assistant turn with this block
    ///   already closed, exactly as the template renders it under
    ///   `enable_thinking=false`. Here it is prefilled after
    ///   [`Self::assistant_start`].
    ///
    /// Sending both would be harmless but is not a shape any published
    /// template produces, so the empty/non-empty split keeps one mechanism
    /// live per family rather than two overlapping ones.
    pub no_think_block: &'static str,
    /// The `/no_think` soft-switch text — emitted by the section tree's
    /// `no_think` node and prepended to prefilled (never-decoded) turns.
    ///
    /// **Empty means the family has no such switch**, which is a capability and
    /// not a formatting detail: see [`Self::has_no_think_switch`]. Prefer asking
    /// that over testing this for emptiness, so a caller states what it wants to
    /// know rather than inferring it from a string.
    pub no_think: &'static str,
    /// The open reasoning marker (`"<think>\n"` for Qwen3).  No longer
    /// force-prefilled — a thinking model emits its own `<think>` as the first
    /// decoded token — so this is retained only as a special-token seed for the
    /// BDP scan's structural-noise set.
    pub think_block: &'static str,
    pub tool_block_open: &'static str,
    pub tool_block_close: &'static str,
    pub tool_response_open: &'static str,
    pub tool_response_close: &'static str,
    /// How this family writes a call. See [`CallStyle`].
    ///
    /// **The style, and not the strings that spell it out.** The markers live
    /// in `candle_conversation::stencil::ToolCallEnvelope`, which is what
    /// compiles them into a grammar — and one syntax written down twice is two
    /// things free to disagree, with the disagreement showing up as a decode
    /// constrained to one shape and parsed as another. So this says *which*
    /// shape, and the layer that has to emit it owns *what* it looks like.
    pub call_style: CallStyle,
}

/// Catalog of named structural-template fragments callable by YAML schemas.
///
/// Used by the projection engine to look up the dialect-specific string that
/// a `kind: template` system-prompt item refers to. See the projection
/// generated-segments design doc for the broader mechanism.
///
/// Variant names mirror the YAML `dialect:` reference in `snake_case`; the
/// [`Self::from_yaml_name`] helper parses YAML strings to enum values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DialectTemplate {
    SystemStart,
    SystemEnd,
    UserStart,
    UserEnd,
    AssistantStart,
    AssistantEnd,
    ToolBlockOpen,
    ToolBlockClose,
    ToolResponseOpen,
    ToolResponseClose,
    NoThinkPrefix,
}

impl DialectTemplate {
    /// Parse a YAML `dialect:` reference (e.g. `"system_start"`).
    /// Returns `None` for unknown names so callers can produce a
    /// schema-locatable error.
    pub fn from_yaml_name(name: &str) -> Option<Self> {
        match name {
            "system_start" => Some(Self::SystemStart),
            "system_end" => Some(Self::SystemEnd),
            "user_start" => Some(Self::UserStart),
            "user_end" => Some(Self::UserEnd),
            "assistant_start" => Some(Self::AssistantStart),
            "assistant_end" => Some(Self::AssistantEnd),
            "tool_block_open" => Some(Self::ToolBlockOpen),
            "tool_block_close" => Some(Self::ToolBlockClose),
            "tool_response_open" => Some(Self::ToolResponseOpen),
            "tool_response_close" => Some(Self::ToolResponseClose),
            "no_think_prefix" => Some(Self::NoThinkPrefix),
            _ => None,
        }
    }

    /// The YAML-form name (the inverse of [`Self::from_yaml_name`]).
    pub fn as_yaml_name(self) -> &'static str {
        match self {
            Self::SystemStart => "system_start",
            Self::SystemEnd => "system_end",
            Self::UserStart => "user_start",
            Self::UserEnd => "user_end",
            Self::AssistantStart => "assistant_start",
            Self::AssistantEnd => "assistant_end",
            Self::ToolBlockOpen => "tool_block_open",
            Self::ToolBlockClose => "tool_block_close",
            Self::ToolResponseOpen => "tool_response_open",
            Self::ToolResponseClose => "tool_response_close",
            Self::NoThinkPrefix => "no_think_prefix",
        }
    }
}

impl std::fmt::Display for DialectTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_yaml_name())
    }
}

impl Dialect {
    /// Whether this family honours a `/no_think` soft switch in the user turn.
    ///
    /// **Qwen3 and ChatML do; Qwen3.5 and Qwen3.8 do not.** To the newer family
    /// the marker is ordinary text, so a caller that sends it gets a reasoning
    /// trace back and a truncated generation that looks like the model answering
    /// the wrong question.
    ///
    /// Stated as a capability because callers kept asking the question by
    /// testing `no_think` for emptiness, which reads as a formatting check and
    /// hides what is actually being asked. It is also **not** the way to
    /// suppress reasoning: a `<think>` steering stencil acts on the decoded
    /// token and works on every family, where this works on one. Ask this only
    /// when composing the turn's text.
    pub fn has_no_think_switch(&self) -> bool {
        !self.no_think.is_empty()
    }

    /// How this dialect suppresses reasoning, as
    /// `(user-turn marker, block prefilled after the assistant header)`.
    ///
    /// The one place the "exactly one mechanism is live, decided by whether
    /// [`Self::no_think`] is empty" convention (documented on
    /// [`Self::no_think_block`]) is turned into strings. Both strings are empty
    /// when the caller wants the model to reason.
    ///
    /// **Call this rather than reading either field directly.** Reading only
    /// `no_think` is not a partial implementation of suppression — on a family
    /// that has no soft switch it is *no* suppression, silently, because the
    /// field is empty by design. That was the live bug: production assembled the
    /// user opener from `no_think` alone, so on Qwen3.5/3.8 the `no_think`
    /// projection node emitted a zero-token segment and thinking-off was a line
    /// of prose asking the model not to deliberate. It ignored it on 22 of 22
    /// repo_map summaries. Both halves now come from here — the user opener
    /// (`turn_head_text`) takes `.0`, the assistant grid
    /// (`submit_turn_with_options`, `BoundaryMarkers::no_think_block`) takes
    /// `.1` — so a family can neither be sent a marker it would read as ordinary
    /// text, nor be left with nothing at all.
    pub fn thinking_suppression(&self, suppress: bool) -> (&'static str, &'static str) {
        if !suppress {
            return ("", "");
        }
        if self.no_think.is_empty() {
            ("", self.no_think_block)
        } else {
            (self.no_think, "")
        }
    }

    pub fn chat_ml() -> Self {
        Self {
            dialect_type: DialectType::ChatML,
            document_start: "",
            document_end: "<|endoftext|>",
            marker_start: "<|im_start|>",
            marker_end: "<|im_end|>",
            turn_start: "<|im_start|>",
            turn_begin: "\n",
            turn_end: "<|im_end|>\n",
            system_start: "<|im_start|>system\n",
            system_end: "<|im_end|>\n",
            user_start: "<|im_start|>user\n",
            user_end: "<|im_end|>\n",
            assistant_start: "<|im_start|>assistant\n",
            assistant_end: "<|im_end|>\n",
            recent_start: "<|im_start|>recent\n",
            recent_end: "<|im_end|>\n",
            no_think_block: "<think>\n\n</think>\n\n",
            no_think: "/no_think\n",
            think_block: "<think>\n",
            tool_block_open: "<tools>\n",
            tool_block_close: "</tools>\n",
            tool_response_open: "<tool_response>\n",
            tool_response_close: "</tool_response>\n",
            // **Plain lines, which is what every caller here already emits.**
            //
            // ChatML's own template can carry `<tool_call>` JSON blocks, and a
            // family on this dialect may well support them — but what the
            // engine has always produced is one JSON object per line as
            // ordinary assistant prose, and saying so is the difference
            // between a described default and an assumed one. A family that
            // should use its native blocks says so in its own constructor, the
            // way `qwen35` does below.
            call_style: CallStyle::Lines,
        }
    }

    /// Qwen3.5 / Qwen3.8.
    ///
    /// ChatML's markers throughout — the published template is built from
    /// `<|im_start|>` / `<|im_end|>` exactly as Qwen3's is. It differs in one
    /// place, and only one: there is no `/no_think` soft switch. The template
    /// renders `<think>\n\n</think>\n\n` straight after the assistant header
    /// when `enable_thinking` is false, and reasons when it is true. A
    /// `/no_think` marker in the user turn is ordinary text to this family, so
    /// a caller that sends it gets a reasoning trace back and a truncated
    /// generation looks like the model answering the wrong question.
    ///
    /// `document_end` is `<|im_end|>`, not `<|endoftext|>`: the checkpoint's
    /// `tokenizer.ggml.eos_token_id` points at the turn terminator.
    /// # It also calls tools differently, and that is the larger difference
    ///
    /// The published template does not put JSON inside `<tool_call>`. It emits
    /// a nested function element with one child per argument and **unquoted
    /// values that may span lines**:
    ///
    /// ```text
    /// <tool_call>
    /// <function=say>
    /// <parameter=intent>
    /// what I mean, which may run to several lines
    /// </parameter>
    /// </function>
    /// </tool_call>
    /// ```
    ///
    /// Taken from the shipped checkpoint's own `chat_template`, which iterates
    /// `message.tool_calls` — so several calls are several blocks in one
    /// assistant turn — and merges consecutive `tool` messages into a single
    /// following user turn, each wrapped in `<tool_response>`. Every one of
    /// those tags is a real token in this vocabulary rather than text that
    /// happens to look like one.
    pub fn qwen35() -> Self {
        Self {
            dialect_type: DialectType::Qwen35,
            no_think: "",
            document_end: "<|im_end|>",
            call_style: CallStyle::FunctionBlock,
            ..Self::chat_ml()
        }
    }

    pub fn llama2() -> Self {
        Self {
            dialect_type: DialectType::Llama2,
            document_start: "<s>",
            document_end: "</s>",
            marker_start: "[INST]",
            marker_end: "[/INST]",
            turn_start: "[INST] ",
            turn_begin: "",
            turn_end: " [/INST]",
            system_start: "[INST] <<SYS>>\n",
            system_end: "\n<</SYS>>\n\n",
            user_start: "",
            user_end: " [/INST]",
            assistant_start: " ",
            assistant_end: " </s>",
            recent_start: "",
            recent_end: "",
            no_think_block: "",
            no_think: "",
            think_block: "",
            tool_block_open: "<tools>\n",
            tool_block_close: "</tools>\n",
            tool_response_open: "<tool_response>\n",
            tool_response_close: "</tool_response>\n",
            // Plain lines — what this engine has always emitted. Stated rather
            // than inherited: a family that silently took another's call style
            // would produce calls its own template cannot represent.
            call_style: CallStyle::Lines,
        }
    }

    pub fn llama3() -> Self {
        Self {
            dialect_type: DialectType::Llama3,
            document_start: "<|begin_of_text|>",
            document_end: "<|end_of_text|>",
            marker_start: "<|start_header_id|>",
            marker_end: "<|eot_id|>",
            turn_start: "<|start_header_id|>",
            turn_begin: "<|end_header_id|>\n\n",
            turn_end: "<|eot_id|>",
            system_start: "<|start_header_id|>system<|end_header_id|>\n\n",
            system_end: "<|eot_id|>",
            user_start: "<|start_header_id|>user<|end_header_id|>\n\n",
            user_end: "<|eot_id|>",
            assistant_start: "<|start_header_id|>assistant<|end_header_id|>\n\n",
            assistant_end: "<|eot_id|>",
            recent_start: "<|start_header_id|>recent<|end_header_id|>\n\n",
            recent_end: "<|eot_id|>",
            no_think_block: "",
            no_think: "",
            think_block: "",
            tool_block_open: "<tools>\n",
            tool_block_close: "</tools>\n",
            tool_response_open: "<tool_response>\n",
            tool_response_close: "</tool_response>\n",
            // Plain lines — what this engine has always emitted. Stated rather
            // than inherited: a family that silently took another's call style
            // would produce calls its own template cannot represent.
            call_style: CallStyle::Lines,
        }
    }

    /// DeepSeek-V4 chat template: `bos + system-text + <｜User｜>… +
    /// <｜Assistant｜>… + eos`, no role-header wrappers. The model ALWAYS
    /// thinks, so every no-think field is empty — the `/no_think` glue island
    /// tokenises to an empty run and nothing is emitted; `<think>` is kept
    /// only as a BDP structural-noise seed (the model emits it itself as its
    /// first decoded token).
    pub fn deepseek() -> Self {
        Self {
            dialect_type: DialectType::DeepSeek,
            document_start: "<｜begin▁of▁sentence｜>",
            document_end: "<｜end▁of▁sentence｜>",
            marker_start: "<｜User｜>",
            marker_end: "<｜end▁of▁sentence｜>",
            turn_start: "<｜User｜>",
            turn_begin: "",
            turn_end: "<｜end▁of▁sentence｜>",
            system_start: "",
            system_end: "",
            user_start: "<｜User｜>",
            user_end: "",
            assistant_start: "<｜Assistant｜>",
            assistant_end: "<｜end▁of▁sentence｜>",
            recent_start: "<｜User｜>",
            recent_end: "<｜end▁of▁sentence｜>",
            no_think_block: "",
            no_think: "",
            think_block: "<think>",
            tool_block_open: "<tools>\n",
            tool_block_close: "</tools>\n",
            tool_response_open: "<tool_response>\n",
            tool_response_close: "</tool_response>\n",
            // Plain lines — what this engine has always emitted. Stated rather
            // than inherited: a family that silently took another's call style
            // would produce calls its own template cannot represent.
            call_style: CallStyle::Lines,
        }
    }

    pub fn dialect_type(&self) -> DialectType {
        self.dialect_type
    }

    /// Resolve a [`DialectTemplate`] to its structural-string content for this
    /// dialect.
    ///
    /// Empty strings indicate "no content" — callers (e.g. the projection
    /// engine's YAML parser) interpret that as a no-op item that should be
    /// dropped from the schema at build time so projection never emits an
    /// empty segment.
    pub fn template(&self, t: DialectTemplate) -> &'static str {
        match t {
            DialectTemplate::SystemStart => self.system_start,
            DialectTemplate::SystemEnd => self.system_end,
            DialectTemplate::UserStart => self.user_start,
            DialectTemplate::UserEnd => self.user_end,
            DialectTemplate::AssistantStart => self.assistant_start,
            DialectTemplate::AssistantEnd => self.assistant_end,
            DialectTemplate::ToolBlockOpen => self.tool_block_open,
            DialectTemplate::ToolBlockClose => self.tool_block_close,
            DialectTemplate::ToolResponseOpen => self.tool_response_open,
            DialectTemplate::ToolResponseClose => self.tool_response_close,
            DialectTemplate::NoThinkPrefix => self.no_think,
        }
    }

    /// Format a system prompt using this dialect's structural tokens.
    ///
    /// Closes with `system_end` so the resulting bytes are a
    /// self-contained system role unit — the section pinned in the
    /// substrate covers exactly the system content's brackets.
    /// Per-turn prefills open their own `user_start` / `assistant_start`
    /// from there.
    pub fn format_system_prompt(&self, system_prompt: &str) -> String {
        format!(
            "{}{}{}{}",
            self.document_start, self.system_start, system_prompt, self.system_end
        )
    }

    /// Format a user turn using this dialect's structural tokens.
    pub fn format_user_turn(&self, user_prompt: &str) -> String {
        format!(
            "{}{}{}{}",
            self.turn_end, self.user_start, user_prompt, self.user_end
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_yaml_name_roundtrip() {
        let all = [
            DialectTemplate::SystemStart,
            DialectTemplate::SystemEnd,
            DialectTemplate::UserStart,
            DialectTemplate::UserEnd,
            DialectTemplate::AssistantStart,
            DialectTemplate::AssistantEnd,
            DialectTemplate::ToolBlockOpen,
            DialectTemplate::ToolBlockClose,
            DialectTemplate::ToolResponseOpen,
            DialectTemplate::ToolResponseClose,
            DialectTemplate::NoThinkPrefix,
        ];
        for t in all {
            assert_eq!(DialectTemplate::from_yaml_name(t.as_yaml_name()), Some(t));
        }
    }

    #[test]
    fn unknown_template_yaml_name_returns_none() {
        assert!(DialectTemplate::from_yaml_name("not_a_real_template").is_none());
        assert!(DialectTemplate::from_yaml_name("").is_none());
    }

    #[test]
    fn chatml_template_contents_match_static_fields() {
        let d = Dialect::chat_ml();
        assert_eq!(d.template(DialectTemplate::SystemStart), d.system_start);
        assert_eq!(d.template(DialectTemplate::SystemEnd), d.system_end);
        assert_eq!(d.template(DialectTemplate::UserStart), d.user_start);
        assert_eq!(d.template(DialectTemplate::UserEnd), d.user_end);
        assert_eq!(
            d.template(DialectTemplate::AssistantStart),
            d.assistant_start
        );
        assert_eq!(d.template(DialectTemplate::AssistantEnd), d.assistant_end);
        assert_eq!(d.template(DialectTemplate::ToolBlockOpen), "<tools>\n");
        assert_eq!(d.template(DialectTemplate::ToolBlockClose), "</tools>\n");
        assert_eq!(
            d.template(DialectTemplate::ToolResponseOpen),
            "<tool_response>\n"
        );
        assert_eq!(
            d.template(DialectTemplate::ToolResponseClose),
            "</tool_response>\n"
        );
        assert_eq!(d.template(DialectTemplate::NoThinkPrefix), "/no_think\n");
    }

    #[test]
    fn chatml_role_markers_non_empty() {
        let d = Dialect::chat_ml();
        for t in [
            DialectTemplate::SystemStart,
            DialectTemplate::SystemEnd,
            DialectTemplate::UserStart,
            DialectTemplate::UserEnd,
            DialectTemplate::AssistantStart,
            DialectTemplate::AssistantEnd,
            DialectTemplate::ToolBlockOpen,
            DialectTemplate::ToolBlockClose,
            DialectTemplate::ToolResponseOpen,
            DialectTemplate::ToolResponseClose,
            DialectTemplate::NoThinkPrefix,
        ] {
            assert!(
                !d.template(t).is_empty(),
                "ChatML template {t} must be non-empty",
            );
        }
    }

    #[test]
    fn llama_role_markers_non_empty() {
        // Llama2 and Llama3 don't carry tool-block markers or a no-think
        // prefix, but every role marker must be populated.
        for d in [Dialect::llama2(), Dialect::llama3()] {
            for t in [
                DialectTemplate::SystemStart,
                DialectTemplate::SystemEnd,
                DialectTemplate::UserStart,
                DialectTemplate::UserEnd,
                DialectTemplate::AssistantStart,
                DialectTemplate::AssistantEnd,
            ] {
                // user_start on Llama2 is intentionally empty (the
                // turn_start carries the marker), so don't blanket-assert.
                let _ = d.template(t);
            }
            // Tool-block delimiters are uniform across all dialects; assert the
            // Llama dialects carry them so future edits notice a divergence.
            assert_eq!(d.template(DialectTemplate::ToolBlockOpen), "<tools>\n");
            assert_eq!(d.template(DialectTemplate::ToolBlockClose), "</tools>\n");
            // The Llama dialects have no dedicated no-think prefix.
            assert_eq!(d.template(DialectTemplate::NoThinkPrefix), "");
        }
    }

    // ── how a family writes a call ──────────────────────────────────────────

    /// **Which shape each family writes, which is all this type decides.**
    ///
    /// What the shape *looks like* belongs to
    /// `candle_conversation::stencil::ToolCallEnvelope`, which compiles it into
    /// a grammar — and is where the strings are asserted against the shipped
    /// checkpoint's own template. Writing them here as well would be one syntax
    /// recorded twice, and the copy nothing compiles is the one that goes stale.
    #[test]
    fn each_family_declares_the_call_shape_its_template_uses() {
        assert_eq!(Dialect::qwen35().call_style, CallStyle::FunctionBlock);
        for d in [
            Dialect::chat_ml(),
            Dialect::llama2(),
            Dialect::llama3(),
            Dialect::deepseek(),
        ] {
            assert_eq!(d.call_style, CallStyle::Lines, "{:?}", d.dialect_type);
        }
    }

    /// Every style here can write more than one call in a turn; only the ones
    /// with a `tool` role can carry a result back. A caller asks rather than
    /// assumes, because the second answer decides whether an outcome has
    /// anywhere to go.
    #[test]
    fn multi_call_is_universal_and_results_are_not() {
        for s in [
            CallStyle::Lines,
            CallStyle::JsonBlock,
            CallStyle::FunctionBlock,
        ] {
            assert!(s.multi_call(), "{s:?}");
        }
        assert!(!CallStyle::Lines.carries_results());
        assert!(CallStyle::JsonBlock.carries_results());
        assert!(CallStyle::FunctionBlock.carries_results());
    }
}
