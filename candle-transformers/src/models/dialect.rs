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
    Llama2,
    Llama3,
    DeepSeek,
}

impl std::fmt::Display for DialectType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DialectType::ChatML => write!(f, "ChatML"),
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
            DialectType::Llama2 => Dialect::llama2(),
            DialectType::Llama3 => Dialect::llama3(),
            DialectType::DeepSeek => Dialect::deepseek(),
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
    /// The empty/closed reasoning block a thinking model emits under `/no_think`
    /// (`"<think>\n\n</think>\n\n"` for Qwen3).  Descriptive only — never
    /// force-prefilled into the assistant header (the model produces it itself,
    /// or the `/no_think` text alone suppresses reasoning); retained as a
    /// structural-noise seed for the BDP scan.
    pub no_think_block: &'static str,
    /// The `/no_think` soft-switch text — emitted by the section tree's
    /// `no_think` node and prepended to prefilled (never-decoded) turns.
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
}
