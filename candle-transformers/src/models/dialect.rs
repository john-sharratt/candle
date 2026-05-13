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
}

impl std::fmt::Display for DialectType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DialectType::ChatML => write!(f, "ChatML"),
            DialectType::Llama2 => write!(f, "Llama2"),
            DialectType::Llama3 => write!(f, "Llama3"),
        }
    }
}

impl DialectType {
    pub fn dialect(&self) -> Dialect {
        match self {
            DialectType::ChatML => Dialect::chat_ml(),
            DialectType::Llama2 => Dialect::llama2(),
            DialectType::Llama3 => Dialect::llama3(),
        }
    }
}

/// Structural tokens for a chat template dialect.
///
/// Each field is a static string fragment used to assemble prompts.
/// Models that support thinking mode use [`no_think_block`]
/// as a prefill to suppress `<think>` blocks.
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
    pub no_think_block: &'static str,
    pub no_think: &'static str,
    pub think_block: &'static str,
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
        }
    }

    pub fn dialect_type(&self) -> DialectType {
        self.dialect_type
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

    /// Whether this dialect supports thinking mode suppression.
    pub fn supports_no_think(&self) -> bool {
        !self.no_think_block.is_empty()
    }

    /// Returns the assistant start token(s), optionally followed by a thinking-mode
    /// prefix or a no-think block.
    ///
    /// | `thinking_capable` | `suppress_thinking` | suffix injected |
    /// |---|---|---|
    /// | `true`  | `false` | `think_block` (`"<think>\n"`) — forces an open think block into
    ///   the prefill so the model continues with reasoning rather than generating
    ///   `<think>` stochastically (or skipping it altogether on abliterated variants). |
    /// | `true`  | `true`  | `no_think_block` (`"<think>\n\n</think>\n\n"`) — closes the block
    ///   immediately so the model never generates reasoning tokens. |
    /// | `false` | *any*   | no suffix — non-thinking model, plain assistant header only. |
    pub fn active_assistant_start(&self, suppress_thinking: bool, thinking_capable: bool) -> String {
        if thinking_capable && suppress_thinking && !self.no_think_block.is_empty() {
            format!("{}{}", self.assistant_start, self.no_think_block)
        } else if thinking_capable && !suppress_thinking && !self.think_block.is_empty() {
            format!("{}{}", self.assistant_start, self.think_block)
        } else {
            self.assistant_start.to_string()
        }
    }
}
