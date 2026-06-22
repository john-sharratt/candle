use crate::config::SamplingConfig;
use crate::token_buffer::TokenBuffer;
use serde::{Deserialize, Serialize};

/// Monotonic turn identifier within a conversation.
pub type TurnId = u64;

/// Role of a turn participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Role {
    /// System prompt (always turn 0, pinned HOT).
    System,
    /// User message.
    User,
    /// Assistant response (generated).
    Assistant,
}

impl Default for Role {
    /// `Role::Assistant` — substrate turn entries represent completed
    /// user→assistant exchanges, so the assistant role is the natural
    /// default for a sealed entry.
    fn default() -> Self {
        Self::Assistant
    }
}

/// A single conversation turn.
#[derive(Debug, Clone)]
pub struct Turn {
    /// Unique, monotonic turn identifier within this conversation.
    pub id: TurnId,

    /// The role (System, User, or Assistant).
    pub role: Role,

    /// Raw text content.
    pub text: String,

    /// Pre-tokenized form (cached to avoid re-tokenization).
    pub token_ids: TokenBuffer,
}

/// Per-turn options, overriding conversation defaults.
#[derive(Debug, Clone, Default)]
pub struct TurnOptions {
    /// Max tokens to generate. `None` = conversation default.
    pub max_tokens: Option<usize>,

    /// Sampling configuration override for this turn.
    /// `None` = use conversation default.
    pub sampling: Option<SamplingConfig>,

    /// Additional tokens to ban for this turn only.
    /// These are merged with the conversation's banned token list.
    pub turn_banned_tokens: Vec<i32>,

    /// Stencil constraint override for this turn.
    /// If set, overrides any conversation-level stencil.
    pub turn_stencil: Option<Vec<i32>>,

    /// Text to prefill as the start of the assistant's response, decoded from
    /// rather than sampled. The model is forced to continue from this prefix —
    /// e.g. seeding `<tool_call>` commits the decode to the tool-call grammar so
    /// it cannot refuse, narrate, or fabricate a result. The prefix is pinned
    /// into the turn's K/V and sealed as part of the assistant content; the
    /// model decodes the continuation. `None` = ordinary free decode.
    pub assistant_prefill: Option<String>,
}

impl TurnOptions {
    /// Create options with just max tokens.
    pub fn with_max_tokens(max_tokens: usize) -> Self {
        Self {
            max_tokens: Some(max_tokens),
            ..Default::default()
        }
    }

    /// Create options with sampling config.
    pub fn with_sampling(sampling: SamplingConfig) -> Self {
        Self {
            sampling: Some(sampling),
            ..Default::default()
        }
    }

    /// Builder: set max tokens.
    pub fn max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Builder: set sampling config.
    pub fn sampling(mut self, config: SamplingConfig) -> Self {
        self.sampling = Some(config);
        self
    }

    /// Builder: add banned tokens for this turn.
    pub fn ban_tokens(mut self, tokens: Vec<i32>) -> Self {
        self.turn_banned_tokens = tokens;
        self
    }

    /// Builder: set stencil constraint for this turn.
    pub fn stencil(mut self, tokens: Vec<i32>) -> Self {
        self.turn_stencil = Some(tokens);
        self
    }

    /// Builder: seed the assistant response with `prefix`, forcing the decode to
    /// continue from it (see [`Self::assistant_prefill`]).
    pub fn assistant_prefill(mut self, prefix: impl Into<String>) -> Self {
        self.assistant_prefill = Some(prefix.into());
        self
    }
}
