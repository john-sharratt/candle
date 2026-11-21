//! The prompts a layer's turns are compressed by, one level of the summary tree
//! at a time.
//!
//! Each summary turn keeps both halves of an exchange, so it stays role-coherent
//! when it is injected again. The user half is always derived from the children's
//! scopes and never decoded, because a decode would invent a question nobody
//! asked. The assistant half is either decoded under the level's prompt or built
//! deterministically from its children, which suits content whose structure is
//! fixed by its input, where a decode would be pure cost and would invite
//! fabrication. A compression prompt is sealed once as a section and injected at
//! the head of every compression pass that uses it.

use super::*;

/// A single compression framing + instruction pair — the unit the compression
/// path consumes.
///
/// The `system_prompt` is sealed once as a section (its [`SectionId`] is
/// allocated from the schema's section pool but never added to
/// `system_prompt.items`, so it only ever materialises via the compression seal,
/// never in a normal projection) and injected zero-copy at the head of a
/// compression pass. The `user_prompt` is prefilled as text after the content —
/// the "user turn" of the compression frame.
#[derive(Debug, Clone)]
pub struct CompressionPrompt {
    pub system_prompt: SectionSchema,
    pub user_prompt: String,
}

impl CompressionPrompt {
    /// A placeholder with an empty prompt and a stand-in section id, used by the
    /// `Default` impls below. Real schemas overwrite both via the YAML/builder
    /// path, which allocates a unique [`SectionId`] per prompt.
    pub(super) fn placeholder(name: &str) -> Self {
        Self {
            system_prompt: SectionSchema {
                id: SectionId::new(1),
                name: name.to_string(),
                content: String::new(),
                priority: 50.0,
                depends_on: None,
                depends_on_absent: None,
                depends_on_configured: None,
                depends_on_unconfigured: None,
                is_template: false,
                template_tokens: None,
            },
            user_prompt: String::new(),
        }
    }
}

/// How a compression level produces a summary's **assistant half** (its
/// content — the answer body).
///
/// Orthogonal to [`Scope`], which derives the *user* half. Any level may pair
/// any `Content` with any `Scope`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Content {
    /// The model decodes the content from the children under this level's
    /// [`CompressionPrompt`]. The default, and right for prose: an assistant
    /// half is exactly what a model decode is good at producing.
    #[default]
    Decode,
    /// The content is built deterministically from the children, with no model
    /// decode at all (`summary_tree::structural`). For content whose structure
    /// is fully determined by its input — directory trees — where a decode is
    /// pure cost and invites fabrication, and a faithful *merge* just unions the
    /// children so the result grows toward the root instead of compressing.
    Structural,
}

/// How one tree-level of a layer's turns is compressed.
///
/// A summary turn is a compressed *exchange* and keeps both bodies of a real
/// turn — the user-message half (`Role::User`, `[user_start, user_end)`) and the
/// assistant-response half (`Role::Assistant`, `[asst_start, total)`) — so it
/// stays role-coherent when re-injected, and carries a scope for retrieval to
/// match against. The two halves are produced by different means:
///
/// - the **user half** is always *derived* from the children's scopes
///   ([`Scope`]) — never decoded, because a decode always speaks as the
///   assistant and would be inventing a question that was never asked;
/// - the **assistant half** is produced per [`Content`]: decoded under
///   [`Self::assistant`], or built deterministically.
///
/// There is deliberately no user-half prompt: nothing would ever run it.
#[derive(Debug, Clone)]
pub struct TurnSummary {
    /// Derives the user-message half (the scope) from the children's scopes.
    pub scope: Scope,
    /// Compresses the assistant-response half of a turn (`Role::Assistant`).
    /// Unused when `content` is [`Content::Structural`].
    pub assistant: CompressionPrompt,
    /// Hard decode-token ceiling for the compressed assistant half.
    pub max_tokens: usize,
    /// How the assistant half is produced: decoded (default) or structural.
    pub content: Content,
}

impl Default for TurnSummary {
    fn default() -> Self {
        Self {
            scope: Scope::default(),
            assistant: CompressionPrompt::placeholder("__summary_assistant__"),
            max_tokens: 0,
            content: Content::Decode,
        }
    }
}

/// How a layer's turns are compressed across the summary tree.
///
/// `turns` drives `SummaryOfTurns` nodes (compressing raw turns); `summaries`
/// drives `SummaryOfSummaries` nodes (compressing already-compressed children).
/// When `summaries` is `None`, those nodes reuse `turns`.
#[derive(Debug, Clone, Default)]
pub struct LayerSummary {
    pub turns: TurnSummary,
    pub summaries: Option<TurnSummary>,
}

impl LayerSummary {
    /// The [`TurnSummary`] driving compression for a node: `summaries` for a
    /// `SummaryOfSummaries` node (falling back to `turns`), else `turns`.
    pub fn for_kind(&self, is_summary_of_summaries: bool) -> &TurnSummary {
        if is_summary_of_summaries {
            self.summaries.as_ref().unwrap_or(&self.turns)
        } else {
            &self.turns
        }
    }

    /// Push every compression-prompt [`SectionId`] (raw) this layer summary owns
    /// into `out` — the `turns` halves and, if present, the `summaries` halves.
    pub(super) fn push_section_ids(&self, out: &mut Vec<u32>) {
        self.turns.push_section_ids(out);
        if let Some(s) = &self.summaries {
            s.push_section_ids(out);
        }
    }
}

impl TurnSummary {
    pub(super) fn push_section_ids(&self, out: &mut Vec<u32>) {
        out.push(self.assistant.system_prompt.id.raw());
    }
}
