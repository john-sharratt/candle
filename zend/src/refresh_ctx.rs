//! Shared context the ingest refresh paths thread through.
//!
//! Bundles the bits that don't change between refresh calls — the
//! engine handle, the projection schema, the dialect config — so
//! the per-call signatures stay small and don't blow past clippy's
//! seven-argument limit.  Per-refresh inputs (workspace path,
//! walked map, prior state, old timeline id, progress sink) remain
//! explicit parameters because they vary per call.

use std::sync::{Arc, Mutex};

use candle_conversation::projection::{Builder, TimelineId};
use candle_conversation::stencil::TriggerRegistry;
use candle_conversation::{ConversationEngine, SequenceConfig};
use zend_tools::ToolContext;

use crate::retrieval_scope::RetrievalScope;

/// Refresh-time context.  Borrows the engine's `Mutex` so the
/// refresh helpers can lock it briefly for the two engine API calls
/// (`new_conversation_with_projection` at the start, then
/// `tombstone_timeline` at the end) and release it across the
/// minutes-long prefill + summary-decode window in between.
///
/// `proj_builder` and `config` are `Clone` (schemas are `Arc`-backed)
/// so the helpers clone what they consume per-call.
#[derive(Clone)]
pub struct RefreshContext<'a> {
    pub engine: &'a Mutex<ConversationEngine>,
    pub proj_builder: Builder,
    pub config: SequenceConfig,
    /// The dialect-formatted system-prompt prelude every REAL conversation
    /// primes on — the same text `base_conv` (the live dialogue's shared
    /// prefix) was built from. `code_reading`'s hidden per-file conversations
    /// use this instead of a bespoke ingest-only prompt, so they frame
    /// identically to a live dialogue turn (`InferenceState::load`'s
    /// `formatted_prompt`).
    pub formatted_prompt: &'a str,
    /// Tool-call grammar + `<think>` steering for `ThinkMode::Quick` — the
    /// lowest thinking level, not fully off: a hidden ingest conversation with
    /// no room to reason at all was measured skipping its `file_read` call
    /// entirely and guessing a summary from the filename. Compiled from the
    /// REAL tool catalog (not an empty placeholder) — see `turn_triggers`.
    pub think_triggers: Arc<TriggerRegistry>,
    /// Tool-execution context a hidden ingest conversation's real `file_read`
    /// calls run against — read-only grants, the daemon's own workspace
    /// (`ToolMode::Restricted`'s context, the same one an unprivileged live
    /// dialogue turn runs tools in). Each unit's conversation runs in a copy
    /// of it with file stores of its own, as every conversation does.
    pub tool_ctx: Arc<ToolContext>,
    /// Told whenever a unit commits, so the next turn's scope can find it.
    pub retrieval: &'a RetrievalScope,
    /// The conversation a unit's new conversation descends from: the priming
    /// chain's end (`crate::branch_ingest::prime`), or — while the chain is
    /// being built — the link before the one being read. Recorded as the
    /// unit's parent before its reading starts, so its turns are projected
    /// with the chain already in context. `None` when there is no chain.
    pub chain_end: Option<TimelineId>,
}
