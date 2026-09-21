//! Shared context the watcher-driven refresh paths thread through.
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

/// Refresh-time context.  Borrows the engine's `Mutex` so the
/// refresh helpers can lock it briefly for the two engine API calls
/// (`new_conversation_with_projection` at the start, then
/// `tombstone_timeline` at the end) and release it across the
/// minutes-long prefill + summary-decode window in between.
///
/// `proj_builder` and `config` are `Clone` (schemas are `Arc`-backed)
/// so the helpers clone what they consume per-call. The whole context is
/// `Clone` for the same reason: `priming_chain::build` re-points
/// `priming_chain_end` per link while it constructs the chain itself.
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
    /// dialogue turn runs tools in).
    pub tool_ctx: Arc<ToolContext>,
    /// The priming chain's final link (`priming_chain::build`'s result), or
    /// `None` when no anchor file was found. Every unit a `refresh_*` pass
    /// mints from now on records this conversation as its PARENT before its
    /// own reading starts, so the anchor documents are already in its
    /// projection (`Substrate::inherited_chain`) — the same starting point
    /// `base_conv` itself is parented onto at boot. A durable metadata
    /// pointer, not a copy: nothing is duplicated and nothing is pinned.
    pub priming_chain_end: Option<TimelineId>,
}
