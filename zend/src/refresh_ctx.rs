//! Shared context the ingest refresh paths thread through.
//!
//! Bundles the bits that don't change between refresh calls — the
//! engine handle, the projection schema, the dialect config — so
//! the per-call signatures stay small and don't blow past clippy's
//! seven-argument limit.  Per-refresh inputs (workspace path,
//! walked map, prior state, old timeline id, progress sink) remain
//! explicit parameters because they vary per call.

use std::sync::Arc;

use candle_conversation::projection::{Builder, GroupId};
use candle_conversation::stencil::{ThinkMode, TriggerRegistry};
use candle_conversation::{ConversationEngine, Sequence, SequenceConfig};
use zend_tools::ToolContext;

use crate::retrieval_scope::RetrievalScope;
use crate::types::ToolMode;

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
    pub engine: &'a ConversationEngine,
    pub proj_builder: Builder,
    pub config: SequenceConfig,
    /// The dialect-formatted system-prompt prelude every REAL conversation
    /// primes on — the same text `base_conv` (the live dialogue's shared
    /// prefix) was built from. `code_reading`'s hidden per-file conversations
    /// use this instead of a bespoke ingest-only prompt, so they frame
    /// identically to a live dialogue turn (`InferenceState::load`'s
    /// `formatted_prompt`).
    pub formatted_prompt: &'a str,
    /// Tool-call grammar + `<think>` steering for `ThinkMode::Off` over the
    /// restricted tools mode: what an analysis unit runs under when
    /// [`Self::think_for`] gives none.
    pub think_triggers: Arc<TriggerRegistry>,
    /// Tool-execution context the ingest reads files and folders through —
    /// read-only grants, the daemon's own workspace (`ToolMode::Restricted`'s
    /// context). Each unit reads in a copy of it with file stores pinned to
    /// the commit it was found on.
    pub tool_ctx: Arc<ToolContext>,
    /// Told whenever a unit commits, so the next turn's scope can find it.
    pub retrieval: &'a RetrievalScope,
    /// The steering for each thinking level and tools mode, for a unit of a
    /// layer that thinks (`crate::layers`) — the same registry a dialogue turn
    /// at that level and mode runs under, so a unit whose tools only a wider
    /// mode's grammar holds can still call them. `None` falls back to
    /// [`Self::think_triggers`].
    pub think_for: Option<ThinkFor>,
    /// The live dialogue a day layer reads: its group. `None` when there is
    /// none.
    pub dialogue: Option<GroupId>,
}

/// The steering for a thinking level, over a tools mode's call grammar.
pub type ThinkFor = Arc<dyn Fn(ThinkMode, ToolMode) -> Arc<TriggerRegistry> + Send + Sync>;

impl RefreshContext<'_> {
    /// `base`'s fork for a new analysis unit.
    pub fn fork_unit(&self, base: &Sequence) -> candle_conversation::Result<Sequence> {
        base.fork()
    }

    /// The steering a turn thinking at `mode` runs under, its tool calls held
    /// to `tools`' grammar.
    pub fn triggers_for(&self, mode: ThinkMode, tools: ToolMode) -> Arc<TriggerRegistry> {
        match &self.think_for {
            Some(f) => f(mode, tools),
            None => Arc::clone(&self.think_triggers),
        }
    }
}
