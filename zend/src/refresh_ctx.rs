//! Shared context the watcher-driven refresh paths thread through.
//!
//! Bundles the bits that don't change between refresh calls — the
//! engine handle, the projection schema, the dialect config — so
//! the per-call signatures stay small and don't blow past clippy's
//! seven-argument limit.  Per-refresh inputs (workspace path,
//! walked map, prior state, old timeline id, progress sink) remain
//! explicit parameters because they vary per call.

use std::sync::Mutex;

use candle_conversation::projection::Builder;
use candle_conversation::{ConversationEngine, SequenceConfig};

/// Refresh-time context.  Borrows the engine's `Mutex` so the
/// refresh helpers can lock it briefly for the two engine API calls
/// (`new_conversation_with_projection` at the start, then
/// `tombstone_timeline` at the end) and release it across the
/// minutes-long prefill + summary-decode window in between.
///
/// `proj_builder` and `config` are `Clone` (schemas are `Arc`-backed)
/// so the helpers clone what they consume per-call.
pub struct RefreshContext<'a> {
    pub engine: &'a Mutex<ConversationEngine>,
    pub proj_builder: Builder,
    pub config: SequenceConfig,
}
