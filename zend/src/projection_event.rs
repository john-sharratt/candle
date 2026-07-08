//! The projection-event payload served on the chat SSE stream (and, on
//! hydrate, the substrate endpoint), docs/zend_ui_redesign.md §2.3.
//!
//! A projection is a POINT on the decode timeline: one event is emitted when the
//! turn opens (SubmitTurn), one on each mid-decode reprojection, and one after
//! the turn seals — each carrying the engine's [`ProjectionEvent`] at that
//! `start_token` position: the materialized-context composition (system /
//! section groups / turns, with per-category token counts) and the
//! materialized-vs-substrate totals — wrapped with the small display fields the
//! GUI timeline needs (a stable id, a region the dot anchors to, and a short step
//! label). The GUI reconstructs each projection's governed interval and its
//! throughput from the sequence of points; we send the engine numerics verbatim.

use candle_conversation::ProjectionEvent;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ProjectionEventOut {
    /// Process-global monotonic id — the GUI keys timeline dots on it.
    pub id: u64,
    /// `"think"` | `"answer"` — which decode region the dot anchors to.
    pub region: &'static str,
    /// Short step label, e.g. `"t=512"`.
    pub step: String,
    /// The engine event, flattened so its fields sit alongside id/region/step.
    #[serde(flatten)]
    pub event: ProjectionEvent,
}

impl ProjectionEventOut {
    /// Wrap a projection point (opening / reprojection / post-seal), anchored to
    /// the answer region.
    pub fn answer(id: u64, event: ProjectionEvent) -> Self {
        // `start_token` is the generated-token position at which this projection
        // was selected — the point on the timeline the dot anchors to.
        let step = format!("t={}", event.start_token);
        Self {
            id,
            region: "answer",
            step,
            event,
        }
    }
}
