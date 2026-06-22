//! The projection-event payload served on the chat SSE stream (and, on
//! hydrate, the substrate endpoint), docs/zend_ui_redesign.md §2.3.
//!
//! One event is emitted per decode, right after the turn seals: it carries the
//! engine's [`ProjectionEvent`] — the materialized-context composition
//! (system / section groups / turns, with per-category token counts), the
//! materialized-vs-substrate totals, and the decode throughput — wrapped with
//! the small display fields the GUI timeline needs (a stable id, a region the
//! dot anchors to, and a short step label). The GUI derives the bar map,
//! legend, and readouts from this; we send the engine numerics verbatim.

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
    /// Wrap a decode-end event, anchored to the answer region.
    pub fn answer(id: u64, event: ProjectionEvent) -> Self {
        let step = format!("t={}", event.end_token);
        Self {
            id,
            region: "answer",
            step,
            event,
        }
    }
}
