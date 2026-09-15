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

use candle_conversation::{ProjectionBucket, ProjectionEvent};
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

/// A projection point as a conversation's history carries it: everything the
/// timeline bars and the hover popover draw, and none of the selection or the
/// materialized spine. Those two are read only by the projection panel and are
/// most of an event's bytes — about ten kilobytes a point, so a conversation of
/// a dozen turns carried a megabyte of them on every load. `turn` and `event`
/// address the full point for the panel, which fetches it when it opens
/// (`GET /v1/conversations/{id}/projections/{turn}/{event}`).
#[derive(Debug, Clone, Serialize)]
pub struct ProjectionSpanOut {
    pub id: u64,
    pub region: &'static str,
    pub step: String,
    pub start_token: u32,
    pub seconds: f64,
    pub materialized_tokens: u32,
    pub substrate_tokens: u32,
    pub buckets: Vec<ProjectionBucket>,
    pub self_reference: bool,
    /// Which assistant turn's record, in the conversation's projection order.
    pub turn: usize,
    /// Which point within that turn.
    pub event: usize,
}

impl ProjectionSpanOut {
    /// The light form of `out`, the `event`-th point of the `turn`-th record.
    pub fn of(out: &ProjectionEventOut, turn: usize, event: usize) -> Self {
        let e = &out.event;
        Self {
            id: out.id,
            region: out.region,
            step: out.step.clone(),
            start_token: e.start_token,
            seconds: e.seconds,
            materialized_tokens: e.materialized_tokens,
            substrate_tokens: e.substrate_tokens,
            buckets: e.buckets.clone(),
            self_reference: e.self_reference,
            turn,
            event,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_conversation::projection::{MaterializedPiece, ProjectionSelection, SystemItem};

    /// The light span carries every field the timeline draws, its address, and
    /// neither of the panel-only fields — however much the full point holds.
    #[test]
    fn a_light_span_drops_the_selection_and_names_its_point() {
        let event = ProjectionEvent {
            start_token: 512,
            seconds: 1.5,
            materialized_tokens: 9000,
            substrate_tokens: 40000,
            selection: ProjectionSelection {
                system: vec![SystemItem::Glue {
                    name: "system_start".into(),
                    content: "<|im_start|>system\n".into(),
                    tokens: 3,
                }],
                turns: Vec::new(),
            },
            materialized: vec![MaterializedPiece::Glue {
                text: "<|im_start|>user\n".into(),
            }],
            ..Default::default()
        };
        let out = ProjectionEventOut::answer(7, event);
        let light = serde_json::to_value(ProjectionSpanOut::of(&out, 2, 4)).unwrap();
        assert_eq!(
            light,
            serde_json::json!({
                "id": 7,
                "region": "answer",
                "step": "t=512",
                "start_token": 512,
                "seconds": 1.5,
                "materialized_tokens": 9000,
                "substrate_tokens": 40000,
                "buckets": [],
                "self_reference": false,
                "turn": 2,
                "event": 4,
            })
        );
    }
}
