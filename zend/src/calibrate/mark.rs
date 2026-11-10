//! Marking every sealed calibration provenance-only, whatever layer wrote it.
//!
//! A calibration's end state is archived and distilled provenance-only
//! ([`super::seal_calibration`]): its K/V is never written, and its signatures
//! stay. The marker is what tells another member the turns are complete
//! without chunks (`docs/replicated_substrate.md` §6.5); a calibration
//! archived without it is held there as still arriving, for good.
//!
//! The pass covers every calibration in the substrate, not one layer's: a
//! layer that no longer declares `calibrate:` runs no pass of its own, and its
//! calibrations would otherwise never be marked.

use candle_conversation::persistence::record::DistillMode;
use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;

use super::run::MARKER_KEY;

/// One calibration as the pass sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sealed {
    pub timeline: TimelineId,
    pub archived: bool,
    pub distilled: bool,
}

/// The calibrations of `found` to mark: archived, and not marked yet. One
/// not archived is still being sealed, or a crash's, which its layer's pass
/// retires.
pub fn unmarked(found: &[Sealed]) -> Vec<TimelineId> {
    found
        .iter()
        .filter(|s| s.archived && !s.distilled)
        .map(|s| s.timeline)
        .collect()
}

/// Mark every archived calibration in `engine`'s substrate that is not marked
/// provenance-only yet. Returns how many it marked.
pub fn mark_sealed(engine: &ConversationEngine) -> usize {
    let found: Vec<Sealed> = engine
        .conversations_with_metadata_key(MARKER_KEY)
        .into_iter()
        .map(|(timeline, _)| Sealed {
            timeline,
            archived: engine.is_conversation_archived(timeline),
            distilled: engine.is_timeline_distilled(timeline),
        })
        .collect();
    let mut marked = 0;
    for timeline in unmarked(&found) {
        match engine.distill_timeline(timeline, DistillMode::ProvenanceOnly) {
            Ok(()) => marked += 1,
            Err(e) => tracing::warn!(
                target: "zend::calibrate",
                timeline = timeline.raw(),
                "marking a sealed calibration provenance-only failed: {e}",
            ),
        }
    }
    if marked > 0 {
        tracing::info!(
            target: "zend::calibrate",
            marked,
            "sealed calibrations marked provenance-only",
        );
    }
    marked
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only an archived calibration not marked yet is marked: one still being
    /// sealed and one marked already are left alone.
    #[test]
    fn only_a_sealed_unmarked_calibration_is_marked() {
        let [sealed, sealing, marked] =
            [21u64, 22, 23].map(|raw| TimelineId::from_raw(raw).unwrap());
        let found = [
            Sealed {
                timeline: sealed,
                archived: true,
                distilled: false,
            },
            Sealed {
                timeline: sealing,
                archived: false,
                distilled: false,
            },
            Sealed {
                timeline: marked,
                archived: true,
                distilled: true,
            },
        ];
        assert_eq!(unmarked(&found), vec![sealed]);
    }
}
