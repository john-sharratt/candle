//! A step's change to the knowledge base's range, laid onto the slot
//! (`docs/placed_section_layout.md` §3.4, §4.4).
//!
//! A section that enters or leaves at a step changes the range and nothing
//! else: the dialogue is the request's and does not move under it (§3.7), and
//! no other section of the system prompt is chosen again. So the slot is laid
//! out from the segments it was last assembled from, with the range rewritten
//! from the files set's scope — no probe, no scan, no projection — and rebuilt
//! as a reprojection rebuilds it, which keeps every piece before the range and
//! splices the range in place. A slot with no assembly on record, or one
//! whose schema places no range, is reprojected whole.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use super::projection_assembler::{held_sections, HeldSection};
use super::reprojection::{PrepareTimes, ReprojectInFlight, Selected};
use super::Scheduler;
use crate::error::ConversationError;
use crate::files::scope::{Place, RangeScope};
use crate::files::SectionKey;
use crate::projection::{
    files_relayout, ProjectionSegment, ResolvedSection, SealedKind, SectionId, TurnKey,
};
use crate::sequence_handle::SequenceId;

/// `scope`'s range from its front to its top as segments: each hole its
/// positions, each section its member in `held`, and a section this process
/// does not hold a hole of its positions, so every position after it stays.
pub(super) fn range_layout(
    scope: &RangeScope,
    held: &HashMap<SectionKey, HeldSection>,
) -> Vec<ProjectionSegment> {
    scope
        .layout()
        .into_iter()
        .map(|place| match place {
            Place::Hole(positions) => ProjectionSegment::Hole { positions },
            Place::Section(e) => match held.get(&e.key) {
                Some(h) => {
                    ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(h.id)))
                }
                None => ProjectionSegment::Hole {
                    positions: e.tokens,
                },
            },
        })
        .collect()
}

impl Scheduler {
    /// Lay `view_id`'s slot out again for the range its files set holds now:
    /// the rebuild a reprojection runs ([`Self::reproject_rebuild`]) over the
    /// slot's last segments with the range rewritten. `None` when the slot
    /// already holds that layout or the decodes are short of their weights.
    pub(super) fn relayout_view_prepare(
        &mut self,
        view_id: SequenceId,
    ) -> Result<Option<ReprojectInFlight>, ConversationError> {
        if !self.decode_floor_holds_after(0) {
            return Ok(None);
        }
        let t_repro = Instant::now();
        let Some(reading) = self.turn_reading(view_id) else {
            return Ok(None);
        };
        let policy = reading.policy;
        let Some(slot) = self.turn_views.get(&view_id).map(|v| v.parent_id) else {
            return Ok(None);
        };
        let recorded = self.slot_projection_state.get(&slot).and_then(|s| {
            s.applied_composition
                .clone()
                .map(|c| (s.applied_segments.clone(), c))
        });
        let Some((applied, composition)) = recorded else {
            return self.reproject_view_prepare(view_id);
        };
        let segments = {
            let read = policy.substrate.read();
            let Some(scope) = read
                .files_set_of(policy.target.timeline)
                .and_then(|s| s.scope.as_ref())
            else {
                return Ok(None);
            };
            let wanted: HashSet<SectionKey> = scope.sections().map(|e| e.key).collect();
            let held = held_sections(&read, &wanted);
            files_relayout(
                policy.projection.schema(),
                &applied,
                range_layout(scope, &held),
            )
        };
        let Some(segments) = segments else {
            return self.reproject_view_prepare(view_id);
        };
        let (sections, turns) = {
            let view = policy.substrate.read();
            let mut sections: Vec<SectionId> = Vec::new();
            let mut turns: Vec<TurnKey> = Vec::new();
            for seg in &segments {
                match seg {
                    ProjectionSegment::Sealed(SealedKind::Section(s))
                        if view.section_tier_state(s.id).is_some() =>
                    {
                        sections.push(s.id)
                    }
                    ProjectionSegment::Sealed(SealedKind::Turn(rt, _)) => {
                        if let Some(key) = rt.key() {
                            turns.push(key);
                        }
                    }
                    _ => {}
                }
            }
            (sections, turns)
        };
        let project_ms = t_repro.elapsed().as_millis() as u64;
        tracing::debug!(
            target: "candle_conversation::scheduler::reproject",
            view = view_id.0,
            sections = sections.len(),
            "relayout: the knowledge base's range laid out again from the step's decision"
        );
        self.reproject_rebuild(
            view_id,
            Selected {
                sections,
                segments,
                turns,
                composition,
            },
            PrepareTimes {
                t_repro,
                probe_ms: 0,
                scan_ms: 0,
                project_ms,
                warm_us: 0,
                score_us: 0,
                observe_us: 0,
                probe_tokens: 0,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The range lays out front to top, each section its member, one this
    /// process does not hold a hole of its positions**: a 100-range holding
    /// a (30) at the top, the 20-hole b left, and c (10) in front of it — c
    /// unheld.
    #[test]
    fn the_range_lays_out_front_to_top_an_unheld_section_a_hole() {
        let mut scope = RangeScope::new(100);
        scope.enter(SectionKey(1), 30, false);
        scope.enter(SectionKey(2), 20, false);
        scope.enter(SectionKey(3), 10, false);
        scope.leave(SectionKey(2));
        let held = HashMap::from([(
            SectionKey(1),
            HeldSection {
                id: SectionId::new(41),
                tokens: 30,
            },
        )]);
        let got: Vec<(char, u64)> = range_layout(&scope, &held)
            .iter()
            .map(|s| match s {
                ProjectionSegment::Sealed(SealedKind::Section(r)) => ('s', r.id.raw() as u64),
                ProjectionSegment::Hole { positions } => ('h', *positions as u64),
                _ => ('?', 0),
            })
            .collect();
        assert_eq!(
            got,
            vec![('h', 40), ('h', 10), ('h', 20), ('s', 41)],
            "front 40, c unheld as its 10, the hole b left, a at the top"
        );
    }
}
