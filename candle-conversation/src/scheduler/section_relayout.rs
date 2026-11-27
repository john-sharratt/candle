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
use crate::files::{PlacedFile, SectionKey};
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

/// The sections of `scope` this process holds, from the range's front to its
/// top, as runs of one file's — what the projection event names as placed.
pub(super) fn placed_runs(
    scope: &RangeScope,
    held: &HashMap<SectionKey, HeldSection>,
) -> Vec<PlacedFile> {
    let mut out: Vec<PlacedFile> = Vec::new();
    for e in scope.layout().iter().filter_map(Place::section) {
        let Some(h) = held.get(&e.key) else {
            continue;
        };
        match out.last_mut() {
            Some(run) if run.file == h.file => run.chunks.push(h.lines),
            _ => out.push(PlacedFile {
                file: h.file,
                chunks: vec![h.lines],
            }),
        }
    }
    out
}

impl Scheduler {
    /// Lay `view_id`'s slot out again for the range its files set holds now:
    /// the rebuild a reprojection runs ([`Self::reproject_rebuild`]) over the
    /// slot's last segments with the range rewritten, its projection event
    /// naming the files the range now places. `None` when the slot already
    /// holds that layout, or the rebuild was declined — the decodes short of
    /// their weights — in which case it is asked again at the next drain.
    pub(super) fn relayout_view_prepare(
        &mut self,
        view_id: SequenceId,
    ) -> Result<Option<ReprojectInFlight>, ConversationError> {
        if !self.decode_floor_holds_after(0) {
            Self::queue_reprojection(&mut self.pending_relayouts, view_id);
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
        let Some((applied, mut composition)) = recorded else {
            return self.reproject_view_prepare(view_id);
        };
        let laid = {
            let read = policy.substrate.read();
            let scope = read
                .files_set_of(policy.target.timeline)
                .and_then(|s| s.scope.as_ref());
            scope.map(|scope| {
                let wanted: HashSet<SectionKey> = scope.sections().map(|e| e.key).collect();
                let held = held_sections(&read, &wanted);
                (
                    files_relayout(
                        policy.projection.schema(),
                        &applied,
                        range_layout(scope, &held),
                    ),
                    placed_runs(scope, &held),
                )
            })
        };
        let Some((segments, placed)) = laid else {
            // No range to lay out: nothing is owed.
            if let Some(state) = self.slot_projection_state.get_mut(&slot) {
                state.layout_owed = false;
            }
            return Ok(None);
        };
        let Some(segments) = segments else {
            return self.reproject_view_prepare(view_id);
        };
        composition.placed = placed;
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
        let rebuilt = self.reproject_rebuild(
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
        )?;
        // Declined, the layout is still owed; kept, the slot held it.
        let owed = self
            .slot_projection_state
            .get(&slot)
            .is_some_and(|s| s.layout_owed);
        if rebuilt.is_none() && owed {
            Self::queue_reprojection(&mut self.pending_relayouts, view_id);
        }
        Ok(rebuilt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::files::{FileKey, LineSpan};

    fn held(id: u32, tokens: usize, file: u64, first: u32, last: u32) -> HeldSection {
        HeldSection {
            id: SectionId::new(id),
            tokens,
            file: FileKey(file),
            lines: LineSpan { first, last },
        }
    }

    /// **The placed files run front to top, a file's sections side by side
    /// one run**: c and b of file 7 in front of a of file 8.
    #[test]
    fn the_placed_files_run_front_to_top() {
        let mut scope = RangeScope::new(100);
        scope.enter(SectionKey(1), 30, false);
        scope.enter(SectionKey(2), 20, false);
        scope.enter(SectionKey(3), 10, false);
        let held = HashMap::from([
            (SectionKey(1), held(41, 30, 8, 1, 50)),
            (SectionKey(2), held(42, 20, 7, 51, 90)),
            (SectionKey(3), held(43, 10, 7, 1, 50)),
        ]);
        let runs = placed_runs(&scope, &held);
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].file, FileKey(7));
        assert_eq!(
            runs[0].chunks,
            vec![
                LineSpan { first: 1, last: 50 },
                LineSpan {
                    first: 51,
                    last: 90
                }
            ]
        );
        assert_eq!(runs[1].file, FileKey(8));
    }

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
        let held = HashMap::from([(SectionKey(1), held(41, 30, 7, 1, 30))]);
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
