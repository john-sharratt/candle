//! An append-only conversation's frame: what its first turn lays onto an empty
//! slot in place of a projection (`ProjectionInputs::frame`).
//!
//! An append-only conversation (`disable_reprojection`) is never projected:
//! its first turn lays its frame onto the slot and every later turn appends to
//! what the slot holds. So the frame is the whole context the conversation's
//! first turn decodes in — the system prompt's fixed sections, then every turn
//! of its lineage, oldest first, exactly what a projection of it would place
//! (`project::ingest_self` selects a lineage whole and scores nothing). It is
//! worked out on the submitting thread from one substrate read; the wave
//! thread is handed the list.
//!
//! **The lineage is in the frame.** A calibration draft is forked onto the
//! reading it writes a file's questions about, and a priming-chain link is
//! read behind the links before it; a frame of the system prompt alone left
//! both without what they are parented on — every calibration draft on
//! BlackWorld from 2026-11-15 on wrote a file's questions knowing only its
//! name, `projection_assembler.rs` asked about a `ProjectionError` it does not
//! hold and never about its `glue_islands`.

use std::sync::Arc;

use crate::projection::{
    LayerId, ProjectionSegment, ProjectionTarget, ResolvedSection, ResolvedTurn, SealedKind,
    SectionId, TimelineId, TurnId, TurnIndex, TurnKey,
};
use crate::substrate::Substrate;
use crate::Role;

/// The sections and turns a first turn lays onto an empty slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Frame {
    /// The system prompt's fixed sections, each tree at its default branch.
    pub sections: Arc<Vec<SectionId>>,
    /// Every turn of the conversation's lineage that carries K/V, oldest first,
    /// the conversation's own last.
    pub lineage: Vec<ResolvedTurn>,
}

/// One timeline of a lineage: its id, the layer that wrote it, and each
/// turn's token count in turn order.
pub(crate) struct ChainLink<'a> {
    pub timeline: TimelineId,
    pub layer: LayerId,
    pub turn_tokens: &'a [usize],
}

impl Frame {
    /// The frame of `target` as `substrate` holds it: `sections`, then every
    /// turn of `target`'s inherited chain (`Substrate::inherited_chain`, its
    /// own timeline last).
    pub(crate) fn of(
        substrate: &Substrate,
        sections: Arc<Vec<SectionId>>,
        target: ProjectionTarget,
    ) -> Self {
        let chain: Vec<(TimelineId, LayerId, Vec<usize>)> = substrate
            .inherited_chain(target.timeline)
            .into_iter()
            .map(|tl| {
                let layer = substrate
                    .timeline_target(tl)
                    .map_or(target.layer, |(layer, _)| layer);
                let tokens = (0..substrate.turn_count(tl))
                    .map(|i| substrate.turn_token_count_of(tl, TurnIndex(i)))
                    .collect();
                (tl, layer, tokens)
            })
            .collect();
        let links: Vec<ChainLink<'_>> = chain
            .iter()
            .map(|(timeline, layer, tokens)| ChainLink {
                timeline: *timeline,
                layer: *layer,
                turn_tokens: tokens,
            })
            .collect();
        Self {
            sections,
            lineage: lineage_turns(&links, target),
        }
    }

    /// The frame as segments: each section, then each turn, sealed, in order.
    pub(crate) fn segments(&self) -> Vec<ProjectionSegment> {
        self.sections
            .iter()
            .map(|&id| ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(id))))
            .chain(
                self.lineage
                    .iter()
                    .map(|&t| ProjectionSegment::Sealed(SealedKind::Turn(t, Role::Assistant))),
            )
            .collect()
    }

    /// The turns to lift onto the card before the frame is laid down.
    pub(crate) fn turn_keys(&self) -> Vec<TurnKey> {
        self.lineage.iter().filter_map(ResolvedTurn::key).collect()
    }
}

/// Every turn of `chain` that carries K/V, in chain order, as `target`'s
/// group's turns — how a projection of an append-only target emits its
/// lineage. A turn of no tokens (a summary's anchor) holds nothing to lay down.
pub(crate) fn lineage_turns(
    chain: &[ChainLink<'_>],
    target: ProjectionTarget,
) -> Vec<ResolvedTurn> {
    chain
        .iter()
        .flat_map(|link| {
            link.turn_tokens
                .iter()
                .enumerate()
                .filter(|&(_, &tokens)| tokens > 0)
                .map(move |(i, _)| ResolvedTurn {
                    id: TurnId {
                        layer_id: link.layer,
                        group_id: target.group,
                        index: TurnIndex(i as u32),
                    },
                    timeline: Some(link.timeline),
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::projection::GroupId;

    fn tl(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    fn turn(layer: LayerId, group: GroupId, timeline: u64, index: u32) -> ResolvedTurn {
        ResolvedTurn {
            id: TurnId {
                layer_id: layer,
                group_id: group,
                index: TurnIndex(index),
            },
            timeline: Some(tl(timeline)),
        }
    }

    /// **A lineage is every turn of the chain that holds tokens, oldest
    /// first**, each in the target's group under the layer that wrote it; the
    /// target's own turns come last, and an empty one is left out.
    #[test]
    fn a_lineage_is_the_chains_turns_in_order() {
        let reading = LayerId::from_raw(2).unwrap();
        let draft = LayerId::from_raw(3).unwrap();
        let group = GroupId::from_raw(5).unwrap();
        let target = ProjectionTarget {
            layer: draft,
            group,
            timeline: tl(30),
        };
        let chain = [
            ChainLink {
                timeline: tl(10),
                layer: reading,
                turn_tokens: &[202, 0, 3_614],
            },
            ChainLink {
                timeline: tl(30),
                layer: draft,
                turn_tokens: &[],
            },
        ];
        assert_eq!(
            lineage_turns(&chain, target),
            vec![turn(reading, group, 10, 0), turn(reading, group, 10, 2)]
        );
    }

    /// **The frame lays its sections down, then its turns**, every one sealed,
    /// and lifts the turns it lays down.
    #[test]
    fn a_frame_is_its_sections_then_its_lineage() {
        let layer = LayerId::from_raw(2).unwrap();
        let group = GroupId::from_raw(5).unwrap();
        let frame = Frame {
            sections: Arc::new(vec![SectionId::new(7), SectionId::new(2)]),
            lineage: vec![turn(layer, group, 10, 0), turn(layer, group, 10, 1)],
        };
        let sealed: Vec<SealedKind> = frame
            .segments()
            .into_iter()
            .map(|s| match s {
                ProjectionSegment::Sealed(kind) => kind,
                other => panic!("a frame segment is sealed, got {other:?}"),
            })
            .collect();
        assert_eq!(
            sealed,
            vec![
                SealedKind::Section(ResolvedSection::new(SectionId::new(7))),
                SealedKind::Section(ResolvedSection::new(SectionId::new(2))),
                SealedKind::Turn(turn(layer, group, 10, 0), Role::Assistant),
                SealedKind::Turn(turn(layer, group, 10, 1), Role::Assistant),
            ]
        );
        assert_eq!(
            frame.turn_keys(),
            vec![
                TurnKey::new(tl(10), TurnIndex(0)),
                TurnKey::new(tl(10), TurnIndex(1))
            ]
        );
    }

    /// **A conversation with no lineage is framed by its sections alone.**
    #[test]
    fn no_lineage_frames_the_sections_alone() {
        let frame = Frame {
            sections: Arc::new(vec![SectionId::new(4)]),
            lineage: Vec::new(),
        };
        assert_eq!(frame.segments().len(), 1);
        assert!(frame.turn_keys().is_empty());
    }
}
