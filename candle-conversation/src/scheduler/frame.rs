//! An append-only conversation's frame, as admission lays it down: the
//! sections and turns its first turn puts onto an empty slot in place of a
//! projection (`ProjectionInputs::frame`, `conversation::frame`).
//!
//! The frame is worked out on the submitting thread; the wave thread only
//! turns it into what a projection would have handed admission — the
//! sections and turns to lift onto the card, and the segments to inject — so a
//! first turn pays the lift of what it lays down and no projection.

use crate::conversation::Frame;
use crate::projection::{prefix_sections, ProjectionSegment, SectionId, TurnKey};

/// Whether a submit lays its frame down in place of projecting: an
/// append-only conversation (`disable_reprojection`) whose slot is still
/// empty.
pub(super) fn frames(disable_reprojection: bool, slot_holds_blocks: bool) -> bool {
    disable_reprojection && !slot_holds_blocks
}

/// The sections and turns to lift and the segments to inject for `frame`, in
/// frame order: every section sealed and every range a hole, then every turn
/// of its lineage, sealed.
pub(super) fn framed_segments(
    frame: &Frame,
) -> (Vec<SectionId>, Vec<ProjectionSegment>, Vec<TurnKey>) {
    (
        prefix_sections(&frame.sections),
        frame.segments(),
        frame.turn_keys(),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::projection::{
        GroupId, LayerId, PrefixItem, ResolvedTurn, SealedKind, TimelineId, TurnId, TurnIndex,
    };

    /// **A frame is laid down whole**: its sections to lift and inject, then
    /// its lineage's turns, lifted and injected after them.
    #[test]
    fn a_frame_is_its_sections_then_its_turns() {
        let timeline = TimelineId::from_raw(10).unwrap();
        let reading = ResolvedTurn {
            id: TurnId {
                layer_id: LayerId::from_raw(2).unwrap(),
                group_id: GroupId::from_raw(5).unwrap(),
                index: TurnIndex(1),
            },
            timeline: Some(timeline),
        };
        let frame = Frame {
            sections: Arc::new(vec![
                PrefixItem::Section(SectionId::new(7)),
                PrefixItem::Hole(32),
                PrefixItem::Section(SectionId::new(2)),
            ]),
            lineage: vec![reading],
        };
        let (sections, segments, turns) = framed_segments(&frame);
        assert_eq!(
            sections,
            vec![SectionId::new(7), SectionId::new(2)],
            "a hole has nothing to lift"
        );
        assert_eq!(turns, vec![TurnKey::new(timeline, TurnIndex(1))]);
        assert_eq!(segments.len(), 4);
        assert!(matches!(
            &segments[1],
            ProjectionSegment::Hole { positions: 32 }
        ));
        assert!(matches!(
            &segments[3],
            ProjectionSegment::Sealed(SealedKind::Turn(t, _)) if *t == reading
        ));
    }

    /// **Only an empty append-only slot is framed**: a later turn appends to
    /// what the slot holds, and a reprojecting conversation projects.
    #[test]
    fn only_an_empty_append_only_slot_is_framed() {
        assert!(frames(true, false), "an append-only first turn");
        assert!(!frames(true, true), "a later turn");
        assert!(!frames(false, false), "a dialogue");
    }
}
