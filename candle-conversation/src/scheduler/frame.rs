//! An append-only conversation's frame: the sections its first turn lays onto
//! an empty slot in place of a projection (`ProjectionInputs::frame`).
//!
//! **Only a conversation with nothing else to place is framed.** A frame is
//! the system prompt and nothing more, so a first turn whose conversation
//! inherits a lineage — a calibration draft forked onto the reading it writes
//! questions about, a priming-chain link read behind the links before it — is
//! projected instead, and the projection places the lineage whole
//! (`project::ingest_self`). Framed, such a turn decoded with the system prompt
//! and its own request alone: every calibration draft on BlackWorld from
//! 2026-11-15 on wrote a file's questions knowing only its name —
//! `projection_assembler.rs` was asked about a `ProjectionError::CycleDetected`
//! it does not hold and never about its `glue_islands`.

use crate::projection::{ProjectionSegment, ResolvedSection, SealedKind, SectionId};

/// Whether a submit lays its frame down in place of projecting: an
/// append-only conversation (`disable_reprojection`) whose slot is still empty,
/// and which inherits no lineage the frame would leave out.
pub(super) fn frames(disable_reprojection: bool, slot_holds_blocks: bool, inherits: bool) -> bool {
    disable_reprojection && !slot_holds_blocks && !inherits
}

/// The sections to elevate and the segments to inject for `frame`, in frame
/// order — what a projection hands admission, with no history and no
/// selection: every section sealed, nothing generated.
pub(super) fn framed_segments(frame: &[SectionId]) -> (Vec<SectionId>, Vec<ProjectionSegment>) {
    let segments = frame
        .iter()
        .map(|&id| ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(id))))
        .collect();
    (frame.to_vec(), segments)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A frame is its sections, sealed, in order** — the same list the slot
    /// was once primed with, and nothing a projection would add.
    #[test]
    fn a_frame_is_its_sections_in_order() {
        let frame = [SectionId::new(7), SectionId::new(2), SectionId::new(9)];
        let (sections, segments) = framed_segments(&frame);
        assert_eq!(sections, frame);
        let ids: Vec<SectionId> = segments
            .iter()
            .map(|s| match s {
                ProjectionSegment::Sealed(SealedKind::Section(rs)) => rs.id,
                other => panic!("a frame segment is a sealed section, got {other:?}"),
            })
            .collect();
        assert_eq!(ids, frame);
    }

    /// **A conversation that inherits a lineage is projected, not framed**;
    /// so is one that reprojects, and one whose slot already holds its prompt.
    #[test]
    fn only_an_empty_append_only_slot_with_no_lineage_is_framed() {
        assert!(frames(true, false, false), "a reading's first turn");
        assert!(!frames(true, false, true), "a draft on its reading");
        assert!(!frames(true, true, false), "a later turn");
        assert!(!frames(false, false, false), "a dialogue");
    }

    /// **An empty frame injects nothing.**
    #[test]
    fn an_empty_frame_injects_nothing() {
        let (sections, segments) = framed_segments(&[]);
        assert!(sections.is_empty());
        assert!(segments.is_empty());
    }
}
