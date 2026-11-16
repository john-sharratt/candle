//! An append-only conversation's frame: the sections its first turn lays onto
//! an empty slot in place of a projection (`ProjectionInputs::frame`).

use crate::projection::{ProjectionSegment, ResolvedSection, SealedKind, SectionId};

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

    /// **An empty frame injects nothing.**
    #[test]
    fn an_empty_frame_injects_nothing() {
        let (sections, segments) = framed_segments(&[]);
        assert!(sections.is_empty());
        assert!(segments.is_empty());
    }
}
