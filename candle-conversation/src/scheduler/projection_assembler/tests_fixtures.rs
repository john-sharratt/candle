//! Builders for the segment and timeline fixtures the assembler tests share.
//!
//! The helpers build timelines, sealed sections and turns, and generated runs. One
//! test checks that each helper produces the kind it names, so a drifting fixture
//! fails in the fixture rather than in a test that uses it.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::projection::{GroupId, LayerId, ResolvedSection, ResolvedTurn, TimelineId, TurnId};

    pub(in super::super) fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).expect("timeline id")
    }

    pub(in super::super) fn section_seg(id: u32) -> ProjectionSegment {
        ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(SectionId::new(
            id,
        ))))
    }

    pub(in super::super) fn turn_seg(layer: u32, group: u32, index: u32) -> ProjectionSegment {
        ProjectionSegment::Sealed(SealedKind::Turn(
            ResolvedTurn {
                id: TurnId {
                    layer_id: LayerId::for_test(layer),
                    group_id: GroupId::for_test(group),
                    index: TurnIndex(index),
                },
                timeline: Some(TimelineId::for_test(group as u64)),
            },
            crate::Role::Assistant,
        ))
    }

    pub(in super::super) fn generated_seg(
        name: &str,
        position: usize,
        tokens: &[u32],
    ) -> ProjectionSegment {
        ProjectionSegment::Generated {
            tokens: Arc::new(tokens.to_vec()),
            identity: crate::projection::GeneratedIdentity {
                name: name.to_string(),
                position,
            },
        }
    }

    #[test]
    fn fixture_helpers_produce_expected_kinds() {
        match section_seg(7) {
            ProjectionSegment::Sealed(SealedKind::Section(rs)) => {
                assert_eq!(rs.id, SectionId::new(7));
            }
            _ => panic!("expected Sealed(Section)"),
        }
        match turn_seg(1, 2, 3) {
            ProjectionSegment::Sealed(SealedKind::Turn(rt, part)) => {
                assert_eq!(rt.group(), GroupId::for_test(2));
                assert_eq!(rt.index(), TurnIndex(3));
                assert_eq!(part, crate::Role::Assistant);
            }
            _ => panic!("expected Sealed(Turn)"),
        }
        match generated_seg("x", 0, &[1, 2, 3]) {
            ProjectionSegment::Generated { tokens, identity } => {
                assert_eq!(*tokens, vec![1, 2, 3]);
                assert_eq!(identity.name, "x");
                assert_eq!(identity.position, 0);
            }
            _ => panic!("expected Generated"),
        }
    }
}
