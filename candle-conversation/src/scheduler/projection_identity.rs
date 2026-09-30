//! The content identity of a projection's segment list — what a slot's prefix
//! is built from.
//!
//! A mid-decode reprojection rebuilds its parent slot from scratch: truncate to
//! zero, re-inject every selected section and turn with its index pages, carve
//! a fresh view. Across most of a turn the selection does not move, and that
//! rebuild then reproduces the prefix the slot already holds — measured at
//! ~190 ms of a ~280 ms reprojection, every ~64 decoded tokens. Two segment
//! lists with the same identity assemble the same prefix, so a reprojection
//! whose identity matches the one its parent was last built from can leave the
//! parent and the view exactly as they are.
//!
//! The identity covers everything the assembly reads from a segment: a
//! section's id AND its stored content (a re-seal under the same id mints new
//! tokens, and must not match), a turn's conversation, index and half, and a
//! generated run's tokens. Positions are implied by order, which the hash
//! folds in.

use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::time::Instant;

use super::Scheduler;
use crate::handle::TurnEvent;
use crate::projection::event::ProjectionEvent;
use crate::projection::{PriorBelief, ProjectionSegment, SealedKind, SectionId};
use crate::sequence_handle::SequenceId;

impl Scheduler {
    /// Close a reprojection whose selection is the one `view_id`'s parent
    /// already holds: the prefix stays, the view keeps decoding into it, and
    /// only what a reprojection reports forward changes — the belief the next
    /// one seeds from, and the projection point the GUI and the seal read.
    pub(super) fn keep_unchanged_projection(
        &mut self,
        view_id: SequenceId,
        composition: ProjectionEvent,
    ) {
        let Some(ds) = self.active_decodes.get_mut(&view_id) else {
            return;
        };
        ds.belief = PriorBelief::from_selection(&composition.selection);
        let start_token = ds.generated_tokens.len() as u32;
        let event = ProjectionEvent {
            start_token,
            seconds: Instant::now().duration_since(ds.decode_start).as_secs_f64(),
            ..composition
        };
        let _ = ds.event_tx.send(TurnEvent::Projection(event));
        ds.last_projection_end = start_token;
    }
}

/// A section's content stamp for [`segments_identity`]: a hash of its sealed
/// tokens and their count.
///
/// The tokens themselves, not the address of the list holding them: a re-seal
/// frees the old list, and the new one of the same length can land at the same
/// address, so an address stamp would call an edited section unchanged and keep
/// decoding against its old K/V.
pub(super) fn section_content_stamp(tokens: &[u32]) -> (u64, usize) {
    let mut h = DefaultHasher::new();
    for &t in tokens {
        h.write_u32(t);
    }
    (h.finish(), tokens.len())
}

/// The identity of `segments`. `section_stamp` names a section's stored
/// content — see [`section_content_stamp`].
pub(super) fn segments_identity(
    segments: &[ProjectionSegment],
    section_stamp: impl Fn(SectionId) -> (u64, usize),
) -> u64 {
    let mut h = DefaultHasher::new();
    h.write_usize(segments.len());
    for seg in segments {
        match seg {
            ProjectionSegment::Sealed(SealedKind::Section(rs)) => {
                let (addr, len) = section_stamp(rs.id);
                h.write_u8(1);
                h.write_u32(rs.id.raw());
                h.write_u64(addr);
                h.write_usize(len);
            }
            ProjectionSegment::Sealed(SealedKind::Turn(rt, role)) => {
                h.write_u8(2);
                h.write_u64(rt.timeline.map_or(0, |t| t.raw()));
                h.write_u32(rt.index().0);
                h.write_u8(*role as u8);
            }
            ProjectionSegment::Sealed(SealedKind::TurnHalf(rt)) => {
                h.write_u8(3);
                h.write_u64(rt.timeline.map_or(0, |t| t.raw()));
                h.write_u32(rt.index().0);
            }
            ProjectionSegment::Generated { tokens, .. } => {
                h.write_u8(4);
                h.write_usize(tokens.len());
                for &t in tokens.iter() {
                    h.write_u32(t);
                }
            }
            ProjectionSegment::NewUserMessage { tokens } => {
                h.write_u8(5);
                h.write_usize(tokens.len());
                for &t in tokens.iter() {
                    h.write_u32(t);
                }
            }
        }
    }
    h.finish()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::segments_identity;
    use crate::projection::{
        GeneratedIdentity, GroupId, LayerId, ProjectionSegment, ResolvedSection, ResolvedTurn,
        SealedKind, SectionId, TimelineId, TurnId, TurnIndex,
    };
    use crate::turn::Role;

    fn section(id: u32) -> ProjectionSegment {
        ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection {
            id: SectionId::new(id),
        }))
    }

    fn turn(timeline: u64, index: u32, role: Role) -> ProjectionSegment {
        ProjectionSegment::Sealed(SealedKind::Turn(
            ResolvedTurn {
                id: TurnId {
                    layer_id: LayerId::for_test(1),
                    group_id: GroupId::for_test(1),
                    index: TurnIndex(index),
                },
                timeline: TimelineId::from_raw(timeline),
            },
            role,
        ))
    }

    fn glue(tokens: &[u32]) -> ProjectionSegment {
        ProjectionSegment::Generated {
            tokens: Arc::new(tokens.to_vec()),
            identity: GeneratedIdentity {
                name: "glue".into(),
                position: 0,
            },
        }
    }

    /// Every section stored at one fixed address.
    fn stamp(id: SectionId) -> (u64, usize) {
        (0x1000 + u64::from(id.raw()), 7)
    }

    fn prefix() -> Vec<ProjectionSegment> {
        vec![
            glue(&[1, 2]),
            section(3),
            turn(9, 4, Role::User),
            turn(9, 4, Role::Assistant),
        ]
    }

    #[test]
    fn the_same_list_has_the_same_identity() {
        assert_eq!(
            segments_identity(&prefix(), stamp),
            segments_identity(&prefix(), stamp)
        );
    }

    /// Each thing the assembly reads, changed alone, changes the identity.
    #[test]
    fn anything_the_assembly_reads_moves_the_identity() {
        let base = segments_identity(&prefix(), stamp);
        let mut variants: Vec<(&str, Vec<ProjectionSegment>)> = Vec::new();
        let mut v = prefix();
        v[2] = turn(9, 5, Role::User);
        variants.push(("another turn", v));
        let mut v = prefix();
        v[2] = turn(8, 4, Role::User);
        variants.push(("another conversation's turn", v));
        let mut v = prefix();
        v[3] = turn(9, 4, Role::User);
        variants.push(("the other half", v));
        let mut v = prefix();
        v[0] = glue(&[1, 3]);
        variants.push(("different glue", v));
        let mut v = prefix();
        v[1] = section(6);
        variants.push(("another section", v));
        let mut v = prefix();
        v.swap(2, 3);
        variants.push(("the same pieces reordered", v));
        let mut v = prefix();
        v.pop();
        variants.push(("a piece dropped", v));
        for (what, v) in variants {
            assert_ne!(segments_identity(&v, stamp), base, "{what}");
        }
    }

    /// A section edited without changing its length stamps differently; the
    /// same tokens in a fresh list stamp the same.
    #[test]
    fn the_section_stamp_is_its_content() {
        use super::section_content_stamp;
        let old = vec![5u32, 6, 7, 8];
        let edited = vec![5u32, 6, 9, 8];
        assert_ne!(section_content_stamp(&old), section_content_stamp(&edited));
        assert_eq!(
            section_content_stamp(&old),
            section_content_stamp(&old.clone())
        );
    }

    /// A section re-sealed under the same id holds new content, so it must not
    /// match the slot built from the old one.
    #[test]
    fn a_resealed_section_moves_the_identity() {
        let resealed = |id: SectionId| (0x9000 + u64::from(id.raw()), 7);
        assert_ne!(
            segments_identity(&prefix(), stamp),
            segments_identity(&prefix(), resealed)
        );
    }
}
