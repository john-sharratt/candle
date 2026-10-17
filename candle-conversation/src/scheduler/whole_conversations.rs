//! Another conversation enters a projection whole, or not at all.
//!
//! The elevate ahead of a rebuild is best-effort: a turn it could not bring
//! into VRAM is dropped by the assembly walk, alone, and the rest of its
//! conversation is placed around the gap. The model then reads a file's call
//! without its response, or the second half of an exchange whose first half is
//! missing — a context that does not make sense, presented as if it did.
//!
//! So once the elevate has run, a conversation any of whose projected turns is
//! not hot is withdrawn from this projection entirely. It is still selected; it
//! arrives whole on a later projection once its K/V is up. The admission caps
//! (`docs/projection_swap_cap.md`) bound how many conversations enter at once,
//! never how much of one.
//!
//! The slot's own timeline is exempt: its turns are the dialogue itself, and
//! withdrawing them would drop the conversation the model is continuing. Its
//! missing turns are reported where the walk drops them.

use std::collections::BTreeSet;

use super::Scheduler;
use crate::projection::{Conversation, ProjectionSegment, SealedKind, TimelineId, TurnIndex};

impl Scheduler {
    /// [`withdraw_partial_conversations`] against `conversation`'s tiers as
    /// they stand after the elevate — the same hot test the elevate's own
    /// lift-list uses. A turn with no tier state holds no K/V to lift (the
    /// elevate skips it by design), so it never makes its conversation
    /// partial: counting it as missing would withdraw that conversation from
    /// every projection, for good. Answers whether anything was withdrawn.
    pub(super) fn keep_conversations_whole(
        conversation: &Conversation,
        own: Option<TimelineId>,
        segments: &mut Vec<ProjectionSegment>,
        whence: &str,
    ) -> bool {
        let read = conversation.read();
        let withdrawn = withdraw_partial_conversations(segments, own, |tl, idx| {
            read.turn_tier_state(tl, idx).is_none_or(|t| t.hot)
        });
        if withdrawn.is_empty() {
            return false;
        }
        tracing::warn!(
            target: "candle_conversation::scheduler::reproject",
            whence,
            withdrawn = ?withdrawn.iter().map(|t| t.raw()).collect::<Vec<_>>(),
            "a selected conversation did not come up whole in VRAM; withdrawn from this \
             projection, to arrive whole on a later one"
        );
        true
    }
}

/// The turn a segment injects, when it injects one of a known conversation.
fn turn_of(seg: &ProjectionSegment) -> Option<(TimelineId, TurnIndex)> {
    match seg {
        ProjectionSegment::Sealed(SealedKind::Turn(rt, _))
        | ProjectionSegment::Sealed(SealedKind::TurnHalf(rt)) => {
            rt.timeline.map(|tl| (tl, rt.index()))
        }
        _ => None,
    }
}

/// Remove every turn of each conversation other than `own` of which any
/// projected turn is not `hot`, and answer the conversations removed.
pub(super) fn withdraw_partial_conversations(
    segments: &mut Vec<ProjectionSegment>,
    own: Option<TimelineId>,
    hot: impl Fn(TimelineId, TurnIndex) -> bool,
) -> Vec<TimelineId> {
    let partial: BTreeSet<TimelineId> = segments
        .iter()
        .filter_map(turn_of)
        .filter(|&(tl, idx)| Some(tl) != own && !hot(tl, idx))
        .map(|(tl, _)| tl)
        .collect();
    if !partial.is_empty() {
        segments.retain(|seg| !turn_of(seg).is_some_and(|(tl, _)| partial.contains(&tl)));
    }
    partial.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{GroupId, LayerId, ResolvedTurn, TurnId};
    use crate::Role;

    fn turn(tl: u64, idx: u32) -> ProjectionSegment {
        ProjectionSegment::Sealed(SealedKind::Turn(
            ResolvedTurn {
                id: TurnId {
                    layer_id: LayerId::for_test(1),
                    group_id: GroupId::for_test(1),
                    index: TurnIndex(idx),
                },
                timeline: Some(TimelineId::for_test(tl)),
            },
            Role::Assistant,
        ))
    }

    fn keys(segments: &[ProjectionSegment]) -> Vec<(u64, u32)> {
        segments
            .iter()
            .filter_map(turn_of)
            .map(|(tl, idx)| (tl.raw(), idx.0))
            .collect()
    }

    /// **One cold turn withdraws its whole conversation, and only that one.**
    /// File 2's turn 1 did not come up hot: none of file 2 is projected, and
    /// files 3 and the dialogue (7) are untouched.
    #[test]
    fn a_conversation_missing_one_turn_is_withdrawn_whole() {
        let mut segments = vec![
            turn(2, 0),
            turn(2, 1),
            turn(2, 2),
            turn(3, 0),
            turn(3, 1),
            turn(7, 0),
        ];
        let cold = [(2u64, 1u32)];
        let withdrawn = withdraw_partial_conversations(
            &mut segments,
            Some(TimelineId::for_test(7)),
            |tl, idx| !cold.contains(&(tl.raw(), idx.0)),
        );
        assert_eq!(withdrawn, vec![TimelineId::for_test(2)]);
        assert_eq!(keys(&segments), vec![(3, 0), (3, 1), (7, 0)]);
    }

    /// **The dialogue's own turns are never withdrawn**, hot or not: they are
    /// the conversation the model is continuing.
    #[test]
    fn the_slots_own_conversation_is_never_withdrawn() {
        let mut segments = vec![turn(7, 0), turn(7, 1), turn(3, 0)];
        let withdrawn = withdraw_partial_conversations(
            &mut segments,
            Some(TimelineId::for_test(7)),
            |tl, _| tl.raw() != 7,
        );
        assert!(withdrawn.is_empty());
        assert_eq!(keys(&segments), vec![(7, 0), (7, 1), (3, 0)]);
    }

    /// Everything hot: nothing moves.
    #[test]
    fn a_fully_hot_projection_is_left_alone() {
        let mut segments = vec![turn(2, 0), turn(3, 0)];
        let withdrawn = withdraw_partial_conversations(&mut segments, None, |_, _| true);
        assert!(withdrawn.is_empty());
        assert_eq!(keys(&segments), vec![(2, 0), (3, 0)]);
    }
}
