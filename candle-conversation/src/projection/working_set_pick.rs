//! What a `working_set` group emits (`docs/zend_working_set.md` §4.4): the
//! target's working-set members in the group, whole, in the working set's own
//! order.
//!
//! Nothing is chosen here. Which conversations are members, and where each
//! sits, is decided when they enter and leave the working set — admission
//! enforces the budget, and the order is insertion order — so a projection
//! emits exactly what the set holds and two projections of an unchanged set
//! emit the same thing.

use super::ids::TimelineId;
use crate::summary_tree::SelectionOrigin;

/// The score a lock carries into the selection record — the level of a full
/// hit on the normalized band. It never ranks anything: a lock is emitted
/// whatever it scores.
pub const PINNED_SCORE: f32 = 1000.0;

/// How a member stands in the working set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Standing {
    /// Locked: served by the fast path, held until the task ends.
    Locked,
    /// Provenance, at this momentum.
    Provenance(f32),
}

/// The projection target's working-set members in one group, in the working
/// set's order.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkingSetMembers {
    pub members: Vec<(TimelineId, Standing)>,
}

/// One conversation a working-set group emits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pick {
    pub timeline: TimelineId,
    pub score: f32,
    pub origin: SelectionOrigin,
}

/// Every member, in order, with its selection record.
pub fn pick(members: &WorkingSetMembers) -> Vec<Pick> {
    members
        .members
        .iter()
        .map(|&(timeline, standing)| match standing {
            Standing::Locked => Pick {
                timeline,
                score: PINNED_SCORE,
                origin: SelectionOrigin::WorkingSetLock,
            },
            Standing::Provenance(momentum) => Pick {
                timeline,
                score: momentum,
                origin: SelectionOrigin::WorkingSetMomentum,
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    /// The members come back in the set's order — never re-sorted by score or
    /// standing — each lock at the pinned score, each provenance member at its
    /// momentum.
    #[test]
    fn members_emit_in_the_sets_order_with_their_standing() {
        let members = WorkingSetMembers {
            members: vec![
                (tl(5), Standing::Provenance(300.0)),
                (tl(2), Standing::Provenance(9_000.0)),
                (tl(9), Standing::Locked),
                (tl(4), Standing::Locked),
            ],
        };
        let picks = pick(&members);
        let order: Vec<u64> = picks.iter().map(|p| p.timeline.raw()).collect();
        assert_eq!(order, vec![5, 2, 9, 4]);
        assert_eq!(picks[1].score, 9_000.0);
        assert_eq!(picks[1].origin, SelectionOrigin::WorkingSetMomentum);
        assert_eq!(picks[2].score, PINNED_SCORE);
        assert_eq!(picks[2].origin, SelectionOrigin::WorkingSetLock);
    }

    #[test]
    fn nothing_in_the_set_emits_nothing() {
        assert!(pick(&WorkingSetMembers::default()).is_empty());
    }
}
