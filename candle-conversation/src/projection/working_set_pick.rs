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

/// The projection target's working-set members in one group, in the working
/// set's order, each with its momentum.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WorkingSetMembers {
    pub members: Vec<(TimelineId, f32)>,
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
        .map(|&(timeline, momentum)| Pick {
            timeline,
            score: momentum,
            origin: SelectionOrigin::WorkingSetMomentum,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    /// The members come back in the set's order — never re-sorted by score —
    /// each at its momentum.
    #[test]
    fn members_emit_in_the_sets_order_at_their_momentum() {
        let members = WorkingSetMembers {
            members: vec![(tl(5), 300.0), (tl(2), 9_000.0), (tl(9), 100.0)],
        };
        let picks = pick(&members);
        let order: Vec<u64> = picks.iter().map(|p| p.timeline.raw()).collect();
        assert_eq!(order, vec![5, 2, 9]);
        assert_eq!(picks[1].score, 9_000.0);
        assert!(picks
            .iter()
            .all(|p| p.origin == SelectionOrigin::WorkingSetMomentum));
    }

    #[test]
    fn nothing_in_the_set_emits_nothing() {
        assert!(pick(&WorkingSetMembers::default()).is_empty());
    }
}
