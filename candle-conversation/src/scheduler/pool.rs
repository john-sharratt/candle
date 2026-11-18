//! Which pieces a dialogue's pool holds (`docs/hit_driven_projection.md` §7.2
//! step 1).
//!
//! The pool stands behind the slot: whole conversations — a file's reading, a
//! guide, the dialogue itself — whose index rows are resident at the sampled
//! layers, so the shadow scan can say which of their rounds the indexer would
//! have read. It holds the pieces the slot places, which hold its floors, and
//! then the working set's members in the order they entered: the working set
//! is what chooses them, by strength, kind by kind, within the card's
//! `PoolSize` (`crate::working_set::WorkingSet`).

use std::collections::HashSet;

use crate::projection::TimelineId;

/// The pool's pieces: every one of `floors`, then `members` in their order,
/// each piece once.
pub fn pool_order(floors: &[TimelineId], members: &[TimelineId]) -> Vec<TimelineId> {
    let mut seen: HashSet<TimelineId> = HashSet::new();
    floors
        .iter()
        .chain(members)
        .copied()
        .filter(|tl| seen.insert(*tl))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(raw: u64) -> TimelineId {
        TimelineId::from_raw(raw).unwrap()
    }

    /// The placed pieces first, then the members in their own order, a piece
    /// both placed and a member once.
    #[test]
    fn floors_then_members_each_once() {
        assert_eq!(
            pool_order(&[tl(1), tl(2)], &[tl(5), tl(2), tl(3)]),
            vec![tl(1), tl(2), tl(5), tl(3)]
        );
        assert_eq!(pool_order(&[], &[tl(3), tl(4)]), vec![tl(3), tl(4)]);
    }
}
