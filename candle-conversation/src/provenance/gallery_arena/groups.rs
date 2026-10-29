//! Which fold groups a resident page holds, or a scan weighs
//! (`docs/gallery_residency.md` §4).
//!
//! A signature's fold groups are its layer regions; a scan weighing one zero
//! adds exactly nothing for it, so neither its words nor its scan are paid
//! for. A set is a bitmask over at most eight groups, in group order.

/// A set of fold groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct GroupSet(u8);

impl GroupSet {
    /// The most groups a set can name.
    pub const MAX: usize = 8;

    /// Every one of `n` groups.
    pub fn all(n: usize) -> GroupSet {
        assert!(n <= Self::MAX, "{n} fold groups");
        GroupSet(((1u16 << n) - 1) as u8)
    }

    /// The groups of `n` that `weights` weighs: those not weighed zero. An
    /// empty `weights` weighs every group alike, as the tally reads it.
    pub fn weighed(weights: &[f32], n: usize) -> GroupSet {
        if weights.is_empty() {
            return GroupSet::all(n);
        }
        (0..n)
            .filter(|&g| weights.get(g).copied().unwrap_or(1.0) != 0.0)
            .fold(GroupSet::default(), |s, g| s.with(g))
    }

    /// This set and group `g`.
    pub fn with(self, g: usize) -> GroupSet {
        assert!(g < Self::MAX, "fold group {g}");
        GroupSet(self.0 | (1 << g))
    }

    pub fn union(self, other: GroupSet) -> GroupSet {
        GroupSet(self.0 | other.0)
    }

    pub fn contains(self, g: usize) -> bool {
        g < Self::MAX && self.0 & (1 << g) != 0
    }

    /// Whether every group of `other` is in this set.
    pub fn covers(self, other: GroupSet) -> bool {
        other.0 & !self.0 == 0
    }

    pub fn len(self) -> usize {
        self.0.count_ones() as usize
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Where group `g` sits among this set's groups — its place in a page that
    /// holds this set, packed in group order. `None` when the set lacks it.
    pub fn place(self, g: usize) -> Option<usize> {
        self.contains(g)
            .then(|| (self.0 & ((1u16 << g) - 1) as u8).count_ones() as usize)
    }

    /// The groups, ascending.
    pub fn iter(self) -> impl Iterator<Item = usize> {
        (0..Self::MAX).filter(move |&g| self.contains(g))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A set of the weighed groups: zero weights are left out, a missing
    /// weight weighs one, and no weights at all weigh every group.
    #[test]
    fn a_set_holds_the_groups_weighed() {
        let content = GroupSet::weighed(&[0.0, 1.0, 1.0], 3);
        assert_eq!(content.iter().collect::<Vec<_>>(), [1, 2]);
        assert_eq!(
            GroupSet::weighed(&[0.0, 0.0, 1.0], 3)
                .iter()
                .collect::<Vec<_>>(),
            [2]
        );
        assert_eq!(GroupSet::weighed(&[0.1, 1.0, 0.1], 3), GroupSet::all(3));
        assert_eq!(GroupSet::weighed(&[], 3), GroupSet::all(3));
        assert_eq!(
            GroupSet::weighed(&[0.0], 3).iter().collect::<Vec<_>>(),
            [1, 2]
        );
        assert_eq!(GroupSet::all(3).len(), 3);
    }

    /// A group's place is its rank among the set's groups; a set covers
    /// another that holds nothing it lacks.
    #[test]
    fn a_group_is_placed_by_its_rank_in_the_set() {
        let s = GroupSet::default().with(1).with(2);
        assert_eq!(s.place(0), None);
        assert_eq!(s.place(1), Some(0));
        assert_eq!(s.place(2), Some(1));
        assert_eq!(GroupSet::all(3).place(2), Some(2));
        assert!(GroupSet::all(3).covers(s));
        assert!(!s.covers(GroupSet::all(3)));
        assert!(s.covers(GroupSet::default().with(2)));
        assert_eq!(s.union(GroupSet::default().with(0)), GroupSet::all(3));
    }
}
