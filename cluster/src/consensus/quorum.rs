//! `max_members` and the quorum (§5.9).
//!
//! The group's size is persisted, not configured: a member that sees more
//! distinct ready members at once than the value writes a **raise**; an
//! operator lowers it with a **set**. The value is the latest set by stamp,
//! raised to the greatest raise stamped after it. Raises commute, and the last
//! set wins, so the value is the same in any arrival order. The quorum is
//! `⌊max_members / 2⌋ + 1`.

use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Change {
    Raise(u32),
    Set(u32),
}

/// `max_members`, over records stamped with `S` — the substrate's version
/// stamp.
#[derive(Debug, Clone)]
pub struct MaxMembers<S: Ord + Copy> {
    set: Option<(S, u32)>,
    /// Raises stamped after the latest set.
    raises: BTreeMap<S, u32>,
}

impl<S: Ord + Copy> Default for MaxMembers<S> {
    fn default() -> Self {
        Self {
            set: None,
            raises: BTreeMap::new(),
        }
    }
}

impl<S: Ord + Copy> MaxMembers<S> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn apply(&mut self, stamp: S, change: Change) {
        match change {
            Change::Raise(n) => {
                if self.set.is_none_or(|(at, _)| stamp > at) {
                    let held = self.raises.entry(stamp).or_insert(n);
                    *held = (*held).max(n);
                }
            }
            Change::Set(n) => {
                if self.set.is_none_or(|(at, _)| stamp > at) {
                    self.set = Some((stamp, n));
                    self.raises = self.raises.split_off(&stamp);
                    self.raises.remove(&stamp);
                }
            }
        }
    }

    /// The group's largest size: at least one.
    pub fn value(&self) -> u32 {
        let set = self.set.map_or(0, |(_, n)| n);
        let raised = self.raises.values().copied().max().unwrap_or(0);
        set.max(raised).max(1)
    }

    pub fn quorum(&self) -> usize {
        self.value() as usize / 2 + 1
    }

    /// The raise to write when `seen` distinct ready members are visible at
    /// once and the value is smaller.
    pub fn observe(&self, seen: u32) -> Option<Change> {
        (seen > self.value()).then_some(Change::Raise(seen))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply_all(changes: &[(u64, Change)]) -> MaxMembers<u64> {
        let mut m = MaxMembers::new();
        for (stamp, change) in changes {
            m.apply(*stamp, *change);
        }
        m
    }

    /// Raises never fall by themselves; a set lowers; a raise stamped after
    /// the set raises again — the same in every arrival order.
    #[test]
    fn raises_and_sets_resolve_the_same_in_any_order() {
        let changes = [
            (1, Change::Raise(3)),
            (2, Change::Raise(5)),
            (3, Change::Set(2)),
            (4, Change::Raise(4)),
            (5, Change::Raise(3)),
        ];
        let mut order: Vec<usize> = (0..changes.len()).collect();
        let mut seen = None;
        // Every permutation of five changes.
        permute(&mut order, 0, &mut |o| {
            let picked: Vec<(u64, Change)> = o.iter().map(|i| changes[*i]).collect();
            let v = apply_all(&picked).value();
            assert_eq!(*seen.get_or_insert(v), v, "{picked:?}");
        });
        assert_eq!(seen, Some(4));
    }

    fn permute(v: &mut Vec<usize>, k: usize, f: &mut impl FnMut(&[usize])) {
        if k == v.len() {
            f(v);
            return;
        }
        for i in k..v.len() {
            v.swap(k, i);
            permute(v, k + 1, f);
            v.swap(k, i);
        }
    }

    #[test]
    fn the_quorum_is_a_majority_of_the_largest_group() {
        let mut m = MaxMembers::<u64>::new();
        assert_eq!((m.value(), m.quorum()), (1, 1));
        assert_eq!(m.observe(3), Some(Change::Raise(3)));
        m.apply(1, Change::Raise(3));
        assert_eq!((m.value(), m.quorum()), (3, 2));
        assert_eq!(m.observe(2), None, "fewer members never lower it");
        m.apply(2, Change::Raise(5));
        assert_eq!(m.quorum(), 3);
        m.apply(3, Change::Set(2));
        assert_eq!((m.value(), m.quorum()), (2, 2));
        m.apply(4, Change::Raise(4));
        assert_eq!(m.quorum(), 3);
    }
}
