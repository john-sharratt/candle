//! The applied set: which `(origin, oseq)`s a member has applied or knows are
//! gone, per origin a version vector **with gaps** (`docs/replicated_substrate.md`
//! §6.2, §6.5).
//!
//! `through` says every `oseq` up to it is covered; the ranges above it are
//! covered too, with gaps between them. A record that overtook a lower one is
//! recorded exactly, and the lower one is still taken when it arrives — so
//! there is no reorder buffer, and nothing waits for anything.

use std::collections::BTreeMap;

/// One origin's coverage: everything through `through`, and the ranges above.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OriginApplied {
    through: u64,
    /// Inclusive ranges `start → end`, each starting above `through + 1`,
    /// none touching another.
    above: BTreeMap<u64, u64>,
}

impl OriginApplied {
    /// Every `oseq` up to this one is covered.
    pub fn through(&self) -> u64 {
        self.through
    }

    /// The highest covered `oseq`: the top of the last range above
    /// [`Self::through`], or `through` itself.
    pub fn top(&self) -> u64 {
        self.above
            .iter()
            .next_back()
            .map_or(self.through, |(_, end)| *end)
    }

    /// The covered ranges above [`Self::through`], in order.
    pub fn ranges_above(&self) -> impl Iterator<Item = (u64, u64)> + '_ {
        self.above.iter().map(|(s, e)| (*s, *e))
    }

    /// The uncovered ranges below [`Self::top`], in order, as inclusive
    /// `(start, end)`.
    pub fn gaps(&self) -> Vec<(u64, u64)> {
        let mut gaps = Vec::with_capacity(self.above.len());
        let mut next = self.through + 1;
        for (s, e) in self.ranges_above() {
            gaps.push((next, s - 1));
            next = e + 1;
        }
        gaps
    }

    pub fn contains(&self, oseq: u64) -> bool {
        oseq <= self.through
            || self
                .above
                .range(..=oseq)
                .next_back()
                .is_some_and(|(_, end)| oseq <= *end)
    }

    /// Cover `start..=end`, returning whether anything in it was not already
    /// covered.
    pub fn insert_range(&mut self, start: u64, end: u64) -> bool {
        if start > end || end <= self.through {
            return false;
        }
        let start = start.max(self.through + 1);
        let mut lo = start;
        let mut hi = end;
        let mut added = true;
        // Absorb every range that overlaps or touches `lo..=hi`.
        let touching: Vec<(u64, u64)> = self
            .above
            .range(..=hi.saturating_add(1))
            .filter(|(s, e)| **e + 1 >= lo && **s <= hi.saturating_add(1))
            .map(|(s, e)| (*s, *e))
            .collect();
        for (s, e) in touching {
            if s <= start && e >= end {
                added = false;
            }
            lo = lo.min(s);
            hi = hi.max(e);
            self.above.remove(&s);
        }
        if lo == self.through + 1 {
            self.through = hi;
            while let Some(e) = self.above.remove(&(self.through + 1)) {
                self.through = e;
            }
        } else {
            self.above.insert(lo, hi);
        }
        added
    }

    /// Cover one `oseq`, returning whether it was not already covered.
    pub fn insert(&mut self, oseq: u64) -> bool {
        self.insert_range(oseq, oseq)
    }

    /// One origin's coverage as `through` and its ranges above.
    pub fn from_parts(through: u64, above: impl IntoIterator<Item = (u64, u64)>) -> OriginApplied {
        let mut o = OriginApplied {
            through,
            above: BTreeMap::new(),
        };
        for (s, e) in above {
            o.insert_range(s, e);
        }
        o
    }

    /// One origin's coverage as its top and the gaps below it — the form a
    /// `HAVE` carries.
    pub fn from_top_and_gaps(
        top: u64,
        gaps: impl IntoIterator<Item = (u64, u64)>,
    ) -> OriginApplied {
        let mut o = OriginApplied::default();
        let mut next = 1;
        for (s, e) in gaps {
            if s > next {
                o.insert_range(next, s - 1);
            }
            next = e.saturating_add(1);
        }
        if top >= next {
            o.insert_range(next, top);
        }
        o
    }

    /// Whether this coverage holds an `oseq` `other` does not.
    pub fn holds_beyond(&self, other: &OriginApplied) -> bool {
        if self.top() == 0 {
            return false;
        }
        if !other.contains_range(1, self.through) {
            return true;
        }
        self.ranges_above()
            .any(|(s, e)| !other.contains_range(s, e))
    }

    /// Whether every `oseq` in `start..=end` is covered.
    pub fn contains_range(&self, start: u64, end: u64) -> bool {
        if start > end || end <= self.through {
            return true;
        }
        let start = start.max(self.through + 1);
        self.above
            .range(..=start)
            .next_back()
            .is_some_and(|(s, e)| *s <= start && *e >= end)
    }
}

/// Every origin's coverage.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AppliedSet {
    origins: BTreeMap<u64, OriginApplied>,
}

impl AppliedSet {
    /// Cover `(origin, oseq)`, returning whether it was not already covered.
    /// A record with no identity — origin or `oseq` zero — is never covered.
    pub fn insert(&mut self, origin: u64, oseq: u64) -> bool {
        if origin == 0 || oseq == 0 {
            return false;
        }
        self.origins.entry(origin).or_default().insert(oseq)
    }

    /// Cover `start..=end` of `origin` — `oseq`s a catch-up learned are gone.
    pub fn insert_range(&mut self, origin: u64, start: u64, end: u64) -> bool {
        if origin == 0 || start == 0 {
            return false;
        }
        self.origins
            .entry(origin)
            .or_default()
            .insert_range(start, end)
    }

    pub fn contains(&self, origin: u64, oseq: u64) -> bool {
        self.origins.get(&origin).is_some_and(|o| o.contains(oseq))
    }

    /// Every `oseq` of `origin` up to this one is covered.
    pub fn through(&self, origin: u64) -> u64 {
        self.origins.get(&origin).map_or(0, OriginApplied::through)
    }

    /// `origin`'s coverage, when this set holds any of it.
    pub fn origin(&self, origin: u64) -> Option<&OriginApplied> {
        self.origins.get(&origin)
    }

    /// Every origin's coverage, in origin order.
    pub fn origins(&self) -> impl Iterator<Item = (u64, &OriginApplied)> + '_ {
        self.origins.iter().map(|(o, a)| (*o, a))
    }

    /// Take in `other`'s coverage as well.
    pub fn merge(&mut self, other: &AppliedSet) {
        for (origin, theirs) in &other.origins {
            let mine = self.origins.entry(*origin).or_default();
            mine.insert_range(1, theirs.through);
            for (s, e) in theirs.ranges_above() {
                mine.insert_range(s, e);
            }
        }
    }

    /// Set `origin`'s coverage outright — what a persisted record says.
    pub fn set_origin(&mut self, origin: u64, applied: OriginApplied) {
        self.origins.insert(origin, applied);
    }

    /// The origins of which this set holds an `oseq` `other` does not, in
    /// origin order: what a member holding `other` catches up on from the
    /// member holding this one (§7.5).
    pub fn origins_beyond(&self, other: &AppliedSet) -> Vec<u64> {
        let empty = OriginApplied::default();
        self.origins
            .iter()
            .filter(|(origin, mine)| mine.holds_beyond(other.origins.get(origin).unwrap_or(&empty)))
            .map(|(origin, _)| *origin)
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.origins.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parts(o: &OriginApplied) -> (u64, Vec<(u64, u64)>) {
        (o.through(), o.ranges_above().collect())
    }

    /// An event that overtook a lower one is recorded exactly, and the lower
    /// one is still taken when it arrives, closing the gap.
    #[test]
    fn an_overtaking_event_leaves_a_gap_the_lower_one_closes() {
        let mut o = OriginApplied::default();
        assert!(o.insert(1));
        assert!(o.insert(2));
        assert!(o.insert(5));
        assert_eq!(parts(&o), (2, vec![(5, 5)]));
        assert!(!o.contains(3) && !o.contains(4) && o.contains(5));
        assert!(o.insert(4));
        assert_eq!(parts(&o), (2, vec![(4, 5)]));
        assert!(o.insert(3));
        assert_eq!(parts(&o), (5, vec![]));
        assert!(!o.insert(4), "already covered");
    }

    /// Ranges merge with every range they touch, and one that reaches
    /// `through + 1` advances it.
    #[test]
    fn ranges_merge_and_advance_through() {
        let mut o = OriginApplied::default();
        o.insert_range(10, 12);
        o.insert_range(20, 25);
        o.insert_range(14, 15);
        assert_eq!(parts(&o), (0, vec![(10, 12), (14, 15), (20, 25)]));
        assert!(o.insert_range(13, 19));
        assert_eq!(parts(&o), (0, vec![(10, 25)]));
        assert!(!o.insert_range(11, 24), "inside a covered range");
        assert!(o.insert_range(1, 9));
        assert_eq!(parts(&o), (25, vec![]));
        assert!(!o.insert_range(3, 7));
    }

    /// Unstamped records are never covered; origins are independent.
    #[test]
    fn origins_are_independent_and_unstamped_records_never_count() {
        let mut a = AppliedSet::default();
        assert!(!a.insert(0, 5));
        assert!(!a.insert(7, 0));
        assert!(a.is_empty());
        a.insert(7, 1);
        a.insert(9, 3);
        assert_eq!(a.through(7), 1);
        assert_eq!(a.through(9), 0);
        assert!(a.contains(9, 3) && !a.contains(7, 3));
    }

    /// Merging is a union, the same in either direction.
    #[test]
    fn merging_is_a_union() {
        let mut a = AppliedSet::default();
        a.insert_range(7, 1, 4);
        a.insert(7, 9);
        let mut b = AppliedSet::default();
        b.insert_range(7, 5, 6);
        b.insert(8, 2);
        let mut ab = a.clone();
        ab.merge(&b);
        let mut ba = b.clone();
        ba.merge(&a);
        assert_eq!(ab, ba);
        assert_eq!(ab.through(7), 6);
        assert!(ab.contains(7, 9) && !ab.contains(7, 8));
        assert!(ab.contains(8, 2) && !ab.contains(8, 1));
    }

    /// Top and gaps are the coverage seen from above, and round-trip.
    #[test]
    fn top_and_gaps_describe_the_same_coverage() {
        let o = OriginApplied::from_parts(3, [(6, 7), (10, 10)]);
        assert_eq!(o.top(), 10);
        assert_eq!(o.gaps(), vec![(4, 5), (8, 9)]);
        assert_eq!(OriginApplied::from_top_and_gaps(10, o.gaps()), o);
        let full = OriginApplied::from_parts(4, []);
        assert_eq!(full.gaps(), vec![]);
        assert_eq!(OriginApplied::from_top_and_gaps(4, []), full);
        assert_eq!(
            OriginApplied::from_top_and_gaps(0, []),
            OriginApplied::default()
        );
        let leading = OriginApplied::from_top_and_gaps(5, [(1, 2)]);
        assert_eq!(parts(&leading), (0, vec![(3, 5)]));
    }

    /// Comparing two sets names exactly the origins one holds more of.
    #[test]
    fn origins_beyond_names_what_the_other_lacks() {
        let mut restarted = AppliedSet::default();
        restarted.insert_range(1, 1, 10);
        restarted.insert_range(2, 1, 5);
        restarted.insert(3, 7);
        restarted.insert_range(4, 1, 2);
        let mut peer = AppliedSet::default();
        peer.insert_range(1, 1, 12);
        peer.insert_range(2, 1, 4);
        peer.insert_range(3, 1, 6);
        peer.insert_range(4, 1, 2);
        assert_eq!(restarted.origins_beyond(&peer), vec![2, 3]);
        assert_eq!(peer.origins_beyond(&restarted), vec![1, 3]);
        assert_eq!(peer.origins_beyond(&peer), Vec::<u64>::new());
    }
}
