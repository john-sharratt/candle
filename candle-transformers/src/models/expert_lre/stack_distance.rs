//! What a few more expert slots would buy: the misses that a zone that many
//! slots larger would have served, measured on the routing as it is served.
//!
//! An access's **LRU stack distance** is the number of distinct other experts
//! routed since that expert was last routed. Under least-recently-used
//! residency at a capacity of `c` slots the access hits exactly when its
//! distance is under `c` (Mattson et al., 1970), so one pass over the routing
//! gives the hit count at every capacity at once. The accesses whose distance
//! falls in `[c, c + window)` are the misses a zone `window` slots larger
//! would have caught — over `window`, what one more slot is worth, in misses.
//! The zone does not evict by recency alone, but its hit curve has the shape
//! of the routing's reuse, and the slope of that curve is what the gallery's
//! keep trade prices weight ground by (`candle-conversation`
//! `Scheduler::update_gallery_keep`). A uniform spread of the misses over the
//! experts not resident — the price before this — reads that slope as the
//! curve's average past the zone, which under skewed routing is several times
//! too flat.
//!
//! The distance is a count of last-access marks after the expert's own: a
//! Fenwick tree over access positions holds one mark per expert, at its last
//! access, so an access is `O(log n)`. Positions are renumbered densely when
//! they run out, which keeps the tree a few times the expert count.

/// Access positions the tree holds per expert before it renumbers.
const SPAN_PER_KEY: usize = 4;

/// The fewest positions the tree holds, so a small model renumbers rarely.
const MIN_SPAN: usize = 1024;

/// The window a marginal count is taken over: an eighth of the capacity it is
/// measured at, never under [`MIN_WINDOW`] slots. Wide enough that a second
/// of routing counts something at it, narrow enough to be the curve's slope
/// at the capacity rather than its average over a span of it.
const WINDOW_SHARE: usize = 8;

/// The narrowest window a marginal count is taken over.
const MIN_WINDOW: usize = 64;

/// The slots past `capacity` a marginal count is taken over.
pub(crate) fn marginal_window(capacity: usize) -> usize {
    (capacity / WINDOW_SHARE).max(MIN_WINDOW)
}

/// LRU stack distances over accesses to `keys` keys (`0..keys`).
pub(crate) struct StackDistance {
    /// Per key, the position of its last access; zero for never.
    last: Vec<u32>,
    /// A Fenwick tree over positions `1..tree.len()`: one mark at each key's
    /// last access.
    tree: Vec<i32>,
    /// The position the next access takes.
    next: u32,
}

impl StackDistance {
    pub(crate) fn new(keys: usize) -> Self {
        let span = (keys * SPAN_PER_KEY).max(MIN_SPAN);
        Self {
            last: vec![0; keys],
            tree: vec![0; span + 1],
            next: 1,
        }
    }

    /// Access `key`, answering its stack distance: the distinct other keys
    /// accessed since its last access. `None` on its first access, which
    /// misses at every capacity, and for a key outside `0..keys`.
    pub(crate) fn access(&mut self, key: usize) -> Option<usize> {
        if key >= self.last.len() {
            return None;
        }
        if self.next as usize >= self.tree.len() {
            self.renumber();
        }
        let at = self.last[key] as usize;
        let next = self.next as usize;
        let distance = (at != 0).then(|| {
            let after = self.prefix(next - 1) - self.prefix(at);
            self.add(at, -1);
            after as usize
        });
        self.add(next, 1);
        self.last[key] = self.next;
        self.next += 1;
        distance
    }

    /// Give every key accessed a dense position, in access order, and rebuild
    /// the tree over them — positions the accesses since have spread out.
    fn renumber(&mut self) {
        let mut order: Vec<(u32, usize)> = self
            .last
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p != 0)
            .map(|(k, &p)| (p, k))
            .collect();
        order.sort_unstable();
        self.tree.iter_mut().for_each(|v| *v = 0);
        for (i, &(_, key)) in order.iter().enumerate() {
            let pos = i + 1;
            self.last[key] = pos as u32;
            self.add(pos, 1);
        }
        self.next = order.len() as u32 + 1;
    }

    fn add(&mut self, mut pos: usize, v: i32) {
        while pos < self.tree.len() {
            self.tree[pos] += v;
            pos += pos.isolate_lowest_one();
        }
    }

    fn prefix(&self, mut pos: usize) -> i32 {
        let mut sum = 0;
        while pos > 0 {
            sum += self.tree[pos];
            pos -= pos.isolate_lowest_one();
        }
        sum
    }
}

/// The marginal count: of the accesses served, the ones a zone
/// [`marginal_window`] slots past its capacity would have caught.
pub(crate) struct MarginalSlots {
    stack: StackDistance,
}

impl MarginalSlots {
    pub(crate) fn new(keys: usize) -> Self {
        Self {
            stack: StackDistance::new(keys),
        }
    }

    /// Access `key` with the zone at `capacity` evictable slots: whether the
    /// access is a miss at `capacity` that a zone of `capacity +
    /// marginal_window(capacity)` would have served.
    pub(crate) fn caught(&mut self, key: usize, capacity: usize) -> bool {
        matches!(
            self.stack.access(key),
            Some(d) if d >= capacity && d < capacity + marginal_window(capacity)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The distance is the distinct other keys between two accesses — repeats
    /// of one key in between count once, and the key itself not at all.
    #[test]
    fn the_distance_counts_distinct_keys_since_the_last_access() {
        let mut s = StackDistance::new(8);
        let got: Vec<Option<usize>> = [0, 1, 2, 1, 1, 0, 0, 3, 2]
            .into_iter()
            .map(|k| s.access(k))
            .collect();
        assert_eq!(
            got,
            vec![
                None,    // 0 first
                None,    // 1 first
                None,    // 2 first
                Some(1), // 1: only 2 since
                Some(0), // 1 again at once
                Some(2), // 0: 1 and 2 since
                Some(0), // 0 again
                None,    // 3 first
                Some(3), // 2: 1, 0, 3 since
            ]
        );
    }

    /// A key past the range is no access, and a first access no distance.
    #[test]
    fn a_key_out_of_range_is_ignored() {
        let mut s = StackDistance::new(2);
        assert_eq!(s.access(2), None);
        assert_eq!(s.access(0), None);
        assert_eq!(s.access(5), None);
        assert_eq!(s.access(0), Some(0), "the stray key left no mark");
    }

    /// **Renumbering keeps every distance.** A cycle over 300 keys runs the
    /// 1024-position tree through several renumberings; each access past the
    /// first lap sees the other 299 keys since its last.
    #[test]
    fn renumbering_preserves_the_distances() {
        let keys = 300;
        let mut s = StackDistance::new(keys);
        for lap in 0..20 {
            for k in 0..keys {
                let d = s.access(k);
                if lap == 0 {
                    assert_eq!(d, None);
                } else {
                    assert_eq!(d, Some(keys - 1), "lap {lap} key {k}");
                }
            }
        }
    }

    /// The window is an eighth of the capacity, never under 64 slots.
    #[test]
    fn the_window_is_a_share_of_the_capacity() {
        assert_eq!(marginal_window(0), 64);
        assert_eq!(marginal_window(511), 64);
        assert_eq!(marginal_window(900), 112);
        assert_eq!(marginal_window(8_000), 1_000);
    }

    /// **The marginal count is the misses a slightly larger zone catches.** A
    /// cycle over 100 keys has distance 99: at a capacity of 99 slots or fewer
    /// down to 36 (99 − 64 + 1) every access past the first lap is caught; at
    /// 100 it is a hit, not a marginal miss; at 35 it is past the window.
    #[test]
    fn a_cycle_is_caught_only_by_a_window_that_reaches_it() {
        let caught_at = |capacity: usize| {
            let mut m = MarginalSlots::new(100);
            let mut n = 0;
            for _ in 0..3 {
                for k in 0..100 {
                    n += usize::from(m.caught(k, capacity));
                }
            }
            n
        };
        assert_eq!(caught_at(99), 200);
        assert_eq!(caught_at(36), 200);
        assert_eq!(caught_at(100), 0, "a hit at its capacity");
        assert_eq!(caught_at(35), 0, "past the window");
    }

    /// **Everything resident, nothing marginal.** Routing that fits the zone
    /// has every distance under the capacity, so one more slot catches
    /// nothing — the keep trade sees no weight pressure.
    #[test]
    fn routing_that_fits_the_zone_has_no_marginal_misses() {
        let mut m = MarginalSlots::new(64);
        let mut n = 0;
        for i in 0..10_000 {
            n += usize::from(m.caught((i * 7) % 64, 64));
        }
        assert_eq!(n, 0);
    }
}
