//! The residency ceiling: Belady's optimal hit rate over a recorded routing.
//!
//! An eviction policy can do no better than evicting, at each miss, the
//! resident expert whose next use is farthest away (Belady's MIN) — or not
//! keeping the missed expert at all when its own next use is farther still.
//! Replayed over the routing a run actually made, at the zone's capacity, that
//! gives the ceiling the measured hit rate is judged against: a policy far
//! below it has room to improve, one near it does not, and the misses left are
//! the link's to hide.
//!
//! The requests are the invocations' routed experts in order, one request per
//! distinct `(row, expert)` of an invocation. Capacity counts the evictable
//! slots — the permanently resident layers are neither requested nor counted.

use std::collections::{BTreeSet, HashMap};

/// One invocation's routed experts: a row and its distinct experts.
pub(crate) type Invocation = (u16, Vec<u16>);

/// Hits and misses of Belady's MIN over `trace` at `capacity` slots,
/// counting only invocations from index `from` on — the earlier ones warm the
/// cache, as the run's own zone was warm when they were measured.
pub(crate) fn belady(trace: &[Invocation], capacity: usize, from: usize) -> (usize, usize) {
    let key = |row: u16, e: u16| (row as u32) << 16 | e as u32;
    // The requests in order, each with the invocation it belongs to.
    let requests: Vec<(usize, u32)> = trace
        .iter()
        .enumerate()
        .flat_map(|(i, (row, experts))| experts.iter().map(move |&e| (i, key(*row, e))))
        .collect();
    // Next use of each request: the position of the next request of the same
    // expert, or `usize::MAX` for never.
    let mut next_use = vec![usize::MAX; requests.len()];
    let mut seen: HashMap<u32, usize> = HashMap::new();
    for (p, &(_, id)) in requests.iter().enumerate().rev() {
        if let Some(n) = seen.insert(id, p) {
            next_use[p] = n;
        }
    }
    // Residents by next use, farthest last, and each one's next use.
    let mut by_next: BTreeSet<(usize, u32)> = BTreeSet::new();
    let mut resident: HashMap<u32, usize> = HashMap::new();
    let (mut hits, mut misses) = (0usize, 0usize);
    for (p, &(i, id)) in requests.iter().enumerate() {
        let counted = i >= from;
        let nu = next_use[p];
        if let Some(old) = resident.get_mut(&id) {
            by_next.remove(&(*old, id));
            *old = nu;
            by_next.insert((nu, id));
            hits += usize::from(counted);
            continue;
        }
        misses += usize::from(counted);
        if capacity == 0 {
            continue;
        }
        if resident.len() == capacity {
            let &(far, victim) = by_next.last().expect("a full cache holds something");
            if nu >= far {
                // Used again no sooner than anything resident: not kept.
                continue;
            }
            by_next.remove(&(far, victim));
            resident.remove(&victim);
        }
        resident.insert(id, nu);
        by_next.insert((nu, id));
    }
    (hits, misses)
}

/// Hits and misses of least-recently-used eviction over the same requests
/// and terms as [`belady`] — the reference a recency policy is judged by:
/// a policy below it is worse than no policy at all, one between it and
/// [`belady`] is using what it knows.
pub(crate) fn lru(trace: &[Invocation], capacity: usize, from: usize) -> (usize, usize) {
    let key = |row: u16, e: u16| (row as u32) << 16 | e as u32;
    // Residents by last use, oldest first, and each one's last use.
    let mut by_last: BTreeSet<(usize, u32)> = BTreeSet::new();
    let mut resident: HashMap<u32, usize> = HashMap::new();
    let (mut hits, mut misses) = (0usize, 0usize);
    let mut p = 0usize;
    for (i, (row, experts)) in trace.iter().enumerate() {
        let counted = i >= from;
        for &e in experts {
            let id = key(*row, e);
            p += 1;
            if let Some(old) = resident.get_mut(&id) {
                by_last.remove(&(*old, id));
                *old = p;
                by_last.insert((p, id));
                hits += usize::from(counted);
                continue;
            }
            misses += usize::from(counted);
            if capacity == 0 {
                continue;
            }
            if resident.len() == capacity {
                let (_, victim) = by_last.pop_first().expect("a full cache holds something");
                resident.remove(&victim);
            }
            resident.insert(id, p);
            by_last.insert((p, id));
        }
    }
    (hits, misses)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inv(row: u16, experts: &[u16]) -> Invocation {
        (row, experts.to_vec())
    }

    /// Two slots, a b c a b: LRU evicts a for c, then b for a, then c for
    /// b — every request misses, where MIN hits twice.
    #[test]
    fn lru_thrashes_a_loop_one_longer_than_the_cache() {
        let t = [
            inv(0, &[1]),
            inv(0, &[2]),
            inv(0, &[3]),
            inv(0, &[1]),
            inv(0, &[2]),
        ];
        assert_eq!(lru(&t, 2, 0), (0, 5));
        assert_eq!(belady(&t, 2, 0), (2, 3));
        // a b a c a: a is always recent — two hits.
        let u = [
            inv(0, &[1]),
            inv(0, &[2]),
            inv(0, &[1]),
            inv(0, &[3]),
            inv(0, &[1]),
        ];
        assert_eq!(lru(&u, 2, 0), (2, 3));
        assert_eq!(lru(&u, 2, 3), (1, 1), "counted from the fourth invocation");
    }

    /// Two slots, requests a b c a b: MIN keeps a and b when c comes (c is
    /// never used again), so the second a and b hit.
    #[test]
    fn min_bypasses_an_expert_never_used_again() {
        let t = [
            inv(0, &[1]),
            inv(0, &[2]),
            inv(0, &[3]),
            inv(0, &[1]),
            inv(0, &[2]),
        ];
        assert_eq!(belady(&t, 2, 0), (2, 3));
    }

    /// One slot, a b a b b: 1 miss; 2 miss and not kept (1 is used sooner);
    /// 1 hit; 2 miss, evicting 1 (never used again); 2 hit.
    #[test]
    fn min_evicts_the_farthest_next_use() {
        let t = [
            inv(0, &[1]),
            inv(0, &[2]),
            inv(0, &[1]),
            inv(0, &[2]),
            inv(0, &[2]),
        ];
        assert_eq!(belady(&t, 1, 0), (2, 3));
    }

    /// Rows are part of the key, and counting starts at `from`.
    #[test]
    fn rows_are_distinct_experts_and_warmup_is_not_counted() {
        let t = [inv(0, &[1]), inv(1, &[1]), inv(0, &[1]), inv(1, &[1])];
        assert_eq!(belady(&t, 2, 0), (2, 2));
        assert_eq!(belady(&t, 2, 2), (2, 0), "the cold misses were warmup");
        assert_eq!(belady(&t, 1, 0), (1, 3));
    }

    /// An invocation's experts are requests in order: 1 2 3 | 3 | 1 at two
    /// slots keeps 1 and 3 and lets 2 go; no capacity misses every request.
    #[test]
    fn an_invocation_is_its_requests_in_order() {
        let t = [inv(0, &[1, 2, 3]), inv(0, &[3]), inv(0, &[1])];
        assert_eq!(belady(&t, 2, 0), (2, 3));
        assert_eq!(belady(&t, 0, 1), (0, 2));
    }
}
