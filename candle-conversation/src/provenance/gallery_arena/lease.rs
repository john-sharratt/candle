//! A scan's hold on a cached index, taken without a lock.
//!
//! A scan reads every page of every turn its index names, so none may be freed
//! or moved until its launch has synchronised. Pinning each turn under the
//! residency lock did that at a cost per turn: 72,582 pins and as many unpins a
//! scan, each unpin its own lock. The scan holds the index as one thing instead.
//!
//! Two atomics on each side, each written before the other is read:
//!
//! - a **scan** adds its lease, then reads the arena's change count;
//! - a **change** to a resident turn adds to the change count, then looks for a
//!   lease on any index naming the turn, and leaves that turn alone if it finds
//!   one.
//!
//! All four are sequentially consistent, so the two cannot miss each other. A
//! change that came first is seen by the scan, whose index no longer matches the
//! count, and it checks every turn under the residency lock — the same check as
//! ever, now paid only when something changed. A scan that came first is seen by
//! the change. A new upload touches no turn an index names and counts nothing.
//!
//! An index leaving the cache is the same handshake on [`ScanHold::retired`]: a
//! scan adds its lease and then reads the flag; the cache sets the flag and,
//! on a later publish, stops listing the index only if it reads no lease.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering::SeqCst};
use std::sync::Arc;

use super::scan::PagedIndex;

/// What scans and the arena's changes agree through, per index.
#[derive(Debug)]
pub(super) struct ScanHold {
    /// Scans reading the index now.
    leases: AtomicU32,
    /// The change count its turns were last checked at.
    verified: AtomicU64,
    /// It has left the cache: a scan leases nothing.
    retired: AtomicBool,
}

impl Default for ScanHold {
    /// Unchecked: the first reuse checks the turns.
    fn default() -> Self {
        Self {
            leases: AtomicU32::new(0),
            verified: AtomicU64::new(u64::MAX),
            retired: AtomicBool::new(false),
        }
    }
}

impl ScanHold {
    /// Add a lease, unless the index has been retired, in which case the
    /// lease is taken back and the caller rebuilds.
    fn enter(&self) -> bool {
        self.leases.fetch_add(1, SeqCst);
        if self.retired.load(SeqCst) {
            self.leave();
            return false;
        }
        true
    }

    fn leave(&self) {
        self.leases.fetch_sub(1, SeqCst);
    }

    /// Whether a scan reads the index now. A change reads this after adding
    /// to the change count.
    pub fn leased(&self) -> bool {
        self.leases.load(SeqCst) > 0
    }

    /// Whether the turns were checked at change count `changes`, read after
    /// the lease was added.
    pub fn current(&self, changes: &AtomicU64) -> bool {
        self.verified.load(SeqCst) == changes.load(SeqCst)
    }

    /// The turns were checked, under the residency lock, at count `changes`.
    pub fn checked(&self, changes: u64) {
        self.verified.store(changes, SeqCst);
    }

    /// Mark the index gone from the cache. Whether it was already marked: only
    /// then, with no lease, may the cache stop listing it.
    pub fn retire(&self) -> bool {
        self.retired.swap(true, SeqCst)
    }

    /// Whether the index has left the cache.
    pub fn retire_seen(&self) -> bool {
        self.retired.load(SeqCst)
    }
}

/// A scan's lease on an index: its turns stay put until it drops.
pub(super) struct IndexLease {
    idx: Arc<PagedIndex>,
}

impl IndexLease {
    /// Lease `idx`, or `None` when it has left the cache.
    pub fn take(idx: Arc<PagedIndex>) -> Option<Self> {
        idx.hold.enter().then_some(Self { idx })
    }

    pub fn index(&self) -> Arc<PagedIndex> {
        self.idx.clone()
    }
}

impl Drop for IndexLease {
    fn drop(&mut self) {
        self.idx.hold.leave();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn a_new_hold_is_unchecked() {
        let changes = AtomicU64::new(0);
        assert!(!ScanHold::default().current(&changes));
    }

    #[test]
    fn a_change_makes_a_checked_hold_stale() {
        let changes = AtomicU64::new(3);
        let hold = ScanHold::default();
        hold.checked(3);
        assert!(hold.current(&changes));
        changes.fetch_add(1, SeqCst);
        assert!(!hold.current(&changes));
    }

    #[test]
    fn leases_count_in_and_out() {
        let hold = ScanHold::default();
        assert!(!hold.leased());
        assert!(hold.enter());
        assert!(hold.enter());
        hold.leave();
        assert!(hold.leased());
        hold.leave();
        assert!(!hold.leased());
    }

    #[test]
    fn a_retired_index_takes_no_lease() {
        let hold = ScanHold::default();
        assert!(!hold.retire(), "the first retirement is news");
        assert!(hold.retire(), "the second says it was already marked");
        assert!(!hold.enter());
        assert!(!hold.leased(), "the refused lease was taken back");
    }

    /// **A scan and a change can never miss each other.** Raced many times: a
    /// scan enters and reads the count; a change bumps the count and reads the
    /// lease. The one outcome that frees a page under a launch — the change
    /// sees no lease AND the scan sees the count unchanged — never happens.
    #[test]
    fn a_scan_and_a_change_always_see_each_other() {
        for _ in 0..20_000 {
            let hold = Arc::new(ScanHold::default());
            let changes = Arc::new(AtomicU64::new(0));
            hold.checked(0);
            let start = Arc::new(Barrier::new(2));
            let scan = {
                let (hold, changes, start) = (hold.clone(), changes.clone(), start.clone());
                thread::spawn(move || {
                    start.wait();
                    assert!(hold.enter());
                    hold.current(&changes)
                })
            };
            let change = {
                let (hold, changes, start) = (hold.clone(), changes.clone(), start.clone());
                thread::spawn(move || {
                    start.wait();
                    changes.fetch_add(1, SeqCst);
                    hold.leased()
                })
            };
            let scan_saw_nothing = scan.join().unwrap();
            let change_saw_a_lease = change.join().unwrap();
            assert!(
                !scan_saw_nothing || change_saw_a_lease,
                "the change freed a turn a scan believed untouched"
            );
        }
    }

    /// The same for an index leaving the cache: it stops being listed only if
    /// no lease was seen, and then no scan may hold one.
    #[test]
    fn a_retirement_and_a_lease_always_see_each_other() {
        for _ in 0..20_000 {
            let hold = Arc::new(ScanHold::default());
            let start = Arc::new(Barrier::new(2));
            let scan = {
                let (hold, start) = (hold.clone(), start.clone());
                thread::spawn(move || {
                    start.wait();
                    hold.enter()
                })
            };
            let cache = {
                let (hold, start) = (hold.clone(), start.clone());
                thread::spawn(move || {
                    start.wait();
                    hold.retire();
                    hold.leased()
                })
            };
            let leased = scan.join().unwrap();
            let seen = cache.join().unwrap();
            assert!(!leased || seen, "an index was unlisted under a live lease");
        }
    }
}
