//! When the bytes of a slot may be overwritten, and when an entry may go to 0.
//!
//! Every routed-expert invocation gets a **ticket** — `seq + 1` of the forward
//! thread's invocation counter, so 0 means "never". The forward thread records
//! each row's latest ticket *before* it enqueues the row's bucketize
//! ([`ReclaimClock::enqueue`]); the host threads record the highest ticket whose
//! routing-summary word they have seen ([`ReclaimClock::observe`]). A summary
//! word for ticket `T` is the last store of that invocation's bucketize, and the
//! compute stream is in order, so every kernel of every invocation with a ticket
//! below `T` has completed.
//!
//! The layer's GEMMs read their weights from bucketize's snapshot of the live
//! table, never from the live table itself (`moe_bucketize.cu`, phase 1b). So:
//!
//! - **Retargeting an entry to another address is always safe.** A kernel that
//!   snapshotted the old address keeps reading it; the old slot must survive
//!   until that kernel is done. The host retargets, issues a full fence, and
//!   reads the row's ticket ([`ReclaimClock::retire_key`]): the slot may be
//!   reused once that ticket is below the observed one
//!   ([`ReclaimClock::reclaimable`]). If the forward thread enqueued a new
//!   invocation of the row while the store was in flight, either its bucketize
//!   saw the new value or its ticket is in the key — the two fences make one of
//!   the two true (a store/load pair on each side, sequentially consistent).
//! - **An entry goes to 0 only while its row is quiet** ([`ReclaimClock::quiet`]):
//!   no invocation of it in flight. A cold expert's workers read the LIVE entry
//!   (that is how they learn it was staged), so a 0 written under an in-flight
//!   invocation that had already classified the expert cold would never be
//!   undone. A row that was quiet at the check and gains an invocation before the
//!   store lands is harmless: that invocation's bucketize reads the old address
//!   into its snapshot (the slot then waits on the retire key) or reads 0 and
//!   classifies the expert cold, and the stager stages it again.
//!
//! The stager has one wider window of its own (Rule R′, `stager.rs`): while it
//! holds an unpublished cold expert of invocation `T`, the GPU is inside `T`'s
//! gate launch, so nothing after `T` has started and everything before it is
//! done.

use std::sync::atomic::{fence, AtomicU64, Ordering};

/// Per-row last-enqueued tickets and the highest observed one.
pub(crate) struct ReclaimClock {
    last: Vec<AtomicU64>,
    observed: AtomicU64,
}

impl ReclaimClock {
    pub(crate) fn new(rows: usize) -> Self {
        Self {
            last: (0..rows).map(|_| AtomicU64::new(0)).collect(),
            observed: AtomicU64::new(0),
        }
    }

    /// The forward thread is about to enqueue invocation `ticket` of `row`.
    /// Called before the row's bucketize is enqueued; the fence orders the
    /// store before the launch and against the host threads' retarget stores.
    pub(crate) fn enqueue(&self, row: usize, ticket: u64) {
        self.last[row].store(ticket, Ordering::SeqCst);
        fence(Ordering::SeqCst);
    }

    /// A host thread has seen the summary word of `ticket`.
    pub(crate) fn observe(&self, ticket: u64) {
        self.observed.fetch_max(ticket, Ordering::SeqCst);
    }

    pub(crate) fn observed(&self) -> u64 {
        self.observed.load(Ordering::SeqCst)
    }

    /// The ticket a slot of `row` must wait past, read after a full fence —
    /// call it after the retarget's stores.
    pub(crate) fn retire_key(&self, row: usize) -> u64 {
        fence(Ordering::SeqCst);
        self.last[row].load(Ordering::SeqCst)
    }

    /// Whether every invocation up to ticket `key` has completed.
    pub(crate) fn reclaimable(&self, key: u64) -> bool {
        key == 0 || key < self.observed()
    }

    /// Whether `row` has no invocation in flight.
    pub(crate) fn quiet(&self, row: usize) -> bool {
        self.reclaimable(self.retire_key(row))
    }
}

/// Slots whose old tenant's readers may still be running, each held until the
/// ticket it waits past is below the observed one.
pub(crate) struct RetireList<T> {
    held: Vec<(u64, T)>,
}

impl<T> RetireList<T> {
    pub(crate) fn new() -> Self {
        Self { held: Vec::new() }
    }

    pub(crate) fn push(&mut self, key: u64, item: T) {
        self.held.push((key, item));
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.held.len()
    }

    /// Everything now reclaimable under `clock`, in the order it was retired.
    pub(crate) fn drain(&mut self, clock: &ReclaimClock) -> Vec<T> {
        let mut out = Vec::new();
        let mut kept = Vec::with_capacity(self.held.len());
        for (key, item) in self.held.drain(..) {
            if clock.reclaimable(key) {
                out.push(item);
            } else {
                kept.push((key, item));
            }
        }
        self.held = kept;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A row never enqueued is quiet; one enqueued is in flight until a later
    /// ticket's summary word is seen — its own word is not enough, since that
    /// is written by its bucketize, before its GEMMs run.
    #[test]
    fn a_row_is_quiet_once_a_later_ticket_is_observed() {
        let c = ReclaimClock::new(4);
        assert!(c.quiet(2), "never enqueued");
        c.enqueue(2, 5);
        assert!(!c.quiet(2), "enqueued, nothing observed");
        c.observe(5);
        assert!(!c.quiet(2), "its own summary word: its GEMMs may still run");
        c.observe(6);
        assert!(c.quiet(2), "a later invocation has started");
        assert!(c.quiet(1), "other rows are untouched");
    }

    /// Observation only moves forward, whichever thread reports first.
    #[test]
    fn observation_is_the_maximum_reported() {
        let c = ReclaimClock::new(1);
        c.observe(9);
        c.observe(4);
        assert_eq!(c.observed(), 9);
    }

    /// The key is the row's ticket at the time of the read, so a row enqueued
    /// again after a retarget holds the slot past its new invocation.
    #[test]
    fn the_retire_key_follows_a_later_enqueue() {
        let c = ReclaimClock::new(3);
        c.enqueue(1, 3);
        assert_eq!(c.retire_key(1), 3);
        c.enqueue(1, 10);
        assert_eq!(c.retire_key(1), 10);
        c.observe(4);
        assert!(!c.reclaimable(10));
        assert!(c.reclaimable(3));
    }

    /// The retire list hands back exactly the items whose key is below the
    /// observed ticket, in retirement order, and keeps the rest.
    #[test]
    fn the_retire_list_drains_what_is_behind_the_observed_ticket() {
        let c = ReclaimClock::new(1);
        let mut r = RetireList::new();
        r.push(7, 'a');
        r.push(3, 'b');
        r.push(12, 'c');
        r.push(0, 'd');
        c.observe(8);
        assert_eq!(r.drain(&c), vec!['a', 'b', 'd']);
        assert_eq!(r.len(), 1);
        c.observe(12);
        assert!(r.drain(&c).is_empty(), "12 is not below 12");
        c.observe(13);
        assert_eq!(r.drain(&c), vec!['c']);
        assert_eq!(r.len(), 0);
    }
}
