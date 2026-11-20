//! When the bytes of a slot may be overwritten, and when an entry may go to 0.
//!
//! Every routed-expert invocation gets a **ticket** — `seq + 1` of the forward
//! thread's invocation counter, so 0 means "never". Bucketize stores its ticket
//! into its row's **started word** before it reads any of the row's live
//! entries ([`StartedRows`]); the host threads record the highest ticket whose
//! routing-summary word they have seen ([`ReclaimClock::observe`]). A summary
//! word for ticket `T` is the last store of that invocation's bucketize, and the
//! compute stream is in order, so every kernel of every invocation with a ticket
//! below `T` has completed. A started word says the same of its own ticket — its
//! bucketize is running, so everything before it has finished — and reuse
//! ([`ReclaimClock::reclaimable`]) takes whichever bound is higher, so a slot
//! never waits on how far behind the GPU a host thread happens to be serving.
//!
//! The layer's GEMMs read their weights from bucketize's snapshot of the live
//! table, never from the live table itself (`moe_bucketize.cu`, phase 1b). So:
//!
//! - **Retargeting an entry to another address is always safe.** A kernel that
//!   snapshotted the old address keeps reading it; the old slot must survive
//!   until that kernel is done. The host retargets, issues a full fence, and
//!   reads the row's started word ([`ReclaimClock::retire_key`]): the slot may be
//!   reused once that ticket is below the observed one
//!   ([`ReclaimClock::reclaimable`]). An invocation of the row that the device
//!   had not begun when the word was read never reads the old address: its
//!   bucketize stores its own ticket and fences before it reads the entry, so
//!   either the host saw that ticket (and the slot waits for it) or the device
//!   saw the new entry — a store/load pair on each side, sequentially
//!   consistent.
//!
//!   **Keyed on what the device has begun, not on what the host has enqueued.**
//!   A key of the row's latest *enqueued* ticket is also safe, but it ties reuse
//!   to how far the forward thread runs ahead: recording a wave as a chain of
//!   graphs costs a tenth of launching it, so the forward thread enqueues a
//!   ticket on nearly every row before the GPU reaches them, and every eviction
//!   would wait a pass on the retire list. The started word is the invocation
//!   the GPU is actually inside, whatever is queued behind it. What is queued
//!   still matters to *which* victims are taken — a row about to run gives up
//!   its experts last — and is kept for that alone ([`ReclaimClock::upcoming`]).
//! - **An entry goes to 0 only while its row is quiet**: no invocation of it
//!   begun and unfinished (`reclaimable(retire_key(row))`). A cold expert's
//!   workers read the LIVE entry (that is how they learn it was staged), so a 0
//!   written under an in-flight invocation that had already classified the
//!   expert cold would never be undone. A row that was quiet at the check and
//!   begins an invocation before the store lands is harmless to the entry: that
//!   invocation's bucketize reads the old address into its snapshot or reads 0
//!   and classifies the expert cold, and the stager stages it again. The slot
//!   behind the old address is another matter — a snapshot may still read it —
//!   so it is overwritten only once the retire key read after the store is
//!   reclaimable; otherwise the entry is restored (`Stager::evict`).
//!
//! The stager has one wider window of its own (Rule R′, `stager.rs`): while it
//! holds an unpublished cold expert of invocation `T`, the GPU is inside `T`'s
//! gate launch, so nothing after `T` has started and everything before it is
//! done.

use super::started::StartedRows;
use std::sync::atomic::{fence, AtomicU64, Ordering};

/// Per-row started tickets, written by the device; the highest one a host
/// thread has observed; and the highest known complete-before bound.
pub(crate) struct ReclaimClock {
    started: StartedRows,
    /// Per row, the latest ticket the forward thread has enqueued — not a
    /// reclaim key, a victim preference: see [`Self::upcoming`].
    enqueued: Box<[AtomicU64]>,
    observed: AtomicU64,
    /// Every invocation below this has completed: the observed ticket, or the
    /// latest started one if higher (`reclaimable`).
    progress: AtomicU64,
}

impl ReclaimClock {
    pub(crate) fn new(started: StartedRows) -> Self {
        let rows = started.rows();
        Self {
            started,
            enqueued: (0..rows).map(|_| AtomicU64::new(0)).collect(),
            observed: AtomicU64::new(0),
            progress: AtomicU64::new(0),
        }
    }

    /// The forward thread has enqueued invocation `ticket` of `row`.
    pub(crate) fn enqueued(&self, row: usize, ticket: u64) {
        self.enqueued[row].fetch_max(ticket, Ordering::Release);
    }

    /// Whether `row` has an invocation enqueued that the device has not begun:
    /// the GPU is about to read the row's experts. Never a question of safety —
    /// an invocation not yet begun reads whatever entry it finds
    /// (`retire_key`) — but evicting one of them buys a miss within the pass,
    /// so victims come from other rows first (`pipeline`'s lazy victims).
    pub(crate) fn upcoming(&self, row: usize) -> bool {
        self.enqueued[row].load(Ordering::Acquire) > self.started.load(row)
    }

    /// The started words' device address, for bucketize.
    pub(crate) fn started_ptr(&self) -> u64 {
        self.started.dev_ptr()
    }

    /// A host thread has seen the summary word of `ticket`.
    pub(crate) fn observe(&self, ticket: u64) {
        self.observed.fetch_max(ticket, Ordering::SeqCst);
        self.progress.fetch_max(ticket, Ordering::SeqCst);
    }

    /// The highest ticket whose summary word a host thread has seen — where
    /// the host is, which is what a check that no bucketize is still running
    /// needs. Reuse asks [`Self::reclaimable`], which follows the device.
    pub(crate) fn observed(&self) -> u64 {
        self.observed.load(Ordering::SeqCst)
    }

    /// The ticket a slot of `row` must wait past, read after a full fence —
    /// call it after the retarget's stores.
    pub(crate) fn retire_key(&self, row: usize) -> u64 {
        fence(Ordering::SeqCst);
        self.started.load(row)
    }

    /// The latest invocation the device has begun, on any row — how far the
    /// GPU is, against the ticket a host thread is serving.
    pub(crate) fn latest_started(&self) -> u64 {
        self.started.latest()
    }

    /// The latest invocation of `row` the device has begun, 0 before any —
    /// where the device is on the row, for a host-side question that is not a
    /// reclaim key (a read-ahead listing is written only while its reader is
    /// still to come, `read_ahead::listing_readable`). No fence: a stale answer
    /// costs a listing nobody reads, never a slot.
    pub(crate) fn begun(&self, row: usize) -> u64 {
        self.started.load(row)
    }

    /// The ticket every reader of a mapped word the host has just rewritten
    /// must finish past, read after a full fence — call it after the stores.
    /// Any row's bucketize may read it (the read-ahead lists), so the key is
    /// the latest invocation begun on any row: one begun later stored its
    /// started word before its first read, so it reads the new value — the
    /// same store/load pairing as [`Self::retire_key`].
    pub(crate) fn readers_key(&self) -> u64 {
        fence(Ordering::SeqCst);
        self.started.latest()
    }

    /// Whether every invocation up to ticket `key` has completed: `key` is
    /// below a ticket whose summary word was seen, or below one the device has
    /// begun on any row — its bucketize runs after every earlier kernel on the
    /// in-order compute stream, exactly as the summary word it writes later
    /// does. The started words are read only when the known bound falls short.
    pub(crate) fn reclaimable(&self, key: u64) -> bool {
        if key == 0 || key < self.progress.load(Ordering::SeqCst) {
            return true;
        }
        let latest = self.started.latest();
        self.progress.fetch_max(latest, Ordering::SeqCst);
        key < latest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn clock(rows: usize) -> ReclaimClock {
        ReclaimClock::new(StartedRows::host(rows))
    }

    /// Whether `row` has no invocation in flight: a slot retired from it now
    /// is reusable at once.
    fn quiet(c: &ReclaimClock, row: usize) -> bool {
        c.reclaimable(c.retire_key(row))
    }

    /// A row the device never began is quiet; one it began is in flight until a
    /// later ticket's summary word is seen — its own word is not enough, since
    /// that is written by its bucketize, before its GEMMs run.
    #[test]
    fn a_row_is_quiet_once_a_later_ticket_is_observed() {
        let c = clock(4);
        assert!(quiet(&c, 2), "never begun");
        c.started.store(2, 5);
        assert!(!quiet(&c, 2), "begun, nothing observed");
        c.observe(5);
        assert!(
            !quiet(&c, 2),
            "its own summary word: its GEMMs may still run"
        );
        c.observe(6);
        assert!(quiet(&c, 2), "a later invocation has started");
        assert!(quiet(&c, 1), "other rows are untouched");
    }

    /// An invocation the forward thread has queued but the device has not begun
    /// holds nothing: its bucketize will read the retargeted entry. However far
    /// ahead the forward thread runs, a row is held only by the invocation the
    /// device is inside.
    #[test]
    fn a_queued_invocation_the_device_has_not_begun_holds_no_slot() {
        let c = clock(3);
        c.started.store(1, 3);
        c.observe(4);
        // Tickets up to 40 queued on every row; the device has begun none.
        for (row, ticket) in [(0, 38), (1, 40), (2, 39)] {
            c.enqueued(row, ticket);
        }
        assert!(c.upcoming(0) && c.upcoming(1) && c.upcoming(2));
        assert_eq!(c.retire_key(1), 3);
        assert!(c.reclaimable(c.retire_key(1)));
        assert!(quiet(&c, 0) && quiet(&c, 1) && quiet(&c, 2));
    }

    /// The device beginning a later invocation on any row completes every
    /// earlier one, whether or not a host thread has read a summary word since:
    /// the compute stream is in order. Reuse follows the GPU, not the host
    /// thread serving behind it — and `observed`, which only the summary words
    /// move, stays where the host is.
    #[test]
    fn a_later_start_on_any_row_completes_every_earlier_invocation() {
        let c = clock(4);
        c.started.store(1, 5);
        c.observe(5);
        assert!(!c.reclaimable(5), "its own summary word only");
        c.started.store(3, 23);
        assert!(c.reclaimable(5) && c.reclaimable(22));
        assert!(!c.reclaimable(23), "the invocation the device is inside");
        assert!(quiet(&c, 1));
        assert!(!quiet(&c, 3));
        assert_eq!(c.observed(), 5);
    }

    /// A row with an invocation enqueued that the device has not begun is
    /// upcoming, and stops being so once the device begins it. Upcoming says
    /// nothing about reuse: the row is still quiet.
    #[test]
    fn an_enqueued_row_the_device_has_not_begun_is_upcoming() {
        let c = clock(3);
        assert!(!c.upcoming(1), "nothing enqueued");
        c.enqueued(1, 7);
        assert!(c.upcoming(1));
        assert!(quiet(&c, 1), "not begun, so nothing holds its slots");
        c.started.store(1, 7);
        assert!(!c.upcoming(1), "begun");
        c.enqueued(1, 12);
        c.started.store(1, 9);
        assert!(c.upcoming(1), "a later invocation still waits");
        assert!(!c.upcoming(0) && !c.upcoming(2));
    }

    /// A rewritten word's readers are every invocation begun so far, on any
    /// row: the key waits for the latest of them, and none before any began.
    #[test]
    fn a_rewritten_words_readers_are_every_invocation_begun() {
        let c = clock(3);
        assert_eq!(c.readers_key(), 0);
        assert!(
            c.reclaimable(c.readers_key()),
            "nothing begun, nothing reads"
        );
        c.started.store(0, 4);
        c.started.store(2, 9);
        let key = c.readers_key();
        assert_eq!(key, 9);
        assert!(!c.reclaimable(key), "invocation 9 may be reading");
        c.started.store(1, 10);
        assert!(c.reclaimable(key), "a later start completes it");
    }

    /// The latest start is the highest word on any row, 0 before any.
    #[test]
    fn the_latest_start_is_the_highest_word_of_any_row() {
        let c = clock(4);
        assert_eq!(c.latest_started(), 0);
        c.started.store(2, 9);
        c.started.store(0, 12);
        c.started.store(3, 11);
        assert_eq!(c.latest_started(), 12);
    }

    /// Where the device is on a row is its started word: 0 before any, and
    /// unmoved by what the forward thread has merely enqueued.
    #[test]
    fn begun_is_the_rows_started_word() {
        let c = clock(3);
        assert_eq!(c.begun(1), 0);
        c.started.store(1, 7);
        c.enqueued(1, 12);
        assert_eq!(c.begun(1), 7, "enqueued is not begun");
        assert_eq!(c.begun(0), 0, "other rows are untouched");
    }

    /// Observation only moves forward, whichever thread reports first.
    #[test]
    fn observation_is_the_maximum_reported() {
        let c = clock(1);
        c.observe(9);
        c.observe(4);
        assert_eq!(c.observed(), 9);
    }

    /// The key is the row's started ticket at the time of the read, so a row
    /// the device begins again before the read holds the slot past that
    /// invocation.
    #[test]
    fn the_retire_key_follows_a_later_start() {
        let c = clock(3);
        c.started.store(1, 3);
        assert_eq!(c.retire_key(1), 3);
        c.started.store(1, 10);
        assert_eq!(c.retire_key(1), 10);
        c.observe(4);
        assert!(!c.reclaimable(10));
        assert!(c.reclaimable(3));
    }
}
