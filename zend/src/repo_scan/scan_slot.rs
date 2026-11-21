//! The gate that bounds how many folder conversations a pool holds open at once,
//! so the pool stays within the card.
//!
//! The decision and the reservation happen under one lock, so no two workers can
//! read the same pre-allocation state and admit together. One conversation is
//! always admitted when none is open, because a pool whose only route to freeing
//! VRAM is finishing work it may not start would deadlock. A wait is bounded, so a
//! gate that never opens stalls the ingest no longer than the arena allocator's
//! own refusal would. The slot is a guard that gives its count back on drop, since
//! the process-wide gauge is never reset and a slot lost to a panic would hold the
//! gate closed for the life of the daemon.

use super::*;

/// Longest a worker will wait for VRAM before claiming its unit anyway.
///
/// A bound, not a timeout to rely on: without it a pathological state where the
/// pool never drops would stall the scan forever. Reaching it means the gate
/// failed to help, and the arena allocator's own refusal is the next line of
/// defence.
pub(super) const SCAN_POOL_WAIT_CAP: std::time::Duration = std::time::Duration::from_secs(20);

/// Live conversation count, readable by [`max_live_conversations`] for its
/// per-conversation estimate without threading the counter through.
pub(super) static SCAN_LIVE_CONVS: AtomicUsize = AtomicUsize::new(0);

/// Block until the KV pool has room for another open conversation, then COUNT
/// THIS ONE IN before releasing the gate.
///
/// `live` is the count of conversations this pool currently holds open. When it
/// is zero the wait is skipped unconditionally — one conversation must always be
/// able to proceed, or a pool whose only route to freeing VRAM is finishing the
/// work it is not allowed to start would deadlock.
///
/// The reservation happens under the SAME lock as the decision, and that is the
/// whole point. Deciding under the lock and incrementing after it re-opens the
/// race the gate exists to close: every waiting worker reads the same
/// pre-increment count, each concludes there is room for one more, and they all
/// proceed — a narrower replay of the herd that put exactly 24 conversations on
/// the card at once and produced `n_failed = 24` under every configuration.
/// Callers must therefore pair this with [`release_scan_slot`] on every exit
/// path, including failures.
pub(super) fn reserve_scan_slot(live: &AtomicUsize) {
    // Serialises the decision so workers cannot all read the same pre-allocation
    // state and admit together (see `max_live_conversations`).
    static GATE: Mutex<()> = Mutex::new(());
    let start = std::time::Instant::now();
    let mut logged = false;
    loop {
        let cap = {
            let _turn = GATE.lock().unwrap_or_else(|e| e.into_inner());
            let cap = max_live_conversations();
            let now = live.load(Ordering::Relaxed);
            // Always let one through: a pool whose only route to freeing VRAM is
            // finishing work it is not allowed to start would deadlock.
            match cap {
                Some(c) if now >= c && now > 0 => c,
                _ => {
                    take_scan_slot(live);
                    return;
                }
            }
        };
        if start.elapsed() >= SCAN_POOL_WAIT_CAP {
            // Waited long enough that stalling the whole ingest is the worse
            // outcome. Still taken under the gate, so the count stays exact.
            let _turn = GATE.lock().unwrap_or_else(|e| e.into_inner());
            take_scan_slot(live);
            return;
        }
        if !logged {
            logged = true;
            tracing::debug!(
                target: "zend::repo_scan",
                live_conversations = live.load(Ordering::Relaxed),
                max_live = cap,
                "scan pool: waiting for KV room before opening another directory",
            );
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
}

/// Count one conversation in, on both the pool-local and process-wide gauges.
/// Called only with the reservation gate held.
pub(super) fn take_scan_slot(live: &AtomicUsize) {
    live.fetch_add(1, Ordering::Relaxed);
    SCAN_LIVE_CONVS.fetch_add(1, Ordering::Relaxed);
}

/// Holds one admitted conversation's slot on both gauges, and gives it back on
/// drop.
///
/// A guard rather than a paired release call: `SCAN_LIVE_CONVS` is
/// process-global and never reset, so a slot lost to an unwind is lost for the
/// life of the daemon. One panic inside an ingest — where the failure paths
/// already tolerate a directory going wrong — would leave the gate permanently
/// believing a conversation is open, and enough of them throttle every later
/// pass toward a width of one.
pub(super) struct ScanSlot<'a> {
    pub(super) live: &'a AtomicUsize,
}

impl<'a> ScanSlot<'a> {
    /// Wait for room, then take a slot. Returns holding it, so no other worker
    /// can decide against this state.
    pub(super) fn reserve(live: &'a AtomicUsize) -> Self {
        reserve_scan_slot(live);
        Self { live }
    }
}

impl Drop for ScanSlot<'_> {
    fn drop(&mut self) {
        self.live.fetch_sub(1, Ordering::Relaxed);
        SCAN_LIVE_CONVS.fetch_sub(1, Ordering::Relaxed);
    }
}
