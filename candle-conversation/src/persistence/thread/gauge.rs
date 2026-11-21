//! The drain gauges the scheduler reads to throttle admission, kept apart so
//! backlog the drain will clear is told from KV it cannot move.
//!
//! The warm deficit is work the drain will reach, which is a reason for
//! admission to slow. The pinned reading is hot KV the drain is skipping because
//! an in-flight decode is using it: it cannot be reclaimed while that holds, so
//! admission subtracts it outright instead of throttling against it. Each pass
//! stamps both halves, and a falling deficit wakes whoever is waiting on it,
//! while a rising or unchanged one does not, so the waiter is not woken for
//! nothing.

use super::*;

/// The persistence loop's drain gauges, shared with every
/// [`PersistenceTrigger`] handed out. Stamped each pass (pre- and post-drain);
/// read by the scheduler's admission throttle every wave.
///
/// The two halves answer different questions and must not be conflated. `warm`
/// is work the drain will get to — a deficit that admission should slow down
/// for. `pinned` is hot KV the drain is *skipping* because it belongs to an
/// in-flight decode's working set: it counts as evictable in the VRAM forecast
/// but cannot be reclaimed at any price, so admission subtracts it outright
/// rather than throttling against it.
#[derive(Clone, Default)]
pub struct BacklogGauge {
    pub(super) warm: Arc<AtomicU64>,
    pub(super) pinned: Arc<AtomicU64>,
    /// Residences holding a warm copy — the purge population, stamped at the
    /// purge gate each pass so readers never take the substrate lock.
    pub(super) warm_residents: Arc<AtomicU64>,
    /// Byte size of the purge population, stamped alongside `warm_residents`.
    pub(super) warm_resident_bytes: Arc<AtomicU64>,
    /// Host bytes of the per-turn payloads held whole outside the warm budget
    /// (`Substrate::held_payload_bytes`), stamped alongside `warm_residents`.
    pub(super) held_payload_bytes: Arc<AtomicU64>,
    /// Called whenever the drainable deficit falls — see
    /// [`PersistenceTrigger::on_backlog_fall`].
    pub(super) on_fall: Arc<OnceLock<BacklogWaker>>,
}

impl BacklogGauge {
    /// Record this pass's readings: the drainable hot→warm deficit and the
    /// pin-deferred bytes the drain skipped.
    pub(super) fn stamp(&self, warm: u64, pinned: u64) {
        self.stamp_warm(warm);
        self.pinned.store(pinned, Ordering::Relaxed);
    }

    /// Record only the drainable deficit, leaving the pinned reading from the
    /// last full split in place. Used at the points in a pass where the cheap
    /// `pending_warm_bytes` sum is refreshed but the split has not been recomputed.
    ///
    /// A fall wakes whoever gates on the figure: admission the backlog shut can
    /// reopen now, and with nothing else running that is the one change the
    /// scheduler could be waiting for.
    pub(super) fn stamp_warm(&self, warm: u64) {
        let before = self.warm.swap(warm, Ordering::Relaxed);
        if warm < before {
            if let Some(wake) = self.on_fall.get() {
                wake();
            }
        }
    }

    /// Record the warm-resident population (see `warm_residents`).
    pub(super) fn stamp_residents(&self, n: u64, bytes: u64, held_payload: u64) {
        self.warm_residents.store(n, Ordering::Relaxed);
        self.warm_resident_bytes.store(bytes, Ordering::Relaxed);
        self.held_payload_bytes
            .store(held_payload, Ordering::Relaxed);
    }
}
