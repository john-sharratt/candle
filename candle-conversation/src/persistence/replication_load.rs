//! Whether replication is busy applying to this substrate — what segment
//! maintenance defers to (`docs/substrate_lock_free.md` §9.5).
//!
//! Every record another member wrote is applied on the log's owner, and so is
//! every maintenance step. A maintenance step that started while a catch-up was
//! landing held the owner for up to 50 s on WhiteWorld (2026-11-01): the records
//! still arriving waited behind it until the member's 128 MiB of arrivals was
//! full and it stopped reading its sockets. So while replication says it is
//! busy, maintenance starts no op and steps aside between batches — for at most
//! [`MAX_DEFER`] at a stretch, since maintenance is what reclaims the disk and a
//! member that always has something arriving must still reclaim it.
//!
//! Replication says so itself ([`ReplicationLoad::note`]), once a second; any
//! thread reads it, without the owner.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// The longest maintenance defers to a busy replication at a stretch.
pub const MAX_DEFER: Duration = Duration::from_secs(10 * 60);

/// Whether replication is busy, and since when — shared by everything holding
/// the substrate.
#[derive(Clone, Debug, Default)]
pub struct ReplicationLoad(Arc<AtomicU64>);

impl ReplicationLoad {
    /// Replication is busy (`true`), or not, at `now_ms`. A busy stretch keeps
    /// the time it began; quiet ends it.
    pub fn note(&self, busy: bool, now_ms: u64) {
        match busy {
            // Only the first busy note of a stretch stamps it; `max(1)` keeps a
            // stretch starting at the epoch distinct from quiet.
            true => {
                let _ =
                    self.0
                        .compare_exchange(0, now_ms.max(1), Ordering::AcqRel, Ordering::Acquire);
            }
            false => self.0.store(0, Ordering::Release),
        }
    }

    /// [`Self::note`] at the wall clock's now.
    pub fn note_now(&self, busy: bool) {
        self.note(busy, now_ms());
    }

    /// Whether maintenance defers at `now_ms`: replication is busy, and has
    /// been for less than [`MAX_DEFER`].
    pub fn defers(&self, now_ms: u64) -> bool {
        match self.0.load(Ordering::Acquire) {
            0 => false,
            since => now_ms.saturating_sub(since) < MAX_DEFER.as_millis() as u64,
        }
    }

    /// [`Self::defers`] at the wall clock's now.
    pub fn defers_now(&self) -> bool {
        self.defers(now_ms())
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: u64 = 1_800_000_000_000;
    const CAP: u64 = MAX_DEFER.as_millis() as u64;

    /// **Busy defers maintenance, quiet lets it run**, and a clone is the same
    /// signal.
    #[test]
    fn busy_defers_and_quiet_releases() {
        let load = ReplicationLoad::default();
        assert!(!load.defers(T0), "quiet from the start");
        let seen = load.clone();
        load.note(true, T0);
        assert!(seen.defers(T0 + 1_000));
        load.note(false, T0 + 2_000);
        assert!(!seen.defers(T0 + 2_000));
    }

    /// **A busy stretch defers for at most [`MAX_DEFER`]**, counted from its
    /// first note — later notes of the same stretch do not push it back — and
    /// a quiet moment starts the count again.
    #[test]
    fn a_busy_stretch_defers_for_at_most_the_cap() {
        let load = ReplicationLoad::default();
        load.note(true, T0);
        load.note(true, T0 + CAP - 1);
        assert!(load.defers(T0 + CAP - 1));
        assert!(!load.defers(T0 + CAP), "reclaim runs past the cap");
        load.note(false, T0 + CAP + 1);
        load.note(true, T0 + CAP + 2);
        assert!(load.defers(T0 + CAP + 3), "a new stretch defers again");
    }
}
