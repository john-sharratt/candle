//! Wall-clock time as an acceptor records it (§5.9): Unix milliseconds.
//!
//! Lease terms are counted on [`super::Millis`], the process's monotonic
//! clock, which begins again at every restart. A restarted acceptor therefore
//! cannot tell from it how much of an accepted lease's term was left. The vote
//! records, beside an accepted lease, the wall-clock time at which this
//! acceptor counts it to end, and a restart re-arms the lease for what is left
//! of it by the wall clock ([`super::rearm`]). The engine is handed a reading
//! of this clock as it is handed one of `Millis`, so a simulation drives both.

use std::ops::Add;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct WallMillis(pub u64);

impl WallMillis {
    /// The system clock now. A clock set before 1970 reads as zero.
    pub fn now() -> WallMillis {
        let since = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        WallMillis(since.as_millis() as u64)
    }
}

impl Add<Duration> for WallMillis {
    type Output = WallMillis;
    fn add(self, d: Duration) -> WallMillis {
        WallMillis(self.0.saturating_add(d.as_millis() as u64))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_term_is_added_in_milliseconds() {
        assert_eq!(
            WallMillis(1_000) + Duration::from_secs(20 * 60),
            WallMillis(1_201_000)
        );
        assert_eq!(
            WallMillis(u64::MAX) + Duration::from_secs(1),
            WallMillis(u64::MAX)
        );
    }

    #[test]
    fn the_system_clock_reads_after_2026() {
        // 2026-01-01T00:00:00Z.
        assert!(WallMillis::now() > WallMillis(1_767_225_600_000));
    }
}
