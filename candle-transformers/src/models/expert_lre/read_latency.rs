//! The stager's per-read latency, by source.
//!
//! A cold expert's GEMM workers spin on its live entry until the stager
//! publishes it, and give it up after `SPIN_LIMIT_NS`, failing the forward. A
//! whole-expert read normally lands in about a millisecond, so a read that takes
//! a large fraction of that limit is the drive or the host stalling, not the
//! queue — and it is the thing that turns into a failed forward. Recording each
//! read's time per source makes a slow drive, or a pageable warm tier the host
//! has paged out, show up in the counters and the log instead of only as a
//! failure.

use std::time::Duration;

/// A read slower than this is reported: a tenth of the worker's spin limit,
/// and about a hundred times a whole-expert read on the dev-box drive.
#[cfg(any(feature = "cuda", test))]
pub(crate) const SLOW_READ_NS: u64 = 100_000_000;

/// Where a staged expert's bytes came from.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReadSource {
    /// A positioned read of the pack record on NVMe.
    Pack,
    /// A `memcpy` from a pageable warm slot — slow only when the host has paged
    /// it out.
    Paged,
}

#[cfg(feature = "cuda")]
impl ReadSource {
    pub(crate) fn name(self) -> &'static str {
        match self {
            ReadSource::Pack => "pack",
            ReadSource::Paged => "pageable warm slot",
        }
    }
}

/// One source's reads: how many, the slowest, and how many were slow.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadLatency {
    pub reads: usize,
    pub max_ns: u64,
    pub slow: usize,
}

impl ReadLatency {
    /// Count one read of `ns`; true when it was slow.
    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn record(&mut self, ns: u64) -> bool {
        self.reads += 1;
        self.max_ns = self.max_ns.max(ns);
        let slow = ns > SLOW_READ_NS;
        if slow {
            self.slow += 1;
        }
        slow
    }

    /// The slowest read, for a report.
    pub fn max(&self) -> Duration {
        Duration::from_nanos(self.max_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_read_over_the_threshold_is_slow_and_the_max_tracks_the_slowest() {
        let mut l = ReadLatency::default();
        assert!(!l.record(900_000));
        assert!(!l.record(SLOW_READ_NS));
        assert!(l.record(SLOW_READ_NS + 1));
        assert!(!l.record(2_000_000));
        assert_eq!(
            l,
            ReadLatency {
                reads: 4,
                max_ns: 100_000_001,
                slow: 1
            }
        );
        assert_eq!(l.max(), Duration::from_nanos(100_000_001));
    }
}
