//! What a sequence's carried-state claim was spent on, by part: restoring a
//! parked state from its snapshot, building a fresh recurrent store, the PLE
//! history, the draft head's seeds.
//!
//! A claim runs inside an admission pass, between forwards, and the scheduler
//! charges its whole time to the admission's `claim` part. Which part of it costs
//! that is what making it faster needs, and only the model knows its parts, so
//! the model charges them here and the scheduler reads the totals before and
//! after the claim ([`ClaimSplit::now`], [`ClaimSplit::since`]).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// The parts of a carried-state claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimPart {
    /// The check that a parked sequence has a snapshot staged for it.
    Parked,
    /// Of `install`: building the store from the snapshot, without the locks
    /// around it.
    Build,
    /// Building the recurrent store from a snapshot held for the sequence.
    Install,
    /// A fresh recurrent store: its state-arena slots claimed and zeroed.
    Recurrent,
    /// The PLE history and its spare.
    Ple,
    /// The draft head's seed and its spare.
    Seeds,
}

/// Every part, in the order the totals read.
pub const CLAIM_PARTS: [ClaimPart; 6] = [
    ClaimPart::Parked,
    ClaimPart::Install,
    ClaimPart::Build,
    ClaimPart::Recurrent,
    ClaimPart::Ple,
    ClaimPart::Seeds,
];

static PART_NS: [AtomicU64; CLAIM_PARTS.len()] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

impl ClaimPart {
    fn index(self) -> usize {
        match self {
            ClaimPart::Parked => 0,
            ClaimPart::Install => 1,
            ClaimPart::Build => 2,
            ClaimPart::Recurrent => 3,
            ClaimPart::Ple => 4,
            ClaimPart::Seeds => 5,
        }
    }

    /// File `took` as this part of a claim.
    pub fn charge(self, took: Duration) {
        PART_NS[self.index()].fetch_add(took.as_nanos() as u64, Ordering::Relaxed);
    }
}

/// The claim parts' totals since the process started, in nanoseconds.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ClaimSplit(pub [u64; CLAIM_PARTS.len()]);

impl ClaimSplit {
    /// The totals now.
    pub fn now() -> Self {
        Self(std::array::from_fn(|i| PART_NS[i].load(Ordering::Relaxed)))
    }

    /// What each part took since `earlier`, a reading taken with [`Self::now`].
    pub fn since(earlier: Self) -> [Duration; CLAIM_PARTS.len()] {
        let now = Self::now();
        std::array::from_fn(|i| Duration::from_nanos(now.0[i].saturating_sub(earlier.0[i])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A charge shows in the difference between two readings, under its own
    /// part and no other.
    #[test]
    fn a_charge_reads_back_under_its_own_part() {
        let before = ClaimSplit::now();
        ClaimPart::Ple.charge(Duration::from_micros(1500));
        let took = ClaimSplit::since(before);
        // Other tests may charge parts concurrently; this one's charge is at
        // least what it filed, under its own index.
        assert!(took[4] >= Duration::from_micros(1500), "{took:?}");
        assert_eq!(CLAIM_PARTS.map(ClaimPart::index), [0, 1, 2, 3, 4, 5]);
    }
}
