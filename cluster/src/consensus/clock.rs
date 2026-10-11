//! Time as consensus counts it: milliseconds on a process's own monotonic
//! clock. Lease terms are counted on it (§5.9), never on the wall clock, which
//! another machine may disagree with and an operator may move. The engine is
//! driven with a reading of it, so a simulation can run thousands of rounds in
//! virtual time.

use std::ops::{Add, Sub};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Millis(pub u64);

impl Add<Duration> for Millis {
    type Output = Millis;
    fn add(self, d: Duration) -> Millis {
        Millis(self.0 + d.as_millis() as u64)
    }
}

impl Sub<Duration> for Millis {
    type Output = Millis;
    fn sub(self, d: Duration) -> Millis {
        Millis(self.0.saturating_sub(d.as_millis() as u64))
    }
}

impl Sub for Millis {
    type Output = Duration;
    fn sub(self, earlier: Millis) -> Duration {
        Duration::from_millis(self.0.saturating_sub(earlier.0))
    }
}
