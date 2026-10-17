//! How long the device waits for cold experts.
//!
//! A live gate launch's worker that holds a cold expert's first row tile adds
//! the time it waited for the expert's publish to a mapped pair `{ns, count}`
//! (`kernel.cuh`, `MoeLive::cold_wait`): one sample per cold expert per layer,
//! from the first wait on it to its publish — what a pack read costs the layer
//! that needs it. The up and down launches find the expert already published,
//! so they carry no counter. Waits on different experts of one layer overlap,
//! so the sum is per expert, not per layer.
//!
//! The pair only grows; [`ColdWait::reset`] takes a baseline and
//! [`ColdWait::since_reset`] reports past it, so the device never has to be
//! drained to start an interval.

use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

use candle::Result;

use super::cpu_pool::rings::Mapped;

/// The mapped counter pair and its interval baseline.
pub(crate) struct ColdWait {
    words: Mapped,
    base_ns: AtomicU64,
    base_n: AtomicU64,
}

impl ColdWait {
    pub(crate) fn new() -> Result<Self> {
        Ok(Self {
            words: Mapped::new(16)?,
            base_ns: AtomicU64::new(0),
            base_n: AtomicU64::new(0),
        })
    }

    /// The pair's device address, for `MoeLive::cold_wait`.
    pub(crate) fn dev(&self) -> u64 {
        self.words.dev()
    }

    /// `(ns, count)` as the device has added them so far.
    fn read(&self) -> (u64, u64) {
        let p = self.words.host() as *const u64;
        // SAFETY: the 16 mapped bytes this struct owns; the device only adds.
        unsafe { (ptr::read_volatile(p), ptr::read_volatile(p.add(1))) }
    }

    /// Start a new interval at the counts as they stand.
    pub(crate) fn reset(&self) {
        let (ns, n) = self.read();
        self.base_ns.store(ns, Ordering::Relaxed);
        self.base_n.store(n, Ordering::Relaxed);
    }

    /// `(ns, count)` since the last [`Self::reset`].
    pub(crate) fn since_reset(&self) -> (u64, u64) {
        let (ns, n) = self.read();
        (
            ns.wrapping_sub(self.base_ns.load(Ordering::Relaxed)),
            n.wrapping_sub(self.base_n.load(Ordering::Relaxed)),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The interval starts at the reset: what the device added before it is
    /// not reported, what it adds after is.
    #[test]
    fn an_interval_reports_past_its_baseline() {
        let _ctx = candle::Device::new_cuda(0).unwrap();
        let w = ColdWait::new().unwrap();
        let p = w.words.host() as *mut u64;
        // SAFETY: the pair just allocated; standing in for the device's adds.
        unsafe {
            ptr::write_volatile(p, 5_000);
            ptr::write_volatile(p.add(1), 2);
        }
        assert_eq!(w.since_reset(), (5_000, 2));
        w.reset();
        assert_eq!(w.since_reset(), (0, 0));
        unsafe {
            ptr::write_volatile(p, 12_000);
            ptr::write_volatile(p.add(1), 3);
        }
        assert_eq!(w.since_reset(), (7_000, 1));
    }
}
