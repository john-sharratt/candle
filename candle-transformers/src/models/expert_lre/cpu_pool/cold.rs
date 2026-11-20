//! Where the pool finds a cold expert (`docs/cpu_expert_compute.md` §5.4).
//!
//! Bucketize hands the pool a cold expert with its snapshot entries 0: the
//! expert is on the pack, and the stager is reading it into a pad slot as it
//! would for the GPU (the summary keeps the cold bit). The pool waits on the
//! same thing a GPU worker would — the expert's live gate entry, which the
//! stager publishes last, after up and down — and computes from the pad
//! directly, so the expert never crosses the link.
//!
//! The pad slot stays put while the pool reads it: the invocation that routed
//! the expert is not done until its down launch has the pool's results, and a
//! slot is reused only once every invocation that could have read it is done
//! (`reclaim`). The stager's demand rule holds too — it evicts only slots the
//! current invocation does not route, and the device cannot reach a later
//! invocation while this one's down launch waits on the pool.
//!
//! **A pageable copy needs no wait.** An expert the warm tier holds in pageable
//! memory is cold to the device, which cannot read that memory, but not to the
//! pool, which can. Those copies are written once at startup and never move or
//! change ([`PagedImages`]), so the pool computes from one at once instead of
//! waiting for the stager's copy of it into the pad.
//!
//! **Giving up.** A wait ends without the expert when the abort word is raised
//! (the stager or the pipeline thread died), when another wait has already
//! claimed the fault word, or past the device's own spin limit. The pool then
//! claims the fault word naming the expert, as a GPU worker would, and
//! publishes no `done`: the down launch's wait on the pool sees the word and
//! gives up at once, and the host fails the forward.

use std::sync::Arc;
use std::time::Duration;

use super::backoff::{Backoff, LOW_POWER_CYCLES};
use crate::models::expert_lre::dispatch::{AbortWord, PinnedRanges};
use crate::models::expert_lre::fault::{ExpertFault, FaultWord};
use crate::models::expert_lre::live_table::{LiveTable, Proj};

/// What the pool reads a cold expert's publish and its give-up through.
pub(crate) struct ColdSource {
    pub(crate) table: Arc<LiveTable>,
    pub(crate) abort: Arc<AbortWord>,
    pub(crate) fault: Arc<FaultWord>,
    /// The pinned ranges the stager publishes a cold expert into — the warm
    /// tier's pinned part and the pad, both host memory the pool can read.
    pub(crate) pinned: PinnedRanges,
    /// The warm tier's pageable copies.
    pub(crate) paged: Arc<PagedImages>,
    pub(crate) spin_limit: Duration,
}

/// Where the warm tier's pageable copy of each expert starts, if it has one:
/// the slot image's first byte, which the live table's per-row projection
/// offsets apply to exactly as they do to a pad slot (the stager stages a
/// pageable copy into the pad byte for byte). Built once at startup; the
/// copies never move, and `owner` — the memory they lie in — lives as long as
/// this does, so every address here stays readable for as long as the pool
/// can read one.
pub(crate) struct PagedImages {
    bases: Vec<u64>,
    n_experts: usize,
    _owner: Arc<dyn Send + Sync>,
}

impl PagedImages {
    /// No pageable copy of any expert yet; the copies [`Self::set`] records
    /// lie in `owner`.
    pub(crate) fn new(rows: usize, n_experts: usize, owner: Arc<dyn Send + Sync>) -> Self {
        Self {
            bases: vec![0; rows * n_experts],
            n_experts,
            _owner: owner,
        }
    }

    /// Record `(row, expert)`'s pageable copy at `base`.
    pub(crate) fn set(&mut self, row: usize, expert: usize, base: u64) {
        self.bases[row * self.n_experts + expert] = base;
    }

    /// `(row, expert)`'s pageable copy, if it has one.
    pub(crate) fn base(&self, row: usize, expert: usize) -> Option<u64> {
        Some(self.bases[row * self.n_experts + expert]).filter(|&b| b != 0)
    }
}

/// A cold expert's publish, as the pool reads it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Published {
    /// Not yet.
    Pending,
    /// Its three projections' host addresses — gate, up, down — in the pad,
    /// the warm tier's pinned part, or its pageable copy.
    Host([u64; 3]),
    /// Published somewhere the pool cannot read — device memory.
    Unreadable,
}

impl ColdSource {
    /// `(row, expert)`'s host copy: its pageable one, which is always there,
    /// else its live entries — the gate entry first, which the stager stores
    /// last.
    pub(crate) fn published(&self, row: usize, expert: usize) -> Published {
        if let Some(base) = self.paged.base(row, expert) {
            return Published::Host([
                base + self.table.offset(Proj::Gate, row),
                base + self.table.offset(Proj::Up, row),
                base + self.table.offset(Proj::Down, row),
            ]);
        }
        let gate = self.table.entry(Proj::Gate, row, expert);
        if gate == 0 {
            return Published::Pending;
        }
        let up = self.table.entry(Proj::Up, row, expert);
        let down = self.table.entry(Proj::Down, row, expert);
        if up == 0 || down == 0 {
            // A clear in flight: gate goes first, so this read caught its tail.
            return Published::Pending;
        }
        let addrs = [gate, up, down];
        if addrs.iter().all(|&a| self.host_readable(a)) {
            Published::Host(addrs)
        } else {
            Published::Unreadable
        }
    }

    pub(crate) fn host_readable(&self, addr: u64) -> bool {
        self.pinned.iter().any(|&(lo, hi)| addr >= lo && addr < hi)
    }

    /// One wait while `(row, expert)` is unpublished (`backoff`): hot spins,
    /// then `umwait` on its gate entry's line — the stager's store wakes it at
    /// once, and until then the core idles in a light power state rather than
    /// spinning beside its hyperthread sibling. Bounded at ~10 µs, so the
    /// caller re-checks the expert's other waits and its give-up conditions.
    pub(crate) fn idle(&self, backoff: &mut Backoff, row: usize, expert: usize) {
        let p = self.table.entry_ptr(Proj::Gate, row, expert);
        // SAFETY: a live entry of the table this source holds.
        unsafe { backoff.wait_u64(p, 0, LOW_POWER_CYCLES) };
    }

    /// Whether a wait must end now: the abort word, another wait's fault, or
    /// `waited` past the spin limit.
    pub(crate) fn must_give_up(&self, waited: Duration) -> bool {
        self.abort.is_raised() || self.fault.is_set() || waited > self.spin_limit
    }

    /// Claim the fault word for `(row, expert)` after `waited`.
    pub(crate) fn give_up(&self, row: usize, expert: usize, waited: Duration) {
        self.fault.claim(ExpertFault {
            aborted: self.abort.is_raised(),
            row,
            expert,
            waited_us: waited.as_micros() as u64,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each `(row, expert)` names its own copy, and one never recorded has
    /// none.
    #[test]
    fn a_pageable_copy_is_per_expert_and_absent_until_set() {
        let mut p = PagedImages::new(3, 4, Arc::new(()));
        assert_eq!(p.base(2, 3), None);
        p.set(2, 3, 0x1_0000);
        p.set(0, 1, 0x2_0000);
        assert_eq!(p.base(2, 3), Some(0x1_0000));
        assert_eq!(p.base(0, 1), Some(0x2_0000));
        assert_eq!(p.base(1, 3), None);
        assert_eq!(p.base(2, 2), None);
    }
}
