//! The pinned host ranges a live expert entry may point into, as a table the
//! bucketize kernel reads (`docs/elastic_warm_tier.md` §5.2).
//!
//! The kernel tells a pinned host copy of an expert from a VRAM one by its
//! address: inside one of these ranges is pinned. The ranges change while
//! kernels run — the warm tier gains and gives back segments — so they live in
//! mapped memory, and every launch copies the table into shared memory
//! (`load_pinned_ranges`, `moe_bucketize_live.cuh`).
//!
//! # The layout, and why a pair never moves
//!
//! `u64` words: the count of slots in use, then `[lo, hi)` pairs. A reader may
//! catch a pair mid-write, so every intermediate state must read as an
//! **empty** interval, never a wrong one — a torn pair spanning VRAM would
//! classify a VRAM expert as pinned, and the CPU pool would read device memory.
//! So:
//!
//! - an add writes `lo`, then `hi`, into a free slot (`(0, 0)`): halfway it is
//!   `(lo, 0)`, empty;
//! - a remove writes `hi = 0`, then `lo = 0`: halfway it is `(lo, 0)`, empty;
//! - a removed slot is reused by the next add, so the count is a high-water
//!   mark and never falls, and nothing is ever compacted.
//!
//! The host's own rules keep a stale copy safe as well: a range is added before
//! any entry points into it, and removed only once none does and no invocation
//! can still read one (`expert_lre::elastic`).

use std::sync::atomic::{fence, AtomicU64, Ordering};

/// The most slots a table has: the kernel copies this many at most.
pub const MAX_PINNED_RANGES: usize = 128;

/// Words a table occupies: the count, then the pairs.
pub const TABLE_WORDS: usize = 1 + 2 * MAX_PINNED_RANGES;

/// Every slot is in use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableFull;

/// The table's words, wherever they live.
pub struct RangeWords<'a>(&'a [AtomicU64]);

impl<'a> RangeWords<'a> {
    /// Over `words`, at least [`TABLE_WORDS`] long.
    pub fn new(words: &'a [AtomicU64]) -> Self {
        assert!(
            words.len() >= TABLE_WORDS,
            "a pinned range table is {TABLE_WORDS} words"
        );
        Self(words)
    }

    fn count(&self) -> usize {
        (self.0[0].load(Ordering::SeqCst) as usize).min(MAX_PINNED_RANGES)
    }

    fn pair(&self, i: usize) -> (u64, u64) {
        (
            self.0[1 + 2 * i].load(Ordering::SeqCst),
            self.0[2 + 2 * i].load(Ordering::SeqCst),
        )
    }

    /// Add `[lo, hi)`: into the first free slot, else a new one past the
    /// count. One writer at a time.
    pub fn add(&self, lo: u64, hi: u64) -> Result<(), TableFull> {
        assert!(lo < hi, "an empty pinned range [{lo:#x}, {hi:#x})");
        let n = self.count();
        let slot = (0..n).find(|&i| self.pair(i) == (0, 0)).unwrap_or(n);
        if slot >= MAX_PINNED_RANGES {
            return Err(TableFull);
        }
        self.0[1 + 2 * slot].store(lo, Ordering::SeqCst);
        self.0[2 + 2 * slot].store(hi, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        if slot == n {
            self.0[0].store((n + 1) as u64, Ordering::SeqCst);
        }
        Ok(())
    }

    /// Remove the range starting at `lo`; whether there was one. One writer at
    /// a time.
    pub fn remove(&self, lo: u64) -> bool {
        let Some(slot) = (0..self.count()).find(|&i| self.pair(i).0 == lo && lo != 0) else {
            return false;
        };
        self.0[2 + 2 * slot].store(0, Ordering::SeqCst);
        self.0[1 + 2 * slot].store(0, Ordering::SeqCst);
        fence(Ordering::SeqCst);
        true
    }

    /// Whether `addr` lies in a range.
    pub fn contains(&self, addr: u64) -> bool {
        (0..self.count()).any(|i| {
            let (lo, hi) = self.pair(i);
            addr >= lo && addr < hi
        })
    }

    /// The ranges in use, in slot order.
    pub fn ranges(&self) -> Vec<(u64, u64)> {
        (0..self.count())
            .map(|i| self.pair(i))
            .filter(|&(lo, hi)| lo < hi)
            .collect()
    }
}

#[cfg(feature = "cuda")]
mod mapped {
    use super::{RangeWords, TableFull, TABLE_WORDS};
    use crate::quantized::cuda::{alloc_host_mapped, HostMappedAlloc};
    use crate::Result;
    use std::sync::atomic::AtomicU64;
    use std::sync::Mutex;

    const _: () = assert!(
        super::MAX_PINNED_RANGES == candle_kernels::simple::moe_bucketize::MAX_PINNED_RANGES
    );

    /// A pinned range table in mapped host memory: what the host writes, and
    /// the device address the bucketize kernel reads it at.
    pub struct PinnedRangeTable {
        host: *const AtomicU64,
        device: u64,
        /// One writer at a time; readers take no lock.
        write: Mutex<()>,
        _alloc: HostMappedAlloc,
    }

    // SAFETY: the words are atomics in pinned memory that lives as long as
    // `_alloc`; writes are serialised by `write`.
    unsafe impl Send for PinnedRangeTable {}
    unsafe impl Sync for PinnedRangeTable {}

    impl PinnedRangeTable {
        /// An empty table.
        pub fn new() -> Result<Self> {
            let (host, device, alloc) = alloc_host_mapped(TABLE_WORDS * 8)?;
            let host = host as *const AtomicU64;
            let t = Self {
                host,
                device,
                write: Mutex::new(()),
                _alloc: alloc,
            };
            // `cuMemHostAlloc` does not zero, and a zero word is an unused one.
            for w in t.words().0 {
                w.store(0, std::sync::atomic::Ordering::SeqCst);
            }
            Ok(t)
        }

        fn words(&self) -> RangeWords<'_> {
            // SAFETY: `TABLE_WORDS` atomics, 8-aligned, alive with `_alloc`.
            RangeWords::new(unsafe { std::slice::from_raw_parts(self.host, TABLE_WORDS) })
        }

        /// The device address of the table.
        pub fn device_ptr(&self) -> u64 {
            self.device
        }

        pub fn add(&self, lo: u64, hi: u64) -> std::result::Result<(), TableFull> {
            let _w = self.write.lock().unwrap_or_else(|e| e.into_inner());
            self.words().add(lo, hi)
        }

        pub fn remove(&self, lo: u64) -> bool {
            let _w = self.write.lock().unwrap_or_else(|e| e.into_inner());
            self.words().remove(lo)
        }

        pub fn contains(&self, addr: u64) -> bool {
            self.words().contains(addr)
        }

        pub fn ranges(&self) -> Vec<(u64, u64)> {
            self.words().ranges()
        }
    }
}

#[cfg(feature = "cuda")]
pub use mapped::PinnedRangeTable;

#[cfg(test)]
mod tests {
    use super::*;

    fn words() -> Vec<AtomicU64> {
        (0..TABLE_WORDS).map(|_| AtomicU64::new(0)).collect()
    }

    /// **Raw words: a count, then pairs**; an add writes the next pair and
    /// bumps the count.
    #[test]
    fn an_add_writes_the_pair_then_the_count() {
        let w = words();
        let t = RangeWords::new(&w);
        t.add(0x7000, 0x8000).unwrap();
        t.add(0x9000, 0x9100).unwrap();
        let raw: Vec<u64> = w[..5].iter().map(|a| a.load(Ordering::SeqCst)).collect();
        assert_eq!(raw, vec![2, 0x7000, 0x8000, 0x9000, 0x9100]);
        assert!(t.contains(0x7fff) && t.contains(0x9000));
        assert!(!t.contains(0x8000) && !t.contains(0x6fff) && !t.contains(0x9100));
    }

    /// **A remove clears the pair in place and the count stays**; the next add
    /// reuses the slot, and no pair ever moves.
    #[test]
    fn a_remove_clears_in_place_and_an_add_reuses_the_slot() {
        let w = words();
        let t = RangeWords::new(&w);
        t.add(0x1000, 0x2000).unwrap();
        t.add(0x3000, 0x4000).unwrap();
        t.add(0x5000, 0x6000).unwrap();
        assert!(t.remove(0x3000));
        assert!(!t.remove(0x3000), "gone");
        let raw: Vec<u64> = w[..7].iter().map(|a| a.load(Ordering::SeqCst)).collect();
        assert_eq!(raw, vec![3, 0x1000, 0x2000, 0, 0, 0x5000, 0x6000]);
        assert!(!t.contains(0x3500));
        t.add(0xa000, 0xb000).unwrap();
        assert_eq!(w[3].load(Ordering::SeqCst), 0xa000, "into the freed slot");
        assert_eq!(
            w[0].load(Ordering::SeqCst),
            3,
            "the count stays a high-water mark"
        );
        assert_eq!(
            t.ranges(),
            vec![(0x1000, 0x2000), (0xa000, 0xb000), (0x5000, 0x6000)]
        );
    }

    /// **Every half-written pair is empty**: `(lo, 0)` contains nothing.
    #[test]
    fn a_half_written_pair_contains_nothing() {
        let w = words();
        w[0].store(1, Ordering::SeqCst);
        w[1].store(0x7000, Ordering::SeqCst);
        let t = RangeWords::new(&w);
        assert!(!t.contains(0x7000) && !t.contains(u64::MAX - 1) && !t.contains(0));
        assert!(t.ranges().is_empty());
    }

    /// A full table refuses; a freed slot takes the next add.
    #[test]
    fn a_full_table_refuses_until_a_slot_is_freed() {
        let w = words();
        let t = RangeWords::new(&w);
        for i in 0..MAX_PINNED_RANGES as u64 {
            t.add(0x1_0000 * (i + 1), 0x1_0000 * (i + 1) + 0x100)
                .unwrap();
        }
        assert_eq!(t.add(0xffff_0000, 0xffff_0100), Err(TableFull));
        assert!(t.remove(0x2_0000));
        t.add(0xffff_0000, 0xffff_0100).unwrap();
    }
}
