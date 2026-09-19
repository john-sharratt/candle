//! The expert warm tier: pinned slots first, pageable slots for the rest.
//!
//! Pinned host memory is the fast source — an upload is a direct DMA — but the
//! driver caps how much of the machine it will page-lock, and that cap is below
//! what the machine can spare. On the 31.5 GiB box the driver pinned 15.5 GiB
//! in one allocation when idle, and a warm pool of ~16 GiB left it unable to
//! lock the staging it needs for any later upload from pageable memory, so the
//! load died in the startup fill with `CUDA_ERROR_OUT_OF_MEMORY`. So the pinned
//! part stops at a page-lock ceiling with a margin for the driver, and whatever
//! the warm draw wants beyond it — or beyond what the driver grants — lives in
//! ordinary pageable memory.
//!
//! A pageable slot is never handed to the driver as an upload source: every
//! upload from one goes through the pinned cold-staging ring first
//! ([`WarmTier::is_pinned`] tells the caller which), exactly like a cold read,
//! minus the drive. The copy is a ~0.1 ms memcpy against a ~1 ms NVMe read.
//!
//! Slot `i` is membership entry `i`: slots `0..pinned` are pinned, the rest
//! pageable. Both are cut to the pack's stride and sector-aligned, so the
//! startup fill's direct reads land in either with nothing in between.

use super::page_pressure::PAGE_PRESSURE_BYTES;
use super::pinned::WarmPool;
use candle::direct_io::DIRECT_IO_SECTOR;
use candle::vram::PinnedUse;
use std::alloc::{alloc, dealloc, Layout};

/// Page-locked host memory left for the driver's own staging once the pinned
/// part is sized: an upload from pageable memory, the embedding shadow, the
/// stagers. The warm pool that left it none measured ~16 GiB of 31.5 GiB.
const DRIVER_LOCK_MARGIN: u64 = 1024 * 1024 * 1024;

/// Most of `total_ram` the pinned part may page-lock, less what is already
/// pinned: half the machine (the driver's page-lock ceiling measured on the
/// 31.5 GiB box) less [`DRIVER_LOCK_MARGIN`].
pub(crate) fn page_lock_ceiling(total_ram: u64, already_pinned: u64) -> u64 {
    (total_ram / 2)
        .saturating_sub(DRIVER_LOCK_MARGIN)
        .saturating_sub(already_pinned)
}

/// Sector-aligned pageable slots.
struct PagedSlots {
    base: *mut u8,
    layout: Option<Layout>,
    slot_size: usize,
    num_slots: usize,
}

impl PagedSlots {
    /// `num_slots` slots of `slot_size`, uninitialised — the startup fill
    /// writes every one before anything reads it. An allocation the machine
    /// refuses yields no slots rather than failing the load.
    fn new(num_slots: usize, slot_size: usize) -> Self {
        let bytes = num_slots.checked_mul(slot_size).unwrap_or(0);
        let layout = Layout::from_size_align(bytes, DIRECT_IO_SECTOR)
            .ok()
            .filter(|l| l.size() > 0);
        // SAFETY: `layout` has a non-zero size.
        let base = layout.map_or(std::ptr::null_mut(), |l| unsafe { alloc(l) });
        if base.is_null() {
            return Self {
                base: std::ptr::null_mut(),
                layout: None,
                slot_size,
                num_slots: 0,
            };
        }
        Self {
            base,
            layout,
            slot_size,
            num_slots,
        }
    }

    fn slot(&self, i: usize, len: usize) -> &[u8] {
        assert!(
            i < self.num_slots && len <= self.slot_size,
            "paged warm slot {i} out of range"
        );
        // SAFETY: bounds checked above; the allocation lives as long as `self`.
        unsafe { std::slice::from_raw_parts(self.base.add(i * self.slot_size), len) }
    }

    fn slot_mut(&mut self, i: usize, len: usize) -> &mut [u8] {
        assert!(
            i < self.num_slots && len <= self.slot_size,
            "paged warm slot {i} out of range"
        );
        // SAFETY: as `slot`, and `&mut self` makes it exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.base.add(i * self.slot_size), len) }
    }
}

impl Drop for PagedSlots {
    fn drop(&mut self) {
        if let Some(l) = self.layout {
            // SAFETY: `base` came from `alloc(l)`.
            unsafe { dealloc(self.base, l) };
        }
    }
}

// SAFETY: plain owned memory; after the startup fill it is only read.
unsafe impl Send for PagedSlots {}
unsafe impl Sync for PagedSlots {}

pub(crate) struct WarmTier {
    pinned: WarmPool,
    paged: PagedSlots,
}

impl WarmTier {
    /// A tier of `want_slots` stride-sized slots: as many pinned as the
    /// page-lock ceiling and the driver allow (`pinned_cap_slots`), the rest
    /// pageable.
    pub(crate) fn new(want_slots: usize, pinned_cap_slots: usize, slot_size: usize) -> Self {
        let pinned = WarmPool::new(
            want_slots.min(pinned_cap_slots),
            slot_size,
            PinnedUse::WeightWarmTier,
        );
        let paged = PagedSlots::new(want_slots.saturating_sub(pinned.num_slots()), slot_size);
        tracing::info!(
            target: "candle_transformers::expert_lre",
            pinned = pinned.num_slots(),
            paged = paged.num_slots,
            pinned_gib = (pinned.num_slots() * slot_size) as f64 / (1u64 << 30) as f64,
            paged_gib = (paged.num_slots * slot_size) as f64 / (1u64 << 30) as f64,
            pressure_gib = PAGE_PRESSURE_BYTES as f64 / (1u64 << 30) as f64,
            "warm tier: pinned part, then pageable for the rest"
        );
        Self { pinned, paged }
    }

    pub(crate) fn num_slots(&self) -> usize {
        self.pinned.num_slots() + self.paged.num_slots
    }

    pub(crate) fn paged_slots(&self) -> usize {
        self.paged.num_slots
    }

    /// Whether slot `i` is page-locked, and so a valid direct upload source.
    pub(crate) fn is_pinned(&self, i: usize) -> bool {
        i < self.pinned.num_slots()
    }

    pub(crate) fn slot_ref(&self, i: usize, len: usize) -> &[u8] {
        match i.checked_sub(self.pinned.num_slots()) {
            None => self.pinned.slot_ref(i, len),
            Some(p) => self.paged.slot(p, len),
        }
    }

    pub(crate) fn slot_mut(&mut self, i: usize, len: usize) -> &mut [u8] {
        match i.checked_sub(self.pinned.num_slots()) {
            None => self.pinned.slot_mut(i, len),
            Some(p) => self.paged.slot_mut(p, len),
        }
    }

    /// The first `n` slots as separate stride-long slices, for one batched
    /// fill across both parts.
    pub(crate) fn slots_mut(&mut self, n: usize, stride: usize) -> Vec<&mut [u8]> {
        let pinned_n = n.min(self.pinned.num_slots());
        let paged_n = n - pinned_n;
        assert!(
            paged_n <= self.paged.num_slots,
            "warm tier: {n} slots past the tier"
        );
        let mut out: Vec<&mut [u8]> = self
            .pinned
            .span_mut(0, pinned_n)
            .chunks_exact_mut(stride)
            .collect();
        if paged_n > 0 {
            // SAFETY: bounds checked above; one exclusive borrow of the block.
            let block = unsafe {
                std::slice::from_raw_parts_mut(self.paged.base, paged_n * self.paged.slot_size)
            };
            out.extend(block.chunks_exact_mut(stride));
        }
        out
    }

    /// Bytes held, both parts.
    pub(crate) fn total_bytes(&self) -> usize {
        self.pinned.total_bytes() + self.paged.num_slots * self.paged.slot_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;

    /// Half the machine less the driver's margin, less what is already pinned.
    #[test]
    fn the_page_lock_ceiling_is_half_less_the_margin() {
        let total = 32 * GIB;
        assert_eq!(page_lock_ceiling(total, 0), 15 * GIB);
        assert_eq!(page_lock_ceiling(total, GIB / 2), 15 * GIB - GIB / 2);
        assert_eq!(page_lock_ceiling(GIB, 0), 0);
    }

    /// Pageable slots are sector-aligned and independent.
    #[test]
    fn paged_slots_are_aligned_and_distinct() {
        let mut p = PagedSlots::new(3, 8192);
        assert_eq!(p.num_slots, 3);
        assert_eq!(p.base as usize % DIRECT_IO_SECTOR, 0);
        p.slot_mut(0, 8192).fill(1);
        p.slot_mut(2, 8192).fill(3);
        assert_eq!(p.slot(0, 8192)[8191], 1);
        assert_eq!(p.slot(2, 8192)[0], 3);
    }

    #[test]
    fn an_empty_paged_block_has_no_slots() {
        let p = PagedSlots::new(0, 8192);
        assert_eq!(p.num_slots, 0);
        assert!(p.layout.is_none());
    }
}
