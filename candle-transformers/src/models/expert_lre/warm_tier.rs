//! The expert warm tier: pinned slots first, pageable slots for the rest.
//!
//! Pinned host memory is the fast source — an upload is a direct DMA — but the
//! driver caps how much of the machine it will page-lock, and the cap is the
//! driver's, not a fraction of RAM: on the 31.5 GiB box it pinned 15.5 GiB in
//! one allocation when idle, while on the 189 GiB box it pinned 119 GiB of
//! warm tier without complaint. A warm pool that took the whole cap on the
//! small box left the driver unable to lock the staging it needs for any later
//! upload from pageable memory, and the load died in the startup fill with
//! `CUDA_ERROR_OUT_OF_MEMORY`.
//!
//! So the pinned part takes as much of the draw as the driver grants and then
//! **proves** the driver still has [`DRIVER_LOCK_MARGIN`] to lock, giving
//! slots back a step at a time until it does. The ceiling is measured on the
//! machine it runs on rather than assumed from one; whatever the draw wants
//! beyond it lives in ordinary pageable memory.
//!
//! A pageable slot is never handed to the driver as an upload source: every
//! upload from one goes through the pinned cold-staging ring first
//! ([`WarmTier::is_pinned`] tells the caller which), exactly like a cold read,
//! minus the drive. That copy is not cheap — 0.59 ms for a 13.5 MiB
//! DeepSeek-V4-Flash expert (`bench_staging_one_pageable_slot`), twice the
//! PCIe 5.0 upload it gates, and more threads only slow it — which is why the
//! pinned part is as large as the driver allows.
//!
//! Slot `i` is membership entry `i`: slots `0..pinned` are pinned, the rest
//! pageable. Both are cut to the pack's stride and sector-aligned, so the
//! startup fill's direct reads land in either with nothing in between.

use super::page_pressure::PAGE_PRESSURE_BYTES;
use super::pinned::{step_down, WarmPool};
use candle::direct_io::DIRECT_IO_SECTOR;
use candle::vram::PinnedUse;
use std::alloc::{alloc, dealloc, Layout};

/// Page-locked host memory the driver must still be able to grant once the
/// pinned part is sized: an upload from pageable memory, the embedding shadow,
/// the stagers. The warm pool that left it none measured ~16 GiB of 31.5 GiB.
///
/// Half a GiB also loads (15.16 GiB pinned) and runs no faster — every rate
/// within run-to-run noise of 1 GiB — so the wider margin stays: it is the
/// driver's room, and running it short fails the load rather than slowing it.
const DRIVER_LOCK_MARGIN: usize = 1024 * 1024 * 1024;

/// GPU-addressable host memory (WDDM's NON_LOCAL budget,
/// [`candle::vram::gpu_addressable_room`]) the pinned part leaves unspent.
///
/// The driver charges every page-locked byte against that budget and keeps
/// granting locks until it is gone, so [`DRIVER_LOCK_MARGIN`] — a lock the
/// driver still grants — says nothing about whether the device can still
/// *address* one: the warm tier pinned 16.02 GiB of a 17.22 GiB budget on the
/// 31.5 GiB box, kept its lock margin, and the startup fill's first upload
/// failed with `CUDA_ERROR_OUT_OF_MEMORY` on most runs. Runs that loaded had
/// left 1.6 GiB or more. Two GiB keeps the pinned part below every failing
/// figure with room for the paging WDDM does when VRAM is full.
pub(crate) const ADDRESSABLE_MARGIN: u64 = 2 * 1024 * 1024 * 1024;

/// The most pinned slots of `slot_size` that leave [`ADDRESSABLE_MARGIN`] of
/// `room` — GPU-addressable host memory not yet in use — unspent, capped at
/// `want`. `None` is a platform with no such budget: the driver's own refusal
/// is the only bound.
pub(crate) fn addressable_pinned_slots(room: Option<u64>, slot_size: usize, want: usize) -> usize {
    match room {
        None => want,
        Some(room) if slot_size == 0 => want.min(room as usize),
        Some(room) => {
            let fits = room.saturating_sub(ADDRESSABLE_MARGIN) / slot_size as u64;
            want.min(fits as usize)
        }
    }
}

/// Whether the driver will still page-lock [`DRIVER_LOCK_MARGIN`] right now:
/// one allocation of that size, freed at once.
fn driver_lock_margin_free() -> bool {
    use cudarc::driver::sys::{cuMemAllocHost_v2, cuMemFreeHost, CUresult};
    let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    // SAFETY: a plain host allocation, freed before returning and never read.
    unsafe {
        if cuMemAllocHost_v2(&mut ptr, DRIVER_LOCK_MARGIN) != CUresult::CUDA_SUCCESS {
            return false;
        }
        cuMemFreeHost(ptr);
    }
    true
}

/// Replace `pool` with one of `slots`, releasing the old pages first: holding
/// both at once would need the very pages the margin probe asks for.
fn retake(pool: &mut WarmPool, slots: usize, slot_size: usize) {
    drop(std::mem::replace(
        pool,
        WarmPool::empty(PinnedUse::WeightWarmTier),
    ));
    *pool = WarmPool::new(slots, slot_size, PinnedUse::WeightWarmTier);
}

/// The largest slot count at or below `held` for which `keeps_margin` holds,
/// stepping down by `step` — `0` when none does.
fn largest_keeping_margin(
    mut held: usize,
    step: impl Fn(usize) -> usize,
    mut keeps_margin: impl FnMut(usize) -> bool,
) -> usize {
    while held > 0 && !keeps_margin(held) {
        held = step(held);
    }
    held
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

    /// One slot, writable — the fill writes the whole tier through
    /// [`WarmTier::slots_mut`]; this is the tests' way in to a single one.
    #[cfg(test)]
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
    /// A tier of `want_slots` stride-sized slots: as many pinned as the driver
    /// grants while still leaving it [`DRIVER_LOCK_MARGIN`] to lock and the
    /// device [`ADDRESSABLE_MARGIN`] of `addressable_room` to address, the rest
    /// pageable.
    pub(crate) fn new(want_slots: usize, slot_size: usize, addressable_room: Option<u64>) -> Self {
        let want_pinned = addressable_pinned_slots(addressable_room, slot_size, want_slots);
        let mut pinned = WarmPool::new(want_pinned, slot_size, PinnedUse::WeightWarmTier);
        // Each probe that fails gives back one step and re-takes the smaller
        // pool: one pool at a time, since holding both would need the pages the
        // margin is asking for.
        let keep = largest_keeping_margin(
            pinned.num_slots(),
            |held| step_down(held, slot_size),
            |held| {
                if held != pinned.num_slots() {
                    retake(&mut pinned, held, slot_size);
                }
                pinned.num_slots() == held && driver_lock_margin_free()
            },
        );
        if keep != pinned.num_slots() {
            retake(&mut pinned, keep, slot_size);
        }
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

    /// Where slot `i` is device-readable — its address, when it is pinned.
    pub(crate) fn pinned_addr(&self, i: usize) -> Option<u64> {
        self.is_pinned(i).then(|| self.pinned.slot_addr(i))
    }

    /// The pinned part's addresses, `[lo, hi)`.
    pub(crate) fn pinned_range(&self) -> (u64, u64) {
        self.pinned.range()
    }

    pub(crate) fn slot_ref(&self, i: usize, len: usize) -> &[u8] {
        match i.checked_sub(self.pinned.num_slots()) {
            None => self.pinned.slot_ref(i, len),
            Some(p) => self.paged.slot(p, len),
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

    const GIB: u64 = 1024 * 1024 * 1024;

    /// The 31.5 GiB box: 17.22 GiB of non-local budget with 0.66 GiB already in
    /// use leaves 16.17 GiB of room, so at 1.84 MiB slots the pinned part stops
    /// at 14.17 GiB — under the 16.02 GiB that failed.
    #[test]
    fn the_pinned_part_leaves_the_addressable_margin() {
        let slot = 1_929_216usize; // 1.84 MiB
        let room = 16_560 * 1024 * 1024u64;
        let pinned = addressable_pinned_slots(Some(room), slot, 10_496);
        assert_eq!(pinned, 7_887);
        assert!(pinned as u64 * slot as u64 <= room - ADDRESSABLE_MARGIN);
        assert!((pinned as u64 + 1) * slot as u64 > room - ADDRESSABLE_MARGIN);
    }

    /// Room for more than is wanted pins what is wanted; no room pins nothing;
    /// a platform with no budget leaves the driver as the only bound.
    #[test]
    fn the_cap_only_ever_lowers_the_pinned_part() {
        assert_eq!(addressable_pinned_slots(Some(64 * GIB), 1 << 20, 100), 100);
        assert_eq!(addressable_pinned_slots(Some(GIB), 1 << 20, 100), 0);
        assert_eq!(addressable_pinned_slots(Some(0), 1 << 20, 100), 0);
        assert_eq!(addressable_pinned_slots(None, 1 << 20, 100), 100);
    }

    /// A pool the driver leaves its margin beside is kept whole.
    #[test]
    fn a_pool_that_keeps_the_margin_is_kept() {
        assert_eq!(largest_keeping_margin(10_496, |h| h - 37, |_| true), 10_496);
    }

    /// A pool that leaves no margin steps down until one is left — the 31.5 GiB
    /// box, whose driver ceiling sits a margin above the size that loads.
    #[test]
    fn a_pool_without_the_margin_steps_down_to_one_that_keeps_it() {
        let mut probed = Vec::new();
        let kept = largest_keeping_margin(
            100,
            |h| h - 10,
            |h| {
                probed.push(h);
                h <= 75
            },
        );
        assert_eq!(kept, 70);
        assert_eq!(probed, vec![100, 90, 80, 70]);
    }

    /// No size keeps the margin: nothing is pinned, and every draw slot is pageable.
    #[test]
    fn no_size_keeping_the_margin_pins_nothing() {
        assert_eq!(
            largest_keeping_margin(30, |h| h.saturating_sub(10), |_| false),
            0
        );
        assert_eq!(largest_keeping_margin(0, |h| h, |_| false), 0);
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

    /// What staging one pageable slot costs on the host: one expert record copied into a
    /// staging buffer, single-threaded as the miss path does it, and cut across threads.
    /// Sources rotate through more slots than the last-level cache holds, so every copy reads
    /// from memory as a cold miss does.
    #[test]
    #[ignore = "benchmark — run explicitly with --ignored --nocapture"]
    fn bench_staging_one_pageable_slot() {
        const SLOT: usize = 14_155_776; // 13.5 MiB, DeepSeek-V4-Flash's expert record.
        const SLOTS: usize = 64;
        let mut src = PagedSlots::new(SLOTS, SLOT);
        for i in 0..SLOTS {
            src.slot_mut(i, SLOT).fill(i as u8);
        }
        let mut dst = vec![0u8; SLOT];
        let reps = 200;
        let time = |threads: usize, dst: &mut [u8]| {
            let t0 = std::time::Instant::now();
            for r in 0..reps {
                let s = src.slot(r % SLOTS, SLOT);
                if threads == 1 {
                    dst.copy_from_slice(s);
                } else {
                    let chunk = SLOT.div_ceil(threads);
                    std::thread::scope(|scope| {
                        for (d, s) in dst.chunks_mut(chunk).zip(s.chunks(chunk)) {
                            scope.spawn(move || d.copy_from_slice(s));
                        }
                    });
                }
            }
            t0.elapsed().as_secs_f64() * 1e3 / reps as f64
        };
        for threads in [1usize, 2, 4, 8] {
            let ms = time(threads, &mut dst);
            println!(
                "{threads} thread(s): {ms:.3} ms per 13.5 MiB slot ({:.1} GB/s)",
                SLOT as f64 / ms / 1e6
            );
        }
        assert_eq!(dst[0], ((reps - 1) % SLOTS) as u8);
    }
}
