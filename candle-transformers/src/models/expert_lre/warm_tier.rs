//! The expert warm tier: pinned segments first, pageable slots for the rest.
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
//! So the pinned part takes as many segments of the draw as the driver grants
//! (`elastic::region`), then **proves** the driver still has
//! [`DRIVER_LOCK_MARGIN`] to lock, giving segments back until it does. The
//! ceiling is measured on the machine it runs on rather than assumed from one;
//! whatever the draw wants beyond it lives in ordinary pageable memory. Past
//! the load, the pinned part is the elastic tier's to grow and shrink
//! (`docs/elastic_warm_tier.md`): its segments are the pad's resident region.
//!
//! A pageable slot is never handed to the driver as an upload source: every
//! upload from one goes through the pinned cold-staging ring first
//! ([`WarmTier::is_pinned`] tells the caller which), exactly like a cold read,
//! minus the drive. That copy is not cheap — 0.59 ms for a 13.5 MiB
//! DeepSeek-V4-Flash expert (`bench_staging_one_pageable_slot`), twice the
//! PCIe 5.0 upload it gates, and more threads only slow it — which is why the
//! pinned part is as large as the driver allows.
//!
//! Slot `i` is membership entry `i`: slots `0..pinned_slots()` — the startup
//! draw's pinned part, fixed for the process — are pinned, the rest pageable.
//! Both are cut to the pack's stride and sector-aligned, so the startup fill's
//! direct reads land in either with nothing in between.

use std::alloc::{alloc, dealloc, Layout};
use std::sync::Arc;

use candle::direct_io::DIRECT_IO_SECTOR;
use candle::quantized::pinned_ranges::PinnedRangeTable;

use super::elastic::region::ResidentRegion;
use super::page_pressure::{press, PAGE_PRESSURE_BYTES};

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
/// failed with `CUDA_ERROR_OUT_OF_MEMORY` on most runs.
///
/// **Two GiB is not enough either.** On driver 610.60, Flash-Next on the same
/// box, a tier the cap bound left exactly 2.00 GiB of a 17.22 GiB budget
/// (15.22 GiB used) and the driver refused everything from then on — the first
/// upload, `cuMemGetInfo`, even `cuMemFreeHost` — with 2.4 GiB of host RAM
/// free; a tier that RAM bound first left 2.15 GiB and loaded. The cap only
/// binds when RAM is plentiful, so the margin decided whether a *cleaner*
/// machine booted. Three GiB keeps the pinned part a GiB clear of the failing
/// figure, with room for the paging WDDM does when VRAM is full.
pub(crate) const ADDRESSABLE_MARGIN: u64 = 3 * 1024 * 1024 * 1024;

/// The most pinned slots of `slot_size` that leave [`ADDRESSABLE_MARGIN`] of
/// `room` — GPU-addressable host memory not yet in use — unspent, capped at
/// `want`. `None` is a driver model with no such budget — TCC or Linux
/// (`docs/driver_model.md` §3): the driver's own refusal is the only bound.
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
pub(crate) fn driver_lock_margin_free() -> bool {
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

/// The largest count at or below `held` for which `keeps_margin` holds,
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
    /// The pinned segments — after the fill, the pad's resident region.
    pinned: Arc<ResidentRegion>,
    /// Pinned slots the startup draw took: the membership index the pageable
    /// part starts at, fixed whatever the region does after.
    startup_pinned: usize,
    paged: PagedSlots,
}

impl WarmTier {
    /// A tier of `want_slots` stride-sized slots: as many pinned, in whole
    /// segments of `per` slots whose ranges go in `ranges`, as the driver
    /// grants while still leaving it [`DRIVER_LOCK_MARGIN`] to lock and the
    /// device [`ADDRESSABLE_MARGIN`] of `addressable_room` to address, the rest
    /// pageable. `per` is fixed for the process — the elastic tier grows and
    /// shrinks by it — so it is the model's, not this moment's
    /// (`segment_slots` of the evictable experts, `elastic::region`).
    pub(crate) fn new(
        want_slots: usize,
        slot_size: usize,
        per: usize,
        addressable_room: Option<u64>,
        ranges: Arc<PinnedRangeTable>,
    ) -> Self {
        let want_pinned = addressable_pinned_slots(addressable_room, slot_size, want_slots);
        let region = Arc::new(ResidentRegion::new(slot_size, per, ranges));
        let per = region.per_segment();
        // Whole segments, as many as the draw wants and the driver grants. A
        // refusal is pressed past once — the pages may be other processes'
        // idle ones — and a second refusal is the answer.
        let mut pressed = false;
        while region.slots() + per <= want_pinned {
            if region.grow() {
                continue;
            }
            if pressed || !press(PAGE_PRESSURE_BYTES) {
                break;
            }
            pressed = true;
        }
        let kept = largest_keeping_margin(
            region.segments(),
            |held| held - 1,
            |held| {
                while region.segments() > held {
                    region.pop();
                }
                driver_lock_margin_free()
            },
        );
        while region.segments() > kept {
            region.pop();
        }
        let startup_pinned = region.slots();
        let paged = PagedSlots::new(want_slots.saturating_sub(startup_pinned), slot_size);
        tracing::info!(
            target: "candle_transformers::expert_lre",
            pinned = startup_pinned,
            segments = region.segments(),
            paged = paged.num_slots,
            pinned_gib = (startup_pinned * slot_size) as f64 / (1u64 << 30) as f64,
            paged_gib = (paged.num_slots * slot_size) as f64 / (1u64 << 30) as f64,
            pressed,
            "warm tier: pinned segments, then pageable for the rest"
        );
        Self {
            pinned: region,
            startup_pinned,
            paged,
        }
    }

    pub(crate) fn num_slots(&self) -> usize {
        self.startup_pinned + self.paged.num_slots
    }

    pub(crate) fn paged_slots(&self) -> usize {
        self.paged.num_slots
    }

    /// Whether membership slot `i` is pinned, and so a valid direct upload
    /// source.
    pub(crate) fn is_pinned(&self, i: usize) -> bool {
        i < self.startup_pinned
    }

    /// The pinned segments, for the pad and the elastic controller.
    pub(crate) fn region(&self) -> &Arc<ResidentRegion> {
        &self.pinned
    }

    /// Slots the startup draw pinned: `0..pinned_slots()`. Once the startup
    /// fill has written them they are the pad's resident region (`pad.rs`).
    pub(crate) fn pinned_slots(&self) -> usize {
        self.startup_pinned
    }

    pub(crate) fn slot_ref(&self, i: usize, len: usize) -> &[u8] {
        match i.checked_sub(self.startup_pinned) {
            None => self.pinned.slot_ref(i, len),
            Some(p) => self.paged.slot(p, len),
        }
    }

    /// The first `n` slots as separate stride-long slices, for one batched
    /// fill across both parts.
    pub(crate) fn slots_mut(&mut self, n: usize, stride: usize) -> Vec<&mut [u8]> {
        let pinned_n = n.min(self.startup_pinned);
        let paged_n = n - pinned_n;
        assert!(
            paged_n <= self.paged.num_slots,
            "warm tier: {n} slots past the tier"
        );
        // SAFETY: the fill is the only writer, before the tier is shared, and
        // each slot is handed out once.
        let mut out: Vec<&mut [u8]> = (0..pinned_n)
            .map(|r| unsafe { &mut self.pinned.slot_mut(r)[..stride] })
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
        self.pinned.held_bytes() as usize + self.paged.num_slots * self.paged.slot_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    /// The 31.5 GiB box: 17.22 GiB of non-local budget with 0.66 GiB already in
    /// use leaves 16.17 GiB of room, so at 1.84 MiB slots the pinned part stops
    /// at 13.17 GiB — three GiB of room unspent, a GiB clear of the 2.00 GiB
    /// that failed.
    #[test]
    fn the_pinned_part_leaves_the_addressable_margin() {
        let slot = 1_929_216usize; // 1.84 MiB
        let room = 16_560 * 1024 * 1024u64;
        let pinned = addressable_pinned_slots(Some(room), slot, 10_496);
        assert_eq!(pinned, 7_331);
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

    /// A pinned part the driver leaves its margin beside is kept whole.
    #[test]
    fn a_pool_that_keeps_the_margin_is_kept() {
        assert_eq!(largest_keeping_margin(14, |h| h - 1, |_| true), 14);
    }

    /// One that leaves no margin gives segments back until one is left — the
    /// 31.5 GiB box, whose driver ceiling sits a margin above the size that
    /// loads.
    #[test]
    fn a_pool_without_the_margin_gives_segments_back_until_it_keeps_it() {
        let mut probed = Vec::new();
        let kept = largest_keeping_margin(
            10,
            |h| h - 1,
            |h| {
                probed.push(h);
                h <= 7
            },
        );
        assert_eq!(kept, 7);
        assert_eq!(probed, vec![10, 9, 8, 7]);
    }

    /// No size keeping the margin pins nothing, and every draw slot is pageable.
    #[test]
    fn no_size_keeping_the_margin_pins_nothing() {
        assert_eq!(largest_keeping_margin(3, |h| h - 1, |_| false), 0);
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

    /// **The pinned part is whole segments, the rest pageable**, and the
    /// startup draw's pinned count is fixed whatever the region does after.
    #[test]
    fn the_pinned_part_is_whole_segments_and_the_startup_count_stays() -> candle::Result<()> {
        let Ok(_dev) = candle::Device::new_cuda(0) else {
            return Ok(());
        };
        let ranges = Arc::new(PinnedRangeTable::new()?);
        let tier = WarmTier::new(13, 8192, 5, None, ranges.clone());
        assert_eq!(tier.pinned_slots(), 10, "two whole segments");
        assert_eq!(
            tier.paged_slots(),
            3,
            "the part short of a segment is pageable"
        );
        assert_eq!(ranges.ranges().len(), 2);
        let mut tier = WarmTier::new(10, 8192, 10, None, ranges.clone());
        assert_eq!(tier.pinned_slots(), 10, "a small tier is one segment");
        assert_eq!(tier.paged_slots(), 0);
        assert_eq!(
            ranges.ranges().len(),
            3,
            "the first tier's two, and this one's"
        );
        {
            let mut slots = tier.slots_mut(10, 8192);
            slots[9].fill(7);
        }
        assert_eq!(tier.slot_ref(9, 8192)[0], 7);
        assert!(tier.region().grow());
        assert_eq!(tier.pinned_slots(), 10, "the startup count is fixed");
        assert_eq!(tier.region().slots(), 20);
        assert!(tier.is_pinned(9) && !tier.is_pinned(10));
        Ok(())
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
