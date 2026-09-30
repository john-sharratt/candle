//! Moving a page's words between the host and its slot in the gallery's arenas.
//!
//! A page lives in one slot of a **gallery-tenant slot arena** (`candle_nn`'s
//! `tenant_arena`): a region of the device reservation cut into page-sized slots,
//! dedicated to the gallery, counted and returned like every other span tenant.
//! The slot's address is the page's address, and the scan kernel receives those
//! addresses directly (the paged-KV `k_ptr` precedent — see
//! `docs/archived/paged_gallery_arena.md` §3.2).
//!
//! These pages once came from the CUDA async pool, outside the reservation, and
//! were never returned. Memory outside the span competes with the span for the
//! same VRAM, and on WDDM the loser is demoted to host RAM rather than refused —
//! measured on the 3.6-35B: 3.7 GiB demoted, 17x on decode, and every individual
//! section of the memory report reading healthy because nothing summed them.

use std::ops::Range;

use candle::cuda_backend::cudarc::driver::result::memcpy_htod_async;
use candle::{Device, Result};
use candle_nn::kv_cache::ArenaSlot;

/// Split slots, by their device addresses, into maximal runs where each slot
/// starts exactly `stride` bytes after the one before — the runs one copy can
/// fill.
pub(super) fn contiguous_runs(ptrs: &[u64], stride: u64) -> Vec<Range<usize>> {
    let mut runs: Vec<Range<usize>> = Vec::new();
    for (i, &ptr) in ptrs.iter().enumerate() {
        match runs.last_mut() {
            Some(run) if ptrs[run.end - 1] + stride == ptr => run.end = i + 1,
            _ => runs.push(i..i + 1),
        }
    }
    runs
}

/// Write a turn's pages into its slots, async on the device's primary stream.
///
/// `pages` holds slot `i`'s bytes at `i * stride`, as
/// [`transpose_to_pages`](super::pages::transpose_to_pages) lays them out at the
/// slots' stride, so each run of address-contiguous slots is one copy. A fresh
/// claim is mostly one run: a 7M-token layer went from ~225k page-sized copies
/// to a handful.
///
/// The scan kernel is launched on the same stream, so it reads the pages only
/// after these copies land.
pub(super) fn write_pages(device: &Device, slots: &[ArenaSlot], pages: &[u64]) -> Result<()> {
    let Device::Cuda(dev) = device else {
        candle::bail!("gallery page upload: the gallery lives on a CUDA device");
    };
    let Some(stride) = slots.first().map(ArenaSlot::stride) else {
        return Ok(());
    };
    if slots.iter().any(|s| s.stride() != stride) {
        candle::bail!("gallery page upload: one turn's slots differ in stride");
    }
    let stride_words = stride / std::mem::size_of::<u64>();
    if pages.len() != slots.len() * stride_words {
        candle::bail!(
            "gallery page upload: {} words for {} slots of {stride} B",
            pages.len(),
            slots.len()
        );
    }
    // The copy below is a raw driver call on a raw stream, so this thread needs the
    // context current — `CudaDevice`'s own methods bind for their callers, but
    // nothing has bound for us here. A warm-up running on the normalization pool's
    // workers is the case that found this.
    dev.bind_to_thread()?;
    let stream = dev.cuda_stream();
    let ptrs: Vec<u64> = slots.iter().map(ArenaSlot::ptr).collect();
    for run in contiguous_runs(&ptrs, stride as u64) {
        let words = &pages[run.start * stride_words..run.end * stride_words];
        // SAFETY: the run's slots are address-contiguous at `stride` (that is what
        // makes it a run), so `[ptrs[run.start], + run.len() * stride)` lies
        // within slots the caller's page run holds for as long as it lives;
        // `words` is exactly that many bytes and outlives the enqueue (a pageable
        // source is staged before the call returns).
        unsafe {
            memcpy_htod_async(ptrs[run.start], words, stream.cu_stream())
                .map_err(|e| candle::Error::Msg(format!("gallery page H2D: {e}")))?;
        }
    }
    Ok(())
}

/// Read `n_words` of a page back to the host — test/verification only (a scan never
/// does this). Synchronous, so the caller sees the bytes on return.
#[cfg(test)]
pub(super) fn read_page(device: &Device, slot: &ArenaSlot, n_words: usize) -> Result<Vec<u64>> {
    use candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_sync;
    let Device::Cuda(dev) = device else {
        candle::bail!("gallery page readback: the gallery lives on a CUDA device");
    };
    dev.bind_to_thread()?;
    let mut out = vec![0u64; n_words];
    // SAFETY: the slot spans at least `n_words` words and is live for the call.
    unsafe {
        memcpy_dtoh_sync(&mut out, slot.ptr())
            .map_err(|e| candle::Error::Msg(format!("gallery page D2H: {e}")))?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::contiguous_runs;

    #[test]
    fn contiguous_slots_form_one_run() {
        assert_eq!(contiguous_runs(&[1000, 1100, 1200, 1300], 100), vec![0..4]);
    }

    #[test]
    fn a_gap_or_a_step_back_starts_a_new_run() {
        let ptrs = [1000, 1100, 1300, 1400, 900, 5000];
        assert_eq!(
            contiguous_runs(&ptrs, 100),
            vec![0..2, 2..4, 4..5, 5..6],
            "a gap (1100→1300), a step back (1400→900) and a jump each split"
        );
    }

    #[test]
    fn no_slots_no_runs() {
        assert!(contiguous_runs(&[], 100).is_empty());
    }
}
