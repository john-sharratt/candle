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

use candle::cuda_backend::cudarc::driver::result::memcpy_htod_async;
use candle::{Device, Result};
use candle_nn::kv_cache::ArenaSlot;

/// Write one page's words into its slot, async on the device's primary stream.
///
/// The scan kernel is launched on the same stream, so it reads the page only after
/// this copy lands.
pub(super) fn write_page(device: &Device, slot: &ArenaSlot, words: &[u64]) -> Result<()> {
    let Device::Cuda(dev) = device else {
        candle::bail!("gallery page upload: the gallery lives on a CUDA device");
    };
    let bytes = std::mem::size_of_val(words);
    if bytes > slot.stride() {
        candle::bail!(
            "gallery page upload: {bytes} B of page into a {} B slot",
            slot.stride()
        );
    }
    // The copy below is a raw driver call on a raw stream, so this thread needs the
    // context current — `CudaDevice`'s own methods bind for their callers, but
    // nothing has bound for us here. A warm-up running on the normalization pool's
    // workers is the case that found this.
    dev.bind_to_thread()?;
    let stream = dev.cuda_stream();
    // SAFETY: the slot spans at least `bytes` (checked above) and is held by the
    // caller's page run for as long as the run lives; `words` is a host slice of
    // exactly that length that outlives the enqueue.
    unsafe {
        memcpy_htod_async(slot.ptr(), words, stream.cu_stream())
            .map_err(|e| candle::Error::Msg(format!("gallery page H2D: {e}")))?;
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
