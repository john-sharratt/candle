//! The physical VRAM slabs behind the gallery arena.
//!
//! Each slab is one **region of the device reservation** — the same 16 MiB unit
//! a KV arena claims, which is not a coincidence: `TARGET_ARENA_BYTES` here is
//! defined as a mirror of the KV arena's slab size, so a gallery slab and a KV
//! region are the same shape and the gallery can simply be another tenant.
//!
//! It did not used to be. These slabs came from the CUDA async pool, outside the
//! reservation, and the arena's own docs recorded the consequence: the slabs
//! "are never returned", so relief could not move them. The deeper cost was
//! invisible until the whole card was reconciled — memory outside the span
//! competes with the span for the same VRAM, and on WDDM the loser is demoted
//! to host RAM rather than refused. Measured on the 3.6-35B: 3.7 GiB demoted,
//! 17x on decode, and every individual section of the memory report reading
//! healthy because nothing summed them.
//!
//! A page slot is written with a single sub-range H2D copy to its device
//! address; the scan kernel receives those addresses directly (the paged-KV
//! `k_ptr` precedent — see `docs/paged_gallery_arena.md` §3.2). That contract is
//! unchanged: `page_addr` always returned a raw address, so only where the bytes
//! come from has moved.

use candle::cuda_backend::cudarc::driver::result::memcpy_htod_async;
use candle::{Device, Result};
use candle_nn::kv_cache::{claim_span_region, SpanRegion};

/// Persistent device slabs for the gallery arena, each one a claimed region of
/// the reservation. Held for the arena's lifetime and returned — to the region
/// free list, not to the OS — when it drops.
pub struct GalleryStorage {
    device: Device,
    page_u64: usize,
    slab_pages: usize,
    /// The claimed regions. Held for their RAII effect; addresses are cached in
    /// `slab_base` because the hot path indexes by slab and must not re-borrow.
    slabs: Vec<SpanRegion>,
    /// Stable device base address (bytes) of each slab, cached at allocation.
    slab_base: Vec<u64>,
}

impl GalleryStorage {
    /// A storage bound to a CUDA device. `page_u64` = words per page,
    /// `slab_pages` = page slots per slab.
    pub fn new(device: &Device, page_u64: usize, slab_pages: usize) -> Result<Self> {
        match device {
            Device::Cuda(_) => {}
            _ => {
                return Err(candle::Error::Msg(
                    "gallery arena requires a CUDA device".into(),
                ))
            }
        }
        Ok(Self {
            device: device.clone(),
            page_u64,
            slab_pages,
            slabs: Vec::new(),
            slab_base: Vec::new(),
        })
    }

    fn cuda(&self) -> &candle::CudaDevice {
        match &self.device {
            Device::Cuda(d) => d,
            _ => unreachable!("checked in new()"),
        }
    }

    /// Reservation bytes held by the slabs.
    ///
    /// **Regions, not payload.** A slab occupies a whole region even when its
    /// pages do not fill one — `slab_pages` is `16 MiB / page_bytes` floored, so
    /// any page size that does not divide the region leaves a remainder that is
    /// held and unusable. The governor reads this to decide what dropping the
    /// arena would give back, and what it gives back is regions.
    #[inline]
    pub fn resident_bytes(&self) -> u64 {
        (self.slabs.len() * SpanRegion::bytes()) as u64
    }

    /// Claim one region for a new slab. Called by the arena immediately before
    /// it grows the pool's capacity, so pool ids never outrun physical slabs.
    ///
    /// A refusal means the reservation itself is full, and it **is** an error.
    /// Routine growth is bounded well before this by the arena's own ceiling
    /// (`evict_to_cap_locked`, run before anything is admitted), so reaching a
    /// refusal is not the cap doing its job — it is the KV side having no region
    /// to spare for a gallery that is already inside its budget. There is no
    /// quieter answer available here: the page pool has been grown to expect
    /// this slab, so returning `Ok` would leave ids naming memory that does not
    /// exist.
    pub fn add_slab(&mut self) -> Result<()> {
        let want = self.slab_pages * self.page_u64 * std::mem::size_of::<u64>();
        // Not a `debug_assert`: this bound is what keeps `page_addr` inside the
        // region. A slab wider than its region hands the scan kernel addresses
        // past the end of the claim — someone else's KV chunks, written through
        // by page uploads — and in a release build the assert is not there to
        // catch it. `slab_pages` floors `16 MiB / page_bytes`, so this can only
        // trip for a page larger than a whole region, where `.max(1)` keeps one
        // page that cannot fit.
        if want > SpanRegion::bytes() {
            return Err(candle::Error::Msg(format!(
                "gallery slab: {want} B of pages does not fit a {} B region — a \
                 single page is wider than the allocator's unit, so the slab \
                 cannot be a region tenant",
                SpanRegion::bytes(),
            )));
        }
        let Some(region) = claim_span_region(&self.device)? else {
            return Err(candle::Error::Msg(
                "gallery slab: the reservation has no free region — the KV side is \
                 full, so the gallery grows no further this pass"
                    .into(),
            ));
        };
        // The region arrives zeroed (mapping touch, or an explicit fill on
        // recycle), which is what the page pool assumes of a fresh slot.
        self.slab_base.push(region.base());
        self.slabs.push(region);
        Ok(())
    }

    /// Write one page's `page_u64` words into slot `page_in_slab` of `slab_idx`.
    pub fn write_page(
        &mut self,
        slab_idx: usize,
        page_in_slab: usize,
        words: &[u64],
    ) -> Result<()> {
        debug_assert_eq!(
            words.len(),
            self.page_u64,
            "page must be exactly page_u64 words"
        );
        let dev = self.cuda();
        let stream = dev.cuda_stream();
        let dst = self.page_addr(slab_idx, page_in_slab);
        // The destination is a raw address inside the reservation rather than a
        // `CudaSlice`, so the copy is the driver call the slice wrapper would
        // have made. Async on the compute stream, which is the ordering every
        // reader of these pages already observes: the scan kernel is launched on
        // the same stream and therefore after this copy.
        //
        // SAFETY: `dst` names `page_u64` words inside the slab's own region
        // (bounds are the page pool's invariant, asserted above), the region is
        // live for as long as `self.slabs` holds its handle, and `words` is a
        // host slice of exactly that length that outlives the call — the copy is
        // enqueued against the stream this thread owns and `words` is not freed
        // until it returns.
        unsafe {
            memcpy_htod_async(dst, words, stream.cu_stream())
                .map_err(|e| candle::Error::Msg(format!("gallery page H2D: {e}")))?;
        }
        Ok(())
    }

    /// The device byte address of a page slot — `slab_base + page_in_slab *
    /// page_bytes`. This is what the scan kernel dereferences.
    #[inline]
    pub fn page_addr(&self, slab_idx: usize, page_in_slab: usize) -> u64 {
        self.slab_base[slab_idx]
            + (page_in_slab * self.page_u64 * std::mem::size_of::<u64>()) as u64
    }

    /// Read a page back to the host — test/verification only (a scan never does
    /// this). The mirror of [`Self::write_page`]: a raw-address copy out of the
    /// region, synchronous so the caller sees the bytes on return.
    #[cfg(test)]
    pub fn read_page(&self, slab_idx: usize, page_in_slab: usize) -> Result<Vec<u64>> {
        use candle::cuda_backend::cudarc::driver::result::memcpy_dtoh_sync;
        let dev = self.cuda();
        dev.cuda_stream()
            .context()
            .bind_to_thread()
            .map_err(|e| candle::Error::Msg(format!("bind_to_thread: {e}")))?;
        let mut out = vec![0u64; self.page_u64];
        let src = self.page_addr(slab_idx, page_in_slab);
        // SAFETY: `src` names `page_u64` words inside a live region this
        // storage holds, and `out` is exactly that many words.
        unsafe {
            memcpy_dtoh_sync(&mut out, src)
                .map_err(|e| candle::Error::Msg(format!("gallery page D2H: {e}")))?;
        }
        Ok(out)
    }
}
