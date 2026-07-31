//! The physical VRAM slabs behind the gallery arena.
//!
//! Each slab is a persistent `CudaSlice<u64>` of `slab_pages * page_u64` words,
//! held resident for the arena's lifetime (freed when the arena drops). A page
//! slot is written with a single sub-range H2D copy; its device address is the
//! slab base plus the slot's byte offset. The scan kernel receives those
//! addresses directly (the paged-KV `k_ptr` precedent — see
//! `docs/paged_gallery_arena.md` §3.2).

use candle::cuda_backend::cudarc::driver::{CudaSlice, DevicePtr};
use candle::{Device, Result};

/// Persistent device slabs for the gallery arena. Slabs are appended on demand
/// and held for the arena's lifetime — freed only when the whole arena drops.
pub struct GalleryStorage {
    device: Device,
    page_u64: usize,
    slab_pages: usize,
    slabs: Vec<CudaSlice<u64>>,
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

    #[inline]
    pub fn slab_count(&self) -> usize {
        self.slabs.len()
    }

    /// Total VRAM held by the slabs, in bytes.
    #[inline]
    pub fn resident_bytes(&self) -> u64 {
        (self.slabs.len() * self.slab_pages * self.page_u64 * std::mem::size_of::<u64>()) as u64
    }

    /// Register one more zeroed slab. Called by the arena immediately before it
    /// grows the pool's capacity, so pool ids never outrun physical slabs.
    pub fn add_slab(&mut self) -> Result<()> {
        let dev = self.cuda();
        let stream = dev.cuda_stream();
        let words = self.slab_pages * self.page_u64;
        let slab = stream
            .alloc_zeros::<u64>(words)
            .map_err(|e| candle::Error::Msg(format!("gallery slab alloc ({words} u64): {e}")))?;
        // Extract the stable base address, then drop the borrow guard so the slab
        // can move into the Vec. The address is fixed for the allocation's life;
        // stream ordering for actual reads is handled at kernel-launch time.
        let base = {
            let (b, _guard) = slab.device_ptr(&stream);
            b
        };
        self.slabs.push(slab);
        self.slab_base.push(base);
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
        let off = page_in_slab * self.page_u64;
        let slab = &mut self.slabs[slab_idx];
        let mut view = slab.slice_mut(off..off + self.page_u64);
        stream
            .memcpy_htod(words, &mut view)
            .map_err(|e| candle::Error::Msg(format!("gallery page H2D: {e}")))?;
        Ok(())
    }

    /// The device byte address of a page slot — `slab_base + page_in_slab *
    /// page_bytes`. This is what the scan kernel dereferences.
    #[inline]
    pub fn page_addr(&self, slab_idx: usize, page_in_slab: usize) -> u64 {
        self.slab_base[slab_idx]
            + (page_in_slab * self.page_u64 * std::mem::size_of::<u64>()) as u64
    }

    /// Read a page back to the host — test/verification only (a scan never does this).
    #[cfg(test)]
    pub fn read_page(&self, slab_idx: usize, page_in_slab: usize) -> Result<Vec<u64>> {
        use candle::cuda_backend::cudarc::driver::DevicePtr as _;
        let dev = self.cuda();
        let stream = dev.cuda_stream();
        let off = page_in_slab * self.page_u64;
        let slab = &self.slabs[slab_idx];
        let view = slab.slice(off..off + self.page_u64);
        let _ = view.device_ptr(&stream);
        stream
            .memcpy_dtov(&view)
            .map_err(|e| candle::Error::Msg(format!("gallery page D2H: {e}")))
    }
}
