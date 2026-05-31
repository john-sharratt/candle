//! Cold-load bridge: pinned-staging path for NVMe → VRAM transfers.
//!
//! ## What this is (and what it isn't)
//!
//! The substrate's cold-load path is, in its target shape, **NVMe → VRAM
//! directly via GPUDirect Storage** (`cuFileReadAsync`): the NVMe
//! controller DMAs straight through the GPU's PCIe BAR into a
//! `cuFileBufRegister`'d region of the VRAM staging scratch, with no
//! host bounce buffer and no copy-stream HtoD. See
//! [`docs/kv_tier_migration.md`](../../../docs/kv_tier_migration.md) §4.
//!
//! GPUDirect Storage (GDS) is a **Linux-only NVIDIA technology** — it
//! depends on the `nvidia-fs` kernel module that NVIDIA does not ship for
//! Windows, and `libcufile` itself is not part of the Windows CUDA
//! Toolkit. Microsoft's DirectStorage is the Windows analogue but is not
//! reachable from CUDA (it is a D3D12 API, tied to NTFS, with
//! GPU-decompression hooks intended for game-asset streaming, not raw
//! KV-cache cold loads).
//!
//! Until the production Linux workstation is in place — and until / if a
//! CUDA-reachable Windows GPU-storage API exists — this module
//! implements the cold-load path as a **bridge**:
//!
//! ```text
//!   chunk records  ─direct-pread─▶  pinned host scratch  ─cuMemcpyHtoDAsync─▶  VRAM staging  ─kv_migrate─▶  arena chunks
//!                  (O_DIRECT /         (cuMemHostAlloc,         (copy stream)
//!                  FILE_FLAG_           no WC)
//!                  NO_BUFFERING)
//! ```
//!
//! The reader pool ([`super::pipeline`]) `pread`s directly into this
//! pinned scratch — no `Vec<u8>` intermediate, no host-to-host copy.
//! The pinned scratch is allocated once at startup (no growth on the
//! hot path) without `CU_MEMHOSTALLOC_WRITECOMBINED` so the NVMe DMA
//! and CPU-side metadata decode both run at full bandwidth.
//!
//! ## Interface
//!
//! The interface here ([`ColdLoadStager`]) is shaped so that the Linux
//! GDS backend can be swapped in as a single `cuFileReadAsync` against a
//! `cuFileBufRegister`'d VRAM scratch — replacing both the pinned-host
//! step and the HtoD step — without changing the cold-load caller.

use candle::cuda_backend::cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use candle::cuda_backend::WrapErr;
use candle::quantized::pinned_staging::PinnedBuf;
use candle::CudaDevice;
use candle::Result;

use super::direct_io::AlignedScratch;

/// The cold-load scratch's backing storage — always 4 KiB-aligned so
/// the chunked direct-I/O reads land safely. The `Pinned` arm is the
/// production path (`cuMemHostAlloc`, no-WC pinned memory);
/// `Aligned` is a no-CUDA test fallback that uses `alloc_zeroed` with
/// a 4 KiB layout — still sector-aligned, just not CUDA-pinned, so
/// HtoD goes through the slower pageable path. CPU-only unit tests
/// take the `Aligned` arm; every other path takes `Pinned`.
enum ColdLoadBuf {
    Pinned(PinnedBuf),
    Aligned(AlignedScratch),
}

impl ColdLoadBuf {
    fn alloc(len: usize) -> Self {
        match PinnedBuf::alloc_owned_default(len) {
            Ok(b) => Self::Pinned(b),
            Err(_) => {
                let mut a = AlignedScratch::new();
                a.ensure(len)
                    .expect("ColdLoadBuf::alloc: AlignedScratch::ensure failed");
                Self::Aligned(a)
            }
        }
    }

    fn capacity(&self) -> usize {
        match self {
            Self::Pinned(b) => b.len(),
            Self::Aligned(a) => a.capacity(),
        }
    }

    fn as_mut_slice(&mut self, len: usize) -> &mut [u8] {
        match self {
            Self::Pinned(b) => &mut b.as_mut_slice()[..len],
            Self::Aligned(a) => a.as_mut_slice(len),
        }
    }

    fn as_slice(&self, len: usize) -> &[u8] {
        match self {
            Self::Pinned(b) => &b.as_slice()[..len],
            Self::Aligned(a) => a.as_slice(len),
        }
    }

    fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            Self::Pinned(b) => b.as_mut_slice().as_mut_ptr(),
            Self::Aligned(a) => a.as_mut_slice(a.capacity()).as_mut_ptr(),
        }
    }
}

/// Per-pinned-scratch initial allocation size, in bytes — used by every
/// scheduler/persistence-thread pinned buffer that wants to skip the
/// first-cold-load `cuMemHostAlloc` cost (~10–30 ms on Windows for
/// 64 MiB).
///
/// 64 MiB is sized against the actual observed cold-load shape on
/// Qwen3-30B-A3B: a typical turn is ~10–40 MiB of compressed KV (depends
/// on compression policy and turn length); 64 MiB covers ~95% of turns
/// without paying allocation cost on the hot path. Longer turns (16K+
/// context, or C0 reference quality) trigger a single grow inside
/// [`ColdLoadStager::ensure_capacity`] which thereafter holds at the
/// new high-water size.
///
/// Total pinned host RAM at init across all three buffers (cold-load,
/// elevate, persistence-thread): 192 MiB. A rounding error on any
/// production-relevant system.
pub const PINNED_PREALLOC_BYTES: usize = 64 * 1024 * 1024;

/// A reusable host-side cold-load scratch: a single fixed-size pinned
/// host buffer that the cold-load chunked read pipeline streams data
/// through, and the HtoD upload sources from. **Never grows.**
///
/// Allocated once at substrate startup via [`Self::with_preallocation`]
/// at [`PINNED_PREALLOC_BYTES`] bytes via `cuMemHostAlloc` **without**
/// `CU_MEMHOSTALLOC_WRITECOMBINED` — the cold-load fast path needs
/// NVMe DMA to write into the buffer at full bandwidth, and CPU reads
/// the metadata bytes for decode; both regress on WC pages.
///
/// `Option<ColdLoadBuf>` is a tri-state:
///  - `None`: stager has been constructed but never allocated (legacy
///    `::new()` path, unit-test-only).
///  - `Some(Pinned/Aligned)`: cuMemHostAlloc'd or aligned-heap fallback.
pub struct ColdLoadStager {
    buf: Option<ColdLoadBuf>,
}

impl Default for ColdLoadStager {
    fn default() -> Self {
        Self::new()
    }
}

impl ColdLoadStager {
    pub fn new() -> Self {
        Self { buf: None }
    }

    /// Construct a stager and immediately allocate a `bytes`-sized
    /// pinned host buffer. The buffer never grows after this.
    pub fn with_preallocation(bytes: usize) -> Self {
        Self {
            buf: Some(ColdLoadBuf::alloc(bytes)),
        }
    }

    pub fn capacity(&self) -> usize {
        self.buf.as_ref().map(|b| b.capacity()).unwrap_or(0)
    }

    /// Borrow the first `len` bytes of the pinned scratch as a mutable
    /// slice — the direct-I/O landing pad for one chunked-read chunk.
    /// Lazy-allocates on the `::new()` test path; thereafter fixed-size.
    pub fn buffer_mut(&mut self, len: usize) -> &mut [u8] {
        if self.buf.is_none() {
            self.buf = Some(ColdLoadBuf::alloc(PINNED_PREALLOC_BYTES.max(len)));
        }
        assert!(
            len <= self.capacity(),
            "ColdLoadStager::buffer_mut: chunk request {len} > capacity {} \
             (the chunk planner should have partitioned the turn to fit)",
            self.capacity(),
        );
        let buf = self.buf.as_mut().unwrap();
        buf.as_mut_slice(len)
    }

    /// Raw mutable pointer to the start of the pinned scratch — used
    /// by the pipelined cold-load to hand a stable pointer to the
    /// reader threads. Lazy-allocates if needed.
    pub fn buffer_ptr_mut(&mut self) -> *mut u8 {
        if self.buf.is_none() {
            self.buf = Some(ColdLoadBuf::alloc(PINNED_PREALLOC_BYTES));
        }
        self.buf.as_mut().unwrap().as_mut_ptr()
    }

    /// Immutable view of the first `len` bytes of the pinned scratch.
    pub fn buffer(&self, len: usize) -> &[u8] {
        debug_assert!(len <= self.capacity());
        match &self.buf {
            Some(b) => b.as_slice(len),
            None => {
                debug_assert_eq!(len, 0);
                &[]
            }
        }
    }

    /// Asynchronously upload the first `len` bytes of the scratch to a
    /// freshly-allocated device buffer via `cuMemcpyHtoDAsync`.
    pub fn upload_async(
        &self,
        dev: &CudaDevice,
        stream: &std::sync::Arc<CudaStream>,
        len: usize,
    ) -> Result<UploadedScratch> {
        let _ = dev;
        if len == 0 {
            let gpu = unsafe { stream.alloc::<u8>(0).w()? };
            return Ok(UploadedScratch {
                slice: gpu,
                base_ptr: 0,
            });
        }
        let src = self.buffer(len);
        let mut gpu: CudaSlice<u8> = unsafe { stream.alloc::<u8>(len).w()? };
        stream.memcpy_htod(src, &mut gpu).w()?;
        let base = {
            let (ptr, _g) = gpu.device_ptr(stream);
            ptr as i64
        };
        Ok(UploadedScratch {
            slice: gpu,
            base_ptr: base,
        })
    }
}

/// Result of [`ColdLoadStager::upload_async`] — a freshly-allocated
/// device slice plus its base pointer ready to feed to a scatter
/// kernel.
pub struct UploadedScratch {
    pub slice: CudaSlice<u8>,
    pub base_ptr: i64,
}

/// Eagerly allocate a `bytes`-sized non-write-combined pinned host
/// buffer for the elevate / persistence-thread DtoH staging paths.
///
/// Same best-effort semantics as
/// [`ColdLoadStager::with_preallocation`]: returns `None` and logs a
/// warning if `cuMemHostAlloc` fails, falling back to lazy allocation
/// on first use. `tag` is included in the warning for diagnostics.
///
/// Use `alloc_owned_default` (no `CU_MEMHOSTALLOC_WRITECOMBINED`)
/// because every consumer of these buffers is a destination of
/// `memcpy_dtoh` followed by a CPU read of the bytes — WC would
/// pessimise the CPU-read leg.
pub fn preallocate_pinned_scratch(bytes: usize, tag: &'static str) -> Option<PinnedBuf> {
    match PinnedBuf::alloc_owned_default(bytes) {
        Ok(b) => Some(b),
        Err(e) => {
            tracing::warn!(
                target: "candle_conversation::persistence::cold_load",
                bytes,
                tag,
                error = %e,
                "pinned scratch preallocation failed — falling back to lazy alloc",
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `::new()` allocates nothing.
    #[test]
    fn stager_new_starts_empty() {
        let s = ColdLoadStager::new();
        assert_eq!(s.capacity(), 0);
        assert!(s.buffer(0).is_empty());
    }

    /// `with_preallocation` sizes the buffer up front.
    #[test]
    fn with_preallocation_sizes_buffer_exactly() {
        let s = ColdLoadStager::with_preallocation(4096);
        assert_eq!(s.capacity(), 4096);
    }
}
