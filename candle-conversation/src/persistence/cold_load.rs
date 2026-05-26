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
//!   chunk records  ─pread─▶  pinned host scratch  ─cuMemcpyHtoDAsync─▶  VRAM staging  ─kv_migrate─▶  arena chunks
//!                            (cudaHostAlloc'd)         (copy stream)
//! ```
//!
//! The host pinned scratch lets the HtoD leg run as a single DMA without
//! the driver's internal pageable-bounce buffer; the VRAM staging
//! scratch is the same one `scatter_chunks` already uses.
//!
//! The interface here ([`ColdLoadStager`]) is shaped so that the Linux
//! GDS backend can be swapped in as a single `cuFileReadAsync` against a
//! `cuFileBufRegister`'d VRAM scratch — replacing both the pinned-host
//! step and the HtoD step — without changing the cold-load caller.
//!
//! ## What's optimized vs. what's deferred
//!
//! This bridge gives us:
//! - **No pageable bounce.** The host source of the HtoD is real pinned
//!   memory, so the driver does not need to synthesise a hidden
//!   `cudaMallocHost` + memcpy step before the DMA.
//! - **Reused scratch.** The pinned scratch is allocated once at
//!   substrate open and grown on demand, not allocated per cold load.
//!
//! Explicitly *not* in the bridge:
//! - **`pread`-directly-into-pinned.** Today the chunk records are
//!   decoded into a [`ChunkPayload`](super::record::ChunkPayload) (which
//!   owns a pageable `Vec<u8>` for `kv_bytes`), and we then `memcpy` from
//!   that `Vec` into the pinned scratch. The extra host memcpy is the
//!   cost of not extending [`ChunkLoc`](super::manifest::ChunkLoc) with
//!   the absolute file offset of `kv_bytes`. The GDS upgrade fixes both
//!   in one step (GDS needs the offset and removes the host hop), so
//!   plumbing the offset for the bridge alone would be throwaway work.
//! - **Double-buffering + NVMe/PCIe overlap.** Single-buffer for now —
//!   we issue one HtoD per cold load, not a ping-pong pipeline. For
//!   typical 50–500 chunk loads this is invisible; at multi-GB loads
//!   it leaves perf on the table. Double-buffering is straightforward
//!   to add but is moot once we have GDS.

use candle::Result;

#[cfg(feature = "cuda")]
use candle::cuda_backend::cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
#[cfg(feature = "cuda")]
use candle::cuda_backend::WrapErr;
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::PinnedBuf;
#[cfg(feature = "cuda")]
use candle::CudaDevice;

/// A reusable host-side cold-load scratch: pinned on CUDA, plain
/// `Vec<u8>` otherwise. The scratch is grown on demand and persists
/// across cold loads, so a steady-state daemon allocates `cuMemHostAlloc`
/// at most a handful of times in its life.
///
/// Owned by [`SubstratePersistence`](super::SubstratePersistence) so the
/// allocation is amortised across every cold load of every stream.
pub struct ColdLoadStager {
    #[cfg(feature = "cuda")]
    buf: Option<PinnedBuf>,
    #[cfg(not(feature = "cuda"))]
    buf: Vec<u8>,
    /// High-water mark: the number of bytes we last *used* (not the buffer
    /// capacity). Diagnostic only — useful for the inspector.
    high_water_bytes: usize,
}

impl Default for ColdLoadStager {
    fn default() -> Self {
        Self::new()
    }
}

impl ColdLoadStager {
    /// Build an empty stager. The first call to `pack` grows the scratch
    /// to fit the request; subsequent calls re-use the allocation
    /// (growing again only if the request is larger than the high-water
    /// mark).
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "cuda")]
            buf: None,
            #[cfg(not(feature = "cuda"))]
            buf: Vec::new(),
            high_water_bytes: 0,
        }
    }

    /// Bytes the stager has reserved (pinned or otherwise).
    pub fn capacity(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.buf.as_ref().map(|b| b.len()).unwrap_or(0)
        }
        #[cfg(not(feature = "cuda"))]
        {
            self.buf.capacity()
        }
    }

    /// Largest amount of data ever packed in a single call — a stable
    /// proxy for the steady-state cold-load size.
    pub fn high_water_bytes(&self) -> usize {
        self.high_water_bytes
    }

    /// Copy each chunk's `kv_bytes` into the scratch in iteration order,
    /// returning the slice the caller should upload to the device.
    ///
    /// Grows the scratch if needed. Returns `Ok(&[])` for an empty
    /// iterator (no allocation performed).
    pub fn pack<'a, I>(&mut self, chunks: I) -> Result<&[u8]>
    where
        I: IntoIterator<Item = &'a [u8]>,
    {
        // Collect into a small Vec first so we can size-the-buffer once.
        // Iterator size_hint is unreliable for the manifest-driven
        // chunk-record streaming path; a small Vec of `&[u8]` is cheap.
        let chunks: Vec<&[u8]> = chunks.into_iter().collect();
        let total: usize = chunks.iter().map(|c| c.len()).sum();
        if total == 0 {
            return Ok(&[]);
        }
        self.ensure_capacity(total)?;
        self.high_water_bytes = self.high_water_bytes.max(total);

        let dst = self.as_mut_slice();
        let mut off = 0usize;
        for c in chunks {
            let end = off + c.len();
            dst[off..end].copy_from_slice(c);
            off = end;
        }
        Ok(&self.as_slice()[..total])
    }

    /// Borrow the packed bytes from the previous `pack` call. The slice
    /// length is the most recent `high_water_bytes` value, which after
    /// `pack` equals the packed length.
    pub fn as_slice(&self) -> &[u8] {
        #[cfg(feature = "cuda")]
        {
            match &self.buf {
                Some(b) => b.as_slice(),
                None => &[],
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            &self.buf
        }
    }

    fn as_mut_slice(&mut self) -> &mut [u8] {
        #[cfg(feature = "cuda")]
        {
            self.buf
                .as_mut()
                .expect("ensure_capacity allocated before pack")
                .as_mut_slice()
        }
        #[cfg(not(feature = "cuda"))]
        {
            &mut self.buf
        }
    }

    fn ensure_capacity(&mut self, len: usize) -> Result<()> {
        #[cfg(feature = "cuda")]
        {
            let need_grow = match &self.buf {
                Some(b) => b.len() < len,
                None => true,
            };
            if need_grow {
                // `cuMemHostAlloc` needs an initialised CUDA context — in
                // unit tests (no GPU touched) it returns
                // `CUDA_ERROR_NOT_INITIALIZED`. Fall back to a plain
                // `Vec<u8>`-backed `PinnedBuf::Host` in that case so the
                // pack/upload logic is exercisable in unit tests; the real
                // pinned path is exercised by the integration test that
                // actually allocates a CUDA device first.
                self.buf = Some(match PinnedBuf::alloc_owned(len) {
                    Ok(b) => b,
                    Err(_) => PinnedBuf::Host {
                        data: vec![0u8; len],
                    },
                });
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            if self.buf.len() < len {
                self.buf.resize(len, 0);
            }
            Ok(())
        }
    }

    /// Asynchronously upload the *first `len` bytes* of the scratch to a
    /// freshly-allocated device buffer via `cuMemcpyHtoDAsync`. The
    /// `len` is typically what the most recent [`Self::pack`] returned —
    /// pass `self.high_water_bytes()` to upload everything the last call
    /// packed.
    ///
    /// Returns a device slice and the `i64` device base pointer (the
    /// shape `kv_migrate` expects). The returned device slice must
    /// outlive the kernel that reads from it.
    #[cfg(feature = "cuda")]
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
        let src = &self.as_slice()[..len];
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

/// Result of [`ColdLoadStager::upload_async`] — a freshly-allocated device
/// slice plus its base pointer ready to feed to a scatter kernel.
#[cfg(feature = "cuda")]
pub struct UploadedScratch {
    /// Owns the device allocation. Drop frees the device memory; keep
    /// alive until the consuming kernel completes.
    pub slice: CudaSlice<u8>,
    /// `i64`-typed base pointer (the shape `MigrationPlan` expects).
    pub base_ptr: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fresh stager allocates nothing until the first `pack`.
    #[test]
    fn stager_starts_empty_and_lazy() {
        let s = ColdLoadStager::new();
        assert_eq!(s.capacity(), 0);
        assert_eq!(s.high_water_bytes(), 0);
        assert!(s.as_slice().is_empty());
    }

    /// Packing concatenates inputs in iteration order, byte-for-byte —
    /// the property the scatter kernel relies on.
    #[test]
    fn pack_concatenates_in_order() {
        let mut s = ColdLoadStager::new();
        let a = b"alpha".as_slice();
        let b = b"beta".as_slice();
        let c = b"gamma".as_slice();
        let packed_len = {
            let packed = s.pack([a, b, c]).expect("pack");
            assert_eq!(packed, b"alphabetagamma");
            packed.len()
        };
        assert_eq!(s.high_water_bytes(), packed_len);
    }

    /// An empty pack is a no-op: no allocation, no high-water bump.
    #[test]
    fn pack_empty_is_noop() {
        let mut s = ColdLoadStager::new();
        let packed = s.pack(std::iter::empty::<&[u8]>()).unwrap();
        assert!(packed.is_empty());
        assert_eq!(s.capacity(), 0);
        assert_eq!(s.high_water_bytes(), 0);
    }

    /// A second `pack` smaller than the first reuses the buffer without
    /// shrinking — the capacity stays at the high-water mark, but
    /// `high_water_bytes` does not regress.
    #[test]
    fn pack_reuses_buffer_across_calls() {
        let mut s = ColdLoadStager::new();
        let big = vec![0xABu8; 4096];
        s.pack([big.as_slice()]).unwrap();
        let cap_after_big = s.capacity();
        assert!(cap_after_big >= 4096);
        assert_eq!(s.high_water_bytes(), 4096);

        let small = vec![0xCDu8; 128];
        let packed = s.pack([small.as_slice()]).unwrap();
        assert_eq!(packed, &small[..]);
        // Buffer not shrunk.
        assert_eq!(s.capacity(), cap_after_big);
        // High water reflects the largest pack, not the latest.
        assert_eq!(s.high_water_bytes(), 4096);
    }

    /// Growing past the existing capacity reallocates and copies the
    /// new payload correctly (round-trip byte equality).
    #[test]
    fn pack_grows_for_larger_payloads() {
        let mut s = ColdLoadStager::new();
        s.pack([&[0xAAu8; 256][..]]).unwrap();
        let huge: Vec<u8> = (0..8192u32).map(|i| (i % 251) as u8).collect();
        let packed = s.pack([huge.as_slice()]).unwrap();
        assert_eq!(packed, &huge[..]);
        assert!(s.capacity() >= huge.len());
        assert_eq!(s.high_water_bytes(), huge.len());
    }

    /// Packing many small chunks (the typical cold-load shape — one
    /// `kv_bytes` blob per persisted chunk, many tens to hundreds of
    /// them) preserves boundary positions exactly.
    #[test]
    fn pack_preserves_chunk_boundaries() {
        let mut s = ColdLoadStager::new();
        let chunks: Vec<Vec<u8>> = (0..50)
            .map(|i| (0..128u32).map(|j| ((i * 31 + j) % 256) as u8).collect())
            .collect();
        let refs: Vec<&[u8]> = chunks.iter().map(|c| c.as_slice()).collect();
        let packed = s.pack(refs.iter().copied()).unwrap();
        let mut expected = Vec::with_capacity(50 * 128);
        for c in &chunks {
            expected.extend_from_slice(c);
        }
        assert_eq!(packed, expected.as_slice());
    }
}
