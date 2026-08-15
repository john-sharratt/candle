//! Pinned host memory pool for the warm tier of the two-tier expert cache.
//!
//! Experts that don't fit in VRAM are stored in physically-locked host
//! memory (`cuMemAllocHost`).  This memory is DMA-safe: H2D/D2H copies
//! from/to pinned memory use the GPU DMA engine at full PCIe bandwidth
//! (~25 GB/s) with no OS page faults.
//!
//! ## Architecture
//!
//! ```text
//!    SSD (GGUF)                                   ┌─────────────┐
//!        │ startup only                           │   VRAM      │
//!        ▼                                        │ (hot tier)  │
//!   GPU repack ──────── VRAM-resident ──────────▶ │  ExpertSlot │
//!        │                                        │  (QMatMul)  │
//!        │ overflow                               └──────┬──────┘
//!        ▼                                           ▲   │
//! ┌─────────────┐     promote (H2D)         evict    │   │
//! │ PinnedPool  │ ◀─────────────────────────(D2H)────┘   │
//! │ (warm tier) │ ──────────────────────────────────────▶ │
//! │  raw K/128  │     on copy_stream
//! └─────────────┘
//! ```
//!
//! At runtime, no SSD I/O occurs — all expert movement is between VRAM
//! and pinned RAM over PCIe.
//!
//! ## Slot layout
//!
//! Each slot stores the concatenated repacked projections:
//!
//! ```text
//! ┌──────────┬──────────┬──────────┬─── (padding) ──┐
//! │ gate K/128│ up K/128 │down K/128│                │
//! └──────────┴──────────┴──────────┴────────────────┘
//!  0         gate_len   +up_len    +down_len       slot_size
//! ```
//!
//! All slots have the same byte size (`max_repacked_expert_bytes`) for
//! uniform addressing.  Actual usage per slot varies by layer (Q4_K vs
//! Q6_K may differ).

#[cfg(feature = "cuda")]
use candle::Result;

/// Per-layer geometry: shapes, dtypes, and repacked byte sizes.
///
/// Within a MoE layer, all experts share the same geometry (but different
/// layers may use different dtypes, e.g. Q4_K_M uses Q6_K for first/last).
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub(crate) struct LayerGeometry {
    pub gate_shape: Vec<usize>,
    pub gate_dtype: candle::quantized::GgmlDType,
    pub gate_repacked_size: usize,

    pub up_shape: Vec<usize>,
    pub up_dtype: candle::quantized::GgmlDType,
    pub up_repacked_size: usize,

    pub down_shape: Vec<usize>,
    pub down_dtype: candle::quantized::GgmlDType,
    pub down_repacked_size: usize,

    /// Total repacked bytes for one expert in this layer.
    pub total_repacked_size: usize,
}

/// Physical location of an expert in the two-tier cache.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExpertLocation {
    /// Expert is resident in VRAM as an [`ExpertSlot`](super::types::ExpertSlot).
    Vram { slot_idx: usize },
    /// Expert is stored as raw K/128 bytes in the pinned host pool.
    Pinned { slot_idx: usize },
}

/// Pinned host memory pool — physically locked RAM for DMA-safe expert storage.
///
/// Memory is allocated once at startup via `cuMemAllocHost` and divided into
/// fixed-size slots.  Each slot holds one expert's concatenated repacked
/// projections (gate + up + down in K/128 format).
///
/// ## Thread safety
///
/// The pool is owned exclusively by the pipeline thread (`&mut self` access).
/// No interior mutability or atomic operations.
#[cfg(feature = "cuda")]
pub(crate) struct PinnedPool {
    /// One `cuMemAllocHost` block per chunk. The pool is split into ≤ [`Self::CHUNK_BYTES`] blocks
    /// because a single ~100+ GB `cuMemAllocHost` OOMs under WDDM even with far more host RAM free
    /// (a per-allocation ceiling, not a total-memory one) — chunking lets the warm tier use the
    /// machine's real RAM (e.g. a 145 GB expert overflow on a 194 GB box).
    chunks: Vec<*mut u8>,
    /// Slots per chunk (uniform; the last chunk may hold fewer). `slot_idx → (idx / slots_per_chunk,
    /// idx % slots_per_chunk)` addresses the block + offset.
    slots_per_chunk: usize,
    /// Per-slot byte size (uniform, = max repacked expert size across layers).
    slot_size: usize,
    /// Number of slots.
    num_slots: usize,
    /// Free slot indices (LIFO stack for O(1) alloc/free).
    pub(crate) free_slots: Vec<usize>,
}

#[cfg(feature = "cuda")]
impl PinnedPool {
    /// Max bytes per `cuMemAllocHost` block. A ~100 GB single allocation OOMs on WDDM even with
    /// more host RAM free, and the 93.6 GB pool that used to work sat right below that wall, so cap
    /// each block at 32 GiB — many small blocks reach the machine's full RAM.
    const CHUNK_BYTES: usize = 32usize << 30;

    /// Allocate a pinned memory pool with `num_slots` × `slot_size` bytes, split across
    /// [`Self::CHUNK_BYTES`] `cuMemAllocHost` blocks.
    ///
    /// Each block physically locks its pages, enabling the GPU DMA engine to access them without OS
    /// page faults.
    pub(crate) fn new(num_slots: usize, slot_size: usize) -> Result<Self> {
        if num_slots == 0 || slot_size == 0 {
            return Ok(Self::empty());
        }

        let slots_per_chunk = (Self::CHUNK_BYTES / slot_size).max(1);
        let n_chunks = num_slots.div_ceil(slots_per_chunk);
        let mut chunks: Vec<*mut u8> = Vec::with_capacity(n_chunks);
        let mut allocated = 0usize;
        for c in 0..n_chunks {
            let this_slots = (num_slots - c * slots_per_chunk).min(slots_per_chunk);
            let sz = this_slots * slot_size;
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let result = unsafe { cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, sz) };
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                // Free the blocks already taken before bailing.
                for &p in &chunks {
                    unsafe {
                        let _ = cudarc::driver::sys::cuMemFreeHost(p as *mut std::ffi::c_void);
                    }
                }
                candle::bail!(
                    "cuMemAllocHost failed: {:?} (chunk {}/{}: {} slots × {} bytes = {:.1} GB; \
                     pool so far {:.1} GB of {} slots)",
                    result,
                    c + 1,
                    n_chunks,
                    this_slots,
                    slot_size,
                    sz as f64 / 1e9,
                    allocated as f64 / 1e9,
                    num_slots,
                );
            }
            chunks.push(ptr as *mut u8);
            allocated += sz;
        }

        tracing::info!(
            "PinnedPool: allocated {:.1} GB pinned RAM in {} chunk(s) ({} slots × {:.1} KB)",
            allocated as f64 / 1e9,
            n_chunks,
            num_slots,
            slot_size as f64 / 1e3,
        );

        Ok(Self {
            chunks,
            slots_per_chunk,
            slot_size,
            num_slots,
            free_slots: (0..num_slots).rev().collect(),
        })
    }

    /// Host pointer to slot `slot_idx`'s bytes, resolving its chunk + offset.
    #[inline]
    pub(crate) fn slot_ptr(&self, slot_idx: usize) -> *mut u8 {
        let c = slot_idx / self.slots_per_chunk;
        let local = slot_idx % self.slots_per_chunk;
        unsafe { self.chunks[c].add(local * self.slot_size) }
    }

    /// Slots per chunk — for the parallel startup fill, which resolves slot pointers itself.
    #[inline]
    pub(crate) fn slots_per_chunk(&self) -> usize {
        self.slots_per_chunk
    }

    /// Chunk base pointers as `usize` (carried across the parallel fill's threads).
    #[inline]
    pub(crate) fn chunk_ptrs(&self) -> Vec<usize> {
        self.chunks.iter().map(|&p| p as usize).collect()
    }

    /// Get a mutable byte slice for a slot.
    ///
    /// `len` must be ≤ `slot_size`.
    #[inline]
    pub(crate) fn slot_mut(&mut self, slot_idx: usize, len: usize) -> &mut [u8] {
        debug_assert!(slot_idx < self.num_slots);
        debug_assert!(len <= self.slot_size);
        unsafe { std::slice::from_raw_parts_mut(self.slot_ptr(slot_idx), len) }
    }

    /// Get a shared byte slice for a slot.
    #[inline]
    pub(crate) fn slot_ref(&self, slot_idx: usize, len: usize) -> &[u8] {
        debug_assert!(slot_idx < self.num_slots);
        debug_assert!(len <= self.slot_size);
        unsafe { std::slice::from_raw_parts(self.slot_ptr(slot_idx), len) }
    }

    /// Allocate a free slot.  Returns `None` if the pool is full.
    #[inline]
    pub(crate) fn alloc(&mut self) -> Option<usize> {
        self.free_slots.pop()
    }

    /// Return a slot to the free list.
    #[inline]
    pub(crate) fn free(&mut self, slot_idx: usize) {
        debug_assert!(slot_idx < self.num_slots);
        self.free_slots.push(slot_idx);
    }

    /// Total number of slots.
    #[inline]
    pub(crate) fn num_slots(&self) -> usize {
        self.num_slots
    }

    #[inline]
    pub(crate) fn slot_size(&self) -> usize {
        self.slot_size
    }

    /// Create an empty pool (no allocation).
    ///
    /// Used when running on a non-CUDA device but the `cuda` feature is enabled.
    pub(crate) fn empty() -> Self {
        Self {
            chunks: Vec::new(),
            slots_per_chunk: 1,
            slot_size: 0,
            num_slots: 0,
            free_slots: Vec::new(),
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for PinnedPool {
    fn drop(&mut self) {
        for &p in &self.chunks {
            let result = unsafe { cudarc::driver::sys::cuMemFreeHost(p as *mut std::ffi::c_void) };
            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                tracing::warn!("PinnedPool: cuMemFreeHost failed: {:?}", result);
            }
        }
    }
}

// SAFETY: The pinned memory is allocated via cuMemAllocHost which returns
// a host-accessible pointer valid for any thread.  The pool is exclusively
// owned by the pipeline thread (no shared access).
#[cfg(feature = "cuda")]
unsafe impl Send for PinnedPool {}

#[cfg(all(test, feature = "cuda"))]
mod probe {
    /// Measure the real per-process pinned (`cuMemAllocHost`) ceiling on this machine: allocate
    /// 4 GiB blocks until one fails, report the total + the exact error, then confirm the same
    /// amount of PAGEABLE RAM allocates fine (so the wall is page-locking, not physical RAM). Run:
    ///   cargo test --release --features cuda -p candle-transformers pinned::probe -- --ignored --nocapture
    /// Re-run after granting "Lock pages in memory" / switching driver model to see if the ceiling moves.
    #[test]
    #[ignore]
    fn probe_pinned_ceiling() {
        // A CUDA context must be current for cuMemAllocHost.
        let _dev = candle::Device::new_cuda(0).expect("cuda device");
        const CHUNK: usize = 4usize << 30; // 4 GiB
        let mut blocks: Vec<*mut std::ffi::c_void> = Vec::new();
        let mut total = 0usize;
        loop {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let r = unsafe { cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, CHUNK) };
            if r != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                eprintln!(
                    "[pinned-probe] cuMemAllocHost FAILED at total={:.1} GiB (next +4 GiB): {:?}",
                    total as f64 / (1u64 << 30) as f64,
                    r
                );
                break;
            }
            blocks.push(ptr);
            total += CHUNK;
            eprintln!(
                "[pinned-probe] pinned OK: {:.1} GiB",
                total as f64 / (1u64 << 30) as f64
            );
            if total >= 180usize << 30 {
                eprintln!("[pinned-probe] reached 180 GiB without failing — stopping");
                break;
            }
        }
        let pinned_ceiling = total;
        for p in blocks {
            unsafe {
                let _ = cudarc::driver::sys::cuMemFreeHost(p);
            }
        }
        // Now show PAGEABLE RAM well past the pinned ceiling is fine (proves it's page-locking).
        let pageable_target = pinned_ceiling + (16usize << 30);
        let mut bufs: Vec<Vec<u8>> = Vec::new();
        let mut pg = 0usize;
        while pg < pageable_target {
            bufs.push(vec![0u8; CHUNK]); // touch via zero-init so pages are committed
            pg += CHUNK;
        }
        eprintln!(
            "[pinned-probe] PAGEABLE allocated {:.1} GiB (past the {:.1} GiB pinned ceiling) — the \
             wall is page-locking, not RAM",
            pg as f64 / (1u64 << 30) as f64,
            pinned_ceiling as f64 / (1u64 << 30) as f64,
        );
    }
}
