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
    /// Base pointer from `cuMemAllocHost`.
    base: *mut u8,
    /// Total allocation size in bytes.
    #[allow(dead_code)]
    total_size: usize,
    /// Per-slot byte size (uniform, = max repacked expert size across layers).
    slot_size: usize,
    /// Number of slots.
    num_slots: usize,
    /// Free slot indices (LIFO stack for O(1) alloc/free).
    pub(crate) free_slots: Vec<usize>,
}

#[cfg(feature = "cuda")]
impl PinnedPool {
    /// Allocate a pinned memory pool with `num_slots` × `slot_size` bytes.
    ///
    /// Uses `cuMemAllocHost` to physically lock the pages, enabling the
    /// GPU DMA engine to access them without OS page faults.
    pub(crate) fn new(num_slots: usize, slot_size: usize) -> Result<Self> {
        if num_slots == 0 || slot_size == 0 {
            return Ok(Self {
                base: std::ptr::null_mut(),
                total_size: 0,
                slot_size,
                num_slots: 0,
                free_slots: Vec::new(),
            });
        }

        let total_size = num_slots * slot_size;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();

        let result = unsafe { cudarc::driver::sys::cuMemAllocHost_v2(&mut ptr, total_size) };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            candle::bail!(
                "cuMemAllocHost failed: {:?} (requested {} slots × {} bytes = {:.1} GB)",
                result,
                num_slots,
                slot_size,
                total_size as f64 / 1e9,
            );
        }

        tracing::info!(
            "PinnedPool: allocated {:.1} GB pinned RAM ({} slots × {:.1} KB)",
            total_size as f64 / 1e9,
            num_slots,
            slot_size as f64 / 1e3,
        );

        Ok(Self {
            base: ptr as *mut u8,
            total_size,
            slot_size,
            num_slots,
            free_slots: (0..num_slots).rev().collect(),
        })
    }

    /// Get a mutable byte slice for a slot.
    ///
    /// `len` must be ≤ `slot_size`.
    #[inline]
    pub(crate) fn slot_mut(&mut self, slot_idx: usize, len: usize) -> &mut [u8] {
        debug_assert!(slot_idx < self.num_slots);
        debug_assert!(len <= self.slot_size);
        unsafe {
            let ptr = self.base.add(slot_idx * self.slot_size);
            std::slice::from_raw_parts_mut(ptr, len)
        }
    }

    /// Get a shared byte slice for a slot.
    #[inline]
    pub(crate) fn slot_ref(&self, slot_idx: usize, len: usize) -> &[u8] {
        debug_assert!(slot_idx < self.num_slots);
        debug_assert!(len <= self.slot_size);
        unsafe {
            let ptr = self.base.add(slot_idx * self.slot_size);
            std::slice::from_raw_parts(ptr, len)
        }
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

    /// Create an empty pool (no allocation).
    ///
    /// Used when running on a non-CUDA device but the `cuda` feature is enabled.
    pub(crate) fn empty() -> Self {
        Self {
            base: std::ptr::null_mut(),
            total_size: 0,
            slot_size: 0,
            num_slots: 0,
            free_slots: Vec::new(),
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for PinnedPool {
    fn drop(&mut self) {
        if !self.base.is_null() {
            let result = unsafe {
                cudarc::driver::sys::cuMemFreeHost(self.base as *mut std::ffi::c_void)
            };
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
