// Kernel-launch wrappers mirror their CUDA signatures argument for argument —
// pointers, extents, strides, scales, stream. Grouping them into structs would
// put a shape between the call site and the kernel it is a transcription of,
// which is the opposite of what makes these auditable against the `.cu`.
#![allow(clippy::too_many_arguments)]

use super::{GgmlDType, Int8Mode, QStorage};
use crate::backend::{BackendDevice, BackendStorage};

use crate::cuda_backend::alloc_inheriting;
use crate::cuda_backend::wave_provenance::{wave_alloc, LeaseOrigin};
use crate::cuda_backend::Backing;
use crate::cuda_backend::INHERIT_ALIGN;
use crate::quantized::k_quants::GgmlType;
use crate::LiveTensor;
use crate::{CudaDevice, CudaStorage, Result, Shape};
use half::{bf16, f16};

use crate::cuda_backend::WrapErr;
use cudarc::driver::{CudaSlice, CudaView, DevicePtr, DevicePtrMut};

// Import the FFI dispatcher functions
use candle_kernels::simple::quantized::{
    run_dequantize_block, run_dequantize_ko, run_dequantize_mul_mat_vec, run_dequantize_q8a128,
    run_mul_mat, run_mul_mat_vec_q8_1, run_quantize_block, run_quantize_ko,
    run_quantize_palette4_convert, run_quantize_q8_1, run_quantize_q8a128,
    run_quantize_transposed_batched, run_quantize_transposed_batched_typed,
    run_reduce_head_stats_format, run_sample_quant_errors_kv_paged, run_sample_quant_errors_paged,
    run_select_kv_format_palette4_paged, run_select_winners_kv_paged,
    run_summarize_winners_side_paged, DequantOutDType, QType,
};

// Fused RMSNorm → q8a128 producer epilogue (B1/B3/B5).
use candle_kernels::simple::reduce::run_rmsnorm_q8a128_op;

// Fused SwiGLU → q8a128 producer epilogue (B4).
use candle_kernels::simple::fused_silu_mul::run_silu_mul_q8a128_op;

// Import the new quantized matmul dispatcher
// K/128 blocks have embedded scales, no external scale extraction needed.
use candle_kernels::quantized::{
    dispatch_info, flush_l2_cache, run_grouped_quantized_matmul, run_qkv_segmented_matmul,
    run_quantized_matmul, MatmulStatus, OutDType, VxSegment, YType,
};

// Import GEMX repacking dispatcher
use candle_kernels::quantized::{get_repacked_size_bytes, is_gemx_supported, run_repack_gemx};

use super::int8_matmul_mode::q8a128_dense_use_mode2;
use super::table_ring::table_ring;

/// Process-cached SM count for the int8 dense tiling (occupancy) heuristic. SM count is a fixed
/// device property; querying the driver attribute on every matmul would add an FFI call to the hot
/// path, so memoize it once. Single-GPU assumption (this fork targets one accelerator) — a tiling
/// heuristic does not need per-device precision, and `0` (query failure) degrades safely (mode-2
/// outside the trap).
fn cached_sm_count(device: &CudaDevice) -> usize {
    use std::sync::OnceLock;
    static SM_COUNT: OnceLock<usize> = OnceLock::new();
    *SM_COUNT.get_or_init(|| device.multiprocessor_count().unwrap_or(0))
}

/// Process-cached L2 size for the grouped grid-order choice — same memoization
/// rationale (fixed device property, hot-path caller) as [`cached_sm_count`].
/// `0` on query failure degrades safely: every activation then "exceeds" L2 and
/// the launch takes the row-fast order, which is the safe default.
fn cached_l2_bytes(device: &CudaDevice) -> usize {
    use std::sync::OnceLock;
    static L2_BYTES: OnceLock<usize> = OnceLock::new();
    *L2_BYTES.get_or_init(|| device.l2_cache_size().unwrap_or(0))
}

/// Grid axis order for a grouped launch (see `quantized_matmul_grouped_entry`):
/// row-tiles-fast whenever the stacked activation cannot stay L2-resident
/// alongside the streaming weights — token-tiles-fast would then re-stream the
/// whole activation from DRAM once per row-tile wave (measured 6× DRAM traffic
/// amplification at prefill scale). When the activation comfortably fits L2,
/// residency is free under either order and token-tiles-fast keeps consecutive
/// blocks on the same weight rows instead. The half-L2 threshold leaves the
/// other half for the weight stream, and `moe_layer_gemm_bench` (both layouts ×
/// decode/cfg8/cfg20, 96 MiB L2) confirms it classifies every measured band
/// correctly: token-fast wins at 2.7 MB (decode, +5%) and 48 MB (down cfg8,
/// +5%); row-fast wins from 100 MB up (+28..40%). Mis-choosing costs 5% on the
/// token-fast side of the line and 40% on the row-fast side, so the threshold
/// deliberately sits well below the row-fast danger zone. Both orders are
/// bit-identical (schedule only) — the crossover is a pure performance band.
pub(crate) fn grouped_grid_row_fast(act_bytes: usize, device: &CudaDevice) -> bool {
    act_bytes * 2 > cached_l2_bytes(device)
}

// ============================================================================
// Host-Mapped / Pinned Memory Primitives
// ============================================================================

/// Guard for CUDA-registered mmap memory. Automatically unregisters on drop.
///
/// Created by [`register_mmap_cuda`]. Ensures `cuMemHostUnregister` is called
/// when the guard is dropped, even if the mmap outlives it.
pub struct MmapRegistration {
    ptr: *mut std::ffi::c_void,
}

// Safety: the pointer is only used in Drop to unregister; the registration
// itself is thread-safe in the CUDA driver.
unsafe impl Send for MmapRegistration {}
unsafe impl Sync for MmapRegistration {}

impl Drop for MmapRegistration {
    fn drop(&mut self) {
        use cudarc::driver::sys;
        unsafe {
            let _ = sys::cuMemHostUnregister(self.ptr).result();
        }
    }
}

/// Guard for `cudaHostAlloc`-allocated memory. Calls `cudaFreeHost` on drop.
///
/// Created by [`alloc_host_mapped`]. The memory is GPU-accessible via PCIe
/// (allocated with `CU_MEMHOSTALLOC_DEVICEMAP`).
pub struct HostMappedAlloc {
    host_ptr: *mut std::ffi::c_void,
    size: usize,
}

// Safety: the pinned memory is accessible from any thread; the host pointer
// is stable for the lifetime of the allocation.
unsafe impl Send for HostMappedAlloc {}
unsafe impl Sync for HostMappedAlloc {}

impl Drop for HostMappedAlloc {
    fn drop(&mut self) {
        unsafe {
            let _ = cudarc::driver::sys::cuMemFreeHost(self.host_ptr).result();
        }
        crate::vram::note_host_pinned_free(self.size as u64);
    }
}

/// Register an mmap with CUDA for DMA-accelerated host-to-device transfers.
///
/// On success, returns a [`MmapRegistration`] guard that unregisters the memory
/// on drop. Returns `None` if registration fails (e.g. alignment issues, driver
/// limitations). Callers should fall back to regular `memcpy` in that case.
///
/// The mmap is registered as read-only and device-mapped, so the GPU can read
/// from it directly over PCIe.
pub fn register_mmap_cuda(mmap: &memmap2::Mmap) -> Option<MmapRegistration> {
    use cudarc::driver::sys;
    let ptr = mmap.as_ptr() as *mut std::ffi::c_void;
    let len = mmap.len();
    let register_result = unsafe {
        sys::cuMemHostRegister_v2(
            ptr,
            len,
            sys::CU_MEMHOSTREGISTER_DEVICEMAP | sys::CU_MEMHOSTREGISTER_READ_ONLY,
        )
    };
    match register_result.result() {
        Ok(_) => Some(MmapRegistration { ptr }),
        Err(e) => {
            // Do not swallow this. Callers degrade to a slower path when
            // registration fails, so a silent `None` turns a driver-level
            // refusal into an unexplained performance difference that looks
            // like a tuning problem.
            tracing::warn!(
                target: "candle_core::quantized::cuda",
                len,
                "cuMemHostRegister(DEVICEMAP|READ_ONLY) failed ({e:?}); \
                 falling back to staged host-to-device copies"
            );
            None
        }
    }
}

/// Allocate host memory that is GPU-accessible via PCIe (`cudaHostAllocMapped`).
///
/// Returns `(host_ptr, device_ptr, guard)`. The `guard` frees the memory on drop.
/// GPU kernels use `device_ptr` transparently — hardware handles PCIe transfers.
///
/// This is the building block for VRAM-overflow weight storage: tensors that
/// don't fit in VRAM can live in pinned host memory and still be used by CUDA
/// kernels (at PCIe bandwidth instead of VRAM bandwidth).
///
/// Allocating here is not interchangeable with registering memory you already
/// have. On WDDM, [`register_mmap_cuda`] succeeds on a multi-gigabyte model
/// mmap and `cuMemHostGetDevicePointer` then *refuses* that range, so a
/// registered mmap is not kernel-addressable however large it is. Forcing the
/// host pointer through anyway — on the theory that unified addressing makes it
/// equivalent — faults with `CUDA_ERROR_ILLEGAL_ADDRESS` on the first forward
/// and poisons the context. Anything a kernel must dereference from host memory
/// has to be copied into an allocation from this function.
pub fn alloc_host_mapped(size: usize) -> Result<(*mut u8, u64, HostMappedAlloc)> {
    use cudarc::driver::sys;
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe {
        // CU_MEMHOSTALLOC_DEVICEMAP = 0x02
        // The size and the standing pinned total are in the message because an
        // out-of-memory without them is unactionable: pinned allocations fail
        // against what every *other* pinned claim in the process has already
        // taken, and the amount asked for is the only clue to which caller this
        // was.
        sys::cuMemHostAlloc(&mut host_ptr, size, 0x02)
            .result()
            .map_err(|e| {
                crate::Error::Msg(format!(
                    "cuMemHostAlloc failed for {:.1} MiB (pinned already held: {:.1} GiB): {:?}",
                    size as f64 / (1024.0 * 1024.0),
                    crate::vram::host_pinned_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                    e,
                ))
            })?;
        let mut dev_ptr: sys::CUdeviceptr = 0;
        let res = sys::cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0);
        if let Err(e) = res.result() {
            let _ = sys::cuMemFreeHost(host_ptr).result();
            crate::bail!("cuMemHostGetDevicePointer failed: {:?}", e);
        }
        // Non-pageable by construction, so the host-RAM budget must count it as
        // structural — tracked here rather than at each call site, so no caller
        // can allocate gigabytes of pinned RAM the budget reads as pageable.
        crate::vram::note_host_pinned_alloc(size as u64);
        let guard = HostMappedAlloc { host_ptr, size };
        Ok((host_ptr as *mut u8, dev_ptr, guard))
    }
}

/// Total VRAM on CUDA device 0, queried without requiring a bound context.
///
/// `cuDeviceTotalMem` needs only driver init and a device handle — unlike
/// [`get_vram_info`], which reads the *current context's* device — so this is
/// safe to call from any thread before any `Device` exists (e.g. when picking
/// which model quant to download at daemon startup).
pub fn get_total_vram_device0() -> Result<usize> {
    use cudarc::driver::sys;
    unsafe {
        sys::cuInit(0)
            .result()
            .map_err(|e| crate::Error::Msg(format!("cuInit failed: {:?}", e)))?;
        let mut dev: sys::CUdevice = 0;
        sys::cuDeviceGet(&mut dev, 0)
            .result()
            .map_err(|e| crate::Error::Msg(format!("cuDeviceGet failed: {:?}", e)))?;
        let mut total: usize = 0;
        sys::cuDeviceTotalMem_v2(&mut total, dev)
            .result()
            .map_err(|e| crate::Error::Msg(format!("cuDeviceTotalMem failed: {:?}", e)))?;
        Ok(total)
    }
}

/// Query total and free VRAM on the current CUDA device.
///
/// Returns `(free_bytes, total_bytes)`.
pub fn get_vram_info() -> Result<(usize, usize)> {
    use cudarc::driver::sys;
    let mut free: usize = 0;
    let mut total: usize = 0;
    unsafe {
        let res = sys::cuMemGetInfo_v2(&mut free, &mut total);
        res.result()
            .map_err(|e| crate::Error::Msg(format!("cuMemGetInfo failed: {:?}", e)))?;
    }
    Ok((free, total))
}

#[derive(Clone, Debug)]
struct PaddedCudaSlice {
    inner: CudaSlice<u8>,
    len: usize,
}

#[derive(Debug)]
pub struct QCudaStorage {
    /// `ManuallyDrop` so [`Drop`] can move the slice out without constructing a
    /// stand-in. `leak` is by-value while `drop` has only `&mut self`, and the
    /// stand-in used to come from `upgrade_device_ptr(0, 0)` — which is not
    /// free: under cudarc's event tracking (on for every production device) it
    /// creates and destroys two `CudaEvent`s and can fail, and it `unwrap`s.
    /// Fallible CUDA work on a drop path panics *inside* `Drop`, which aborts
    /// during an unwind. `CudaStorage` avoids this with its `Empty` slice
    /// variant; this is the same trick for a struct that has no spare variant.
    ///
    /// Every read still goes through `Deref`, so `self.data.inner` is unchanged
    /// at the ~18 use sites. The obligation this adds is that **every path that
    /// destroys or replaces `data` must dispose of the old value explicitly** —
    /// [`Drop`] and [`Self::quantize`] are the only two.
    data: std::mem::ManuallyDrop<PaddedCudaSlice>,
    dtype: GgmlDType,
    device: CudaDevice,
    /// Whether dropping this storage may free its device memory.
    /// [`Backing::Owned`] for everything allocated here — which is everything
    /// except the slot views built by [`QCudaStorage::from_leased_device_ptr`].
    backing: Backing,
}

impl Clone for QCudaStorage {
    /// A clone is always **owned**, never a second lease.
    ///
    /// `CudaSlice::clone` is a device-to-device copy, so a clone has its own
    /// allocation regardless of what the source was. Carrying `Lease` across
    /// would leak that fresh allocation on every clone.
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            dtype: self.dtype,
            device: self.device.clone(),
            backing: Backing::Owned,
        }
    }
}

impl Drop for QCudaStorage {
    fn drop(&mut self) {
        // Exhaustive rather than `!= Lease`, matching `CudaStorage::drop`: a
        // future `Backing` variant must state which side it falls on here rather
        // than silently inheriting the free path.
        match self.backing {
            Backing::Owned => {
                // Dispose normally. `data` is `ManuallyDrop`, so without this the
                // allocation is never freed.
                // SAFETY: `self` is being destroyed; `data` is not read again and
                // is dropped exactly once (this arm and the lease arm below are
                // mutually exclusive and both return).
                unsafe { std::mem::ManuallyDrop::drop(&mut self.data) };
                return;
            }
            Backing::Lease(_) => {}
        }
        // Same discipline as `CudaStorage::drop`: calling `leak` rather than
        // merely suppressing the drop is load-bearing — it waits on the
        // slice's read/write events, destroys them, and decrements the
        // stream's `Arc`. Bare suppression would strand two `CudaEvent`s and a
        // stream refcount per lease, and leases are minted per band per
        // forward.
        //
        // SAFETY: `self` is being destroyed and `data` is never read again;
        // `ManuallyDrop::take` is the move-out that `leak`'s by-value signature
        // needs, and it touches no CUDA API on the way (see the field's note).
        let data = unsafe { std::mem::ManuallyDrop::take(&mut self.data) };
        data.inner.leak();
    }
}

impl QCudaStorage {
    /// Wrap `elem_count` elements of quantized device memory at `ptr` as a
    /// storage that does **not** own them.
    ///
    /// This is the quantized counterpart of
    /// [`CudaStorage::from_leased_device_ptr`], and it exists for exactly one
    /// caller: a KV arena slot. Under size classes an arena is a run of untyped
    /// byte slots, so the quantize / dequantize kernels can no longer be handed
    /// "the arena's QTensor" — they are handed a view of one slot instead. See
    /// `docs/archived/arena_unification.md` principle 8.
    ///
    /// The view carries **no matrix-row padding**, unlike
    /// [`Self::zeros`]: a slot is exactly its own bytes and the next slot
    /// belongs to another chunk. That makes it unsuitable for the matmul
    /// kernels, which read into the padding — and correct for the block
    /// quantize / dequantize paths, which do not.
    ///
    /// # Safety
    /// `ptr` must point to at least `ceil(elem_count / block_size) * type_size`
    /// bytes of device memory that stays live, and un-aliased for writes, for
    /// the storage's lifetime.
    pub unsafe fn from_leased_device_ptr(
        ptr: u64,
        elem_count: usize,
        dtype: GgmlDType,
        device: &CudaDevice,
        origin: LeaseOrigin,
    ) -> Result<Self> {
        if !elem_count.is_multiple_of(dtype.block_size()) {
            crate::bail!(
                "leased quantized view of {elem_count} elements is not a whole number of \
                 {dtype:?} blocks ({})",
                dtype.block_size()
            );
        }
        let size_in_bytes = (elem_count / dtype.block_size()) * dtype.type_size();
        let inner = device
            .cuda_stream()
            .upgrade_device_ptr::<u8>(ptr, size_in_bytes);
        Ok(QCudaStorage {
            data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            }),
            device: device.clone(),
            dtype,
            backing: Backing::Lease(origin),
        })
    }
}

static FORCE_DMMV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

/// Get a human-readable description of the kernel dispatch plan.
///
/// Returns a string like "s2i8(16)" or "tc32(32)+s8(8)" describing which
/// kernels will be used for the given batch size and weight tensor size.
///
/// # Arguments
/// * `batch_size` - Number of vectors to process
/// * `weight_bytes` - Weight tensor size in bytes (determines L2 vs DRAM path)
///
/// # Returns
/// A String describing the dispatch plan.
pub fn get_dispatch_info(batch_size: i32, weight_bytes: usize) -> String {
    dispatch_info(batch_size, weight_bytes)
}

/// Flush L2 cache using a CUDA kernel.
///
/// This is useful for benchmarking to simulate realistic cache conditions
/// where different matrices alternate and cannot all fit in L2 cache.
///
/// # Arguments
/// * `buffer` - A device buffer larger than L2 cache (recommend 2x L2 size)
/// * `device` - The CUDA device
///
/// The buffer should be created once and reused. This function synchronizes
/// the device before returning.
pub fn cuda_flush_l2(buffer: &CudaSlice<u8>, device: &CudaDevice) {
    let stream = device.cuda_stream();
    let (ptr, _guard) = buffer.device_ptr(&stream);
    let size = buffer.len();
    unsafe {
        flush_l2_cache(ptr as *const core::ffi::c_void, size);
    }
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
pub const GGML_CUDA_MMV_X: usize = 32;
pub const GGML_CUDA_MMV_Y: usize = 1;
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

/// Convert GgmlDType to QType for dispatcher
fn dtype_to_qtype(dtype: GgmlDType) -> Result<QType> {
    Ok(match dtype {
        GgmlDType::Q4_0 => QType::Q4_0,
        GgmlDType::Q4_1 => QType::Q4_1,
        GgmlDType::Q5_0 => QType::Q5_0,
        GgmlDType::Q5_1 => QType::Q5_1,
        GgmlDType::Q8_0 => QType::Q8_0,
        GgmlDType::Q8_1 => QType::Q8_1,
        GgmlDType::Q2_K => QType::Q2_K,
        GgmlDType::Q3_K => QType::Q3_K,
        GgmlDType::Q4_K => QType::Q4_K,
        GgmlDType::Q5_K => QType::Q5_K,
        GgmlDType::Q6_K => QType::Q6_K,
        GgmlDType::Q8_K => QType::Q8_K,
        GgmlDType::QAWQ => QType::QAWQ,
        GgmlDType::QAWQ_G64 => QType::QAWQ_G64,
        GgmlDType::Q4_KS => QType::Q4_KS,
        GgmlDType::Q8_KS => QType::Q8_KS,
        GgmlDType::Q2_0 => QType::Q2_0,
        GgmlDType::Q3_0 => QType::Q3_0,
        GgmlDType::R16 => QType::R16,
        GgmlDType::Q0 => QType::Q0,
        GgmlDType::Q1_S => QType::Q1_S,
        GgmlDType::Q2_S => QType::Q2_S,
        GgmlDType::Q2_A => QType::Q2_A,
        GgmlDType::Q2_1 => QType::Q2_1,
        GgmlDType::Q3_1 => QType::Q3_1,
        GgmlDType::Q0_V => QType::Q0_V,
        GgmlDType::Q1_A => QType::Q1_A,
        GgmlDType::Q0_X => QType::Q0_X,
        GgmlDType::Q0_M2 => QType::Q0_M2,
        GgmlDType::Q4_KO => QType::Q4_KO,
        GgmlDType::Q5_KO => QType::Q5_KO,
        GgmlDType::Q6_KO => QType::Q6_KO,
        GgmlDType::Q8_KO => QType::Q8_KO,
        GgmlDType::MXFP4_KO => QType::MXFP4_KO,
        GgmlDType::Q2_KO => QType::Q2_KO,
        GgmlDType::Q0_M4 => QType::Q0_M4,
        _ => crate::bail!("unsupported dtype for quantized op: {dtype:?}"),
    })
}

fn quantize_q8_1(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    elem_count: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let stream = dev.cuda_stream();
    let (src_ptr, _src_guard) = src.device_ptr(&stream);
    let (dst_ptr, _dst_guard) = dst.device_ptr_mut(&stream);
    unsafe {
        run_quantize_q8_1(
            src_ptr as *const f32,
            dst_ptr as *mut std::ffi::c_void,
            elem_count as i32,
            ky as i32,
        );
    }
    Ok(())
}

/// Quantize f32 data to any supported quantized format on GPU.
///
/// # Arguments
/// * `src` - Source f32 data view
/// * `dst` - Destination buffer (must be pre-allocated with correct size)
/// * `elem_count` - Total number of f32 elements to quantize
/// * `dtype` - Target quantized dtype (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K, QAWQ, QAWQG64)
/// * `dev` - CUDA device
///
/// # Returns
/// Result indicating success or error
pub fn quantize_to_dtype(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    elem_count: usize,
    dtype: GgmlDType,
    dev: &CudaDevice,
) -> Result<()> {
    let qtype = dtype_to_qtype(dtype)?;
    let stream = dev.cuda_stream();
    let (src_ptr, _src_guard) = src.device_ptr(&stream);
    let (dst_ptr, _dst_guard) = dst.device_ptr_mut(&stream);
    unsafe {
        run_quantize_block(
            src_ptr as *const f32,
            dst_ptr as *mut std::ffi::c_void,
            elem_count as i32,
            qtype as i32,
        );
    }
    Ok(())
}

/// Quantize f32 data with fused transpose from [H, T, D] to [H, D, T] layout on GPU.
///
/// This fuses the memory layout transformation with quantization to avoid
/// intermediate allocations. Used for KV cache quantization where:
/// - Input layout: [n_head, chunk_size, head_dim] - channel-oriented float
/// - Output layout: [n_head, head_dim, chunk_size] - token-oriented quant
///
/// # Arguments
/// * `src` - Source f32 tensor with shape [n_head, chunk_size, head_dim]
/// * `dst` - Destination buffer (must be pre-allocated for n_head * head_dim quantized blocks)
/// * `n_head` - Number of KV heads
/// * `chunk_size` - Number of tokens (must be 32 for standard quants)
/// * `head_dim` - Dimension per head
/// * `dtype` - Target quantized dtype (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
/// * `dev` - CUDA device
///
/// # Returns
/// Result indicating success or error
pub fn quantize_transposed_to_dtype(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    n_head: usize,
    chunk_size: usize,
    head_dim: usize,
    dtype: GgmlDType,
    dev: &CudaDevice,
) -> Result<()> {
    // Standard 32-element formats support fused transpose+quantize
    let qtype = match dtype {
        GgmlDType::Q4_0 => QType::Q4_0,
        GgmlDType::Q4_1 => QType::Q4_1,
        GgmlDType::Q5_0 => QType::Q5_0,
        GgmlDType::Q5_1 => QType::Q5_1,
        GgmlDType::Q8_0 => QType::Q8_0,
        GgmlDType::Q8_1 => QType::Q8_1,
        GgmlDType::Q4_KS => QType::Q4_KS,
        GgmlDType::Q8_KS => QType::Q8_KS,
        GgmlDType::Q2_0 => QType::Q2_0,
        GgmlDType::Q3_0 => QType::Q3_0,
        GgmlDType::R16 => QType::R16,
        GgmlDType::Q0 => QType::Q0,
        GgmlDType::Q1_S => QType::Q1_S,
        GgmlDType::Q2_S => QType::Q2_S,
        GgmlDType::Q2_A => QType::Q2_A,
        GgmlDType::Q2_1 => QType::Q2_1,
        GgmlDType::Q3_1 => QType::Q3_1,
        _ => crate::bail!("quantize_transposed_to_dtype: only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q4_KS/Q8_KS supported, got {:?}", dtype),
    };

    // Validate chunk_size matches block size (all standard quants use 32)
    if chunk_size != 32 {
        crate::bail!(
            "quantize_transposed_to_dtype: chunk_size must be 32 for standard quants, got {}",
            chunk_size
        );
    }

    let stream = dev.cuda_stream();
    let (src_ptr, _src_guard) = src.device_ptr(&stream);
    let (dst_ptr, _dst_guard) = dst.device_ptr_mut(&stream);
    unsafe {
        // Use batched version with num_chunks=1
        run_quantize_transposed_batched(
            src_ptr as *const f32,
            dst_ptr as *mut std::ffi::c_void,
            std::ptr::null(), // No src offsets (contiguous)
            std::ptr::null(), // No dst offsets (contiguous)
            1,                // Single chunk
            n_head as i32,
            chunk_size as i32,
            head_dim as i32,
            qtype as i32,
        );
    }
    Ok(())
}

/// Batched quantize f32 data with fused transpose for multiple chunks.
///
/// Processes multiple chunks in a single kernel launch for efficient KV cache migration.
/// Fuses the memory layout transformation with quantization:
/// - Input layout: [num_chunks, n_head, chunk_size, head_dim] - channel-oriented float
/// - Output layout: [num_chunks, n_head, head_dim] Q blocks - token-oriented quant
///
/// # Arguments
/// * `src` - Source f32 data pointer (contiguous for all chunks or use offsets)
/// * `dst` - Destination quantized data pointer
/// * `src_offsets` - Per-chunk element offsets into src (None for contiguous)
/// * `dst_offsets` - Per-chunk byte offsets into dst (None for contiguous)
/// * `num_chunks` - Number of chunks to process
/// * `n_head` - Number of KV heads per chunk
/// * `chunk_size` - Tokens per chunk (must be 32 for standard quants)
/// * `head_dim` - Dimension per head
/// * `dtype` - Target dtype (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
/// * `dev` - CUDA device
///
/// # Safety
/// The caller must ensure that `src` and `dst` are valid GPU pointers with
/// sufficient allocated memory for the operation.
#[allow(clippy::too_many_arguments)]
pub unsafe fn quantize_transposed_batched_to_dtype(
    src: *const f32,
    dst: *mut u8,
    src_offsets: Option<&CudaSlice<i32>>,
    dst_offsets: Option<&CudaSlice<i32>>,
    num_chunks: usize,
    n_head: usize,
    chunk_size: usize,
    head_dim: usize,
    dtype: GgmlDType,
    dev: &CudaDevice,
) -> Result<()> {
    // Only standard 32-element quants are supported
    let qtype = match dtype {
        GgmlDType::Q4_0 => QType::Q4_0,
        GgmlDType::Q4_1 => QType::Q4_1,
        GgmlDType::Q5_0 => QType::Q5_0,
        GgmlDType::Q5_1 => QType::Q5_1,
        GgmlDType::Q8_0 => QType::Q8_0,
        GgmlDType::Q8_1 => QType::Q8_1,
        GgmlDType::Q4_KS => QType::Q4_KS,
        GgmlDType::Q8_KS => QType::Q8_KS,
        GgmlDType::Q2_0 => QType::Q2_0,
        GgmlDType::Q3_0 => QType::Q3_0,
        GgmlDType::R16 => QType::R16,
        GgmlDType::Q0 => QType::Q0,
        GgmlDType::Q1_S => QType::Q1_S,
        GgmlDType::Q2_S => QType::Q2_S,
        GgmlDType::Q2_A => QType::Q2_A,
        GgmlDType::Q2_1 => QType::Q2_1,
        GgmlDType::Q3_1 => QType::Q3_1,
        GgmlDType::Q0_V => QType::Q0_V,
        GgmlDType::Q1_A => QType::Q1_A,
        GgmlDType::Q0_X => QType::Q0_X,
        GgmlDType::Q0_M2 => QType::Q0_M2,
        GgmlDType::Q0_M4 => QType::Q0_M4,
        _ => crate::bail!(
            "quantize_transposed_batched: only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q4_KS/Q8_KS supported, got {:?}",
            dtype
        ),
    };

    // Validate chunk_size matches block size
    if chunk_size != 32 {
        crate::bail!(
            "quantize_transposed_batched: chunk_size must be 32, got {}",
            chunk_size
        );
    }

    let stream = dev.cuda_stream();

    // Get src_offsets pointer (or null for contiguous)
    let src_off_ptr = if let Some(offsets) = src_offsets {
        let (ptr, _guard) = offsets.device_ptr(&stream);
        ptr as *const i32
    } else {
        std::ptr::null()
    };

    // Get dst_offsets pointer (or null for contiguous)
    let dst_off_ptr = if let Some(offsets) = dst_offsets {
        let (ptr, _guard) = offsets.device_ptr(&stream);
        ptr as *const i32
    } else {
        std::ptr::null()
    };

    run_quantize_transposed_batched(
        src,
        dst as *mut std::ffi::c_void,
        src_off_ptr,
        dst_off_ptr,
        num_chunks as i32,
        n_head as i32,
        chunk_size as i32,
        head_dim as i32,
        qtype as i32,
    );
    Ok(())
}

/// Convert candle DType to kernel SrcDType code.
/// Returns: 0=F32, 1=F16, 2=BF16, 3=F8E4M3
/// Map a candle `DType` to the integer src-dtype code expected by the
/// `transpose_quant_batch_typed` CUDA dispatcher. The codes match
/// `GgmlDType` / `QType` / `ArenaFormat` values so the Rust side never
/// has to translate between two numbering schemes — the CUDA enum
/// `SrcDType` in `transpose_batch.cuh` uses the same values.
pub fn dtype_to_src_dtype_code(dtype: crate::DType) -> Result<i32> {
    match dtype {
        crate::DType::F32 => Ok(GgmlDType::F32 as i32),   // 0
        crate::DType::F16 => Ok(GgmlDType::F16 as i32),   // 1
        crate::DType::BF16 => Ok(GgmlDType::BF16 as i32), // 2
        crate::DType::F8E4M3 => Ok(GgmlDType::F8E4M3 as i32), // 34
        _ => crate::bail!(
            "quantize_transposed: source dtype must be F32/F16/BF16/F8E4M3, got {:?}",
            dtype
        ),
    }
}

/// Convert candle DType to GgmlDType for float source types.
pub fn dtype_to_ggml_float(dtype: crate::DType) -> Result<GgmlDType> {
    match dtype {
        crate::DType::F32 => Ok(GgmlDType::F32),
        crate::DType::F16 => Ok(GgmlDType::F16),
        crate::DType::BF16 => Ok(GgmlDType::BF16),
        _ => crate::bail!("dtype_to_ggml_float: expected float dtype, got {:?}", dtype),
    }
}

/// Batched quantize typed data with fused transpose for multiple chunks.
///
/// Like `quantize_transposed_batched_to_dtype` but accepts any supported source dtype
/// (F32, F16, BF16, F8E4M3) and converts inline in the kernel.
///
/// # Arguments
/// * `src` - Source data pointer (F32/F16/BF16/F8E4M3)
/// * `src_dtype` - Source data type
/// * `dst` - Destination quantized data pointer
/// * `src_offsets` - Per-chunk element offsets into src (None for contiguous)
/// * `dst_offsets` - Per-chunk byte offsets into dst (None for contiguous)
/// * `num_chunks` - Number of chunks to process
/// * `n_head` - Number of KV heads per chunk
/// * `chunk_size` - Tokens per chunk (must be 32 for standard quants)
/// * `head_dim` - Dimension per head
/// * `dtype` - Target quantized dtype (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
/// * `dev` - CUDA device
///
/// # Safety
/// The caller must ensure that `src` and `dst` are valid GPU pointers with
/// sufficient allocated memory for the operation.
#[allow(clippy::too_many_arguments)]
pub unsafe fn quantize_transposed_batched_typed(
    src: *const std::ffi::c_void,
    src_dtype: crate::DType,
    dst: *mut u8,
    src_offsets: Option<&CudaSlice<i32>>,
    dst_offsets: Option<&CudaSlice<i32>>,
    num_chunks: usize,
    n_head: usize,
    chunk_size: usize,
    head_dim: usize,
    dtype: GgmlDType,
    dev: &CudaDevice,
) -> Result<()> {
    // Only standard 32-element quants are supported
    let qtype = match dtype {
        GgmlDType::Q4_0 => QType::Q4_0,
        GgmlDType::Q4_1 => QType::Q4_1,
        GgmlDType::Q5_0 => QType::Q5_0,
        GgmlDType::Q5_1 => QType::Q5_1,
        GgmlDType::Q8_0 => QType::Q8_0,
        GgmlDType::Q8_1 => QType::Q8_1,
        GgmlDType::Q4_KS => QType::Q4_KS,
        GgmlDType::Q8_KS => QType::Q8_KS,
        GgmlDType::Q2_0 => QType::Q2_0,
        GgmlDType::Q3_0 => QType::Q3_0,
        GgmlDType::R16 => QType::R16,
        GgmlDType::Q0 => QType::Q0,
        GgmlDType::Q1_S => QType::Q1_S,
        GgmlDType::Q2_S => QType::Q2_S,
        GgmlDType::Q2_A => QType::Q2_A,
        GgmlDType::Q2_1 => QType::Q2_1,
        GgmlDType::Q3_1 => QType::Q3_1,
        GgmlDType::Q0_V => QType::Q0_V,
        GgmlDType::Q1_A => QType::Q1_A,
        GgmlDType::Q0_X => QType::Q0_X,
        GgmlDType::Q0_M2 => QType::Q0_M2,
        GgmlDType::Q0_M4 => QType::Q0_M4,
        _ => crate::bail!(
            "quantize_transposed_batched: only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q4_KS/Q8_KS supported, got {:?}",
            dtype
        ),
    };

    let src_dtype_code = dtype_to_src_dtype_code(src_dtype)?;

    // Validate chunk_size matches block size
    if chunk_size != 32 {
        crate::bail!(
            "quantize_transposed_batched: chunk_size must be 32, got {}",
            chunk_size
        );
    }

    let stream = dev.cuda_stream();

    // Get src_offsets pointer (or null for contiguous)
    let src_off_ptr = if let Some(offsets) = src_offsets {
        let (ptr, _guard) = offsets.device_ptr(&stream);
        ptr as *const i32
    } else {
        std::ptr::null()
    };

    // Get dst_offsets pointer (or null for contiguous)
    let dst_off_ptr = if let Some(offsets) = dst_offsets {
        let (ptr, _guard) = offsets.device_ptr(&stream);
        ptr as *const i32
    } else {
        std::ptr::null()
    };

    run_quantize_transposed_batched_typed(
        src,
        dst as *mut std::ffi::c_void,
        src_off_ptr,
        dst_off_ptr,
        num_chunks as i32,
        n_head as i32,
        chunk_size as i32,
        head_dim as i32,
        qtype as i32,
        src_dtype_code,
    );
    Ok(())
}

/// Palette4 KV-cache format conversion.
///
/// Converts K or V data between arbitrary arena formats using 4-palette
/// metadata from KvHead structs. Source and destination may have independent
/// palette maps; routing is handled by a merged xlat[128] table built on-GPU.
///
/// # Arguments
/// * `src_kvhead_ptrs` - Device pointers to src KvHead structs [num_layers × num_kv_heads]
/// * `dst_kvhead_ptrs` - Device pointers to dst KvHead structs [num_layers × num_kv_heads]
/// * `num_kv_heads` - Number of KV heads per layer
/// * `num_layers` - Number of layers
/// * `num_chunks` - Number of 32-token chunks per head to convert
/// * `is_k` - true for K conversion, false for V
pub fn quantize_palette4_convert(
    heads_base: &CudaSlice<u8>,
    num_heads: usize,
    num_kv_heads: usize,
    num_layers: usize,
    num_chunks: usize,
    is_k: bool,
    head_dim: usize,
    dev: &CudaDevice,
) -> Result<()> {
    if num_kv_heads == 0 || num_layers == 0 || num_chunks == 0 {
        return Ok(());
    }

    let stream = dev.cuda_stream();
    let (base_ptr, _guard) = heads_base.device_ptr(&stream);

    unsafe {
        crate::set_kernel_breadcrumb(
            if is_k {
                "run_quantize_palette4_convert (K, single)"
            } else {
                "run_quantize_palette4_convert (V, single)"
            },
            file!(),
            line!(),
        );
        run_quantize_palette4_convert(
            base_ptr as *const u8,
            num_heads as i32,
            num_kv_heads as i32,
            num_layers as i32,
            num_chunks as i32,
            if is_k { 1 } else { 0 },
            head_dim as i32,
            stream.cu_stream() as *mut _,
        );
    }
    Ok(())
}

// ============================================================================
// Palette4 buffered conversion API
// ============================================================================

/// KvHead byte layout — parameterized by head_dim (128 or 256), mirroring
/// `kv_head_byte_size<HD>()` and the accessors in slot_types.cuh. The record
/// is `[k_pal hd/4][v_pal hd/4][k_ptr 4×u64][v_ptr 4×u64][k_fmt 4][v_fmt 4]
/// [k_scale 4×f32][v_scale 4×f32]` = hd/2 + 104 bytes.
pub const KVHEAD_N_PAL: usize = 4;
/// The widest head the palette records support; pal-map buffers are sized for
/// it and only the first `head_dim / 4` bytes are live.
pub const KVHEAD_MAX_HD: usize = 256;
/// A 2-bit-packed palette assignment map, sized for the widest head.
pub type PalMapBytes = [u8; KVHEAD_MAX_HD / 4];

const fn kvhead_size(hd: usize) -> usize {
    hd / 2 + 104
}
const fn kvhead_v_pal_off(hd: usize) -> usize {
    hd / 4
}
const fn kvhead_k_ptr_off(hd: usize) -> usize {
    hd / 2
}
const fn kvhead_v_ptr_off(hd: usize) -> usize {
    hd / 2 + 32
}
const fn kvhead_k_fmt_off(hd: usize) -> usize {
    hd / 2 + 64
}
const fn kvhead_v_fmt_off(hd: usize) -> usize {
    hd / 2 + 68
}
const fn kvhead_k_scale_off(hd: usize) -> usize {
    hd / 2 + 72
}
const fn kvhead_v_scale_off(hd: usize) -> usize {
    hd / 2 + 88
}

/// The head widths the palette4 convert kernel is instantiated for.
pub fn kvhead_supported_head_dim(hd: usize) -> bool {
    matches!(hd, 128 | 256)
}

/// Maps a `GgmlDType` to the 1-byte ArenaFormat code expected by the palette4
/// convert kernel.  Returns an error for types that the kernel does not
/// support.
pub fn ggml_dtype_to_arena_fmt_code(dtype: GgmlDType) -> Result<u8> {
    let code: u8 = match dtype {
        GgmlDType::F32 => 0,
        GgmlDType::F16 => 1,
        GgmlDType::BF16 => 2,
        GgmlDType::R16 => 3,
        GgmlDType::P2 => 4,
        GgmlDType::QAWQ => 5,
        GgmlDType::QAWQ_G64 => 6,
        GgmlDType::Q8_0 => 7,
        GgmlDType::Q8_1 => 8,
        GgmlDType::Q8_K => 9,
        GgmlDType::Q8_KS => 10,
        GgmlDType::Q6_K => 11,
        GgmlDType::Q5_0 => 12,
        GgmlDType::Q5_1 => 13,
        GgmlDType::Q5_K => 14,

        GgmlDType::Q4_0 => 15,
        GgmlDType::Q4_1 => 16,
        GgmlDType::Q4_K => 17,
        GgmlDType::Q4_KS => 18,

        GgmlDType::Q3_0 => 19,
        GgmlDType::Q3_1 => 20,
        GgmlDType::Q3_K => 21,

        GgmlDType::Q2_0 => 22,
        GgmlDType::Q2_1 => 23,
        GgmlDType::Q2_K => 24,
        GgmlDType::Q2_S => 25,
        GgmlDType::Q2_A => 26,

        GgmlDType::Q1_S => 27,

        GgmlDType::Q0_V => 28,
        GgmlDType::Q1_A => 29,
        GgmlDType::Q0_X => 30,
        GgmlDType::Q0_M2 => 31,
        GgmlDType::Q0_M4 => 32,
        GgmlDType::Q0 => 33,

        _ => crate::bail!(
            "ggml_dtype_to_arena_fmt_code: unsupported dtype {:?}",
            dtype
        ),
    };
    Ok(code)
}

/// Per-head descriptor for `quantize_palette4_convert_buffered`.
///
/// Describes one `(layer, kv_head)` conversion job.  Separate src and dst
/// arena pointers and format codes are required because the kernel reads from
/// `src_kvhead_ptrs` arena pointers and writes to `dst_kvhead_ptrs` arena
/// pointers (different GPU memory regions, potentially different formats).
///
/// `k_pal_map` / `v_pal_map` are 2-bit-packed arrays covering `head_dim`
/// dimensions (the first `head_dim / 4` bytes are live). Each 2-bit field
/// selects which palette (0-3) owns that dimension. Use
/// `identity_pal_map(head_dim)` to get the standard identity map
/// (dims 0..head_dim/4 → pal 0, …). Src and dst may have independent pal_maps.
pub struct PalHeadDesc {
    /// Raw device pointers to source K arena data for palettes 0-3.
    pub k_src_arena_ptrs: [u64; KVHEAD_N_PAL],
    /// Raw device pointers to source V arena data for palettes 0-3.
    pub v_src_arena_ptrs: [u64; KVHEAD_N_PAL],
    /// GgmlDType format of each source K palette arena.
    pub k_src_fmts: [GgmlDType; KVHEAD_N_PAL],
    /// GgmlDType format of each source V palette arena.
    pub v_src_fmts: [GgmlDType; KVHEAD_N_PAL],
    /// 2-bit-packed source K palette assignment map.
    pub k_src_pal_map: PalMapBytes,
    /// 2-bit-packed source V palette assignment map.
    pub v_src_pal_map: PalMapBytes,
    /// Outer scale baked into the source K arena per palette (f32, 1.0 for
    /// float-typed sources). The encoder kernel divides dequantized source
    /// values by this so a re-compression chain (quant → quant) round-trips
    /// correctly. Pass 1.0 for any float source (F16, F32, BF16, R16) — those
    /// have no outer-scale concept.
    pub k_src_scales: [f32; KVHEAD_N_PAL],
    /// Outer scale baked into the source V arena per palette (f32, 1.0 for
    /// float-typed sources). Same convention as k_src_scales.
    pub v_src_scales: [f32; KVHEAD_N_PAL],
    /// Raw device pointers to destination K arena data for palettes 0-3.
    pub k_dst_arena_ptrs: [u64; KVHEAD_N_PAL],
    /// Raw device pointers to destination V arena data for palettes 0-3.
    pub v_dst_arena_ptrs: [u64; KVHEAD_N_PAL],
    /// GgmlDType format of each destination K palette arena.
    pub k_dst_fmts: [GgmlDType; KVHEAD_N_PAL],
    /// GgmlDType format of each destination V palette arena.
    pub v_dst_fmts: [GgmlDType; KVHEAD_N_PAL],
    /// 2-bit-packed destination K palette assignment map.
    pub k_dst_pal_map: PalMapBytes,
    /// 2-bit-packed destination V palette assignment map.
    pub v_dst_pal_map: PalMapBytes,
    /// Post-dequant scale written into the dst KvHead for K (f32, default 1.0).
    /// The decode kernel multiplies dequantized K values by this scale per palette.
    pub k_dst_scales: [f32; KVHEAD_N_PAL],
    /// Post-dequant scale written into the dst KvHead for V (f32, default 1.0).
    pub v_dst_scales: [f32; KVHEAD_N_PAL],
}

/// Build the identity 2-bit-packed palette map for `head_dim` (only the first
/// `head_dim / 4` bytes are live; the rest stay zero).
pub fn identity_pal_map(head_dim: usize) -> PalMapBytes {
    debug_assert!(kvhead_supported_head_dim(head_dim));
    let pal_dim = head_dim / KVHEAD_N_PAL;
    let mut out = [0u8; KVHEAD_MAX_HD / 4];
    for d in 0..head_dim {
        let p = (d / pal_dim) as u8;
        out[d / 4] |= (p & 0x3) << (2 * (d % 4));
    }
    out
}

/// Build a balanced pseudo-random 2-bit-packed palette map for `head_dim`.
///
/// Assigns each dim to one of 4 palettes using a Fisher-Yates shuffle. Every
/// palette is assigned exactly `head_dim / 4` dims but the assignment is
/// non-contiguous and pseudo-random. The caller supplies a seed/IV so different
/// randomization events can generate different maps while remaining reproducible.
pub fn shuffled_pal_map(head_dim: usize, seed: u64) -> PalMapBytes {
    debug_assert!(kvhead_supported_head_dim(head_dim));
    let pal_dim = head_dim / KVHEAD_N_PAL;
    // Start with exactly pal_dim dims per palette (sequential assignment), then
    // shuffle with a caller-provided seed so the palette sizes stay balanced
    // while the dim routing varies between randomization events.
    let mut assign = [0u8; KVHEAD_MAX_HD];
    for (d, a) in assign.iter_mut().enumerate().take(head_dim) {
        *a = (d / pal_dim) as u8;
    }

    // Fisher-Yates with Knuth multiplicative LCG.
    let mut rng: u64 = seed ^ 0x9e3779b97f4a7c15u64;
    if rng == 0 {
        rng = 0x9e3779b97f4a7c15u64;
    }
    for i in (1..head_dim).rev() {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        assign.swap(i, j);
    }

    // Pack: 4 dims per byte, 2 bits per dim in little-endian order.
    let mut out = [0u8; KVHEAD_MAX_HD / 4];
    for d in 0..head_dim {
        out[d / 4] |= (assign[d] & 0x3) << (2 * (d % 4));
    }
    out
}

/// Serialize arena pointers, formats, pal_map, and scales into a KvHead byte
/// record for `head_dim` (`head_dim / 2 + 104` bytes).
#[allow(clippy::too_many_arguments)]
pub fn build_kvhead_bytes_raw(
    head_dim: usize,
    k_arena_ptrs: &[u64; KVHEAD_N_PAL],
    v_arena_ptrs: &[u64; KVHEAD_N_PAL],
    k_fmts: &[GgmlDType; KVHEAD_N_PAL],
    v_fmts: &[GgmlDType; KVHEAD_N_PAL],
    k_pal_map: &PalMapBytes,
    v_pal_map: &PalMapBytes,
    k_scales: &[f32; KVHEAD_N_PAL],
    v_scales: &[f32; KVHEAD_N_PAL],
) -> Result<Vec<u8>> {
    if !kvhead_supported_head_dim(head_dim) {
        crate::bail!("build_kvhead_bytes_raw: unsupported head_dim {head_dim}");
    }
    let pal_bytes = head_dim / 4;
    let mut head = vec![0u8; kvhead_size(head_dim)];
    head[..pal_bytes].copy_from_slice(&k_pal_map[..pal_bytes]);
    head[kvhead_v_pal_off(head_dim)..kvhead_v_pal_off(head_dim) + pal_bytes]
        .copy_from_slice(&v_pal_map[..pal_bytes]);
    for p in 0..KVHEAD_N_PAL {
        let k_ptr_off = kvhead_k_ptr_off(head_dim) + p * 8;
        head[k_ptr_off..k_ptr_off + 8].copy_from_slice(&k_arena_ptrs[p].to_le_bytes());
        let v_ptr_off = kvhead_v_ptr_off(head_dim) + p * 8;
        head[v_ptr_off..v_ptr_off + 8].copy_from_slice(&v_arena_ptrs[p].to_le_bytes());
        head[kvhead_k_fmt_off(head_dim) + p] = ggml_dtype_to_arena_fmt_code(k_fmts[p])?;
        head[kvhead_v_fmt_off(head_dim) + p] = ggml_dtype_to_arena_fmt_code(v_fmts[p])?;
        let k_f32 = k_scales[p].to_le_bytes();
        let v_f32 = v_scales[p].to_le_bytes();
        let ks = kvhead_k_scale_off(head_dim) + p * 4;
        let vs = kvhead_v_scale_off(head_dim) + p * 4;
        head[ks..ks + 4].copy_from_slice(&k_f32);
        head[vs..vs + 4].copy_from_slice(&v_f32);
    }
    Ok(head)
}

/// Palette4 KV-cache format conversion with CPU-side buffer construction.
///
/// Higher-level wrapper around [`quantize_palette4_convert`] that accepts
/// structured `PalHeadDesc` descriptors, serialises them into 152-byte
/// KvHead GPU structs, and invokes the kernel for both K and V in a single
/// call.
///
/// # Arguments
/// * `descs` - Row-major `[num_layers][num_kv_heads]` slice of head descriptors.
/// * `num_kv_heads` - KV heads per layer.
/// * `num_layers` - Number of transformer layers.
/// * `num_chunks` - 32-token chunks per head to convert.
/// * `stager` - Optional pinned staging allocator for fully async upload.
/// * `dev` - CUDA device.
///
/// `descs.len()` must equal `num_layers * num_kv_heads`.
pub fn quantize_palette4_convert_buffered(
    head_dim: usize,
    descs: &[PalHeadDesc],
    num_kv_heads: usize,
    num_layers: usize,
    num_chunks: usize,
    generation: &super::pinned_staging::Generation,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<()> {
    if num_kv_heads == 0 || num_layers == 0 || num_chunks == 0 {
        return Ok(());
    }
    if !kvhead_supported_head_dim(head_dim) {
        crate::bail!("quantize_palette4_convert_buffered: unsupported head_dim {head_dim}");
    }
    let kvhead_bytes = kvhead_size(head_dim);

    // Guard against i32 overflow in the kernel's grid/count parameters.
    for (name, val) in [
        ("num_kv_heads", num_kv_heads),
        ("num_layers", num_layers),
        ("num_chunks", num_chunks),
    ] {
        if val > i32::MAX as usize {
            crate::bail!("quantize_palette4_convert_buffered: {name}={val} exceeds i32::MAX");
        }
    }

    let expected = num_layers * num_kv_heads;
    if descs.len() != expected {
        crate::bail!(
            "quantize_palette4_convert_buffered: descs.len()={} != num_layers*num_kv_heads={}",
            descs.len(),
            expected
        );
    }

    // Validate that every arena pointer referenced by the pal_map is non-null.
    // A zero pointer would cause CUDA_ERROR_ILLEGAL_ADDRESS inside the kernel,
    // which is much harder to diagnose than a Rust-level error here.
    for (i, desc) in descs.iter().enumerate() {
        // Determine which palettes are actually referenced by each pal_map.
        let check_ptrs = |ptrs: &[u64; KVHEAD_N_PAL],
                          pal_map: &PalMapBytes,
                          side: &str,
                          kv: &str|
         -> Result<()> {
            let mut used = [false; KVHEAD_N_PAL];
            for d in 0..head_dim {
                let p = ((pal_map[d / 4] >> (2 * (d % 4))) & 0x3) as usize;
                used[p] = true;
            }
            for p in 0..KVHEAD_N_PAL {
                if used[p] && ptrs[p] == 0 {
                    crate::bail!(
                        "quantize_palette4_convert_buffered: desc[{i}] {kv} {side} \
                         arena pointer for palette {p} is null (0) but pal_map references it"
                    );
                }
            }
            Ok(())
        };
        check_ptrs(&desc.k_src_arena_ptrs, &desc.k_src_pal_map, "src", "K")?;
        check_ptrs(&desc.k_dst_arena_ptrs, &desc.k_dst_pal_map, "dst", "K")?;
        check_ptrs(&desc.v_src_arena_ptrs, &desc.v_src_pal_map, "src", "V")?;
        check_ptrs(&desc.v_dst_arena_ptrs, &desc.v_dst_pal_map, "dst", "V")?;
    }

    // Pack everything into one GPU allocation:
    //
    //   [ src KvHead[0..N]    ]   offset 0
    //   [ dst KvHead[0..N]    ]   offset N × KVHEAD_SIZE
    //
    // Total = N × 2 × KVHEAD_SIZE bytes.  The kernel computes head pointers
    // as base + job * KVHEAD_SIZE (src) and base + (N + job) * KVHEAD_SIZE (dst),
    // so no pointer arrays are needed.
    let n = expected;
    let src_heads_off: usize = 0;
    let dst_heads_off: usize = n * kvhead_bytes;
    // Layout: [src KvHeads][dst KvHeads]. Per-palette outer scales live
    // inside each dst KvHead struct (f32 at HD/2+72 / HD/2+88), so the encoder
    // (multiply by outer) and decoder (divide by outer) share a single source
    // of truth.
    let total_bytes = 2 * n * kvhead_bytes;

    // Build the CPU image directly in pinned memory (no intermediate Vec).
    let mut buf = generation.alloc(total_bytes)?;

    for (i, desc) in descs.iter().enumerate() {
        let src_bytes = build_kvhead_bytes_raw(
            head_dim,
            &desc.k_src_arena_ptrs,
            &desc.v_src_arena_ptrs,
            &desc.k_src_fmts,
            &desc.v_src_fmts,
            &desc.k_src_pal_map,
            &desc.v_src_pal_map,
            &desc.k_src_scales,
            &desc.v_src_scales,
        )?;
        let dst_bytes = build_kvhead_bytes_raw(
            head_dim,
            &desc.k_dst_arena_ptrs,
            &desc.v_dst_arena_ptrs,
            &desc.k_dst_fmts,
            &desc.v_dst_fmts,
            &desc.k_dst_pal_map,
            &desc.v_dst_pal_map,
            &desc.k_dst_scales,
            &desc.v_dst_scales,
        )?;

        let src_off = src_heads_off + i * kvhead_bytes;
        let dst_off = dst_heads_off + i * kvhead_bytes;
        buf[src_off..src_off + kvhead_bytes].copy_from_slice(&src_bytes);
        buf[dst_off..dst_off + kvhead_bytes].copy_from_slice(&dst_bytes);
    }

    // Async H2D upload via stager (deferred cleanup).
    let gpu_buf = generation.submit(buf)?;

    // Launch K and V conversion passes.
    // Grid: dim3(num_kv_heads, num_layers) → exactly n blocks per launch.
    let base_ptr = gpu_buf.dev_ptr();
    let raw_stream = stream.cu_stream() as *mut _;
    unsafe {
        crate::set_kernel_breadcrumb("run_quantize_palette4_convert (K)", file!(), line!());
        run_quantize_palette4_convert(
            base_ptr as *const u8,
            n as i32,
            num_kv_heads as i32,
            num_layers as i32,
            num_chunks as i32,
            1, // K
            head_dim as i32,
            raw_stream,
        );
        crate::set_kernel_breadcrumb("run_quantize_palette4_convert (V)", file!(), line!());
        run_quantize_palette4_convert(
            base_ptr as *const u8,
            n as i32,
            num_kv_heads as i32,
            num_layers as i32,
            num_chunks as i32,
            0, // V
            head_dim as i32,
            raw_stream,
        );
    }
    // gpu_buf drops here — stream-ordered after the kernel launches.
    Ok(())
}

/// Thin unsafe wrapper around `run_select_kv_format_palette4_paged`.
///
/// # Safety
/// All device pointers must be valid. `head_gids_ptr` must have
/// `total_heads * 8` i64 entries (K/V GIDs for each of the 4 palette bands,
/// in HeadGids order: `head * 8 + palette * 2 + is_v`).
#[allow(clippy::too_many_arguments)]
pub unsafe fn select_kv_format_palette4_paged(
    per_head_table_ptr: u64,
    head_gids_ptr: u64,
    q_relevance_median_ptr: u64,
    q_relevance_spread_ptr: u64,
    k_head_amax_ptr: u64,
    v_head_amax_ptr: u64,
    k_head_p95_ptr: u64,
    v_head_p95_ptr: u64,
    k_cand_ptr: u64,
    v_cand_ptr: u64,
    n_k_cand: i32,
    n_v_cand: i32,
    k_threshold_hi: f32,
    k_threshold_lo: f32,
    v_threshold_hi: f32,
    v_threshold_lo: f32,
    total_heads: i32,
    blocks_per_head: i32,
    n_kv_head: i32,
    arena_chunks: i32,
    valid_ranges_ptr: u64,
    k_palette_tags_ptr: u64,
    v_palette_tags_ptr: u64,
    k_palette_scale_ptr: u64,
    v_palette_scale_ptr: u64,
    k_palette_map_ptr: u64,
    v_palette_map_ptr: u64,
    k_effective_block_tags_ptr: u64,
    v_effective_block_tags_ptr: u64,
    k_head_tags_ptr: u64,
    v_head_tags_ptr: u64,
    q_relevance_out_ptr: u64,
    stream: *mut std::ffi::c_void,
) {
    crate::set_kernel_breadcrumb("run_select_kv_format_palette4_paged", file!(), line!());
    run_select_kv_format_palette4_paged(
        per_head_table_ptr as *const i64,
        head_gids_ptr as *const i64,
        q_relevance_median_ptr as *mut f32,
        q_relevance_spread_ptr as *mut f32,
        k_head_amax_ptr as *mut f32,
        v_head_amax_ptr as *mut f32,
        k_head_p95_ptr as *mut f32,
        v_head_p95_ptr as *mut f32,
        k_cand_ptr as *const i32,
        v_cand_ptr as *const i32,
        n_k_cand,
        n_v_cand,
        k_threshold_hi,
        k_threshold_lo,
        v_threshold_hi,
        v_threshold_lo,
        total_heads,
        blocks_per_head,
        n_kv_head,
        arena_chunks,
        valid_ranges_ptr as *const i32,
        k_palette_tags_ptr as *mut i32,
        v_palette_tags_ptr as *mut i32,
        k_palette_scale_ptr as *mut f32,
        v_palette_scale_ptr as *mut f32,
        k_palette_map_ptr as *mut i32,
        v_palette_map_ptr as *mut i32,
        k_effective_block_tags_ptr as *mut i32,
        v_effective_block_tags_ptr as *mut i32,
        k_head_tags_ptr as *mut i32,
        v_head_tags_ptr as *mut i32,
        q_relevance_out_ptr as *mut f32,
        stream,
    );
}

/// Gids staged per (chunk, head) for the selection kernels: K and V for each
/// of the 4 palette bands, in HeadGids order (`head * 8 + palette * 2 + is_v`).
/// Must match `GIDS_PER_HEAD` in candle-nn's `head_gids.rs` and `arena_table.cuh`.
const SELECT_GIDS_PER_HEAD: usize = 8;

/// Paged format selection — runs the fused per-head palette-4 selection kernel
/// and returns the per-block effective format tags. Used by tests.
///
/// Wraps `select_kv_format_palette4_paged_batched_raw_from_device_ptrs` and keeps
/// only the `k_eff_block_tags` / `v_eff_block_tags` outputs (per-block tags after
/// palette slot expansion); the per-slot palette tags, scales, pal_maps, and
/// other diagnostic outputs are discarded.
pub fn select_kv_format_paged_batched_raw(
    per_head_table_gpu: &CudaSlice<i64>,
    head_gids: &[i64],
    k_candidates: &[GgmlDType],
    v_candidates: &[GgmlDType],
    k_threshold_hi: f32,
    k_threshold_lo: f32,
    v_threshold_hi: f32,
    v_threshold_lo: f32,
    blocks_per_head: usize,
    n_kv_head: usize,
    arena_chunks: usize,
    dev: &CudaDevice,
) -> Result<(CudaSlice<i32>, CudaSlice<i32>)> {
    let n_chunks = head_gids.len() / (n_kv_head * SELECT_GIDS_PER_HEAD);
    let total_heads = n_chunks * n_kv_head;
    let total_blocks = total_heads * blocks_per_head;

    if total_blocks == 0 {
        let empty = dev.alloc_zeros::<i32>(0)?;
        return Ok((empty, dev.alloc_zeros::<i32>(0)?));
    }

    let gids_gpu = dev.memcpy_stod(head_gids)?;
    // Test-oriented wrapper: full chunks only (offset 0, all 32 tokens).
    let full_ranges = vec![32i32; n_chunks];

    let (_kpt, _vpt, _ksi, _vsi, _kpm, _vpm, _ka, _va, k_eff, v_eff, _kht, _vht, _qr) = unsafe {
        let stream = dev.cuda_stream();
        let (pht_ptr, _pht_guard) = per_head_table_gpu.device_ptr(&stream);
        let (gids_ptr, _gids_guard) = gids_gpu.device_ptr(&stream);
        select_kv_format_palette4_paged_batched_raw_from_device_ptrs(
            pht_ptr,
            gids_ptr,
            n_chunks,
            k_candidates,
            v_candidates,
            k_threshold_hi,
            k_threshold_lo,
            v_threshold_hi,
            v_threshold_lo,
            blocks_per_head,
            n_kv_head,
            arena_chunks,
            &full_ranges,
            dev,
            &stream,
        )?
    };

    Ok((k_eff, v_eff))
}

/// Fused per-head palette-4 selection from pre-uploaded device pointers.
///
/// Calls the fused `run_select_kv_format_palette4_paged` kernel (one block per head)
/// which performs format selection and palette-4 grouping in a single pass.
///
/// Returns a 13-tuple:
/// `(k_palette_tags, v_palette_tags, k_pal_scale, v_pal_scale,
///   k_palette_map, v_palette_map, k_head_amax, v_head_amax,
///   k_eff_block_tags, v_eff_block_tags, k_head_tags, v_head_tags, q_rel_out)`
///
/// # Safety
///
/// `per_head_table_ptr` and `head_gids_ptr` are raw DEVICE addresses, passed
/// straight to the kernel without validation. The caller guarantees that both:
///
/// * point into live allocations on the same device and stream this call uses,
///   and stay live for the duration of the launch;
/// * are laid out as the kernel expects — `PerHeadEntry[n_chunks]` and the
///   matching `HeadGids` block — since a wrong stride reads neighbouring memory
///   rather than faulting;
/// * are correctly aligned for those element types.
///
/// Passing a host pointer, a freed allocation, or a buffer from another device
/// is undefined behaviour and will not necessarily fault.
#[allow(clippy::type_complexity)]
pub unsafe fn select_kv_format_palette4_paged_batched_raw_from_device_ptrs(
    per_head_table_ptr: u64,
    head_gids_ptr: u64,
    n_chunks: usize,
    k_candidates: &[GgmlDType],
    v_candidates: &[GgmlDType],
    k_threshold_hi: f32,
    k_threshold_lo: f32,
    v_threshold_hi: f32,
    v_threshold_lo: f32,
    blocks_per_head: usize,
    n_kv_head: usize,
    arena_chunks: usize,
    // Per-chunk valid token range, packed (offset << 8) | len, len in
    // [1, 32]. Partial chunks' dead slots are zero (arena zeroing at
    // creation/recycle); the range corrects the count-normalized error
    // metrics and sink statistics for the missing lanes.
    valid_ranges: &[i32],
    dev: &CudaDevice,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<(
    CudaSlice<i32>, // k_palette_tags          [total_heads * 4]
    CudaSlice<i32>, // v_palette_tags          [total_heads * 4]
    CudaSlice<f32>, // k_pal_scale             [total_heads * 4] outer scale per slot
    CudaSlice<f32>, // v_pal_scale             [total_heads * 4] outer scale per slot
    CudaSlice<i32>, // k_palette_map           [total_heads * blocks_per_head]
    CudaSlice<i32>, // v_palette_map           [total_heads * blocks_per_head]
    CudaSlice<f32>, // k_head_amax             [total_heads]
    CudaSlice<f32>, // v_head_amax             [total_heads]
    CudaSlice<i32>, // k_eff_block_tags        [total_heads * blocks_per_head]
    CudaSlice<i32>, // v_eff_block_tags        [total_heads * blocks_per_head]
    CudaSlice<i32>, // k_head_tags             [total_heads]
    CudaSlice<i32>, // v_head_tags             [total_heads]
    CudaSlice<f32>, // q_rel_out               [total_heads * blocks_per_head]
)> {
    let total_heads = n_chunks * n_kv_head;
    let total_blocks = total_heads * blocks_per_head;
    if total_heads == 0 {
        return Ok((
            dev.alloc_zeros::<i32>(0)?,
            dev.alloc_zeros::<i32>(0)?,
            dev.alloc_zeros::<f32>(0)?,
            dev.alloc_zeros::<f32>(0)?,
            dev.alloc_zeros::<i32>(0)?,
            dev.alloc_zeros::<i32>(0)?,
            dev.alloc_zeros::<f32>(0)?,
            dev.alloc_zeros::<f32>(0)?,
            dev.alloc_zeros::<i32>(0)?,
            dev.alloc_zeros::<i32>(0)?,
            dev.alloc_zeros::<i32>(0)?,
            dev.alloc_zeros::<i32>(0)?,
            dev.alloc_zeros::<f32>(0)?,
        ));
    }

    let mut k_codes: Vec<i32> = k_candidates
        .iter()
        .map(|d| ggml_to_select_qtype(*d))
        .collect::<Result<Vec<_>>>()?;
    let mut v_codes: Vec<i32> = v_candidates
        .iter()
        .map(|d| ggml_to_select_qtype(*d))
        .collect::<Result<Vec<_>>>()?;
    // Sort candidates ascending by BPE so the kernel sees the lowest-BPE
    // (most aggressive) format first — that's the kernel's expected
    // contract (`select_kv_format.cuh` walks `cands[ci]` in ascending-BPE
    // order and stops at the first format hitting the 32-block quota).
    //
    // `sort_by_key` is a STABLE sort: when two formats have equal BPE,
    // their relative order from the caller's input is preserved. This is
    // load-bearing — it lets the caller express priority within an
    // equal-BPE tier by ordering the input list. E.g. passing
    // [Q0_V, Q0_X] vs [Q0_X, Q0_V] (both 0.5 BPE) keeps the user's
    // chosen first-tried format first; the kernel's lowest-aerr tiebreak
    // can still override at the winner-pick stage.
    k_codes.sort_by_key(|c| select_qtype_bpe_x4(*c));
    v_codes.sort_by_key(|c| select_qtype_bpe_x4(*c));

    let k_cand_gpu = stream.memcpy_stod(&k_codes).w()?;
    let v_cand_gpu = stream.memcpy_stod(&v_codes).w()?;
    if valid_ranges.len() != n_chunks {
        crate::bail!(
            "valid_ranges length {} != n_chunks {}",
            valid_ranges.len(),
            n_chunks
        );
    }
    let valid_ranges_gpu = stream.memcpy_stod(valid_ranges).w()?;

    let mut q_rel_median_out = stream.alloc_zeros::<f32>(total_heads).w()?;
    let mut q_rel_spread_out = stream.alloc_zeros::<f32>(total_heads).w()?;
    let mut k_head_amax_out = stream.alloc_zeros::<f32>(total_heads).w()?;
    let mut v_head_amax_out = stream.alloc_zeros::<f32>(total_heads).w()?;
    let mut k_head_p95_out = stream.alloc_zeros::<f32>(total_heads).w()?;
    let mut v_head_p95_out = stream.alloc_zeros::<f32>(total_heads).w()?;
    let mut k_palette_tags = stream.alloc_zeros::<i32>(total_heads * 4).w()?;
    let mut v_palette_tags = stream.alloc_zeros::<i32>(total_heads * 4).w()?;
    let mut k_pal_scale = stream.alloc_zeros::<f32>(total_heads * 4).w()?;
    let mut v_pal_scale = stream.alloc_zeros::<f32>(total_heads * 4).w()?;
    let mut k_palette_map = stream.alloc_zeros::<i32>(total_blocks).w()?;
    let mut v_palette_map = stream.alloc_zeros::<i32>(total_blocks).w()?;
    let mut k_eff_block_tags = stream.alloc_zeros::<i32>(total_blocks).w()?;
    let mut v_eff_block_tags = stream.alloc_zeros::<i32>(total_blocks).w()?;
    let mut k_head_tags_out = stream.alloc_zeros::<i32>(total_heads).w()?;
    let mut v_head_tags_out = stream.alloc_zeros::<i32>(total_heads).w()?;
    let mut q_rel_out = stream.alloc_zeros::<f32>(total_blocks).w()?;
    // Source outer scales: 1.0 when source is R16/float (no outer scale).
    // For re-compression from already-quantized data these must be supplied by
    // the caller; passing 1.0 here is correct for the initial R16→quant path.
    {
        let (q_med_ptr, _qm_guard) = q_rel_median_out.device_ptr_mut(stream);
        let (q_spd_ptr, _qs_guard) = q_rel_spread_out.device_ptr_mut(stream);
        let (k_amax_ptr, _ka_guard) = k_head_amax_out.device_ptr_mut(stream);
        let (v_amax_ptr, _va_guard) = v_head_amax_out.device_ptr_mut(stream);
        let (k_p95_ptr, _kp_guard) = k_head_p95_out.device_ptr_mut(stream);
        let (v_p95_ptr, _vp_guard) = v_head_p95_out.device_ptr_mut(stream);
        let (k_cand_ptr, _kc_guard) = k_cand_gpu.device_ptr(stream);
        let (v_cand_ptr, _vc_guard) = v_cand_gpu.device_ptr(stream);
        let (k_pal_tag_ptr, _kpt) = k_palette_tags.device_ptr_mut(stream);
        let (v_pal_tag_ptr, _vpt) = v_palette_tags.device_ptr_mut(stream);
        let (k_pal_scale_ptr, _kpsi) = k_pal_scale.device_ptr_mut(stream);
        let (v_pal_scale_ptr, _vpsi) = v_pal_scale.device_ptr_mut(stream);
        let (k_pal_map_ptr, _kpm) = k_palette_map.device_ptr_mut(stream);
        let (v_pal_map_ptr, _vpm) = v_palette_map.device_ptr_mut(stream);
        let (k_eff_ptr, _keff) = k_eff_block_tags.device_ptr_mut(stream);
        let (v_eff_ptr, _veff) = v_eff_block_tags.device_ptr_mut(stream);
        let (k_htag_ptr, _kht) = k_head_tags_out.device_ptr_mut(stream);
        let (v_htag_ptr, _vht) = v_head_tags_out.device_ptr_mut(stream);
        let (q_rel_ptr, _qrel) = q_rel_out.device_ptr_mut(stream);
        let (valid_ranges_ptr, _vr_guard) = valid_ranges_gpu.device_ptr(stream);

        select_kv_format_palette4_paged(
            per_head_table_ptr,
            head_gids_ptr,
            q_med_ptr,
            q_spd_ptr,
            k_amax_ptr,
            v_amax_ptr,
            k_p95_ptr,
            v_p95_ptr,
            k_cand_ptr,
            v_cand_ptr,
            k_codes.len() as i32,
            v_codes.len() as i32,
            k_threshold_hi,
            k_threshold_lo,
            v_threshold_hi,
            v_threshold_lo,
            total_heads as i32,
            blocks_per_head as i32,
            n_kv_head as i32,
            arena_chunks as i32,
            valid_ranges_ptr,
            k_pal_tag_ptr,
            v_pal_tag_ptr,
            k_pal_scale_ptr,
            v_pal_scale_ptr,
            k_pal_map_ptr,
            v_pal_map_ptr,
            k_eff_ptr,
            v_eff_ptr,
            k_htag_ptr,
            v_htag_ptr,
            q_rel_ptr,
            stream.cu_stream() as *mut _,
        );
    }

    Ok((
        k_palette_tags,
        v_palette_tags,
        k_pal_scale,
        v_pal_scale,
        k_palette_map,
        v_palette_map,
        k_head_amax_out,
        v_head_amax_out,
        k_eff_block_tags,
        v_eff_block_tags,
        k_head_tags_out,
        v_head_tags_out,
        q_rel_out,
    ))
}

#[allow(clippy::type_complexity)]
pub fn reduce_head_format_stats(
    k_dev_ptr: u64,
    v_dev_ptr: u64,
    blocks_per_head: usize,
    n_kv_head: usize,
    num_chunks: usize,
    dev: &CudaDevice,
) -> Result<(
    CudaSlice<i32>,
    CudaSlice<i32>,
    CudaSlice<i32>,
    CudaSlice<i32>,
)> {
    let total_heads = num_chunks * n_kv_head;
    let total_blocks = total_heads * blocks_per_head;
    let mut k_head_out = dev.alloc_zeros::<i32>(total_heads)?;
    let mut v_head_out = dev.alloc_zeros::<i32>(total_heads)?;
    let mut k_eff_out = dev.alloc_zeros::<i32>(total_blocks)?;
    let mut v_eff_out = dev.alloc_zeros::<i32>(total_blocks)?;

    {
        let stream = dev.cuda_stream();
        let k_blk_ptr = k_dev_ptr;
        let v_blk_ptr = v_dev_ptr;
        let (k_head_ptr, _kh_guard) = k_head_out.device_ptr_mut(&stream);
        let (v_head_ptr, _vh_guard) = v_head_out.device_ptr_mut(&stream);
        let (k_eff_ptr, _ke_guard) = k_eff_out.device_ptr_mut(&stream);
        let (v_eff_ptr, _ve_guard) = v_eff_out.device_ptr_mut(&stream);

        unsafe {
            crate::set_kernel_breadcrumb("run_reduce_head_stats_format", file!(), line!());
            run_reduce_head_stats_format(
                k_blk_ptr as *const i32,
                v_blk_ptr as *const i32,
                k_head_ptr as *mut i32,
                v_head_ptr as *mut i32,
                k_eff_ptr as *mut i32,
                v_eff_ptr as *mut i32,
                blocks_per_head as i32,
                n_kv_head as i32,
                num_chunks as i32,
            );
        }
    }
    Ok((k_head_out, v_head_out, k_eff_out, v_eff_out))
}

/// Per-head format selection: returns most conservative per-(chunk, head) format.
/// Uses the fused palette4 kernel; slot 0 = highest-amax blocks = most conservative format.
#[allow(clippy::too_many_arguments)]
pub fn select_kv_format_paged_per_head(
    per_head_table_gpu: &CudaSlice<i64>,
    head_gids: &[i64],
    k_candidates: &[GgmlDType],
    v_candidates: &[GgmlDType],
    k_threshold_hi: f32,
    k_threshold_lo: f32,
    v_threshold_hi: f32,
    v_threshold_lo: f32,
    blocks_per_head: usize,
    n_kv_head: usize,
    arena_chunks: usize,
    dev: &CudaDevice,
) -> Result<(CudaSlice<i32>, CudaSlice<i32>)> {
    let n_chunks = head_gids.len() / (n_kv_head * SELECT_GIDS_PER_HEAD);
    let total_heads = n_chunks * n_kv_head;

    if total_heads == 0 || n_kv_head == 0 {
        let empty = dev.alloc_zeros::<i32>(0)?;
        return Ok((empty, dev.alloc_zeros::<i32>(0)?));
    }

    let gids_gpu = dev.memcpy_stod(head_gids)?;
    // Test-oriented wrapper: full chunks only (offset 0, all 32 tokens).
    let full_ranges = vec![32i32; n_chunks];

    let (_kpt, _vpt, _ksi, _vsi, _kpm, _vpm, _ka, _va, _keff, _veff, k_head_tags, v_head_tags, _qr) = unsafe {
        let stream = dev.cuda_stream();
        let (pht_ptr, _pht_guard) = per_head_table_gpu.device_ptr(&stream);
        let (gids_ptr, _gids_guard) = gids_gpu.device_ptr(&stream);
        select_kv_format_palette4_paged_batched_raw_from_device_ptrs(
            pht_ptr,
            gids_ptr,
            n_chunks,
            k_candidates,
            v_candidates,
            k_threshold_hi,
            k_threshold_lo,
            v_threshold_hi,
            v_threshold_lo,
            blocks_per_head,
            n_kv_head,
            arena_chunks,
            &full_ranges,
            dev,
            &stream,
        )?
    };

    Ok((k_head_tags, v_head_tags))
}

/// Run the new paged-batched sampled-error kernel.
///
/// Output logical layout:
/// [batch_item][head_dim][quant_index][head]
#[allow(clippy::too_many_arguments)]
pub fn sample_quant_errors_paged(
    per_head_table_dev_ptr: u64,
    head_gids_dev_ptr: u64,
    candidates: &[GgmlDType],
    sample_token: usize,
    side_is_k: bool,
    num_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    arena_chunks: usize,
    dev: &CudaDevice,
) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
    if sample_token >= 32 {
        crate::bail!(
            "sample_quant_errors_paged: sample_token {} out of range",
            sample_token
        );
    }
    let total = num_chunks
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(candidates.len()))
        .and_then(|v| v.checked_mul(n_kv_head))
        .ok_or_else(|| crate::Error::Msg("sample_quant_errors_paged: size overflow".into()))?;
    if total == 0 {
        return Ok((dev.alloc_zeros::<f32>(0)?, dev.alloc_zeros::<f32>(0)?));
    }

    let cand_codes: Vec<i32> = candidates
        .iter()
        .map(|d| ggml_to_select_qtype(*d))
        .collect::<Result<Vec<_>>>()?;
    let cand_gpu = dev.memcpy_stod(&cand_codes)?;
    let rel_total = num_chunks
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(n_kv_head))
        .ok_or_else(|| {
            crate::Error::Msg("sample_quant_errors_paged: relevance size overflow".into())
        })?;
    let mut error_out = unsafe { dev.alloc::<f32>(total)? };
    let mut q_relevance_out = unsafe { dev.alloc::<f32>(rel_total)? };

    {
        let stream = dev.cuda_stream();
        let pht_ptr = per_head_table_dev_ptr;
        let (cand_ptr, _cand_guard) = cand_gpu.device_ptr(&stream);
        let (out_ptr, _out_guard) = error_out.device_ptr_mut(&stream);
        let (qrel_ptr, _qrel_guard) = q_relevance_out.device_ptr_mut(&stream);

        unsafe {
            crate::set_kernel_breadcrumb("run_sample_quant_errors_paged", file!(), line!());
            run_sample_quant_errors_paged(
                pht_ptr as *const i64,
                head_gids_dev_ptr as *const i64,
                cand_ptr as *const i32,
                cand_codes.len() as i32,
                out_ptr as *mut f32,
                qrel_ptr as *mut f32,
                sample_token as i32,
                if side_is_k { 1 } else { 0 },
                num_chunks as i32,
                n_kv_head as i32,
                head_dim as i32,
                arena_chunks as i32,
            );
        }
    }

    Ok((error_out, q_relevance_out))
}

/// Fused KV sampled-error computation — processes K and V in a single kernel launch.
///
/// Returns `(k_errors, v_errors, q_relevance)` where:
/// - `k_errors`: `[num_chunks × head_dim × num_candidates × n_kv_head]` K quantization errors
/// - `v_errors`: same shape, V quantization errors weighted by Q·K relevance
/// - `q_relevance`: `[num_chunks × head_dim × n_kv_head]` per-block attention relevance scores
///
/// K and V must share the same candidate list.
pub fn sample_quant_errors_kv_paged(
    per_head_table_dev_ptr: u64,
    head_gids_dev_ptr: u64,
    candidates: &[GgmlDType],
    sample_token: usize,
    num_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    arena_chunks: usize,
    dev: &CudaDevice,
) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
    if sample_token >= 32 {
        crate::bail!(
            "sample_quant_errors_kv_paged: sample_token {} out of range",
            sample_token
        );
    }
    let total = num_chunks
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(candidates.len()))
        .and_then(|v| v.checked_mul(n_kv_head))
        .ok_or_else(|| crate::Error::Msg("sample_quant_errors_kv_paged: size overflow".into()))?;
    if total == 0 {
        return Ok((dev.alloc_zeros::<f32>(0)?, dev.alloc_zeros::<f32>(0)?));
    }

    let cand_codes: Vec<i32> = candidates
        .iter()
        .map(|d| ggml_to_select_qtype(*d))
        .collect::<Result<Vec<_>>>()?;
    let cand_gpu = dev.memcpy_stod(&cand_codes)?;
    let mut k_error_out = unsafe { dev.alloc::<f32>(total)? };
    let mut v_error_out = unsafe { dev.alloc::<f32>(total)? };

    {
        let stream = dev.cuda_stream();
        let pht_ptr = per_head_table_dev_ptr;
        let (cand_ptr, _cand_guard) = cand_gpu.device_ptr(&stream);
        let (k_out_ptr, _k_out_guard) = k_error_out.device_ptr_mut(&stream);
        let (v_out_ptr, _v_out_guard) = v_error_out.device_ptr_mut(&stream);

        unsafe {
            crate::set_kernel_breadcrumb(
                "run_sample_quant_errors_kv_paged (uniform)",
                file!(),
                line!(),
            );
            run_sample_quant_errors_kv_paged(
                pht_ptr as *const i64,
                head_gids_dev_ptr as *const i64,
                cand_ptr as *const i32,
                cand_codes.len() as i32,
                k_out_ptr as *mut f32,
                v_out_ptr as *mut f32,
                sample_token as i32,
                num_chunks as i32,
                n_kv_head as i32,
                head_dim as i32,
                arena_chunks as i32,
            );
        }
    }

    Ok((k_error_out, v_error_out))
}

/// Like `sample_quant_errors_kv_paged` but with candidates already staged on the device.
///
/// Use this when the candidate list is constant across many calls — upload once via
/// `dev.memcpy_stod(&cand_codes)` and pass the raw device pointer here to avoid
/// re-uploading on every invocation.
pub fn sample_quant_errors_kv_paged_staged(
    per_head_table_dev_ptr: u64,
    head_gids_dev_ptr: u64,
    candidates_dev_ptr: u64, // pre-staged &[i32] QType codes on device
    n_candidates: usize,
    sample_token: usize,
    num_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    arena_chunks: usize,
    dev: &CudaDevice,
) -> Result<(CudaSlice<f32>, CudaSlice<f32>)> {
    if sample_token >= 32 {
        crate::bail!(
            "sample_quant_errors_kv_paged_staged: sample_token {} out of range",
            sample_token
        );
    }
    let total = num_chunks
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(n_candidates))
        .and_then(|v| v.checked_mul(n_kv_head))
        .ok_or_else(|| {
            crate::Error::Msg("sample_quant_errors_kv_paged_staged: size overflow".into())
        })?;
    if total == 0 {
        return Ok((dev.alloc_zeros::<f32>(0)?, dev.alloc_zeros::<f32>(0)?));
    }

    let mut k_error_out = unsafe { dev.alloc::<f32>(total)? };
    let mut v_error_out = unsafe { dev.alloc::<f32>(total)? };

    {
        let stream = dev.cuda_stream();
        let (k_out_ptr, _k_out_guard) = k_error_out.device_ptr_mut(&stream);
        let (v_out_ptr, _v_out_guard) = v_error_out.device_ptr_mut(&stream);

        unsafe {
            crate::set_kernel_breadcrumb(
                "run_sample_quant_errors_kv_paged (raw)",
                file!(),
                line!(),
            );
            run_sample_quant_errors_kv_paged(
                per_head_table_dev_ptr as *const i64,
                head_gids_dev_ptr as *const i64,
                candidates_dev_ptr as *const i32,
                n_candidates as i32,
                k_out_ptr as *mut f32,
                v_out_ptr as *mut f32,
                sample_token as i32,
                num_chunks as i32,
                n_kv_head as i32,
                head_dim as i32,
                arena_chunks as i32,
            );
        }
    }

    Ok((k_error_out, v_error_out))
}

/// F16 code from CUDA `select_kv_format.cuh` (`#define SELECT_FMT_F16 1`).
pub const SELECT_FMT_F16: i32 = 1;
/// BF16 code from CUDA `select_kv_format.cuh` (`#define SELECT_FMT_BF16 2`).
pub const SELECT_FMT_BF16: i32 = 2;

/// Select winner candidate indices from pre-computed K/V error surfaces.
///
/// Error surfaces are expected on device (output of `sample_quant_errors_kv_paged`).
/// For each (chunk, head, dim) cell and each threshold, writes the index of the
/// first candidate whose error ≤ threshold into the output arrays.
///
/// Winner layout: `[n_thresholds × n_cells]` as `u8`, where
/// `cell = (chunk * n_kv_head + head) * head_dim + dim`.
/// This matches the layout consumed by `batch_summarize_from_winners`.
///
/// Returns `(k_winners, v_winners)` already downloaded to CPU.
pub fn select_kv_winners_paged(
    k_errors: &CudaSlice<f32>,
    v_errors: &CudaSlice<f32>,
    k_thresholds: &[f32],
    v_thresholds: &[f32],
    n_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    n_quant: usize,
    dev: &CudaDevice,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let n_cells = n_chunks
        .checked_mul(n_kv_head)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| crate::Error::Msg("select_kv_winners_paged: n_cells overflow".into()))?;
    if n_cells == 0 || n_quant == 0 {
        let last = n_quant.saturating_sub(1) as u8;
        return Ok((
            vec![last; k_thresholds.len() * n_cells],
            vec![last; v_thresholds.len() * n_cells],
        ));
    }
    let k_thr_gpu = dev.memcpy_stod(k_thresholds)?;
    let v_thr_gpu = dev.memcpy_stod(v_thresholds)?;
    let mut k_winners = unsafe { dev.alloc::<u8>(k_thresholds.len() * n_cells)? };
    let mut v_winners = unsafe { dev.alloc::<u8>(v_thresholds.len() * n_cells)? };

    {
        let stream = dev.cuda_stream();
        let (k_err_ptr, _ke_guard) = k_errors.device_ptr(&stream);
        let (v_err_ptr, _ve_guard) = v_errors.device_ptr(&stream);
        let (k_thr_ptr, _kt_guard) = k_thr_gpu.device_ptr(&stream);
        let (v_thr_ptr, _vt_guard) = v_thr_gpu.device_ptr(&stream);
        let (k_win_ptr, _kw_guard) = k_winners.device_ptr_mut(&stream);
        let (v_win_ptr, _vw_guard) = v_winners.device_ptr_mut(&stream);

        unsafe {
            crate::set_kernel_breadcrumb(
                "run_select_winners_kv_paged (per-head-amax)",
                file!(),
                line!(),
            );
            run_select_winners_kv_paged(
                k_err_ptr as *const f32,
                v_err_ptr as *const f32,
                k_thr_ptr as *const f32,
                v_thr_ptr as *const f32,
                k_win_ptr as *mut u8,
                v_win_ptr as *mut u8,
                k_thresholds.len() as i32,
                v_thresholds.len() as i32,
                n_cells as i32,
                n_quant as i32,
                n_kv_head as i32,
                head_dim as i32,
            );
        }
    }

    Ok((dev.memcpy_dtov(&k_winners)?, dev.memcpy_dtov(&v_winners)?))
}

/// Like `select_kv_winners_paged` but with threshold arrays already staged on the device.
///
/// Use this when K and V thresholds are constant across many calls — upload once and
/// pass the raw device pointers here to avoid re-uploading on every invocation.
pub fn select_kv_winners_paged_staged(
    k_errors: &CudaSlice<f32>,
    v_errors: &CudaSlice<f32>,
    k_thresholds_dev_ptr: u64, // pre-staged &[f32] on device
    n_k_thresholds: usize,
    v_thresholds_dev_ptr: u64, // pre-staged &[f32] on device
    n_v_thresholds: usize,
    n_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    n_quant: usize,
    dev: &CudaDevice,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let n_cells = n_chunks
        .checked_mul(n_kv_head)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| {
            crate::Error::Msg("select_kv_winners_paged_staged: n_cells overflow".into())
        })?;
    if n_cells == 0 || n_quant == 0 {
        let last = n_quant.saturating_sub(1) as u8;
        return Ok((
            vec![last; n_k_thresholds * n_cells],
            vec![last; n_v_thresholds * n_cells],
        ));
    }
    let mut k_winners = unsafe { dev.alloc::<u8>(n_k_thresholds * n_cells)? };
    let mut v_winners = unsafe { dev.alloc::<u8>(n_v_thresholds * n_cells)? };

    {
        let stream = dev.cuda_stream();
        let (k_err_ptr, _ke_guard) = k_errors.device_ptr(&stream);
        let (v_err_ptr, _ve_guard) = v_errors.device_ptr(&stream);
        let (k_win_ptr, _kw_guard) = k_winners.device_ptr_mut(&stream);
        let (v_win_ptr, _vw_guard) = v_winners.device_ptr_mut(&stream);

        unsafe {
            crate::set_kernel_breadcrumb(
                "run_select_winners_kv_paged (uniform-threshold)",
                file!(),
                line!(),
            );
            run_select_winners_kv_paged(
                k_err_ptr as *const f32,
                v_err_ptr as *const f32,
                k_thresholds_dev_ptr as *const f32,
                v_thresholds_dev_ptr as *const f32,
                k_win_ptr as *mut u8,
                v_win_ptr as *mut u8,
                n_k_thresholds as i32,
                n_v_thresholds as i32,
                n_cells as i32,
                n_quant as i32,
                n_kv_head as i32,
                head_dim as i32,
            );
        }
    }

    Ok((dev.memcpy_dtov(&k_winners)?, dev.memcpy_dtov(&v_winners)?))
}

/// Fused winner selection + GPU summarization — avoids downloading the full winner array.
///
/// Runs `select_winners_kv_paged` to produce K and V winner buffers on device, then
/// immediately runs `summarize_winners_side_paged` on each buffer, producing compact
/// `[n_thresholds × 3]` f32 partial-sum arrays (ideal_bits, head_bits, pal4_bits).
///
/// Download cost: `(n_k_thresholds + n_v_thresholds) × 3 × 4` bytes ≈ 168 bytes
/// instead of `(n_k_thresholds + n_v_thresholds) × n_cells` bytes (~3.6 MB).
///
/// # Arguments
/// * `k_errors` / `v_errors`  — K/V error surfaces from `sample_quant_errors_kv_paged_staged`
/// * `k_thresholds_dev_ptr` / `v_thresholds_dev_ptr` — pre-staged threshold arrays on device
/// * `candidates_bpe_dev_ptr` — pre-staged `[n_quant]` f32 bits-per-element on device
/// * `n_chunks`, `n_kv_head`, `head_dim`, `n_quant`, `chunk_size` — problem dimensions
/// * `pal_overhead` — palette metadata bits per head `= head_dim * 2 + 4 * 8`
///
/// # Returns
/// `(k_sums, v_sums)` where each `Vec<f32>` has length `n_thresholds * 3`:
///   `sums[t * 3 + 0]` = ideal_bits,  `[t * 3 + 1]` = head_bits,  `[t * 3 + 2]` = pal4_bits
/// Like [`select_and_summarize_kv_winners_paged_staged`] but accepts pre-allocated
/// scratch buffers to eliminate per-call device allocation overhead, and performs
/// an **asynchronous** DtoH copy on a dedicated `dtoh_stream` into a caller-supplied
/// pinned host buffer.
///
/// * `k_winners_scratch` — device buffer of at least `n_k_thresholds × n_cells` bytes.
/// * `v_winners_scratch` — device buffer of at least `n_v_thresholds × n_cells` bytes.
/// * `kv_sums` — device buffer of exactly `(n_k_thresholds + n_v_thresholds) × 3`
///   `f32` values. **Must be zeroed by the caller** (the kernel uses
///   `atomicAdd`). The K sums occupy the first `n_k_thresholds × 3`
///   elements and the V sums the remainder.
/// * `dtoh_stream` — secondary stream used exclusively for the DtoH transfer.
///   The function records an event on the compute stream, makes
///   `dtoh_stream` GPU-wait on it, then enqueues the DMA.
/// * `pinned_dst` — pre-allocated pinned host slice of length
///   `(n_k_thresholds + n_v_thresholds) × 3`.  The DMA is enqueued
///   but **not waited on** — the caller must `synchronize()` the
///   returned `CudaEvent` before reading the data.
///
/// Returns a `CudaEvent` recorded on `dtoh_stream` after the DMA.  Call
/// `event.synchronize()` to block until the data is available in `pinned_dst`.
pub fn select_and_summarize_kv_winners_paged_staged(
    k_errors: &CudaSlice<f32>,
    v_errors: &CudaSlice<f32>,
    k_thresholds_dev_ptr: u64,
    n_k_thresholds: usize,
    v_thresholds_dev_ptr: u64,
    n_v_thresholds: usize,
    candidates_bpe_dev_ptr: u64,
    n_chunks: usize,
    n_kv_head: usize,
    head_dim: usize,
    n_quant: usize,
    chunk_size: usize,
    pal_overhead: f32,
    k_winners_scratch: &mut CudaSlice<u8>,
    v_winners_scratch: &mut CudaSlice<u8>,
    kv_sums: &mut CudaSlice<f32>,
    dev: &CudaDevice,
    dtoh_stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    pinned_dst: &mut [f32],
) -> Result<cudarc::driver::CudaEvent> {
    let n_sums = (n_k_thresholds + n_v_thresholds) * 3;
    let n_cells = n_chunks
        .checked_mul(n_kv_head)
        .and_then(|v| v.checked_mul(head_dim))
        .ok_or_else(|| crate::Error::Msg("select_and_summarize: n_cells overflow".into()))?;
    let n_bh = n_chunks
        .checked_mul(n_kv_head)
        .ok_or_else(|| crate::Error::Msg("select_and_summarize: n_bh overflow".into()))?;

    if n_cells == 0 || n_quant == 0 {
        // Fill pinned destination on the CPU (no kernel launch needed) and return
        // an event that fires as soon as the GPU processes the (empty) dtoh_stream.
        let last = (n_quant.saturating_sub(1) as f32) * chunk_size as f32 * (n_cells as f32);
        pinned_dst[..n_sums].fill(last);
        return dtoh_stream.record_event(None).map_err(crate::Error::wrap);
    }

    {
        let stream = dev.cuda_stream();
        let (k_err_ptr, _ke) = k_errors.device_ptr(&stream);
        let (v_err_ptr, _ve) = v_errors.device_ptr(&stream);
        let (k_win_ptr, _kw) = k_winners_scratch.device_ptr_mut(&stream);
        let (v_win_ptr, _vw) = v_winners_scratch.device_ptr_mut(&stream);
        let (kv_sums_ptr, _ks) = kv_sums.device_ptr_mut(&stream);
        // V sums start immediately after the K sums in the combined buffer.
        let v_sums_ptr = kv_sums_ptr + (n_k_thresholds * 3 * std::mem::size_of::<f32>()) as u64;

        unsafe {
            // 1. Select winners into on-device scratch buffers (never downloaded).
            crate::set_kernel_breadcrumb(
                "run_select_winners_kv_paged (fused-sweep)",
                file!(),
                line!(),
            );
            run_select_winners_kv_paged(
                k_err_ptr as *const f32,
                v_err_ptr as *const f32,
                k_thresholds_dev_ptr as *const f32,
                v_thresholds_dev_ptr as *const f32,
                k_win_ptr as *mut u8,
                v_win_ptr as *mut u8,
                n_k_thresholds as i32,
                n_v_thresholds as i32,
                n_cells as i32,
                n_quant as i32,
                n_kv_head as i32,
                head_dim as i32,
            );

            // 2. Summarize K winners → kv_sums[0 .. n_k_thresholds*3].
            crate::set_kernel_breadcrumb("run_summarize_winners_side_paged (K)", file!(), line!());
            run_summarize_winners_side_paged(
                k_win_ptr as *const u8,
                candidates_bpe_dev_ptr as *const f32,
                kv_sums_ptr as *mut f32,
                n_k_thresholds as i32,
                n_cells as i32,
                n_bh as i32,
                head_dim as i32,
                n_quant as i32,
                chunk_size as i32,
                pal_overhead,
            );

            // 3. Summarize V winners → kv_sums[n_k_thresholds*3 ..].
            crate::set_kernel_breadcrumb("run_summarize_winners_side_paged (V)", file!(), line!());
            run_summarize_winners_side_paged(
                v_win_ptr as *const u8,
                candidates_bpe_dev_ptr as *const f32,
                v_sums_ptr as *mut f32,
                n_v_thresholds as i32,
                n_cells as i32,
                n_bh as i32,
                head_dim as i32,
                n_quant as i32,
                chunk_size as i32,
                pal_overhead,
            );
        }

        // Record when all three kernels are done on the compute stream, then let
        // the D2H stream GPU-wait for that event before issuing the async DMA.
        // This serialises compute → DMA on the GPU without blocking the CPU.
        let e_compute_done = stream.record_event(None).map_err(crate::Error::wrap)?;
        dtoh_stream
            .wait(&e_compute_done)
            .map_err(crate::Error::wrap)?;
    } // All device_ptr guards released here.

    // Enqueue async DtoH into the pre-allocated pinned host slice.
    // `stream.memcpy_dtoh` does NOT synchronise the CPU — it merely enqueues the DMA.
    let kv_slice = kv_sums.slice(0..n_sums);
    dtoh_stream
        .memcpy_dtoh(&kv_slice, pinned_dst)
        .map_err(crate::Error::wrap)?;

    // Record an event AFTER the DMA so callers can `event.synchronize()` when
    // they need the data, rather than stalling immediately.
    dtoh_stream.record_event(None).map_err(crate::Error::wrap)
}

pub fn ggml_to_select_qtype(dtype: GgmlDType) -> Result<i32> {
    match dtype {
        GgmlDType::Q4_0 => Ok(QType::Q4_0 as i32),
        GgmlDType::Q4_1 => Ok(QType::Q4_1 as i32),
        GgmlDType::Q5_0 => Ok(QType::Q5_0 as i32),
        GgmlDType::Q5_1 => Ok(QType::Q5_1 as i32),
        GgmlDType::Q8_0 => Ok(QType::Q8_0 as i32),
        GgmlDType::Q8_1 => Ok(QType::Q8_1 as i32),
        GgmlDType::Q4_KS => Ok(QType::Q4_KS as i32),
        GgmlDType::Q8_KS => Ok(QType::Q8_KS as i32),
        GgmlDType::Q2_0 => Ok(QType::Q2_0 as i32),
        GgmlDType::Q3_0 => Ok(QType::Q3_0 as i32),
        GgmlDType::R16 => Ok(QType::R16 as i32),
        GgmlDType::Q0 => Ok(QType::Q0 as i32),
        GgmlDType::Q1_S => Ok(QType::Q1_S as i32),
        GgmlDType::Q2_S => Ok(QType::Q2_S as i32),
        GgmlDType::Q2_A => Ok(QType::Q2_A as i32),
        GgmlDType::Q2_1 => Ok(QType::Q2_1 as i32),
        GgmlDType::Q3_1 => Ok(QType::Q3_1 as i32),
        GgmlDType::Q0_V => Ok(QType::Q0_V as i32),
        GgmlDType::Q1_A => Ok(QType::Q1_A as i32),
        GgmlDType::Q0_X => Ok(QType::Q0_X as i32),
        GgmlDType::Q0_M2 => Ok(QType::Q0_M2 as i32),
        GgmlDType::Q0_M4 => Ok(QType::Q0_M4 as i32),
        GgmlDType::F16 => Ok(SELECT_FMT_F16),
        GgmlDType::BF16 => Ok(SELECT_FMT_BF16),
        _ => crate::bail!("select_kv_format: unsupported format {:?}", dtype),
    }
}

/// Returns 4× the bits-per-element for a QType code (matches CUDA `format_bpe_x4`).
/// Used to sort candidates ascending by BPE before uploading to the GPU kernel.
///
/// Derives from `GgmlDType::type_size() * 32 / GgmlDType::block_size()` which is
/// `bits_per_elem × 4` (×4 gives exact integers for fractional-bpe formats).
/// This chains through `size_of::<BlockX>()`, which is itself locked to the CUDA
/// `block_x` struct via the `static_assert(sizeof(block_x) == N)` in
/// `candle-kernels/src/blocks.cuh`. So Rust block struct → CUDA struct →
/// kernel's `format_bpe_x4` all stay in sync automatically.
///
/// Unknown codes (or formats without a block struct that's a kernel candidate)
/// return 256 — the worst-case sentinel matching CUDA's `format_bpe_x4` default.
pub fn select_qtype_bpe_x4(code: i32) -> i32 {
    match select_qtype_to_ggml(code) {
        Ok(d) => {
            let ts = d.type_size() as i32;
            let bs = d.block_size() as i32;
            // bpe_x4 = (bytes × 8 / block_size) × 4 = bytes × 32 / block_size.
            // For 32-element blocks this collapses to `bytes`, which matches
            // the kernel's "bpe_x4 ≡ block_bytes for 32-element blocks" identity.
            ts * 32 / bs
        }
        Err(_) => 256, // unknown → worst, matches CUDA default
    }
}

#[cfg(test)]
mod select_qtype_bpe_x4_tests {
    use super::*;

    /// Cross-check against the CUDA `format_bpe_x4` table in
    /// `candle-kernels/src/quantize/select_kv_format.cuh:346-373`. If a block
    /// struct changes size, both this test and the kernel `static_assert`
    /// would fire — so a single failure here is the canary.
    #[test]
    fn matches_cuda_format_bpe_x4() {
        let cases: &[(i32, i32)] = &[
            (SELECT_FMT_F16, 64),
            (SELECT_FMT_BF16, 64),
            (QType::Q8_KS as i32, 36),
            (QType::Q8_1 as i32, 36),
            (QType::Q8_0 as i32, 34),
            (QType::Q5_1 as i32, 24),
            (QType::Q5_0 as i32, 22),
            (QType::Q4_KS as i32, 20),
            (QType::Q4_1 as i32, 20),
            (QType::Q4_0 as i32, 18),
            (QType::Q3_1 as i32, 16),
            (QType::Q3_0 as i32, 14),
            (QType::Q2_1 as i32, 12),
            (QType::Q2_A as i32, 10),
            (QType::Q2_0 as i32, 10),
            (QType::Q2_S as i32, 9),
            (QType::Q0_M4 as i32, 8),
            (QType::Q1_A as i32, 6),
            (QType::Q1_S as i32, 5),
            (QType::Q0_M2 as i32, 3),
            (QType::Q0_V as i32, 2),
            (QType::Q0_X as i32, 2),
            (QType::Q0 as i32, 1),
        ];
        for &(code, expected) in cases {
            let got = select_qtype_bpe_x4(code);
            assert_eq!(
                got, expected,
                "select_qtype_bpe_x4({code}) = {got}, expected {expected} (kernel format_bpe_x4)"
            );
        }
    }
}

/// Convert a QType integer code (from GPU output) back to GgmlDType.
pub fn select_qtype_to_ggml(code: i32) -> Result<GgmlDType> {
    match code {
        c if c == QType::Q4_0 as i32 => Ok(GgmlDType::Q4_0),
        c if c == QType::Q4_1 as i32 => Ok(GgmlDType::Q4_1),
        c if c == QType::Q5_0 as i32 => Ok(GgmlDType::Q5_0),
        c if c == QType::Q5_1 as i32 => Ok(GgmlDType::Q5_1),
        c if c == QType::Q8_0 as i32 => Ok(GgmlDType::Q8_0),
        c if c == QType::Q8_1 as i32 => Ok(GgmlDType::Q8_1),
        c if c == QType::Q4_KS as i32 => Ok(GgmlDType::Q4_KS),
        c if c == QType::Q8_KS as i32 => Ok(GgmlDType::Q8_KS),
        c if c == QType::Q2_0 as i32 => Ok(GgmlDType::Q2_0),
        c if c == QType::Q3_0 as i32 => Ok(GgmlDType::Q3_0),
        c if c == QType::R16 as i32 => Ok(GgmlDType::R16),
        c if c == QType::Q0 as i32 => Ok(GgmlDType::Q0),
        c if c == QType::Q1_S as i32 => Ok(GgmlDType::Q1_S),
        c if c == QType::Q2_S as i32 => Ok(GgmlDType::Q2_S),
        c if c == QType::Q2_A as i32 => Ok(GgmlDType::Q2_A),
        c if c == QType::Q2_1 as i32 => Ok(GgmlDType::Q2_1),
        c if c == QType::Q3_1 as i32 => Ok(GgmlDType::Q3_1),
        c if c == QType::Q0_V as i32 => Ok(GgmlDType::Q0_V),
        c if c == QType::Q1_A as i32 => Ok(GgmlDType::Q1_A),
        c if c == QType::Q0_X as i32 => Ok(GgmlDType::Q0_X),
        c if c == QType::Q0_M2 as i32 => Ok(GgmlDType::Q0_M2),
        c if c == QType::Q0_M4 as i32 => Ok(GgmlDType::Q0_M4),
        SELECT_FMT_F16 => Ok(GgmlDType::F16),
        SELECT_FMT_BF16 => Ok(GgmlDType::BF16),
        _ => crate::bail!("select_qtype_to_ggml: unknown QType code {}", code),
    }
}

/// Calculate the required buffer size in bytes for quantizing `elem_count` f32 elements.
///
/// # Arguments
/// * `elem_count` - Number of f32 elements to quantize
/// * `dtype` - Target quantized dtype
///
/// # Returns
/// Required buffer size in bytes
pub fn quantized_size(elem_count: usize, dtype: GgmlDType) -> usize {
    let block_size = dtype.block_size();
    let num_blocks = elem_count.div_ceil(block_size);
    num_blocks * dtype.type_size()
}

fn dequantize_f32(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if dtype == GgmlDType::MXFP4 {
        // MXFP4 has no QType slot (kept off the locked QTYPE tables) — standalone kernel.
        let dst = unsafe { dev.alloc::<f32>(elem_count)? };
        {
            let stream = dev.cuda_stream();
            let (data_ptr, _dg) = data.inner.device_ptr(&stream);
            let (dst_ptr, _og) = dst.device_ptr(&stream);
            unsafe {
                candle_kernels::simple::quantized::run_dequantize_mxfp4(
                    data_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    elem_count as i32,
                    0,
                );
            }
        }
        return Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()));
    }
    let qtype = dtype_to_qtype(dtype)?;
    let dst = unsafe { dev.alloc::<f32>(elem_count)? };
    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = data.inner.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
        unsafe {
            run_dequantize_block(
                data_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                elem_count as i32,
                qtype as i32,
                DequantOutDType::F32 as i32,
            );
        }
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_f16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if dtype == GgmlDType::MXFP4 {
        let dst = unsafe { dev.alloc::<f16>(elem_count)? };
        {
            let stream = dev.cuda_stream();
            let (data_ptr, _dg) = data.inner.device_ptr(&stream);
            let (dst_ptr, _og) = dst.device_ptr(&stream);
            unsafe {
                candle_kernels::simple::quantized::run_dequantize_mxfp4(
                    data_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    elem_count as i32,
                    1,
                );
            }
        }
        return Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()));
    }
    let qtype = dtype_to_qtype(dtype)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = data.inner.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
        unsafe {
            run_dequantize_block(
                data_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                elem_count as i32,
                qtype as i32,
                DequantOutDType::F16 as i32,
            );
        }
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_bf16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    use half::bf16;
    if dtype == GgmlDType::MXFP4 {
        let dst = unsafe { dev.alloc::<bf16>(elem_count)? };
        {
            let stream = dev.cuda_stream();
            let (data_ptr, _dg) = data.inner.device_ptr(&stream);
            let (dst_ptr, _og) = dst.device_ptr(&stream);
            unsafe {
                candle_kernels::simple::quantized::run_dequantize_mxfp4(
                    data_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    elem_count as i32,
                    2,
                );
            }
        }
        return Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()));
    }
    let qtype = dtype_to_qtype(dtype)?;
    let dst = unsafe { dev.alloc::<bf16>(elem_count)? };
    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = data.inner.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
        unsafe {
            run_dequantize_block(
                data_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                elem_count as i32,
                qtype as i32,
                DequantOutDType::BF16 as i32,
            );
        }
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_mul_mat_vec(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    let qtype = dtype_to_qtype(dtype)?;
    let dst = unsafe { dev.alloc::<f32>(nrows)? };
    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = data.inner.device_ptr(&stream);
        let (y_ptr, _y_guard) = y.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
        unsafe {
            run_dequantize_mul_mat_vec(
                data_ptr as *const std::ffi::c_void,
                y_ptr as *const f32,
                dst_ptr as *mut f32,
                ncols as i32,
                nrows as i32,
                qtype as i32,
            );
        }
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// `inherit` is the **activation's** arena. The weight is a model parameter and
/// names none, so a quantized projection's output belongs wherever the value
/// flowing through the layer came from — and so does the q8_1 staging buffer,
/// which dies at the end of this call.
#[allow(clippy::too_many_arguments)]
fn mul_mat_vec_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
    inherit: Backing,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if b_size == 0 || b_size > 8 {
        crate::bail!("only bsize between 1 and 8 are supported, got {b_size}")
    }
    // Start by quantizing y
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    // The staging buffer stays on the pool. It is a bare `CudaSlice` that this
    // function drops, and dropping a lease means `cuMemFreeAsync` on an address
    // inside the VMM reservation — which the driver rejects and cudarc records,
    // once per call. Only `dst` inherits, because only `dst` is wrapped in a
    // storage that carries its backing.
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    quantize_q8_1(y, &mut y_q8_1, ncols, b_size, dev)?;

    let qtype = dtype_to_qtype(dtype)?;
    let (dst, dst_backing) = unsafe { alloc_inheriting::<f32>(dev, nrows * b_size, inherit)? };
    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = data.inner.device_ptr(&stream);
        let (y_q8_1_ptr, _y_guard) = y_q8_1.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
        unsafe {
            run_mul_mat_vec_q8_1(
                data_ptr as *const std::ffi::c_void,
                y_q8_1_ptr as *const std::ffi::c_void,
                dst_ptr as *mut f32,
                ncols as i32,
                nrows as i32,
                ncols_padded as i32,
                nrows as i32,
                b_size as i32,
                qtype as i32,
            );
        }
    }
    Ok(CudaStorage::wrap_cuda_slice_backed(
        dst,
        dev.clone(),
        dst_backing,
    ))
}

/// As [`mul_mat_vec_via_q8_1`], including what `inherit` means.
#[allow(clippy::too_many_arguments)]
fn mul_mat_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &CudaDevice,
    inherit: Backing,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        crate::bail!("unexpected lhs size {}, {x_rows} {x_cols}", data_elems)
    }
    if y.len() != y_rows * y_cols {
        crate::bail!("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if x_cols != y_rows {
        crate::bail!("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    let k = x_cols;
    // Start by quantizing y
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        k_padded * y_cols * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    // Pool, for the reason given in [`mul_mat_vec_via_q8_1`].
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    quantize_q8_1(y, &mut y_q8_1, k, y_cols, dev)?;

    let qtype = dtype_to_qtype(dtype)?;
    let (dst, dst_backing) = unsafe { alloc_inheriting::<f32>(dev, x_rows * y_cols, inherit)? };
    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = data.inner.device_ptr(&stream);
        let (y_q8_1_ptr, _y_guard) = y_q8_1.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
        unsafe {
            run_mul_mat(
                data_ptr as *const std::ffi::c_void,
                y_q8_1_ptr as *const std::ffi::c_void,
                dst_ptr as *mut f32,
                x_cols as i32,
                x_rows as i32,
                y_cols as i32,
                k_padded as i32,
                x_rows as i32,
                qtype as i32,
            );
        }
    }
    Ok(CudaStorage::wrap_cuda_slice_backed(
        dst,
        dev.clone(),
        dst_backing,
    ))
}

/// Float dense quantized matmul shared by `QCudaStorage::matmul_gemx` and the `Float`
/// arm of [`dense_qmatmul`]: quantized weight `[nrows(N) x ncols(K)]` (qtype, `weight_len`
/// bytes at `weight_ptr`) times float activations `rhs` -> same-dtype output `[.. x N]`.
/// An unsupported activation dtype is converted to BF16 and the output converted back.
#[allow(clippy::too_many_arguments)]
fn dense_qmatmul_float(
    weight_ptr: u64,
    qtype: i32,
    weight_len: usize,
    nrows: usize,
    ncols: usize,
    rhs: &CudaStorage,
    rhs_l: &crate::Layout,
    device: &CudaDevice,
) -> Result<(CudaStorage, crate::Shape)> {
    use crate::cuda_backend::CudaStorageSlice;

    let (batch_size, k) = match rhs_l.shape().dims() {
        [b, m, k] => (b * m, *k),
        [b, k] => (*b, *k),
        _ => crate::bail!(
            "unexpected rhs shape in quantized_matmul {:?}",
            rhs_l.shape()
        ),
    };
    if ncols != k {
        crate::bail!(
            "mismatch on matmul dim N={nrows} K={ncols} vs rhs {:?}",
            rhs_l.shape()
        )
    }

    let input_dtype = rhs.dtype();
    let needs_conversion = !matches!(
        input_dtype,
        crate::DType::F16 | crate::DType::BF16 | crate::DType::F32
    );
    let rhs_converted: Option<CudaStorage> = if needs_conversion {
        Some(rhs.to_dtype(rhs_l, crate::DType::BF16)?)
    } else {
        None
    };
    let rhs_storage = rhs_converted.as_ref().unwrap_or(rhs);
    let rhs_layout_owned;
    let rhs_layout = if rhs_converted.is_some() {
        rhs_layout_owned = crate::Layout::contiguous(rhs_l.shape());
        &rhs_layout_owned
    } else {
        rhs_l
    };

    let ytype = match &rhs_storage.slice {
        CudaStorageSlice::F16(_) => YType::F16,
        CudaStorageSlice::BF16(_) => YType::BF16,
        CudaStorageSlice::F32(_) => YType::F32,
        _ => unreachable!("should have been converted to BF16"),
    };

    enum OutputSlice {
        F16(CudaSlice<f16>),
        BF16(CudaSlice<bf16>),
        F32(CudaSlice<f32>),
    }
    let dst_slice = match ytype {
        YType::F16 => OutputSlice::F16(unsafe { device.alloc::<f16>(nrows * batch_size)? }),
        YType::BF16 => OutputSlice::BF16(unsafe { device.alloc::<bf16>(nrows * batch_size)? }),
        YType::F32 => OutputSlice::F32(unsafe { device.alloc::<f32>(nrows * batch_size)? }),
        YType::Q8A128 => {
            crate::bail!("YType::Q8A128 is the int8 matmul INPUT format, not an FP output dtype")
        }
    };

    {
        let stream = device.cuda_stream();
        let segment = VxSegment {
            weights: weight_ptr as *const std::ffi::c_void,
            batch_count: batch_size as i32,
        };
        macro_rules! run_matmul {
            ($y_ptr:expr, $dst_ptr:expr) => {{
                let status = unsafe {
                    run_quantized_matmul(
                        &segment as *const VxSegment,
                        1,
                        $y_ptr as *const std::ffi::c_void,
                        $dst_ptr as *mut std::ffi::c_void,
                        ncols as i32,
                        nrows as i32,
                        k as i32,
                        nrows as i32,
                        qtype,
                        ytype as i32,
                        weight_len,
                        0, // force_mode2: FP path ignores it
                        // out_dtype: the FP kernels store at the activation dtype, so this
                        // only has to be a valid code — it is not consulted.
                        OutDType::F32 as i32,
                    )
                };
                check_matmul_status(status, "dense_qmatmul_float")?;
            }};
        }
        match (&rhs_storage.slice, &dst_slice) {
            (CudaStorageSlice::F16(y_data), OutputSlice::F16(dst)) => {
                let y_view = match rhs_layout.contiguous_offsets() {
                    Some((o1, o2)) => y_data.slice(o1..o2),
                    None => {
                        return Err(crate::Error::RequiresContiguous {
                            op: "quantized_matmul",
                        }
                        .bt())?
                    }
                };
                let (y_ptr, _y_guard) = y_view.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
                run_matmul!(y_ptr, dst_ptr);
            }
            (CudaStorageSlice::BF16(y_data), OutputSlice::BF16(dst)) => {
                let y_view = match rhs_layout.contiguous_offsets() {
                    Some((o1, o2)) => y_data.slice(o1..o2),
                    None => {
                        return Err(crate::Error::RequiresContiguous {
                            op: "quantized_matmul",
                        }
                        .bt())?
                    }
                };
                let (y_ptr, _y_guard) = y_view.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
                run_matmul!(y_ptr, dst_ptr);
            }
            (CudaStorageSlice::F32(y_data), OutputSlice::F32(dst)) => {
                let y_view = match rhs_layout.contiguous_offsets() {
                    Some((o1, o2)) => y_data.slice(o1..o2),
                    None => {
                        return Err(crate::Error::RequiresContiguous {
                            op: "quantized_matmul",
                        }
                        .bt())?
                    }
                };
                let (y_ptr, _y_guard) = y_view.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
                run_matmul!(y_ptr, dst_ptr);
            }
            _ => unreachable!("ytype and rhs_storage slice should match"),
        }
    }

    let out_storage = match dst_slice {
        OutputSlice::F16(dst) => CudaStorage::wrap_cuda_slice(dst, device.clone()),
        OutputSlice::BF16(dst) => CudaStorage::wrap_cuda_slice(dst, device.clone()),
        OutputSlice::F32(dst) => CudaStorage::wrap_cuda_slice(dst, device.clone()),
    };
    let mut out_shape = rhs_l.shape().dims().to_vec();
    out_shape.pop();
    out_shape.push(nrows);
    let out_shape: crate::Shape = out_shape.into();
    if needs_conversion {
        let out_layout = crate::Layout::contiguous(&out_shape);
        let converted_out = out_storage.to_dtype(&out_layout, input_dtype)?;
        Ok((converted_out, out_shape))
    } else {
        Ok((out_storage, out_shape))
    }
}

impl QCudaStorage {
    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes)?;
        Ok(QCudaStorage {
            data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            }),
            device: device.clone(),
            dtype,
            backing: Backing::Owned,
        })
    }

    /// Same as [`QCudaStorage::zeros`], but the bytes are left as they are.
    ///
    /// For storage a kernel fills immediately, where the zero-fill is a full
    /// memset of the buffer discarded microseconds later by the write that
    /// follows it.
    ///
    /// # Safety
    ///
    /// The caller must write the whole `len`-byte data region before anything
    /// reads it. The `MATRIX_ROW_PADDING` tail past `len` is uninitialised too,
    /// so this is wrong for any storage handed to a q-matmul kernel — those
    /// over-read into the padding by design, and [`QCudaStorage::zeros`] is
    /// what makes that read a defined zero.
    pub unsafe fn uninit(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        unsafe { Self::uninit_from(device, el_count, dtype, Backing::Owned) }
    }

    /// [`Self::uninit`] with the storage taken from `origin`'s arena.
    ///
    /// For quantized staging that is written and consumed inside one scope — the
    /// embedding gather, whose bytes are read once by the dequantize that
    /// immediately follows. Carving that from the wave costs no VRAM at all: the
    /// span is already reserved, so the bytes come out of a budget that has
    /// been paid for whether or not anything uses it.
    ///
    /// # Safety
    /// As [`Self::uninit`]: the storage is uninitialised, so every byte a reader
    /// touches must be written first.
    pub unsafe fn uninit_from(
        device: &CudaDevice,
        el_count: usize,
        dtype: GgmlDType,
        origin: Backing,
    ) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let (inner, backing) =
            unsafe { alloc_inheriting::<u8>(device, padded_size_in_bytes, origin)? };
        Ok(QCudaStorage {
            data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            }),
            device: device.clone(),
            dtype,
            backing,
        })
    }

    /// Create a `QCudaStorage` backed by host-mapped (pinned) memory.
    ///
    /// The quantized data is copied into a `cudaHostAlloc`-allocated buffer
    /// with `CU_MEMHOSTALLOC_DEVICEMAP`. CUDA kernels access this memory
    /// transparently over PCIe â€” **no VRAM is consumed**.
    ///
    /// Returns `(storage, guard)`. The caller must keep `guard` alive for the
    /// lifetime of the storage; dropping it frees the pinned host buffer.
    ///
    /// This is the correct overflow path when VRAM budget is exceeded: tensors
    /// still work with all CUDA matmul kernels, just at PCIe bandwidth.
    pub fn from_host_mapped(
        data: &[u8],
        elem_count: usize,
        dtype: GgmlDType,
        device: &CudaDevice,
    ) -> Result<(Self, HostMappedAlloc)> {
        let size_in_bytes = data.len();
        let padded_size_in_bytes =
            ceil_div(elem_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let (host_ptr, _dev_ptr, guard) = alloc_host_mapped(padded_size_in_bytes)?;

        // Copy quantized data into the pinned host buffer
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), host_ptr, size_in_bytes);
            // Zero the padding region
            if padded_size_in_bytes > size_in_bytes {
                std::ptr::write_bytes(
                    host_ptr.add(size_in_bytes),
                    0u8,
                    padded_size_in_bytes - size_in_bytes,
                );
            }
        }

        // Wrap the pinned host buffer as a CudaSlice via a device H2D copy: the
        // QCudaStorage API needs an owned CudaSlice, and cudarc has no way to wrap an
        // external device pointer (from cuMemHostGetDevicePointer) into a CudaSlice
        // without ownership, so the mapped-host zero-copy path is not expressible here.
        // The HostMappedAlloc guard keeps the pinned buffer alive alongside the VRAM copy.
        let pinned_slice = unsafe { std::slice::from_raw_parts(host_ptr, size_in_bytes) };
        let mut inner = unsafe { device.alloc::<u8>(padded_size_in_bytes)? };
        device.memcpy_htod(pinned_slice, &mut inner.slice_mut(..size_in_bytes))?;

        let storage = QCudaStorage {
            data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            }),
            device: device.clone(),
            dtype,
            backing: Backing::Owned,
        };
        Ok((storage, guard))
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
        fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) {
            let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
            let vec = slice.to_vec();
            T::to_float(&vec, dst)
        }

        let fast_kernel = matches!(
            self.dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q8_1
                | GgmlDType::Q2_K
                | GgmlDType::Q3_K
                | GgmlDType::Q4_K
                | GgmlDType::Q5_K
                | GgmlDType::Q6_K
                | GgmlDType::Q8_K
                | GgmlDType::QAWQ
                | GgmlDType::QAWQ_G64
                | GgmlDType::Q4_KS
                | GgmlDType::Q8_KS
                | GgmlDType::Q2_0
                | GgmlDType::Q3_0
                | GgmlDType::R16
                | GgmlDType::Q0
                | GgmlDType::Q0_V
                | GgmlDType::Q1_A
                | GgmlDType::Q0_X
                | GgmlDType::Q0_M2
                | GgmlDType::Q0_M4
                | GgmlDType::Q1_S
                | GgmlDType::Q2_S
                | GgmlDType::Q2_A
                | GgmlDType::Q2_1
                | GgmlDType::Q3_1
                | GgmlDType::MXFP4
        );
        if fast_kernel {
            return dequantize_f32(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.

        let buffer = self
            .device
            .memcpy_dtov(&self.data.inner.slice(..self.data.len))?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F64 => deq::<f64>(&buffer, block_len, &mut out),
            GgmlDType::F32 => deq::<f32>(&buffer, block_len, &mut out),
            GgmlDType::U8 => deq::<u8>(&buffer, block_len, &mut out),
            GgmlDType::I8 => deq::<i8>(&buffer, block_len, &mut out),
            GgmlDType::U16 => deq::<u16>(&buffer, block_len, &mut out),
            GgmlDType::I16 => deq::<i16>(&buffer, block_len, &mut out),
            GgmlDType::U32 => deq::<u32>(&buffer, block_len, &mut out),
            GgmlDType::I32 => deq::<i32>(&buffer, block_len, &mut out),
            GgmlDType::U64 => deq::<u64>(&buffer, block_len, &mut out),
            GgmlDType::I64 => deq::<i64>(&buffer, block_len, &mut out),
            GgmlDType::F16 => deq::<half::f16>(&buffer, block_len, &mut out),
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, block_len, &mut out),
            GgmlDType::F8E4M3 => panic!("not implemented"),
            GgmlDType::F8E5M2 => panic!("not implemented"),
            GgmlDType::MXFP4 => {
                deq::<crate::quantized::k_quants::BlockMXFP4>(&buffer, block_len, &mut out)
            }
            GgmlDType::Q4_0 => deq::<crate::quantized::BlockQ4_0>(&buffer, block_len, &mut out),
            GgmlDType::Q4_1 => deq::<crate::quantized::BlockQ4_1>(&buffer, block_len, &mut out),
            GgmlDType::Q5_0 => deq::<crate::quantized::BlockQ5_0>(&buffer, block_len, &mut out),
            GgmlDType::Q5_1 => deq::<crate::quantized::BlockQ5_1>(&buffer, block_len, &mut out),
            GgmlDType::Q8_0 => deq::<crate::quantized::BlockQ8_0>(&buffer, block_len, &mut out),
            GgmlDType::Q8_1 => deq::<crate::quantized::BlockQ8_1>(&buffer, block_len, &mut out),
            GgmlDType::Q2_K => deq::<crate::quantized::BlockQ2_K>(&buffer, block_len, &mut out),
            GgmlDType::Q3_K => deq::<crate::quantized::BlockQ3_K>(&buffer, block_len, &mut out),
            GgmlDType::Q4_K => deq::<crate::quantized::BlockQ4_K>(&buffer, block_len, &mut out),
            GgmlDType::Q5_K => deq::<crate::quantized::BlockQ5_K>(&buffer, block_len, &mut out),
            GgmlDType::Q6_K => deq::<crate::quantized::BlockQ6_K>(&buffer, block_len, &mut out),
            GgmlDType::Q8_K => deq::<crate::quantized::BlockQ8_K>(&buffer, block_len, &mut out),
            GgmlDType::QAWQ => deq::<crate::quantized::BlockQAWQ>(&buffer, block_len, &mut out),
            GgmlDType::QAWQ_G64 => {
                deq::<crate::quantized::BlockQAWQ_G64>(&buffer, block_len, &mut out)
            }
            GgmlDType::Q4_KS => deq::<crate::quantized::BlockQ4_KS>(&buffer, block_len, &mut out),
            GgmlDType::Q8_KS => deq::<crate::quantized::BlockQ8_KS>(&buffer, block_len, &mut out),
            GgmlDType::Q2_0 => deq::<crate::quantized::BlockQ2_0>(&buffer, block_len, &mut out),
            GgmlDType::Q3_0 => deq::<crate::quantized::BlockQ3_0>(&buffer, block_len, &mut out),
            GgmlDType::R16 => deq::<crate::quantized::BlockR16>(&buffer, block_len, &mut out),
            GgmlDType::Q0 => deq::<crate::quantized::BlockQ0>(&buffer, block_len, &mut out),
            GgmlDType::Q1_S => deq::<crate::quantized::BlockQ1S>(&buffer, block_len, &mut out),
            GgmlDType::Q2_S => deq::<crate::quantized::BlockQ2S>(&buffer, block_len, &mut out),
            GgmlDType::Q2_A => deq::<crate::quantized::BlockQ2A>(&buffer, block_len, &mut out),
            GgmlDType::Q2_1 => deq::<crate::quantized::BlockQ2_1>(&buffer, block_len, &mut out),
            GgmlDType::Q3_1 => deq::<crate::quantized::BlockQ3_1>(&buffer, block_len, &mut out),
            GgmlDType::P2 => deq::<crate::quantized::BlockP2>(&buffer, block_len, &mut out),
            GgmlDType::Q0_V => deq::<crate::quantized::BlockQ0V>(&buffer, block_len, &mut out),
            GgmlDType::Q1_A => deq::<crate::quantized::BlockQ1A>(&buffer, block_len, &mut out),
            GgmlDType::Q0_X => deq::<crate::quantized::BlockQ0X>(&buffer, block_len, &mut out),
            GgmlDType::Q0_M2 => deq::<crate::quantized::BlockQ0M2>(&buffer, block_len, &mut out),
            GgmlDType::Q0_M4 => deq::<crate::quantized::BlockQ0M4>(&buffer, block_len, &mut out),
            GgmlDType::Q4_KO => deq::<crate::quantized::BlockQ4_KO>(&buffer, block_len, &mut out),
            GgmlDType::Q5_KO => deq::<crate::quantized::BlockQ5_KO>(&buffer, block_len, &mut out),
            GgmlDType::Q6_KO => deq::<crate::quantized::BlockQ6_KO>(&buffer, block_len, &mut out),
            GgmlDType::Q8_KO => deq::<crate::quantized::BlockQ8_KO>(&buffer, block_len, &mut out),
            // MXFP4_KO is a GPU-only lane-major chunk with no per-128 host block codec; it is
            // never CPU-dequantized (the per-sub fold lives in the int8 kernel).
            GgmlDType::MXFP4_KO => {
                crate::bail!("MXFP4_KO has no CPU dequant path; it is a GPU-only int8 weight")
            }
            // Q2_KO is a GPU-only lane-major crumb chunk; the shape-aware `ko_quant::dequant_q2_ko`
            // (test/prepare only) reconstructs it — this flat per-block path does not apply.
            GgmlDType::Q2_KO => {
                crate::bail!("Q2_KO has no CPU dequant path; it is a GPU-only int8 weight")
            }
        }

        self.device
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<CudaStorage> {
        dequantize_f16(&self.data, self.dtype, elem_count, self.device())
    }

    pub fn dequantize_bf16(&self, elem_count: usize) -> Result<CudaStorage> {
        dequantize_bf16(&self.data, self.dtype, elem_count, self.device())
    }

    pub fn byte_len(&self) -> usize {
        self.data.len
    }

    pub fn bytes(&self) -> CudaView<'_, u8> {
        self.data.inner.slice(..self.data.len)
    }

    pub fn bytes_mut(&mut self) -> cudarc::driver::CudaViewMut<'_, u8> {
        // `len` is read out first: `data` is `ManuallyDrop`, so both halves of
        // this expression go through one `DerefMut` borrow rather than being
        // disjoint field accesses.
        let len = self.data.len;
        self.data.inner.slice_mut(..len)
    }

    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.memcpy_dtov(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(data.as_ref(), &mut inner.slice_mut(..data.len()))?;
        // Dispose of the buffer being replaced before overwriting it: `data` is
        // `ManuallyDrop`, so an assignment alone would leak it. A lease must be
        // `leak`ed rather than dropped — its memory belongs to an arena — while
        // an owned buffer is freed.
        // SAFETY: the old value is moved out exactly once and not read again.
        let old = unsafe { std::mem::ManuallyDrop::take(&mut self.data) };
        if matches!(self.backing, Backing::Lease(_)) {
            old.inner.leak();
        } else {
            drop(old);
        }
        self.data = std::mem::ManuallyDrop::new(PaddedCudaSlice {
            inner,
            len: data.len(),
        });
        // The buffer identity just changed, so the old `backing` no longer
        // describes it. On a storage built by `from_leased_device_ptr` this is
        // load-bearing twice over: the quantized bytes have landed in the fresh
        // allocation rather than the arena slot the lease pointed at, and
        // leaving `Lease` set makes `Drop` `leak()` that fresh allocation —
        // a permanent VRAM leak of `padded_len` per call. `Clone` was already
        // taught to force `Owned` for exactly this reason; this is the other
        // place the buffer is replaced.
        self.backing = Backing::Owned;
        Ok(())
    }

    /// Quantize f32 data directly into this storage at the given byte offset.
    ///
    /// This avoids intermediate allocations by using the GPU quantization kernel
    /// to write directly into the target buffer.
    ///
    /// # Arguments
    /// * `src` - Source f32 CUDA storage
    /// * `elem_count` - Number of f32 elements to quantize
    /// * `byte_offset` - Byte offset in destination where quantized data should be written
    ///
    /// # Safety
    /// The caller must ensure:
    /// - byte_offset is block-aligned (multiple of dtype.type_size())
    /// - byte_offset + quantized_size(elem_count, dtype) <= self.storage_size_in_bytes()
    pub fn quantize_into(
        &mut self,
        src: &CudaStorage,
        elem_count: usize,
        byte_offset: usize,
    ) -> Result<()> {
        // Validate input dtype
        let src_f32 = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => data,
            _ => crate::bail!("quantize_into: only f32 can be quantized"),
        };

        // Validate alignment
        let block_size = self.dtype.block_size();
        let type_size = self.dtype.type_size();
        if !byte_offset.is_multiple_of(type_size) {
            crate::bail!(
                "quantize_into: byte_offset {} not aligned to block type_size {}",
                byte_offset,
                type_size
            );
        }

        // Validate bounds
        let quantized_bytes = quantized_size(elem_count, self.dtype);
        if byte_offset + quantized_bytes > self.data.len {
            crate::bail!(
                "quantize_into: offset {} + size {} exceeds storage size {}",
                byte_offset,
                quantized_bytes,
                self.data.len
            );
        }

        // Get pointers and call kernel
        let qtype = dtype_to_qtype(self.dtype)?;
        let stream = self.device.cuda_stream();
        let (src_ptr, _src_guard) = src_f32.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = self.data.inner.device_ptr_mut(&stream);

        unsafe {
            run_quantize_block(
                src_ptr as *const f32,
                (dst_ptr as *mut u8).add(byte_offset) as *mut std::ffi::c_void,
                elem_count as i32,
                qtype as i32,
            );
        }

        // Note: we silence the "unused" warning on block_size since it's used for documentation
        let _ = block_size;

        Ok(())
    }

    /// Quantize f32 data with fused transpose from [H, T, D] to [H, D, T] layout.
    ///
    /// This fuses the memory layout transformation with quantization to avoid
    /// intermediate allocations. Used for KV cache quantization where:
    /// - Input layout: [n_head, chunk_size, head_dim] - channel-oriented float
    /// - Output layout: [n_head, head_dim, chunk_size] - token-oriented quant
    ///
    /// # Arguments
    /// * `src` - Source f32 CUDA storage with shape [n_head, chunk_size, head_dim]
    /// * `n_head` - Number of KV heads
    /// * `chunk_size` - Number of tokens (must be 32 for standard quants)
    /// * `head_dim` - Dimension per head
    /// * `byte_offset` - Byte offset in destination where quantized data should be written
    ///
    /// # Supported Types
    /// Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 (all 32-element formats).
    ///
    /// # Safety
    /// The caller must ensure:
    /// - byte_offset is block-aligned
    /// - byte_offset + n_head * head_dim * type_size <= storage_size_in_bytes()
    /// - src contains n_head * chunk_size * head_dim elements (F32/F16/BF16/F8E4M3)
    pub fn quantize_transposed_into(
        &mut self,
        src: &CudaStorage,
        n_head: usize,
        chunk_size: usize,
        head_dim: usize,
        byte_offset: usize,
    ) -> Result<()> {
        // Standard 32-element formats support fused transpose+quantize
        let qtype = match self.dtype {
            GgmlDType::Q4_0 => QType::Q4_0,
            GgmlDType::Q4_1 => QType::Q4_1,
            GgmlDType::Q5_0 => QType::Q5_0,
            GgmlDType::Q5_1 => QType::Q5_1,
            GgmlDType::Q8_0 => QType::Q8_0,
            GgmlDType::Q8_1 => QType::Q8_1,
            GgmlDType::Q4_KS => QType::Q4_KS,
            GgmlDType::Q8_KS => QType::Q8_KS,
            GgmlDType::Q2_0 => QType::Q2_0,
            GgmlDType::Q3_0 => QType::Q3_0,
            GgmlDType::R16 => QType::R16,
            GgmlDType::Q0 => QType::Q0,
            GgmlDType::Q0_X => QType::Q0_X,
            GgmlDType::Q1_A => QType::Q1_A,
            GgmlDType::Q0_V => QType::Q0_V,
            GgmlDType::Q0_M2 => QType::Q0_M2,
            GgmlDType::Q0_M4 => QType::Q0_M4,
            GgmlDType::Q1_S => QType::Q1_S,
            GgmlDType::Q2_S => QType::Q2_S,
            GgmlDType::Q2_A => QType::Q2_A,
            GgmlDType::Q2_1 => QType::Q2_1,
            GgmlDType::Q3_1 => QType::Q3_1,
            _ => crate::bail!(
                "quantize_transposed_into: only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q4_KS/Q8_KS supported, got {:?}",
                self.dtype
            ),
        };

        // Validate chunk_size (all standard quants use 32)
        if chunk_size != 32 {
            crate::bail!(
                "quantize_transposed_into: chunk_size must be 32 for standard quants, got {}",
                chunk_size
            );
        }

        // Get source pointer and dtype - supports F32/F16/BF16/F8E4M3
        let (src_ptr, src_dtype) = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => {
                let stream = self.device.cuda_stream();
                let (ptr, _guard) = data.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, crate::DType::F32)
            }
            crate::cuda_backend::CudaStorageSlice::F16(data) => {
                let stream = self.device.cuda_stream();
                let (ptr, _guard) = data.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, crate::DType::F16)
            }
            crate::cuda_backend::CudaStorageSlice::BF16(data) => {
                let stream = self.device.cuda_stream();
                let (ptr, _guard) = data.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, crate::DType::BF16)
            }
            crate::cuda_backend::CudaStorageSlice::F8E4M3(data) => {
                let stream = self.device.cuda_stream();
                let (ptr, _guard) = data.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, crate::DType::F8E4M3)
            }
            _ => crate::bail!(
                "quantize_transposed_into: source dtype must be F32/F16/BF16/F8E4M3, got {:?}",
                src.dtype()
            ),
        };

        let src_dtype_code = dtype_to_src_dtype_code(src_dtype)?;

        // Validate alignment
        let type_size = self.dtype.type_size();
        if !byte_offset.is_multiple_of(type_size) {
            crate::bail!(
                "quantize_transposed_into: byte_offset {} not aligned to type_size {}",
                byte_offset,
                type_size
            );
        }

        // Validate bounds: n_head * head_dim Q8_0/Q4_0 blocks
        let num_blocks = n_head * head_dim;
        let required_bytes = num_blocks * type_size;
        if byte_offset + required_bytes > self.data.len {
            crate::bail!(
                "quantize_transposed_into: offset {} + size {} exceeds storage size {}",
                byte_offset,
                required_bytes,
                self.data.len
            );
        }

        // Get destination pointer and call typed kernel
        let stream = self.device.cuda_stream();
        let (dst_ptr, _dst_guard) = self.data.inner.device_ptr_mut(&stream);

        unsafe {
            // Use typed batched version with num_chunks=1
            run_quantize_transposed_batched_typed(
                src_ptr,
                (dst_ptr as *mut u8).add(byte_offset) as *mut std::ffi::c_void,
                std::ptr::null(), // No src offsets (contiguous)
                std::ptr::null(), // No dst offsets (contiguous)
                1,                // Single chunk
                n_head as i32,
                chunk_size as i32,
                head_dim as i32,
                qtype as i32,
                src_dtype_code,
            );
        }

        Ok(())
    }

    /// Dequantize data from this storage at the given byte offset into a destination buffer.
    ///
    /// This avoids intermediate allocations by using the GPU dequantization kernel
    /// to read from a specific offset and write directly into the target buffer.
    ///
    /// # Arguments
    /// * `dst` - Destination CUDA storage (f16/bf16/f32)
    /// * `elem_count` - Number of elements to dequantize
    /// * `src_byte_offset` - Byte offset in source (this storage) to read from
    /// * `dst_elem_offset` - Element offset in destination where dequantized data should be written
    ///
    /// # Safety
    /// The caller must ensure:
    /// - src_byte_offset is block-aligned (multiple of dtype.type_size())
    /// - src_byte_offset + quantized_size(elem_count, dtype) <= self.storage_size_in_bytes()
    /// - dst_elem_offset + elem_count <= dst element count
    pub fn dequantize_into(
        &self,
        dst: &mut CudaStorage,
        elem_count: usize,
        src_byte_offset: usize,
        dst_elem_offset: usize,
    ) -> Result<()> {
        use crate::cuda_backend::CudaStorageSlice;
        use candle_kernels::simple::quantized::run_dequantize_block;

        // Validate alignment
        let type_size = self.dtype.type_size();
        if !src_byte_offset.is_multiple_of(type_size) {
            crate::bail!(
                "dequantize_into: src_byte_offset {} not aligned to block type_size {}",
                src_byte_offset,
                type_size
            );
        }

        // Validate source bounds
        let quantized_bytes = quantized_size(elem_count, self.dtype);
        if src_byte_offset + quantized_bytes > self.data.len {
            crate::bail!(
                "dequantize_into: src offset {} + size {} exceeds storage size {}",
                src_byte_offset,
                quantized_bytes,
                self.data.len
            );
        }

        let qtype = dtype_to_qtype(self.dtype)?;
        let stream = self.device.cuda_stream();
        let (src_ptr, _src_guard) = self.data.inner.device_ptr(&stream);

        match &mut dst.slice {
            CudaStorageSlice::F16(data) => {
                let (dst_ptr, _dst_guard) = data.device_ptr_mut(&stream);
                unsafe {
                    run_dequantize_block(
                        (src_ptr as *const u8).add(src_byte_offset) as *const std::ffi::c_void,
                        (dst_ptr as *mut f16).add(dst_elem_offset) as *mut std::ffi::c_void,
                        elem_count as i32,
                        qtype as i32,
                        DequantOutDType::F16 as i32,
                    );
                }
            }
            CudaStorageSlice::BF16(data) => {
                let (dst_ptr, _dst_guard) = data.device_ptr_mut(&stream);
                unsafe {
                    run_dequantize_block(
                        (src_ptr as *const u8).add(src_byte_offset) as *const std::ffi::c_void,
                        (dst_ptr as *mut half::bf16).add(dst_elem_offset) as *mut std::ffi::c_void,
                        elem_count as i32,
                        qtype as i32,
                        DequantOutDType::BF16 as i32,
                    );
                }
            }
            CudaStorageSlice::F32(data) => {
                let (dst_ptr, _dst_guard) = data.device_ptr_mut(&stream);
                unsafe {
                    run_dequantize_block(
                        (src_ptr as *const u8).add(src_byte_offset) as *const std::ffi::c_void,
                        (dst_ptr as *mut f32).add(dst_elem_offset) as *mut std::ffi::c_void,
                        elem_count as i32,
                        qtype as i32,
                        DequantOutDType::F32 as i32,
                    );
                }
            }
            _ => crate::bail!("dequantize_into: destination must be f16, bf16, or f32"),
        }

        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len
    }

    /// Get the device pointer for the raw quantized data.
    /// This is used by paged attention kernels that need direct GPU access.
    ///
    /// # Safety
    /// The returned pointer is only valid while the QCudaStorage is alive.
    /// The caller must ensure the stream is synchronized before using the pointer.
    pub fn data_ptr(&self) -> u64 {
        let stream = self.device.cuda_stream();
        let (ptr, _guard) = self.data.inner.device_ptr(&stream);
        ptr
    }

    /// Get a mutable device pointer for the raw quantized data.
    /// This is used by batched quantization kernels that write directly to GPU memory.
    ///
    /// # Safety
    /// The returned pointer is only valid while the QCudaStorage is alive.
    /// The caller must ensure proper synchronization and exclusive access.
    pub fn data_ptr_mut(&mut self) -> u64 {
        let stream = self.device.cuda_stream();
        let (ptr, _guard) = self.data.inner.device_ptr_mut(&stream);
        ptr
    }

    /// Copy the raw quantized data from GPU to CPU as a Vec<u8>
    pub fn data(&self) -> Result<Vec<u8>> {
        let stream = self.device.cuda_stream();
        // Only copy the actual data, not the padding
        let slice = self.data.inner.slice(0..self.data.len);
        let cpu_data = stream.memcpy_dtov(&slice).map_err(crate::Error::wrap)?;
        Ok(cpu_data)
    }

    /// Copy a byte range of the quantized data from GPU to CPU as a `Vec<u8>`.
    ///
    /// Issues a single `cuMemcpyDtoH` for exactly `range.len()` bytes
    /// starting at `range.start` — no kernel, no full-buffer copy.  Use
    /// this whenever the caller only needs a slice; calling
    /// [`Self::data`] and slicing the result on the CPU side would
    /// transfer the entire arena over PCIe just to throw most of it away.
    pub fn data_range(&self, range: std::ops::Range<usize>) -> Result<Vec<u8>> {
        if range.end > self.data.len {
            crate::bail!(
                "data_range: range {:?} exceeds storage byte_len {}",
                range,
                self.data.len,
            );
        }
        let stream = self.device.cuda_stream();
        let slice = self.data.inner.slice(range);
        let cpu_data = stream.memcpy_dtov(&slice).map_err(crate::Error::wrap)?;
        Ok(cpu_data)
    }

    /// Overwrite the contents of this VRAM buffer from host bytes.
    ///
    /// Used by the expert LRU cache to swap expert weights into pre-allocated
    /// VRAM slots without reallocating. The source slice must be exactly
    /// `self.storage_size_in_bytes()` bytes.
    pub fn copy_from_host(&mut self, src: &[u8]) -> Result<()> {
        if src.len() != self.data.len {
            crate::bail!(
                "copy_from_host: expected {} bytes, got {}",
                self.data.len,
                src.len()
            );
        }
        self.device
            .memcpy_htod(src, &mut self.data.inner.slice_mut(..src.len()))?;
        Ok(())
    }

    /// Like [`copy_from_host`](Self::copy_from_host), but issues the H2D
    /// copy on a secondary [`CudaStream`] for DMA overlap.
    ///
    /// The caller must synchronise (event + wait) before using the buffer
    /// on the device's main stream.
    pub fn copy_from_host_on_stream(
        &mut self,
        src: &[u8],
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        if src.len() != self.data.len {
            crate::bail!(
                "copy_from_host_on_stream: expected {} bytes, got {}",
                self.data.len,
                src.len()
            );
        }
        stream
            .memcpy_htod(src, &mut self.data.inner.slice_mut(..src.len()))
            .map_err(crate::Error::wrap)?;
        Ok(())
    }

    /// Copy raw quantized VRAM bytes to a pre-existing host buffer on a stream.
    ///
    /// When `dst` is backed by pinned memory (`cuMemAllocHost`), the copy is
    /// truly asynchronous â€” the CPU returns immediately and the DMA engine
    /// handles the transfer.  This is the D2H path used for VRAM â†’ pinned
    /// eviction in the two-tier expert cache.
    ///
    /// `dst` must be at least `self.storage_size_in_bytes()` bytes.
    pub fn copy_to_host_on_stream(
        &self,
        dst: &mut [u8],
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        let len = self.data.len;
        if dst.len() < len {
            crate::bail!(
                "copy_to_host_on_stream: dst too small ({} < {})",
                dst.len(),
                len,
            );
        }
        let src_slice = self.data.inner.slice(0..len);
        stream
            .memcpy_dtoh(&src_slice, &mut dst[..len])
            .map_err(crate::Error::wrap)?;
        Ok(())
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        // A leased view is exactly `payload` bytes of somebody else's arena
        // slot; it carries no `MATRIX_ROW_PADDING`. Both matmul kernels below
        // gate on `data.len / type_size * block_size >= ncols * nrows` and then
        // address `pad(ncols, MATRIX_ROW_PADDING)` columns, so a lease passes
        // the guard and reads up to 512 elements past the slot — into the next
        // chunk's bytes, or past the region at the last slot. The restriction is
        // documented on `from_leased_device_ptr` and `QTensor::from_leased_cuda_ptr`
        // but nothing enforced it, and `backing` is right here to check.
        if matches!(self.backing, Backing::Lease(_)) {
            crate::bail!(
                "QMatMul on a leased quantized view: a lease is exactly its payload and \
                 carries no {MATRIX_ROW_PADDING}-element row padding, which both matmul \
                 kernels address unconditionally. Copy it into an owned QTensor first."
            )
        }
        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            8
        };
        // Q8K doesn't have MMQ support (QI8_K > WARP_SIZE), always use vec kernel
        let force_vec = self.dtype == GgmlDType::Q8_K;
        let use_vec_kernel = force_vec
            || match layout.shape().dims() {
                [b, m, _k] => b * m <= max_bm,
                [b, _k] => *b <= max_bm,
                _ => false,
            };

        let output_type = storage.dtype();

        let (mut storage, shape) = if use_vec_kernel {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }?;

        if storage.dtype() != output_type {
            storage = storage.to_dtype(layout, output_type)?;
        }
        Ok((storage, shape))
    }

    pub fn fwd_via_gemx(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        self.matmul_gemx(self_shape, storage, layout)
    }
}

impl QCudaStorage {
    /// Use the new quantized matmul kernel that takes F16/BF16 activations directly.
    /// Outputs BF16. For unsupported input dtypes, converts to BF16 before and back after.
    /// K/128 blocks have embedded scales, so no external scales needed.
    fn matmul_gemx(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let qtype = dtype_to_qtype(self.dtype)? as i32;
        let stream = self.device.cuda_stream();
        let (weight_ptr, _weight_guard) = self.data.inner.device_ptr(&stream);
        dense_qmatmul_float(
            weight_ptr,
            qtype,
            self.data.len,
            nrows,
            ncols,
            rhs,
            rhs_l,
            &self.device,
        )
    }

    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let inherit = rhs.backing;
        let rhs = rhs.as_cuda_slice::<f32>()?;
        let rhs = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt())?,
        };
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            dequantize_mul_mat_vec(&self.data, &rhs, self.dtype, ncols, nrows, self.device())?
        } else {
            mul_mat_vec_via_q8_1(
                &self.data,
                &rhs,
                self.dtype,
                ncols,
                nrows,
                b_size,
                self.device(),
                inherit,
            )?
        };
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }

    fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            let data_f32 = self.dequantize(n * k)?;
            let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
            storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?
        } else {
            let inherit = storage.backing;
            let storage = storage.as_cuda_slice::<f32>()?;
            let storage = match layout.contiguous_offsets() {
                Some((o1, o2)) => storage.slice(o1..o2),
                None => Err(crate::Error::RequiresContiguous {
                    op: "quantized-matmul",
                }
                .bt())?,
            };
            mul_mat_via_q8_1(
                &self.data,
                &storage,
                self.dtype,
                /* x_rows */ n,
                /* x_cols */ k,
                /* y_rows */ k,
                /* y_cols */ b * m,
                self.device(),
                inherit,
            )?
        };
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }

    /// Repack quantized weights to GEMX format (K/128 with embedded scales).
    ///
    /// This removes scale data from the weights (scales should be extracted
    /// separately via extract_scales before calling this) and reorders the
    /// quant bytes for optimal tensor core access patterns.
    ///
    /// # Returns
    /// A new QCudaStorage with the repacked data (smaller since scales removed)
    pub fn repack_gemx(&self, shape: &Shape) -> Result<Self> {
        let (nrows, ncols) = shape.dims2()?;
        let qtype = dtype_to_qtype(self.dtype)? as i32;

        // Check if this qtype supports GEMX repacking
        let supported = unsafe { is_gemx_supported(qtype) };
        if supported == 0 {
            crate::bail!("GEMX repacking not supported for dtype {:?}", self.dtype);
        }

        // Get expected output size
        let new_size = unsafe { get_repacked_size_bytes(nrows as i32, ncols as i32, qtype) };
        if new_size < 0 {
            crate::bail!("Failed to get repacked size for dtype {:?}", self.dtype);
        }
        let new_size = new_size as usize;

        // Allocate new buffer for repacked data
        let dst_data = self.device.alloc_zeros::<u8>(new_size)?;

        // Launch repacking kernel (src -> dst)
        // Scope the guards so they're dropped before we move dst_data
        {
            let stream = self.device.cuda_stream();
            let (src_ptr, _src_guard) = self.data.inner.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst_data.device_ptr(&stream);
            let result = unsafe {
                run_repack_gemx(
                    src_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    nrows as i32,
                    ncols as i32,
                    qtype,
                )
            };

            if result < 0 {
                crate::bail!("repack_gemx kernel failed");
            }
        }

        Ok(Self {
            data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
                inner: dst_data,
                len: new_size,
            }),
            dtype: self.dtype,
            device: self.device.clone(),
            backing: Backing::Owned,
        })
    }

    /// Re-quantize this (compact) weight `[nrows × ncols]` into the lane-major KO format
    /// `ko_dtype` for the q8a128 int8 matmul: dequantize to f32, then per-128 affine KO
    /// quantize. Both stages run on the GPU and the dequantized f32 is fed straight into
    /// `run_quantize_ko` without ever leaving VRAM. The int8-mode counterpart of
    /// [`Self::repack_gemx`] used by `QMatMul::repack_for_optimization`.
    pub fn repack_ko(&self, shape: &Shape, ko_dtype: GgmlDType) -> Result<Self> {
        let (nrows, ncols) = shape.dims2()?;
        // The KO chunk layout packs 8 rows × 128 K per chunk, but the q8a128 matmul kernel that
        // reads the result tiles N in blocks of 32 — so require `nrows % 32` (not just 8), matching
        // the matmul. Callers (`repack_for_optimization` → `qlinear_int8`) treat the bail as "not
        // KO-tileable" and fall back to a dense weight (e.g. the tiny mHC `fn_w`, `mix_hc=24`).
        if nrows % 32 != 0 || ncols % 128 != 0 {
            crate::bail!(
                "repack_ko: shape [{nrows}, {ncols}] must have nrows % 32 == 0 and ncols % 128 == 0"
            );
        }
        // MXFP4 → MXFP4_KO is an EXACT byte permutation (nibbles + per-sub E8M0 copied
        // verbatim, per-row dm baked) — never a dequant/requant, which would re-derive
        // E8M0 scales lossily and needs a quantize kernel that deliberately does not
        // exist (`run_quantize_ko` has no MXFP4 arm). Load-time only: pull the native
        // bytes to the host, reorder with the same routine the engine's prepare path
        // uses, upload the chunk tensor.
        if ko_dtype == GgmlDType::MXFP4_KO {
            if self.dtype != GgmlDType::MXFP4 {
                crate::bail!(
                    "repack_ko(MXFP4_KO): source must be MXFP4, got {:?}",
                    self.dtype
                );
            }
            let need = nrows * (ncols / 32) * GgmlDType::MXFP4.type_size();
            if self.data.len < need {
                crate::bail!(
                    "repack_ko(MXFP4_KO): storage holds {} bytes, need {need}",
                    self.data.len
                );
            }
            let native = self
                .device
                .memcpy_dtov(&self.data.inner.slice(..need))
                .map_err(crate::Error::wrap)?;
            let ko =
                crate::quantized::ko_quant::mxfp4_native_to_ko_gpu_chunk(&native, nrows, ncols);
            let mut out = unsafe { self.device.alloc::<u8>(ko.len())? };
            self.device
                .memcpy_htod(&ko, &mut out.slice_mut(..ko.len()))?;
            return Ok(Self {
                data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
                    inner: out,
                    len: ko.len(),
                }),
                dtype: ko_dtype,
                device: self.device.clone(),
                backing: Backing::Owned,
            });
        }
        let qtype = dtype_to_qtype(ko_dtype)? as i32;
        let bytes =
            (nrows / 8) * (ncols / 128) * crate::quantized::ko_quant::ko_chunk_bytes(ko_dtype);
        // dequantize stays on-device; quantize reads that f32 buffer directly (no D2H/H2D).
        let f32_storage = self.dequantize(nrows * ncols)?;
        let f32_slice = f32_storage.as_cuda_slice::<f32>()?;
        let mut out = unsafe { self.device.alloc::<u8>(bytes)? };
        {
            let stream = self.device.cuda_stream();
            let (wp, _gw) = f32_slice.device_ptr(&stream);
            let (op, _go) = out.device_ptr_mut(&stream);
            unsafe {
                run_quantize_ko(
                    wp as *const f32,
                    op as *mut std::ffi::c_void,
                    nrows as i32,
                    ncols as i32,
                    qtype,
                );
            }
        }
        Ok(Self {
            data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
                inner: out,
                len: bytes,
            }),
            dtype: ko_dtype,
            device: self.device.clone(),
            backing: Backing::Owned,
        })
    }

    /// Get the size in bytes after GEMX repacking, without actually repacking.
    pub fn repacked_size(&self, shape: &Shape) -> Result<usize> {
        let (nrows, ncols) = shape.dims2()?;
        let qtype = dtype_to_qtype(self.dtype)? as i32;

        let size = unsafe { get_repacked_size_bytes(nrows as i32, ncols as i32, qtype) };
        if size < 0 {
            crate::bail!("GEMX repacking not supported for dtype {:?}", self.dtype);
        }
        Ok(size as usize)
    }

    /// Check if this storage's dtype supports GEMX repacking.
    pub fn supports_gemx_repacking(&self) -> bool {
        if let Ok(qtype) = dtype_to_qtype(self.dtype) {
            unsafe { is_gemx_supported(qtype as i32) != 0 }
        } else {
            false
        }
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = unsafe { device.alloc::<u8>(padded_len)? };
    device.memcpy_htod(data, &mut inner.slice_mut(..data.len()))?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
            inner,
            len: data.len(),
        }),
        device: device.clone(),
        dtype,
        backing: Backing::Owned,
    }))
}

/// Like [`load_quantized`], but issues the alloc + H2D copy on a secondary
/// [`CudaStream`] so it can overlap with compute on the device's main stream.
///
/// The caller is responsible for synchronisation: record a
/// [`CudaEvent`](cudarc::driver::CudaEvent) on the copy stream *after* calling
/// this, then make the compute stream [`wait`](cudarc::driver::CudaStream::wait)
/// on that event before first use.
///
/// # Safety
///
/// The returned `QStorage` must not be accessed on the device's main stream
/// until the copy-stream work has been synchronised via an event.
pub fn load_quantized_on_stream<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = unsafe { stream.alloc::<u8>(padded_len).map_err(crate::Error::wrap)? };
    stream
        .memcpy_htod(data, &mut inner.slice_mut(..data.len()))
        .map_err(crate::Error::wrap)?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
            inner,
            len: data.len(),
        }),
        device: device.clone(),
        dtype,
        backing: Backing::Owned,
    }))
}

// =============================================================================
// Repacked-format helpers (swap file support)
// =============================================================================

/// Load pre-repacked K/128 GEMX data from host bytes into a new CUDA buffer.
///
/// Unlike [`load_quantized_on_stream`], this does **not** add
/// [`MATRIX_ROW_PADDING`] â€” the repacked data is already correctly sized
/// by `get_repacked_size_bytes()`.
///
/// Used when loading experts from a swap file where weights are already in
/// K/128 format.  No GPU repack kernel is needed.
pub fn load_repacked_on_stream(
    device: &CudaDevice,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    repacked_data: &[u8],
    dtype: GgmlDType,
) -> Result<super::QStorage> {
    let len = repacked_data.len();
    let mut inner = unsafe { stream.alloc::<u8>(len).map_err(crate::Error::wrap)? };
    stream
        .memcpy_htod(repacked_data, &mut inner.slice_mut(..len))
        .map_err(crate::Error::wrap)?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: std::mem::ManuallyDrop::new(PaddedCudaSlice { inner, len }),
        device: device.clone(),
        dtype,
        backing: Backing::Owned,
    }))
}

/// Upload `repacked_data` to `dst_ptr` — device memory owned by somebody else —
/// and wrap it as a storage that will not free those bytes.
///
/// The weight-zone counterpart of [`load_repacked_on_stream`]. An expert slot is
/// a fixed-size range of the device reservation, handed out by
/// `candle_nn::kv_cache::WeightZone`, so the bytes outlive any one `QTensor`
/// built over them: a miss overwrites the slot in place and an eviction returns
/// it to the zone's free list. Dropping the storage must therefore release the
/// *view* and nothing else, which is exactly what [`Backing::Lease`] means here.
///
/// No `MATRIX_ROW_PADDING` tail, matching what the pool path allocated: the
/// expert kernels read the repacked layout exactly and never over-read it.
///
/// # Safety
///
/// `dst_ptr` must point to at least `repacked_data.len()` bytes of device memory
/// that stays live, and un-aliased for writes, for the storage's lifetime.
pub unsafe fn load_repacked_into(
    device: &CudaDevice,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    dst_ptr: u64,
    repacked_data: &[u8],
    dtype: GgmlDType,
) -> Result<super::QStorage> {
    let len = repacked_data.len();
    // Upgraded once and moved into the storage: a second upgrade would need its
    // own `leak` on every path out of here, and forgetting one strands a pair of
    // `CudaEvent`s per expert load.
    let mut inner = stream.upgrade_device_ptr::<u8>(dst_ptr, len);
    stream
        .memcpy_htod(repacked_data, &mut inner.slice_mut(..len))
        .map_err(crate::Error::wrap)?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: std::mem::ManuallyDrop::new(PaddedCudaSlice { inner, len }),
        device: device.clone(),
        dtype,
        backing: Backing::Lease(LeaseOrigin::Foreign),
    }))
}

/// Wrap `bytes` of already-populated device memory at `ptr` as a storage that
/// will not free it — [`load_repacked_into`] without the upload.
///
/// For a slot that has been **relocated**: the bytes were copied device-to-device
/// and are already correct, but the three `QMatMul`s hold device pointers, so
/// their storages have to be rebuilt over the new address.
///
/// # Safety
///
/// `ptr` must point to at least `bytes` of live device memory holding a valid
/// `dtype` payload, un-aliased for writes for the storage's lifetime.
pub unsafe fn view_repacked(
    device: &CudaDevice,
    ptr: u64,
    bytes: usize,
    dtype: GgmlDType,
) -> Result<super::QStorage> {
    let inner = device.cuda_stream().upgrade_device_ptr::<u8>(ptr, bytes);
    Ok(QStorage::Cuda(QCudaStorage {
        data: std::mem::ManuallyDrop::new(PaddedCudaSlice { inner, len: bytes }),
        device: device.clone(),
        dtype,
        backing: Backing::Lease(LeaseOrigin::Foreign),
    }))
}

/// Like [`load_repacked_on_stream`], but uses the device's default stream.
pub fn load_repacked(
    device: &CudaDevice,
    repacked_data: &[u8],
    dtype: GgmlDType,
) -> Result<super::QStorage> {
    let len = repacked_data.len();
    let mut inner = unsafe { device.alloc::<u8>(len)? };
    device.memcpy_htod(repacked_data, &mut inner.slice_mut(..len))?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: std::mem::ManuallyDrop::new(PaddedCudaSlice { inner, len }),
        device: device.clone(),
        dtype,
        backing: Backing::Owned,
    }))
}

/// One-shot repack: CPU (GGML bytes) â†’ GPU â†’ repack â†’ GPU â†’ CPU (K/128 bytes).
///
/// Used during swap file creation.  Not performance-critical (one-time cost).
/// Allocates scratch VRAM internally and frees on return.
///
/// Returns the repacked weight bytes in K/128 GEMX format.
pub fn repack_to_host(
    device: &CudaDevice,
    ggml_bytes: &[u8],
    nrows: usize,
    ncols: usize,
    dtype: GgmlDType,
    target_dtype: GgmlDType,
) -> Result<Vec<u8>> {
    // KO target: dequantize the compact source (`dtype`) and re-quantize to the KO twin
    // (`target_dtype`) ONCE. The host bytes are cached in the expert pinned pool exactly like the
    // gemx repack, so a miss DMA-reloads them with no per-miss re-quant. The int8 KO matmul reads
    // these; the staging pipeline is format-agnostic (gemx K/128 or KO — both just tensors).
    // Already the target format (a prepared / pre-repacked weight, e.g. MXFP4_KO on disk):
    // nothing to do — hand the bytes straight through so the staging path stays uniform and
    // pays no reorder. This is what lets the engine load a pre-KO GGUF with no runtime repack.
    //
    // **This must stay unconditional.** The dtype pair says what the bytes ARE, not whether they
    // still need work: `(X, X)` means "the source is already in the target format". Narrowing
    // this to `is_ko()` targets made the loader repack ALREADY-REPACKED weights a second time —
    // per expert, that is an `alloc` + `alloc_zeros` + H2D + a kernel with an internal
    // `cudaDeviceSynchronize` + D2H into a fresh host `Vec`, ~11,008 times. Measured: 157 GB of
    // private commit on a 189 GB box, the GPU at 6%, and the model gate never finishing.
    //
    // A caller holding RAW GGML that wants the K/128 layout must say so by calling
    // [`repack_gemx_to_host`] directly — the source's provenance (raw vs already-prepared) is
    // the caller's knowledge and cannot be recovered from `(dtype, target_dtype)`.
    if dtype == target_dtype {
        return Ok(ggml_bytes.to_vec());
    }
    // MXFP4_KO: the native MXFP4 experts repack by an EXACT byte-reorder (no dequant/requant),
    // keeping the weights 4-bit. Done on the host straight from the GGUF bytes — the per-sub
    // fold lives in the int8 kernel, so this only permutes nibbles + E8M0 and bakes the per-row
    // e_max scale. (Bypasses the affine `repack_ko` F32-requant route.)
    if target_dtype == GgmlDType::MXFP4_KO {
        if dtype != GgmlDType::MXFP4 {
            crate::bail!("repack_to_host(MXFP4_KO): source must be MXFP4, got {dtype:?}");
        }
        return Ok(crate::quantized::ko_quant::mxfp4_native_to_ko_gpu_chunk(
            ggml_bytes, nrows, ncols,
        ));
    }
    if target_dtype.is_ko() {
        let src = load_repacked(device, ggml_bytes, dtype)?;
        let src_cuda = match &src {
            QStorage::Cuda(s) => s,
            _ => crate::bail!("repack_to_host(KO): expected CUDA storage"),
        };
        let shape: Shape = vec![nrows, ncols].into();
        let ko = src_cuda.repack_ko(&shape, target_dtype)?;
        let host = device
            .memcpy_dtov(&ko.data.inner.slice(..ko.data.len))
            .map_err(crate::Error::wrap)?;
        return Ok(host);
    }

    repack_gemx_to_host(device, ggml_bytes, nrows, ncols, dtype)
}

/// Repack **raw GGML bytes** into the K/128 GEMX layout, keeping the dtype.
///
/// Split out of [`repack_to_host`] because the two cannot be distinguished by their arguments:
/// a GEMX repack keeps the dtype and changes only the byte layout, so it and "the source is
/// already in the target format" both present as `(X, X)`. Which one is meant depends on the
/// **source's provenance** — raw out of a GGUF, or already prepared on disk — which only the
/// caller knows. [`repack_to_host`] therefore treats `(X, X)` as already-prepared (the loader's
/// case, where repacking twice is both wrong and ruinously expensive), and a caller holding raw
/// bytes calls this instead.
///
/// Not performance-critical: one-time, one weight per call, and the underlying kernel carries an
/// internal device sync. Do not use it to convert a whole expert set at load time.
pub fn repack_gemx_to_host(
    device: &CudaDevice,
    ggml_bytes: &[u8],
    nrows: usize,
    ncols: usize,
    dtype: GgmlDType,
) -> Result<Vec<u8>> {
    let qtype = dtype_to_qtype(dtype)? as i32;

    let supported = unsafe { is_gemx_supported(qtype) };
    if supported == 0 {
        crate::bail!("GEMX repacking not supported for dtype {:?}", dtype);
    }

    let repacked_size = unsafe { get_repacked_size_bytes(nrows as i32, ncols as i32, qtype) };
    if repacked_size < 0 {
        crate::bail!(
            "Failed to get repacked size for {:?} ({nrows}Ã—{ncols})",
            dtype
        );
    }
    let repacked_size = repacked_size as usize;

    // Allocate scratch buffers on the default stream.
    let mut src_buf = unsafe { device.alloc::<u8>(ggml_bytes.len())? };
    let dst_buf = device.alloc_zeros::<u8>(repacked_size)?;

    // H2D: GGML bytes â†’ src VRAM.
    device.memcpy_htod(ggml_bytes, &mut src_buf.slice_mut(..ggml_bytes.len()))?;

    // GPU repack kernel (includes cudaDeviceSynchronize).
    {
        let stream = device.cuda_stream();
        let (src_ptr, _sg) = src_buf.device_ptr(&stream);
        let (dst_ptr, _dg) = dst_buf.device_ptr(&stream);
        let result = unsafe {
            run_repack_gemx(
                src_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                nrows as i32,
                ncols as i32,
                qtype,
            )
        };
        if result < 0 {
            crate::bail!("repack_gemx failed for {:?} ({nrows}Ã—{ncols})", dtype);
        }
    }

    // D2H: repacked VRAM â†’ host Vec.
    let repacked = device
        .memcpy_dtov(&dst_buf.slice(..repacked_size))
        .map_err(crate::Error::wrap)?;

    Ok(repacked)
}

/// Get the repacked byte size for a weight matrix of the given dimensions
/// and quantization type, without actually repacking.
pub fn repacked_size_bytes(nrows: usize, ncols: usize, dtype: GgmlDType) -> Result<usize> {
    // KO target: per-128 chunk bytes (the format the expert pinned pool caches for int8 experts).
    if dtype.is_ko() {
        if !nrows.is_multiple_of(8) || !ncols.is_multiple_of(128) {
            crate::bail!("repacked_size_bytes(KO): nrows={nrows} %8, ncols={ncols} %128 required");
        }
        return Ok((nrows / 8) * (ncols / 128) * crate::quantized::ko_quant::ko_chunk_bytes(dtype));
    }
    let qtype = dtype_to_qtype(dtype)? as i32;
    let size = unsafe { get_repacked_size_bytes(nrows as i32, ncols as i32, qtype) };
    if size < 0 {
        crate::bail!("GEMX repacking not supported for dtype {:?}", dtype);
    }
    Ok(size as usize)
}

// =============================================================================
// GROUPED EXPERT MATMUL (segmented dispatch)
// =============================================================================

/// Grouped expert matmul: processes multiple experts via segmented dispatch.
///
/// Builds a `VxSegment` array (one per expert) and makes a single call to
/// `run_quantized_matmul`. The C dispatcher loops over segments internally,
/// giving each expert full greedy batch decomposition (TC, iter, bulk, remainder).
/// No device table allocation or memcpy â€” segment descriptors are host-side.
///
/// # Arguments
/// * `weight_ptrs` - GPU device pointers to each expert's weight data (K/128 format)
/// * `weight_dtype` - Quantization type (same for all experts in this call)
/// * `nrows` - N (output features per expert, same for all)
/// * `ncols` - K (input features per expert, same for all)
/// * `activations` - Stacked activations `[total_batch, K]` on GPU
/// * `act_layout` - Layout of activations tensor
/// * `expert_offsets` - `[num_experts + 1]` prefix sum of per-expert batch counts
///   (CPU slice: `[0, n0, n0+n1, ..., total_batch]`)
/// * `device` - CUDA device
///
/// # Returns
/// A `Tensor` with shape `[total_batch, N]` on the same CUDA device.
fn grouped_matmul_gemx_impl<'w>(
    weight_ptrs: &[u64],
    weight_dtype: GgmlDType,
    nrows: usize,
    ncols: usize,
    activations: &CudaStorage,
    act_layout: &crate::Layout,
    expert_offsets: &[i32],
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    use crate::cuda_backend::CudaStorageSlice;

    let num_experts = weight_ptrs.len();
    if num_experts == 0 {
        crate::bail!("grouped_matmul_gemx: no experts provided");
    }
    // The grouped kernel writes full 32-row (N_TILE) blocks with no partial-row
    // guard, so nrows must be a multiple of 32 (all MoE expert dims are).
    if !nrows.is_multiple_of(32) {
        crate::bail!("grouped_matmul_gemx: nrows={nrows} must be a multiple of 32");
    }
    if expert_offsets.len() != num_experts + 1 {
        crate::bail!(
            "grouped_matmul_gemx: expert_offsets.len() = {} but expected {} (num_experts + 1)",
            expert_offsets.len(),
            num_experts + 1
        );
    }
    let total_batch = *expert_offsets.last().unwrap() as usize;
    if total_batch == 0 {
        // No tokens to process â€” return zero output
        let out_shape: Shape = vec![0, nrows].into();
        let out_slice = unsafe { device.alloc::<f16>(0)? };
        let out_storage = CudaStorage::wrap_cuda_slice(out_slice, device.clone());
        return Ok(crate::tensor::from_storage(
            crate::Storage::Cuda(out_storage),
            out_shape,
            crate::op::BackpropOp::none(),
            false,
        ));
    }

    let k = ncols;
    let (batch_size_from_act, k_act) = match act_layout.shape().dims() {
        [b, m, k] => (b * m, *k),
        [b, k] => (*b, *k),
        _ => crate::bail!(
            "grouped_matmul_gemx: unexpected activation shape {:?}",
            act_layout.shape()
        ),
    };
    if k_act != k {
        crate::bail!("grouped_matmul_gemx: K mismatch: weight ncols={k}, activation K={k_act}");
    }
    if batch_size_from_act != total_batch {
        crate::bail!(
            "grouped_matmul_gemx: batch mismatch: activation batch={batch_size_from_act}, offsets total={total_batch}"
        );
    }

    // Convert dtype
    let qtype = dtype_to_qtype(weight_dtype)? as i32;

    // Determine Y type and element size
    let ytype = match &activations.slice {
        CudaStorageSlice::F16(_) => YType::F16,
        CudaStorageSlice::BF16(_) => YType::BF16,
        CudaStorageSlice::F32(_) => YType::F32,
        _ => crate::bail!(
            "grouped_matmul_gemx: unsupported activation dtype {:?}",
            activations.dtype()
        ),
    };

    // The output inherits the activations' arena and matches their dtype (the FP
    // grouped kernels store at the activation width). Only the pool arm owns a
    // slice; the wave arm stays a bare pointer until the wrap, so a `?` between
    // here and there cannot drop a `CudaSlice` over arena memory.
    let out_elems = nrows * total_batch;
    let (dst_ptr, owned_out, out_backing) =
        resolve_out(activations.dtype(), activations.backing, device, out_elems)?;

    // Grouped dispatch: split each expert's [offset..offset+cnt) batch range into
    // <=16-token MMA tiles, then run ALL experts in ONE kernel launch. The kernel
    // grid (num_tiles x row_tiles) spans every expert at once, so both the launch
    // count (N -> 1 per projection) and per-expert occupancy are fixed together.
    let mut tile_expert: Vec<i32> = Vec::new();
    let mut tile_b_start: Vec<i32> = Vec::new();
    let mut tile_b_cnt: Vec<i32> = Vec::new();
    for e in 0..num_experts {
        let mut s = expert_offsets[e];
        let end = expert_offsets[e + 1];
        while s < end {
            let cnt = (end - s).min(16);
            tile_expert.push(e as i32);
            tile_b_start.push(s);
            tile_b_cnt.push(cnt);
            s += cnt;
        }
    }
    let num_tiles = tile_expert.len();
    // Token tiles ride grid.y (row tiles are the fast axis for L2 reuse).
    if num_tiles > 65535 {
        crate::bail!("grouped_matmul_gemx: num_tiles={num_tiles} exceeds grid.y max 65535");
    }

    {
        let stream = device.cuda_stream();

        // Pack the weight-pointer array + all 3 i32 tile tables into ONE blob.
        // weight_ptrs (u64) go first (8-aligned at the 16-aligned ring slot);
        // the three i32 tables follow at 4-aligned offsets. The kernel reads
        // each table via base + offset, IN PLACE from the device-mapped pinned
        // table ring — no per-launch H2D copy, no device allocation.
        let off_te = num_experts * 8;
        let off_tbs = off_te + num_tiles * 4;
        let off_tbc = off_tbs + num_tiles * 4;
        let total_bytes = off_tbc + num_tiles * 4;
        let mut packed: Vec<u8> = Vec::with_capacity(total_bytes);
        for &w in weight_ptrs {
            packed.extend_from_slice(&w.to_le_bytes());
        }
        for &x in &tile_expert {
            packed.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &tile_b_start {
            packed.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &tile_b_cnt {
            packed.extend_from_slice(&x.to_le_bytes());
        }
        let ring = table_ring(device)?;
        let y_elem = match ytype {
            YType::F32 => 4usize,
            _ => 2, // F16 / BF16
        };
        let row_fast = grouped_grid_row_fast(total_batch * k * y_elem, device) as i32;

        macro_rules! dispatch_grouped {
            ($y_data:expr) => {{
                let y_view = match act_layout.contiguous_offsets() {
                    Some((o1, o2)) => $y_data.slice(o1..o2),
                    None => {
                        return Err(crate::Error::RequiresContiguous {
                            op: "grouped_matmul_gemx",
                        }
                        .bt())?
                    }
                };
                let (y_ptr, _y_guard) = y_view.device_ptr(&stream);
                ring.with_table(&packed, |base| {
                    let wptr_ptr = base;
                    let te_ptr = base + off_te as u64;
                    let tbs_ptr = base + off_tbs as u64;
                    let tbc_ptr = base + off_tbc as u64;
                    unsafe {
                        run_grouped_quantized_matmul(
                            wptr_ptr as *const std::ffi::c_void,
                            te_ptr as *const std::ffi::c_void,
                            tbs_ptr as *const std::ffi::c_void,
                            tbc_ptr as *const std::ffi::c_void,
                            y_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            k as i32,     // ncols_x = K
                            nrows as i32, // nrows_x = N
                            k as i32,     // y_stride = K (stacked activations)
                            nrows as i32, // dst_stride = N (stacked output)
                            num_tiles as i32,
                            qtype,
                            ytype as i32,
                            2, // FP grouped kernels ignore the int8 tile mode
                            row_fast,
                        );
                    }
                })?;
            }};
        }

        match &activations.slice {
            CudaStorageSlice::F16(y_data) => {
                dispatch_grouped!(y_data);
            }
            CudaStorageSlice::BF16(y_data) => {
                dispatch_grouped!(y_data);
            }
            CudaStorageSlice::F32(y_data) => {
                dispatch_grouped!(y_data);
            }
            _ => unreachable!("ytype and activation slice should match"),
        }
    }

    let out_shape: Shape = vec![total_batch, nrows].into();
    tensor_from_owned_out(owned_out, dst_ptr, out_backing, out_shape, device)
}

/// Single-launch grouped MoE expert matmul over FP activations (F16/BF16/F32):
/// FP16 tensor-core MMA with Q4_K weights dequantized on the fly. The INT8-MMA path
/// is `grouped_matmul_gemx_q8a128` (q8a128 activations). See docs/q8_matmul_pipeline.md.
pub fn grouped_matmul_gemx(
    weight_ptrs: &[u64],
    weight_dtype: GgmlDType,
    nrows: usize,
    ncols: usize,
    activations: &CudaStorage,
    act_layout: &crate::Layout,
    expert_offsets: &[i32],
    device: &CudaDevice,
) -> Result<crate::Tensor> {
    grouped_matmul_gemx_impl(
        weight_ptrs,
        weight_dtype,
        nrows,
        ncols,
        activations,
        act_layout,
        expert_offsets,
        device,
    )
}

/// The minimum token count M at which the q8a1024 int8 matmul flips from mode-1 to
/// mode-2. Measured crossover: below this, the per-token weight re-read of mode-2's
/// Bm=32 tile costs more than it saves; at/above it, weight reuse wins (the prefill
/// regime). M is the absolute number of token rows fed to the quantizer.
/// q8a128-quantized int8 operand for the tensor-core qmatmul: the packed blocks (stored in
/// the q8a1024 flat-grouped super-block layout — see blocks.cuh), the logical shape
/// `[rows × cols]` = `[M × K]`, and `lead` — the original activation's leading dims
/// (`prod(lead) == rows`) so the matmul can rebuild the output rank (`[lead.., N]`) to match the
/// float path on 3D+ activations. The mode-1/mode-2 flip is NOT stored here: the q8a1024 layout
/// is mode-independent, so the dispatcher chooses the kernel tiling from the token count. Named
/// "operand" (not "acts") because either matmul side could be supplied this way.
/// Backing bytes for a [`Q8a128Operand`]: either an owned `CudaSlice` (the quantizer / fused
/// RMSNorm / SwiGLU producers `alloc` their own buffer) or a `U8` [`Tensor`] of q8a1024 bytes.
/// The latter lets a producer that returns through `apply_op1` (e.g. the B2 decode op, whose
/// output is a tensor) feed the int8 matmul WITHOUT a device copy — the operand just borrows the
/// tensor's bytes. Both expose a device pointer via [`Q8a128Operand::with_device_ptr`].
pub enum Q8a128Data<'w> {
    Owned(CudaSlice<u8>),
    Tensor(crate::LiveTensor<'w>),
}

/// Where a kernel's output goes, and how the storage wrapping it must be marked.
///
/// Returned as a triple — `(ptr, owned, backing)` — rather than as separate
/// decisions, because the pointer and the backing have to agree. A wave range
/// marked `Owned` is a double free; a pool buffer marked `Lease` is a permanent
/// leak. Resolving both in one place is what stops them drifting.
///
/// # Why the wave arm is a bare pointer
///
/// The pool arm owns a `CudaSlice`; the wave arm deliberately does not
/// materialise one. A `CudaSlice` frees on drop, and between resolving an output
/// and wrapping it the kernel launch can fail — every `?` in that window would
/// otherwise put a `cuMemFreeAsync` on arena memory. Staying a raw `u64` until
/// the caller wraps it removes the window entirely.
type ResolvedOut<T> = (u64, Option<CudaSlice<T>>, Backing);

/// Resolve a `u8` output from `origin`'s arena, or from the pool.
///
/// `origin` is the operand's backing: a wave-backed operand yields a wave-backed
/// output (the inheritance rule), anything else allocates normally. A phase's
/// *first* buffer has no arena to inherit — its operand is the residual stream,
/// which must stay pooled because it crosses layers — so those call sites pass a
/// synthesised `Backing::Lease(Wave(ticket))` to seed the chain instead.
fn resolve_u8_out(origin: Backing, dev: &CudaDevice, bytes: usize) -> Result<ResolvedOut<u8>> {
    if let Some(ticket) = origin.inherit_ticket() {
        if let Some(ptr) = wave_alloc(ticket, bytes, INHERIT_ALIGN) {
            return Ok((ptr, None, Backing::Lease(LeaseOrigin::Wave(ticket))));
        }
    }
    // SAFETY: the kernel the caller launches next fills this before anything
    // reads it, exactly as when it allocated inline.
    let slice = unsafe { dev.alloc::<u8>(bytes)? };
    let ptr = {
        let stream = dev.cuda_stream();
        let (ptr, _guard) = slice.device_ptr(&stream);
        ptr
    };
    Ok((ptr, Some(slice), Backing::Owned))
}

/// [`resolve_u8_out`] for a typed output.
///
/// Separate because the byte count is `elems * size_of::<T>()`, and getting that
/// wrong is the one mistake this whole mechanism exists to make impossible.
fn resolve_typed_out<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    origin: Backing,
    dev: &CudaDevice,
    elems: usize,
) -> Result<ResolvedOut<T>> {
    if let Some(ticket) = origin.inherit_ticket() {
        let bytes = elems * std::mem::size_of::<T>();
        if let Some(ptr) = wave_alloc(ticket, bytes, INHERIT_ALIGN) {
            return Ok((ptr, None, Backing::Lease(LeaseOrigin::Wave(ticket))));
        }
    }
    // SAFETY: filled by the kernel the caller launches next.
    let slice = unsafe { dev.alloc::<T>(elems)? };
    let ptr = {
        let stream = dev.cuda_stream();
        let (ptr, _guard) = slice.device_ptr(&stream);
        ptr
    };
    Ok((ptr, Some(slice), Backing::Owned))
}

/// A resolved matmul destination whose element type was picked at runtime.
///
/// The pool arm owns a typed `CudaSlice`, so the type has to be carried from the
/// resolve to the wrap; the wave arm holds `None` and rides on the raw pointer.
/// One enum for all three matmul entries — they make the same decision and would
/// otherwise each grow their own copy of it.
enum OwnedOut {
    F16(Option<CudaSlice<f16>>),
    BF16(Option<CudaSlice<bf16>>),
    F32(Option<CudaSlice<f32>>),
}

/// [`resolve_typed_out`] with the element type chosen from a runtime [`DType`].
fn resolve_out(
    dtype: crate::DType,
    origin: Backing,
    dev: &CudaDevice,
    elems: usize,
) -> Result<(u64, OwnedOut, Backing)> {
    match dtype {
        crate::DType::F16 => {
            let (p, o, b) = resolve_typed_out::<f16>(origin, dev, elems)?;
            Ok((p, OwnedOut::F16(o), b))
        }
        crate::DType::BF16 => {
            let (p, o, b) = resolve_typed_out::<bf16>(origin, dev, elems)?;
            Ok((p, OwnedOut::BF16(o), b))
        }
        crate::DType::F32 => {
            let (p, o, b) = resolve_typed_out::<f32>(origin, dev, elems)?;
            Ok((p, OwnedOut::F32(o), b))
        }
        other => crate::bail!("quantized matmul: unsupported output dtype {other:?}"),
    }
}

/// [`tensor_from_out`] for a destination resolved by [`resolve_out`].
fn tensor_from_owned_out<'w>(
    owned: OwnedOut,
    ptr: u64,
    backing: Backing,
    shape: crate::Shape,
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    match owned {
        OwnedOut::F16(o) => tensor_from_out::<f16>(o, ptr, backing, shape, device),
        OwnedOut::BF16(o) => tensor_from_out::<bf16>(o, ptr, backing, shape, device),
        OwnedOut::F32(o) => tensor_from_out::<f32>(o, ptr, backing, shape, device),
    }
}

/// The dispatcher's store-width code for an output dtype.
fn out_dtype_code(dtype: crate::DType) -> Result<i32> {
    let code = match dtype {
        crate::DType::F16 => OutDType::F16,
        crate::DType::BF16 => OutDType::BF16,
        crate::DType::F32 => OutDType::F32,
        other => crate::bail!("quantized matmul: unsupported output dtype {other:?}"),
    };
    Ok(code as i32)
}

/// Turn a launcher status code into an error.
///
/// The launchers pick a kernel from a static table; a miss leaves the destination
/// holding whatever the arena last put there. Checking the code is what makes that
/// a failure instead of wrong numbers.
fn check_matmul_status(code: i32, op: &str) -> Result<()> {
    match MatmulStatus::from_code(code).failure() {
        None => Ok(()),
        Some(reason) => crate::bail!("{op}: kernel not launched ({reason}, status {code})"),
    }
}

/// Build the q8a128 operand for whichever destination the kernel just wrote.
///
/// Kept next to [`resolve_u8_out`] because the two are halves of one decision:
/// resolve picks where the bytes go, this wraps them without the caller having
/// to re-derive which arm was taken.
fn q8a128_from_out<'w>(
    owned: Option<CudaSlice<u8>>,
    ptr: u64,
    backing: Backing,
    bytes: usize,
    rows: usize,
    cols: usize,
    device: &CudaDevice,
) -> Result<Q8a128Operand<'w>> {
    match owned {
        Some(slice) => Ok(Q8a128Operand::new(slice, rows, cols)),
        None => {
            let dev = crate::Device::Cuda(device.clone());
            let origin = match backing {
                Backing::Lease(o) => o,
                Backing::Owned => LeaseOrigin::Foreign,
            };
            // SAFETY: the kernel above filled `bytes` at `ptr`, and `'w` is the
            // generation that range was carved from, so the operand cannot be
            // named after the span is reset.
            let backing_tensor = unsafe {
                crate::LiveTensor::from_leased_cuda_ptr(ptr, crate::DType::U8, bytes, &dev, origin)?
            };
            Ok(Q8a128Operand::from_tensor(backing_tensor, rows, cols))
        }
    }
}

/// Wrap whichever destination a kernel wrote into as a tensor bounded by `'w`.
///
/// The leased arm is why these kernels carry a lifetime: an inherited output
/// lives in a span the next layer resets, so returning `Tensor` (`'static`) over
/// it would be a lie the borrow checker could not catch.
fn tensor_from_out<
    'w,
    T: crate::WithDType + crate::cuda_backend::CudaDType + cudarc::driver::DeviceRepr,
>(
    owned: Option<CudaSlice<T>>,
    ptr: u64,
    backing: Backing,
    shape: crate::Shape,
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    match owned {
        Some(slice) => {
            let storage = CudaStorage::wrap_cuda_slice(slice, device.clone());
            // SAFETY: the kernel wrote `shape.elem_count()` elements of `T`.
            Ok(unsafe { crate::LiveTensor::from_cuda_storage(storage, shape) })
        }
        None => {
            let dev = crate::Device::Cuda(device.clone());
            let origin = match backing {
                Backing::Lease(o) => o,
                Backing::Owned => LeaseOrigin::Foreign,
            };
            // SAFETY: as above, and `'w` bounds the carved range.
            unsafe { crate::LiveTensor::from_leased_cuda_ptr(ptr, T::DTYPE, shape, &dev, origin) }
        }
    }
}

/// `'w` is the backing tensor's: the B2 decode context is written into an
/// inference wave's transient half, so an operand wrapping it must not outlive
/// that wave. The owned variant allocates and is free to pick any `'w`.
pub struct Q8a128Operand<'w> {
    pub data: Q8a128Data<'w>,
    pub rows: usize,      // M (flattened token count)
    pub cols: usize,      // K
    pub lead: Vec<usize>, // output leading dims; prod == rows (defaults to [rows])
}

impl<'w> Q8a128Operand<'w> {
    /// Copy off whatever arena this operand lives in, yielding one that owns its
    /// bytes and so may be `'static`.
    ///
    /// The sanctioned escape from a wave, and the only one: it really copies
    /// rather than laundering the lifetime. Needed where an operand crosses a
    /// boundary the borrow checker cannot follow — the expert-pipeline thread
    /// receives its work over a channel, so a `MoeWorkRequest` cannot carry a
    /// lifetime and must own what it holds.
    pub fn to_owned(&self) -> Result<Q8a128Operand<'static>> {
        let data = match &self.data {
            Q8a128Data::Owned(slice) => Q8a128Data::Owned(slice.try_clone().w()?),
            Q8a128Data::Tensor(t) => Q8a128Data::Tensor(t.to_owned_tensor()?),
        };
        Ok(Q8a128Operand {
            data,
            rows: self.rows,
            cols: self.cols,
            lead: self.lead.clone(),
        })
    }

    /// The arena this operand's bytes came from, for a kernel that wants to
    /// allocate its output alongside them.
    ///
    /// The `Owned` arm holds a pool `CudaSlice` and so reports
    /// [`Backing::Owned`]; the `Tensor` arm forwards whatever its storage
    /// carries, which is how a wave-backed activation propagates through a
    /// whole expert chain without any call site naming a generation.
    pub fn backing(&self) -> Backing {
        match &self.data {
            Q8a128Data::Owned(_) => Backing::Owned,
            Q8a128Data::Tensor(t) => t.cuda_backing(),
        }
    }

    /// Wrap owned q8a128 blocks of flattened logical shape `[rows × cols]`. `lead` defaults to
    /// `[rows]` (a 2D `[M, N]` output); use [`Self::with_lead`] to preserve higher activation
    /// ranks. The matmul mode (mode-1 vs mode-2 weight-reuse) is NOT stored here — it is a
    /// kernel/tiling property the dispatcher derives from the token count, not an attribute of
    /// the (mode-independent) activation bytes.
    pub fn new(data: CudaSlice<u8>, rows: usize, cols: usize) -> Self {
        Self {
            data: Q8a128Data::Owned(data),
            rows,
            cols,
            lead: vec![rows],
        }
    }

    /// Wrap a contiguous `U8` [`Tensor`] of q8a1024 bytes as an operand, no copy. For producers
    /// that emit through `apply_op1` (the B2 decode context), whose result is a tensor.
    pub fn from_tensor(data: crate::LiveTensor<'w>, rows: usize, cols: usize) -> Self {
        Self {
            data: Q8a128Data::Tensor(data),
            rows,
            cols,
            lead: vec![rows],
        }
    }

    /// Set the output leading dims (the activation's dims minus K); `prod(lead)` must equal
    /// `rows`. Lets the int8 matmul reproduce the activation's rank, e.g. `[B, M, K]→[B, M, N]`.
    pub fn with_lead(mut self, lead: Vec<usize>) -> Self {
        debug_assert_eq!(
            lead.iter().product::<usize>(),
            self.rows,
            "lead must multiply to rows"
        );
        self.lead = lead;
        self
    }

    /// Borrow the owned backing slice; bails on the tensor-backed variant. For callers/tests that
    /// need the raw `CudaSlice` directly (e.g. `dequantize_q8a128`).
    pub fn data_slice(&self) -> Result<&CudaSlice<u8>> {
        match &self.data {
            Q8a128Data::Owned(s) => Ok(s),
            Q8a128Data::Tensor(_) => {
                crate::bail!("Q8a128Operand::data_slice: operand is tensor-backed, not owned")
            }
        }
    }

    /// Consume into the owned backing slice; bails on the tensor-backed variant.
    pub fn into_owned_data(self) -> Result<CudaSlice<u8>> {
        match self.data {
            Q8a128Data::Owned(s) => Ok(s),
            Q8a128Data::Tensor(_) => {
                crate::bail!("Q8a128Operand::into_owned_data: operand is tensor-backed, not owned")
            }
        }
    }

    /// Re-describe these bytes as a `'static` operand that **borrows** them.
    ///
    /// The counterpart to [`Self::to_owned`], and the one to prefer: `to_owned`
    /// copies the whole activation off the arena, which on the MoE path is a
    /// per-layer device copy of the entire ln2 output. This copies nothing — it
    /// wraps the same address as a [`LeaseOrigin::Foreign`] lease, so dropping
    /// the result frees nothing and the producer stays the owner.
    ///
    /// `'static` here is a statement about *ownership*, not lifetime: the result
    /// owns no memory, so there is nothing for a lifetime to bound. That is what
    /// lets it cross a channel — a `MoeWorkRequest` cannot carry a borrow — and
    /// it is also exactly why the caller carries the obligation below.
    ///
    /// # Safety
    /// `self` must outlive every use of the returned operand. On the expert
    /// pipeline that is structural rather than hoped for: `submit_moe_work`
    /// sends the request and immediately blocks on the response channel, so the
    /// submitting frame — and the activation it holds — is live for the whole of
    /// the worker's use, and both threads issue on the same stream.
    pub unsafe fn as_foreign_lease(&self, device: &CudaDevice) -> Result<Q8a128Operand<'static>> {
        let bytes = self.byte_len();
        let dev = crate::Device::Cuda(device.clone());
        let leased = self.with_device_ptr(device, |ptr| {
            LiveTensor::from_leased_cuda_ptr(
                ptr,
                crate::DType::U8,
                bytes,
                &dev,
                LeaseOrigin::Foreign,
            )
        })?;
        Ok(Q8a128Operand {
            data: Q8a128Data::Tensor(leased),
            rows: self.rows,
            cols: self.cols,
            lead: self.lead.clone(),
        })
    }

    /// Packed size of the q8a1024 blocks backing this operand.
    ///
    /// Delegates to [`q8a1024_byte_len`] rather than recomputing: this is the
    /// size `as_foreign_lease` stamps on the lease it hands the expert pipeline,
    /// and a figure that disagrees with what the quantizer actually allocated is
    /// a lease that lies about its own extent.
    pub fn byte_len(&self) -> usize {
        q8a1024_byte_len(self.rows, self.cols)
    }

    /// Run `f` with the raw device pointer to the q8a128 bytes, keeping the backing's
    /// storage/stream guards alive for the duration so the pointer stays valid. Unifies the
    /// owned-slice and tensor-backed variants for the kernel dispatch.
    pub fn with_device_ptr<R>(
        &self,
        device: &CudaDevice,
        f: impl FnOnce(u64) -> Result<R>,
    ) -> Result<R> {
        let stream = device.cuda_stream();
        match &self.data {
            Q8a128Data::Owned(s) => {
                let (ptr, _g) = s.device_ptr(&stream);
                f(ptr)
            }
            Q8a128Data::Tensor(t) => {
                let (storage, layout) = t.storage_and_layout();
                let cuda = match &*storage {
                    crate::Storage::Cuda(c) => c,
                    _ => crate::bail!("Q8a128Operand: tensor backing must be a CUDA tensor"),
                };
                let slice = cuda.as_cuda_slice::<u8>()?.slice(layout.start_offset()..);
                let (ptr, _g) = slice.device_ptr(&stream);
                f(ptr)
            }
        }
    }
}

/// A qmatmul activation/LHS that is either a plain float tensor (F16/BF16/F32 — the
/// dequant-weight float path) or pre-quantized q8a1024 int8 blocks (the int8 tensor-core
/// path). Threading one type through the matmul hides whether it runs in float or int8
/// mode: the int8 arm runs the q8a128 tensor-core path (mode-1/mode-2 chosen by the occupancy
/// formula `q8a128_dense_use_mode2` at dispatch), the float arm derives it from the tensor's dtype.
/// Two lifetimes, deliberately distinct: `'a` is how long this borrow of the
/// operand lasts, `'w` is the generation the operand's memory belongs to.
///
/// Collapsing them (`Float(&'a LiveTensor<'a>)`) forces the result of a matmul
/// to be bounded by the borrow of a *local variable* rather than by the arena
/// the bytes came from — so a pool-backed activation held in a local could not
/// produce a result that outlives the local, even though the result is owned.
/// Keeping them apart lets `'w` stay `'static` for pool operands and contract
/// to the guard only for wave-backed ones, which is the whole point.
pub enum DynamicTensor<'a, 'w> {
    Float(&'a crate::LiveTensor<'w>),
    Int8(&'a Q8a128Operand<'w>),
}

impl DynamicTensor<'_, '_> {
    /// The arena this activation came from, so a matmul over it can allocate its
    /// output alongside. Forwarding this at every call site is the whole of the
    /// inheritance rule on the consumer side.
    pub fn backing(&self) -> Backing {
        match self {
            Self::Float(t) => t.cuda_backing(),
            Self::Int8(op) => op.backing(),
        }
    }
}

/// Owned activation operand produced by [`to_dynamic`]: a float tensor ([`Int8Mode::Off`]) or
/// pre-quantized q8a128 blocks (any int8 mode). `DynamicTensor` borrows, so this owns the
/// chosen representation and hands out a borrow via [`DynamicActs::as_dynamic`] for the matmul.
pub enum DynamicActs<'w> {
    Float(crate::LiveTensor<'w>),
    Int8(Q8a128Operand<'w>),
}

impl<'w> DynamicActs<'w> {
    /// Borrow as the matmul-facing [`DynamicTensor`].
    pub fn as_dynamic(&self) -> DynamicTensor<'_, 'w> {
        match self {
            DynamicActs::Float(t) => DynamicTensor::Float(t),
            DynamicActs::Int8(op) => DynamicTensor::Int8(op),
        }
    }
}

/// Convert activations `[M × K]` (or `[B × M × K]`) to the matmul operand selected by `mode`,
/// the single knob that drives the whole inference's numeric mode: any int8 mode → quantize to
/// q8a128 (the int8 tensor-core path, paired with KO weights), [`Int8Mode::Off`] → keep the float
/// tensor (the dequant-weight float path). The q8a128 activation is identical for
/// [`Int8Mode::Performance`] and [`Int8Mode::Precision`] — only the weight twin differs — so this
/// branches solely on [`Int8Mode::is_int8`]. Paired with `QMatMul::repack_for_optimization` on the
/// weight side; the matmul's KO⇔int8 guard keeps the two consistent.
pub fn to_dynamic<'w>(
    xs: &crate::LiveTensor<'w>,
    mode: Int8Mode,
    device: &CudaDevice,
) -> Result<DynamicActs<'w>> {
    use crate::cuda_backend::CudaStorageSlice;
    if !mode.is_int8() {
        return Ok(DynamicActs::Float(xs.clone()));
    }
    let (rows, cols) = match xs.dims() {
        &[m, k] => (m, k),
        &[b, m, k] => (b * m, k),
        s => crate::bail!("to_dynamic(int8): expected 2D/3D activations, got {s:?}"),
    };
    let dtype_code: i32 = match xs.dtype() {
        crate::DType::F16 => 0,
        crate::DType::BF16 => 1,
        crate::DType::F32 => 2,
        d => crate::bail!("to_dynamic(int8): unsupported activation dtype {d:?}"),
    };
    let (storage, layout) = xs.storage_and_layout();
    let (o1, o2) = layout
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::RequiresContiguous { op: "to_dynamic" }.bt())?;
    let cuda = match &*storage {
        crate::Storage::Cuda(c) => c,
        _ => crate::bail!("to_dynamic(int8): activations must be a CUDA tensor"),
    };
    let stream = device.cuda_stream();
    let op = match &cuda.slice {
        CudaStorageSlice::F16(s) => {
            let v = s.slice(o1..o2);
            let (ptr, _g) = v.device_ptr(&stream);
            quantize_acts_q8a128(ptr, dtype_code, rows, cols, device)?
        }
        CudaStorageSlice::BF16(s) => {
            let v = s.slice(o1..o2);
            let (ptr, _g) = v.device_ptr(&stream);
            quantize_acts_q8a128(ptr, dtype_code, rows, cols, device)?
        }
        CudaStorageSlice::F32(s) => {
            let v = s.slice(o1..o2);
            let (ptr, _g) = v.device_ptr(&stream);
            quantize_acts_q8a128(ptr, dtype_code, rows, cols, device)?
        }
        _ => crate::bail!("to_dynamic(int8): activation slice dtype must be F16/BF16/F32"),
    };
    // Preserve the activation's leading dims (everything but K) so the int8 matmul rebuilds
    // the output rank ([B,M,K]→[B,M,N]) exactly like the float path.
    let lead: Vec<usize> = xs.dims()[..xs.rank() - 1].to_vec();
    Ok(DynamicActs::Int8(op.with_lead(lead)))
}

/// Bytes a `[rows, cols]` activation occupies in the q8a1024 flat-grouped layout:
/// 8 × 128-element tiles per 1152-byte super-block (see blocks.cuh).
///
/// The single definition of this size. `cols` is only required to be a multiple of
/// 128, so the tile count is rounded up to whole super-blocks across the whole
/// buffer — dividing `cols` by 1024 instead truncates for every K that is not a
/// multiple of 1024, which includes the MoE intermediate widths (Qwen3-30B-A3B's
/// is 768, where that form yields zero).
pub(crate) fn q8a1024_byte_len(rows: usize, cols: usize) -> usize {
    (rows * (cols / 128)).div_ceil(8) * 1152
}

/// Quantize activations `[rows, cols]` (dtype 0=F16,1=BF16,2=F32 at `act_ptr`) →
/// q8a1024 flat-grouped blocks (8 × 128-tiles per 1152-byte super-block; qs
/// de-interleaved from the per-32 ds — see blocks.cuh). The matmul mode is not chosen here; it is
/// derived later by the occupancy formula `q8a128_dense_use_mode2` at dispatch.
pub fn quantize_acts_q8a128(
    act_ptr: u64,
    dtype: i32,
    rows: usize,
    cols: usize,
    device: &CudaDevice,
) -> Result<Q8a128Operand<'static>> {
    let bytes = q8a1024_byte_len(rows, cols);
    let mut out = unsafe { device.alloc::<u8>(bytes)? };
    {
        let stream = device.cuda_stream();
        let (op, _g) = out.device_ptr_mut(&stream);
        unsafe {
            run_quantize_q8a128(
                act_ptr as *const std::ffi::c_void,
                op as *mut std::ffi::c_void,
                rows as i32,
                cols as i32,
                dtype,
            );
        }
    }
    Ok(Q8a128Operand::new(out, rows, cols))
}

/// Fused RMSNorm → q8a128: normalize each row of `xs` `[.. × K]` by `alpha` `[K]` and emit the
/// q8a128 activation operand directly, in ONE kernel — the producer epilogue for B1/B3/B5. This
/// replaces the unfused `rms_norm` (FP store) + [`quantize_acts_q8a128`] (re-read) pair, removing
/// the standalone quantize launch + FP round-trip that dominates M=1 decode. The result tracks
/// that two-call path within float margin (the quant grid is per-128; the normalized values are
/// rounded through the input dtype before quantization, mirroring the FP store). `K` must be a
/// multiple of 128 and ≤ 8192 (the row is cached in shared memory). Leading dims are preserved on
/// the operand so the int8 matmul rebuilds the output rank like the float path.
pub fn rms_norm_q8a128<'w>(
    xs: &crate::Tensor,
    alpha: &crate::Tensor,
    eps: f32,
    device: &CudaDevice,
    origin: Backing,
) -> Result<Q8a128Operand<'w>> {
    use crate::cuda_backend::CudaStorageSlice;
    let (rows, cols) = match xs.dims() {
        &[m, k] => (m, k),
        &[b, m, k] => (b * m, k),
        s => crate::bail!("rms_norm_q8a128: expected 2D/3D activations, got {s:?}"),
    };
    if cols % 128 != 0 {
        crate::bail!("rms_norm_q8a128: K={cols} must be a multiple of 128");
    }
    if cols > 8192 {
        crate::bail!("rms_norm_q8a128: K={cols} exceeds the cached-row limit (8192)");
    }
    if alpha.dims() != [cols] {
        crate::bail!(
            "rms_norm_q8a128: alpha must be [{cols}], got {:?}",
            alpha.dims()
        );
    }
    if xs.dtype() != alpha.dtype() {
        crate::bail!(
            "rms_norm_q8a128: xs/alpha dtype mismatch {:?} vs {:?}",
            xs.dtype(),
            alpha.dtype()
        );
    }
    // FloatDType code for the reduce dispatcher: 0=f32, 2=f16, 3=bf16.
    let dtype_code: i32 = match xs.dtype() {
        crate::DType::F32 => 0,
        crate::DType::F16 => 2,
        crate::DType::BF16 => 3,
        d => crate::bail!("rms_norm_q8a128: unsupported dtype {d:?}"),
    };

    let bytes = q8a1024_byte_len(rows, cols);
    let out_bytes = bytes;
    let (out_ptr_planned, owned, out_backing) = resolve_u8_out(origin, device, out_bytes)?;

    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let (xo1, xo2) = xs_layout.contiguous_offsets().ok_or_else(|| {
        crate::Error::RequiresContiguous {
            op: "rms_norm_q8a128(xs)",
        }
        .bt()
    })?;
    let xs_cuda = match &*xs_storage {
        crate::Storage::Cuda(c) => c,
        _ => crate::bail!("rms_norm_q8a128: xs must be a CUDA tensor"),
    };
    let (a_storage, a_layout) = alpha.storage_and_layout();
    let (ao1, ao2) = a_layout.contiguous_offsets().ok_or_else(|| {
        crate::Error::RequiresContiguous {
            op: "rms_norm_q8a128(alpha)",
        }
        .bt()
    })?;
    let a_cuda = match &*a_storage {
        crate::Storage::Cuda(c) => c,
        _ => crate::bail!("rms_norm_q8a128: alpha must be a CUDA tensor"),
    };

    let stream = device.cuda_stream();
    {
        let out_ptr = out_ptr_planned;
        // The src/alpha slices and their device-ptr guards must outlive the launch, so the kernel
        // call happens inside the match arm where all are still alive.
        macro_rules! launch {
            ($xsl:expr, $asl:expr) => {{
                let xv = $xsl.slice(xo1..xo2);
                let av = $asl.slice(ao1..ao2);
                let (sp, _gx) = xv.device_ptr(&stream);
                let (ap, _ga) = av.device_ptr(&stream);
                unsafe {
                    run_rmsnorm_q8a128_op(
                        dtype_code,
                        sp as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        ap as *const std::ffi::c_void,
                        rows as i32,
                        cols as i32,
                        eps,
                    );
                }
            }};
        }
        match (&xs_cuda.slice, &a_cuda.slice) {
            (CudaStorageSlice::F16(x), CudaStorageSlice::F16(a)) => launch!(x, a),
            (CudaStorageSlice::BF16(x), CudaStorageSlice::BF16(a)) => launch!(x, a),
            (CudaStorageSlice::F32(x), CudaStorageSlice::F32(a)) => launch!(x, a),
            _ => crate::bail!("rms_norm_q8a128: xs/alpha must be F16/BF16/F32 and match"),
        }
    }

    let lead: Vec<usize> = xs.dims()[..xs.rank() - 1].to_vec();
    q8a128_from_out(
        owned,
        out_ptr_planned,
        out_backing,
        bytes,
        rows,
        cols,
        device,
    )
    .map(|o| o.with_lead(lead))
}

/// Fused SwiGLU → q8a128: compute `silu(gate) · up` element-wise and emit the q8a128 activation
/// operand directly, in ONE kernel — the producer epilogue for B4 (feeds the down projection).
/// Replaces the unfused `silu_mul` (FP store) + [`quantize_acts_q8a128`] (re-read). `gate`/`up`
/// must share shape `[.. × K]` and dtype; `K` (and the total element count) must be a multiple of
/// 128. Tracks the two-call path within float margin (silu uses the same fast-exp path, the
/// result is rounded through the input dtype before quantization). Leading dims are preserved.
pub fn silu_mul_q8a128<'w>(
    gate: &crate::Tensor,
    up: &crate::Tensor,
    device: &CudaDevice,
    origin: Backing,
) -> Result<Q8a128Operand<'w>> {
    use crate::cuda_backend::CudaStorageSlice;
    if gate.dims() != up.dims() {
        crate::bail!(
            "silu_mul_q8a128: gate/up shape mismatch {:?} vs {:?}",
            gate.dims(),
            up.dims()
        );
    }
    if gate.dtype() != up.dtype() {
        crate::bail!(
            "silu_mul_q8a128: gate/up dtype mismatch {:?} vs {:?}",
            gate.dtype(),
            up.dtype()
        );
    }
    let (rows, cols) = match gate.dims() {
        &[m, k] => (m, k),
        &[b, m, k] => (b * m, k),
        s => crate::bail!("silu_mul_q8a128: expected 2D/3D activations, got {s:?}"),
    };
    if (rows * cols) % 128 != 0 {
        crate::bail!(
            "silu_mul_q8a128: rows*cols ({}) must be a multiple of 128",
            rows * cols
        );
    }
    // FusedSiluMul dtype code: 0=f32, 1=f16, 2=bf16.
    let dtype_code: i32 = match gate.dtype() {
        crate::DType::F32 => 0,
        crate::DType::F16 => 1,
        crate::DType::BF16 => 2,
        d => crate::bail!("silu_mul_q8a128: unsupported dtype {d:?}"),
    };

    let bytes = q8a1024_byte_len(rows, cols);
    let out_bytes = bytes;
    let (out_ptr_planned, owned, out_backing) = resolve_u8_out(origin, device, out_bytes)?;

    let (g_storage, g_layout) = gate.storage_and_layout();
    let (go1, go2) = g_layout.contiguous_offsets().ok_or_else(|| {
        crate::Error::RequiresContiguous {
            op: "silu_mul_q8a128(gate)",
        }
        .bt()
    })?;
    let g_cuda = match &*g_storage {
        crate::Storage::Cuda(c) => c,
        _ => crate::bail!("silu_mul_q8a128: gate must be a CUDA tensor"),
    };
    let (u_storage, u_layout) = up.storage_and_layout();
    let (uo1, uo2) = u_layout.contiguous_offsets().ok_or_else(|| {
        crate::Error::RequiresContiguous {
            op: "silu_mul_q8a128(up)",
        }
        .bt()
    })?;
    let u_cuda = match &*u_storage {
        crate::Storage::Cuda(c) => c,
        _ => crate::bail!("silu_mul_q8a128: up must be a CUDA tensor"),
    };

    let stream = device.cuda_stream();
    {
        let out_ptr = out_ptr_planned;
        macro_rules! launch {
            ($gsl:expr, $usl:expr) => {{
                let gv = $gsl.slice(go1..go2);
                let uv = $usl.slice(uo1..uo2);
                let (gp, _gg) = gv.device_ptr(&stream);
                let (upp, _gu) = uv.device_ptr(&stream);
                unsafe {
                    run_silu_mul_q8a128_op(
                        dtype_code,
                        gp as *const std::ffi::c_void,
                        upp as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        rows as i32,
                        cols as i32,
                    );
                }
            }};
        }
        match (&g_cuda.slice, &u_cuda.slice) {
            (CudaStorageSlice::F16(g), CudaStorageSlice::F16(u)) => launch!(g, u),
            (CudaStorageSlice::BF16(g), CudaStorageSlice::BF16(u)) => launch!(g, u),
            (CudaStorageSlice::F32(g), CudaStorageSlice::F32(u)) => launch!(g, u),
            _ => crate::bail!("silu_mul_q8a128: gate/up must be F16/BF16/F32 and match"),
        }
    }

    let lead: Vec<usize> = gate.dims()[..gate.rank() - 1].to_vec();
    q8a128_from_out(
        owned,
        out_ptr_planned,
        out_backing,
        bytes,
        rows,
        cols,
        device,
    )
    .map(|o| o.with_lead(lead))
}

/// Quantize F32 weights `[nrows × ncols]` (row-major) → a GPU buffer in the lane-major KO
/// layout the int8 KO matmul reads (`Q4_KO`/`Q5_KO`/`Q6_KO`/`Q8_KO`). The de-interleave runs
/// on the GPU (`run_quantize_ko`), byte-identical to the CPU `ko_quant::quantize_ko` — the
/// symmetric weight counterpart to `quantize_acts_q8a128`. `nrows` is N (output rows), `ncols`
/// is K; `nrows` must be a multiple of 8, `ncols` a multiple of 128.
pub fn quantize_ko_weights(
    data: &[f32],
    nrows: usize,
    ncols: usize,
    dtype: GgmlDType,
    device: &CudaDevice,
) -> Result<CudaSlice<u8>> {
    if !nrows.is_multiple_of(8) || !ncols.is_multiple_of(128) || data.len() != nrows * ncols {
        crate::bail!(
            "quantize_ko_weights: bad shape nrows={nrows} ncols={ncols} len={}",
            data.len()
        );
    }
    let qtype = dtype_to_qtype(dtype)? as i32;
    let bytes = (nrows / 8) * (ncols / 128) * crate::quantized::ko_quant::ko_chunk_bytes(dtype);
    let w_dev = device.memcpy_stod(data)?;
    let mut out = unsafe { device.alloc::<u8>(bytes)? };
    {
        let stream = device.cuda_stream();
        let (wp, _g0) = w_dev.device_ptr(&stream);
        let (op, _g1) = out.device_ptr_mut(&stream);
        unsafe {
            run_quantize_ko(
                wp as *const f32,
                op as *mut std::ffi::c_void,
                nrows as i32,
                ncols as i32,
                qtype,
            );
        }
    }
    Ok(out)
}

/// Dequantize a lane-major KO chunk buffer → f32 `[nrows × ncols]` (row-major) on the GPU
/// (`run_dequantize_ko`). Inverse of [`quantize_ko_weights`]; matches `ko_quant::dequant_ko`.
pub fn dequant_ko_weights(
    chunk: &CudaSlice<u8>,
    nrows: usize,
    ncols: usize,
    dtype: GgmlDType,
    device: &CudaDevice,
) -> Result<CudaSlice<f32>> {
    let qtype = dtype_to_qtype(dtype)? as i32;
    let mut out = unsafe { device.alloc::<f32>(nrows * ncols)? };
    {
        let stream = device.cuda_stream();
        let (cp, _g0) = chunk.device_ptr(&stream);
        let (op, _g1) = out.device_ptr_mut(&stream);
        unsafe {
            run_dequantize_ko(
                cp as *const std::ffi::c_void,
                op as *mut f32,
                nrows as i32,
                ncols as i32,
                qtype,
            );
        }
    }
    Ok(out)
}

/// Dequantize q8a1024 flat-grouped blocks → f32 `[rows, cols]`.
pub fn dequantize_q8a128(
    blocks: &CudaSlice<u8>,
    rows: usize,
    cols: usize,
    device: &CudaDevice,
) -> Result<CudaSlice<f32>> {
    let mut out = unsafe { device.alloc::<f32>(rows * cols)? };
    {
        let stream = device.cuda_stream();
        let (bp, _g1) = blocks.device_ptr(&stream);
        let (op, _g2) = out.device_ptr_mut(&stream);
        unsafe {
            run_dequantize_q8a128(
                bp as *const std::ffi::c_void,
                op as *mut std::ffi::c_void,
                rows as i32,
                cols as i32,
                2, // F32 output
            );
        }
    }
    Ok(out)
}

/// INT8 grouped matmul on PRE-QUANTIZED q8a128 activations
/// (`block_q8a128[total_batch][K/128]`). The caller quantizes once (amortized
/// across gate/up). Tensor-core only (the q8a128 input has no FP fallback);
/// output is F32. Routes through the unified `run_grouped_quantized_matmul`
/// with ytype = Q8A128. The grouped/MoE path has no mode-2 kernel, so it always
/// runs mode-1 (Q8A128V) regardless of M. Internal building block for the `Int8`
/// arm of [`grouped_qmatmul`].
/// Token width of one mode-2 (`Bm=32`, `N_SUB=2`) grouped-GEMM expert tile —
/// the width the DEVICE-side tile builder (`moe_bucketize`) segments at, and
/// the mode its consumers launch (`n_sub = 2`; the decode regime, where 32 is
/// already the full reuse win). The host tile builder
/// (`grouped_matmul_gemx_q8a128_with_mode`) derives its width from the chosen
/// mode instead (`16·n_sub`, up to `Bm=128` for prefill-scale rows-per-expert);
/// a builder/launch width divergence silently mis-strides the kernel's batch
/// slices, so each launch site pairs its table width with its `n_sub`.
pub const GROUPED_GEMM_TILE_W: usize = 32;

/// Kernel bounds of the MoE bucketize, re-exported so every gate that decides
/// "can this routing take the device-table path?" reads the SAME constants the
/// kernel and its wrapper validate against — a hand-copied bound here is the
/// one place the mirrored-constant scheme couldn't reach.
pub use candle_kernels::simple::moe_bucketize::{
    MAX_EXPERTS as MOE_MAX_EXPERTS, MAX_TOPK as MOE_MAX_TOPK,
};

fn grouped_matmul_gemx_q8a128<'w>(
    act_ptr: u64,
    weight_ptrs: &[u64],
    weight_dtype: GgmlDType,
    nrows: usize,
    ncols: usize,
    total_batch: usize,
    expert_offsets: &[i32],
    device: &CudaDevice,
    origin: Backing,
) -> Result<crate::LiveTensor<'w>> {
    let num_experts = weight_ptrs.len();
    if num_experts == 0 {
        crate::bail!("grouped_matmul_gemx_q8a128: no experts provided");
    }
    // Token-tile mode by rows-per-active-expert. The grouped int8 kernel loads
    // + dequants each expert weight chunk ONCE per tile and sweeps the
    // m16n8k32 core across the tile's 16-row sub-tiles, so the tile width IS
    // the weight-reuse factor: at decode's 1–32 rows/expert the 32-wide
    // mode-2 tile is already optimal (a partial tile costs the same weight
    // traffic), but at PREFILL's ~100–300 rows/expert it re-streams and
    // re-dequants every expert 4×+ per launch — measured as the flat
    // ~0.87 ms/token marginal prefill cost. Wide modes (Bm 64 / 128) exist
    // for the KO rows (the only int8 formats); thresholds sit at the widths
    // where the wider tile's weight-traffic saving is guaranteed even for a
    // final partial tile.
    let active: usize = (0..num_experts)
        .filter(|&e| expert_offsets[e + 1] > expert_offsets[e])
        .count();
    let avg_rows = total_batch.checked_div(active).unwrap_or(0);
    // Mode choice is BENCH-derived (`moe_layer_gemm_bench`, real shapes:
    // 256 experts, gate/up [2048,7168] / down [7168,2048] MXFP4_KO): at
    // ~91 rows/expert Bm-128 wins (29.3 vs 36.3 ms), but at ~192 rows/expert
    // Bm-64 beats Bm-128 (52.9 vs 60.3 ms) — the widest tile loses more
    // occupancy than its extra reuse pays back once several tiles per expert
    // exist. The bands encode those two measured points.
    let n_sub: usize = if !weight_dtype.is_ko() {
        2
    } else if avg_rows >= 128 {
        4
    } else if avg_rows >= 64 {
        8
    } else if avg_rows >= 32 {
        4
    } else {
        2
    };
    // q8a128 activations are ~1 B/elem (int8 qs + per-128 scales).
    let row_fast = grouped_grid_row_fast(total_batch * ncols, device);
    grouped_matmul_gemx_q8a128_with_mode(
        act_ptr,
        weight_ptrs,
        weight_dtype,
        nrows,
        ncols,
        total_batch,
        expert_offsets,
        device,
        origin,
        n_sub,
        row_fast,
    )
}

/// [`grouped_matmul_gemx_q8a128`] with the token-tile mode AND grid axis order
/// chosen by the caller — the test seam that proves the wide modes and both
/// grid orders bit-equal mode-2.
#[allow(clippy::too_many_arguments)]
pub(crate) fn grouped_matmul_gemx_q8a128_with_mode<'w>(
    act_ptr: u64,
    weight_ptrs: &[u64],
    weight_dtype: GgmlDType,
    nrows: usize,
    ncols: usize,
    total_batch: usize,
    expert_offsets: &[i32],
    device: &CudaDevice,
    origin: Backing,
    n_sub: usize,
    row_fast: bool,
) -> Result<crate::LiveTensor<'w>> {
    let num_experts = weight_ptrs.len();
    if num_experts == 0 {
        crate::bail!("grouped_matmul_gemx_q8a128: no experts provided");
    }
    if !nrows.is_multiple_of(32) {
        crate::bail!("grouped_matmul_gemx_q8a128: nrows={nrows} must be a multiple of 32");
    }
    if !ncols.is_multiple_of(128) {
        crate::bail!("grouped_matmul_gemx_q8a128: ncols={ncols} must be a multiple of 128");
    }
    if !matches!(n_sub, 2 | 4 | 8) || (n_sub > 2 && !weight_dtype.is_ko()) {
        crate::bail!(
            "grouped_matmul_gemx_q8a128: tile mode n_sub={n_sub} unsupported for {weight_dtype:?}"
        );
    }
    let qtype = dtype_to_qtype(weight_dtype)? as i32;

    let tile_w = 16 * n_sub;
    let mut tile_expert: Vec<i32> = Vec::new();
    let mut tile_b_start: Vec<i32> = Vec::new();
    let mut tile_b_cnt: Vec<i32> = Vec::new();
    for e in 0..num_experts {
        let mut s = expert_offsets[e];
        let end = expert_offsets[e + 1];
        while s < end {
            let cnt = (end - s).min(tile_w as i32);
            tile_expert.push(e as i32);
            tile_b_start.push(s);
            tile_b_cnt.push(cnt);
            s += cnt;
        }
    }
    let num_tiles = tile_expert.len();
    // Token tiles ride grid.y (row tiles are the fast axis for L2 reuse).
    if num_tiles > 65535 {
        crate::bail!("grouped_matmul_gemx_q8a128: num_tiles={num_tiles} exceeds grid.y max 65535");
    }

    let (dst_ptr, owned_dst, out_backing) =
        resolve_typed_out::<f32>(origin, device, nrows * total_batch)?;

    {
        let off_te = num_experts * 8;
        let off_tbs = off_te + num_tiles * 4;
        let off_tbc = off_tbs + num_tiles * 4;
        let total_bytes = off_tbc + num_tiles * 4;
        let mut packed: Vec<u8> = Vec::with_capacity(total_bytes);
        for &w in weight_ptrs {
            packed.extend_from_slice(&w.to_le_bytes());
        }
        for &x in &tile_expert {
            packed.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &tile_b_start {
            packed.extend_from_slice(&x.to_le_bytes());
        }
        for &x in &tile_b_cnt {
            packed.extend_from_slice(&x.to_le_bytes());
        }
        // The kernel reads the descriptor blob IN PLACE from the device-mapped
        // pinned table ring — no per-launch H2D copy, no device allocation.
        table_ring(device)?.with_table(&packed, |base| {
            let wptr_ptr = base;
            let te_ptr = base + off_te as u64;
            let tbs_ptr = base + off_tbs as u64;
            let tbc_ptr = base + off_tbc as u64;
            unsafe {
                crate::set_kernel_breadcrumb("run_grouped_quantized_matmul", file!(), line!());
                run_grouped_quantized_matmul(
                    wptr_ptr as *const std::ffi::c_void,
                    te_ptr as *const std::ffi::c_void,
                    tbs_ptr as *const std::ffi::c_void,
                    tbc_ptr as *const std::ffi::c_void,
                    act_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    ncols as i32, // ncols_x = K
                    nrows as i32, // nrows_x = N
                    ncols as i32, // y_stride (unused by int8 kernel; ABI)
                    nrows as i32, // dst_stride = N
                    num_tiles as i32,
                    qtype,
                    YType::Q8A128 as i32,
                    n_sub as i32,
                    row_fast as i32,
                );
            }
        })?;
    }

    let out_shape: Shape = vec![total_batch, nrows].into();
    tensor_from_out::<f32>(owned_dst, dst_ptr, out_backing, out_shape, device)
}

/// Grouped (MoE) quantized matmul over a [`DynamicTensor`] activation, dispatched by
/// numeric mode: `Int8` runs the q8a128 tensor-core grouped path (mode-1 only — the grouped
/// kernel has no mode-2, so an operand carrying `Q8A128X` from M≥64 still runs mode-1; the
/// operand's `ytype` is ignored here), `Float` runs the dequant-weight float grouped path.
/// M and K come from the activation (operand shape / tensor shape); `nrows` is N per
/// expert, `expert_offsets` partitions the M token rows across `weight_ptrs`.
/// KO weights and int8 (q8a128) activations are EXCLUSIVELY paired: the int8 tensor-core
/// kernels' per-128 deferred-scale fold reads only KO weights (the per-128 grid), and KO weights
/// are only consumed through the int8 path. Plain k-quants (Q2_K/Q4_K/…) have finer per-sub
/// scales the per-128 int8 fold does not apply — they must be re-quantized to a KO twin (`to_ko`)
/// to run int8; through the FP grouped path they stay non-KO. So `DynamicTensor::Int8` must pair
/// with a KO weight and `DynamicTensor::Float` with a non-KO weight — any cross combination has
/// no kernel and is rejected here rather than silently producing garbage.
fn ensure_qmatmul_pairing(input: &DynamicTensor, weight_dtype: GgmlDType) -> Result<()> {
    let is_int8 = matches!(input, DynamicTensor::Int8(_));
    if weight_dtype.is_ko() != is_int8 {
        crate::bail!(
            "qmatmul: KO weights and int8 q8a128 activations are exclusively paired — got \
             weight {weight_dtype:?} (KO={}) with {} activations. Pair Int8 activations with \
             KO weights, and Float activations with non-KO weights.",
            weight_dtype.is_ko(),
            if is_int8 { "Int8" } else { "Float" },
        );
    }
    Ok(())
}

pub fn grouped_qmatmul<'w>(
    input: DynamicTensor,
    weight_ptrs: &[u64],
    weight_dtype: GgmlDType,
    nrows: usize,
    expert_offsets: &[i32],
    device: &CudaDevice,
    origin: Backing,
) -> Result<crate::LiveTensor<'w>> {
    ensure_qmatmul_pairing(&input, weight_dtype)?;
    match input {
        DynamicTensor::Int8(op) => op.with_device_ptr(device, |act_ptr| {
            grouped_matmul_gemx_q8a128(
                act_ptr,
                weight_ptrs,
                weight_dtype,
                nrows,
                op.cols,
                op.rows,
                expert_offsets,
                device,
                origin,
            )
        }),
        DynamicTensor::Float(t) => {
            let (storage, layout) = t.storage_and_layout();
            let cuda = match &*storage {
                crate::Storage::Cuda(c) => c,
                _ => crate::bail!("grouped_qmatmul: Float input must be a CUDA tensor"),
            };
            let ncols = *layout.shape().dims().last().unwrap();
            // The FP grouped path writes through `&mut CudaSlice`, so it owns
            // its output rather than taking a planned slot. An owned tensor
            // coerces into any `'w` — it outlives every wave — so the caller
            // sees the same type either way. This is the non-int8 path: the
            // expert GEMMs of an F16/BF16 config still allocate, and the
            // detector reports them as such.
            let _ = origin;
            grouped_matmul_gemx_impl(
                weight_ptrs,
                weight_dtype,
                nrows,
                ncols,
                cuda,
                layout,
                expert_offsets,
                device,
            )
        }
    }
}

/// Grouped (MoE) q8a128 quantized matmul with **device-resident dispatch tables**:
/// the GPU-native twin of [`grouped_qmatmul`]'s int8 arm. Where the host path
/// packs per-active-expert weight pointers + tile tables on the CPU (requiring
/// the routing indices to round-trip GPU→CPU first), this variant reads
/// everything from device memory:
///  * `weight_ptrs_dev` — the resident expert pointer table (u64 rows, built
///    once at load; every expert VRAM-resident); `expert_base` is the element
///    offset of THIS layer's `[n_experts]` row block inside it, so
///    `tile_expert` carries RAW expert ids straight from `moe_bucketize`;
///  * `tile_expert` / `tile_b_start` / `tile_b_cnt` — `moe_bucketize`'s device
///    tile tables, launched at the `launch_tiles` upper bound; padding tiles
///    (`b_cnt == 0`) exit in the kernel without touching the pointer table.
///
/// Output is `[total_batch, nrows]` f32 with padding rows UNWRITTEN — the
/// deterministic scatter's segment tables never reference them. Bit-identical
/// to the host path for every valid row: same tables (proven by the bucketize
/// tests), same ascending-expert tile order, same kernel.
#[allow(clippy::too_many_arguments)]
pub fn grouped_qmatmul_dev_q8a128<'w>(
    op: &Q8a128Operand<'w>,
    weight_ptrs_dev: &CudaSlice<u64>,
    expert_base: usize,
    n_experts: usize,
    weight_dtype: GgmlDType,
    nrows: usize,
    tile_expert: &CudaSlice<i32>,
    tile_b_start: &CudaSlice<i32>,
    tile_b_cnt: &CudaSlice<i32>,
    launch_tiles: usize,
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    ensure_qmatmul_pairing(&DynamicTensor::Int8(op), weight_dtype)?;
    if !nrows.is_multiple_of(32) {
        crate::bail!("grouped_qmatmul_dev_q8a128: nrows={nrows} must be a multiple of 32");
    }
    if !op.cols.is_multiple_of(128) {
        crate::bail!(
            "grouped_qmatmul_dev_q8a128: ncols={} must be a multiple of 128",
            op.cols
        );
    }
    if launch_tiles == 0 {
        crate::bail!("grouped_qmatmul_dev_q8a128: launch_tiles must be > 0");
    }
    // Token tiles ride grid.y (row tiles are the fast axis for L2 reuse).
    if launch_tiles > 65535 {
        crate::bail!(
            "grouped_qmatmul_dev_q8a128: launch_tiles={launch_tiles} exceeds grid.y max 65535"
        );
    }
    let qtype = dtype_to_qtype(weight_dtype)? as i32;
    let total_batch = op.rows;
    let ncols = op.cols;

    // `tile_expert` values are bounded by the ROUTER's expert count (the
    // bucketize input), so the whole `[n_experts]` row this layer indexes must
    // lie inside the table — checking only the base would let a router/table
    // width mismatch dereference past the end (or into the next layer's row).
    if n_experts == 0
        || expert_base
            .checked_add(n_experts)
            .is_none_or(|end| end > weight_ptrs_dev.len())
    {
        crate::bail!(
            "grouped_qmatmul_dev_q8a128: expert row [{expert_base}, {expert_base}+{n_experts}) \
             exceeds table len {}",
            weight_ptrs_dev.len()
        );
    }
    // Inherits the gathered activation's arena, so the whole expert chain —
    // gate, up, SwiGLU, down — stays in the FFN generation the gather seeded.
    let (dst_ptr, owned_dst, out_backing) =
        resolve_typed_out::<f32>(op.backing(), device, nrows * total_batch)?;
    {
        let stream = device.cuda_stream();
        let (wp_base, _g0) = weight_ptrs_dev.device_ptr(&stream);
        // Offset to this layer's `[n_experts]` row inside the flat table.
        let wp = wp_base + (expert_base * std::mem::size_of::<u64>()) as u64;
        let (te, _g1) = tile_expert.device_ptr(&stream);
        let (tbs, _g2) = tile_b_start.device_ptr(&stream);
        let (tbc, _g3) = tile_b_cnt.device_ptr(&stream);
        // q8a128 activations are ~1 B/elem.
        let row_fast = grouped_grid_row_fast(total_batch * ncols, device) as i32;
        op.with_device_ptr(device, |act_ptr| {
            unsafe {
                run_grouped_quantized_matmul(
                    wp as *const std::ffi::c_void,
                    te as *const std::ffi::c_void,
                    tbs as *const std::ffi::c_void,
                    tbc as *const std::ffi::c_void,
                    act_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    ncols as i32, // ncols_x = K
                    nrows as i32, // nrows_x = N
                    ncols as i32, // y_stride (unused by int8 kernel; ABI)
                    nrows as i32, // dst_stride = N
                    launch_tiles as i32,
                    qtype,
                    YType::Q8A128 as i32,
                    // moe_bucketize builds 32-wide tiles (decode regime).
                    2,
                    row_fast,
                );
            }
            Ok(())
        })?;
    }

    let out_shape: Shape = vec![total_batch, nrows].into();
    tensor_from_out::<f32>(owned_dst, dst_ptr, out_backing, out_shape, device)
}

/// Dense (non-MoE) quantized matmul over a [`DynamicTensor`] activation: a single weight
/// `[nrows(N) × ncols(K)]` (KO format for the `Int8` arm, FP GEMX K/128-repack for the
/// `Float` arm) × the activation `[.. × K]` → `[.., N]` (leading dims preserved on both arms).
/// `Int8` runs the q8a128 tensor-core path (mode-1/mode-2 chosen by `q8a128_dense_use_mode2` at
/// dispatch), output F32; `Float` runs the dequant-weight float path
/// ([`dense_qmatmul_float`]), output matching the activation dtype. The caller stays
/// agnostic to which numeric mode runs. `weight_len` is the quantized-weight byte length
/// used by the float path; the int8 path ignores it. See docs/q8_matmul_pipeline.md.
/// KO format code for the segmented qkv kernel's `fmt` field (0=Q4_KO … 3=Q8_KO).
fn ko_fmt_code(dtype: GgmlDType) -> Result<i32> {
    Ok(match dtype {
        GgmlDType::Q4_KO => 0,
        GgmlDType::Q5_KO => 1,
        GgmlDType::Q6_KO => 2,
        GgmlDType::Q8_KO => 3,
        other => {
            crate::bail!("qkv_segmented_matmul: unsupported segment dtype {other:?} (KO only)")
        }
    })
}

/// Fused qkv int8 dense matmul: a SINGLE launch multiplies the shared q8a128 activation `op`
/// (`[lead.., K]`) by up to three KO weights of possibly-different formats, writing the
/// **concatenated** `[lead.., ΣN]` output at `out_dtype`. Per thread-block the global N-tile resolves to one
/// segment (boundaries align to the 32-row tile → no divergence) and runs that segment's format on
/// the shared activation. Float-identical to running the segments as separate `q8a128_dense_matmul`
/// calls, but one launch with full GPU occupancy (the tiny k/v GEMVs no longer starve).
///
/// `segments`: `(weight device ptr, KO dtype, N)` for each of q, k, v — N must be a multiple of 32.
pub(crate) fn qkv_segmented_matmul<'w>(
    op: &Q8a128Operand<'w>,
    segments: &[(u64, GgmlDType, usize)],
    out_dtype: crate::DType,
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    if !op.cols.is_multiple_of(128) {
        crate::bail!(
            "qkv_segmented_matmul: K={} must be a multiple of 128",
            op.cols
        );
    }
    if segments.is_empty() {
        crate::bail!("qkv_segmented_matmul: no segments");
    }
    let k = op.cols;
    let m = op.rows;

    // Build the device-side segment table (24-byte qkv_seg_t each): weights(8) fmt(4)
    // n_tile_start(4) n_size(4) dst_col_off(4).
    let mut bytes: Vec<u8> = Vec::with_capacity(segments.len() * 24);
    let mut tile_start: i32 = 0;
    let mut col_off: i32 = 0;
    let mut n_total: usize = 0;
    for &(wptr, dtype, n) in segments {
        if n % 32 != 0 {
            crate::bail!("qkv_segmented_matmul: segment N={n} must be a multiple of 32");
        }
        let fmt = ko_fmt_code(dtype)?;
        bytes.extend_from_slice(&wptr.to_le_bytes());
        bytes.extend_from_slice(&fmt.to_le_bytes());
        bytes.extend_from_slice(&tile_start.to_le_bytes());
        bytes.extend_from_slice(&(n as i32).to_le_bytes());
        bytes.extend_from_slice(&col_off.to_le_bytes());
        tile_start += n.div_ceil(32) as i32;
        col_off += n as i32;
        n_total += n;
    }
    let total_n_tiles = tile_start;

    let mode2 = q8a128_dense_use_mode2(m, n_total, k, cached_sm_count(device)) as i32;
    let out_code = out_dtype_code(out_dtype)?;
    // Inherits the fused ln1 activation's arena, so q/k/v land in the attention
    // generation alongside the norm that produced their operand.
    let (dst_ptr, owned_dst, out_backing) =
        resolve_out(out_dtype, op.backing(), device, m * n_total)?;
    op.with_device_ptr(device, |act_ptr| {
        // `bytes` is the HOST segment table (≤3 × 24B); the launcher copies it into by-value kernel
        // params, so there is no per-call device upload.
        let status = unsafe {
            run_qkv_segmented_matmul(
                bytes.as_ptr() as *const std::ffi::c_void,
                segments.len() as i32,
                act_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                k as i32,
                total_n_tiles,
                m as i32,
                n_total as i32,
                mode2,
                out_code,
            )
        };
        check_matmul_status(status, "qkv_segmented_matmul")
    })?;

    let mut out_dims = op.lead.clone();
    out_dims.push(n_total);
    let out_shape: Shape = out_dims.into();
    tensor_from_owned_out(owned_dst, dst_ptr, out_backing, out_shape, device)
}

/// The q8a128 int8 dense launch with an **explicit** tiling mode (`mode2`: false = mode-1 `Bm=16`,
/// true = mode-2 `Bm=32` weight-reuse). Production reaches this from [`dense_qmatmul`] with the mode
/// chosen by [`q8a128_dense_use_mode2`]; the crossover benchmark calls it directly to time each mode
/// at a fixed `(M, N, K)`. Result is the `[lead.., N]` output (rank rebuilt from `op.lead`) stored at
/// `out_dtype` — the MMA accumulates in F32 registers and converts on the store, so a narrow
/// `out_dtype` is bit-identical to the F32 kernel followed by a cast, minus the cast.
pub(crate) fn q8a128_dense_matmul<'w>(
    op: &Q8a128Operand<'w>,
    weight_ptr: u64,
    weight_dtype: GgmlDType,
    nrows: usize,
    weight_len: usize,
    mode2: bool,
    out_dtype: crate::DType,
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    if !nrows.is_multiple_of(32) {
        crate::bail!("q8a128_dense_matmul: nrows={nrows} must be a multiple of 32");
    }
    if !op.cols.is_multiple_of(128) {
        crate::bail!(
            "q8a128_dense_matmul: K={} must be a multiple of 128",
            op.cols
        );
    }
    let qtype = dtype_to_qtype(weight_dtype)? as i32;
    let (ncols, total_batch) = (op.cols, op.rows);
    let out_code = out_dtype_code(out_dtype)?;
    // Inherits the activation's arena: on the decode path this is o_proj over a
    // wave-backed attention context, so the projection lands in the same
    // generation rather than in the pool.
    let (dst_ptr, owned_dst, out_backing) =
        resolve_out(out_dtype, op.backing(), device, nrows * total_batch)?;
    op.with_device_ptr(device, |act_ptr| {
        let segment = VxSegment {
            weights: weight_ptr as *const std::ffi::c_void,
            batch_count: total_batch as i32,
        };
        let status = unsafe {
            run_quantized_matmul(
                &segment as *const VxSegment,
                1, // num_segments = 1 (non-MoE)
                act_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                ncols as i32,       // ncols_x = K
                nrows as i32,       // nrows_x = N
                total_batch as i32, // nrows_y = M
                nrows as i32,       // nrows_dst = N
                qtype,
                YType::Q8A128 as i32, // int8 activation; tiling forced via `mode2`
                weight_len,           // weight_bytes (FP path only; int8 ignores)
                mode2 as i32,         // 0 = mode-1, 1 = mode-2 (weight-reuse)
                out_code,             // store width
            )
        };
        check_matmul_status(status, "q8a128_dense_matmul")
    })?;
    // Rebuild the output rank from the operand's leading dims ([lead.., N]) so the int8 arm matches
    // the float arm on 3D+ activations (data is row-major [M, N]).
    let mut out_dims = op.lead.clone();
    out_dims.push(nrows);
    let out_shape: Shape = out_dims.into();
    tensor_from_owned_out(owned_dst, dst_ptr, out_backing, out_shape, device)
}

/// `out_dtype` is the width the result is stored at. The int8 arm passes it to the
/// kernel, which converts the F32 accumulator on the way out — the caller gets the
/// dtype it asked for with no cast launch. The FP arm has no such choice: those
/// kernels store at the activation dtype, so `out_dtype` must equal the input's.
pub fn dense_qmatmul<'w>(
    input: DynamicTensor<'_, 'w>,
    weight_ptr: u64,
    weight_dtype: GgmlDType,
    nrows: usize,
    weight_len: usize,
    out_dtype: crate::DType,
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    ensure_qmatmul_pairing(&input, weight_dtype)?;
    let qtype = dtype_to_qtype(weight_dtype)? as i32;
    match input {
        DynamicTensor::Int8(op) => {
            // Tiling choice (mode-1 Bm=16 vs mode-2 Bm=32, N_SUB=2): an occupancy decision driven by
            // M and N (block count vs SM count) plus the [17,32] trap — not weight bytes. The bench
            // reaches the same launch with a forced mode via `q8a128_dense_matmul`.
            let mode2 = q8a128_dense_use_mode2(op.rows, nrows, op.cols, cached_sm_count(device));
            q8a128_dense_matmul(
                op,
                weight_ptr,
                weight_dtype,
                nrows,
                weight_len,
                mode2,
                out_dtype,
                device,
            )
        }
        DynamicTensor::Float(t) => {
            let (storage, layout) = t.storage_and_layout();
            let cuda = match &*storage {
                crate::Storage::Cuda(c) => c,
                _ => crate::bail!("dense_qmatmul: Float input must be a CUDA tensor"),
            };
            let ncols = *layout.shape().dims().last().unwrap();
            let (out_storage, out_shape) = dense_qmatmul_float(
                weight_ptr, qtype, weight_len, nrows, ncols, cuda, layout, device,
            )?;
            // Checked against what the path actually produced rather than re-deriving
            // it from the activation: the FP kernels store at the activation dtype,
            // but an exotic activation is promoted first, so the input dtype is not
            // always the answer.
            if out_storage.dtype() != out_dtype {
                crate::bail!(
                    "dense_qmatmul: the FP path produced {:?}, but {out_dtype:?} was requested",
                    out_storage.dtype()
                );
            }
            Ok(crate::tensor::from_storage(
                crate::Storage::Cuda(out_storage),
                out_shape,
                crate::op::BackpropOp::none(),
                false,
            ))
        }
    }
}

// =============================================================================
// FUSED MoE GATHER / WEIGHTED-SCATTER-ADD
// =============================================================================
// These wrap the CUDA kernels in candle-kernels so that compute.rs can call
// them from Tensor-level code without manually extracting device pointers.

/// Map Candle DType â†’ MoeScatterDType enum value for the CUDA dispatcher.
fn dtype_to_moe_scatter_dtype(dtype: crate::DType) -> Result<i32> {
    use candle_kernels::simple::moe_scatter::MoeScatterDType;
    match dtype {
        crate::DType::F32 => Ok(MoeScatterDType::F32 as i32),
        crate::DType::F16 => Ok(MoeScatterDType::F16 as i32),
        crate::DType::BF16 => Ok(MoeScatterDType::BF16 as i32),
        other => crate::bail!("moe_scatter: unsupported dtype {other:?}"),
    }
}

/// Fused gather: `out[i] = xs[token_ids[i]]` for 2-D tensors.
///
/// Single kernel launch replaces `Tensor::new(ids) + xs.index_select`.
/// Returns a new tensor `[total_rows, hidden_dim]`.
///
/// * `xs` â€” input activations `[num_tokens, hidden_dim]`
/// * `ids_dev` â€” pre-uploaded GPU u32 index buffer
/// * `total_rows` â€” number of rows to gather
/// * `device` â€” CUDA device
pub fn fused_moe_gather(
    xs: &crate::Tensor,
    ids_dev: &CudaSlice<u32>,
    total_rows: usize,
    device: &CudaDevice,
) -> Result<crate::Tensor> {
    use crate::cuda_backend::CudaStorageSlice;

    let (_, hidden_dim) = xs.dims2()?;

    let xs_storage = xs.storage_and_layout().0;
    let xs_cuda = match &*xs_storage {
        crate::Storage::Cuda(s) => s,
        _ => crate::bail!("fused_moe_gather: expected CUDA input"),
    };

    let dtype = xs.dtype();
    let moe_dtype = dtype_to_moe_scatter_dtype(dtype)?;

    // Allocate output
    let stream = device.cuda_stream();

    macro_rules! dispatch_gather {
        ($src_slice:expr, $elem_ty:ty) => {{
            let out_slice: CudaSlice<$elem_ty> =
                unsafe { device.alloc::<$elem_ty>(total_rows * hidden_dim)? };
            {
                let (src_ptr, _sg) = $src_slice.device_ptr(&stream);
                let (out_ptr, _og) = out_slice.device_ptr(&stream);
                let (ids_ptr, _ig) = ids_dev.device_ptr(&stream);
                unsafe {
                    candle_kernels::simple::moe_scatter::run_moe_gather(
                        moe_dtype,
                        out_ptr as *mut std::ffi::c_void,
                        src_ptr as *const std::ffi::c_void,
                        ids_ptr as *const u32,
                        total_rows,
                        hidden_dim,
                    );
                }
            }
            CudaStorage::wrap_cuda_slice(out_slice, device.clone())
        }};
    }

    let out_storage = match &xs_cuda.slice {
        CudaStorageSlice::BF16(s) => dispatch_gather!(s, half::bf16),
        CudaStorageSlice::F16(s) => dispatch_gather!(s, half::f16),
        CudaStorageSlice::F32(s) => dispatch_gather!(s, f32),
        _ => crate::bail!("fused_moe_gather: unsupported dtype {dtype:?}"),
    };

    let out_shape: Shape = vec![total_rows, hidden_dim].into();
    Ok(crate::tensor::from_storage(
        crate::Storage::Cuda(out_storage),
        out_shape,
        crate::op::BackpropOp::none(),
        false,
    ))
}

/// Fused MoE router: softmax + top-k select + (optional) renormalize over `logits`
/// `[num_tokens, n_experts]` in **one** kernel launch, replacing the
/// `softmax → sort(desc) → narrow(k) → renorm → flatten` op chain (≈6 launches over a tiny
/// tensor). Returns `(weights, indices)`, each `[num_tokens, k]`: f32 routing weights and u32
/// expert ids in descending-logit order. With `norm_topk` the selected top-k softmax weights are
/// renormalized (which cancels the global softmax denominator exactly, so the kernel never
/// computes the 128-wide softmax); without it they are the plain full-softmax values.
pub fn moe_route<'w>(
    logits: &LiveTensor<'w>,
    k: usize,
    norm_topk: bool,
) -> Result<(LiveTensor<'w>, LiveTensor<'w>)> {
    use crate::cuda_backend::CudaStorageSlice;

    let (num_tokens, n_experts) = logits.dims2()?;
    if k == 0 || k > 16 {
        crate::bail!("moe_route: k={k} must be in 1..=16 (kernel top-k register bound)");
    }
    if k > n_experts {
        crate::bail!("moe_route: k={k} exceeds n_experts={n_experts}");
    }
    if n_experts > 256 {
        crate::bail!("moe_route: n_experts={n_experts} exceeds 256 (warp slot bound)");
    }
    let device = match logits.device() {
        crate::Device::Cuda(d) => d.clone(),
        _ => crate::bail!("moe_route: expected a CUDA tensor"),
    };
    let dtype = logits.dtype();
    let moe_dtype = dtype_to_moe_scatter_dtype(dtype)?;

    let (storage, layout) = logits.storage_and_layout();
    let (o1, o2) = layout
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::RequiresContiguous { op: "moe_route" }.bt())?;
    let cuda = match &*storage {
        crate::Storage::Cuda(c) => c,
        _ => crate::bail!("moe_route: expected CUDA storage"),
    };

    let stream = device.cuda_stream();
    // The routing pair is read by the bucketize kernel inside the same FFN
    // scope as the logits it was derived from, so it belongs beside them.
    let inherit = cuda.backing;
    let (out_idx, idx_backing) =
        unsafe { alloc_inheriting::<u32>(&device, num_tokens * k, inherit)? };
    let (out_w, w_backing) = unsafe { alloc_inheriting::<f32>(&device, num_tokens * k, inherit)? };

    macro_rules! launch {
        ($s:expr) => {{
            let v = $s.slice(o1..o2);
            let (lp, _lg) = v.device_ptr(&stream);
            let (ip, _ig) = out_idx.device_ptr(&stream);
            let (wp, _wg) = out_w.device_ptr(&stream);
            unsafe {
                candle_kernels::simple::moe_scatter::run_moe_route(
                    moe_dtype,
                    lp as *const std::ffi::c_void,
                    ip as *mut u32,
                    wp as *mut f32,
                    num_tokens as i32,
                    n_experts as i32,
                    k as i32,
                    norm_topk as i32,
                );
            }
        }};
    }
    match &cuda.slice {
        CudaStorageSlice::F32(s) => launch!(s),
        CudaStorageSlice::F16(s) => launch!(s),
        CudaStorageSlice::BF16(s) => launch!(s),
        _ => crate::bail!("moe_route: unsupported logits dtype {dtype:?}"),
    }

    let shape: Shape = vec![num_tokens, k].into();
    let weights = crate::tensor::from_storage(
        crate::Storage::Cuda(CudaStorage::wrap_cuda_slice_backed(
            out_w,
            device.clone(),
            w_backing,
        )),
        shape.clone(),
        crate::op::BackpropOp::none(),
        false,
    );
    let indices = crate::tensor::from_storage(
        crate::Storage::Cuda(CudaStorage::wrap_cuda_slice_backed(
            out_idx,
            device.clone(),
            idx_backing,
        )),
        shape,
        crate::op::BackpropOp::none(),
        false,
    );
    Ok((weights, indices))
}

/// B3: gather pre-quantized q8a1024 activations by token id into a stacked q8a1024 operand the
/// experts consume directly. The q8a1024 layout is token-contiguous (`hidden % 1024 == 0`), so
/// this is a byte-row copy of each token's `hidden/1024 · 1152` bytes — no gather-then-quantize.
/// Mirrors [`fused_moe_gather`] for the int8 path; pairs with [`rms_norm_q8a128`]-fused ln2.
pub fn fused_moe_gather_q8a128<'w>(
    xs_q8: &Q8a128Operand<'_>,
    ids_dev: &CudaSlice<u32>,
    total_rows: usize,
    device: &CudaDevice,
    origin: Backing,
) -> Result<Q8a128Operand<'w>> {
    let hidden = xs_q8.cols;
    if !hidden.is_multiple_of(1024) {
        crate::bail!(
            "fused_moe_gather_q8a128: hidden={hidden} must be a multiple of 1024 (token-contiguous \
             q8a1024)"
        );
    }
    let row_bytes = (hidden / 1024) * 1152;
    let out_bytes = total_rows * row_bytes;
    let (out_ptr_planned, owned, out_backing) = resolve_u8_out(origin, device, out_bytes)?;
    {
        let stream = device.cuda_stream();
        let out_ptr = out_ptr_planned;
        let (ids_ptr, _ig) = ids_dev.device_ptr(&stream);
        xs_q8.with_device_ptr(device, |src_ptr| {
            unsafe {
                candle_kernels::simple::moe_scatter::run_moe_gather(
                    3, // u8 (q8a1024 byte-row)
                    out_ptr as *mut std::ffi::c_void,
                    src_ptr as *const std::ffi::c_void,
                    ids_ptr as *const u32,
                    total_rows,
                    row_bytes,
                );
            }
            Ok(())
        })?;
    }
    q8a128_from_out(
        owned,
        out_ptr_planned,
        out_backing,
        out_bytes,
        total_rows,
        hidden,
        device,
    )
}

/// Deterministic MoE scatter: sequential per-token reduce, no atomicAdd.
///
/// `down_out` is in expert-major order. `perm[i]` maps token-major index `i` to the
/// corresponding row in `down_out`, so no CPU-side reorder pass is required.
///
/// `token_starts` is a prefix-sum array of length `num_tokens + 1` on the GPU,
/// where `token_starts[t]` is the start index in the token-major arrays for token t.
/// Variable k per token is supported via these offsets.
///
/// `ys` is ACCUMULATED into (+=); initialize to zero before the first call.
pub fn fused_deterministic_scatter(
    ys: &LiveTensor<'_>,
    down_out: &LiveTensor<'_>,
    perm: &CudaSlice<u32>,
    weights_flat: &LiveTensor<'_>,
    reordered_weight_ids: &CudaSlice<u32>,
    token_starts: &CudaSlice<i32>,
    num_tokens: usize,
    device: &CudaDevice,
) -> Result<()> {
    use crate::cuda_backend::CudaStorageSlice;

    if num_tokens == 0 {
        return Ok(());
    }
    let (_, hidden_dim) = down_out.dims2()?;
    let dtype = down_out.dtype();
    let moe_dtype = dtype_to_moe_scatter_dtype(dtype)?;

    if !ys.layout().is_contiguous() || ys.layout().start_offset() != 0 {
        crate::bail!("fused_deterministic_scatter: ys must be contiguous with zero offset");
    }
    if !down_out.layout().is_contiguous() || down_out.layout().start_offset() != 0 {
        crate::bail!("fused_deterministic_scatter: down_out must be contiguous with zero offset");
    }

    let ys_storage_guard = ys.storage_and_layout();
    let ys_cuda = match &*ys_storage_guard.0 {
        crate::Storage::Cuda(s) => s,
        _ => crate::bail!("fused_deterministic_scatter: expected CUDA ys"),
    };

    let src_storage = down_out.storage_and_layout().0;
    let src_cuda = match &*src_storage {
        crate::Storage::Cuda(s) => s,
        _ => crate::bail!("fused_deterministic_scatter: expected CUDA down_out"),
    };

    let wf_storage = weights_flat.storage_and_layout().0;
    let wf_cuda = match &*wf_storage {
        crate::Storage::Cuda(s) => s,
        _ => crate::bail!("fused_deterministic_scatter: expected CUDA weights_flat"),
    };
    let wf_f32 = match &wf_cuda.slice {
        CudaStorageSlice::F32(s) => s,
        _ => crate::bail!("fused_deterministic_scatter: weights_flat must be F32"),
    };

    let stream = device.cuda_stream();

    macro_rules! dispatch_var_k_scatter {
        ($ys_slice:expr, $src_slice:expr) => {{
            let (ys_ptr, _yg) = $ys_slice.device_ptr(&stream);
            let (src_ptr, _sg) = $src_slice.device_ptr(&stream);
            let (perm_ptr, _pg) = perm.device_ptr(&stream);
            let (wf_ptr, _wg) = wf_f32.device_ptr(&stream);
            let (wid_ptr, _widg) = reordered_weight_ids.device_ptr(&stream);
            let (ts_ptr, _tsg) = token_starts.device_ptr(&stream);
            unsafe {
                candle_kernels::simple::moe_scatter::run_deterministic_scatter(
                    moe_dtype,
                    ys_ptr as *mut std::ffi::c_void,
                    src_ptr as *const std::ffi::c_void,
                    perm_ptr as *const u32,
                    wf_ptr as *const f32,
                    wid_ptr as *const u32,
                    ts_ptr as *const i32,
                    num_tokens as i32,
                    hidden_dim as i32,
                );
            }
        }};
    }

    match (&ys_cuda.slice, &src_cuda.slice) {
        (CudaStorageSlice::BF16(ys_s), CudaStorageSlice::BF16(src_s)) => {
            dispatch_var_k_scatter!(ys_s, src_s);
        }
        (CudaStorageSlice::F16(ys_s), CudaStorageSlice::F16(src_s)) => {
            dispatch_var_k_scatter!(ys_s, src_s);
        }
        (CudaStorageSlice::F32(ys_s), CudaStorageSlice::F32(src_s)) => {
            dispatch_var_k_scatter!(ys_s, src_s);
        }
        _ => crate::bail!(
            "fused_deterministic_scatter: dtype mismatch ys={:?} src={:?}",
            ys.dtype(),
            down_out.dtype()
        ),
    }

    Ok(())
}

/// Reusable device workspace for [`moe_bucketize`]. All output/scratch buffers
/// live in VRAM and are sized to a maximum `n_tokens × k`; one workspace is
/// allocated per forward pipeline and reused across every MoE layer, so the
/// per-layer bucketize costs a single kernel launch — no allocations, no
/// host↔device traffic.
pub struct MoeBucketizeWorkspace {
    cap_assign: usize,
    cap_starts: usize,
    /// Expert-grouped token ids (gather rows); padding rows are `!0`.
    pub tok_ids: CudaSlice<u32>,
    /// Expert-grouped `widx` into the flattened routing weights; padding `!0`.
    pub weight_ids: CudaSlice<u32>,
    /// RAW owning expert id per grouped-GEMM tile; padding tiles are expert 0.
    pub tile_expert: CudaSlice<i32>,
    /// Stacked-batch start row per tile.
    pub tile_b_start: CudaSlice<i32>,
    /// Tokens per tile; 0 marks a padding tile the grouped kernel skips.
    pub tile_b_cnt: CudaSlice<i32>,
    /// Token-major valid assignment → its expert-grouped row (scatter `perm`).
    pub perm: CudaSlice<u32>,
    /// Token-major valid assignment → its `widx` (scatter `reordered_weight_ids`).
    pub rw_ids: CudaSlice<u32>,
    /// Per-token scatter segment boundaries, `[n_tokens + 1]`.
    pub token_starts: CudaSlice<i32>,
    /// Device header `{n_active, total_valid, num_tiles, 0}` — diagnostic only;
    /// the pipeline launches at the `n_tokens × k` upper bound and never reads
    /// this on the host.
    pub header: CudaSlice<i32>,
    inv: CudaSlice<u32>,
    scan: CudaSlice<i32>,
}

impl MoeBucketizeWorkspace {
    /// Allocate for up to `max_tokens × k` routing assignments.
    pub fn new(device: &CudaDevice, max_tokens: usize, k: usize) -> Result<Self> {
        let cap_assign = max_tokens.max(1) * k.max(1);
        let cap_starts = max_tokens.max(1) + 1;
        Ok(Self {
            cap_assign,
            cap_starts,
            tok_ids: unsafe { device.alloc::<u32>(cap_assign)? },
            weight_ids: unsafe { device.alloc::<u32>(cap_assign)? },
            tile_expert: unsafe { device.alloc::<i32>(cap_assign)? },
            tile_b_start: unsafe { device.alloc::<i32>(cap_assign)? },
            tile_b_cnt: unsafe { device.alloc::<i32>(cap_assign)? },
            perm: unsafe { device.alloc::<u32>(cap_assign)? },
            rw_ids: unsafe { device.alloc::<u32>(cap_assign)? },
            token_starts: unsafe { device.alloc::<i32>(cap_starts)? },
            header: unsafe { device.alloc::<i32>(4)? },
            inv: unsafe { device.alloc::<u32>(cap_assign)? },
            scan: unsafe { device.alloc::<i32>(cap_assign)? },
        })
    }

    /// Grow (never shrink) to cover `n_tokens × k`.
    fn ensure(&mut self, device: &CudaDevice, n_tokens: usize, k: usize) -> Result<()> {
        if n_tokens * k > self.cap_assign || n_tokens + 1 > self.cap_starts {
            *self = Self::new(device, n_tokens, k)?;
        }
        Ok(())
    }
}

/// GPU-native expert bucketize: turn `moe_route`'s `[n_tokens, k]` u32 index
/// tensor into every device table the grouped expert pipeline consumes —
/// expert-grouped gather lists, grouped-GEMM tile tables (padded to the
/// `n_tokens × k` launch bound with `b_cnt = 0`), and the deterministic
/// scatter's token-major segment tables — in ONE launch on the compute stream,
/// with no GPU→CPU readback. The grouping is stable in (token, slot) order,
/// bit-identical to the CPU counting-sort it replaces; an index
/// `≥ n_experts` is the router's empty-slot sentinel and is skipped. See
/// `candle-kernels/src/simple/moe_bucketize.cu` for the padding contract.
pub fn moe_bucketize(
    indices: &LiveTensor<'_>,
    n_experts: usize,
    tile_w: usize,
    ws: &mut MoeBucketizeWorkspace,
) -> Result<()> {
    use crate::cuda_backend::CudaStorageSlice;

    use candle_kernels::simple::moe_bucketize::{MAX_EXPERTS, MAX_TOPK};

    let (n_tokens, k) = indices.dims2()?;
    // These bounds mirror the launcher's own guards (via the shared constants),
    // so an invalid call errors HERE instead of the launcher silently skipping
    // the launch and leaving the workspace holding the previous layer's tables.
    if n_tokens == 0 || k == 0 {
        crate::bail!("moe_bucketize: empty indices [{n_tokens}, {k}]");
    }
    if n_experts == 0 || n_experts > MAX_EXPERTS {
        crate::bail!("moe_bucketize: n_experts={n_experts} must be in 1..={MAX_EXPERTS}");
    }
    if k > MAX_TOPK {
        crate::bail!("moe_bucketize: k={k} exceeds {MAX_TOPK} (kernel per-token sort bound)");
    }
    if tile_w == 0 {
        crate::bail!("moe_bucketize: tile_w must be > 0");
    }
    let device = match indices.device() {
        crate::Device::Cuda(d) => d.clone(),
        _ => crate::bail!("moe_bucketize: expected a CUDA tensor"),
    };
    if indices.dtype() != crate::DType::U32 {
        crate::bail!(
            "moe_bucketize: indices must be u32, got {:?}",
            indices.dtype()
        );
    }
    ws.ensure(&device, n_tokens, k)?;

    let (storage, layout) = indices.storage_and_layout();
    let (o1, o2) = layout.contiguous_offsets().ok_or_else(|| {
        crate::Error::RequiresContiguous {
            op: "moe_bucketize",
        }
        .bt()
    })?;
    let cuda = match &*storage {
        crate::Storage::Cuda(c) => c,
        _ => crate::bail!("moe_bucketize: expected CUDA storage"),
    };
    let ids = match &cuda.slice {
        CudaStorageSlice::U32(s) => s.slice(o1..o2),
        _ => crate::bail!("moe_bucketize: expected u32 storage"),
    };

    let stream = device.cuda_stream();
    let (idp, _g0) = ids.device_ptr(&stream);
    let (tok, _g1) = ws.tok_ids.device_ptr(&stream);
    let (wid, _g2) = ws.weight_ids.device_ptr(&stream);
    let (te, _g3) = ws.tile_expert.device_ptr(&stream);
    let (tbs, _g4) = ws.tile_b_start.device_ptr(&stream);
    let (tbc, _g5) = ws.tile_b_cnt.device_ptr(&stream);
    let (pm, _g6) = ws.perm.device_ptr(&stream);
    let (rw, _g7) = ws.rw_ids.device_ptr(&stream);
    let (ts, _g8) = ws.token_starts.device_ptr(&stream);
    let (hd, _g9) = ws.header.device_ptr(&stream);
    let (iv, _g10) = ws.inv.device_ptr(&stream);
    let (sc, _g11) = ws.scan.device_ptr(&stream);
    unsafe {
        candle_kernels::simple::moe_bucketize::run_moe_bucketize(
            idp as *const std::ffi::c_void,
            n_tokens as i32,
            k as i32,
            n_experts as i32,
            tile_w as i32,
            tok as *mut std::ffi::c_void,
            wid as *mut std::ffi::c_void,
            te as *mut std::ffi::c_void,
            tbs as *mut std::ffi::c_void,
            tbc as *mut std::ffi::c_void,
            pm as *mut std::ffi::c_void,
            rw as *mut std::ffi::c_void,
            ts as *mut std::ffi::c_void,
            hd as *mut std::ffi::c_void,
            iv as *mut std::ffi::c_void,
            sc as *mut std::ffi::c_void,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "cuda_tests.rs"]
mod test;
