use super::{GgmlDType, QStorage};
use crate::backend::{BackendDevice, BackendStorage};

use crate::quantized::k_quants::GgmlType;
use crate::{CudaDevice, CudaStorage, Result, Shape};
use half::f16;

use crate::cuda_backend::WrapErr;
use cudarc::driver::{CudaSlice, CudaView, DevicePtr, DevicePtrMut};

// Import the FFI dispatcher functions
use candle_kernels::simple::quantized::{
    run_arena_compact_copy, run_arena_compact_patch, run_dequantize_block,
    run_dequantize_mul_mat_vec, run_mul_mat, run_mul_mat_vec_q8_1, run_quantize_block,
    run_quantize_palette4_convert, run_quantize_q8_1, run_quantize_transposed_batched,
    run_quantize_transposed_batched_typed, run_reduce_head_stats_format,
    run_sample_quant_errors_kv_paged, run_sample_quant_errors_paged,
    run_select_kv_format_palette4_paged, run_select_winners_kv_paged,
    run_summarize_winners_side_paged, DequantOutDType, QType,
};

// Import the new quantized matmul dispatcher
// K/128 blocks have embedded scales, no external scale extraction needed.
use candle_kernels::quantized::{
    dispatch_info, flush_l2_cache, run_quantized_matmul, VxSegment, YType,
};

// Import GEMX repacking dispatcher
use candle_kernels::quantized::{get_repacked_size_bytes, is_gemx_supported, run_repack_gemx};

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
    #[allow(dead_code)]
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
            (sys::CU_MEMHOSTREGISTER_DEVICEMAP | sys::CU_MEMHOSTREGISTER_READ_ONLY) as u32,
        )
    };
    match register_result.result() {
        Ok(_) => Some(MmapRegistration { ptr }),
        Err(_) => None,
    }
}

/// Allocate host memory that is GPU-accessible via PCIe (`cudaHostAllocMapped`).
///
/// Returns `(host_ptr, device_ptr, guard)`. The `guard` frees the memory on drop.
/// GPU kernels use `device_ptr` transparently â€” hardware handles PCIe transfers.
///
/// This is the building block for VRAM-overflow weight storage: tensors that
/// don't fit in VRAM can live in pinned host memory and still be used by CUDA
/// kernels (at PCIe bandwidth instead of VRAM bandwidth).
pub fn alloc_host_mapped(size: usize) -> Result<(*mut u8, u64, HostMappedAlloc)> {
    use cudarc::driver::sys;
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe {
        // CU_MEMHOSTALLOC_DEVICEMAP = 0x02
        sys::cuMemHostAlloc(&mut host_ptr, size, 0x02)
            .result()
            .map_err(|e| crate::Error::Msg(format!("cuMemHostAlloc failed: {:?}", e)))?;
        let mut dev_ptr: sys::CUdeviceptr = 0;
        let res = sys::cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0);
        if let Err(e) = res.result() {
            let _ = sys::cuMemFreeHost(host_ptr).result();
            crate::bail!("cuMemHostGetDevicePointer failed: {:?}", e);
        }
        let guard = HostMappedAlloc { host_ptr, size };
        Ok((host_ptr as *mut u8, dev_ptr, guard))
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

#[derive(Clone, Debug)]
pub struct QCudaStorage {
    data: PaddedCudaSlice,
    dtype: GgmlDType,
    device: CudaDevice,
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
// Arena compaction wrappers
// ============================================================================

/// CompactMove struct matching the CUDA side (24 bytes, padded).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CompactMove {
    pub dst: u64, // device pointer
    pub src: u64, // device pointer
    pub stride_bytes: u32,
    pub _pad: u32,
}

// SAFETY: CompactMove is #[repr(C)] with no padding holes and contains only
// plain-old-data types. It is safe to copy to/from GPU memory.
unsafe impl cudarc::driver::DeviceRepr for CompactMove {}
// SAFETY: CompactMove is all POD (u64, u64, u32, u32) — all-zeros is valid.
unsafe impl cudarc::driver::ValidAsZeroBits for CompactMove {}

/// Launch the arena compaction copy kernel for one bucket.
///
/// `moves` must contain `CompactMove` structs with valid device pointers.
/// Each move carries its own `stride_bytes`, so mixed formats work in one call.
/// `block_dim` controls threads per block (128 recommended).
pub fn arena_compact_copy(
    moves: &CudaSlice<CompactMove>,
    num_moves: usize,
    block_dim: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<()> {
    if num_moves == 0 {
        return Ok(());
    }
    let (moves_ptr, _guard) = moves.device_ptr(stream);
    unsafe {
        run_arena_compact_copy(
            moves_ptr as *const std::ffi::c_void,
            num_moves as i32,
            block_dim as i32,
            stream.cu_stream() as *mut _,
        );
    }
    Ok(())
}

/// Fully async arena compaction copy: pinned-alloc → async H2D → kernel launch.
///
/// Takes host-side `moves` slice, uploads via the provided [`PinnedStager`] so
/// the entire pipeline (alloc + copy + launch) is non-blocking on the host.
/// The stager manages deferred freeing of spent pinned buffers — keep it alive
/// for the duration of the batch to get full async benefits.
///
/// `block_dim` controls threads per block (128 recommended).
pub fn arena_compact_copy_async(
    moves: &[CompactMove],
    block_dim: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    stager: &super::pinned_staging::PinnedStager,
) -> Result<()> {
    let num_moves = moves.len();
    if num_moves == 0 {
        return Ok(());
    }

    // Allocate pinned host buffer, copy moves into it, then async-upload.
    let byte_len = num_moves * std::mem::size_of::<CompactMove>();
    let mut pinned = stager.alloc(byte_len)?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            moves.as_ptr() as *const u8,
            pinned.as_mut_slice().as_mut_ptr(),
            byte_len,
        );
    }
    let moves_gpu = stager.submit(pinned)?;

    // Launch kernel — reinterpret the u8 slice as CompactMove pointers.
    let moves_ptr = moves_gpu.dev_ptr();
    unsafe {
        run_arena_compact_copy(
            moves_ptr as *const std::ffi::c_void,
            num_moves as i32,
            block_dim as i32,
            stream.cu_stream() as *mut _,
        );
    }
    Ok(())
}

/// Launch the arena compaction patch kernel.
///
/// Rewrites `block_table` entries in-place: any entry matching a `src_gid`
/// is replaced with the corresponding `dst_gid`. `src_gids` must be sorted
/// ascending.
pub fn arena_compact_patch(
    block_table: &mut CudaSlice<i32>,
    num_entries: usize,
    src_gids: &CudaSlice<i32>,
    dst_gids: &CudaSlice<i32>,
    num_moves: usize,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
) -> Result<()> {
    if num_entries == 0 || num_moves == 0 {
        return Ok(());
    }
    let (bt_ptr, _bt_guard) = block_table.device_ptr_mut(stream);
    let (src_ptr, _src_guard) = src_gids.device_ptr(stream);
    let (dst_ptr, _dst_guard) = dst_gids.device_ptr(stream);
    unsafe {
        run_arena_compact_patch(
            bt_ptr as *mut i32,
            num_entries as i32,
            src_ptr as *const i32,
            dst_ptr as *const i32,
            num_moves as i32,
            stream.cu_stream() as *mut _,
        );
    }
    Ok(())
}

/// Fully async arena compaction patch: pinned-alloc → async H2D → kernel launch.
///
/// Like [`arena_compact_patch`] but uploads `src_gids` and `dst_gids` from host
/// slices via the [`PinnedStager`] so the entire pipeline is non-blocking.
/// `block_table` must already be on the GPU.
pub fn arena_compact_patch_async(
    block_table: &mut CudaSlice<i32>,
    num_entries: usize,
    src_gids: &[i32],
    dst_gids: &[i32],
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    stager: &super::pinned_staging::PinnedStager,
) -> Result<()> {
    let num_moves = src_gids.len();
    if num_entries == 0 || num_moves == 0 {
        return Ok(());
    }
    assert_eq!(src_gids.len(), dst_gids.len());

    let i32_bytes = std::mem::size_of::<i32>();

    // Upload src_gids via pinned staging
    let src_byte_len = num_moves * i32_bytes;
    let mut src_pinned = stager.alloc(src_byte_len)?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            src_gids.as_ptr() as *const u8,
            src_pinned.as_mut_slice().as_mut_ptr(),
            src_byte_len,
        );
    }
    let src_gpu = stager.submit(src_pinned)?;

    // Upload dst_gids via pinned staging
    let dst_byte_len = num_moves * i32_bytes;
    let mut dst_pinned = stager.alloc(dst_byte_len)?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            dst_gids.as_ptr() as *const u8,
            dst_pinned.as_mut_slice().as_mut_ptr(),
            dst_byte_len,
        );
    }
    let dst_gpu = stager.submit(dst_pinned)?;

    let (bt_ptr, _bt_guard) = block_table.device_ptr_mut(stream);
    let src_ptr = src_gpu.dev_ptr();
    let dst_ptr = dst_gpu.dev_ptr();
    unsafe {
        run_arena_compact_patch(
            bt_ptr as *mut i32,
            num_entries as i32,
            src_ptr as *const i32,
            dst_ptr as *const i32,
            num_moves as i32,
            stream.cu_stream() as *mut _,
        );
    }
    Ok(())
}

// ============================================================================
// Palette4 buffered conversion API
// ============================================================================

/// KvHead byte layout constants for HD=128.
const KVHEAD_HD: usize = 128;
const KVHEAD_N_PAL: usize = 4;
const KVHEAD_PAL_DIM: usize = KVHEAD_HD / KVHEAD_N_PAL; // 32
const KVHEAD_K_PAL_OFF: usize = 0; // k_pal[32]: 32 bytes
const KVHEAD_V_PAL_OFF: usize = KVHEAD_HD / 4; // v_pal[32]: offset 32
const KVHEAD_K_PTR_OFF: usize = KVHEAD_HD / 2; // k_ptr[4 × u64]: offset 64
const KVHEAD_V_PTR_OFF: usize = KVHEAD_HD / 2 + 32; // v_ptr[4 × u64]: offset 96
const KVHEAD_K_FMT_OFF: usize = KVHEAD_HD / 2 + 64; // k_fmt[4 × u8]: offset 128
const KVHEAD_V_FMT_OFF: usize = KVHEAD_HD / 2 + 68; // v_fmt[4 × u8]: offset 132
const KVHEAD_K_SCALE_OFF: usize = KVHEAD_HD / 2 + 72; // k_scale[4 × f32]: offset 136
const KVHEAD_V_SCALE_OFF: usize = KVHEAD_HD / 2 + 88; // v_scale[4 × f32]: offset 152
const KVHEAD_SIZE: usize = KVHEAD_HD / 2 + 104; // 168 bytes total — must match kv_head_byte_size<HD>() in slot_types.cuh

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
/// `k_pal_map` / `v_pal_map` are 2-bit-packed arrays: 32 bytes covering 128
/// dimensions.  Each 2-bit field selects which palette (0-3) owns that
/// dimension.  Use `identity_pal_map_128()` to get the standard identity map
/// (dims 0-31 → pal 0, …).  Src and dst may have independent pal_maps.
pub struct PalHeadDesc {
    /// Raw device pointers to source K arena data for palettes 0-3.
    pub k_src_arena_ptrs: [u64; KVHEAD_N_PAL],
    /// Raw device pointers to source V arena data for palettes 0-3.
    pub v_src_arena_ptrs: [u64; KVHEAD_N_PAL],
    /// GgmlDType format of each source K palette arena.
    pub k_src_fmts: [GgmlDType; KVHEAD_N_PAL],
    /// GgmlDType format of each source V palette arena.
    pub v_src_fmts: [GgmlDType; KVHEAD_N_PAL],
    /// 2-bit-packed source K palette assignment map (32 bytes).
    pub k_src_pal_map: [u8; KVHEAD_HD / 4],
    /// 2-bit-packed source V palette assignment map (32 bytes).
    pub v_src_pal_map: [u8; KVHEAD_HD / 4],
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
    /// 2-bit-packed destination K palette assignment map (32 bytes).
    pub k_dst_pal_map: [u8; KVHEAD_HD / 4],
    /// 2-bit-packed destination V palette assignment map (32 bytes).
    pub v_dst_pal_map: [u8; KVHEAD_HD / 4],
    /// Post-dequant scale written into the dst KvHead for K (f32, default 1.0).
    /// The decode kernel multiplies dequantized K values by this scale per palette.
    pub k_dst_scales: [f32; KVHEAD_N_PAL],
    /// Post-dequant scale written into the dst KvHead for V (f32, default 1.0).
    pub v_dst_scales: [f32; KVHEAD_N_PAL],
}

/// Build the identity 2-bit-packed palette map for HD=128.
pub fn identity_pal_map_128() -> [u8; KVHEAD_HD / 4] {
    let mut out = [0u8; KVHEAD_HD / 4];
    for d in 0..KVHEAD_HD {
        let p = (d / KVHEAD_PAL_DIM) as u8;
        out[d / 4] |= (p & 0x3) << (2 * (d % 4));
    }
    out
}

/// Build a balanced pseudo-random 2-bit-packed palette map for HD=128.
///
/// Assigns each of the 128 dims to one of 4 palettes using a Fisher-Yates
/// shuffle. Every palette is assigned exactly 32 dims but the assignment is
/// non-contiguous and pseudo-random. The caller supplies a seed/IV so different
/// randomization events can generate different maps while remaining reproducible.
pub fn shuffled_pal_map_128(seed: u64) -> [u8; KVHEAD_HD / 4] {
    // Start with exactly 32 dims per palette (sequential assignment), then
    // shuffle with a caller-provided seed so the palette sizes stay balanced
    // while the dim routing varies between randomization events.
    let mut assign = [0u8; KVHEAD_HD];
    for d in 0..KVHEAD_HD {
        assign[d] = (d / KVHEAD_PAL_DIM) as u8;
    }

    // Fisher-Yates with Knuth multiplicative LCG.
    let mut rng: u64 = seed ^ 0x9e3779b97f4a7c15u64;
    if rng == 0 {
        rng = 0x9e3779b97f4a7c15u64;
    }
    for i in (1..KVHEAD_HD).rev() {
        rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (rng >> 33) as usize % (i + 1);
        assign.swap(i, j);
    }

    // Pack: 4 dims per byte, 2 bits per dim in little-endian order.
    let mut out = [0u8; KVHEAD_HD / 4];
    for d in 0..KVHEAD_HD {
        out[d / 4] |= (assign[d] & 0x3) << (2 * (d % 4));
    }
    out
}

/// Serialize arena pointers, formats, pal_map, and scales into a 168-byte KvHead struct.
pub fn build_kvhead_bytes_raw(
    k_arena_ptrs: &[u64; KVHEAD_N_PAL],
    v_arena_ptrs: &[u64; KVHEAD_N_PAL],
    k_fmts: &[GgmlDType; KVHEAD_N_PAL],
    v_fmts: &[GgmlDType; KVHEAD_N_PAL],
    k_pal_map: &[u8; KVHEAD_HD / 4],
    v_pal_map: &[u8; KVHEAD_HD / 4],
    k_scales: &[f32; KVHEAD_N_PAL],
    v_scales: &[f32; KVHEAD_N_PAL],
) -> Result<Vec<u8>> {
    let mut head = vec![0u8; KVHEAD_SIZE];
    head[KVHEAD_K_PAL_OFF..KVHEAD_K_PAL_OFF + KVHEAD_HD / 4].copy_from_slice(k_pal_map);
    head[KVHEAD_V_PAL_OFF..KVHEAD_V_PAL_OFF + KVHEAD_HD / 4].copy_from_slice(v_pal_map);
    for p in 0..KVHEAD_N_PAL {
        let k_ptr_off = KVHEAD_K_PTR_OFF + p * 8;
        head[k_ptr_off..k_ptr_off + 8].copy_from_slice(&k_arena_ptrs[p].to_le_bytes());
        let v_ptr_off = KVHEAD_V_PTR_OFF + p * 8;
        head[v_ptr_off..v_ptr_off + 8].copy_from_slice(&v_arena_ptrs[p].to_le_bytes());
        head[KVHEAD_K_FMT_OFF + p] = ggml_dtype_to_arena_fmt_code(k_fmts[p])?;
        head[KVHEAD_V_FMT_OFF + p] = ggml_dtype_to_arena_fmt_code(v_fmts[p])?;
        let k_f32 = k_scales[p].to_le_bytes();
        let v_f32 = v_scales[p].to_le_bytes();
        head[KVHEAD_K_SCALE_OFF + p * 4..KVHEAD_K_SCALE_OFF + p * 4 + 4].copy_from_slice(&k_f32);
        head[KVHEAD_V_SCALE_OFF + p * 4..KVHEAD_V_SCALE_OFF + p * 4 + 4].copy_from_slice(&v_f32);
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
                          pal_map: &[u8; KVHEAD_HD / 4],
                          side: &str,
                          kv: &str|
         -> Result<()> {
            let mut used = [false; KVHEAD_N_PAL];
            for d in 0..KVHEAD_HD {
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
    let dst_heads_off: usize = n * KVHEAD_SIZE;
    // Layout: [src KvHeads][dst KvHeads]. Per-palette outer scales live
    // inside each dst KvHead struct (f32 at HD/2+72 / HD/2+88), so the encoder
    // (multiply by outer) and decoder (divide by outer) share a single source
    // of truth.
    let total_bytes = 2 * n * KVHEAD_SIZE;

    // Build the CPU image directly in pinned memory (no intermediate Vec).
    let mut buf = generation.alloc(total_bytes)?;

    for (i, desc) in descs.iter().enumerate() {
        let src_bytes = build_kvhead_bytes_raw(
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
            &desc.k_dst_arena_ptrs,
            &desc.v_dst_arena_ptrs,
            &desc.k_dst_fmts,
            &desc.v_dst_fmts,
            &desc.k_dst_pal_map,
            &desc.v_dst_pal_map,
            &desc.k_dst_scales,
            &desc.v_dst_scales,
        )?;

        let src_off = src_heads_off + i * KVHEAD_SIZE;
        let dst_off = dst_heads_off + i * KVHEAD_SIZE;
        buf[src_off..src_off + KVHEAD_SIZE].copy_from_slice(&src_bytes);
        buf[dst_off..dst_off + KVHEAD_SIZE].copy_from_slice(&dst_bytes);
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
            KVHEAD_HD as i32,
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
            KVHEAD_HD as i32,
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
/// `total_heads * 2` i64 entries (interleaved K/V GIDs).
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
    let n_chunks = head_gids.len() / (n_kv_head * 2);
    let total_heads = n_chunks * n_kv_head;
    let total_blocks = total_heads * blocks_per_head;

    if total_blocks == 0 {
        let empty = dev.alloc_zeros::<i32>(0)?;
        return Ok((empty, dev.alloc_zeros::<i32>(0)?));
    }

    let gids_gpu = dev.memcpy_stod(head_gids)?;

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
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
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
    let n_chunks = head_gids.len() / (n_kv_head * 2);
    let total_heads = n_chunks * n_kv_head;

    if total_heads == 0 || n_kv_head == 0 {
        let empty = dev.alloc_zeros::<i32>(0)?;
        return Ok((empty, dev.alloc_zeros::<i32>(0)?));
    }

    let gids_gpu = dev.memcpy_stod(head_gids)?;

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
#[allow(clippy::too_many_arguments)]
/// Like [`select_and_summarize_kv_winners_paged_staged`] but accepts pre-allocated
/// scratch buffers to eliminate per-call device allocation overhead, and performs
/// an **asynchronous** DtoH copy on a dedicated `dtoh_stream` into a caller-supplied
/// pinned host buffer.
///
/// * `k_winners_scratch` — device buffer of at least `n_k_thresholds × n_cells` bytes.
/// * `v_winners_scratch` — device buffer of at least `n_v_thresholds × n_cells` bytes.
/// * `kv_sums`           — device buffer of exactly `(n_k_thresholds + n_v_thresholds) × 3`
///                         `f32` values. **Must be zeroed by the caller** (the kernel uses
///                         `atomicAdd`). The K sums occupy the first `n_k_thresholds × 3`
///                         elements and the V sums the remainder.
/// * `dtoh_stream`       — secondary stream used exclusively for the DtoH transfer.
///                         The function records an event on the compute stream, makes
///                         `dtoh_stream` GPU-wait on it, then enqueues the DMA.
/// * `pinned_dst`        — pre-allocated pinned host slice of length
///                         `(n_k_thresholds + n_v_thresholds) × 3`.  The DMA is enqueued
///                         but **not waited on** — the caller must `synchronize()` the
///                         returned `CudaEvent` before reading the data.
///
/// Returns a `CudaEvent` recorded on `dtoh_stream` after the DMA.  Call
/// `event.synchronize()` to block until the data is available in `pinned_dst`.
#[allow(clippy::too_many_arguments)]
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

fn mul_mat_vec_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
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
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    quantize_q8_1(y, &mut y_q8_1, ncols, b_size, dev)?;

    let qtype = dtype_to_qtype(dtype)?;
    let dst = unsafe { dev.alloc::<f32>(nrows * b_size)? };
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
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

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
    let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    quantize_q8_1(y, &mut y_q8_1, k, y_cols, dev)?;

    let qtype = dtype_to_qtype(dtype)?;
    let dst = unsafe { dev.alloc::<f32>(x_rows * y_cols)? };
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
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

impl QCudaStorage {
    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes)?;
        Ok(QCudaStorage {
            data: PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            },
            device: device.clone(),
            dtype,
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

        // Wrap the pinned host buffer as a CudaSlice via device H2D copy.
        // Note: we still need a CudaSlice for the QCudaStorage API.
        // We allocate VRAM and copy â€” but the data also lives in pinned host
        // memory for future zero-copy patterns.
        //
        // TODO(perf): When cudarc supports wrapping an external device pointer
        // (from cuMemHostGetDevicePointer) into a CudaSlice without ownership,
        // we can eliminate this VRAM copy entirely. For now, the HostMappedAlloc
        // guard keeps the pinned buffer alive as a correctness guarantee.
        let pinned_slice = unsafe { std::slice::from_raw_parts(host_ptr, size_in_bytes) };
        let mut inner = unsafe { device.alloc::<u8>(padded_size_in_bytes)? };
        device.memcpy_htod(pinned_slice, &mut inner.slice_mut(..size_in_bytes))?;

        let storage = QCudaStorage {
            data: PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            },
            device: device.clone(),
            dtype,
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
        self.data.inner.slice_mut(..self.data.len)
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
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
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
        use crate::cuda_backend::CudaStorageSlice;
        use half::bf16;

        let (nrows, ncols) = self_shape.dims2()?;
        let (batch_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!(
                "unexpected rhs shape in quantized_matmul {:?}",
                rhs_l.shape()
            ),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let input_dtype = rhs.dtype();

        // Check if we need to convert input to BF16
        // F32 is now supported for Q4_K via dedicated kernel
        let needs_conversion = !matches!(
            input_dtype,
            crate::DType::F16 | crate::DType::BF16 | crate::DType::F32
        );

        // If input is not a supported type, convert to BF16
        let rhs_converted: Option<CudaStorage> = if needs_conversion {
            Some(rhs.to_dtype(rhs_l, crate::DType::BF16)?)
        } else {
            None
        };

        // Get the actual storage and layout to use
        let rhs_storage = rhs_converted.as_ref().unwrap_or(rhs);
        let rhs_layout_owned;
        let rhs_layout = if rhs_converted.is_some() {
            // Converted storage is contiguous
            rhs_layout_owned = crate::Layout::contiguous(rhs_l.shape());
            &rhs_layout_owned
        } else {
            rhs_l
        };

        // Convert GgmlDType to qtype
        let qtype = dtype_to_qtype(self.dtype)? as i32;

        // Determine Y type
        let ytype = match &rhs_storage.slice {
            CudaStorageSlice::F16(_) => YType::F16,
            CudaStorageSlice::BF16(_) => YType::BF16,
            CudaStorageSlice::F32(_) => YType::F32,
            _ => unreachable!("should have been converted to BF16"),
        };

        // NOTE: GEMX tensor core kernels are integrated into run_quantized_matmul
        // The dispatcher automatically uses tensor cores for batch >= 16 on SM80+ with F16

        // Allocate output based on ytype - enum to hold different slice types
        enum OutputSlice {
            F16(CudaSlice<f16>),
            BF16(CudaSlice<bf16>),
            F32(CudaSlice<f32>),
        }

        let dst_slice = match ytype {
            YType::F16 => {
                OutputSlice::F16(unsafe { self.device.alloc::<f16>(nrows * batch_size)? })
            }
            YType::BF16 => {
                OutputSlice::BF16(unsafe { self.device.alloc::<bf16>(nrows * batch_size)? })
            }
            YType::F32 => {
                OutputSlice::F32(unsafe { self.device.alloc::<f32>(nrows * batch_size)? })
            }
        };

        // Run kernel in a block so all guards drop before we wrap the output
        {
            let stream = self.device.cuda_stream();

            // Get data pointer (quantized weights with embedded scales in K/128 format)
            let (data_ptr, _data_guard) = self.data.inner.device_ptr(&stream);

            // Build single segment descriptor (non-MoE: one segment, all batches)
            let segment = VxSegment {
                weights: data_ptr as *const std::ffi::c_void,
                batch_count: batch_size as i32,
            };

            // Helper macro to run the matmul with the correct Y/output pointers
            // This avoids code duplication across all the y_type match arms
            // NOTE: K/128 blocks have embedded scales, no external scales needed
            macro_rules! run_matmul {
                ($y_ptr:expr, $dst_ptr:expr) => {
                    unsafe {
                        run_quantized_matmul(
                            &segment as *const VxSegment,
                            1, // num_segments = 1 (non-MoE)
                            $y_ptr as *const std::ffi::c_void,
                            $dst_ptr as *mut std::ffi::c_void,
                            ncols as i32,
                            nrows as i32,
                            k as i32,
                            nrows as i32,
                            qtype,
                            ytype as i32,
                            self.data.len,
                        );
                    }
                };
            }

            // Get Y pointer based on type
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
        } // All guards dropped here, kernel is synchronized

        // Wrap output - guards have been dropped so we can move the slice
        let out_storage = match dst_slice {
            OutputSlice::F16(dst) => CudaStorage::wrap_cuda_slice(dst, self.device.clone()),
            OutputSlice::BF16(dst) => CudaStorage::wrap_cuda_slice(dst, self.device.clone()),
            OutputSlice::F32(dst) => CudaStorage::wrap_cuda_slice(dst, self.device.clone()),
        };

        // Build output shape
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        let out_shape: crate::Shape = out_shape.into();

        // Convert output back to input dtype if we converted the input
        if needs_conversion {
            let out_layout = crate::Layout::contiguous(&out_shape);
            let converted_out = out_storage.to_dtype(&out_layout, input_dtype)?;
            Ok((converted_out, out_shape))
        } else {
            Ok((out_storage, out_shape))
        }
    }

    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
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
            data: PaddedCudaSlice {
                inner: dst_data,
                len: new_size,
            },
            dtype: self.dtype,
            device: self.device.clone(),
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
        data: PaddedCudaSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
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
        data: PaddedCudaSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
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
        data: PaddedCudaSlice { inner, len },
        device: device.clone(),
        dtype,
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
        data: PaddedCudaSlice { inner, len },
        device: device.clone(),
        dtype,
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
    use crate::cuda_backend::CudaStorageSlice;
    use half::bf16;

    let num_experts = weight_ptrs.len();
    if num_experts == 0 {
        crate::bail!("grouped_matmul_gemx: no experts provided");
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

    // Allocate output
    enum OutputSlice {
        F16(CudaSlice<f16>),
        BF16(CudaSlice<bf16>),
        F32(CudaSlice<f32>),
    }

    let dst_slice = match ytype {
        YType::F16 => OutputSlice::F16(unsafe { device.alloc::<f16>(nrows * total_batch)? }),
        YType::BF16 => OutputSlice::BF16(unsafe { device.alloc::<bf16>(nrows * total_batch)? }),
        YType::F32 => OutputSlice::F32(unsafe { device.alloc::<f32>(nrows * total_batch)? }),
    };

    // Segmented dispatch: build VxSegment array (one per expert), single call.
    // No device table allocation, no memcpy â€” segment descriptors are host-side.
    {
        let stream = device.cuda_stream();

        // Build VxSegment array: one entry per expert
        let segments: Vec<VxSegment> = (0..num_experts)
            .map(|e| {
                let expert_batch = expert_offsets[e + 1] - expert_offsets[e];
                VxSegment {
                    weights: weight_ptrs[e] as *const std::ffi::c_void,
                    batch_count: expert_batch,
                }
            })
            .collect();

        // Helper macro to extract pointers and call run_quantized_matmul
        macro_rules! dispatch_segments {
            ($y_data:expr, $dst:expr) => {{
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
                let (dst_ptr, _dst_guard) = $dst.device_ptr(&stream);
                unsafe {
                    run_quantized_matmul(
                        segments.as_ptr(),
                        num_experts as i32,
                        y_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        ncols as i32,
                        nrows as i32,
                        k as i32,
                        nrows as i32,
                        qtype,
                        ytype as i32,
                        0, // weight_bytes: 0 â†’ assume L2-cached (small expert weights)
                    );
                }
            }};
        }

        match (&activations.slice, &dst_slice) {
            (CudaStorageSlice::F16(y_data), OutputSlice::F16(dst)) => {
                dispatch_segments!(y_data, dst);
            }
            (CudaStorageSlice::BF16(y_data), OutputSlice::BF16(dst)) => {
                dispatch_segments!(y_data, dst);
            }
            (CudaStorageSlice::F32(y_data), OutputSlice::F32(dst)) => {
                dispatch_segments!(y_data, dst);
            }
            _ => unreachable!("ytype and activation slice should match"),
        }
    }

    // Wrap output
    let out_storage = match dst_slice {
        OutputSlice::F16(dst) => CudaStorage::wrap_cuda_slice(dst, device.clone()),
        OutputSlice::BF16(dst) => CudaStorage::wrap_cuda_slice(dst, device.clone()),
        OutputSlice::F32(dst) => CudaStorage::wrap_cuda_slice(dst, device.clone()),
    };

    let out_shape: Shape = vec![total_batch, nrows].into();
    Ok(crate::tensor::from_storage(
        crate::Storage::Cuda(out_storage),
        out_shape,
        crate::op::BackpropOp::none(),
        false,
    ))
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
    ys: &crate::Tensor,
    down_out: &crate::Tensor,
    perm: &CudaSlice<u32>,
    weights_flat: &crate::Tensor,
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

#[cfg(test)]
#[path = "cuda_tests.rs"]
mod test;
