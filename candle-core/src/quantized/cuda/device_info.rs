//! Reports the device properties the matmul heuristics and the VRAM budget read.
//!
//! The tiling and grid-order choices consult the SM count and L2 size on every
//! matmul, and those are fixed for the device, so each is looked up once per
//! process and remembered. A failed query reads as zero and degrades to the safe
//! default. Free and total VRAM are reported for the current context, and total
//! VRAM for device 0 is also available without a bound context, so startup can
//! pick a model quantization before any device exists. A benchmark helper flushes
//! L2 between matrices so timings reflect realistic cache conditions.

use super::*;

/// Process-cached SM count for the int8 dense tiling (occupancy) heuristic. SM count is a fixed
/// device property; querying the driver attribute on every matmul would add an FFI call to the hot
/// path, so memoize it once. Single-GPU assumption (this fork targets one accelerator) — a tiling
/// heuristic does not need per-device precision, and `0` (query failure) degrades safely (mode-2
/// outside the trap).
pub(super) fn cached_sm_count(device: &CudaDevice) -> usize {
    use std::sync::OnceLock;
    static SM_COUNT: OnceLock<usize> = OnceLock::new();
    *SM_COUNT.get_or_init(|| device.multiprocessor_count().unwrap_or(0))
}

/// Process-cached L2 size for the grouped grid-order choice — same memoization
/// rationale (fixed device property, hot-path caller) as [`cached_sm_count`].
/// `0` on query failure degrades safely: every activation then "exceeds" L2 and
/// the launch takes the row-fast order, which is the safe default.
pub(super) fn cached_l2_bytes(device: &CudaDevice) -> usize {
    use std::sync::OnceLock;
    static L2_BYTES: OnceLock<usize> = OnceLock::new();
    *L2_BYTES.get_or_init(|| device.l2_cache_size().unwrap_or(0))
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
/// the device's stream before returning.
pub fn cuda_flush_l2(buffer: &CudaSlice<u8>, device: &CudaDevice) {
    let stream = device.cuda_stream();
    let (ptr, _guard) = buffer.device_ptr(&stream);
    let size = buffer.len();
    unsafe {
        flush_l2_cache(
            ptr as *const core::ffi::c_void,
            size,
            stream.cu_stream() as *mut core::ffi::c_void,
        );
    }
}
