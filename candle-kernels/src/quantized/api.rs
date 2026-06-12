//! FFI bindings for quantized batched matmul kernels
//!
//! Dispatcher for quantized matrix-vector multiplication kernels.
//! Selects the appropriate kernel based on quantization type, Y type, and tensor core usage.

use core::ffi::c_void;

/// Segment descriptor for segmented dispatch.
/// One per expert (MoE) or one total (non-MoE).
/// Matches C-side `vx_segment_t` in dispatcher.cu.
#[repr(C)]
pub struct VxSegment {
    /// Device pointer to quantized weight data
    pub weights: *const c_void,
    /// Number of batches in this segment (greedy decomposition boundary)
    pub batch_count: i32,
}

// Safety: VxSegment contains a raw pointer that is only dereferenced on the GPU side.
// It is constructed on the host and passed to CUDA kernels via FFI.
unsafe impl Send for VxSegment {}
unsafe impl Sync for VxSegment {}

/// Quantization type enum for the matmul dispatcher (`run_quantized_matmul`).
///
/// Integer values MUST match `GgmlDType` (candle-core) — `GgmlDType` is the
/// single source of truth for quant-format numbering across this whole
/// workspace. The CUDA `dispatcher.cu` uses its own 14-entry kernel-lookup
/// table internally, accessed via a `qtype_to_matmul_kernel_index` helper
/// (see `block_compact.cuh`) — the enum numbering no longer has to be
/// contiguous 0..13 just to index it.
///
/// Only the formats the matmul dispatcher actually has kernels for are
/// listed here; the KV-quant-only formats (Q4_KS/Q8_KS/Q0 family/etc.) are
/// intentionally absent because `run_quantized_matmul` would reject them
/// anyway.
#[repr(i32)]
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QType {
    // Values are `GgmlDType as u32` discriminants.
    Q_AWQ = 5,
    Q_AWQ_G64 = 6,
    Q8_0 = 7,
    Q8_1 = 8,
    Q8_K = 9,
    Q6_K = 11,
    Q5_0 = 12,
    Q5_1 = 13,
    Q5_K = 14,
    Q4_0 = 15,
    Q4_1 = 16,
    Q4_K = 17,
    Q3_K = 21,
    Q2_K = 24,
}

/// Y vector type enum (matches dispatcher ytype parameter).
/// MUST match C++ dispatcher ordering: 0=F16, 1=BF16, 2=F32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum YType {
    F16 = 0,
    BF16 = 1,
    F32 = 2,
}

extern "C" {
    /// Dispatches to the appropriate quantized matmul kernel.
    ///
    /// # Parameters
    /// - `segments`: Host array of segment descriptors (one per expert or single for non-MoE)
    /// - `num_segments`: Length of segments array
    /// - `vy`: Y vector (activations), type determined by `ytype`
    /// - `dst`: Output buffer, same type as Y
    /// - `ncols_x`: Number of columns in X (input features)
    /// - `nrows_x`: Number of rows in X (output features)
    /// - `nrows_y`: Number of rows in Y
    /// - `nrows_dst`: Number of rows in output
    /// - `qtype`: Quantization type (0-9, see QType enum)
    /// - `ytype`: Y vector type (0-2, see YType enum). Note: F32 (2) only for Q4_K.
    /// - `weight_bytes`: Weight tensor size in bytes (for L2 cache dispatch decision)
    pub fn run_quantized_matmul(
        segments: *const VxSegment,
        num_segments: i32,
        vy: *const c_void,
        dst: *mut c_void,
        ncols_x: i32,
        nrows_x: i32,
        nrows_y: i32,
        nrows_dst: i32,
        qtype: i32,
        ytype: i32,
        weight_bytes: usize,
    );

    /// Repack quantized weights to GEMX format (K/128 with embedded scales).
    ///
    /// This removes scale data from the weights (scales should be extracted
    /// separately via extract_scales before calling this) and reorders the
    /// quant bytes for optimal tensor core access patterns.
    ///
    /// # Parameters
    /// - `data`: Weight tensor data (device pointer, modified in-place)
    /// - `nrows`: Number of rows in tensor
    /// - `ncols`: Number of columns in tensor
    /// - `qtype`: Quantization type (0-9, see QType enum)
    ///
    /// # Returns
    /// New size in bytes of the repacked data, or -1 on error
    ///
    /// # Safety
    /// - src_data must be a valid device pointer to the source quantized data
    /// - dst_data must be a valid device pointer with at least get_repacked_size_bytes() bytes
    /// - Returns 0 on success, -1 on error
    #[link_name = "run_repack_gemx"]
    pub fn run_repack_gemx(
        src_data: *const core::ffi::c_void,
        dst_data: *mut core::ffi::c_void,
        nrows: i32,
        ncols: i32,
        qtype: i32,
    ) -> i32;

    /// Get the size of repacked weights without actually repacking.
    ///
    /// # Parameters
    /// - `nrows`: Number of rows in tensor
    /// - `ncols`: Number of columns in tensor
    /// - `qtype`: Quantization type (0-9)
    ///
    /// # Returns
    /// Size in bytes of repacked data, or -1 if format not supported
    pub fn get_repacked_size_bytes(nrows: i32, ncols: i32, qtype: i32) -> i64;

    /// Check if a quantization type supports GEMX repacking.
    ///
    /// # Parameters
    /// - `qtype`: Quantization type (0-9)
    ///
    /// # Returns
    /// 1 if supported, 0 if not
    #[link_name = "is_gemx_supported"]
    pub fn is_gemx_supported(qtype: i32) -> i32;

    /// Dequantize repacked quantized tensor to float32.
    ///
    /// Uses the same element mapping as the matmul loader, allowing direct
    /// debugging of the loader's element indexing without Y multiplication.
    ///
    /// # Parameters
    /// - `x`: Repacked quantized blocks (device pointer)
    /// - `scales`: External scales (device pointer, format depends on qtype)
    /// - `out`: Output float32 buffer (device pointer, nrows × ncols)
    /// - `nrows`: Number of rows
    /// - `ncols`: Number of columns (must be multiple of block size)
    /// - `qtype`: Quantization type (0-9, see QType enum)
    ///
    /// Note: K/128 blocks have embedded scales - no external scales parameter.
    ///
    /// # Returns
    /// 0 on success, -1 on error
    pub fn run_dequantize(
        x: *const c_void,
        out: *mut c_void,
        nrows: i32,
        ncols: i32,
        qtype: i32,
    ) -> i32;

    /// Get the output size (in floats) for dequantizing a tensor.
    ///
    /// # Parameters
    /// - `nrows`: Number of rows
    /// - `ncols`: Number of columns
    ///
    /// # Returns
    /// Number of float32 output elements (nrows × ncols)
    pub fn get_dequantize_output_size(nrows: i32, ncols: i32) -> i64;

    /// Get dispatch info string describing which kernels will be used.
    ///
    /// Returns a string describing the kernel dispatch plan for a given
    /// batch size and weight tensor size. Useful for benchmarking and debugging.
    ///
    /// # Parameters
    /// - `batch_size`: Number of vectors to process
    /// - `weight_bytes`: Weight tensor size in bytes (determines L2 vs DRAM path)
    /// - `buffer`: Output buffer to write the kernel description (C string)
    /// - `buffer_len`: Size of output buffer (recommend 64+ bytes)
    ///
    /// # Returns
    /// Number of characters written (excluding null terminator), or -1 on error
    ///
    /// # Examples
    /// - "s2i8(16)" - single s2_iter8 kernel for batch 16
    /// - "s2i4(8)+s3(3)" - s2_iter4 for 8 batches + s3 for 3 batches
    /// - "tc32(32)+s8(8)" - tensor core for 32 + s8 for remainder
    pub fn get_dispatch_info(
        batch_size: i32,
        weight_bytes: usize,
        buffer: *mut i8,
        buffer_len: i32,
    ) -> i32;

    /// Flush L2 cache by reading through a buffer larger than L2.
    ///
    /// This is useful for benchmarking to simulate realistic cache conditions
    /// where different matrices alternate and cannot all fit in L2 cache.
    ///
    /// # Parameters
    /// - `buffer`: Pre-allocated device buffer (should be >= 2x L2 cache size)
    /// - `size`: Size of buffer in bytes
    ///
    /// # Safety
    /// - buffer must be a valid device pointer
    /// - Synchronizes the device before returning
    pub fn flush_l2_cache(buffer: *const c_void, size: usize);
}

// =============================================================================
// GEMX TENSOR CORE SUPPORT
// =============================================================================
// GEMX tensor core kernels are dispatched through run_quantized_matmul
// when USE_TC=true in the kernel instantiation. The dispatcher automatically
// uses tensor cores for batch >= 16 on SM80+ when F16 activations are used.
//
// The following utility functions remain for workspace allocation checks.
// =============================================================================

// NOTE: GEMX kernel is integrated into the standard quantized_matmul dispatch path.
// Use run_quantized_matmul with GEMX-repacked weights (K/128 with embedded scales).

// =============================================================================
// SAFE RUST WRAPPERS
// =============================================================================

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
/// A String describing the dispatch plan, or an error message on failure.
pub fn dispatch_info(batch_size: i32, weight_bytes: usize) -> String {
    let mut buffer = [0i8; 128];
    let result = unsafe {
        get_dispatch_info(
            batch_size,
            weight_bytes,
            buffer.as_mut_ptr(),
            buffer.len() as i32,
        )
    };
    if result < 0 {
        return "error".to_string();
    }
    // Convert C string to Rust String
    let len = result as usize;
    let bytes: Vec<u8> = buffer[..len].iter().map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).to_string()
}

#[cfg(test)]
mod matmul_qtype_lock_tests {
    //! Pin the exact integer value for every `QType` variant in this file.
    //! Values must match `GgmlDType` in candle-core. Any drift will also
    //! break the C++ `QTYPE_*` lock in `block_compact.cuh` (which uses the
    //! same values) and the `qtype_to_matmul_kernel_index` mapping in
    //! `block_compact.cuh`.
    use super::QType;

    #[test]
    fn matmul_qtype_values_are_stable() {
        assert_eq!(QType::Q_AWQ as i32, 5);
        assert_eq!(QType::Q_AWQ_G64 as i32, 6);
        assert_eq!(QType::Q8_0 as i32, 7);
        assert_eq!(QType::Q8_1 as i32, 8);
        assert_eq!(QType::Q8_K as i32, 9);
        assert_eq!(QType::Q6_K as i32, 11);
        assert_eq!(QType::Q5_0 as i32, 12);
        assert_eq!(QType::Q5_1 as i32, 13);
        assert_eq!(QType::Q5_K as i32, 14);
        assert_eq!(QType::Q4_0 as i32, 15);
        assert_eq!(QType::Q4_1 as i32, 16);
        assert_eq!(QType::Q4_K as i32, 17);
        assert_eq!(QType::Q3_K as i32, 21);
        assert_eq!(QType::Q2_K as i32, 24);
    }
}
