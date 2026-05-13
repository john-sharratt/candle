//! FFI bindings for sort.cu kernels
//!
//! This module provides Rust FFI declarations for CUDA argsort kernels.
//! The kernels perform bitonic sort to compute argsort indices.

use std::ffi::c_void;

/// DType enum for sort dispatcher functions
/// Matches the dtype values expected by run_argsort_asc/desc and run_sort_asc/desc
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDType {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
    U8 = 4,
    U32 = 5,
    I64 = 6,
}

extern "C" {
    // =========================================================================
    // Dispatcher functions (sort_dispatcher.cu)
    // =========================================================================

    /// Argsort ascending - returns indices that would sort the array in ascending order
    ///
    /// # Parameters
    /// - `dtype`: Data type enum (SortDType)
    /// - `x`: Input data pointer
    /// - `dst`: Output indices pointer (uint32_t)
    /// - `ncols`: Number of columns (elements per row to sort)
    /// - `ncols_pad`: Padded column count (must be power of 2 for bitonic sort)
    /// - `nrows`: Number of rows (batch dimension)
    /// - `shared_mem_size`: Shared memory size in bytes (ncols_pad * sizeof(int))
    /// - `stream`: CUDA stream
    pub fn run_argsort_asc(
        dtype: i32,
        x: *const c_void,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
        nrows: i32,
        shared_mem_size: usize,
        stream: *mut c_void,
    );

    /// Argsort descending - returns indices that would sort the array in descending order
    pub fn run_argsort_desc(
        dtype: i32,
        x: *const c_void,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
        nrows: i32,
        shared_mem_size: usize,
        stream: *mut c_void,
    );

    /// Sort ascending - currently returns indices (same as argsort)
    /// Future: may implement direct value sorting
    pub fn run_sort_asc(
        dtype: i32,
        x: *const c_void,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
        nrows: i32,
        shared_mem_size: usize,
        stream: *mut c_void,
    );

    /// Sort descending - currently returns indices (same as argsort)
    /// Future: may implement direct value sorting
    pub fn run_sort_desc(
        dtype: i32,
        x: *const c_void,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
        nrows: i32,
        shared_mem_size: usize,
        stream: *mut c_void,
    );

    // =========================================================================
    // Individual kernel functions (sort.cu)
    // =========================================================================

    // bf16 argsort kernels (CUDA_ARCH >= 800)
    pub fn asort_asc_bf16(
        x: *const c_void,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
    );
    pub fn asort_desc_bf16(
        x: *const c_void,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
    );

    // f16 argsort kernels (CUDA_ARCH >= 530)
    pub fn asort_asc_f16(
        x: *const c_void,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
    );
    pub fn asort_desc_f16(
        x: *const c_void,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
    );

    // f32 argsort kernels
    pub fn asort_asc_f32(
        x: *const f32,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
    );
    pub fn asort_desc_f32(
        x: *const f32,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
    );

    // f64 argsort kernels
    pub fn asort_asc_f64(
        x: *const f64,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
    );
    pub fn asort_desc_f64(
        x: *const f64,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
    );

    // u8 argsort kernels
    pub fn asort_asc_u8(
        x: *const u8,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
    );
    pub fn asort_desc_u8(
        x: *const u8,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
    );

    // u32 argsort kernels
    pub fn asort_asc_u32(
        x: *const u32,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
    );
    pub fn asort_desc_u32(
        x: *const u32,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
    );

    // i64 argsort kernels
    pub fn asort_asc_i64(
        x: *const i64,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
    );
    pub fn asort_desc_i64(
        x: *const i64,
        dst: *mut u32,
        ncols: i32,
        ncols_pad: i32,
    );
}
