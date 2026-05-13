//! FFI bindings for indexing.cu kernels
//!
//! This module provides Rust FFI declarations for CUDA indexing kernels.
//! The kernels perform index_select, gather, index_add, scatter_add, and scatter operations.
//!
//! Macros in the CUDA file:
//! - IS_OP: index_select (is_*)
//! - GATHER_OP: gather (gather_*)
//! - IA_OP: index_add (ia_*)
//! - SA_OP: scatter_add (sa_*)
//! - S_OP: scatter (s_*)

use std::ffi::c_void;

// =============================================================================
// Dispatcher enums and bindings
// =============================================================================

/// Index data type enum for indexing operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexDType {
    I16 = 0,
    I32 = 1,
    I64 = 2,
    U32 = 3,
    U8 = 4,
}

/// Data type enum for indexing operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexingDataDType {
    F32 = 0,
    F64 = 1,
    U8 = 2,
    U32 = 3,
    I64 = 4,
    F16 = 5,
    BF16 = 6,
    F8E4M3 = 7,
}

extern "C" {
    /// Dispatches to the appropriate index_select kernel.
    ///
    /// # Parameters
    /// - `idx_dtype`: Index data type (0=i16, 1=i32, 2=i64, 3=u32, 4=u8)
    /// - `data_dtype`: Data type (0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3)
    /// - `numel`: Number of elements in output
    /// - `num_dims`: Number of dimensions
    /// - `info`: Pointer to dims and strides array
    /// - `ids`: Index tensor
    /// - `inp`: Input tensor
    /// - `out`: Output tensor
    /// - `left_size`: Size of dimensions to the left of the indexed dimension
    /// - `src_dim_size`: Size of the source dimension being indexed
    /// - `ids_dim_size`: Size of the index dimension
    /// - `right_size`: Size of dimensions to the right of the indexed dimension
    pub fn run_index_select(
        idx_dtype: i32,
        data_dtype: i32,
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const c_void,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );

    /// Dispatches to the appropriate gather kernel.
    ///
    /// # Parameters
    /// - `idx_dtype`: Index data type (0=i16, 1=i32, 2=i64, 3=u32, 4=u8)
    /// - `data_dtype`: Data type (0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3)
    /// - `numel`: Number of elements in output
    /// - `ids`: Index tensor
    /// - `inp`: Input tensor
    /// - `out`: Output tensor
    /// - `left_size`: Size of dimensions to the left of the gathered dimension
    /// - `src_dim_size`: Size of the source dimension being gathered from
    /// - `ids_dim_size`: Size of the index dimension
    /// - `right_size`: Size of dimensions to the right of the gathered dimension
    pub fn run_gather(
        idx_dtype: i32,
        data_dtype: i32,
        numel: usize,
        ids: *const c_void,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );

    /// Dispatches to the appropriate index_add kernel.
    ///
    /// # Parameters
    /// - `idx_dtype`: Index data type (0=i16, 1=i32, 2=i64, 3=u32, 4=u8)
    /// - `data_dtype`: Data type (0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3)
    /// - `ids`: Index tensor
    /// - `ids_dim_size`: Size of the index dimension
    /// - `inp`: Input tensor (source values to add)
    /// - `out`: Output tensor (destination, modified in-place)
    /// - `left_size`: Size of dimensions to the left of the indexed dimension
    /// - `src_dim_size`: Size of the source dimension
    /// - `dst_dim_size`: Size of the destination dimension
    /// - `right_size`: Size of dimensions to the right of the indexed dimension
    pub fn run_index_add(
        idx_dtype: i32,
        data_dtype: i32,
        ids: *const c_void,
        ids_dim_size: usize,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    /// Dispatches to the appropriate scatter_add kernel.
    ///
    /// # Parameters
    /// - `idx_dtype`: Index data type (0=i16, 1=i32, 2=i64, 3=u32, 4=u8)
    /// - `data_dtype`: Data type (0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3)
    /// - `ids`: Index tensor (same shape as input)
    /// - `inp`: Input tensor (source values to scatter and add)
    /// - `out`: Output tensor (destination, modified in-place)
    /// - `left_size`: Size of dimensions to the left of the scattered dimension
    /// - `src_dim_size`: Size of the source dimension
    /// - `dst_dim_size`: Size of the destination dimension
    /// - `right_size`: Size of dimensions to the right of the scattered dimension
    pub fn run_scatter_add(
        idx_dtype: i32,
        data_dtype: i32,
        ids: *const c_void,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    /// Dispatches to the appropriate scatter kernel.
    ///
    /// # Parameters
    /// - `idx_dtype`: Index data type (0=i16, 1=i32, 2=i64, 3=u32, 4=u8)
    /// - `data_dtype`: Data type (0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16)
    ///   Note: f8_e4m3 is not supported for scatter
    /// - `ids`: Index tensor (same shape as input)
    /// - `inp`: Input tensor (source values to scatter)
    /// - `out`: Output tensor (destination, modified in-place)
    /// - `left_size`: Size of dimensions to the left of the scattered dimension
    /// - `src_dim_size`: Size of the source dimension
    /// - `dst_dim_size`: Size of the destination dimension
    /// - `right_size`: Size of dimensions to the right of the scattered dimension
    pub fn run_scatter(
        idx_dtype: i32,
        data_dtype: i32,
        ids: *const c_void,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
}

// =============================================================================
// Individual kernel bindings (original API)
// =============================================================================

extern "C" {
    // ============================================================
    // INDEX_SELECT (IS_OP) - bf16 (CUDA_ARCH >= 800)
    // ============================================================
    pub fn is_i64_bf16(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const i64,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_u32_bf16(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u32,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_u8_bf16(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u8,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // INDEX_SELECT (IS_OP) - f8_e4m3 (CUDA_ARCH >= 890)
    // ============================================================
    pub fn is_i16_f8_e4m3(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const i16,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_i32_f8_e4m3(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const i32,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_i64_f8_e4m3(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const i64,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_u32_f8_e4m3(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u32,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_u8_f8_e4m3(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u8,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // INDEX_SELECT (IS_OP) - f16 (CUDA_ARCH >= 530)
    // ============================================================
    pub fn is_i64_f16(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const i64,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_u32_f16(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u32,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_u8_f16(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u8,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // INDEX_SELECT (IS_OP) - Standard types
    // ============================================================
    // i64 index variants
    pub fn is_i64_f32(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const i64,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_i64_f64(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const i64,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_i64_u8(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const i64,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_i64_u32(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const i64,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_i64_i64(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const i64,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );

    // u32 index variants
    pub fn is_u32_f32(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u32,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_u32_f64(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u32,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_u32_u8(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u32,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_u32_i64(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u32,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_u32_u32(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u32,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );

    // u8 index variants
    pub fn is_u8_f32(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u8,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_u8_f64(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u8,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_u8_u8(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u8,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_u8_u32(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u8,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn is_u8_i64(
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const u8,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // GATHER (GATHER_OP) - bf16 (CUDA_ARCH >= 800)
    // ============================================================
    pub fn gather_i64_bf16(
        numel: usize,
        ids: *const i64,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_u32_bf16(
        numel: usize,
        ids: *const u32,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_u8_bf16(
        numel: usize,
        ids: *const u8,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // GATHER (GATHER_OP) - f8_e4m3 (CUDA_ARCH >= 890)
    // ============================================================
    pub fn gather_i16_f8_e4m3(
        numel: usize,
        ids: *const i16,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_i32_f8_e4m3(
        numel: usize,
        ids: *const i32,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_i64_f8_e4m3(
        numel: usize,
        ids: *const i64,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_u32_f8_e4m3(
        numel: usize,
        ids: *const u32,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_u8_f8_e4m3(
        numel: usize,
        ids: *const u8,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // GATHER (GATHER_OP) - f16 (CUDA_ARCH >= 530)
    // ============================================================
    pub fn gather_i64_f16(
        numel: usize,
        ids: *const i64,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_u32_f16(
        numel: usize,
        ids: *const u32,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_u8_f16(
        numel: usize,
        ids: *const u8,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // GATHER (GATHER_OP) - Standard types
    // ============================================================
    // i64 index variants
    pub fn gather_i64_f32(
        numel: usize,
        ids: *const i64,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_i64_f64(
        numel: usize,
        ids: *const i64,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_i64_u8(
        numel: usize,
        ids: *const i64,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_i64_u32(
        numel: usize,
        ids: *const i64,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_i64_i64(
        numel: usize,
        ids: *const i64,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );

    // u32 index variants
    pub fn gather_u32_f32(
        numel: usize,
        ids: *const u32,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_u32_f64(
        numel: usize,
        ids: *const u32,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_u32_u8(
        numel: usize,
        ids: *const u32,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_u32_i64(
        numel: usize,
        ids: *const u32,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_u32_u32(
        numel: usize,
        ids: *const u32,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );

    // u8 index variants
    pub fn gather_u8_f32(
        numel: usize,
        ids: *const u8,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_u8_f64(
        numel: usize,
        ids: *const u8,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_u8_u8(
        numel: usize,
        ids: *const u8,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_u8_u32(
        numel: usize,
        ids: *const u8,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );
    pub fn gather_u8_i64(
        numel: usize,
        ids: *const u8,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        ids_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // INDEX_ADD (IA_OP) - bf16 (CUDA_ARCH >= 800)
    // ============================================================
    pub fn ia_i64_bf16(
        ids: *const i64,
        ids_dim_size: usize,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_u32_bf16(
        ids: *const u32,
        ids_dim_size: usize,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_u8_bf16(
        ids: *const u8,
        ids_dim_size: usize,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // INDEX_ADD (IA_OP_F8) - f8_e4m3 (CUDA_ARCH >= 890)
    // ============================================================
    pub fn ia_i16_f8_e4m3(
        ids: *const i16,
        ids_dim_size: usize,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_i32_f8_e4m3(
        ids: *const i32,
        ids_dim_size: usize,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_i64_f8_e4m3(
        ids: *const i64,
        ids_dim_size: usize,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_u32_f8_e4m3(
        ids: *const u32,
        ids_dim_size: usize,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_u8_f8_e4m3(
        ids: *const u8,
        ids_dim_size: usize,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // INDEX_ADD (IA_OP) - f16 (CUDA_ARCH >= 530)
    // ============================================================
    pub fn ia_i64_f16(
        ids: *const i64,
        ids_dim_size: usize,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_u32_f16(
        ids: *const u32,
        ids_dim_size: usize,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_u8_f16(
        ids: *const u8,
        ids_dim_size: usize,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // INDEX_ADD (IA_OP) - Standard types
    // ============================================================
    // i64 index variants
    pub fn ia_i64_f32(
        ids: *const i64,
        ids_dim_size: usize,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_i64_f64(
        ids: *const i64,
        ids_dim_size: usize,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_i64_u8(
        ids: *const i64,
        ids_dim_size: usize,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_i64_i64(
        ids: *const i64,
        ids_dim_size: usize,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_i64_u32(
        ids: *const i64,
        ids_dim_size: usize,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // u32 index variants
    pub fn ia_u32_f32(
        ids: *const u32,
        ids_dim_size: usize,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_u32_f64(
        ids: *const u32,
        ids_dim_size: usize,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_u32_u8(
        ids: *const u32,
        ids_dim_size: usize,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_u32_i64(
        ids: *const u32,
        ids_dim_size: usize,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_u32_u32(
        ids: *const u32,
        ids_dim_size: usize,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // u8 index variants
    pub fn ia_u8_f32(
        ids: *const u8,
        ids_dim_size: usize,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_u8_f64(
        ids: *const u8,
        ids_dim_size: usize,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_u8_u8(
        ids: *const u8,
        ids_dim_size: usize,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_u8_u32(
        ids: *const u8,
        ids_dim_size: usize,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn ia_u8_i64(
        ids: *const u8,
        ids_dim_size: usize,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // SCATTER_ADD (SA_OP) - bf16 (CUDA_ARCH >= 800)
    // ============================================================
    pub fn sa_i64_bf16(
        ids: *const i64,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_u32_bf16(
        ids: *const u32,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_u8_bf16(
        ids: *const u8,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // SCATTER_ADD (SA_OP_F8) - f8_e4m3 (CUDA_ARCH >= 890)
    // ============================================================
    pub fn sa_i16_f8_e4m3(
        ids: *const i16,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_i32_f8_e4m3(
        ids: *const i32,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_i64_f8_e4m3(
        ids: *const i64,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_u32_f8_e4m3(
        ids: *const u32,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_u8_f8_e4m3(
        ids: *const u8,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // SCATTER_ADD (SA_OP) - f16 (CUDA_ARCH >= 530)
    // ============================================================
    pub fn sa_i64_f16(
        ids: *const i64,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_u32_f16(
        ids: *const u32,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_u8_f16(
        ids: *const u8,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // SCATTER_ADD (SA_OP) - Standard types
    // ============================================================
    // i64 index variants
    pub fn sa_i64_f32(
        ids: *const i64,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_i64_f64(
        ids: *const i64,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_i64_u8(
        ids: *const i64,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_i64_i64(
        ids: *const i64,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_i64_u32(
        ids: *const i64,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // u32 index variants
    pub fn sa_u32_f32(
        ids: *const u32,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_u32_f64(
        ids: *const u32,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_u32_u8(
        ids: *const u32,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_u32_i64(
        ids: *const u32,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_u32_u32(
        ids: *const u32,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // u8 index variants
    pub fn sa_u8_f32(
        ids: *const u8,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_u8_f64(
        ids: *const u8,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_u8_u8(
        ids: *const u8,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_u8_u32(
        ids: *const u8,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn sa_u8_i64(
        ids: *const u8,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // SCATTER (S_OP) - bf16 (CUDA_ARCH >= 800)
    // ============================================================
    pub fn s_i64_bf16(
        ids: *const i64,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_u32_bf16(
        ids: *const u32,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_u8_bf16(
        ids: *const u8,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // SCATTER (S_OP) - f16 (CUDA_ARCH >= 530)
    // ============================================================
    pub fn s_i64_f16(
        ids: *const i64,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_u32_f16(
        ids: *const u32,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_u8_f16(
        ids: *const u8,
        inp: *const c_void,
        out: *mut c_void,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // ============================================================
    // SCATTER (S_OP) - Standard types
    // ============================================================
    // i64 index variants
    pub fn s_i64_f32(
        ids: *const i64,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_i64_f64(
        ids: *const i64,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_i64_u8(
        ids: *const i64,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_i64_i64(
        ids: *const i64,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_i64_u32(
        ids: *const i64,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // u32 index variants
    pub fn s_u32_f32(
        ids: *const u32,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_u32_f64(
        ids: *const u32,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_u32_u8(
        ids: *const u32,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_u32_i64(
        ids: *const u32,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_u32_u32(
        ids: *const u32,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );

    // u8 index variants
    pub fn s_u8_f32(
        ids: *const u8,
        inp: *const f32,
        out: *mut f32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_u8_f64(
        ids: *const u8,
        inp: *const f64,
        out: *mut f64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_u8_u8(
        ids: *const u8,
        inp: *const u8,
        out: *mut u8,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_u8_u32(
        ids: *const u8,
        inp: *const u32,
        out: *mut u32,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
    pub fn s_u8_i64(
        ids: *const u8,
        inp: *const i64,
        out: *mut i64,
        left_size: usize,
        src_dim_size: usize,
        dst_dim_size: usize,
        right_size: usize,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_dtype_values() {
        assert_eq!(IndexDType::I16 as i32, 0);
        assert_eq!(IndexDType::I32 as i32, 1);
        assert_eq!(IndexDType::I64 as i32, 2);
        assert_eq!(IndexDType::U32 as i32, 3);
        assert_eq!(IndexDType::U8 as i32, 4);
    }

    #[test]
    fn test_indexing_data_dtype_values() {
        assert_eq!(IndexingDataDType::F32 as i32, 0);
        assert_eq!(IndexingDataDType::F64 as i32, 1);
        assert_eq!(IndexingDataDType::U8 as i32, 2);
        assert_eq!(IndexingDataDType::U32 as i32, 3);
        assert_eq!(IndexingDataDType::I64 as i32, 4);
        assert_eq!(IndexingDataDType::F16 as i32, 5);
        assert_eq!(IndexingDataDType::BF16 as i32, 6);
        assert_eq!(IndexingDataDType::F8E4M3 as i32, 7);
    }
}
