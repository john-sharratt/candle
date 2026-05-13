//! FFI bindings for sub_at_indices_with_values CUDA kernels
//!
//! These kernels subtract values at specified indices: data[indices[i]] -= values[i]

use std::ffi::c_void;

extern "C" {
    /// Subtract values at specified indices for f32 data
    /// data[indices[i]] -= values[i] for each i
    pub fn sub_at_indices_with_values_f32(
        data: *mut f32,
        indices: *const u32,
        values: *const f32,
        num_indices: usize,
    );

    /// Subtract values at specified indices for f16 data
    /// data[indices[i]] -= values[i] for each i
    /// Note: values are provided as f32 and converted to f16 internally
    pub fn sub_at_indices_with_values_f16(
        data: *mut c_void, // __half*
        indices: *const u32,
        values: *const f32,
        num_indices: usize,
    );

    /// Subtract values at specified indices for bf16 data
    /// data[indices[i]] -= values[i] for each i
    /// Note: values are provided as f32 and converted to bf16 internally
    pub fn sub_at_indices_with_values_bf16(
        data: *mut c_void, // __nv_bfloat16*
        indices: *const u32,
        values: *const f32,
        num_indices: usize,
    );

    /// Subtract values at specified indices for f64 data
    /// data[indices[i]] -= values[i] for each i
    pub fn sub_at_indices_with_values_f64(
        data: *mut f64,
        indices: *const u32,
        values: *const f64,
        num_indices: usize,
    );
}
