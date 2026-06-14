//! FFI bindings for sub_at_indices CUDA kernels
//!
//! These kernels subtract a value from tensor elements at specified indices.

use std::ffi::c_void;

extern "C" {
    /// Subtract a value from f32 tensor elements at specified indices
    ///
    /// # Arguments
    /// * `data` - Pointer to the f32 tensor data
    /// * `indices` - Pointer to the indices array
    /// * `num_indices` - Number of indices
    /// * `value` - Value to subtract
    pub fn sub_at_indices_f32(data: *mut f32, indices: *const u32, num_indices: usize, value: f32);

    /// Subtract a value from f16 tensor elements at specified indices
    ///
    /// # Arguments
    /// * `data` - Pointer to the f16 tensor data (as c_void)
    /// * `indices` - Pointer to the indices array
    /// * `num_indices` - Number of indices
    /// * `value` - Value to subtract (as f32, converted internally)
    pub fn sub_at_indices_f16(
        data: *mut c_void,
        indices: *const u32,
        num_indices: usize,
        value: f32,
    );

    /// Subtract a value from bf16 tensor elements at specified indices
    ///
    /// # Arguments
    /// * `data` - Pointer to the bf16 tensor data (as c_void)
    /// * `indices` - Pointer to the indices array
    /// * `num_indices` - Number of indices
    /// * `value` - Value to subtract (as f32, converted internally)
    pub fn sub_at_indices_bf16(
        data: *mut c_void,
        indices: *const u32,
        num_indices: usize,
        value: f32,
    );

    /// Subtract a value from f64 tensor elements at specified indices
    ///
    /// # Arguments
    /// * `data` - Pointer to the f64 tensor data
    /// * `indices` - Pointer to the indices array
    /// * `num_indices` - Number of indices
    /// * `value` - Value to subtract
    pub fn sub_at_indices_f64(data: *mut f64, indices: *const u32, num_indices: usize, value: f64);
}
