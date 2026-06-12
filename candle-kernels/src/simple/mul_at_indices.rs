//! FFI bindings for mul_at_indices CUDA kernels
//!
//! These kernels multiply elements in a data array at positions specified by indices.

use std::ffi::c_void;

extern "C" {
    /// Multiply at indices for f32 type using atomic operations
    ///
    /// # Arguments
    /// * `data` - Pointer to the f32 data array to modify
    /// * `indices` - Pointer to the indices array (uint32_t)
    /// * `num_indices` - Number of indices to process
    /// * `value` - The value to multiply by
    pub fn mul_at_indices_f32(data: *mut f32, indices: *const u32, num_indices: usize, value: f32);

    /// Multiply at indices for f16 type using atomic operations
    ///
    /// # Arguments
    /// * `data` - Pointer to the f16 data array to modify (as c_void)
    /// * `indices` - Pointer to the indices array (uint32_t)
    /// * `num_indices` - Number of indices to process
    /// * `value` - The value to multiply by (as f32, converted internally)
    pub fn mul_at_indices_f16(
        data: *mut c_void,
        indices: *const u32,
        num_indices: usize,
        value: f32,
    );

    /// Multiply at indices for bf16 type using atomic operations
    ///
    /// # Arguments
    /// * `data` - Pointer to the bf16 data array to modify (as c_void)
    /// * `indices` - Pointer to the indices array (uint32_t)
    /// * `num_indices` - Number of indices to process
    /// * `value` - The value to multiply by (as f32, converted internally)
    pub fn mul_at_indices_bf16(
        data: *mut c_void,
        indices: *const u32,
        num_indices: usize,
        value: f32,
    );

    /// Multiply at indices for f64 type using atomic operations
    ///
    /// # Arguments
    /// * `data` - Pointer to the f64 data array to modify
    /// * `indices` - Pointer to the indices array (uint32_t)
    /// * `num_indices` - Number of indices to process
    /// * `value` - The value to multiply by
    pub fn mul_at_indices_f64(data: *mut f64, indices: *const u32, num_indices: usize, value: f64);
}
