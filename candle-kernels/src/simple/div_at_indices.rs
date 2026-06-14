//! FFI bindings for div_at_indices CUDA kernels
//!
//! These kernels divide tensor elements at specified indices by a value.

use std::ffi::c_void;

extern "C" {
    /// Divide f32 tensor elements at specified indices by a value
    ///
    /// # Arguments
    /// * `data` - Pointer to the f32 tensor data
    /// * `indices` - Pointer to the indices array
    /// * `num_indices` - Number of indices
    /// * `value` - Value to divide by
    pub fn div_at_indices_f32(data: *mut f32, indices: *const u32, num_indices: usize, value: f32);

    /// Divide f16 tensor elements at specified indices by a value
    ///
    /// # Arguments
    /// * `data` - Pointer to the f16 tensor data (as c_void)
    /// * `indices` - Pointer to the indices array
    /// * `num_indices` - Number of indices
    /// * `value` - Value to divide by (as f32, converted internally)
    pub fn div_at_indices_f16(
        data: *mut c_void,
        indices: *const u32,
        num_indices: usize,
        value: f32,
    );

    /// Divide bf16 tensor elements at specified indices by a value
    ///
    /// # Arguments
    /// * `data` - Pointer to the bf16 tensor data (as c_void)
    /// * `indices` - Pointer to the indices array
    /// * `num_indices` - Number of indices
    /// * `value` - Value to divide by (as f32, converted internally)
    pub fn div_at_indices_bf16(
        data: *mut c_void,
        indices: *const u32,
        num_indices: usize,
        value: f32,
    );

    /// Divide f64 tensor elements at specified indices by a value
    ///
    /// # Arguments
    /// * `data` - Pointer to the f64 tensor data
    /// * `indices` - Pointer to the indices array
    /// * `num_indices` - Number of indices
    /// * `value` - Value to divide by
    pub fn div_at_indices_f64(data: *mut f64, indices: *const u32, num_indices: usize, value: f64);
}
