//! FFI bindings for add_at_indices CUDA kernels
//!
//! These kernels atomically add a value to tensor elements at specified indices.
//! The host-side dispatcher functions handle CUDA kernel launch configuration.

use half::{bf16, f16};

extern "C" {
    /// Atomically add a value to f32 tensor elements at specified indices
    ///
    /// # Arguments
    /// * `data` - Pointer to the f32 data buffer
    /// * `indices` - Pointer to the u32 indices array
    /// * `num_indices` - Number of indices to process
    /// * `value` - The f32 value to add
    /// * `stride` - Stride between elements
    ///
    /// # Safety
    /// - `data` must be a valid device pointer
    /// - `indices` must be a valid device pointer with at least `num_indices` elements
    /// - All indices must be valid within the data buffer
    pub fn add_at_indices_f32(
        data: *mut f32,
        indices: *const u32,
        num_indices: usize,
        value: f32,
        stride: usize,
    );

    /// Atomically add a value to f16 tensor elements at specified indices
    ///
    /// # Arguments
    /// * `data` - Pointer to the f16 data buffer
    /// * `indices` - Pointer to the u32 indices array
    /// * `num_indices` - Number of indices to process
    /// * `value` - The f16 value to add
    /// * `stride` - Stride between elements
    ///
    /// # Safety
    /// - `data` must be a valid device pointer
    /// - `indices` must be a valid device pointer with at least `num_indices` elements
    /// - All indices must be valid within the data buffer
    pub fn add_at_indices_f16(
        data: *mut f16,
        indices: *const u32,
        num_indices: usize,
        value: f16,
        stride: usize,
    );

    /// Atomically add a value to bf16 tensor elements at specified indices
    ///
    /// # Arguments
    /// * `data` - Pointer to the bf16 data buffer
    /// * `indices` - Pointer to the u32 indices array
    /// * `num_indices` - Number of indices to process
    /// * `value` - The bf16 value to add
    /// * `stride` - Stride between elements
    ///
    /// # Safety
    /// - `data` must be a valid device pointer
    /// - `indices` must be a valid device pointer with at least `num_indices` elements
    /// - All indices must be valid within the data buffer
    pub fn add_at_indices_bf16(
        data: *mut bf16,
        indices: *const u32,
        num_indices: usize,
        value: bf16,
        stride: usize,
    );

    /// Atomically add a value to f64 tensor elements at specified indices
    ///
    /// # Arguments
    /// * `data` - Pointer to the f64 data buffer
    /// * `indices` - Pointer to the u32 indices array
    /// * `num_indices` - Number of indices to process
    /// * `value` - The f64 value to add
    /// * `stride` - Stride between elements
    ///
    /// # Safety
    /// - `data` must be a valid device pointer
    /// - `indices` must be a valid device pointer with at least `num_indices` elements
    /// - All indices must be valid within the data buffer
    pub fn add_at_indices_f64(
        data: *mut f64,
        indices: *const u32,
        num_indices: usize,
        value: f64,
        stride: usize,
    );
}
