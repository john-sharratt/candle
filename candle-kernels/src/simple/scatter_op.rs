//! FFI bindings for scatter operation dispatcher
//!
//! Provides a unified interface to dispatch scatter operations (add, sub, mul, div)
//! at indices based on operation type and data type enums.
//!
//! Scatter operations: data[indices[i]] op= value
//! - Add: data[idx] += value
//! - Sub: data[idx] -= value  
//! - Mul: data[idx] *= value
//! - Div: data[idx] /= value

use core::ffi::c_void;

/// Scatter operations enum
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterOp {
    /// Add: data[indices[i]] += value
    Add = 0,
    /// Sub: data[indices[i]] -= value
    Sub = 1,
    /// Mul: data[indices[i]] *= value
    Mul = 2,
    /// Div: data[indices[i]] /= value
    Div = 3,
}

/// Data type enum for scatter operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatterDType {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
}

extern "C" {
    /// Dispatches to the appropriate scatter operation kernel.
    ///
    /// Scatter operations modify tensor data in-place at specified indices:
    /// `data[indices[i] * stride] op= value`
    ///
    /// # Parameters
    /// - `op`: Operation type (0=add, 1=sub, 2=mul, 3=div)
    /// - `dtype`: Data type (0=f32, 1=f64, 2=f16, 3=bf16)
    /// - `data`: Pointer to tensor data (modified in-place)
    /// - `indices`: Pointer to indices array (u32)
    /// - `num_indices`: Number of indices
    /// - `value_f32`: Value as f32 (used for f32/f16/bf16 operations)
    /// - `value_f64`: Value as f64 (used for f64 operations)
    /// - `stride`: Stride between elements (only used by add operation currently)
    ///
    /// # Notes
    /// - For f16/bf16, `value_f32` is converted to the appropriate type internally
    /// - For f64, use `value_f64`; `value_f32` is ignored
    /// - The `stride` parameter is only used by the add operation; for sub/mul/div
    ///   the kernels currently don't support strided access
    pub fn run_scatter_op_at_indices(
        op: i32,
        dtype: i32,
        data: *mut c_void,
        indices: *const u32,
        num_indices: usize,
        value_f32: f32,
        value_f64: f64,
        stride: usize,
    );

    /// Dispatches to the sub_at_indices_with_values kernel.
    ///
    /// Unlike the regular scatter operations which use a single value for all indices,
    /// this operation uses a separate value for each index:
    /// `data[indices[i]] -= values[i]`
    ///
    /// # Parameters
    /// - `dtype`: Data type (0=f32, 1=f64, 2=f16, 3=bf16)
    /// - `data`: Pointer to tensor data (modified in-place)
    /// - `indices`: Pointer to indices array (u32)
    /// - `values`: Pointer to values array
    ///   - For f32: values is `*const f32`
    ///   - For f64: values is `*const f64`
    ///   - For f16/bf16: values is `*const f32` (converted internally)
    /// - `num_indices`: Number of indices
    pub fn run_sub_at_indices_with_values(
        dtype: i32,
        data: *mut c_void,
        indices: *const u32,
        values: *const c_void,
        num_indices: usize,
    );
}

/// Helper trait for scatter operations with type-safe value passing
pub trait ScatterValue {
    /// Get the dtype enum value
    fn dtype() -> ScatterDType;
    /// Get value as f32 (for f32/f16/bf16)
    fn as_f32(&self) -> f32;
    /// Get value as f64 (for f64)
    fn as_f64(&self) -> f64;
}

impl ScatterValue for f32 {
    fn dtype() -> ScatterDType {
        ScatterDType::F32
    }
    fn as_f32(&self) -> f32 {
        *self
    }
    fn as_f64(&self) -> f64 {
        *self as f64
    }
}

impl ScatterValue for f64 {
    fn dtype() -> ScatterDType {
        ScatterDType::F64
    }
    fn as_f32(&self) -> f32 {
        *self as f32
    }
    fn as_f64(&self) -> f64 {
        *self
    }
}

/// Safe wrapper for scatter operations
pub struct ScatterDispatcher;

impl ScatterDispatcher {
    /// Perform a scatter operation at the specified indices.
    ///
    /// # Safety
    /// - `data` must be a valid pointer to a tensor of the specified dtype
    /// - `indices` must point to `num_indices` valid u32 values
    /// - All indices must be valid positions in the data tensor
    pub unsafe fn scatter_op<V: ScatterValue>(
        op: ScatterOp,
        dtype: ScatterDType,
        data: *mut c_void,
        indices: *const u32,
        num_indices: usize,
        value: V,
        stride: usize,
    ) {
        run_scatter_op_at_indices(
            op as i32,
            dtype as i32,
            data,
            indices,
            num_indices,
            value.as_f32(),
            value.as_f64(),
            stride,
        )
    }

    /// Perform scatter add operation
    pub unsafe fn add_at_indices<V: ScatterValue>(
        dtype: ScatterDType,
        data: *mut c_void,
        indices: *const u32,
        num_indices: usize,
        value: V,
        stride: usize,
    ) {
        Self::scatter_op(
            ScatterOp::Add,
            dtype,
            data,
            indices,
            num_indices,
            value,
            stride,
        )
    }

    /// Perform scatter sub operation
    pub unsafe fn sub_at_indices<V: ScatterValue>(
        dtype: ScatterDType,
        data: *mut c_void,
        indices: *const u32,
        num_indices: usize,
        value: V,
    ) {
        Self::scatter_op(ScatterOp::Sub, dtype, data, indices, num_indices, value, 1)
    }

    /// Perform scatter mul operation
    pub unsafe fn mul_at_indices<V: ScatterValue>(
        dtype: ScatterDType,
        data: *mut c_void,
        indices: *const u32,
        num_indices: usize,
        value: V,
    ) {
        Self::scatter_op(ScatterOp::Mul, dtype, data, indices, num_indices, value, 1)
    }

    /// Perform scatter div operation
    pub unsafe fn div_at_indices<V: ScatterValue>(
        dtype: ScatterDType,
        data: *mut c_void,
        indices: *const u32,
        num_indices: usize,
        value: V,
    ) {
        Self::scatter_op(ScatterOp::Div, dtype, data, indices, num_indices, value, 1)
    }

    /// Perform sub at indices with per-element values
    ///
    /// # Safety
    /// - Same requirements as scatter_op
    /// - Additionally, `values` must point to `num_indices` valid values
    pub unsafe fn sub_at_indices_with_values(
        dtype: ScatterDType,
        data: *mut c_void,
        indices: *const u32,
        values: *const c_void,
        num_indices: usize,
    ) {
        run_sub_at_indices_with_values(dtype as i32, data, indices, values, num_indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scatter_op_enum_values() {
        assert_eq!(ScatterOp::Add as i32, 0);
        assert_eq!(ScatterOp::Sub as i32, 1);
        assert_eq!(ScatterOp::Mul as i32, 2);
        assert_eq!(ScatterOp::Div as i32, 3);
    }

    #[test]
    fn test_scatter_dtype_enum_values() {
        assert_eq!(ScatterDType::F32 as i32, 0);
        assert_eq!(ScatterDType::F64 as i32, 1);
        assert_eq!(ScatterDType::F16 as i32, 2);
        assert_eq!(ScatterDType::BF16 as i32, 3);
    }

    #[test]
    fn test_scatter_value_f32() {
        let val: f32 = 3.14;
        assert_eq!(f32::dtype() as i32, ScatterDType::F32 as i32);
        assert_eq!(val.as_f32(), 3.14f32);
        assert!((val.as_f64() - 3.14f64).abs() < 0.001);
    }

    #[test]
    fn test_scatter_value_f64() {
        let val: f64 = 2.718;
        assert_eq!(f64::dtype() as i32, ScatterDType::F64 as i32);
        assert!((val.as_f32() - 2.718f32).abs() < 0.001);
        assert_eq!(val.as_f64(), 2.718f64);
    }
}
