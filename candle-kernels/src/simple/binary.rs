//! FFI bindings for binary operation dispatcher
//!
//! Provides a unified interface to dispatch binary operations based on
//! operation type and data type enums.

use core::ffi::c_void;

/// Binary arithmetic operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryArithOp {
    Add = 0,
    Div = 1,
    Mul = 2,
    Sub = 3,
    Minimum = 4,
    Maximum = 5,
}

/// Binary comparison operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryCmpOp {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}

/// Unified binary operation enum (for single dispatcher)
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Arithmetic operations (0-5)
    Add = 0,
    Div = 1,
    Mul = 2,
    Sub = 3,
    Minimum = 4,
    Maximum = 5,
    // Comparison operations (6-11) - output is u8
    Eq = 6,
    Ne = 7,
    Lt = 8,
    Le = 9,
    Gt = 10,
    Ge = 11,
}

impl BinaryOp {
    /// Returns true if this is an arithmetic operation (output same type as input)
    pub fn is_arithmetic(&self) -> bool {
        (*self as i32) < 6
    }

    /// Returns true if this is a comparison operation (output is u8)
    pub fn is_comparison(&self) -> bool {
        (*self as i32) >= 6
    }
}

/// Data type enum for binary operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryDType {
    F32 = 0,
    F64 = 1,
    U8 = 2,
    U32 = 3,
    I64 = 4,
    F16 = 5,
    BF16 = 6,
    F8E4M3 = 7,
}

/// In-place binary arithmetic operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryInplaceOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
    Div = 3,
    Min = 4,
    Max = 5,
}

extern "C" {
    /// Dispatches to the appropriate binary arithmetic kernel.
    ///
    /// # Parameters
    /// - `op`: Arithmetic operation (0=add, 1=div, 2=mul, 3=sub, 4=minimum, 5=maximum)
    /// - `dtype`: Data type (0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3)
    /// - `numel`: Number of elements
    /// - `num_dims`: Number of dimensions
    /// - `dims_and_strides`: Pointer to dims and strides array (dims, lhs_strides, rhs_strides)
    /// - `lhs`: Left-hand side input tensor
    /// - `rhs`: Right-hand side input tensor
    /// - `out`: Output tensor (same type as input)
    pub fn run_binary_arith_op(
        op: i32,
        dtype: i32,
        numel: usize,
        num_dims: usize,
        dims_and_strides: *const usize,
        lhs: *const c_void,
        rhs: *const c_void,
        out: *mut c_void,
    );

    /// Dispatches to the appropriate binary comparison kernel.
    ///
    /// # Parameters
    /// - `op`: Comparison operation (0=eq, 1=ne, 2=lt, 3=le, 4=gt, 5=ge)
    /// - `dtype`: Data type of inputs (0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3)
    /// - `numel`: Number of elements
    /// - `num_dims`: Number of dimensions
    /// - `dims_and_strides`: Pointer to dims and strides array (dims, lhs_strides, rhs_strides)
    /// - `lhs`: Left-hand side input tensor
    /// - `rhs`: Right-hand side input tensor
    /// - `out`: Output tensor (always u8)
    pub fn run_binary_cmp_op(
        op: i32,
        dtype: i32,
        numel: usize,
        num_dims: usize,
        dims_and_strides: *const usize,
        lhs: *const c_void,
        rhs: *const c_void,
        out: *mut u8,
    );

    /// Unified dispatcher for all binary operations.
    ///
    /// # Parameters
    /// - `op`: Operation type (0-5 for arithmetic, 6-11 for comparison)
    ///   - 0=add, 1=div, 2=mul, 3=sub, 4=minimum, 5=maximum
    ///   - 6=eq, 7=ne, 8=lt, 9=le, 10=gt, 11=ge
    /// - `dtype`: Data type (0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3)
    /// - `numel`: Number of elements
    /// - `num_dims`: Number of dimensions
    /// - `dims_and_strides`: Pointer to dims and strides array (dims, lhs_strides, rhs_strides)
    /// - `lhs`: Left-hand side input tensor
    /// - `rhs`: Right-hand side input tensor
    /// - `out`: Output tensor (same type as input for arithmetic ops, u8 for comparison ops)
    pub fn run_binary_op(
        op: i32,
        dtype: i32,
        numel: usize,
        num_dims: usize,
        dims_and_strides: *const usize,
        lhs: *const c_void,
        rhs: *const c_void,
        out: *mut c_void,
    );

    /// Dispatches to the appropriate in-place binary arithmetic kernel.
    /// Modifies lhs in-place: lhs = lhs OP rhs
    ///
    /// # Parameters
    /// - `op`: In-place operation (0=add, 1=sub, 2=mul, 3=div, 4=min, 5=max)
    /// - `dtype`: Data type (0=f32, 1=f64, 2=u8, 3=u32, 4=i64, 5=f16, 6=bf16, 7=f8_e4m3)
    /// - `numel`: Number of elements
    /// - `num_dims`: Number of dimensions
    /// - `dims_and_strides`: Pointer to dims and strides array (dims, lhs_strides, rhs_strides)
    ///   Note: lhs_strides is ignored (lhs must be contiguous for in-place ops)
    /// - `lhs`: Left-hand side tensor (modified in-place, MUST be contiguous)
    /// - `rhs`: Right-hand side input tensor (can be strided)
    pub fn run_binary_inplace_op(
        op: i32,
        dtype: i32,
        numel: usize,
        num_dims: usize,
        dims_and_strides: *const usize,
        lhs: *mut c_void,
        rhs: *const c_void,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enum_values() {
        // Verify arithmetic ops are 0-5
        assert_eq!(BinaryOp::Add as i32, 0);
        assert_eq!(BinaryOp::Div as i32, 1);
        assert_eq!(BinaryOp::Mul as i32, 2);
        assert_eq!(BinaryOp::Sub as i32, 3);
        assert_eq!(BinaryOp::Minimum as i32, 4);
        assert_eq!(BinaryOp::Maximum as i32, 5);

        // Verify comparison ops are 6-11
        assert_eq!(BinaryOp::Eq as i32, 6);
        assert_eq!(BinaryOp::Ne as i32, 7);
        assert_eq!(BinaryOp::Lt as i32, 8);
        assert_eq!(BinaryOp::Le as i32, 9);
        assert_eq!(BinaryOp::Gt as i32, 10);
        assert_eq!(BinaryOp::Ge as i32, 11);

        // Verify dtype values
        assert_eq!(BinaryDType::F32 as i32, 0);
        assert_eq!(BinaryDType::F64 as i32, 1);
        assert_eq!(BinaryDType::U8 as i32, 2);
        assert_eq!(BinaryDType::U32 as i32, 3);
        assert_eq!(BinaryDType::I64 as i32, 4);
        assert_eq!(BinaryDType::F16 as i32, 5);
        assert_eq!(BinaryDType::BF16 as i32, 6);
        assert_eq!(BinaryDType::F8E4M3 as i32, 7);
    }

    #[test]
    fn test_op_classification() {
        assert!(BinaryOp::Add.is_arithmetic());
        assert!(BinaryOp::Mul.is_arithmetic());
        assert!(!BinaryOp::Add.is_comparison());

        assert!(BinaryOp::Eq.is_comparison());
        assert!(BinaryOp::Lt.is_comparison());
        assert!(!BinaryOp::Eq.is_arithmetic());
    }
}
