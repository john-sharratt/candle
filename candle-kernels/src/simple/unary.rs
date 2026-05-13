//! FFI bindings for unary operation dispatcher
//!
//! Provides a unified interface to dispatch unary operations based on
//! operation type and data type enums.

use core::ffi::c_void;

/// Unary operations (standard, no extra parameter)
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Copy = 0,
    Neg = 1,
    Recip = 2,
    Exp = 3,
    Log = 4,
    Sin = 5,
    Cos = 6,
    Tanh = 7,
    Erf = 8,
    Ceil = 9,
    Floor = 10,
    Round = 11,
    Normcdf = 12,
    Abs = 13,
    Sqr = 14,
    Sqrt = 15,
    Gelu = 16,
    GeluErf = 17,
    Relu = 18,
    Silu = 19,
    Sign = 20,
    Sigmoid = 21,
}

/// Parametric unary operations (require an extra float parameter)
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryParamOp {
    /// ELU activation: alpha * (exp(x) - 1) for x < 0, x for x >= 0
    Elu = 0,
    /// Power function: x^param
    Powf = 1,
}

/// Data type enum for unary operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryDType {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
    F8E4M3 = 4,
    /// Only supported for Copy operation
    U8 = 5,
    /// Only supported for Copy operation
    U32 = 6,
    /// Only supported for Copy operation
    I64 = 7,
}

impl UnaryDType {
    /// Returns true if this dtype is only supported for copy operations
    pub fn is_copy_only(&self) -> bool {
        matches!(self, UnaryDType::U8 | UnaryDType::U32 | UnaryDType::I64)
    }

    /// Returns true if this dtype is a floating-point type
    pub fn is_float(&self) -> bool {
        !self.is_copy_only()
    }
}

impl UnaryOp {
    /// Returns true if this operation supports integer types (only Copy does)
    pub fn supports_integers(&self) -> bool {
        matches!(self, UnaryOp::Copy)
    }
}

extern "C" {
    /// Dispatches to the appropriate standard unary kernel.
    ///
    /// # Parameters
    /// - `op`: Operation type (see UnaryOp enum values)
    /// - `dtype`: Data type (see UnaryDType enum values)
    /// - `numel`: Number of elements
    /// - `num_dims`: Number of dimensions
    /// - `info`: Pointer to dims and strides array
    /// - `inp`: Input tensor (can be null for in-place operations)
    /// - `out`: Output tensor
    ///
    /// # Note
    /// Only the Copy operation (op=0) supports u8, u32, and i64 dtypes.
    /// All other operations only support float types (f32, f64, f16, bf16, f8_e4m3).
    pub fn run_unary_op(
        op: i32,
        dtype: i32,
        numel: usize,
        num_dims: usize,
        info: *const usize,
        inp: *const c_void,
        out: *mut c_void,
    );

    /// Dispatches to the appropriate parametric unary kernel (elu, powf).
    ///
    /// # Parameters
    /// - `op`: Operation type (0=elu, 1=powf)
    /// - `dtype`: Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3)
    /// - `param`: The operation parameter (alpha for elu, exponent for powf)
    /// - `numel`: Number of elements
    /// - `num_dims`: Number of dimensions
    /// - `info`: Pointer to dims and strides array
    /// - `inp`: Input tensor
    /// - `out`: Output tensor
    ///
    /// # Note
    /// Only float types are supported for parametric operations.
    pub fn run_unary_param_op(
        op: i32,
        dtype: i32,
        param: f32,
        numel: usize,
        num_dims: usize,
        info: *const usize,
        inp: *const c_void,
        out: *mut c_void,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unary_op_enum_values() {
        assert_eq!(UnaryOp::Copy as i32, 0);
        assert_eq!(UnaryOp::Neg as i32, 1);
        assert_eq!(UnaryOp::Recip as i32, 2);
        assert_eq!(UnaryOp::Exp as i32, 3);
        assert_eq!(UnaryOp::Log as i32, 4);
        assert_eq!(UnaryOp::Sin as i32, 5);
        assert_eq!(UnaryOp::Cos as i32, 6);
        assert_eq!(UnaryOp::Tanh as i32, 7);
        assert_eq!(UnaryOp::Erf as i32, 8);
        assert_eq!(UnaryOp::Ceil as i32, 9);
        assert_eq!(UnaryOp::Floor as i32, 10);
        assert_eq!(UnaryOp::Round as i32, 11);
        assert_eq!(UnaryOp::Normcdf as i32, 12);
        assert_eq!(UnaryOp::Abs as i32, 13);
        assert_eq!(UnaryOp::Sqr as i32, 14);
        assert_eq!(UnaryOp::Sqrt as i32, 15);
        assert_eq!(UnaryOp::Gelu as i32, 16);
        assert_eq!(UnaryOp::GeluErf as i32, 17);
        assert_eq!(UnaryOp::Relu as i32, 18);
        assert_eq!(UnaryOp::Silu as i32, 19);
        assert_eq!(UnaryOp::Sign as i32, 20);
        assert_eq!(UnaryOp::Sigmoid as i32, 21);
    }

    #[test]
    fn test_unary_param_op_enum_values() {
        assert_eq!(UnaryParamOp::Elu as i32, 0);
        assert_eq!(UnaryParamOp::Powf as i32, 1);
    }

    #[test]
    fn test_dtype_enum_values() {
        assert_eq!(UnaryDType::F32 as i32, 0);
        assert_eq!(UnaryDType::F64 as i32, 1);
        assert_eq!(UnaryDType::F16 as i32, 2);
        assert_eq!(UnaryDType::BF16 as i32, 3);
        assert_eq!(UnaryDType::F8E4M3 as i32, 4);
        assert_eq!(UnaryDType::U8 as i32, 5);
        assert_eq!(UnaryDType::U32 as i32, 6);
        assert_eq!(UnaryDType::I64 as i32, 7);
    }

    #[test]
    fn test_dtype_classification() {
        assert!(!UnaryDType::F32.is_copy_only());
        assert!(!UnaryDType::F64.is_copy_only());
        assert!(!UnaryDType::F16.is_copy_only());
        assert!(!UnaryDType::BF16.is_copy_only());
        assert!(!UnaryDType::F8E4M3.is_copy_only());
        assert!(UnaryDType::U8.is_copy_only());
        assert!(UnaryDType::U32.is_copy_only());
        assert!(UnaryDType::I64.is_copy_only());

        assert!(UnaryDType::F32.is_float());
        assert!(!UnaryDType::U8.is_float());
    }

    #[test]
    fn test_op_integer_support() {
        assert!(UnaryOp::Copy.supports_integers());
        assert!(!UnaryOp::Neg.supports_integers());
        assert!(!UnaryOp::Exp.supports_integers());
        assert!(!UnaryOp::Gelu.supports_integers());
    }
}
