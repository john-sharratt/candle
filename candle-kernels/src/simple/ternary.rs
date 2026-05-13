//! FFI bindings for ternary (where) operation dispatcher
//!
//! Provides a unified interface to dispatch where operations based on
//! condition type and data type enums.

use core::ffi::c_void;

/// Condition dtype enum for where operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhereCondDType {
    I64 = 0,
    U32 = 1,
    U8 = 2,
    /// Only supported for fp8_e4m3 data type
    I16 = 3,
    /// Only supported for fp8_e4m3 data type
    I32 = 4,
}

/// Data dtype enum for where operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WhereDataDType {
    F32 = 0,
    F64 = 1,
    U8 = 2,
    U32 = 3,
    I64 = 4,
    F16 = 5,
    BF16 = 6,
    F8E4M3 = 7,
}

impl WhereCondDType {
    /// Returns true if this condition dtype is only supported for fp8_e4m3 data type
    pub fn is_fp8_only(&self) -> bool {
        matches!(self, WhereCondDType::I16 | WhereCondDType::I32)
    }

    /// Returns true if this condition dtype supports all data types
    pub fn supports_all_data_types(&self) -> bool {
        !self.is_fp8_only()
    }
}

impl WhereDataDType {
    /// Returns true if this data dtype is a floating-point type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            WhereDataDType::F32
                | WhereDataDType::F64
                | WhereDataDType::F16
                | WhereDataDType::BF16
                | WhereDataDType::F8E4M3
        )
    }

    /// Returns true if this data dtype is an integer type
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            WhereDataDType::U8 | WhereDataDType::U32 | WhereDataDType::I64
        )
    }
}

extern "C" {
    /// Dispatches to the appropriate where kernel.
    ///
    /// # Parameters
    /// - `cond_dtype`: Condition data type (see WhereCondDType enum values)
    /// - `data_dtype`: Data type (see WhereDataDType enum values)
    /// - `numel`: Number of elements
    /// - `num_dims`: Number of dimensions
    /// - `info`: Pointer to dims and strides array (dims + strides + strides_t + strides_f)
    /// - `ids`: Condition tensor (boolean values)
    /// - `t`: True branch tensor
    /// - `f`: False branch tensor
    /// - `out`: Output tensor
    ///
    /// # Note
    /// - I16 and I32 condition dtypes only support F8E4M3 data type.
    /// - All other condition dtypes (I64, U32, U8) support all data types.
    pub fn run_where(
        cond_dtype: i32,
        data_dtype: i32,
        numel: usize,
        num_dims: usize,
        info: *const usize,
        ids: *const c_void,
        t: *const c_void,
        f: *const c_void,
        out: *mut c_void,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cond_dtype_enum_values() {
        assert_eq!(WhereCondDType::I64 as i32, 0);
        assert_eq!(WhereCondDType::U32 as i32, 1);
        assert_eq!(WhereCondDType::U8 as i32, 2);
        assert_eq!(WhereCondDType::I16 as i32, 3);
        assert_eq!(WhereCondDType::I32 as i32, 4);
    }

    #[test]
    fn test_data_dtype_enum_values() {
        assert_eq!(WhereDataDType::F32 as i32, 0);
        assert_eq!(WhereDataDType::F64 as i32, 1);
        assert_eq!(WhereDataDType::U8 as i32, 2);
        assert_eq!(WhereDataDType::U32 as i32, 3);
        assert_eq!(WhereDataDType::I64 as i32, 4);
        assert_eq!(WhereDataDType::F16 as i32, 5);
        assert_eq!(WhereDataDType::BF16 as i32, 6);
        assert_eq!(WhereDataDType::F8E4M3 as i32, 7);
    }

    #[test]
    fn test_cond_dtype_classification() {
        assert!(!WhereCondDType::I64.is_fp8_only());
        assert!(!WhereCondDType::U32.is_fp8_only());
        assert!(!WhereCondDType::U8.is_fp8_only());
        assert!(WhereCondDType::I16.is_fp8_only());
        assert!(WhereCondDType::I32.is_fp8_only());

        assert!(WhereCondDType::I64.supports_all_data_types());
        assert!(WhereCondDType::U32.supports_all_data_types());
        assert!(WhereCondDType::U8.supports_all_data_types());
        assert!(!WhereCondDType::I16.supports_all_data_types());
        assert!(!WhereCondDType::I32.supports_all_data_types());
    }

    #[test]
    fn test_data_dtype_classification() {
        assert!(WhereDataDType::F32.is_float());
        assert!(WhereDataDType::F64.is_float());
        assert!(WhereDataDType::F16.is_float());
        assert!(WhereDataDType::BF16.is_float());
        assert!(WhereDataDType::F8E4M3.is_float());
        assert!(!WhereDataDType::U8.is_float());
        assert!(!WhereDataDType::U32.is_float());
        assert!(!WhereDataDType::I64.is_float());

        assert!(WhereDataDType::U8.is_integer());
        assert!(WhereDataDType::U32.is_integer());
        assert!(WhereDataDType::I64.is_integer());
        assert!(!WhereDataDType::F32.is_integer());
        assert!(!WhereDataDType::F64.is_integer());
        assert!(!WhereDataDType::F16.is_integer());
        assert!(!WhereDataDType::BF16.is_integer());
        assert!(!WhereDataDType::F8E4M3.is_integer());
    }
}
