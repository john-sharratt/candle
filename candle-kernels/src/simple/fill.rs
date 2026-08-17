//! FFI bindings for fill operation dispatcher
//!
//! Provides a unified interface to dispatch fill operations based on
//! data type enum.

use core::ffi::c_void;

/// Data type enum for fill operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillDType {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
    F8E4M3 = 4,
    U8 = 5,
    U32 = 6,
    I64 = 7,
}

impl FillDType {
    /// Returns the size in bytes of this dtype
    pub fn size_in_bytes(&self) -> usize {
        match self {
            FillDType::F32 => 4,
            FillDType::F64 => 8,
            FillDType::F16 => 2,
            FillDType::BF16 => 2,
            FillDType::F8E4M3 => 1,
            FillDType::U8 => 1,
            FillDType::U32 => 4,
            FillDType::I64 => 8,
        }
    }
}

// =============================================================================
// Helper functions to convert values to bits for the dispatcher
// =============================================================================

/// Convert f32 to bits for fill dispatcher
#[inline]
pub fn f32_to_bits(v: f32) -> u64 {
    v.to_bits() as u64
}

/// Convert f64 to bits for fill dispatcher
#[inline]
pub fn f64_to_bits(v: f64) -> u64 {
    v.to_bits()
}

/// Convert u8 to bits for fill dispatcher
#[inline]
pub fn u8_to_bits(v: u8) -> u64 {
    v as u64
}

/// Convert u16 to bits for fill dispatcher (for f16/bf16)
#[inline]
pub fn u16_to_bits(v: u16) -> u64 {
    v as u64
}

/// Convert u32 to bits for fill dispatcher
#[inline]
pub fn u32_to_bits(v: u32) -> u64 {
    v as u64
}

/// Convert i64 to bits for fill dispatcher
#[inline]
pub fn i64_to_bits(v: i64) -> u64 {
    v as u64
}

extern "C" {
    /// Dispatches to the appropriate fill kernel.
    ///
    /// Fills a contiguous buffer with a constant value.
    /// The value is passed as bits (use helper functions to convert).
    ///
    /// # Parameters
    /// - `dtype`: Data type (see FillDType enum)
    /// - `buf`: Buffer to fill
    /// - `value_bits`: Value as bits
    /// - `numel`: Number of elements
    pub fn run_fill_op(dtype: i32, buf: *mut c_void, value_bits: u64, numel: usize);

    /// Dispatches to the appropriate arange (integer iota) kernel.
    ///
    /// Writes `buf[i] = start + i*step` (exact integer arithmetic) into a contiguous
    /// buffer of `numel` elements. INTEGER dtypes only (U8/U32/I64) — float aranges
    /// keep the host build, whose repeated-addition rounding this closed form would
    /// not reproduce bit-for-bit. Start/step are passed as bits like `run_fill_op`.
    pub fn run_arange_op(
        dtype: i32,
        buf: *mut c_void,
        start_bits: u64,
        step_bits: u64,
        numel: usize,
    );

    /// Dispatches to the appropriate copy2d kernel.
    ///
    /// 2D strided copy operation.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see FillDType enum)
    /// - `src`: Source buffer
    /// - `dst`: Destination buffer
    /// - `d1`: First dimension
    /// - `d2`: Second dimension
    /// - `src_s`: Source stride
    /// - `dst_s`: Destination stride
    pub fn run_copy2d_op(
        dtype: i32,
        src: *const c_void,
        dst: *mut c_void,
        d1: u32,
        d2: u32,
        src_s: u32,
        dst_s: u32,
    );

    /// Dispatches to the appropriate const_set kernel.
    ///
    /// Strided fill operation with layout info.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see FillDType enum)
    /// - `numel`: Number of elements
    /// - `num_dims`: Number of dimensions
    /// - `info`: Pointer to dims and strides
    /// - `value_bits`: Value as bits
    /// - `out`: Output buffer
    pub fn run_const_set_op(
        dtype: i32,
        numel: usize,
        num_dims: usize,
        info: *const usize,
        value_bits: u64,
        out: *mut c_void,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fill_dtype_enum_values() {
        assert_eq!(FillDType::F32 as i32, 0);
        assert_eq!(FillDType::F64 as i32, 1);
        assert_eq!(FillDType::F16 as i32, 2);
        assert_eq!(FillDType::BF16 as i32, 3);
        assert_eq!(FillDType::F8E4M3 as i32, 4);
        assert_eq!(FillDType::U8 as i32, 5);
        assert_eq!(FillDType::U32 as i32, 6);
        assert_eq!(FillDType::I64 as i32, 7);
    }

    #[test]
    fn test_f32_to_bits() {
        assert_eq!(f32_to_bits(1.0), 0x3f800000);
        assert_eq!(f32_to_bits(0.0), 0x00000000);
    }

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(FillDType::F32.size_in_bytes(), 4);
        assert_eq!(FillDType::F64.size_in_bytes(), 8);
        assert_eq!(FillDType::F16.size_in_bytes(), 2);
        assert_eq!(FillDType::BF16.size_in_bytes(), 2);
        assert_eq!(FillDType::F8E4M3.size_in_bytes(), 1);
        assert_eq!(FillDType::U8.size_in_bytes(), 1);
        assert_eq!(FillDType::U32.size_in_bytes(), 4);
        assert_eq!(FillDType::I64.size_in_bytes(), 8);
    }
}
