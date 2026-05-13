//! FFI bindings for affine.cu kernels and affine_dispatcher.cu
use core::ffi::c_void;

/// Data type enum for affine operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AffineDType {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
    F8E4M3 = 4,
    U8 = 5,
    U32 = 6,
    I16 = 7,
    I32 = 8,
    I64 = 9,
}

impl AffineDType {
    /// Returns true if this dtype is a floating-point type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            AffineDType::F32
                | AffineDType::F64
                | AffineDType::F16
                | AffineDType::BF16
                | AffineDType::F8E4M3
        )
    }

    /// Returns true if this dtype is an integer type
    pub fn is_integer(&self) -> bool {
        !self.is_float()
    }
}

extern "C" {
    // Individual typed kernel bindings
    pub fn affine_f32(numel: usize, num_dims: usize, info: *const usize, inp: *const f32, out: *mut f32, mul: f32, add: f32);
    pub fn affine_f64(numel: usize, num_dims: usize, info: *const usize, inp: *const f64, out: *mut f64, mul: f64, add: f64);
    pub fn affine_u8(numel: usize, num_dims: usize, info: *const usize, inp: *const u8, out: *mut u8, mul: u8, add: u8);
    pub fn affine_u32(numel: usize, num_dims: usize, info: *const usize, inp: *const u32, out: *mut u32, mul: u32, add: u32);
    pub fn affine_i16(numel: usize, num_dims: usize, info: *const usize, inp: *const i16, out: *mut i16, mul: i16, add: i16);
    pub fn affine_i32(numel: usize, num_dims: usize, info: *const usize, inp: *const i32, out: *mut i32, mul: i32, add: i32);
    pub fn affine_i64(numel: usize, num_dims: usize, info: *const usize, inp: *const i64, out: *mut i64, mul: i64, add: i64);
    pub fn affine_f16(numel: usize, num_dims: usize, info: *const usize, inp: *const c_void, out: *mut c_void, mul: u16, add: u16);
    pub fn affine_bf16(numel: usize, num_dims: usize, info: *const usize, inp: *const c_void, out: *mut c_void, mul: u16, add: u16);
    pub fn affine_f8_e4m3(numel: usize, num_dims: usize, info: *const usize, inp: *const c_void, out: *mut c_void, mul: u8, add: u8);

    /// Dispatches to the appropriate affine kernel based on dtype.
    ///
    /// # Parameters
    /// - `dtype`: Data type (see AffineDType enum values)
    /// - `numel`: Number of elements
    /// - `num_dims`: Number of dimensions
    /// - `info`: Pointer to dims and strides array
    /// - `inp`: Input tensor (can be null for in-place operations)
    /// - `out`: Output tensor
    /// - `mul`: Multiplier value (as f64, converted internally)
    /// - `add`: Addend value (as f64, converted internally)
    ///
    /// # Note
    /// The mul and add parameters are passed as f64 and converted to the
    /// appropriate type internally by the dispatcher.
    pub fn run_affine(
        dtype: i32,
        numel: usize,
        num_dims: usize,
        info: *const usize,
        inp: *const c_void,
        out: *mut c_void,
        mul: f64,
        add: f64,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_dtype_enum_values() {
        assert_eq!(AffineDType::F32 as i32, 0);
        assert_eq!(AffineDType::F64 as i32, 1);
        assert_eq!(AffineDType::F16 as i32, 2);
        assert_eq!(AffineDType::BF16 as i32, 3);
        assert_eq!(AffineDType::F8E4M3 as i32, 4);
        assert_eq!(AffineDType::U8 as i32, 5);
        assert_eq!(AffineDType::U32 as i32, 6);
        assert_eq!(AffineDType::I16 as i32, 7);
        assert_eq!(AffineDType::I32 as i32, 8);
        assert_eq!(AffineDType::I64 as i32, 9);
    }

    #[test]
    fn test_affine_dtype_is_float() {
        assert!(AffineDType::F32.is_float());
        assert!(AffineDType::F64.is_float());
        assert!(AffineDType::F16.is_float());
        assert!(AffineDType::BF16.is_float());
        assert!(AffineDType::F8E4M3.is_float());
        assert!(!AffineDType::U8.is_float());
        assert!(!AffineDType::U32.is_float());
        assert!(!AffineDType::I16.is_float());
        assert!(!AffineDType::I32.is_float());
        assert!(!AffineDType::I64.is_float());
    }

    #[test]
    fn test_affine_dtype_is_integer() {
        assert!(!AffineDType::F32.is_integer());
        assert!(!AffineDType::F64.is_integer());
        assert!(!AffineDType::F16.is_integer());
        assert!(!AffineDType::BF16.is_integer());
        assert!(!AffineDType::F8E4M3.is_integer());
        assert!(AffineDType::U8.is_integer());
        assert!(AffineDType::U32.is_integer());
        assert!(AffineDType::I16.is_integer());
        assert!(AffineDType::I32.is_integer());
        assert!(AffineDType::I64.is_integer());
    }
}
