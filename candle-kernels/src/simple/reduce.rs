//! FFI bindings for reduce operation dispatchers
//!
//! Provides a unified interface to dispatch reduce operations based on
//! operation type and data type enums.

use core::ffi::c_void;

/// Fast reduce operations (sum, min, max)
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastReduceOp {
    Sum = 0,
    Min = 1,
    Max = 2,
}

/// Fast arg reduce operations (argmin, argmax)
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastArgReduceOp {
    ArgMin = 0,
    ArgMax = 1,
}

/// Data type enum for fast reduce operations
/// Supports both float and integer types
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FastReduceDType {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
    U32 = 4,
    I64 = 5,
    U8 = 6,
    /// FP8 E4M3 format (SM89+, Ada/Hopper). Only supports min/max/argmin/argmax (no sum due to lack of atomicAdd)
    F8E4M3 = 7,
}

/// Data type enum for sum operation (with atomicAdd)
/// Limited to types that support atomic operations
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SumDType {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
    U32 = 4,
}

/// Data type enum for float-only operations (softmax, rmsnorm, layernorm, rope)
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatDType {
    F32 = 0,
    F64 = 1,
    F16 = 2,
    BF16 = 3,
    /// FP8 E4M3 format (SM89+, Ada/Hopper)
    F8E4M3 = 4,
}

impl FastReduceDType {
    /// Returns true if this dtype is a floating-point type
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            FastReduceDType::F32
                | FastReduceDType::F64
                | FastReduceDType::F16
                | FastReduceDType::BF16
                | FastReduceDType::F8E4M3
        )
    }

    /// Returns true if this dtype is an integer type
    pub fn is_integer(&self) -> bool {
        !self.is_float()
    }
}

extern "C" {
    // =========================================================================
    // Fast reduce dispatcher (sum, min, max)
    // =========================================================================
    /// Dispatches to the appropriate fast reduce kernel.
    ///
    /// # Parameters
    /// - `op`: Operation type (0=sum, 1=min, 2=max)
    /// - `dtype`: Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=u32, 5=i64, 6=u8, 7=f8_e4m3)
    /// - `src_numel`: Total number of elements in source
    /// - `el_to_sum_per_block`: Elements to reduce per block
    /// - `num_dims`: Number of dimensions
    /// - `info`: Pointer to dims and strides array
    /// - `src`: Source tensor
    /// - `dst`: Destination tensor (same type as src)
    ///
    /// Note: F8E4M3 (7) does not support sum operation due to lack of atomicAdd
    pub fn run_fast_reduce_op(
        op: i32,
        dtype: i32,
        src_numel: usize,
        el_to_sum_per_block: usize,
        num_dims: usize,
        info: *const usize,
        src: *const c_void,
        dst: *mut c_void,
    );

    // =========================================================================
    // Fast arg reduce dispatcher (argmin, argmax)
    // =========================================================================
    /// Dispatches to the appropriate fast arg reduce kernel.
    ///
    /// # Parameters
    /// - `op`: Operation type (0=argmin, 1=argmax)
    /// - `dtype`: Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=u32, 5=i64, 6=u8, 7=f8_e4m3)
    /// - `src_numel`: Total number of elements in source
    /// - `el_to_sum_per_block`: Elements to reduce per block
    /// - `num_dims`: Number of dimensions
    /// - `info`: Pointer to dims and strides array
    /// - `src`: Source tensor
    /// - `dst`: Destination tensor (always u32 indices)
    pub fn run_fast_arg_reduce_op(
        op: i32,
        dtype: i32,
        src_numel: usize,
        el_to_sum_per_block: usize,
        num_dims: usize,
        info: *const usize,
        src: *const c_void,
        dst: *mut u32,
    );

    // =========================================================================
    // Sum dispatcher (with atomicAdd, multi-dimensional reduce)
    // =========================================================================
    /// Dispatches to the appropriate sum kernel (uses atomicAdd).
    ///
    /// # Parameters
    /// - `dtype`: Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=u32)
    /// - `numel`: Total number of elements
    /// - `num_dims`: Number of dimensions
    /// - `num_sum_dims`: Number of dimensions to sum over
    /// - `info`: Pointer to dims, strides, sum_dims_l, sum_dims_s
    /// - `inp`: Input tensor
    /// - `out`: Output tensor
    pub fn run_sum_op(
        dtype: i32,
        numel: usize,
        num_dims: usize,
        num_sum_dims: usize,
        info: *const usize,
        inp: *const c_void,
        out: *mut c_void,
    );

    // =========================================================================
    // Softmax dispatcher
    // =========================================================================
    /// Dispatches to the appropriate softmax kernel.
    ///
    /// # Parameters
    /// - `dtype`: Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3)
    /// - `src`: Source tensor
    /// - `dst`: Destination tensor
    /// - `n_rows`: Number of rows (batch dimension)
    /// - `n_cols`: Number of columns (softmax dimension size)
    pub fn run_softmax_op(
        dtype: i32,
        src: *const c_void,
        dst: *mut c_void,
        n_rows: i32,
        n_cols: i32,
    );

    // =========================================================================
    // RMSNorm dispatcher
    // =========================================================================
    /// Dispatches to the appropriate rmsnorm kernel.
    ///
    /// # Parameters
    /// - `dtype`: Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3)
    /// - `src`: Source tensor
    /// - `dst`: Destination tensor
    /// - `alpha`: Scale weights (can be null)
    /// - `n_rows`: Number of rows (batch dimension)
    /// - `n_cols`: Number of columns (normalization dimension size)
    /// - `eps`: Epsilon for numerical stability
    pub fn run_rmsnorm_op(
        dtype: i32,
        src: *const c_void,
        dst: *mut c_void,
        alpha: *const c_void,
        n_rows: i32,
        n_cols: i32,
        eps: f32,
    );

    /// Fused RMSNorm → q8a128: normalizes each row and writes the q8a128 activation
    /// block directly (producer epilogue for B1/B3/B5). `out` is the flat-grouped
    /// q8a1024 buffer. Requires `n_cols % 128 == 0` and `n_cols <= 8192`.
    ///
    /// # Parameters
    /// - `dtype`: input/alpha dtype (0=f32, 2=f16, 3=bf16)
    /// - `src`: source activations `[n_rows × n_cols]`
    /// - `out`: q8a1024 output buffer
    /// - `alpha`: RMSNorm weight `[n_cols]`
    /// - `n_rows` / `n_cols`: row count / normalization size
    /// - `eps`: epsilon
    pub fn run_rmsnorm_q8a128_op(
        dtype: i32,
        src: *const c_void,
        out: *mut c_void,
        alpha: *const c_void,
        n_rows: i32,
        n_cols: i32,
        eps: f32,
    );

    // =========================================================================
    // LayerNorm dispatcher
    // =========================================================================
    /// Dispatches to the appropriate layernorm kernel.
    ///
    /// # Parameters
    /// - `dtype`: Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3)
    /// - `src`: Source tensor
    /// - `dst`: Destination tensor
    /// - `alpha`: Scale weights (gamma, can be null)
    /// - `beta`: Bias weights (beta, can be null)
    /// - `n_rows`: Number of rows (batch dimension)
    /// - `n_cols`: Number of columns (normalization dimension size)
    /// - `eps`: Epsilon for numerical stability
    pub fn run_layernorm_op(
        dtype: i32,
        src: *const c_void,
        dst: *mut c_void,
        alpha: *const c_void,
        beta: *const c_void,
        n_rows: i32,
        n_cols: i32,
        eps: f32,
    );

    // =========================================================================
    // RoPE (Rotary Position Embedding) dispatchers
    // =========================================================================

    /// Dispatches to the appropriate rope_i (interleaved) kernel.
    ///
    /// # Parameters
    /// - `dtype`: Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3)
    /// - `src`: Source tensor
    /// - `cos`: Cosine values
    /// - `sin`: Sine values
    /// - `dst`: Destination tensor
    /// - `bh`: Batch * heads
    /// - `td`: Tokens * dimension
    /// - `stride_b`: Stride for batch dimension (0 for no batch handling)
    pub fn run_rope_i_op(
        dtype: i32,
        src: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        dst: *mut c_void,
        bh: u32,
        td: u32,
        stride_b: u32,
    );

    /// Dispatches to the appropriate rope (non-interleaved) kernel.
    ///
    /// # Parameters
    /// - `dtype`: Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3)
    /// - `src`: Source tensor
    /// - `cos`: Cosine values
    /// - `sin`: Sine values
    /// - `dst`: Destination tensor
    /// - `bh`: Batch * heads
    /// - `td`: Tokens * dimension
    /// - `d`: Dimension
    /// - `stride_b`: Stride for batch dimension (0 for no batch handling)
    pub fn run_rope_op(
        dtype: i32,
        src: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        dst: *mut c_void,
        bh: u32,
        td: u32,
        d: u32,
        stride_b: u32,
    );

    /// Dispatches to the appropriate rope_thd (t, h, d layout) kernel.
    ///
    /// # Parameters
    /// - `dtype`: Data type (0=f32, 1=f64, 2=f16, 3=bf16, 4=f8_e4m3)
    /// - `src`: Source tensor
    /// - `cos`: Cosine values
    /// - `sin`: Sine values
    /// - `dst`: Destination tensor
    /// - `b`: Batch size
    /// - `t`: Tokens
    /// - `h`: Heads
    /// - `d`: Dimension
    /// - `stride_b`: Stride for batch dimension (0 for no batch handling)
    pub fn run_rope_thd_op(
        dtype: i32,
        src: *const c_void,
        cos: *const c_void,
        sin: *const c_void,
        dst: *mut c_void,
        b: u32,
        t: u32,
        h: u32,
        d: u32,
        stride_b: u32,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_reduce_op_enum_values() {
        assert_eq!(FastReduceOp::Sum as i32, 0);
        assert_eq!(FastReduceOp::Min as i32, 1);
        assert_eq!(FastReduceOp::Max as i32, 2);
    }

    #[test]
    fn test_fast_arg_reduce_op_enum_values() {
        assert_eq!(FastArgReduceOp::ArgMin as i32, 0);
        assert_eq!(FastArgReduceOp::ArgMax as i32, 1);
    }

    #[test]
    fn test_fast_reduce_dtype_enum_values() {
        assert_eq!(FastReduceDType::F32 as i32, 0);
        assert_eq!(FastReduceDType::F64 as i32, 1);
        assert_eq!(FastReduceDType::F16 as i32, 2);
        assert_eq!(FastReduceDType::BF16 as i32, 3);
        assert_eq!(FastReduceDType::U32 as i32, 4);
        assert_eq!(FastReduceDType::I64 as i32, 5);
        assert_eq!(FastReduceDType::U8 as i32, 6);
        assert_eq!(FastReduceDType::F8E4M3 as i32, 7);
    }

    #[test]
    fn test_sum_dtype_enum_values() {
        assert_eq!(SumDType::F32 as i32, 0);
        assert_eq!(SumDType::F64 as i32, 1);
        assert_eq!(SumDType::F16 as i32, 2);
        assert_eq!(SumDType::BF16 as i32, 3);
        assert_eq!(SumDType::U32 as i32, 4);
    }

    #[test]
    fn test_float_dtype_enum_values() {
        assert_eq!(FloatDType::F32 as i32, 0);
        assert_eq!(FloatDType::F64 as i32, 1);
        assert_eq!(FloatDType::F16 as i32, 2);
        assert_eq!(FloatDType::BF16 as i32, 3);
        assert_eq!(FloatDType::F8E4M3 as i32, 4);
    }

    #[test]
    fn test_fast_reduce_dtype_classification() {
        assert!(FastReduceDType::F32.is_float());
        assert!(FastReduceDType::F64.is_float());
        assert!(FastReduceDType::F16.is_float());
        assert!(FastReduceDType::BF16.is_float());
        assert!(FastReduceDType::F8E4M3.is_float());
        assert!(!FastReduceDType::U32.is_float());
        assert!(!FastReduceDType::I64.is_float());
        assert!(!FastReduceDType::U8.is_float());

        assert!(!FastReduceDType::F32.is_integer());
        assert!(!FastReduceDType::F8E4M3.is_integer());
        assert!(FastReduceDType::U32.is_integer());
        assert!(FastReduceDType::I64.is_integer());
        assert!(FastReduceDType::U8.is_integer());
    }
}
