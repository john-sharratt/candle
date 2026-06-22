//! FFI bindings for fused SiLU-Mul kernel dispatcher.
//!
//! Computes `out = silu(gate) * up` in a single kernel launch,
//! eliminating the intermediate allocation and extra launch of
//! separate `silu()` + `mul()` calls.
//!
//! This is the core SwiGLU activation used in MoE expert FFNs.

use core::ffi::c_void;

/// Data type enum for fused SiLU-Mul operations.
/// Must match the switch cases in the CUDA dispatcher.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedSiluMulDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    F8E4M3 = 3,
}

extern "C" {
    /// Dispatches to the appropriate fused silu-mul kernel.
    ///
    /// Computes: `out[i] = silu(gate[i]) * up[i]`
    ///
    /// # Parameters
    /// - `dtype`: Data type (0=f32, 1=f16, 2=bf16, 3=f8_e4m3)
    /// - `numel`: Number of elements
    /// - `num_dims`: Number of dimensions
    /// - `dims_and_strides`: Pointer to dims and strides array (dims, gate_strides, up_strides)
    /// - `gate`: Gate projection output tensor (lhs for silu)
    /// - `up`: Up projection output tensor (rhs for mul)
    /// - `out`: Output tensor (same type as input)
    pub fn run_fused_silu_mul(
        dtype: i32,
        numel: usize,
        num_dims: usize,
        dims_and_strides: *const usize,
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
    );

    /// Fused SwiGLU → q8a128: `out = quantize(silu(gate) * up)` in one kernel (producer
    /// epilogue B4). `out` is the flat-grouped q8a1024 buffer. Requires `(rows*cols) % 128 == 0`;
    /// `gate`/`up` are contiguous `[rows × cols]`.
    ///
    /// - `dtype`: gate/up dtype (0=f32, 1=f16, 2=bf16)
    pub fn run_silu_mul_q8a128_op(
        dtype: i32,
        gate: *const c_void,
        up: *const c_void,
        out: *mut c_void,
        rows: i32,
        cols: i32,
    );
}
