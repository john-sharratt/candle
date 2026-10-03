//! FFI binding for the MoE shared-expert residual (`simple/moe_shared_residual.cu`):
//! `x += narrow(routed + shared · sigmoid(gate))` in one launch, bit-identical to
//! the sigmoid, broadcast multiply, add and residual add it replaces.
//!
//! dtype codes are [`MoeScatterDType`](super::moe_scatter::MoeScatterDType)'s.

use std::ffi::c_void;

/// The launcher ran (or had no rows to run).
pub const MOE_SHARED_RESIDUAL_LAUNCHED: i32 = 0;

/// The launcher refused its dtype pair or shape and wrote nothing.
pub const MOE_SHARED_RESIDUAL_REFUSED: i32 = 1;

extern "C" {
    /// `x[t, j] += narrow(routed[t, j] + shared[t, j] · sigmoid(gate[t · gate_stride]))`.
    ///
    /// - `parts_dtype`: `routed`, `shared` and `gate`'s type — the FFN's working width
    /// - `x_dtype`: the residual's type. Instantiated pairs (parts, residual): F32/F32,
    ///   BF16/BF16, BF16/F16; any other pair returns [`MOE_SHARED_RESIDUAL_REFUSED`].
    /// - `x`, `routed`, `shared`: dense `[n, d]` device pointers at their start offsets
    /// - `gate`: one scalar per row, `gate_stride` elements apart
    pub fn run_moe_shared_residual(
        parts_dtype: i32,
        x_dtype: i32,
        x: *mut c_void,
        routed: *const c_void,
        shared: *const c_void,
        gate: *const c_void,
        n: i32,
        d: i32,
        gate_stride: i32,
        stream: *mut c_void,
    ) -> i32;
}
