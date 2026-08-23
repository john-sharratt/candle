// FFI binding for the fused MoE router epilogue.
//
// See `simple/router_topk.cu` for the authoritative contract: elementwise
// score functions only, the bit-exactness guarantees against the eager op
// chain, and the launch bounds. All array arguments are device pointers on
// `stream`; `bias` may be null.

use std::ffi::c_void;

/// Kernel bound on experts, mirrored from `32 * RT_MAX_PER_LANE` in the `.cu`.
pub const MAX_EXPERTS: usize = 1024;
/// Kernel bound on top-k width, mirrored from `RT_MAX_TOPK` in the `.cu`.
pub const MAX_TOPK: usize = 16;
/// Score-function selectors, mirrored from the `.cu`.
pub const SCORE_SIGMOID: i32 = 1;
pub const SCORE_SQRT_SOFTPLUS: i32 = 2;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn run_router_topk(
        logits: *const c_void,
        bias: *const c_void,
        n_tokens: i32,
        n_experts: i32,
        k: i32,
        score_func: i32,
        route_scale: f32,
        out_w: *mut c_void,
        out_i: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
}
