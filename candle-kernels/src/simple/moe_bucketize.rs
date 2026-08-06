// FFI binding for the GPU MoE expert-bucketize kernel.
//
// See `simple/moe_bucketize.cu` for the authoritative contract: outputs,
// stability guarantees, and the padding conventions. All array arguments are
// device pointers on `stream`; every output/scratch buffer is sized to the
// `n_tokens × k` launch bound; `header` is `i32[4]` and `token_starts` is
// `i32[n_tokens + 1]`.

use std::ffi::c_void;

/// Kernel bound on experts (`n_experts`), mirrored from `MAX_EXPERTS` in the
/// `.cu`. The Rust wrapper validates against THESE constants so its checks can
/// never drift from the launcher's silent-return guards (a drifted wrapper
/// would skip the launch and leave the workspace holding the PREVIOUS layer's
/// tables).
pub const MAX_EXPERTS: usize = 256;
/// Kernel bound on top-k width (`k`), mirrored from `MAX_TOPK` in the `.cu`.
pub const MAX_TOPK: usize = 32;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn run_moe_bucketize(
        topk_ids: *const c_void,
        n_tokens: i32,
        k: i32,
        n_experts: i32,
        tile_w: i32,
        tok_ids: *mut c_void,
        weight_ids: *mut c_void,
        tile_expert: *mut c_void,
        tile_b_start: *mut c_void,
        tile_b_cnt: *mut c_void,
        perm: *mut c_void,
        rw_ids: *mut c_void,
        token_starts: *mut c_void,
        header: *mut c_void,
        inv: *mut c_void,
        scan: *mut c_void,
        stream: *mut c_void,
    );
}
