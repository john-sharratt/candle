// FFI binding for the GPU MoE expert-bucketize kernel.
//
// See `simple/moe_bucketize.cu` for the authoritative contract: outputs,
// stability guarantees, and the padding conventions. All array arguments are
// device pointers on `stream`; every output/scratch buffer is sized to the
// `n_tokens × k` launch bound; `header` is `i32[5]` and `token_starts` is
// `i32[n_tokens + 1]`.

use std::ffi::c_void;

/// Kernel bound on experts (`n_experts`), mirrored from `MAX_EXPERTS` in the
/// `.cu`. The Rust wrapper validates against THESE constants so its checks can
/// never drift from the launcher's silent-return guards (a drifted wrapper
/// would skip the launch and leave the workspace holding the PREVIOUS layer's
/// tables).
pub const MAX_EXPERTS: usize = 512;
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
        // This layer's row of the live gate table, or null when every expert is
        // in VRAM. The up and down rows are `table_plane` and `2·table_plane`
        // entries after it.
        gate_row: *const c_void,
        table_plane: i64,
        // `u64[3][n_experts]` snapshot of the routed experts' entries (gate,
        // up, down); required when `gate_row` is set.
        snap: *mut c_void,
        // The two pinned host ranges `[lo, hi)` a remote (non-VRAM) entry lies in.
        pinned0_lo: u64,
        pinned0_hi: u64,
        pinned1_lo: u64,
        pinned1_hi: u64,
        // Tokens `[0, decode_tokens)` are decode rows (the summary's decode bit).
        decode_tokens: i32,
        // `u32[n_experts + 1]` routing summary, or null; `summary_seq` is
        // stored in its last word after every count is visible system-wide.
        summary: *mut c_void,
        summary_seq: u32,
        // `i32[n_experts][4]` remote list, or null.
        remote: *mut c_void,
        // `i32[3]` launch work counters to zero, or null.
        counters: *mut c_void,
        // The promotion ring in mapped host memory, or null: `u64[cap]` slot
        // images, `u64[cap]` log (`summary_seq << 32 | row << 16 | expert`),
        // and the `u32` head (this kernel's) and tail (the host's) counters.
        promo_slots: *const c_void,
        promo_log: *mut c_void,
        promo_head: *mut c_void,
        promo_tail: *const c_void,
        promo_cap: u32,
        // `u32[rows][n_experts]` in-flight promotion marks (mapped), with the
        // ring.
        promo_marks: *mut c_void,
        // This layer's row, recorded in the log.
        row: i32,
        // `u64[n_experts]` promotion slot per remote expert (0 = none), or null.
        remote_dst: *mut c_void,
        stream: *mut c_void,
    );
}
