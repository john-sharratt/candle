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
/// Kernel bound on the decode-scored token ranges, mirrored from
/// `MAX_DECODE_RANGES` in the `.cu`.
pub const MAX_DECODE_RANGES: usize = 32;

/// [`run_moe_bucketize`] launched the kernel.
pub const BUCKETIZE_LAUNCHED: i32 = 0;
/// [`run_moe_bucketize`]'s argument guards refused the call; nothing was written.
pub const BUCKETIZE_REFUSED: i32 = 1;
/// The launch itself returned an error.
pub const BUCKETIZE_LAUNCH_FAILED: i32 = 2;
/// An earlier launch on the calling thread had left an error pending; nothing
/// was launched.
pub const BUCKETIZE_EARLIER_FAILURE: i32 = 3;

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
        // Tokens inside one of the `decode_ranges` ranges `[decode_lo[i],
        // decode_hi[i])` are decode-scored (the summary's decode bit). Host
        // pointers, read by the launcher into the launch parameters; at most
        // `MAX_DECODE_RANGES`.
        decode_lo: *const u32,
        decode_hi: *const u32,
        decode_ranges: i32,
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
        // Mapped `u32`, with the ring: a prompt-only expert takes a slot only
        // while more than this is stocked; null = never.
        promo_reserve: *const c_void,
        // This layer's row, recorded in the log.
        row: i32,
        // `u64[n_experts]` promotion slot per remote expert (0 = none), or null.
        remote_dst: *mut c_void,
        started_rows: *mut c_void,
        ticket: u64,
        stream: *mut c_void,
    ) -> i32;
}
