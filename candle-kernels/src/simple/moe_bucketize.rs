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

/// A promotion offer with no resident expert behind it, mirrored from
/// `PROMO_EMPTY` in the `.cu`.
pub const PROMO_EMPTY: u64 = u64::MAX;
/// The expert a skipped victim's log entry names, mirrored from `PROMO_SKIP`.
pub const PROMO_SKIP: u32 = 0xffff;

/// Read-ahead items one bucketize may write, mirrored from `AHEAD_MAX` in
/// `moe_read_ahead.cuh` — which documents the item layout.
pub const AHEAD_MAX: usize = 64;
/// `u64` words per read-ahead item, mirrored from `AHEAD_ITEM_WORDS`.
pub const AHEAD_ITEM_WORDS: usize = 11;
/// Pieces each read-ahead item is copied in, mirrored from `AHEAD_CHUNKS`.
pub const AHEAD_CHUNKS: usize = 16;
/// Set in a promotion log entry's expert field for a read-ahead claim,
/// mirrored from `AHEAD_FLAG`.
pub const AHEAD_FLAG: u32 = 0x4000;

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
        // Mapped `u32`, required with the ring: a launch with more claiming
        // experts than this is a sweep and claims nothing.
        promo_sweep: *const c_void,
        // Mapped, required with the ring: `u64[cap]` the victim behind each
        // offer (`row · n_experts + expert` in the gate plane, or
        // [`PROMO_EMPTY`]), and `u64[cap][3]` the entries (gate, up, down) a
        // claimed victim is retargeted to.
        promo_victims: *const c_void,
        promo_retarget: *const c_void,
        // Mapped `u32[zone_slots]` slot tags `(row + 1) << 16 | expert`, or null:
        // every VRAM entry snapshotted is checked against its slot's tag, and a
        // mismatch traps. Slot `s` spans `[zone_end - (s + 1) · zone_slot_bytes,
        // zone_end - s · zone_slot_bytes)`.
        slot_owner: *const c_void,
        zone_end: u64,
        zone_slot_bytes: u64,
        zone_slots: u32,
        // This layer's row, recorded in the log.
        row: i32,
        // `u64[n_experts]` promotion slot per remote expert (0 = none), or null.
        remote_dst: *mut c_void,
        started_rows: *mut c_void,
        ticket: u64,
        // Read-ahead, or a null `ahead_items` (which requires the ring): the
        // mapped `u32` window (slot images the link moves in one layer) and
        // depth (targets `row + 2 ..= row + depth`), the mapped prediction
        // lists `u32[rows]` counts, `u32[rows][ahead_cap]` experts and
        // `u64[rows][ahead_cap]` vetted source images — an expert is read
        // ahead only while its entries still point at that image; the device
        // `u64[rows][4]` row layout (gate, up, down offset, image bytes), item
        // buffer `u64[1 + AHEAD_MAX · AHEAD_ITEM_WORDS]` and `u32[AHEAD_MAX]`
        // piece counters.
        ahead_window: *const c_void,
        ahead_depth: *const c_void,
        ahead_n: *const c_void,
        ahead_list: *const c_void,
        ahead_src: *const c_void,
        ahead_cap: u32,
        rows: i32,
        row_layout: *const c_void,
        ahead_items: *mut c_void,
        ahead_done: *mut c_void,
        stream: *mut c_void,
    ) -> i32;
}
