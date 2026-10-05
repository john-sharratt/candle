//! FFI binding for the QSA block-selection kernel.
//!
//! `run_qsa_topk_entries` turns one row of indexer scores per query into that
//! query's packed selection — the device half of
//! `models::qwen4exp::qsa_select::selection_entries`, whose tests pin the two
//! against each other. See `simple/qsa_topk.cu` for the ordering argument, the
//! buffer bound and the stratified windows.
//!
//!   scores          device f32[n_rows * score_stride], valid on `[0, n_cand[r])`
//!   n_cand/qpos     device u32[n_rows]
//!   tail_len        device u32[n_rows] — cells of its own block the query has
//!   prompt_blocks   device u32[n_rows] — blocks wholly inside the system prompt
//!   entries         device u32[n_rows * entry_stride] — ascending by block
//!   cnt             device u32[n_rows] — entries written, or `DENSE_ROW`
//!   window_blocks   candidate blocks per window; `0` is one window
//!   recent_blocks   the recent span nearest the query
//!   recent_forced   `1` attends the recent span whole; `0` ranks it everywhere
//!   ent_cap         the union buffer, a power of two `≤ MAX_ENTRIES`, at least
//!                   every window's keep
//!   cand_max        the most candidate blocks any row has — sizes the split
//!   split_keys      device scratch of `SPLIT_KEYS` u64, used when a launch is
//!                   too narrow to fill the device and splits each row across
//!                   blocks (`qsa_topk_split_parts`); every key it reads back
//!                   was written by the same launch. Null runs the single pass

use std::ffi::c_void;

/// The kernel's survivor bound: `top_k / ratio + 1` must not exceed it, or the
/// streaming buffer could not absorb a chunk. Mirrors `qsa_topk::MAX_KEEP` in
/// the `.cu` — the released checkpoint's 2048/4 sits at 513.
pub const MAX_KEEP: usize = 1024 - 256;

/// The kernel's union buffer ceiling, in entries. Mirrors
/// `qsa_topk::MAX_ENTRIES` in the `.cu`: with the survivor buffer it is the
/// kernel's dynamic shared memory, 8 KiB + 64 KiB.
pub const MAX_ENTRIES: usize = 16384;

/// The most blocks a split launch runs. Mirrors `qsa_topk::SPLIT_BLOCKS`.
pub const SPLIT_BLOCKS: usize = 512;

/// The split launch's scratch, in u64 keys: `MAX_KEEP` per block.
pub const SPLIT_KEYS: usize = SPLIT_BLOCKS * MAX_KEEP;

extern "C" {
    /// How many slices each window of a launch over `n_rows` rows splits into;
    /// `0` runs the single pass. Depends on the current device's SM count.
    pub fn qsa_topk_split_parts(
        n_rows: i32,
        cand_max: i32,
        window_blocks: i32,
        top_k: i32,
        ratio: i32,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    pub fn run_qsa_topk_entries(
        scores: *const f32,
        score_stride: i32,
        n_cand: *const u32,
        qpos: *const u32,
        tail_len: *const u32,
        prompt_blocks: *const u32,
        entries: *mut u32,
        entry_stride: i32,
        cnt: *mut u32,
        ratio: i32,
        top_k: i32,
        window_blocks: i32,
        recent_blocks: i32,
        recent_forced: i32,
        ent_cap: i32,
        cand_max: i32,
        split_keys: *mut c_void,
        n_rows: i32,
        stream: *mut c_void,
    );
}
