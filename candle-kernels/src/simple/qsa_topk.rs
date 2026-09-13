//! FFI binding for the QSA block-selection kernel.
//!
//! `run_qsa_topk_entries` turns one row of indexer scores per query into that
//! query's packed selection — the device half of
//! `models::qwen4exp::qsa_select::selection_entries`, whose tests pin the two
//! against each other. See `simple/qsa_topk.cu` for the ordering argument and
//! the buffer bound.
//!
//!   scores       device f32[n_rows * score_stride], valid on `[0, n_cand[r])`
//!   n_cand/qpos  device u32[n_rows]
//!   entries      device u32[n_rows * entry_stride] — ascending by block
//!   cnt          device u32[n_rows] — entries written, or `DENSE_ROW`

use std::ffi::c_void;

/// The kernel's survivor bound: `top_k / ratio + 1` must not exceed it, or the
/// streaming buffer could not absorb a chunk. Mirrors `qsa_topk::MAX_KEEP` in
/// the `.cu` — the released checkpoint's 2048/4 sits at 513.
pub const MAX_KEEP: usize = 1024 - 256;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn run_qsa_topk_entries(
        scores: *const f32,
        score_stride: i32,
        n_cand: *const u32,
        qpos: *const u32,
        tail_len: *const u32,
        entries: *mut u32,
        entry_stride: i32,
        cnt: *mut u32,
        ratio: i32,
        top_k: i32,
        n_rows: i32,
        stream: *mut c_void,
    );
}
