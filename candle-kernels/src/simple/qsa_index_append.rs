//! FFI binding for the QSA index-cache append.
//!
//! `run_qsa_index_append` prepares one indexer key per completing block —
//! pool, RMS-norm, RoPE — for every sequence in a wave in ONE launch, writing
//! each key straight into its sequence's cache. `run_qsa_index_carry` moves the
//! trailing rows that did not complete a block into each sequence's open-block
//! buffer. Together they replace the ~30 launches a sequence used to cost per
//! layer per wave; see `simple/qsa_index_append.cu` for the measurement and the
//! descriptor-table argument.
//!
//!   jobs      device i64[n_jobs * JOB_WORDS]  — {dst, src0, n0, src1, pos}
//!   k_norm    device f32[d]
//!   cos/sin   device f32[max_pos * rope_dim/2]
//!   carries   device i64[n_carry * CARRY_WORDS] — {dst, src, rows}

use std::ffi::c_void;

/// i64 words per job in the append table. Mirrors `QSA_APPEND_JOB_WORDS`.
pub const JOB_WORDS: usize = 5;
/// i64 words per entry in the carry table. Mirrors `QSA_APPEND_CARRY_WORDS`.
pub const CARRY_WORDS: usize = 3;
/// i64 words per flush job — `{dst, src, count, pos}`. Mirrors
/// `QSA_FLUSH_JOB_WORDS`.
pub const FLUSH_WORDS: usize = 4;
/// The widest channel count one block can own, from the kernel's `MAX_D`. The
/// launch uses `d` threads, so this is also the block-size bound.
pub const MAX_D: usize = 1024;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn run_qsa_index_append(
        jobs: *const i64,
        k_norm: *const f32,
        cos_tab: *const f32,
        sin_tab: *const f32,
        d: i32,
        rope_dim: i32,
        ratio: i32,
        eps: f32,
        n_jobs: i32,
        stream: *mut c_void,
    );

    pub fn run_qsa_index_carry(carries: *const i64, d: i32, n_carry: i32, stream: *mut c_void);

    /// Close a turn on a block boundary: pool the carried rows into ONE short
    /// block, over the count actually present rather than `ratio`.
    ///
    /// A turn's index is a self-contained page only if it ends on a block
    /// boundary, and `T mod ratio` rows are left carried otherwise — belonging
    /// to a block the next turn finishes. See the kernel's own notes.
    #[allow(clippy::too_many_arguments)]
    pub fn run_qsa_index_flush(
        jobs: *const i64,
        k_norm: *const f32,
        cos_tab: *const f32,
        sin_tab: *const f32,
        d: i32,
        rope_dim: i32,
        eps: f32,
        n_jobs: i32,
        stream: *mut c_void,
    );
}
