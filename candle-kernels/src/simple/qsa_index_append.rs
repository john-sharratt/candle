//! FFI binding for the QSA index-cache append.
//!
//! `run_qsa_index_append` prepares one indexer key per completing block —
//! pool, then RMS-norm — for every sequence in a wave in ONE launch, writing
//! each key straight into its sequence's cache. `run_qsa_index_carry` moves the
//! trailing rows that did not complete a block into each sequence's open-block
//! buffer. Together they replace the ~30 launches a sequence used to cost per
//! layer per wave; see `simple/qsa_index_append.cu` for the measurement and the
//! descriptor-table argument.
//!
//! Keys are stored **un-rotated**: the scorer rotates each one as it loads it.
//!
//!   jobs      device i64[n_jobs * JOB_WORDS]  — {dst, src0, n0, src1}
//!   k_norm    device f32[d]
//!   carries   device i64[n_carry * CARRY_WORDS] — {dst, src, rows}

use std::ffi::c_void;

/// i64 words per job in the append table. Mirrors `QSA_APPEND_JOB_WORDS`.
pub const JOB_WORDS: usize = 4;
/// i64 words per entry in the carry table. Mirrors `QSA_APPEND_CARRY_WORDS`.
pub const CARRY_WORDS: usize = 3;
/// i64 words per flush job — `{dst, src, count}`. Mirrors
/// `QSA_FLUSH_JOB_WORDS`.
pub const FLUSH_WORDS: usize = 3;
/// The widest channel count one block can own, from the kernel's `MAX_D`. The
/// launch uses `d` threads, so this is also the block-size bound.
pub const MAX_D: usize = 1024;

extern "C" {
    pub fn run_qsa_index_append(
        jobs: *const i64,
        k_norm: *const f32,
        d: i32,
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
    pub fn run_qsa_index_flush(
        jobs: *const i64,
        k_norm: *const f32,
        d: i32,
        eps: f32,
        n_jobs: i32,
        stream: *mut c_void,
    );
}
