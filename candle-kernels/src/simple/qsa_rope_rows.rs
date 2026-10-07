//! FFI binding for rotating QSA rows at their positions.
//!
//! The indexer's queries, and the live tail's keys on the spans scored by
//! cuBLAS, are rotated from the model's factored RoPE rungs
//! (`models::rope_schedule`) at each row's position and rung. See
//! `simple/qsa_rope_rows.cu`.
//!
//! ```text
//!   src, dst     device f32[n_rows, d], contiguous
//!   src_pages    device i64[⌈n_rows / rows_per_src_page⌉] page base addresses,
//!                or null to read `src`; row r at
//!                src_pages[r / rows_per_src_page] + (r % rows_per_src_page)·d
//!   pos          device u32[n_rows / rows_per_pos], or null for
//!                pos(group) = pos_base + group · pos_step
//!   rungs        every rung's table and m², by value
//!   group_rung   device u32[n_rows / rows_per_pos], or null for `rung`
//!   q_scale      nonzero: the rotated channels take the rung's m²
//! ```

use std::ffi::c_void;

use crate::rope::RopeRungsFfi;

extern "C" {
    #[allow(clippy::too_many_arguments)]
    pub fn run_qsa_rope_rows(
        src: *const f32,
        src_pages: *const i64,
        rows_per_src_page: i32,
        dst: *mut f32,
        n_rows: i32,
        d: i32,
        rows_per_pos: i32,
        pos: *const u32,
        pos_base: i64,
        pos_step: i32,
        rungs: RopeRungsFfi,
        group_rung: *const u32,
        rung: u32,
        q_scale: i32,
        stream: *mut c_void,
    );

    /// The rows of a dense `src`, RMS-normed — `x / sqrt(Σx²·(1/d) + eps) · norm_w`
    /// — then rotated exactly as [`run_qsa_rope_rows`] rotates them, in one launch.
    /// `d` must not exceed [`QSA_ROPE_NORM_MAX_D`]; the kernel writes nothing for a
    /// shape outside its bounds, which the caller checks first.
    #[allow(clippy::too_many_arguments)]
    pub fn run_qsa_rope_rows_norm(
        src: *const f32,
        dst: *mut f32,
        n_rows: i32,
        d: i32,
        rows_per_pos: i32,
        pos: *const u32,
        pos_base: i64,
        pos_step: i32,
        rungs: RopeRungsFfi,
        group_rung: *const u32,
        rung: u32,
        q_scale: i32,
        norm_w: *const f32,
        eps: f32,
        stream: *mut c_void,
    );
}

/// The widest row [`run_qsa_rope_rows_norm`] takes, mirroring `NORM_MAX_D` in
/// `simple/qsa_rope_rows.cu`.
pub const QSA_ROPE_NORM_MAX_D: usize = 256;
