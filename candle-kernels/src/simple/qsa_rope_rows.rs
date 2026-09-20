//! FFI binding for rotating QSA rows at their positions.
//!
//! The indexer's queries, and the live tail's keys on the spans scored by
//! cuBLAS, are rotated from the model's factored RoPE rungs
//! (`models::rope_schedule`) at each row's position and rung. See
//! `simple/qsa_rope_rows.cu`.
//!
//! ```text
//!   src, dst     device f32[n_rows, d], contiguous
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
}
