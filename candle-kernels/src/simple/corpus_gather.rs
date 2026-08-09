// FFI binding for the fused compressed-corpus hot-cache gather.
//
// `run_corpus_gather_rows` gathers a session's `k` selected gallery rows —
// `nope_i8` `[len,448]` u8, `nope_scale` `[len,14]` f32, `rope_bf` `[len,64]`
// bf16, `pos` `[len]` u32 — into a shared output block at `row_offset`, in ONE
// launch (one block per gathered row). Replaces the per-region `index_select` +
// cross-session `cat` the decode wave used to assemble its corpus cache. See
// `simple/corpus_gather.cu`.

use std::ffi::c_void;

extern "C" {
    /// Batched form: gather EVERY hot decode session's selected rows in ONE
    /// launch. `ptrs` is the packed device pointer table `[5*n_hot]`
    /// (`nope|scale|rope|pos|gids` base addresses, each `n_hot`); `meta` is
    /// `[2*n_hot]` (`out_off|cnt`, each `n_hot`). Byte-identical per row to
    /// `run_corpus_gather_rows`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_corpus_gather_rows_batched(
        ptrs: *const i64,
        meta: *const u32,
        out_nope: *mut u32,
        out_scale: *mut u32,
        out_rope: *mut u32,
        out_pos: *mut u32,
        n_hot: i32,
        max_k: i32,
        stream: *mut c_void,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    pub fn run_corpus_gather_rows(
        nope_i8: *const u32,
        nope_scale: *const u32,
        rope_bf: *const u32,
        pos: *const u32,
        gids: *const u32,
        out_nope: *mut u32,
        out_scale: *mut u32,
        out_rope: *mut u32,
        out_pos: *mut u32,
        k: i32,
        row_offset: i32,
        stream: *mut c_void,
    ) -> i32;
}
