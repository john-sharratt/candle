// FFI binding for the batched row scatter.
//
// Replaces a `slice_set` per array per session per layer — the corpus-gallery
// append — with ONE launch driven by a descriptor table. See
// `simple/rows_scatter.cu`.

use std::ffi::c_void;

/// i64 words per run in the descriptor table, mirroring `ROWS_SCATTER_WORDS` in
/// `simple/rows_scatter.cu`. Array-of-structs: run `i` occupies
/// `ROWS_SCATTER_WORDS` consecutive words.
///
/// | word | meaning                          |
/// |------|----------------------------------|
/// | 0    | source base pointer              |
/// | 1    | source row stride, in 32-bit WORDS |
/// | 2    | destination base pointer         |
/// | 3    | destination row stride, in words |
/// | 4    | rows to copy                     |
/// | 5    | words per row                    |
pub const ROWS_SCATTER_WORDS: usize = 6;

extern "C" {
    /// Copy each run's `rows × words` 32-bit words from its source to its
    /// destination. Runs may target different arrays of different element
    /// types; every row width must be a multiple of 4 bytes.
    ///   desc:      device i64 `[ROWS_SCATTER_WORDS * n_runs]` (layout above)
    ///   max_elems: the widest run's `rows × words`, for grid sizing
    pub fn run_rows_scatter(desc: *const i64, n_runs: i32, max_elems: i32, stream: *mut c_void);
}
