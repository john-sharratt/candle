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

/// Runs whose descriptor fits in kernel parameters, mirroring
/// `ROWS_SCATTER_INLINE_MAX` in `simple/rows_scatter.cu`. At or below this the
/// table rides in constant memory and the device-side copy is never read, so
/// the caller need not stage one at all.
pub const ROWS_SCATTER_INLINE_MAX: usize = 8;

extern "C" {
    /// Copy each run's `rows × words` 32-bit words from its source to its
    /// destination. Runs may target different arrays of different element
    /// types; every row width must be a multiple of 4 bytes.
    ///
    /// The grid is (column tiles, runs, row chunks), so **both** extents are
    /// needed and both are the maximum over the runs — the grid is sized by the
    /// widest and longest, and shorter or narrower runs exit on their bounds
    /// checks. Passing `rows × words` as `max_elems` (the old flat form) would
    /// size the column axis by an area and launch orders of magnitude too many
    /// blocks.
    /// With few enough runs the table is passed in **kernel parameters**
    /// (constant memory) instead of being read from the pinned arena over PCIe.
    /// That read is per block and uncached, and at a wide split's block count it
    /// was the entire cost — 54.8 µs of kernel time at 0.73% SM for a 2.1 MB
    /// copy. `host_desc` is the same words `desc` holds, which the caller has
    /// already built; pass null to force the pointer path.
    ///
    ///   desc:      device i64 `[ROWS_SCATTER_WORDS * n_runs]` (layout above)
    ///   host_desc: the same words, host-side, or null
    ///   max_elems: the widest run's `words`, for the column axis
    ///   max_rows:  the longest run's `rows`, for the row axis
    ///
    /// Returns the launch's `cudaGetLastError` code: 0 on success (and for an
    /// empty set, which launches nothing), or negative when it launched and an
    /// earlier launch on this thread had failed unread ([`status_error`]).
    pub fn run_rows_scatter(
        desc: *const i64,
        host_desc: *const i64,
        n_runs: i32,
        max_elems: i32,
        max_rows: i32,
        stream: *mut c_void,
    ) -> i32;
}

/// What a [`run_rows_scatter`] `status` of a launch of `n_runs` over
/// `max_elems` × `max_rows` says went wrong; `None` when nothing did.
pub fn status_error(status: i32, n_runs: usize, max_elems: i32, max_rows: i32) -> Option<String> {
    match status {
        0 => None,
        s if s < 0 => Some(format!(
            "rows_scatter: an earlier launch on this thread failed with CUDA error {}, and \
             nothing read it; the {n_runs}-run scatter itself ran",
            -s
        )),
        s => Some(format!(
            "rows_scatter: launch failed with CUDA error {s} ({n_runs} runs, {max_elems} words \
             wide, {max_rows} rows)"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::status_error;

    /// **Another launch's failure is not reported as this one's**, and this
    /// one's says the grid it asked for.
    #[test]
    fn a_status_names_whose_failure_it_is() {
        assert_eq!(status_error(0, 4, 32, 1), None);
        assert_eq!(
            status_error(9, 4, 32, 1).unwrap(),
            "rows_scatter: launch failed with CUDA error 9 (4 runs, 32 words wide, 1 rows)"
        );
        assert_eq!(
            status_error(-9, 4, 32, 1).unwrap(),
            "rows_scatter: an earlier launch on this thread failed with CUDA error 9, and \
             nothing read it; the 4-run scatter itself ran"
        );
    }
}
