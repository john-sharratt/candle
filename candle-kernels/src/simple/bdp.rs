// FFI bindings for the corpus-selection recall kernels.
//
// `run_sign_pack` packs the sign bits of `[n, dim]` f32 rows into
// `[n, ceil(dim/32)]` u32 words (bit = value >= 0). `run_bdp_recall`
// scores every corpus entry's packed signs against `n_heads` packed query
// signs by XNOR+popcount agreement, summed over heads — the training-free
// recall stage of the two-stage BDP-recall → Indexer-precision selection.
// Both run on-device; top-M / top-k selection over the outputs uses candle's
// existing sort ops. See `simple/bdp.cu`.

use std::ffi::c_void;

// Each entry returns the post-launch `cudaGetLastError` code (0 = success) —
// callers assert on it, so a bad launch can never silently hand back stale
// buffer contents.
extern "C" {
    pub fn run_sign_pack(
        x: *const f32,
        out: *mut u32,
        n: i32,
        dim: i32,
        stream: *mut c_void,
    ) -> i32;

    /// Exact top-`m` entry ids by bounded u32 key (histogram-threshold select,
    /// tie order arbitrary) — the recall shortlist selector, valid at any `g`
    /// (unlike the single-block bitonic argsort, which caps at 1024 columns).
    /// `hist` is `[bins]` u32 scratch, `meta` `[4]` u32 scratch; both are
    /// zeroed inside. Stream-ordered, no host round-trip.
    #[allow(clippy::too_many_arguments)]
    pub fn run_topm_select(
        counts: *const u32,
        hist: *mut u32,
        meta: *mut u32,
        out_ids: *mut u32,
        g: i32,
        m: i32,
        bins: i32,
        stream: *mut c_void,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    pub fn run_bdp_recall(
        q_signs: *const u32,
        signs: *const u32,
        counts: *mut u32,
        n_heads: i32,
        g: i32,
        words: i32,
        dim: i32,
        stream: *mut c_void,
    ) -> i32;
}
