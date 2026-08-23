// FFI binding for the fused Indexer score reduction.
//
// Collapses the seven eager launches of the two-stage selection's precision
// stage — relu, per-head weight, sum over the head axis, and the four ops that
// build and add the padding mask — into one. Shared by the decode wave's
// `two_stage_select_batched` and prefill's `batched_causal_select_device`; see
// `simple/indexer_score.cu`.

use std::ffi::c_void;

extern "C" {
    /// `out[b, j] = Σ_h relu(scores[b, h, j]) · w[b, h]
    ///              + (j < counts[b] ? 0 : -1e30)`.
    ///   scores: device f32 `[b, h, m]`, read through `sc_s{b,h,m}` (ELEMENTS)
    ///   w:      device f32 `[b, h]`, read through `w_s{b,h}`
    ///   counts: device u32 `[b]`, read through `cnt_s`, or NULL to leave every
    ///           column unmasked
    ///   out:    device f32[b * m], packed
    ///
    /// `h == 0` is a valid call and writes the reduction over zero heads (0,
    /// plus the mask) rather than leaving `out` untouched — the caller allocates
    /// it uninitialised under hot-path invariant 6.
    #[allow(clippy::too_many_arguments)]
    pub fn run_indexer_score_reduce(
        scores: *const f32,
        w: *const f32,
        counts: *const u32,
        out: *mut f32,
        b: i32,
        h: i32,
        m: i32,
        sc_sb: i64,
        sc_sh: i64,
        sc_sm: i64,
        w_sb: i64,
        w_sh: i64,
        cnt_s: i64,
        stream: *mut c_void,
    );
}
