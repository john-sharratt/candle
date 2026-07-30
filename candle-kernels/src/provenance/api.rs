//! FFI binding for the batched Binary Directional Provenance (BDP) flat scan.
//!
//! One launch scores a wave of probes against a resident gallery, emitting the
//! per-(query-token, layer-group, SEGMENT) `(leading_case, z*margin)` pairs. The
//! needle gate + per-case tally run host-side (identical to the CPU reference).
//! A segment is a code-read file / timeline: the gallery tokens are sorted by
//! segment and each segment owns a contiguous token AND case range, so z/margin
//! are per-segment (the belief scan's per-file normalization) in ONE launch. The
//! non-segmented global-z scan is `n_segments == 1`.

use core::ffi::c_void;

extern "C" {
    /// Launch the segmented batched BDP scan on `stream`.
    ///
    /// - `gallery_words`: group-major `[group][token][gw]` sign words, tokens
    ///   sorted by segment, device ptr. Total tokens = `seg_tok_start[n_segments]`.
    /// - `gallery_case`: `n_tokens` GLOBAL case ids, device ptr.
    /// - `probe_words`: `n_probe_tokens * wpt` sign words (token-major), device ptr.
    /// - `seg_tok_start` / `seg_case_start`: `n_segments+1` prefix boundaries —
    ///   segment `s` owns tokens `[seg_tok_start[s], seg_tok_start[s+1])` and cases
    ///   `[seg_case_start[s], seg_case_start[s+1])`, device ptrs.
    /// - `n_groups` / `gw` / `wpt`: folded-signature geometry (`wpt = n_groups*gw`).
    /// - `max_seg_cases`: the largest segment's case count (shared-mem stride).
    /// - `out_case` / `out_vote`: `n_probe_tokens * n_groups * n_segments` outputs
    ///   (device) — the per-(query token, group, segment) leading GLOBAL case
    ///   (`-1` if none) and its `z*margin`.
    /// - `stream`: the CUDA stream handle so the kernel orders with the copies.
    pub fn run_batched_bdp_scan(
        gallery_words: *const u64,
        gallery_case: *const u32,
        probe_words: *const u64,
        seg_tok_start: *const i32,
        seg_case_start: *const i32,
        n_probe_tokens: i32,
        n_groups: i32,
        n_segments: i32,
        max_seg_cases: i32,
        gw: i32,
        wpt: i32,
        out_case: *mut i32,
        out_vote: *mut f32,
        stream: *mut c_void,
    );
}
