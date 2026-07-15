//! FFI binding for the batched Binary Directional Provenance (BDP) flat scan.
//!
//! One launch scores a whole wave of probes against a resident gallery, emitting
//! the per-(query-token, layer-group) `(leading_case, z*margin)` pairs. The needle
//! gate + per-case tally run host-side (identical to the CPU `score_packed`), so
//! the output is bit-identical to the CPU reference. See `bdp_scan.cu`.

use core::ffi::c_void;

extern "C" {
    /// Launch the batched BDP scan on `stream`.
    ///
    /// - `gallery_words`: `n_tokens * wpt` sign words (token-major), device ptr.
    /// - `gallery_case`: `n_tokens` case ids, device ptr.
    /// - `probe_words`: `n_probe_tokens * wpt` sign words — every request's probe
    ///   tokens concatenated (the batch), device ptr.
    /// - `n_groups` / `gw` / `wpt`: folded-signature geometry (`wpt = n_groups*gw`).
    /// - `n_cases`: number of cases (slots) voted over.
    /// - `out_case` / `out_vote`: `n_probe_tokens * n_groups` outputs (device),
    ///   the leading case (`-1` if none) and its `z*margin` per (query token, group).
    /// - `stream`: the CUDA stream handle (`CudaStream::cu_stream()`), so the
    ///   kernel orders with the surrounding copies.
    pub fn run_batched_bdp_scan(
        gallery_words: *const u64,
        gallery_case: *const u32,
        probe_words: *const u64,
        n_tokens: i32,
        n_probe_tokens: i32,
        n_groups: i32,
        gw: i32,
        wpt: i32,
        n_cases: i32,
        out_case: *mut i32,
        out_vote: *mut f32,
        stream: *mut c_void,
    );
}
