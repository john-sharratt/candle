//! FFI bindings for the fused PLE gate and conv kernels.
//!
//! Qwen3.8-Flash-Next's per-layer n-gram embedding block, which runs once per
//! forward over every row of the wave. See `simple/ple_fused.cu` for the algebra
//! and for what each launch replaces; the reference they are asserted against is
//! `models::qwen4exp::ple`'s eager chain, which stays as the CPU implementation.
//!
//! All pointers are device pointers on `stream`; every buffer is F32.

use std::ffi::c_void;

/// Words in one span descriptor `run_ple_conv` reads: `start`, `len`, the
/// history pointer, the new-history pointer — mirrors `SpanDesc` in
/// `simple/ple_fused.cu`.
pub const PLE_SPAN_WORDS: usize = 4;

extern "C" {
    /// The keyed injection and the conv's input, per (row, stream):
    /// `res[t,s,·] += gated` in place, and `normalized[t, s·d + ·]` written.
    ///
    ///   kv `[n, kv_stride]` (key `[hc·d]` then value `[d]`), res `[n, hc, d]`,
    ///   gains `[hc·d]` each, normalized `[n, hc·d]`
    ///
    /// `vec_ok` states that every base is 16-byte aligned and `kv_stride` a
    /// multiple of four.
    pub fn run_ple_gate(
        kv: *const f32,
        res: *mut f32,
        gk: *const f32,
        gq: *const f32,
        gc: *const f32,
        normalized: *mut f32,
        n: i32,
        hc: i32,
        d: i32,
        kv_stride: i32,
        eps: f32,
        inv_sqrt_d: f32,
        vec_ok: i32,
        stream: *mut c_void,
    );

    /// The dilated causal conv and its SiLU, added into `res` in place, then
    /// each sequence's next history written into its span's `new_hist`.
    ///
    ///   normalized `[n, hcd]`, wt `[kern, hcd]` (the checkpoint's `[hcd, kern]`
    ///   transposed), res `[n, hcd]`; `spans` a device array of `n_spans`
    ///   [`PLE_SPAN_WORDS`]-word descriptors tiling `0..n`
    ///
    /// `vec_ok` states that every base — including each span's two history
    /// pointers — is 16-byte aligned.
    pub fn run_ple_conv(
        normalized: *const f32,
        wt: *const f32,
        spans: *const c_void,
        res: *mut f32,
        n: i32,
        n_spans: i32,
        hcd: i32,
        kern: i32,
        dil: i32,
        hist_rows: i32,
        vec_ok: i32,
        stream: *mut c_void,
    );
}
