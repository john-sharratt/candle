//! FFI bindings for the Gated-Residual fused pre-mix / combine kernels.
//!
//! Qwen3.8-Flash-Next's hyper-connection bracket, which runs 96 times per
//! forward (48 layers × 2 sub-blocks) and has no layer norm to hide behind.
//! See `simple/gr_hyper.cu` for the algebra and for why these are new kernels
//! rather than an extension of `hyper_mhc.cu`; the reference they are asserted
//! against is `models::qwen4exp::hyper`'s eager path, which stays as the CPU
//! implementation.
//!
//! All pointers are device pointers on `stream`; every buffer is F32.
//!
//! `vec_ok` is the caller's statement that every wide operand's base is
//! 16-byte aligned, which decides whether the kernels take their `float4`
//! path. Operands arrive as views and a view's base carries its start offset,
//! so the row width alone cannot answer it — see `gr_hyper.cu`.

use std::ffi::c_void;

/// The widest hyper-connection count [`run_gr_combine`] is built for, mirroring
/// `GR_MAX_HC` in `simple/gr_hyper.cu`, where the per-stream weights are held in
/// a fixed-size register array.
///
/// The kernel's own bound check can only `return` — the launcher is
/// `extern "C" void` — and its destination is allocated uninitialised, so a
/// caller that ignores this constant propagates an uninitialised residual with
/// no error. Refuse it host-side, as `MHC_MAX_HC` is refused.
pub const GR_MAX_HC: usize = 16;

extern "C" {
    /// Grouped RMS norm over the wide residual, with the `[hc·d]` gain folded
    /// in: `xn[t, s·d + j] = x[t,s,j] · rsqrt(mean_j(x[t,s,·]²) + eps) · gain[s·d + j]`.
    ///
    ///   x    `[n, hc, d]`, gain `[hc·d]`, xn `[n, hc·d]`
    pub fn run_gr_norm(
        x: *const f32,
        gain: *const f32,
        xn: *mut f32,
        n: i32,
        hc: i32,
        d: i32,
        eps: f32,
        vec_ok: i32,
        stream: *mut c_void,
    );

    /// The read collapse: `mixed[t,j] = (1/hc) · Σ_s xn[t, s·d+j] · sigmoid(gate_raw[t, s·d+j])`.
    ///
    ///   xn/gate_raw `[n, hc·d]`, mixed `[n, d]`
    pub fn run_gr_mix(
        xn: *const f32,
        gate_raw: *const f32,
        mixed: *mut f32,
        n: i32,
        hc: i32,
        d: i32,
        vec_ok: i32,
        stream: *mut c_void,
    );

    /// The write scatter, into a fresh residual:
    /// `out[t,s,j] = res[t,s,j] + block_out[t,j] · 2·sigmoid(inject[t,s] / hc)`.
    ///
    ///   res/out `[n, hc, d]`, block_out `[n, d]`, inject `[n, hc]` (`hc ≤ 16`)
    pub fn run_gr_combine(
        res: *const f32,
        block_out: *const f32,
        inject: *const f32,
        out: *mut f32,
        n: i32,
        hc: i32,
        d: i32,
        vec_ok: i32,
        stream: *mut c_void,
    );
}
