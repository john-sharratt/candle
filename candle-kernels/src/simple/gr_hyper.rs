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

/// The widest hyper-connection count [`run_gr_mix`] and [`run_gr_combine`] are
/// built for, mirroring `GR_MAX_HC` in `simple/gr_hyper.cu`. Both kernels are
/// instantiated per stream count — every power of two up to this — so the
/// stream loop unrolls and the per-stream weights stay in registers.
///
/// A launcher handed any other count can only `return` — it is
/// `extern "C" void` — leaving its destination unwritten with no error. Refuse
/// it host-side with [`gr_hc_supported`].
pub const GR_MAX_HC: usize = 16;

/// A q8a128-emitting launcher ran (or had no rows to run). Mirrors `GR_Q8_LAUNCHED` in
/// `simple/gr_hyper.cu`.
pub const GR_Q8_LAUNCHED: i32 = 0;

/// A q8a128-emitting launcher refused its shape and wrote nothing. Mirrors `GR_Q8_REFUSED`.
pub const GR_Q8_REFUSED: i32 = 1;

/// Whether the mix and combine launchers are instantiated for `hc` streams.
pub const fn gr_hc_supported(hc: usize) -> bool {
    hc.is_power_of_two() && hc <= GR_MAX_HC
}

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

    /// [`run_gr_norm`] that also writes `xn` as a q8a128 operand into `q8`
    /// (`q8a1024` layout, `n·hc·d/128` tiles) — bit-identical to running the
    /// norm and then quantizing its output. `sum_norm` is `SumScale::as_code()`.
    ///
    /// Returns [`GR_Q8_LAUNCHED`], or [`GR_Q8_REFUSED`] when `d` is not a multiple of
    /// 128 — having written nothing, which the caller must treat as an error. Every
    /// operand must be 16-byte aligned; that is the caller's check, not the launcher's.
    pub fn run_gr_norm_q8(
        x: *const f32,
        gain: *const f32,
        xn: *mut f32,
        q8: *mut c_void,
        n: i32,
        hc: i32,
        d: i32,
        eps: f32,
        sum_norm: i32,
        stream: *mut c_void,
    ) -> i32;

    /// `q8 ← q8a128(silu(proj[t, 0..cols]))`, reading `proj` `[n, row_stride]`
    /// through its row stride — bit-identical to a dense `silu` then a quantize.
    ///
    /// Returns [`GR_Q8_LAUNCHED`], or [`GR_Q8_REFUSED`] when `cols` is not a multiple
    /// of 128 or exceeds `row_stride` — having written nothing. `proj` and
    /// `row_stride` must be 16-byte aligned; that is the caller's check.
    pub fn run_gr_silu_q8(
        proj: *const f32,
        q8: *mut c_void,
        n: i32,
        cols: i32,
        row_stride: i32,
        sum_norm: i32,
        stream: *mut c_void,
    ) -> i32;

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

    /// [`run_gr_mix`] that also writes `mixed` as a q8a128 operand into `q8`
    /// (`q8a1024` layout, `n·d/128` tiles) — bit-identical to the collapse then a
    /// quantize.
    ///
    /// Returns [`GR_Q8_LAUNCHED`], or [`GR_Q8_REFUSED`] when `d` is not a multiple of
    /// 128 or `hc` is not one of 1, 2, 4, 8, 16 — having written nothing. Every
    /// operand must be 16-byte aligned; that is the caller's check.
    pub fn run_gr_mix_q8(
        xn: *const f32,
        gate_raw: *const f32,
        mixed: *mut f32,
        q8: *mut c_void,
        n: i32,
        hc: i32,
        d: i32,
        sum_norm: i32,
        stream: *mut c_void,
    ) -> i32;

    /// The write scatter, in place on the residual:
    /// `res[t,s,j] += block_out[t,j] · 2·sigmoid(inject[t,s] / hc)`.
    ///
    ///   res `[n, hc, d]`, block_out `[n, d]`, inject `[n, hc]` (`hc ≤ 16`) with
    ///   its rows `inject_stride ≥ hc` elements apart
    pub fn run_gr_combine(
        res: *mut f32,
        block_out: *const f32,
        inject: *const f32,
        n: i32,
        hc: i32,
        d: i32,
        inject_stride: i32,
        vec_ok: i32,
        stream: *mut c_void,
    );

    /// [`run_gr_combine`] with the MoE's block output assembled in the same pass:
    /// `res[t,s,j] += (routed[t,j] + shared[t,j] · sigmoid(gate[t])) · 2·sigmoid(inject[t,s] / hc)`
    /// — bit-identical to the sigmoid, broadcast multiply and add it replaces.
    ///
    ///   routed/shared `[n, d]`, gate `[n]` with rows `gate_stride ≥ 1` elements
    ///   apart, the rest as [`run_gr_combine`]
    pub fn run_gr_combine_gated(
        res: *mut f32,
        routed: *const f32,
        shared: *const f32,
        gate: *const f32,
        inject: *const f32,
        n: i32,
        hc: i32,
        d: i32,
        inject_stride: i32,
        gate_stride: i32,
        vec_ok: i32,
        stream: *mut c_void,
    );
}
