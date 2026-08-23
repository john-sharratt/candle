//! FFI bindings for the Gated DeltaNet kernels.
//!
//! The parity oracle is the sequential reference in
//! `candle-transformers/src/models/delta_net/mix.rs`; these kernels
//! implement the same recurrence with F32 state math (the state is a running
//! sum — half precision drifts over a long decode). The decode step fuses
//! decay, the delta-rule correction, the in-place state update, and the
//! post-update read into one launch per token; the conv step advances the
//! carried causal-conv tail in place.

use core::ffi::c_void;

/// Chunk width of the fused prefill scan — the token count between sequential
/// state carries, and the row width of the `kq` handoff buffer. 64 is the
/// width at which the chunk's strictly-lower triangle lives in shared memory.
pub const DELTA_NET_PREFILL_CHUNK: usize = 64;

/// The head width the prefill-scan kernels are compiled for (`d_k == d_v`).
/// Other widths take the tensor-op fallback in `delta_chunked`.
pub const DELTA_NET_PREFILL_DIM: usize = 128;

extern "C" {
    /// One gated-delta-rule token step per decode sequence, batched over the
    /// wave: grid `(n_v_heads, n_decode)`. Each sequence's state lives in its
    /// own allocation, so `states` is a device array of f32 base pointers
    /// (as i64) and `rows` gives each sequence's row in the wave tensors —
    /// the table one host upload per forward builds. `conved [T_wave,
    /// tok_stride]` is the conv output (Q|K normed columns leading, V
    /// trailing); `alpha`/`beta_lin [T_wave, n_v_heads]` raw gate
    /// projections; `o [T_wave, n_v_heads·d_v]`. Gates computed in-kernel;
    /// GQA broadcast by index; q scaled on load. `d_k`/`d_v` capped at 256
    /// (shared-memory staging); the launcher silently no-ops beyond that —
    /// callers validate dims at model load.
    pub fn run_delta_net_decode_step_f32(
        states: *const i64,
        conved: *const f32,
        rows: *const u32,
        alpha: *const f32,
        beta_lin: *const f32,
        dt_bias: *const f32,
        a_neg: *const f32,
        o: *mut f32,
        n_decode: i32,
        n_v_heads: i32,
        n_k_heads: i32,
        d_k: i32,
        d_v: i32,
        tok_stride: i32,
        q_scale: f32,
        stream: *mut c_void,
    );

    /// One causal depthwise-conv token step per decode sequence, batched over
    /// the wave, with the SiLU + Q|K-norm epilogue: each sequence's output
    /// row of `y [T_wave, channels]` is the post-activation operand row the
    /// decode step reads. `x [T_wave, channels]` (the wave's raw QKV rows),
    /// `tails [n_decode]` device f32 pointers to each sequence's
    /// `[channels, kwidth−1]` carried tail (shifted in place, raw values),
    /// `rows [n_decode]` wave rows. `qk_channels` (= `2·h_k·128`, a multiple
    /// of 256) bounds the l2-normed columns; `eps` is the norm's root floor.
    pub fn run_delta_net_conv_decode_f32(
        x: *const f32,
        kernel: *const f32,
        tails: *const i64,
        rows: *const u32,
        y: *mut f32,
        n_decode: i32,
        channels: i32,
        kwidth: i32,
        qk_channels: i32,
        eps: f32,
        stream: *mut c_void,
    );

    /// Write the per-matrix address arrays `cublas<t>trsmBatched` reads:
    /// `a_ptrs[i] = a_base + i·a_stride`, `b_ptrs[i] = b_base + i·b_stride`,
    /// for `i < batch`. Strides are in **elements**.
    ///
    /// cuBLAS has no strided-batched trsm, so the addresses must be
    /// materialised; doing it on the device keeps a host upload out of the
    /// chunk loop.
    pub fn run_delta_net_batch_ptrs(
        a_base: *const f32,
        a_stride: i64,
        b_base: *mut f32,
        b_stride: i64,
        a_ptrs: *mut *const f32,
        b_ptrs: *mut *mut f32,
        batch: i32,
        stream: *mut c_void,
    );

    /// Token-parallel causal conv over a whole prefill segment, with the
    /// SiLU + Q|K-norm epilogue: `y` IS the operand buffer the scan kernels
    /// read q/k/v from through strides. `x [t_len, channels]` token-major,
    /// `kernel [channels, kwidth]`, `tail [channels, kwidth−1]` (entering
    /// tail, read-only), `y [t_len, channels]`,
    /// `tail_out [channels, kwidth−1]` (the advanced RAW tail; a separate
    /// buffer because blocks computing the first `kwidth−1` outputs read the
    /// entering tail concurrently). `qk_channels`/`eps` as in the conv step.
    pub fn run_delta_net_conv_prefill_f32(
        x: *const f32,
        kernel: *const f32,
        tail: *const f32,
        y: *mut f32,
        tail_out: *mut f32,
        t_len: i32,
        channels: i32,
        kwidth: i32,
        qk_channels: i32,
        eps: f32,
        stream: *mut c_void,
    );

    /// Intra-chunk half of the fused prefill scan, parallel over
    /// (chunk, V head): in-kernel gates from the raw projections, per-chunk
    /// log-decay cumsum, the strictly-lower decay triangle, one
    /// forward-substitution solve for both right-hand sides, and the inclusive
    /// `(q_t·k_s)·D[t,s]` dot grid. `qk` and `v` are both strided views of
    /// the conv output (token stride `tok_stride` = conv_dim): `qk` at the
    /// Q|K columns (V head `h` reads K head `h % n_k_heads`, q scaled on
    /// load), `v` at the V column; `alpha`/`blin [t_len, n_v_heads]` raw;
    /// emits `u`/`w [n_v_heads, t_len, 128]`,
    /// `kq [n_v_heads, t_len, DELTA_NET_PREFILL_CHUNK]` (rows valid for
    /// `s ≤ t` only) and `g_cs [n_v_heads, t_len]`.
    pub fn run_delta_net_prefill_intra_f32(
        qk: *const f32,
        v: *const f32,
        alpha: *const f32,
        blin: *const f32,
        dt_bias: *const f32,
        a_neg: *const f32,
        u: *mut f32,
        w: *mut f32,
        kq: *mut f32,
        g_cs: *mut f32,
        t_len: i32,
        n_v_heads: i32,
        n_k_heads: i32,
        tok_stride: i32,
        q_scale: f32,
        stream: *mut c_void,
    );

    /// Sequential half of the fused prefill scan: walks the chunks in order
    /// with the state tile register-resident, computes the fused output
    /// (inter-chunk read of the pre-update state + intra-chunk reads of the
    /// chunk's own writes) directly into the span's rows of the whole-wave
    /// output `o [.., n_v_heads·128]`, and updates `state [n_v_heads, 128,
    /// 128]` in place in the stored orientation. `qk`/`tok_stride` as in the
    /// intra kernel.
    pub fn run_delta_net_prefill_state_f32(
        state: *mut f32,
        qk: *const f32,
        u: *const f32,
        w: *const f32,
        kq: *const f32,
        g_cs: *const f32,
        o: *mut f32,
        t_len: i32,
        n_v_heads: i32,
        n_k_heads: i32,
        tok_stride: i32,
        q_scale: f32,
        stream: *mut c_void,
    );

    /// Row-wise epilogue over the whole wave: per `(token, V head)` row,
    /// `out = (o / sqrt(mean(o²) + eps)) ⊙ gain ⊙ SiLU(z)` — the per-head
    /// RMS norm and the z-gate in one launch. `rows = T · n_v_heads`,
    /// `gain [d]`, `d ≤ 256`.
    pub fn run_delta_net_norm_gate_f32(
        o: *const f32,
        z: *const f32,
        gain: *const f32,
        out: *mut f32,
        rows: i32,
        d: i32,
        eps: f32,
        stream: *mut c_void,
    );
}
