//! FFI bindings for the Gated DeltaNet kernels.
//!
//! The parity oracle is the sequential reference in
//! `candle-transformers/src/models/delta_net/mix.rs`; these kernels
//! implement the same recurrence with F32 state math (the state is a running
//! sum — half precision drifts over a long decode). A span of at most
//! [`DELTA_NET_TIMELINE_ROWS`] rows — a decode row, a verify block, a short
//! prompt — reads its state and writes a **timeline** of the generators its
//! tokens add (`docs/deltanet_timeline.md`); [`run_delta_net_fold_f32`] applies
//! the rows a step keeps. A longer span runs the conv + scan pair and writes
//! its advanced state to a separate buffer.

use core::ffi::c_void;

/// Chunk width of the fused prefill scan — the token count between sequential
/// state carries, and the row width of the `kq` handoff buffer. 64 is the
/// width at which the chunk's strictly-lower triangle lives in shared memory.
pub const DELTA_NET_PREFILL_CHUNK: usize = 64;

/// The head width the prefill-scan kernels are compiled for (`d_k == d_v`).
/// Other widths take the tensor-op fallback in `delta_chunked`.
pub const DELTA_NET_PREFILL_DIM: usize = 128;

/// The most spans one prefill launch covers — the grid's z extent. The
/// launchers refuse a table past it.
pub const DELTA_NET_MAX_SPANS: usize = 65_535;

/// The longest span [`run_delta_net_short_span_f32`] takes: a speculative
/// verify block (the drafts plus the accepted token) with room to spare. The
/// kernel's per-row buffers are sized by it (`DNS_ROWS`).
pub const DELTA_NET_SHORT_SPAN_ROWS: usize = 8;

/// Rows a timeline holds (`DNT_ROWS`) — every span the short-span kernel runs
/// keeps one, so the two are one number.
pub const DELTA_NET_TIMELINE_ROWS: usize = DELTA_NET_SHORT_SPAN_ROWS;

/// The causal-conv width the short-span kernel is compiled for (`DNS_KW`).
pub const DELTA_NET_SHORT_SPAN_CONV: usize = 4;

/// The widest causal-conv window [`run_delta_net_conv_prefill_f32`] takes: each
/// thread holds its channel's window in registers sized by it (`DNC_KMAX`), and
/// the launcher refuses a wider kernel.
pub const DELTA_NET_PREFILL_CONV_MAX: usize = 8;

extern "C" {
    /// One gated-delta-rule token step per decode sequence, batched over the
    /// wave: grid `(n_v_heads, n_decode)`. Each sequence's state lives in its
    /// own allocation, so `states` is a device array of f32 base pointers
    /// (as i64) and `rows` gives each sequence's row in the wave tensors —
    /// the table one host upload per forward builds. `conved [T_wave,
    /// tok_stride]` is the conv output (Q|K normed columns leading, V
    /// trailing); `alpha`/`beta_lin [T_wave, n_v_heads]` raw gate
    /// projections; `o [T_wave, n_v_heads·d_v]`. Gates computed in-kernel;
    /// GQA broadcast by index; q scaled on load.
    ///
    /// The state is read and never written: the token's generators go to row 0
    /// of each sequence's timeline (`timelines`, f32 pointers as i64), for
    /// [`run_delta_net_fold_f32`]. `d_k == d_v == 128` and `tok_stride ==
    /// (2·h_k + h_v)·128`; returns 1 with nothing written otherwise, 0 when
    /// launched.
    pub fn run_delta_net_decode_step_f32(
        states: *const i64,
        timelines: *const i64,
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
    ) -> i32;

    /// One causal depthwise-conv token step per decode sequence, batched over
    /// the wave, with the SiLU + Q|K-norm epilogue: each sequence's output
    /// row of `y [T_wave, channels]` is the post-activation operand row the
    /// decode step reads. `x [T_wave, channels]` (the wave's raw QKV rows),
    /// `tails [n_decode]` device f32 pointers to each sequence's entering
    /// `[channels, kwidth−1]` tail (read only), `timelines [n_decode]` each
    /// sequence's timeline, whose row 0 receives the token's raw input,
    /// `rows [n_decode]` wave rows. `qk_channels` (= `2·h_k·128`, a multiple
    /// of 256) bounds the l2-normed columns; `eps` is the norm's root floor.
    /// Returns 1 with nothing written for a refused shape, 0 when launched.
    pub fn run_delta_net_conv_decode_f32(
        x: *const f32,
        kernel: *const f32,
        tails: *const i64,
        timelines: *const i64,
        rows: *const u32,
        y: *mut f32,
        n_decode: i32,
        channels: i32,
        kwidth: i32,
        qk_channels: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;

    /// Apply the first `m[i]` rows of entry `i`'s timeline to its state and
    /// conv tail, in place, for every entry in one launch: grid
    /// `(n_v_heads, 128/32, n)`. `table [4, n]` i64 holds each entry's conv
    /// tail, state and timeline (f32 pointers), then its row count `m[i]` in
    /// `1..=DELTA_NET_TIMELINE_ROWS` — one upload for the whole fold. The state
    /// update is the scan's state phase over those rows, bit for bit; the tail
    /// becomes the last `kwidth−1` of `[tail | x[0..m)]`. Returns 1 with
    /// nothing written for a refused shape, 0 when launched.
    pub fn run_delta_net_fold_f32(
        table: *const i64,
        n: i32,
        n_v_heads: i32,
        n_k_heads: i32,
        channels: i32,
        kwidth: i32,
        stream: *mut c_void,
    ) -> i32;

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
    /// SiLU + Q|K-norm epilogue, for every span of a wave in one launch: `y` IS
    /// the operand buffer the scan kernels read q/k/v from through strides.
    /// `x_wave [T_wave, channels]` token-major, `kernel [channels, kwidth]`,
    /// `y_wave [T_wave, channels]`; `ptrs [4, n_spans]` each span's entering
    /// tail (read-only), advanced tail, entering state and advanced state, and
    /// `spans [2, n_spans]` its first row and length. The advanced RAW tail is a
    /// separate buffer because blocks computing the first `kwidth−1` outputs
    /// read the entering tail concurrently. `qk_channels`/`eps` as in the conv
    /// step. `kwidth` is at most [`DELTA_NET_PREFILL_CONV_MAX`] and `n_spans` at
    /// most [`DELTA_NET_MAX_SPANS`]; a refused shape returns `1` with nothing
    /// written (`0` when launched).
    pub fn run_delta_net_conv_prefill_f32(
        x_wave: *const f32,
        kernel: *const f32,
        y_wave: *mut f32,
        ptrs: *const i64,
        spans: *const u32,
        n_spans: i32,
        max_len: i32,
        channels: i32,
        kwidth: i32,
        qk_channels: i32,
        eps: f32,
        stream: *mut c_void,
    ) -> i32;

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
    /// `s ≤ t` only) and `g_cs [n_v_heads, t_len]`. The q/k rows are read four
    /// floats at a time, so `qk_wave` must be 16-byte aligned.
    pub fn run_delta_net_prefill_intra_f32(
        qk_wave: *const f32,
        v_wave: *const f32,
        alpha_wave: *const f32,
        blin_wave: *const f32,
        dt_bias: *const f32,
        a_neg: *const f32,
        u: *mut f32,
        w: *mut f32,
        kq: *mut f32,
        g_cs: *mut f32,
        spans: *const u32,
        n_spans: i32,
        max_len: i32,
        t_tran: i32,
        n_v_heads: i32,
        n_k_heads: i32,
        tok_stride: i32,
        q_scale: f32,
        stream: *mut c_void,
    );

    /// Sequential half of the fused prefill scan: walks the chunks in order
    /// with the state tile resident in shared memory, computes the fused output
    /// (inter-chunk read of the pre-update state + intra-chunk reads of the
    /// chunk's own writes) directly into the span's rows of the whole-wave
    /// output `o [.., n_v_heads·128]`, and writes each span's advanced state
    /// `[n_v_heads, 128, 128]` in the stored orientation to its state-out
    /// (`ptrs` as in [`run_delta_net_conv_prefill_f32`]), leaving the entering
    /// state as it was. `qk`/`tok_stride` as in the intra kernel.
    ///
    /// The pass reads `qk`, `u`, `w`, `kq` and every span's states four floats
    /// at a time, so each of those bases must be 16-byte aligned (`tok_stride`
    /// is a multiple of 128, which keeps every row aligned with its base).
    pub fn run_delta_net_prefill_state_f32(
        qk_wave: *const f32,
        u: *const f32,
        w: *const f32,
        kq: *const f32,
        g_cs: *const f32,
        o_wave: *mut f32,
        ptrs: *const i64,
        spans: *const u32,
        n_spans: i32,
        t_tran: i32,
        n_v_heads: i32,
        n_k_heads: i32,
        tok_stride: i32,
        q_scale: f32,
        stream: *mut c_void,
    );

    /// The conv, the intra-chunk solve and the state pass's reads of the
    /// prefill scan in one launch, for spans of at most
    /// [`DELTA_NET_SHORT_SPAN_ROWS`] rows: grid `(n_v_heads, 128/32, n_spans)`,
    /// each block convolving the q/k/v rows it reads itself and keeping every
    /// intermediate on chip. Writes exactly what the three-launch path writes
    /// into `o` — the same bits — and each span's timeline rows (G, k, v_new
    /// and the raw conv input per row); it writes neither state nor conv tail,
    /// and no conv-output rows.
    ///
    /// `x_wave [T_wave, channels]` is the raw QKV projection (token-major),
    /// `kernel [channels, DELTA_NET_SHORT_SPAN_CONV]`; `alpha`/`blin
    /// [T_wave, n_v_heads]` raw; `o_wave [T_wave, n_v_heads·128]`; `ptrs
    /// [3, n_spans]` each span's conv tail, state and timeline, `spans
    /// [2, n_spans]` its first row and length. `max_len` is the table's longest
    /// span. Returns 0 when launched, 1 when the shape is refused and nothing
    /// was written: `max_len` outside `1..=DELTA_NET_SHORT_SPAN_ROWS`, `kwidth`
    /// other than [`DELTA_NET_SHORT_SPAN_CONV`], or `channels` not
    /// `(2·h_k + h_v)·128`.
    pub fn run_delta_net_short_span_f32(
        x_wave: *const f32,
        kernel: *const f32,
        alpha_wave: *const f32,
        blin_wave: *const f32,
        dt_bias: *const f32,
        a_neg: *const f32,
        o_wave: *mut f32,
        ptrs: *const i64,
        spans: *const u32,
        n_spans: i32,
        max_len: i32,
        n_v_heads: i32,
        n_k_heads: i32,
        channels: i32,
        kwidth: i32,
        eps: f32,
        q_scale: f32,
        stream: *mut c_void,
    ) -> i32;

    /// Row-wise epilogue over the whole wave: per `(token, V head)` row,
    /// `out = (o / sqrt(mean(o²) + eps)) ⊙ gain ⊙ zgate(z)` — the per-head
    /// RMS norm and the z-gate in one launch. `rows = T · n_v_heads`,
    /// `gain [d]`, `d ≤ 256`. `sigmoid_gate` selects the gate instantiation:
    /// 0 = SiLU(z) (the Qwen3.5 lineage), nonzero = sigmoid(z) (qwen4exp).
    ///
    /// `q8`, when non-null, also receives `out` as a q8a1024 operand (row-major
    /// tiles, the bytes a standalone quantize of `out` writes; `sum_norm` its
    /// Σx convention) — only at `d = 128`, one tile per row. Returns 0 when
    /// launched, 1 when the shape is refused and nothing was written.
    pub fn run_delta_net_norm_gate_f32(
        o: *const f32,
        z: *const f32,
        gain: *const f32,
        out: *mut f32,
        rows: i32,
        d: i32,
        eps: f32,
        sigmoid_gate: i32,
        q8: *mut u8,
        sum_norm: i32,
        stream: *mut c_void,
    ) -> i32;
}
