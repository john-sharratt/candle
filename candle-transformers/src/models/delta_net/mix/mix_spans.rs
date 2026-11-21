//! The DeltaNet mixer over a packed buffer of several sequences, run as the
//! row-wise majority once and the two carried steps once per sequence.
//!
//! Only the causal conv, which carries a tail, and the delta rule, which carries
//! the state, are per sequence. Every other op computes each row from that row
//! alone, so running it once over the wave keeps weight reads from scaling with
//! the session count. The tensor-op path is the arithmetic the fused kernels are
//! checked against, and it reads value head `j` from key head `j % h_k`, as ggml
//! tiles the broadcast. The read scale is applied before the rule, since the
//! gated norm's epsilon makes it a real term. The spans must tile the buffer in
//! order, and this is checked, because a gap or an overlap would feed one
//! sequence another's rows.

use super::*;

/// Everything between the input projections and the output projection, over a
/// packed buffer of **several** sequences: conv+SiLU → split → l2-norm → GQA
/// broadcast → read scale → recurrence → gated norm. Returns the
/// `[T, value_dim]` gated activations the output projection consumes; each
/// sequence's state is advanced in place.
///
/// # Why this takes the whole wave
///
/// Only two steps here are per sequence — the conv, which carries a tail, and
/// the delta rule, which carries `S`. The other forty-odd ops compute each row
/// from that row alone. Running the *whole* mixer once per sequence, which is
/// what a per-sequence caller forces, therefore re-does all of that N times and,
/// worse, re-reads every projection weight N times: a decode step is weight-
/// bandwidth-bound, so N sessions cost N× a single session and batching buys
/// nothing. Measured on the 0.8B, decode went 1 → 4 contexts for 1.37× where the
/// dense models in the same harness get 3.1×.
///
/// So the row-wise majority runs once over all `T` rows and the two carried
/// steps run per sequence, each writing its slice of a shared buffer.
// `table` is the pre-uploaded per-sequence pointer table the batched decode
// kernel reads; without `cuda` there is no kernel to hand it to.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
#[allow(clippy::too_many_arguments)]
pub fn delta_net_mix_spans<'w>(
    p: &DeltaNetProjections<'w>,
    c: &DeltaNetConstants<'_>,
    dims: &DeltaNetDims,
    seqs: &mut [DeltaNetSeq<'_>],
    rms_eps: f64,
    table: Option<&DeltaNetLayerTable>,
    zgate: ZGate,
    // The open layer phase, for the per-layer tables built below.
    ticket: Option<WaveTicket>,
) -> Result<LiveTensor<'w>> {
    let (t, _) = p.qkv.dims2()?;
    // **The spans must tile the buffer exactly, in order.** The two carried
    // steps below slice their inputs by `start`/`len` and their results are
    // concatenated back in span order, so a gap, an overlap or a reordering
    // silently feeds one sequence another's rows — a wrong answer, and one that
    // looks like a model fault rather than a packing fault. Checked rather than
    // assumed: the caller derives these from the wave's own packing, and this is
    // the one place that can see both.
    check_span_tiling(seqs, t)?;
    check_span_writes(seqs)?;
    check_span_hooks(seqs)?;
    #[cfg(feature = "cuda")]
    let d = dims.head_dim;

    #[cfg(feature = "cuda")]
    if let Some(o) = fused_mix(p, c, dims, seqs, rms_eps, table, ticket)? {
        // The z-gate is a kernel template instantiation: SiLU for the Qwen3.5
        // lineage, sigmoid for qwen4exp (§12.6 — the one numerical difference
        // between the two generations' GDN).
        return super::super::cuda::delta_net_norm_gate(&o, &p.z, c.norm, d, rms_eps as f32, zgate);
    }
    mix_spans_by_ops(p, c, dims, seqs, rms_eps, zgate)
}

/// What [`delta_net_mix_spans_q8`] hands an int8 output projection: its operand
/// emitted by the fused epilogue, or — where there is no fused epilogue to emit
/// from — the F32 activations for the projection to quantize itself.
#[cfg(feature = "cuda")]
pub enum MixOut<'w> {
    /// The gated activations as F32.
    F32(LiveTensor<'w>),
    /// The gated activations as the projection's q8a128 operand, with no F32
    /// copy stored.
    Q8(Q8a128Operand<'w>),
}

/// [`delta_net_mix_spans`] for a layer whose output projection runs int8: on
/// the fused path the epilogue writes its output as the projection's q8a128
/// operand under `sum_scale` (`delta_net_norm_gate_q8`) and stores no F32 copy
/// nothing would read. The tensor-op path — CPU, a reference dtype, another
/// head geometry — has no fused epilogue to emit from and returns the F32
/// activations, leaving the projection to quantize them itself.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn delta_net_mix_spans_q8<'w>(
    p: &DeltaNetProjections<'w>,
    c: &DeltaNetConstants<'_>,
    dims: &DeltaNetDims,
    seqs: &mut [DeltaNetSeq<'_>],
    rms_eps: f64,
    table: Option<&DeltaNetLayerTable>,
    zgate: ZGate,
    ticket: Option<WaveTicket>,
    sum_scale: SumScale,
) -> Result<MixOut<'w>> {
    let (t, _) = p.qkv.dims2()?;
    check_span_tiling(seqs, t)?;
    check_span_writes(seqs)?;
    check_span_hooks(seqs)?;
    let d = dims.head_dim;
    match fused_mix(p, c, dims, seqs, rms_eps, table, ticket)? {
        Some(o) if d == 128 => Ok(MixOut::Q8(super::super::cuda::delta_net_norm_gate_q8(
            &o,
            &p.z,
            c.norm,
            d,
            rms_eps as f32,
            zgate,
            sum_scale,
        )?)),
        Some(o) => Ok(MixOut::F32(super::super::cuda::delta_net_norm_gate(
            &o,
            &p.z,
            c.norm,
            d,
            rms_eps as f32,
            zgate,
        )?)),
        None => Ok(MixOut::F32(mix_spans_by_ops(
            p, c, dims, seqs, rms_eps, zgate,
        )?)),
    }
}

/// The mixer as tensor ops — the arithmetic the fused kernels are
/// parity-locked to, for CPU, reference dtypes and any other head geometry.
pub(super) fn mix_spans_by_ops<'w>(
    p: &DeltaNetProjections<'w>,
    c: &DeltaNetConstants<'_>,
    dims: &DeltaNetDims,
    seqs: &mut [DeltaNetSeq<'_>],
    rms_eps: f64,
    zgate: ZGate,
) -> Result<LiveTensor<'w>> {
    let (t, _) = p.qkv.dims2()?;
    let (h_k, h_v, d) = (dims.n_k_heads, dims.n_v_heads, dims.head_dim);
    let (qkv, w) = (&p.qkv, c);
    let key_dim = dims.key_dim();
    // A record's rows are the fused conv's output, Q|K already normed; the ops
    // path holds no such buffer to capture from or replay into.
    if seqs.iter().any(|s| !s.hook.is_none()) {
        candle::bail!(
            "delta_net_mix_spans: a span carries a capture or replay, which runs on the fused \
             path only"
        );
    }

    // Causal conv, then SiLU.
    //
    // Per sequence, because the tail is: token `i` of a sequence convolves with
    // that sequence's own previous tokens, and its first `K−1` come from the
    // carried tail rather than from whatever sequence precedes it in the buffer.
    let conved = conv_silu_spans(qkv, w.conv, seqs)?; // [T, conv_dim]

    // The l2-normed Q | K stack.
    //
    // Q and K are adjacent in the conv output and take the same per-head l2
    // norm, so they are normed as one `[T, 2·H_k, d]` tensor and split after.
    // The norm reduces over the last axis, so treating the two projections as
    // one stack of `2·H_k` heads is the same arithmetic on each — and it halves
    // a four-launch reduction that runs twice per DeltaNet layer.
    let qk = l2_norm(
        &conved.narrow(1, 0, 2 * key_dim)?.reshape((t, 2 * h_k, d))?,
        rms_eps,
    )?;

    let beta = candle_nn::ops::sigmoid(&p.beta_lin)?; // [T, H_v]

    // g = a ⊙ softplus(α + dt_bias): per-head log-decay, ≤ 0 since a < 0.
    let g_log = softplus(&p.alpha_lin.broadcast_add(c.dt_bias)?)?.broadcast_mul(c.a)?;

    let q = qk.narrow(1, 0, h_k)?;
    let k = qk.narrow(1, h_k, h_k)?;
    let v = conved
        .narrow(1, 2 * key_dim, dims.value_dim())?
        .reshape((t, h_v, d))?;
    let group = h_v / h_k;
    // **V head `j` reads K head `j % h_k`, not `j / group`.**
    //
    // ggml broadcasts this with `ggml_repeat`, which *tiles* its source along
    // the head axis: `[k0, k1, …]` becomes `[k0, k1, …, k0, k1, …]`. The
    // blocked reading — each K head repeated `group` times in place — is the
    // other natural way to write it and is what an `expand` over an inserted
    // axis gives you, so it is easy to arrive at and impossible to tell apart
    // on any model where `h_v == h_k`.
    //
    // Which is exactly what happened: the 0.8B the reference was validated
    // against on llama.cpp has 16 K heads and 16 V heads, so `group == 1` and
    // both conventions are the identity. The 9B is 16/32 and the 32B family
    // wider still — there the two differ, and the symptom is not a crash but a
    // model that is *almost* right: it still knows that Paris is the capital
    // of France and still writes English, it just cannot hold an instruction.
    let repeat = |x: &LiveTensor<'w>| -> Result<LiveTensor<'w>> {
        // [T, H_k, d] → [T, group, H_k, d] → [T, H_v, d], so index
        // `g · h_k + i` reads K head `i` — the tiling ggml performs.
        x.unsqueeze(1)?
            .expand((t, group, h_k, d))?
            .reshape((t, h_v, d))
    };
    let (q, k) = (repeat(&q)?, repeat(&k)?);

    // Read scale, exactly as `build_delta_net_*` applies it to `q` before the
    // rule. This looks like a no-op — `o = S q` is linear in `q`, and the
    // gated RMSNorm downstream is scale-invariant — but it is not, because
    // that norm has an epsilon floor:
    //
    //   rms(o/√d) = (o/√d) / √(mean(o²)/d + ε) = o / √(mean(o²) + d·ε)
    //
    // so dropping the scale is equivalent to shrinking the norm's epsilon by
    // `d` (128×). `o` here is `β(q·k)v`, small enough that the floor is a
    // live term rather than a guard, and the difference is large enough to
    // move the argmax. Omitting this produced fluent-looking but wrong text
    // that passed every structural check.
    let q = (q * (1.0 / (d as f64).sqrt()))?;

    // The chunked parallel-scan form: identical numbers to the sequential
    // rule (locked by test at every chunk width), one state carry per chunk
    // instead of one per token. A single-token decode degenerates to one
    // 1-wide chunk, so one code path serves prefill and decode alike.
    //
    // Per sequence, because `S` is: this is the other place a row depends on
    // the rows before it *within its own sequence*. Each call reads its slice of
    // the five operands and writes its slice of one shared output.
    let lay = TimelineLayout::of(dims);
    let mut o_parts: Vec<LiveTensor<'w>> = Vec::with_capacity(seqs.len());
    for s in seqs.iter_mut() {
        // Free when the span is the whole buffer: a row range of a contiguous
        // tensor is contiguous, so `contiguous` hands the view straight back.
        let rows = |x: &LiveTensor<'w>| -> Result<LiveTensor<'w>> {
            x.narrow(0, s.start, s.len)?.contiguous()
        };
        match &s.write {
            // A short span reads the state and writes its timeline: the rows'
            // `G` and `v_new` against the entering state, the per-K-head key
            // (normed, unscaled) and the raw conv input — what the kernels
            // write, by the same algebra.
            SpanWrite::Timeline(tl) => {
                let (o, g_cs, v_new) = delta_span_rows(
                    &s.state.s,
                    &rows(&q)?,
                    &rows(&k)?,
                    &rows(&v)?,
                    &rows(&g_log)?,
                    &rows(&beta)?,
                )?;
                let k_heads = qk.narrow(1, h_k, h_k)?.narrow(0, s.start, s.len)?;
                lay.write_rows(tl, &g_cs, &k_heads, &v_new, &rows(qkv)?)?;
                o_parts.push(o);
            }
            // A long span advances into its write buffers — the prefill
            // scratch, which `commit_wave` exchanges in — and leaves the live
            // state as it entered. On the single-span entry points the two are
            // one buffer, and the advance is in place.
            SpanWrite::State(out) => {
                if !out.s.same_storage(&s.state.s) {
                    out.s.slice_set(&s.state.s, 0, 0)?;
                }
                let mut advanced = out.s.clone();
                o_parts.push(delta_chunked(
                    &mut advanced,
                    &rows(&q)?,
                    &rows(&k)?,
                    &rows(&v)?,
                    &rows(&g_log)?,
                    &rows(&beta)?,
                    DELTA_CHUNK,
                )?);
            }
        }
    }
    let o = LiveTensor::cat(&o_parts, 0)?;
    drop(o_parts);

    // Gated per-head norm, then flatten for the output projection.
    let z_heads = p.z.reshape((t, h_v, d))?;
    let z_gate = match zgate {
        ZGate::Silu => silu(&z_heads)?,
        ZGate::Sigmoid => candle_nn::ops::sigmoid(&z_heads)?,
    };
    let gated = rms_norm_per_head(&o, w.norm, rms_eps)?.mul(&z_gate)?;
    gated.reshape((t, dims.value_dim()))
}
