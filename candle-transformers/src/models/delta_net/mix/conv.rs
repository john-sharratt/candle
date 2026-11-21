//! The causal depthwise convolution, and the normalisation and activation steps
//! around it, in the tensor-op form and in the form the fused path runs.
//!
//! The convolution carries a tail of the previous segment's inputs, so a sequence
//! split across calls gives the same output as one call over all of it. Tensors
//! are assembled in their carried form without a concatenation, because a
//! concatenation would inherit the carried tail's arena and place the layer's
//! widest buffer on the wrong span. On CUDA in F32 the conv kernels read the
//! token-major rows directly, so the channel-major transposes never exist. The
//! fused path writes each row exactly once into a buffer the mixer reads, which
//! is why that buffer is allocated uninitialised.

use super::*;

/// Causal 1-D conv over channels with a carried tail.
///
/// `x` `[C, T]` (channel-major), `kernel` `[C, K]` (a depthwise kernel per
/// channel), `tail` `[C, K − 1]` — the trailing inputs of the previous
/// segment, zeros at sequence start. Returns `(y [C, T], new_tail)`.
///
/// `y[c, t] = Σ_j kernel[c, j] · xpad[c, t + j]` where `xpad = [tail | x]` —
/// i.e. output `t` sees inputs `t − K + 1 ..= t`, the standard causal form.
pub fn causal_conv1d<'w>(
    x: &LiveTensor<'w>,
    kernel: &Tensor,
    tail: &Tensor,
) -> Result<(LiveTensor<'w>, LiveTensor<'w>)> {
    let (c, t) = x.dims2()?;
    let (ck, k) = kernel.dims2()?;
    if ck != c {
        candle::bail!("causal_conv1d: {c} channels but kernel has {ck}");
    }
    // `[tail | x]`, assembled rather than concatenated. `cat` inherits its arena
    // from its **first** argument, and that one is the carried tail — a buffer
    // the sequence owns across every wave, so it names no arena and the whole
    // conv would land on the pool. Allocating beside `x` instead puts the widest
    // buffer in this layer on the wave's span; the two writes are the same two
    // copies `cat` would have made.
    let padded = x.empty_beside((c, k - 1 + t), x.dtype())?; // [C, K-1+T]
    padded.slice_set(tail, 1, 0)?;
    padded.slice_set(x, 1, k - 1)?;
    let mut acc: Option<LiveTensor<'w>> = None;
    for j in 0..k {
        let win = padded.narrow(1, j, t)?;
        let term = win.broadcast_mul(&kernel.narrow(1, j, 1)?)?;
        acc = Some(match acc {
            Some(a) => a.add(&term)?,
            None => term,
        });
    }
    let y = acc.expect("kernel width is at least 1");
    let new_tail = padded.narrow(1, t, k - 1)?.contiguous()?;
    Ok((y, new_tail))
}

/// Per-row L2 normalization: `x / max(sqrt(Σ x²), eps)` over the last dim —
/// ggml's `ggml_l2_norm` semantics, which the llama.cpp graph applies to the
/// post-conv Q and K heads.
///
/// Note the floor is on the *root*, not on the sum (`max(√Σ, ε)`, not
/// `√max(Σ, ε)`). The two differ by `ε²` in where the clamp bites, which
/// only matters for near-zero rows — but this family's epsilon terms have
/// already proven load-bearing once (see the read scale in
/// [`delta_net_layer_forward`]), so it follows the reference exactly.
pub fn l2_norm<'w>(x: &LiveTensor<'w>, eps: f64) -> Result<LiveTensor<'w>> {
    // `maximum` against the scalar, not against a materialised tensor of it:
    // the tensor form allocated and filled a buffer the size of the reduction
    // on every call, twice per DeltaNet layer, for a constant.
    let sumsq = x.sqr()?.sum_keepdim(candle::D::Minus1)?;
    let denom = sumsq.sqrt()?.maximum(eps)?;
    x.broadcast_div(&denom)
}

/// SiLU (x · sigmoid(x)) on any shape.
pub fn silu<'w>(x: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    x.broadcast_mul(&candle_nn::ops::sigmoid(x)?)
}

/// Numerically-stable softplus: `ln(1 + eˣ) = max(x, 0) + ln(1 + e^{−|x|})`.
pub fn softplus<'w>(x: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    // `relu`, not `maximum(0f64)`. They are the same function, but the scalar
    // form goes through the tensor-or-scalar path, which materialises the
    // constant as a device tensor — a host-to-device copy and an allocation on
    // every call, per layer, per sequence, per token.
    let max0 = x.relu()?;
    let expneg = x.abs()?.neg()?.exp()?;
    let ln1p = (expneg + 1.0)?.log()?;
    max0.add(&ln1p)
}

/// Per-V-head RMSNorm (weighted), `x [T, H, d]` → same shape.
pub(super) fn rms_norm_per_head<'w>(
    x: &LiveTensor<'w>,
    weight: &Tensor,
    eps: f64,
) -> Result<LiveTensor<'w>> {
    let ms = x.sqr()?.mean_keepdim(candle::D::Minus1)?;
    let denom = (ms + eps)?.sqrt()?;
    x.broadcast_div(&denom)?.broadcast_mul(weight)
}

/// Advance the causal conv over a channel-major segment — the tensor-op path
/// [`conv_silu_spans`] falls back to where the CUDA kernels do not apply.
/// Same numbers as the kernels — [`super::cuda`]'s tests lock them to
/// [`causal_conv1d`] column by column — so which one runs is never which
/// arithmetic.
pub(super) fn conv_advance<'w>(
    x: &LiveTensor<'w>,
    kernel: &Tensor,
    tail: &Tensor,
    tail_out: &Tensor,
) -> Result<LiveTensor<'w>> {
    // The multi-token scan derives the advanced tail from the padded input
    // rather than shifting it, so it produces a *value*, which is written into
    // the destination buffer rather than replacing it: a sequence's state
    // buffers are allocated once and keep their identity, and the advanced
    // value is wave-scoped where the carried buffer outlives the wave. The copy
    // is the tensor-op path's alone — the CUDA conv kernels write the
    // destination directly.
    let (y, advanced) = causal_conv1d(x, kernel, tail)?;
    tail_out.slice_set(&advanced, 1, 0)?;
    Ok(y)
}

/// The conv + SiLU stage over the wave's spans: token-major `[T, conv_dim]`
/// in, token-major `[T, conv_dim]` out, every sequence's carried tail advanced
/// in place.
///
/// Two layouts serve one arithmetic. On CUDA in F32 the kernels read the
/// projection's own token-major rows — a narrow on the token axis of a
/// contiguous buffer, so the channel-major transpose (and its twin on the way
/// back out, two full copies of the widest buffer in the layer) never exists.
/// Everywhere else the tensor-op conv wants channel-major and the two
/// transposes are the price of admission.
pub(super) fn conv_silu_spans<'w>(
    qkv: &LiveTensor<'w>,
    kernel: &Tensor,
    seqs: &mut [DeltaNetSeq<'_>],
) -> Result<LiveTensor<'w>> {
    // Channel-major tensor-op conv (CPU, non-F32 reference modes — the fused
    // CUDA path never reaches here).
    let x_cm = qkv.t()?.contiguous()?; // [conv_dim, T]
    let mut parts: Vec<LiveTensor<'w>> = Vec::with_capacity(seqs.len());
    for s in seqs.iter_mut() {
        let xs = x_cm.narrow(1, s.start, s.len)?.contiguous()?;
        parts.push(match &s.write {
            SpanWrite::State(out) => conv_advance(&xs, kernel, &s.state.conv_tail, &out.conv_tail)?,
            // A short span leaves the tail as it is: its raw rows go to the
            // timeline, and the fold shifts in the ones the step keeps.
            SpanWrite::Timeline(_) => causal_conv1d(&xs, kernel, &s.state.conv_tail)?.0,
        });
    }
    // `cat`, not a preallocated buffer written span by span. The two are the
    // same work for several sequences, and **not** the same for one: `cat` of a
    // single argument hands it straight back, where writing into a shared buffer
    // costs a full copy of it.
    let conved = LiveTensor::cat(&parts, 1)?;
    silu(&conved)?.t()?.contiguous() // back to [T, conv_dim]
}

/// The conv stage of the fused CUDA path: the token-parallel conv kernels
/// with the SiLU + Q|K-norm epilogue, writing every span's rows directly into
/// one whole-wave `[T, conv_dim]` buffer — which IS the mixer's operand
/// buffer. No split, activation, norm launch, or concatenation exists
/// downstream, and the wave's decode spans are ONE launch regardless of how
/// many sessions it carries.
#[cfg(feature = "cuda")]
pub(super) fn conv_fused_spans<'w>(
    qkv: &LiveTensor<'w>,
    kernel: &Tensor,
    qk_channels: usize,
    eps: f32,
    table: Option<&DeltaNetLayerTable>,
    long: Option<&DeltaNetSpanTable>,
) -> Result<LiveTensor<'w>> {
    let (t, channels) = qkv.dims2()?;
    // Every row a kernel reads is written exactly once — by the batched decode
    // conv or by a long span's conv — so the buffer is allocated uninitialised
    // (hot-path invariant 6), in `qkv`'s arena. Short spans convolve inside
    // the short-span kernel, which reads none of these rows, so theirs stay
    // unwritten.
    let conved = qkv.empty_beside((t, channels), DType::F32)?;
    if let Some(tbl) = table {
        // All decode spans in one launch: tails via the pointer table, rows
        // scattered into the shared buffer, each row's raw input into its
        // timeline.
        super::super::cuda::delta_net_conv_decode(qkv, kernel, tbl, &conved, qk_channels, eps)?;
    }
    if let Some(sp) = long {
        // Every long span in ONE launch, token-parallel over the widest of
        // them, each writing its own rows of the shared buffer. The advanced
        // tails land in the prefill scratch — separate allocations from the
        // entering ones, which the kernel's readers of the first `kwidth−1`
        // outputs are still reading concurrently — so there is nothing to copy
        // back.
        super::super::cuda::delta_net_conv_prefill(qkv, kernel, sp, qk_channels, eps, &conved)?;
    }
    Ok(conved)
}
