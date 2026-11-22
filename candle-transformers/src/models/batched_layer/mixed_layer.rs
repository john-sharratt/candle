//! One transformer layer over a mixed wave, where each row type's attention runs
//! on its own rows and the feed-forward runs once over all of them.
//!
//! Each group's attention is added straight into its own row range of the
//! residual stream, so the mixed wave never concatenates its groups' outputs.
//! The feed-forward and its expert load are shared by decode, prefill and glue
//! rows alike, so one expert load in a layer serves every row type. The attention
//! and feed-forward transients each live in a scope bounded by this one layer,
//! which keeps their memory from accumulating with depth.

use super::*;

// Layer Processing Functions

/// One row-type group of a mixed continuous-fair-wave layer
/// (`docs/continuous_fair_waves.md`): the sequences of a single attention flavour
/// (decode / prefill / glue), their per-layer caches + offsets, the attention
/// metadata to run, and the shape of their slice of the combined residual buffer.
pub struct WaveAttnGroup<'a, 'c> {
    /// This group's sequences' caches for the current layer (disjoint sub-slice
    /// of the wave's caches).
    pub caches: &'c mut [&'a mut KvCache],
    /// Per-sequence cached-token offsets.
    pub offsets: &'c [usize],
    /// The attention metadata for this group's kernel (decode `SlotHeader` /
    /// prefill `cu_seqlens` / glue).
    pub params: &'c BatchedAttentionParams<'c>,
    /// Token-rows this group contributes to the combined buffer.
    pub rows: usize,
    /// Decode rows sit in the flat buffer as a `[1, rows, hidden]` slice but the
    /// decode kernel wants `[rows, 1, hidden]`; prefill/glue stay `[1, rows, hidden]`.
    pub decode_layout: bool,
    /// QSA: this group's block-sparse selection for the CURRENT layer, in the
    /// group's own row order. `None` — every model but Qwen3.8-Flash-Next's
    /// full-attention layers — is a full causal read.
    pub qsa: Option<&'c QsaSelection>,
}

/// Mixed-wave transformer layer — the co-batched form of
/// `docs/continuous_fair_waves.md`.
///
/// Runs each row-type's attention kernel on its **own row-slice** of the combined
/// residual buffer (no kernel changes — the existing decode / prefill / glue
/// paths, each over its rows), concatenates the per-type attention outputs, then
/// runs o_proj (already fused inside each attention call) + the residual +
/// `ffn_norm` + the **single shared FFN/MoE grouped GEMM** over the whole buffer.
///
/// Because o_proj is linear and the FFN/MoE is token-flat (per row), per-type
/// attention followed by shared post-attention is bit-identical to a fused pass —
/// and it is the "one expert load per layer serves decode + prefill + glue
/// together" amortisation. A single group (`groups.len() == 1`) is the ordinary
/// homogeneous forward: attention over the whole buffer with no slicing.
///
/// `decode` names the buffer's decode-attributed rows, for the MoE's residency
/// scoring ([`BatchedAttentionLayer::ffn_residual`]).
pub fn forward_layer_batched_mixed<L: BatchedAttentionLayer>(
    layer: &L,
    groups: &mut [WaveAttnGroup<'_, '_>],
    x: &mut TensorCat,
    act_dtype: DType,
    layer_idx: usize,
    decode: &DecodeRows,
) -> Result<()> {
    let orig_dtype = x.dtype();

    // ── Attention: each row-group's kernel over its OWN slice, concatenated ──
    // The combined buffer is flat `[1, total, hidden]` with each group a
    // contiguous row-range; a decode group reshapes its slice to the kernel's
    // `[rows, 1, hidden]` layout (`decode_layout`), prefill/glue stay flat. A
    // single homogeneous group is the same loop with one iteration — it must
    // ALSO honour `decode_layout` (a flat multi-decode group would otherwise be
    // misrouted), so there is no shape-based fast path here. Routing decode vs
    // prefill is by the group's declared headers (see `forward_attn_batched`),
    // not by tensor shape, so a 1-token prefill or a multi-token decode is safe.
    let hidden = x.dim(2)?;
    let xt = x.to_tensor();
    // The attention generation spans every group's attention, o_proj, and the
    // residual add that consumes the result —
    // the same shape as the FFN scope below. Opening it inside
    // `forward_attn_batched` instead would close it one step before the value
    // dies, which is the difference between o_proj's output living on the wave
    // and having to be allocated off it.
    //
    // Scoped to one layer, not the forward: consumption stays at a single
    // layer's working set rather than accumulating with depth. Halves
    // alternate per layer, so
    // layer N's reads are separated from layer N+2's writes by a whole layer of
    // same-stream work.
    let t_attn_host = profile_now();
    {
        #[cfg(feature = "cuda")]
        let attn_wave = match xt.device() {
            Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Attention)?),
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let attn_wave: Option<()> = None;

        let mut row0 = 0usize;
        for g in groups.iter_mut() {
            if g.rows == 0 {
                continue;
            }
            // A row range of the flat `[1, total, hidden]` buffer is itself dense
            // (the leading axis is one), so each group reads its rows in place —
            // a view that starts at row `row0` of the shared buffer, which the
            // attention producers address from its own first element.
            let slice = xt.narrow(1, row0, g.rows)?;
            let x_g = if g.decode_layout {
                TensorCat::from_cat_tensor(slice.reshape((g.rows, 1, hidden))?, 0)?
            } else {
                TensorCat::from_cat_tensor(slice, 0)?
            };
            expect_dense_view(x_g.as_cat_tensor(), "attention group rows")?;
            let h = forward_attn_batched(
                layer,
                g.caches,
                &x_g,
                g.offsets,
                g.params,
                layer_idx,
                g.qsa,
                attn_wave.as_ref(),
            )?;
            let h = if g.decode_layout {
                h.reshape((1, g.rows, hidden))?
            } else {
                h
            };
            // First residual, per group: x[rows] += attn(rows). The add lands
            // on the group's own row range of the residual stream, so a mixed
            // wave never concatenates its groups' outputs (hot-path invariant
            // 2) and the residual never takes a wave allocation. Later groups
            // read disjoint rows, so adding before they run changes nothing
            // they see.
            //
            // VALIDATED, not converted: both output-projection sites store the
            // residual's own dtype, so the two agree by construction. Rewriting
            // `x` here to meet the attention output would be a full-tensor pass
            // per layer per wave, and one that silently absorbs a producer that
            // later starts emitting the wrong width (hot-path invariant 1b).
            expect_dtype(
                &h,
                orig_dtype,
                "attention residual: attn(x) vs the residual stream",
            )?;
            x.as_cat_tensor_mut().narrow(1, row0, g.rows)?.add_mut(&h)?;
            row0 += g.rows;
        }
        // No `drop(attn_wave)`: each `h` borrows the guard, so the compiler
        // refuses any order but this one. The guard dies at the brace.
    }
    // Host time of the layer's attention half, from opening its generation to
    // closing it: against the stream time of the spans inside, what the host
    // spends issuing the work rather than the device spends doing it.
    pipeline_record("layer:attn_host", t_attn_host);

    // ── Shared FFN/MoE over the WHOLE combined buffer — one grouped GEMM whose
    // per-layer expert load serves every row-type at once. ──
    let mlp_dtype = if act_dtype == DType::F16 {
        DType::BF16
    } else {
        act_dtype
    };
    // The layer's other transient scope, and the same shape as the attention
    // one: it spans the FFN through the residual add that consumes its result,
    // after which nothing the expert forward produced is live. The MoE combine
    // target is what this bounds — it is returned from the expert forward to
    // the residual update in `ffn_residual`, so no scope inside the expert code
    // could have bounded it (§3.6, and see `wave_buffers`).
    #[cfg(feature = "cuda")]
    let ffn_wave = match x.as_cat_tensor().device() {
        Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Ffn)?),
        _ => None,
    };
    #[cfg(not(feature = "cuda"))]
    let ffn_wave: Option<()> = None;
    let t_ffn_host = profile_now();
    let acts = layer.ffn_norm(x.as_cat_tensor(), layer.int8mode(), ffn_wave.as_ref())?;
    layer.ffn_residual(
        x.as_cat_tensor_mut(),
        acts,
        mlp_dtype,
        decode,
        ffn_wave.as_ref(),
    )?;
    pipeline_record("layer:ffn_host", t_ffn_host);
    // The guard falls out of scope at the end of the function, after the
    // residual update that consumed everything the FFN put on it, fencing the
    // stream and rewinding the half.
    Ok(())
}

/// `x += h`, the FFN residual add for an implementation of
/// [`BatchedAttentionLayer::ffn_residual`] with nothing to fold into it.
///
/// VALIDATED, not converted: an FFN's down projection stores the residual's own
/// type, so the two agree by construction. Rewriting `x` to meet the FFN output
/// would be a full-tensor pass per layer per wave, and one that silently absorbs
/// a producer that later starts emitting the wrong width (hot-path invariant 1b).
pub fn add_ffn_residual(x: &mut Tensor, h: &LiveTensor<'_>) -> Result<()> {
    expect_dtype(h, x.dtype(), "ffn residual: ffn(x) vs the residual stream")?;
    x.add_mut(h)
}
