//! Multi-token attention for prefill and reprojection glue, over the flat packed
//! varlen layout.
//!
//! The packed tokens are described by cumulative sequence lengths, and the paged
//! prefill kernel applies RoPE itself, so the model does not rotate them first.
//! Ordinary prefill takes the int8 prefix-attention kernel. A reprojection-glue
//! forward is refused on any route other than its own kernel, since only that
//! kernel carries the glue's forward window and true-position columns, and a
//! route that cannot apply a layer's selection refuses rather than answering
//! densely. The output projection stores the residual stream's width, so the
//! residual never needs a full-tensor conversion to meet the attention output.

use super::*;

/// Multi-token batched attention (prefill path).
#[allow(clippy::too_many_arguments)]
pub(super) fn forward_attn_batched_multi<'w, L: BatchedAttentionLayer>(
    layer: &L,
    caches: &mut [&mut KvCache],
    x: &TensorCat,
    offsets: &[usize],
    q_lens: &[usize],
    model_rope: &LazyRope<'_>,
    rope_interleaved: bool,
    prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
    glue_meta: Option<&GlueMeta>,
    rope: &RopeRungs,
    generation: &Generation,
    shared_pm: &std::cell::RefCell<Option<SharedPm>>,
    image_rotation: Option<&ImageRotation>,
    qsa: Option<&QsaSelection>,
    wave: WaveRef<'w>,
) -> Result<LiveTensor<'w>> {
    // The flat-packed activation has leading dim 1 (x.len() == 1), so validate
    // against the sequence count carried by q_lens instead.
    validate_batch_sizes(caches.len(), offsets.len(), q_lens.len())?;

    // FLAT-packed activation: x_tensor is [1, total_q, hidden] where
    // total_q = Σ q_lens and per-sequence token ranges live in cu_seqlens
    // (prefill_meta). `n_seqs` is the sequence count (= caches.len()), NOT the
    // leading tensor dim (which is 1 for the flat batch-of-one layout).
    let n_seqs = caches.len();
    let x_tensor = x.as_cat_tensor();
    let (_one, total_q, _n_embd) = x_tensor.dims3()?;

    // Project Q/K/V over the fused attention_norm (B1). Prefill is high-M (compute-bound), so
    // the fused ln1→q8a128 still saves a launch; the per-matmul cost dominates.
    //
    // No drain before the mark. The old host-synced timer needed one here: it was
    // the first synced span of an attention layer, so without it everything
    // queued and not yet awaited — on a hybrid stack, whole DeltaNet layers —
    // drained inside this span and was reported as Q/K/V projection time. Events
    // are stream-ordered, so the span measures the work between its own two
    // records and nothing earlier.
    let g_qkv = gpu_span("prefill:qkv_proj", x_tensor.device());
    let kv_dtype = attention_operand_dtype(caches, x_tensor.dtype());
    let QkvProjection { q, k, v, gate } = {
        let acts = layer.attention_norm(x_tensor, layer.int8mode(), wave)?;
        layer.project_qkv(&acts, kv_dtype, wave)?
    };

    let n_head = layer.n_head();
    let n_kv_head = layer.n_kv_head();
    let head_dim = layer.head_dim();

    // Varlen packed layout: (1, total_q, H*D) -> (total_q, H, D).
    let q = q.reshape((total_q, n_head, head_dim))?;
    let k = k.reshape((total_q, n_kv_head, head_dim))?;
    let v = v.reshape((total_q, n_kv_head, head_dim))?.contiguous()?;
    let q = ensure_contiguous(&q)?;
    let k = ensure_contiguous(&k)?;
    g_qkv.end();

    // An image's rows turned by their place in the image, once, before the
    // kernel rotates Q by its slot and before K is written — so every later
    // read's slot rotation composes to the multimodal position, wherever the
    // chunk is placed (`docs/flash_next_vision.md` §4.4).
    if let Some(rotation) = image_rotation {
        rotation.apply(&q)?;
        rotation.apply(&k)?;
    }

    // Check for paged (chunked) CUDA path BEFORE applying model-side RoPE.
    // The paged prefill kernel applies RoPE internally; applying it here first
    // would double-rotate Q/K.
    let is_cuda_paged = caches
        .first()
        .and_then(|c| c.k_cache().chunked_arena_chunks())
        .is_some();

    // Model-side RoPE only on the non-paged (CPU) path. rope() wants [B, H, L, D];
    // for the flat layout that's [1, n_head, total_q, head_dim], with cos/sin
    // carrying the ragged per-token positions ([1, total_q, rotary]).
    let (q, k) = if is_cuda_paged {
        (q, k)
    } else {
        // Model-side cos/sin are rung 0's; see `refuse_rung_past_zero`.
        let reach: Vec<usize> = offsets.iter().zip(q_lens).map(|(o, l)| o + l).collect();
        refuse_rung_past_zero(rope, &reach, 0)?;
        let (cos, sin) = model_rope.tables()?;
        let q4 = q
            .reshape((1, total_q, n_head, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k4 = k
            .reshape((1, total_q, n_kv_head, head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let (q4, k4) = if rope_interleaved {
            (
                candle_nn::rotary_emb::rope_i(&q4, cos, sin)?,
                candle_nn::rotary_emb::rope_i(&k4, cos, sin)?,
            )
        } else {
            (
                candle_nn::rotary_emb::rope(&q4, cos, sin)?,
                candle_nn::rotary_emb::rope(&k4, cos, sin)?,
            )
        };
        let q = q4.transpose(1, 2)?.reshape((total_q, n_head, head_dim))?;
        let k = k4
            .transpose(1, 2)?
            .reshape((total_q, n_kv_head, head_dim))?;
        (ensure_contiguous(&q)?, ensure_contiguous(&k)?)
    };

    // The cache is already sized and truncated for this wave: `wave_admit` did
    // both, for every layer, before the forward began. Nothing on this path may
    // claim a chunk — the transient tier is placed against the arena frontier
    // and a claim here would move it.
    // Flat attention output: [total_q, n_head, head_dim]. A reprojection-glue
    // forward (HD128, chunked) routes to the paged-glue kernel — it streams the
    // quantized slot once and reuses it across all glue rows (dequant-once),
    // positioning every column by its chunk `rope_base` (`slice_rope`) and
    // masking each glue token by `cpos > row_pos + fwd_ahead[t]`. Everything else
    // (ordinary prefill, non-128 head dims) stays on the plain prefill kernel.
    // Both the attention context and o_proj's result live on the wave's
    // transient half — o_proj allocates from whichever arena its operand came
    // from, so the projection is wave-backed too. `wave` is the caller's
    // generation, opened in `forward_layer_batched_mixed` around every group's
    // attention *and* the residual add that consumes it. Opening it here instead
    // would end the scope one step before the value dies.
    // The reprojection-glue kernel is 128-only, and no other prefill path
    // carries its masking — a glue forward routed elsewhere would silently
    // lose the forward-bridge window and the true-position columns rather
    // than fail. Refuse it up front instead of falling through. The dtype
    // that gates the route is the **cache compute dtype** (the glue kernel is
    // compiled for the half types), not the activation dtype:
    // `paged_glue_attn` casts Q and the new K/V to the cache's dtype itself,
    // so an F32-activation reference session over a half-typed arena runs the
    // glue kernel like any other.
    if glue_meta.is_some() {
        let glue_compute = caches.first().map(|c| c.k_cache().dtype());
        let ok = is_cuda_paged
            && head_dim == 128
            && matches!(glue_compute, Some(DType::F16 | DType::BF16));
        if !ok {
            candle::bail!(
                "glue prefill requires the paged head_dim-128 kernel (got head_dim \
                 {head_dim}, paged {is_cuda_paged}, cache compute dtype {:?}) — no \
                 other prefill path carries glue masking",
                glue_compute
            );
        }
    }
    // Only the int8 prefix-attention kernel reads a selection. The glue and
    // float-fallback routes would answer densely, which is a wrong number
    // rather than a slow one, so they refuse instead.
    if qsa.is_some()
        && (glue_meta.is_some()
            || !(is_cuda_paged
                && int8_prefill_head_dim(head_dim)
                && int8_prefill_act_dtype(q.dtype())))
    {
        candle::bail!(
            "QSA selection on a prefill route that reads the whole prefix (glue \
             {}, paged {is_cuda_paged}, head_dim {head_dim}, q dtype {:?}) — only the \
             int8 prefix-attention kernel carries block-sparse masking",
            glue_meta.is_some(),
            q.dtype()
        );
    }
    let out_packed = match glue_meta {
        Some(g) => paged_glue_attn(
            wave,
            caches,
            offsets,
            &q,
            &k,
            &v,
            n_seqs,
            q_lens,
            n_head,
            n_kv_head,
            head_dim,
            prefill_meta,
            &g.glue_write_slice,
            &g.glue_write_in_blk,
            &g.fwd_ahead,
            rope,
            rope_interleaved,
            generation,
            shared_pm,
        )?,
        // Shapes and dtypes the int8 prefix-attention kernel is not built for.
        // the float fallback, which keeps the paged cache contract
        // (unrotated K/V in the arena) and pays a per-sequence materialized
        // score matrix instead of a fused kernel.
        None if is_cuda_paged
            && !(int8_prefill_head_dim(head_dim) && int8_prefill_act_dtype(q.dtype())) =>
        {
            paged_prefill_float_fallback(
                caches,
                offsets,
                &q,
                &k,
                &v,
                q_lens,
                n_head,
                n_kv_head,
                head_dim,
                rope,
                rope_interleaved,
            )?
        }
        // Ordinary prefill (fresh or over an existing prefix): the INT8
        // prefix-attention kernel —
        // GQA-packed M, slice-aligned tiles, int8 MMA directly over the
        // quantized arena, split-KV for the short-q/long-prefix regime.
        // Both RoPE pairings are applied in-kernel (half-split and
        // interleaved).
        None => paged_prefill_batched(
            wave,
            caches,
            offsets,
            &q,
            &k,
            &v,
            n_seqs,
            q_lens,
            n_head,
            n_kv_head,
            head_dim,
            prefill_meta,
            rope,
            rope_interleaved,
            generation,
            shared_pm,
            qsa,
        )?,
    };

    // Project per-token: [total_q, n_head*head_dim] -> [total_q, hidden_out].
    // (n_head*head_dim may differ from n_embd, e.g. Qwen3-MoE.)
    let reshaped_ctx = out_packed
        .contiguous()?
        .reshape((total_q, n_head * head_dim))?;
    // B2: prefill/glue feed the FP context as `Float`; the int8 override quantizes at the matmul
    // (launch amortized against the large prefill GEMM).
    let g_out_proj = gpu_span("prefill:out_proj", reshaped_ctx.device());
    let output = {
        let gated = apply_attention_gate(reshaped_ctx, gate)?;
        // Store the RESIDUAL STREAM's width, as the decode path does — not the
        // context's own. `x + attn(x)` needs one dtype, and naming it here means
        // the projection's store performs the conversion instead of a
        // full-tensor pass rewriting the residual to meet the attention output
        // (hot-path invariant 1).
        layer.output_projection(DynamicActs::Float(gated), x.dtype())?
    };
    g_out_proj.end();
    // Restore the flat-packed [1, total_q, hidden_out] activation.
    let hidden_out = output.dim(1)?;
    output.reshape((1, total_q, hidden_out))
}
