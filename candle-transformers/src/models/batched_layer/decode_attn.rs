//! Attention for the single-token decode step over the paged KV cache, and the
//! fallback for a cache that is not paged.
//!
//! The paged kernel rotates Q and K itself at each slot's own rung, so the model
//! must not rotate them first, or they would be rotated twice. The attention
//! context is emitted directly in the int8 form the output projection consumes,
//! and a gated layer's output gate is folded into the same kernel pass, so the
//! gate costs no launch. A slot's usage advances once per step for all layers
//! together: a per-layer advance would leave the layers within one step
//! disagreeing about the cache, which the layer-invariance guard reads as
//! corruption.

use super::*;

/// Single-token batched attention (decode path).
#[allow(clippy::too_many_arguments)]
pub(super) fn forward_attn_batched_single<'w, L: BatchedAttentionLayer>(
    layer: &L,
    caches: &mut [&mut KvCache],
    x: &TensorCat,
    offsets: &[usize],
    model_rope: &LazyRope<'_>,
    rope_interleaved: bool,
    rope: &RopeRungs,
    #[allow(unused_variables)] generation: &Generation,
    #[allow(unused_variables)] decode_headers_ptr: u64,
    #[allow(unused_variables)] qsa: Option<&QsaSelection>,
    wave: WaveRef<'w>,
) -> Result<LiveTensor<'w>> {
    validate_batch_sizes(caches.len(), offsets.len(), x.len())?;

    // `x` is the PRE-norm activation; B1 fuses ln1 → q/k/v here. Shapes are norm-invariant.
    let x_tensor = x.as_cat_tensor();
    let (b_sz, seq_len, _n_embd) = x_tensor.dims3()?;
    debug_assert_eq!(seq_len, 1);

    // Project Q/K/V over the fused attention_norm (q8a128 on int8, FP on Off / non-CUDA).
    let g_qkv = gpu_span("decode:qkv_proj", x_tensor.device());
    let kv_dtype = attention_operand_dtype(caches, x_tensor.dtype());
    let QkvProjection { q, k, v, gate } = {
        let acts = layer.attention_norm(x_tensor, layer.int8mode(), wave)?;
        layer.project_qkv(&acts, kv_dtype, wave)?
    };

    // Reshape for attention: (B, seq_len, H*D) -> (B, H, seq_len, D)
    // For seq_len=1, we can reshape directly without transpose
    let n_head = layer.n_head();
    let n_kv_head = layer.n_kv_head();
    let head_dim = layer.head_dim();

    let q = q.reshape((b_sz, n_head, 1, head_dim))?;
    let k = k.reshape((b_sz, n_kv_head, 1, head_dim))?;
    let v = v.reshape((b_sz, n_kv_head, 1, head_dim))?;

    let q = ensure_contiguous(&q)?;
    let k = ensure_contiguous(&k)?;
    let v = ensure_contiguous(&v)?;
    g_qkv.end();

    // Check for chunked (paged) KV cache BEFORE applying model-side RoPE.
    // For the paged path the decode kernel rotates Q and K internally, at their
    // own positions under the slot's rung, so we must NOT pre-rotate Q/K here.
    let use_paged = caches
        .first()
        .and_then(|c| c.k_cache().chunked_arena_chunks())
        .is_some();

    // Apply model-side RoPE only for the non-paged path.
    // Paged kernel always applies RoPE internally — applying it here would double-rotate.
    let (q, k) = if use_paged {
        // Kernel will rotate; skip model-side rotation.
        (q, k)
    } else {
        // The model-side cos/sin are rung 0's frequencies; a slot the schedule
        // has moved up a rung rotates only in the paged kernels.
        refuse_rung_past_zero(rope, offsets, 1)?;
        let (cos, sin) = model_rope.tables()?;
        if cos.dtype() != q.dtype() || sin.dtype() != q.dtype() {
            candle::bail!(
                "rope cos/sin dtype mismatch: q={:?} cos={:?} sin={:?}",
                q.dtype(),
                cos.dtype(),
                sin.dtype()
            );
        }
        if cos.dim(0)? != b_sz || sin.dim(0)? != b_sz {
            candle::bail!(
                "rope cos/sin batch mismatch: b_sz={} cos_b={} sin_b={}",
                b_sz,
                cos.dim(0)?,
                sin.dim(0)?
            );
        }
        if rope_interleaved {
            (
                candle_nn::rotary_emb::rope_i(&q, cos, sin)?,
                candle_nn::rotary_emb::rope_i(&k, cos, sin)?,
            )
        } else {
            (
                candle_nn::rotary_emb::rope(&q, cos, sin)?,
                candle_nn::rotary_emb::rope(&k, cos, sin)?,
            )
        }
    };

    reset_caches_at_zero(caches, offsets);

    // B2: emit the decode context directly as q8a1024 (head_dim 128 or 256 — whole q8a128
    // tiles per head) so o_proj runs int8 with no standalone quantize. Only on the paged CUDA
    // decode path; false → FP context. A gated layer folds its output gate into the same
    // kernel pass, so the gate costs no launches either.
    let want_q8 = layer.int8mode().is_int8()
        && use_paged
        && seq_len == 1
        && paged_decode_q8_head_dim(head_dim);

    // Both the attention context and o_proj's result live on the wave's
    // transient half — o_proj allocates from whichever arena its operand came
    // from, so the projection is wave-backed too. `wave` is the caller's
    // generation, opened in `forward_layer_batched_mixed` around every group's
    // attention *and* the residual add that consumes it. Opening it here instead
    // would end the scope one step before the value dies.
    let outputs = if use_paged && seq_len == 1 {
        paged_decode_attention(
            wave,
            caches,
            offsets,
            &q,
            &k,
            &v,
            n_head,
            n_kv_head,
            head_dim,
            rope,
            rope_interleaved,
            generation,
            decode_headers_ptr,
            want_q8,
            if want_q8 { gate.as_ref() } else { None },
            qsa,
        )?
    } else {
        // Non-chunked fallback: standard per-sequence attention. It carries no
        // selection, so a layer that has one must not be silently answered
        // densely — that is a wrong number, not a slower one.
        if qsa.is_some() {
            candle::bail!(
                "QSA selection on the non-paged decode fallback: the contiguous-KV path \
                 reads the whole prefix, so it cannot honour a block-sparse selection"
            );
        }
        standard_batched_attention(caches, &q, &k, &v, head_dim, n_head, n_kv_head)?
    };

    // B2: o_proj over the attention context. On CUDA, `want_q8` → `outputs` is the flat U8 q8a1024
    // context wrapped (no copy) into an int8 operand; otherwise the FP context (the int8 override,
    // if any, quantizes it at the matmul). Non-CUDA never sets `want_q8` and has no dynamic o_proj.
    let g_out_proj = gpu_span("decode:out_proj", x_tensor.device());
    let attn_out = if want_q8 {
        // The gate, if any, was folded into the decode kernel's combine pass
        // (`sigmoid(g) ⊙ ctx` before the quantize), so the q8a1024 bytes are
        // already the gated context.
        let op = Q8a128Operand::from_tensor(outputs, b_sz, n_head * head_dim)
            .with_lead(vec![b_sz, seq_len]);
        layer.output_projection(DynamicActs::Int8(op), x_tensor.dtype())?
    } else {
        let out = outputs.reshape((b_sz, 1, n_head * head_dim))?;
        let out = apply_attention_gate(out, gate)?;
        layer.output_projection(DynamicActs::Float(out), x_tensor.dtype())?
    };
    g_out_proj.end();

    // Bounded by the caller's generation, not wrapped and copied off it: the
    // layer consumes this into the residual add while that generation is still
    // open, so `'w` is what keeps the two in step.
    Ok(attn_out)
}

// Attention Implementations

/// Paged decode attention using chunked KV cache.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(super) fn paged_decode_attention<'w>(
    wave: Option<&'w WaveGeneration>,
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &LiveTensor<'_>,
    k: &LiveTensor<'_>,
    v: &LiveTensor<'_>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rope: &RopeRungs,
    rope_interleaved: bool,
    _generation: &Generation,
    decode_headers_ptr: u64,
    // B2: emit the attention context as q8a1024 (returns a flat U8 tensor) instead of an FP
    // context, so o_proj runs int8 with no standalone quantize. Head_dim 128 or 256.
    emit_q8: bool,
    // Output gate folded into the q8 emit (`sigmoid(g) ⊙ ctx` inside the combine kernel).
    // Only meaningful with `emit_q8`; the FP path applies its gate on the FP context instead.
    gate: Option<&LiveTensor<'_>>,
    // QSA: this group's block-sparse selection, one row per slot.
    qsa: Option<&QsaSelection>,
) -> Result<LiveTensor<'w>> {
    // Host spans, not GPU ones: both regions are pure host work in the common
    // path — validation, then metadata assembly whose only possible launches are
    // dtype casts that do not fire when Q already matches the arena. An event
    // pair around a region that enqueues nothing measures the host gap when the
    // stream is idle and the outstanding backlog when it is not, which is the
    // ambiguity the event timer exists to remove. Host time is also the answer
    // that matters here: this is launch-side cost.
    let s_alloc = span("decode:alloc");
    KvCache::validate_chunked_decode_batch(caches, offsets)?;
    s_alloc.end();

    let s_meta = span("decode:meta");
    let _batch_indices: Vec<usize> = caches
        .iter()
        .map(|c| c.k_cache().chunked_batch_idx().unwrap_or(0))
        .collect();

    let (arena_dtype, _chunk_size) =
        {
            let first = caches
                .first()
                .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;

            // Get format tags for dtype dispatch from the backing's default K/V formats.
            let (k_format_tag, v_format_tag) = first
                .k_cache()
                .chunked_arena_format_tags()
                .ok_or_else(|| candle::Error::Msg("expected chunked arena format tags".into()))?;

            let dispatch_dtype =
                |tag: candle_nn::kv_cache::ArenaFormatTag| -> candle::Result<DType> {
                    if tag.is_quantized() {
                        Ok(first.dtype())
                    } else {
                        tag.to_dtype().ok_or_else(|| {
                            candle::Error::Msg(format!(
                                "paged-decode-attention: invalid arena format tag {:?}",
                                tag
                            ))
                        })
                    }
                };
            let k_dtype = dispatch_dtype(k_format_tag)?;
            let v_dtype = dispatch_dtype(v_format_tag)?;
            if k_dtype != v_dtype {
                candle::bail!(
                "K and V arena formats require different compute dtypes: K={:?}({:?}) V={:?}({:?})",
                k_format_tag, k_dtype, v_format_tag, v_dtype
            );
            }
            let dtype = k_dtype;
            let chunk_size = first
                .k_cache()
                .chunked_chunk_size()
                .unwrap_or(CHUNK_SIZE as usize);

            (dtype, chunk_size)
        };

    // Squeeze from (B, H, 1, D) to (B, H, D) for paged decode kernel
    let q_3d = q.squeeze(2)?;
    let k_3d = k.squeeze(2)?;
    let v_3d = v.squeeze(2)?;

    let q_dtype = q.dtype(); // Store for output dtype

    // Mixed-precision handling:
    // - FP8 arenas: Q must be BF16 (for precision), k_new/v_new must also be BF16.
    //   The decode kernel reads k_new/v_new as BF16* and arena_store_element converts BF16→FP8
    //   when writing to the arena. Passing FP8 bytes to a BF16* kernel produces garbage.
    // - Other arenas: Q/k_new/v_new must all match arena dtype
    let (q_kernel, k_kernel, v_kernel) = if arena_dtype == DType::F8E4M3 {
        // FP8 KV cache with BF16 compute: Q and new K/V are all BF16.
        // The kernel writes BF16→FP8 to the arena via arena_store_element (correct conversion).
        if q_3d.dtype() != DType::BF16 {
            candle::bail!(
                "paged-decode: FP8 arenas require BF16 Q, got {:?}",
                q_3d.dtype()
            );
        }
        (
            q_3d,
            k_3d.to_dtype(DType::BF16)?,
            v_3d.to_dtype(DType::BF16)?,
        )
    } else if q_3d.dtype() != arena_dtype {
        // Non-FP8 arenas: convert everything to arena dtype
        (
            q_3d.to_dtype(arena_dtype)?,
            k_3d.to_dtype(arena_dtype)?,
            v_3d.to_dtype(arena_dtype)?,
        )
    } else {
        (q_3d, k_3d, v_3d)
    };

    let softmax_scale = 1.0 / (head_dim as f32).sqrt();

    let out = {
        s_meta.end();

        let g_kernel = gpu_span("decode:kernel", q_kernel.device());
        // B2: emit q8a1024 (flat U8 tensor) instead of the FP context when requested. The U8
        // bytes are the operand for o_proj's int8 matmul — never dtype-converted.
        let raw_out = if emit_q8 {
            // The kernel reads the gate with the queries' element type. On the
            // nominal gated decode the projection already emits it in that type
            // (activation dtype == arena dtype) and the strided view passes
            // through untouched; the conversion below runs only on the
            // arena-mismatch sessions whose Q/K/V were converted above — the
            // same off-nominal sessions, paying one more pass of the same kind.
            let gate_kernel = match gate {
                Some(g) if g.dtype() != q_kernel.dtype() => Some(g.to_dtype(q_kernel.dtype())?),
                Some(g) => Some(g.clone()),
                None => None,
            };
            paged_decode_attn_q8(
                wave,
                &q_kernel,
                decode_headers_ptr,
                arena_dtype,
                n_head,
                n_kv_head,
                head_dim,
                softmax_scale,
                &k_kernel,
                &v_kernel,
                rope,
                rope_interleaved,
                gate_kernel.as_ref(),
                qsa,
            )?
        } else {
            paged_decode_attn(
                wave,
                &q_kernel,
                decode_headers_ptr,
                arena_dtype,
                n_head,
                n_kv_head,
                head_dim,
                softmax_scale,
                &k_kernel,
                &v_kernel,
                rope,
                rope_interleaved,
                qsa,
            )?
        };
        g_kernel.end();

        if !emit_q8 && q_dtype != arena_dtype {
            raw_out.to_dtype(q_dtype)?
        } else {
            raw_out
        }
    };
    // The token's usage advance does NOT happen here, and must not. This runs
    // once per layer as the sweep passes through, but a decode step with a
    // creep group active is split across several `forward_wave` segment calls —
    // and each of those rebuilds the decode metadata from the caches. A
    // per-layer advance makes the layers between two segments genuinely
    // disagree (swept layers one token ahead), which the layer-invariance
    // guard then reads as corruption. In-step attention never needs the
    // advance: the new token is read through the position map's write-slot
    // entry, built against the pre-step usage. The advance is bookkeeping for
    // the NEXT step, and `forward_wave_contexts` performs it once, for every
    // layer at once, when the step's head completes.
    Ok(out)
}
