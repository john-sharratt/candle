//! Attention fallbacks for shapes and layouts the fused paged kernels do not
//! cover: correct at any width, and slower than the kernels.
//!
//! The float prefill fallback handles any head width by working through the
//! batch one sequence at a time and materialising the score matrix. It writes K
//! and V unrotated into the cache, which every paged reader expects, and rotates
//! only for its own attention. It reads the cached prefix back only when a
//! prefill follows an existing prefix, since a prefill from offset zero already
//! holds exactly the tokens it attends to. The non-paged paths refuse any row
//! past the first RoPE rung, since they would rotate it at the wrong frequencies.

use super::*;

/// Float prefill for head dims outside the int8 prefix-attention kernel's
/// {64, 128, 256} instantiation set (see [`int8_prefill_head_dim`]).
///
/// Correct at any head width, and much slower than the kernel: it walks the
/// batch one sequence at a time, expands K/V for GQA, and materializes the
/// full `[1, H, T, T]` score matrix and causal mask per call.
///
/// The paged-path contract is kept exactly:
/// - K/V are written to the chunked cache **unrotated** — the arena
///   convention every paged reader (decode, glue, reprojection) depends on.
///   Rotation happens locally, for this call's own attention only.
/// - RoPE comes from the same rungs the kernels read: each sequence rotates by
///   the rung its reach selects, K at unit scale and Q with the rung's `m²`,
///   through the host mirror of the kernels' lookup.
///
/// Takes the flat-packed varlen operands (`q [total_q, n_head, head_dim]`,
/// `k`/`v [total_q, n_kv_head, head_dim]`) and returns the flat-packed
/// context `[total_q, n_head, head_dim]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn paged_prefill_float_fallback(
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &LiveTensor<'_>,
    k: &LiveTensor<'_>,
    v: &LiveTensor<'_>,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rope: &RopeRungs,
    rope_interleaved: bool,
) -> Result<Tensor> {
    // Pool copies: this path feeds `chunked_write_kv` and plain tensor math,
    // none of which is wave-aware (same posture as the non-paged fallback).
    let q = &q.to_owned_tensor()?;
    let k = &k.to_owned_tensor()?;
    let v = &v.to_owned_tensor()?;
    let n_rep = n_head / n_kv_head;
    let half = head_dim / 2;

    // The cos/sin planes at `at`, one row per position, by `rung` and `scale`
    // (K at 1, Q at the rung's `m²`). Full width on purpose: a pass-through
    // frequency is exactly `(cos 1, sin 0)`, unscaled, so rotating all
    // `head_dim/2` pairs is the identity on the non-rotary dims, as the kernels
    // leave them.
    let dtype = q.dtype();
    let device = q.device().clone();
    let planes = |rung: u32, scale: f32, at: &[usize]| -> Result<(Tensor, Tensor)> {
        let mut cos = Vec::with_capacity(at.len() * half);
        let mut sin = Vec::with_capacity(at.len() * half);
        for &pos in at {
            for f in 0..half {
                let (c, s) = rope.cos_sin(rung, pos, f);
                let k = if f < rope.pairs() { scale } else { 1.0 };
                cos.push(c * k);
                sin.push(s * k);
            }
        }
        Ok((
            Tensor::from_vec(cos, (at.len(), half), &device)?.to_dtype(dtype)?,
            Tensor::from_vec(sin, (at.len(), half), &device)?.to_dtype(dtype)?,
        ))
    };

    // Rotate `x [1, h, len, d]` by `(cos, sin)`, one row per position.
    let rot = |x: &Tensor, (cos, sin): &(Tensor, Tensor)| {
        if rope_interleaved {
            candle_nn::rotary_emb::rope_i(&x.contiguous()?, cos, sin)
        } else {
            candle_nn::rotary_emb::rope(&x.contiguous()?, cos, sin)
        }
    };

    let mut rows: Vec<Tensor> = Vec::with_capacity(q_lens.len());
    let mut row_start = 0usize;
    for (cache, (&len, &offset)) in caches.iter_mut().zip(q_lens.iter().zip(offsets.iter())) {
        if len == 0 {
            continue;
        }
        // Flat rows → [1, heads, len, d].
        let take = |x: &Tensor| -> Result<Tensor> {
            x.narrow(0, row_start, len)?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .contiguous()
        };
        let qs = take(q)?;
        let ks = take(k)?;
        let vs = take(v)?;

        // Write UNROTATED K/V through the chunked cache. The head_dim-128
        // kernel path scatters inside the attention kernel; this one cannot, so
        // it goes through `write_contiguous`, which walks 32-token block ×
        // KV head × palette band. That is the span this records: it is the
        // candidate for the fallback's cost, and an attention layer here is
        // ~45x its own arithmetic bound.
        let t_kv = profile_now();
        KvCache::ensure_chunked_capacity_batch(std::slice::from_mut(cache), &[offset], len)?;
        cache.chunked_write_kv(offset, &ks, &vs)?;
        cache.commit_written_tokens(offset, len)?;
        pipeline_record("prefill_fb:kv_write", t_kv);

        // Assemble the prefix to attend, **without re-reading what we just
        // wrote**. `chunked_read_kv` gathers token by token — a paged sequence
        // is not a flat grid, so each logical position resolves to its own
        // `(chunk, slot)` and is copied individually. At 649 tokens that is
        // thousands of tiny copies *per layer*, and it was 71–80% of prefill
        // (147 ms per attention layer on the 0.8B, 294 ms on the 9B) for data
        // already sitting in registers.
        //
        // A prefill at offset 0 — the whole-prompt case — needs no read at all:
        // `ks`/`vs` are exactly the tokens the cache now holds. Only a prefill
        // over an existing prefix reads, and then only the prefix.
        //
        // The values are the same either way: the arena stores at the cache
        // dtype and the activations arrive at `activation_dtype` of it, so the
        // round trip was returning what went in.
        let total = offset + len;
        let (k_all, v_all) = if offset == 0 {
            (ks.clone(), vs.clone())
        } else {
            let (kp, vp) = cache.chunked_read_kv(0, offset)?;
            // Per the note above: the arena stores at the cache dtype and the
            // activations arrive at `activation_dtype` of it, so the prefix
            // comes back in the width the new tokens are already in. All four
            // conversions here were no-ops that would have silently absorbed a
            // producer handing over the wrong width (invariant 1b) — on the
            // dominant per-layer prefill cost for every head_dim-256 model,
            // which is this whole lineage.
            expect_dtype(&kp, ks.dtype(), "prefill fallback: cached K prefix")?;
            expect_dtype(&vp, vs.dtype(), "prefill fallback: cached V prefix")?;
            (
                Tensor::cat(&[kp, ks.clone()], 2)?,
                Tensor::cat(&[vp, vs.clone()], 2)?,
            )
        };
        let t_rope = profile_now();
        expect_dtype(&k_all, qs.dtype(), "prefill fallback: K vs Q")?;
        expect_dtype(&v_all, qs.dtype(), "prefill fallback: V vs Q")?;
        // Every key at the position the slot places it — past the slot's holes
        // (`docs/placed_section_layout.md` §4.1) — and each query at its own
        // key's, as the kernels read them; the rung by the deepest of them.
        let runs = cache.k_cache().chunked_position_runs()?;
        let key_at: Vec<usize> = (0..total)
            .map(|t| ChunkedKvBacking::token_position(t, &runs))
            .collect();
        let rung = rope.rung_for(key_at.last().map_or(0, |p| p + 1))?;
        let k_rot = rot(&k_all, &planes(rung, 1.0, &key_at)?)?;
        let q_rot = rot(&qs, &planes(rung, rope.q_scale(rung), &key_at[offset..])?)?;
        let k_rep = repeat_kv(k_rot, n_rep)?;
        let v_rep = repeat_kv(v_all, n_rep)?;
        pipeline_record("prefill_fb:rope_repeat", t_rope);

        let t_attn = profile_now();
        let out = standard_attention_prefill(&q_rot, &k_rep, &v_rep, head_dim)?;
        rows.push(out.squeeze(0)?.transpose(0, 1)?.contiguous()?); // [len, H, d]
        pipeline_record("prefill_fb:attention", t_attn);

        row_start += len;
    }

    let refs: Vec<&Tensor> = rows.iter().collect();
    Tensor::cat(&refs, 0)
}

/// Simple per-sequence prefill attention fallback.
///
/// This processes each sequence independently using flash-attn (when available)
/// or standard attention with causal masking. Works with both contiguous and chunked KV caches.
/// Used for quantized KV cache mode where paged CUDA kernels aren't available.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
pub(super) fn prefill_attention_simple(
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &LiveTensor<'_>,
    k: &LiveTensor<'_>,
    v: &LiveTensor<'_>,
    head_dim: usize,
    n_head: usize,
    n_kv_head: usize,
) -> Result<Tensor> {
    // The non-paged fallback keeps its operands in the ordinary pool: it feeds
    // `KvCache::append` and `standard_attention_prefill`, neither of which is
    // wave-aware, and it runs only where the paged kernels do not apply. The
    // copy is explicit rather than a lifetime launder — the alternative would be
    // claiming `'static` over memory the layer is about to reclaim.
    let q = &q.to_owned_tensor()?;
    let k = &k.to_owned_tensor()?;
    let v = &v.to_owned_tensor()?;
    let b_sz = q.dim(0)?;
    let seq_len = q.dim(2)?; // Q is (B, H, L, D)
    let mut all_outputs = Vec::with_capacity(b_sz);
    let n_rep = n_head / n_kv_head;
    #[cfg(feature = "flash-attn")]
    let scale = 1.0 / (head_dim as f32).sqrt();

    for (batch_idx, (cache, &offset)) in caches.iter_mut().zip(offsets.iter()).enumerate() {
        let q_seq = q.narrow(0, batch_idx, 1)?;
        let k_seq = k.narrow(0, batch_idx, 1)?;
        let v_seq = v.narrow(0, batch_idx, 1)?;

        // Get cached K/V - different path for chunked vs contiguous
        let (k_cached, v_cached) = if cache.k_cache().is_chunked() {
            // Chunked: write new data, then read all data back
            // Ensure capacity and write new data
            KvCache::ensure_chunked_capacity_batch(
                std::slice::from_mut(cache),
                &[offset],
                seq_len,
            )?;
            cache.chunked_write_kv(offset, &k_seq, &v_seq)?;
            cache.commit_written_tokens(offset, seq_len)?;

            // Read all data from 0 to offset+seq_len
            let total_len = offset + seq_len;
            cache.chunked_read_kv(0, total_len)?
        } else {
            // Contiguous: use append
            cache.append(&k_seq, &v_seq)?
        };

        // GQA: repeat K/V to match Q head count
        let k_cached = repeat_kv(k_cached, n_rep)?;
        let v_cached = repeat_kv(v_cached, n_rep)?;

        // Try flash attention first (requires BF16/F16)
        #[cfg(feature = "flash-attn")]
        let out_seq = {
            let q_fa = q_seq.transpose(1, 2)?.to_dtype(DType::BF16)?;
            let k_fa = k_cached.transpose(1, 2)?.to_dtype(DType::BF16)?;
            let v_fa = v_cached.transpose(1, 2)?.to_dtype(DType::BF16)?;

            match candle_flash_attn::flash_attn(&q_fa, &k_fa, &v_fa, scale, true) {
                Ok(out) => out.to_dtype(DType::F32)?.transpose(1, 2)?,
                Err(_) => standard_attention_prefill(&q_seq, &k_cached, &v_cached, head_dim)?,
            }
        };
        #[cfg(not(feature = "flash-attn"))]
        let out_seq = standard_attention_prefill(&q_seq, &k_cached, &v_cached, head_dim)?;

        all_outputs.push(out_seq);
    }

    let out_refs: Vec<&Tensor> = all_outputs.iter().collect();
    Tensor::cat(&out_refs, 0)
}

/// Standard attention for prefill (multi-token) with causal masking.
#[allow(dead_code)]
pub(super) fn standard_attention_prefill(
    q: &Tensor,
    k_cached: &Tensor,
    v_cached: &Tensor,
    head_dim: usize,
) -> Result<Tensor> {
    // Note: K/V should already be expanded for GQA before calling this function
    let (_b, _h, q_len, _d) = q.dims4()?;
    let kv_len = k_cached.dim(2)?;

    let scale = 1.0 / (head_dim as f64).sqrt();
    let k_t = k_cached.t()?;
    let att = (q.matmul(&k_t)? * scale)?;

    // Causal mask: positions can only attend to earlier positions
    let cache_len = kv_len; // Total KV length including prefix
    let prefix_len = cache_len - q_len; // How much prefix exists before this chunk

    // Built on the device from two index vectors. As a host `Vec` this is
    // `q_len × cache_len` floats constructed one at a time in a Rust loop and
    // uploaded — 421k elements per attention layer at a 649-token prefill, for
    // a mask that is identical at every layer and derivable from two `arange`s.
    let dev = q.device();
    let idx_q = Tensor::arange(0u32, q_len as u32, dev)?
        .reshape((q_len, 1))?
        .broadcast_add(&Tensor::new(prefix_len as u32, dev)?)?;
    let idx_k = Tensor::arange(0u32, cache_len as u32, dev)?.reshape((1, cache_len))?;
    // Query at absolute position `prefix_len + i` may attend keys `0..=` it.
    let allowed = idx_k.broadcast_le(&idx_q)?;
    let keep = Tensor::zeros((q_len, cache_len), att.dtype(), dev)?;
    let block = Tensor::full(f32::NEG_INFINITY, (q_len, cache_len), dev)?.to_dtype(att.dtype())?;
    let mask = allowed
        .where_cond(&keep, &block)?
        .reshape((1, 1, q_len, cache_len))?;
    let att = att.broadcast_add(&mask)?;

    let att = candle_nn::ops::softmax_last_dim(&att)?;
    att.matmul(&v_cached.contiguous()?)
}

/// Standard (non-paged) batched attention fallback.
pub(super) fn standard_batched_attention(
    caches: &mut [&mut KvCache],
    q: &LiveTensor<'_>,
    k: &LiveTensor<'_>,
    v: &LiveTensor<'_>,
    head_dim: usize,
    n_head: usize,
    n_kv_head: usize,
) -> Result<Tensor> {
    // As in `prefill_attention_simple`: this fallback hands its operands to
    // `KvCache::append` and the non-paged attention helpers, none of which are
    // wave-aware, so it copies off the wave rather than laundering the lifetime.
    let q = &q.to_owned_tensor()?;
    let k = &k.to_owned_tensor()?;
    let v = &v.to_owned_tensor()?;
    let b_sz = q.dim(0)?;
    let mut all_outputs = Vec::with_capacity(b_sz);

    // GQA: compute repeat factor for K/V expansion
    let n_rep = n_head / n_kv_head;

    for (batch_idx, cache) in caches.iter_mut().enumerate() {
        let k_seq = k.narrow(0, batch_idx, 1)?;
        let v_seq = v.narrow(0, batch_idx, 1)?;
        let q_seq = q.narrow(0, batch_idx, 1)?;

        let (k_cached, v_cached) = cache.append(&k_seq, &v_seq)?;

        // GQA: repeat K/V to match Q head count
        let k_cached = repeat_kv(k_cached, n_rep)?;
        let v_cached = repeat_kv(v_cached, n_rep)?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let k_t = k_cached.t()?;
        let att = (q_seq.matmul(&k_t)? * scale)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out_seq = att.matmul(&v_cached)?;
        all_outputs.push(out_seq);
    }

    let out_refs: Vec<&Tensor> = all_outputs.iter().collect();
    Tensor::cat(&out_refs, 0)
}

// Helper Functions

/// Reset caches where offset == 0.
pub fn reset_caches_at_zero(caches: &mut [&mut KvCache], offsets: &[usize]) {
    for (cache, &offset) in caches.iter_mut().zip(offsets.iter()) {
        if offset == 0 {
            cache.reset();
        }
    }
}

/// Refuse a non-paged attention over any row the RoPE schedule has moved past
/// its first rung, or whose first rung carries a temperature.
///
/// The non-paged paths rotate with the model-level cos/sin, which are rung
/// 0's frequencies at unit scale; only the paged kernels read a sequence's
/// rung from its header and put the rung's `m²` on Q. A row past rung 0, or
/// on a rung 0 with `m² ≠ 1` (a file's static declared YaRN), answered here
/// would rotate by the wrong frequencies or miss the scale — a wrong number,
/// not a slow one. `ahead` is how far past each offset in `reach` the step
/// writes.
pub(super) fn refuse_rung_past_zero(rope: &RopeRungs, reach: &[usize], ahead: usize) -> Result<()> {
    for &r in reach {
        let rung = rope.rung_for(r + ahead)?;
        if rung != 0 || rope.q_scale(rung) != 1.0 {
            candle::bail!(
                "non-paged attention at a reach of {} positions: the schedule puts it on rope \
                 rung {rung} (Q scale {}), which only the paged kernels rotate by",
                r + ahead,
                rope.q_scale(rung)
            );
        }
    }
    Ok(())
}

/// Validate that cache and offset counts match the batch size.
pub fn validate_batch_sizes(
    caches_len: usize,
    offsets_len: usize,
    batch_size: usize,
) -> Result<()> {
    if caches_len != batch_size {
        candle::bail!(
            "Cache count mismatch: expected {} caches, got {}",
            batch_size,
            caches_len
        );
    }
    if offsets_len != batch_size {
        candle::bail!(
            "Offset count mismatch: expected {} offsets, got {}",
            batch_size,
            offsets_len
        );
    }
    Ok(())
}

/// Ensure tensor is contiguous.
pub(super) fn ensure_contiguous<'w>(t: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}
