//! Runs paged prefill attention over chunked KV arenas, scattering each prompt's
//! new tokens in and attending the prefix without materialising it.
//!
//! The write region is claimed before the call, so this path scatters, computes
//! and commits lengths and nothing else. Arenas are kept in a half-precision type
//! the kernel supports, and full-precision reference caches are collapsed to
//! BF16. The output is packed in query order, so the caller reshapes it and
//! projects each token. A CPU fallback exists only for caches that are not
//! chunked, and it refuses a block-sparse selection rather than answer it
//! densely.

use super::*;

/// Chunked KV + paged prefill attention over chunks.
///
/// This path avoids pooled KV materialization by:
/// 1) Ensuring (or creating) chunked KV backing
/// 2) Scattering this prefill segment into chunked arenas
/// 3) Running the paged prefill attention kernel that reads via `block_table`.
///
/// Returns `Some(outputs_per_sequence)` on success, `None` if inapplicable or if the attempt fails.
// A kernel-launch transcription: pointers, extents, per-sequence lengths, RoPE
// tables, stream generation. Same reasoning as the arena layer in
// `candle_nn::kv_cache::chunked` — the flat list IS the launch, and a struct
// between it and the `.cu` is what makes a mismatch hard to audit.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
pub(super) fn paged_prefill_batched_impl<'w>(
    wave: Option<&'w WaveGeneration>,
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &LiveTensor<'_>,
    k: &LiveTensor<'_>,
    v: &LiveTensor<'_>,
    b_sz: usize,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
    rope: &RopeRungs,
    rope_interleaved: bool,
    generation: &Generation,
    shared_pm: &std::cell::RefCell<Option<SharedPm>>,
    qsa: Option<&QsaSelection>,
    bounds: PrefillLaunchBounds,
) -> Result<LiveTensor<'w>> {
    // Ragged/varlen prefill. q/k/v arrive FLAT-packed:
    //   q: [total_q, n_head, head_dim], k/v: [total_q, n_kv_head, head_dim]
    // where total_q = Σ q_lens and the per-sequence token ranges are described
    // by `prefill_meta` (cu_seqlens_q / q_lens / kv_lens). `b_sz` is the number
    // of sequences (= caches.len() = q_lens.len()). Uniform prefill is the
    // special case where all q_lens are equal.
    let total_q: usize = q_lens.iter().sum();
    let max_add = q_lens.iter().copied().max().unwrap_or(0);
    if !(max_add >= 1
        && matches!(q.device(), Device::Cuda(_))
        && head_dim.is_multiple_of(32)
        && head_dim <= 256)
    {
        candle::bail!(
            "paged prefill batched attention not applicable: \
             total_q={}, device={:?}, head_dim={}",
            total_q,
            q.device(),
            head_dim
        );
    }

    // No reset and no truncation here: `wave_admit` did both for every layer
    // before the forward began, and repeating them would discard the chunks that
    // pass claimed.
    let mut use_chunks = caches
        .first()
        .and_then(|c| c.k_cache().chunked_arena_chunks())
        .is_some();

    // If not already chunked, create chunked backing and (if needed) migrate the existing prefix.
    if !use_chunks {
        let max_len = offsets.iter().copied().max().unwrap_or(0);
        let max_after = offsets
            .iter()
            .zip(q_lens.iter())
            .map(|(&o, &l)| o.saturating_add(l))
            .max()
            .unwrap_or(max_add);

        // If we're creating chunked backing from scratch (no prefix to migrate), use the
        // incoming K/V dtype. Empty caches may otherwise report a default dtype (often F32)
        // even when the caller intends BF16/F16 execution.
        //
        // IMPORTANT: The paged prefill kernel only supports BF16/F16. If F32 is detected,
        // we use BF16 for the arenas and convert K/V on scatter. The Q/K/V inputs to the
        // attention kernel itself will be converted to BF16 by ensure_dtype anyway.
        let raw_dtype = if max_len > 0 {
            caches
                .first()
                .map(|c| c.dtype())
                .unwrap_or_else(|| k.dtype())
        } else {
            k.dtype()
        };
        let dtype = match raw_dtype {
            DType::F32 | DType::F64 => DType::BF16, // Paged prefill only supports BF16/F16
            other => other,
        };

        // Migration from contiguous to chunked KV is not supported.
        // Callers should ensure they start with chunked caches.
        if max_len > 0 {
            candle::bail!(
                "migration from contiguous KV to chunked backing is not supported; \
                    please ensure caches start with chunked backing for batched prefill"
            );
        }

        let backing =
            ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, q.device(), max_after)?;

        // Allocate chunks for exactly each sequence's own new-token count — never
        // over-allocate to max(q_lens), which would leave extra tail chunks that
        // desync the decode writer slice.
        backing.ensure_for_offsets(offsets, q_lens)?;

        // Switch caches to shared chunked backing.
        for (i, cache) in caches.iter_mut().enumerate() {
            cache.set_chunked_backing(&backing, i, None)?;
            cache.set_current_seq_len(offsets[i])?;
        }
        use_chunks = true;
    }

    if !use_chunks {
        candle::bail!("chunked backing unavailable");
    }

    // No entry drain. The span that used to sit here existed only to absorb the
    // caller's in-flight tail so the first host-synced span did not inherit it —
    // an artefact of timing by synchronisation. GPU spans bracket their own work
    // with stream-ordered events, so a caller's tail is simply not inside them.
    let t_meta = profile_now();
    let (compute_dtype, _chunk_size) = {
        let first = caches
            .first()
            .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;

        // Use the cache's dtype — for quantized backings this returns F16 (the dequant
        // output dtype), and for float backings it returns the arena's actual dtype.
        // The paged attention kernels run in F16/BF16 only, so collapse F32/F64 reference-mode
        // float arenas to BF16 — matching the decode path (decode_utils compute_dtype) and the
        // chunked-backing dtype selection above. Q/K/V are cast to this dtype before launch.
        let collapse_compute = |d: DType| match d {
            DType::F32 | DType::F64 => DType::BF16,
            other => other,
        };
        let k_compute = collapse_compute(first.k_cache().dtype());
        let v_compute = collapse_compute(first.v_cache().dtype());
        if k_compute != v_compute {
            candle::bail!(
                "K and V caches require different compute dtypes: K={:?} V={:?}",
                k_compute,
                v_compute
            );
        }
        let compute_dtype = k_compute;

        let chunk_size = first
            .k_cache()
            .chunked_chunk_size()
            .ok_or_else(|| candle::Error::Msg("expected chunked chunk_size".into()))?;
        (compute_dtype, chunk_size)
    };

    pipeline_record("prefill:metadata", t_meta);

    let g_pack = gpu_span("prefill:pack", q.device());
    // --- prefill:pack spans tensor packing, chunk_meta, head_gids construction ---
    // Some model paths provide non-contiguous K/V; ensure contiguity once.
    let k = if k.is_contiguous() {
        k.clone()
    } else {
        k.contiguous()?
    };
    let v = if v.is_contiguous() {
        v.clone()
    } else {
        v.contiguous()?
    };

    // Q/K/V already arrive FLAT-packed in cu_seqlens_q token order:
    //   q_packed: [total_q, n_head, head_dim], k/v: [total_q, n_kv_head, head_dim]
    // (k/v were made contiguous just above). No transpose/reshape needed.
    let q_packed = if q.is_contiguous() {
        q.clone()
    } else {
        q.contiguous()?
    };
    let k_packed = k;
    let v_packed = v;
    debug_assert_eq!(q_packed.dim(0)?, total_q);

    // Varlen metadata (device tensors). Renamed to `*_dev` so the host
    // `q_lens: &[usize]` param stays in scope for the per-seq seqlen update.
    let (cu_seqlens_q, q_lens_dev, kv_lens) = if let Some((cu, ql, kv)) = prefill_meta {
        (cu.clone(), ql.clone(), kv.clone())
    } else {
        // Fallback: rebuild the ragged metadata from the host q_lens.
        let mut cu = Vec::with_capacity(b_sz + 1);
        cu.push(0u32);
        let mut acc = 0u32;
        for &l in q_lens {
            acc += l as u32;
            cu.push(acc);
        }
        let cu_seqlens_q = Tensor::from_vec(cu, b_sz + 1, q.device())?;
        let q_lens_dev = Tensor::from_vec(
            q_lens.iter().map(|&l| l as u32).collect::<Vec<_>>(),
            b_sz,
            q.device(),
        )?;
        let kv_lens = Tensor::from_vec(
            offsets
                .iter()
                .zip(q_lens.iter())
                .map(|(&o, &l)| (o + l) as u32)
                .collect::<Vec<_>>(),
            b_sz,
            q.device(),
        )?;
        (cu_seqlens_q, q_lens_dev, kv_lens)
    };

    let softmax_scale = 1f32 / (head_dim as f32).sqrt();

    // Any quantized storage policy must be kernel-native.
    {
        let first = caches
            .first()
            .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;
        if let Some(policy) = first.k_cache().chunked_storage_policy() {
            if policy.is_quantized() && !policy.is_kernel_native() {
                candle::bail!(
                    "storage policy uses a quantized format that the kernel cannot read natively; \
                     all quantized formats must be kernel-native"
                );
            }
        }
    }

    let header_upload = build_slot_headers(
        caches,
        q_lens,
        generation,
        shared_pm,
        offsets,
        rope,
        q.device(),
        wave,
    )?;
    let headers_ptr = header_upload.headers_ptr;

    g_pack.end();

    let g_kernel = gpu_span("prefill:kernel", q.device());
    let t_launch = profile_now();
    let out_packed = paged_prefill_attn_varlen_chunks(
        wave,
        &q_packed,
        &cu_seqlens_q,
        &q_lens_dev,
        &kv_lens,
        &k_packed,
        &v_packed,
        headers_ptr,
        compute_dtype,
        n_head,
        n_kv_head,
        head_dim,
        softmax_scale,
        rope,
        rope_interleaved,
        max_add,
        PrefillLaunchCut::new(q_lens, offsets, n_head, n_kv_head, head_dim, bounds),
        qsa,
    )?;
    g_kernel.end();
    pipeline_record("prefill:launch", t_launch);
    // Per-sequence written length (each sequence advanced by its own q_lens[i],
    // not the over-allocated max_add). Written by this kernel, not the decode
    // kernel, so the cached decode slot buffers are marked for the next sync to
    // bring up to date — a verify block is exactly this commit on a sequence
    // mid-decode. One commit for the layer's whole batch: one state lock, not
    // one per sequence.
    let t_commit = profile_now();
    KvCache::commit_written_tokens_batch(caches, offsets, q_lens)?;
    pipeline_record("prefill:commit", t_commit);

    // Nothing is primed for the decode that follows. The slot-state buffers
    // this launch's headers synced are the ones decode reads, and the commit
    // above marked them for decode's own sync to bring the writer region up to
    // date; decode claims its write chunk before its forward begins
    // (`ensure_for_batch_entries_all`). Priming here instead was a second
    // eager section per attention layer, every forward, for a buffer that
    // already existed.

    // Return the attention output FLAT-packed in cu_seqlens_q token order:
    // [total_q, n_head, head_dim]. The caller reshapes to [total_q, n_head*head_dim]
    // and runs the output projection per-token (no per-sequence split needed).
    Ok(out_packed)
}

/// Public wrapper: full-context prefill through the INT8 prefix-attention
/// kernel (docs/archived/prefill_optimization.md) — GQA-packed M,
/// slice-aligned tiles, int8 m16n8k32 QK/PV computed directly over the
/// quantized arena.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn paged_prefill_batched<'w>(
    wave: Option<&'w WaveGeneration>,
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &LiveTensor<'_>,
    k: &LiveTensor<'_>,
    v: &LiveTensor<'_>,
    b_sz: usize,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
    rope: &RopeRungs,
    rope_interleaved: bool,
    generation: &Generation,
    shared_pm: &std::cell::RefCell<Option<SharedPm>>,
    qsa: Option<&QsaSelection>,
) -> Result<LiveTensor<'w>> {
    paged_prefill_batched_impl(
        wave,
        caches,
        offsets,
        q,
        k,
        v,
        b_sz,
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
        PrefillLaunchBounds::PRODUCTION,
    )
}

/// [`paged_prefill_batched`] with the launch cut by `bounds` instead of the
/// production key chunk and row group — what the bit-identity tests use to cut
/// small launches into many windows and groups. The output is the same bits
/// whatever the bounds.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn paged_prefill_batched_bounded<'w>(
    wave: Option<&'w WaveGeneration>,
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &LiveTensor<'_>,
    k: &LiveTensor<'_>,
    v: &LiveTensor<'_>,
    b_sz: usize,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
    rope: &RopeRungs,
    rope_interleaved: bool,
    generation: &Generation,
    shared_pm: &std::cell::RefCell<Option<SharedPm>>,
    qsa: Option<&QsaSelection>,
    bounds: PrefillLaunchBounds,
) -> Result<LiveTensor<'w>> {
    paged_prefill_batched_impl(
        wave,
        caches,
        offsets,
        q,
        k,
        v,
        b_sz,
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
        bounds,
    )
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn paged_prefill_batched(
    // Concrete, not generic: `W` would be unconstrained at the `None` call
    // sites and force a turbofish on every one of them. Without CUDA there is
    // no wave domain, so the unit type is the honest stand-in.
    _wave: Option<&()>,
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &LiveTensor<'_>,
    k: &LiveTensor<'_>,
    v: &LiveTensor<'_>,
    b_sz: usize,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    _prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
    _rope: &RopeRungs,
    _rope_interleaved: bool,
    _generation: &Generation,
    _shared_pm: &std::cell::RefCell<Option<SharedPm>>,
    qsa: Option<&QsaSelection>,
) -> Result<Tensor> {
    // CPU fallback: per-sequence standard attention. The paged CUDA kernel is
    // the production path; this exists only for non-chunked CPU caches. It
    // reads the whole prefix, so a block-sparse selection is refused rather
    // than silently answered densely.
    if qsa.is_some() {
        candle::bail!(
            "QSA selection on the CPU prefill fallback: it reads the whole causal \
             prefix and cannot honour a block-sparse selection"
        );
    }
    // Q/K/V arrive FLAT-packed [total_q, n_*head, head_dim] in cu_seqlens order;
    // we slice each sequence's rows via the running prefix sum of q_lens.
    let first_cache = caches
        .first()
        .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;
    if first_cache.k_cache().chunked_arena_chunks().is_some() {
        candle::bail!(
                "batched multi-token attention cannot use chunked caches without paged prefill support; \
                 paged prefill requires head_dim to be a multiple of 32 and <= 256; device must be CUDA; \
                 head_dim={}, has_chunked_caches=true, supports_paged_prefill=false",
                head_dim
            );
    }

    let mut all_outputs = Vec::with_capacity(b_sz);
    let mut cu = 0usize;
    for (batch_idx, cache) in caches.iter_mut().enumerate() {
        let seq_len = q_lens[batch_idx];
        // Slice this sequence's flat rows and restore [1, n_*head, seq_len, head_dim].
        let q_seq = q
            .narrow(0, cu, seq_len)?
            .reshape((1, seq_len, n_head, head_dim))?
            .transpose(1, 2)?;
        let k_seq = k
            .narrow(0, cu, seq_len)?
            .reshape((1, seq_len, n_kv_head, head_dim))?
            .transpose(1, 2)?;
        let v_seq = v
            .narrow(0, cu, seq_len)?
            .reshape((1, seq_len, n_kv_head, head_dim))?
            .transpose(1, 2)?;
        cu += seq_len;

        // Append new K/V to cache
        let (k_cached, v_cached) = cache.append(&k_seq, &v_seq)?;
        let cache_len = k_cached.dim(2)?;

        // Repeat K/V heads for MQA/GQA
        let k_rep = crate::utils::repeat_kv(k_cached, n_head / n_kv_head)?;
        let v_rep = crate::utils::repeat_kv(v_cached, n_head / n_kv_head)?;

        // Standard scaled dot-product attention with causal mask
        let scale = 1.0 / (head_dim as f64).sqrt();
        let att = (q_seq.matmul(&k_rep.t()?)? * scale)?;

        let offset = offsets[batch_idx];
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..cache_len).map(move |j| {
                    if j > offset + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0f32
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, seq_len, cache_len), q.device())?;
        let mask = mask.to_dtype(att.dtype())?;
        let att = att.broadcast_add(&mask)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        // [1, n_head, seq_len, head_dim] -> [seq_len, n_head, head_dim] (flat row block)
        let attn_out = att
            .matmul(&v_rep.contiguous()?)?
            .transpose(1, 2)?
            .reshape((seq_len, n_head, head_dim))?;
        all_outputs.push(attn_out);
    }

    // Concatenate per-sequence outputs back into the flat [total_q, n_head, head_dim].
    Tensor::cat(&all_outputs, 0)
}
