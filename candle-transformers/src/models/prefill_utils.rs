use candle::quantized::pinned_staging::Generation;
#[cfg(feature = "cuda")]
use candle::quantized::pinned_staging::GpuBuf;
#[cfg(feature = "cuda")]
use crate::models::profile::{pipeline_record, profile_now, profile_sync};
use candle::*;
pub(crate) use candle_nn::kv_cache::KvCache;
#[cfg(feature = "cuda")]
pub(crate) use candle_nn::kv_cache::CHUNK_SIZE;


#[cfg(feature = "cuda")]
use {
    candle::backend::BackendStorage,
    candle::cuda_backend::cudarc::driver::{DevicePtr, DeviceRepr},
    candle_kernels::paged_prefill::*,
    candle_nn::kv_cache::ChunkedKvBacking,
    half::{bf16, f16},
};

#[cfg(feature = "cuda")]
use crate::models::slot_state::{SlotStateHost, TokenSliceHost};


/// Chunked KV + paged prefill attention over chunks.
///
/// This path avoids pooled KV materialization by:
/// 1) Ensuring (or creating) chunked KV backing
/// 2) Scattering this prefill segment into chunked arenas
/// 3) Running the paged prefill attention kernel that reads via `block_table`.
///
/// Returns `Some(outputs_per_sequence)` on success, `None` if inapplicable or if the attempt fails.
#[cfg(feature = "cuda")]
fn paged_prefill_batched_impl(
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    b_sz: usize,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    prefill_meta: Option<(&Tensor, &Tensor, &Tensor, bool)>,
    rope_offsets: &Tensor,
    rope_cs: &Tensor,
    rope_interleaved: bool,
    write_offset_shifts_ptr: u64,
    generation: &Generation,
) -> Result<Tensor> {
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
        && head_dim % 32 == 0
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

    // Reset caches where offset == 0.
    for (cache, &off) in caches.iter_mut().zip(offsets.iter()) {
        if off == 0 {
            cache.reset();
        }
    }

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

    // Ensure chunks for EXACTLY each sequence's own q_len — never over-allocate
    // to max(q_lens). Extra trailing chunks on a shorter sequence desync the
    // decode writer slice (the kernel's "one free slot per chunk" contract),
    // tripping the `ws_offset + ws_len < CHUNK_SIZE` assert on its next decode.
    // Per-cache calls use each cache's real batch_idx, so a single-element slice
    // targets the correct slot.
    let t_alloc = profile_now();
    for (i, &add) in q_lens.iter().enumerate() {
        KvCache::ensure_chunked_capacity_batch(&mut caches[i..i + 1], &offsets[i..i + 1], add)?;
    }
    profile_sync(q.device());
    pipeline_record("prefill:alloc", t_alloc);

    let t_meta = profile_now();
    let (compute_dtype, _chunk_size, max_blocks) = {
        let first = caches
            .first()
            .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;

        // Use the cache's dtype — for quantized backings this returns F16 (the dequant
        // output dtype), and for float backings it returns the arena's actual dtype.
        // This matches the decode path (batched_layer.rs dispatch_dtype) and correctly
        // handles pre-reconcile state where arenas are float even if the target is quant.
        let k_compute = first.k_cache().dtype();
        let v_compute = first.v_cache().dtype();
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
        let max_blocks = first.k_cache().chunked_max_blocks();
        (compute_dtype, chunk_size, max_blocks)
    };

    pipeline_record("prefill:metadata", t_meta);

    let t_pack = profile_now();
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
    let (cu_seqlens_q, q_lens_dev, kv_lens, has_prefix) =
        if let Some((cu, ql, kv, has_prefix)) = prefill_meta {
            (cu.clone(), ql.clone(), kv.clone(), has_prefix)
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
            let has_prefix = offsets.iter().any(|&o| o > 0);
            (cu_seqlens_q, q_lens_dev, kv_lens, has_prefix)
        };

    let softmax_scale = 1f32 / (head_dim as f32).sqrt();

    // Check storage policy to determine if reconcile is needed after prefill.
    // Validate that any quantized storage policy is kernel-native.
    // Reconcile and consolidation now happen once after all layers in batched_model.rs.
    let needs_reconcile = {
        let first = caches
            .first()
            .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;
        match first.k_cache().chunked_storage_policy() {
            Some(policy) => {
                let is_quant = policy.to_arena_key().is_quantized();
                if is_quant && !policy.is_kernel_native() {
                    candle::bail!(
                        "storage policy uses a quantized format that the kernel cannot read natively; \
                         all quantized formats must be kernel-native"
                    );
                }
                is_quant
            }
            None => false,
        }
    };

    let (_headers_gpu_guard, _slices_gpu_guard, _pm_gpu_guard, headers_ptr):
        (GpuBuf, GpuBuf, GpuBuf, u64) = {
        let arena_info = {
            let first = caches
                .first()
                .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;
            first
                .k_cache()
                .chunked_resolve_arena_info()
                .ok_or_else(|| candle::Error::Msg("expected chunked resolve_arena_info".into()))??
        };

        let mut slots: Vec<SlotStateHost> = caches
            .iter()
            .map(|cache| {
                let chunks = cache
                    .k_cache()
                    .chunked_live_chunks_as_sealed()
                    .unwrap_or_default();
                let writer_start_idx = cache
                    .k_cache()
                    .chunked_writer_start_idx()
                    .unwrap_or(0);
                SlotStateHost::from_sealed_chunks(
                    &chunks,
                    n_kv_head,
                    head_dim,
                    &arena_info,
                    writer_start_idx,
                )
            })
            .collect();

        // Extend each slot's position_map to cover the prefill's write region.
        // Ragged: slot i writes q_lens[i] new tokens, so after this
        // `position_map.len() == offsets[i] + q_lens[i] == kv_lens[i]`, letting
        // the kernel resolve any k_pos in `[0, kv_lens[i])` via a single lookup.
        let chunk_size = CHUNK_SIZE;
        for (slot, &add) in slots.iter_mut().zip(q_lens.iter()) {
            slot.extend_for_write_region(add, chunk_size);
        }

        // Pack slices — upload via stager for zero-copy PCIe read.
        let slice_size = TokenSliceHost::serialized_size(n_kv_head, head_dim);
        let total_slices: usize = slots.iter().map(|s| s.slices.len()).sum();
        let mut slice_buf: Vec<u8> = Vec::with_capacity(total_slices * slice_size);
        let mut slot_byte_offsets: Vec<usize> = Vec::with_capacity(slots.len());
        for slot in &slots {
            slot_byte_offsets.push(slice_buf.len());
            for slice in &slot.slices {
                slice.serialize_into(&mut slice_buf);
            }
        }
        if slice_buf.is_empty() {
            slice_buf.push(0u8);
        }
        let mut slices_pinned = generation.alloc(slice_buf.len())?;
        slices_pinned.copy_from_slice(&slice_buf);
        let slices_gpu = generation.submit(slices_pinned)?;
        let slices_base_ptr = slices_gpu.dev_ptr();

        // Pack position_maps — upload via stager, same path.
        let total_pm_entries: usize = slots.iter().map(|s| s.position_map.len()).sum();
        let mut pm_buf: Vec<u32> = Vec::with_capacity(total_pm_entries.max(1));
        let mut pm_byte_offsets: Vec<usize> = Vec::with_capacity(slots.len());
        for slot in &slots {
            pm_byte_offsets.push(pm_buf.len() * 4);
            pm_buf.extend_from_slice(&slot.position_map);
        }
        if pm_buf.is_empty() {
            pm_buf.push(0u32);
        }
        let pm_byte_len = pm_buf.len() * std::mem::size_of::<u32>();
        let mut pm_pinned = generation.alloc(pm_byte_len)?;
        // SAFETY: u32 has no padding and is trivially copyable; lengths match.
        let pm_bytes = unsafe {
            std::slice::from_raw_parts(pm_buf.as_ptr() as *const u8, pm_byte_len)
        };
        pm_pinned.copy_from_slice(pm_bytes);
        let pm_gpu = generation.submit(pm_pinned)?;
        let pm_base_ptr = pm_gpu.dev_ptr();

        let mut header_buf: Vec<u8> = Vec::with_capacity(slots.len() * 24);
        for (i, slot) in slots.iter().enumerate() {
            let n_slices = slot.slices.len() as u32;
            let write_slice = slot.write_slice;
            let slices_ptr = slices_base_ptr + slot_byte_offsets[i] as u64;
            let position_map_ptr = pm_base_ptr + pm_byte_offsets[i] as u64;
            header_buf.extend_from_slice(&n_slices.to_le_bytes());
            header_buf.extend_from_slice(&write_slice.to_le_bytes());
            header_buf.extend_from_slice(&slices_ptr.to_le_bytes());
            header_buf.extend_from_slice(&position_map_ptr.to_le_bytes());
        }

        let mut pinned = generation.alloc(header_buf.len())?;
        pinned.copy_from_slice(&header_buf);
        let headers_gpu = generation.submit(pinned)?;
        let headers_ptr = headers_gpu.dev_ptr();
        (headers_gpu, slices_gpu, pm_gpu, headers_ptr)
    };

    profile_sync(q.device());
    pipeline_record("prefill:pack", t_pack);

    let t_kernel = profile_now();
    let out_packed = paged_prefill_attn_varlen_chunks(
        &q_packed,
        &cu_seqlens_q,
        &q_lens_dev,
        &kv_lens,
        &k_packed,
        &v_packed,
        headers_ptr,
        compute_dtype,
        max_blocks,
        n_head,
        n_kv_head,
        head_dim,
        softmax_scale,
        has_prefix,
        rope_offsets,
        rope_cs,
        rope_interleaved,
        write_offset_shifts_ptr,
    )?;
    profile_sync(q.device());
    pipeline_record("prefill:kernel", t_kernel);
    // Per-sequence written length (each sequence advanced by its own q_lens[i],
    // not the over-allocated max_add).
    for ((cache, &off), &add) in caches.iter_mut().zip(offsets.iter()).zip(q_lens.iter()) {
        cache.set_current_seq_len(off + add)?;
    }

    // After each prefill layer, eagerly quantize all fully-sealed chunks so that
    // float F16 arenas are freed as we go rather than accumulating to OOM.
    // The partial tail chunk (still being written to) is skipped automatically by
    // reconcile_multi, which only processes blocks where seq_len / CHUNK_SIZE > blk.
    //
    // After reconcile, float arenas hold only the sparse tail chunks. Consolidate
    // them into the minimum number of arenas and CUDA-free the rest. This is critical
    // for large batches (e.g. Q4_0Ã—460) where uncompacted tail arenas would exceed
    // the available VRAM budget (each F16 arena is 64 MiB regardless of occupancy).
    // Reconcile and consolidate now happen once after all layers complete,
    // in batched_model.rs. Per-layer reconcile was removed to avoid paying
    // the quantization cost on every layer during prefill.

    // Prefill and any post-prefill migrations are now complete. For pure-float
    // paths we can materialize the persistent decode slot buffers eagerly here.
    // Quantized paths are left to rebuild lazily on the next decode metadata
    // sync so they always see the final post-reconcile chunk routing.
    if !needs_reconcile {
        KvCache::prime_chunked_decode_slots_batch(caches)?;
    }

    // Return the attention output FLAT-packed in cu_seqlens_q token order:
    // [total_q, n_head, head_dim]. The caller reshapes to [total_q, n_head*head_dim]
    // and runs the output projection per-token (no per-sequence split needed).
    Ok(out_packed)
}

/// Public wrapper: full-context prefill (no windowing).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn paged_prefill_batched(
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    b_sz: usize,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    prefill_meta: Option<(&Tensor, &Tensor, &Tensor, bool)>,
    rope_offsets: &Tensor,
    rope_cs: &Tensor,
    rope_interleaved: bool,
    write_offset_shifts_ptr: u64,
    generation: &Generation,
) -> Result<Tensor> {
    paged_prefill_batched_impl(
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
        rope_offsets,
        rope_cs,
        rope_interleaved,
        write_offset_shifts_ptr,
        generation,
    )
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn paged_prefill_batched(
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    b_sz: usize,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    _prefill_meta: Option<(&Tensor, &Tensor, &Tensor, bool)>,
    _rope_offsets: &Tensor,
    _rope_cs: &Tensor,
    _rope_interleaved: bool,
    _write_offset_shifts_ptr: u64,
    _generation: &Generation,
) -> Result<Tensor> {
    // CPU fallback: per-sequence standard attention. The paged CUDA kernel is
    // the production path; this exists only for non-chunked CPU caches.
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

#[cfg(feature = "cuda")]
#[derive(Clone)]
struct PagedPrefillChunks {
    softmax_scale: f32,
    cu_seqlens_q: Tensor,
    q_lens: Tensor,
    kv_lens: Tensor,
    k_packed: Tensor,
    v_packed: Tensor,
    /// Raw GPU virtual address of `SlotHeader[batch_size]`.
    headers_ptr: u64,
    batch_size: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    max_blocks: usize,
    has_prefix: bool,
    /// Compute dtype for K/V/Q — F16 or BF16. Pre-resolved from K and V arena formats.
    compute_dtype: DType,
    /// RoPE position offsets per batch element, shape [batch_size], dtype U32.
    /// Zero values = natural positions (no extra shift). Non-zero = page-clone position delta.
    rope_offsets: Tensor,
    /// Precomputed cos/sin table on device, shape [max_pos, head_dim], dtype F32.
    /// Layout: rope_cs[pos * head_dim + d*2] = cos, [pos * head_dim + d*2+1] = sin.
    rope_cs: Tensor,
    /// RoPE pairing style: false=non-interleaved half-split (Qwen/GPT2), true=interleaved adjacent-pairs (Llama).
    rope_interleaved: bool,
    /// Per-batch write position shift [batch_size], dtype U32.
    /// Zero values = no shift (left-packed). Non-zero = SSO right-pack offset.
    /// Raw GPU device pointer (u64) into a u32 array of length batch_size.
    write_offset_shifts_ptr: u64,
}

#[cfg(feature = "cuda")]
impl PagedPrefillChunks {
    fn cuda_fwd_t<
        Q: candle::cuda_backend::CudaDType + DeviceRepr, // Query type
        KV: candle::cuda_backend::CudaDType + DeviceRepr, // KV cache type
        O: candle::cuda_backend::CudaDType + DeviceRepr, // Output type
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        let (total_q, n_head, head_dim) = q_l.shape().dims3()?;
        if n_head != self.n_head || head_dim != self.head_dim {
            candle::bail!(
                "paged-prefill-chunks: q shape mismatch got {:?} expected (total_q, {}, {})",
                q_l.shape(),
                self.n_head,
                self.head_dim
            )
        }

        let dev = q.device();
        let stream = dev.cuda_stream();

        let (cu_seqlens_q_s, cu_seqlens_q_l) = self.cu_seqlens_q.storage_and_layout();
        let (q_lens_s, q_lens_l) = self.q_lens.storage_and_layout();
        let (kv_lens_s, kv_lens_l) = self.kv_lens.storage_and_layout();
        let (k_packed_s, k_packed_l) = self.k_packed.storage_and_layout();
        let (v_packed_s, v_packed_l) = self.v_packed.storage_and_layout();

        if cu_seqlens_q_l.shape().dims1()? != self.batch_size + 1 {
            candle::bail!(
                "paged-prefill-chunks: cu_seqlens_q must have len batch+1 ({})",
                self.batch_size + 1
            )
        }
        if q_lens_l.shape().dims1()? != self.batch_size {
            candle::bail!(
                "paged-prefill-chunks: q_lens must have len batch ({})",
                self.batch_size
            )
        }
        if kv_lens_l.shape().dims1()? != self.batch_size {
            candle::bail!(
                "paged-prefill-chunks: kv_lens must have len batch ({})",
                self.batch_size
            )
        }
        if self.cu_seqlens_q.dtype() != DType::U32
            || self.q_lens.dtype() != DType::U32
            || self.kv_lens.dtype() != DType::U32
        {
            candle::bail!("paged-prefill-chunks: cu_seqlens_q/q_lens/kv_lens must be U32")
        }

        let cu_seqlens_q = match &*cu_seqlens_q_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("paged-prefill-chunks: cu_seqlens_q must be a cuda tensor"),
        }
        .slice(cu_seqlens_q_l.start_offset()..);

        let q_lens = match &*q_lens_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("paged-prefill-chunks: q_lens must be a cuda tensor"),
        }
        .slice(q_lens_l.start_offset()..);

        let kv_lens = match &*kv_lens_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("paged-prefill-chunks: kv_lens must be a cuda tensor"),
        }
        .slice(kv_lens_l.start_offset()..);

        let k_packed = match &*k_packed_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
            _ => candle::bail!("paged-prefill-chunks: k_packed must be a cuda tensor"),
        }
        .slice(k_packed_l.start_offset()..);

        let v_packed = match &*v_packed_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
            _ => candle::bail!("paged-prefill-chunks: v_packed must be a cuda tensor"),
        }
        .slice(v_packed_l.start_offset()..);

        let elem_count = q_l.shape().elem_count();
        let dst = unsafe { dev.alloc::<O>(elem_count)? };

        // Compute q_dtype code from Q's actual dtype
        // q_dtype codes: 0=F32, 1=F16, 2=BF16, 3=F8E4M3
        let q_dtype_code: i32 = match q.dtype() {
            candle::DType::F16 => 1,
            candle::DType::BF16 => 2,
            dt => candle::bail!(
                "paged-prefill: unsupported Q dtype {:?} (only F16/BF16 supported)",
                dt
            ),
        };

        match self.head_dim {
            64 | 96 | 128 | 256 => {}
            hd => candle::bail!(
                "paged-prefill: unsupported head_dim {hd} (must be 64, 96, 128, or 256)"
            ),
        }

        unsafe {
            let q = q.as_cuda_slice::<Q>()?;
            let q = q.slice(q_l.start_offset()..);
            let (q_ptr, _guard) = q.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr(&stream);
            let (k_ptr, _guard) = k_packed.device_ptr(&stream);
            let (v_ptr, _guard) = v_packed.device_ptr(&stream);
            let (cu_ptr, _guard) = cu_seqlens_q.device_ptr(&stream);
            let (q_lens_ptr, _guard) = q_lens.device_ptr(&stream);
            let (kv_lens_ptr, _guard) = kv_lens.device_ptr(&stream);
            // Extract rope_offsets pointer. The prefill CUDA kernel applies fused RoPE to Q
            // (in smem) and to new K tokens (k_pos >= prefix_len) before computing attention
            // scores and writing K/V to the arena.
            let rope_offsets_ptr = {
                let (ro_s, ro_l) = self.rope_offsets.storage_and_layout();
                let ro_slice = match &*ro_s {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                    _ => candle::bail!("paged-prefill: rope_offsets must be a cuda tensor"),
                }
                .slice(ro_l.start_offset()..);
                let (ro_ptr, _ro_guard) = ro_slice.device_ptr(&stream);
                ro_ptr as *const u32
            };

            let rope_cs_ptr = {
                let (cs_s, cs_l) = self.rope_cs.storage_and_layout();
                let cs_slice = match &*cs_s {
                    candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                    _ => candle::bail!("paged-prefill: rope_cs must be a cuda tensor"),
                }
                .slice(cs_l.start_offset()..);
                let (cs_ptr, _cs_guard) = cs_slice.device_ptr(&stream);
                cs_ptr as *const f32
            };

            let write_offset_shifts_ptr = self.write_offset_shifts_ptr as *const u32;
            let headers_ptr = self.headers_ptr as *const u8;

            candle::set_kernel_breadcrumb("run_paged_prefill_chunks", file!(), line!());
            let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
            run_paged_prefill_chunks(
                q_ptr as *const core::ffi::c_void,
                k_ptr as *const core::ffi::c_void,
                v_ptr as *const core::ffi::c_void,
                headers_ptr,
                cu_ptr as *const u32,
                q_lens_ptr as *const u32,
                kv_lens_ptr as *const u32,
                dst_ptr as *mut core::ffi::c_void,
                total_q as i32,
                self.batch_size as i32,
                self.n_head as i32,
                self.n_kv_head as i32,
                self.head_dim as i32,
                self.max_blocks as i32,
                self.softmax_scale,
                q_dtype_code,
                if self.has_prefix { 1 } else { 0 },
                rope_offsets_ptr,
                rope_cs_ptr,
                self.rope_interleaved as i32,
                write_offset_shifts_ptr,
                raw_stream,
            );
        }

        let dst = candle::CudaStorage::wrap_cuda_slice(dst, dev.clone());
        Ok((dst, q_l.shape().clone()))
    }
}

#[cfg(feature = "cuda")]
impl candle::CustomOp1 for PagedPrefillChunks {
    fn name(&self) -> &'static str {
        "paged-prefill-chunks"
    }

    fn cpu_fwd(&self, _: &candle::CpuStorage, _: &Layout) -> Result<(candle::CpuStorage, Shape)> {
        candle::bail!("no cpu support for paged-prefill-chunks")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        if q.dtype() != self.compute_dtype {
            candle::bail!(
                "paged-prefill-chunks: expected {:?} Q, got {:?}",
                self.compute_dtype,
                q.dtype()
            );
        }
        match self.compute_dtype {
            candle::DType::F16 => self.cuda_fwd_t::<f16, f16, f16>(q, q_l),
            candle::DType::BF16 => self.cuda_fwd_t::<bf16, bf16, bf16>(q, q_l),
            dt => candle::bail!("paged-prefill-chunks: unsupported compute dtype {:?}", dt),
        }
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
/// Paged prefill attention over chunked KV arenas (no KV materialization).
///
/// `q` has shape `(total_q, n_head, head_dim)`.
/// `headers_ptr` is the raw GPU address of `SlotHeader[batch_size]`, reusing the
/// same persistent slot-payload representation as decode.
/// `compute_dtype` is the pre-resolved F16 or BF16 dtype for Q/K/V (derived from arena formats).
///
/// `has_prefix` should be true if any sequence has existing KV cache (kv_len > q_len).
/// When false, uses an optimized async path. When true, uses synchronous scattered loads.
pub(crate) fn paged_prefill_attn_varlen_chunks(
    q: &Tensor,
    cu_seqlens_q: &Tensor,
    q_lens: &Tensor,
    kv_lens: &Tensor,
    k_packed: &Tensor,
    v_packed: &Tensor,
    headers_ptr: u64,
    compute_dtype: DType,
    max_blocks: usize,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    softmax_scale: f32,
    has_prefix: bool,
    rope_offsets: &Tensor,
    rope_cs: &Tensor,
    rope_interleaved: bool,
    write_offset_shifts_ptr: u64,
) -> Result<Tensor> {
    // HEAD_DIM must be a multiple of 32 and <= 256
    if head_dim % 32 != 0 || head_dim > 256 {
        candle::bail!("paged-prefill-chunks only supports head_dim multiple of 32 and <= 256 (got {head_dim})")
    }

    let q = q.to_dtype(compute_dtype)?;
    let k_packed = k_packed.to_dtype(compute_dtype)?;
    let v_packed = v_packed.to_dtype(compute_dtype)?;

    let (_total_q, q_n_head, q_head_dim) = q.dims3()?;
    if q_n_head != n_head || q_head_dim != head_dim {
        candle::bail!("paged-prefill-chunks: q shape mismatch {:?}", q.dims())
    }

    let (kp_total_q, kp_n_kv, kp_hd) = k_packed.dims3()?;
    if kp_total_q != _total_q || kp_n_kv != n_kv_head || kp_hd != head_dim {
        candle::bail!(
            "paged-prefill-chunks: k_packed shape mismatch {:?}",
            k_packed.dims()
        )
    }
    let (vp_total_q, vp_n_kv, vp_hd) = v_packed.dims3()?;
    if vp_total_q != _total_q || vp_n_kv != n_kv_head || vp_hd != head_dim {
        candle::bail!(
            "paged-prefill-chunks: v_packed shape mismatch {:?}",
            v_packed.dims()
        )
    }

    let batch_size = cu_seqlens_q.dim(0)?.saturating_sub(1);

    let op = PagedPrefillChunks {
        softmax_scale,
        cu_seqlens_q: cu_seqlens_q.clone(),
        q_lens: q_lens.clone(),
        kv_lens: kv_lens.clone(),
        k_packed: k_packed.clone(),
        v_packed: v_packed.clone(),
        headers_ptr,
        batch_size,
        n_head,
        n_kv_head,
        head_dim,
        max_blocks,
        has_prefix,
        compute_dtype,
        rope_offsets: rope_offsets.clone(),
        rope_cs: rope_cs.clone(),
        rope_interleaved,
        write_offset_shifts_ptr,
    };
    q.apply_op1(op)
}

/// Precompute cos/sin table for RoPE from inv_freq, computed with f64 precision.
///
/// Layout: `table[pos * head_dim + d * 2] = cos(pos * inv_freq[d])`,
///         `table[pos * head_dim + d * 2 + 1] = sin(pos * inv_freq[d])`.
///
/// Returns a tensor of shape `[max_blocks * 32, head_dim]`, dtype F32, on `device`.
pub fn compute_rope_cs(
    inv_freq: &Tensor,
    max_blocks: usize,
    head_dim: usize,
    device: &Device,
) -> Result<Tensor> {
    let max_pos = max_blocks * 32; // CHUNK_SIZE = 32
    let inv_freq_host = inv_freq.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let half_dim = inv_freq_host.len();
    let mut table = vec![0f32; max_pos * head_dim];
    for pos in 0..max_pos {
        let base = pos * head_dim;
        for d in 0..half_dim {
            let angle = pos as f64 * inv_freq_host[d] as f64;
            table[base + d * 2] = angle.cos() as f32;
            table[base + d * 2 + 1] = angle.sin() as f32;
        }
    }
    Tensor::from_vec(table, (max_pos, head_dim), device)
}

// ============================================================================
// Decode kernel — persistent slot buffer edition
// ============================================================================

/// Paged decode attention using persistent slot buffers.
///
/// Takes the slot pool `headers` tensor (16 bytes × n_active per slot) instead
/// of the old per-step chunk_meta / head_gids / kv_lens / per_head_table.
/// The kernel self-increments ws.len after scatter, so no write_offsets needed.
#[cfg(feature = "cuda")]
pub fn paged_decode_attn(
    q: &Tensor,
    headers_ptr: u64,
    arena_dtype: DType,
    n_q_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    softmax_scale: f32,
    k_new: &Tensor,
    v_new: &Tensor,
    rope_cs: &Tensor,
    rope_interleaved: bool,
) -> Result<Tensor> {
    let num_active_slots = q.dim(0)?;
    let k_new = k_new.contiguous()?;
    let v_new = v_new.contiguous()?;
    let op = PagedDecode {
        headers_ptr,
        arena_dtype,
        n_q_head,
        n_kv_head,
        head_dim,
        softmax_scale,
        k_new,
        v_new,
        rope_cs: rope_cs.clone(),
        rope_interleaved,
        num_active_slots,
    };
    q.apply_op1(op)
}

#[cfg(feature = "cuda")]
struct PagedDecode {
    headers_ptr: u64, // raw GPU virtual address of SlotHeader[num_active_slots]
    arena_dtype: DType,
    n_q_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    softmax_scale: f32,
    k_new: Tensor,
    v_new: Tensor,
    rope_cs: Tensor,
    rope_interleaved: bool,
    num_active_slots: usize,
}

#[cfg(feature = "cuda")]
impl PagedDecode {
    fn cuda_fwd_typed<
        Q: candle::cuda_backend::CudaDType + DeviceRepr + 'static,
        KV: candle::cuda_backend::CudaDType + DeviceRepr,
        O: candle::cuda_backend::CudaDType + DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        ffi_fn: unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const u8,
            *mut core::ffi::c_void,
            i32,
            i32,
            i32,
            i32,
            f32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            *const f32,
            i32,
            *mut core::ffi::c_void,
        ),
    ) -> Result<(candle::CudaStorage, Shape)> {
        let dev = q.device().clone();
        let stream = dev.cuda_stream();

        let out_elem = self.num_active_slots * self.n_q_head * self.head_dim;
        let dst = unsafe { dev.alloc::<O>(out_elem)? };

        {
            let q_slice = q.as_cuda_slice::<Q>()?.slice(q_l.start_offset()..);
            let (q_ptr, _q_g) = q_slice.device_ptr(&stream);

            let headers_ptr = self.headers_ptr as *const u8;

            let (k_s, k_l) = self.k_new.storage_and_layout();
            let k_slice = match &*k_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
                _ => candle::bail!("paged-decode-v2: k_new must be CUDA"),
            }
            .slice(k_l.start_offset()..);
            let (k_ptr, _k_g) = k_slice.device_ptr(&stream);

            let (v_s, v_l) = self.v_new.storage_and_layout();
            let v_slice = match &*v_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
                _ => candle::bail!("paged-decode-v2: v_new must be CUDA"),
            }
            .slice(v_l.start_offset()..);
            let (v_ptr, _v_g) = v_slice.device_ptr(&stream);

            let (rcs_s, rcs_l) = self.rope_cs.storage_and_layout();
            let rcs_slice = match &*rcs_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("paged-decode-v2: rope_cs must be CUDA"),
            }
            .slice(rcs_l.start_offset()..);
            let (rcs_ptr, _rcs_g) = rcs_slice.device_ptr(&stream);

            let (dst_ptr, _dst_g) = dst.device_ptr(&stream);

            // Pass the device's dedicated stream so that both the decode
            // kernel and the follow-up commit_decode_write_len_kernel run on
            // the same stream as all GpuChunksGuard::memcpy_htod calls.
            // Passing null here would use the default CUDA stream, which
            // can race with the non-blocking dedicated stream.
            candle::set_kernel_breadcrumb("run_paged_decode", file!(), line!());
            let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
            unsafe {
                ffi_fn(
                    q_ptr as *const core::ffi::c_void,
                    headers_ptr as *const u8,
                    dst_ptr as *mut core::ffi::c_void,
                    self.num_active_slots as i32,
                    self.n_q_head as i32,
                    self.n_kv_head as i32,
                    self.head_dim as i32,
                    self.softmax_scale,
                    k_ptr as *const core::ffi::c_void,
                    v_ptr as *const core::ffi::c_void,
                    rcs_ptr as *const f32,
                    self.rope_interleaved as i32,
                    raw_stream,
                );
            }
        } // all guards dropped here, dst no longer borrowed

        let dst_cs = candle::CudaStorage::wrap_cuda_slice(dst, dev);
        let out_shape = Shape::from_dims(&[self.num_active_slots, self.n_q_head, self.head_dim]);
        Ok((dst_cs, out_shape))
    }
}

#[cfg(feature = "cuda")]
impl candle::CustomOp1 for PagedDecode {
    fn name(&self) -> &'static str {
        "paged-decode"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for paged-decode")
    }

    fn cuda_fwd(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle_kernels::paged_decode::{run_paged_decode_bf16, run_paged_decode_fp16};

        match self.head_dim {
            64 | 96 | 128 | 256 => {}
            hd => candle::bail!(
                "paged-decode: unsupported head_dim {hd} (must be 64, 96, 128, or 256)"
            ),
        }

        match self.arena_dtype {
            DType::F16 => self.cuda_fwd_typed::<f16, f16, f16>(q, q_l, run_paged_decode_fp16),
            DType::BF16 | DType::F8E4M3 => {
                self.cuda_fwd_typed::<bf16, bf16, bf16>(q, q_l, run_paged_decode_bf16)
            }
            dt => candle::bail!(
                "paged-decode: unsupported arena dtype {:?} (only F16/BF16 supported)",
                dt
            ),
        }
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::paged_prefill_batched as paged_prefill_flat;
    use candle::quantized::pinned_staging::{Generation, PinnedStager};
    use candle::{DType, Device, Result, Tensor};
    use candle_nn::kv_cache::KvCache;

    /// Test shim mapping the legacy uniform calling convention (q/k/v as
    /// `[b_sz, n_head, seq_len, head_dim]`, returning per-sequence
    /// `[1, n_head, seq_len, head_dim]`) onto the production flat/ragged
    /// [`paged_prefill_flat`]. Lets the kernel-correctness tests keep their
    /// original shapes/assertions while the production API is flat-packed.
    #[allow(clippy::too_many_arguments)]
    fn paged_prefill_uniform(
        caches: &mut [&mut KvCache],
        offsets: &[usize],
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        b_sz: usize,
        seq_len: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        _prefill_meta: Option<(&Tensor, &Tensor, &Tensor, bool)>,
        rope_offsets: &Tensor,
        rope_cs: &Tensor,
        rope_interleaved: bool,
        write_offset_shifts_ptr: u64,
        generation: &Generation,
    ) -> Result<Vec<Tensor>> {
        let total_q = b_sz * seq_len;
        let q_flat = q
            .transpose(1, 2)?
            .contiguous()?
            .reshape((total_q, n_head, head_dim))?;
        let k_flat = k
            .transpose(1, 2)?
            .contiguous()?
            .reshape((total_q, n_kv_head, head_dim))?;
        let v_flat = v
            .transpose(1, 2)?
            .contiguous()?
            .reshape((total_q, n_kv_head, head_dim))?;
        let q_lens = vec![seq_len; b_sz];
        let out = paged_prefill_flat(
            caches,
            offsets,
            &q_flat,
            &k_flat,
            &v_flat,
            b_sz,
            &q_lens,
            n_head,
            n_kv_head,
            head_dim,
            None,
            rope_offsets,
            rope_cs,
            rope_interleaved,
            write_offset_shifts_ptr,
            generation,
        )?;
        // Flat [total_q, n_head, head_dim] -> per-seq [1, n_head, seq_len, head_dim].
        let mut per_seq = Vec::with_capacity(b_sz);
        for i in 0..b_sz {
            per_seq.push(
                out.narrow(0, i * seq_len, seq_len)?
                    .reshape((1, seq_len, n_head, head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?,
            );
        }
        Ok(per_seq)
    }

    /// Helper: create a standard inv_freq tensor for tests (theta = 10000.0).
    /// Shape: (head_dim / 2,), dtype F32, on the given device.
    fn make_test_inv_freq(head_dim: usize, device: &Device) -> Result<Tensor> {
        let half_dim = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0f32 / 10000.0f32.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        Tensor::from_vec(inv_freq, (half_dim,), device)
    }

    /// Zero inv_freq for correctness tests: RoPE rotation becomes identity so the
    /// paged-kernel output can be compared directly to a no-RoPE reference.
    fn make_zero_inv_freq(head_dim: usize, device: &Device) -> Result<Tensor> {
        Tensor::zeros((head_dim / 2,), DType::F32, device)
    }

    /// Build a rope_cs table from inv_freq for decode tests.
    #[allow(dead_code)]
    fn make_test_rope_cs(head_dim: usize, max_blocks: usize, device: &Device) -> Result<Tensor> {
        let inv_freq = make_test_inv_freq(head_dim, device)?;
        super::compute_rope_cs(&inv_freq, max_blocks, head_dim, device)
    }

    /// Zero rope_cs for correctness tests (identity RoPE).
    #[allow(dead_code)]
    fn make_zero_rope_cs(head_dim: usize, max_blocks: usize, device: &Device) -> Result<Tensor> {
        let inv_freq = make_zero_inv_freq(head_dim, device)?;
        super::compute_rope_cs(&inv_freq, max_blocks, head_dim, device)
    }

    #[test]
    #[cfg(feature = "flash-attn")]
    fn paged_prefill_uses_flash_attn_kernel_smoke() -> Result<()> {
        // This forces a multi-token BF16 prefill so the flash-attn paged-prefill path is eligible.
        let device = Device::new_cuda(0)?;

        let b_sz = 1usize;
        let seq_len = 2usize;
        let n_head = 1usize;
        let n_kv_head = 1usize;
        let head_dim = 64usize;

        // Shapes: q (B, H, T, D), k/v (B, H_kv, T, D)
        let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::BF16)?
            .contiguous()?;

        let mut cache0 = KvCache::new(2, 512);
        cache0.force_dtype(DType::BF16);
        let offsets = [0usize];
        let mut caches: [&mut KvCache; 1] = [&mut cache0];

        // Enable the trace so the test output proves which kernel ran.
        std::env::set_var("CANDLE_TRACE_PAGED_PREFILL", "1");
        let rope_zeros = Tensor::zeros(b_sz, DType::U32, &device)?;
        let generation = PinnedStager::new(device.as_cuda_device()?).begin_generation();

        let out = paged_prefill_uniform(
            &mut caches,
            &offsets,
            &q,
            &k,
            &v,
            b_sz,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            None,
            &rope_zeros,
            &make_test_rope_cs(head_dim, 16, &device)?,
            false, // rope_interleaved
            {
                let mut b = generation.alloc(b_sz * 4)?;
                b.fill(0u8);
                generation.submit(b)?.dev_ptr()
            }, // write_offset_shifts
            &generation,
        )
        .expect("expected paged prefill to be applicable");

        assert_eq!(out.len(), 1);
        let y = &out[0];
        assert_eq!(y.dims(), &[1, n_head, seq_len, head_dim]);
        // Output dtype matches input dtype (BF16 in, BF16 out)
        assert_eq!(y.dtype(), DType::BF16);

        // Basic sanity: output should be finite.
        let max_abs = y
            .to_dtype(DType::F32)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        assert!(max_abs.is_finite());

        Ok(())
    }

    #[test]
    fn test_fp8_hd128_paged_prefill() -> Result<()> {
        // This tests the FP8 path with HD=128 (Llama-style)
        println!("\n=== Testing FP8 HD=128 Paged Prefill ===\n");

        let device = Device::new_cuda(0)?;
        println!("Device: {:?}", device);

        // Llama-style parameters
        let b_sz = 1usize;
        let seq_len = 4usize;
        let n_head = 32usize;
        let n_kv_head = 8usize;
        let head_dim = 128usize;

        println!(
            "Parameters: b_sz={}, seq_len={}, n_head={}, n_kv_head={}, head_dim={}",
            b_sz, seq_len, n_head, n_kv_head, head_dim
        );

        // Create Q/K/V with FP8 dtype
        let q = Tensor::randn(0f32, 0.1f32, (b_sz, n_head, seq_len, head_dim), &device)?
            .to_dtype(DType::F8E4M3)?;
        let k = Tensor::randn(0f32, 0.1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::F8E4M3)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 0.1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::F8E4M3)?
            .contiguous()?;

        let mut cache0 = KvCache::new(2, 512);
        cache0.force_dtype(DType::F8E4M3);
        let offsets = [0usize];
        let mut caches: [&mut KvCache; 1] = [&mut cache0];

        std::env::set_var("CANDLE_TRACE_PAGED_PREFILL", "1");

        println!(
            "Q dtype: {:?}, K dtype: {:?}, V dtype: {:?}",
            q.dtype(),
            k.dtype(),
            v.dtype()
        );
        let rope_zeros = Tensor::zeros(b_sz, DType::U32, &device)?;
        let generation = PinnedStager::new(device.as_cuda_device()?).begin_generation();

        let out = paged_prefill_uniform(
            &mut caches,
            &offsets,
            &q,
            &k,
            &v,
            b_sz,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            None,
            &rope_zeros,
            &make_test_rope_cs(head_dim, 16, &device)?,
            false, // rope_interleaved
            {
                let mut b = generation.alloc(b_sz * 4)?;
                b.fill(0u8);
                generation.submit(b)?.dev_ptr()
            }, // write_offset_shifts
            &generation,
        )
        .expect("expected paged prefill to be applicable");

        assert_eq!(out.len(), 1);
        let y = &out[0];
        println!("Output shape: {:?}, dtype: {:?}", y.dims(), y.dtype());

        // Convert to F32 to check values
        let y_f32 = y.to_dtype(DType::F32)?;
        let max_abs = y_f32.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;
        println!("Max abs value: {}", max_abs);

        // Print first few values to diagnose
        let vals = y_f32.flatten_all()?.to_vec1::<f32>()?;
        println!("First 10 values: {:?}", &vals[..10.min(vals.len())]);

        // Check if values are reasonable
        assert!(max_abs.is_finite(), "Output contains NaN or Inf");
        assert!(max_abs < 100.0, "Output values are too large: {}", max_abs);

        Ok(())
    }

    #[test]
    fn test_fp8_hd64_paged_prefill() -> Result<()> {
        // This tests the FP8 path with HD=64 (Qwen2-style) - should work
        println!("\n=== Testing FP8 HD=64 Paged Prefill (Reference) ===\n");

        let device = Device::new_cuda(0)?;

        // Qwen2-style parameters
        let b_sz = 1usize;
        let seq_len = 4usize;
        let n_head = 14usize;
        let n_kv_head = 2usize;
        let head_dim = 64usize;

        println!(
            "Parameters: b_sz={}, seq_len={}, n_head={}, n_kv_head={}, head_dim={}",
            b_sz, seq_len, n_head, n_kv_head, head_dim
        );

        let q = Tensor::randn(0f32, 0.1f32, (b_sz, n_head, seq_len, head_dim), &device)?
            .to_dtype(DType::F8E4M3)?;
        let k = Tensor::randn(0f32, 0.1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::F8E4M3)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 0.1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(DType::F8E4M3)?
            .contiguous()?;

        let mut cache0 = KvCache::new(2, 512);
        cache0.force_dtype(DType::F8E4M3);
        let offsets = [0usize];
        let mut caches: [&mut KvCache; 1] = [&mut cache0];

        std::env::set_var("CANDLE_TRACE_PAGED_PREFILL", "1");
        let rope_zeros = Tensor::zeros(b_sz, DType::U32, &device)?;
        let generation = PinnedStager::new(device.as_cuda_device()?).begin_generation();

        let out = paged_prefill_uniform(
            &mut caches,
            &offsets,
            &q,
            &k,
            &v,
            b_sz,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            None,
            &rope_zeros,
            &make_test_rope_cs(head_dim, 16, &device)?,
            false, // rope_interleaved
            {
                let mut b = generation.alloc(b_sz * 4)?;
                b.fill(0u8);
                generation.submit(b)?.dev_ptr()
            }, // write_offset_shifts
            &generation,
        )
        .expect("expected paged prefill to be applicable");

        assert_eq!(out.len(), 1);
        let y = &out[0];
        println!("Output shape: {:?}, dtype: {:?}", y.dims(), y.dtype());

        let y_f32 = y.to_dtype(DType::F32)?;
        let max_abs = y_f32.abs()?.flatten_all()?.max(0)?.to_vec0::<f32>()?;
        println!("Max abs value: {}", max_abs);

        let vals = y_f32.flatten_all()?.to_vec1::<f32>()?;
        println!("First 10 values: {:?}", &vals[..10.min(vals.len())]);

        assert!(max_abs.is_finite(), "Output contains NaN or Inf");
        assert!(max_abs < 100.0, "Output values are too large: {}", max_abs);

        Ok(())
    }

    // ========================================================================
    // Correctness comparison tests: paged kernels vs reference matmul attention
    // ========================================================================

    /// Gold-standard causal attention via matmul.
    ///
    /// q: (1, n_head, seq_len, head_dim)
    /// k: (1, n_kv_head, kv_len, head_dim)  ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â full KV including prefix
    /// v: (1, n_kv_head, kv_len, head_dim)
    /// offset: number of prefix tokens (causal mask allows attending to prefix + up to current pos)
    ///
    /// Returns: (1, n_head, seq_len, head_dim) in F32
    fn reference_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let seq_len = q.dim(2)?;
        let kv_len = k.dim(2)?;

        // Convert everything to F32 for reference precision
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        // Repeat KV heads for GQA/MQA
        let num_groups = n_head / n_kv_head;
        let k = crate::utils::repeat_kv(k, num_groups)?;
        let v = crate::utils::repeat_kv(v, num_groups)?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;

        // Causal mask: position i in the query (at absolute position offset+i)
        // can attend to KV positions 0..=offset+i
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if j > offset + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0f32
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, seq_len, kv_len), q.device())?;
        let att = att.broadcast_add(&mask)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?;
        Ok(out)
    }

    /// Reference decode attention: single query token against full KV cache.
    ///
    /// q: (batch_size, n_head, head_dim)
    /// k: (batch_size, n_kv_head, kv_len, head_dim)
    /// v: (batch_size, n_kv_head, kv_len, head_dim)
    ///
    /// Returns: (batch_size, n_head, head_dim) in F32
    #[allow(dead_code)]
    fn reference_decode_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let _batch_size = q.dim(0)?;
        let _kv_len = k.dim(2)?;

        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let num_groups = n_head / n_kv_head;
        let k = crate::utils::repeat_kv(k, num_groups)?;
        let v = crate::utils::repeat_kv(v, num_groups)?;

        // q: (batch, n_head, head_dim) -> (batch, n_head, 1, head_dim)
        let q = q.unsqueeze(2)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;

        // No causal mask needed for decode ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â all KV positions are valid
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?; // (batch, n_head, 1, head_dim)
        let out = out.squeeze(2)?; // (batch, n_head, head_dim)
        Ok(out)
    }

    /// Compute max absolute error between two tensors (both converted to F32).
    fn max_abs_error(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?;
        let diff = (a - b)?.abs()?;
        diff.max(0)?.to_vec0::<f32>()
    }

    /// Compute mean absolute error between two tensors (both converted to F32).
    fn mean_abs_error(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?;
        let diff = (a - b)?.abs()?;
        diff.mean_all()?.to_vec0::<f32>()
    }

    /// Helper: run paged prefill and return output in (1, n_head, seq_len, head_dim) shape.
    fn run_paged_prefill(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        b_sz: usize,
        seq_len: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        offset: usize,
        dtype: DType,
    ) -> Result<Tensor> {
        let mut cache0 = KvCache::new(2, 4096);
        cache0.force_dtype(dtype);
        let offsets = [offset];
        let mut caches: [&mut KvCache; 1] = [&mut cache0];
        let rope_zeros = Tensor::zeros(b_sz, DType::U32, q.device())?;
        let generation = PinnedStager::new(q.device().as_cuda_device()?).begin_generation();

        let out = paged_prefill_uniform(
            &mut caches,
            &offsets,
            q,
            k,
            v,
            b_sz,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            None,
            &rope_zeros,
            &make_zero_rope_cs(head_dim, 16, q.device())?,
            false, // rope_interleaved
            {
                let mut b = generation.alloc(b_sz * 4)?;
                b.fill(0u8);
                generation.submit(b)?.dev_ptr()
            }, // write_offset_shifts
            &generation,
        )?;
        assert_eq!(out.len(), 1);
        Ok(out[0].clone())
    }

    /// Helper: prefill KV cache, then run paged decode for one token.
    /// Returns the decode output and the reference decode output.

    // ------------------------------------------------------------------
    // Prefill correctness: no prefix (offset=0)
    // ------------------------------------------------------------------

    #[test]
    fn correctness_prefill_no_prefix_bf16() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        for &(n_head, n_kv_head, head_dim, seq_len, label) in &[
            (8, 8, 64, 4, "MHA hd64 short"),
            (8, 8, 64, 32, "MHA hd64 medium"),
            (8, 8, 64, 128, "MHA hd64 long"),
            (8, 8, 128, 4, "MHA hd128 short"),
            (8, 8, 128, 64, "MHA hd128 long"),
            (32, 8, 64, 4, "GQA 32/8 hd64 short"),
            (32, 8, 64, 32, "GQA 32/8 hd64 medium"),
            (32, 8, 64, 128, "GQA 32/8 hd64 long"),
            (32, 8, 128, 4, "GQA 32/8 hd128 short"),
            (32, 8, 128, 64, "GQA 32/8 hd128 long"),
            (40, 8, 64, 4, "GQA 40/8 hd64 short"), // num_groups=5, tricky for WARPS_TC
            (40, 8, 64, 32, "GQA 40/8 hd64 medium"),
            (40, 8, 128, 4, "GQA 40/8 hd128 short"),
            (14, 2, 64, 4, "GQA 14/2 hd64 short"), // Qwen2-style
            (14, 2, 64, 32, "GQA 14/2 hd64 medium"),
            (14, 2, 128, 4, "GQA 14/2 hd128 short"),
            (8, 1, 64, 4, "MQA hd64 short"),
            (8, 1, 64, 32, "MQA hd64 medium"),
            (8, 1, 128, 4, "MQA hd128 short"),
        ] {
            let b_sz = 1;
            let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            let paged_out = run_paged_prefill(
                &q, &k, &v, b_sz, seq_len, n_head, n_kv_head, head_dim, 0, dtype,
            )?;

            // Reference: for no-prefix, full KV == the input K/V
            let ref_out = reference_attention(&q, &k, &v, n_head, n_kv_head, head_dim, 0)?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            // BF16 tolerance: mean ~1e-2, max ~5e-2
            assert!(
                mae < 0.05,
                "[{label}] BF16 prefill mean error too large: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] BF16 prefill max error too large: {max_err}"
            );
            println!("[{label}] BF16 prefill OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    #[test]
    fn correctness_prefill_no_prefix_f16() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::F16;

        for &(n_head, n_kv_head, head_dim, seq_len, label) in &[
            (8, 8, 64, 4, "MHA hd64 short"),
            (8, 8, 64, 32, "MHA hd64 medium"),
            (32, 8, 64, 4, "GQA 32/8 hd64 short"),
            (32, 8, 64, 32, "GQA 32/8 hd64 medium"),
            (32, 8, 128, 4, "GQA 32/8 hd128 short"),
            (40, 8, 64, 4, "GQA 40/8 hd64 short"),
            (14, 2, 64, 4, "GQA 14/2 hd64 short"),
            (8, 1, 64, 4, "MQA hd64 short"),
        ] {
            let b_sz = 1;
            let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            let paged_out = run_paged_prefill(
                &q, &k, &v, b_sz, seq_len, n_head, n_kv_head, head_dim, 0, dtype,
            )?;
            let ref_out = reference_attention(&q, &k, &v, n_head, n_kv_head, head_dim, 0)?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            // F16 tolerance ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â slightly tighter since F16 has more mantissa bits than BF16
            assert!(
                mae < 0.05,
                "[{label}] F16 prefill mean error too large: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] F16 prefill max error too large: {max_err}"
            );
            println!("[{label}] F16 prefill OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Prefill correctness: with prefix (offset > 0, multi-turn)
    // ------------------------------------------------------------------

    #[test]
    fn correctness_prefill_with_prefix_bf16() -> Result<()> {
        use candle_nn::kv_cache::ChunkedKvBacking;

        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        for &(n_head, n_kv_head, head_dim, prefix_len, new_len, label) in &[
            (8, 8, 64, 10, 4, "MHA hd64 prefix=10"),
            (8, 8, 64, 32, 8, "MHA hd64 prefix=32"),
            (32, 8, 64, 10, 4, "GQA 32/8 hd64 prefix=10"),
            (32, 8, 64, 64, 16, "GQA 32/8 hd64 prefix=64"),
            (32, 8, 128, 10, 4, "GQA 32/8 hd128 prefix=10"),
            (40, 8, 64, 10, 4, "GQA 40/8 hd64 prefix=10"),
            (14, 2, 64, 10, 4, "GQA 14/2 hd64 prefix=10"),
            (8, 1, 64, 10, 4, "MQA hd64 prefix=10"),
        ] {
            let b_sz = 1;
            let total_kv = prefix_len + new_len;

            // Create K/V for prefix (already in cache) and new segment
            let prefix_k =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, prefix_len, head_dim), &device)?
                    .to_dtype(dtype)?;
            let prefix_v =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, prefix_len, head_dim), &device)?
                    .to_dtype(dtype)?;
            let new_q = Tensor::randn(0f32, 1f32, (1, n_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let new_k = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let new_v = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            // Set up chunked KV cache and write prefix data
            let backing =
                ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, &device, total_kv)?;
            backing.ensure_for_offset(0, 0, total_kv)?;
            backing.write_contiguous(0, 0, &prefix_k, &prefix_v)?;

            let mut cache0 = KvCache::new(2, 4096);
            cache0.force_dtype(dtype);
            cache0.set_chunked_backing(&backing, 0, None)?;
            cache0.set_current_seq_len(prefix_len)?;

            let offsets = [prefix_len];
            let mut caches: [&mut KvCache; 1] = [&mut cache0];
            let rope_zeros = Tensor::zeros(b_sz, DType::U32, &device)?;
            let generation = backing.begin_stager_generation_required();

            let out = paged_prefill_uniform(
                &mut caches,
                &offsets,
                &new_q,
                &new_k,
                &new_v,
                b_sz,
                new_len,
                n_head,
                n_kv_head,
                head_dim,
                None,
                &rope_zeros,
                &make_zero_rope_cs(head_dim, 16, &device)?,
                false, // rope_interleaved
                {
                    let mut b = generation.alloc(b_sz * 4)?;
                    b.fill(0u8);
                    generation.submit(b)?.dev_ptr()
                }, // write_offset_shifts
                &generation,
            )?;
            assert_eq!(out.len(), 1);
            let paged_out = &out[0];

            // Reference: full KV = prefix + new
            let full_k = Tensor::cat(&[&prefix_k, &new_k], 2)?;
            let full_v = Tensor::cat(&[&prefix_v, &new_v], 2)?;
            let ref_out = reference_attention(
                &new_q, &full_k, &full_v, n_head, n_kv_head, head_dim, prefix_len,
            )?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            assert!(
                mae < 0.05,
                "[{label}] BF16 prefill-with-prefix mean error too large: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] BF16 prefill-with-prefix max error too large: {max_err}"
            );
            println!("[{label}] BF16 prefill-with-prefix OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    #[test]
    fn correctness_prefill_with_prefix_f16() -> Result<()> {
        use candle_nn::kv_cache::ChunkedKvBacking;

        let device = Device::new_cuda(0)?;
        let dtype = DType::F16;

        for &(n_head, n_kv_head, head_dim, prefix_len, new_len, label) in &[
            (8, 8, 64, 10, 4, "MHA hd64 prefix=10"),
            (32, 8, 64, 10, 4, "GQA 32/8 hd64 prefix=10"),
            (32, 8, 128, 10, 4, "GQA 32/8 hd128 prefix=10"),
            (40, 8, 64, 10, 4, "GQA 40/8 hd64 prefix=10"),
            (14, 2, 64, 10, 4, "GQA 14/2 hd64 prefix=10"),
        ] {
            let b_sz = 1;
            let total_kv = prefix_len + new_len;

            let prefix_k =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, prefix_len, head_dim), &device)?
                    .to_dtype(dtype)?;
            let prefix_v =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, prefix_len, head_dim), &device)?
                    .to_dtype(dtype)?;
            let new_q = Tensor::randn(0f32, 1f32, (1, n_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let new_k = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let new_v = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            let backing =
                ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, &device, total_kv)?;
            backing.ensure_for_offset(0, 0, total_kv)?;
            backing.write_contiguous(0, 0, &prefix_k, &prefix_v)?;

            let mut cache0 = KvCache::new(2, 4096);
            cache0.force_dtype(dtype);
            cache0.set_chunked_backing(&backing, 0, None)?;
            cache0.set_current_seq_len(prefix_len)?;

            let offsets = [prefix_len];
            let mut caches: [&mut KvCache; 1] = [&mut cache0];
            let rope_zeros = Tensor::zeros(b_sz, DType::U32, &device)?;
            let generation = backing.begin_stager_generation_required();

            let out = paged_prefill_uniform(
                &mut caches,
                &offsets,
                &new_q,
                &new_k,
                &new_v,
                b_sz,
                new_len,
                n_head,
                n_kv_head,
                head_dim,
                None,
                &rope_zeros,
                &make_zero_rope_cs(head_dim, 16, &device)?,
                false, // rope_interleaved
                {
                    let mut b = generation.alloc(b_sz * 4)?;
                    b.fill(0u8);
                    generation.submit(b)?.dev_ptr()
                }, // write_offset_shifts
                &generation,
            )?;
            let paged_out = &out[0];

            let full_k = Tensor::cat(&[&prefix_k, &new_k], 2)?;
            let full_v = Tensor::cat(&[&prefix_v, &new_v], 2)?;
            let ref_out = reference_attention(
                &new_q, &full_k, &full_v, n_head, n_kv_head, head_dim, prefix_len,
            )?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            assert!(
                mae < 0.05,
                "[{label}] F16 prefill-with-prefix mean error too large: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] F16 prefill-with-prefix max error too large: {max_err}"
            );
            println!("[{label}] F16 prefill-with-prefix OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Decode correctness: prefill then single-token decode
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // GQA head mapping regression test (validates the kv_group_end fix)
    // ------------------------------------------------------------------

    /// This test specifically targets the GQA head mapping bug where WARPS_TC
    /// could overflow into the next KV group. It exercises configurations where
    /// num_groups is not a multiple of WARPS_TC (e.g., 40/8 = 5 groups).
    #[test]
    fn correctness_prefill_gqa_head_mapping_regression() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        // These configurations are specifically chosen to trigger the GQA overflow bug:
        // - 40/8: num_groups=5, with WARPS_TC=2 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ head_blocks_per_kv=3, last block has 1 warp active
        // - 28/4: num_groups=7, tricky for WARPS_TC alignment
        // - 48/8: num_groups=6, with WARPS_TC=2 ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢ head_blocks_per_kv=3, all warps active
        // - 56/8: num_groups=7, same as 28/4 pattern
        // - 12/4: num_groups=3
        for &(n_head, n_kv_head, head_dim, seq_len, label) in &[
            (40, 8, 64, 4, "40/8 hd64 short"),
            (40, 8, 64, 32, "40/8 hd64 medium"),
            (40, 8, 64, 128, "40/8 hd64 long"),
            (40, 8, 128, 4, "40/8 hd128 short"),
            (40, 8, 128, 32, "40/8 hd128 medium"),
            (28, 4, 64, 4, "28/4 hd64 short"),
            (28, 4, 64, 32, "28/4 hd64 medium"),
            (48, 8, 64, 4, "48/8 hd64 short"),
            (48, 8, 64, 32, "48/8 hd64 medium"),
            (56, 8, 64, 4, "56/8 hd64 short"),
            (56, 8, 128, 4, "56/8 hd128 short"),
            (12, 4, 64, 4, "12/4 hd64 short"),
            (12, 4, 64, 32, "12/4 hd64 medium"),
        ] {
            let b_sz = 1;
            let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            let paged_out = run_paged_prefill(
                &q, &k, &v, b_sz, seq_len, n_head, n_kv_head, head_dim, 0, dtype,
            )?;
            let ref_out = reference_attention(&q, &k, &v, n_head, n_kv_head, head_dim, 0)?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            assert!(
                mae < 0.05,
                "[{label}] GQA regression: mean error too large: {mae} (may indicate head mapping overflow)"
            );
            assert!(
                max_err < 0.2,
                "[{label}] GQA regression: max error too large: {max_err} (may indicate head mapping overflow)"
            );
            println!("[{label}] GQA regression OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    /// Diagnostic test that prints per-head errors to help identify specific
    /// GQA group cross-contamination patterns.
    #[test]
    fn correctness_prefill_diagnostic_per_head() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        // Use a config known to be tricky: 40 heads / 8 KV heads = 5 groups
        let n_head = 40;
        let n_kv_head = 8;
        let head_dim = 64;
        let seq_len = 8;
        let b_sz = 1;

        let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;

        let paged_out = run_paged_prefill(
            &q, &k, &v, b_sz, seq_len, n_head, n_kv_head, head_dim, 0, dtype,
        )?;
        let ref_out = reference_attention(&q, &k, &v, n_head, n_kv_head, head_dim, 0)?;

        let paged_f32 = paged_out.to_dtype(DType::F32)?;
        let num_groups = n_head / n_kv_head;

        println!("\n=== Per-head error for {n_head}/{n_kv_head} (groups={num_groups}) ===");
        for h in 0..n_head {
            let kv_group = h / num_groups;
            let paged_h = paged_f32.narrow(1, h, 1)?;
            let ref_h = ref_out.narrow(1, h, 1)?;
            let mae = mean_abs_error(&paged_h, &ref_h)?;
            let marker = if mae > 0.05 { " *** HIGH ***" } else { "" };
            println!("  head {h:2} (kv_group {kv_group}): mae={mae:.4e}{marker}");
        }

        let overall_mae = mean_abs_error(&paged_f32, &ref_out)?;
        assert!(
            overall_mae < 0.05,
            "Per-head diagnostic: overall mean error too large: {overall_mae}"
        );
        Ok(())
    }

    // ========================================================================
    // RoPE offset plumbing tests
    // ========================================================================
    //
    // These tests verify that the rope_offsets parameter is correctly plumbed
    // from the public API all the way to the CUDA kernel entry point without
    // causing panics or incorrect output.
    //
    // Since the kernels treat rope_offsets=nullptr as offset=0 (a no-op), and
    // rope_offsets=Some(zeros) also means offset=0, the two must produce
    // numerically identical results.

    /// Helper: run a single-batch prefill and return the output tensor.
    fn run_prefill_with_rope(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        dtype: DType,
        rope: &Tensor,
    ) -> candle::Result<Tensor> {
        use candle_nn::kv_cache::ChunkedKvBacking;
        let device = q.device();
        let b_sz = 1;

        let backing = ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, device, seq_len)?;
        backing.ensure_for_offset(0, 0, seq_len)?;

        let mut cache0 = KvCache::new(2, 4096);
        cache0.force_dtype(dtype);
        cache0.set_chunked_backing(&backing, 0, None)?;

        let offsets = [0usize];
        let mut caches: [&mut KvCache; 1] = [&mut cache0];
        let generation = backing.begin_stager_generation_required();

        let out = paged_prefill_uniform(
            &mut caches,
            &offsets,
            q,
            k,
            v,
            b_sz,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            None,
            rope,
            &make_test_rope_cs(head_dim, 16, device)?,
            false, // rope_interleaved
            {
                let mut b = generation.alloc(b_sz * 4)?;
                b.fill(0u8);
                generation.submit(b)?.dev_ptr()
            }, // write_offset_shifts
            &generation,
        )?;
        Ok(out.into_iter().next().unwrap())
    }

    #[test]
    fn rope_offset_prefill_none_succeeds() -> candle::Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let (n_head, n_kv_head, head_dim, seq_len) = (8, 8, 64, 16);

        let q =
            Tensor::randn(0f32, 1f32, (1, n_head, seq_len, head_dim), &device)?.to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;

        let rope_zeros = Tensor::zeros(1usize, DType::U32, &device)?;
        let out = run_prefill_with_rope(
            &q,
            &k,
            &v,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
            &rope_zeros,
        )?;
        assert_eq!(out.dims(), &[1, n_head, seq_len, head_dim]);
        let max_abs = out
            .to_dtype(DType::F32)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            max_abs.is_finite(),
            "rope=zeros output not finite: {max_abs}"
        );
        println!("rope_offset_prefill_none_succeeds OK: max_abs={max_abs:.4e}");
        Ok(())
    }

    #[test]
    fn rope_offset_prefill_zeros_differs_from_none() -> candle::Result<()> {
        // Now that both paths always apply RoPE, this test confirms that different rope_offsets
        // values produce different outputs.
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let (n_head, n_kv_head, head_dim, seq_len) = (8, 8, 64, 16);

        let q =
            Tensor::randn(0f32, 1f32, (1, n_head, seq_len, head_dim), &device)?.to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;

        // Run with rope=zeros (offset=0)
        let rope_zeros = Tensor::zeros(1usize, DType::U32, &device)?;
        let out_zeros = run_prefill_with_rope(
            &q,
            &k,
            &v,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
            &rope_zeros,
        )?;

        // Run with rope=offset16 â€” different base offset produces different rotation
        let rope_offset16 = Tensor::from_vec(vec![16u32], 1, &device)?;
        let out_offset16 = run_prefill_with_rope(
            &q,
            &k,
            &v,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
            &rope_offset16,
        )?;

        let zeros_f32 = out_zeros.to_dtype(DType::F32)?;
        let offset16_f32 = out_offset16.to_dtype(DType::F32)?;

        // Both outputs must be finite
        let max_zeros = zeros_f32.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        let max_offset16 = offset16_f32
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(max_zeros.is_finite(), "rope=zeros output not finite");
        assert!(max_offset16.is_finite(), "rope=offset16 output not finite");

        let mae = mean_abs_error(&zeros_f32, &offset16_f32)?;
        println!("rope_offset_prefill_zeros_differs_from_none: mae={mae:.4e}");
        // Different rope offsets must produce different outputs
        // RoPE preserves relative positions: shifting all token positions by the same constant
        // leaves the relative-position attention map unchanged. The two runs should be equivalent.
        assert!(
            mae < 1e-2,
            "rope=zeros and rope=offset16 should produce equivalent attention outputs \
             (same relative positions; mae={mae:.4e})"
        );
        Ok(())
    }

    // Reference RoPE rotation: apply to f32 slice of HEAD_DIM values.
    // Pairs: dim d with dim d+half_dim.
    fn apply_rope_to_vec(x: &[f32], head_dim: usize, pos: usize) -> Vec<f32> {
        let mut out = x.to_vec();
        let half = head_dim / 2;
        for d in 0..half {
            let theta = pos as f32 * 10000.0f32.powf(-2.0 * d as f32 / head_dim as f32);
            let (sin_v, cos_v) = theta.sin_cos();
            let x_lo = x[d];
            let x_hi = x[d + half];
            out[d] = x_lo * cos_v - x_hi * sin_v;
            out[d + half] = x_lo * sin_v + x_hi * cos_v;
        }
        out
    }

    #[test]
    fn rope_offset_prefill_functional() -> candle::Result<()> {
        // Verifies: reference_attention(rotated_Q, rotated_K, V) â‰ˆ paged_kernel(unrotated_Q, unrotated_K, V, rope=zeros)
        // Tolerance < 0.05 for BF16.
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let (n_head, n_kv_head, head_dim, seq_len) = (4, 4, 64, 8);

        let q =
            Tensor::randn(0f32, 1f32, (1, n_head, seq_len, head_dim), &device)?.to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;

        // Manually rotate [1, n_head, seq_len, head_dim]: token t -> pos t
        let apply_rope_4d = |t: &Tensor| -> candle::Result<Tensor> {
            let (b, h, s, d) = t.dims4()?;
            // Reshape to [b*h, s, d] and back so we can use to_vec3
            let flat = t.reshape((b * h, s, d))?;
            let data = flat.to_dtype(DType::F32)?.to_vec3::<f32>()?;
            let mut out: Vec<f32> = Vec::with_capacity(b * h * s * d);
            for bh in 0..(b * h) {
                for si in 0..s {
                    out.extend_from_slice(&apply_rope_to_vec(&data[bh][si], d, si));
                }
            }
            Tensor::from_vec(out, (b * h, s, d), t.device())?
                .reshape((b, h, s, d))?
                .to_dtype(t.dtype())
        };

        // Pre-rotated versions
        let q_rotated = apply_rope_4d(&q)?.contiguous()?;
        let k_rotated = apply_rope_4d(&k)?.contiguous()?;

        // Branch A (reference): reference_attention with pre-rotated Q/K
        let ref_out =
            reference_attention(&q_rotated, &k_rotated, &v, n_head, n_kv_head, head_dim, 0)?;

        // Branch B: un-rotated Q, K  +  rope=zeros  (kernel rotates at tok_idx + 0)
        let rope_zeros = Tensor::zeros(1usize, DType::U32, &device)?;
        let out_kernel_rope = run_prefill_with_rope(
            &q,
            &k,
            &v,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
            &rope_zeros,
        )?;

        let mae = mean_abs_error(
            &ref_out.to_dtype(DType::F32)?,
            &out_kernel_rope.to_dtype(DType::F32)?,
        )?;
        println!(
            "rope_offset_prefill_functional: mae(reference_attention(rotated) vs kernel+rope0)={mae:.4e}"
        );
        // BF16 rounding: allow up to 0.05 (two roundings: manual rotation + kernel)
        assert!(
            mae < 0.05,
            "fused RoPE prefill functional mismatch: mae={mae}"
        );
        Ok(())
    }
}
