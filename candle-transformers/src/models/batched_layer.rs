//! Batched transformer layer processing for continuous batching.
//!
//! This module provides traits and utilities for processing transformer layers
//! in batched mode, enabling efficient continuous batching across multiple sequences.
//!
//! The key abstraction is [`BatchedAttentionLayer`] which defines the interface
//! that transformer layers must implement to support batched attention processing.
//! The actual batched attention computation is implemented generically in this module.

#[cfg(feature = "cuda")]
use candle::quantized::cuda::{DynamicActs, Q8a128Operand};
use candle::quantized::pinned_staging::{Generation, GpuBuf};
use candle::quantized::Int8Mode;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::KvCache;

#[cfg(feature = "cuda")]
use crate::models::prefill_utils::paged_decode_attn;
#[cfg(feature = "cuda")]
use crate::models::prefill_utils::paged_decode_attn_q8;
use crate::models::prefill_utils::{paged_glue_attn, paged_prefill_batched, SharedPm};
use crate::models::profile::{pipeline_record, profile_now, profile_sync};
use crate::models::quantized_matmul::QMatMul;
use crate::utils::repeat_kv;

#[cfg(feature = "cuda")]
use candle_kernels::CHUNK_SIZE;

use super::tensor_cat::TensorCat;

// ============================================================================
// Batched Attention Parameters
// ============================================================================

/// Decode attention headers for a single forward pass.
///
/// Passed to `forward_batch` to indicate whether this is a prefill or decode pass,
/// and carries the mode-specific metadata for each path.
pub enum DecodeHeaders {
    /// Prefill pass (multi-token input): carries the paged prefill metadata.
    Prefill(BatchedPrefillMeta),
    /// Decode pass (single-token input): headers for all layers packed contiguously
    /// in a single GPU buffer with constant byte stride between layers.
    /// `buf` is `None` on non-CUDA builds or when paged decode is unavailable;
    /// the kernel falls back to standard attention.
    Decode {
        /// Single GPU buffer containing all layers' slot headers
        /// (`SlotHeader[n_active]` per layer, packed contiguously).
        /// Layer `i` starts at byte offset `i * stride`.
        buf: Option<GpuBuf>,
        /// Byte stride between successive layers (`n_active * 16`).
        stride: u64,
    },
}

/// Parameters for batched attention computation.
///
/// These are precomputed values passed to batched attention to avoid redundant
/// computation across layers. RoPE tables are always precomputed at the model level
/// and passed down to layers.
pub struct BatchedAttentionParams<'a> {
    /// Precomputed RoPE (cos, sin) for the current batch positions.
    /// Always provided - computed once at model level.
    pub rope_cos: &'a Tensor,
    /// Precomputed RoPE sin for the current batch positions.
    pub rope_sin: &'a Tensor,
    /// Whether RoPE uses interleaved format.
    pub rope_interleaved: bool,
    /// Per-dimension inverse frequencies for the CUDA paged-attention kernels.
    /// Shape: [head_dim/2], dtype F32, stored on the model device.
    pub inv_freq_device: &'a Tensor,
    /// Precomputed cos/sin table for decode RoPE, shape [max_pos, head_dim], dtype F32.
    /// Computed once from inv_freq at model level and cached.
    pub rope_cs: &'a Tensor,
    /// Whether this is a prefill or decode pass, and — for decode — the GPU buffer
    /// of per-layer slot headers packed contiguously with constant byte stride.
    pub decode_headers: DecodeHeaders,
    /// Per-sequence new-token (query) lengths for the batch. For prefill these are
    /// the ragged per-sequence chunk lengths (Σ = the flat-packed token count);
    /// for decode they are all 1. Host-side; drives the varlen prefill plumbing.
    pub q_lens: &'a [usize],
    /// Pinned-stager generation guard for quantization kernel metadata allocations.
    /// Threaded from forward_batched through to reconcile → quantize_palette4_convert_buffered.
    pub generation: &'a Generation,
    /// Per-forward cache of the layer-invariant prefill `position_map`. The first
    /// layer of a prefill forward populates it; later layers reuse the uploaded
    /// buffer instead of rebuilding + re-uploading it. Empty (`None`) at forward
    /// start; unused on the decode and CPU paths.
    pub shared_prefill_pm: &'a std::cell::RefCell<Option<SharedPm>>,
}

impl<'a> BatchedAttentionParams<'a> {
    /// Create params with all fields populated.
    ///
    /// RoPE (cos, sin) must always be provided - they are computed once at the model level.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cos: &'a Tensor,
        sin: &'a Tensor,
        rope_interleaved: bool,
        inv_freq_device: &'a Tensor,
        rope_cs: &'a Tensor,
        decode_headers: DecodeHeaders,
        q_lens: &'a [usize],
        generation: &'a Generation,
        shared_prefill_pm: &'a std::cell::RefCell<Option<SharedPm>>,
    ) -> Self {
        Self {
            rope_cos: cos,
            rope_sin: sin,
            rope_interleaved,
            inv_freq_device,
            rope_cs,
            decode_headers,
            q_lens,
            generation,
            shared_prefill_pm,
        }
    }
}

/// Reprojection-glue metadata. Present only when the multi-token forward is a
/// gap-fill (glue) pass: it routes the layer's attention to the paged-glue
/// kernel instead of plain prefill. The glue tokens are reserved IN PLACE as gap
/// chunks, so there is no `col_actual_pos` — every column's position comes from
/// its chunk `rope_base` (`slice_rope`), the convention decode also reads. These
/// flat `[Σ q_lens]` U32 tensors carry only what the kernel can't derive from
/// the slot: per glue token, where it scatters and how far it bridges forward.
#[derive(Clone)]
pub struct GlueMeta {
    /// Gap chunk block index each glue token's K/V scatters into.
    pub glue_write_slice: Tensor,
    /// In-block offset within the gap chunk.
    pub glue_write_in_blk: Tensor,
    /// Forward bridge window in tokens (`0` == backward-only).
    pub fwd_ahead: Tensor,
}

/// Precomputed metadata for paged prefill attention.
///
/// This avoids rebuilding the same small tensors (cu_seqlens, q_lens, kv_lens)
/// for every layer during multi-token prefill.
#[derive(Clone)]
pub struct BatchedPrefillMeta {
    pub cu_seqlens_q: Tensor,
    pub q_lens: Tensor,
    pub kv_lens: Tensor,
    /// Set when this prefill is a reprojection-glue forward (HD128 routes to the
    /// paged-glue kernel). `None` for an ordinary prefill.
    pub glue: Option<GlueMeta>,
}

impl BatchedPrefillMeta {
    /// Build paged-prefill metadata for a uniform batch — every sequence
    /// prefills the same `seq_len` new tokens. Thin wrapper over
    /// [`Self::new_ragged`] (uniform is the special case where all query
    /// lengths are equal), kept for the decode/uniform-prefill call sites.
    pub fn new(offsets: &[usize], seq_len: usize, device: &Device) -> Result<Self> {
        let q_lens = vec![seq_len; offsets.len()];
        Self::new_ragged(offsets, &q_lens, device)
    }

    /// Build paged-prefill metadata for a **ragged** batch — each sequence has
    /// its own query length `q_lens[i]`. This is the flash-attention varlen
    /// format: `cu_seqlens_q` is the exclusive prefix sum of `q_lens` (so the
    /// packed Q layout is `[Σ q_lens, n_head, head_dim]`), `q_lens` are the
    /// per-sequence new-token counts, and `kv_lens[i] = offsets[i] + q_lens[i]`
    /// is each sequence's total context length after this prefill.
    pub fn new_ragged(offsets: &[usize], q_lens: &[usize], device: &Device) -> Result<Self> {
        if offsets.len() != q_lens.len() {
            candle::bail!(
                "BatchedPrefillMeta::new_ragged: {} offsets vs {} q_lens",
                offsets.len(),
                q_lens.len()
            );
        }
        let batch_size = offsets.len();
        let mut cu = Vec::with_capacity(batch_size + 1);
        cu.push(0u32);
        let mut acc = 0u32;
        for &ql in q_lens {
            acc += ql as u32;
            cu.push(acc);
        }
        let cu_seqlens_q = Tensor::from_vec(cu, batch_size + 1, device)?;
        let q_lens_t = Tensor::from_vec(
            q_lens.iter().map(|&l| l as u32).collect::<Vec<_>>(),
            batch_size,
            device,
        )?;
        let kv_lens = Tensor::from_vec(
            offsets
                .iter()
                .zip(q_lens.iter())
                .map(|(&o, &l)| (o + l) as u32)
                .collect::<Vec<_>>(),
            batch_size,
            device,
        )?;
        Ok(Self {
            cu_seqlens_q,
            q_lens: q_lens_t,
            kv_lens,
            glue: None,
        })
    }
}

/// Precomputed decode metadata for the paged-decode attention path.
///
// ============================================================================
// Layer-Level Trait
// ============================================================================

/// Q/K/V tensors after projection but before RoPE.
///
/// Shape: (batch, seq_len, hidden_dim) for each tensor.
pub struct QkvProjection {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

/// Trait for transformer layers that support batched attention processing.
///
/// Implement this trait to enable batched forward passes through transformer layers.
/// The trait exposes the primitives needed for attention computation, and the
/// generic batched attention logic is implemented in this module.
///
/// Note: RoPE is applied generically using precomputed (cos, sin) tables passed
/// in [`BatchedAttentionParams`]. Layers do not need to implement RoPE logic.
pub trait BatchedAttentionLayer {
    /// Number of attention heads.
    fn n_head(&self) -> usize;

    /// Number of KV heads (for GQA/MQA).
    fn n_kv_head(&self) -> usize;

    /// Dimension of each attention head.
    fn head_dim(&self) -> usize;

    /// Attention layer norm (ln1) as a producer epilogue (B1): returns the matmul-ready
    /// `DynamicActs` — q8a128 for an int8 `mode` (fused RMSNorm→quant in one kernel), `Float` for
    /// `Off`. Every layer fuses its own ln1.
    fn attention_norm(&self, x: &Tensor, mode: Int8Mode) -> Result<DynamicActs>;

    /// FFN layer norm (ln2) as a producer epilogue (B3): q8a128 for an int8 `mode`, `Float` for
    /// `Off`.
    fn ffn_norm(&self, x: &Tensor, mode: Int8Mode) -> Result<DynamicActs>;

    /// FFN/MoE module consuming the producer-prepared `DynamicActs` (the fused `ffn_norm`): the
    /// router + expert gather take the q8a128 directly (no standalone quantize). `mlp_dtype` is the
    /// FP-stable accumulation dtype for the `Float` path.
    fn ffn_forward(&self, acts: DynamicActs, mlp_dtype: DType) -> Result<Tensor>;

    /// Project Q/K/V over the producer-prepared `DynamicActs` (the fused `attention_norm`), folding
    /// in any Q/K/V bias and q/k-norm. q/k/v share the single quantize (B1).
    ///
    /// # Returns
    /// QkvProjection containing Q, K, V tensors.
    /// - Q shape: (batch, seq_len, n_head * head_dim)
    /// - K shape: (batch, seq_len, n_kv_head * head_dim)
    /// - V shape: (batch, seq_len, n_kv_head * head_dim)
    fn project_qkv(&self, acts: &DynamicActs, out_dtype: DType) -> Result<QkvProjection>;

    /// The output-projection weight (`attention_wo` / `self_attn.o_proj`). Backs the generalized
    /// `int8mode` + `output_projection` defaults so any KO-loaded model gets B2 + the mode for free
    /// — the weight already carries its numeric mode.
    fn o_proj(&self) -> &QMatMul;

    /// Numeric mode for this layer's dense projections (q/k/v/o), read straight off the o_proj
    /// weight: `Off` (FP16) for an FP-loaded model, an int8 mode for a KO-loaded one. Drives the
    /// producer-fused q8a128 path (B1/B2).
    fn int8mode(&self) -> Int8Mode {
        self.o_proj().int8mode()
    }

    /// Output projection (B2) consuming the attention context as a `DynamicActs` — q8a1024 (decode,
    /// fused by the paged kernel) or `Float` (prefill/glue, quantized at the matmul). Default runs
    /// `forward_dynamic` off the o_proj weight: an `Int8` operand (decode) goes straight to the KO
    /// matmul; a `Float` operand is the FP path (or quantized at the matmul for a KO weight).
    /// Generic across all models.
    fn output_projection(&self, attn: DynamicActs, out_dtype: DType) -> Result<Tensor> {
        self.o_proj().forward_dynamic(attn.as_dynamic(), out_dtype)
    }
}

// ============================================================================
// Layer Processing Functions
// ============================================================================

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
pub fn forward_layer_batched_mixed<L: BatchedAttentionLayer>(
    layer: &L,
    groups: &mut [WaveAttnGroup<'_, '_>],
    x: &mut TensorCat,
    act_dtype: DType,
    layer_idx: usize,
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
    let mut parts: Vec<Tensor> = Vec::with_capacity(groups.len());
    let mut row0 = 0usize;
    for g in groups.iter_mut() {
        if g.rows == 0 {
            continue;
        }
        let slice = xt.narrow(1, row0, g.rows)?;
        let x_g = if g.decode_layout {
            TensorCat::from_cat_tensor(slice.reshape((g.rows, 1, hidden))?.contiguous()?, 0)?
        } else {
            TensorCat::from_cat_tensor(slice.contiguous()?, 0)?
        };
        let h = forward_attn_batched(layer, g.caches, &x_g, g.offsets, g.params, layer_idx)?
            .to_tensor();
        let h = if g.decode_layout {
            h.reshape((1, g.rows, hidden))?
        } else {
            h
        };
        parts.push(h);
        row0 += g.rows;
    }
    let h_attn = if parts.len() == 1 {
        parts.pop().unwrap()
    } else {
        Tensor::cat(&parts, 1)?
    };

    // First residual: x = x + attn(h).
    x.to_dtype_mut(h_attn.dtype())?;
    x.add_mut(&h_attn)?;
    drop(h_attn);

    // ── Shared FFN/MoE over the WHOLE combined buffer — one grouped GEMM whose
    // per-layer expert load serves every row-type at once. ──
    let mlp_dtype = if act_dtype == DType::F16 {
        DType::BF16
    } else {
        act_dtype
    };
    let mut h2 = {
        let acts = layer.ffn_norm(x.as_cat_tensor(), layer.int8mode())?;
        layer.ffn_forward(acts, mlp_dtype)?
    };
    h2.to_dtype_mut(orig_dtype)?;
    x.to_dtype_mut(orig_dtype)?;
    x.add_mut(&h2)?;
    Ok(())
}

/// Compute batched attention for a layer.
///
/// Dispatches to single-token decode or multi-token prefill paths.
pub fn forward_attn_batched<L: BatchedAttentionLayer>(
    layer: &L,
    caches: &mut [&mut KvCache],
    x: &TensorCat,
    offsets: &[usize],
    params: &BatchedAttentionParams<'_>,
    layer_idx: usize,
) -> Result<TensorCat> {
    // Route by the batch's DECLARED flavour (its headers), not tensor shape: a
    // mixed-wave group hands each type its own kernel, and a single-token prefill
    // ([1,1,h]) or a multi-token decode group ([D,1,h]) must still take the
    // correct path — a shape test (`dim(1)==1`) would misroute both.
    let is_decode = matches!(params.decode_headers, DecodeHeaders::Decode { .. });
    if is_decode {
        let ret = forward_attn_batched_single(
            layer,
            caches,
            x,
            offsets,
            params.rope_cos,
            params.rope_sin,
            params.rope_interleaved,
            params.inv_freq_device,
            params.rope_cs,
            params.generation,
            match &params.decode_headers {
                DecodeHeaders::Decode {
                    buf: Some(b),
                    stride,
                } => b.dev_ptr() + layer_idx as u64 * stride,
                _ => 0,
            },
        )?;
        Ok(ret)
    } else {
        let prefill_meta = match &params.decode_headers {
            DecodeHeaders::Prefill(m) => Some((&m.cu_seqlens_q, &m.q_lens, &m.kv_lens)),
            _ => None,
        };
        let glue_meta = match &params.decode_headers {
            DecodeHeaders::Prefill(m) => m.glue.as_ref(),
            _ => None,
        };
        let ret = forward_attn_batched_multi(
            layer,
            caches,
            x,
            offsets,
            params.q_lens,
            params.rope_cos,
            params.rope_sin,
            params.rope_interleaved,
            prefill_meta,
            glue_meta,
            params.rope_cs,
            params.generation,
            params.shared_prefill_pm,
        )?;
        Ok(ret)
    }
}

/// Single-token batched attention (decode path).
#[allow(clippy::too_many_arguments)]
fn forward_attn_batched_single<L: BatchedAttentionLayer>(
    layer: &L,
    caches: &mut [&mut KvCache],
    x: &TensorCat,
    offsets: &[usize],
    cos: &Tensor,
    sin: &Tensor,
    rope_interleaved: bool,
    #[allow(unused_variables)] inv_freq_device: &Tensor,
    #[allow(unused_variables)] rope_cs: &Tensor,
    #[allow(unused_variables)] generation: &Generation,
    #[allow(unused_variables)] decode_headers_ptr: u64,
) -> Result<TensorCat> {
    validate_batch_sizes(caches.len(), offsets.len(), x.len())?;

    // `x` is the PRE-norm activation; B1 fuses ln1 → q/k/v here. Shapes are norm-invariant.
    let x_tensor = x.as_cat_tensor();
    let (b_sz, seq_len, _n_embd) = x_tensor.dims3()?;
    debug_assert_eq!(seq_len, 1);
    let _act_dtype = x_tensor.dtype();

    // Project Q/K/V over the fused attention_norm (q8a128 on int8, FP on Off / non-CUDA).
    let t_qkv = profile_now();
    let QkvProjection { q, k, v } = {
        let acts = layer.attention_norm(x_tensor, layer.int8mode())?;
        layer.project_qkv(&acts, x_tensor.dtype())?
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
    profile_sync(q.device());
    pipeline_record("decode:qkv_proj", t_qkv);

    // Check for chunked (paged) KV cache BEFORE applying model-side RoPE.
    // For the paged path the decode kernel handles RoPE internally (via zeros rope_offsets
    // meaning natural positions), so we must NOT pre-rotate Q/K here.
    let use_paged = caches
        .first()
        .and_then(|c| c.k_cache().chunked_arena_chunks())
        .is_some();

    // Apply model-side RoPE only for the non-paged path.
    // Paged kernel always applies RoPE internally â€” applying it here would double-rotate.
    let (q, k) = if use_paged {
        // Kernel will rotate; skip model-side rotation.
        (q, k)
    } else {
        // Validate RoPE cos/sin for non-paged path
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

    // B2: emit the decode context directly as q8a1024 (head_dim 128 only) so o_proj runs int8
    // with no standalone quantize. Only on the paged CUDA decode path; false → FP context.
    let want_q8 = layer.int8mode().is_int8() && use_paged && seq_len == 1 && head_dim == 128;

    let outputs = if use_paged && seq_len == 1 {
        paged_decode_attention(
            caches,
            offsets,
            &q,
            &k,
            &v,
            n_head,
            n_kv_head,
            head_dim,
            rope_cs,
            rope_interleaved,
            generation,
            decode_headers_ptr,
            want_q8,
        )?
    } else {
        // Non-chunked fallback: standard per-sequence attention
        standard_batched_attention(caches, &q, &k, &v, head_dim, n_head, n_kv_head)?
    };

    // B2: o_proj over the attention context. On CUDA, `want_q8` → `outputs` is the flat U8 q8a1024
    // context wrapped (no copy) into an int8 operand; otherwise the FP context (the int8 override,
    // if any, quantizes it at the matmul). Non-CUDA never sets `want_q8` and has no dynamic o_proj.
    let t_out_proj = profile_now();
    let attn_out = if want_q8 {
        let op = Q8a128Operand::from_tensor(outputs, b_sz, n_head * head_dim)
            .with_lead(vec![b_sz, seq_len]);
        layer.output_projection(DynamicActs::Int8(op), x_tensor.dtype())?
    } else {
        let out = outputs.reshape((b_sz, 1, n_head * head_dim))?;
        layer.output_projection(DynamicActs::Float(out), x_tensor.dtype())?
    };
    profile_sync(attn_out.device());
    pipeline_record("decode:out_proj", t_out_proj);

    TensorCat::from_tensors(0, std::iter::once(attn_out))
}

/// Multi-token batched attention (prefill path).
#[allow(clippy::too_many_arguments)]
fn forward_attn_batched_multi<L: BatchedAttentionLayer>(
    layer: &L,
    caches: &mut [&mut KvCache],
    x: &TensorCat,
    offsets: &[usize],
    q_lens: &[usize],
    cos: &Tensor,
    sin: &Tensor,
    rope_interleaved: bool,
    prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
    glue_meta: Option<&GlueMeta>,
    rope_cs: &Tensor,
    generation: &Generation,
    shared_pm: &std::cell::RefCell<Option<SharedPm>>,
) -> Result<TensorCat> {
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
    let t_qkv = profile_now();
    let QkvProjection { q, k, v } = {
        let acts = layer.attention_norm(x_tensor, layer.int8mode())?;
        layer.project_qkv(&acts, x_tensor.dtype())?
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
    profile_sync(q.device());
    pipeline_record("prefill:qkv_proj", t_qkv);

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

    // Truncate each sequence to its prefill offset first: a fresh sequence is
    // untouched, but a re-prefill at the same offset (the bench harness's repeat
    // loop) discards the stale tail chunks the previous run left — otherwise they
    // stack up and push the decode writer past a gap of empty chunks.
    truncate_caches_to_offset(caches, offsets);

    let rope_zeros = Tensor::zeros(n_seqs, DType::U32, q.device())?;
    // Flat attention output: [total_q, n_head, head_dim]. A reprojection-glue
    // forward (HD128, chunked) routes to the paged-glue kernel — it streams the
    // quantized slot once and reuses it across all glue rows (dequant-once),
    // positioning every column by its chunk `rope_base` (`slice_rope`) and
    // masking each glue token by `cpos > row_pos + fwd_ahead[t]`. Everything else
    // (ordinary prefill, non-128 head dims) stays on the plain prefill kernel.
    let out_packed = match glue_meta {
        Some(g) if is_cuda_paged && head_dim == 128 => paged_glue_attn(
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
            rope_cs,
            rope_interleaved,
            generation,
            shared_pm,
        )?,
        // Ordinary prefill (fresh or over an existing prefix): the INT8
        // prefix-attention kernel (docs/archived/prefill_optimization.md) —
        // GQA-packed M, slice-aligned tiles, int8 MMA directly over the
        // quantized arena, split-KV for the short-q/long-prefix regime.
        // Head dims outside {64, 128} and interleaved RoPE fail loudly in
        // paged_prefill_attn_varlen_chunks.
        _ => paged_prefill_batched(
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
            &rope_zeros,
            rope_cs,
            rope_interleaved,
            generation,
            shared_pm,
        )?,
    };

    // Project per-token: [total_q, n_head*head_dim] -> [total_q, hidden_out].
    // (n_head*head_dim may differ from n_embd, e.g. Qwen3-MoE.)
    let reshaped_ctx = out_packed
        .contiguous()?
        .reshape((total_q, n_head * head_dim))?;
    // B2: prefill/glue feed the FP context as `Float`; the int8 override quantizes at the matmul
    // (launch amortized against the large prefill GEMM).
    let t_out_proj = profile_now();
    let output = {
        let dt = reshaped_ctx.dtype();
        layer.output_projection(DynamicActs::Float(reshaped_ctx), dt)?
    };
    profile_sync(output.device());
    pipeline_record("prefill:out_proj", t_out_proj);
    // Restore the flat-packed [1, total_q, hidden_out] activation.
    let hidden_out = output.dim(1)?;
    let output = output.reshape((1, total_q, hidden_out))?;
    TensorCat::from_cat_tensor(output, 1)
}

/// Simple per-sequence prefill attention fallback.
///
/// This processes each sequence independently using flash-attn (when available)
/// or standard attention with causal masking. Works with both contiguous and chunked KV caches.
/// Used for quantized KV cache mode where paged CUDA kernels aren't available.
#[allow(dead_code)]
#[allow(clippy::too_many_arguments)]
fn prefill_attention_simple(
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_head: usize,
    n_kv_head: usize,
) -> Result<Tensor> {
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
            cache.set_current_seq_len(offset + seq_len)?;

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
fn standard_attention_prefill(
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

    let mask: Vec<f32> = (0..q_len)
        .flat_map(|i| {
            (0..cache_len).map(move |j| {
                // Position i in query corresponds to absolute position (prefix_len + i)
                // It can attend to positions 0..=(prefix_len + i)
                if j > prefix_len + i {
                    f32::NEG_INFINITY
                } else {
                    0.0f32
                }
            })
        })
        .collect();
    let mask = Tensor::from_vec(mask, (1, 1, q_len, cache_len), q.device())?;
    let mask = mask.to_dtype(att.dtype())?;
    let att = att.broadcast_add(&mask)?;

    let att = candle_nn::ops::softmax_last_dim(&att)?;
    att.matmul(&v_cached.contiguous()?)
}

// ============================================================================
// Attention Implementations
// ============================================================================

/// Paged decode attention using chunked KV cache.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn paged_decode_attention(
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rope_cs: &Tensor,
    rope_interleaved: bool,
    _generation: &Generation,
    decode_headers_ptr: u64,
    // B2: emit the attention context as q8a1024 (returns a flat U8 tensor) instead of an FP
    // context, so o_proj runs int8 with no standalone quantize. Head_dim 128 only.
    emit_q8: bool,
) -> Result<Tensor> {
    let t_alloc = profile_now();
    KvCache::validate_chunked_decode_batch(caches, offsets)?;
    profile_sync(q.device());
    pipeline_record("decode:alloc", t_alloc);

    let t_meta = profile_now();
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
    //   The decode kernel reads k_new/v_new as BF16* and arena_store_element converts BF16â†’FP8
    //   when writing to the arena. Passing FP8 bytes to a BF16* kernel produces garbage.
    // - Other arenas: Q/k_new/v_new must all match arena dtype
    let (q_kernel, k_kernel, v_kernel) = if arena_dtype == DType::F8E4M3 {
        // FP8 KV cache with BF16 compute: Q and new K/V are all BF16.
        // The kernel writes BF16â†’FP8 to the arena via arena_store_element (correct conversion).
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
        profile_sync(q_kernel.device());
        pipeline_record("decode:meta", t_meta);

        let t_kernel = profile_now();
        // B2: emit q8a1024 (flat U8 tensor) instead of the FP context when requested. The U8
        // bytes are the operand for o_proj's int8 matmul — never dtype-converted.
        let raw_out = if emit_q8 {
            paged_decode_attn_q8(
                &q_kernel,
                decode_headers_ptr,
                arena_dtype,
                n_head,
                n_kv_head,
                head_dim,
                softmax_scale,
                &k_kernel,
                &v_kernel,
                rope_cs,
                rope_interleaved,
            )?
        } else {
            paged_decode_attn(
                &q_kernel,
                decode_headers_ptr,
                arena_dtype,
                n_head,
                n_kv_head,
                head_dim,
                softmax_scale,
                &k_kernel,
                &v_kernel,
                rope_cs,
                rope_interleaved,
            )?
        };
        profile_sync(q_kernel.device());
        pipeline_record("decode:kernel", t_kernel);

        if !emit_q8 && q_dtype != arena_dtype {
            raw_out.to_dtype(q_dtype)?
        } else {
            raw_out
        }
    };
    for (cache, &offset) in caches.iter_mut().zip(offsets.iter()) {
        cache.set_current_seq_len(offset + 1)?;
    }

    // After writing each decode token, eagerly quantize any newly-sealed chunk.
    Ok(out)
}

/// Standard (non-paged) batched attention fallback.
fn standard_batched_attention(
    caches: &mut [&mut KvCache],
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    head_dim: usize,
    n_head: usize,
    n_kv_head: usize,
) -> Result<Tensor> {
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

// ============================================================================
// Helper Functions
// ============================================================================

/// Reset caches where offset == 0.
pub fn reset_caches_at_zero(caches: &mut [&mut KvCache], offsets: &[usize]) {
    for (cache, &offset) in caches.iter_mut().zip(offsets.iter()) {
        if offset == 0 {
            cache.reset();
        }
    }
}

/// Prefill idempotency: truncate each sequence's KV to exactly its `offset`
/// cum-tokens before the prefill writes its tokens. A fresh sequence (or offset
/// 0) is a no-op / full clear; a re-prefill at the same offset (e.g. the bench
/// harness's repeat loop) discards the stale tail chunks the previous run left,
/// so the prefill never stacks duplicate chunks (which would push the decode
/// writer past a gap of empty chunks and corrupt attention).
pub fn truncate_caches_to_offset(caches: &mut [&mut KvCache], offsets: &[usize]) {
    for (cache, &offset) in caches.iter_mut().zip(offsets.iter()) {
        cache.truncate_to_offset(offset);
    }
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
fn ensure_contiguous(t: &Tensor) -> Result<Tensor> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_meta_ragged_builds_varlen_layout() {
        let dev = Device::Cpu;
        // Three sequences with different prefix offsets and different new-token
        // (query) lengths — the ragged case.
        let offsets = [0usize, 5, 100];
        let q_lens = [3usize, 7, 2];
        let m = BatchedPrefillMeta::new_ragged(&offsets, &q_lens, &dev).unwrap();

        // cu_seqlens_q is the exclusive prefix sum of q_lens → packed Q has
        // 3+7+2 = 12 rows.
        assert_eq!(m.cu_seqlens_q.to_vec1::<u32>().unwrap(), vec![0, 3, 10, 12]);
        // q_lens passthrough.
        assert_eq!(m.q_lens.to_vec1::<u32>().unwrap(), vec![3, 7, 2]);
        // kv_lens[i] = offset[i] + q_len[i].
        assert_eq!(m.kv_lens.to_vec1::<u32>().unwrap(), vec![3, 12, 102]);
    }

    #[test]
    fn prefill_meta_uniform_matches_ragged_special_case() {
        let dev = Device::Cpu;
        let offsets = [0usize, 0, 0];
        let seq_len = 4usize;
        let u = BatchedPrefillMeta::new(&offsets, seq_len, &dev).unwrap();

        // Uniform is exactly the ragged case with equal q_lens.
        assert_eq!(u.cu_seqlens_q.to_vec1::<u32>().unwrap(), vec![0, 4, 8, 12]);
        assert_eq!(u.q_lens.to_vec1::<u32>().unwrap(), vec![4, 4, 4]);
        assert_eq!(u.kv_lens.to_vec1::<u32>().unwrap(), vec![4, 4, 4]);
    }
}
