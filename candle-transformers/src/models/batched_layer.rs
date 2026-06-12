//! Batched transformer layer processing for continuous batching.
//!
//! This module provides traits and utilities for processing transformer layers
//! in batched mode, enabling efficient continuous batching across multiple sequences.
//!
//! The key abstraction is [`BatchedAttentionLayer`] which defines the interface
//! that transformer layers must implement to support batched attention processing.
//! The actual batched attention computation is implemented generically in this module.

use candle::quantized::pinned_staging::{Generation, GpuBuf};
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::KvCache;

#[cfg(feature = "cuda")]
use crate::models::prefill_utils::paged_decode_attn;
use crate::models::prefill_utils::paged_prefill_batched;
use crate::models::prefill_utils::paged_prefill_batched_gap_fill;
use crate::models::profile::{pipeline_record, profile_now, profile_sync};
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
        }
    }
}

/// Precomputed metadata for paged prefill attention.
///
/// This avoids rebuilding the same small tensors (cu_seqlens, q_lens, kv_lens)
/// for every layer during multi-token prefill.
/// GAP_FILL descriptor: routes a prefill through the gap-fill kernel. Sealed
/// content is the (packed) contiguous prefix; all glue is the new region,
/// appended as the writer tail. `col_actual_pos` relabels every column with its
/// TRUE sequence position so each glue token attends only logically-earlier
/// columns (sealed or glue) regardless of physical packing order. The resulting
/// `[sealed | glue]` slot decodes correctly because attention is order-invariant
/// over keys and the glue K/V already carries its logical RoPE.
#[derive(Clone)]
pub struct GapFillDescriptor {
    /// Flat (per-slot kv_len) array of each column's TRUE sequence position.
    pub col_actual_pos: std::sync::Arc<Vec<u32>>,
}

#[derive(Clone)]
pub struct BatchedPrefillMeta {
    pub cu_seqlens_q: Tensor,
    pub q_lens: Tensor,
    pub kv_lens: Tensor,
    pub has_prefix: bool,
    /// `Some` → run the gap-fill kernel with this descriptor; `None` = normal.
    pub gap_fill: Option<GapFillDescriptor>,
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
        let has_prefix = offsets.iter().any(|&o| o > 0);
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
            has_prefix,
            gap_fill: None,
        })
    }

    /// Attach the GAP_FILL descriptor — routes this prefill through the gap-fill
    /// kernel.
    pub fn with_gap_fill(mut self, desc: GapFillDescriptor) -> Self {
        self.gap_fill = Some(desc);
        self
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

    /// Apply attention layer normalization.
    fn attention_norm(&self, x: &Tensor) -> Result<Tensor>;

    /// Apply FFN layer normalization.
    fn ffn_norm(&self, x: &Tensor) -> Result<Tensor>;

    /// Apply the FFN/MoE module.
    fn ffn_forward(&self, x: &Tensor) -> Result<Tensor>;

    /// Project input to Q, K, V tensors.
    ///
    /// Implementations should include any Q/K/V biases in the projection output.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape (batch, seq_len, hidden_dim)
    ///
    /// # Returns
    /// QkvProjection containing Q, K, V tensors.
    /// - Q shape: (batch, seq_len, n_head * head_dim)
    /// - K shape: (batch, seq_len, n_kv_head * head_dim)
    /// - V shape: (batch, seq_len, n_kv_head * head_dim)
    fn project_qkv(&self, x: &Tensor) -> Result<QkvProjection>;

    /// Project attention output back to hidden dimension.
    ///
    /// # Arguments
    /// * `attn_output` - Attention output of shape (batch, seq_len, n_head * head_dim)
    ///
    /// # Returns
    /// Projected output of shape (batch, seq_len, hidden_dim)
    fn output_projection(&self, attn_output: &Tensor) -> Result<Tensor>;
}

// ============================================================================
// Layer Processing Functions
// ============================================================================

/// Process a full transformer layer in batched mode.
///
/// This function implements the standard pre-norm transformer layer:
/// ```text
/// h = attention_norm(x)
/// x = x + attention(h)
/// h = ffn_norm(x)
/// x = x + ffn(h)
/// ```
///
/// It handles dtype conversions for mixed-precision training (e.g., FP8 activations
/// with BF16 attention).
///
/// # Arguments
/// * `act_dtype` - The activation dtype for MLP matmuls (e.g., F8E4M3 for FP8 mode).
///   This should be the original embedding dtype, not the layer input dtype which
///   may have been converted to higher precision by previous layers.
pub fn forward_layer_batched<L: BatchedAttentionLayer>(
    layer: &L,
    caches: &mut [&mut KvCache],
    x: &mut TensorCat,
    offsets: &[usize],
    params: &BatchedAttentionParams<'_>,
    act_dtype: DType,
    write_offset_shifts_ptr: u64,
    layer_idx: usize,
) -> Result<()> {
    // Apply attention norm and compute attention
    let stage_is_decode = x.dim(1)? == 1;
    let attn_total_name = if stage_is_decode {
        "decode:model:attn:total"
    } else {
        "prefill:model:attn:total"
    };
    let attn_norm_name = if stage_is_decode {
        "decode:model:attn:norm"
    } else {
        "prefill:model:attn:norm"
    };
    let attn_core_name = if stage_is_decode {
        "decode:model:attn:core"
    } else {
        "prefill:model:attn:core"
    };
    let attn_residual_name = if stage_is_decode {
        "decode:model:attn:resid"
    } else {
        "prefill:model:attn:resid"
    };

    let t_attn_total = profile_now();
    let orig_dtype = x.dtype();

    let t_attn_norm = profile_now();
    let h = x.transform(|t, _| layer.attention_norm(t))?;
    profile_sync(h.as_cat_tensor().device());
    pipeline_record(attn_norm_name, t_attn_norm);

    let t_attn_core = profile_now();
    let h_attn = forward_attn_batched(
        layer,
        caches,
        h,
        offsets,
        params,
        write_offset_shifts_ptr,
        layer_idx,
    )?
    .to_tensor();
    profile_sync(h_attn.device());
    pipeline_record(attn_core_name, t_attn_core);

    // First residual: x = x + attention(h)
    // Convert x to attention dtype and add in-place
    let t_attn_residual = profile_now();
    x.to_dtype_mut(h_attn.dtype())?;
    x.add_mut(&h_attn)?;
    drop(h_attn); // Free attention output memory
    profile_sync(x.as_cat_tensor().device());
    pipeline_record(attn_residual_name, t_attn_residual);
    pipeline_record(attn_total_name, t_attn_total);

    let mlp_total_name = if stage_is_decode {
        "decode:model:mlp:total"
    } else {
        "prefill:model:mlp:total"
    };
    let mlp_norm_name = if stage_is_decode {
        "decode:model:mlp:norm"
    } else {
        "prefill:model:mlp:norm"
    };
    let mlp_cast_name = if stage_is_decode {
        "decode:model:mlp:cast"
    } else {
        "prefill:model:mlp:cast"
    };
    let mlp_ffn_name = if stage_is_decode {
        "decode:model:ffn:total"
    } else {
        "prefill:model:ffn:total"
    };
    let mlp_residual_name = if stage_is_decode {
        "decode:model:mlp:resid"
    } else {
        "prefill:model:mlp:resid"
    };

    let t_mlp_total = profile_now();
    // FFN: h2 = ffn(ffn_norm(x))
    let t_mlp_norm = profile_now();
    let mut h2 = layer.ffn_norm(x.as_cat_tensor())?;
    profile_sync(h2.device());
    pipeline_record(mlp_norm_name, t_mlp_norm);

    // F16 has limited dynamic range and can overflow in MLP intermediate values.
    // F16 max ~65504 - insufficient for MLP intermediates that can reach 10000+.
    // Use BF16 for MLP computation when activation dtype is F16.
    // Note: FP8 already uses BF16 internally via QMatMul, so no conversion needed.
    let mlp_dtype = if act_dtype == DType::F16 {
        DType::BF16
    } else {
        act_dtype
    };

    let t_mlp_cast_up = profile_now();
    h2.to_dtype_mut(mlp_dtype)?; // MLP in safe dtype for numerical stability
    profile_sync(h2.device());
    pipeline_record(mlp_cast_name, t_mlp_cast_up);

    let t_mlp_ffn = profile_now();
    h2 = layer.ffn_forward(&h2)?;
    profile_sync(h2.device());
    pipeline_record(mlp_ffn_name, t_mlp_ffn);

    // Second residual: x = x + ffn(h)
    // Convert h2 to x's dtype and add in-place
    let t_mlp_residual = profile_now();
    h2.to_dtype_mut(orig_dtype)?;
    x.to_dtype_mut(orig_dtype)?;
    x.add_mut(&h2)?;
    profile_sync(x.as_cat_tensor().device());
    pipeline_record(mlp_residual_name, t_mlp_residual);
    pipeline_record(mlp_total_name, t_mlp_total);

    Ok(())
}

/// Compute batched attention for a layer.
///
/// Dispatches to single-token decode or multi-token prefill paths.
pub fn forward_attn_batched<L: BatchedAttentionLayer>(
    layer: &L,
    caches: &mut [&mut KvCache],
    x: TensorCat,
    offsets: &[usize],
    params: &BatchedAttentionParams<'_>,
    write_offset_shifts_ptr: u64,
    layer_idx: usize,
) -> Result<TensorCat> {
    let seq_len = x.dim(1)?;
    // Use optimized batched implementation for single-token generation
    if seq_len == 1 {
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
        let (prefill_meta, gap_fill) = match &params.decode_headers {
            DecodeHeaders::Prefill(m) => (
                Some((&m.cu_seqlens_q, &m.q_lens, &m.kv_lens, m.has_prefix)),
                m.gap_fill.as_ref(),
            ),
            _ => (None, None),
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
            params.rope_cs,
            write_offset_shifts_ptr,
            params.generation,
            gap_fill,
        )?;
        Ok(ret)
    }
}

/// Single-token batched attention (decode path).
#[allow(clippy::too_many_arguments)]
fn forward_attn_batched_single<L: BatchedAttentionLayer>(
    layer: &L,
    caches: &mut [&mut KvCache],
    x: TensorCat,
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

    let x_tensor = x.as_cat_tensor();
    let (b_sz, seq_len, _n_embd) = x_tensor.dims3()?;
    debug_assert_eq!(seq_len, 1);
    let _act_dtype = x_tensor.dtype();

    // Project Q/K/V
    let t_qkv = profile_now();
    let QkvProjection { q, k, v } = layer.project_qkv(x_tensor)?;

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
    #[cfg(feature = "cuda")]
    let use_paged = caches
        .first()
        .and_then(|c| c.k_cache().chunked_arena_chunks())
        .is_some();
    #[cfg(not(feature = "cuda"))]
    let use_paged = false;

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

    let outputs = if use_paged && seq_len == 1 {
        #[cfg(feature = "cuda")]
        {
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
            )?
        }
        #[cfg(not(feature = "cuda"))]
        {
            standard_batched_attention(caches, &q, &k, &v, head_dim, n_head, n_kv_head)?
        }
    } else {
        // Non-chunked fallback: standard per-sequence attention
        standard_batched_attention(caches, &q, &k, &v, head_dim, n_head, n_kv_head)?
    };

    // Project back: attention output is (B, H, 1, D) => (B, 1, H*D)
    // Note: n_head*head_dim may differ from n_embd (e.g. Qwen3-MoE)
    let out = outputs.reshape((b_sz, 1, n_head * head_dim))?;

    let t_out_proj = profile_now();
    let attn_out = layer.output_projection(&out)?;
    profile_sync(attn_out.device());
    pipeline_record("decode:out_proj", t_out_proj);

    TensorCat::from_tensors(0, std::iter::once(attn_out))
}

/// Multi-token batched attention (prefill path).
#[allow(clippy::too_many_arguments)]
fn forward_attn_batched_multi<L: BatchedAttentionLayer>(
    layer: &L,
    caches: &mut [&mut KvCache],
    x: TensorCat,
    offsets: &[usize],
    q_lens: &[usize],
    cos: &Tensor,
    sin: &Tensor,
    rope_interleaved: bool,
    prefill_meta: Option<(&Tensor, &Tensor, &Tensor, bool)>,
    rope_cs: &Tensor,
    write_offset_shifts_ptr: u64,
    generation: &Generation,
    gap_fill: Option<&GapFillDescriptor>,
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

    // Project Q/K/V
    let t_qkv = profile_now();
    let QkvProjection { q, k, v } = layer.project_qkv(x_tensor)?;

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
    #[cfg(feature = "cuda")]
    let is_cuda_paged = caches
        .first()
        .and_then(|c| c.k_cache().chunked_arena_chunks())
        .is_some();
    #[cfg(not(feature = "cuda"))]
    let is_cuda_paged = false;

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

    reset_caches_at_zero(caches, offsets);

    let rope_zeros = Tensor::zeros(n_seqs, DType::U32, q.device())?;
    // Flat attention output: [total_q, n_head, head_dim].
    let out_packed = if let Some(desc) = gap_fill {
        paged_prefill_batched_gap_fill(
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
            write_offset_shifts_ptr,
            generation,
            &desc.col_actual_pos,
        )?
    } else {
        paged_prefill_batched(
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
            write_offset_shifts_ptr,
            generation,
        )?
    };

    // Project per-token: [total_q, n_head*head_dim] -> [total_q, hidden_out].
    // (n_head*head_dim may differ from n_embd, e.g. Qwen3-MoE.)
    let reshaped_ctx = out_packed
        .contiguous()?
        .reshape((total_q, n_head * head_dim))?;
    let t_out_proj = profile_now();
    let output = layer.output_projection(&reshaped_ctx)?;
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
        let raw_out = paged_decode_attn(
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
        )?;
        profile_sync(q_kernel.device());
        pipeline_record("decode:kernel", t_kernel);

        if q_dtype != arena_dtype {
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
        // A non-zero offset is present.
        assert!(m.has_prefix);
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
        assert!(!u.has_prefix);
    }
}
