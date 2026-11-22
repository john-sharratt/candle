//! The batched layer abstraction: the trait a transformer layer implements and
//! the generic attention path that drives it over a wave.
//!
//! A layer supplies its norms, projections, output projection and feed-forward,
//! and this module runs attention for decode or prefill against the batch's
//! headers. RoPE comes from the tables passed in the shared parameters, so no
//! layer implements it. Each layer's attention and feed-forward outputs are
//! carved from a generation over the wave, so a result cannot outlive the memory
//! it was taken from.
//!
//! - `decode_attn`: single-token attention for decode.
//! - `fallbacks`: correct but slower attention for shapes the kernels do not cover.
//! - `mixed_layer`: one layer over a mixed wave.
//! - `prefill_attn`: multi-token attention for prefill and glue.
//! - `projection_meta`: the shared prefill tables and projection types.
//! - `tests_prefill`: tests of prefill metadata and the float fallback.

#[cfg(feature = "cuda")]
use candle::quantized::cuda::{DynamicActs, Q8a128Operand};
use candle::quantized::decode_rows::DecodeRows;
use candle::quantized::pinned_staging::{Generation, GpuBuf};
use candle::quantized::Int8Mode;
#[cfg(feature = "cuda")]
use candle::quantized::SumScale;
use candle::{DType, Device, Result, Tensor};
use candle_nn::kv_cache::KvCache;
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::{begin_wave, LayerPhase};

use crate::models::lazy_rope::LazyRope;
use crate::models::operand_guard::{expect_dense_view, expect_dtype};
#[cfg(feature = "cuda")]
use crate::models::prefill_utils::paged_decode_attn;
#[cfg(feature = "cuda")]
use crate::models::prefill_utils::paged_decode_attn_q8;
use crate::models::prefill_utils::{
    int8_prefill_act_dtype, int8_prefill_head_dim, paged_decode_q8_head_dim, paged_glue_attn,
    paged_prefill_batched, SharedPm,
};
use crate::models::profile::{gpu_span, pipeline_record, profile_now, span};
use crate::models::qsa_selection::QsaSelection;
use crate::models::quantized_matmul::QMatMul;
use crate::models::rope_schedule::RopeRungs;
use crate::utils::repeat_kv;

#[cfg(feature = "cuda")]
use candle_kernels::CHUNK_SIZE;

use super::tensor_cat::TensorCat;
use candle::LiveTensor;
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::WaveGeneration;

use candle::wave_provenance::WaveTicket;

use crate::models::wave_buffers::wave_from_vec_ticketed;

/// A borrow of the attention generation, threaded from the layer down to the
/// kernels that allocate inside it.
///
/// There is no wave domain without CUDA, so the non-CUDA arm carries a unit
/// borrow: the plumbing keeps one shape across both configurations, and `'w`
/// still bounds every result the way it does on the CUDA path.
#[cfg(feature = "cuda")]
pub(crate) type WaveRef<'w> = Option<&'w WaveGeneration>;

#[cfg(not(feature = "cuda"))]
pub(crate) type WaveRef<'w> = Option<&'w ()>;

// Batched Attention Parameters

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
        /// Byte stride between successive layers (`n_active * SLOT_HEADER_BYTES`).
        stride: u64,
    },
}

/// Parameters for batched attention computation.
///
/// Values shared by every layer of one forward for one attention group.
pub struct BatchedAttentionParams<'a> {
    /// Model-side RoPE (cos, sin) at the group's positions. Only the non-paged
    /// path reads them — the paged kernels rotate from `rope` — so they are
    /// built on the first non-paged layer, never on a paged forward.
    pub model_rope: &'a LazyRope<'a>,
    /// Whether RoPE uses interleaved format.
    pub rope_interleaved: bool,
    /// The model's RoPE rungs: every paged kernel rotates each sequence by the
    /// rung its `SlotHeader` names (`docs/progressive_yarn.md` §6).
    pub rope: &'a RopeRungs,
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_rope: &'a LazyRope<'a>,
        rope_interleaved: bool,
        rope: &'a RopeRungs,
        decode_headers: DecodeHeaders,
        q_lens: &'a [usize],
        generation: &'a Generation,
        shared_prefill_pm: &'a std::cell::RefCell<Option<SharedPm>>,
    ) -> Self {
        Self {
            model_rope,
            rope_interleaved,
            rope,
            decode_headers,
            q_lens,
            generation,
            shared_prefill_pm,
        }
    }
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
    /// Seeds the attention phase. This is the first allocation made *inside* the
    /// scope that holds the generation, so it names the wave directly; the
    /// projections, the context and o_proj after it all inherit from their
    /// operands. `'w` comes from the guard, which is what stops the activation
    /// outliving the span it was carved from.
    fn attention_norm<'w>(
        &self,
        x: &Tensor,
        mode: Int8Mode,
        wave: WaveRef<'w>,
    ) -> Result<DynamicActs<'w>>;

    /// FFN layer norm (ln2) as a producer epilogue (B3): q8a128 for an int8 `mode`, `Float` for
    /// `Off`.
    /// Seeds the FFN phase, exactly as [`Self::attention_norm`] seeds attention:
    /// it is the first allocation made inside the scope holding the FFN
    /// generation, so it names the wave directly and the router, the gather, the
    /// expert GEMMs and the combine all inherit from their operands. `'w` comes
    /// from the guard, which is what stops the activation outliving the span it
    /// was carved from.
    fn ffn_norm<'w>(
        &self,
        x: &Tensor,
        mode: Int8Mode,
        wave: WaveRef<'w>,
    ) -> Result<DynamicActs<'w>>;

    /// The FFN/MoE sub-block's residual update, `x += ffn(acts)`, consuming the
    /// producer-prepared `DynamicActs` (the fused `ffn_norm`): the router + expert
    /// gather take the q8a128 directly (no standalone quantize).
    ///
    /// The residual is the implementation's to update rather than a result to
    /// hand back, because the last step of an FFN is often cheapest folded into
    /// that add — a gated shared expert's sigmoid, multiply and combine run in
    /// the same launch as the residual add ([`super::qwen35::shared_residual`]).
    /// An implementation with nothing to fold adds its output with
    /// [`add_ffn_residual`].
    ///
    /// `work_dtype` is the FP-stable width the SwiGLU intermediates need — an
    /// F16 activation runs them in BF16, whose range they can exceed F16's. The
    /// residual's own type is what lands in it: a down projection stores it
    /// directly, so nothing narrows a full tensor per layer per wave to undo a
    /// widening only the intermediates needed.
    ///
    /// `'w` bounds the FFN's transients: the activations come from
    /// [`Self::ffn_norm`], and the MoE combine target is taken from the wave.
    /// `decode` names the rows (of this call's combined buffer) that are
    /// decode-attributed. Only an MoE implementation reads it, to weight
    /// expert-cache residency scoring — a decode row's reuse of a given expert
    /// is near-certain step to step, a prefill row's is close to zero, so the
    /// two must not bid for slots on equal footing. A dense FFN ignores it.
    fn ffn_residual<'w>(
        &self,
        x: &mut Tensor,
        acts: DynamicActs<'w>,
        work_dtype: DType,
        decode: &DecodeRows,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<()>;

    /// Project Q/K/V over the producer-prepared `DynamicActs` (the fused `attention_norm`), folding
    /// in any Q/K/V bias and q/k-norm. q/k/v share the single quantize (B1).
    ///
    /// # Returns
    /// QkvProjection containing Q, K, V tensors.
    /// - Q shape: (batch, seq_len, n_head * head_dim)
    /// - K shape: (batch, seq_len, n_kv_head * head_dim)
    /// - V shape: (batch, seq_len, n_kv_head * head_dim)
    ///
    /// `wave` is the open attention phase, for the one case the operand cannot
    /// supply: a `Float` activation is the residual cloned rather than a fresh
    /// allocation, so on a stack whose residual is pool-backed the projection
    /// group has no ticket to inherit and its split block fell to the pool.
    fn project_qkv<'w>(
        &self,
        acts: &DynamicActs<'w>,
        out_dtype: DType,
        wave: WaveRef<'w>,
    ) -> Result<QkvProjection<'w>>;

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
    /// Generic over `'w` so the decode context — which lives on the wave's
    /// transient half — can be projected without being copied off it first.
    /// The result inherits `'w` from the context: the matmul writes into the
    /// arena its operand came from, so a wave-scoped context yields a
    /// wave-scoped projection, and the guard that reclaims the context also
    /// bounds the result.
    fn output_projection<'w>(
        &self,
        attn: DynamicActs<'w>,
        out_dtype: DType,
    ) -> Result<LiveTensor<'w>> {
        self.o_proj().forward_dynamic(attn.as_dynamic(), out_dtype)
    }
}

/// Compute batched attention for a layer.
///
/// Dispatches to single-token decode or multi-token prefill paths.
///
/// `qsa` is this LAYER's block-sparse selection for these rows, which is why
/// it is an argument rather than a field of the wave-invariant `params`: a
/// hybrid stack's indexer produces a new one at every full-attention layer.
#[allow(clippy::too_many_arguments)]
pub fn forward_attn_batched<'w, L: BatchedAttentionLayer>(
    layer: &L,
    caches: &mut [&mut KvCache],
    x: &TensorCat,
    offsets: &[usize],
    params: &BatchedAttentionParams<'_>,
    layer_idx: usize,
    qsa: Option<&QsaSelection>,
    wave: WaveRef<'w>,
) -> Result<LiveTensor<'w>> {
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
            params.model_rope,
            params.rope_interleaved,
            params.rope,
            params.generation,
            match &params.decode_headers {
                DecodeHeaders::Decode {
                    buf: Some(b),
                    stride,
                } => b.dev_ptr() + layer_idx as u64 * stride,
                _ => 0,
            },
            qsa,
            wave,
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
            params.model_rope,
            params.rope_interleaved,
            prefill_meta,
            glue_meta,
            params.rope,
            params.generation,
            params.shared_prefill_pm,
            qsa,
            wave,
        )?;
        Ok(ret)
    }
}

mod decode_attn;
mod fallbacks;
mod mixed_layer;
mod prefill_attn;
mod projection_meta;
#[cfg(test)]
mod tests_prefill;

use decode_attn::*;
pub use fallbacks::*;
pub use mixed_layer::*;
use prefill_attn::*;
pub use projection_meta::*;
