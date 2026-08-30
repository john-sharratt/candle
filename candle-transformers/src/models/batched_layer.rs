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
use candle_nn::kv_cache::{begin_wave, LayerPhase};

use crate::models::operand_guard::expect_dtype;
#[cfg(feature = "cuda")]
use crate::models::prefill_utils::paged_decode_attn;
#[cfg(feature = "cuda")]
use crate::models::prefill_utils::paged_decode_attn_q8;
use crate::models::prefill_utils::{
    int8_prefill_act_dtype, int8_prefill_head_dim, paged_decode_q8_head_dim, paged_glue_attn,
    paged_prefill_batched, SharedPm,
};
use crate::models::profile::{gpu_span, pipeline_record, profile_now, span};
use crate::models::quantized_matmul::QMatMul;
use crate::utils::repeat_kv;

#[cfg(feature = "cuda")]
use candle_kernels::CHUNK_SIZE;

use super::tensor_cat::TensorCat;
use candle::LiveTensor;
#[cfg(feature = "cuda")]
use candle_nn::kv_cache::WaveGeneration;

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

// ============================================================================
// Layer-Level Trait
// ============================================================================

/// Q/K/V tensors after projection but before RoPE.
///
/// Shape: (batch, seq_len, hidden_dim) for each tensor.
///
/// `'w` is the generation the projections were allocated from — inherited from
/// the activation operand, since a matmul writes its output into whichever arena
/// its input came from. For a pool-backed activation `'w` is `'static` and this
/// is an ordinary owned triple.
pub struct QkvProjection<'w> {
    pub q: LiveTensor<'w>,
    pub k: LiveTensor<'w>,
    pub v: LiveTensor<'w>,
    /// Output gate for a **gated-attention** layer, `[.., n_head·head_dim]`,
    /// pre-sigmoid. `None` on the classic layers.
    ///
    /// The Qwen3-Next/Qwen3.5 lineage projects `2·head_dim` per head and
    /// splits it into `[query | gate]`; the gate is neither normed nor
    /// roped, and is applied as `sigmoid(gate) ⊙ context` after attention
    /// and before the output projection. It is produced with q/k/v — same
    /// weight, same fused activation — but consumed at the far end of the
    /// attention block, so it rides along here rather than being recomputed
    /// or stashed on the layer (which would not survive a layer being run
    /// for several row-groups in one wave).
    pub gate: Option<LiveTensor<'w>>,
}

/// The width Q/K/V and the output gate are projected in.
///
/// **The KV arena's, not the residual stream's** — and the distinction only has
/// teeth when a model computes in a width its KV storage does not use.
///
/// K and V *become* the cache: they are appended to the live arena, and the
/// attention kernel then reads them back alongside Q. So the projection that
/// produces them has to emit the width the arena holds, or the append converts
/// a full tensor per layer per wave (hot-path invariant 1) — or, worse, writes
/// one width's bits into a buffer read as another's. Q and the gate follow K
/// and V because the kernel consumes all of them together.
///
/// The residual stream is a separate question with a separate answer: the
/// output projection stores *its* width, which is where the two meet, and the
/// `expect_dtype` on that boundary is what holds the pair honest.
///
/// A cache with no float dtype of its own falls back to the residual's, which
/// is every model whose activations and KV agree — for those this is the
/// identity and nothing changes.
fn attention_operand_dtype(caches: &[&mut KvCache], residual: DType) -> DType {
    caches.first().map(|c| c.dtype()).unwrap_or(residual)
}

/// Apply a gated-attention layer's output gate to the attention context.
///
/// `sigmoid(gate) ⊙ context`, matching `qwen35.cpp`'s
/// `ggml_mul(cur, ggml_sigmoid(gate))` between attention and the output
/// projection. A `None` gate returns the context untouched, which is every
/// classic attention layer and costs nothing.
///
/// The gate is reshaped to the context's shape rather than assumed to match:
/// decode carries `[b, 1, n_head·head_dim]` while prefill is flat
/// `[total_q, n_head·head_dim]`, and the projection that produced the gate
/// followed the activation's layout, not the context's.
fn apply_attention_gate<'w>(
    ctx: LiveTensor<'w>,
    gate: Option<LiveTensor<'w>>,
) -> Result<LiveTensor<'w>> {
    let Some(gate) = gate else {
        return Ok(ctx);
    };
    if gate.elem_count() != ctx.elem_count() {
        candle::bail!(
            "attention gate has {} elements against a {}-element context",
            gate.elem_count(),
            ctx.elem_count()
        );
    }
    // The gate is the `wq` projection's second half, stored at the same width as
    // the context it multiplies — validated, not converted (invariant 1b).
    let gate = gate.reshape(ctx.shape())?;
    expect_dtype(&gate, ctx.dtype(), "attention gate vs the context it gates")?;
    &ctx * &candle_nn::ops::sigmoid(&gate)?
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

    /// FFN/MoE module consuming the producer-prepared `DynamicActs` (the fused `ffn_norm`): the
    /// router + expert gather take the q8a128 directly (no standalone quantize).
    ///
    /// Two widths, because they are two different questions. `work_dtype` is the
    /// FP-stable width the SwiGLU intermediates need — an F16 activation runs
    /// them in BF16, whose range they can exceed F16's. `out_dtype` is what the
    /// residual stream wants back, and the implementation **returns that**: the
    /// down projection stores it, so nothing narrows a full tensor per layer per
    /// wave to undo a widening only the intermediates needed.
    ///
    /// `'w` bounds the *result*, not the input: the FFN activations always come
    /// from [`Self::ffn_norm`], which allocates, while the MoE combine target is
    /// taken from the wave. A dense MLP hands its activations to a `Module` and
    /// so could not accept a wave-scoped operand anyway.
    fn ffn_forward<'w>(
        &self,
        acts: DynamicActs<'w>,
        work_dtype: DType,
        out_dtype: DType,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>>;

    /// Project Q/K/V over the producer-prepared `DynamicActs` (the fused `attention_norm`), folding
    /// in any Q/K/V bias and q/k-norm. q/k/v share the single quantize (B1).
    ///
    /// # Returns
    /// QkvProjection containing Q, K, V tensors.
    /// - Q shape: (batch, seq_len, n_head * head_dim)
    /// - K shape: (batch, seq_len, n_kv_head * head_dim)
    /// - V shape: (batch, seq_len, n_kv_head * head_dim)
    fn project_qkv<'w>(
        &self,
        acts: &DynamicActs<'w>,
        out_dtype: DType,
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
    // The attention generation spans every group's attention, the concatenation
    // of their contexts, o_proj, and the residual add that consumes the result —
    // the same shape as the FFN scope below. Opening it inside
    // `forward_attn_batched` instead would close it one step before the value
    // dies, which is the difference between o_proj's output living on the wave
    // and having to be allocated off it.
    //
    // Scoped to one layer, not the forward: consumption stays at a single
    // layer's working set rather than accumulating with depth
    // (`docs/archived/arena_unification.md` §3.6). Halves alternate per layer, so
    // layer N's reads are separated from layer N+2's writes by a whole layer of
    // same-stream work.
    {
        #[cfg(feature = "cuda")]
        let attn_wave = match xt.device() {
            Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Attention)?),
            _ => None,
        };
        #[cfg(not(feature = "cuda"))]
        let attn_wave: Option<()> = None;

        let mut parts: Vec<LiveTensor<'_>> = Vec::with_capacity(groups.len());
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
            let h = forward_attn_batched(
                layer,
                g.caches,
                &x_g,
                g.offsets,
                g.params,
                layer_idx,
                attn_wave.as_ref(),
            )?;
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
            LiveTensor::cat(&parts, 1)?
        };

        // First residual: x = x + attn(h). `add_mut` reads `h_attn` in place, so
        // the residual stream never takes a wave allocation and never escapes.
        //
        // VALIDATED, not converted: both output-projection sites store the
        // residual's own dtype, so the two agree by construction. Rewriting `x`
        // here to meet the attention output would be a full-tensor pass per
        // layer per wave, and one that silently absorbs a producer that later
        // starts emitting the wrong width (hot-path invariant 1b).
        expect_dtype(
            &h_attn,
            orig_dtype,
            "attention residual: attn(x) vs the residual stream",
        )?;
        x.add_mut(&h_attn)?;
        // No `drop(h_attn)` / `drop(attn_wave)`: `h_attn` borrows the guard, so
        // the compiler refuses any order but this one. Both die at the brace.
    }

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
    // target is what this bounds — it is returned from `ffn_forward`, so no
    // scope inside the MoE code could have bounded it (§3.6, and see
    // `wave_buffers`).
    #[cfg(feature = "cuda")]
    let ffn_wave = match x.as_cat_tensor().device() {
        Device::Cuda(d) => Some(begin_wave(&d.cuda_stream(), LayerPhase::Ffn)?),
        _ => None,
    };
    #[cfg(not(feature = "cuda"))]
    let ffn_wave: Option<()> = None;
    // `ffn_forward` returns `orig_dtype` — its down projection stores it — so
    // the residual add takes the result as it stands. Narrowing here instead
    // cost a full-tensor pass per layer per wave to undo the widening only the
    // SwiGLU intermediates needed.
    let h2 = {
        let acts = layer.ffn_norm(x.as_cat_tensor(), layer.int8mode(), ffn_wave.as_ref())?;
        layer.ffn_forward(acts, mlp_dtype, orig_dtype, ffn_wave.as_ref())?
    };
    // Same contract as the attention residual above: `ffn_forward` stores
    // `orig_dtype`, and the residual never left it, so this is an assertion.
    expect_dtype(
        &h2,
        orig_dtype,
        "ffn residual: ffn(x) vs the residual stream",
    )?;
    x.add_mut(&h2)?;
    // No `drop(h2)` / `drop(ffn_wave)` here: `h2` borrows `ffn_wave`, so the
    // compiler already refuses any order but this one. The guard falls out of
    // scope at the end of the function, fencing the stream and rewinding the
    // half — which is exactly where the hand-written drops used to put it.
    Ok(())
}

/// Compute batched attention for a layer.
///
/// Dispatches to single-token decode or multi-token prefill paths.
pub fn forward_attn_batched<'w, L: BatchedAttentionLayer>(
    layer: &L,
    caches: &mut [&mut KvCache],
    x: &TensorCat,
    offsets: &[usize],
    params: &BatchedAttentionParams<'_>,
    layer_idx: usize,
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
            params.rope_cos,
            params.rope_sin,
            params.rope_interleaved,
            prefill_meta,
            glue_meta,
            params.rope_cs,
            params.generation,
            params.shared_prefill_pm,
            wave,
        )?;
        Ok(ret)
    }
}

/// Single-token batched attention (decode path).
#[allow(clippy::too_many_arguments)]
fn forward_attn_batched_single<'w, L: BatchedAttentionLayer>(
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
        layer.project_qkv(&acts, kv_dtype)?
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
            rope_cs,
            rope_interleaved,
            generation,
            decode_headers_ptr,
            want_q8,
            if want_q8 { gate.as_ref() } else { None },
        )?
    } else {
        // Non-chunked fallback: standard per-sequence attention
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

/// Multi-token batched attention (prefill path).
#[allow(clippy::too_many_arguments)]
fn forward_attn_batched_multi<'w, L: BatchedAttentionLayer>(
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
        layer.project_qkv(&acts, kv_dtype)?
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

    // The cache is already sized and truncated for this wave: `wave_admit` did
    // both, for every layer, before the forward began. Nothing on this path may
    // claim a chunk — the transient tier is placed against the arena frontier
    // and a claim here would move it (`docs/elastic_vram_partition.md` §7).
    let rope_zeros = Tensor::zeros(n_seqs, DType::U32, q.device())?;
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
            rope_cs,
            rope_interleaved,
            generation,
            shared_pm,
        )?,
        // Shapes and dtypes the int8 prefix-attention kernel is not built for:
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
                rope_cs,
                rope_interleaved,
            )?
        }
        // Ordinary prefill (fresh or over an existing prefix): the INT8
        // prefix-attention kernel (docs/archived/prefill_optimization.md) —
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
/// - RoPE comes from the same shared `rope_cs` table the kernels read
///   (cos/sin interleaved per frequency, rows indexed by absolute position).
///
/// Takes the flat-packed varlen operands (`q [total_q, n_head, head_dim]`,
/// `k`/`v [total_q, n_kv_head, head_dim]`) and returns the flat-packed
/// context `[total_q, n_head, head_dim]`.
#[allow(clippy::too_many_arguments)]
fn paged_prefill_float_fallback(
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &LiveTensor<'_>,
    k: &LiveTensor<'_>,
    v: &LiveTensor<'_>,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rope_cs: &Tensor,
    rope_interleaved: bool,
) -> Result<Tensor> {
    // Pool copies: this path feeds `chunked_write_kv` and plain tensor math,
    // none of which is wave-aware (same posture as the non-paged fallback).
    let q = &q.to_owned_tensor()?;
    let k = &k.to_owned_tensor()?;
    let v = &v.to_owned_tensor()?;
    let n_rep = n_head / n_kv_head;
    let half = head_dim / 2;

    // The cos/sin planes, split from the shared interleaved table once for the
    // longest prefix any sequence here reaches — the table never changes
    // within a forward, so deriving it per sequence (narrow → reshape →
    // to_dtype → two contiguous copies, per call, per layer) was pure rework.
    let max_total = q_lens
        .iter()
        .zip(offsets.iter())
        .map(|(&l, &o)| o + l)
        .max()
        .unwrap_or(0);
    let cs_full = rope_cs
        .narrow(0, 0, max_total)?
        .reshape((max_total, half, 2))?
        .to_dtype(q.dtype())?;
    let cos_full = cs_full.narrow(2, 0, 1)?.squeeze(2)?.contiguous()?;
    let sin_full = cs_full.narrow(2, 1, 1)?.squeeze(2)?.contiguous()?;

    // Rotate `x [1, h, len, d]` at absolute positions `start..start + len`.
    //
    // Full-width rotation on purpose: the table's row covers every frequency
    // pair of the head, and on partial-rotary models `RotaryLayout::rope_table`
    // fills the pass-through pairs with exact `(cos 1, sin 0)` — rotation by
    // zero — so rotating all `head_dim/2` pairs is the identity on the
    // non-rotary dims. That padding IS the contract this fallback relies on;
    // a table with real frequencies in those rows would rotate dims the
    // kernels leave alone.
    let rot = |x: &Tensor, start: usize, len: usize| -> Result<Tensor> {
        let cos = cos_full.narrow(0, start, len)?;
        let sin = sin_full.narrow(0, start, len)?;
        if rope_interleaved {
            candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
        } else {
            candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
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
        cache.set_current_seq_len(offset + len)?;
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
        let k_rot = rot(&k_all, 0, total)?;
        let q_rot = rot(&qs, offset, len)?;
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
fn prefill_attention_simple(
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

// ============================================================================
// Attention Implementations
// ============================================================================

/// Paged decode attention using chunked KV cache.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn paged_decode_attention<'w>(
    wave: Option<&'w WaveGeneration>,
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &LiveTensor<'_>,
    k: &LiveTensor<'_>,
    v: &LiveTensor<'_>,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rope_cs: &Tensor,
    rope_interleaved: bool,
    _generation: &Generation,
    decode_headers_ptr: u64,
    // B2: emit the attention context as q8a1024 (returns a flat U8 tensor) instead of an FP
    // context, so o_proj runs int8 with no standalone quantize. Head_dim 128 or 256.
    emit_q8: bool,
    // Output gate folded into the q8 emit (`sigmoid(g) ⊙ ctx` inside the combine kernel).
    // Only meaningful with `emit_q8`; the FP path applies its gate on the FP context instead.
    gate: Option<&LiveTensor<'_>>,
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
                rope_cs,
                rope_interleaved,
                gate_kernel.as_ref(),
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
                rope_cs,
                rope_interleaved,
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

/// Standard (non-paged) batched attention fallback.
fn standard_batched_attention(
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
fn ensure_contiguous<'w>(t: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
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

    /// The head_dim-256 float prefill fallback (Qwen3.5/3.8 attention shape):
    /// varlen sequences through the real chunked cache must match plain
    /// rotated causal attention, and — the paged-cache interop invariant —
    /// the arena must hold the UNROTATED keys.
    #[cfg(feature = "cuda")]
    #[test]
    fn float_fallback_prefill_hd256_matches_reference() -> Result<()> {
        use crate::models::prefill_utils::compute_rope_cs;
        use candle::DType;
        use candle_nn::kv_cache::ChunkedKvBacking;

        let device = Device::new_cuda(0)?;
        let (n_head, n_kv_head, head_dim) = (16usize, 2usize, 256usize);
        let n_rep = n_head / n_kv_head;
        // One short sequence and one that crosses a chunk boundary.
        let q_lens = [7usize, 40usize];
        let offsets = [0usize, 0usize];
        let total_q: usize = q_lens.iter().sum();

        let backing = ChunkedKvBacking::new(2, n_kv_head, head_dim, DType::BF16, &device, 64)?;
        let mut caches: Vec<KvCache> = (0..2)
            .map(|seq| {
                let mut c = KvCache::new(2, 64);
                c.force_dtype(DType::BF16);
                c.set_chunked_backing(&backing, seq, None)?;
                Ok(c)
            })
            .collect::<Result<_>>()?;
        let mut cache_refs: Vec<&mut KvCache> = caches.iter_mut().collect();

        let q = Tensor::randn(0f32, 1f32, (total_q, n_head, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1f32, (total_q, n_kv_head, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let v = Tensor::randn(0f32, 1f32, (total_q, n_kv_head, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1f32 / 1e6f32.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let inv_freq_t = Tensor::from_vec(inv_freq, (head_dim / 2,), &device)?;
        let rope_cs = compute_rope_cs(&inv_freq_t, 4, head_dim, &device)?;

        let q_live = q.clone();
        let k_live = k.clone();
        let v_live = v.clone();
        let out = paged_prefill_float_fallback(
            &mut cache_refs,
            &offsets,
            &q_live,
            &k_live,
            &v_live,
            &q_lens,
            n_head,
            n_kv_head,
            head_dim,
            &rope_cs,
            false,
        )?;
        assert_eq!(out.dims(), &[total_q, n_head, head_dim]);

        // Reference: per sequence, rotate q/k at absolute positions and run
        // plain causal attention over bf16-rounded values.
        let half = head_dim / 2;
        let cs_ref = rope_cs.reshape((rope_cs.dim(0)?, half, 2))?;
        let rot_ref = |x: &Tensor, len: usize| -> Result<Tensor> {
            let cos = cs_ref
                .narrow(0, 0, len)?
                .narrow(2, 0, 1)?
                .squeeze(2)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            let sin = cs_ref
                .narrow(0, 0, len)?
                .narrow(2, 1, 1)?
                .squeeze(2)?
                .to_dtype(DType::BF16)?
                .contiguous()?;
            candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
        };
        let mut row_start = 0usize;
        for &len in &q_lens {
            let take = |x: &Tensor| -> Result<Tensor> {
                x.narrow(0, row_start, len)?
                    .transpose(0, 1)?
                    .unsqueeze(0)?
                    .contiguous()
            };
            let q_r = rot_ref(&take(&q)?, len)?.to_dtype(DType::F32)?;
            let k_r = rot_ref(&take(&k)?, len)?.to_dtype(DType::F32)?;
            let v_s = take(&v)?.to_dtype(DType::F32)?;
            let k_rep = repeat_kv(k_r, n_rep)?;
            let v_rep = repeat_kv(v_s, n_rep)?;
            let expect = standard_attention_prefill(&q_r, &k_rep, &v_rep, head_dim)?;
            let got = out
                .narrow(0, row_start, len)?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .to_dtype(DType::F32)?;
            let diff = got
                .sub(&expect)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_vec0::<f32>()?;
            assert!(
                diff < 5e-2,
                "seq len {len}: fallback diverged from reference (max abs {diff})"
            );
            row_start += len;
        }

        // Interop invariant: the arena holds the UNROTATED keys.
        let (k_stored, _v_stored) = cache_refs[1].chunked_read_kv(0, q_lens[1])?;
        let k_expect = k
            .narrow(0, q_lens[0], q_lens[1])?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;
        let sd = k_stored
            .to_dtype(DType::F32)?
            .sub(&k_expect.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        assert!(
            sd < 1e-6,
            "arena keys must be stored unrotated (max abs diff {sd})"
        );
        Ok(())
    }
}
