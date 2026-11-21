//! The metadata and projection types that carry a prefill's shared tables, a
//! glue forward's scatter targets and a gated layer's output gate between stages.
//!
//! The varlen tables for a prefill wave are built once and read by every layer,
//! so each layer does not rebuild them. A reprojection-glue forward carries, for
//! each glue token, the block and offset it scatters into and how far it bridges
//! forward. K, V, and the query and gate beside them, are projected in the width
//! the KV arena holds, so appending to the cache converts nothing, and the gate
//! is applied to the context before the output projection.

use super::*;

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
        Self::new_ragged(offsets, &q_lens, device, None)
    }

    /// Build paged-prefill metadata for a **ragged** batch — each sequence has
    /// its own query length `q_lens[i]`. This is the flash-attention varlen
    /// format: `cu_seqlens_q` is the exclusive prefix sum of `q_lens` (so the
    /// packed Q layout is `[Σ q_lens, n_head, head_dim]`), `q_lens` are the
    /// per-sequence new-token counts, and `kv_lens[i] = offsets[i] + q_lens[i]`
    /// is each sequence's total context length after this prefill.
    /// `ticket`, when present, is the forward-scoped span these three tables
    /// belong to — they are built once per wave and read by every layer, which
    /// is exactly what that span is for. `None` uploads them the ordinary way.
    pub fn new_ragged(
        offsets: &[usize],
        q_lens: &[usize],
        device: &Device,
        ticket: Option<WaveTicket>,
    ) -> Result<Self> {
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
        let cu_seqlens_q = wave_from_vec_ticketed(cu, batch_size + 1, device, ticket)?;
        let q_lens_t = wave_from_vec_ticketed(
            q_lens.iter().map(|&l| l as u32).collect::<Vec<_>>(),
            batch_size,
            device,
            ticket,
        )?;
        let kv_lens = wave_from_vec_ticketed(
            offsets
                .iter()
                .zip(q_lens.iter())
                .map(|(&o, &l)| (o + l) as u32)
                .collect::<Vec<_>>(),
            batch_size,
            device,
            ticket,
        )?;
        Ok(Self {
            cu_seqlens_q,
            q_lens: q_lens_t,
            kv_lens,
            glue: None,
        })
    }
}

// Layer-Level Trait

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
pub(super) fn attention_operand_dtype(caches: &[&mut KvCache], residual: DType) -> DType {
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
pub(super) fn apply_attention_gate<'w>(
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
