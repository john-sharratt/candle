//! Runs the attention and feed-forward halves of a Qwen3-MoE layer, with the
//! producer-fused paths the batched engine uses.
//!
//! The q, k and v projections run as one segmented int8 launch over their three
//! weights. That is float-identical to three separate matmuls, and it keeps the
//! GPU occupied while the small k and v products run. Under the int8 mode the
//! layer norms are fused into the quantising producer, and the MoE router and
//! experts consume the q8a128 output. A dense MLP stays in float, since it has
//! no int8 grouped path. The MoE combine narrows to the width its experts ran in.
//! A `PendingLayer` carries everything except the expert cache, which is grafted
//! on once the weight span is reserved.

use super::*;

// Attention Weights (copied from quantized_qwen3)

#[derive(Debug, Clone)]
pub(super) struct AttentionWeights {
    pub(super) q_proj: Option<QMatMul>,
    pub(super) k_proj: Option<QMatMul>,
    pub(super) v_proj: Option<QMatMul>,
    pub(super) o_proj: QMatMul,
    pub(super) q_norm: RmsNorm,
    pub(super) k_norm: RmsNorm,
    pub(super) num_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) rotary_emb: Arc<RotaryEmbedding>,
}

impl AttentionWeights {
    /// q/k/v projection over a producer-prepared [`DynamicActs`] (the fused `ln1` output).
    #[cfg(feature = "cuda")]
    pub(super) fn project_qkv<'w>(
        &self,
        acts: &DynamicActs<'w>,
        out_dtype: DType,
    ) -> Result<(LiveTensor<'w>, LiveTensor<'w>, LiveTensor<'w>)> {
        let q_dim = self.num_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let q_proj = self
            .q_proj
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("missing q_proj".into()))?;
        let k_proj = self
            .k_proj
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("missing k_proj".into()))?;
        let v_proj = self
            .v_proj
            .as_ref()
            .ok_or_else(|| candle::Error::Msg("missing v_proj".into()))?;
        match acts {
            // int8: ONE segmented launch over the three KO weights (q/k/v kept separate — no
            // concat). Float-identical to three separate matmuls, with full GPU occupancy so the
            // tiny k/v GEMVs no longer starve. Output [lead.., q_dim+2·kv_dim] → narrow to q/k/v.
            DynamicActs::Int8(op) => {
                let qkv = candle::quantized::QMatMul::qkv_segmented(
                    op,
                    &[q_proj.inner(), k_proj.inner(), v_proj.inner()],
                    out_dtype,
                )?;
                let r = qkv.rank() - 1;
                Ok((
                    qkv.narrow(r, 0, q_dim)?,
                    qkv.narrow(r, q_dim, kv_dim)?,
                    qkv.narrow(r, q_dim + kv_dim, kv_dim)?,
                ))
            }
            // FP (Off): three separate matmuls.
            DynamicActs::Float(_) => Ok((
                q_proj.forward_dynamic(acts.as_dynamic(), out_dtype)?,
                k_proj.forward_dynamic(acts.as_dynamic(), out_dtype)?,
                v_proj.forward_dynamic(acts.as_dynamic(), out_dtype)?,
            )),
        }
    }
}

/// A layer with everything except its expert cache.
pub(super) struct PendingLayer {
    pub(super) self_attn: AttentionWeights,
    pub(super) ffn: PendingFfn,
    pub(super) ln1: RmsNorm,
    pub(super) ln2: RmsNorm,
}

impl PendingLayer {
    /// Graft the expert cache on. `cache` is `None` only for a model with no MoE
    /// layers, where no `PendingFfn::MoE` can exist either.
    pub(super) fn resolve(
        self,
        cache: Option<&Arc<ExpertCache>>,
        n_expert_used: usize,
        norm_topk_prob: bool,
    ) -> Result<LayerWeights> {
        let ffn = match self.ffn {
            PendingFfn::Mlp(mlp) => FeedForward::Mlp(mlp),
            PendingFfn::MoE {
                gate,
                moe_layer_idx,
            } => {
                let cache = cache.ok_or_else(|| {
                    candle::Error::Msg(
                        "a MoE layer was loaded but no expert cache was built".into(),
                    )
                })?;
                FeedForward::MoE(SparseMoeBlock {
                    gate,
                    cache: cache.clone(),
                    moe_layer_idx,
                    num_experts_per_tok: n_expert_used,
                    norm_topk_prob,
                })
            }
        };
        Ok(LayerWeights {
            self_attn: self.self_attn,
            ffn,
            ln1: self.ln1,
            ln2: self.ln2,
        })
    }
}

// Layer Weights

pub struct LayerWeights {
    pub(super) self_attn: AttentionWeights,
    pub(super) ffn: FeedForward,
    pub(super) ln1: RmsNorm,
    pub(super) ln2: RmsNorm,
}

#[cfg(feature = "cuda")]
impl BatchedAttentionLayer for LayerWeights {
    fn n_head(&self) -> usize {
        self.self_attn.num_heads
    }

    fn n_kv_head(&self) -> usize {
        self.self_attn.num_kv_heads
    }

    fn head_dim(&self) -> usize {
        self.self_attn.head_dim
    }

    fn o_proj(&self) -> &QMatMul {
        &self.self_attn.o_proj
    }

    // ── Producer-fused (q8a128) overrides for the batched paged path ──
    // `int8mode` + `output_projection` (B2) come from the trait defaults (o_proj-driven);
    // only the model-specific producers (ln1/ln2 fusion, q/k-norm qkv, MoE ffn) are overridden.

    /// B1 producer: fuse ln1 → q8a128 (int8) or FP rms_norm (Off) in one kernel.
    #[cfg(feature = "cuda")]
    fn attention_norm<'w>(
        &self,
        x: &Tensor,
        mode: Int8Mode,
        wave: WaveRef<'w>,
    ) -> Result<DynamicActs<'w>> {
        self.ln1.forward_dynamic(x, mode, wave_root(wave))
    }

    /// B1 consumer: q/k/v over the fused ln1 activation, then q/k/v RMSNorm + reshapes.
    #[cfg(feature = "cuda")]
    fn project_qkv<'w>(
        &self,
        acts: &DynamicActs<'w>,
        out_dtype: DType,
        // As llama: separate q/k/v weights, no stacked group to split.
        _wave: WaveRef<'w>,
    ) -> Result<QkvProjection<'w>> {
        let (b_sz, seq_len) = match acts {
            DynamicActs::Float(t) => {
                let (b, s, _) = t.dims3()?;
                (b, s)
            }
            DynamicActs::Int8(op) => match op.lead.as_slice() {
                &[b, s] => (b, s),
                other => {
                    candle::bail!("project_qkv: expected [b, seq] lead, got {other:?}")
                }
            },
        };
        let (q, k, v) = self.self_attn.project_qkv(acts, out_dtype)?;

        let n_head = self.self_attn.num_heads;
        let n_kv_head = self.self_attn.num_kv_heads;
        let head_dim = self.self_attn.head_dim;

        // The per-head norm is over the last axis, so it runs on the token-major
        // rows as they are — `[tokens·heads, head_dim]` is the same rows a
        // head-major transpose would visit, and the transpose (with its copies
        // in and out) buys nothing.
        let q = self
            .self_attn
            .q_norm
            .forward_live(&q.reshape((b_sz * seq_len * n_head, head_dim))?)?
            .reshape((b_sz, seq_len, n_head * head_dim))?;
        let k = self
            .self_attn
            .k_norm
            .forward_live(&k.reshape((b_sz * seq_len * n_kv_head, head_dim))?)?
            .reshape((b_sz, seq_len, n_kv_head * head_dim))?;

        Ok(QkvProjection {
            q,
            k,
            v,
            gate: None,
        })
    }

    /// B3: ln2 as a producer epilogue. Only the MoE path emits q8a128 (its router + expert gather
    /// consume it); a dense MLP layer stays FP (it has no int8 grouped path).
    #[cfg(feature = "cuda")]
    fn ffn_norm<'w>(
        &self,
        x: &Tensor,
        mode: Int8Mode,
        wave: WaveRef<'w>,
    ) -> Result<DynamicActs<'w>> {
        match &self.ffn {
            FeedForward::MoE(_) => self.ln2.forward_dynamic(x, mode, wave_root(wave)),
            FeedForward::Mlp(_) => Ok(DynamicActs::Float(self.ln2.forward(x)?)),
        }
    }

    /// B3 consumer: the MoE/MLP over the producer-fused ln2 activation. MoE routes the q8a128 (or
    /// Float) through `forward_dynamic`; a dense MLP runs [`QuantizedMlp::forward_dynamic`], which
    /// runs silu/mul/down in `mlp_dtype` (the F16-overflow stability cast).
    #[cfg(feature = "cuda")]
    fn ffn_residual<'w>(
        &self,
        x: &mut Tensor,
        acts: DynamicActs<'w>,
        work_dtype: DType,
        decode: &DecodeRows,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<()> {
        let out_dtype = x.dtype();
        let h = match &self.ffn {
            FeedForward::MoE(m) => {
                // FP acts get the F16→BF16 stability cast; q8a128 is range-safe (no cast).
                let acts = match acts {
                    DynamicActs::Float(t) => DynamicActs::Float(t.to_dtype(work_dtype)?),
                    int8 => int8,
                };
                // The MoE combine writes the width its experts ran in — the
                // router logits and the device dispatch share that one dtype —
                // so this path narrows on return. Giving the combine its own
                // store width is the same change one level down.
                let mut out = m.forward_dynamic(acts, work_dtype, decode, wave)?;
                out.to_dtype_mut(out_dtype)?;
                out
            }
            FeedForward::Mlp(m) => m.forward_dynamic(&acts, work_dtype, out_dtype)?,
        };
        add_ffn_residual(x, &h)
    }
}
