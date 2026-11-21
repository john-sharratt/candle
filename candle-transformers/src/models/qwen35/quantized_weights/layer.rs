//! The weights of one trunk layer, and the residue that stays resident when its
//! projections are streamed.
//!
//! A streamed layer keeps only its norms and the recurrence's constants
//! resident, a few hundred kilobytes against the projections a weight-zone slot
//! holds, and its projections are views over that slot. The DeltaNet constants
//! stay F32 because the recurrence accumulates and must not drift. Whether a
//! layer's FFN is dense or routed is decided by tensor presence, layer by layer.

use super::*;

/// A full-attention layer's production weights.
pub struct QuantAttentionWeights {
    /// The `q`, `k` and `v` projections in that order — `q` itself interleaves
    /// `[q|gate]` per head, so its rows are `2·head_dim·n_head` against
    /// `head_dim·n_kv` for each of the others.
    ///
    /// One weight when the loader could row-concatenate them, three when it
    /// could not — see `QuantDeltaNetWeights::proj` for why that is the
    /// loader's decision and not a fallback.
    pub wqkv: Vec<QMatMul>,
    /// Rows of `q`, and of each of `k`/`v` — the widths
    /// [`crate::models::stacked_proj::split_group`] splits on, since a stacked
    /// weight's shape no longer distinguishes the three.
    pub q_rows: usize,
    pub kv_rows: usize,
    pub wo: QMatMul,
    /// Per-head Q/K norms, folded into the projection by `project_qkv`.
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
}

/// A dense FFN — the shared gated-MLP implementation, which fuses gate+up
/// into one launch where the device and dtypes allow.
pub type QuantFfnWeights = QuantizedMlp;

/// Output rows the int8 KO weight layout tiles together.
///
/// The shared-expert gate is the one projection in this stack narrower than a
/// tile, so it is stored padded to this width and read back at output 0.
pub const SHARED_GATE_TILE: usize = 32;

/// A layer's FFN half: dense on the small variants, sparse on the MoE ones.
///
/// Which one a layer has is decided by tensor presence, per layer — the same
/// rule the reference loader uses, and the reason the 35B (whose every layer
/// is MoE, DeltaNet and attention alike) needs no special case.
pub(crate) enum QuantFfn {
    Dense(QuantFfnWeights),
    #[cfg(feature = "cuda")]
    Moe(Qwen35MoeBlock),
}

pub enum QuantLayerMix {
    DeltaNet(QuantDeltaNetWeights),
    Attention(QuantAttentionWeights),
}

pub struct QuantLayer {
    /// ln1 / ln2 over the residual stream — fused producers, hence [`RmsNorm`].
    pub attn_norm: RmsNorm,
    pub post_attn_norm: RmsNorm,
    pub mix: QuantLayerMix,
    pub(crate) ffn: QuantFfn,
}

/// The DeltaNet parts of a layer that never leave VRAM.
///
/// The F32 constants the recurrence reads (which must not be requantized —
/// see this module's header) and the two sub-tile gates, which are
/// `[n_v_heads, hidden]` and so cannot clear the int8 matmul's `nrows % 32`.
#[derive(Debug, Clone)]
pub struct DeltaNetResidue {
    pub w_beta: QMatMul,
    pub w_alpha: QMatMul,
    pub dt_bias: Tensor,
    pub a: Tensor,
    pub conv: Tensor,
    pub norm: Tensor,
}

/// The attention parts of a layer that never leave VRAM.
#[derive(Debug, Clone)]
pub struct AttentionResidue {
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
}

/// Everything in a layer that a slot does **not** hold.
///
/// Kept once per layer for the life of the process and shared into every
/// assembly by handle: `RmsNorm` is an `Arc<QTensor>` and `Tensor` is a handle,
/// so a clone here is a refcount bump rather than device traffic. It is ~0.1%
/// of a layer — see `layer_stream::descriptor` for why streaming it would buy
/// nothing.
#[derive(Debug, Clone)]
pub struct ResidentResidue {
    pub attn_norm: RmsNorm,
    pub post_attn_norm: RmsNorm,
    pub(super) mix: ResidueMix,
}

#[derive(Debug, Clone)]
pub(super) enum ResidueMix {
    DeltaNet(DeltaNetResidue),
    Attention(AttentionResidue),
}

impl ResidentResidue {
    /// The DeltaNet residue, or an error naming the mismatch.
    pub fn delta_net(&self) -> Result<&DeltaNetResidue> {
        match &self.mix {
            ResidueMix::DeltaNet(d) => Ok(d),
            ResidueMix::Attention(_) => candle::bail!(
                "layer residue: asked for DeltaNet parts of an attention layer — the \
                 image's kind and the residue's disagree"
            ),
        }
    }

    /// Re-materialise every norm weight in the dtype activations will arrive in.
    ///
    /// **Every norm a layer has is in the residue** — the two residual-stream
    /// norms and, on an attention layer, the Q/K norms — so this is the whole of
    /// a layer's dtype change and works identically whether the layer is
    /// resident or a slot tenant. `RmsNorm` holds its materialised weight behind
    /// a shared `Arc<RwLock<_>>`, so a residue and every assembly cloned from it
    /// see the same change.
    ///
    /// Called when a session is created, never inside a wave.
    /// **Two widths, because these norms sit on two different tensors.**
    ///
    /// `attn_norm` and `post_attn_norm` read the RESIDUAL stream, so they are
    /// materialised in the session's activation dtype. `q_norm` and `k_norm`
    /// read Q and K, which are projected in the KV ARENA's dtype because they
    /// become its contents (see `batched_layer::attention_operand_dtype`). The
    /// two coincide for every model whose activations and KV agree, and diverge
    /// for one that computes wider than it stores.
    pub fn set_activation_dtype(
        &self,
        dtype: candle::DType,
        kv_dtype: candle::DType,
    ) -> Result<()> {
        self.attn_norm.maybe_change_dtype(dtype)?;
        self.post_attn_norm.maybe_change_dtype(dtype)?;
        if let ResidueMix::Attention(a) = &self.mix {
            a.q_norm.maybe_change_dtype(kv_dtype)?;
            a.k_norm.maybe_change_dtype(kv_dtype)?;
        }
        Ok(())
    }

    /// The attention residue, or an error naming the mismatch.
    pub fn attention(&self) -> Result<&AttentionResidue> {
        match &self.mix {
            ResidueMix::Attention(a) => Ok(a),
            ResidueMix::DeltaNet(_) => candle::bail!(
                "layer residue: asked for attention parts of a DeltaNet layer — the \
                 image's kind and the residue's disagree"
            ),
        }
    }
}

impl QuantLayer {
    /// Take the residue out of a loaded layer, leaving the streamable
    /// projections behind.
    ///
    /// What the layer cache keeps per layer once its slot-borne weights have
    /// gone into the pack.
    pub fn residue(&self) -> ResidentResidue {
        ResidentResidue {
            attn_norm: self.attn_norm.clone(),
            post_attn_norm: self.post_attn_norm.clone(),
            mix: match &self.mix {
                QuantLayerMix::DeltaNet(d) => ResidueMix::DeltaNet(DeltaNetResidue {
                    // Entries 2 and 3 of the group — this loader never stacks,
                    // so they are the `β`/`α` projections themselves. They stay
                    // resident because they are sub-tile (48 rows does not
                    // clear `nrows % 32`), which is also why they could not join
                    // a stacked weight even if the schedule allowed it.
                    w_beta: d.proj[2].clone(),
                    w_alpha: d.proj[3].clone(),
                    dt_bias: d.dt_bias.clone(),
                    a: d.a.clone(),
                    conv: d.conv.clone(),
                    norm: d.norm.clone(),
                }),
                QuantLayerMix::Attention(a) => ResidueMix::Attention(AttentionResidue {
                    q_norm: a.q_norm.clone(),
                    k_norm: a.k_norm.clone(),
                }),
            },
        }
    }

    /// The projection a streamable role names, borrowed.
    ///
    /// Borrowed and never cloned: a `QMatMul` clone is a device-to-device copy
    /// of the weight, so the pack build reads these in place and copies only
    /// the bytes it is about to write.
    pub fn streamed_projection(&self, role: LayerTensor) -> Result<&QMatMul> {
        let ffn = match &self.ffn {
            QuantFfn::Dense(m) => m,
            #[cfg(feature = "cuda")]
            QuantFfn::Moe(_) => candle::bail!(
                "layer stream: a routed layer's FFN has no streamable projection — its \
                 experts are streamed by the expert cache instead"
            ),
        };
        match (role, &self.mix) {
            // `proj` and `wqkv` are groups, and this lineage's loader never
            // stacks them (its narrowing schedule forbids it), so each entry is
            // one streamable role — in the canonical order the loader pushes,
            // `[qkv, z, β, α]` and `[q, k, v]`.
            (LayerTensor::Wqkv, QuantLayerMix::DeltaNet(d)) => Ok(&d.proj[0]),
            (LayerTensor::Wz, QuantLayerMix::DeltaNet(d)) => Ok(&d.proj[1]),
            (LayerTensor::WOut, QuantLayerMix::DeltaNet(d)) => Ok(&d.w_out),
            (LayerTensor::Wq, QuantLayerMix::Attention(a)) => Ok(&a.wqkv[0]),
            (LayerTensor::Wk, QuantLayerMix::Attention(a)) => Ok(&a.wqkv[1]),
            (LayerTensor::Wv, QuantLayerMix::Attention(a)) => Ok(&a.wqkv[2]),
            (LayerTensor::Wo, QuantLayerMix::Attention(a)) => Ok(&a.wo),
            (LayerTensor::FfnGateUp, _) => ffn.fused_gate_up().ok_or_else(|| {
                candle::Error::Msg(
                    "layer stream: this FFN did not fuse, so it has no gate_up weight".into(),
                )
            }),
            (LayerTensor::FfnGate, _) => ffn.split_gate().ok_or_else(|| {
                candle::Error::Msg("layer stream: this FFN fused, so it has no gate weight".into())
            }),
            (LayerTensor::FfnUp, _) => ffn.split_up().ok_or_else(|| {
                candle::Error::Msg("layer stream: this FFN fused, so it has no up weight".into())
            }),
            (LayerTensor::FfnDown, _) => Ok(ffn.down()),
            (r, _) => {
                candle::bail!("layer stream: {r:?} does not belong to this layer's mixer kind")
            }
        }
    }

    /// Assemble a layer whose projections are views over a weight-zone slot.
    ///
    /// Dense by construction: a streamed layer's FFN is a plain gated MLP, so
    /// there is no routed arm here. A MoE checkpoint does not take this path —
    /// its experts are already streamed by `expert_lre`, and a model is one or
    /// the other.
    pub fn from_streamed(
        attn_norm: RmsNorm,
        post_attn_norm: RmsNorm,
        mix: QuantLayerMix,
        ffn: QuantFfnWeights,
    ) -> Self {
        Self {
            attn_norm,
            post_attn_norm,
            mix,
            ffn: QuantFfn::Dense(ffn),
        }
    }

    /// The numeric mode this layer's FFN projections were repacked for.
    pub(crate) fn ffn_int8mode(&self) -> candle::quantized::Int8Mode {
        match &self.ffn {
            QuantFfn::Dense(m) => m.int8mode(),
            #[cfg(feature = "cuda")]
            QuantFfn::Moe(m) => m.shared.int8mode(),
        }
    }
}
