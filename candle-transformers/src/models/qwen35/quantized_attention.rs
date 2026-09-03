//! The production gated-attention layer, as the engine's per-layer traits.
//!
//! Two things separate this from a classic attention layer, and both are
//! handled here rather than in the generic machinery:
//!
//! * **The output gate.** `wq` projects `2 · head_dim` per head, interleaved
//!   `[query | gate]`. The gate is neither normed nor roped; it rides out of
//!   [`BatchedAttentionLayer::project_qkv`] in [`QkvProjection::gate`] and
//!   the generic attention block applies `sigmoid(gate) ⊙ context` before the
//!   output projection.
//! * **Partial rotary.** Only `rope_dim` of `head_dim` dims rotate, and the
//!   paged kernels only know full-width RoPE, so Q and K are reordered into
//!   the kernel's pairing by [`RotaryLayout`] — see that module for why this
//!   is exact.

use candle::quantized::cuda::DynamicActs;
use candle::quantized::Int8Mode;
use candle::{DType, LiveTensor, Result, Tensor};

use candle_nn::kv_cache::WaveGeneration;

use super::quantized_weights::{QuantFfn, QuantLayer};
use crate::models::batched_layer::{BatchedAttentionLayer, QkvProjection, WaveRef};
use crate::models::quantized_matmul::QMatMul;
use crate::models::rotary_layout::RotaryLayout;
use crate::models::stacked_proj::split_group;
use crate::models::wave_buffers::wave_root;

/// One full-attention layer of the hybrid stack, bound to the geometry it
/// needs. Holds no weights of its own — it borrows the layer's — so it can
/// be built per wave without copying anything.
pub struct Qwen35AttentionLayer<'a> {
    pub layer: &'a QuantLayer,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
    pub rotary: &'a RotaryLayout,
}

impl Qwen35AttentionLayer<'_> {
    fn attn(&self) -> Result<&super::quantized_weights::QuantAttentionWeights> {
        match &self.layer.mix {
            super::quantized_weights::QuantLayerMix::Attention(a) => Ok(a),
            super::quantized_weights::QuantLayerMix::DeltaNet(_) => candle::bail!(
                "Qwen35AttentionLayer wraps a DeltaNet layer — the wave loop \
                 dispatched on the wrong kind"
            ),
        }
    }
}

impl BatchedAttentionLayer for Qwen35AttentionLayer<'_> {
    fn n_head(&self) -> usize {
        self.n_head
    }

    fn int8mode(&self) -> Int8Mode {
        self.layer.ffn_int8mode()
    }

    #[cfg(feature = "cuda")]
    fn ffn_norm<'w>(
        &self,
        x: &Tensor,
        mode: Int8Mode,
        wave: WaveRef<'w>,
    ) -> Result<DynamicActs<'w>> {
        self.layer
            .post_attn_norm
            .forward_dynamic(x, mode, wave_root(wave))
    }

    #[cfg(feature = "cuda")]
    fn ffn_forward<'w>(
        &self,
        acts: DynamicActs<'w>,
        work_dtype: DType,
        out_dtype: DType,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        match &self.layer.ffn {
            QuantFfn::Dense(m) => m.forward_dynamic(&acts, work_dtype, out_dtype),
            // See the qwen3-MoE arm: the shared+routed combine writes the width
            // its experts ran in, so this path narrows on return.
            QuantFfn::Moe(m) => {
                let mut out = m.forward_dynamic(acts, work_dtype, wave)?;
                out.to_dtype_mut(out_dtype)?;
                Ok(out)
            }
        }
    }

    fn n_kv_head(&self) -> usize {
        self.n_kv_head
    }

    fn head_dim(&self) -> usize {
        self.head_dim
    }

    #[cfg(feature = "cuda")]
    fn attention_norm<'w>(
        &self,
        x: &Tensor,
        mode: Int8Mode,
        wave: WaveRef<'w>,
    ) -> Result<DynamicActs<'w>> {
        self.layer
            .attn_norm
            .forward_dynamic(x, mode, wave_root(wave))
    }

    #[cfg(feature = "cuda")]
    fn project_qkv<'w>(
        &self,
        acts: &DynamicActs<'w>,
        out_dtype: DType,
    ) -> Result<QkvProjection<'w>> {
        project_qkv_gated(
            self.attn()?,
            self.rotary,
            self.n_head,
            self.n_kv_head,
            self.head_dim,
            acts,
            out_dtype,
        )
    }

    fn o_proj(&self) -> &QMatMul {
        // Unwrapping is sound: the wave loop only builds this wrapper for a
        // layer whose kind is Attention, and `attn()` states that contract.
        match &self.layer.mix {
            super::quantized_weights::QuantLayerMix::Attention(a) => &a.wo,
            super::quantized_weights::QuantLayerMix::DeltaNet(_) => {
                unreachable!("Qwen35AttentionLayer over a DeltaNet layer")
            }
        }
    }
}

/// The gated-attention Q/K/V projection over prepared activations — the whole
/// of [`BatchedAttentionLayer::project_qkv`] for this lineage's attention
/// shape (interleaved `[q | gate]`, per-head Q/K norms, partial-rotary
/// permutation), shared by every model that carries it: the hybrid's own
/// wrapper above, and qwen4exp's (whose block input arrives pre-mixed by the
/// Gated Residual rather than pre-normed, but whose projection is this one
/// exactly — `docs/qwen38_flash_next.md` §12.5).
#[cfg(feature = "cuda")]
pub fn project_qkv_gated<'w>(
    w: &super::quantized_weights::QuantAttentionWeights,
    rotary: &RotaryLayout,
    n_head: usize,
    n_kv: usize,
    d: usize,
    acts: &DynamicActs<'w>,
    out_dtype: DType,
) -> Result<QkvProjection<'w>> {
    // `wq` is `[q | gate]` interleaved per head, so the projection is one
    // matmul and the split is a view: dim ordering per token is
    // `[h0_q(d) h0_gate(d) h1_q(d) …]`.
    //
    // One launch per stacked group — three when the loader kept q/k/v apart,
    // one when it stacked them — and the split is a single ragged scatter into
    // one arena bump, not a copy per part.
    let mut outs = Vec::with_capacity(w.wqkv.len());
    for m in &w.wqkv {
        outs.push(m.forward_dynamic(acts.as_dynamic(), out_dtype)?);
    }
    let mut parts = split_group(
        outs,
        &[w.q_rows, w.kv_rows, w.kv_rows],
        "attention q/k/v projections",
    )?
    .into_iter();
    let qg = parts.next().expect("three parts requested");
    let k = parts.next().expect("three parts requested");
    let v = parts.next().expect("three parts requested");

    let lead: Vec<usize> = qg.dims()[..qg.rank() - 1].to_vec();
    let mut q_shape = lead.clone();
    q_shape.extend_from_slice(&[n_head, 2, d]);
    let qg = qg.reshape(q_shape)?;
    let split = qg.rank() - 2;
    let q = qg.narrow(split, 0, 1)?.squeeze(split)?;
    let gate = qg.narrow(split, 1, 1)?.squeeze(split)?;

    // Per-head RMSNorm on Q and K, then the rotary reordering. Norm
    // first: it is elementwise over the head dim, so it commutes with a
    // permutation of that dim only if the gain is permuted too — norming
    // in model order and permuting afterwards keeps the gain and the dims
    // in step without a second permuted copy of the weight.
    let mut k_shape = lead.clone();
    k_shape.extend_from_slice(&[n_kv, d]);
    let k = k.reshape(k_shape)?;
    let q = w.q_norm.forward_live(&q.flatten_to(q.rank() - 2)?)?;
    let k = w.k_norm.forward_live(&k.flatten_to(k.rank() - 2)?)?;
    let q = rotary.permute_last_dim_live(&q)?;
    let k = rotary.permute_last_dim_live(&k)?;

    // Back to the flat `[.., n_head · head_dim]` the caller reshapes from.
    let mut flat_q = lead.clone();
    flat_q.push(n_head * d);
    let mut flat_kv = lead.clone();
    flat_kv.push(n_kv * d);
    let mut flat_gate = lead;
    flat_gate.push(n_head * d);

    Ok(QkvProjection {
        q: q.reshape(flat_q)?,
        k: k.reshape(flat_kv.clone())?,
        v: v.reshape(flat_kv)?,
        gate: Some(gate.reshape(flat_gate)?),
    })
}
