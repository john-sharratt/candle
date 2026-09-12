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
use crate::models::lora::{adapt, LayerLora};
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
    /// This layer's LoRA pairs, resolved once by the wave loop. Default — all
    /// `None` — is the unadapted layer, which is the overwhelmingly common case
    /// and takes the same code path.
    pub lora: LayerLora<'a>,
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

/// The activation an adapter's `A` matmul reads — **the operand the base
/// projection multiplies, whichever form it is in.**
///
/// On the float path this shares the tensor (a `LiveTensor` clone is an `Arc`
/// bump, not a copy). On the int8 path the post-norm float does not exist —
/// `attention_norm` fuses RMSNorm and quantize into one kernel — so this
/// reconstructs it from the q8a128 blocks.
///
/// **The int8 round trip is the right input, not a compromise.** What comes
/// back is the activation the base projection actually multiplies: the same
/// int8 codes, times the same per-128 scale. An adapter computed against a float
/// the matmul never sees would be the odd one out. And it costs one kernel per
/// *projection group* — `project_qkv_gated` resolves it once for q, k and v —
/// against the alternative of dropping the whole layer out of int8, which loses
/// the norm/quantize fusion on every projection it has.
///
/// `dtype` is the width the adapter's `A` is resident in, which is the base
/// output's: an adapted layer asks for the pair it already materialised rather
/// than forcing a second copy at some other width. `like` supplies the device,
/// and is the base projection this adapter adds to — the one tensor every call
/// site already has, and by construction on the device the operand lives on.
///
/// Free rather than a method because [`project_qkv_gated`] needs it too — the
/// projection body is shared with qwen4exp, so it cannot reach through `self`.
#[cfg(feature = "cuda")]
pub(crate) fn lora_input<'w>(
    acts: &DynamicActs<'w>,
    like: &LiveTensor<'w>,
    site: &str,
) -> Result<LiveTensor<'w>> {
    match acts {
        DynamicActs::Float(t) => Ok(t.clone()),
        DynamicActs::Int8(op) => {
            let candle::Device::Cuda(dev) = like.device().clone() else {
                candle::bail!("{site}: a q8a128 operand must be on a CUDA device");
            };
            op.dequantize(like.dtype(), &dev)
        }
    }
}

impl BatchedAttentionLayer for Qwen35AttentionLayer<'_> {
    fn n_head(&self) -> usize {
        self.n_head
    }

    /// The layer's numeric mode — **the weight's own, adapted or not.**
    ///
    /// An adapter does not change how this layer computes. Its `A` matmul reads
    /// the same post-norm activation the base projection does, and on the int8
    /// path that activation exists only as q8a128, because `attention_norm`
    /// fuses RMSNorm and quantize into one kernel; [`lora_input`] reconstructs
    /// it from the blocks rather than making the layer produce a float one.
    ///
    /// The alternative — reporting `Off` for an adapted layer — keeps the float
    /// alive but pays for it everywhere: the norm/quantize fusion is lost on
    /// every projection of every adapted layer, and `want_q8` (gated on
    /// `int8mode().is_int8()`) stops the attention context being emitted as
    /// q8a1024 too. That is a numeric-mode change driven by whether an adapter
    /// happens to be attached, which is exactly the coupling this avoids: the
    /// adapted and unadapted paths now run the same kernels, in the same mode,
    /// over the same resident weights.
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
            QuantFfn::Dense(m) => {
                m.forward_dynamic_adapted(&acts, work_dtype, out_dtype, self.lora)
            }
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
            self.lora,
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

    /// The output projection, with its adapter.
    ///
    /// The adapter's input here is the **attention context** — `o_proj`'s
    /// operand, not the layer input — which is why an adapted layer must not
    /// emit that context as q8a1024. `int8mode()` returning `Off` is what stops
    /// it: `want_q8` is gated on the mode, so the context arrives as `Float` and
    /// the gate has already been applied to it.
    #[cfg(feature = "cuda")]
    fn output_projection<'w>(
        &self,
        attn: DynamicActs<'w>,
        out_dtype: DType,
    ) -> Result<LiveTensor<'w>> {
        let base = self
            .o_proj()
            .forward_dynamic(attn.as_dynamic(), out_dtype)?;
        if self.lora.o.is_none() {
            return Ok(base);
        }
        let x = lora_input(&attn, &base, "output_projection")?;
        adapt(self.lora.o, base, &x)
    }
}

/// The gated-attention Q/K/V projection over prepared activations — the whole
/// of [`BatchedAttentionLayer::project_qkv`] for this lineage's attention
/// shape (interleaved `[q | gate]`, per-head Q/K norms, partial-rotary
/// permutation), shared by every model that carries it: the hybrid's own
/// wrapper above, and qwen4exp's (whose block input arrives pre-mixed by the
/// Gated Residual rather than pre-normed, but whose projection is this one
/// exactly — `docs/qwen38_flash_next.md` §12.5).
///
/// `lora` is this layer's adapter pairs, or [`LayerLora::default`] — all `None`
/// — for the unadapted layer, which is the overwhelmingly common case and takes
/// the same code path.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn project_qkv_gated<'w>(
    w: &super::quantized_weights::QuantAttentionWeights,
    rotary: &RotaryLayout,
    n_head: usize,
    n_kv: usize,
    d: usize,
    acts: &DynamicActs<'w>,
    out_dtype: DType,
    lora: LayerLora<'_>,
) -> Result<QkvProjection<'w>> {
    // The adapter reads the same post-norm activation the base projections do,
    // and adds to their raw output — before the q/k norms and before the rotary
    // reordering. That ordering is the definition, not a choice: PEFT trained
    // `B(Ax)` against `q_proj`'s output, which in this architecture is the
    // pre-norm, pre-RoPE `[q | gate]`.
    //
    // The adapter lands on the *split* parts rather than on the stacked group
    // below, which is the same arithmetic: stacking concatenates the three
    // projections on the output axis, so adding each pair's term to its own part
    // after the split is exactly adding them to their own rows before it. Doing
    // it here also means the three adapters stay separate weights, which is what
    // PEFT trained and what `Target::{AttnQ, AttnK, AttnV}` name.
    //
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

    // Resolved once for all three, after the projections so `qg` can name the
    // device and width — on the int8 path this reconstructs the operand from its
    // q8a128 blocks, and doing it per projection would triple that.
    let lora_x = match lora.is_empty() {
        true => None,
        false => Some(lora_input(acts, &qg, "project_qkv")?),
    };

    // The `q` adapter's `B` is `[n_head · 2 · head_dim, r]` — twice the hidden
    // width — because the reference implementation this was trained against also
    // projects the gate through `q_proj`. So it adds to `qg` whole, and the
    // split below divides the adapted result exactly as it divides the base one.
    let (qg, k, v) = match &lora_x {
        Some(x) => (
            adapt(lora.q, qg, x)?,
            adapt(lora.k, k, x)?,
            adapt(lora.v, v, x)?,
        ),
        None => (qg, k, v),
    };

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
