//! Production weights for the hybrid stack: quantized projections held as
//! [`QMatMul`], everything else as F32 device tensors.
//!
//! The split is by *role*, not by size:
//!
//! * a **projection** is a matmul and goes through the quantized kernels;
//! * a **residual-stream norm** ([`RmsNorm`]) is a fused producer — it emits
//!   q8a128 straight into the matmul that follows, so it must be the shared
//!   type that can re-materialise itself in the session's activation dtype,
//!   not a bare tensor;
//! * an **elementwise constant** read by hand-written tensor algebra — the
//!   conv kernel, `ssm_a`, the `dt` bias, and the DeltaNet per-head norm
//!   gain — is dequantized once at load and kept in F32, because the
//!   recurrence accumulates and must not drift (design doc §8, and the
//!   checkpoint itself declares `mamba_ssm_dtype: float32`).
//!
//! `ssm_norm` is the one that looks like it belongs in the second group and
//! does not: it is a *per-head* gain applied inside the mixer's own algebra
//! (`delta_net_mix`), never a producer feeding a matmul, so it stays a plain
//! F32 tensor.
//!
//! This module only *loads*. The forward lives alongside the reference it is
//! validated against, so that the load-bearing algebra is written once.

use candle::quantized::{gguf_file, GgmlDType, Int8Mode, QTensor};
use candle::{Device, Result, Tensor};
use std::io::{Read, Seek};
#[cfg(feature = "cuda")]
use std::sync::Arc;

use super::config::Qwen35Config;
use super::embedding::EmbeddingTable;
use super::layer_store::LayerStore;
use super::mtp::{MtpHead, MtpInput};
#[cfg(feature = "cuda")]
use super::quantized_moe::Qwen35MoeBlock;
use crate::models::delta_net::{LayerKind, QuantDeltaNetWeights};
use crate::models::dense_span;
#[cfg(feature = "cuda")]
use crate::models::expert_lre::ExpertCache;
use crate::models::host_embedding::HostEmbedding;
use crate::models::layer_stream::LayerTensor;
use crate::models::quantized_matmul::{QMatMul, WeightResidency};
use crate::models::quantized_mlp::QuantizedMlp;
#[cfg(feature = "cuda")]
use crate::models::quantized_qwen3_moe::SparseMoeBlock;
use crate::quantized_nn::RmsNorm;

/// A full-attention layer's production weights.
pub struct QuantAttentionWeights {
    /// `[2·head_dim·n_head, hidden]` — interleaved `[q|gate]` per head.
    pub wq: QMatMul,
    pub wk: QMatMul,
    pub wv: QMatMul,
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

/// A layer's FFN before the expert cache exists.
///
/// The cache is sized from a live measurement of the span the **dense** weights
/// left behind, so it cannot be built until every dense tensor is resident —
/// which means the MoE layers must be loadable without it. Everything a routed
/// layer owns other than the experts themselves is loaded here; the cache is
/// grafted on afterwards by [`PendingLayer::resolve`].
enum PendingFfn {
    Dense(QuantFfnWeights),
    #[cfg(feature = "cuda")]
    Moe {
        gate: QMatMul,
        shared: QuantFfnWeights,
        shared_gate: QMatMul,
        /// Position among the MoE layers, which is what the cache keys on — a
        /// stack that mixes dense and routed layers still indexes it densely.
        moe_layer_idx: usize,
    },
}

/// A layer with everything but its experts.
/// `pub(crate)` for the layer-streaming pack build, which loads a layer and
/// resolves it with no expert cache — see [`PendingLayer::resolve_dense`].
pub(crate) struct PendingLayer {
    attn_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    mix: QuantLayerMix,
    ffn: PendingFfn,
}

/// One trunk layer, read from the checkpoint.
///
/// Extracted from [`load_quantized_model`]'s loop rather than inlined in it,
/// because a layer's tensors are also read **one at a time and then dropped**
/// by the layer-streaming pack build (`docs/qwen38_layer_streaming.md` §12.2):
/// it repacks each layer, writes the record, and lets the layer go, so its peak
/// is one layer rather than the whole model. Two transcriptions of a layer's
/// tensor names would be two chances for the pack to describe a layer the
/// trunk loop does not actually build.
///
/// `moe_layer_idx_next` is threaded through because the expert cache keys on a
/// dense MoE-layer index rather than the trunk index, and the counter has to
/// advance in trunk order however this is called.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(crate) fn load_layer<R: Read + Seek>(
    g: &mut Loader<'_, R>,
    cfg: &Qwen35Config,
    li: usize,
    mode: Int8Mode,
    #[cfg(feature = "cuda")] moe_layer_idx_next: &mut usize,
) -> Result<PendingLayer> {
    let p = format!("blk.{li}");
    let mix = match cfg.layer_kinds[li] {
        LayerKind::Attention => QuantLayerMix::Attention(g.attention(&p, cfg.rms_norm_eps)?),
        LayerKind::DeltaNet => QuantLayerMix::DeltaNet(QuantDeltaNetWeights {
            wqkv: g.proj(&format!("{p}.attn_qkv.weight"))?,
            wz: g.proj(&format!("{p}.attn_gate.weight"))?,
            w_beta: g.proj(&format!("{p}.ssm_beta.weight"))?,
            w_alpha: g.proj(&format!("{p}.ssm_alpha.weight"))?,
            w_out: g.proj(&format!("{p}.ssm_out.weight"))?,
            dt_bias: g.f32(&format!("{p}.ssm_dt.bias"))?,
            a: g.f32(&format!("{p}.ssm_a"))?,
            conv: g.f32(&format!("{p}.ssm_conv1d.weight"))?,
            norm: g.f32(&format!("{p}.ssm_norm.weight"))?,
        }),
    };
    let ffn = pending_ffn(
        g,
        cfg,
        &p,
        mode,
        #[cfg(feature = "cuda")]
        moe_layer_idx_next,
    )?;
    Ok(PendingLayer {
        attn_norm: g.norm(&format!("{p}.attn_norm.weight"), cfg.rms_norm_eps)?,
        post_attn_norm: g.norm(&format!("{p}.post_attention_norm.weight"), cfg.rms_norm_eps)?,
        mix,
        ffn,
    })
}

/// The NextN / MTP draft head, read from whichever GGUF holds it.
///
/// Extracted from [`load_quantized_model`] for the same reason [`load_layer`]
/// was, and then for one more: the head is not always in the checkpoint that
/// holds the trunk. Unsloth embeds it, ggml-org ships it as a sidecar file, and
/// the two carry **identical tensor names** — so the only difference is which
/// reader they come from, and a generic function is what lets one transcription
/// of those names serve both. The alternative, a loader type unified across the
/// two files, does not exist: their readers are different types.
///
/// `mtp_layers` is the head count as *the head's own source* declares it, which
/// on the sidecar convention is the only place it appears at all.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
fn load_mtp_head<R: Read + Seek>(
    g: &mut Loader<'_, R>,
    cfg: &Qwen35Config,
    mtp_layers: usize,
    mode: Int8Mode,
    #[cfg(feature = "cuda")] moe_layer_idx_next: &mut usize,
) -> Result<PendingMtp> {
    let mi = cfg.num_layers;
    let p = format!("blk.{mi}");
    if mtp_layers != 1 {
        candle::bail!(
            "qwen35: {mtp_layers} nextn_predict_layers — this stack drafts from a \
             single MTP block (llama.cpp asserts the same for this architecture), \
             and chaining several is a different recurrence, not more of this one"
        );
    }
    for t in ["nextn.enorm", "nextn.hnorm", "nextn.eh_proj"] {
        if !g.has(&format!("{p}.{t}.weight")) {
            candle::bail!(
                "qwen35: the head's source declares nextn_predict_layers = \
                 {mtp_layers} but {p}.{t}.weight is missing — a conversion that \
                 kept the metadata and dropped the tensors"
            );
        }
    }
    // Dedicated embedding / head tensors exist in the format for checkpoints
    // trained with them. These are not (`mtp_use_dedicated_embeddings: false`),
    // and the head sharing the target's `token_embd` + `output.weight` is what
    // makes it cost one block rather than a second model — so a checkpoint that
    // carries them is a different animal and is refused rather than silently
    // drafted with the wrong tables.
    //
    // A *sidecar* carries `token_embd`/`output` of its own, because it has to be
    // a well-formed GGUF on its own terms. Those are not these tensors and are
    // simply never read: the head is built from `blk.{mi}.*` and takes the
    // target's tables, which is the whole point of a NextN head.
    for t in ["nextn.embed_tokens", "nextn.shared_head_head"] {
        if g.has(&format!("{p}.{t}.weight")) {
            candle::bail!(
                "qwen35: {p}.{t}.weight is present — this checkpoint's MTP head \
                 has its own embedding/LM head, which this drafter does not read \
                 (it shares the target's)"
            );
        }
    }
    // **A routed head is loaded, not refused.** On a MoE checkpoint the head is
    // a complete MoE block — its own router, 256 experts and shared expert, at
    // the trunk's geometry — so it takes the same `pending_ffn` the trunk layers
    // take and the same deferred resolve, and its experts land at the
    // `moe_layer_idx` straight after the last trunk MoE layer.
    // `expert_host_refs` walks the checkpoint in that same order, so the cache
    // holds the head's experts at exactly the index the router asks for.
    let block = PendingLayer {
        // Built by the SAME helpers a trunk layer is: `Loader::attention` and
        // `pending_ffn` already read exactly these tensor names, because the
        // head is `blk.{num_layers}` and is named like every other block.
        attn_norm: g.norm(&format!("{p}.attn_norm.weight"), cfg.rms_norm_eps)?,
        post_attn_norm: g.norm(&format!("{p}.post_attention_norm.weight"), cfg.rms_norm_eps)?,
        mix: QuantLayerMix::Attention(g.attention(&p, cfg.rms_norm_eps)?),
        ffn: pending_ffn(
            g,
            cfg,
            &p,
            mode,
            #[cfg(feature = "cuda")]
            moe_layer_idx_next,
        )?,
    };
    let head = PendingMtp {
        block,
        input: MtpInput {
            enorm: g.norm(&format!("{p}.nextn.enorm.weight"), cfg.rms_norm_eps)?,
            hnorm: g.norm(&format!("{p}.nextn.hnorm.weight"), cfg.rms_norm_eps)?,
            eh_proj: g.proj(&format!("{p}.nextn.eh_proj.weight"))?,
        },
        // `shared_head_norm` is the head's own final norm. llama.cpp falls back
        // to the model's `output_norm` when it is absent; these checkpoints ship
        // it, and quietly substituting a different norm would be a drafter that
        // is subtly wrong in a way only acceptance could show — so it is
        // required here.
        head_norm: g.norm(
            &format!("{p}.nextn.shared_head_norm.weight"),
            cfg.rms_norm_eps,
        )?,
        layer_index: mi,
    };
    tracing::info!(
        target: "candle_transformers::qwen35",
        block = mi,
        routed = matches!(head.block.ffn, PendingFfn::Dense(_)).then_some(false).unwrap_or(true),
        "qwen35 MTP draft head loaded"
    );
    Ok(head)
}

/// The part of a layer that a streaming slot does **not** hold.
///
/// Reads only the norms, the DeltaNet F32 constants and the two sub-tile gates
/// — no projection, no repack — so it costs a few hundred KB per layer against
/// the ~240 MB [`load_layer`] moves. That difference is the reason this exists:
/// the layer cache needs a residue for *every* layer at every startup, and
/// getting them by loading every layer would cost a full repack pass on a model
/// that by definition does not fit.
///
/// The two gates are projections but not streamable ones: `[n_v_heads, hidden]`
/// is 48 rows on the 27B, which packs into the storage chunk and is still
/// rejected by the matmul tile, so they have no KO twin and stay resident with
/// the rest of the residue.
pub fn load_residue<R: Read + Seek>(
    g: &mut Loader<'_, R>,
    cfg: &Qwen35Config,
    li: usize,
) -> Result<ResidentResidue> {
    let p = format!("blk.{li}");
    let eps = cfg.rms_norm_eps;
    let mix = match cfg.layer_kinds[li] {
        LayerKind::DeltaNet => ResidueMix::DeltaNet(DeltaNetResidue {
            w_beta: g.proj(&format!("{p}.ssm_beta.weight"))?,
            w_alpha: g.proj(&format!("{p}.ssm_alpha.weight"))?,
            dt_bias: g.f32(&format!("{p}.ssm_dt.bias"))?,
            a: g.f32(&format!("{p}.ssm_a"))?,
            conv: g.f32(&format!("{p}.ssm_conv1d.weight"))?,
            norm: g.f32(&format!("{p}.ssm_norm.weight"))?,
        }),
        LayerKind::Attention => ResidueMix::Attention(AttentionResidue {
            q_norm: g.norm(&format!("{p}.attn_q_norm.weight"), eps)?,
            k_norm: g.norm(&format!("{p}.attn_k_norm.weight"), eps)?,
        }),
    };
    Ok(ResidentResidue {
        attn_norm: g.norm(&format!("{p}.attn_norm.weight"), eps)?,
        post_attn_norm: g.norm(&format!("{p}.post_attention_norm.weight"), eps)?,
        mix,
    })
}

/// The FFN half of a block at tensor prefix `p`, routed or dense.
///
/// **Shared by the trunk loop and the MTP draft head**, which is the point: the
/// head is `blk.{num_layers}` with a trunk block's tensor set, and on a routed
/// checkpoint that includes a full 256-expert FFN with its own router and
/// shared expert. Two transcriptions of this would be two chances for the
/// head's experts to be indexed differently from the trunk's, and the expert
/// cache keys on a dense `moe_layer_idx` that both sides have to agree on.
///
/// `moe_layer_idx_next` is that counter, threaded through so the head — built
/// after the trunk loop — takes the index straight after the last trunk MoE
/// layer. `expert_host_refs` enumerates the checkpoint in the same order (trunk
/// layers, then the head), so the two agree by construction rather than by
/// coincidence.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
fn pending_ffn<R: Read + Seek>(
    g: &mut Loader<'_, R>,
    cfg: &Qwen35Config,
    p: &str,
    mode: Int8Mode,
    #[cfg(feature = "cuda")] moe_layer_idx_next: &mut usize,
) -> Result<PendingFfn> {
    if !g.has(&format!("{p}.ffn_gate_inp.weight")) {
        return Ok(PendingFfn::Dense(g.dense_ffn(p)?));
    }
    #[cfg(not(feature = "cuda"))]
    {
        candle::bail!("{p} is a MoE layer — the expert cache is a CUDA-only path");
    }
    #[cfg(feature = "cuda")]
    {
        // The shared gate is stored as a `[hidden]` vector; the matmul
        // that consumes it wants a `[1, hidden]` row.
        //
        // **Padded to a full KO tile.** This projection has exactly one
        // output — a scalar gate per token — and the int8 KO layout
        // tiles output rows in groups of 32, so a 1-row weight cannot
        // be represented (`repack_ko` refuses it). Without a KO twin
        // the weight stays float, and a float weight fed the q8a128
        // activations the fused norms emit has no kernel at all: the
        // model loads and then dies on its first MoE layer.
        //
        // So the row is padded with zeros to 32 and the consumer takes
        // output 0 back (`shared_expert_contribution`). The padding is
        // unconditional rather than mode-dependent, so there is one
        // shape here regardless of numeric path; it costs a
        // `[32, hidden]` matmul in place of `[1, hidden]`, which is
        // three orders of magnitude below the expert chain beside it.
        let gate_row = g
            .raw(&format!("{p}.ffn_gate_inp_shexp.weight"))?
            .dequantize(&g.device)?
            .reshape((1, cfg.hidden_size))?;
        let pad = Tensor::zeros(
            (SHARED_GATE_TILE - 1, cfg.hidden_size),
            gate_row.dtype(),
            &g.device,
        )?;
        let gate_vec = Tensor::cat(&[&gate_row, &pad], 0)?;
        let pending = PendingFfn::Moe {
            gate: g.proj(&format!("{p}.ffn_gate_inp.weight"))?,
            shared: QuantFfnWeights::from_weights(
                g.raw(&format!("{p}.ffn_gate_shexp.weight"))?,
                g.raw(&format!("{p}.ffn_up_shexp.weight"))?,
                g.raw(&format!("{p}.ffn_down_shexp.weight"))?,
                mode,
            )?,
            shared_gate: QMatMul::from_qtensor_with_mode(
                QTensor::quantize(&gate_vec, GgmlDType::F32)?,
                mode,
            )?,
            moe_layer_idx: *moe_layer_idx_next,
        };
        *moe_layer_idx_next += 1;
        Ok(pending)
    }
}

/// The draft head with everything but its experts.
///
/// Deferred for exactly the reason a routed trunk layer is: the expert cache is
/// sized from a measurement of the span the dense weights leave behind, so it
/// cannot exist until every dense tensor is resident. On a routed checkpoint
/// the head's FFN is a 256-expert block like any other, so it waits with them.
struct PendingMtp {
    input: MtpInput,
    block: PendingLayer,
    head_norm: RmsNorm,
    layer_index: usize,
}

impl PendingMtp {
    fn resolve(
        self,
        experts: Option<&std::sync::Arc<ExpertCache>>,
        n_experts_used: usize,
        norm_topk_prob: bool,
    ) -> Result<MtpHead> {
        Ok(MtpHead {
            input: self.input,
            block: self
                .block
                .resolve(experts, n_experts_used, norm_topk_prob)?,
            head_norm: self.head_norm,
            layer_index: self.layer_index,
        })
    }
}

impl PendingLayer {
    /// Finish a layer that has no experts to graft on.
    ///
    /// The dense case of [`Self::resolve`], named so the layer-streaming pack
    /// build does not have to pass an expert cache it knows is absent and two
    /// arguments that mean nothing without one. A routed layer reaching here is
    /// a caller that confused the two subsystems, and it is refused rather than
    /// resolved against a cache of nothing.
    #[cfg(feature = "cuda")]
    pub(crate) fn resolve_dense(self) -> Result<QuantLayer> {
        if matches!(self.ffn, PendingFfn::Moe { .. }) {
            candle::bail!(
                "layer stream: this layer is routed, and a routed layer's bulk is its \
                 experts — it belongs to the expert cache, not the layer cache"
            );
        }
        self.resolve(None, 0, false)
    }

    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    fn resolve(
        self,
        experts: Option<&std::sync::Arc<ExpertCache>>,
        n_experts_used: usize,
        norm_topk_prob: bool,
    ) -> Result<QuantLayer> {
        let ffn = match self.ffn {
            PendingFfn::Dense(m) => QuantFfn::Dense(m),
            #[cfg(feature = "cuda")]
            PendingFfn::Moe {
                gate,
                shared,
                shared_gate,
                moe_layer_idx,
            } => {
                let cache = experts
                    .ok_or_else(|| {
                        candle::Error::Msg(
                            "qwen35: a routed layer resolved with no expert cache — the \
                             span measurement produced none for a checkpoint that has experts"
                                .into(),
                        )
                    })?
                    .clone();
                QuantFfn::Moe(Qwen35MoeBlock {
                    routed: SparseMoeBlock {
                        gate,
                        cache,
                        moe_layer_idx,
                        num_experts_per_tok: n_experts_used,
                        norm_topk_prob,
                    },
                    shared,
                    shared_gate,
                })
            }
        };
        Ok(QuantLayer {
            attn_norm: self.attn_norm,
            post_attn_norm: self.post_attn_norm,
            mix: self.mix,
            ffn,
        })
    }
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
    mix: ResidueMix,
}

#[derive(Debug, Clone)]
enum ResidueMix {
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
                    w_beta: d.w_beta.clone(),
                    w_alpha: d.w_alpha.clone(),
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
            (LayerTensor::Wqkv, QuantLayerMix::DeltaNet(d)) => Ok(&d.wqkv),
            (LayerTensor::Wz, QuantLayerMix::DeltaNet(d)) => Ok(&d.wz),
            (LayerTensor::WOut, QuantLayerMix::DeltaNet(d)) => Ok(&d.w_out),
            (LayerTensor::Wq, QuantLayerMix::Attention(a)) => Ok(&a.wq),
            (LayerTensor::Wk, QuantLayerMix::Attention(a)) => Ok(&a.wk),
            (LayerTensor::Wv, QuantLayerMix::Attention(a)) => Ok(&a.wv),
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

/// The loaded production model.
///
/// `embed` is deliberately **off the card**: a `vocab × hidden` table is 4 GB at
/// the 9B's geometry for one row read per token, the worst VRAM-per-access ratio
/// in the model. Where it lives instead, and why the production path no longer
/// pays a GPU→CPU touch for it, is [`EmbeddingTable`]. The tied LM head is a
/// separate device-side [`QMatMul`] over the same checkpoint tensor, because
/// that one *is* a matmul and reads its whole weight every call.
pub struct QuantModel {
    pub cfg: Qwen35Config,
    /// The token-embedding table, off the card either way — see
    /// [`EmbeddingTable`] for what the two residencies cost.
    pub embed: EmbeddingTable,
    /// The trunk's layers, resident or streamed — see [`LayerStore`].
    ///
    /// Not a `Vec`: on a dense checkpoint the layers are tenants of the weight
    /// zone's slots and only a working set of them exists at any moment, so
    /// there is no vector to index. Everything that wants a layer's *weights*
    /// goes through [`LayerStore::ensure`], and everything that wants only its
    /// norms or the recurrence's constants goes through
    /// [`LayerStore::residue`], which is resident either way.
    pub layers: LayerStore,
    /// The NextN / MTP draft head, when the conversion kept it.
    ///
    /// `None` on a plain GGUF — which is not a property of the architecture but
    /// of the file: Qwen3.5/3.6 are trained with multi-token prediction and
    /// ship the head, and unsloth publishes a parallel `…-MTP-GGUF` repo per
    /// model that keeps it. With `None` the lineage drafts through the
    /// weightless fallback instead. See [`super::mtp`].
    pub mtp: Option<MtpHead>,
    /// The shared expert cache, `None` on a dense checkpoint.
    ///
    /// Also reachable through any MoE layer's router, but held here as well
    /// because the model-level answers the scheduler asks — how much weight
    /// ground can be ceded, what the resident footprint is, the pipeline's
    /// telemetry — are about the cache as a whole and must not depend on the
    /// stack having a MoE layer at a particular index.
    #[cfg(feature = "cuda")]
    pub experts: Option<Arc<ExpertCache>>,
    /// The head's norm — a fused producer feeding `lm_head`, so [`RmsNorm`]
    /// rather than a bare gain, exactly like the per-layer norms.
    pub final_norm: RmsNorm,
    pub lm_head: QMatMul,
    pub device: Device,
    /// VRAM the DENSE tensors hold — everything above except the paged experts.
    ///
    /// Kept because a footprint that is only ever handed to the governor is a
    /// footprint no report can read back. `resident_weight_bytes` is documented
    /// as "fixed base + time-varying resident experts" and could answer only the
    /// second half without this, so the whole-card decomposition subtracted the
    /// experts from themselves and reported the dense half as ZERO — several
    /// GiB of VRAM invisible to every consumer, including the budget that is
    /// supposed to size the reservation around it.
    pub dense_bytes: usize,
}

/// `pub` so the layer-streaming loader can drive the same reader the trunk loop
/// does — one transcription of a tensor's name, not two. Its fields stay
/// private, so the only things a caller can do with one are the reads
/// [`load_layer`] and [`load_residue`] expose.
pub struct Loader<'a, R: Read + Seek> {
    content: &'a gguf_file::Content,
    reader: &'a mut R,
    device: Device,
    mode: Int8Mode,
    /// Where this load's repacked projections live.
    ///
    /// `Span` for a load whose weights stay — the trunk, the MTP head, the
    /// residues. `Pool` for the pack build, which materialises a layer only to
    /// read it back and drop it, and would otherwise claim dense-block ground
    /// for all 64 layers of a model that by definition does not fit.
    residency: WeightResidency,
    /// Checkpoint bytes pulled onto the device so far.
    ///
    /// Summed from the tensors themselves rather than bracketed as a driver
    /// delta: the governor's balloon releases into the CUDA pool, the weights
    /// then allocate out of those cached blocks, and driver-used barely moves —
    /// a delta reads near zero for a model that is really there. Qwen3-MoE
    /// learned this twice, at 3936 MiB and at 733 MiB against a true ~990.
    device_bytes: usize,
    /// Narrow this weight's KO twin to [`NARROW_TWIN`], for as long as it is set.
    ///
    /// Set around the two loads that produce **permanently resident** weights on a card that
    /// streams: the output head and the MTP draft block. Everything else — the trunk's layers —
    /// is a slot tenant whose width is already the ladder's business, so narrowing it here
    /// would be deciding the same thing twice.
    ///
    /// State on the loader rather than an argument because those two loads reach `proj` through
    /// a dozen call sites (`attention`, `dense_ffn`, the mixer), and threading a parameter
    /// through all of them to say one thing about two tensors is how a parameter ends up
    /// ignored at the one site that mattered.
    narrow: Option<GgmlDType>,
    /// Apply [`streaming_twin`] to the trunk's layers, carrying `num_layers`.
    ///
    /// Separate from [`Self::narrow`] because they answer different questions: `narrow` forces
    /// one width over whatever is being loaded right now (the resident weights), while this is
    /// a per-tensor *schedule* over the streamed trunk. `narrow` wins where both apply, so the
    /// draft block keeps its own answer even though its tensors are named like a layer's.
    stream_narrow: Option<usize>,
}

impl<'a, R: Read + Seek> Loader<'a, R> {
    /// A reader over `content`, for a caller outside [`load_quantized_model`].
    ///
    /// The layer-streaming pack build drives the same helpers the trunk loop
    /// does — which is the point, since a second transcription of a tensor's
    /// name is how a pack comes to describe a layer the loader does not build.
    pub(crate) fn new(
        content: &'a gguf_file::Content,
        reader: &'a mut R,
        device: &Device,
        mode: Int8Mode,
        residency: WeightResidency,
    ) -> Self {
        Self {
            content,
            reader,
            device: device.clone(),
            mode,
            residency,
            device_bytes: 0,
            narrow: None,
            stream_narrow: None,
        }
    }

    /// Narrow every projection loaded until this is cleared. See [`Loader::narrow`].
    pub(crate) fn set_narrow(&mut self, narrow: Option<GgmlDType>) {
        self.narrow = narrow;
    }

    /// Apply the streamed trunk's narrowing schedule for a stack of `num_layers`.
    /// See [`Loader::stream_narrow`] and [`streaming_twin`].
    pub(crate) fn set_stream_narrow(&mut self, num_layers: Option<usize>) {
        self.stream_narrow = num_layers;
    }
}

impl<R: Read + Seek> Loader<'_, R> {
    fn raw(&mut self, name: &str) -> Result<QTensor> {
        let t = self.content.tensor(self.reader, name, &self.device)?;
        self.device_bytes += t.storage_size_in_bytes();
        Ok(t)
    }

    /// The twin this tensor should take: the scoped override if one is set, else the streaming
    /// schedule if this load is narrowing, else whatever the mode picks.
    fn twin_for(&self, name: &str) -> Option<GgmlDType> {
        self.narrow
            .or_else(|| self.stream_narrow.and_then(|n| streaming_twin(name, n)))
    }

    /// A projection: quantized, repacked for the numeric mode.
    fn proj(&mut self, name: &str) -> Result<QMatMul> {
        let want = self.twin_for(name);
        let qt = self.raw(name)?;
        match want {
            None => QMatMul::from_qtensor_in(qt, self.mode, self.residency),
            // A source already at or below the narrowing width has nothing to give up, and
            // asking for a *wider* twin than the mode picked is refused downstream — so pass
            // the request only when it can actually shrink this tensor. That is what lets the
            // schedule name one target per role and leave the checkpoint's own carve-outs
            // alone: a block already at or below the target simply keeps what it has.
            Some(n) => {
                let picked = qt.dtype().to_ko(self.mode).ok();
                if picked.is_some_and(|p| n.bits_per_weight() < p.bits_per_weight()) {
                    QMatMul::from_qtensor_narrowed(qt, self.mode, self.residency, n)
                } else {
                    QMatMul::from_qtensor_in(qt, self.mode, self.residency)
                }
            }
        }
    }

    /// An elementwise constant: dequantized to F32 once, at load.
    fn f32(&mut self, name: &str) -> Result<Tensor> {
        self.raw(name)?.dequantize(&self.device)
    }

    /// A residual-stream norm, retained quantized so it can re-materialise
    /// itself in whatever dtype the session's activations arrive in.
    fn norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let qt = self.raw(name)?;
        RmsNorm::from_qtensor(qt, eps)
    }

    fn has(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }

    /// A gated-attention block's projections and Q/K norms, at tensor prefix
    /// `p`. Shared by the trunk's attention layers and the MTP draft head,
    /// which carries exactly the same tensor set.
    fn attention(&mut self, p: &str, eps: f64) -> Result<QuantAttentionWeights> {
        Ok(QuantAttentionWeights {
            wq: self.proj(&format!("{p}.attn_q.weight"))?,
            wk: self.proj(&format!("{p}.attn_k.weight"))?,
            wv: self.proj(&format!("{p}.attn_v.weight"))?,
            wo: self.proj(&format!("{p}.attn_output.weight"))?,
            q_norm: self.norm(&format!("{p}.attn_q_norm.weight"), eps)?,
            k_norm: self.norm(&format!("{p}.attn_k_norm.weight"), eps)?,
        })
    }

    /// A dense gated MLP at tensor prefix `p`.
    fn dense_ffn(&mut self, p: &str) -> Result<QuantFfnWeights> {
        // **One target per projection, each asked for by name.**
        //
        // This used to derive a single width from `ffn_down` and hand it to all
        // three, on the reasoning that gate and up are already at or below it so
        // naming it leaves them untouched. That is a property of some
        // checkpoints, not of the schedule: narrowing is a floor that *shrinks*,
        // so a gate or up whose twin is wider than the down-projection's target
        // — a Q4_K gate at `Int8Mode::Performance` against a `Q3_KO` down — is
        // narrowed too. `streaming_twin` names no target for those roles and
        // `an_unnamed_role_is_untouched` asserts they are untouched, so the
        // loader was implementing a different schedule from the one its tests
        // describe. Asking per tensor is the schedule.
        let narrow = |t: &str| {
            self.narrow
                .or_else(|| self.twin_for(&format!("{p}.{t}.weight")))
        };
        let narrow = (narrow("ffn_gate"), narrow("ffn_up"), narrow("ffn_down"));
        let (mode, residency) = (self.mode, self.residency);
        QuantFfnWeights::from_weights_in(
            self.raw(&format!("{p}.ffn_gate.weight"))?,
            self.raw(&format!("{p}.ffn_up.weight"))?,
            self.raw(&format!("{p}.ffn_down.weight"))?,
            mode,
            residency,
            narrow,
        )
    }
}

/// Trunk blocks held at their checkpoint width at each end of the stack.
///
/// The first and last layers are where a quantization schedule's own carve-outs live — this
/// checkpoint puts `ffn_down` at Q5_K on blocks 0-3 and nowhere else — and they are where the
/// literature puts the outlier ("super") weights. Five covers at least four DeltaNet layers at
/// each end under the lineage's 3:1 interleave.
const NARROW_STREAM_ENDS: usize = 5;

/// The twin a **streamed trunk tensor** takes when the card cannot hold the model.
///
/// `None` leaves the mode's own choice, which is every tensor not named here and every load
/// that is not narrowing. Applied through [`Loader::twin_for`], which only ever *narrows*: a
/// target wider than what the mode picked is ignored, so naming one width per role cannot
/// accidentally inflate a block the checkpoint deliberately kept small.
///
/// # Why these three and not the rest
///
/// The bytes that matter are the KO twin's, not the source quant's — the twin is what crosses
/// PCIe on every forward. Sorted by what that yields on the 27B, the trunk's tensors above
/// `Q3_KO` are `ffn_down` (638 MiB), `attn_qkv` (300), `ssm_out` (450), `attn_q` (60). Two are
/// deliberately absent:
///
/// * **`ssm_out` is not narrowed at all.** It is the DeltaNet output projection — the most
///   sensitive tensor in this architecture, and unrescuable, because it has no preceding norm
///   for a calibrated repack to correct against. The checkpoint's own schedule agrees only
///   partly (front and back at Q8_0, the middle already at Q3_K), which is itself worth
///   knowing: the middle third is already the model's weakest point and this must not deepen
///   it.
/// * **`attn_output` likewise** — an out-projection, same argument, and only 60 MiB.
///
/// What is left cuts **in**-projections and the FFN's down-projection, and leaves the recurrent
/// path's own weights (`ssm_*`) at their checkpoint precision entirely.
pub(crate) fn streaming_twin(name: &str, num_layers: usize) -> Option<GgmlDType> {
    let rest = name.strip_prefix("blk.")?;
    let (idx, role) = rest.split_once('.')?;
    let li: usize = idx.parse().ok()?;
    // The draft block sits at `num_layers` and is resident, not streamed — `narrow_resident`
    // governs it, and reaching it here would apply a schedule meant for the trunk.
    if li >= num_layers {
        return None;
    }
    let interior = li >= NARROW_STREAM_ENDS && li + NARROW_STREAM_ENDS < num_layers;
    match role {
        // Blocks 0-3 carry the checkpoint's Q5_K carve-out; leave it standing.
        "ffn_down.weight" => (li >= 4).then_some(GgmlDType::Q3_KO),
        // One step (Q5_KO → Q4_KO), interior only. In-projections are norm-preceded, which is
        // what makes them the defensible half of the attention weights to cut.
        "attn_qkv.weight" => interior.then_some(GgmlDType::Q4_KO),
        // One step (Q6_KO → Q5_KO). The blocks holding Q6_K are llama.cpp's positional
        // `use_more_bits` bump, not a sensitivity finding — `attn_v` shows the identical
        // pattern — so a single step off it is cheap.
        "attn_q.weight" => Some(GgmlDType::Q5_KO),
        _ => None,
    }
}

/// The twin the two permanently-resident weights take when the card cannot hold the model.
///
/// `Q3_KO` — the narrowest twin with a real 3-bit grid. `Q2_KO` exists and is narrower still,
/// but its four levels floor at `rel_l2 ≈ 0.325`, which is a different thing from a repack.
const NARROW_TWIN: GgmlDType = GgmlDType::Q3_KO;

/// Whether this load should narrow the output head and the draft block to [`NARROW_TWIN`].
///
/// **Both halves of the condition are load-bearing, and neither is a preference.**
///
/// *Dense*, because a routed checkpoint's weight is in its experts, which `expert_lre` already
/// streams — its resident set is small and narrowing the head buys it almost nothing while
/// costing the same precision. Only a dense model pays for residency in layer slots.
///
/// *Tight*, meaning the checkpoint does not fit in the card with room to serve from. When it
/// does fit, every layer is resident, no byte crosses PCIe after load, and narrowing would be
/// giving up precision for memory nobody needs. The comparison is against the GGUF's own size
/// plus the KV side's opening reserve, both known before a tensor is read.
///
/// Returns `None` when either half fails, and `None` is the whole of the "off" path — there is
/// no second code path here, only a twin that is or is not overridden.
pub(crate) fn narrow_resident_twin(
    device: &Device,
    cfg: &Qwen35Config,
    content: &gguf_file::Content,
) -> Option<GgmlDType> {
    if cfg.moe.is_some() {
        return None;
    }
    let Device::Cuda(_) = device else {
        return None;
    };
    let total = candle::quantized::get_vram_info().ok()?.1;
    // Everything the checkpoint's tensors weigh, from the header — the same arithmetic the
    // ladder uses, and free.
    let weights: usize = content
        .tensor_infos
        .values()
        .map(|i| {
            let elems: usize = i.shape.dims().iter().product();
            elems / i.ggml_dtype.block_size() * i.ggml_dtype.type_size()
        })
        .sum();
    // "Tight" is: the weights plus the KV side's opening reserve do not both fit. A model that
    // clears this has slots for every layer and nothing to gain here.
    let needed = weights + candle_nn::kv_cache::INITIAL_KV_RESERVE;
    (needed > total).then_some(NARROW_TWIN)
}

/// The `build_layers` type of a load that keeps every layer resident.
///
/// `None` carries no type of its own, and `load_quantized_model` is generic over
/// the builder, so a caller that does not stream needs *some* concrete type to
/// spell the absence with: `None::<ResidentLayers>`.
pub type ResidentLayers =
    fn(&gguf_file::Content, &Qwen35Config, Arc<Vec<ResidentResidue>>) -> Result<LayerStore>;

/// The `build_experts` type of a load with no expert cache — a dense checkpoint.
///
/// The same absence-needs-a-type problem [`ResidentLayers`] solves, one argument
/// over: `build_experts` is not an `Option`, so a dense load has to spell the
/// no-op, and a bare `|_, _| Ok(None)` per call site is one anonymous type per
/// call site — which is what stops [`LoadInputs::resident`] from naming a single
/// shared value.
pub type NoExperts = fn(&gguf_file::Content, &Qwen35Config) -> Result<Option<Arc<ExpertCache>>>;

/// What a load is given beyond the checkpoint, the reader and the device.
///
/// Four inputs that vary independently, passed as one value. Positionally they
/// were four arguments, and every one of them is an absence at a test call site
/// — `None, None, |_, _| Ok(None), None::<ResidentLayers>` names nothing, so a
/// reader had to count commas against the signature to find out which absence
/// was which. Now those call sites read [`LoadInputs::resident`].
pub struct LoadInputs<'a, F, L> {
    /// The embedding table, already pinned on the host by the caller. `None`
    /// dequantizes it to a plain host tensor instead — see [`EmbeddingTable`].
    pub host_embed: Option<HostEmbedding>,
    /// The NextN draft head's own GGUF and its mapped bytes, for a checkpoint
    /// whose head ships as a sidecar rather than embedded in the trunk file.
    /// `None` reads the head from `content` if it declares one, and leaves the
    /// model without a drafter if it does not.
    pub mtp_src: Option<(&'a gguf_file::Content, &'a [u8])>,
    /// Builds the expert cache at the one point the span means what it says —
    /// see this module's [`load_quantized_model`] header.
    pub build_experts: F,
    /// Builds the layer store, and by being `Some` at all decides that the
    /// trunk loop does not run. See [`load_quantized_model`].
    pub build_layers: Option<L>,
}

impl LoadInputs<'_, NoExperts, ResidentLayers> {
    /// A dense checkpoint loaded whole: no expert cache, no sidecar head, every
    /// layer resident in the dense block.
    ///
    /// What a test harness wants — it dequantizes projections to build a
    /// reference, which needs the weights in hand rather than a slot's view of
    /// them — and never what production wants, since §7 of
    /// `docs/qwen38_layer_streaming.md` streams every dense checkpoint.
    pub fn resident() -> Self {
        Self {
            host_embed: None,
            mtp_src: None,
            build_experts: |_, _| Ok(None),
            build_layers: None,
        }
    }
}

/// Load the production model from a GGUF onto `device`.
///
/// `mode` selects the numeric path for every projection uniformly: an int8
/// mode makes each one a KO twin so the q8a128 activations the fused producers
/// emit always meet a KO weight, and [`Int8Mode::Off`] keeps the standard
/// dequant-weight path.
///
/// # Why the cache arrives as a callback
///
/// The expert cache's capacity is `(span − reserve) / slot_bytes`, and the span
/// is *measured*, not declared (`docs/elastic_vram_partition.md` §4). That
/// measurement is only meaningful once every dense tensor is resident: taken
/// before, it reads the model's own weights as free ground and hands the expert
/// side room the KV side is about to need. Qwen3-MoE hit exactly this and its
/// loader carries the fixed ordering as a comment.
///
/// So the ordering is *structural* here rather than a caller's obligation: this
/// function loads every dense tensor, then calls `build_experts` — at the one
/// point where the span means what it says — then grafts the cache onto the
/// layers that route. A dense checkpoint passes a closure returning `Ok(None)`.
///
/// # `build_layers`, and why it is an `Option` rather than a returned `Option`
///
/// The layer cache has the same ordering constraint for the same reason — its
/// zone is carved from the span's measured remainder — but it also decides
/// something the expert cache does not: **whether the trunk loop runs at all**.
/// A streamed checkpoint must not load its layers into the dense block, because
/// not loading them there is the entire point. That decision has to be made
/// before the loop, and a closure whose answer arrives after it cannot make it.
///
/// So `Some(f)` *is* the decision: the trunk loop is skipped, the per-layer
/// residues are read in its place (a few hundred KB against ~240 MB a layer),
/// and `f` builds the store once the span is closed. `None` loads every layer
/// resident.
///
/// One case overrides it. A **routed** checkpoint keeps its layers resident
/// whatever the caller asked, because its weight is in the experts — which
/// `expert_lre` already streams — and its per-layer projections are a rounding
/// error beside them. That is not the fits/does-not-fit branch
/// `docs/qwen38_layer_streaming.md` §7 forbids: every *dense* checkpoint
/// streams, and "it fits" is the degenerate case where nothing is ever evicted.
pub fn load_quantized_model<R, F, L>(
    content: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    mode: Int8Mode,
    inputs: LoadInputs<'_, F, L>,
) -> Result<QuantModel>
where
    R: Read + Seek,
    F: FnOnce(&gguf_file::Content, &Qwen35Config) -> Result<Option<Arc<ExpertCache>>>,
    L: FnOnce(&gguf_file::Content, &Qwen35Config, Arc<Vec<ResidentResidue>>) -> Result<LayerStore>,
{
    let LoadInputs {
        host_embed,
        mtp_src,
        build_experts,
        build_layers,
    } = inputs;

    let arch = super::loader::detect_arch(content);
    let mut cfg = Qwen35Config::from_gguf_metadata(&arch, &content.metadata)?;

    // **Claim the span before a single weight is read** — see `dense_span`. The
    // reservation used to be created lazily by the expert cache, which runs at
    // the *end* of this function, so it was sized from the VRAM left after the
    // weights had already been taken from the pool and could not contain them.
    dense_span::open_for_load(device, content)?;

    // A routed checkpoint's layers stay resident however the caller asked — see
    // this function's header. Resolved here, where the config is known, so the
    // caller does not have to parse it a second time to find out.
    let build_layers = build_layers.filter(|_| cfg.moe.is_none());
    // Whether this load narrows its two permanently-resident weights. A hardware-and-model
    // fact, derived here where both halves are known, rather than a knob: see
    // [`narrow_resident_twin`].
    let narrow_resident = narrow_resident_twin(device, &cfg, content);

    let mut g = Loader::new(content, reader, device, mode, WeightResidency::Span);

    // The embedding table, off the card under either residency (see
    // [`EmbeddingTable`]). Host-mapped when the caller could pin it, which is
    // strictly better and therefore not a choice made here.
    let embed = match host_embed {
        Some(h) => EmbeddingTable::HostMapped(h),
        None => {
            // The quantized tensor lands on the device to be read and the F32
            // copy on the host, so the device bytes it touched are transient and
            // must come back off the count — charging them would inflate the
            // dense footprint by the largest single tensor in the model (594 MiB
            // on the 35B).
            let embed_q = g.raw("token_embd.weight")?;
            let embed_transient = embed_q.storage_size_in_bytes();
            let table = embed_q.dequantize(&Device::Cpu)?;
            g.device_bytes -= embed_transient.min(g.device_bytes);
            EmbeddingTable::Host(table)
        }
    };
    let final_norm = g.norm("output_norm.weight", cfg.rms_norm_eps)?;
    // **The output head is narrowed on a card that streams.** It is one of only two weights
    // that stay resident whatever the trunk does, it is often published *wider* than the trunk
    // (bartowski ships the 27B's at Q6_K against a Q3_K body), and every byte it holds is a
    // layer slot the zone does not have — which is bandwidth on every forward, not VRAM once.
    // Its own precision is the cheapest to give up: it decides a token from a 248k-way argmax,
    // where the trunk's projections decide the residual stream the whole rest of the forward
    // reads. `narrow_resident` is `None` on a card that holds the model, so nothing changes
    // there. See `Loader::narrow`.
    g.set_narrow(narrow_resident);
    let lm_head = if g.has("output.weight") {
        g.proj("output.weight")?
    } else {
        g.proj("token_embd.weight")?
    };
    g.set_narrow(None);

    // MoE layers are numbered among themselves — the expert cache keys on
    // `(moe_layer_idx, expert)`, not on the trunk layer index, so a stack
    // that mixes dense and MoE layers still indexes the cache densely.
    #[cfg(feature = "cuda")]
    let mut moe_layer_idx_next = 0usize;

    // The trunk, in whichever of its two forms this load takes.
    //
    // Streamed: the projections do NOT come into the dense block — that is the
    // whole point — so only each layer's residue is read, and it is read *here*,
    // inside the load window, so it lands in the span with the rest of the
    // resident model rather than in the pool after the block is frozen.
    let mut pending_layers = Vec::new();
    let mut residues = Vec::new();
    if build_layers.is_some() {
        residues.reserve_exact(cfg.num_layers);
        for li in 0..cfg.num_layers {
            residues.push(load_residue(&mut g, &cfg, li)?);
        }
    } else {
        pending_layers.reserve_exact(cfg.num_layers);
        // The trunk's narrowing schedule applies only here — to the layers themselves. It is
        // set for the loop and cleared after, so nothing loaded outside it (the head, the draft
        // block, the residues) can pick up a per-layer rule meant for a slot tenant.
        g.set_stream_narrow(narrow_resident.map(|_| cfg.num_layers));
        for li in 0..cfg.num_layers {
            pending_layers.push(load_layer(
                &mut g,
                &cfg,
                li,
                mode,
                #[cfg(feature = "cuda")]
                &mut moe_layer_idx_next,
            )?);
        }
        g.set_stream_narrow(None);
    }
    let residues = std::sync::Arc::new(residues);

    // ── Every dense tensor is resident. NOW the span means what it says. ──
    //
    // Record the dense footprint with the governor first. `kv_floor` is
    // `abs + pct × (C − weights)`, so leaving it unrecorded computes the floor
    // against the whole card as though the model were free — inflating the
    // floor and taking the difference straight out of the expert budget, which
    // is the scarcest thing on a tight card. Set, not add: this is the whole
    // dense footprint, and adding it twice drives `C − weights` to zero.
    #[cfg(feature = "cuda")]
    if let candle::DeviceLocation::Cuda { gpu_id } = device.location() {
        if let Some(gov) = candle::vram::get(gpu_id) {
            gov.set_class(candle::vram::AllocClass::Weights, g.device_bytes as u64);
        }
        tracing::info!(
            target: "candle_transformers::qwen35",
            dense_mib = g.device_bytes >> 20,
            layers = cfg.num_layers,
            "qwen35 dense weights resident"
        );
    }
    // ── The NextN / MTP draft head, when the conversion kept it ──
    //
    // `nextn_predict_layers` blocks sit past the trunk, and the trunk loop
    // above stops before them because `cfg.num_layers` already excludes them.
    // The head is structurally a trunk attention layer (see `super::mtp`), so
    // it loads through the same two helpers.
    //
    // Absent on a plain GGUF conversion, which drops the MTP tensors — that is
    // the common case and is not an error; the model simply has no drafter and
    // decodes a token at a time. Present-but-malformed IS an error.
    //
    // # The head may live in a **second file**
    //
    // Two conventions are in the wild and this reads both. Unsloth embeds the
    // NextN tensors in the main GGUF, so `cfg.num_mtp_layers` is non-zero and
    // `blk.{num_layers}.*` resolves against the checkpoint already open. ggml-org
    // splits them into a sidecar — `mtp-Qwen3.8-27B-Q4_0.gguf` beside
    // `Qwen3.8-27B-Q4_K_M.gguf` — whose main file declares **no**
    // `nextn_predict_layers` at all, so a loader that only looked there would
    // decode the 27B a token at a time and report nothing amiss.
    //
    // The sidecar's tensors are named exactly as the embedded convention names
    // them (`blk.64.nextn.eh_proj.weight` and the rest of a trunk attention
    // block), so the only difference is which reader they come from. It also
    // carries its own `token_embd`/`output`, which are ignored: the head shares
    // the target's, which is what makes it cost one block rather than a second
    // model.
    let mut mtp_cursor = mtp_src.map(|(c, bytes)| (c, std::io::Cursor::new(bytes)));
    let (mtp_layers, mut head_loader) = match mtp_cursor {
        Some((content2, ref mut cur)) => {
            let n = Qwen35Config::mtp_layers_in(&content2.metadata, &arch);
            // **The config has to learn the head exists**, and it cannot learn
            // it from the checkpoint: `num_mtp_layers` was parsed from the main
            // file, which under the sidecar convention says nothing about a
            // head. Everything downstream gates on this field —
            // `engine::mtp_kv_layer` returns `None` without it, so the head gets
            // no paged KV layer to write, `session_kv_layers` sizes the cache
            // without one, and `wave_kv_range` never covers the head's pass.
            //
            // The head then drafts into nothing and **every proposal is
            // rejected**: measured on the 27B as `draft budget 2` with
            // `1.00 accepted/step`, which reads as a drafter that is merely bad
            // rather than one with nowhere to write.
            //
            // `num_layers` is *not* adjusted. `from_gguf_metadata` subtracts the
            // head from `block_count` because the embedded convention counts it
            // there; a sidecar's head was never in the main file's count, so the
            // trunk is already the right size.
            // **A named sidecar that declares no head is an error, not a
            // fallback.** The caller went to the trouble of pointing at a file;
            // reading zero out of it means the arch string does not match or the
            // conversion dropped the key, and silently continuing produces
            // exactly the failure the knob exists to remove — a model that loads
            // clean, drafts nothing, and reports `draft budget 0`.
            if n == 0 {
                candle::bail!(
                    "qwen35: the MTP sidecar declares no {arch}.nextn_predict_layers, so it \
                     carries no draft head this loader can use. Either it is not an MTP \
                     conversion or it is for a different architecture."
                );
            }
            cfg.num_mtp_layers = n;
            (
                n,
                Some(Loader::new(
                    content2,
                    cur,
                    device,
                    mode,
                    WeightResidency::Span,
                )),
            )
        }
        None => (cfg.num_mtp_layers, None),
    };
    // The head reads from the sidecar when there is one and from the checkpoint
    // otherwise. Two reader *types*, so this dispatches rather than unifying
    // them — one transcription of the tensor names either way.
    // **The drafter is narrowed for the same reason, and it is the safer of the two.** The MTP
    // block is a full layer's worth of permanently-resident weight earning a 1-in-65 share of
    // the forward, so on a streaming card it is the most expensive residency in the model. And
    // speculation is *lossless*: the trunk verifies every proposal, so a coarser drafter moves
    // the acceptance rate and can never move a token. Precision here buys throughput, not
    // correctness, which is exactly the thing worth trading for layer slots.
    let mtp = match (mtp_layers, head_loader.as_mut()) {
        (0, _) => None,
        (n, Some(h)) => {
            h.set_narrow(narrow_resident);
            let head = load_mtp_head(
                h,
                &cfg,
                n,
                mode,
                #[cfg(feature = "cuda")]
                &mut moe_layer_idx_next,
            )?;
            h.set_narrow(None);
            Some(head)
        }
        (n, None) => {
            g.set_narrow(narrow_resident);
            let head = load_mtp_head(
                &mut g,
                &cfg,
                n,
                mode,
                #[cfg(feature = "cuda")]
                &mut moe_layer_idx_next,
            )?;
            g.set_narrow(None);
            Some(head)
        }
    };

    // **The load phase ends here, headroom included, and it cannot end later.**
    //
    // Every dense tensor is placed; the block's right edge is final. It has to
    // close before `build_experts` — the expert cache sizes its zone from the
    // span's free ground, which is not knowable while the block can still grow.
    //
    // Deferring the *headroom* reclaim past `build_layers` was tried, because
    // that pass repacks a whole layer at a time through `WeightResidency::Pool`
    // and `peak_load_pool_bytes` is what reserves pool room for it. It cannot
    // be: `build_layers` carves the layer zone itself (`carve_zone` before
    // `upload_pinned`), and growing the span moves the right edge every
    // weight-side address is measured from — so `reclaim_load_headroom` refuses
    // once a zone is installed, and would refuse there. The two deadlines
    // genuinely conflict, and the address one wins.
    //
    // What must therefore cover the layer-pack repack is the CUDA pool's own
    // scratch margin, not the headroom. See `DEFAULT_SCRATCH_MARGIN_MB`.
    dense_span::close_load(device)?;

    let experts = build_experts(content, &cfg)?;
    let (n_used, norm_topk) = cfg
        .moe
        .map_or((0, false), |m| (m.n_experts_used, m.norm_topk_prob));
    // The layer store, built at the same point and for the same reason as the
    // expert cache: its zone is carved from the span's *measured* remainder,
    // which only means anything now that the dense block is frozen.
    let layers = match build_layers {
        Some(build) => build(content, &cfg, residues)?,
        None => LayerStore::Resident(
            pending_layers
                .into_iter()
                .map(|l| {
                    l.resolve(experts.as_ref(), n_used, norm_topk)
                        .map(std::sync::Arc::new)
                })
                .collect::<Result<Vec<_>>>()?,
        ),
    };
    // The head resolves against the same cache, after it — see `PendingMtp`.
    let mtp = mtp
        .map(|m| m.resolve(experts.as_ref(), n_used, norm_topk))
        .transpose()?;

    Ok(QuantModel {
        cfg,
        embed,
        layers,
        mtp,
        #[cfg(feature = "cuda")]
        experts,
        final_norm,
        lm_head,
        device: device.clone(),
        dense_bytes: g.device_bytes,
    })
}

#[cfg(test)]
mod narrowing_tests {
    use super::{streaming_twin, NARROW_STREAM_ENDS, NARROW_TWIN};
    use candle::quantized::GgmlDType;

    const N: usize = 64;

    fn twin(li: usize, role: &str) -> Option<GgmlDType> {
        streaming_twin(&format!("blk.{li}.{role}"), N)
    }

    /// The whole schedule, at the four positions where it changes answer.
    ///
    /// Written out rather than derived, because this table *is* the quality decision — a value
    /// that drifts here is a checkpoint quantized differently than it was measured, and nothing
    /// downstream would notice.
    #[test]
    fn the_schedule_is_the_table_it_was_derived_from() {
        // `ffn_down`: the checkpoint's Q5_K carve-out on blocks 0-3 stands; everything after it
        // goes to Q3_KO.
        for li in 0..4 {
            assert_eq!(twin(li, "ffn_down.weight"), None, "block {li}");
        }
        for li in [4, 5, 32, N - 1] {
            assert_eq!(
                twin(li, "ffn_down.weight"),
                Some(GgmlDType::Q3_KO),
                "block {li}"
            );
        }

        // `attn_qkv`: interior only — the first and last `NARROW_STREAM_ENDS` blocks keep theirs.
        for li in [0, NARROW_STREAM_ENDS - 1, N - NARROW_STREAM_ENDS, N - 1] {
            assert_eq!(twin(li, "attn_qkv.weight"), None, "block {li}");
        }
        for li in [NARROW_STREAM_ENDS, 32, N - NARROW_STREAM_ENDS - 1] {
            assert_eq!(
                twin(li, "attn_qkv.weight"),
                Some(GgmlDType::Q4_KO),
                "block {li}"
            );
        }

        // `attn_q`: every block, no carve-out.
        for li in [0, 1, 32, N - 1] {
            assert_eq!(
                twin(li, "attn_q.weight"),
                Some(GgmlDType::Q5_KO),
                "block {li}"
            );
        }
    }

    /// Roles the schedule does not name are left alone — including the ones sitting next to a
    /// role it does name, which is where a match arm typo lands.
    #[test]
    fn an_unnamed_role_is_untouched() {
        for role in [
            "ffn_gate.weight",
            "ffn_up.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ssm_out.weight",
            "attn_gate.weight",
            "attn_q_norm.weight",
        ] {
            assert_eq!(twin(32, role), None, "{role}");
        }
    }

    /// The draft head sits at `blk.{num_layers}` and is resident, not streamed. It is governed by
    /// `narrow_resident_twin`, so a schedule meant for the trunk must not reach it — the two
    /// disagree (the head takes `NARROW_TWIN` on every tensor, the trunk takes it on one role).
    #[test]
    fn the_draft_head_is_not_a_trunk_layer() {
        for role in ["ffn_down.weight", "attn_q.weight", "attn_qkv.weight"] {
            assert_eq!(
                streaming_twin(&format!("blk.{N}.{role}"), N),
                None,
                "{role}"
            );
            assert_eq!(
                streaming_twin(&format!("blk.{}.{role}", N + 3), N),
                None,
                "{role}"
            );
        }
        // The head's own answer is the resident one, and it is a different width.
        assert_eq!(NARROW_TWIN, GgmlDType::Q3_KO);
    }

    /// Tensors outside the `blk.{n}.` namespace — `output.weight`, `token_embd.weight`, the
    /// final norm — are not layers and must not be matched by prefix accident.
    #[test]
    fn a_non_layer_tensor_is_not_scheduled() {
        for name in [
            "output.weight",
            "token_embd.weight",
            "output_norm.weight",
            "blk.attn_q.weight",
            "blkX.4.ffn_down.weight",
        ] {
            assert_eq!(streaming_twin(name, N), None, "{name}");
        }
    }

    /// Every width the schedule names is a KO twin, and narrower than the source it replaces.
    ///
    /// `Loader::proj` and `slot_form` both clamp a request that is not narrower — so a schedule
    /// entry that never shrinks anything would be silently inert rather than an error.
    #[test]
    fn every_scheduled_width_can_actually_shrink_its_source() {
        use candle::quantized::Int8Mode;
        // (role, the source dtype this checkpoint carries there)
        for (role, src) in [
            ("ffn_down.weight", GgmlDType::Q4_K),
            ("attn_qkv.weight", GgmlDType::Q5_K),
            ("attn_q.weight", GgmlDType::Q6_K),
        ] {
            let n = twin(32, role).unwrap_or_else(|| panic!("{role} is scheduled"));
            assert!(n.is_ko(), "{role}: {n:?} is not a KO twin");
            let picked = src.to_ko(Int8Mode::Precision).unwrap();
            assert!(
                n.bits_per_weight() < picked.bits_per_weight(),
                "{role}: {n:?} does not shrink {picked:?}"
            );
        }
    }
}
