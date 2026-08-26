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
use super::mtp::{MtpHead, MtpInput};
#[cfg(feature = "cuda")]
use super::quantized_moe::Qwen35MoeBlock;
use crate::models::delta_net::{LayerKind, QuantDeltaNetWeights};
#[cfg(feature = "cuda")]
use crate::models::expert_lre::ExpertCache;
use crate::models::host_embedding::HostEmbedding;
use crate::models::quantized_matmul::QMatMul;
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
struct PendingLayer {
    attn_norm: RmsNorm,
    post_attn_norm: RmsNorm,
    mix: QuantLayerMix,
    ffn: PendingFfn,
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

impl QuantLayer {
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
    pub layers: Vec<QuantLayer>,
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
}

struct Loader<'a, R: Read + Seek> {
    content: &'a gguf_file::Content,
    reader: &'a mut R,
    device: Device,
    mode: Int8Mode,
    /// Checkpoint bytes pulled onto the device so far.
    ///
    /// Summed from the tensors themselves rather than bracketed as a driver
    /// delta: the governor's balloon releases into the CUDA pool, the weights
    /// then allocate out of those cached blocks, and driver-used barely moves —
    /// a delta reads near zero for a model that is really there. Qwen3-MoE
    /// learned this twice, at 3936 MiB and at 733 MiB against a true ~990.
    device_bytes: usize,
}

impl<R: Read + Seek> Loader<'_, R> {
    fn raw(&mut self, name: &str) -> Result<QTensor> {
        let t = self.content.tensor(self.reader, name, &self.device)?;
        self.device_bytes += t.storage_size_in_bytes();
        Ok(t)
    }

    /// A projection: quantized, repacked for the numeric mode.
    fn proj(&mut self, name: &str) -> Result<QMatMul> {
        let qt = self.raw(name)?;
        QMatMul::from_qtensor_with_mode(qt, self.mode)
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
        let mode = self.mode;
        QuantFfnWeights::from_weights(
            self.raw(&format!("{p}.ffn_gate.weight"))?,
            self.raw(&format!("{p}.ffn_up.weight"))?,
            self.raw(&format!("{p}.ffn_down.weight"))?,
            mode,
        )
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
pub fn load_quantized_model<R, F>(
    content: &gguf_file::Content,
    reader: &mut R,
    device: &Device,
    mode: Int8Mode,
    host_embed: Option<HostEmbedding>,
    build_experts: F,
) -> Result<QuantModel>
where
    R: Read + Seek,
    F: FnOnce(&gguf_file::Content, &Qwen35Config) -> Result<Option<std::sync::Arc<ExpertCache>>>,
{
    let arch = super::loader::detect_arch(content);
    let cfg = Qwen35Config::from_gguf_metadata(&arch, &content.metadata)?;
    let mut g = Loader {
        content,
        reader,
        device: device.clone(),
        mode,
        device_bytes: 0,
    };

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
    let lm_head = if g.has("output.weight") {
        g.proj("output.weight")?
    } else {
        g.proj("token_embd.weight")?
    };

    let mut pending_layers = Vec::with_capacity(cfg.num_layers);
    // MoE layers are numbered among themselves — the expert cache keys on
    // `(moe_layer_idx, expert)`, not on the trunk layer index, so a stack
    // that mixes dense and MoE layers still indexes the cache densely.
    #[cfg(feature = "cuda")]
    let mut moe_layer_idx_next = 0usize;
    for li in 0..cfg.num_layers {
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
            &mut g,
            &cfg,
            &p,
            mode,
            #[cfg(feature = "cuda")]
            &mut moe_layer_idx_next,
        )?;
        pending_layers.push(PendingLayer {
            attn_norm: g.norm(&format!("{p}.attn_norm.weight"), cfg.rms_norm_eps)?,
            post_attn_norm: g.norm(&format!("{p}.post_attention_norm.weight"), cfg.rms_norm_eps)?,
            mix,
            ffn,
        });
    }

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
    let mtp = if cfg.num_mtp_layers == 0 {
        None
    } else {
        let mi = cfg.num_layers;
        let p = format!("blk.{mi}");
        if cfg.num_mtp_layers != 1 {
            candle::bail!(
                "qwen35: {} nextn_predict_layers — this stack drafts from a single \
                 MTP block (llama.cpp asserts the same for this architecture), and \
                 chaining several is a different recurrence, not more of this one",
                cfg.num_mtp_layers
            );
        }
        for t in ["nextn.enorm", "nextn.hnorm", "nextn.eh_proj"] {
            if !g.has(&format!("{p}.{t}.weight")) {
                candle::bail!(
                    "qwen35: the checkpoint declares nextn_predict_layers = {} but \
                     {p}.{t}.weight is missing — a conversion that kept the metadata \
                     and dropped the tensors",
                    cfg.num_mtp_layers
                );
            }
        }
        // Dedicated embedding / head tensors exist in the format for
        // checkpoints trained with them. These are not (`mtp_use_dedicated_
        // embeddings: false`), and the head sharing the target's `token_embd` +
        // `output.weight` is what makes it cost one block rather than a second
        // model — so a checkpoint that carries them is a different animal and
        // is refused rather than silently drafted with the wrong tables.
        for t in ["nextn.embed_tokens", "nextn.shared_head_head"] {
            if g.has(&format!("{p}.{t}.weight")) {
                candle::bail!(
                    "qwen35: {p}.{t}.weight is present — this checkpoint's MTP head \
                     has its own embedding/LM head, which this drafter does not read \
                     (it shares the target's)"
                );
            }
        }
        // **A routed head is loaded, not refused.** On a MoE checkpoint the
        // head is a complete MoE block — its own router, 256 experts and shared
        // expert, at the trunk's geometry — so it takes the same `pending_ffn`
        // the trunk layers take and the same deferred resolve, and its experts
        // land at the `moe_layer_idx` straight after the last trunk MoE layer.
        // `expert_host_refs` walks the checkpoint in that same order, so the
        // cache holds the head's experts at exactly the index the router asks
        // for. (This used to bail: the cache was sized and indexed over the
        // trunk only, and drafting would have read experts nothing staged.)
        let block = PendingLayer {
            // Built by the SAME helpers a trunk layer is: `Loader::attention`
            // and `pending_ffn` already read exactly these tensor names,
            // because the head is `blk.{num_layers}` in the checkpoint and is
            // named like every other block. Nothing here is head-specific,
            // which is the point — it runs the production path unchanged.
            attn_norm: g.norm(&format!("{p}.attn_norm.weight"), cfg.rms_norm_eps)?,
            post_attn_norm: g.norm(&format!("{p}.post_attention_norm.weight"), cfg.rms_norm_eps)?,
            mix: QuantLayerMix::Attention(g.attention(&p, cfg.rms_norm_eps)?),
            ffn: pending_ffn(
                &mut g,
                &cfg,
                &p,
                mode,
                #[cfg(feature = "cuda")]
                &mut moe_layer_idx_next,
            )?,
        };
        let head = PendingMtp {
            block,
            input: MtpInput {
                enorm: g.norm(&format!("{p}.nextn.enorm.weight"), cfg.rms_norm_eps)?,
                hnorm: g.norm(&format!("{p}.nextn.hnorm.weight"), cfg.rms_norm_eps)?,
                eh_proj: g.proj(&format!("{p}.nextn.eh_proj.weight"))?,
            },
            // `shared_head_norm` is the head's own final norm. llama.cpp falls
            // back to the model's `output_norm` when it is absent; these
            // checkpoints ship it, and quietly substituting a different norm
            // would be a drafter that is subtly wrong in a way only acceptance
            // could show — so it is required here.
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
        Some(head)
    };

    let experts = build_experts(content, &cfg)?;
    let (n_used, norm_topk) = cfg
        .moe
        .map_or((0, false), |m| (m.n_experts_used, m.norm_topk_prob));
    let layers = pending_layers
        .into_iter()
        .map(|l| l.resolve(experts.as_ref(), n_used, norm_topk))
        .collect::<Result<Vec<_>>>()?;
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
    })
}
