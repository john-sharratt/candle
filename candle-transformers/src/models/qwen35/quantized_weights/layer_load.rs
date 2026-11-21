//! Reading a trunk layer and the MTP draft head from the checkpoint.
//!
//! A layer is read in two parts: everything but its experts, which are grafted
//! on once the expert cache exists, since that cache is sized from the span the
//! dense weights leave. The trunk loop and the draft head share one
//! transcription of the tensor names, so the head's experts are indexed the way
//! the trunk's are. A head that declares dedicated embeddings is refused rather
//! than drafted with the wrong tables.

use super::*;

/// A layer's FFN before the expert cache exists.
///
/// The cache is sized from a live measurement of the span the **dense** weights
/// left behind, so it cannot be built until every dense tensor is resident —
/// which means the MoE layers must be loadable without it. Everything a routed
/// layer owns other than the experts themselves is loaded here; the cache is
/// grafted on afterwards by [`PendingLayer::resolve`].
pub(super) enum PendingFfn {
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
    pub(super) attn_norm: RmsNorm,
    pub(super) post_attn_norm: RmsNorm,
    pub(super) mix: QuantLayerMix,
    pub(super) ffn: PendingFfn,
}

/// One trunk layer, read from the checkpoint.
///
/// Extracted from [`load_quantized_model`]'s loop rather than inlined in it,
/// because a layer's tensors are also read **one at a time and then dropped**
/// by the layer-streaming pack build (`docs/archived/qwen38_layer_streaming.md` §12.2):
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
        // **Four weights, not one stacked one, and that is this lineage's
        // constraint rather than an oversight.** `streaming_twin` narrows
        // `attn_qkv` to `Q4_KO` in the interior and leaves the other three
        // alone; a stacked weight takes exactly one target, and a narrowed
        // weight is a KO twin whose lane-major rows cannot be row-appended
        // afterwards. The schedule and the stacking are mutually exclusive
        // here, and the schedule wins — it is what fits this model on a card
        // that cannot hold it.
        LayerKind::DeltaNet => QuantLayerMix::DeltaNet(QuantDeltaNetWeights {
            proj: vec![
                g.proj(&format!("{p}.attn_qkv.weight"))?,
                g.proj(&format!("{p}.attn_gate.weight"))?,
                g.proj(&format!("{p}.ssm_beta.weight"))?,
                g.proj(&format!("{p}.ssm_alpha.weight"))?,
            ],
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
pub(super) fn load_mtp_head<R: Read + Seek>(
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
    // The pack's expert section holds the blocks in that same order
    // (`model_pack::experts::qwen_expert_blocks`), so the cache holds the head's
    // experts at exactly the index the router asks for.
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
/// layer. The pack's expert section holds the blocks in the same order (trunk
/// layers, then the head — `model_pack::experts::qwen_expert_blocks`), so the
/// two agree by construction rather than by coincidence.
#[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
pub(super) fn pending_ffn<R: Read + Seek>(
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
pub(super) struct PendingMtp {
    pub(super) input: MtpInput,
    pub(super) block: PendingLayer,
    pub(super) head_norm: RmsNorm,
    pub(super) layer_index: usize,
}

impl PendingMtp {
    pub(super) fn resolve(
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
    pub(super) fn resolve(
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
