//! The qwen4exp production engine: GPU-resident trunk + streamed experts,
//! loaded from the model pack built over the prepared engine artifact
//! (`qwen4exp::prepare`, then `model_pack`).
//!
//! The weight split mirrors the oracle's (`docs/qwen38_flash_next.md` §12.8),
//! moved onto the card:
//!
//! - **F32, resident**: the PLE norms and conv, the GDN/attention F32
//!   constants, and each HC module's norm gain — the width the oracle computes
//!   in, a load-time decision, never an in-loop conversion (invariant 1).
//! - **HC projections KO** (`HcWeightsKo`): each module's `down` and `up`,
//!   zero-padded to the int8 tiling and run on the same matmul as every other
//!   dense projection.
//! - **The token embedding stays Q8_0, resident** ([`DeviceEmbedding`]): one
//!   kernel gathers a wave's rows and writes them as F32 straight into the
//!   wide residual.
//! - **KO**: every dense projection — attention q/k/v/o, the GDN projections,
//!   routers, shared experts, the PLE key|value stack, the LM head (Q8_0 →
//!   Q8_KO here).
//!
//! Every KO weight is stored in the pack as the engine runs it
//! (`qwen4exp::pack_build`, named by `trunk_names`): the load copies each one
//! from the host into the dense block, and builds nothing.
//! - **Routed experts stay KO in the `ExpertCache`**: VRAM hot slots over
//!   pinned warm and pad RAM over the pack's expert section, the same device-side expert
//!   forward (`ExpertCache::forward_routed`) DeepSeek-V4 and Qwen3.6-35B run
//!   through. 512 experts is exactly `moe_bucketize`'s `MAX_EXPERTS`.
//! - **The PLE table stays on NVMe** behind the §0.1 row cache
//!   (`loader::open_cached_ple` — the same source the oracle reads, so the
//!   two consume identical records).

use std::path::Path;
use std::sync::Arc;

use candle::quantized::gguf_file::Value;
use candle::quantized::GgmlDType;
use candle::{Device, Result, Tensor};

use super::config::Qwen4ExpConfig;
use super::hyper::ko::HcWeightsKo;
use super::loader::{load_headroom_bytes, offloaded_bytes, open_cached_ple, OUTPUT, TOKEN_EMBD};
use super::model::PleSource;
use super::mtp::{MtpDense, MtpHead};
use super::ple_fused::PleFusedWeights;
use super::qsa::IndexerWeights;
use super::sum_scale::ACT_SUM_SCALE;
use super::trunk_names::{
    attention_stack, block_kind, delta_net_stack, hc_down, hc_up, ple_key_value, shared_gate,
    shared_gate_up,
};
use crate::models::delta_net::{KvLayerMap, QuantDeltaNetWeights};
use crate::models::dense_span;
use crate::models::device_embedding::DeviceEmbedding;
use crate::models::expert_lre::section::open_section;
use crate::models::expert_lre::ExpertCache;
use crate::models::latent_moe::GgufModel;
use crate::models::load_phase::{LoadPhase, LoadProgress};
use crate::models::model_pack::ModelPack;
use crate::models::quantized_matmul::QMatMul;
use crate::models::quantized_mlp::QuantizedMlp;
use crate::models::quantized_qwen3_moe::SparseMoeBlock;
use crate::models::qwen35::expert_loader::build_expert_cache_for;
use crate::models::qwen35::quantized_moe::Qwen35MoeBlock;
use crate::models::qwen35::quantized_weights::QuantAttentionWeights;
use crate::models::rotary_layout::RotaryLayout;
use crate::quantized_nn::RmsNorm;

/// The token-mixing half of a production layer.
pub enum GpuLayerMix {
    DeltaNet(QuantDeltaNetWeights),
    /// The 12 full-attention layers, each with its QSA indexer.
    ///
    /// The indexer scores compressed blocks and the attention reads only the
    /// winners (§3.1). Its selection is the identity at ≤ `indexer_budget +
    /// ratio − 1` = 2051 attended positions (§12.5), so a short context runs
    /// exactly the dense arithmetic — but the index keys are cached at every
    /// depth, because the wave that crosses the budget scores the blocks the
    /// waves below it built.
    Attention {
        w: QuantAttentionWeights,
        indexer: IndexerWeights,
        /// `attention.compress_ratios[li]`: cells per index block, 0 for a
        /// layer the checkpoint declares dense.
        compress_ratio: usize,
    },
}

/// One production decoder layer.
pub struct GpuLayer {
    pub hc_attn: HcWeightsKo,
    pub hc_ffn: HcWeightsKo,
    pub mix: GpuLayerMix,
    /// `pub(crate)` because the routed half ([`Qwen35MoeBlock`]) is the
    /// engine's shared machinery, not part of any public surface.
    pub(crate) moe: Qwen35MoeBlock,
}

/// The resident engine.
pub struct Qwen4ExpGpu {
    pub cfg: Qwen4ExpConfig,
    pub device: Device,
    /// `[vocab, hidden]` in the checkpoint's Q8_0, resident — 644 MiB for the
    /// released 248,320-row table, where the widened BF16 copy it replaces took
    /// 1,212 MiB from the expert zone.
    pub embed: DeviceEmbedding,
    pub layers: Vec<GpuLayer>,
    /// The PLE block's weights, stacked and transposed for the fused launches.
    pub ple_w: PleFusedWeights,
    /// The final hyper-connection mix — the output norm (no inject).
    pub out_hc: HcWeightsKo,
    pub lm_head: QMatMul,
    pub rotary: RotaryLayout,
    pub kv_map: KvLayerMap,
    pub experts: Arc<ExpertCache>,
    /// The routed experts' stored format — the rung the artifact was prepared
    /// at. The KV threshold row is calibrated per format (`kv_factors_for`),
    /// because narrower experts leave the model less margin for K/V error.
    pub expert_format: GgmlDType,
    pub ple_table: Box<dyn PleSource>,
    /// The NextN draft head, when the artifact carries one.
    ///
    /// `None` on a checkpoint with no head — every GGUF of this release's own
    /// lineage — in which case the engine decodes exactly as it did before and
    /// `draft_budget` stays 0.
    pub mtp: Option<MtpHead>,
}

impl Qwen4ExpGpu {
    /// Load the engine from its model pack. Order is load-bearing: every dense
    /// tensor resident first, then the expert cache is sized from a live measurement of what
    /// they left behind.
    pub fn load(pack: &Path, device: &Device) -> Result<Self> {
        Self::load_with_progress(pack, device, None)
    }

    /// [`Self::load`] reporting each phase of the load as it runs: the dense
    /// blocks mounted, then the expert cache's fill.
    pub fn load_with_progress(
        pack_path: &Path,
        device: &Device,
        progress: Option<LoadProgress<'_>>,
    ) -> Result<Self> {
        // The KV span is sized from the governor's balloon-measured capacity;
        // without one it falls back to the small test constant and the expert
        // zone measures a floor-violating handful of slots.
        crate::models::batched_model::ensure_vram_governor(device);
        let pack = ModelPack::open(pack_path)?;
        let int8mode = pack.int8_mode;
        // The pack's GGUF part is an ordinary GGUF, so the tensor reader opens it
        // as one; the sections after it are the expert cache's to read.
        let mut gguf = GgufModel::open(&[pack_path.to_path_buf()])?;
        match gguf.metadata.get("general.architecture") {
            Some(Value::String(a)) if a == "qwen4exp" => {}
            other => candle::bail!(
                "qwen4exp engine: general.architecture is {other:?} — wrong checkpoint"
            ),
        }
        let cfg = Qwen4ExpConfig::from_gguf_metadata(&gguf.metadata)?;
        let eps = cfg.rms_norm_eps;
        let content = &pack.content;
        // Claim the reservation before the first tensor, so every KO weight is
        // carved into its dense block and the span is sized from the whole card
        // rather than from what a lazily-created span found free mid-load. The
        // guard holds the load window open until this loader returns.
        let _load = dense_span::open_for_load_sized(device, load_headroom_bytes(content))?;

        // A weight the model keeps as loaded, for its whole life: placed in the
        // dense block rather than left on the driver's ground
        // (`place_tensor_in_dense`).
        let f32t = |g: &mut GgufModel, name: &str| -> Result<Tensor> {
            dense_span::place_tensor_in_dense(device, g.qtensor(name, device)?.dequantize(device)?)
        };
        // A plain tensor the model keeps on the CUDA pool, where the reads
        // that need it find it: the QSA indexer's weights, the PLE block's
        // norms and conv, every hyper-connection module's gain.
        let f32_source = |g: &mut GgufModel, name: &str| -> Result<Tensor> {
            g.qtensor(name, device)?.dequantize(device)
        };
        // Every int8 weight is stored as the KO twin the engine runs, in the
        // form the pack build made it (`trunk_names`): copied from the host
        // straight into the dense block, so nothing repacks and no source
        // stands on the card. Every weight declares the model's activation sum
        // convention, so every operand quantized for it — by `forward`, a fused
        // producer, or the wave — stores `Σx` the one way
        // (`sum_scale::ACT_SUM_SCALE`).
        let stored = |g: &GgufModel, name: &str| -> Result<QMatMul> {
            let (bytes, info) = g.host_bytes(name)?;
            if !info.ggml_dtype.is_ko() {
                candle::bail!(
                    "qwen4exp engine: {name} is stored as {:?}, not a KO twin — a pack of \
                     another family",
                    info.ggml_dtype
                );
            }
            let qt = dense_span::place_host_in_dense(
                device,
                &bytes,
                info.shape.dims(),
                info.ggml_dtype,
            )?;
            Ok(QMatMul::from_qtensor_view(qt, int8mode)?.with_sum_scale(ACT_SUM_SCALE))
        };
        // A hyper-connection module: its two projections as the pack built
        // them — the inject rows stacked under the gate's, the gate's `1/hc`
        // folded in, both padded to the int8 tiling (`HcWeightsKo`) — and its
        // norm gain as the checkpoint has it.
        let hc = |g: &mut GgufModel, prefix: &str, injects: bool| -> Result<HcWeightsKo> {
            HcWeightsKo::from_stored(
                f32_source(g, &format!("{prefix}_norm.weight"))?,
                stored(g, &hc_down(prefix))?,
                stored(g, &hc_up(prefix))?,
                injects,
            )
        };

        // Left in the checkpoint's format. Dequantizing it here put a 2.5 GiB
        // F32 intermediate on the device to keep a 1.2 GiB BF16 table, for a
        // lookup that reads one row per token; the gather kernel widens only
        // those rows, into the residual's F32.
        // Copied from the host straight into the dense block, so the table
        // never stands on the card a second time beside its dense copy — the
        // largest tensor the load reads, which the load headroom therefore
        // leaves out (`load_headroom_bytes`).
        let embed = {
            let (bytes, info) = gguf.host_bytes(TOKEN_EMBD)?;
            DeviceEmbedding::new(dense_span::place_host_in_dense(
                device,
                &bytes,
                info.shape.dims(),
                info.ggml_dtype,
            )?)?
        };
        // The pack stores the head's twin as `output.weight` whether the
        // checkpoint carries one or projects through its embedding table, as
        // the oracle reads a checkpoint with tied embeddings.
        let lm_head = stored(&gguf, OUTPUT)?;
        let out_hc = hc(&mut gguf, "output_hc", false)?;

        use crate::models::delta_net::LayerKind;
        let mut trunk: Vec<(HcWeightsKo, HcWeightsKo, GpuLayerMix)> =
            Vec::with_capacity(cfg.num_layers);
        let mut pending: Vec<(QMatMul, QuantizedMlp, QMatMul)> = Vec::with_capacity(cfg.num_layers);
        let mut ple_w: Option<PleFusedWeights> = None;
        // **The draft head is a block of the model, so it loads as one.** It
        // sits at `blk.{num_layers}` under the same tensor names every trunk
        // block uses (`qwen35::mtp`: "it is a layer of the model, not a
        // sidecar"), so extending the bound builds it with the identical code
        // — its hyper-connections, its 512-expert MoE, its QSA indexer. The
        // trailing entries are split off into the head below.
        //
        // `num_mtp_layers` is 0 on a checkpoint without a head, which is every
        // GGUF of this release's own lineage, so that case loads exactly as it
        // did before.
        let n_blocks = cfg.num_layers + cfg.num_mtp_layers;
        for li in 0..n_blocks {
            if let Some(report) = progress {
                report(LoadPhase::Weights, li, n_blocks);
            }
            let p = format!("blk.{li}");
            let g = &mut gguf;
            let hc_attn = hc(g, &format!("{p}.hc_attn"), true)?;
            let hc_ffn = hc(g, &format!("{p}.hc_ffn"), true)?;

            // `layer_kinds` and `compress_ratios` describe the TRUNK — the
            // shared config parser truncates both to `num_layers`, which is
            // what makes the head's block index land past their end. The head
            // is full-attention by declaration (`"mtp": {"layer_types":
            // ["full_attention"]}`), and it carries an indexer.
            //
            // Its ratio is the checkpoint's own when the array covers the
            // head's block, and only otherwise the trunk's last attention
            // ratio. Preferring the inference outright discarded a declared
            // ratio: a head declared dense would run under block selection,
            // which raises nothing and keeps proposals fluent — visible only as
            // an accept rate that never justifies the head.
            let is_head = li >= cfg.num_layers;
            let kind = block_kind(&cfg, li);
            let ratio = if is_head {
                cfg.head_compress_ratio.unwrap_or_else(|| {
                    cfg.layer_kinds
                        .iter()
                        .enumerate()
                        .rev()
                        .find(|(_, k)| matches!(k, LayerKind::Attention))
                        .map(|(i, _)| cfg.compress_ratios[i])
                        .unwrap_or(0)
                })
            } else {
                cfg.compress_ratios[li]
            };
            let mix = match kind {
                // The input projections read one activation over one
                // contraction, so the pack stores them row-concatenated into a
                // single weight and the block issues one GEMM launch instead of
                // N. Exact: an output column depends only on its own weight
                // row, so the stacked rows *are* the parts.
                LayerKind::DeltaNet => GpuLayerMix::DeltaNet(QuantDeltaNetWeights {
                    proj: vec![stored(g, &delta_net_stack(&p))?],
                    w_out: stored(g, &format!("{p}.ssm_out.weight"))?,
                    dt_bias: f32t(g, &format!("{p}.ssm_dt.bias"))?,
                    a: f32t(g, &format!("{p}.ssm_a"))?,
                    conv: f32t(g, &format!("{p}.ssm_conv1d.weight"))?,
                    norm: f32t(g, &format!("{p}.ssm_norm.weight"))?,
                }),
                LayerKind::Attention => GpuLayerMix::Attention {
                    w: QuantAttentionWeights {
                        wqkv: vec![stored(g, &attention_stack(&p))?],
                        // `q` interleaves `[q|gate]` per head, hence the 2.
                        q_rows: 2 * cfg.attn_head_dim * cfg.num_attention_heads,
                        kv_rows: cfg.attn_head_dim * cfg.num_kv_heads,
                        wo: stored(g, &format!("{p}.attn_output.weight"))?,
                        q_norm: RmsNorm::from_qtensor(
                            g.qtensor(&format!("{p}.attn_q_norm.weight"), device)?,
                            eps,
                        )?,
                        k_norm: RmsNorm::from_qtensor(
                            g.qtensor(&format!("{p}.attn_k_norm.weight"), device)?,
                            eps,
                        )?,
                    },
                    // F32 at load, like every other constant the selection
                    // path reads: the indexer's arithmetic is small and its
                    // rounding decides ranks at the cut (hot-path invariant 1
                    // — the conversion the design calls for, done once).
                    // On the driver's ground, not the dense block: placed there
                    // (`place_tensor_in_dense`), the selection read them wrong
                    // and a 4K prefill's logits came back non-finite
                    // (`strata_bench_single_session`, 2026-11-10), with the
                    // SSM weights beside them placed and reading right.
                    indexer: IndexerWeights {
                        q_proj: f32_source(g, &format!("{p}.indexer.q_proj.weight"))?,
                        k_proj: f32_source(g, &format!("{p}.indexer.k_proj.weight"))?,
                        q_norm: f32_source(g, &format!("{p}.indexer.q_norm.weight"))?,
                        k_norm: f32_source(g, &format!("{p}.indexer.k_norm.weight"))?,
                    },
                    compress_ratio: ratio,
                },
            };

            if li == cfg.ple.layer {
                ple_w = Some(PleFusedWeights::from_stored(
                    stored(g, &ple_key_value(&p))?,
                    f32_source(g, &format!("{p}.ple_norm_key.weight"))?,
                    f32_source(g, &format!("{p}.ple_norm_query.weight"))?,
                    f32_source(g, &format!("{p}.ple_norm_conv.weight"))?,
                    &f32_source(g, &format!("{p}.ple_conv1d.weight"))?,
                )?);
            }

            // The shared expert: gate and up fused into one weight when the
            // pack built them so, the pair apart otherwise.
            let shexp = |s: &str| format!("{p}.ffn_{s}_shexp.weight");
            let fused = shared_gate_up(&p);
            let shared = if g.info(&fused).is_some() {
                QuantizedMlp::from_repacked(
                    Some(stored(g, &fused)?),
                    None,
                    None,
                    stored(g, &shexp("down"))?,
                )?
            } else {
                QuantizedMlp::from_repacked(
                    None,
                    Some(stored(g, &shexp("gate"))?),
                    Some(stored(g, &shexp("up"))?),
                    stored(g, &shexp("down"))?,
                )?
            };
            // The routed half waits for the cache (built AFTER the dense loop,
            // from the span measurement) — the same pending-then-graft order
            // the qwen35 loader runs. The shared-expert gate is one row, stored
            // padded to a KO tile and read back at output 0.
            pending.push((
                stored(g, &format!("{p}.ffn_gate_inp.weight"))?,
                shared.with_sum_scale(ACT_SUM_SCALE),
                stored(g, &shared_gate(&p))?,
            ));

            trunk.push((hc_attn, hc_ffn, mix));
        }
        let ple_w = ple_w.ok_or_else(|| {
            candle::Error::Msg(format!(
                "qwen4exp engine: PLE layer {} produced no weights",
                cfg.ple.layer
            ))
        })?;

        let rotary = RotaryLayout::new(cfg.attn_head_dim, cfg.rope_dim, device)?;
        // The TRUNK's layers only. The draft head holds its keys in this same
        // paged cache, but at a layer past every trunk layer — `mtp_kv_layer`,
        // which this map deliberately cannot name, so no sweep driven by it can
        // reach the head.
        let kv_map = KvLayerMap::new(&cfg.layer_kinds);

        // ── Experts: measure the span the dense weights left, carve the zone,
        // fill the cache, over the `Content` and mapping read above. ──
        // The head's dense weights belong with the rest of the dense stack,
        // ahead of the expert-zone measurement below — see [`MtpDense`].
        let mtp_dense = match cfg.num_mtp_layers {
            0 => None,
            1 => Some(MtpDense::load(&mut gguf, &cfg, eps, device, &stored)?),
            n => candle::bail!(
                "qwen4exp engine: {n} draft-head blocks declared — this engine reads the \
                 one-block NextN form (`mtp_num_hidden_layers: 1`), which is what the \
                 released checkpoint carries"
            ),
        };

        // The routed experts' stored format — the rung the artifact was prepared
        // at — is the checkpoint dtype the expert section records for them.
        let Some(at) = pack.experts else {
            candle::bail!("qwen4exp engine: the pack has no expert section");
        };
        let Device::Cuda(cuda) = device else {
            candle::bail!("qwen4exp engine: the expert cache is a CUDA-only path");
        };
        let expert_format = open_section(&pack.path, at.offset, cuda)?
            .header()
            .layers
            .first()
            .map(|l| l.gate.src_dtype)
            .ok_or_else(|| {
                candle::Error::Msg("qwen4exp engine: the expert section is empty".into())
            })?;
        // The dense stack is resident: lock the block's edge before the expert
        // zone below is placed from the span's right edge.
        dense_span::close_load(device)?;
        // `SparseMoeBlock::moe_layer_idx` indexes the expert section's layers,
        // and every qwen4exp block routes — 512 experts on all of them, the head
        // included — so the section must hold blocks `0..n_blocks` in order.
        // `build_expert_cache_for` refuses one that does not: a missing block
        // would have every block above the gap compute with the previous MoE
        // layer's experts while the last fell off the end of the list.
        if let Some(report) = progress {
            report(LoadPhase::Weights, n_blocks, n_blocks);
        }
        // The PLE table reads through direct-I/O handles of its own
        // (`DiskPleFetch`); the GGUF reader — a buffered handle on the pack —
        // closes here, before the expert cache fills its warm tier from the
        // same file. The pack's own mapping (`ModelPack::mmap`) stays open
        // through the fill; nothing reads it after load.
        let ple_table = open_cached_ple(&gguf, &cfg, device)?;
        drop(gguf);
        let experts = build_expert_cache_for(
            &pack,
            cfg.moe.n_experts,
            cfg.moe.n_experts_used,
            // The head's experts join the same grid — which is the whole reason
            // it is merged in as a block: they stream over PCIe and offload
            // through the same three tiers instead of sitting resident, and
            // prefill never touches them so they cost nothing while the trunk
            // needs the room most.
            n_blocks,
            device,
            progress,
            // Nothing outside the experts is read from the host after load: the
            // n-gram table is served by its own row cache, and every other
            // tensor is resident on the card. Counted as live weight, the host
            // budget would reserve the table's 54 GB and the dense stack's
            // 5.2 GiB as page cache out of the warm tier's RAM.
            offloaded_bytes(content)?,
        )?
        .ok_or_else(|| candle::Error::Msg("qwen4exp engine: no expert tensors found".into()))?;
        let mut layers: Vec<GpuLayer> = trunk
            .into_iter()
            .zip(pending)
            .enumerate()
            .map(
                |(li, ((hc_attn, hc_ffn, mix), (gate, shared, shared_gate)))| GpuLayer {
                    hc_attn,
                    hc_ffn,
                    mix,
                    moe: Qwen35MoeBlock {
                        routed: SparseMoeBlock {
                            gate,
                            cache: experts.clone(),
                            moe_layer_idx: li,
                            num_experts_per_tok: cfg.moe.n_experts_used,
                            norm_topk_prob: cfg.moe.norm_topk_prob,
                        },
                        shared,
                        shared_gate,
                    },
                },
            )
            .collect();

        // Split the head off the tail of the block list. Its `GpuLayer` is the
        // same type a trunk layer is — same weights, same shapes, same
        // production path — so what makes it a *head* is only the NextN input
        // assembly below, which nothing in the trunk has.
        let mtp = match mtp_dense {
            None => None,
            Some(dense) => {
                let block = layers.pop().ok_or_else(|| {
                    candle::Error::Msg(
                        "qwen4exp engine: a draft head was declared but no block was built".into(),
                    )
                })?;
                Some(dense.with_block(block))
            }
        };

        Ok(Self {
            cfg,
            device: device.clone(),
            embed,
            layers,
            ple_w,
            out_hc,
            lm_head,
            rotary,
            kv_map,
            mtp,
            experts,
            expert_format,
            ple_table,
        })
    }
}
