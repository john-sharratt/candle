//! The qwen4exp production engine: GPU-resident trunk + streamed experts,
//! loaded from the ONE prepared engine GGUF (`prepare::prepare_engine`).
//!
//! The weight split mirrors the oracle's (`docs/qwen38_flash_next.md` §12.8),
//! moved onto the card:
//!
//! - **F32 at load, resident**: every HC module, the PLE projections, the
//!   GDN/attention F32 constants; the token embedding is resident too, stored
//!   BF16 and widened per gathered row. The GR mix and the PLE
//!   injection run as eager F32 tensor ops for this bring-up (their fusion is
//!   recorded §0.4 work), so their weights stay the width the oracle computes
//!   in — a load-time decision, never an in-loop conversion (invariant 1).
//! - **KO at load**: every dense projection — attention q/k/v/o, the GDN
//!   projections, routers, shared experts, the LM head — through the same
//!   `QMatMul` repack every production model uses (Q8_0 → Q8_KO here).
//! - **Routed experts stay at the artifact's KO width in the `ExpertCache`**
//!   (the rung `quant_ladder` picked when it was prepared): VRAM hot slots over
//!   pinned warm RAM over the mmap, exactly the three-tier machinery
//!   DeepSeek-V4 and Qwen3.6-35B stream through. 512 experts rides the host
//!   dispatch (`moe_bucketize` declines >256 by design).
//! - **The PLE table stays on NVMe** behind the §0.1 row cache
//!   (`loader::open_cached_ple` — the same source the oracle reads, so the
//!   two consume identical records).

use std::path::Path;
use std::sync::Arc;

use candle::quantized::gguf_file::{Content, Value};
use candle::quantized::{GgmlDType, Int8Mode, QTensor};
use candle::{Device, Result, Tensor};

use super::config::Qwen4ExpConfig;
use super::hyper::ko::HcWeightsKo;
use super::hyper::HcWeights;
use super::loader::{load_headroom_bytes, offloaded_bytes, open_cached_ple};
use super::model::PleSource;
use super::mtp::{MtpDense, MtpHead};
use super::ple::PleWeights;
use super::qsa::IndexerWeights;
use crate::models::delta_net::{KvLayerMap, QuantDeltaNetWeights};
use crate::models::dense_span;
use crate::models::expert_lre::ExpertCache;
use crate::models::latent_moe::GgufModel;
use crate::models::quantized_matmul::QMatMul;
use crate::models::quantized_mlp::QuantizedMlp;
use crate::models::quantized_qwen3_moe::SparseMoeBlock;
use crate::models::qwen35::expert_loader::build_expert_cache_for;
use crate::models::qwen35::quantized_moe::Qwen35MoeBlock;
use crate::models::qwen35::quantized_weights::{QuantAttentionWeights, SHARED_GATE_TILE};
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
    /// `[vocab, hidden]` BF16, resident.
    pub embed: Tensor,
    pub layers: Vec<GpuLayer>,
    pub ple_w: PleWeights,
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
    /// Load the engine from the merged GGUF. Order is load-bearing
    /// (`docs/archived/elastic_vram_partition.md` §4): every dense tensor resident
    /// first, then the expert cache is sized from a live measurement of what
    /// they left behind.
    pub fn load(merged: &Path, device: &Device, int8mode: Int8Mode) -> Result<Self> {
        // The KV span is sized from the governor's balloon-measured capacity;
        // without one it falls back to the small test constant and the expert
        // zone measures a floor-violating handful of slots.
        crate::models::batched_model::ensure_vram_governor(device);
        let mut gguf = GgufModel::open(&[merged.to_path_buf()])?;
        match gguf.metadata.get("general.architecture") {
            Some(Value::String(a)) if a == "qwen4exp" => {}
            other => candle::bail!(
                "qwen4exp engine: general.architecture is {other:?} — wrong checkpoint"
            ),
        }
        let cfg = Qwen4ExpConfig::from_gguf_metadata(&gguf.metadata)?;
        let eps = cfg.rms_norm_eps;
        // One `Content` and one mapping of the artifact, shared by the load
        // bracket below and the expert cache after the dense stack.
        let content = Content::read(&mut std::fs::File::open(merged)?)?;
        let mmap = Arc::new(unsafe { memmap2::Mmap::map(&std::fs::File::open(merged)?)? });
        // Claim the reservation before the first tensor, so every KO weight is
        // carved into its dense block and the span is sized from the whole card
        // rather than from what a lazily-created span found free mid-load; the
        // headroom it concedes to the pool is returned at `close_load` below.
        dense_span::open_for_load_sized(device, load_headroom_bytes(&content))?;

        let f32t = |g: &mut GgufModel, name: &str| -> Result<Tensor> {
            g.qtensor(name, device)?.dequantize(device)
        };
        let qm = |g: &mut GgufModel, name: &str| -> Result<QMatMul> {
            QMatMul::from_qtensor_with_mode(g.qtensor(name, device)?, int8mode)
        };
        // Several projections that read one activation over one contraction,
        // row-concatenated into a single weight so the block issues one GEMM
        // launch instead of N. Exact: an output column depends only on its own
        // weight row, so the stacked rows *are* the parts.
        //
        // **Before the repack, necessarily.** `concat_rows_cuda` is a byte
        // append over the GGUF block layout, where a row is a contiguous run of
        // blocks and no scale is shared across the join; after
        // `from_qtensor_with_mode` the weight is a lane-major KO twin with its
        // rows interleaved, and the append is refused by name. This engine has
        // no per-tensor narrowing schedule — every projection takes the mode's
        // twin — so unlike the qwen35 streaming loader it is free to stack.
        let stack = |g: &mut GgufModel, names: &[String]| -> Result<QMatMul> {
            let parts = names
                .iter()
                .map(|n| g.qtensor(n, device))
                .collect::<Result<Vec<_>>>()?;
            let refs: Vec<&QTensor> = parts.iter().collect();
            QMatMul::from_qtensor_with_mode(QTensor::concat_rows_cuda(&refs)?, int8mode)
        };
        // The inject rows are stacked under the down projection at load, which
        // is what makes the read gate and the write weights one GEMM instead of
        // two over the same operand — see `HcWeights::down` for why that is
        // exact and what it measures. The `cat` is a load-time cost paid once
        // per module, not the hot-path copy invariant 2 is about.
        //
        // Held KO-quantized (`HcWeightsKo`): the F32 checkpoint form is 26 MB a
        // module and 2.6 GB across the stack, resident beside a card that
        // streams its experts. The F32 module is a load-time intermediate.
        let hc = |g: &mut GgufModel, prefix: &str, with_inject: bool| -> Result<HcWeightsKo> {
            let down = g
                .qtensor(&format!("{prefix}_down.weight"), device)?
                .dequantize(device)?;
            let down = if with_inject {
                let inject = g
                    .qtensor(&format!("{prefix}_inject.weight"), device)?
                    .dequantize(device)?;
                Tensor::cat(&[&down, &inject], 0)?.contiguous()?
            } else {
                down
            };
            let f32_module = HcWeights::from_checkpoint(
                g.qtensor(&format!("{prefix}_norm.weight"), device)?
                    .dequantize(device)?,
                down,
                g.qtensor(&format!("{prefix}_up.weight"), device)?
                    .dequantize(device)?,
                cfg.hc.count,
            )?;
            HcWeightsKo::from_weights(&f32_module, int8mode)
        };

        // Resident, stored BF16 — a table-storage width chosen at load, 1.27 GB
        // for the 248,320-row table; the per-wave gather widens its rows to the
        // Gated Residual's F32. Dequantized straight to BF16 where the device
        // path takes the source type — through F32 it holds a 2.5 GB
        // intermediate in the pool beside the result — and through F32
        // otherwise, which `load_headroom_bytes` prices for exactly those types.
        let embed = {
            let q = gguf.qtensor("token_embd.weight", device)?;
            if q.dtype().dequantizes_to_bf16() {
                q.dequantize_bf16(device)?
            } else {
                q.dequantize(device)?.to_dtype(candle::DType::BF16)?
            }
        };
        // Tied embeddings, as the oracle loader handles them: a checkpoint
        // without `output.weight` projects through the embedding table. Reading
        // it unconditionally made the engine refuse artifacts the oracle
        // accepts, which is exactly the pair the oracle-vs-engine diff needs to
        // be able to run on the same checkpoint.
        let lm_head = if gguf.info("output.weight").is_some() {
            qm(&mut gguf, "output.weight")?
        } else {
            qm(&mut gguf, "token_embd.weight")?
        };
        let out_hc = hc(&mut gguf, "output_hc", false)?;

        use crate::models::delta_net::LayerKind;
        let mut trunk: Vec<(HcWeightsKo, HcWeightsKo, GpuLayerMix)> =
            Vec::with_capacity(cfg.num_layers);
        let mut pending: Vec<(QMatMul, QuantizedMlp, QMatMul)> = Vec::with_capacity(cfg.num_layers);
        let mut ple_w: Option<PleWeights> = None;
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
            let kind = if is_head {
                LayerKind::Attention
            } else {
                cfg.layer_kinds[li]
            };
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
                LayerKind::DeltaNet => GpuLayerMix::DeltaNet(QuantDeltaNetWeights {
                    proj: vec![stack(
                        g,
                        &[
                            format!("{p}.attn_qkv.weight"),
                            format!("{p}.attn_gate.weight"),
                            format!("{p}.ssm_beta.weight"),
                            format!("{p}.ssm_alpha.weight"),
                        ],
                    )?],
                    w_out: qm(g, &format!("{p}.ssm_out.weight"))?,
                    dt_bias: f32t(g, &format!("{p}.ssm_dt.bias"))?,
                    a: f32t(g, &format!("{p}.ssm_a"))?,
                    conv: f32t(g, &format!("{p}.ssm_conv1d.weight"))?,
                    norm: f32t(g, &format!("{p}.ssm_norm.weight"))?,
                }),
                LayerKind::Attention => GpuLayerMix::Attention {
                    w: QuantAttentionWeights {
                        wqkv: vec![stack(
                            g,
                            &[
                                format!("{p}.attn_q.weight"),
                                format!("{p}.attn_k.weight"),
                                format!("{p}.attn_v.weight"),
                            ],
                        )?],
                        // `q` interleaves `[q|gate]` per head, hence the 2.
                        q_rows: 2 * cfg.attn_head_dim * cfg.num_attention_heads,
                        kv_rows: cfg.attn_head_dim * cfg.num_kv_heads,
                        wo: qm(g, &format!("{p}.attn_output.weight"))?,
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
                    indexer: IndexerWeights {
                        q_proj: f32t(g, &format!("{p}.indexer.q_proj.weight"))?,
                        k_proj: f32t(g, &format!("{p}.indexer.k_proj.weight"))?,
                        q_norm: f32t(g, &format!("{p}.indexer.q_norm.weight"))?,
                        k_norm: f32t(g, &format!("{p}.indexer.k_norm.weight"))?,
                    },
                    compress_ratio: ratio,
                },
            };

            if li == cfg.ple.layer {
                ple_w = Some(PleWeights {
                    key: f32t(g, &format!("{p}.ple_key.weight"))?,
                    value: f32t(g, &format!("{p}.ple_value.weight"))?,
                    norm_key: f32t(g, &format!("{p}.ple_norm_key.weight"))?,
                    norm_query: f32t(g, &format!("{p}.ple_norm_query.weight"))?,
                    norm_conv: f32t(g, &format!("{p}.ple_norm_conv.weight"))?,
                    conv: f32t(g, &format!("{p}.ple_conv1d.weight"))?,
                });
            }

            // The shared-expert gate is one row; stored padded to a KO tile and
            // read back at output 0, exactly as the qwen35 loader stores it.
            let gate_row = f32t(g, &format!("{p}.ffn_gate_inp_shexp.weight"))?
                .reshape((1, cfg.hidden_size))?;
            let pad = Tensor::zeros(
                (SHARED_GATE_TILE - 1, cfg.hidden_size),
                gate_row.dtype(),
                device,
            )?;
            let gate_vec = Tensor::cat(&[&gate_row, &pad], 0)?;
            // The routed half waits for the cache (built AFTER the dense loop,
            // from the span measurement) — the same pending-then-graft order
            // the qwen35 loader runs.
            pending.push((
                qm(g, &format!("{p}.ffn_gate_inp.weight"))?,
                QuantizedMlp::from_weights(
                    g.qtensor(&format!("{p}.ffn_gate_shexp.weight"), device)?,
                    g.qtensor(&format!("{p}.ffn_up_shexp.weight"), device)?,
                    g.qtensor(&format!("{p}.ffn_down_shexp.weight"), device)?,
                    int8mode,
                )?,
                QMatMul::from_qtensor_with_mode(
                    QTensor::quantize(&gate_vec, candle::quantized::GgmlDType::F32)?,
                    int8mode,
                )?,
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
        // fill the cache. One Content + mmap over the merged file. ──
        // The head's dense weights belong with the rest of the dense stack,
        // ahead of the expert-zone measurement below — see [`MtpDense`].
        let mtp_dense = match cfg.num_mtp_layers {
            0 => None,
            1 => Some(MtpDense::load(&mut gguf, &cfg, eps, int8mode, device)?),
            n => candle::bail!(
                "qwen4exp engine: {n} draft-head blocks declared — this engine reads the \
                 one-block NextN form (`mtp_num_hidden_layers: 1`), which is what the \
                 released checkpoint carries"
            ),
        };

        // `SparseMoeBlock::moe_layer_idx` indexes the expert cache's COMPACTED
        // layer list: `expert_host_refs_for` skips a block carrying no expert
        // tensors, because a mixed stack is legal for the lineage at large. The
        // block index below is therefore the same number only while every block
        // is MoE — which qwen4exp is, 512 experts on all of them, the head
        // included. Check it rather than assume it: a block missing its slabs
        // still loads its router, so nothing else would object, and every block
        // above the gap would then compute with the previous MoE layer's
        // experts while the last fell off the end of the list.
        for li in 0..n_blocks {
            let p = format!("blk.{li}");
            let missing = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                .iter()
                .any(|t| {
                    !content
                        .tensor_infos
                        .contains_key(&format!("{p}.{t}.weight"))
                });
            if missing {
                candle::bail!(
                    "qwen4exp engine: blk.{li} carries no expert tensors, but every qwen4exp \
                     block is MoE — the expert cache indexes only blocks that have experts, so \
                     a dense block would shift every later block onto the wrong expert slab"
                );
            }
        }
        // Every block was just checked to carry its experts, and the recipe
        // writes one format across all of them.
        let expert_format = content.tensor_infos["blk.0.ffn_gate_exps.weight"].ggml_dtype;
        // The dense stack is resident: lock the block's edge and return the
        // load's pool headroom to the span, before the expert zone below is
        // placed from the span's right edge.
        dense_span::close_load(device)?;
        let experts = build_expert_cache_for(
            &content,
            cfg.moe.n_experts,
            // The head's experts join the same grid — which is the whole reason
            // it is merged in as a block: they stream over PCIe and offload
            // through the same three tiers instead of sitting resident, and
            // prefill never touches them so they cost nothing while the trunk
            // needs the room most.
            n_blocks,
            device,
            merged,
            mmap,
            int8mode,
            // The pack lives beside the artifact it was built from: its name
            // carries that artifact's identity, so a rebuilt artifact gets a new
            // pack and every later load of this one reads the pack rather than
            // repacking tens of GiB of experts.
            merged.parent(),
            // Nothing outside the experts is read from the host after load: the
            // n-gram table is served by its own row cache, and every other
            // tensor is resident on the card. Counted as live weight, the host
            // budget would reserve the table's 54 GB and the dense stack's
            // 5.2 GiB as page cache out of the warm tier's RAM.
            offloaded_bytes(&content)?,
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

        let gguf = Arc::new(gguf);
        let ple_table = open_cached_ple(gguf, &cfg, device)?;

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
