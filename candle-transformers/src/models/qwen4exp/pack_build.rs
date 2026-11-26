//! Building the model pack of a Flash-Next engine artifact.
//!
//! The pack is the routed family's — the GGUF part, then the expert section —
//! with the dense trunk stored as the engine runs it: every int8 matmul weight
//! as its KO twin, and every weight the engine assembles from several — the
//! stacked input projections, the hyper-connection modules padded to the int8
//! tiling with their gate's `1/hc` folded in, the shared expert's padded gate
//! and fused gate|up, the PLE key|value stack, the draft head's `eh_proj`
//! halves and its narrower LM head — in its built form ([`trunk_names`]).
//!
//! The built forms are made here on the device, through the constructions the
//! engine's weights are defined by — `HcWeightsKo::from_weights`,
//! `PleFusedWeights::from_weights`, `QuantizedMlp`'s fusion, the KO repack —
//! and read back byte for byte into the pack ([`BuiltWriter`]). A load from the
//! pack copies each weight to its destination and builds nothing.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use candle::quantized::gguf_file::Value;
use candle::quantized::{GgmlDType, Int8Mode, QTensor};
use candle::{CudaDevice, Device, Result, Tensor};

use super::config::Qwen4ExpConfig;
use super::hyper::ko::HcWeightsKo;
use super::hyper::HcWeights;
use super::loader::{lm_head_name, OUTPUT};
use super::mtp::MtpDense;
use super::ple::PleWeights;
use super::ple_fused::PleFusedWeights;
use super::trunk_names::{
    attention_stack, block_kind, delta_net_stack, hc_down, hc_up, ple_key_value, replaced_sources,
    shared_gate, shared_gate_up, stack_parts, DRAFT_LM_HEAD, TRUNK_BUILD_KEY, TRUNK_BUILD_VERSION,
};
use crate::models::delta_net::LayerKind;
use crate::models::latent_moe::GgufModel;
use crate::models::model_pack::build::{write_pack, PackBuild, TokenizerSource};
use crate::models::model_pack::built::BuiltWriter;
use crate::models::model_pack::compose::{checkpoint_bytes, Composition, MappedSource};
use crate::models::model_pack::experts::qwen_expert_blocks;
use crate::models::model_pack::family::{cuda_of, expert_section};
use crate::models::model_pack::keys::TRUNK_PAIRS;
use crate::models::model_pack::prints::{from_metadata, to_metadata};
use crate::models::model_pack::provenance::Provenance;
use crate::models::model_pack::request::PackRequest;
use crate::models::quantized_matmul::{QMatMul, WeightResidency};
use crate::models::quantized_mlp::QuantizedMlp;
use crate::models::qwen35::quantized_weights::SHARED_GATE_TILE;
use crate::models::repack_fingerprint::{check_prints, pair_fingerprint, prints_for};

/// Where the built tensors go, and the `(source → stored)` repack pairs they
/// went through.
struct Store<'w> {
    out: &'w mut BuiltWriter,
    pairs: BTreeSet<(u32, u32)>,
}

impl Store<'_> {
    /// Store `m`'s weight as `name`, built from a source of `src`.
    fn put(&mut self, name: &str, m: &QMatMul, src: GgmlDType) -> Result<()> {
        let Some(qt) = m.inner().qtensor() else {
            candle::bail!(
                "qwen4exp pack: {name} built as a float weight — its shape does not tile the \
                 int8 matmul, and the engine runs every dense projection there"
            );
        };
        if !qt.dtype().is_ko() {
            candle::bail!(
                "qwen4exp pack: {name} built as {:?}, not a KO twin",
                qt.dtype()
            );
        }
        self.pairs.insert((src.to_u32(), qt.dtype().to_u32()));
        self.out.push_qtensor(name, qt)
    }

    /// The pairs recorded, in a fixed order.
    fn pairs(&self) -> Result<Vec<(GgmlDType, GgmlDType)>> {
        self.pairs
            .iter()
            .map(|&(s, d)| Ok((GgmlDType::from_u32(s)?, GgmlDType::from_u32(d)?)))
            .collect()
    }
}

/// The KO twin of checkpoint tensor `name`, built on the pool.
fn twin(
    g: &mut GgufModel,
    name: &str,
    mode: Int8Mode,
    device: &Device,
) -> Result<(QMatMul, GgmlDType)> {
    let qt = g.qtensor(name, device)?;
    let src = qt.dtype();
    Ok((
        QMatMul::from_qtensor_in(qt, mode, WeightResidency::Pool)?,
        src,
    ))
}

/// One hyper-connection module at `prefix`, in the form the engine runs: the
/// inject rows stacked under the gate's, the gate's `1/hc` folded in, both
/// projections padded to the int8 tiling (`HcWeightsKo`).
fn hc_module(
    g: &mut GgufModel,
    prefix: &str,
    with_inject: bool,
    cfg: &Qwen4ExpConfig,
    mode: Int8Mode,
    device: &Device,
) -> Result<HcWeightsKo> {
    let mut f32t =
        |name: String| -> Result<Tensor> { g.qtensor(&name, device)?.dequantize(device) };
    let down = f32t(format!("{prefix}_down.weight"))?;
    let down = if with_inject {
        let inject = f32t(format!("{prefix}_inject.weight"))?;
        Tensor::cat(&[&down, &inject], 0)?.contiguous()?
    } else {
        down
    };
    let module = HcWeights::from_checkpoint(
        f32t(format!("{prefix}_norm.weight"))?,
        down,
        f32t(format!("{prefix}_up.weight"))?,
        cfg.hc.count,
    )?;
    HcWeightsKo::from_weights(&module, mode)
}

/// Build and store `prefix`'s module.
fn store_hc(
    store: &mut Store<'_>,
    g: &mut GgufModel,
    prefix: &str,
    with_inject: bool,
    cfg: &Qwen4ExpConfig,
    mode: Int8Mode,
    device: &Device,
) -> Result<()> {
    let ko = hc_module(g, prefix, with_inject, cfg, mode, device)?;
    let (down, up) = ko.stored_projections();
    // Quantized to Q8_0 on the device, then repacked: the pair is Q8_0's.
    store.put(&hc_down(prefix), down, GgmlDType::Q8_0)?;
    store.put(&hc_up(prefix), up, GgmlDType::Q8_0)
}

/// Build every dense weight of the trunk and the draft head into `out`, and
/// answer the repack pairs they went through.
pub(crate) fn build_trunk(
    g: &mut GgufModel,
    cfg: &Qwen4ExpConfig,
    mode: Int8Mode,
    head: &str,
    device: &Device,
    out: &mut BuiltWriter,
) -> Result<Vec<(GgmlDType, GgmlDType)>> {
    if !mode.is_int8() {
        candle::bail!(
            "qwen4exp pack: the engine runs an int8 session on a CUDA card, so its pack is built \
             for an int8 mode, not {mode:?}"
        );
    }
    let mut store = Store {
        out,
        pairs: BTreeSet::new(),
    };
    let (lm_head, src) = twin(g, head, mode, device)?;
    store.put(OUTPUT, &lm_head, src)?;
    drop(lm_head);
    store_hc(&mut store, g, "output_hc", false, cfg, mode, device)?;

    let n_blocks = cfg.num_layers + cfg.num_mtp_layers;
    for li in 0..n_blocks {
        tracing::info!(
            target: "candle_transformers::model_pack",
            block = li,
            of = n_blocks,
            "model pack: building the dense trunk"
        );
        let p = format!("blk.{li}");
        store_hc(
            &mut store,
            g,
            &format!("{p}.hc_attn"),
            true,
            cfg,
            mode,
            device,
        )?;
        store_hc(
            &mut store,
            g,
            &format!("{p}.hc_ffn"),
            true,
            cfg,
            mode,
            device,
        )?;

        // The block's input projections read one activation over one
        // contraction, so they are row-concatenated into one weight — before
        // the repack, where a row is a contiguous run of blocks and no scale
        // is shared across the join.
        let kind = block_kind(cfg, li);
        let parts = stack_parts(&p, kind);
        let qts = parts
            .iter()
            .map(|n| g.qtensor(n, device))
            .collect::<Result<Vec<_>>>()?;
        let src = qts[0].dtype();
        let refs: Vec<&QTensor> = qts.iter().collect();
        let stacked = QMatMul::from_qtensor_in(
            QTensor::concat_rows_cuda(&refs)?,
            mode,
            WeightResidency::Pool,
        )?;
        let (stack_name, out_name) = match kind {
            LayerKind::DeltaNet => (delta_net_stack(&p), format!("{p}.ssm_out.weight")),
            LayerKind::Attention => (attention_stack(&p), format!("{p}.attn_output.weight")),
        };
        store.put(&stack_name, &stacked, src)?;
        drop((stacked, qts));
        let (w_out, src) = twin(g, &out_name, mode, device)?;
        store.put(&out_name, &w_out, src)?;

        if li == cfg.ple.layer {
            let mut f32t =
                |name: String| -> Result<Tensor> { g.qtensor(&name, device)?.dequantize(device) };
            let fused = PleFusedWeights::from_weights(
                &PleWeights {
                    key: f32t(format!("{p}.ple_key.weight"))?,
                    value: f32t(format!("{p}.ple_value.weight"))?,
                    norm_key: f32t(format!("{p}.ple_norm_key.weight"))?,
                    norm_query: f32t(format!("{p}.ple_norm_query.weight"))?,
                    norm_conv: f32t(format!("{p}.ple_norm_conv.weight"))?,
                    conv: f32t(format!("{p}.ple_conv1d.weight"))?,
                },
                mode,
            )?;
            store.put(&ple_key_value(&p), &fused.key_value, GgmlDType::Q8_0)?;
        }

        // The shared-expert gate is one row, padded to a KO tile and read back
        // at output 0.
        let gate_row = g
            .qtensor(&format!("{p}.ffn_gate_inp_shexp.weight"), device)?
            .dequantize(device)?
            .reshape((1, cfg.hidden_size))?;
        let pad = Tensor::zeros(
            (SHARED_GATE_TILE - 1, cfg.hidden_size),
            gate_row.dtype(),
            device,
        )?;
        let gate_vec = Tensor::cat(&[&gate_row, &pad], 0)?;
        let gate = QMatMul::from_qtensor_in(
            QTensor::quantize(&gate_vec, GgmlDType::F32)?,
            mode,
            WeightResidency::Pool,
        )?;
        store.put(&shared_gate(&p), &gate, GgmlDType::F32)?;

        let (router, src) = twin(g, &format!("{p}.ffn_gate_inp.weight"), mode, device)?;
        store.put(&format!("{p}.ffn_gate_inp.weight"), &router, src)?;

        let shexp = |s: &str| format!("{p}.ffn_{s}_shexp.weight");
        let (gate_w, up_w, down_w) = (
            g.qtensor(&shexp("gate"), device)?,
            g.qtensor(&shexp("up"), device)?,
            g.qtensor(&shexp("down"), device)?,
        );
        let (gate_src, down_src) = (gate_w.dtype(), down_w.dtype());
        let mlp = QuantizedMlp::from_weights_in(
            gate_w,
            up_w,
            down_w,
            mode,
            WeightResidency::Pool,
            (None, None, None),
        )?;
        match (mlp.fused_gate_up(), mlp.split_gate(), mlp.split_up()) {
            (Some(gate_up), None, None) => store.put(&shared_gate_up(&p), gate_up, gate_src)?,
            (None, Some(gate), Some(up)) => {
                store.put(&shexp("gate"), gate, gate_src)?;
                store.put(&shexp("up"), up, gate_src)?;
            }
            _ => candle::bail!("qwen4exp pack: {p}'s shared expert built neither form"),
        }
        store.put(&shexp("down"), mlp.down(), down_src)?;
    }

    match cfg.num_mtp_layers {
        0 => {}
        1 => {
            // The source each of the head's weights was repacked from: the LM
            // head's table, `eh_proj` for its halves, and Q8_0 for the mixer,
            // quantized on the device as every module is.
            let dtype_of = |name: &str| {
                g.info(name).map(|i| i.ggml_dtype).ok_or_else(|| {
                    candle::Error::Msg(format!("qwen4exp pack: the checkpoint has no {name}"))
                })
            };
            let head_src = dtype_of(head)?;
            let eh_src = dtype_of(&format!("blk.{}.nextn.eh_proj.weight", cfg.num_layers))?;
            MtpDense::build(g, cfg, mode, head, device, &mut |name, m| {
                let src = if name == DRAFT_LM_HEAD {
                    head_src
                } else if name.contains(".nextn.eh_proj.") {
                    eh_src
                } else {
                    GgmlDType::Q8_0
                };
                store.put(name, m, src)
            })?
        }
        n => candle::bail!(
            "qwen4exp pack: {n} draft-head blocks declared — the engine reads the one-block \
             NextN form"
        ),
    }
    store.pairs()
}

/// Build a Flash-Next engine artifact's pack into `dir`, returning its path.
pub(crate) fn build_qwen4exp(
    request: &PackRequest,
    mode: Int8Mode,
    sources: &[MappedSource],
    provenance: Provenance,
    tokenizer: TokenizerSource,
    device: &Device,
    dir: &Path,
) -> Result<PathBuf> {
    let cuda = cuda_of(device)?;
    let primary = &sources[0];
    let cfg = Qwen4ExpConfig::from_gguf_metadata(&primary.content.metadata)?;
    let layers = qwen_expert_blocks(&primary.content);
    if layers.is_empty() {
        candle::bail!(
            "model pack: {} carries no merged expert tensors — not a routed checkpoint",
            primary.path.display()
        );
    }

    let mut out = BuiltWriter::create(dir)?;
    let pairs = {
        let mut gguf = GgufModel::open(std::slice::from_ref(&primary.path))?;
        build_trunk(
            &mut gguf,
            &cfg,
            mode,
            lm_head_name(&primary.content),
            device,
            &mut out,
        )?
    };
    let built = out.finish()?;

    let (section, experts) = expert_section(primary, &layers, mode, cuda)?;
    let replaced: HashSet<String> = replaced_sources(&cfg).into_iter().collect();
    let mut composition =
        Composition::from_checkpoint(sources, &|n| !experts.contains(n) && !replaced.contains(n));
    composition.add_built(&built);
    composition.set_metadata(TRUNK_BUILD_KEY, Value::U32(TRUNK_BUILD_VERSION));
    for (k, v) in to_metadata(TRUNK_PAIRS, &prints_for(cuda, &pairs)) {
        composition.set_metadata(&k, v);
    }

    let out = dir.join(request.file_name(mode, None));
    write_pack(
        PackBuild {
            sources,
            provenance,
            composition,
            built: Some(&built),
            int8_mode: mode,
            narrow: None,
            checkpoint_bytes: checkpoint_bytes(&primary.content),
            tokenizer,
            experts: Some(section),
            layers: None,
        },
        &out,
    )?;
    Ok(out)
}

/// Refuse a Flash-Next pack this build would not have written: one of the old
/// family, with no built trunk or another version of it, or one whose repack
/// pairs no longer reproduce on this device.
pub(crate) fn check_trunk(metadata: &HashMap<String, Value>, cuda: &CudaDevice) -> Result<()> {
    match metadata.get(TRUNK_BUILD_KEY).map(|v| v.to_u32()) {
        Some(Ok(TRUNK_BUILD_VERSION)) => {}
        other => candle::bail!(
            "model pack: a Flash-Next pack's trunk is at {other:?}, this build writes version \
             {TRUNK_BUILD_VERSION}"
        ),
    }
    check_prints(&from_metadata(TRUNK_PAIRS, metadata)?, |s, d| {
        pair_fingerprint(cuda, s, d)
    })
}
