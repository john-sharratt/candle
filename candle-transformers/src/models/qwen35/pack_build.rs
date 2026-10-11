//! Building the model pack of a Qwen3.5-lineage checkpoint.
//!
//! The lineage's checkpoints are the ones that arrive in more than one file:
//!
//! * a **draft head** shipped as a sidecar beside the trunk (ggml-org's
//!   convention — the trunk file declares no head at all);
//! * a **gate donor**, the base checkpoint a fine-tune's quantized DeltaNet
//!   recurrent path is read from instead of its own;
//! * **tensor overrides**, named tensors taken from another release.
//!
//! The build resolves all of them by the loader's own rules — the sidecar's
//! head blocks and its `nextn_predict_layers`, `donor_gates`'s repair set, each
//! override's tensor — into one composition, so a load of the pack opens one
//! file and finds the model exactly as the multi-file load assembled it.
//!
//! Then the section: a routed checkpoint's experts, or a dense checkpoint's
//! streamed trunk projections, built one layer at a time through the same
//! `load_layer` a resident load runs, so a record is byte for byte what the
//! loader would have produced.

use super::config::Qwen35Config;
use super::layer_loader::loaded_layer;
use super::loader::detect_arch;
use super::quantized_weights::{
    donor_gates, load_layer, narrow_resident_twin, streaming_twin, undersized_gates, Loader,
    RECURRENT_PATH,
};
use super::tensor_override::{TensorOverride, TensorOverrides};
use crate::models::delta_net::LayerKind;
use crate::models::layer_stream::build::{images_from_gguf, tensor_suffix};
use crate::models::layer_stream::descriptor::{LayerTensor, Projection};
use crate::models::layer_stream::pack::{header_for, section_len, PackWriter};
use crate::models::model_pack::build::{write_pack, PackBuild, Section, TokenizerSource};
use crate::models::model_pack::compose::{checkpoint_bytes, Composition, MappedSource};
use crate::models::model_pack::experts::qwen_expert_blocks;
use crate::models::model_pack::family::{cuda_of, expert_section};
use crate::models::model_pack::provenance::Provenance;
use crate::models::model_pack::request::PackRequest;
use crate::models::quantized_matmul::WeightResidency;
use crate::models::repack_fingerprint::pair_fingerprint;
use candle::quantized::gguf_file::{Content, Value};
use candle::quantized::{GgmlDType, Int8Mode};
use candle::{Device, Result};
use std::collections::HashSet;
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};

/// The layer narrowing a load of this checkpoint takes on `device`:
/// `Some(num_layers)` when a dense checkpoint of `checkpoint_bytes` does not
/// fit the card with the KV side's opening reserve, `None` otherwise.
pub(crate) fn qwen35_narrowing(
    content: &Content,
    checkpoint_bytes: u64,
    device: &Device,
) -> Result<Option<usize>> {
    let arch = detect_arch(content);
    let cfg = Qwen35Config::from_gguf_metadata(&arch, &content.metadata)?;
    Ok(narrow_resident_twin(device, &cfg, checkpoint_bytes).map(|_| cfg.num_layers))
}

/// Every trunk tensor a dense load streams instead of holding resident: each
/// layer's mixer projections and its FFN's three.
fn streamed_names(cfg: &Qwen35Config) -> HashSet<String> {
    let mut out = HashSet::new();
    for (li, kind) in cfg.layer_kinds.iter().enumerate() {
        let mix = match kind {
            LayerKind::DeltaNet => LayerTensor::DELTA_NET_MIX,
            LayerKind::Attention => LayerTensor::ATTENTION_MIX,
        };
        for role in mix {
            if let Some(suffix) = tensor_suffix(*role) {
                out.insert(format!("blk.{li}.{suffix}"));
            }
        }
        for ffn in ["ffn_gate", "ffn_up", "ffn_down"] {
            out.insert(format!("blk.{li}.{ffn}.weight"));
        }
    }
    out
}

/// The wider of two formats, by bits per weight.
fn wider(a: GgmlDType, b: GgmlDType) -> bool {
    a.bits_per_weight() > b.bits_per_weight()
}

/// Build a Qwen3.5-lineage pack into `dir`, returning its path.
pub(crate) fn build_qwen35(
    request: &PackRequest,
    mode: Int8Mode,
    sources: &[MappedSource],
    provenance: Provenance,
    tokenizer: TokenizerSource,
    device: &Device,
    dir: &Path,
) -> Result<PathBuf> {
    let primary = &sources[0];
    let arch = detect_arch(&primary.content);
    let cfg = Qwen35Config::from_gguf_metadata(&arch, &primary.content.metadata)?;
    let role_of = |role: &str| request.sources.iter().position(|s| s.role == role);
    let mtp = role_of("mtp");
    let donor = role_of("gate-donor");
    let override_specs: Vec<(usize, TensorOverride)> = request
        .sources
        .iter()
        .enumerate()
        .filter_map(|(i, s)| {
            s.role
                .strip_prefix("override:")
                .map(|t| (i, TensorOverride::new(t, &sources[i].path)))
        })
        .collect();
    let overrides = TensorOverrides::new(
        &primary.content,
        override_specs
            .iter()
            .map(|(i, spec)| (spec, &sources[*i].content, &sources[*i].mmap[..])),
    )?;
    // The donor repairs only what is broken — `load_quantized_model`'s rule.
    let donor_names: Vec<String> = match donor {
        Some(d) if !undersized_gates(&primary.content).is_empty() => {
            donor_gates(&primary.content, &sources[d].content)?
        }
        _ => Vec::new(),
    };
    let gate_src = donor
        .filter(|_| !donor_names.is_empty())
        .map(|d| (&sources[d].content, &sources[d].mmap[..]));

    let checkpoint = checkpoint_bytes(&primary.content);
    let dense = cfg.moe.is_none();
    let narrow = if dense {
        qwen35_narrowing(&primary.content, checkpoint, device)?
    } else {
        None
    };
    let expert_layers = if dense {
        Vec::new()
    } else {
        qwen_expert_blocks(&primary.content)
    };
    let mut leaves: HashSet<String> = if dense {
        streamed_names(&cfg)
    } else {
        HashSet::new()
    };
    for l in &expert_layers {
        leaves.extend(l.all().map(str::to_string));
    }

    let mut composition = Composition::from_checkpoint(sources, &|n| !leaves.contains(n));
    if let Some(m) = mtp {
        let side = &sources[m].content;
        let n = Qwen35Config::mtp_layers_in(&side.metadata, &arch);
        if n == 0 {
            candle::bail!(
                "model pack: the MTP sidecar declares no {arch}.nextn_predict_layers, so it \
                 carries no draft head"
            );
        }
        let mut head: Vec<&String> = side
            .tensor_infos
            .keys()
            .filter(|k| {
                k.strip_prefix("blk.")
                    .and_then(|r| r.split_once('.'))
                    .and_then(|(idx, _)| idx.parse::<usize>().ok())
                    .is_some_and(|li| li >= cfg.num_layers)
            })
            .collect();
        head.sort();
        for name in head {
            composition.take(sources, m, name)?;
        }
        // The embedding table is read from whichever copy is wider — the
        // loader's `EmbeddingTable::widest_host_mapped` rule.
        let embed = "token_embd.weight";
        if let (Some(theirs), Some(mine)) = (
            side.tensor_infos.get(embed),
            primary.content.tensor_infos.get(embed),
        ) {
            if wider(theirs.ggml_dtype, mine.ggml_dtype) {
                composition.take(sources, m, embed)?;
            }
        }
        // The embedded convention: the head counts in `block_count`, and
        // `nextn_predict_layers` says how much of it is head.
        composition.set_metadata(
            &format!("{arch}.nextn_predict_layers"),
            Value::U32(n as u32),
        );
        composition.set_metadata(
            &format!("{arch}.block_count"),
            Value::U32((cfg.num_layers + n) as u32),
        );
    }
    if let Some(d) = donor {
        for name in donor_names.iter().filter(|n| !leaves.contains(*n)) {
            composition.take(sources, d, name)?;
        }
    }
    for (i, spec) in &override_specs {
        if !leaves.contains(&spec.tensor) {
            composition.take(sources, *i, &spec.tensor)?;
        }
    }

    let cuda = cuda_of(device)?;
    let experts = if expert_layers.is_empty() {
        None
    } else {
        Some(expert_section(primary, &expert_layers, mode, cuda)?.0)
    };
    let layers = if dense {
        Some(layer_section(
            primary, &cfg, mode, narrow, gate_src, &overrides, device,
        )?)
    } else {
        None
    };
    let out = dir.join(request.file_name(mode, narrow));
    write_pack(
        PackBuild {
            sources,
            provenance,
            composition,
            int8_mode: mode,
            narrow,
            checkpoint_bytes: checkpoint,
            tokenizer,
            experts,
            layers,
        },
        &out,
    )?;
    Ok(out)
}

/// The layer section of a dense checkpoint: every trunk layer, loaded and
/// repacked one at a time through `load_layer`, so a record is what a resident
/// load would have built.
fn layer_section<'a>(
    primary: &'a MappedSource,
    cfg: &'a Qwen35Config,
    mode: Int8Mode,
    narrow: Option<usize>,
    gate_src: Option<(&'a Content, &'a [u8])>,
    overrides: &'a TensorOverrides<'a>,
    device: &'a Device,
) -> Result<Section<'a>> {
    let cuda = cuda_of(device)?;
    let narrowing = |name: &str| narrow.and_then(|n| streaming_twin(name, n));
    // The donor's dtype wherever the donor supplies the tensor, and an
    // override's wherever one does: the slot is sized for what gets written.
    let substitute = |name: &str| -> Option<GgmlDType> {
        if let Some(info) = overrides.effective_info(&primary.content, name) {
            if info.ggml_dtype != primary.content.tensor_infos.get(name)?.ggml_dtype {
                return Some(info.ggml_dtype);
            }
        }
        let (donor, _) = gate_src?;
        if !RECURRENT_PATH.iter().any(|r| name.ends_with(r)) {
            return None;
        }
        let theirs = donor.tensor_infos.get(name)?.ggml_dtype;
        let mine = primary.content.tensor_infos.get(name)?.ggml_dtype;
        (theirs != mine).then_some(theirs)
    };
    let images = images_from_gguf(
        &primary.content,
        &cfg.layer_kinds,
        mode,
        &narrowing,
        &substitute,
    )?;
    // The checkpoint's dtype for a projection — a fused FFN's halves share one.
    let src_dtype = |li: usize, p: &Projection| -> GgmlDType {
        let suffix = tensor_suffix(p.role).unwrap_or("ffn_gate.weight");
        let name = format!("blk.{li}.{suffix}");
        substitute(&name)
            .or_else(|| {
                primary
                    .content
                    .tensor_infos
                    .get(&name)
                    .map(|i| i.ggml_dtype)
            })
            .unwrap_or(p.dtype)
    };
    let header = header_for(&images, &src_dtype, mode as u32, &|s, d| {
        pair_fingerprint(cuda, s, d)
    });
    let len = section_len(&header);
    let write = Box::new(move |w: &mut dyn Write| -> Result<u64> {
        let stream = cuda.cuda_stream();
        let mut reader = Cursor::new(&primary.mmap[..]);
        // Pool, not span: each layer is materialised only to be read back and
        // dropped. The donor and the overrides ride along, because what this
        // writes is the model a load will run.
        let mut g = Loader::new(
            &primary.content,
            &mut reader,
            device,
            mode,
            WeightResidency::Pool,
        )
        .with_gate_src(gate_src)
        .with_overrides(Some(overrides));
        g.set_stream_narrow(narrow);
        let mut pw = PackWriter::new(w, header)?;
        for (li, image) in images.iter().enumerate() {
            let layer = load_layer(&mut g, cfg, li, mode, &mut 0)?.resolve_dense()?;
            let loaded = loaded_layer(&layer, image)?;
            let bufs = loaded.read_back(&stream)?;
            let refs: Vec<&[u8]> = bufs.iter().map(|b| b.as_slice()).collect();
            pw.write_layer(li, &refs)?;
            if (li + 1) % 8 == 0 || li + 1 == images.len() {
                tracing::info!(
                    target: "candle_transformers::model_pack",
                    layer = li + 1,
                    of = images.len(),
                    "model pack: repacking layers"
                );
            }
        }
        pw.finish()
    });
    Ok(Section { len, write })
}
