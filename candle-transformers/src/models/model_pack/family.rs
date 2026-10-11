//! Each family's build, its layer narrowing, and its check of a pack on disk.

use super::build::{write_pack, PackBuild, Section, TokenizerSource};
use super::compose::{checkpoint_bytes, Composition, MappedSource};
use super::experts::{expert_refs, latent_expert_blocks, qwen_expert_blocks, ExpertNames};
use super::open::ModelPack;
use super::provenance::Provenance;
use super::request::{PackFamily, PackRequest};
use crate::models::expert_lre::pack::section_len;
use crate::models::expert_lre::section::{
    geometries_of, open_section, section_header, write_section,
};
use crate::models::layer_stream::section::open_layer_section;
use crate::models::qwen35::pack_build::{build_qwen35, qwen35_narrowing};
use candle::quantized::gguf_file::Content;
use candle::quantized::Int8Mode;
use candle::{CudaDevice, Device, Result};
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};

pub(crate) fn cuda_of(device: &Device) -> Result<&CudaDevice> {
    match device {
        Device::Cuda(c) => Ok(c),
        other => {
            candle::bail!("model pack: repacked sections are built on a CUDA device, got {other:?}")
        }
    }
}

/// The expert section of `layers` in `source`, and the tensor names it takes
/// out of the GGUF part.
pub(crate) fn expert_section<'a>(
    source: &'a MappedSource,
    layers: &[ExpertNames],
    mode: Int8Mode,
    cuda: &'a CudaDevice,
) -> Result<(Section<'a>, HashSet<String>)> {
    let (refs, _) = expert_refs(&source.content, layers)?;
    let blocks: Vec<u32> = layers.iter().map(|l| l.block).collect();
    let header = section_header(&refs, &blocks, mode, cuda)?;
    let len = section_len(&header);
    let names: HashSet<String> = layers
        .iter()
        .flat_map(|l| l.all().map(str::to_string))
        .collect();
    let total = header.total_experts();
    let write = Box::new(move |w: &mut dyn Write| -> Result<u64> {
        // A line every tenth of the way: the repack is the long part of a build,
        // and a silent one reads as a hang.
        let step = (total / 10).max(1);
        let progress = |done: usize, of: usize| {
            if done.is_multiple_of(step) || done == of {
                tracing::info!(
                    target: "candle_transformers::model_pack",
                    done,
                    of,
                    "model pack: repacking experts"
                );
            }
        };
        write_section(w, header, &source.mmap, &refs, mode, cuda, Some(&progress))
    });
    Ok((Section { len, write }, names))
}

/// Build `request`'s pack from its mapped `sources` into `dir`, returning its
/// path.
pub(crate) fn build_pack(
    request: &PackRequest,
    mode: Int8Mode,
    sources: &[MappedSource],
    provenance: Provenance,
    tokenizer_json: String,
    device: &Device,
    dir: &Path,
) -> Result<PathBuf> {
    let tokenizer = TokenizerSource {
        repo: request.tokenizer_repo.clone(),
        rev: request.tokenizer_rev.clone(),
        json: tokenizer_json,
    };
    let checkpoint = checkpoint_bytes(&sources[0].content);
    match request.family {
        PackFamily::Plain => {
            let out = dir.join(request.file_name(mode, None));
            write_pack(
                PackBuild {
                    sources,
                    provenance,
                    composition: Composition::from_checkpoint(sources, &|_| true),
                    int8_mode: mode,
                    narrow: None,
                    checkpoint_bytes: checkpoint,
                    tokenizer,
                    experts: None,
                    layers: None,
                },
                &out,
            )?;
            Ok(out)
        }
        PackFamily::Routed | PackFamily::Latent(_) => {
            let cuda = cuda_of(device)?;
            let layers = match request.family {
                PackFamily::Latent(arch) => latent_expert_blocks(&sources[0].content, arch),
                _ => qwen_expert_blocks(&sources[0].content),
            };
            if layers.is_empty() {
                candle::bail!(
                    "model pack: {} carries no merged expert tensors — not a routed checkpoint",
                    sources[0].path.display()
                );
            }
            let (section, names) = expert_section(&sources[0], &layers, mode, cuda)?;
            let out = dir.join(request.file_name(mode, None));
            write_pack(
                PackBuild {
                    sources,
                    provenance,
                    composition: Composition::from_checkpoint(sources, &|n| !names.contains(n)),
                    int8_mode: mode,
                    narrow: None,
                    checkpoint_bytes: checkpoint,
                    tokenizer,
                    experts: Some(section),
                    layers: None,
                },
                &out,
            )?;
            Ok(out)
        }
        PackFamily::Qwen35 => {
            build_qwen35(request, mode, sources, provenance, tokenizer, device, dir)
        }
    }
}

/// The layer narrowing a load of this model takes on `device` — what a pack for
/// this card must have been built with.
pub(crate) fn narrowing(
    family: PackFamily,
    content: &Content,
    checkpoint_bytes: u64,
    device: &Device,
) -> Result<Option<usize>> {
    match family {
        PackFamily::Plain | PackFamily::Routed | PackFamily::Latent(_) => Ok(None),
        PackFamily::Qwen35 => qwen35_narrowing(content, checkpoint_bytes, device),
    }
}

/// Open every section of `pack` the way a load would, so a pack this build
/// cannot read is found by the resolver — and rebuilt — rather than by the
/// load that would have failed on it.
pub(crate) fn validate(family: PackFamily, pack: &ModelPack, device: &Device) -> Result<()> {
    let routed = matches!(family, PackFamily::Routed | PackFamily::Latent(_));
    if routed && pack.experts.is_none() {
        candle::bail!("model pack: a routed model's pack has no expert section");
    }
    if let Some(s) = pack.experts {
        let section = open_section(&pack.path, s.offset, cuda_of(device)?)?;
        geometries_of(section.header())?;
    }
    if let Some(s) = pack.layers {
        open_layer_section(&pack.path, s.offset, cuda_of(device)?)?;
    }
    Ok(())
}
