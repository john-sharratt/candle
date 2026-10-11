//! What a model spec asks the model pack system for, and the pack it gets.
//!
//! Every model loads from a model pack (`docs/self_contained_model_packs.md`):
//! one file holding the checkpoint's tensors in the form this card runs them,
//! its tokenizer, and where it came from. This module turns a [`ModelSpec`]
//! into the request for that pack and resolves it — from the cache when it is
//! there, otherwise by fetching the sources, building, and releasing them.

use super::{ModelArch, ModelSpec};
use crate::error::ConversationError;
use candle::quantized::gguf_file::Content;
use candle::quantized::Int8Mode;
use candle::Device;
use candle_transformers::models::deepseek4::DEEPSEEK_V4;
use candle_transformers::models::model_pack::keys::VERSION_KEY;
use candle_transformers::models::model_pack::{
    cache_root, existing_pack, local_label, local_rev, model_pack, Fetched, LocalFetch, ModelPack,
    PackFamily, PackRequest, SourceFetch, SourceRef,
};
use candle_transformers::models::quantized_qwen38_moe::prepared_engine_pack;
use candle_transformers::models::qwen35::quantized_weights::undersized_gates;
use std::path::{Path, PathBuf};

/// A model pack, resolved, and the tokenizer it carries.
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    pub pack: PathBuf,
    /// `tokenizer.json`'s text, as the pack recorded it.
    pub tokenizer_json: String,
}

impl ResolvedModel {
    /// The pack at `path`, with its tokenizer read out of it.
    pub fn open(pack: PathBuf) -> crate::Result<Self> {
        let tokenizer_json = ModelPack::open(&pack)
            .and_then(|p| p.tokenizer_json().map(str::to_owned))
            .map_err(ConversationError::Model)?;
        Ok(Self {
            pack,
            tokenizer_json,
        })
    }

    pub fn tokenizer(&self) -> crate::Result<tokenizers::Tokenizer> {
        self.tokenizer_json
            .parse()
            .map_err(|e: tokenizers::Error| ConversationError::Tokenizer(e.to_string()))
    }
}

/// Which pack build an architecture takes.
pub(crate) fn pack_family(arch: ModelArch) -> PackFamily {
    match arch {
        ModelArch::Qwen3Moe | ModelArch::Qwen4Exp => PackFamily::Routed,
        ModelArch::DeepSeekV4 => PackFamily::Latent(&DEEPSEEK_V4),
        ModelArch::Qwen35Hybrid | ModelArch::Qwen35Dense => PackFamily::Qwen35,
        ModelArch::Qwen3 | ModelArch::Qwen2 | ModelArch::Llama => PackFamily::Plain,
    }
}

/// The int8 numeric mode an architecture's pack is built for on `device` —
/// or `None` for the one `Int8Mode::auto_sized` picks for the checkpoint.
///
/// - **The routed hybrid and Flash-Next take `auto`**, not the size-weighed
///   default. `auto_sized` asks whether the file fits in 70% of the card's VRAM,
///   which is a dense model's question: a routed checkpoint pages its experts
///   through the three-tier cache, so its file size is not its resident size —
///   and the question answers `Performance` for a 21.7 GB file on a 24 GB card
///   that runs `Precision` with every gate row valid.
/// - **DeepSeek-V4 is pinned to `Performance`**, the mode its engine is
///   measured at.
/// - **Llama and Qwen2 take `auto`**, their loaders' own default.
/// - **The rest take `auto_sized`**: the stepped-up twin only where the weights
///   leave headroom.
pub(crate) fn pack_mode(arch: ModelArch, device: &Device) -> Option<Int8Mode> {
    match arch {
        ModelArch::DeepSeekV4 => Some(Int8Mode::Performance),
        ModelArch::Qwen35Hybrid | ModelArch::Qwen4Exp | ModelArch::Llama | ModelArch::Qwen2 => {
            Some(Int8Mode::auto(device))
        }
        ModelArch::Qwen3 | ModelArch::Qwen3Moe | ModelArch::Qwen35Dense => None,
    }
}

/// The request for a spec's own checkpoint and the tensors it takes from
/// others — everything but a gate donor, which [`resolve_spec`] decides on.
pub(crate) fn spec_request(spec: &ModelSpec, device: &Device) -> PackRequest {
    let mut request = PackRequest::of(
        pack_family(spec.arch),
        (&spec.model_repo, &spec.model_rev, &spec.model_filename),
        (&spec.tokenizer_repo, &spec.tokenizer_rev),
        pack_mode(spec.arch, device),
    );
    for t in &spec.tensor_overrides {
        request = request.with_source(
            &format!("override:{}", t.tensor),
            (&t.repo, &t.revision, &t.filename),
        );
    }
    request
}

/// A fetch that answers tokenizers from one local file, and everything else
/// from `inner` — for a checkpoint the caller supplied with its tokenizer.
pub(crate) struct FileTokenizer<'a> {
    pub path: PathBuf,
    pub inner: &'a dyn SourceFetch,
}

impl SourceFetch for FileTokenizer<'_> {
    fn fetch(&self, source: &SourceRef) -> candle::Result<Fetched> {
        self.inner.fetch(source)
    }

    fn tokenizer_json(&self, _repo: &str, _rev: &str) -> candle::Result<String> {
        std::fs::read_to_string(&self.path)
            .map_err(|e| candle::Error::Msg(format!("read {}: {e}", self.path.display())))
    }

    fn release(&self, fetched: &Fetched) -> candle::Result<()> {
        self.inner.release(fetched)
    }
}

/// A fetch with nowhere to fetch from — for a build without the `hub` feature,
/// where everything a pack needs must already be on disk.
#[cfg(any(not(feature = "hub"), test))]
pub(crate) struct NoFetch;

#[cfg(any(not(feature = "hub"), test))]
impl SourceFetch for NoFetch {
    fn fetch(&self, source: &SourceRef) -> candle::Result<Fetched> {
        candle::bail!(
            "{} is read from {}, and this build has no `hub` feature to fetch it with",
            source.file,
            source.repo
        )
    }

    fn tokenizer_json(&self, repo: &str, _rev: &str) -> candle::Result<String> {
        candle::bail!(
            "the tokenizer is read from {repo}, and this build has no `hub` feature to fetch \
             it with"
        )
    }

    fn release(&self, fetched: &Fetched) -> candle::Result<()> {
        candle::bail!("{} was not fetched here", fetched.path.display())
    }
}

/// Whether `path` is a model pack rather than a checkpoint.
pub(crate) fn is_pack(path: &Path) -> bool {
    ModelPack::read_header(path).is_ok_and(|c| c.metadata.contains_key(VERSION_KEY))
}

/// The pack for a checkpoint the caller supplied by path, built into the model
/// cache under [`local_label`] — never beside the caller's file, and the file
/// is never released. `tokenizer` is the caller's `tokenizer.json`; the spec's
/// overrides and gate donor still come from `fetch`.
pub(crate) fn resolve_local(
    spec: &ModelSpec,
    checkpoint: &Path,
    tokenizer: &Path,
    device: &Device,
    fetch: &dyn SourceFetch,
) -> crate::Result<ResolvedModel> {
    let dir = checkpoint
        .parent()
        .ok_or_else(|| ConversationError::Other(format!("{checkpoint:?} has no directory")))?;
    let file = checkpoint
        .file_name()
        .ok_or_else(|| ConversationError::Other(format!("{checkpoint:?} names no file")))?
        .to_string_lossy()
        .into_owned();
    let label = local_label(dir);
    let mut local_spec = spec.clone();
    local_spec.model_repo = label.clone();
    local_spec.model_rev = local_rev(checkpoint);
    local_spec.model_filename = file;
    local_spec.tokenizer_repo = label.clone();
    local_spec.tokenizer_rev = local_rev(tokenizer);
    let tokenizers = FileTokenizer {
        path: tokenizer.to_path_buf(),
        inner: fetch,
    };
    let local = LocalFetch {
        repo: label,
        dir: dir.to_path_buf(),
        other: &tokenizers,
    };
    resolve_spec(&local_spec, device, &local)
}

/// The pack a spec names, from the model cache — built from its sources when
/// it is not there.
///
/// **A gate donor is folded in only when the checkpoint needs one.** The
/// donor is the preset an override displaced, held in reserve for a fine-tune
/// whose conversion quantized the DeltaNet recurrent gates; a stock conversion
/// stores them at F32 and has nothing to repair. So a pack already built
/// either way is taken as it is, and otherwise the checkpoint's header decides
/// — the donor's several gigabytes are fetched only for a checkpoint that
/// cannot run without them.
///
/// Flash-Next's artifact is prepared rather than published, and is never
/// prepared here: that fetches ~190 GB of pinned sources, which loading a model
/// must not start as a side effect.
pub(crate) fn resolve_spec(
    spec: &ModelSpec,
    device: &Device,
    fetch: &dyn SourceFetch,
) -> crate::Result<ResolvedModel> {
    let root = cache_root();
    if spec.prepared_from_source {
        let tokenizer = |repo: &str, rev: &str| fetch.tokenizer_json(repo, rev);
        let pack = prepared_engine_pack(device, pack_mode(spec.arch, device), &tokenizer)
            .map_err(ConversationError::Model)?;
        // The engine pack is this card's rung; a preset naming another rung's
        // artifact would otherwise be served this one under its name.
        let built_from = ModelPack::open(&pack)
            .map_err(ConversationError::Model)?
            .sources
            .first()
            .map(|s| s.file.clone())
            .unwrap_or_default();
        if built_from != spec.model_filename {
            return Err(ConversationError::Other(format!(
                "this card's engine rung is {built_from}, and the preset names {} — pick the \
                 preset for this card's rung",
                spec.model_filename
            )));
        }
        return ResolvedModel::open(pack);
    }
    let base = spec_request(spec, device);
    let request = match &spec.gate_donor {
        None => base,
        Some((repo, rev, file)) => {
            let with = base.clone().with_source("gate-donor", (repo, rev, file));
            if let Some(pack) =
                existing_pack(&base, &root, device).or_else(|| existing_pack(&with, &root, device))
            {
                return ResolvedModel::open(pack);
            }
            let primary = fetch
                .fetch(base.checkpoint())
                .map_err(ConversationError::Model)?;
            let mut f = std::fs::File::open(&primary.path)?;
            let content = Content::read(&mut f).map_err(ConversationError::Model)?;
            let bad = undersized_gates(&content);
            if bad.is_empty() {
                base
            } else {
                tracing::warn!(
                    "this checkpoint stores {} DeltaNet recurrent gates below F32 (`{}` is \
                     {:?}); its pack reads them from the base checkpoint {repo} instead, which \
                     is what this override replaced",
                    bad.len(),
                    bad[0].0,
                    bad[0].1,
                );
                with
            }
        }
    };
    let pack = model_pack(&request, &root, device, fetch).map_err(ConversationError::Model)?;
    ResolvedModel::open(pack)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_arch_takes_its_familys_build() {
        assert_eq!(pack_family(ModelArch::Qwen3Moe), PackFamily::Routed);
        assert_eq!(pack_family(ModelArch::Qwen4Exp), PackFamily::Routed);
        assert_eq!(
            pack_family(ModelArch::DeepSeekV4),
            PackFamily::Latent(&DEEPSEEK_V4)
        );
        assert_eq!(pack_family(ModelArch::Qwen35Hybrid), PackFamily::Qwen35);
        assert_eq!(pack_family(ModelArch::Qwen35Dense), PackFamily::Qwen35);
        assert_eq!(pack_family(ModelArch::Qwen3), PackFamily::Plain);
        assert_eq!(pack_family(ModelArch::Llama), PackFamily::Plain);
    }

    /// A caller's tokenizer file answers every tokenizer request; sources still
    /// go to the inner fetch.
    #[test]
    fn a_file_tokenizer_answers_tokenizers_and_passes_sources_on() {
        let dir = tempfile::tempdir().unwrap();
        let tok = dir.path().join("tokenizer.json");
        std::fs::write(&tok, "{\"v\":1}").unwrap();
        let fetch = FileTokenizer {
            path: tok,
            inner: &NoFetch,
        };
        assert_eq!(fetch.tokenizer_json("any", "rev").unwrap(), "{\"v\":1}");
        let e = fetch
            .fetch(&SourceRef::checkpoint("org/m", "", "m.gguf"))
            .err()
            .expect("NoFetch fetches nothing")
            .to_string();
        assert!(e.contains("no `hub` feature"), "{e}");
    }
}
