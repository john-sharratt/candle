/// Shared helpers for RULER and other benchmark tests.
///
/// These are only compiled under `#[cfg(test)]` — they exist solely to reduce
/// boilerplate in per-model test modules.
///
/// HF downloads go through [`hf_get`] / the [`api`] wrapper, which resolve
/// through [`hub_download::repo_file`]: the local caches first, then one
/// resumable, timeout-protected IPv4 download — never hf-hub's own downloader,
/// which has no read timeout and can hang on a stalled connection forever.
use std::path::PathBuf;

use hf_hub::{Cache, Repo, RepoType};

use crate::models::hub_download::{self, cached, download_path};
#[cfg(feature = "cuda")]
use crate::models::model_pack::{
    cache_root, model_pack, HubFetch, ModelPack, PackFamily, PackRequest,
};
use crate::models::qwen4exp::prepare::{SourceFile, SourceStore};
#[cfg(feature = "cuda")]
use candle::quantized::Int8Mode;

/// The engine-build [`SourceStore`] over the Hugging Face caches: fetched with
/// [`hf_get`], and cached in the hub cache or the download cache — both are
/// reported, so releasing a source clears whichever copy exists.
pub struct HfSourceStore;

impl SourceStore for HfSourceStore {
    fn fetch(&self, file: &SourceFile) -> candle::Result<PathBuf> {
        hf_get(file.repo, RepoType::Model, file.revision, file.path)
    }

    fn cached_copies(&self, file: &SourceFile) -> Vec<PathBuf> {
        let repo = Repo::with_revision(
            file.repo.to_string(),
            RepoType::Model,
            file.revision.to_string(),
        );
        let mut copies: Vec<PathBuf> = cached(&Cache::default(), &repo, file.path)
            .into_iter()
            .collect();
        copies.push(download_path(&repo, file.path));
        copies
    }
}

/// The model pack a gate asks for, from the model cache — built on first use
/// from the pinned sources, which are then released.
#[cfg(feature = "cuda")]
pub fn gate_pack(request: &PackRequest, device: &candle::Device) -> candle::Result<PathBuf> {
    model_pack(request, &cache_root(), device, &HubFetch)
}

/// A plain model's pack — the checkpoint `(repo, rev, file)` verbatim, with its
/// tokenizer `(repo, rev)` and provenance — and the tokenizer's JSON, read out
/// of the pack, so the checkpoint is downloaded once and then released.
///
/// `mode` is the int8 mode the gate loads at. A plain pack holds no repacked
/// records — its loader repacks at load — so the mode only names the file; it is
/// the gate's own so a daemon running the same checkpoint at the same mode
/// shares the pack.
#[cfg(feature = "cuda")]
pub fn plain_pack(
    checkpoint: (&str, &str, &str),
    tokenizer: (&str, &str),
    mode: Int8Mode,
    device: &candle::Device,
) -> candle::Result<(PathBuf, String)> {
    let request = PackRequest::of(PackFamily::Plain, checkpoint, tokenizer, Some(mode));
    let pack = gate_pack(&request, device)?;
    let tokenizer_json = ModelPack::open(&pack)?.tokenizer_json()?.to_string();
    Ok((pack, tokenizer_json))
}

/// A model-repo file at `revision`, local — see [`hub_download::repo_file`].
pub fn hf_get(
    repo: &str,
    repo_type: RepoType,
    revision: &str,
    filename: &str,
) -> candle::Result<PathBuf> {
    let r = Repo::with_revision(repo.to_string(), repo_type, revision.to_string());
    hub_download::repo_file(&r, filename)
}

// ---------------------------------------------------------------------------
// Drop-in resilient `Api` wrapper.
//
// Mirrors the small slice of `hf_hub::api::sync::Api` that tests use
// (`.model()/.repo()/.dataset()` → `.get()`), but every `.get()` routes through
// `hub_download`. Tests swap `hf_hub::api::sync::Api::new()` →
// `test_helpers::api()` and the rest of the call site is unchanged.
// ---------------------------------------------------------------------------

pub struct ResilientApi;

/// Construct a resilient HF API handle (infallible; returns `Result` to match
/// the `hf_hub::api::sync::Api::new()?` call shape).
pub fn api() -> candle::Result<ResilientApi> {
    Ok(ResilientApi)
}

impl ResilientApi {
    pub fn model(&self, repo_id: String) -> ResilientRepo {
        ResilientRepo {
            repo: Repo::new(repo_id, RepoType::Model),
        }
    }
    pub fn dataset(&self, repo_id: String) -> ResilientRepo {
        ResilientRepo {
            repo: Repo::new(repo_id, RepoType::Dataset),
        }
    }
    pub fn repo(&self, repo: Repo) -> ResilientRepo {
        ResilientRepo { repo }
    }
}

pub struct ResilientRepo {
    repo: Repo,
}

impl ResilientRepo {
    pub fn get(&self, filename: &str) -> candle::Result<PathBuf> {
        hub_download::repo_file(&self.repo, filename)
    }
}

/// Download `{hf_repo}/tokenizer.json` via the HF Hub and return a `Tokenizer`.
pub fn load_hf_tokenizer(hf_repo: &str) -> candle::Result<tokenizers::Tokenizer> {
    let path = hf_get(hf_repo, RepoType::Model, "main", "tokenizer.json")?;
    let json = std::fs::read_to_string(&path)
        .map_err(|e| candle::Error::Msg(format!("tokenizer read: {e}")))?;
    tokenizers::Tokenizer::from_bytes(json.as_bytes())
        .map_err(|e| candle::Error::Msg(format!("tokenizer parse: {e}")))
}

/// Download a GGUF file from the HF Hub and return its local `PathBuf`.
pub fn download_hf_gguf(hf_repo: &str, filename: &str, revision: &str) -> candle::Result<PathBuf> {
    hf_get(hf_repo, RepoType::Model, revision, filename)
}

/// Read a GGUF file from disk and return its parsed
/// [`candle::quantized::gguf_file::Content`] together with the open file handle.
pub fn open_gguf(
    path: &std::path::Path,
) -> candle::Result<(candle::quantized::gguf_file::Content, std::fs::File)> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| candle::Error::Msg(format!("open {:?}: {e}", path)))?;
    // Buffered: the header's tokenizer vocabulary is hundreds of thousands of
    // fields, each its own syscall when read straight off the file.
    let content = candle::quantized::gguf_file::Content::read(
        &mut std::io::BufReader::with_capacity(1 << 19, &mut file),
    )
    .map_err(|e| candle::Error::Msg(format!("read gguf {:?}: {e}", path)))?;
    Ok((content, file))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises the download path and the `Api` wrapper over it. Downloads
    /// tiny public files.
    #[test]
    #[ignore = "downloads from huggingface.co"]
    fn test_hf_download() {
        let repo = Repo::with_revision("gpt2".into(), RepoType::Model, "main".into());
        let dest = download_path(&repo, "config.json");
        let _ = std::fs::remove_file(&dest);
        let p = hub_download::repo_file(&repo, "config.json").expect("gpt2 config.json");
        let len = std::fs::metadata(&p).expect("metadata").len();
        assert!(len > 100, "config.json suspiciously small: {len} bytes");

        let t = hf_get("gpt2", RepoType::Model, "main", "tokenizer.json")
            .expect("hf_get gpt2 tokenizer.json");
        assert!(std::fs::metadata(&t).expect("metadata").len() > 1000);
        let t2 = api()
            .unwrap()
            .model("gpt2".to_string())
            .get("tokenizer.json")
            .expect("api().model().get()");
        assert!(std::fs::metadata(&t2).expect("metadata").len() > 1000);
    }
}
