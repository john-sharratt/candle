//! A model pack for a checkpoint an example names by path or hub coordinate.
//!
//! The routed models load from a model pack, not a GGUF
//! (`docs/self_contained_model_packs.md`). An example points at a checkpoint —
//! a file it was given, or one it downloaded — so this packs that file into the
//! model cache, reading it in place: it is never released, and nothing is
//! written beside it. The tokenizer comes from the hub.

use std::path::{Path, PathBuf};

use candle::quantized::Int8Mode;
use candle::{Device, Result};
use candle_transformers::models::model_pack::{
    cache_root, local_label, local_rev, model_pack, HubFetch, LocalFetch, PackFamily, PackRequest,
};

/// The `family` pack of `checkpoint` at `mode` (`None` for the card's own),
/// with its tokenizer from `tokenizer_repo` — from the model cache, built on
/// first use.
pub fn pack_checkpoint(
    family: PackFamily,
    checkpoint: &Path,
    tokenizer_repo: &str,
    mode: Option<Int8Mode>,
    device: &Device,
) -> Result<PathBuf> {
    let dir = checkpoint
        .parent()
        .ok_or_else(|| candle::Error::Msg(format!("{checkpoint:?} has no directory")))?;
    let file = checkpoint
        .file_name()
        .ok_or_else(|| candle::Error::Msg(format!("{checkpoint:?} names no file")))?
        .to_string_lossy()
        .into_owned();
    let label = local_label(dir);
    let rev = local_rev(checkpoint);
    let request = PackRequest::of(family, (&label, &rev, &file), (tokenizer_repo, ""), mode);
    // The request names no source but the local file, so the hub is asked for
    // the tokenizer alone.
    let fetch = LocalFetch {
        repo: label,
        dir: dir.to_path_buf(),
        other: &HubFetch,
    };
    model_pack(&request, &cache_root(), device, &fetch)
}
