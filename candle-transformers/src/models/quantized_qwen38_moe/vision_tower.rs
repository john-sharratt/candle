//! Where Flash-Next's vision tower comes from: a twinned model pack
//! (`model_pack::twins`) built from the pinned `mmproj` GGUF, which is fetched
//! from the hub, checked against its published length and digest, and released
//! once the pack holds it.
//!
//! The pack stores every projection as the KO twin the tower's matmuls read,
//! and the position table — which the tower gathers rows from — as its file's
//! F16. A load copies each tensor to the card as it is: no source upload, no
//! repack, no scratch.
//!
//! Resolved by whoever is about to need it, on that caller's thread — never by
//! the guest drain that reads it, which runs with every wave stopped.

use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::SystemTime;

use candle::quantized::Int8Mode;
use candle::{Device, Result};
use hf_hub::{Repo, RepoType};
use sha2::{Digest, Sha256};

use super::checkpoint_pins::{
    QWEN4EXP_MMPROJ_FILE, QWEN4EXP_MMPROJ_REV, QWEN4EXP_REPO, TOKENIZER_REPO, TOKENIZER_REV,
};
use crate::models::hub_download::repo_file;
use crate::models::model_pack::{
    cache_root, existing_pack, model_pack, Fetched, HubFetch, PackFamily, PackRequest, SourceFetch,
    SourceRef,
};
use crate::models::qwen_vision::tower::POSITION_TABLE;

/// The int8 mode a card's tower pack is built at: the tower's projections are
/// F16, whose twin is `Q8_KO` in either int8 mode, so one pack serves every card
/// with the int8 matmul.
pub const VISION_TOWER_MODE: Int8Mode = Int8Mode::Precision;

/// The tower's pack at `mode`. Its tokenizer is the text model's: the tower's
/// rows stand for that tokenizer's `<|image_pad|>`s.
fn request(mode: Int8Mode) -> PackRequest {
    PackRequest::of(
        PackFamily::Twinned(&[POSITION_TABLE]),
        (QWEN4EXP_REPO, QWEN4EXP_MMPROJ_REV, QWEN4EXP_MMPROJ_FILE.0),
        (TOKENIZER_REPO, TOKENIZER_REV),
        Some(mode),
    )
}

/// The pack each mode last resolved to in this process.
///
/// Judging a pack opens its header, and the tokenizer it carries makes that
/// megabytes of metadata. The drain asks for the path with every wave stopped,
/// so it is judged once a process — and again only if the file is gone.
static RESOLVED: Mutex<Vec<(Int8Mode, PathBuf)>> = Mutex::new(Vec::new());

fn remembered(mode: Int8Mode) -> Option<PathBuf> {
    let resolved = RESOLVED.lock().unwrap_or_else(|e| e.into_inner());
    resolved
        .iter()
        .find(|(m, p)| *m == mode && p.is_file())
        .map(|(_, p)| p.clone())
}

fn remember(mode: Int8Mode, path: &Path) {
    let mut resolved = RESOLVED.lock().unwrap_or_else(|e| e.into_inner());
    resolved.retain(|(m, _)| *m != mode);
    resolved.push((mode, path.to_path_buf()));
}

/// The tower's pack at `mode`, if this machine already holds it, with no
/// network and no build.
///
/// The request names its mode, so no card is consulted to pick one, and the
/// device the check is given is the host's.
pub fn cached_vision_tower_pack(mode: Int8Mode) -> Option<PathBuf> {
    if let Some(path) = remembered(mode) {
        return Some(path);
    }
    let path = existing_pack(&request(mode), &cache_root(), &Device::Cpu)?;
    remember(mode, &path);
    Some(path)
}

/// The tower's pack at `mode`, built — from the pinned `mmproj`, fetched when
/// it is not cached — when there is none. The build is the host's: the twins
/// are converted on the CPU.
pub fn vision_tower_pack(mode: Int8Mode) -> Result<PathBuf> {
    if let Some(path) = remembered(mode) {
        return Ok(path);
    }
    let path = model_pack(&request(mode), &cache_root(), &Device::Cpu, &PinnedFetch)?;
    remember(mode, &path);
    Ok(path)
}

/// The hub, with the `mmproj` held to its pin before a pack is built from it.
struct PinnedFetch;

impl SourceFetch for PinnedFetch {
    fn fetch(&self, source: &SourceRef) -> Result<Fetched> {
        if source.file != QWEN4EXP_MMPROJ_FILE.0 {
            candle::bail!(
                "the vision tower's pack is built from {}, not {}",
                QWEN4EXP_MMPROJ_FILE.0,
                source.file
            );
        }
        Ok(Fetched {
            path: pinned_mmproj()?,
            cached: true,
        })
    }

    fn tokenizer_json(&self, repo: &str, rev: &str) -> Result<String> {
        HubFetch.tokenizer_json(repo, rev)
    }

    fn release(&self, fetched: &Fetched) -> Result<()> {
        HubFetch.release(fetched)
    }
}

fn repo() -> Repo {
    Repo::with_revision(
        QWEN4EXP_REPO.to_string(),
        RepoType::Model,
        QWEN4EXP_MMPROJ_REV.to_string(),
    )
}

/// The path whose digest matched the pin, and its modification time then.
static VERIFIED: Mutex<Option<(PathBuf, SystemTime)>> = Mutex::new(None);

/// The `mmproj`, fetched when it is not cached, its length checked every call
/// and its SHA-256 whenever the file is not the one last verified. Only a match
/// is remembered, so a file fetched again after a failure is checked again.
fn pinned_mmproj() -> Result<PathBuf> {
    let (name, bytes, sha) = QWEN4EXP_MMPROJ_FILE;
    let path = repo_file(&repo(), name)?;
    let meta = fs::metadata(&path)?;
    if meta.len() != bytes {
        candle::bail!("{path:?} is {} bytes, the pin is {bytes}", meta.len());
    }
    let modified = meta.modified()?;
    let mut verified = VERIFIED.lock().unwrap_or_else(|e| e.into_inner());
    if verified.as_ref() == Some(&(path.clone(), modified)) {
        return Ok(path);
    }
    let mut f = File::open(&path)?;
    let mut h = Sha256::new();
    io::copy(&mut f, &mut h)?;
    let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if got != sha {
        candle::bail!("{path:?} has SHA-256 {got}, the pin is {sha}");
    }
    *verified = Some((path.clone(), modified));
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tower's pack is named for its `mmproj` and mode, in the repo's
    /// directory, and keeps the position table out of its twins.
    #[test]
    fn the_tower_pack_is_named_for_its_mmproj() {
        let r = request(VISION_TOWER_MODE);
        assert_eq!(
            r.file_name(VISION_TOWER_MODE, None),
            "mmproj-F16.precision.pack.gguf"
        );
        assert_eq!(r.family, PackFamily::Twinned(&["v.position_embd.weight"]));
        assert_eq!(r.checkpoint().rev, QWEN4EXP_MMPROJ_REV);
    }
}
