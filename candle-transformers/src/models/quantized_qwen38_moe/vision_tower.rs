//! Where Flash-Next's vision tower comes from: the pinned `mmproj` GGUF,
//! fetched from the hub into its cache and checked against its published
//! length and digest.
//!
//! Fetched by whoever is about to need it, on that caller's thread — never by
//! the guest drain that reads it, which runs with every wave stopped.

use std::path::PathBuf;
use std::sync::OnceLock;

use candle::Result;
use hf_hub::{Cache, Repo, RepoType};
use sha2::{Digest, Sha256};

use super::checkpoint_pins::{QWEN4EXP_MMPROJ_FILE, QWEN4EXP_MMPROJ_REV, QWEN4EXP_REPO};
use crate::models::hub_download::{cached, repo_file};

fn repo() -> Repo {
    Repo::with_revision(
        QWEN4EXP_REPO.to_string(),
        RepoType::Model,
        QWEN4EXP_MMPROJ_REV.to_string(),
    )
}

/// The tower's file, if it is already in the hub cache — no network.
pub fn cached_vision_tower_file() -> Option<PathBuf> {
    cached(&Cache::default(), &repo(), QWEN4EXP_MMPROJ_FILE.0)
}

/// The tower's file, fetched when it is not cached, its length checked every
/// call and its SHA-256 once per process.
pub fn vision_tower_file() -> Result<PathBuf> {
    let (name, bytes, sha) = QWEN4EXP_MMPROJ_FILE;
    let path = repo_file(&repo(), name)?;
    let len = std::fs::metadata(&path)?.len();
    if len != bytes {
        candle::bail!("{path:?} is {len} bytes, the pin is {bytes}");
    }
    static CHECKED: OnceLock<std::result::Result<(), String>> = OnceLock::new();
    CHECKED
        .get_or_init(|| {
            let mut f = std::fs::File::open(&path).map_err(|e| format!("{path:?}: {e}"))?;
            let mut h = Sha256::new();
            std::io::copy(&mut f, &mut h).map_err(|e| format!("{path:?}: {e}"))?;
            let got: String = h.finalize().iter().map(|b| format!("{b:02x}")).collect();
            if got != sha {
                return Err(format!("{path:?} has SHA-256 {got}, the pin is {sha}"));
            }
            Ok(())
        })
        .clone()
        .map_err(candle::Error::Msg)?;
    Ok(path)
}
