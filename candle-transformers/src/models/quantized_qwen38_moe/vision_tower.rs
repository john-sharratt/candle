//! Where Flash-Next's vision tower comes from: the pinned `mmproj` GGUF,
//! fetched from the hub into its cache and checked against its published
//! length and digest.
//!
//! The digest is a 900 MB read, so a file once verified is not hashed again
//! while its path and modification time stay the same.
//!
//! Fetched by whoever is about to need it, on that caller's thread — never by
//! the guest drain that reads it, which runs with every wave stopped.

use std::fs::{self, File};
use std::io;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::SystemTime;

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

/// The path whose digest matched the pin, and its modification time then.
static VERIFIED: Mutex<Option<(PathBuf, SystemTime)>> = Mutex::new(None);

/// The tower's file, fetched when it is not cached, its length checked every
/// call and its SHA-256 whenever the file is not the one last verified. Only a
/// match is remembered, so a file fetched again after a failure is checked
/// again.
pub fn vision_tower_file() -> Result<PathBuf> {
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
