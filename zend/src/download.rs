//! First-run model download with progress logging.
//!
//! `ensure_model()` resolves files in this priority order:
//!   1. `~/.cache/zend/models/`         — our own cache
//!   2. `~/.cache/huggingface/hub/`     — hf-hub cache (already downloaded)
//!   3. Download from HuggingFace       — streams with 5%-step progress to log pane
//!
//! Set `HF_TOKEN` in the environment for gated / private models.

use std::path::{Path, PathBuf};

use futures::StreamExt;
use tokio::io::AsyncWriteExt;

// ── Model coordinates ─────────────────────────────────────────────────────────

const MODEL_REPO: &str  = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF";
const MODEL_FILE: &str  = "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf";
const TOK_REPO:   &str  = "Qwen/Qwen3-30B-A3B-Instruct-2507";
const TOK_FILE:   &str  = "tokenizer.json";
const MODEL_BYTES: u64  = 17_100_000_000; // ~17 GB; fallback when Content-Length absent

// ── Public API ────────────────────────────────────────────────────────────────

/// Ensure the model and tokenizer are present and return their local paths.
///
/// Progress is published on `status` so callers can surface it to users.
pub async fn ensure_model(
    status: &tokio::sync::watch::Sender<String>,
) -> anyhow::Result<(PathBuf, PathBuf)> {
    let dir = cache_dir();
    tokio::fs::create_dir_all(&dir).await?;

    let model_path = resolve_file(MODEL_REPO, MODEL_FILE, Some(MODEL_BYTES), &dir, status).await?;
    let tok_path   = resolve_file(TOK_REPO,   TOK_FILE,   None,             &dir, status).await?;

    Ok((model_path, tok_path))
}

// ── Resolution ────────────────────────────────────────────────────────────────

/// Resolve a model file: our cache → HF hub cache → download.
async fn resolve_file(
    repo:      &str,
    filename:  &str,
    size_hint: Option<u64>,
    our_dir:   &Path,
    status:    &tokio::sync::watch::Sender<String>,
) -> anyhow::Result<PathBuf> {
    // 1. Our own cache.
    let our_path = our_dir.join(filename);
    if our_path.exists() {
        let gb = tokio::fs::metadata(&our_path).await.map(|m| m.len()).unwrap_or(0) as f64 / 1e9;
        tracing::info!("cache hit: {} ({:.2} GB)", filename, gb);
        status.send(format!("Found {} ({:.1} GB)", filename, gb)).ok();
        return Ok(our_path);
    }

    // 2. HuggingFace hub cache (hf-hub or huggingface-cli may have already downloaded it).
    if let Some(hf_path) = hf_hub_path(repo, filename) {
        let gb = hf_path.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1e9;
        tracing::info!("found in HF hub cache: {} ({:.2} GB)", hf_path.display(), gb);
        status.send(format!("Found {} ({:.1} GB)", filename, gb)).ok();
        return Ok(hf_path);
    }

    // 3. Download.
    fetch(&hf_url(repo, filename), &our_path, size_hint, status).await?;
    Ok(our_path)
}

/// Look up a file in the standard HuggingFace hub cache.
///
/// Reads `refs/main` for the commit hash, then checks
/// `snapshots/{commit}/{filename}`.
fn hf_hub_path(repo: &str, filename: &str) -> Option<PathBuf> {
    let dir_name = format!("models--{}", repo.replace('/', "--"));
    let hub_root = hf_hub_root();
    let model_dir = hub_root.join(&dir_name);

    let commit = std::fs::read_to_string(model_dir.join("refs").join("main"))
        .ok()?
        .trim()
        .to_owned();

    let path = model_dir.join("snapshots").join(&commit).join(filename);
    // Sanity-check: must exist and be non-trivially sized (> 1 KB).
    if path.exists() && path.metadata().map(|m| m.len()).unwrap_or(0) > 1024 {
        Some(path)
    } else {
        None
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn hf_url(repo: &str, file: &str) -> String {
    format!("https://huggingface.co/{repo}/resolve/main/{file}")
}

/// `~/.cache/zend/models/`
pub fn cache_dir() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .or_else(|| std::env::var_os("USERPROFILE").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("zend")
        .join("models")
}

/// `~/.cache/huggingface/hub/` — respects `HF_HOME`.
fn hf_hub_root() -> PathBuf {
    if let Ok(hf_home) = std::env::var("HF_HOME") {
        return PathBuf::from(hf_home).join("hub");
    }
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .or_else(|| std::env::var_os("USERPROFILE").map(|h| PathBuf::from(h).join(".cache")))
        .unwrap_or_else(std::env::temp_dir)
        .join("huggingface")
        .join("hub")
}

/// Stream `url` to `dest`, logging progress every 5 %.
///
/// Writes to a `.part` sidecar and atomically renames on success.
async fn fetch(
    url:       &str,
    dest:      &Path,
    size_hint: Option<u64>,
    status:    &tokio::sync::watch::Sender<String>,
) -> anyhow::Result<()> {
    let part = dest.with_extension("part");
    if part.exists() {
        tokio::fs::remove_file(&part).await.ok();
    }

    let mut req = reqwest::Client::new().get(url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        req = req.bearer_auth(token);
    }

    let resp  = req.send().await?.error_for_status()?;
    let total = resp.content_length().or(size_hint).unwrap_or(0);
    let name  = dest.file_name().unwrap_or_default().to_string_lossy().to_string();

    if total > 0 {
        let msg = format!("Downloading {} ({:.1} GB)…", name, total as f64 / 1e9);
        tracing::info!("{}", msg);
        status.send(msg).ok();
    } else {
        let msg = format!("Downloading {}…", name);
        tracing::info!("{}", msg);
        status.send(msg).ok();
    }

    let mut file       = tokio::fs::File::create(&part).await?;
    let mut stream     = resp.bytes_stream();
    let mut downloaded = 0u64;
    let mut last_pct   = 0u64;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;

        if total > 0 {
            let pct = downloaded * 100 / total;
            if pct >= last_pct + 5 {
                last_pct = pct;
                let msg = format!(
                    "Downloading {} — {:.1}/{:.1} GB ({pct}%)",
                    name, downloaded as f64 / 1e9, total as f64 / 1e9,
                );
                tracing::info!("{}", msg);
                status.send(msg).ok();
            }
        }
    }

    file.flush().await?;
    drop(file);
    tokio::fs::rename(&part, dest).await?;
    tracing::info!("download complete: {}  ({:.2} GB)", name, downloaded as f64 / 1e9);
    Ok(())
}
