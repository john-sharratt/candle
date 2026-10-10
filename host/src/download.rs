//! First-run model download with progress logging — for every tenant, so the
//! file fetched and the model built are the same one whichever daemon loads it.
//!
//! `ensure_model()` resolves files in this priority order:
//!   1. `~/.cache/zend/models/`         — our own cache
//!   2. `~/.cache/huggingface/hub/`     — hf-hub cache (already downloaded)
//!   3. Download from HuggingFace       — streams with 5%-step progress to log pane
//!
//! Set `HF_TOKEN` in the environment for gated / private models.

use std::cmp::Reverse;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use futures::StreamExt;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use tokio::sync::watch::Sender;
use tokio::sync::Mutex;

use candle_conversation::models::Model;

// ── Model coordinates ─────────────────────────────────────────────────────────
//
// The model repo/filename/size all come from the spec of the model being
// loaded (`crate::MODEL`, or a harness's preset) — the downloader never names a
// checkpoint itself, so it cannot drift from what is loaded.

const TOK_FILE: &str = "tokenizer.json";

// ── Public API ────────────────────────────────────────────────────────────────

/// One resolution at a time, process-wide. Every tenant of a host resolves the
/// same files, and two fetching one at once would share its `.part` sidecar —
/// the second's start deleting the first's half-written file. The second
/// waits, then finds the file the first fetched.
static RESOLVING: Mutex<()> = Mutex::const_new(());

/// Ensure the model and tokenizer are present and return their local paths.
///
/// Progress is published on `status` so callers can surface it to users.
pub async fn ensure_model(
    model: &Model,
    status: &Sender<String>,
) -> anyhow::Result<(PathBuf, PathBuf)> {
    let _one_at_a_time = RESOLVING.lock().await;
    let dir = cache_dir();
    tokio::fs::create_dir_all(&dir).await?;

    let spec = model.clone().spec();
    let model_path = if spec.prepared_from_source {
        resolve_prepared(&spec.model_repo, &spec.model_filename, &dir, status)?
    } else {
        resolve_file(
            &spec.model_repo,
            // The checkpoint carries no pinned revision on the spec; its
            // published length is what distinguishes it here.
            "",
            &spec.model_filename,
            Some(spec.model_bytes),
            &dir,
            status,
        )
        .await?
    };
    let tok_path = resolve_file(
        &spec.tokenizer_repo,
        &spec.tokenizer_rev,
        TOK_FILE,
        None,
        &dir,
        status,
    )
    .await?;

    Ok((model_path, tok_path))
}

// ── Resolution ────────────────────────────────────────────────────────────────

/// Locate an engine artifact this codebase **prepares** rather than downloads.
///
/// Same cache layout as everything else — `<cache>/<repo-with-dashes>/<file>` —
/// so a prepared artifact sits beside the published ones and one convention
/// covers both. What differs is the miss: there is no URL to fall back to,
/// because the name was never published, so the miss is reported as the missing
/// build step it actually is.
///
/// Also accepts the file in the hub cache, for the case where a prepare wrote
/// beside the source shards it read.
fn resolve_prepared(
    repo: &str,
    filename: &str,
    our_dir: &Path,
    status: &Sender<String>,
) -> anyhow::Result<PathBuf> {
    let our_path = our_dir.join(repo.replace('/', "--")).join(filename);
    for candidate in [Some(our_path.clone()), hf_hub_path(repo, "", filename)]
        .into_iter()
        .flatten()
    {
        if candidate.exists() {
            let gb = candidate.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1e9;
            tracing::info!("prepared artifact: {} ({:.2} GB)", candidate.display(), gb);
            status
                .send(format!("Found {} ({:.1} GB)", filename, gb))
                .ok();
            return Ok(candidate);
        }
    }
    // No fetch, and no partial-file placeholder left behind. Naming the step is
    // the whole point: a 404 on `{filename}` would be a confusing way to say
    // "the merge has not been run on this machine".
    anyhow::bail!(
        "engine artifact {filename} is not on this machine.\n\
         It is BUILT, not downloaded — nothing publishes it under that name, and the name \
         carries the digest of the recipe that builds it. Produce it on this card with the \
         `test_parallel_batched_forwarding` gate in \
         `candle-transformers/src/models/quantized_qwen38_moe.rs` (`qwen4exp::prepare`: it \
         fetches the recipe's pinned {repo} sources, brings the routed experts to this card's \
         rung, folds in the MTP head and merges), which writes it to {}.",
        our_path.display(),
    )
}

/// Resolve a model file: our cache → HF hub cache → download.
async fn resolve_file(
    repo: &str,
    revision: &str,
    filename: &str,
    size_hint: Option<u64>,
    our_dir: &Path,
    status: &Sender<String>,
) -> anyhow::Result<PathBuf> {
    // 1. Our own cache, keyed on REPO **and** filename, and the length is
    //    checked on top when the spec states one.
    //
    //    Keying on the filename alone let one model's file shadow another's
    //    forever. `tokenizer.json` is the same name in every repo and has no
    //    published length to check, so a Qwen3-30B tokenizer left here by an
    //    earlier model was served to Qwen3.6-35B — a completely different
    //    vocabulary (`<|endoftext|>` is 151643 there and 248044 here). Every
    //    prompt was then encoded to token ids that meant something else, so the
    //    model was fed token soup, answered incoherently, and the substrate
    //    filled with 349 folder summaries written against garbage. Nothing
    //    failed; the daemon reported a clean load throughout.
    //
    //    Scoping by repo makes that unrepresentable rather than guarded against.
    let our_path = our_dir.join(repo.replace('/', "--")).join(filename);
    if let Some(parent) = our_path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    if our_path.exists() {
        let len = tokio::fs::metadata(&our_path).await.map(|m| m.len()).ok();
        let gb = len.unwrap_or(0) as f64 / 1e9;
        match (len, size_hint) {
            (Some(len), Some(want)) if len != want => {
                tracing::warn!(
                    "cached {} is {} B, expected {} B — ignoring it and re-resolving",
                    filename,
                    len,
                    want,
                );
            }
            _ => {
                tracing::info!("cache hit: {} ({:.2} GB)", filename, gb);
                status
                    .send(format!("Found {} ({:.1} GB)", filename, gb))
                    .ok();
                return Ok(our_path);
            }
        }
    }

    // 2. HuggingFace hub cache (hf-hub or huggingface-cli may have already
    //    downloaded it). A pinned revision selects its snapshot directly, so
    //    the resolved file is the one the gates verified rather than whichever
    //    snapshot `refs/main` currently names.
    if let Some(hf_path) = hf_hub_path(repo, revision, filename) {
        let gb = hf_path.metadata().map(|m| m.len()).unwrap_or(0) as f64 / 1e9;
        tracing::info!(
            "found in HF hub cache: {} ({:.2} GB)",
            hf_path.display(),
            gb
        );
        status
            .send(format!("Found {} ({:.1} GB)", filename, gb))
            .ok();
        return Ok(hf_path);
    }

    // 3. Download, from the pinned revision when there is one.
    fetch(
        &hf_url(repo, revision, filename),
        &our_path,
        size_hint,
        status,
    )
    .await?;
    Ok(our_path)
}

/// Look up a file in the standard HuggingFace hub cache.
///
/// Prefers `refs/main` for the commit hash, then checks
/// `snapshots/{commit}/{filename}`.
///
/// Falls back to searching the snapshot directories when there is no
/// `refs/main`. A repo fetched **by pinned revision** — which is how this
/// codebase pins every checkpoint it gates on, so the upstream cannot drift
/// under a test — never writes that ref, so a `refs/main`-only lookup reports
/// a 22 GB file that is already on disk as missing and downloads it a second
/// time under a different path.
fn hf_hub_path(repo: &str, revision: &str, filename: &str) -> Option<PathBuf> {
    let dir_name = format!("models--{}", repo.replace('/', "--"));
    let model_dir = hf_hub_root().join(&dir_name);
    // A pinned revision names its snapshot outright. Falling through to the
    // `refs/main`-or-search path below would resolve to whatever else happens
    // to be cached, which is the drift the pin exists to prevent.
    if !revision.is_empty() {
        let pinned = model_dir.join("snapshots").join(revision).join(filename);
        if pinned.exists() && pinned.metadata().map(|m| m.len()).unwrap_or(0) > 1024 {
            return Some(pinned);
        }
    }
    cached_in_model_dir(&model_dir, filename)
}

/// The lookup itself, against one `models--*` directory.
fn cached_in_model_dir(model_dir: &Path, filename: &str) -> Option<PathBuf> {
    let snapshots = model_dir.join("snapshots");

    let usable = |p: PathBuf| -> Option<PathBuf> {
        // Must exist and be non-trivially sized (> 1 KB).
        (p.exists() && p.metadata().map(|m| m.len()).unwrap_or(0) > 1024).then_some(p)
    };

    if let Some(commit) = std::fs::read_to_string(model_dir.join("refs").join("main"))
        .ok()
        .map(|c| c.trim().to_owned())
    {
        if let Some(p) = usable(snapshots.join(&commit).join(filename)) {
            return Some(p);
        }
    }

    // No usable `refs/main`: take the newest snapshot holding the file, so a
    // repo pinned by revision resolves and a later re-pin wins over an older
    // one still on disk.
    let mut found: Vec<(SystemTime, PathBuf)> = std::fs::read_dir(&snapshots)
        .ok()?
        .flatten()
        .filter_map(|e| usable(e.path().join(filename)))
        .map(|p| {
            let t = p
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH);
            (t, p)
        })
        .collect();
    found.sort_by_key(|(t, _)| Reverse(*t));
    found.into_iter().next().map(|(_, p)| p)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn hf_url(repo: &str, revision: &str, file: &str) -> String {
    // `main` moves; a pinned revision is what makes a download reproducible.
    let rev = if revision.is_empty() {
        "main"
    } else {
        revision
    };
    format!("https://huggingface.co/{repo}/resolve/{rev}/{file}")
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
    url: &str,
    dest: &Path,
    size_hint: Option<u64>,
    status: &Sender<String>,
) -> anyhow::Result<()> {
    let part = dest.with_extension("part");
    if part.exists() {
        tokio::fs::remove_file(&part).await.ok();
    }

    let mut req = reqwest::Client::new().get(url);
    if let Ok(token) = std::env::var("HF_TOKEN") {
        req = req.bearer_auth(token);
    }

    let resp = req.send().await?.error_for_status()?;
    let total = resp.content_length().or(size_hint).unwrap_or(0);
    let name = dest
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    if total > 0 {
        let msg = format!("Downloading {} ({:.1} GB)…", name, total as f64 / 1e9);
        tracing::info!("{}", msg);
        status.send(msg).ok();
    } else {
        let msg = format!("Downloading {}…", name);
        tracing::info!("{}", msg);
        status.send(msg).ok();
    }

    let mut file = File::create(&part).await?;
    let mut stream = resp.bytes_stream();
    let mut downloaded = 0u64;
    let mut last_pct = 0u64;

    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        downloaded += chunk.len() as u64;

        if let Some(pct) = (downloaded * 100).checked_div(total) {
            if pct >= last_pct + 5 {
                last_pct = pct;
                let msg = format!(
                    "Downloading {} — {:.1}/{:.1} GB ({pct}%)",
                    name,
                    downloaded as f64 / 1e9,
                    total as f64 / 1e9,
                );
                tracing::info!("{}", msg);
                status.send(msg).ok();
            }
        }
    }

    file.flush().await?;
    drop(file);
    tokio::fs::rename(&part, dest).await?;
    tracing::info!(
        "download complete: {}  ({:.2} GB)",
        name,
        downloaded as f64 / 1e9
    );
    Ok(())
}

#[cfg(test)]
mod url_tests {
    use super::*;

    /// A pinned revision replaces `main` in the resolve URL, so a download is
    /// reproducible rather than whatever the branch points at today.
    #[test]
    fn a_pinned_revision_replaces_main_in_the_resolve_url() {
        assert_eq!(
            hf_url("Qwen/Qwen3.6-35B-A3B", "995ad96eac", "tokenizer.json"),
            "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/995ad96eac/tokenizer.json"
        );
        assert_eq!(
            hf_url("Qwen/Qwen3.6-35B-A3B", "", "tokenizer.json"),
            "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/resolve/main/tokenizer.json",
            "an unpinned spec still resolves, against main"
        );
    }
}

/// The hub-cache lookup, which is model-agnostic.
#[cfg(test)]
mod hub_cache_tests {
    use super::cached_in_model_dir;

    /// Build a `models--*` dir holding `filename` in one snapshot, optionally
    /// with a `refs/main` pointing at `main_ref`. Files are 2 KB — over the
    /// 1 KB "non-trivially sized" floor.
    fn hub_dir(snapshots: &[&str], filename: &str, main_ref: Option<&str>) -> tempfile::TempDir {
        let td = tempfile::tempdir().unwrap();
        for s in snapshots {
            let dir = td.path().join("snapshots").join(s);
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(filename), vec![0u8; 2048]).unwrap();
        }
        if let Some(r) = main_ref {
            std::fs::create_dir_all(td.path().join("refs")).unwrap();
            std::fs::write(td.path().join("refs").join("main"), r).unwrap();
        }
        td
    }

    #[test]
    fn refs_main_is_used_when_present() {
        let td = hub_dir(&["aaa", "bbb"], "m.gguf", Some("bbb\n"));
        let got = cached_in_model_dir(td.path(), "m.gguf").expect("resolves");
        assert!(got.ends_with("snapshots/bbb/m.gguf") || got.ends_with(r"snapshots\bbb\m.gguf"));
    }

    /// A repo fetched by pinned revision writes no `refs/main`. Before this
    /// fallback the file was reported missing and re-downloaded — 22 GB for the
    /// hybrid, which is pinned exactly that way.
    #[test]
    fn a_revision_pinned_snapshot_resolves_without_refs_main() {
        let td = hub_dir(&["5bc3e238"], "m.gguf", None);
        let got = cached_in_model_dir(td.path(), "m.gguf").expect("resolves without refs/main");
        assert!(got.exists());
    }

    #[test]
    fn a_dangling_refs_main_falls_back_rather_than_missing_the_file() {
        // Ref names a snapshot that was garbage-collected; the file is elsewhere.
        let td = hub_dir(&["aaa"], "m.gguf", Some("deadbeef"));
        assert!(cached_in_model_dir(td.path(), "m.gguf").is_some());
    }

    #[test]
    fn an_absent_file_is_none_not_a_wrong_path() {
        let td = hub_dir(&["aaa"], "m.gguf", None);
        assert!(cached_in_model_dir(td.path(), "other.gguf").is_none());
    }

    #[test]
    fn a_truncated_file_is_not_a_cache_hit() {
        let td = tempfile::tempdir().unwrap();
        let dir = td.path().join("snapshots").join("aaa");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("m.gguf"), b"stub").unwrap(); // under the 1 KB floor
        assert!(cached_in_model_dir(td.path(), "m.gguf").is_none());
    }
}
