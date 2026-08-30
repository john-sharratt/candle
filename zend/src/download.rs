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

use crate::model_choice::model;

// ── Model coordinates ─────────────────────────────────────────────────────────
//
// The model repo/filename/size all come from `model_choice::model()` via the
// library spec — the downloader never names a checkpoint itself, so it cannot
// drift from what the session loads.

const TOK_FILE: &str = "tokenizer.json";

// ── Public API ────────────────────────────────────────────────────────────────

/// Ensure the model and tokenizer are present and return their local paths.
///
/// Progress is published on `status` so callers can surface it to users.
pub async fn ensure_model(
    status: &tokio::sync::watch::Sender<String>,
) -> anyhow::Result<(PathBuf, PathBuf)> {
    let dir = cache_dir();
    tokio::fs::create_dir_all(&dir).await?;

    let spec = model().spec();
    let model_path = resolve_file(
        &spec.model_repo,
        // The checkpoint carries no pinned revision on the spec; its published
        // length is what distinguishes it here.
        "",
        &spec.model_filename,
        Some(spec.model_bytes),
        &dir,
        status,
    )
    .await?;
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

/// Resolve a model file: our cache → HF hub cache → download.
async fn resolve_file(
    repo: &str,
    revision: &str,
    filename: &str,
    size_hint: Option<u64>,
    our_dir: &Path,
    status: &tokio::sync::watch::Sender<String>,
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
    let mut found: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(&snapshots)
        .ok()?
        .flatten()
        .filter_map(|e| usable(e.path().join(filename)))
        .map(|p| {
            let t = p
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            (t, p)
        })
        .collect();
    found.sort_by_key(|(t, _)| std::cmp::Reverse(*t));
    found.into_iter().next().map(|(_, p)| p)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn hf_url(repo: &str, revision: &str, file: &str) -> String {
    // `main` moves; a pinned revision is what makes a download reproducible.
    let rev = if revision.is_empty() { "main" } else { revision };
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
    status: &tokio::sync::watch::Sender<String>,
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

    let mut file = tokio::fs::File::create(&part).await?;
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

// ── DeepSeek-V4-Flash + DSpark drafter ──────────────────────────────────────────
//
// The main model ships as 4 MXFP4 GGUF splits nested in a same-named subfolder of
// the bartowski repo; the DSpark speculative-decode drafter is a single 10.9 GB GGUF
// at the repo root. `ensure_deepseek` fetches whichever are missing into `dir` (flat
// local names matching the engine's on-disk layout) so first-run — and adding
// speculative decode to an existing install — needs no manual `curl`.

/// bartowski GGUF repo holding both the main MXFP4 splits and the DSpark drafter.
const DSV4_REPO: &str = "bartowski/DeepSeek-V4-Flash-0731-GGUF";
/// Number of main-model GGUF splits.
const DSV4_SPLITS: usize = 4;
/// The DSpark drafter filename (identical in the repo root and on disk).
const DSPARK_FILE: &str = "dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf";

/// One file to fetch: its path *within* the HF repo and the flat local filename it
/// lands under. The repo nests the main splits in a subfolder, but the engine loads
/// them flat next to the drafter — so `path_in_repo` and `local_name` differ for the
/// splits and coincide for the drafter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RemoteFile {
    pub repo: String,
    pub path_in_repo: String,
    pub local_name: String,
}

/// The `i`-th (1-based) main-model split.
fn dsv4_split(i: usize) -> RemoteFile {
    let local = format!("DeepSeek-V4-Flash-0731-MXFP4-{i:05}-of-{DSV4_SPLITS:05}.gguf");
    RemoteFile {
        repo: DSV4_REPO.to_string(),
        // The splits live under a same-named subfolder of the repo (the root-level
        // name 404s — verified against the resolve endpoint).
        path_in_repo: format!("DeepSeek-V4-Flash-0731-MXFP4/{local}"),
        local_name: local,
    }
}

/// The DSpark drafter (repo root == local name).
fn dspark_file() -> RemoteFile {
    RemoteFile {
        repo: DSV4_REPO.to_string(),
        path_in_repo: DSPARK_FILE.to_string(),
        local_name: DSPARK_FILE.to_string(),
    }
}

/// The full DeepSeek-V4-Flash source set: the 4 main splits then the DSpark drafter.
pub fn dsv4_files() -> Vec<RemoteFile> {
    let mut v: Vec<RemoteFile> = (1..=DSV4_SPLITS).map(dsv4_split).collect();
    v.push(dspark_file());
    v
}

/// The default on-disk directory for the DeepSeek-V4-Flash GGUFs.
pub fn deepseek_dir() -> PathBuf {
    cache_dir().join("deepseek-v4-flash-mxfp4")
}

/// Local paths of the resolved DeepSeek-V4-Flash source files.
pub struct DeepseekPaths {
    /// The 4 main MXFP4 GGUF splits (offline `prepare` merges + repacks these).
    pub splits: Vec<PathBuf>,
    /// The DSpark speculative-decode drafter GGUF.
    pub dspark: PathBuf,
}

/// Ensure the DeepSeek-V4-Flash main splits + the DSpark drafter are present in
/// `dir`, downloading only whichever are missing from the HF Hub. Files already on
/// disk are kept, so adding speculative decode to an existing main-model install
/// pulls just the ~10.9 GB drafter.
pub async fn ensure_deepseek(
    dir: &Path,
    status: &tokio::sync::watch::Sender<String>,
) -> anyhow::Result<DeepseekPaths> {
    tokio::fs::create_dir_all(dir).await?;
    let mut splits = Vec::with_capacity(DSV4_SPLITS);
    for i in 1..=DSV4_SPLITS {
        splits.push(ensure_remote_file(dir, &dsv4_split(i), status).await?);
    }
    let dspark = ensure_remote_file(dir, &dspark_file(), status).await?;
    Ok(DeepseekPaths { splits, dspark })
}

/// Resolve one [`RemoteFile`] into `dir/local_name`: cache-hit when already present
/// and non-trivially sized, else stream it from `{repo}/resolve/main/{path_in_repo}`.
async fn ensure_remote_file(
    dir: &Path,
    f: &RemoteFile,
    status: &tokio::sync::watch::Sender<String>,
) -> anyhow::Result<PathBuf> {
    let local = dir.join(&f.local_name);
    let have = tokio::fs::metadata(&local)
        .await
        .map(|m| m.len())
        .unwrap_or(0);
    // > 1 MiB guards against a truncated/aborted prior write masquerading as a hit.
    if have > (1 << 20) {
        tracing::info!("cache hit: {} ({:.2} GB)", f.local_name, have as f64 / 1e9);
        status
            .send(format!(
                "Found {} ({:.1} GB)",
                f.local_name,
                have as f64 / 1e9
            ))
            .ok();
        return Ok(local);
    }
    // The DeepSeek manifest names no revision; `main` is the only coordinate
    // this path has.
    fetch(&hf_url(&f.repo, "", &f.path_in_repo), &local, None, status).await?;
    Ok(local)
}

#[cfg(test)]
mod deepseek_tests {
    use super::*;

    #[test]
    fn split_maps_subfolder_repo_path_to_flat_local() {
        let s1 = dsv4_split(1);
        assert_eq!(s1.repo, "bartowski/DeepSeek-V4-Flash-0731-GGUF");
        assert_eq!(
            s1.path_in_repo,
            "DeepSeek-V4-Flash-0731-MXFP4/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf"
        );
        assert_eq!(
            s1.local_name,
            "DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf"
        );
        // The last split uses the same 5-digit zero-padded index/count.
        assert_eq!(
            dsv4_split(4).local_name,
            "DeepSeek-V4-Flash-0731-MXFP4-00004-of-00004.gguf"
        );
    }

    #[test]
    fn dspark_file_is_repo_root() {
        let d = dspark_file();
        assert_eq!(d.path_in_repo, "dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf");
        assert_eq!(d.path_in_repo, d.local_name, "drafter is flat at the root");
    }

    #[test]
    fn full_set_is_four_splits_plus_drafter() {
        let files = dsv4_files();
        assert_eq!(files.len(), DSV4_SPLITS + 1);
        assert_eq!(files.last().unwrap().local_name, DSPARK_FILE);
        assert!(files[..DSV4_SPLITS]
            .iter()
            .all(|f| f.local_name.contains("-of-00004.gguf")));
    }

    #[test]
    fn resolve_url_matches_verified_endpoint() {
        // The exact URLs confirmed (HTTP 200) against the HF resolve endpoint.
        assert_eq!(
            hf_url(&dsv4_split(1).repo, "", &dsv4_split(1).path_in_repo),
            "https://huggingface.co/bartowski/DeepSeek-V4-Flash-0731-GGUF/resolve/main/\
             DeepSeek-V4-Flash-0731-MXFP4/DeepSeek-V4-Flash-0731-MXFP4-00001-of-00004.gguf"
        );
        assert_eq!(
            hf_url(&dspark_file().repo, "", &dspark_file().path_in_repo),
            "https://huggingface.co/bartowski/DeepSeek-V4-Flash-0731-GGUF/resolve/main/\
             dspark-DeepSeek-V4-Flash-0731-MXFP4.gguf"
        );
    }

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
