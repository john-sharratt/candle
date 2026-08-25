/// Shared helpers for RULER and other benchmark tests.
///
/// These are only compiled under `#[cfg(test)]` — they exist solely to reduce
/// boilerplate in per-model test modules.
///
/// HF downloads go through [`hf_get`] / the [`api`] wrapper, which fall back
/// from IPv6 to IPv4 (some networks have broken IPv6 routing to the HF CDN —
/// the TCP connect succeeds but the TLS handshake is reset, so the default
/// client can't recover) and resume large transfers via HTTP Range.
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// ureq resolver that returns only the IPv4 addresses for a host.
struct Ipv4Resolver;

impl ureq::Resolver for Ipv4Resolver {
    fn resolve(&self, netloc: &str) -> std::io::Result<Vec<SocketAddr>> {
        use std::net::ToSocketAddrs;
        let all: Vec<SocketAddr> = netloc.to_socket_addrs()?.collect();
        let v4: Vec<SocketAddr> = all.iter().copied().filter(|a| a.is_ipv4()).collect();
        Ok(if v4.is_empty() { all } else { v4 })
    }
}

/// Local cache dir for files fetched via the IPv4 fallback.
fn hf_fallback_cache() -> PathBuf {
    if let Ok(h) = std::env::var("HF_HOME") {
        return PathBuf::from(h).join("ipv4_fallback");
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home)
        .join(".cache")
        .join("huggingface")
        .join("ipv4_fallback")
}

/// Resumable, timeout-protected, IPv4-only download of `url` → `dest`. Follows
/// redirects (to the LFS/CDN host); a read timeout turns a stalled socket into
/// an error, and each error resumes via `Range: bytes=N-` from the bytes already
/// on disk (large GGUFs over a flaky path reset mid-stream, so a plain GET hangs).
fn download_ipv4(url: &str, dest: &Path) -> candle::Result<PathBuf> {
    let err = |m: String| candle::Error::Msg(m);
    if let Some(p) = dest.parent() {
        std::fs::create_dir_all(p).map_err(|e| err(format!("fallback cache mkdir: {e}")))?;
    }
    if dest.exists() {
        return Ok(dest.to_path_buf());
    }
    let tmp = dest.with_extension("part");

    let agent = ureq::AgentBuilder::new()
        .resolver(Ipv4Resolver)
        .redirects(10)
        .timeout_connect(Duration::from_secs(30))
        .timeout_read(Duration::from_secs(30))
        .build();

    let mut total: Option<u64> = None;
    let mut last_have = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
    let mut stalls = 0usize;
    const MAX_STALLS: usize = 200; // consecutive no-progress attempts before giving up

    loop {
        let have = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
        if have > last_have {
            stalls = 0; // made progress since last attempt
            last_have = have;
        }
        if let Some(t) = total {
            if have >= t {
                break;
            }
        }

        let mut req = agent.get(url);
        if have > 0 {
            req = req.set("Range", &format!("bytes={have}-"));
        }
        let resp = match req.call() {
            Ok(r) => r,
            Err(e) => {
                stalls += 1;
                if stalls > MAX_STALLS {
                    return Err(err(format!("IPv4 download {url}: {e}")));
                }
                std::thread::sleep(Duration::from_secs(2));
                continue;
            }
        };

        let status = resp.status();
        let mut file = match status {
            200 => {
                total = resp.header("Content-Length").and_then(|s| s.parse().ok());
                last_have = 0;
                std::fs::OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&tmp)
            }
            206 => {
                if total.is_none() {
                    total = resp
                        .header("Content-Range")
                        .and_then(|cr| cr.rsplit('/').next())
                        .and_then(|t| t.trim().parse().ok());
                }
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&tmp)
            }
            416 => break, // requested range past EOF → already complete
            s => return Err(err(format!("IPv4 download {url}: HTTP {s}"))),
        }
        .map_err(|e| err(format!("open {tmp:?}: {e}")))?;

        let mut reader = resp.into_reader();
        if let Err(e) = std::io::copy(&mut reader, &mut file) {
            stalls += 1;
            drop(file);
            if stalls > MAX_STALLS {
                return Err(err(format!("IPv4 download {url}: too many stalls ({e})")));
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    if let Some(t) = total {
        let got = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
        if got != t {
            return Err(err(format!("IPv4 download incomplete: {got}/{t} bytes")));
        }
    }
    std::fs::rename(&tmp, dest).map_err(|e| err(format!("rename {tmp:?}: {e}")))?;
    Ok(dest.to_path_buf())
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string())
}

/// Fetch a file from the HF Hub via an explicit `hf_hub::Repo`, returning a local
/// path. Answers from the local cache when it can; otherwise tries hf-hub's
/// network path, then falls back to a resumable IPv4-only download.
///
/// The cache is consulted **first and on its own**. `Api::get` also ends at the
/// cache, but only after asking the hub which revision it should be holding —
/// so a file already on disk cannot be opened while the hub is unreachable, and
/// an unanswered socket stalls for as long as the HTTP client will wait. Every
/// caller here pins an explicit revision, which is exactly the case where a
/// cache hit needs no confirmation.
fn hf_get_repo(repo: &hf_hub::Repo, filename: &str) -> candle::Result<PathBuf> {
    if let Some(p) = hf_hub::Cache::default().repo(repo.clone()).get(filename) {
        return Ok(p);
    }
    if let Ok(api) = hf_hub::api::sync::Api::new() {
        if let Ok(p) = api.repo(repo.clone()).get(filename) {
            return Ok(p);
        }
    }
    let url = format!(
        "{}/{}/resolve/{}/{filename}",
        endpoint(),
        repo.url(),
        repo.revision()
    );
    let dest = hf_fallback_cache()
        .join(repo.folder_name())
        .join(repo.revision())
        .join(filename);
    download_ipv4(&url, &dest)
}

/// Fetch a model-repo file (`revision = "main"`) with IPv6→IPv4 fallback.
pub fn hf_get(
    repo: &str,
    repo_type: hf_hub::RepoType,
    revision: &str,
    filename: &str,
) -> candle::Result<PathBuf> {
    let r = hf_hub::Repo::with_revision(repo.to_string(), repo_type, revision.to_string());
    hf_get_repo(&r, filename)
}

// ---------------------------------------------------------------------------
// Drop-in resilient `Api` wrapper.
//
// Mirrors the small slice of `hf_hub::api::sync::Api` that tests use
// (`.model()/.repo()/.dataset()` → `.get()`), but every `.get()` routes through
// the IPv6→IPv4 fallback. Tests swap `hf_hub::api::sync::Api::new()` →
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
            repo: hf_hub::Repo::new(repo_id, hf_hub::RepoType::Model),
        }
    }
    pub fn dataset(&self, repo_id: String) -> ResilientRepo {
        ResilientRepo {
            repo: hf_hub::Repo::new(repo_id, hf_hub::RepoType::Dataset),
        }
    }
    pub fn repo(&self, repo: hf_hub::Repo) -> ResilientRepo {
        ResilientRepo { repo }
    }
}

pub struct ResilientRepo {
    repo: hf_hub::Repo,
}

impl ResilientRepo {
    pub fn get(&self, filename: &str) -> candle::Result<PathBuf> {
        hf_get_repo(&self.repo, filename)
    }
}

/// Download `{hf_repo}/tokenizer.json` via the HF Hub and return a `Tokenizer`.
pub fn load_hf_tokenizer(hf_repo: &str) -> candle::Result<tokenizers::Tokenizer> {
    let path = hf_get(hf_repo, hf_hub::RepoType::Model, "main", "tokenizer.json")?;
    let json = std::fs::read_to_string(&path)
        .map_err(|e| candle::Error::Msg(format!("tokenizer read: {e}")))?;
    tokenizers::Tokenizer::from_bytes(json.as_bytes())
        .map_err(|e| candle::Error::Msg(format!("tokenizer parse: {e}")))
}

/// Download a GGUF file from the HF Hub and return its local `PathBuf`.
pub fn download_hf_gguf(
    hf_repo: &str,
    filename: &str,
    revision: &str,
) -> candle::Result<std::path::PathBuf> {
    hf_get(hf_repo, hf_hub::RepoType::Model, revision, filename)
}

/// Read a GGUF file from disk and return its parsed
/// [`candle::quantized::gguf_file::Content`] together with the open file handle.
pub fn open_gguf(
    path: &std::path::Path,
) -> candle::Result<(candle::quantized::gguf_file::Content, std::fs::File)> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| candle::Error::Msg(format!("open {:?}: {e}", path)))?;
    let content = candle::quantized::gguf_file::Content::read(&mut file)
        .map_err(|e| candle::Error::Msg(format!("read gguf {:?}: {e}", path)))?;
    Ok((content, file))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercises the resilient download path (hf-hub first, IPv4 fallback) and
    /// the IPv4 downloader directly. Downloads tiny public files.
    #[test]
    #[ignore]
    fn test_hf_ipv4_fallback() {
        // Direct IPv4 download (forces the fallback path).
        let url = format!("{}/gpt2/resolve/main/config.json", endpoint());
        let dest = hf_fallback_cache()
            .join("gpt2")
            .join("main")
            .join("config.json");
        let _ = std::fs::remove_file(&dest);
        let p = download_ipv4(&url, &dest).expect("download_ipv4 gpt2 config.json");
        let len = std::fs::metadata(&p).expect("metadata").len();
        println!("download_ipv4 -> {p:?} ({len} bytes)");
        assert!(len > 100, "config.json suspiciously small: {len} bytes");

        // Full resilient helper + the Api wrapper.
        let t = hf_get("gpt2", hf_hub::RepoType::Model, "main", "tokenizer.json")
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
