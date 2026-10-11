//! Files from the HuggingFace hub: the local caches first, then one resumable,
//! timeout-protected download.
//!
//! **Never hf-hub's own downloader.** Its client has no read timeout, so a
//! connection that opens and then never sends a byte waits forever without
//! returning an error. Some networks route IPv6 to the HF CDN brokenly — the TCP
//! connect succeeds and the TLS handshake is reset — and a sweep sat on a 0-byte
//! `.part` that way. [`download`] resolves IPv4 only, times out a stalled
//! socket, and resumes by HTTP Range from the bytes already on disk, so it is
//! the one network path.

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::time::Duration;

use candle::Result;
use hf_hub::{Cache, Repo};

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

/// Where downloads made here are kept: beside the hub cache, which they do not
/// share a layout with.
fn download_cache() -> PathBuf {
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

/// Where a download of `filename` of `repo` lands.
pub fn download_path(repo: &Repo, filename: &str) -> PathBuf {
    download_cache()
        .join(repo.folder_name())
        .join(repo.revision())
        .join(filename)
}

/// Whether an HTTP error status can clear on its own: a timeout, a rate limit,
/// or a server fault. Every other error status is a refusal.
fn retryable(code: u16) -> bool {
    matches!(code, 408 | 429) || code >= 500
}

fn endpoint() -> String {
    std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string())
}

/// Resumable, timeout-protected, IPv4-only download of `url` → `dest`. Follows
/// redirects (to the LFS/CDN host); a read timeout turns a stalled socket into
/// an error, and each error resumes via `Range: bytes=N-` from the bytes already
/// on disk (large GGUFs over a flaky path reset mid-stream, so a plain GET hangs).
pub fn download(url: &str, dest: &Path) -> Result<PathBuf> {
    let err = |m: String| candle::Error::Msg(m);
    if let Some(p) = dest.parent() {
        std::fs::create_dir_all(p).map_err(|e| err(format!("download cache mkdir: {e}")))?;
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
        if let Ok(token) = std::env::var("HF_TOKEN") {
            req = req.set("Authorization", &format!("Bearer {token}"));
        }
        if have > 0 {
            req = req.set("Range", &format!("bytes={have}-"));
        }
        let resp = match req.call() {
            Ok(r) => r,
            // **A refusal is not a stall.** Only a status that can clear on its
            // own — a timeout, a rate limit, a server fault — is worth asking
            // again; any other 4xx is the same answer every time, and retrying
            // it turned an immediate error into minutes of an empty `.part`.
            // Range past EOF: everything is already on disk.
            Err(ureq::Error::Status(416, _)) => break,
            Err(ureq::Error::Status(code, r)) if !retryable(code) => {
                return Err(err(format!(
                    "download {url}: HTTP {code} from {}",
                    r.get_url()
                )));
            }
            Err(e) => {
                stalls += 1;
                tracing::warn!(
                    target: "candle_transformers::hub_download",
                    attempt = stalls,
                    %url,
                    error = %e,
                    "download attempt failed; retrying"
                );
                if stalls > MAX_STALLS {
                    return Err(err(format!("download {url}: {e}")));
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
            s => return Err(err(format!("download {url}: HTTP {s}"))),
        }
        .map_err(|e| err(format!("open {tmp:?}: {e}")))?;

        let mut reader = resp.into_reader();
        if let Err(e) = std::io::copy(&mut reader, &mut file) {
            stalls += 1;
            drop(file);
            if stalls > MAX_STALLS {
                return Err(err(format!("download {url}: too many stalls ({e})")));
            }
            std::thread::sleep(Duration::from_secs(2));
        }
    }

    if let Some(t) = total {
        let got = std::fs::metadata(&tmp).map(|m| m.len()).unwrap_or(0);
        if got != t {
            return Err(err(format!("download incomplete: {got}/{t} bytes")));
        }
    }
    std::fs::rename(&tmp, dest).map_err(|e| err(format!("rename {tmp:?}: {e}")))?;
    Ok(dest.to_path_buf())
}

/// `repo`'s `filename` in the hub cache `cache`, or `None` if it is not there.
///
/// # A pinned revision does not live behind a ref
///
/// `CacheRepo::get` resolves in one way only: read the commit hash out of
/// `refs/<revision>`, then look under `snapshots/<hash>/`. That is right for a
/// branch or a tag, which is what a ref *is* — and wrong for a revision pinned
/// to a commit, because the hub writes `refs/<branch>` and never `refs/<sha>`.
/// Asked for a sha it reads a path that cannot exist and reports a miss, so a
/// pinned checkpoint already on disk would be downloaded again. So: the ref
/// lookup first, since a branch has to keep resolving through the ref it is
/// named by, and the snapshot directly when that finds nothing. A revision the
/// cache genuinely does not hold misses both and is still a miss.
pub fn cached(cache: &Cache, repo: &Repo, filename: &str) -> Option<PathBuf> {
    if let Some(found) = cache.repo(repo.clone()).get(filename) {
        return Some(found);
    }
    let snapshot = cache
        .path()
        .join(repo.folder_name())
        .join("snapshots")
        .join(repo.revision())
        .join(filename);
    snapshot.is_file().then_some(snapshot)
}

/// `repo`'s `filename`, local: the hub cache, a file an earlier download left,
/// or a [`download`] now.
///
/// The caches are consulted **first and on their own**: a pinned file already
/// on disk needs no confirmation from the hub, and asking makes a load depend on
/// the network for nothing.
pub fn repo_file(repo: &Repo, filename: &str) -> Result<PathBuf> {
    if let Some(p) = cached(&Cache::default(), repo, filename) {
        return Ok(p);
    }
    let url = format!(
        "{}/{}/resolve/{}/{filename}",
        endpoint(),
        repo.url(),
        repo.revision()
    );
    download(&url, &download_path(repo, filename))
}

#[cfg(test)]
mod tests {
    use super::{cached, retryable};

    #[test]
    fn only_a_transient_status_is_retried() {
        for code in [408, 429, 500, 502, 503] {
            assert!(retryable(code), "{code}");
        }
        for code in [400, 401, 403, 404, 410, 416] {
            assert!(!retryable(code), "{code}");
        }
    }
    use hf_hub::{Cache, Repo, RepoType};

    fn pinned(repo: &str, rev: &str) -> Repo {
        Repo::with_revision(repo.to_string(), RepoType::Model, rev.to_string())
    }

    /// **A cached file resolves without the network, and a miss says so** —
    /// through the ref for a branch, through the snapshot for a pinned commit.
    #[test]
    fn a_cached_file_is_found_without_touching_the_network() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let cache = Cache::new(tmp.path().to_path_buf());
        let repo = "acme/widget-GGUF";

        assert!(
            cached(&cache, &pinned(repo, "main"), "widget.gguf").is_none(),
            "an empty cache must report a miss, not a phantom hit"
        );

        // Lay the file down the way hf-hub itself does — a ref pointing at a
        // commit, and the file under that commit's snapshot — so this exercises
        // the real lookup rather than a re-implementation of its path rules.
        let commit = "0123456789abcdef0123456789abcdef01234567";
        cache
            .model(repo.to_string())
            .create_ref(commit)
            .expect("create ref");
        let snapshot = tmp
            .path()
            .join(Repo::model(repo.to_string()).folder_name())
            .join("snapshots")
            .join(commit);
        std::fs::create_dir_all(&snapshot).expect("mkdir");
        std::fs::write(snapshot.join("widget.gguf"), b"weights").expect("write");

        assert_eq!(
            cached(&cache, &pinned(repo, "main"), "widget.gguf"),
            Some(snapshot.join("widget.gguf")),
            "a file already in the cache must resolve from it"
        );
        // **A pinned revision resolves to that revision's snapshot, not to
        // whatever the ref happens to point at.**
        assert_eq!(
            cached(&cache, &pinned(repo, commit), "widget.gguf"),
            Some(snapshot.join("widget.gguf")),
            "the pinned commit's own snapshot did not resolve"
        );
        assert!(
            cached(&cache, &pinned(repo, "cafebabe"), "widget.gguf").is_none(),
            "a revision the cache does not hold reported a hit — the pin is being ignored"
        );
    }
}
