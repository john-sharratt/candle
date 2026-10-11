//! The web UIs, served from disk (`docs/shared_model_host.md` §3.4).
//!
//! Every UI file is read from its directory when it is requested, so saving a
//! file is the deployment: the next request serves it, with no rebuild and no
//! restart. The shared front-end library (`web/content/common/`) is served once,
//! at `/common/`, for every UI.
//!
//! # `build`
//!
//! Both UIs reload themselves when `/v1/status`'s `build` changes. It is a hash
//! over every file under the content directories — each one's path, size and
//! modification time — the binary's version, and the running executable's own
//! size and modification time, computed when asked. Saving a file moves its
//! modification time, so an open page picks up an edit on its next status
//! poll, and a rebuilt binary changes it too. The walk reads metadata only,
//! never contents: a few hundred `stat`s per poll.
//!
//! # Where the UIs are
//!
//! In the source tree the binary was built from, by default — the daemons run
//! from their checkout, and a workspace folder elsewhere must not make a UI
//! vanish. A directory given on the command line resolves against the working
//! directory, as any path argument does.

use std::env::{current_dir, current_exe};
use std::fs::{self, Metadata};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use axum::extract::{Path as UrlPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use sha2::{Digest, Sha256};
use web::asset::respond;
use web::config::Cache;
use web::content::safe_rel;
use web::Roots;

/// Where the shared front-end library lives: `web/content/common` in the
/// source tree this binary was built from.
pub const COMMON_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../web/content/common");

/// Refuse a content directory that does not exist, naming it — a UI that would
/// serve nothing is a misconfiguration, not a quiet 404 per request.
pub fn require_dir(what: &str, dir: &Path) -> anyhow::Result<PathBuf> {
    dir.canonicalize()
        .ok()
        .filter(|d| d.is_dir())
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{what}: content directory {} does not exist (a relative path resolves \
                 against the working directory, {})",
                dir.display(),
                current_dir()
                    .map(|d| d.display().to_string())
                    .unwrap_or_default()
            )
        })
}

/// The build id over `dirs` as they are on disk now, `version`, and the
/// running executable.
pub fn build_id(dirs: &[PathBuf], version: &str) -> String {
    let mut files: Vec<(String, u64, u128)> = Vec::new();
    for (i, dir) in dirs.iter().enumerate() {
        walk(dir, &format!("{i}/"), &mut files);
    }
    // Sorted, because the id must depend on what is there and not on the order
    // a directory happened to be read in.
    files.sort_unstable();
    let mut h = Sha256::new();
    h.update(version.as_bytes());
    if let Some((len, mtime)) = current_exe().ok().and_then(|exe| stamp(&exe)) {
        h.update(len.to_le_bytes());
        h.update(mtime.to_le_bytes());
    }
    for (rel, len, mtime) in files {
        h.update(rel.as_bytes());
        h.update([0]);
        h.update(len.to_le_bytes());
        h.update(mtime.to_le_bytes());
    }
    let digest = h.finalize();
    digest[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// Every file under `dir`, depth first, as `(prefix + relative path, size,
/// modification time in nanoseconds)`. Hidden entries (a leading `.`) are not
/// part of any UI — editor swap files, metadata a tool drops beside the pages —
/// so they never move the build id.
fn walk(dir: &Path, prefix: &str, out: &mut Vec<(String, u64, u128)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let Ok(meta) = entry.metadata() else {
            continue;
        };
        let rel = format!("{prefix}{name}");
        if meta.is_dir() {
            walk(&entry.path(), &format!("{rel}/"), out);
        } else if meta.is_file() {
            out.push((rel, meta.len(), mtime_nanos(&meta)));
        }
    }
}

/// A file's size and modification time in nanoseconds.
fn stamp(path: &Path) -> Option<(u64, u128)> {
    let meta = fs::metadata(path).ok()?;
    Some((meta.len(), mtime_nanos(&meta)))
}

fn mtime_nanos(meta: &Metadata) -> u128 {
    meta.modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_nanos())
}

/// `/common/*path` — the shared front-end library, from disk.
pub async fn common(
    State(roots): State<Roots>,
    UrlPath(path): UrlPath<String>,
    headers: HeaderMap,
) -> Response {
    serve(&roots, &path, &headers).await
}

/// `/brand/*path` — the estate's marks, from the shared front-end's `brand/`.
/// The switcher addresses them as `/brand/<id>` on every host, so a site's
/// mark shows even while that site is down.
pub async fn brand(
    State(roots): State<Roots>,
    UrlPath(path): UrlPath<String>,
    headers: HeaderMap,
) -> Response {
    serve(&roots, &format!("brand/{path}"), &headers).await
}

/// `path` under `roots`, through `web::asset` — an entity tag, a conditional
/// answer, gzip and the short cache age; refused when it escapes the root.
async fn serve(roots: &Roots, path: &str, headers: &HeaderMap) -> Response {
    let Some(rel) = safe_rel(path) else {
        return StatusCode::BAD_REQUEST.into_response();
    };
    match roots.read(&rel).await {
        Some(bytes) => respond(&rel, bytes, Cache::default(), headers),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use axum::body::Body;
    use axum::http::Request;
    use axum::routing::get;
    use axum::Router;
    use tempfile::TempDir;
    use tower::ServiceExt;

    fn tree() -> TempDir {
        let t = TempDir::new().unwrap();
        fs::create_dir_all(t.path().join("lib")).unwrap();
        fs::write(t.path().join("index.html"), "<p>one</p>").unwrap();
        fs::write(t.path().join("lib").join("a.js"), "export const a = 1;").unwrap();
        t
    }

    /// The same tree gives the same id; a file rewritten, added or removed
    /// changes it; so does the version.
    #[test]
    fn the_build_id_follows_the_files_on_disk() {
        let t = tree();
        let dirs = vec![t.path().to_path_buf()];
        let a = build_id(&dirs, "1");
        assert_eq!(a, build_id(&dirs, "1"), "stable while nothing changes");
        assert_eq!(a.len(), 16);
        assert_ne!(a, build_id(&dirs, "2"), "a new binary is a new build");

        fs::write(t.path().join("index.html"), "<p>two, longer</p>").unwrap();
        let b = build_id(&dirs, "1");
        assert_ne!(a, b, "an edit is a new build");

        fs::write(t.path().join("lib").join("b.js"), "").unwrap();
        let c = build_id(&dirs, "1");
        assert_ne!(b, c, "a new file is a new build");

        fs::remove_file(t.path().join("lib").join("b.js")).unwrap();
        assert_ne!(c, build_id(&dirs, "1"), "a removed file is a new build");
    }

    /// A hidden file or directory — an editor's swap file, a tool's metadata —
    /// is not part of the UI and does not change the build.
    #[test]
    fn hidden_entries_do_not_move_the_build_id() {
        let t = tree();
        let dirs = vec![t.path().to_path_buf()];
        let a = build_id(&dirs, "1");
        fs::write(t.path().join(".substrate.yaml"), "x: 1").unwrap();
        fs::write(t.path().join("lib").join(".a.js.swp"), "swap").unwrap();
        fs::create_dir_all(t.path().join(".cache")).unwrap();
        fs::write(t.path().join(".cache").join("entry"), "c").unwrap();
        assert_eq!(a, build_id(&dirs, "1"));
    }

    #[test]
    fn a_missing_content_directory_is_named() {
        let t = tree();
        assert!(require_dir("zend", t.path()).is_ok());
        let err = require_dir("zend", &t.path().join("nope"))
            .unwrap_err()
            .to_string();
        assert!(err.starts_with("zend: content directory"), "{err}");
        assert!(err.contains("nope"), "{err}");
    }

    /// `/common/` reads the file at request time — an edit is served on the
    /// next request with the process untouched — and refuses an escape.
    #[tokio::test]
    async fn common_serves_from_disk_per_request() {
        let t = tree();
        let app = Router::new()
            .route("/common/*path", get(common))
            .with_state(Roots::disk(&[t.path().to_path_buf()]));
        let get_body = |uri: &'static str| {
            let app = app.clone();
            async move {
                let res = app
                    .oneshot(Request::get(uri).body(Body::empty()).unwrap())
                    .await
                    .unwrap();
                let status = res.status();
                let body = axum::body::to_bytes(res.into_body(), 1 << 20)
                    .await
                    .unwrap();
                (status, String::from_utf8_lossy(&body).into_owned())
            }
        };
        assert_eq!(
            get_body("/common/lib/a.js").await,
            (StatusCode::OK, "export const a = 1;".to_string())
        );
        fs::write(t.path().join("lib").join("a.js"), "export const a = 2;").unwrap();
        assert_eq!(get_body("/common/lib/a.js").await.1, "export const a = 2;");
        assert_eq!(
            get_body("/common/lib/none.js").await.0,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            get_body("/common/..%2fsecret").await.0,
            StatusCode::BAD_REQUEST
        );
    }

    /// `/brand/<id>` is the shared front-end's `brand/<id>`.
    #[tokio::test]
    async fn brand_serves_the_estates_marks() {
        let t = tree();
        fs::create_dir_all(t.path().join("brand")).unwrap();
        fs::write(t.path().join("brand").join("npcd.svg"), "<svg/>").unwrap();
        let app = Router::new()
            .route("/brand/*path", get(brand))
            .with_state(Roots::disk(&[t.path().to_path_buf()]));
        let res = app
            .oneshot(Request::get("/brand/npcd.svg").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let body = axum::body::to_bytes(res.into_body(), 1 << 20)
            .await
            .unwrap();
        assert_eq!(&body[..], b"<svg/>");
    }
}
