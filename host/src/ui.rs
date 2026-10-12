//! The web UIs, compiled in (`docs/shared_model_host.md` §3.4).
//!
//! Every tenant's UI and the shared front-end library (`web/content/common/`)
//! are embedded in the binary, so a shipped binary serves its own pages with no
//! checkout beside it. The shared library is served once, at `/common/`, for
//! every UI. Iterating on a page needs no rebuild of the daemon: the gateway
//! serves the same directories from disk (`web/web.yaml`) and forwards only
//! the API here.
//!
//! # `build`
//!
//! Both UIs reload themselves when `/v1/status`'s `build` changes. It is a hash
//! over every embedded file — its path and its bytes — and the binary's
//! version, taken once at startup: the content cannot change while the process
//! runs, so a new id means a new binary with a different UI.

use axum::extract::{Path as UrlPath, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use include_dir::{include_dir, Dir};
use sha2::{Digest, Sha256};
use web::asset::respond;
use web::config::Cache;
use web::content::safe_rel;
use web::Roots;

/// The shared front-end library, `web/content/common`.
pub static COMMON: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/../web/content/common");

/// The build id over `dirs` and `version`.
pub fn build_id(dirs: &[&'static Dir<'static>], version: &str) -> String {
    let mut files: Vec<(String, &'static [u8])> = Vec::new();
    for (i, dir) in dirs.iter().enumerate() {
        collect(dir, &format!("{i}/"), &mut files);
    }
    // Sorted, because the id must depend on what is there and not on the order
    // the embedding listed it in.
    files.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let mut h = Sha256::new();
    h.update(version.as_bytes());
    for (rel, bytes) in files {
        h.update(rel.as_bytes());
        h.update([0]);
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    }
    let digest = h.finalize();
    digest[..8].iter().map(|b| format!("{b:02x}")).collect()
}

/// Every file under `dir`, as `(prefix + its path, its bytes)`.
fn collect(dir: &'static Dir<'static>, prefix: &str, out: &mut Vec<(String, &'static [u8])>) {
    for f in dir.files() {
        let rel = f.path().to_string_lossy().replace('\\', "/");
        out.push((format!("{prefix}{rel}"), f.contents()));
    }
    for d in dir.dirs() {
        collect(d, prefix, out);
    }
}

/// `/common/*path` — the shared front-end library.
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
    use tower::ServiceExt;

    static BRAND: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/../web/content/common/brand");

    /// The same content gives the same id; a new version, or different
    /// content, gives another.
    #[test]
    fn the_build_id_follows_the_content_and_the_version() {
        let a = build_id(&[&COMMON], "1");
        assert_eq!(a, build_id(&[&COMMON], "1"), "stable while nothing changes");
        assert_eq!(a.len(), 16);
        assert_ne!(a, build_id(&[&COMMON], "2"), "a new version is a new build");
        assert_ne!(
            a,
            build_id(&[&BRAND], "1"),
            "other content is another build"
        );
    }

    async fn get_from(app: &Router, uri: &str) -> (StatusCode, Vec<u8>) {
        let res = app
            .clone()
            .oneshot(Request::get(uri).body(Body::empty()).unwrap())
            .await
            .unwrap();
        let status = res.status();
        let body = axum::body::to_bytes(res.into_body(), 1 << 22)
            .await
            .unwrap();
        (status, body.to_vec())
    }

    /// `/common/` serves the compiled-in library as it is on disk, answers a
    /// missing file with 404, and refuses an escape.
    #[tokio::test]
    async fn common_serves_the_compiled_in_library() {
        let app = Router::new()
            .route("/common/*path", get(common))
            .with_state(Roots::embedded(&[&COMMON]));
        let (status, body) = get_from(&app, "/common/base.css").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, COMMON.get_file("base.css").unwrap().contents());
        assert_eq!(
            get_from(&app, "/common/lib/none.js").await.0,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            get_from(&app, "/common/..%2fsecret").await.0,
            StatusCode::BAD_REQUEST
        );
    }

    /// `/brand/<id>` is the shared front-end's `brand/<id>`.
    #[tokio::test]
    async fn brand_serves_the_estates_marks() {
        let app = Router::new()
            .route("/brand/*path", get(brand))
            .with_state(Roots::embedded(&[&COMMON]));
        let first = BRAND
            .files()
            .next()
            .expect("the brand directory has a mark");
        let name = first.path().file_name().unwrap().to_string_lossy();
        let (status, body) = get_from(&app, &format!("/brand/{name}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, first.contents());
    }
}
