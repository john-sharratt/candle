use std::sync::Arc;

use axum::{
    body::Body,
    http::{header, Request, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use include_dir::{include_dir, Dir};

use crate::session::ZendSession;

pub mod chat;
pub mod conversations;
pub mod files;
pub mod models;
pub mod status;
pub mod ws_logs;

static WEB: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/web");

/// Stable identifier for the embedded web build — a hash of the served UI
/// assets, computed once. The frontend captures this on load and forces a
/// reload when it changes (i.e. the daemon was rebuilt with new HTML/JS) so a
/// hot rebuild doesn't leave a stale UI talking to a fresh daemon.
pub fn build_id() -> &'static str {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::sync::OnceLock;
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(|| {
        let mut h = DefaultHasher::new();
        for name in [
            "index.html",
            "zend-api.js",
            "zend-api.mock.js",
            "zend-api.live.js",
        ] {
            if let Some(f) = WEB.get_file(name) {
                f.contents().hash(&mut h);
            }
        }
        format!("{:016x}", h.finish())
    })
}

/// Build the axum router.
pub fn router(session: Arc<ZendSession>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat::completions))
        .route("/v1/models", get(models::list))
        .route("/v1/status", get(status::status))
        .route("/v1/debug/maintenance", post(status::force_maintenance))
        .route("/v1/conversations", get(conversations::list))
        .route(
            "/v1/conversations/:id",
            get(conversations::get).delete(conversations::delete),
        )
        .route(
            "/v1/conversations/:id/files",
            get(files::list).post(files::upload),
        )
        .route(
            "/v1/conversations/:id/files/:file_id",
            get(files::content).delete(files::delete),
        )
        .route(
            "/v1/conversations/:id/archive",
            post(conversations::archive),
        )
        .route("/ws/logs", get(ws_logs::handler))
        .with_state(session)
        .fallback(embedded_asset)
}

async fn embedded_asset(req: Request<Body>) -> Response {
    let path = req.uri().path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };

    match WEB.get_file(path) {
        Some(f) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            // `no-store` so a hot rebuild never leaves the browser running a
            // cached `index.html` against a stale `zend-api.*.js` (or vice
            // versa) — the two would disagree on the API surface and silently
            // break (e.g. a method the new HTML calls is absent in old JS).
            (
                [
                    (header::CONTENT_TYPE, mime.as_ref()),
                    (header::CACHE_CONTROL, "no-store"),
                ],
                f.contents(),
            )
                .into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}
