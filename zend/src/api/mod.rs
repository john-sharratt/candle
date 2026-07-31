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
pub mod substrate;
pub mod telemetry;
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
            "perf.html",
            "substrate.html",
            "project.html",
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
        .route("/v1/telemetry", get(telemetry::telemetry))
        .route("/v1/phases", get(telemetry::phases))
        .route("/v1/promotes", get(telemetry::promotes))
        .route("/v1/substrate", get(substrate::overview))
        .route("/v1/substrate/system-prompt", get(substrate::system_prompt))
        .route("/v1/substrate/tools", get(substrate::tools))
        .route("/v1/substrate/layer/:name", get(substrate::layer))
        .route(
            "/v1/substrate/layer/:name/toggle",
            post(substrate::toggle_layer),
        )
        .route("/v1/substrate/timeline/:tl", get(substrate::timeline))
        .route("/v1/substrate/project", post(substrate::project))
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
    let raw = req.uri().path().trim_start_matches('/');
    let want = if raw.is_empty() { "index.html" } else { raw };
    // Clean extensionless routes (`/perf`, `/substrate`) resolve to the matching
    // `<name>.html`, so those pages get pretty URLs without a `.html` suffix.
    let name: String = if WEB.get_file(want).is_some() {
        want.to_string()
    } else if !want.contains('.') && WEB.get_file(&format!("{want}.html")).is_some() {
        format!("{want}.html")
    } else {
        want.to_string()
    };

    match WEB.get_file(&name) {
        Some(f) => {
            let mime = mime_guess::from_path(&name).first_or_octet_stream();
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
