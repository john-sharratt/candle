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
pub mod models;
pub mod status;
pub mod ws_logs;

static WEB: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/web");

/// Build the axum router.
pub fn router(session: Arc<ZendSession>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat::completions))
        .route("/v1/models", get(models::list))
        .route("/v1/status", get(status::status))
        .route("/v1/conversations", get(conversations::list))
        .route("/v1/conversations/:id", get(conversations::get))
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
            ([(header::CONTENT_TYPE, mime.as_ref())], f.contents()).into_response()
        }
        None => StatusCode::NOT_FOUND.into_response(),
    }
}
