//! Image routes (`docs/flash_next_vision.md` §6.1).
//!
//!   POST /v1/images        the image's bytes as the body -> its id, size and rows
//!   GET  /v1/images/{id}   the bytes the image arrived as
//!
//! A kept image is read by the tower when it is kept, so a turn that names its
//! id in an `image_url` part reads its rows from the substrate and waits on no
//! tower. Both answer `503` until the model is loaded.

use std::sync::Arc;

use axum::{
    body::Bytes,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
};
use candle_conversation::vision::prepare::mime_of;

use crate::session::ZendSession;

/// The largest image a request carries: past a 4096-token image's pixels at
/// any sane encoding.
pub const IMAGE_BYTES: usize = 32 << 20;

/// The chat route's body limit: a message's images as `data:` URLs, base64
/// a third over their bytes, beside its text.
pub const CHAT_BODY_LIMIT: usize = 4 * IMAGE_BYTES;

/// POST — keep the body as an image.
pub async fn keep(State(session): State<Arc<ZendSession>>, body: Bytes) -> Response {
    let bytes = body.to_vec();
    let kept = tokio::task::spawn_blocking(move || session.keep_image(bytes)).await;
    match kept {
        Ok(Some(Ok(image))) => Json(image).into_response(),
        Ok(Some(Err(why))) => refused(StatusCode::BAD_REQUEST, "invalid_image", &why),
        Ok(None) => StatusCode::SERVICE_UNAVAILABLE.into_response(),
        Err(e) => refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "image_failed",
            &format!("keeping the image panicked: {e}"),
        ),
    }
}

/// GET — the bytes image `id` arrived as, typed by their format.
pub async fn original(State(session): State<Arc<ZendSession>>, Path(id): Path<String>) -> Response {
    match session.kept_image(&id) {
        Some(Ok(Some(bytes))) => ([(header::CONTENT_TYPE, mime_of(&bytes))], bytes).into_response(),
        Some(Ok(None)) => StatusCode::NOT_FOUND.into_response(),
        Some(Err(why)) => refused(StatusCode::INTERNAL_SERVER_ERROR, "image_failed", &why),
        None => StatusCode::SERVICE_UNAVAILABLE.into_response(),
    }
}

fn refused(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(serde_json::json!({
            "error": { "message": message, "type": "invalid_request_error", "code": code }
        })),
    )
        .into_response()
}
