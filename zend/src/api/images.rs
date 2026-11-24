//! Image routes (`docs/flash_next_vision.md` §6.1).
//!
//!   POST /v1/images        the image's bytes as the body -> its id, size and rows
//!   GET  /v1/images/{id}   the bytes the image arrived as
//!
//! A kept image is read by the tower when it is kept, so a turn that names its
//! id in an `image_url` part reads its rows from the substrate and waits on no
//! tower. Both answer `503` until the model is loaded.
//!
//! An image is its sender's (`vision::tenancy`): the GET answers a person who
//! sent it, or an admin, and `404` to anyone else exactly as to an id nothing
//! is kept under.

use std::sync::Arc;

use axum::{
    body::Bytes,
    extract::{Path, State},
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Json, Response},
};
use candle_conversation::vision::prepare::mime_of;

use crate::mcp::caller::Caller;
use crate::session::ZendSession;

/// The largest image a request carries: past a 4096-token image's pixels at
/// any sane encoding.
pub const IMAGE_BYTES: usize = 32 << 20;

/// The chat route's body limit: a message's images as `data:` URLs, base64
/// a third over their bytes, beside its text.
pub const CHAT_BODY_LIMIT: usize = 4 * IMAGE_BYTES;

/// POST — keep the body as an image the caller sent.
pub async fn keep(
    State(session): State<Arc<ZendSession>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let Some(caller) = Caller::of(&headers, session.roles()) else {
        return StatusCode::FORBIDDEN.into_response();
    };
    let bytes = body.to_vec();
    let kept = tokio::task::spawn_blocking(move || session.keep_images(vec![bytes], &caller)).await;
    match kept {
        Ok(Some(Ok(mut images))) if images.len() == 1 => Json(images.remove(0)).into_response(),
        Ok(Some(Ok(images))) => refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "image_failed",
            &format!("one image kept as {}", images.len()),
        ),
        Ok(Some(Err(why))) => refused(StatusCode::BAD_REQUEST, "invalid_image", &why),
        Ok(None) => StatusCode::SERVICE_UNAVAILABLE.into_response(),
        Err(e) => refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "image_failed",
            &format!("keeping the image panicked: {e}"),
        ),
    }
}

/// GET — the bytes image `id` arrived as, typed by their format, for a caller
/// who sent it. The blob is read on the blocking pool.
pub async fn original(
    State(session): State<Arc<ZendSession>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Response {
    let Some(caller) = Caller::of(&headers, session.roles()) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let read = tokio::task::spawn_blocking(move || session.kept_image(&id, &caller)).await;
    match read {
        Ok(Some(Ok(Some(bytes)))) => {
            ([(header::CONTENT_TYPE, mime_of(&bytes))], bytes).into_response()
        }
        Ok(Some(Ok(None))) => StatusCode::NOT_FOUND.into_response(),
        Ok(Some(Err(why))) => refused(StatusCode::INTERNAL_SERVER_ERROR, "image_failed", &why),
        Ok(None) => StatusCode::SERVICE_UNAVAILABLE.into_response(),
        Err(e) => refused(
            StatusCode::INTERNAL_SERVER_ERROR,
            "image_failed",
            &format!("reading the image panicked: {e}"),
        ),
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
