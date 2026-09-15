//! JSON replies, gzipped for a client that takes it.
//!
//! axum's `Json` sends the bytes as serialised. The conversation replies are
//! text that compresses around eight to one, and a phone reaching the daemon
//! through the gateway pays for every byte of them, so these go through the
//! same gzip the embedded assets use.

use axum::{
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde::Serialize;

/// `body` as JSON, compressed when `req` takes gzip.
pub fn json<T: Serialize>(body: &T, req: &HeaderMap) -> Response {
    match serde_json::to_vec(body) {
        Ok(bytes) => web::asset::respond_json(bytes, req),
        Err(e) => {
            tracing::error!("reply did not serialise: {e}");
            StatusCode::INTERNAL_SERVER_ERROR.into_response()
        }
    }
}
