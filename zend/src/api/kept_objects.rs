//! `GET /v1/substrate/zend/objects/export` and `POST
//! /v1/substrate/zend/objects/import` — every
//! kept custom object out as JSON Lines and back
//! (`candle_conversation::persistence::kept_objects`), for a substrate moved
//! aside under a new model key (`docs/hit_driven_projection.md` §13.4).

use std::sync::Arc;

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio::task::spawn_blocking;

use crate::session::ZendSession;

/// The daemon an export names.
const DAEMON: &str = "zend";

/// Every kept object, its tombstones and the blobs it names. Read off the
/// async runtime: it reads every blob a kept object names.
pub async fn export(State(session): State<Arc<ZendSession>>) -> Response {
    let Some(shared) = session.shared_substrate() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "the substrate is not open").into_response();
    };
    match spawn_blocking(move || shared.export_kept_objects(DAEMON)).await {
        Ok(Ok(text)) => ([(header::CONTENT_TYPE, "application/x-ndjson")], text).into_response(),
        Ok(Err(e)) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("export failed: {e:#}"),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("export task: {e}"),
        )
            .into_response(),
    }
}

/// Write the union of the exports the body holds, one after another — one
/// from each member — leaving every key already live as it is, and answer
/// what was written.
pub async fn import(State(session): State<Arc<ZendSession>>, body: String) -> Response {
    let Some(shared) = session.shared_substrate() else {
        return (StatusCode::SERVICE_UNAVAILABLE, "the substrate is not open").into_response();
    };
    match spawn_blocking(move || shared.import_kept_objects(&body, DAEMON)).await {
        Ok(Ok(report)) => Json(report).into_response(),
        Ok(Err(e)) => (StatusCode::BAD_REQUEST, format!("import refused: {e:#}")).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("import task: {e}"),
        )
            .into_response(),
    }
}
