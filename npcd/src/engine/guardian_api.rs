//! `GET /v1/pulse/guardian` — what the guardian has found and done.

use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::api::{owner_of, Authored};
use crate::engine::no_engine;

/// The guardian's records, oldest first. `enabled: false` with no records when
/// no guardian is configured. Admin: it names every character.
pub async fn records(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("reading the guardian");
    };
    match rt.guardian_log() {
        Some(log) => Json(json!({ "enabled": true, "records": log.records() })).into_response(),
        None => Json(json!({ "enabled": false, "records": [] })).into_response(),
    }
}
