//! `GET /v1/conversations/{id}/prompt-tokens` — the tokens the conversation
//! would be given if it decoded now, piece by piece: each system-prompt section
//! and collection member by name, each turn by layer and index, and the glue
//! between them, with the decoded text of exactly those tokens and the score
//! that selected each piece (`candle_conversation::PromptDump`, the same view
//! npcd serves for a character).
//!
//! It answers what the model actually reads — a journal section is either in
//! it or not — where a reply describing its own context can only say what the
//! model believes. `404` when the conversation is not open on this member: the
//! projection is computed from its live slot.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::session::ZendSession;

pub async fn get(State(session): State<Arc<ZendSession>>, Path(id): Path<String>) -> Response {
    match session.prompt_dump(&id).await {
        Some(Some(dump)) => Json(dump).into_response(),
        Some(None) => (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": "not_open",
                "message": "the conversation is not open on this member — its prompt is \
                            projected from its live slot, so send it a message first",
            })),
        )
            .into_response(),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "error": "not_loaded", "message": "the model is not loaded yet" })),
        )
            .into_response(),
    }
}
