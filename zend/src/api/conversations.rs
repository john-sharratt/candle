//! `GET /v1/conversations` and `GET /v1/conversations/{id}` — sidebar
//! population and recovered-history retrieval.
//!
//! - `list` returns the per-workspace label sidecar entries that still have
//!   recovered turns in the substrate.
//! - `get` decodes the on-disk token grid for a single conversation back to
//!   `{role, text}` turns the UI can render.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::Serialize;

use crate::session::{ConvEntry, ZendSession};
use crate::types::Role;

pub async fn list(State(session): State<Arc<ZendSession>>) -> Json<ListBody> {
    Json(ListBody {
        conversations: session.list_conversations(),
    })
}

pub async fn get(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
) -> Result<Json<HistoryBody>, StatusCode> {
    let history = session
        .conversation_history(&id)
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let messages = history
        .into_iter()
        .map(|(role, content)| HistoryMessage {
            role: role_str(role),
            content,
        })
        .collect();
    Ok(Json(HistoryBody { id, messages }))
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
    }
}

#[derive(Serialize)]
pub struct ListBody {
    pub conversations: Vec<ConvEntry>,
}

#[derive(Serialize)]
pub struct HistoryBody {
    pub id: String,
    pub messages: Vec<HistoryMessage>,
}

#[derive(Serialize)]
pub struct HistoryMessage {
    pub role: &'static str,
    pub content: String,
}
