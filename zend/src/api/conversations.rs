//! `GET /v1/conversations` — sidebar population.
//! `GET /v1/conversations/{id}` — recovered turn history.
//! `POST /v1/conversations/{id}/archive` — set archived = true.
//! `POST /v1/conversations/{id}/unarchive` — set archived = false.
//!
//! Archive/unarchive append a `RecordType::ConvState` record
//! (last-writer-wins) and update the in-RAM substrate. The sidebar
//! filters archived entries out unless `?include_archived=true` is
//! set on the list call — that's the "show archived" checkbox at
//! the bottom of the sidebar.

use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};

use crate::session::{ConvEntry, ZendSession};
use crate::types::Role;

#[derive(Debug, Default, Deserialize)]
pub struct ListQuery {
    /// When true, archived conversations are included in the response.
    /// Defaults to false — the sidebar hides archived rows behind the
    /// "show archived" checkbox.
    #[serde(default)]
    pub include_archived: bool,
}

pub async fn list(
    State(session): State<Arc<ZendSession>>,
    Query(q): Query<ListQuery>,
) -> Json<ListBody> {
    Json(ListBody {
        conversations: session.list_conversations(q.include_archived),
    })
}

pub async fn archive(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    set_archived(&session, &id, true)
}

pub async fn unarchive(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    set_archived(&session, &id, false)
}

fn set_archived(
    session: &ZendSession,
    id: &str,
    archived: bool,
) -> Result<StatusCode, StatusCode> {
    match session.set_conversation_archived(id, archived) {
        Some(Ok(())) => Ok(StatusCode::NO_CONTENT),
        Some(Err(e)) => {
            tracing::warn!(conv_id = %id, "archive write failed: {e}");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
        // Model not loaded yet — same shape as `get` returns.
        None => Err(StatusCode::SERVICE_UNAVAILABLE),
    }
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
