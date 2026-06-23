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

fn set_archived(session: &ZendSession, id: &str, archived: bool) -> Result<StatusCode, StatusCode> {
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
    // Each recovered turn is one stored ChatML stream; split it back into
    // role-attributed bubbles server-side (docs/zend_ui_redesign.md decision 9)
    // so the client renders one bubble per role without any ChatML parsing.
    let mut messages: Vec<HistoryMessage> = history
        .into_iter()
        .flat_map(|(role, content)| crate::chatml::split_turn(role, &content))
        .map(|(role, content)| HistoryMessage {
            role: role_str(role),
            content,
            spans: Vec::new(),
        })
        .collect();

    // Re-attach projection-event timelines banked this daemon session. Buckets
    // correspond to the most recent decodes, so align them to the *trailing*
    // assistant bubbles — that way conversations recovered from disk (no
    // buckets) keep their older turns dot-free without shifting the mapping.
    let buckets = session.conversation_projections(&id);
    let assistant_idxs: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == "assistant")
        .map(|(i, _)| i)
        .collect();
    let take = buckets.len().min(assistant_idxs.len());
    for j in 0..take {
        let mi = assistant_idxs[assistant_idxs.len() - take + j];
        messages[mi].spans = buckets[buckets.len() - take + j].clone();
    }

    // Glue + section content are workspace-wide (the dialect markers and the
    // schema's authored section text) — returned here as first-class fields so
    // the projection panel renders the framing and expands sections with no
    // extra round-trip. Computed on demand; never persisted in the event.
    let glue = session.glue_markers().map(Glue::from);
    let section_content = session
        .section_content(&id)
        .unwrap_or_default()
        .into_iter()
        .map(|(name, content)| SectionContent { name, content })
        .collect();

    // Memory-tier turn bodies (every projected layer except the dialogue, whose
    // bodies the GUI already holds): read from the substrate so the panel can
    // expand them, exactly like section content. Deduped across spans.
    let mut seen: std::collections::HashSet<(String, u32)> = std::collections::HashSet::new();
    let mut turn_content: Vec<TurnContent> = Vec::new();
    for span_list in &buckets {
        for ev in span_list {
            for t in &ev.event.selection.turns {
                if t.layer == "dialogue" || t.index == u32::MAX {
                    continue;
                }
                if seen.insert((t.group.clone(), t.index)) {
                    if let Some((user, assistant)) = session.resolve_turn_text(&t.group, t.index) {
                        turn_content.push(TurnContent {
                            group: t.group.clone(),
                            index: t.index,
                            user,
                            assistant,
                        });
                    }
                }
            }
        }
    }

    let target_layer = session.target_layer_name().unwrap_or_default();

    Ok(Json(HistoryBody {
        id,
        messages,
        glue,
        section_content,
        turn_content,
        target_layer,
    }))
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
    /// Dialect framing markers — the glue the assembler wraps around the prompt
    /// and turns. `None` until the model is loaded.
    pub glue: Option<Glue>,
    /// Authored content for every schema section, keyed by name; the panel shows
    /// a section's text when it is expanded.
    pub section_content: Vec<SectionContent>,
    /// Verbatim bodies of projected memory-tier turns (non-dialogue layers),
    /// keyed by `(group, index)`; the panel expands a turn to show its text.
    pub turn_content: Vec<TurnContent>,
    /// The target layer's name (e.g. `dialogue`) — the panel prefixes the
    /// conversation messages with it.
    pub target_layer: String,
}

/// One projected turn's two halves, read from the substrate on demand. Returned
/// unframed; the GUI places the dialect glue around and between them.
#[derive(Serialize)]
pub struct TurnContent {
    pub group: String,
    pub index: u32,
    pub user: String,
    pub assistant: String,
}

/// The dialect framing markers the assembler wraps around the prompt and turns.
#[derive(Serialize)]
pub struct Glue {
    pub system_start: String,
    pub system_end: String,
    pub user_start: String,
    pub user_end: String,
    pub assistant_start: String,
    pub assistant_end: String,
}

impl From<candle_conversation::GlueMarkers> for Glue {
    fn from(m: candle_conversation::GlueMarkers) -> Self {
        Glue {
            system_start: m.system_start,
            system_end: m.system_end,
            user_start: m.user_start,
            user_end: m.user_end,
            assistant_start: m.assistant_start,
            assistant_end: m.assistant_end,
        }
    }
}

#[derive(Serialize)]
pub struct SectionContent {
    pub name: String,
    pub content: String,
}

#[derive(Serialize)]
pub struct HistoryMessage {
    pub role: &'static str,
    pub content: String,
    /// Projection-event timeline for this bubble (assistant turns only).
    /// Omitted from the wire when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub spans: Vec<crate::projection_event::ProjectionEventOut>,
}
