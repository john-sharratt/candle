//! `GET /v1/conversations` — sidebar population.
//! `GET /v1/conversations/{id}` — recovered turn history, with its projection
//!   points light; the projection panel fetches one point in full from
//!   `GET /v1/conversations/{id}/projections/{turn}/{event}` (see `projections`).
//! `POST /v1/conversations/{id}/archive` — archive (one-way): set archived = true
//!   and mark the timeline for TextOnly distillation. There is no unarchive —
//!   distillation drops the KV, so an archived conversation can't be resumed.
//! `DELETE /v1/conversations/{id}` — tombstone (permanent; reclaimed at compaction).
//!
//! Archive appends a `RecordType::ConvState` record (last-writer-wins) plus a
//! `RecordType::Distilled` marker, and updates the in-RAM substrate. The sidebar
//! filters archived entries out unless `?include_archived=true` is set on the
//! list call — that's the "show archived" checkbox at the bottom of the sidebar.

use std::collections::BTreeMap;
use std::sync::Arc;

use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::Response,
    Json,
};
use candle_conversation::turn_layout::ThinkingLength;
use candle_conversation::Role as TurnRole;
use serde::{Deserialize, Serialize};

use super::compressed;
use crate::chatml::split_turn;
use crate::projection_event::ProjectionSpanOut;
use crate::session::{ConvEntry, ConversationDials, UploadInfo, UploadStats, ZendSession};
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
    match session.archive_conversation(&id) {
        Some(Ok(())) => Ok(StatusCode::NO_CONTENT),
        Some(Err(e)) => {
            tracing::warn!(conv_id = %id, "archive failed: {e}");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
        // Model not loaded yet — same shape as `get` returns.
        None => Err(StatusCode::SERVICE_UNAVAILABLE),
    }
}

pub async fn delete(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    match session.tombstone_conversation(&id) {
        Some(Ok(())) => Ok(StatusCode::NO_CONTENT),
        Some(Err(e)) => {
            tracing::warn!(conv_id = %id, "tombstone failed: {e}");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
        // Model not loaded yet — same shape as `get`/`archive` return.
        None => Err(StatusCode::SERVICE_UNAVAILABLE),
    }
}

pub async fn get(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Response, StatusCode> {
    let history = session
        .conversation_history(&id)
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;

    // Uploaded files recorded against this conversation (substrate event),
    // grouped by their turn position so a burst dropped together tiles into
    // one inline marker. Recovered with the conversation, so they replay on
    // resume. Also returned as a flat `uploads` list for the files pane.
    let recovered_uploads = session.conversation_uploads(&id);
    let uploads: Vec<UploadOut> = recovered_uploads.iter().map(UploadOut::from).collect();
    let mut groups: BTreeMap<u32, Vec<UploadOut>> = BTreeMap::new();
    for u in &recovered_uploads {
        groups
            .entry(u.turn_index)
            .or_default()
            .push(UploadOut::from(u));
    }
    // Emit an `upload` marker message for every upload group whose turn
    // boundary is `<= boundary`, draining them in turn order.
    let mut emit_uploads = |messages: &mut Vec<HistoryMessage>, boundary: u32| {
        while let Some(&k) = groups.keys().next() {
            if k > boundary {
                break;
            }
            let files = groups.remove(&k).unwrap();
            messages.push(HistoryMessage {
                role: "upload",
                content: String::new(),
                no_think: false,
                thinking: None,
                tool_tokens: Vec::new(),
                spans: Vec::new(),
                files,
            });
        }
    };

    // Each recovered turn is one stored ChatML stream; split it back into
    // role-attributed bubbles server-side (docs/zend_ui_redesign.md decision 9)
    // so the client renders one bubble per role without any ChatML parsing.
    // Upload markers are interleaved at their recorded turn boundaries.
    let mut messages: Vec<HistoryMessage> = Vec::new();
    emit_uploads(&mut messages, 0); // uploads before the first turn
    let mut turn_no: u32 = 0;
    for entry in history {
        turn_no += 1;
        // The reasoning length belongs to the first assistant bubble the turn
        // splits into — the one its thinking block renders in.
        let mut thinking = entry.thinking;
        // Likewise the tool results' lengths, to the first user bubble.
        let mut tool_tokens = entry.tool_tokens;
        for (r, c) in split_turn(chat_role(entry.role), &entry.text) {
            // The turn's `no_think` belongs on the USER bubble only — a bundled
            // turn can split into both roles, so tag the assistant half `false`.
            let user_no_think = entry.no_think && r == Role::User;
            let thinking = if r == Role::Assistant {
                thinking.take()
            } else {
                None
            };
            let tool_tokens = if r == Role::User {
                std::mem::take(&mut tool_tokens)
            } else {
                Vec::new()
            };
            messages.push(HistoryMessage {
                role: role_str(r),
                content: c,
                no_think: user_no_think,
                thinking,
                tool_tokens,
                spans: Vec::new(),
                files: Vec::new(),
            });
        }
        emit_uploads(&mut messages, turn_no);
    }
    // Any uploads recorded past the last turn (uploaded after the final turn)
    // append at the end.
    emit_uploads(&mut messages, u32::MAX);

    // Re-attach the recorded projection points, light. Records correspond to
    // the most recent decodes, so align them to the *trailing* assistant
    // bubbles — that way conversations recovered from disk (no records) keep
    // their older turns dot-free without shifting the mapping. Each point keeps
    // its `(turn, event)` address so the panel can fetch it in full.
    let records = session.conversation_projections(&id);
    let assistant_idxs: Vec<usize> = messages
        .iter()
        .enumerate()
        .filter(|(_, m)| m.role == "assistant")
        .map(|(i, _)| i)
        .collect();
    let take = records.len().min(assistant_idxs.len());
    for j in 0..take {
        let mi = assistant_idxs[assistant_idxs.len() - take + j];
        let turn = records.len() - take + j;
        messages[mi].spans = records[turn]
            .iter()
            .enumerate()
            .map(|(event, point)| ProjectionSpanOut::of(point, turn, event))
            .collect();
    }

    let title = session.conversation_label(&id);
    let dials = session.conversation_dials(&id);
    Ok(compressed::json(
        &HistoryBody {
            id,
            title,
            messages,
            uploads,
            dials,
        },
        &headers,
    ))
}

fn chat_role(role: TurnRole) -> Role {
    match role {
        TurnRole::User => Role::User,
        TurnRole::Assistant => Role::Assistant,
        TurnRole::System => Role::System,
    }
}

fn role_str(role: Role) -> &'static str {
    match role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::System => "system",
        Role::Tool => "tool",
    }
}

#[derive(Serialize)]
pub struct ListBody {
    pub conversations: Vec<ConvEntry>,
}

#[derive(Serialize)]
pub struct HistoryBody {
    pub id: String,
    /// The conversation's title, when something has labelled it — the same
    /// label the sidebar lists.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    pub messages: Vec<HistoryMessage>,
    /// Every file uploaded to this conversation (recovered from the
    /// substrate), newest-last — hydrates the files pane on resume.
    pub uploads: Vec<UploadOut>,
    /// The composer dials this conversation last ran under. The GUI sets its
    /// dials from these when the conversation is opened, so a conversation
    /// continues at the settings it was held at rather than at whatever the
    /// composer happened to be showing. Absent for one that has never taken a
    /// turn — the client keeps its own defaults there.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dials: Option<ConversationDials>,
}

#[derive(Serialize)]
pub struct HistoryMessage {
    pub role: &'static str,
    pub content: String,
    /// Whether this turn was generated with thinking suppressed (the `/no_think`
    /// dial active at submit).  Set on USER bubbles; the GUI re-renders the
    /// `/no_think` soft-switch (`Glue.no_think`) right after `user_start` on each
    /// prior user bubble where this is true — mirroring what the engine's
    /// assembler now injects into the real model input.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub no_think: bool,
    /// On an assistant bubble whose turn reasoned: how long the reasoning was,
    /// in tokens — exact from the turn's record, or estimated (`exact: false`)
    /// when its K/V was dropped. The thinking block shows it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingLength>,
    /// On a user bubble that carries tool results: each result's length in
    /// tokens as it sits in the context, one per `<tool_response>` block in
    /// order. The expanded tool card's output header shows it.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_tokens: Vec<u32>,
    /// Projection points for this bubble (assistant turns only), light — the
    /// timeline's fields and each point's address. Omitted from the wire when
    /// empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub spans: Vec<ProjectionSpanOut>,
    /// Uploaded files — set only on `role: "upload"` marker messages, which
    /// the GUI renders as an inline row of clickable file tiles. Omitted
    /// (empty) on ordinary user/assistant bubbles.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<UploadOut>,
}

/// One uploaded file as the history/pane wire shape — the metadata the GUI
/// needs to render a tile and open the file's content by `id`.
#[derive(Serialize)]
pub struct UploadOut {
    pub id: u64,
    pub name: String,
    pub ext: String,
    pub kind: String,
    pub size: String,
    pub added: String,
    /// Measured throughput of the upload batch (shared by every file dropped
    /// together). Absent on older events or model-less uploads; drives the
    /// inline stat line and the file viewer's upload-time note.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<UploadStats>,
}

impl From<&UploadInfo> for UploadOut {
    fn from(u: &UploadInfo) -> Self {
        UploadOut {
            id: u.id,
            name: u.name.clone(),
            ext: u.ext.clone(),
            kind: u.kind.clone(),
            size: u.size.clone(),
            added: u.added.clone(),
            stats: u.stats.clone(),
        }
    }
}
