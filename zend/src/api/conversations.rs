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

    // Uploaded files recorded against this conversation (substrate event),
    // grouped by their turn position so a burst dropped together tiles into
    // one inline marker. Recovered with the conversation, so they replay on
    // resume. Also returned as a flat `uploads` list for the files pane.
    let recovered_uploads = session.conversation_uploads(&id);
    let uploads: Vec<UploadOut> = recovered_uploads.iter().map(UploadOut::from).collect();
    let mut groups: std::collections::BTreeMap<u32, Vec<UploadOut>> =
        std::collections::BTreeMap::new();
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
    for (role, content, no_think) in history {
        turn_no += 1;
        for (r, c) in crate::chatml::split_turn(role, &content) {
            // The turn's `no_think` belongs on the USER bubble only — a bundled
            // turn can split into both roles, so tag the assistant half `false`.
            let user_no_think = no_think && r == Role::User;
            messages.push(HistoryMessage {
                role: role_str(r),
                content: c,
                no_think: user_no_think,
                spans: Vec::new(),
                files: Vec::new(),
            });
        }
        emit_uploads(&mut messages, turn_no);
    }
    // Any uploads recorded past the last turn (uploaded after the final turn)
    // append at the end.
    emit_uploads(&mut messages, u32::MAX);

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

    // Bodies for EVERY projected turn — memory tiers AND the dialogue, including
    // summary nodes — read from the substrate so the projection panel renders the
    // materialized KV exactly as selected (summaries shown in place of the turns
    // they replaced), not the raw message history. Deduped across spans. The live
    // user message (`u32::MAX`) has no sealed body and is skipped.
    let mut seen: std::collections::HashSet<(String, u32)> = std::collections::HashSet::new();
    let mut turn_content: Vec<TurnContent> = Vec::new();
    for span_list in &buckets {
        for ev in span_list {
            for t in &ev.event.selection.turns {
                if t.index == u32::MAX {
                    continue;
                }
                if seen.insert((t.group.clone(), t.index)) {
                    // Resolve the body by the turn's STAMPED timeline identity
                    // (`SelectedTurn::timeline`), never by group: the shared
                    // substrate registers many conversations under one group, so a
                    // group→timeline lookup is non-deterministic. A turn with no
                    // stamped timeline (only the live user message) is skipped.
                    let Some(timeline) = t
                        .timeline
                        .and_then(candle_conversation::projection::TimelineId::from_raw)
                    else {
                        continue;
                    };
                    // The whole turn, continuous (what the panel renders). Fall
                    // back to the split halves only to populate the legacy fields.
                    let text = session.resolve_turn_full_text(timeline, t.index);
                    let (user, assistant) = session
                        .resolve_turn_text(timeline, t.index)
                        .unwrap_or_default();
                    let layout = session.turn_layout(timeline, t.index);
                    if let Some(text) = text {
                        turn_content.push(TurnContent {
                            group: t.group.clone(),
                            index: t.index,
                            text,
                            user,
                            assistant,
                            layout,
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
        uploads,
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
    /// Every file uploaded to this conversation (recovered from the
    /// substrate), newest-last — hydrates the files pane on resume.
    pub uploads: Vec<UploadOut>,
}

/// One projected turn's body, read from the substrate on demand. `text` is the
/// ENTIRE turn as one continuous string — the full sealed token range decoded
/// verbatim (user content, the baked intra-turn boundary, and assistant content)
/// — which the panel renders as a single card; the turn is stored continuously,
/// so this is the truth, not two re-glued halves. `user`/`assistant` are the
/// legacy split halves, retained only for the pre-materialized fallback.
#[derive(Serialize)]
pub struct TurnContent {
    pub group: String,
    pub index: u32,
    pub text: String,
    pub user: String,
    pub assistant: String,
    /// The turn's segment-vector layout (real/ethereal glue, user, thinking,
    /// assistant) — the complete K/V description, surfaced so the panel renders
    /// the exact segments instead of re-splitting the text on markers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layout: Option<candle_conversation::turn_layout::TurnLayout>,
}

/// The dialect framing markers the assembler wraps around the prompt and turns.
/// These are the role markers the backend frames turns with, plus `no_think` —
/// the `/no_think` soft-switch the scheduler emits as live glue right after
/// `user_start` on a suppressed turn. The reasoning *block* is deliberately NOT
/// here: it's never glue (a suppressed turn decodes its own empty
/// `<think></think>` into the body), only the `/no_think` directive is.
#[derive(Serialize)]
pub struct Glue {
    pub system_start: String,
    pub system_end: String,
    pub user_start: String,
    pub user_end: String,
    pub assistant_start: String,
    pub assistant_end: String,
    pub no_think: String,
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
            no_think: m.no_think,
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
    /// Whether this turn was generated with thinking suppressed (the `/no_think`
    /// dial active at submit).  Set on USER bubbles; the GUI re-renders the
    /// `/no_think` soft-switch (`Glue.no_think`) right after `user_start` on each
    /// prior user bubble where this is true — mirroring what the engine's
    /// assembler now injects into the real model input.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub no_think: bool,
    /// Projection-event timeline for this bubble (assistant turns only).
    /// Omitted from the wire when empty.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub spans: Vec<crate::projection_event::ProjectionEventOut>,
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
    pub stats: Option<crate::session::UploadStats>,
}

impl From<&crate::session::UploadInfo> for UploadOut {
    fn from(u: &crate::session::UploadInfo) -> Self {
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
