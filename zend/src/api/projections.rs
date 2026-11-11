//! The projection panel's data, fetched when the panel opens.
//!
//! `GET /v1/conversations/{id}/projections/{turn}/{event}` — one recorded
//! projection point in full, with its panel context.
//! `POST /v1/conversations/{id}/projection-context` — the panel context alone,
//! for a point the client already holds in full.
//!
//! A conversation's history carries its projection points light
//! ([`ProjectionSpanOut`](crate::projection_event::ProjectionSpanOut)): the
//! fields the timeline draws, addressed by `(turn, event)`. The panel opens on
//! one point and asks for it here — its selection and materialized spine, plus
//! the context it renders beside them: the dialect glue, the authored text of
//! the system-prompt sections, and the bodies of the turns that point selected.
//! Sent with the history, all of that made a dozen-turn conversation a megabyte
//! and a half, nearly all of it for a panel that is rarely opened.
//!
//! A point streamed live during a decode arrives in full but unaddressed — its
//! turn has not sealed, and the history that would address it has not been
//! fetched — so for that one the client posts the selected turns and those of
//! its materialized spine, and asks for the context alone.

use std::collections::HashSet;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::Response,
    Json,
};
use candle_conversation::projection::{MaterializedPiece, ProjectionEvent, TimelineId};
use candle_conversation::turn_layout::TurnLayout;
use candle_conversation::GlueMarkers;
use serde::{Deserialize, Serialize};

use super::compressed;
use crate::projection_event::ProjectionEventOut;
use crate::session::ZendSession;

pub async fn get(
    State(session): State<Arc<ZendSession>>,
    Path((id, turn, event)): Path<(String, usize, usize)>,
    headers: HeaderMap,
) -> Result<Response, StatusCode> {
    // Every part of the answer reads the loaded engine; the glue is the gate.
    let glue = session
        .glue_markers()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let span = session
        .conversation_projections(&id)
        .into_iter()
        .nth(turn)
        .and_then(|points| points.into_iter().nth(event))
        .ok_or(StatusCode::NOT_FOUND)?;
    let context = panel_context(&session, &id, glue, panel_turns(&span.event));
    Ok(compressed::json(&DetailBody { span, context }, &headers))
}

pub async fn context(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Json(query): Json<ContextQuery>,
) -> Result<Response, StatusCode> {
    let glue = session
        .glue_markers()
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)?;
    let context = panel_context(&session, &id, glue, query.turns.into_iter());
    Ok(compressed::json(&context, &headers))
}

/// Every turn a point's panel draws a body for: those its selection names, and
/// those its materialized spine names.
///
/// The selection names what was chosen; the spine names what was injected,
/// and the cards are drawn from the spine. They differ where a dialogue holds
/// an ingest unit's opening by the unit's face: the record names the unit, the
/// spine the face (`Substrate::face_of`). Bodies for the selection alone left
/// every face's card without one — drawn blank, under the face's token count.
fn panel_turns(event: &ProjectionEvent) -> impl Iterator<Item = TurnKey> + '_ {
    let spine = event.materialized.iter().filter_map(|p| match p {
        MaterializedPiece::Turn { turn } => Some(turn),
        MaterializedPiece::Glue { .. } => None,
    });
    event.selection.turns.iter().chain(spine).map(|t| TurnKey {
        group: t.group.clone(),
        timeline: t.timeline,
        index: t.index,
    })
}

/// Everything the panel renders beside a point's selection.
fn panel_context(
    session: &ZendSession,
    conv_id: &str,
    glue: GlueMarkers,
    turns: impl Iterator<Item = TurnKey>,
) -> PanelContext {
    PanelContext {
        glue: Glue::from(glue),
        section_content: session
            .section_content(conv_id)
            .unwrap_or_default()
            .into_iter()
            .map(|(name, content)| SectionContent { name, content })
            .collect(),
        turn_content: turn_bodies(session, turns),
        target_layer: session.target_layer_name().unwrap_or_default(),
    }
}

/// The body of every selected turn — memory tiers and the dialogue, summary
/// nodes included — read from the substrate, so the panel renders the
/// materialized K/V exactly as selected (a summary in place of the turns it
/// replaced), not the raw message history. The live user message (`u32::MAX`)
/// has no sealed body and is skipped.
fn turn_bodies(session: &ZendSession, turns: impl Iterator<Item = TurnKey>) -> Vec<TurnContent> {
    let mut seen: HashSet<(String, u64, u32)> = HashSet::new();
    let mut out = Vec::new();
    for t in turns {
        if t.index == u32::MAX {
            continue;
        }
        // Resolve the body by the turn's STAMPED timeline identity
        // (`SelectedTurn::timeline`), never by group: the shared substrate
        // registers many conversations under one group, so a group→timeline
        // lookup is non-deterministic — and the dedup / panel key MUST carry the
        // timeline too, because one group routinely holds same-index turns from
        // different file conversations. A turn with no stamped timeline (only
        // the live user message) is skipped.
        let Some(timeline) = t.timeline.and_then(TimelineId::from_raw) else {
            continue;
        };
        if !seen.insert((t.group.clone(), timeline.raw(), t.index)) {
            continue;
        }
        // The whole turn, continuous (what the panel renders). A turn whose
        // `Tokens` record was lost (async writer + hard kill) decodes no full
        // text but still carries its layout text — emit the entry whenever ANY
        // body source resolves, and let the panel fall back from `text` to the
        // halves.
        let text = session.resolve_turn_full_text(timeline, t.index);
        let (user, assistant) = session
            .resolve_turn_text(timeline, t.index)
            .unwrap_or_default();
        let layout = session.turn_layout(timeline, t.index);
        if text.is_some() || !user.is_empty() || !assistant.is_empty() || layout.is_some() {
            out.push(TurnContent {
                group: t.group,
                timeline: timeline.raw(),
                index: t.index,
                text,
                user,
                assistant,
                layout,
            });
        }
    }
    out
}

/// The turns a live point selected, as the client holds them.
#[derive(Deserialize)]
pub struct ContextQuery {
    pub turns: Vec<TurnKey>,
}

/// A selected turn's identity — the fields of `SelectedTurn` that key its body.
/// The client posts its `selection.turns` and its spine's turns as they are;
/// the rest are ignored.
#[derive(Deserialize)]
pub struct TurnKey {
    pub group: String,
    #[serde(default)]
    pub timeline: Option<u64>,
    pub index: u32,
}

#[derive(Serialize)]
pub struct DetailBody {
    /// The point in full — its selection and materialized spine included.
    pub span: ProjectionEventOut,
    #[serde(flatten)]
    pub context: PanelContext,
}

#[derive(Serialize)]
pub struct PanelContext {
    /// Dialect framing markers — the glue the assembler wraps around the prompt
    /// and turns.
    pub glue: Glue,
    /// Authored content for every schema section, keyed by name; the panel shows
    /// a section's text when it is expanded.
    pub section_content: Vec<SectionContent>,
    /// Verbatim bodies of the selected turns, keyed by `(group, timeline,
    /// index)`; the panel expands a turn to show its text.
    pub turn_content: Vec<TurnContent>,
    /// The target layer's name (e.g. `dialogue`) — the panel prefixes the
    /// conversation messages with it.
    pub target_layer: String,
}

/// One projected turn's body, read from the substrate on demand. `text` is the
/// ENTIRE turn as one continuous string — the full sealed token range decoded
/// verbatim (user content, the baked intra-turn boundary, and assistant content)
/// — which the panel renders as a single card; the turn is stored continuously,
/// so this is the truth, not two re-glued halves. `text` is absent when the
/// turn's `Tokens` record was lost (async writer + hard kill) — the panel then
/// renders the layout-derived `user`/`assistant` halves instead.
#[derive(Serialize)]
pub struct TurnContent {
    pub group: String,
    /// The turn's resolved timeline identity — part of the panel key, because
    /// one group holds many conversations and indices repeat across them.
    pub timeline: u64,
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    pub user: String,
    pub assistant: String,
    /// The turn's segment-vector layout (real/ethereal glue, user, thinking,
    /// assistant) — the complete K/V description, surfaced so the panel renders
    /// the exact segments instead of re-splitting the text on markers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layout: Option<TurnLayout>,
}

/// The dialect framing markers the assembler wraps around the prompt and turns.
///
/// These are the role markers the backend frames turns with, plus BOTH halves of
/// thinking suppression — `no_think`, the soft-switch emitted as live glue right
/// after `user_start`, and `no_think_block`, the already-closed
/// `<think></think>` prefilled straight after `assistant_start`. A dialect uses
/// exactly one of them, so on any given model one of the two is empty.
///
/// The block half used to be omitted here on the reasoning that it was "never
/// glue — a suppressed turn decodes its own empty block into the body". That
/// stopped being true when suppression became structural: the block is prefilled
/// into the grid, never decoded, so a panel without it rendered less than the
/// turn actually holds on precisely the family that relies on it.
#[derive(Serialize)]
pub struct Glue {
    pub system_start: String,
    pub system_end: String,
    pub user_start: String,
    pub user_end: String,
    pub assistant_start: String,
    pub assistant_end: String,
    pub no_think: String,
    pub no_think_block: String,
}

impl From<GlueMarkers> for Glue {
    fn from(m: GlueMarkers) -> Self {
        Glue {
            system_start: m.system_start,
            system_end: m.system_end,
            user_start: m.user_start,
            user_end: m.user_end,
            assistant_start: m.assistant_start,
            assistant_end: m.assistant_end,
            no_think: m.no_think,
            no_think_block: m.no_think_block,
        }
    }
}

#[derive(Serialize)]
pub struct SectionContent {
    pub name: String,
    pub content: String,
}

#[cfg(test)]
mod tests {
    use candle_conversation::projection::{
        MaterializedPiece, ProjectionEvent, ProjectionSelection, SelectedTurn,
    };
    use candle_conversation::summary_tree::TurnKind;

    use super::panel_turns;

    fn turn(timeline: u64, index: u32, tokens: u32) -> SelectedTurn {
        SelectedTurn {
            layer: "dialogue".into(),
            group: "primary_conversation".into(),
            index,
            role: "assistant".into(),
            tokens,
            kind: TurnKind::Normal,
            reason: None,
            timeline: Some(timeline),
            selected: true,
            score: 0.0,
            qualified: false,
        }
    }

    /// **A face's card gets its body**: the selection names the unit's
    /// opening, the spine the face the model was shown, and the panel asks
    /// for both, so the card drawn from the spine finds its text.
    #[test]
    fn the_panel_asks_for_the_spines_turns_as_well_as_the_selections() {
        let unit = 1_881_470_790_753_706_315;
        let face = 1_881_470_905_646_194_696;
        let event = ProjectionEvent {
            selection: ProjectionSelection {
                turns: vec![turn(unit, 0, 45), turn(unit, 1, 743)],
                ..ProjectionSelection::default()
            },
            materialized: vec![
                MaterializedPiece::Glue {
                    text: "<|im_start|>".into(),
                },
                MaterializedPiece::Turn {
                    turn: turn(face, 0, 45),
                },
                MaterializedPiece::Turn {
                    turn: turn(unit, 1, 743),
                },
            ],
            ..ProjectionEvent::default()
        };
        let keys: Vec<(u64, u32)> = panel_turns(&event)
            .map(|k| (k.timeline.unwrap(), k.index))
            .collect();
        assert_eq!(keys, vec![(unit, 0), (unit, 1), (face, 0), (unit, 1)]);
    }
}
