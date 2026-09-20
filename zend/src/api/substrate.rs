//! Read-only substrate viewer API (`/substrate.html`). A lazy hierarchy — the
//! overview is deliberately lightweight (storage + counts + layer metadata), and
//! each drill-down level is its own endpoint the page fetches only when the user
//! expands that node:
//!
//! - `GET /v1/substrate`                     — storage footprint, counts, layers.
//! - `GET /v1/substrate/system-prompt`       — the target layer's framing.
//! - `GET /v1/substrate/tools`               — the live tool catalog.
//! - `GET /v1/substrate/layer/{name}`        — the conversations in one layer.
//! - `GET /v1/substrate/timeline/{tl}`       — one conversation's summary forest.
//! - `GET /v1/substrate/timeline/{tl}/selection` — its most recent score-density
//!   selection: which nodes made the slot, why, and the pending/token/budget
//!   counters around it. Unlike the other routes this is NOT a substrate read —
//!   it is in-memory, last-write-wins per timeline, and empties on restart.
//!
//! All read the daemon's live `Substrate` through a cloned `Conversation` handle
//! (engine lock released immediately, only the substrate read guard held for the
//! walk), never a rebuild from the multi-GB redo log, and return `503` until the
//! model is loaded. They never mutate.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use candle_conversation::summary_tree::SelectionOrigin;
use candle_conversation::turn_layout::TurnLayout;

use crate::session::ZendSession;

pub async fn overview(
    State(session): State<Arc<ZendSession>>,
) -> Result<Json<SubstrateOverview>, StatusCode> {
    session
        .substrate_overview()
        .map(Json)
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)
}

pub async fn system_prompt(
    State(session): State<Arc<ZendSession>>,
) -> Result<Json<SystemPromptView>, StatusCode> {
    session
        .substrate_system_prompt()
        .map(Json)
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)
}

pub async fn tools(State(session): State<Arc<ZendSession>>) -> Result<Json<ToolsView>, StatusCode> {
    session
        .substrate_tools()
        .map(Json)
        .ok_or(StatusCode::SERVICE_UNAVAILABLE)
}

pub async fn layer(
    State(session): State<Arc<ZendSession>>,
    Path(name): Path<String>,
) -> Result<Json<LayerConversations>, StatusCode> {
    session
        .substrate_layer(&name)
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

/// `POST /v1/substrate/layer/:name/toggle` — flip a layer's runtime projection
/// kill switch (in-memory, non-persistent). Excluding a populated layer from
/// assembly lets you A/B whether it's the source of an incoherent response.
pub async fn toggle_layer(
    State(session): State<Arc<ZendSession>>,
    Path(name): Path<String>,
) -> Result<Json<LayerToggleResult>, StatusCode> {
    session
        .toggle_projection_layer(&name)
        .map(|disabled| Json(LayerToggleResult { name, disabled }))
        .ok_or(StatusCode::NOT_FOUND)
}

#[derive(Serialize)]
pub struct LayerToggleResult {
    pub name: String,
    /// The layer's new state after the flip (`true` = now excluded).
    pub disabled: bool,
}

pub async fn project(
    State(session): State<Arc<ZendSession>>,
    Json(req): Json<ProjectReq>,
) -> Result<Json<ProjectView>, StatusCode> {
    // The probe is a real GPU prefill + belief scan — keep it off the async
    // runtime's worker threads so an in-flight projection never stalls other
    // requests. `substrate_project` is synchronous and internally blocking.
    let out = tokio::task::spawn_blocking(move || session.substrate_project(&req.text))
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    out.map(Json).ok_or(StatusCode::SERVICE_UNAVAILABLE)
}

pub async fn timeline(
    State(session): State<Arc<ZendSession>>,
    Path(tl): Path<String>,
) -> Result<Json<TimelineDetail>, StatusCode> {
    // Timeline ids are raw u64 — carried as a string on the wire so JS never
    // rounds them through its 2^53 float mantissa.
    let raw: u64 = tl.parse().map_err(|_| StatusCode::BAD_REQUEST)?;
    session
        .substrate_timeline(raw)
        // Model not loaded, or no such timeline — either way there's nothing to
        // show; the viewer only requests this after a successful overview.
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

/// `GET /v1/substrate/timeline/{tl}/selection` — see the module doc.
pub async fn selection(
    State(session): State<Arc<ZendSession>>,
    Path(tl): Path<String>,
) -> Result<Json<SelectionView>, StatusCode> {
    let raw: u64 = tl.parse().map_err(|_| StatusCode::BAD_REQUEST)?;
    session
        .substrate_selection(raw)
        // Model not loaded, unknown timeline, or no projection has run for it
        // yet (or it used the rule-based path, which records no diagnostic).
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

/// `DELETE /v1/substrate/timeline/{tl}` — tombstone one timeline by its RAW id.
///
/// The conversation-scoped `DELETE /v1/conversations/{id}` cannot reach these.
/// It derives the timeline as `hash(conv_id)`, which only addresses dialogue
/// conversations; a derived-layer timeline (`repo_map`, `code_reading`) is
/// created with its own id and carries an empty `conv_id`, so there is no string
/// that hashes to it. This route takes the id the inspection views already
/// print.
///
/// Tombstoning is the supported repair for a conversation whose *records* are
/// intact but whose *content* is wrong — a directory ingest that sealed its
/// turns and lost the couplings joining them, a turn whose chunk count says its
/// seal was interrupted. The marker makes the timeline invisible to the resume
/// probe (`metadata_values_for_key` and `timelines_with_metadata_key` both skip
/// tombstoned timelines), so the next ingest pass no longer counts the
/// directory as done and rebuilds it properly; a following compaction then
/// reclaims the dead records.
///
/// Idempotent — tombstoning an already-dead or unknown timeline is a no-op that
/// still reports success, so a bulk repair pass can be re-run without
/// bookkeeping.
pub async fn delete_timeline(
    State(session): State<Arc<ZendSession>>,
    Path(tl): Path<String>,
) -> Result<StatusCode, StatusCode> {
    let raw: u64 = tl.parse().map_err(|_| StatusCode::BAD_REQUEST)?;
    match session.tombstone_timeline_raw(raw) {
        Some(Ok(())) => Ok(StatusCode::NO_CONTENT),
        Some(Err(e)) => {
            tracing::warn!(timeline = raw, "tombstone_timeline_raw failed: {e}");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
        // Model not loaded yet, or the id is not a valid timeline — the same
        // shape the sibling conversation routes return.
        None => Err(StatusCode::SERVICE_UNAVAILABLE),
    }
}

/// `GET /v1/substrate` — the lightweight top of the tree. No conversations,
/// section text, or tool catalog: those are fetched per-expansion so init and
/// the periodic refresh stay cheap regardless of corpus size.
#[derive(Serialize)]
pub struct SubstrateOverview {
    pub storage: Storage,
    pub counts: Counts,
    /// The projection target layer (`dialogue`) — where the live chat lives.
    pub target_layer: String,
    /// Every declared projection layer, low → high, with its config and a count
    /// of the conversations targeting it (fetched on expand).
    pub layers: Vec<LayerView>,
}

/// On-disk footprint of the segmented redo log.
#[derive(Serialize)]
pub struct Storage {
    /// One entry per `seg-*.log` file, ascending by id; the last is active.
    pub segments: Vec<SegmentView>,
    /// Sum of every segment file's size on disk.
    pub total_bytes: u64,
    /// Live KV chunk records currently indexed in RAM.
    pub live_chunks: usize,
    /// Fraction of record bytes that are dead (superseded + tombstoned) and
    /// reclaimable by compaction. `None` when the persistence lock was
    /// momentarily held (a compaction in flight) — the viewer shows "—".
    pub dead_ratio: Option<f32>,
}

#[derive(Serialize)]
pub struct SegmentView {
    pub id: u64,
    pub active: bool,
    pub bytes: u64,
}

#[derive(Serialize)]
pub struct Counts {
    pub timelines: usize,
    pub conversations: usize,
    pub sections: usize,
}

/// One projection layer's schema config plus a conversation count (the list
/// itself is fetched from `/v1/substrate/layer/{name}` on expand).
#[derive(Serialize)]
pub struct LayerView {
    pub name: String,
    pub description: String,
    /// Per-target token budget when this layer is the projection target.
    pub window: usize,
    /// Flex priority when another layer is the target.
    pub priority: f32,
    /// Declared turn groups (name + selection rule), in declaration order.
    pub groups: Vec<GroupView>,
    /// This layer's dial overrides for the shared system prompt, as
    /// `"<selector> = <option>"` labels. Empty inherits the section-tree defaults.
    pub dials: Vec<String>,
    /// Number of conversations targeting this layer.
    pub conv_count: usize,
    /// Total sealed tokens across every conversation in this layer.
    pub tokens: usize,
    /// Runtime projection kill switch: `true` while this layer is toggled OFF
    /// (excluded from projection assembly). In-memory only — clears on restart.
    pub disabled: bool,
}

#[derive(Serialize)]
pub struct GroupView {
    pub name: String,
    pub selection: String,
}

/// `GET /v1/substrate/layer/{name}` — the conversations in one layer.
#[derive(Serialize)]
pub struct LayerConversations {
    pub conversations: Vec<ConvView>,
}

/// One conversation timeline as it sits in a layer.
#[derive(Serialize)]
pub struct ConvView {
    /// Raw timeline id as a decimal string (see [`SubstrateOverview`] note).
    pub timeline: String,
    pub conv_id: String,
    pub label: String,
    pub archived: bool,
    pub group: String,
    /// Turn count including summary nodes.
    pub turns: u32,
    /// Total sealed tokens across the timeline (turns + any timeline sections).
    pub tokens: usize,
    /// Summary-forest nodes (SoT + SoS) — how much has been compressed.
    pub summary_nodes: usize,
}

/// `GET /v1/substrate/system-prompt` — the single shared system prompt. Fetched
/// when the "system prompt" card expands.
#[derive(Serialize)]
pub struct SystemPromptView {
    /// The layer currently targeted — whose dials pick the active section-tree
    /// branch of this shared prompt.
    pub target_layer: String,
    /// Ordered system-prompt item labels (sections / collection / tree).
    pub items: Vec<String>,
    /// Always-emit framing sections, with authored text + token cost.
    pub sections: Vec<SectionView>,
    /// Live tool-catalog size (the catalog itself loads from `/tools` on expand).
    pub tool_count: usize,
}

#[derive(Serialize)]
pub struct SectionView {
    pub name: String,
    pub tokens: usize,
    pub blocks: usize,
    /// The section's authored text — what gets tokenized into these blocks.
    /// Shown when the viewer expands the section.
    pub content: String,
}

/// `GET /v1/substrate/tools` — the live tool catalog. Fetched when the tools
/// collection expands.
#[derive(Serialize)]
pub struct ToolsView {
    pub tools: Vec<ToolView>,
}

/// One tool in the live catalog.
#[derive(Serialize)]
pub struct ToolView {
    pub name: String,
    pub description: String,
    /// `true` for high-risk tools (dropped from the Restricted tools mode).
    pub high_risk: bool,
    /// JSON Schema for the call arguments. The GUI lists every parameter on a
    /// tool-call card from it — the ones a call left out included, with the
    /// default they took — since a call names only what it overrides.
    pub parameters: Value,
}

/// `GET /v1/substrate/timeline/{tl}` body — one conversation's forest.
#[derive(Serialize)]
pub struct TimelineDetail {
    pub timeline: String,
    pub conv_id: String,
    pub label: String,
    pub archived: bool,
    pub layer: String,
    pub group: String,
    pub total_tokens: usize,
    /// Forest peaks — the orphan summary nodes that are the window entry points.
    pub peaks: Vec<u32>,
    /// Turn indices that open a coupled exchange (a tool-call turn joined with
    /// its response turn) — the `from` side of each `TurnCoupling`.
    pub couplings: Vec<u32>,
    /// Every turn in append order.
    pub turns: Vec<TurnView>,
}

/// One turn / summary node in the forest.
#[derive(Serialize)]
pub struct TurnView {
    pub index: u32,
    /// `"normal"` | `"sot"` (summary of turns) | `"sos"` (summary of summaries).
    pub kind: &'static str,
    /// Tree height (0 for a normal leaf turn).
    pub height: u8,
    pub tokens: usize,
    pub no_think: bool,
    /// Child turn indices this summary node compresses (empty for a normal turn).
    pub children: Vec<u32>,
    /// `true` when this turn opens a coupled exchange (its response turn follows).
    pub coupled: bool,
    /// `true` when this node is a forest peak (a window entry point).
    pub peak: bool,
    /// Decoded user half (the derived scope for a summary node).
    pub user: String,
    /// Decoded assistant half (the summary text for a summary node).
    pub assistant: String,
    /// K/V segment layout (real vs ethereal glue, user, thinking, assistant) —
    /// present for normal turns, letting the viewer colorize the exact segments.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layout: Option<TurnLayout>,
}

/// `GET /v1/substrate/timeline/{tl}/selection` body — the most recent
/// score-density selection recorded for this timeline.
#[derive(Serialize)]
pub struct SelectionView {
    /// In selection order — oldest first, most recent last.
    pub selected: Vec<SelectedNodeView>,
    /// Pending turns at the moment of selection (bigger pending ⇒ smaller
    /// selection region).
    pub pending_count: usize,
    /// Total token cost of the selected set (excludes pending).
    pub selected_tokens: u32,
    /// Layer window budget used, for scale.
    pub budget: u32,
}

/// One selected node: which turn/summary, why it was chosen, and the score
/// that won it the slot (`None` for a node selection never scored, such as a
/// `Pending` or `HardAnchor` origin).
#[derive(Serialize)]
pub struct SelectedNodeView {
    pub node_id: u32,
    pub origin: SelectionOrigin,
    pub effective_score: Option<f32>,
}

/// `POST /v1/substrate/project` request — the typed query to project.
#[derive(Deserialize)]
pub struct ProjectReq {
    pub text: String,
}

/// `POST /v1/substrate/project` body — what the query would retrieve, as tiles.
#[derive(Serialize)]
pub struct ProjectView {
    /// Query token count (the probe's wide-Q window length).
    pub query_tokens: usize,
    /// Selected turns + scored section members, sorted by belief score descending.
    pub tiles: Vec<ProjectTile>,
}

/// One retrieved item — a conversation turn/summary or a scored system-prompt
/// section (collection member) the query would pull in.
#[derive(Serialize)]
pub struct ProjectTile {
    /// `"normal"` | `"sot"` | `"sos"` | `"section"`.
    pub kind: &'static str,
    pub score: f32,
    pub selected: bool,
    pub layer: String,
    pub group: String,
    /// Turn: its conversation title (label / conv_id / file path) or `#index`.
    /// Section: the section/tool name.
    pub label: String,
    pub tokens: u32,
    /// Turn identity (raw timeline id as a string) — absent for sections.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeline: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<u32>,
    /// The full body (turn text, or the tool description) — clipped by the client.
    pub text: String,
}

impl ProjectTile {
    /// Ceiling on a tile's REPORTED score.
    ///
    /// A tile's score is a belief, and a belief is deliberately unbounded
    /// (`ToolBelief::update`): a lock-on settles at `fresh/β`, measured near
    /// ~207,000, which is useless to read. The bound is applied to the tile and
    /// nowhere upstream, because a tile is rendered for a human and never
    /// consumed again — a clamp on anything read back (the accumulator, or the
    /// persisted selection `PriorBelief::from_selection` reseeds from) ties the
    /// leaders and makes the selection arbitrary.
    ///
    /// It bounds every tile kind alike: turn tiles and section tiles are both
    /// beliefs on the same normalized band.
    ///
    /// 5,000 leaves 6.25x of headroom over `CommittedToolScope`'s `min_score` of
    /// 800, so a bounded readout still separates a strong hit from a marginal
    /// one.
    pub const SCORE_CAP: f32 = 5_000.0;

    /// Bound this tile's score for the readout.
    ///
    /// Call AFTER ranking: tiles sort on the raw belief, and clamping first ties
    /// every leader past the cap, so the readout's order — the thing its margin
    /// is read for — becomes arbitrary.
    pub fn cap_score(&mut self) {
        self.score = self.score.min(Self::SCORE_CAP);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tile(score: f32) -> ProjectTile {
        ProjectTile {
            kind: "section",
            score,
            selected: true,
            layer: String::new(),
            group: String::new(),
            label: String::new(),
            tokens: 0,
            timeline: None,
            index: None,
            text: String::new(),
        }
    }

    /// A score past the cap reports AT the cap; one inside it reports untouched.
    #[test]
    fn a_tile_score_is_bounded_at_the_cap_and_untouched_below_it() {
        let mut past = tile(143_035.0);
        past.cap_score();
        assert_eq!(past.score, ProjectTile::SCORE_CAP);

        let mut inside = tile(802.2);
        inside.cap_score();
        assert_eq!(inside.score, 802.2);
    }

    /// The catalog carries each tool's argument schema, so the GUI can list the
    /// parameters a call left out.
    #[test]
    fn a_tool_view_serializes_its_parameter_schema() {
        let view = ToolView {
            name: "file_read".to_string(),
            description: String::new(),
            high_risk: false,
            parameters: serde_json::json!({"properties": {"path": {"type": "string"}}}),
        };
        let json = serde_json::to_value(&view).unwrap();
        assert_eq!(json["parameters"]["properties"]["path"]["type"], "string");
    }
}
