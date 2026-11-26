//! The recorded account of each decode: what provenance selected, how many tokens
//! each part took, and how fast the model decoded against it.
//!
//! A projection event is produced once per decode. It pairs the decode's
//! throughput with the composition of the materialized context, grouped into a
//! system bucket, one bucket per contributing section collection, and one per
//! conversation group. Aggregation is pure and tested on its own, and an adapter
//! classifies real segments against the live schema and resolver. Events are
//! stored as JSON in the redo log. Decoding returns nothing for malformed input,
//! so a corrupt timeline entry cannot break a conversation's hydrate.
//!
//! - `classify`: classifies a projection's segments into an event.
//! - `constructors`: aggregation, and the events for ingest and summary nodes.
//! - `selection`: the selected sections and turns with their belief scores.
//! - `tests_events`: tests of aggregation, classification and encoding.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use super::fill_report::{PieceReport, Unpooled};
use super::ids::{CollectionId, GroupId, SectionId, TurnIndex, TurnKey};
use super::project::{ProjectionSegment, SealedKind};
use super::scan_stats::GroupScan;
use super::schema::{Schema, SystemPromptItem};
use crate::files::PlacedFile;
use crate::substrate::ContentResolver;
use crate::summary_tree::{SelectionOrigin, TurnKind};

/// One category of the materialized context, with its summed token count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectionBucket {
    /// Display label: `"system"`, or the collection / group name.
    pub label: String,
    pub kind: BucketKind,
    pub tokens: u32,
}

/// The exact materialized-context selection for a projection — what provenance
/// pulled in, structured for the GUI's per-section "selected / not" view (vs the
/// token-count [`ProjectionBucket`]s, which are just the aggregate totals).
///
/// Only the *contributing* layer's items appear: a bare section is listed when
/// selected; a collection is listed (with all its members) when ≥1 of its
/// sections was selected — which naturally scopes the view to the projection's
/// own layer without threading a layer id through.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProjectionSelection {
    /// The system prompt in materialized order: structural glue, bare content
    /// sections, and contributing collections interleaved exactly as emitted.
    pub system: Vec<SystemItem>,
    /// Conversation turns selected this projection.
    pub turns: Vec<SelectedTurn>,
}

/// A recorded projection point: the `start_token` position it governs, the
/// wall-clock `seconds` since decode start, and the materialized-context
/// composition. Throughput and the governed interval are derived by the consumer
/// from the sequence of points (see `start_token` / `seconds`).
///
/// All fields are `#[serde(default)]` so the persisted redo-log record stays
/// forward-compatible as the shape evolves.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct ProjectionEvent {
    /// Generated-token position at which this projection was SELECTED — the start
    /// of the interval it governs. A projection is a point that governs everything
    /// forward until the next event supersedes it, so the consumer derives each
    /// projection's effective interval `[start_token_i, start_token_{i+1})` (the
    /// last one running to the turn's final token count) from the event sequence.
    #[serde(default)]
    pub start_token: u32,
    /// Wall-clock seconds since decode start, at selection. The consumer derives
    /// each interval's throughput from the `(start_token, seconds)` deltas between
    /// consecutive events.
    #[serde(default)]
    pub seconds: f64,
    /// Sum of all bucket tokens — what provenance materialized into the window.
    #[serde(default)]
    pub materialized_tokens: u32,
    /// Total tokens stored across the whole substrate (all layers/timelines).
    #[serde(default)]
    pub substrate_tokens: u32,
    /// Per-category breakdown, ordered system → section groups → turns.
    #[serde(default)]
    pub buckets: Vec<ProjectionBucket>,
    /// Structured, per-section selection (what provenance picked AND skipped) —
    /// backs the projection bubble's substrate view.
    #[serde(default)]
    pub selection: ProjectionSelection,
    /// The conversation in materialized injection order, exactly as the assembler
    /// laid it out (`assemble_pieces`): real boundary-glue islands interleaved
    /// with the sealed turns. The panel renders the dialogue region from this so
    /// its glue is what the engine truly injected. Empty on records written
    /// before this field, or when the boundary markers weren't available to the
    /// builder (the panel then falls back to reconstructing the framing).
    #[serde(default)]
    pub materialized: Vec<MaterializedPiece>,
    /// When `true`, this event marks a **self-referencing retrieval sub-window**:
    /// the belief scan bounds the interval `[start_token, next_start_token)` as one
    /// focused window, but tags every window of a turn with the SAME slot, so the
    /// scorer aggregates them into ONE case (best-token agreement across the whole
    /// turn) rather than letting the regions compete as separate cases and split
    /// the turn's vote. So a query matching one structural region of a prefilled
    /// listing (a subdirectory of a repo_map cluster) surfaces the whole cluster
    /// without diluting against the rest of the listing. Ordinary reprojection /
    /// staged-ingest events leave it `false` — their target is their `selection`,
    /// and the turn is scored as one window.
    #[serde(default)]
    pub self_reference: bool,
    /// The raw-score statistics of every turn group the belief scan scored for
    /// this projection, selected or not — what each layer's need is read from.
    /// Empty on records written before this field and on synthesized events.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub groups: Vec<GroupScanRecord>,
    /// Every working piece the dialogue's slot fill was offered — the scores
    /// it ranked the piece's rounds by, what it placed, and where the piece
    /// stands in the pool — strongest first. Empty for a target with no fill.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pieces: Vec<PieceReport>,
    /// The strongest conversations this boundary's scan scored that the pool
    /// does not hold.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unpooled: Vec<Unpooled>,
    /// The knowledge base's files this point placed, in the order they sit,
    /// each with its placed chunks' lines
    /// (`docs/provenance_system_collection.md` §3). Empty when it placed none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub placed: Vec<PlacedFile>,
}

/// One turn group's scan, named for the record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GroupScanRecord {
    pub group: String,
    #[serde(flatten)]
    pub scan: GroupScan,
}

/// One classified segment before aggregation — `(kind, label, tokens)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentTokens {
    pub kind: BucketKind,
    pub label: String,
    pub tokens: u32,
}

/// Encode a turn's projection events to a redo-log record payload — UTF-8 JSON,
/// the same encoding `StreamDecl` uses. Paired with [`decode_events`].
pub fn encode_events(events: &[ProjectionEvent]) -> Vec<u8> {
    serde_json::to_vec(events).expect("ProjectionEvent JSON encoding is infallible")
}

/// Decode a projection-events record payload back into the turn's events.
/// Returns an empty vec on malformed input — a corrupt timeline entry must not
/// break conversation hydrate.
pub fn decode_events(payload: &[u8]) -> Vec<ProjectionEvent> {
    serde_json::from_slice(payload).unwrap_or_default()
}

/// A projection point as a conversation's timeline draws it: a
/// [`ProjectionEvent`] without its `selection` and `materialized` spine.
///
/// Those two are read only by the projection panel and are most of a point's
/// bytes — about ten kilobytes each. Decoded into this type they are skipped
/// as the JSON is read, never built, so loading a conversation's history costs
/// its timeline and not its panel.
#[derive(Debug, Clone, Default, PartialEq, Deserialize)]
pub struct ProjectionPoint {
    #[serde(default)]
    pub start_token: u32,
    #[serde(default)]
    pub seconds: f64,
    #[serde(default)]
    pub materialized_tokens: u32,
    #[serde(default)]
    pub substrate_tokens: u32,
    #[serde(default)]
    pub buckets: Vec<ProjectionBucket>,
    #[serde(default)]
    pub self_reference: bool,
}

/// Decode a projection-events record payload into its points, light — see
/// [`ProjectionPoint`]. Empty on malformed input, like [`decode_events`].
pub fn decode_points(payload: &[u8]) -> Vec<ProjectionPoint> {
    serde_json::from_slice(payload).unwrap_or_default()
}

mod classify;
mod constructors;
mod selection;
#[cfg(test)]
mod tests_events;

pub use classify::*;
pub use constructors::*;
pub use selection::*;
