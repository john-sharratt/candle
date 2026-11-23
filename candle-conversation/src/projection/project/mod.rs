//! The projection pipeline: schema, target and resolver in, emitted context out.
//!
//! A projection runs mask, score, select, reconcile and emit. The target layer
//! sees only its own group, layers below it are fully visible, and the turn
//! budget comes from the target layer's window. The belief of one reprojection
//! seeds the next through `PriorBelief`, and selection is carried out by the
//! belief, rule and fill paths. Each file owns one concern of that flow.
//!
//! - `belief`: the carried `PriorBelief` and its decay and challenger rules.
//! - `collection`: which members of a section collection are shown.
//! - `emit`: the system prompt's sections, in declaration order.
//! - `exchange`: tool round-trips as units of selection and budget.
//! - `mode`: the decode and prefill scoring modes.
//! - `pipeline`: the mask, score, select, reconcile and emit steps.
//! - `resolved`: the output types of a projection.
//! - `section_tree`: selector ids, optional states and runtime overrides.
//!
//! Test files:
//!
//! - `tests_belief`: merge, decay and challenger tests for the carried belief.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, Ordering};

use candle::engine_events::note;

use super::adaptive::AnchorMember;
use super::dialogue_fill::{self, ContextExchange, DialogueSlot, WorkingPiece};
use super::fill::{minimum_cap, reading_piece_cap, Caps, Kind};
use super::fill_report::{line, piece_reports, PieceReport, Unpooled, UNPOOLED_KEPT};
use super::ids::{
    CollectionId, GroupId, LayerId, SectionId, TimelineId, TurnId, TurnIndex, TurnKey,
};
use super::pins::{within_budget, Pin};
use super::prompt_tokens::segment_tokens;
use super::reconcile::{flexbox_distribute, FlexItem};
use super::schema::{
    GroupSchema, LayerSchema, Schema, ScoreFormula, SectionCollection, SectionSchema,
    SelectionRule, SystemPromptItem, SystemPromptSchema, TreeCollection, WorkingSetShare,
    LINEAGE_BASE,
};
use super::selection::LINEAGE_RANK_STEP;
use super::selection::{apply_selection, trim_to_budget_low_score_first};
use super::swap_cap::{cap_new_members, SwapCap};
use super::working_set_pick;
use crate::substrate::ContentResolver;
use crate::summary_tree::{NodeId, SelectionDiagnostics, SelectionOrigin};

/// Fixed scoring formula used for all turn scoring and section selection.
/// Calibrated against real Qwen3-30B-A3B Q-vector data; span α=2.0 with
/// pragmatic-only depth weights dominates all other formulas for tool
/// selection (min_ratio 5.54 vs 1.14–1.40 for alternatives).
pub(super) const FIXED_FORMULA: ScoreFormula = ScoreFormula::Span { alpha: 2.0 };

/// One position in the projection.
///
/// Three shapes — sealed (substrate-pinned), generated (live-prefilled
/// structural template), and new-user-message (this turn's user input
/// captured into the slot's pending-user-part buffer).  The assembler
/// walks a `Vec<ProjectionSegment>` in declaration order and routes
/// each variant to its inject path.
///
/// All three carry their token payloads behind an [`Arc`] so the
/// segment list is cheap to clone across the LCP comparison + rebuild
/// path the assembler performs on every reprojection.
#[derive(Debug, Clone)]
pub enum ProjectionSegment {
    /// Substrate-pinned K/V — Arc-cloned onto the slot.
    Sealed(SealedKind),
    /// Live-prefilled structural template tokens (role markers, block
    /// envelopes) whose K/V is computed under the current runtime left
    /// context, re-derived every projection rather than stored in the
    /// substrate.
    Generated {
        tokens: std::sync::Arc<Vec<u32>>,
        identity: GeneratedIdentity,
    },
    /// The user message being submitted this turn.  Prefilled onto the
    /// slot and captured into the slot's pending-user-part buffer;
    /// committed to the substrate at seal time as one half of the
    /// resulting turn record.
    NewUserMessage { tokens: std::sync::Arc<Vec<u32>> },
}

/// Discriminator under [`ProjectionSegment::Sealed`] — distinguishes a
/// system-prompt section from a turn-half entry.  Each turn carries a
/// `user` half and an `assistant` half in the substrate; a `Turn`
/// segment names which half to inject.  Under today's seal path the
/// user half is empty and the assistant half holds all content, so
/// the projection engine emits `part: Role::Assistant`; once the
/// `NewUserMessage` capture path lights up both halves are real and
/// each turn projects as a `User` segment followed by an `Assistant`
/// segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SealedKind {
    Section(ResolvedSection),
    Turn(ResolvedTurn, crate::Role),
}

/// Diagnostic identity for a [`ProjectionSegment::Generated`] run.
///
/// Does not affect assembly — used only for log messages and progress
/// traces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GeneratedIdentity {
    /// Schema-level name for logs (e.g. `"system_open"`,
    /// `"tools_close"`).
    pub name: String,
    /// Monotonic position in the projection's segment list, assigned
    /// at emit time.  Stable across same-shape reprojections so it can
    /// serve as an LCP tie-breaker even when two runs happen to share
    /// the same surface text.
    pub position: usize,
}

/// Result of a [`super::Builder::project`] call.
///
/// Segments emit in (system-prompt items, declaration order) followed
/// by (turns: layer order × ascending-group-score × turn insertion
/// order).  Sealed segments interleave with Generated runs around
/// every turn-boundary template and around any structural template
/// items declared in the system prompt.
#[derive(Debug, Clone, Default)]
pub struct Projection {
    pub segments: Vec<ProjectionSegment>,
    /// Section-tree selectors as resolved for this projection — which option each
    /// selector emitted, addressed by their string ids.  Empty when the target
    /// layer has no [`super::SectionTree`].
    pub selections: Vec<ResolvedSelection>,
    /// Per-section/turn belief confidence produced by the online selection loop,
    /// stamped onto the recorded [`super::ProjectionEvent`] so the next
    /// reprojection seeds its belief from this one.
    pub selection_scores: super::SelectionScores,
    /// Why each selected turn entered the slot, keyed by `(group, turn)`. The
    /// score-density path tags each pick from the summary forest (hard anchor /
    /// provenance / coverage / …); the rule-based path tags `Seed` / `Inherited`
    /// (the lineage), `Recent` (inside the recency window) or `Historical`
    /// (top-k by score). Read by
    /// [`super::event::from_projection_with_origins`] so the persisted projection
    /// record / GUI can show "why is this turn in my context?". Empty entries
    /// (e.g. the live user message) carry no origin.
    pub selection_origins: HashMap<TurnKey, SelectionOrigin>,
    /// What the dialogue's slot fill weighed each working piece at, and where
    /// each stands in the pool (`fill_report`). Empty for a target with no
    /// fill.
    pub fill: Vec<PieceReport>,
    /// The strongest conversations the scan scored that the pool does not
    /// hold.
    pub unpooled: Vec<Unpooled>,
}

/// One section-tree selector as resolved for a projection — the selector id and
/// the option it emitted.  Carried on [`Projection::selections`] so callers
/// (GUI, runtime) know exactly which option fired and can address it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSelection {
    pub selector: String,
    pub option: String,
}

/// Identifies which layer and which group inside that layer the projection
/// is **for**. Masking semantics flow from this target.
///
/// ```text
///   rank    < target rank         → fully visible
///   layer  == target.layer        → only target.group visible (siblings hidden)
///   group  == target.group        → only target.timeline visible
///                                   (Phase 3 of the substrate refactor — see
///                                    [`crate::projection::resolver`])
///   rank   >= target rank         → entirely hidden
/// ```
///
/// A layer's rank is its declaration index unless it declares one — see
/// [`LayerSchema::rank`].
#[derive(Debug, Clone, Copy)]
pub struct ProjectionTarget {
    pub layer: LayerId,
    pub group: GroupId,
    /// The specific instance within `(layer, group)` that this projection
    /// is for.  Multiple parallel conversations of the same group shape
    /// each have their own `TimelineId`; only the target's timeline emits
    /// from `target.group`, while lower-layer groups remain visible
    /// across all timelines (Phase 3 of the substrate refactor enforces
    /// this masking; Phase 1 still aggregates the first-registered
    /// timeline per group).
    pub timeline: TimelineId,
}

mod belief;
mod collection;
mod emit;
mod exchange;
mod mode;
mod pipeline;
mod resolved;
mod section_tree;
#[cfg(test)]
mod tests_belief;
mod turn_dials;

pub use belief::*;
use collection::*;
use emit::*;
use exchange::*;
pub use mode::*;
pub use pipeline::*;
pub use resolved::*;
pub use section_tree::*;
pub use turn_dials::*;
