//! Full projection pipeline: **mask → score → select → reconcile → emit**.
//!
//! This module is the orchestrator that ties together [`super::selection`],
//! [`super::score`], [`super::reconcile`], and the [`super::ContentResolver`]
//! into the 12-step flow described in the design doc §9.
//!
//! # Pipeline
//!
//! ```text
//!  ┌────────────────────────────────────────────────────────────────────┐
//!  │  Inputs:  Schema, ProjectionTarget, &resolver                      │
//!  ├────────────────────────────────────────────────────────────────────┤
//!  │  STEP 1.  Mask                                                     │
//!  │           layers below target.layer  → fully visible               │
//!  │           target layer               → only target group visible   │
//!  │           layers above target.layer  → entirely hidden             │
//!  ├────────────────────────────────────────────────────────────────────┤
//!  │  STEP 2.  Score every visible turn      (resolver.turn_score)      │
//!  ├────────────────────────────────────────────────────────────────────┤
//!  │  STEP 3.  Group threshold gate                                     │
//!  │           drop turns with score < group.score_threshold            │
//!  ├────────────────────────────────────────────────────────────────────┤
//!  │  STEP 4.  Unbounded selection per group                            │
//!  │           apply_selection(rule, threshold, turns, budget=None)     │
//!  │           → "natural" set the rule would pick if unconstrained     │
//!  ├────────────────────────────────────────────────────────────────────┤
//!  │  STEP 5.  Group score = FIXED_FORMULA.aggregate(natural scores)    │
//!  ├────────────────────────────────────────────────────────────────────┤
//!  │  STEP 6.  Layer threshold gate                                     │
//!  │           drop groups with derived score < layer.score_threshold   │
//!  ├────────────────────────────────────────────────────────────────────┤
//!  │  STEP 7.  Filter empty groups; layers with no surviving groups     │
//!  │           drop out as a side effect                                │
//!  ├────────────────────────────────────────────────────────────────────┤
//!  │  STEP 8.  Emit target layer's system-prompt sections               │
//!  │           Each layer carries its own system_prompt; the TARGET     │
//!  │           layer's sections frame this projection. Sections emit    │
//!  │           in declaration order, in full — no reconciliation.       │
//!  ├────────────────────────────────────────────────────────────────────┤
//!  │  STEP 9.  Reconcile turn budget — TWO-LEVEL FLEXBOX                │
//!  │                                                                    │
//!  │           Total turn-budget for this projection comes from         │
//!  │           target.layer.window — different targets get different    │
//!  │           pies (e.g. zend gives the dialogue layer 16K, reasoning  │
//!  │           layers 8K, foundational layers 6K).                      │
//!  │                                                                    │
//!  │           outer: layers (flexbox capped at natural consumption)    │
//!  │           inner: groups within each layer (same)                   │
//!  │                                                                    │
//!  │           Natural-consumption caps make the single-pass flexbox    │
//!  │           equivalent to the doc's iterative redistribution: an     │
//!  │           under-consuming layer/group saturates at its cap, and    │
//!  │           the freed budget redistributes to others in the same    │
//!  │           pass.                                                    │
//!  ├────────────────────────────────────────────────────────────────────┤
//!  │  STEP 10. Bounded selection per group with allocated budget        │
//!  │           apply_selection(..., budget=Some(allocated), ...)        │
//!  │           Each rule has its own trim policy (see selection.rs).   │
//!  ├────────────────────────────────────────────────────────────────────┤
//!  │  STEP 11. (Iterative redistribution — implicit in the natural-cap  │
//!  │           strategy. See "Why single-pass" below.)                  │
//!  ├────────────────────────────────────────────────────────────────────┤
//!  │  STEP 12. Emit                                                     │
//!  │           sections — declaration order                             │
//!  │           layers  — declaration order (layer 0 first)              │
//!  │           groups  — sorted by ASCENDING group score (so highest    │
//!  │                     emits LAST, near the model's recency bias)    │
//!  │           turns   — insertion order within each group              │
//!  └────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Why single-pass instead of the doc's iterative loop?
//!
//! The design doc §5 sketches an iterative algorithm where freed budget from
//! under-consuming groups is redistributed across `MAX_ITERATIONS` passes.
//! The naive transcription has a subtle bug: the doc's `remaining = freed`
//! line shrinks the budget on each iteration, which can drop already-selected
//! turns when the freed amount is smaller than a single turn's tokens.
//!
//! Our approach achieves the same result in one flexbox pass:
//!
//! ```text
//!   1. Phase-1 unbounded selection determines each group's natural token
//!      consumption.
//!   2. Each group's FlexItem.max_tokens is set to that natural consumption.
//!   3. flexbox_distribute, in a single call, gives saturated groups exactly
//!      their natural amount and redistributes the rest.
//!   4. Phase-2 bounded selection then trims any group whose allocation came
//!      out below its natural consumption.
//! ```
//!
//! This is functionally equivalent to the iterative loop (assuming the loop
//! converged), strictly correct in edge cases the literal algorithm got
//! wrong, and visibly simpler.

use std::collections::HashMap;

use super::ids::{CollectionId, GroupId, LayerId, SectionId, TimelineId, TurnId, TurnIndex};
use super::reconcile::{flexbox_distribute, FlexItem};
use super::schema::{
    GroupSchema, LayerSchema, Schema, ScoreFormula, SectionCollection, SectionSchema,
    SelectionRule, SystemPromptItem,
};
use super::selection::apply_selection;
use crate::substrate::ContentResolver;
use crate::summary_tree::{NodeId, SelectionDiagnostics};

/// Fixed scoring formula used for all turn scoring and section selection.
/// Calibrated against real Qwen3-30B-A3B Q-vector data; span α=2.0 with
/// pragmatic-only depth weights dominates all other formulas for tool
/// selection (min_ratio 5.54 vs 1.14–1.40 for alternatives).
pub(super) const FIXED_FORMULA: ScoreFormula = ScoreFormula::Span { alpha: 2.0 };

/// Which inference phase a projection is being computed for.
///
/// Decode and prefill produce structurally different Q vectors and so need
/// different collection-scoring configs (calibrated 2026-05-17 against real
/// Qwen3-30B-A3B data — see `tests/projection_harness/cases/`):
///
/// - **Decode** — Q vectors captured while the model *generates* a tool-call
///   response.  They form a coherent run, so `Span{alpha:2.0}` on pragmatic
///   depth dominates (min_ratio 5.96).  This is the steady-state mode used by
///   continuous reprojection during decode.
/// - **Prefill** — Q vectors captured while the model *reads* the user prompt.
///   The signal is weak and lives *below* the hit threshold, so run-based
///   (`Span`) and extreme-value (`Max`) formulas miss it.  `PerTokenExcess`
///   on pragmatic depth — recentered on the noise baseline, per-probe-token,
///   threshold-free — recovers it (full-corpus top-1 58% / top-3 74%, vs
///   `Max` 38% and `Span` 43%).  The intra/inter ratio is still thin, so the
///   `score_threshold` gate is skipped; selection is pure top-k by rank.
///   Used for the initial-guess section injection before decode refines it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjectionMode {
    Prefill,
    Decode,
}

/// Section-scoring configuration resolved from a [`ProjectionMode`].
struct CollectionScoring {
    /// When `false`, `score_threshold` is not used as a gate — every section
    /// competes on rank alone (the prefill ratio is too thin to threshold).
    /// Belief-driven collections ignore this (the policy's `min_score` gates).
    apply_threshold: bool,
}

impl ProjectionMode {
    fn collection_scoring(self) -> CollectionScoring {
        match self {
            ProjectionMode::Decode => CollectionScoring {
                apply_threshold: true,
            },
            ProjectionMode::Prefill => CollectionScoring {
                apply_threshold: false,
            },
        }
    }
}

// ── Output types ──────────────────────────────────────────────────────────────

/// One emitted system-prompt section reference. The caller resolves the
/// content from [`SectionId`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSection {
    pub id: SectionId,
}

/// One emitted turn reference, fully self-describing via [`TurnId`].
///
/// Convenience accessors for the legacy `(group, index)` pair are provided
/// so call sites that don't care about conversation/layer info can stay
/// terse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedTurn {
    pub id: TurnId,
}

impl ResolvedTurn {
    #[inline]
    pub fn group(&self) -> GroupId {
        self.id.group_id
    }

    #[inline]
    pub fn index(&self) -> TurnIndex {
        self.id.index
    }
}

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
    /// The *user half* of a turn — its user-message body, derived on
    /// demand as a zero-copy window view over the turn's existing chunks.
    /// Distinct from [`SealedKind::Turn`], which always injects the whole
    /// turn: a `TurnHalf` injects only the user content (no per-turn
    /// boundary-marker wrapping), so the compression path can assemble
    /// many turns' user-halves into one coherent block.  The assistant
    /// half is text-prefilled rather than injected — its assistant-role
    /// K/V are incoherent in the compression's user-input frame — so only
    /// the user half is ever injected.  Used solely by the summary-tree
    /// compression passes.
    TurnHalf(ResolvedTurn),
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
}

/// The prior projection's per-collection belief, used to seed the current
/// projection so the online decay/reinforcement carries across a turn's
/// reprojections. Built by the scheduler from the previous
/// [`super::ProjectionEvent`]; empty (the default) means a fresh, stateless
/// belief for this projection.
#[derive(Debug, Clone, Default)]
pub struct PriorBelief {
    collections: std::collections::HashMap<String, std::collections::HashMap<String, (f32, bool)>>,
}

impl PriorBelief {
    /// Record a section's prior belief (score + whether it was selected).
    pub fn set(&mut self, collection: &str, section: &str, score: f32, selected: bool) {
        self.collections
            .entry(collection.to_string())
            .or_default()
            .insert(section.to_string(), (score, selected));
    }

    /// Rebuild the prior belief from a completed projection's selection — every
    /// collection member's `(score, selected)`. The scheduler stores this on the
    /// decode state after each reprojection so the next one seeds from it,
    /// carrying the online belief across a turn's reprojections.
    pub fn from_selection(sel: &super::ProjectionSelection) -> PriorBelief {
        let mut pb = PriorBelief::default();
        for item in &sel.system {
            if let super::SystemItem::Collection { name, sections } = item {
                for s in sections {
                    pb.set(name, &s.name, s.score, s.selected);
                }
            }
        }
        pb
    }

    /// Aligned `(scores, selected)` for a collection's sections in declaration
    /// order — the seed for [`crate::provenance::belief_step`]. Unknown sections
    /// read `(0.0, false)`.
    fn collection(&self, name: &str, sections: &[SectionSchema]) -> (Vec<f32>, Vec<bool>) {
        let map = self.collections.get(name);
        let mut scores = vec![0.0f32; sections.len()];
        let mut selected = vec![false; sections.len()];
        if let Some(m) = map {
            for (i, s) in sections.iter().enumerate() {
                if let Some(&(sc, sel)) = m.get(&s.name) {
                    scores[i] = sc;
                    selected[i] = sel;
                }
            }
        }
        (scores, selected)
    }
}

/// The section-tree selector id the engine reads to drive per-turn thinking
/// suppression.  The dialogue schema authors an `optional` node with this id;
/// [`OptionalState::Present`] suppresses reasoning for the turn, [`Absent`]
/// enables it.  The contract lives here (not as scattered `"no_think"` literals)
/// so the schema author, the dial mapping, and the suppression read can't drift.
///
/// [`Absent`]: OptionalState::Absent
pub const NO_THINK_SELECTOR: &str = "no_think";

/// The two states of an `optional` section-tree node: whether its content is
/// projected (`Present`) or omitted (`Absent`).
///
/// Unlike a selector's *option* ids (which are authored in YAML and so are
/// inherently dynamic strings), these two ids are a fixed convention the
/// `optional` lowering synthesizes — a closed set the engine owns.  This enum is
/// the typed form; [`Self::as_id`] / [`Self::from_id`] are the only place the
/// `"present"` / `"absent"` strings are defined, so a typo can't silently slip
/// past the compiler and fall through to a default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptionalState {
    Present,
    Absent,
}

impl OptionalState {
    /// The synthesized option id this state serializes to in a [`SelectionState`].
    pub const fn as_id(self) -> &'static str {
        match self {
            OptionalState::Present => "present",
            OptionalState::Absent => "absent",
        }
    }

    /// Parse an `optional` node's option id back to its typed state; `None` for
    /// any id that isn't one of the two synthesized optional ids.
    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "present" => Some(OptionalState::Present),
            "absent" => Some(OptionalState::Absent),
            _ => None,
        }
    }
}

/// Runtime override of section-tree selectors for one projection.
///
/// Maps a selector id (a dimension node's declared name) to the chosen option
/// id.  Any selector absent from the map falls back to its authored default, so
/// an empty state reproduces the schema defaults exactly.
///
/// Selector and option ids are `String`s by necessity — they are authored in
/// the projection YAML, so the override map is data-driven and can't be a closed
/// Rust type.  For the engine's own closed conventions (the `optional`
/// present/absent state), use the typed [`Self::optional`] / [`Self::set_optional`]
/// accessors rather than bare string literals.
#[derive(Debug, Clone, Default)]
pub struct SelectionState {
    chosen: HashMap<String, String>,
}

impl SelectionState {
    /// An empty selection — every selector falls back to its authored default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select a selector's option by id.  Returns `&mut self` for chaining.
    pub fn select(&mut self, selector: impl Into<String>, option: impl Into<String>) -> &mut Self {
        self.chosen.insert(selector.into(), option.into());
        self
    }

    /// Set an `optional` selector to a typed [`OptionalState`].  Returns
    /// `&mut self` for chaining.  Prefer this over `select(id, "present")` so the
    /// present/absent ids stay in one place.
    pub fn set_optional(&mut self, selector: impl Into<String>, state: OptionalState) -> &mut Self {
        self.select(selector, state.as_id())
    }

    /// The chosen option id for a selector, if one was set.
    pub fn get(&self, selector: &str) -> Option<&str> {
        self.chosen.get(selector).map(String::as_str)
    }

    /// The typed state of an `optional` selector.  `None` when the selector is
    /// unset (the caller falls back to its schema/config default) or resolves to
    /// an id that isn't one of the optional present/absent ids.
    pub fn optional(&self, selector: &str) -> Option<OptionalState> {
        self.get(selector).and_then(OptionalState::from_id)
    }
}

/// One section-tree selector as resolved for a projection — the selector id and
/// the option it emitted.  Carried on [`Projection::selections`] so callers
/// (GUI, runtime) know exactly which option fired and can address it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSelection {
    pub selector: String,
    pub option: String,
}

impl Projection {
    /// Iter over only the sealed-section payloads, in emission order.
    /// Convenience over walking [`Self::segments`] manually; skips
    /// generated runs and new-user-message segments.
    pub fn sealed_sections(&self) -> impl Iterator<Item = &ResolvedSection> + '_ {
        self.segments.iter().filter_map(|seg| match seg {
            ProjectionSegment::Sealed(SealedKind::Section(s)) => Some(s),
            _ => None,
        })
    }

    /// Iter over only the sealed-turn payloads, in emission order.
    /// Yields `&ResolvedTurn` — the `Role` discriminator under
    /// `SealedKind::Turn` is dropped here so callers that only care
    /// about which `(group, index)` pair was selected don't have to
    /// destructure the part.  Callers that need the part too use
    /// [`Self::sealed_turn_segments`].
    pub fn sealed_turns(&self) -> impl Iterator<Item = &ResolvedTurn> + '_ {
        self.segments.iter().filter_map(|seg| match seg {
            ProjectionSegment::Sealed(SealedKind::Turn(t, _)) => Some(t),
            _ => None,
        })
    }

    /// Iter over each sealed-turn segment as `(ResolvedTurn, Role)` so
    /// callers can distinguish the user half from the assistant half
    /// of the same turn.  Used by the projection assembler when
    /// injecting per-part residence bytes.
    pub fn sealed_turn_segments(&self) -> impl Iterator<Item = (&ResolvedTurn, crate::Role)> + '_ {
        self.segments.iter().filter_map(|seg| match seg {
            ProjectionSegment::Sealed(SealedKind::Turn(t, role)) => Some((t, *role)),
            _ => None,
        })
    }
}

/// Identifies which layer and which group inside that layer the projection
/// is **for**. Masking semantics flow from this target.
///
/// ```text
///   layers  < target.layer        → fully visible
///   layer  == target.layer        → only target.group visible (siblings hidden)
///   group  == target.group        → only target.timeline visible
///                                   (Phase 3 of the substrate refactor — see
///                                    [`crate::projection::resolver`])
///   layers  > target.layer        → entirely hidden
/// ```
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

// ── Main entry-point ──────────────────────────────────────────────────────────

/// Run the full projection pipeline.
///
/// Called from [`super::Builder::project`]. See module-level docs for the
/// 12-step pipeline. Pure given a valid schema and a working resolver —
/// never errors.
pub fn run<R: ContentResolver>(
    schema: &Schema,
    target: ProjectionTarget,
    resolver: &R,
    mode: ProjectionMode,
    selection: &SelectionState,
) -> Projection {
    run_with_sink(
        schema,
        target,
        resolver,
        mode,
        selection,
        &PriorBelief::default(),
        &mut |_| {},
    )
}

/// Variant of [`run`] that delivers score-density [`SelectionDiagnostics`]
/// to a sink the caller provides.  Used by the scheduler to ferry the
/// diagnostic straight into the substrate's
/// [`Substrate::set_last_selection`](crate::substrate::Substrate::set_last_selection)
/// side-channel without polluting [`Projection`] with a test-only field.
///
/// When the projection used the rule-based path (no summary tree for
/// the target timeline), the sink is never invoked.  When score-density
/// did run, the sink is called exactly once with the fully-assembled
/// diagnostic — token counts, origin tags, effective scores, pending
/// backpressure metric — built from the resolver's existing
/// `turn_token_count` + `pending_summary_len` methods.
#[allow(clippy::too_many_arguments)]
pub fn run_with_sink<R: ContentResolver>(
    schema: &Schema,
    target: ProjectionTarget,
    resolver: &R,
    mode: ProjectionMode,
    selection: &SelectionState,
    prior: &PriorBelief,
    sink: &mut dyn FnMut(SelectionDiagnostics),
) -> Projection {
    let mut selection_scores = super::SelectionScores::default();
    // ── Step 1: Mask ─────────────────────────────────────────────────────────
    let target_layer_idx = schema
        .layers
        .iter()
        .position(|l| l.id == target.layer)
        .unwrap_or(0);

    let visible_layers: Vec<&LayerSchema> = schema
        .layers
        .iter()
        .enumerate()
        .filter_map(|(li, layer)| {
            // All groups in lower layers are visible; on the target
            // layer, groups are filtered individually further below.
            if li <= target_layer_idx {
                Some(layer)
            } else {
                None
            }
        })
        .collect();

    // ── Step 2–4: Score, threshold-gate, unbounded selection ─────────────────
    struct GroupState<'a> {
        schema: &'a GroupSchema,
        layer_idx: usize,
        selected: Vec<(TurnIndex, f32)>, // (index, score), insertion order
        group_score: f32,
    }

    let mut group_states: Vec<GroupState> = Vec::new();

    for (li, layer) in visible_layers.iter().enumerate() {
        let layer_is_target = li == target_layer_idx;
        for group in &layer.groups {
            // Masking: for the target layer, only the target group is visible.
            if layer_is_target && group.id != target.group {
                continue;
            }

            let count = resolver.turn_count(group.id);
            let all_turns: Vec<(TurnIndex, f32)> = (0..count)
                .map(|i| {
                    let idx = TurnIndex(i);
                    let score = resolver.turn_score(group.id, idx);
                    (idx, score)
                })
                .collect();

            let tc = |idx: TurnIndex| resolver.turn_token_count(group.id, idx);

            // Score-density override: when this is the *target* group
            // and a summary tree exists for the target timeline, swap
            // out the rule-based unbounded selection for the §8
            // algorithm.  The result is already budget-fitted into
            // `layer.window` so the flexbox step caps the group's
            // natural consumption at the score-density total.
            let mut score_density_used = false;
            let selected: Vec<(TurnIndex, f32)> = if layer_is_target && group.id == target.group {
                if let Some(picks) =
                    resolver.summary_tree_select(target.timeline, layer.window as u32)
                {
                    score_density_used = true;
                    // Score-density diagnostics for the test harness
                    // (§10.8.4): assemble per-node selection metadata
                    // and hand it to the caller-supplied sink.  The
                    // scheduler's sink writes this to the substrate's
                    // last-selection side-channel; production callers
                    // pass a no-op.  Skipped entirely when the
                    // rule-based path runs (no tree → no `picks`).
                    let mut diag = SelectionDiagnostics::new(layer.window as u32);
                    for (turn_idx, origin, score) in &picks {
                        let tokens = resolver.turn_token_count(group.id, *turn_idx) as u32;
                        diag.push(NodeId(turn_idx.0), *origin, *score, tokens);
                    }
                    diag.pending_count = resolver.pending_summary_len(target.timeline);
                    sink(diag);
                    picks
                        .iter()
                        .map(|(idx, _origin, score)| (*idx, *score))
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            // Fall through to the rule-based unbounded selection path
            // when score-density wasn't applicable.
            let selected: Vec<(TurnIndex, f32)> = if score_density_used {
                selected
            } else {
                let selected_indices = apply_selection(
                    &group.selection,
                    group.score_threshold,
                    &all_turns,
                    None,
                    &tc,
                );

                selected_indices
                    .iter()
                    .map(|&idx| {
                        let score = all_turns
                            .iter()
                            .find(|(i, _)| *i == idx)
                            .map(|(_, s)| *s)
                            .unwrap_or(0.0);
                        (idx, score)
                    })
                    .collect()
            };

            tracing::trace!(
                group = %group.name,
                selected = format!("{}/{}", selected.len(), all_turns.len()),
                score_density = score_density_used,
                "projection"
            );

            group_states.push(GroupState {
                schema: group,
                layer_idx: li,
                selected,
                group_score: 0.0,
            });
        }
    }

    // ── Step 5: Group scores ──────────────────────────────────────────────────
    for gs in &mut group_states {
        let scores: Vec<f32> = gs.selected.iter().map(|(_, s)| *s).collect();
        gs.group_score = FIXED_FORMULA.aggregate(&scores);
    }

    // ── Step 6: Layer score threshold ────────────────────────────────────────
    // Doc §9.6: "Apply layer score thresholds → surviving groups". Groups whose
    // derived score falls below their layer's threshold are dropped wholesale.
    group_states.retain(|gs| {
        let layer = &visible_layers[gs.layer_idx];
        gs.group_score >= layer.score_threshold
    });

    // ── Step 7: Filter empty groups and (transitively) empty layers ───────────
    // Doc §5: "Empty nodes do not consume budget; their min reservations are
    // released." A group whose selection produced zero turns is filtered out
    // before reconciliation. A layer that has lost all its groups is filtered
    // when we collect surviving_layer_indices below.
    group_states.retain(|gs| !gs.selected.is_empty());

    // Collect surviving layer indices.
    let surviving_layer_indices: Vec<usize> = {
        let mut v: Vec<usize> = group_states.iter().map(|gs| gs.layer_idx).collect();
        v.sort();
        v.dedup();
        v
    };

    // ── Step 8: System prompt — emit target layer's items ────────────────────
    //
    // The target layer's `system_prompt.items` is an ordered list:
    // each item is either a single section (always emits) or a
    // [`SectionCollection`] (named bucket with its own selection rule).
    // Items emit in declaration order; sections inside a collection
    // also emit in their declaration order, but only the surviving
    // subset of the collection's selection rule survives.
    //
    // This separation — static framing as plain sections vs.
    // selectable catalogs as collections — is what lets a layer's
    // system prompt mix always-emit framing (role, grounding,
    // dialect markers) with dynamic catalogs (tool definitions,
    // retrieval candidates) at well-defined positions.
    let mut resolved_selections: Vec<ResolvedSelection> = Vec::new();
    let system_prompt_segments: Vec<ProjectionSegment> = schema
        .layers
        .iter()
        .find(|l| l.id == target.layer)
        .map(|l| {
            emit_system_prompt_items(
                l,
                resolver,
                mode,
                selection,
                prior,
                &mut selection_scores,
                &mut resolved_selections,
            )
        })
        .unwrap_or_default();

    if group_states.is_empty() {
        return Projection {
            segments: system_prompt_segments,
            selections: resolved_selections,
            selection_scores,
        };
    }

    // ── Steps 9–11: Turn budget reconciliation with natural-cap redistribution ──
    //
    // The turn budget comes from the **target layer's `window`** — different
    // targets get different total pies to slice. Distribute via flex over all
    // visible layers below + the target layer; group-level flex within each
    // layer; bounded selection within each group.
    //
    // Strategy: compute each group's natural token consumption (from the
    // unbounded-selected turns), use that as a max_tokens cap in FlexItem so
    // that flexbox_distribute automatically redistributes surplus to groups that
    // are still hungry. Then run bounded selection once per group.
    let turn_budget = schema
        .layers
        .iter()
        .find(|l| l.id == target.layer)
        .map(|l| l.window)
        .unwrap_or(0);

    // Natural token consumption per group (unbounded selection result).
    let natural_tokens: HashMap<GroupId, usize> = group_states
        .iter()
        .map(|gs| {
            let tokens: usize = gs
                .selected
                .iter()
                .map(|(idx, _)| resolver.turn_token_count(gs.schema.id, *idx))
                .sum();
            (gs.schema.id, tokens)
        })
        .collect();

    // Natural consumption per layer (sum of groups).
    let layer_natural: Vec<usize> = surviving_layer_indices
        .iter()
        .map(|&li| {
            group_states
                .iter()
                .filter(|gs| gs.layer_idx == li)
                .map(|gs| natural_tokens[&gs.schema.id])
                .sum()
        })
        .collect();

    // Layer-level flex with natural cap so freed budget redistributes.
    let layer_items: Vec<FlexItem> = surviving_layer_indices
        .iter()
        .enumerate()
        .map(|(slot, &li)| {
            let mut item = FlexItem::from_budget(&visible_layers[li].budget, turn_budget);
            let nat = layer_natural[slot];
            item.max_tokens = Some(item.max_tokens.map_or(nat, |m| m.min(nat)));
            item
        })
        .collect();
    let layer_budgets = flexbox_distribute(&layer_items, turn_budget);

    let mut final_selected: Vec<(GroupId, TurnIndex)> = vec![];

    for (slot, &li) in surviving_layer_indices.iter().enumerate() {
        let layer_budget = layer_budgets[slot];

        let layer_groups: Vec<&GroupState> = group_states
            .iter()
            .filter(|gs| gs.layer_idx == li)
            .collect();

        if layer_groups.is_empty() {
            continue;
        }

        // Group-level flex within layer, also capped at natural consumption.
        let group_items: Vec<FlexItem> = layer_groups
            .iter()
            .map(|gs| {
                let mut item = FlexItem::from_budget(&gs.schema.budget, layer_budget);
                let nat = natural_tokens[&gs.schema.id];
                item.max_tokens = Some(item.max_tokens.map_or(nat, |m| m.min(nat)));
                item
            })
            .collect();
        let group_budgets = flexbox_distribute(&group_items, layer_budget);

        for (gi, gs) in layer_groups.iter().enumerate() {
            let group_budget = group_budgets[gi];
            let tc = |idx: TurnIndex| resolver.turn_token_count(gs.schema.id, idx);

            let selected_indices = apply_selection(
                &gs.schema.selection,
                gs.schema.score_threshold,
                &gs.selected,
                Some(group_budget),
                &tc,
            );

            for idx in selected_indices {
                final_selected.push((gs.schema.id, idx));
            }
        }
    }

    // ── Step 12: Emit ─────────────────────────────────────────────────────────
    // Groups are emitted in declaration order per layer, but sorted by
    // descending group score within each layer (ties: declaration order).
    let mut turns: Vec<ResolvedTurn> = vec![];

    for &li in &surviving_layer_indices {
        let layer_id_for_walk = visible_layers[li].id;
        // Groups in this layer, sorted by descending group score (ties: declaration order).
        let mut layer_groups: Vec<&GroupState> = group_states
            .iter()
            .filter(|gs| gs.layer_idx == li)
            .collect();
        // Higher-scored groups emit last (closer to the model's recency bias).
        layer_groups.sort_by(|a, b| {
            a.group_score
                .partial_cmp(&b.group_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for gs in layer_groups {
            // Turns for this group in insertion order.
            let mut group_turns: Vec<TurnIndex> = final_selected
                .iter()
                .filter_map(|(gid, idx)| {
                    if *gid == gs.schema.id {
                        Some(*idx)
                    } else {
                        None
                    }
                })
                .collect();
            group_turns.sort();

            for idx in group_turns {
                // Ghost summary turns (`record_summary_turn` →
                // `append_with_blocks(0..0)`) are zero-token tree-meta anchors
                // with no K/V in any tier. They must never become a
                // `Sealed(Turn)` segment: there is nothing to inject, so
                // emitting one only wastes a window slot and makes the
                // assembler wrap empty boundary markers around no content
                // (which then drops at inject time). Skip them at the
                // projection source so every consumer — the elevate set and the
                // assembler alike — sees only turns that carry injectable K/V.
                if resolver.turn_token_count(gs.schema.id, idx) == 0 {
                    continue;
                }
                // Prefer the producer's `layer_id` from the resolver
                // record; fall back to projector context if the
                // resolver doesn't track origins (e.g. test mocks).
                let origin_layer = resolver
                    .turn_origin(gs.schema.id, idx)
                    .unwrap_or(layer_id_for_walk);
                turns.push(ResolvedTurn {
                    id: TurnId {
                        layer_id: origin_layer,
                        group_id: gs.schema.id,
                        index: idx,
                    },
                });
            }
        }
    }

    let mut segments: Vec<ProjectionSegment> =
        Vec::with_capacity(system_prompt_segments.len() + turns.len());
    segments.extend(system_prompt_segments);
    // Past turns emit as a single `Sealed::Turn` segment each.  The
    // projection engine doesn't know about dialect tokens or boundary
    // wrapping — the projection assembler owns that concern and
    // injects `user_start` before / `assistant_end` after every
    // `Sealed::Turn` from its own pre-tokenised `BoundaryMarkers`
    // (held on the scheduler, threaded through `ApplyContext`).
    segments.extend(
        turns
            .into_iter()
            .map(|t| ProjectionSegment::Sealed(SealedKind::Turn(t, crate::Role::Assistant))),
    );
    Projection {
        segments,
        selections: resolved_selections,
        selection_scores,
    }
}

/// Walk a layer's `system_prompt.items` in declaration order, emitting
/// either each plain section verbatim or each collection's surviving
/// subset (after applying its selection rule).
///
/// A [`SectionSchema`] with `depends_on = Some(cid)` only emits if the
/// named collection materialised ≥ 1 section in this same emission pass.
#[allow(clippy::too_many_arguments)]
fn emit_system_prompt_items<R: ContentResolver>(
    layer: &LayerSchema,
    resolver: &R,
    mode: ProjectionMode,
    selection_state: &SelectionState,
    prior: &PriorBelief,
    scores: &mut super::SelectionScores,
    resolved_selections: &mut Vec<ResolvedSelection>,
) -> Vec<ProjectionSegment> {
    let scoring = mode.collection_scoring();
    let mut out: Vec<ProjectionSegment> = Vec::new();
    // First pass: resolve every Collection in declaration order so
    // their materialised section sets are known when we walk the
    // items for emission. Cached by CollectionId.
    let mut collection_results: std::collections::HashMap<CollectionId, Vec<ProjectionSegment>> =
        std::collections::HashMap::new();
    for item in &layer.system_prompt.items {
        if let SystemPromptItem::Collection(coll) = item {
            let selected = select_collection_sections(
                coll,
                resolver,
                &scoring,
                selection_state,
                prior,
                scores,
            );
            collection_results.insert(coll.id, selected);
        }
    }
    // Collections that materialised ≥1 section, captured up front. The second
    // pass DRAINS `collection_results` (`remove`) as each collection emits, so a
    // `depends_on` template that appears AFTER its collection — e.g. a
    // `tool_block_close` closing the tool block — must consult this set, not the
    // drained map. Reading the drained map would see `None` and silently drop the
    // closing marker, leaving the block unclosed in the materialized context.
    let non_empty_collections: std::collections::HashSet<CollectionId> = collection_results
        .iter()
        .filter(|(_, segs)| !segs.is_empty())
        .map(|(cid, _)| *cid)
        .collect();
    // Second pass: walk items in declaration order, applying the
    // `depends_on` predicate to Sections and using the cached
    // collection results for Collections.
    for item in &layer.system_prompt.items {
        match item {
            SystemPromptItem::Section(s) => {
                let should_emit = match s.depends_on {
                    None => true,
                    Some(cid) => non_empty_collections.contains(&cid),
                };
                if should_emit {
                    push_section_segment(&mut out, s);
                }
            }
            SystemPromptItem::Collection(coll) => {
                if let Some(selected) = collection_results.remove(&coll.id) {
                    // Emit the catalog summary just before the members when the
                    // selection is a *proper subset* (top-k / threshold dropped at
                    // least one member) and the summary section is actually sealed.
                    // When all members survive, nothing was dropped, so the summary
                    // adds nothing and is omitted.
                    if let Some(sum_id) = coll.summary_section {
                        let partial = selected.len() < coll.sections.len();
                        if partial && resolver.section_token_count(sum_id) > 0 {
                            out.push(ProjectionSegment::Sealed(SealedKind::Section(
                                ResolvedSection { id: sum_id },
                            )));
                        }
                    }
                    out.extend(selected);
                }
            }
            SystemPromptItem::SectionTree(tree) => {
                // Resolve the active selection (runtime overrides over authored
                // defaults), then emit each node's chosen option's PRE-SEALED
                // variant for that branch — the one whose substrate K/V was
                // prefilled with exactly this branch's prefix, so the nodes
                // attend to each other correctly with no re-prefill. Each
                // selector's resolved option is recorded so the caller knows what
                // fired, addressed by id.
                let selection = tree.selection(|id| selection_state.get(id));
                for node in &tree.nodes {
                    let opt_idx = node.chosen(&selection);
                    let option = &node.options[opt_idx];
                    if let Some(sel_id) = node.selector_id() {
                        resolved_selections.push(ResolvedSelection {
                            selector: sel_id.to_string(),
                            option: option.id.clone(),
                        });
                    }
                    // Empty options (e.g. a binary node's `absent`) emit nothing.
                    if let Some(v) = option.variant_for(tree.pack(&selection, node.ancestor_dims)) {
                        out.push(ProjectionSegment::Sealed(SealedKind::Section(
                            ResolvedSection { id: v.id },
                        )));
                    }
                }
            }
        }
    }
    out
}

/// Emit a single [`SectionSchema`] as either a [`ProjectionSegment::Sealed`]
/// (content section — K/V comes from substrate) or a
/// [`ProjectionSegment::Generated`] (template section — K/V is
/// live-prefilled at apply time under the runtime left context).
///
/// Panics if a template section has not been tokenised — the schema
/// must run through [`super::Builder::tokenize_templates`] before
/// `project()` is called on a schema containing template items.
fn push_section_segment(out: &mut Vec<ProjectionSegment>, s: &SectionSchema) {
    if s.is_template {
        let tokens = s.template_tokens.clone().unwrap_or_else(|| {
            panic!(
                "projection: template section {:?} (id {}) has no pre-tokenised tokens — \
                 call Builder::tokenize_templates before project()",
                s.name,
                s.id.raw(),
            )
        });
        out.push(ProjectionSegment::Generated {
            tokens,
            identity: GeneratedIdentity {
                name: s.name.clone(),
                position: out.len(),
            },
        });
    } else {
        out.push(ProjectionSegment::Sealed(SealedKind::Section(
            ResolvedSection { id: s.id },
        )));
    }
}

/// Apply a collection's selection rule, returning the surviving
/// sections in **declaration order**.
///
/// Selection picks by salience (score, then priority); emission
/// preserves authored structure (declaration order).  Sections below
/// `score_threshold` are filtered out before selection.
#[allow(clippy::too_many_arguments)]
fn select_collection_sections<R: ContentResolver>(
    coll: &SectionCollection,
    resolver: &R,
    scoring: &CollectionScoring,
    selection_state: &SelectionState,
    prior: &PriorBelief,
    scores: &mut super::SelectionScores,
) -> Vec<ProjectionSegment> {
    if coll.sections.is_empty() {
        return Vec::new();
    }
    match &coll.selection {
        SelectionRule::AlwaysVisible => {
            let mut out = Vec::with_capacity(coll.sections.len());
            for s in &coll.sections {
                push_section_segment(&mut out, s);
            }
            out
        }
        SelectionRule::TopK { .. } => {
            // Belief-driven selection: the collection's policy (RelLeak budget +
            // hysteresis) decides the surviving set from the per-section scores,
            // seeded from the prior projection's belief so decay/reinforcement
            // carries across a turn. The count budget subsumes the old `top_k` k
            // and `min_score` the old `score_threshold`; emission stays in
            // declaration order. Each member's belief is recorded on `scores`.
            let fresh: Vec<f32> = coll
                .sections
                .iter()
                .map(|s| resolver.section_score(s.id))
                .collect();
            let (prior_scores, prior_selected) = prior.collection(&coll.name, &coll.sections);
            let beliefs = crate::provenance::belief_step(
                &fresh,
                &prior_scores,
                &prior_selected,
                coll.policy.config.section_policy(0),
                coll.policy.config.budget(),
            );
            if tracing::enabled!(tracing::Level::TRACE) {
                let scores_str = coll
                    .sections
                    .iter()
                    .zip(&beliefs)
                    .map(|(s, b)| {
                        format!(
                            "{}={:.1}{}",
                            s.name,
                            b.score,
                            if b.selected { "*" } else { "" }
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                tracing::trace!(collection = %coll.name, scores = %scores_str, "belief selection");
            }
            let mut out = Vec::new();
            for (s, b) in coll.sections.iter().zip(&beliefs) {
                scores.set_section(s.id, b.score);
                if b.selected {
                    push_section_segment(&mut out, s);
                }
            }
            out
        }
        SelectionRule::Single => {
            let best = coll
                .sections
                .iter()
                .map(|s| {
                    let score = resolver.section_score(s.id);
                    (s, score)
                })
                .filter(|(_, score)| !scoring.apply_threshold || *score >= coll.score_threshold)
                .max_by(|(a, asc), (b, bsc)| {
                    asc.partial_cmp(bsc)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(
                            a.priority
                                .partial_cmp(&b.priority)
                                .unwrap_or(std::cmp::Ordering::Equal),
                        )
                });
            let mut out = Vec::new();
            if let Some((s, _)) = best {
                push_section_segment(&mut out, s);
            }
            out
        }
        SelectionRule::Named { selector } => {
            // Explicit by-name pick: emit exactly the member whose `name`
            // matches the runtime selector value. Score-independent — no
            // threshold, no provenance relevance. Nothing is emitted if the selector
            // is unset or names no member.
            let mut out = Vec::new();
            if let Some(target) = selection_state.get(selector) {
                if let Some(s) = coll.sections.iter().find(|s| s.name == target) {
                    push_section_segment(&mut out, s);
                }
            }
            tracing::trace!(
                collection = %coll.name,
                selector = %selector,
                target = selection_state.get(selector).unwrap_or(""),
                selected = out.len(),
                "projection (named)"
            );
            out
        }
        SelectionRule::Sequence { .. } => {
            // No sensible "recent" semantics for sections.  Fall back
            // to AlwaysVisible.
            let mut out = Vec::with_capacity(coll.sections.len());
            for s in &coll.sections {
                push_section_segment(&mut out, s);
            }
            out
        }
    }
}
