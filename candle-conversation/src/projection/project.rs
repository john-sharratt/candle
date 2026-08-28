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

use std::collections::{HashMap, HashSet};

use super::adaptive::AnchorMember;
use super::ids::{
    CollectionId, GroupId, LayerId, SectionId, TimelineId, TurnId, TurnIndex, TurnKey,
};
use super::reconcile::{flexbox_distribute, FlexItem};
use super::schema::{
    GroupSchema, LayerSchema, Schema, ScoreFormula, SectionCollection, SectionSchema,
    SelectionRule, SystemPromptItem, SystemPromptSchema, TreeCollection,
};
use super::selection::{apply_selection, resolve_default_turn, trim_to_budget_low_score_first};
use crate::substrate::ContentResolver;
use crate::summary_tree::{NodeId, SelectionDiagnostics, SelectionOrigin};

/// Fixed scoring formula used for all turn scoring and section selection.
/// Calibrated against real Qwen3-30B-A3B Q-vector data; span α=2.0 with
/// pragmatic-only depth weights dominates all other formulas for tool
/// selection (min_ratio 5.54 vs 1.14–1.40 for alternatives).
pub(super) const FIXED_FORMULA: ScoreFormula = ScoreFormula::Span { alpha: 2.0 };

/// Which inference phase a projection is being computed for.
///
/// Decode and prefill produce structurally different Q vectors and so need
/// different collection-scoring configs (calibrated 2026-05-17 against real
/// Qwen3-30B-A3B data):
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
/// Carries the resolved [`TimelineId`] — the *conversation* this turn belongs to
/// — stamped ONCE at projection where the target is unambiguous (see
/// [`ContentResolver::turn_timeline`]).  Downstream consumers read [`Self::key`]
/// directly instead of re-deriving the timeline from the group: a turn's
/// identity is `(timeline, index)`, and re-resolving `group → timeline` per
/// consumer is exactly what once let the reproject pick the wrong conversation's
/// timeline and drop a slot's whole history.  `None` only for mock resolvers in
/// tests (which never reach the apply path).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedTurn {
    pub id: TurnId,
    pub timeline: Option<TimelineId>,
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

    /// The fully-resolved `(timeline, index)` identity — the only correct key for
    /// substrate turn lookups.  `None` for mock-resolver turns in tests.
    #[inline]
    pub fn key(&self) -> Option<TurnKey> {
        self.timeline.map(|timeline| TurnKey {
            timeline,
            index: self.id.index,
        })
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
    /// Why each selected turn entered the slot, keyed by `(group, turn)`. The
    /// score-density path tags each pick from the summary forest (hard anchor /
    /// provenance / coverage / …); the rule-based path tags `Recent` (inside the
    /// recency window) vs `Historical` (top-k by score). Read by
    /// [`super::event::from_projection_with_origins`] so the persisted projection
    /// record / GUI can show "why is this turn in my context?". Empty entries
    /// (e.g. the live user message) carry no origin.
    pub selection_origins: HashMap<TurnKey, SelectionOrigin>,
}

/// A belief node's identity: a named section collection (the tool catalog) or a
/// belief-driven turn group (repo_map clusters, code scopes). Group names are
/// globally unique in a schema, but a collection and a turn group could share a
/// name, so the two namespaces are kept distinct by the variant. Turn-group
/// members are sub-keyed by their `TurnIndex`, collection members by section name.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum GroupKey {
    Collection(String),
    TurnGroup(String),
}

/// The prior projection's belief, per belief node (collection **or** turn group),
/// used to seed the current projection so online decay/reinforcement carries
/// across a turn's reprojections. Built by the scheduler from the previous
/// [`super::ProjectionEvent`]; empty (the default) means a fresh, stateless
/// belief for this projection.
#[derive(Debug, Clone, Default)]
pub struct PriorBelief {
    beliefs: HashMap<GroupKey, HashMap<String, (f32, bool, bool)>>,
}

impl PriorBelief {
    /// Record a section's prior belief (score + whether it was selected).
    pub fn set(
        &mut self,
        collection: &str,
        section: &str,
        score: f32,
        selected: bool,
        qualified: bool,
    ) {
        self.beliefs
            .entry(GroupKey::Collection(collection.to_string()))
            .or_default()
            .insert(section.to_string(), (score, selected, qualified));
    }

    /// Record a turn's prior belief in a belief-driven turn group (repo_map
    /// clusters, code scopes). The turn-axis analogue of [`Self::set`].
    pub fn set_turn(
        &mut self,
        group: &str,
        turn: TurnKey,
        score: f32,
        selected: bool,
        qualified: bool,
    ) {
        self.beliefs
            .entry(GroupKey::TurnGroup(group.to_string()))
            .or_default()
            .insert(turn_belief_key(turn), (score, selected, qualified));
    }

    /// Rebuild the prior belief from a completed projection's selection — every
    /// collection member's `(score, selected)`, plus each **selected** memory
    /// turn of a belief-driven turn group. The scheduler stores this on the
    /// decode state after each reprojection so the next one seeds from it,
    /// carrying the online belief across a turn's reprojections.
    ///
    /// Unlike the collection axis — which carries *every* member so an
    /// unselected tool keeps its decaying score — the turn axis carries **only
    /// selected** turns. This divergence is deliberate: the candidate set of a
    /// belief-driven group is bounded (clusters, scopes — not unbounded dialogue,
    /// which is a recency `Sequence` group and never consulted here), and an
    /// unselected turn with a strong fresh score is re-admitted by the next scan
    /// anyway, so an unselected turn reseeds fresh (score 0) instead of decaying —
    /// which keeps the carry bounded by the group's budget without losing lock-on.
    /// The live user message (`index == u32::MAX`) is the probe, not a memory.
    pub fn from_selection(sel: &super::ProjectionSelection) -> PriorBelief {
        let mut pb = PriorBelief::default();
        for item in &sel.system {
            if let super::SystemItem::Collection { name, sections } = item {
                for s in sections {
                    pb.set(name, &s.name, s.score, s.selected, s.qualified);
                }
            }
        }
        for t in &sel.turns {
            if t.index != u32::MAX && t.selected {
                // Carry the belief under the turn's full `(timeline, index)`
                // identity — the persisted selection stamps the timeline exactly
                // so consumers never re-derive `group → timeline`, which is
                // ambiguous once a group holds many conversations.
                if let Some(tl) = t.timeline.and_then(TimelineId::from_raw) {
                    pb.set_turn(
                        &t.group,
                        TurnKey::new(tl, TurnIndex(t.index)),
                        t.score,
                        true,
                        t.qualified,
                    );
                }
            }
        }
        pb
    }

    /// Overlay `newer` onto this belief, per node: nodes present in `newer`
    /// replace this belief's entry wholesale; nodes absent from `newer` keep
    /// their existing state. A turn whose projection never ran a collection (e.g.
    /// tools dial off → the tools collection materialised nothing and is absent
    /// from the selection) says nothing about that node's belief, so carrying it
    /// across the turn must not erase it.
    pub fn merge_from(&mut self, newer: &PriorBelief) {
        for (key, members) in &newer.beliefs {
            self.beliefs.insert(key.clone(), members.clone());
        }
    }

    /// Scale every carried score by `factor`, turning the belief from a *hard
    /// pin* into a *soft prior* for the next turn's opening projection.
    ///
    /// The carried belief exists so a genuine multi-turn tool workflow keeps
    /// its lock-on instead of resetting to catalog order. But at full strength
    /// it also lets the *previous* turn's tool outrank the correct fresh signal
    /// for a topic-changed turn: the incoming query's evidence is prefill-Q at
    /// token 0 (the call↔definition domain gap, ≈0 for the right tool) and only
    /// builds from the assistant's decode-Q over the first ~64 tokens, so a
    /// full-strength prior owns the whole opening window and the model can
    /// commit to a wrong framing before the correct tool is ever selected.
    /// Halving the carried scores lets that fresh decode-Q overtake a stale
    /// tool within a few tokens; a real continuation re-accumulates its belief
    /// just as fast from its own decode-Q, so continuity is preserved. The
    /// `selected` flags are untouched — the score is the lever selection ranks
    /// on. See `docs/tool_selection_provenance_results.md` §24.
    pub fn decay_scores(&mut self, factor: f32) {
        for members in self.beliefs.values_mut() {
            for (score, _selected, _qualified) in members.values_mut() {
                *score *= factor;
            }
        }
    }

    /// Turn-boundary challenger rule for a top-N collection (`budget_max ≥ 3`):
    /// give the strongest FRESH signal a slot even when it's below the
    /// selection threshold, so a topic-changed turn's new intent can break in
    /// **without lowering `min_score`** (which would admit the noise floor
    /// mid-turn). Applied once, at the turn's first fresh-scored reprojection.
    ///
    /// `fresh` is `(section_name, fresh_score)` for the collection this scan.
    /// The challenger is the highest fresh-scored section that isn't already
    /// selected. It's seeded as a selected member at `seed_score` (≥
    /// `evict_score`, so it survives the belief step); if the carried selection
    /// is already at `budget_max` the lowest-scored incumbent is evicted first.
    /// A carried selection below capacity keeps all its members — the strong
    /// signals survive and the challenger simply fills a spare slot; only a full
    /// selection displaces its weakest. RelLeak then decays the challenger back
    /// out over the turn if its fresh score doesn't hold up.
    pub fn seat_turn_boundary_challenger(
        &mut self,
        key: GroupKey,
        fresh: &[(String, f32)],
        budget_max: usize,
        seed_score: f32,
    ) {
        if budget_max < 3 {
            return;
        }
        let members = self.beliefs.entry(key).or_default();
        let is_selected = |name: &str| members.get(name).map(|&(_, sel, _)| sel).unwrap_or(false);
        // The challenger: strongest fresh signal not already selected.
        let challenger = fresh
            .iter()
            .filter(|(name, score)| *score > 0.0 && !is_selected(name))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(name, _)| name.clone());
        let Some(challenger) = challenger else {
            return;
        };
        // Currently-selected incumbents, by carried score.
        let selected: Vec<(String, f32)> = members
            .iter()
            .filter(|(_, (_, sel, _))| *sel)
            .map(|(name, (score, _, _))| (name.clone(), *score))
            .collect();
        if selected.len() >= budget_max {
            if let Some((low, _)) = selected
                .iter()
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                members.insert(low.clone(), (0.0, false, false));
            }
        }
        // Seated, but NOT qualified: the challenger is deliberately placed below
        // the selection threshold ("break in without lowering `min_score`"), so it
        // has shown no evidence of crossing a bar. Marking it qualified would hand
        // it to the early-decode carry floor, which would pin it at the seed score
        // for the whole grace window — turning a speculative slot into a lock-on
        // and squatting the budget the real tool needs. It has to earn the floor by
        // reaching `min_score` on its own belief, like any other member.
        members.insert(challenger, (seed_score, true, false));
    }

    /// Aligned `(scores, selected)` for a collection's sections in declaration
    /// order — the seed for [`crate::provenance::belief_step`]. Unknown sections
    /// read `(0.0, false)`.
    fn collection(
        &self,
        name: &str,
        sections: &[SectionSchema],
    ) -> (Vec<f32>, Vec<bool>, Vec<bool>) {
        let map = self.beliefs.get(&GroupKey::Collection(name.to_string()));
        let mut scores = vec![0.0f32; sections.len()];
        let mut selected = vec![false; sections.len()];
        let mut qualified = vec![false; sections.len()];
        if let Some(m) = map {
            for (i, s) in sections.iter().enumerate() {
                if let Some(&(sc, sel, q)) = m.get(&s.name) {
                    scores[i] = sc;
                    selected[i] = sel;
                    qualified[i] = q;
                }
            }
        }
        (scores, selected, qualified)
    }

    /// Aligned `(scores, selected)` for a turn group's candidate turns, in the
    /// given order — the seed for [`crate::provenance::belief_step`] on the turn
    /// axis. Unknown turns read `(0.0, false)`.
    fn turn_group(
        &self,
        group: &str,
        turns: &[(TurnKey, f32)],
    ) -> (Vec<f32>, Vec<bool>, Vec<bool>) {
        let map = self.beliefs.get(&GroupKey::TurnGroup(group.to_string()));
        let mut scores = vec![0.0f32; turns.len()];
        let mut selected = vec![false; turns.len()];
        let mut qualified = vec![false; turns.len()];
        if let Some(m) = map {
            for (i, (key, _)) in turns.iter().enumerate() {
                if let Some(&(sc, sel, q)) = m.get(&turn_belief_key(*key)) {
                    scores[i] = sc;
                    selected[i] = sel;
                    qualified[i] = q;
                }
            }
        }
        (scores, selected, qualified)
    }
}

/// Belief-map key for a turn: `"<timeline>:<index>"`.
///
/// Qualified by timeline because a belief-driven group holds many conversations
/// (`code_reading` declares one per file) and turn indices restart per timeline —
/// a bare index would make turn 3 of every file share one carried belief. Every
/// producer of a turn-group belief key (the carry, and the scheduler's
/// turn-boundary challenger) MUST route through here so they stay aligned.
pub(crate) fn turn_belief_key(turn: TurnKey) -> String {
    format!("{}:{}", turn.timeline.raw(), turn.index.0)
}

#[cfg(test)]
impl PriorBelief {
    /// A collection's member map (test-only readable view over `beliefs`).
    /// `pub(crate)` so the scheduler's resume-seeding test can assert the
    /// recovered belief's content, not merely its presence.
    pub(crate) fn coll(&self, name: &str) -> &HashMap<String, (f32, bool, bool)> {
        &self.beliefs[&GroupKey::Collection(name.to_string())]
    }
    /// A turn group's member map (test-only readable view over `beliefs`).
    fn tgroup(&self, name: &str) -> &HashMap<String, (f32, bool, bool)> {
        &self.beliefs[&GroupKey::TurnGroup(name.to_string())]
    }
    fn is_empty(&self) -> bool {
        self.beliefs.is_empty()
    }
}

#[cfg(test)]
mod prior_belief_tests {
    use super::{GroupKey, PriorBelief, TimelineId, TurnIndex, TurnKey};

    /// Belief-map turn keys are timeline-qualified (`"<timeline>:<index>"`), so
    /// turn 3 of one conversation can't collide with turn 3 of another.
    fn tk(n: u32) -> TurnKey {
        TurnKey::new(TimelineId::for_test(1), TurnIndex(n))
    }
    fn tkey(n: u32) -> String {
        format!("{}:{}", TimelineId::for_test(1).raw(), n)
    }

    #[test]
    fn merge_from_overlays_present_collections_and_preserves_absent_ones() {
        // The carried belief holds a tools lock-on from earlier turns.
        let mut carried = PriorBelief::default();
        carried.set("tools", "calculator", 2500.0, true, true);
        carried.set("tools", "datetime", 400.0, false, false);

        // A tools-off turn harvests a belief WITHOUT the tools key (the
        // collection materialised nothing, so the selection omitted it) but
        // with fresh state for another collection.
        let mut newer = PriorBelief::default();
        newer.set("notes", "pinned", 40.0, true, true);

        carried.merge_from(&newer);

        // The absent collection survives untouched — the tools lock-on is
        // NOT erased by a turn that said nothing about tools...
        assert_eq!(carried.coll("tools")["calculator"], (2500.0, true, true));
        assert_eq!(carried.coll("tools")["datetime"], (400.0, false, false));
        // ...and the present collection is overlaid.
        assert_eq!(carried.coll("notes")["pinned"], (40.0, true, true));
    }

    #[test]
    fn merge_from_replaces_a_present_collection_wholesale() {
        let mut carried = PriorBelief::default();
        carried.set("tools", "calculator", 2500.0, true, true);
        carried.set("tools", "datetime", 400.0, false, false);

        // A newer turn re-projected tools with a different member set: the
        // collection entry is replaced, not unioned — sections absent from the
        // newer selection are gone (their belief genuinely ended).
        let mut newer = PriorBelief::default();
        newer.set("tools", "web_search", 900.0, true, true);

        carried.merge_from(&newer);

        let tools = carried.coll("tools");
        assert_eq!(tools.get("web_search"), Some(&(900.0, true, true)));
        assert_eq!(tools.get("calculator"), None);
        assert_eq!(tools.len(), 1);
    }

    #[test]
    fn merge_from_empty_belief_is_a_no_op() {
        // A projection-skipped turn harvests a default belief: merging it must
        // not disturb the carry.
        let mut carried = PriorBelief::default();
        carried.set("tools", "calculator", 2500.0, true, true);

        carried.merge_from(&PriorBelief::default());

        assert_eq!(carried.coll("tools")["calculator"], (2500.0, true, true));
    }

    #[test]
    fn decay_scores_scales_scores_and_leaves_selected_flags() {
        let mut carried = PriorBelief::default();
        // A carried tool lock-on (sub_run) plus a runner-up (datetime), as at
        // a topic-changed turn boundary.
        carried.set("tools", "sub_run", 1009.0, true, true);
        carried.set("tools", "datetime", 0.0, false, false);
        carried.set("notes", "pinned", 40.0, true, true);

        carried.decay_scores(0.5);

        // Scores halve so the fresh decode-Q of a new query can overtake the
        // stale leader early; the `selected` flags are the ranking input, not
        // the lever, so they carry unchanged.
        assert_eq!(carried.coll("tools")["sub_run"], (504.5, true, true));
        assert_eq!(carried.coll("tools")["datetime"], (0.0, false, false));
        assert_eq!(carried.coll("notes")["pinned"], (20.0, true, true));
    }

    #[test]
    fn decay_scores_on_empty_belief_is_a_no_op() {
        let mut carried = PriorBelief::default();
        carried.decay_scores(0.5);
        assert!(carried.is_empty());
    }

    // ── Turn-group belief (the generalized axis) ────────────────────────────

    #[test]
    fn from_selection_carries_only_selected_memory_turns_by_group() {
        use crate::projection::event::{ProjectionSelection, SelectedTurn};
        let mk = |group: &str, index: u32, selected: bool, score: f32| SelectedTurn {
            // These fixtures stand for turns that won their slot on their own
            // belief, so qualification tracks selection.
            qualified: selected,
            layer: "repo_map".to_string(),
            group: group.to_string(),
            index,
            role: "user".to_string(),
            tokens: 10,
            kind: crate::summary_tree::TurnKind::Normal,
            reason: None,
            timeline: Some(1),
            selected,
            score,
        };
        let sel = ProjectionSelection {
            system: Vec::new(),
            turns: vec![
                mk("structure", 0, true, 500.0),       // carried
                mk("structure", 1, false, 200.0),      // unselected → not carried
                mk("structure", u32::MAX, false, 0.0), // live message → skipped
            ],
        };
        let pb = PriorBelief::from_selection(&sel);
        let group = pb.tgroup("structure");
        assert_eq!(group.get(&tkey(0)), Some(&(500.0, true, true)));
        assert_eq!(group.get(&tkey(1)), None);
        assert_eq!(group.len(), 1, "only the selected memory turn is carried");
    }

    #[test]
    fn turn_group_aligns_carried_belief_to_candidate_order() {
        let mut pb = PriorBelief::default();
        pb.set_turn("structure", tk(3), 500.0, true, true);
        pb.set_turn("structure", tk(7), 200.0, false, false);
        // Candidates in a different order than they were carried; the seed must
        // realign by turn index, and unknown turns read (0.0, false).
        let turns = vec![(tk(7), 0.0), (tk(3), 0.0), (tk(5), 0.0)];
        let (scores, selected, _qualified) = pb.turn_group("structure", &turns);
        assert_eq!(scores, vec![200.0, 500.0, 0.0]);
        assert_eq!(selected, vec![false, true, false]);
    }

    #[test]
    fn challenger_seats_a_turn_group_member_by_index_key() {
        let mut pb = PriorBelief::default();
        // One carried cluster; a spare slot at budget_max 3.
        pb.set_turn("clusters", tk(0), 900.0, true, true);
        pb.seat_turn_boundary_challenger(
            GroupKey::TurnGroup("clusters".into()),
            &[(tkey(4), 246.0), (tkey(9), 30.0)],
            3,
            100.0,
        );
        let g = pb.tgroup("clusters");
        assert_eq!(g[&tkey(0)], (900.0, true, true), "incumbent survives");
        assert_eq!(
            g[&tkey(4)],
            (100.0, true, false),
            "strongest fresh turn seated, but unqualified — it never crossed the bar"
        );
        assert!(g.get(&tkey(9)).is_none(), "weaker rival not seated");
    }

    #[test]
    fn merge_from_and_decay_span_both_axes() {
        let mut carried = PriorBelief::default();
        carried.set("tools", "calc", 400.0, true, true);
        carried.set_turn("structure", tk(2), 800.0, true, true);
        // decay scales scores on both the collection and turn-group axes.
        carried.decay_scores(0.5);
        assert_eq!(carried.coll("tools")["calc"], (200.0, true, true));
        assert_eq!(carried.tgroup("structure")[&tkey(2)], (400.0, true, true));
        // merge overlays a turn group wholesale, leaving the collection intact.
        let mut newer = PriorBelief::default();
        newer.set_turn("structure", tk(5), 900.0, true, true);
        carried.merge_from(&newer);
        assert_eq!(
            carried.tgroup("structure").get(&tkey(2)),
            None,
            "replaced wholesale"
        );
        assert_eq!(carried.tgroup("structure")[&tkey(5)], (900.0, true, true));
        assert_eq!(
            carried.coll("tools")["calc"],
            (200.0, true, true),
            "other axis intact"
        );
    }

    #[test]
    fn challenger_fills_a_spare_slot_without_evicting() {
        // Carried a single (topic-stale) tool from the previous turn; the new
        // query's strongest fresh signal is a below-threshold datetime.
        let mut b = PriorBelief::default();
        b.set("tools", "sub_run", 350.0, true, true);
        // budget_max 3 with one incumbent → room for the challenger, no eviction.
        b.seat_turn_boundary_challenger(
            GroupKey::Collection("tools".into()),
            &[("datetime".into(), 246.0), ("weather".into(), 30.0)],
            3,
            1000.0,
        );
        // The incumbent survives; datetime is seated selected at the seed score.
        assert_eq!(b.coll("tools")["sub_run"], (350.0, true, true));
        // Seated selected, but unqualified: it is below the bar by construction.
        assert_eq!(b.coll("tools")["datetime"], (1000.0, true, false));
        // A weaker fresh rival is NOT seated — only the single strongest.
        assert!(b.coll("tools").get("weather").is_none());
    }

    #[test]
    fn challenger_evicts_the_weakest_incumbent_when_full() {
        let mut b = PriorBelief::default();
        b.set("tools", "http_request", 1800.0, true, true);
        b.set("tools", "sub_run", 400.0, true, true); // weakest incumbent
        b.set("tools", "code_run", 1200.0, true, true);
        // At budget_max=3 → the challenger displaces the lowest-scored incumbent.
        b.seat_turn_boundary_challenger(
            GroupKey::Collection("tools".into()),
            &[("datetime".into(), 246.0)],
            3,
            1000.0,
        );
        assert_eq!(b.coll("tools")["sub_run"], (0.0, false, false)); // evicted
        assert_eq!(b.coll("tools")["datetime"], (1000.0, true, false)); // seated, unqualified
                                                                        // The strong incumbents survive.
        assert_eq!(b.coll("tools")["http_request"], (1800.0, true, true));
        assert_eq!(b.coll("tools")["code_run"], (1200.0, true, true));
    }

    #[test]
    fn challenger_is_a_no_op_when_top_fresh_is_already_selected() {
        // Continuation: the strongest fresh signal is the carried tool itself,
        // and no other fresh signal exists → nothing to seat, carry untouched.
        let mut b = PriorBelief::default();
        b.set("tools", "datetime", 1500.0, true, true);
        b.seat_turn_boundary_challenger(
            GroupKey::Collection("tools".into()),
            &[("datetime".into(), 900.0)],
            3,
            1000.0,
        );
        assert_eq!(b.coll("tools")["datetime"], (1500.0, true, true));
        assert_eq!(b.coll("tools").len(), 1);
    }

    #[test]
    fn challenger_does_not_apply_below_top_3() {
        // A top-1/top-2 collection has no spare slot to lend a challenger.
        let mut b = PriorBelief::default();
        b.set("tools", "sub_run", 350.0, true, true);
        b.seat_turn_boundary_challenger(
            GroupKey::Collection("tools".into()),
            &[("datetime".into(), 246.0)],
            2,
            1000.0,
        );
        assert_eq!(b.coll("tools")["sub_run"], (350.0, true, true));
        assert!(b.coll("tools").get("datetime").is_none());
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

/// Selector for the `tools_enabled` optional_group that gates the WHOLE tool block
/// (overview, `<tools>` markers, catalog, summary). `Absent` omits it entirely —
/// what a tools-off layer (e.g. `code_reading`) sets so the projection never
/// selects (or elevates) any tool section. Kept here so the string can't drift
/// between the schema, chat.rs, and the ingest path.
pub const TOOLS_ENABLED_SELECTOR: &str = "tools_enabled";

/// Selector that force-pins a single member of a belief-driven (`TopK`) tool
/// catalog. When set to a tool's section name the `TopK` arm emits **exactly**
/// that member and skips belief selection entirely — score-independent, the same
/// by-name pick `SelectionRule::Named` does, but available on the production
/// `tools` collection without switching its rule. The code_read summary path
/// sets it to `file_read` so the prefilled `read_file` tool_call/response is
/// backed by a coherent, present tool definition (tools ON, one tool, forced).
/// Unset ⇒ ordinary belief selection. Kept here so the string can't drift.
pub const FORCE_TOOL_SELECTOR: &str = "force_tool";

/// Separator between member names in a [`FORCE_TOOL_SELECTOR`] value. A pin may
/// name several tools when one ingest turn prefills calls to more than one.
pub const FORCE_TOOL_SEPARATOR: char = ',';

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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
        None,
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
    decode_pos: Option<usize>,
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
            // NOTE: this stays a contiguous prefix (visible index == schema
            // index) so `li == target_layer_idx` below is correct — the
            // diagnostic kill switch is applied INSIDE the loop, not here, to
            // preserve that correspondence.
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
        selected: Vec<(TurnKey, f32)>, // ((timeline, index), score), insertion order
        group_score: f32,
        /// The §8 score-density selector produced `selected` (already
        /// budget-fit and in chronological order — summaries ABOVE the turns
        /// they cover). When set, downstream steps must NOT re-run the
        /// rule-based selection or re-sort by raw `TurnIndex` (that would place
        /// every summary AFTER its own content); the order is emitted verbatim.
        score_density: bool,
    }

    let mut group_states: Vec<GroupState> = Vec::new();
    // Why each selected turn entered the slot — collected as we select, surfaced
    // on the returned `Projection` so the persisted record / GUI can show it.
    let mut selection_origins: HashMap<TurnKey, SelectionOrigin> = HashMap::new();

    // Hoisted out of the per-layer loop below: the kill switch is process-global,
    // so "is anything disabled?" has one answer for the whole assembly. Asking it
    // per layer made this the second-hottest resolved symbol in an ingest profile.
    let any_layer_disabled = super::layer_toggle::any_layer_disabled();

    for (li, layer) in visible_layers.iter().enumerate() {
        let layer_is_target = li == target_layer_idx;
        // Runtime diagnostic kill switch: a non-target layer toggled off
        // contributes nothing to the assembly (its groups are never scored or
        // selected). The target layer is never skipped — that would leave the
        // projection with nothing to emit. See `super::layer_toggle`.
        if !layer_is_target
            && any_layer_disabled
            && super::layer_toggle::is_layer_disabled(&layer.name)
        {
            continue;
        }
        for group in &layer.groups {
            // Masking: for the target layer, only the target group is visible.
            if layer_is_target && group.id != target.group {
                continue;
            }

            // Candidates are every turn in the group — raw turns AND summary
            // forest nodes alike — across EVERY conversation the group holds.
            // A group is a shape, not a conversation: `code_reading` declares one
            // timeline per file, so candidates are keyed by `(timeline, index)`.
            // A summary carries its own K/V and provenance signature, so a
            // cross-corpus hit can score a summary node directly and pull it into
            // the window when the raw turns it covers didn't themselves score high
            // enough. The `score_density` path (target group with a tree) still
            // overrides this list wholesale; on the belief and rule-based paths,
            // summaries compete on score and the descendant-dedup below keeps the
            // SPECIFIC over the coarse.
            let all_turns: Vec<(TurnKey, f32)> = resolver
                .group_turns(group.id)
                .into_iter()
                .map(|key| {
                    let score = resolver.turn_score(key);
                    (key, score)
                })
                .collect();

            let tc = |key: TurnKey| resolver.turn_token_count(key);

            // Score-density override: when this is the *target* group
            // and a summary tree exists for the target timeline, swap
            // out the rule-based unbounded selection for the §8
            // algorithm.  The result is already budget-fitted into
            // `layer.window` so the flexbox step caps the group's
            // natural consumption at the score-density total.
            let mut score_density_used = false;
            let selected: Vec<(TurnKey, f32)> = if layer_is_target && group.id == target.group {
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
                    // Score-density picks come from the target timeline's own
                    // forest, so every pick keys to `target.timeline`.
                    let mut diag = SelectionDiagnostics::new(layer.window as u32);
                    for (turn_idx, origin, score) in &picks {
                        let key = TurnKey::new(target.timeline, *turn_idx);
                        let tokens = resolver.turn_token_count(key) as u32;
                        diag.push(NodeId(turn_idx.0), *origin, *score, tokens);
                        selection_origins.insert(key, *origin);
                    }
                    diag.pending_count = resolver.pending_summary_len(target.timeline);
                    sink(diag);
                    picks
                        .iter()
                        .map(|(idx, _origin, score)| (TurnKey::new(target.timeline, *idx), *score))
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            // Fall through to the rule-based unbounded selection path
            // when score-density wasn't applicable.
            let mut selected: Vec<(TurnKey, f32)> = if score_density_used {
                selected
            } else if resolver.target_is_ingest_self() && group.id == target.group {
                // An append-only ingest conversation reading its OWN turns while
                // it generates. Every candidate is selected, unconditionally.
                //
                // Belief selection is for RETRIEVAL — ranking a corpus against a
                // probe. It is the wrong instrument for a conversation reading
                // its own history: an ingest turn is inserted and decoded against
                // immediately, so it carries no wide-Q belief yet, scores zero,
                // and is filtered out by its own group's band (repo_map
                // `min_score` 250, code_reading `score_threshold` 100). The
                // decode is then left holding only its own user turn — which is
                // how a folder-summary decode came to answer "the user hasn't
                // asked a specific question yet" with the request one turn back.
                //
                // `group_turns` has already masked the candidates to the target
                // timeline for an append-only target, so "every candidate" is
                // exactly "this conversation's own turns" — 2-4 for a repo_map
                // folder, one prior turn for a forked code_read scope. Phase-1
                // selection is unbounded by design (see the module header): the
                // flexbox pass below still trims to `layer.window`.
                //
                // Dialogue retrieving this same content is untouched — the target
                // is then the dialogue layer, not an append-only one, so this arm
                // never fires and belief gates as before.
                all_turns
                    .iter()
                    .map(|(key, score)| {
                        selection_scores.set_turn(*key, *score, true);
                        selection_origins.insert(*key, SelectionOrigin::IngestSelf);
                        (*key, *score)
                    })
                    .collect()
            } else if group.is_belief_driven() {
                // Belief-driven turn selection: RelLeak (hysteresis + budget) over
                // the fresh per-turn wide-Q scores, seeded from the carried prior
                // so a locked-on cluster/scope survives across the turn's
                // reprojections. The turn-axis analogue of `select_collection_sections`'s
                // TopK arm — same `belief_step`, keyed by turn instead of section.
                let (prior_scores, prior_selected, prior_qualified) =
                    prior.turn_group(&group.name, &all_turns);
                let fresh: Vec<f32> = all_turns.iter().map(|(_, s)| *s).collect();
                // Early-decode grace: within the opening window the selection band
                // is lowered and carried picks are floored (see `PolicyConfig::windowed`).
                let (mut cfg, floor) = group.belief_config(all_turns.len()).windowed(decode_pos);
                // Concept B: attention mass extends the member budget within the
                // declared rail (`budget_adaptive.absolute_max`).
                if let Some(ba) = &group.budget_adaptive {
                    cfg.budget_max =
                        ba.effective_max(cfg.budget_max, resolver.group_attention_mass(group.id));
                }
                let beliefs = crate::provenance::belief_step(
                    &fresh,
                    &prior_scores,
                    &prior_selected,
                    &prior_qualified,
                    cfg.section_policy(0),
                    cfg.budget(),
                    floor,
                );
                let mut out = Vec::new();
                for ((key, _), b) in all_turns.iter().zip(&beliefs) {
                    // Record every candidate's belief so `build_selection` stamps
                    // it and the next reprojection can seed from it.
                    selection_scores.set_turn(*key, b.score, b.qualified);
                    if b.selected {
                        out.push((*key, b.score));
                        // Concept C: a pick whose score came from neighbor drag
                        // (not its own vote) is stamped `Locality` so the record
                        // answers "why is this here?" truthfully.
                        let origin = if resolver.turn_locality_boosted(*key) {
                            SelectionOrigin::Locality
                        } else {
                            SelectionOrigin::Belief
                        };
                        selection_origins.insert(*key, origin);
                    }
                }
                out
            } else {
                let selected_indices = apply_selection(
                    &group.selection,
                    group.score_threshold.unwrap_or(0.0),
                    &all_turns,
                    None,
                    &tc,
                );

                let selected: Vec<(TurnKey, f32)> = selected_indices
                    .iter()
                    .map(|&key| {
                        let score = all_turns
                            .iter()
                            .find(|(k, _)| *k == key)
                            .map(|(_, s)| *s)
                            .unwrap_or(0.0);
                        (key, score)
                    })
                    .collect();
                selected
            };

            // Descendant-dedup: with summaries in the candidate pool, a summary
            // node and one of the turns it covers can both survive selection
            // (both clear the score cut / belief band). Prefer the SPECIFIC —
            // drop any selected node that transitively covers another selected
            // node, leaving the finest antichain (never a summary stacked over
            // its own content). Applies to both the belief and rule-based paths;
            // the score-density path already emits a deduped antichain. Only pays
            // the coverage walk when a summary was actually picked; a pure-Normal
            // selection is untouched.
            if !score_density_used {
                let any_summary = selected
                    .iter()
                    .any(|(key, _)| resolver.turn_kind(*key).is_summary());
                if any_summary {
                    let picked: HashSet<TurnKey> = selected.iter().map(|(key, _)| *key).collect();
                    selected.retain(|(key, _)| {
                        // A summary only covers turns on its OWN timeline, so the
                        // covered indices re-key against that same timeline.
                        resolver
                            .node_covers(*key)
                            .into_iter()
                            .all(|covered| !picked.contains(&TurnKey::new(key.timeline, covered)))
                    });
                }
            }

            // Default fallback: if belief/scores/rule selected nothing, bring in
            // the group's declared default turn (by tag) so the group — and its
            // layer — never drops out of the projection at the empty-group
            // retain below. The sentinel score clears both the layer and group
            // score gates; it fires only when empty, so it never double-selects
            // or fights the belief challenger.
            if selected.is_empty() {
                if let Some(key) = resolve_default_turn(group.default.as_ref(), group.id, resolver)
                {
                    let sentinel = layer
                        .score_threshold
                        .max(group.score_threshold.unwrap_or(0.0));
                    selected.push((key, sentinel));
                    selection_origins.insert(key, SelectionOrigin::Fallback);
                }
            }

            // Concept D — timeline anchor: whenever ANY exchange of a timeline
            // is selected in an anchored group, the timeline's anchor member
            // (`first` = the file-header exchange carrying imports + module
            // doc) rides along at the timeline's best selected score — it
            // travels WITH the hit, never above it. Fires only on non-empty
            // per-timeline selections (an empty group is `default`'s job) and
            // never double-injects an organically-selected head.
            if let Some(anchor) = &group.anchor {
                let AnchorMember::First = anchor.member;
                let mut per_timeline: HashMap<TimelineId, f32> = HashMap::new();
                for (key, score) in &selected {
                    let best = per_timeline.entry(key.timeline).or_insert(f32::MIN);
                    *best = best.max(*score);
                }
                for (timeline, best) in per_timeline {
                    let head = TurnKey::new(timeline, TurnIndex(0));
                    if selected.iter().any(|(k, _)| *k == head) {
                        continue;
                    }
                    // A timeline with no turn 0 (never true for real
                    // conversations) contributes nothing.
                    if resolver.turn_token_count(head) == 0 {
                        continue;
                    }
                    selected.push((head, best));
                    // Pushed into `selected` above, so it qualified by
                    // construction — the anchor is admitted regardless of where
                    // `best` sits relative to the band.
                    selection_scores.set_turn(head, best, true);
                    selection_origins.insert(head, SelectionOrigin::Anchor);
                }
            }

            // Rule-based path has no per-pick origin; derive it from the
            // `Sequence` rule's own recency split so the panel can still answer
            // "why?": a turn inside the last `recent` window is `Recent`
            // (inviolate), an older selected turn is `Historical` (top-k by
            // score). Mirrors `select_conversation`'s `split_at = len - recent`.
            if !score_density_used {
                if let SelectionRule::Sequence { recent, .. } = &group.selection {
                    // `Sequence` is the live-conversation rule, so the group holds
                    // one timeline and candidate order is index order.
                    let split = (all_turns.len() as u32).saturating_sub(*recent as u32);
                    for (key, _) in &selected {
                        let origin = if key.index.0 >= split {
                            SelectionOrigin::Recent
                        } else {
                            SelectionOrigin::Historical
                        };
                        selection_origins.insert(*key, origin);
                    }
                }
            }

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
                score_density: score_density_used,
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

    // ── Step 8: System prompt — emit the shared prompt, framed by the target ──
    //
    // The schema's single `system_prompt.items` is an ordered list: each item is
    // a single section (always emits), a [`SectionCollection`] (named bucket with
    // its own selection rule), or a section-tree (pre-sealed selector branches).
    // It is shared by every projection target — the target layer contributes only
    // its `dials`, which seed the section-tree selection so this layer picks its
    // pre-sealed branch. The caller's per-turn `selection` overrides those dials;
    // any selector neither the caller nor the layer sets falls back to the tree's
    // authored default.
    //
    // This separation — static framing as plain sections vs. selectable catalogs
    // as collections vs. dial-selected branches — is what lets one shared system
    // prompt mix always-emit framing (role, grounding, dialect markers), dynamic
    // catalogs (tool definitions), and per-layer thinking/length/tool dials while
    // its sealed K/V is reused across all layers.
    let mut resolved_selections: Vec<ResolvedSelection> = Vec::new();
    // Seed the section-tree selection from the target layer's dials beneath the
    // caller's per-turn selection. Only clone when the layer actually overrides a
    // dial — the dialogue layer has none, and this runs per reprojection (per
    // decoded token), so an empty-dial merge would clone the selection for nothing.
    let target_dials = schema
        .layers
        .iter()
        .find(|l| l.id == target.layer)
        .map(|l| &l.dials)
        .filter(|d| !d.is_empty());
    let merged_selection;
    let effective_selection: &SelectionState = match target_dials {
        Some(dials) => {
            let mut eff = selection.clone();
            for (sel, opt) in dials.iter() {
                if eff.get(sel).is_none() {
                    eff.select(sel.to_string(), opt.to_string());
                }
            }
            merged_selection = eff;
            &merged_selection
        }
        None => selection,
    };
    let system_prompt_segments: Vec<ProjectionSegment> = emit_system_prompt_items(
        &schema.system_prompt,
        resolver,
        mode,
        effective_selection,
        prior,
        decode_pos,
        &mut selection_scores,
        &mut resolved_selections,
    );

    if group_states.is_empty() {
        return Projection {
            segments: system_prompt_segments,
            selections: resolved_selections,
            selection_scores,
            selection_origins,
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
                .map(|(key, _)| resolver.turn_token_count(*key))
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

    // Layer-level flex with natural cap so freed budget redistributes. Concept
    // B: a layer's attention mass (the sum over its groups') scales its
    // priority within the declared rails — a tour-shaped probe lifts repo_map
    // above its static share, a code probe grows the scopes layer.
    let layer_items: Vec<FlexItem> = surviving_layer_indices
        .iter()
        .enumerate()
        .map(|(slot, &li)| {
            let layer_mass: f32 = group_states
                .iter()
                .filter(|gs| gs.layer_idx == li)
                .map(|gs| resolver.group_attention_mass(gs.schema.id))
                .sum();
            let mut item = FlexItem::from_budget_with_mass(
                &visible_layers[li].budget,
                turn_budget,
                layer_mass,
            );
            let nat = layer_natural[slot];
            item.max_tokens = Some(item.max_tokens.map_or(nat, |m| m.min(nat)));
            item
        })
        .collect();
    let layer_budgets = flexbox_distribute(&layer_items, turn_budget);

    let mut final_selected: Vec<(GroupId, TurnKey)> = vec![];

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
        // Concept B applies one level down with the group's own mass.
        let group_items: Vec<FlexItem> = layer_groups
            .iter()
            .map(|gs| {
                let mut item = FlexItem::from_budget_with_mass(
                    &gs.schema.budget,
                    layer_budget,
                    resolver.group_attention_mass(gs.schema.id),
                );
                let nat = natural_tokens[&gs.schema.id];
                item.max_tokens = Some(item.max_tokens.map_or(nat, |m| m.min(nat)));
                item
            })
            .collect();
        let group_budgets = flexbox_distribute(&group_items, layer_budget);

        for (gi, gs) in layer_groups.iter().enumerate() {
            if gs.score_density {
                // The §8 selector already fit these picks into `layer.window`
                // (the flexbox natural-cap equals their total, so the group
                // gets exactly this budget back) and ordered them
                // chronologically with each summary ABOVE the turns it covers.
                // Re-running the rule-based `apply_selection` would re-sort by
                // raw `TurnIndex` — placing every summary AFTER its content —
                // and could silently drop picks via `historical_top_k`. Emit
                // the picks verbatim, in their existing order.
                for (key, _) in &gs.selected {
                    final_selected.push((gs.schema.id, *key));
                }
                continue;
            }
            let group_budget = group_budgets[gi];
            let tc = |key: TurnKey| resolver.turn_token_count(key);

            let mut selected_indices = if gs.schema.is_belief_driven() {
                // Belief already decided the surviving set (RelLeak + the rule's
                // budget); the bounded pass must ONLY trim to the token budget,
                // not re-apply the rule's `score_threshold` to the post-leak
                // belief scores — a selected turn whose leaked score fell below
                // the threshold would otherwise be silently dropped here,
                // diverging from the belief decision. Trim low-score-first, then
                // restore turn order for emission.
                let mut kept = gs.selected.clone();
                trim_to_budget_low_score_first(&mut kept, group_budget, &tc);
                kept.sort_by_key(|(idx, _)| *idx);
                kept.into_iter().map(|(idx, _)| idx).collect()
            } else {
                apply_selection(
                    &gs.schema.selection,
                    gs.schema.score_threshold.unwrap_or(0.0),
                    &gs.selected,
                    Some(group_budget),
                    &tc,
                )
            };

            // Budget-trim floor: the token trim must never drop a group to
            // nothing. If it did, re-inject the group's declared default turn so
            // the group stays present regardless of budget pressure.
            if selected_indices.is_empty() {
                if let Some(idx) =
                    resolve_default_turn(gs.schema.default.as_ref(), gs.schema.id, resolver)
                {
                    selected_indices.push(idx);
                }
            }

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
            let mut group_turns: Vec<TurnKey> = final_selected
                .iter()
                .filter_map(|(gid, key)| {
                    if *gid == gs.schema.id {
                        Some(*key)
                    } else {
                        None
                    }
                })
                .collect();
            // Raw turns carry a chronological `TurnIndex`, so sorting yields the
            // right reading order. On the rule-based path a summary node may also
            // survive (score cut + descendant-dedup), and its index is the storage
            // slot — higher than the turns it covers — so it sorts to the recent
            // end of this group's block. That is correct here: the dedup already
            // guarantees NONE of the turns it covers are also selected, so it
            // stands alone as the coarse cover of an older span (reference context
            // for a non-target group), not stacked on top of its own content. The
            // score-density path emits its own chronological order (summary above
            // the turns it refines); leave it untouched.
            if !gs.score_density {
                group_turns.sort();
            }

            for key in group_turns {
                // Ghost summary turns (`record_summary_turn` →
                // `append_with_blocks(0..0)`) are zero-token tree-meta anchors
                // with no K/V in any tier. They must never become a
                // `Sealed(Turn)` segment: there is nothing to inject, so
                // emitting one only wastes a window slot and makes the
                // assembler wrap empty boundary markers around no content
                // (which then drops at inject time). Skip them at the
                // projection source so every consumer — the elevate set and the
                // assembler alike — sees only turns that carry injectable K/V.
                if resolver.turn_token_count(key) == 0 {
                    continue;
                }
                // Prefer the producer's `layer_id` from the resolver
                // record; fall back to projector context if the
                // resolver doesn't track origins (e.g. test mocks).
                let origin_layer = resolver.turn_origin(key).unwrap_or(layer_id_for_walk);
                turns.push(ResolvedTurn {
                    id: TurnId {
                        layer_id: origin_layer,
                        group_id: gs.schema.id,
                        index: key.index,
                    },
                    // Stamp the conversation ONCE, here, where the target-aware
                    // resolver knows it — so no downstream consumer re-derives it.
                    timeline: Some(key.timeline),
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
        selection_origins,
    }
}

/// Walk the shared system prompt's `items` in declaration order, emitting each
/// plain section verbatim or each collection's surviving subset (after applying
/// its selection rule), with section-tree branches resolved from `selection_state`.
///
/// A [`SectionSchema`] with `depends_on = Some(cid)` only emits if the
/// named collection materialised ≥ 1 section in this same emission pass.
#[allow(clippy::too_many_arguments)]
fn emit_system_prompt_items<R: ContentResolver>(
    sp: &SystemPromptSchema,
    resolver: &R,
    mode: ProjectionMode,
    selection_state: &SelectionState,
    prior: &PriorBelief,
    decode_pos: Option<usize>,
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
    // Catalog summaries to emit just before each collection's opening structural
    // marker (OUTSIDE it). A summary emits only when the selection is a proper
    // non-empty subset — top-k/threshold dropped at least one member, so the
    // summary names the full set — and the summary section is actually sealed.
    let mut pending_summaries: std::collections::HashMap<CollectionId, SectionId> =
        std::collections::HashMap::new();
    let mut record = |coll: &SectionCollection, selected: Vec<ProjectionSegment>| {
        if let Some(sum_id) = coll.summary_section {
            let partial = !selected.is_empty() && selected.len() < coll.sections.len();
            if partial && resolver.section_token_count(sum_id) > 0 {
                pending_summaries.insert(coll.id, sum_id);
            }
        }
        collection_results.insert(coll.id, selected);
    };
    for item in &sp.items {
        match item {
            SystemPromptItem::Collection(coll) => {
                // Top-level collection (e.g. `tools`): belief-driven selection.
                let selected = select_collection_sections(
                    coll,
                    resolver,
                    &scoring,
                    selection_state,
                    prior,
                    decode_pos,
                    scores,
                );
                record(coll, selected);
            }
            // Collections embedded as section-tree nodes resolve here too, so
            // their materialised set feeds `depends_on` gating like any other.
            // Tree-node collections use the provenance-scored top-k path (their
            // members are structural reasoning sections, not belief-gathered tools).
            SystemPromptItem::SectionTree(tree) => {
                let selection = tree.selection(|id| selection_state.get(id));
                for node in &tree.nodes {
                    if let Some(tc) = &node.collection {
                        let active_key = tree.pack(&selection, node.ancestor_dims);
                        let selected =
                            select_tree_collection_segments(tc, active_key, resolver, &scoring);
                        record(&tc.collection, selected);
                    }
                }
            }
            _ => {}
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
    for item in &sp.items {
        match item {
            SystemPromptItem::Section(s) => {
                let should_emit = match (s.depends_on, s.depends_on_absent) {
                    // `depends_on`: emit only when the collection materialised ≥1.
                    (Some(cid), _) => non_empty_collections.contains(&cid),
                    // `depends_on_absent`: the inverse — emit only when it
                    // materialised zero (the no-tools variant).
                    (None, Some(cid)) => !non_empty_collections.contains(&cid),
                    (None, None) => true,
                };
                if should_emit {
                    // Emit the collection's catalog summary just before its opening
                    // structural template (e.g. `<tools>`), so it sits OUTSIDE the
                    // markers. Triggered by the first emitting *template* gated on
                    // the collection (the open marker; a plain prose section like
                    // `tools_overview` does not trigger it); drained so the closing
                    // marker doesn't re-emit it.
                    if s.is_template {
                        if let Some(cid) = s.depends_on {
                            if let Some(sum_id) = pending_summaries.remove(&cid) {
                                out.push(ProjectionSegment::Sealed(SealedKind::Section(
                                    ResolvedSection { id: sum_id },
                                )));
                            }
                        }
                    }
                    push_section_segment(&mut out, s);
                }
            }
            SystemPromptItem::Collection(coll) => {
                if let Some(selected) = collection_results.remove(&coll.id) {
                    // Fallback for a collection with no opening structural template
                    // to hang the summary on: emit it just before the members, as
                    // before. (When the collection IS wrapped — e.g. `<tools>` —
                    // the open template already drained and emitted it outside.)
                    if let Some(sum_id) = pending_summaries.remove(&coll.id) {
                        out.push(ProjectionSegment::Sealed(SealedKind::Section(
                            ResolvedSection { id: sum_id },
                        )));
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
                    // A live-prefilled structural marker (`<tools>` etc.): emit a
                    // Generated run when the active branch is one it lives in (e.g.
                    // tools on).  Prefix-transparent — never a Sealed segment.
                    if let Some(g) = &node.glue {
                        let key = tree.pack(&selection, node.ancestor_dims);
                        if g.active_keys.contains(&key) {
                            if let Some(tokens) = &g.tokens {
                                out.push(ProjectionSegment::Generated {
                                    tokens: tokens.clone(),
                                    identity: GeneratedIdentity {
                                        name: node.name.clone(),
                                        position: out.len(),
                                    },
                                });
                            }
                        }
                        continue;
                    }
                    // A prefix-transparent embedded collection emits its cached
                    // active-branch selection — the same drain path as a
                    // top-level collection (summary outside its markers).  When
                    // `deferred_projection` is set its selection is emitted by a
                    // placeholder node below (`inject_collection`), so it skips
                    // its own emission and leaves the cached results in place.
                    if let Some(tc) = &node.collection {
                        if tc.deferred_projection {
                            continue;
                        }
                        if let Some(selected) = collection_results.remove(&tc.collection.id) {
                            if let Some(sum_id) = pending_summaries.remove(&tc.collection.id) {
                                out.push(ProjectionSegment::Sealed(SealedKind::Section(
                                    ResolvedSection { id: sum_id },
                                )));
                            }
                            out.extend(selected);
                        }
                        continue;
                    }
                    // A placeholder node: it sealed its own anchor content (so the
                    // nodes below attend to a stable prefix), but at projection its
                    // content is REPLACED by the injected collection's top-k — the
                    // provenance-selected real members, glued by their own trailing
                    // newlines.  The anchor's K/V is intentionally not emitted here.
                    // (The catalog summary is its own sealed tree section above the
                    // tool block, not drained here.)
                    if let Some(cid) = node.inject_collection {
                        if let Some(selected) = collection_results.remove(&cid) {
                            out.extend(selected);
                        }
                        continue;
                    }
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

/// Emit a single [`SectionSchema`] as a [`ProjectionSegment::Sealed`] — its
/// K/V comes from the substrate.
///
/// **Template sections seal like any other.** They used to emit a
/// [`ProjectionSegment::Generated`] run, live-prefilled at apply time under the
/// runtime left context. They are dialect structural text — a handful of tokens
/// whose content and position are both fixed — and what varied was only their
/// left context, because a `depends_on` template emits just when its collection
/// materialises.
///
/// A live-prefilled run is a glue **island**: a hole in the middle of the
/// sequence the engine has to gap-fill. That is impossible for a model whose
/// per-sequence memory is a recurrence, so the choice is between an approximate
/// bake — the same approximation a collection member's K/V already carries — and
/// a wave that refuses to run. See `docs/deltanet_state_persistence.md` §4.7d.
///
/// The distinction survives in the schema (`is_template` still selects the
/// dialect text at build time) but no longer changes what a projection emits.
fn push_section_segment(out: &mut Vec<ProjectionSegment>, s: &SectionSchema) {
    out.push(ProjectionSegment::Sealed(SealedKind::Section(
        ResolvedSection { id: s.id },
    )));
}

/// Apply a collection's selection rule, returning the surviving
/// sections in **declaration order**.
///
/// Selection picks by salience (score, then priority); emission
/// preserves authored structure (declaration order).  Sections below
/// `score_threshold` are filtered out before selection.
/// The selected member declaration indices (returned in declaration order) for
/// `coll` under `scoring`.  Shared by top-level collections and collections
/// embedded as section-tree nodes; scoring is always by the member's canonical
/// [`SectionSchema::id`], so a tree-node collection's selection is stable across
/// the outer-branch (no_think) — only the *sealed K/V* it emits differs.
fn select_collection_indices<R: ContentResolver>(
    coll: &SectionCollection,
    resolver: &R,
    scoring: &CollectionScoring,
) -> Vec<usize> {
    if coll.sections.is_empty() {
        return Vec::new();
    }
    use std::cmp::Ordering::Equal;
    // Provenance-scored top-k — this branch's scoring architecture. Structural
    // tree-node sections rank by the same per-section provenance score the belief
    // path uses; the tree-node algorithm just top-k sorts rather than accumulating.
    let score_of = |s: &SectionSchema| resolver.section_score(s.id);
    match &coll.selection {
        // No sensible "recent" semantics for sections → all, declaration order.
        SelectionRule::AlwaysVisible | SelectionRule::Sequence { .. } => {
            (0..coll.sections.len()).collect()
        }
        SelectionRule::TopK { k } => {
            let mut scored: Vec<(usize, f32, f32)> = coll
                .sections
                .iter()
                .enumerate()
                .map(|(decl, s)| (decl, score_of(s), s.priority))
                .filter(|(_, score, _)| !scoring.apply_threshold || *score >= coll.score_threshold)
                .collect();
            if tracing::enabled!(tracing::Level::TRACE) {
                let mut by_score = scored.clone();
                by_score.sort_by(|(_, a, _), (_, b, _)| b.partial_cmp(a).unwrap_or(Equal));
                let scores_str = by_score
                    .iter()
                    .map(|(i, sc, _)| format!("{}={:.1}", coll.sections[*i].name, sc))
                    .collect::<Vec<_>>()
                    .join(", ");
                tracing::trace!(collection = %coll.name, threshold = coll.score_threshold, scores = %scores_str, "projection scores");
            }
            // Score desc, then priority desc, then declaration order.
            scored.sort_by(|(ai, asc, ap), (bi, bsc, bp)| {
                bsc.partial_cmp(asc)
                    .unwrap_or(Equal)
                    .then(bp.partial_cmp(ap).unwrap_or(Equal))
                    .then(ai.cmp(bi))
            });
            scored.truncate(*k);
            let mut idx: Vec<usize> = scored.into_iter().map(|(i, _, _)| i).collect();
            idx.sort_unstable(); // re-emit in declaration order
            idx
        }
        SelectionRule::Single => coll
            .sections
            .iter()
            .enumerate()
            .map(|(decl, s)| (decl, score_of(s), s.priority))
            .filter(|(_, score, _)| !scoring.apply_threshold || *score >= coll.score_threshold)
            .max_by(|(_, asc, ap), (_, bsc, bp)| {
                asc.partial_cmp(bsc)
                    .unwrap_or(Equal)
                    .then(ap.partial_cmp(bp).unwrap_or(Equal))
            })
            .map(|(i, _, _)| vec![i])
            .unwrap_or_default(),
        // `Named` is a score-independent by-name pick that needs the projection's
        // `SelectionState` to resolve the selector. It is only used on the
        // top-level `tools` collection (handled by the belief path
        // `select_collection_sections`); the score-based tree-node path never sees
        // it, so select nothing.
        SelectionRule::Named { .. } => Vec::new(),
    }
}

/// Emit the collection's `member_glue` (a live-prefilled structural token, e.g. a
/// newline) into `out` BEFORE the next member — but never before the first one.
/// The glue is independent of which members provenance selected: it is not baked
/// into any member's seal, so a dropped member never takes its separator with it.
fn push_member_glue(out: &mut Vec<ProjectionSegment>, coll: &SectionCollection) {
    if out.is_empty() {
        return; // no glue leads the first member
    }
    let Some(tokens) = &coll.member_glue_tokens else {
        return;
    };
    let position = out.len();
    out.push(ProjectionSegment::Generated {
        tokens: tokens.clone(),
        identity: GeneratedIdentity {
            name: format!("{}__member_glue", coll.name),
            position,
        },
    });
}

/// Resolve a top-level collection (e.g. `tools`) to its emitted segments —
/// **belief-driven** selection (RelLeak + hysteresis + budget) over each
/// section's provenance score. This is the perfected tool-selection mechanism;
/// tree-node collections use the provenance-scored top-k path in
/// [`select_collection_indices`] instead.
#[allow(clippy::too_many_arguments)]
fn select_collection_sections<R: ContentResolver>(
    coll: &SectionCollection,
    resolver: &R,
    scoring: &CollectionScoring,
    selection_state: &SelectionState,
    prior: &PriorBelief,
    decode_pos: Option<usize>,
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
            // Forced-member pin: when the runtime sets `FORCE_TOOL_SELECTOR`, emit
            // exactly the named members and skip belief selection, so a prefilled
            // tool_call is always backed by a present tool definition. Same by-name
            // pick as `Named`, score-independent. Unset ⇒ belief path.
            //
            // The value is a comma-separated list because an ingest turn can
            // prefill calls to more than one tool: the repo_map folder round-trip
            // uses `file_list` then `file_read`, and pinning only one would leave
            // the other call naming a tool absent from the `<tools>` block — the
            // incoherence this pin exists to prevent. Emitted in the order named
            // (`FORCE_TOOL_SEPARATOR`); unknown names are skipped.
            if let Some(target) = selection_state.get(FORCE_TOOL_SELECTOR) {
                let mut out = Vec::new();
                for name in target.split(FORCE_TOOL_SEPARATOR).map(str::trim) {
                    if let Some(s) = coll.sections.iter().find(|s| s.name == name) {
                        push_section_segment(&mut out, s);
                        scores.set_section(s.id, coll.score_threshold.max(1.0), true);
                    }
                }
                if !out.is_empty() {
                    return out;
                }
            }
            // Belief-driven selection: the collection's policy (RelLeak budget +
            // hysteresis) decides the surviving set from the per-section scores,
            // seeded from the prior projection's belief so decay/reinforcement
            // carries across a turn. Each member's belief is recorded on `scores`.
            let fresh: Vec<f32> = coll
                .sections
                .iter()
                .map(|s| resolver.section_score(s.id))
                .collect();
            let (prior_scores, prior_selected, prior_qualified) =
                prior.collection(&coll.name, &coll.sections);
            // Early-decode grace: within the opening window the selection band is
            // lowered and carried picks are floored (see `PolicyConfig::windowed`),
            // so the submit guess and a still-accruing correct tool stay in scope.
            let (mut cfg, floor) = coll.policy.config.windowed(decode_pos);
            // Concept B: the collection's attention mass extends its member
            // budget within the declared rail.
            if let Some(ba) = &coll.budget_adaptive {
                cfg.budget_max = ba.effective_max(
                    cfg.budget_max,
                    resolver.collection_attention_mass(coll.id.raw()),
                );
            }
            let beliefs = crate::provenance::belief_step(
                &fresh,
                &prior_scores,
                &prior_selected,
                &prior_qualified,
                cfg.section_policy(0),
                cfg.budget(),
                floor,
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
                scores.set_section(s.id, b.score, b.qualified);
                if b.selected {
                    push_section_segment(&mut out, s);
                }
            }
            // Default fallback: if the belief loop selected no member, emit the
            // collection's declared default section (by name) so the collection
            // always contributes at least one section. Fires only when empty.
            if out.is_empty() {
                if let Some(def) = &coll.default {
                    if let Some(s) = coll.sections.iter().find(|s| s.name == def.tag) {
                        push_section_segment(&mut out, s);
                    }
                }
            }
            out
        }
        SelectionRule::Single => {
            let best = coll
                .sections
                .iter()
                .map(|s| (s, resolver.section_score(s.id)))
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
            // Explicit by-name pick: emit exactly the member whose `name` matches
            // the runtime selector value. Score-independent.
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
            let mut out = Vec::with_capacity(coll.sections.len());
            for s in &coll.sections {
                push_section_segment(&mut out, s);
            }
            out
        }
    }
}

/// Resolve a section-tree collection node to its emitted segments — the same
/// top-k selection (over canonical ids), but emitting each selected member's
/// ACTIVE-branch sealed variant for `active_key`.
fn select_tree_collection_segments<R: ContentResolver>(
    tc: &TreeCollection,
    active_key: u32,
    resolver: &R,
    scoring: &CollectionScoring,
) -> Vec<ProjectionSegment> {
    let selected = select_collection_indices(&tc.collection, resolver, scoring);
    let mut out = Vec::with_capacity(selected.len());
    for i in selected {
        if let Some(v) = tc.member_variant(i, active_key) {
            push_member_glue(&mut out, &tc.collection);
            out.push(ProjectionSegment::Sealed(SealedKind::Section(
                ResolvedSection { id: v.id },
            )));
        }
    }
    out
}
