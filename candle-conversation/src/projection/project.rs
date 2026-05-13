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
//!  │  STEP 5.  Group score = layer.score_formula(natural turn scores)   │
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

use super::ids::{GroupId, LayerId, SectionId, TurnId, TurnIndex};
use super::reconcile::{flexbox_distribute, FlexItem};
use crate::substrate::ContentResolver;
use super::schema::{
    GroupSchema, LayerSchema, Schema, SectionCollection, SectionSchema, SelectionRule,
    SystemPromptItem,
};
use super::selection::apply_selection;

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

/// Result of a [`super::Builder::project`] call.
///
/// `system_prompt` is in declaration order. `turns` is in (layer order ×
/// ascending-group-score × turn insertion order) — see step 12 of the
/// pipeline above.
#[derive(Debug, Clone)]
pub struct Projection {
    pub system_prompt: Vec<ResolvedSection>,
    pub turns: Vec<ResolvedTurn>,
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
    pub timeline: super::ids::TimelineId,
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
) -> Projection {
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
            if li < target_layer_idx {
                Some(layer) // all groups in lower layers are visible
            } else if li == target_layer_idx {
                Some(layer) // groups filtered per-group below
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
        let formula = layer.score_formula;
        let weights = layer.depth_weights;
        for group in &layer.groups {
            // Masking: for the target layer, only the target group is visible.
            if layer_is_target && group.id != target.group {
                continue;
            }

            let count = resolver.turn_count(group.id);
            let all_turns: Vec<(TurnIndex, f32)> = (0..count)
                .map(|i| {
                    let idx = TurnIndex(i);
                    let score = resolver.turn_score(group.id, idx, formula, &weights);
                    (idx, score)
                })
                .collect();

            let tc = |idx: TurnIndex| resolver.turn_token_count(group.id, idx);

            // Unbounded selection (no budget constraint yet).
            let selected_indices =
                apply_selection(&group.selection, group.score_threshold, &all_turns, None, &tc);

            let selected: Vec<(TurnIndex, f32)> = selected_indices
                .iter()
                .map(|&idx| {
                    let score = all_turns
                        .iter()
                        .find(|(i, _)| *i == idx)
                        .map(|(_, s)| *s)
                        .unwrap_or(0.0);
                    (idx, score)
                })
                .collect();

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
        let layer = &visible_layers[gs.layer_idx];
        gs.group_score = layer.score_formula.aggregate(&scores);
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
    let system_prompt_sections: Vec<ResolvedSection> = schema
        .layers
        .iter()
        .find(|l| l.id == target.layer)
        .map(|l| emit_system_prompt_items(l, resolver))
        .unwrap_or_default();

    if group_states.is_empty() {
        return Projection {
            system_prompt: system_prompt_sections,
            turns: vec![],
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
                .filter_map(|(gid, idx)| if *gid == gs.schema.id { Some(*idx) } else { None })
                .collect();
            group_turns.sort();

            for idx in group_turns {
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

    Projection {
        system_prompt: system_prompt_sections,
        turns,
    }
}

/// Walk a layer's `system_prompt.items` in declaration order, emitting
/// either each plain section verbatim or each collection's surviving
/// subset (after applying its selection rule).  Result is the
/// `Vec<ResolvedSection>` that goes into the final [`Projection`].
fn emit_system_prompt_items<R: ContentResolver>(
    layer: &LayerSchema,
    resolver: &R,
) -> Vec<ResolvedSection> {
    let mut out: Vec<ResolvedSection> = Vec::new();
    for item in &layer.system_prompt.items {
        match item {
            SystemPromptItem::Section(s) => {
                out.push(ResolvedSection { id: s.id });
            }
            SystemPromptItem::Collection(coll) => {
                out.extend(select_collection_sections(coll, layer, resolver));
            }
        }
    }
    out
}

/// Apply a collection's selection rule, returning the surviving
/// sections in **declaration order**.
///
/// Selection picks by salience (score, then priority); emission
/// preserves authored structure (declaration order).  Sections below
/// `score_threshold` are filtered out before selection.
fn select_collection_sections<R: ContentResolver>(
    coll: &SectionCollection,
    layer: &LayerSchema,
    resolver: &R,
) -> Vec<ResolvedSection> {
    if coll.sections.is_empty() {
        return Vec::new();
    }
    match &coll.selection {
        SelectionRule::AlwaysVisible => coll
            .sections
            .iter()
            .map(|s| ResolvedSection { id: s.id })
            .collect(),
        SelectionRule::TopK { k } => {
            let mut scored: Vec<(usize, &SectionSchema, f32)> = coll
                .sections
                .iter()
                .enumerate()
                .map(|(decl, s)| {
                    let score = resolver.section_score(
                        s.id,
                        coll.score_formula,
                        &layer.depth_weights,
                    );
                    (decl, s, score)
                })
                .filter(|(_, _, score)| *score >= coll.score_threshold)
                .collect();
            scored.sort_by(|(ai, a, asc), (bi, b, bsc)| {
                bsc.partial_cmp(asc)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal))
                    .then(ai.cmp(bi))
            });
            scored.truncate(*k);
            scored.sort_by_key(|(i, _, _)| *i);
            scored
                .into_iter()
                .map(|(_, s, _)| ResolvedSection { id: s.id })
                .collect()
        }
        SelectionRule::Single => coll
            .sections
            .iter()
            .map(|s| {
                let score = resolver.section_score(
                    s.id,
                    coll.score_formula,
                    &layer.depth_weights,
                );
                (s, score)
            })
            .filter(|(_, score)| *score >= coll.score_threshold)
            .max_by(|(a, asc), (b, bsc)| {
                asc.partial_cmp(bsc)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.priority.partial_cmp(&b.priority).unwrap_or(std::cmp::Ordering::Equal))
            })
            .map(|(s, _)| vec![ResolvedSection { id: s.id }])
            .unwrap_or_default(),
        SelectionRule::Sequence { .. } => {
            // No sensible "recent" semantics for sections.  Fall back
            // to AlwaysVisible.
            coll.sections
                .iter()
                .map(|s| ResolvedSection { id: s.id })
                .collect()
        }
    }
}

