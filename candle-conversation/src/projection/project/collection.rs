//! Chooses which members of a section collection a projection shows.
//!
//! Belief-driven selection applies the collection's budget and hysteresis to each
//! member's provenance score. The explicit rules, top-k by score, by name, or the
//! declared default, cover collections whose members are fixed rather than scored.
//! Mandatory members are emitted on every projection and sit outside the budget.
//! Emission keeps the authored catalog order, and a collection with no provenance
//! opinion shows everything it was offered.

use std::collections::HashSet;

use super::range_fill::fit_by_tokens;
use super::*;

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
pub(super) fn select_collection_indices<R: ContentResolver>(
    coll: &SectionCollection,
    resolver: &R,
    scoring: &CollectionScoring,
    selection_state: &SelectionState,
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
        SelectionRule::TopK { k } => top_scored_indices(coll, resolver, scoring, *k, None),
        SelectionRule::Offered { selector, k } => top_scored_indices(
            coll,
            resolver,
            scoring,
            *k,
            Some(selection_state.members(selector)),
        ),
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
        // Score-independent: whatever the runtime selector names.
        SelectionRule::Named { selector } => named_indices(coll, selector, selection_state),
        // A collection carries no conversations (the YAML loader refuses it).
        SelectionRule::WorkingSet { .. } => Vec::new(),
    }
}

/// The `k` highest-scored members of `coll`, in declaration order. `offered`
/// restricts the candidates to the members it names (`None` ranks them all).
pub(super) fn top_scored_indices<R: ContentResolver>(
    coll: &SectionCollection,
    resolver: &R,
    scoring: &CollectionScoring,
    k: usize,
    offered: Option<&[String]>,
) -> Vec<usize> {
    use std::cmp::Ordering::Equal;
    let mut scored: Vec<(usize, f32, f32)> = coll
        .sections
        .iter()
        .enumerate()
        .filter(|(_, s)| offered.is_none_or(|named| named.contains(&s.name)))
        .map(|(decl, s)| (decl, resolver.section_score(s.id), s.priority))
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
    scored.truncate(k);
    let mut idx: Vec<usize> = scored.into_iter().map(|(i, _, _)| i).collect();
    idx.sort_unstable(); // re-emit in declaration order
    idx
}

/// The declaration indices of the members a [`SelectionRule::Named`] selector
/// names, in declaration order — the order the collection was authored in, not
/// the order the caller happened to list them. A name that is not a member is
/// skipped, and a name listed twice emits once.
pub(super) fn named_indices(
    coll: &SectionCollection,
    selector: &str,
    selection_state: &SelectionState,
) -> Vec<usize> {
    let named = selection_state.members(selector);
    coll.sections
        .iter()
        .enumerate()
        .filter(|(_, s)| named.contains(&s.name))
        .map(|(i, _)| i)
        .collect()
}

/// Resolve a top-level collection (e.g. `tools`) to its emitted segments —
/// **belief-driven** selection (RelLeak + hysteresis + budget) over each
/// section's provenance score. This is the perfected tool-selection mechanism;
/// tree-node collections use the provenance-scored top-k path in
/// [`select_collection_indices`] instead.
#[allow(clippy::too_many_arguments)]
pub(super) fn select_collection_sections<R: ContentResolver>(
    coll: &SectionCollection,
    resolver: &R,
    scoring: &CollectionScoring,
    selection_state: &SelectionState,
    prior: &PriorBelief,
    decode_pos: Option<usize>,
    scores: &mut super::super::SelectionScores,
) -> Vec<ProjectionSegment> {
    if coll.sections.is_empty() {
        return Vec::new();
    }
    if let Some(members) = &selection_state.scripted_members {
        let mut out = Vec::new();
        for s in coll.sections.iter().filter(|s| members.contains(&s.name)) {
            push_member_glue(&mut out, coll);
            push_section_segment(&mut out, s);
        }
        return out;
    }
    match &coll.selection {
        SelectionRule::AlwaysVisible => {
            let mut out = Vec::with_capacity(coll.sections.len());
            for s in &coll.sections {
                push_member_glue(&mut out, coll);
                push_section_segment(&mut out, s);
            }
            out
        }
        SelectionRule::TopK { .. } | SelectionRule::Offered { .. } => {
            // `Offered` ranks only the members the runtime selector names: the
            // rest are neither candidates nor emitted, whatever they score.
            let offered: Option<&[String]> = match &coll.selection {
                SelectionRule::Offered { selector, .. } => Some(selection_state.members(selector)),
                _ => None,
            };
            // A member the round's grammar holds out is never shown either.
            let is_offered = |s: &SectionSchema| {
                offered.is_none_or(|named| named.contains(&s.name))
                    && !selection_state.is_withheld(&s.name)
            };
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
                        push_member_glue(&mut out, coll);
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
            // Mandatory members are emitted on every projection and sit OUTSIDE
            // the belief budget: the policy picks its top-k from the other
            // members only, so a mandatory `file_read` adds to the selection
            // rather than taking one of its slots.
            let candidates: Vec<&SectionSchema> = coll
                .sections
                .iter()
                .filter(|s| !coll.mandatory.contains(&s.id) && is_offered(s))
                .collect();
            let fresh: Vec<f32> = candidates
                .iter()
                .map(|s| resolver.section_score(s.id))
                .collect();
            let (prior_scores, prior_selected, prior_qualified) =
                prior.collection(&coll.name, &candidates);
            // Early-decode grace: within the opening window the selection band is
            // lowered and carried picks are floored (see `PolicyConfig::windowed`),
            // so the submit guess and a still-accruing correct tool stay in scope.
            let (mut cfg, floor) = coll.policy.config.windowed(decode_pos);
            if let SelectionRule::Offered { k, .. } = &coll.selection {
                cfg.budget_max = *k;
            }
            // Concept B: the collection's attention mass extends its member
            // budget within the declared rail.
            if let Some(ba) = &coll.budget_adaptive {
                cfg.budget_max = ba.effective_max(
                    cfg.budget_max,
                    resolver.collection_attention_mass(coll.id.raw()),
                );
            }
            // A ranged collection is filled by tokens, not by a count: the
            // range is its size, so the selection may choose every member and
            // the range takes them by score while they fit (`range_fill`).
            if coll.range.is_some() {
                cfg.budget_max = cfg.budget_max.max(candidates.len());
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
                let scores_str = candidates
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
            // Emit in catalog order: a mandatory member always, the rest as the
            // belief selected them. A mandatory member's score is recorded for
            // the event view, qualified by the same bar as everyone else.
            let min_score = cfg.section_policy(0).min_score;
            // An offered collection whose provenance has no opinion (every
            // offered member scores nothing: cold scores, or content that
            // matches none of them) shows everything offered rather than the
            // first `k` by declaration order — the room already narrowed it to
            // acts the body can take.
            let no_opinion = offered.is_some() && fresh.iter().all(|&f| f <= 0.0);
            // What a ranged collection's range holds of what was selected:
            // the mandatory members' tokens first, then by score while each
            // fits (`docs/placed_section_layout.md` §3.5).
            let fits: Option<HashSet<SectionId>> = coll.range.map(|range| {
                let room = range.positions(resolver.trained_window());
                let mandatory: usize = coll
                    .sections
                    .iter()
                    .filter(|s| coll.mandatory.contains(&s.id) && is_offered(s))
                    .map(|s| resolver.section_token_count(s.id))
                    .sum();
                let chosen: Vec<(SectionId, f32, usize)> = candidates
                    .iter()
                    .zip(&beliefs)
                    .filter(|(_, b)| b.selected || no_opinion)
                    .map(|(s, b)| (s.id, b.score, resolver.section_token_count(s.id)))
                    .collect();
                fit_by_tokens(&chosen, room, mandatory)
            });
            let mut beliefs = beliefs.into_iter();
            let mut out = Vec::new();
            for s in coll.sections.iter().filter(|s| is_offered(s)) {
                let emit = if coll.mandatory.contains(&s.id) {
                    let score = resolver.section_score(s.id);
                    scores.set_section(s.id, score, score >= min_score);
                    true
                } else {
                    let b = beliefs
                        .next()
                        .expect("one belief per non-mandatory member, in catalog order");
                    scores.set_section(s.id, b.score, b.qualified);
                    (b.selected || no_opinion) && fits.as_ref().is_none_or(|f| f.contains(&s.id))
                };
                if emit {
                    push_member_glue(&mut out, coll);
                    push_section_segment(&mut out, s);
                }
            }
            // Default fallback: if the belief loop selected no member, emit the
            // collection's declared default section (by name) so the collection
            // always contributes at least one section. Fires only when empty, so
            // there is nothing for glue to separate it from.
            if out.is_empty() {
                if let Some(def) = &coll.default {
                    if let Some(s) = coll
                        .sections
                        .iter()
                        .find(|s| s.name == def.tag && is_offered(s))
                    {
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
                push_member_glue(&mut out, coll);
                push_section_segment(&mut out, s);
            }
            out
        }
        SelectionRule::Named { selector } => {
            // Explicit by-name pick: every member the runtime selector names, in
            // the collection's declaration order. Score-independent.
            let mut out = Vec::new();
            for i in named_indices(coll, selector, selection_state) {
                push_member_glue(&mut out, coll);
                push_section_segment(&mut out, &coll.sections[i]);
            }
            tracing::trace!(
                collection = %coll.name,
                selector = %selector,
                named = selection_state.members(selector).len(),
                selected = out.len(),
                "projection (named)"
            );
            out
        }
        SelectionRule::Sequence { .. } => {
            let mut out = Vec::with_capacity(coll.sections.len());
            for s in &coll.sections {
                push_member_glue(&mut out, coll);
                push_section_segment(&mut out, s);
            }
            out
        }
        // A working set carries whole conversations into turn groups; a
        // collection has none to carry (the YAML loader refuses the pairing).
        SelectionRule::WorkingSet { .. } => Vec::new(),
    }
}

/// Resolve a section-tree collection node to its emitted segments — the same
/// top-k selection (over canonical ids), but emitting each selected member's
/// ACTIVE-branch sealed variant for `active_key`.
pub(super) fn select_tree_collection_segments<R: ContentResolver>(
    tc: &TreeCollection,
    active_key: u32,
    resolver: &R,
    scoring: &CollectionScoring,
    selection_state: &SelectionState,
) -> Vec<ProjectionSegment> {
    let selected = match &selection_state.scripted_members {
        Some(members) => (0..tc.collection.sections.len())
            .filter(|&i| members.contains(&tc.collection.sections[i].name))
            .collect(),
        None => select_collection_indices(&tc.collection, resolver, scoring, selection_state),
    };
    let mut out = Vec::with_capacity(selected.len());
    for i in selected {
        if let Some(v) = tc.member_variant(i, active_key) {
            push_member_glue(&mut out, &tc.collection);
            out.push(ProjectionSegment::Sealed(SealedKind::Section(
                ResolvedSection::new(v.id),
            )));
        }
    }
    out
}
