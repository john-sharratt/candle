//! Runs the projection pipeline and reports how score-density selection chose what
//! it emitted.
//!
//! The scheduler uses this variant to ferry the selection diagnostic into the
//! substrate's side channel without adding a test-only field to the projection.
//! The sink is called once, and only when score-density selection ran rather than
//! the rule-based path. The pipeline masks the visible layers, scores and gates
//! each group's turns, selects an unbounded set, and trims it to the budget in one
//! flexbox pass. A group with nothing that qualifies contributes nothing, and it is
//! not refilled with a default that would defeat the threshold.

use super::*;

/// The system-prompt section the knowledge base's files are spliced after.
const FILES_OPEN_SECTION: &str = "files_open";

/// The fill kind a working-set group's members count toward.
fn share_kind(share: Option<WorkingSetShare>) -> Kind {
    match share {
        Some(WorkingSetShare::Folders) => Kind::Folder,
        Some(WorkingSetShare::Guides) => Kind::Guide,
        Some(WorkingSetShare::Journals) => Kind::Journal,
        Some(WorkingSetShare::Readings) | None => Kind::Reading,
    }
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
    let mut selection_scores = super::super::super::SelectionScores::default();
    // ── Step 1: Mask ─────────────────────────────────────────────────────────
    // The stack is by rank, not by declaration: a layer appended to a live
    // schema keeps every existing id where it was and still sits wherever its
    // rank puts it. See `LayerSchema::rank`.
    let target_rank = schema
        .layers
        .iter()
        .find(|l| l.id == target.layer)
        .map(|l| l.rank)
        .unwrap_or(i32::MIN);

    let mut visible_layers: Vec<&LayerSchema> = schema
        .layers
        .iter()
        // All groups in lower-ranked layers are visible; on the target layer,
        // groups are filtered individually further below. The diagnostic kill
        // switch is applied INSIDE the loop, not here, so the target is always
        // in this list.
        .filter(|layer| layer.id == target.layer || layer.rank < target_rank)
        .collect();
    // Bottom of the stack first, so the layers are walked — and the target
    // reached — in the order the stack describes, whatever order they were
    // declared in. Stable, so equal ranks keep declaration order.
    visible_layers.sort_by_key(|layer| layer.rank);

    // ── Step 2–4: Score, threshold-gate, unbounded selection ─────────────────
    struct GroupState<'a> {
        schema: &'a GroupSchema,
        layer_idx: usize,
        selected: Vec<(TurnKey, f32)>, // ((timeline, index), score), insertion order
        /// Exchange membership for every candidate: representative → the run's
        /// turns in ascending order. `selected` holds representatives, so this is
        /// what prices a candidate for the budget and what expands it at emit —
        /// a tool round-trip is selected, trimmed and emitted whole.
        members: HashMap<TurnKey, Vec<TurnKey>>,
        group_score: f32,
        /// The §8 score-density selector produced `selected` (already
        /// budget-fit and in chronological order — summaries ABOVE the turns
        /// they cover). When set, downstream steps must NOT re-run the
        /// rule-based selection or re-sort by raw `TurnIndex` (that would place
        /// every summary AFTER its own content); the order is emitted verbatim.
        score_density: bool,
        /// A `working_set` group of a target that carries one: filled from the
        /// target's working set once every group has been walked, sized by the
        /// working set's own budget, kept off the flexbox and emitted verbatim.
        working_set: Option<WorkingSetShare>,
        /// The target's own group while it generates: every turn selected
        /// unconditionally, and the bounded pass only trims to the budget.
        ingest_self: bool,
    }

    let mut group_states: Vec<GroupState> = Vec::new();
    // Why each selected turn entered the slot — collected as we select, surfaced
    // on the returned `Projection` so the persisted record / GUI can show it.
    let mut selection_origins: HashMap<TurnKey, SelectionOrigin> = HashMap::new();

    // Hoisted out of the per-layer loop below: the kill switch is the
    // substrate's, so "is anything disabled?" has one answer for the whole
    // assembly. Asking it per layer made this the second-hottest resolved symbol
    // in an ingest profile.
    let any_layer_disabled = resolver.any_layer_disabled();
    let tool_round = selection.optional(TOOL_ROUND_SELECTOR) == Some(OptionalState::Present);
    // The turn's projection dial: a lower level selects none of the retrieved
    // memory it leaves out (`turn_dials`).
    let level = selection.projection_level();

    for (li, layer) in visible_layers.iter().enumerate() {
        let layer_is_target = layer.id == target.layer;
        // OUT OF SERVICE for the whole process (`--disable-layer`, applied via
        // `Builder::set_layer_gathered`): the layer contributes nothing to the
        // assembly. Its belief groups are already excluded at scoring time, but a
        // `Sequence` (recency) group is not belief-driven and is never scored, so
        // without this it would still emit its window — the exclusion has to be
        // restated where the assembly walks layers, or "excluded from provenance"
        // would hold for retrieved turns and quietly fail for recency ones.
        //
        // The target layer is exempt, for the same reason the diagnostic toggle
        // below exempts it: skipping it leaves the projection with nothing to
        // emit. Disabling the layer you are projecting FOR is not a meaningful
        // request, and answering it with an empty context would be worse than
        // ignoring it.
        if !layer_is_target && !layer.gathered {
            continue;
        }
        // Runtime diagnostic kill switch: a non-target layer toggled off
        // contributes nothing to the assembly (its groups are never scored or
        // selected). The target layer is never skipped — that would leave the
        // projection with nothing to emit. See `super::layer_toggle`.
        if !layer_is_target && any_layer_disabled && resolver.layer_disabled(&layer.name) {
            continue;
        }
        // A tool round leaves out the layers whose turns would compete with
        // the user's question — see `LayerSchema::in_tool_rounds`.
        if !layer_is_target && tool_round && !layer.in_tool_rounds {
            continue;
        }
        for group in &layer.groups {
            // Masking: for the target layer, only the target group is visible.
            if layer_is_target && group.id != target.group {
                continue;
            }
            // A focused projection takes nothing from another layer but the
            // working-set members it was served (`Schema::focus`).
            if schema.focus
                && !layer_is_target
                && !matches!(group.selection, SelectionRule::WorkingSet { .. })
            {
                continue;
            }
            // Nor, at a lower projection level, anything another layer holds
            // outside the working set.
            if !layer_is_target
                && !level.places(Kind::Layer(li))
                && !matches!(group.selection, SelectionRule::WorkingSet { .. })
            {
                continue;
            }

            // A working-set group is filled from the target's working set once
            // every group has been walked (Step 4b) — the folder groups' share
            // has to be known before the file groups can be sized, whatever
            // order the layers run in — so its corpus is never walked here. The
            // exception is an ingest conversation generating into this very
            // group, which reads its own turns below like any other.
            if let SelectionRule::WorkingSet { share } = group.selection {
                if !(resolver.target_is_ingest_self() && group.id == target.group) {
                    group_states.push(GroupState {
                        schema: group,
                        layer_idx: li,
                        selected: Vec::new(),
                        members: HashMap::new(),
                        group_score: 0.0,
                        score_density: false,
                        working_set: Some(share),
                        ingest_self: false,
                    });
                    continue;
                }
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
            // A tagged group admits only the turns carrying one of its tags: one
            // group holding every owner's conversations, each reader scoped to
            // its own. An untagged group admits everything, as it always has.
            //
            // **Never the target's own group.** That is already pinned to the
            // conversation's own timeline, so scoping it can only take turns
            // away — and the first one it takes is the turn being written, which
            // carries no tags until it seals. A prefill whose own turn is
            // scoped out of its own projection never finishes.
            //
            // Scoped BEFORE the exchange closure below, so the closure only ever
            // runs over turns this reader may see. An exchange lives on one
            // timeline, and a timeline in a tagged group is one owner's, so
            // closing a scoped turn's exchange cannot pull in another owner's.
            let scoped = !group.policy.tags.is_empty() && group.id != target.group;
            let in_scope: Vec<TurnKey> = resolver
                .group_turns(group.id)
                .into_iter()
                .filter(|key| !scoped || resolver.turn_carries(*key, &group.policy.tags))
                .collect();

            // A candidate is an EXCHANGE, not a turn. A tool round-trip is
            // recorded as coupled turns (`Sequence::couple_turn`) and is one
            // indivisible projection unit: the rules below rank whole runs and
            // the budget trims whole runs, so a `<tool_call>` is never seated
            // without the `<tool_response>` that answers it. An uncoupled turn
            // is its own singleton exchange, so a group with no round-trips
            // behaves exactly as before.
            //
            // The exchange is also the unit provenance voted on — the belief
            // scan aggregates a whole run into one case and stamps every member
            // with the shared score — so ranking exchanges here ranks exactly
            // what the scan decided, instead of re-splitting it into halves.
            let mut members: HashMap<TurnKey, Vec<TurnKey>> = HashMap::new();
            let all_turns: Vec<(TurnKey, f32)> =
                exchange_candidates(resolver, &in_scope, &mut members);

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
                    // §8 picks nodes, and a node can be one half of a tool
                    // round-trip on the live timeline. Close the picks under
                    // exchange so the emitted set is whole runs — the same unit
                    // every other path here selects. The scores stay §8's own:
                    // `exchange_candidates` takes the best over the members, and
                    // an unpicked member scores 0, so a closed exchange carries
                    // exactly the score of the pick that earned it.
                    let picked: Vec<TurnKey> = picks
                        .iter()
                        .map(|(idx, _origin, _score)| TurnKey::new(target.timeline, *idx))
                        .collect();
                    let pick_score: HashMap<TurnKey, f32> = picks
                        .iter()
                        .map(|(idx, _origin, score)| (TurnKey::new(target.timeline, *idx), *score))
                        .collect();
                    exchange_candidates(resolver, &picked, &mut members)
                        .into_iter()
                        .map(|(rep, _)| {
                            let score = members[&rep]
                                .iter()
                                .filter_map(|k| pick_score.get(k).copied())
                                .fold(f32::MIN, f32::max);
                            (rep, score)
                        })
                        .collect()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            // `members` is complete from here (every candidate exchange, plus any
            // the score-density path closed), so the budget can price a candidate
            // by what emitting it actually costs.
            let tc = |key: TurnKey| exchange_token_count(resolver, &members, key);

            // Fall through to the rule-based unbounded selection path
            // when score-density wasn't applicable.
            let ingest_self = resolver.target_is_ingest_self() && group.id == target.group;
            let mut selected: Vec<(TurnKey, f32)> = if score_density_used {
                selected
            } else if ingest_self {
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
                // exactly "this conversation's own turns", its fork lineage
                // first — 2-4 for a repo_map folder, every page read so far for
                // a code_read file. Neither is trimmed by the flexbox pass
                // below: an ingest conversation keeps its lineage whole, and
                // the own turns are the conversation itself. (A dialogue's
                // lineage is scored instead — `select_conversation`.)
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
                let mut beliefs = crate::provenance::belief_step(
                    &fresh,
                    &prior_scores,
                    &prior_selected,
                    &prior_qualified,
                    cfg.section_policy(0),
                    cfg.budget(),
                    floor,
                );
                // A mid-decode reprojection brings at most `max_swaps` new
                // conversations into the group and keeps a leaver for each one
                // it holds back (`swap_cap`). The opening projection of a turn
                // (position 0) is a new question and reselects freely.
                if decode_pos.is_some_and(|pos| pos > 0) {
                    let keys: Vec<TurnKey> = all_turns.iter().map(|(k, _)| *k).collect();
                    let mut selected: Vec<bool> = beliefs.iter().map(|b| b.selected).collect();
                    let scores: Vec<f32> = beliefs.iter().map(|b| b.score).collect();
                    let capped = cap_new_members(
                        &keys,
                        &mut selected,
                        &prior_selected,
                        &scores,
                        group.max_swaps,
                    );
                    if capped != SwapCap::default() {
                        tracing::debug!(
                            target: "candle_conversation::scheduler::reproject",
                            group = %group.name,
                            held_back = capped.held_back,
                            kept = capped.kept,
                            max_swaps = group.max_swaps,
                            "reprojection swap cap: newcomers held back"
                        );
                        for (b, s) in beliefs.iter_mut().zip(selected) {
                            b.selected = s;
                        }
                    }
                }
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
                    Some(target.timeline),
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

            // An empty selection stays empty. A group whose members all scored
            // below the threshold has nothing relevant to contribute, and Step 7
            // drops it — taking its layer with it if no sibling group survived.
            // That is the intended outcome, not a hole to be plugged.
            //
            // This used to inject the group's declared `default` turn with a
            // SENTINEL score synthesised to clear the very gates the turn had
            // just failed. It defeated the threshold by construction: repo_map
            // sets `score_threshold: 1.0` precisely so a cold probe (every folder
            // scoring exactly 0) selects nothing, and the fallback then put the
            // workspace-root exchange back regardless — ~4,000 tokens of folder
            // listing and summary in front of a model that had asked for none of
            // it. Measured on a live greeting: several exchanges injected, every
            // one either `reason: fallback` or scoring 0.0.
            //
            // Selection is by score, on the same terms as the tool catalog: pass
            // the threshold or stay out.

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
                    // Turn 0 opens its timeline's first exchange, so it is that
                    // exchange's representative; closing here records the
                    // membership for a timeline whose head was never a candidate.
                    let head = close_under_exchange(
                        resolver,
                        &mut members,
                        TurnKey::new(timeline, TurnIndex(0)),
                    );
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
            // `Sequence` rule's own split so the panel can still answer "why?",
            // mirroring `select_conversation`:
            // - a turn on another timeline is inherited, inviolate and outside
            //   the window: `Seed` when it is a link of this member's seed,
            //   `Inherited` for any other ancestor;
            // - of the conversation's own turns, one inside the last `recent` is
            //   `Recent` (inviolate), an older one `Historical` (top-k by score).
            if !score_density_used {
                if let SelectionRule::Sequence { recent, .. } = &group.selection {
                    // The window splits the own turns alone. Read the split off
                    // each pick's POSITION among them rather than its turn index:
                    // the candidates are exchanges, so position and index only
                    // coincide when nothing on the timeline is coupled.
                    let own: Vec<TurnKey> = all_turns
                        .iter()
                        .map(|(key, _)| *key)
                        .filter(|key| key.timeline == target.timeline)
                        .collect();
                    let split = own.len().saturating_sub(*recent);
                    let position: HashMap<TurnKey, usize> =
                        own.iter().enumerate().map(|(i, key)| (*key, i)).collect();
                    let seed = resolver.seed_timelines();
                    for (key, _) in &selected {
                        let origin = if key.timeline != target.timeline {
                            if seed.contains(&key.timeline) {
                                SelectionOrigin::Seed
                            } else {
                                SelectionOrigin::Inherited
                            }
                        } else if position.get(key).is_some_and(|p| *p >= split) {
                            SelectionOrigin::Recent
                        } else {
                            SelectionOrigin::Historical
                        };
                        selection_origins.insert(*key, origin);
                    }
                }
            }

            // Every non-representative member of a selected exchange rides in on
            // its representative's decision, so it carries the same score and the
            // same origin. Stamping them is what makes the selection record
            // explain every turn the projection emits — a `<tool_response>` that
            // appears "unselected" in the panel reads as a leak. Representatives
            // are the only keys the next reprojection seeds from
            // (`Carried::turn_group` looks up `all_turns`), so this cannot
            // perturb the belief carry.
            for (rep, score) in &selected {
                let Some(m) = members.get(rep) else { continue };
                let origin = selection_origins.get(rep).copied();
                for key in m.iter().skip(1) {
                    selection_scores.set_turn(*key, *score, true);
                    if let Some(origin) = origin {
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
                members,
                group_score: 0.0,
                score_density: score_density_used,
                working_set: None,
                ingest_self,
            });
        }
    }

    // ── Step 4b: Working-set groups ─────────────────────────────────────────
    // `docs/zend_working_set.md` §4.4. Each group emits the target's working-set
    // members in it, whole, in the working set's own order — the set enforced
    // the budget as they entered, so nothing is chosen here. A target whose
    // layer declares no working set — or that holds none — selects nothing.
    let declares_working_set = schema
        .layers
        .iter()
        .any(|l| l.id == target.layer && l.working_set.is_some());
    if declares_working_set {
        for gs in group_states
            .iter_mut()
            .filter(|gs| gs.working_set.is_some())
        {
            // Focused, the projection is held to the conversation itself
            // (`Schema::focus`): it carries no working set.
            if schema.focus {
                continue;
            }
            // A kind the projection dial leaves out picks nothing.
            if !level.places(share_kind(gs.working_set)) {
                continue;
            }
            let Some(members) = resolver.working_set_members(gs.schema.id) else {
                continue;
            };
            for p in working_set_pick::pick(&members) {
                for key in resolver.timeline_turns(p.timeline) {
                    selection_scores.set_turn(key, p.score, true);
                    selection_origins.insert(key, p.origin);
                    gs.selected.push((key, p.score));
                }
            }
        }
    }
    // Below Full the target's group keeps its own turns and drops the lineage
    // it inherited.
    if !level.places_lineage() {
        if let Some(gs) = group_states
            .iter_mut()
            .find(|gs| gs.schema.id == target.group)
        {
            gs.selected.retain(|(k, _)| k.timeline == target.timeline);
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
    // A working-set group is exempt: its pinned members are promises already
    // made to the model, not candidates a threshold may turn away.
    group_states.retain(|gs| {
        let layer = &visible_layers[gs.layer_idx];
        gs.working_set.is_some() || gs.group_score >= layer.score_threshold
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
    let mut system_prompt_segments: Vec<ProjectionSegment> = emit_system_prompt_items(
        &schema.system_prompt,
        resolver,
        mode,
        effective_selection,
        prior,
        decode_pos,
        &mut selection_scores,
        &mut resolved_selections,
    );
    mark_pins(&schema.system_prompt, resolver, &mut system_prompt_segments);

    // ── Step 8b: The knowledge base — the files the target's files set holds,
    // each kind under its share of the room, spliced after `files_open`
    // (docs/provenance_system_collection.md §3, §4). Before any turn group is
    // weighed, so an opening, which has none yet, places them too. The fill
    // below counts the system prompt without them: the kinds' caps and the
    // dialogue's share are shares of one room.
    let mut files_placed: Vec<FileKey> = Vec::new();
    let mut files_tokens = 0usize;
    let knowledge_base = schema
        .layers
        .iter()
        .find(|l| l.id == target.layer)
        .and_then(|l| l.working_set.as_ref())
        .zip(resolver.trained_window())
        .filter(|_| !schema.focus);
    let files_open = schema
        .system_prompt
        .items
        .iter()
        .find_map(|item| match item {
            SystemPromptItem::Section(s) if s.name == FILES_OPEN_SECTION => Some(s.id),
            _ => None,
        });
    // A pinned knowledge base — a calibration's frame (§12 D11) — is placed
    // whatever the target's layer declares: a pin takes no share of a room.
    let pinned = resolver
        .files_set()
        .is_some_and(|s| s.pin != FilesPin::Free);
    let shares = knowledge_base.map(|(ws, trained)| {
        let room = ws
            .slot_window(trained)
            .window
            .saturating_sub(ws.reserve_tokens)
            .saturating_sub(segment_tokens(&system_prompt_segments, resolver))
            .saturating_sub(decode_pos.unwrap_or(0));
        ws.fill_shares.caps(room)
    });
    if let (true, Some(files_open)) = (shares.is_some() || pinned, files_open) {
        let kind_cap = |share: WorkingSetShare| {
            let Some(shares) = &shares else {
                return 0;
            };
            if !level.places(share_kind(Some(share))) {
                return 0;
            }
            match share {
                WorkingSetShare::Readings => shares.readings,
                WorkingSetShare::Folders => shares.folders,
                WorkingSetShare::Guides => shares.guides,
                WorkingSetShare::Journals => shares.journals,
            }
        };
        let splice = files_splice(schema, resolver, kind_cap, effective_selection.seeds());
        let at = system_prompt_segments.iter().position(|seg| {
            matches!(seg, ProjectionSegment::Sealed(SealedKind::Section(s)) if s.id == files_open)
        });
        if let (Some(at), false) = (at, splice.segments.is_empty()) {
            files_tokens = segment_tokens(&splice.segments, resolver);
            system_prompt_segments.splice(at + 1..at + 1, splice.segments);
            files_placed = splice.order;
        }
    }

    if group_states.is_empty() {
        return Projection {
            segments: system_prompt_segments,
            selections: resolved_selections,
            selection_scores,
            selection_origins,
            fill: Vec::new(),
            unpooled: Vec::new(),
            files_placed,
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
    let own_window = schema
        .layers
        .iter()
        .find(|l| l.id == target.layer)
        .map(|l| l.window)
        .unwrap_or(0);

    // **A dialogue whose engine selects is filled, not flexed**
    // (`docs/hit_driven_projection.md` §7.2): its lineage, its working set, its
    // own exchanges and every other group's selections go into one
    // score-ordered fill of the engine's trained window, and nothing takes
    // part in the flexbox.
    let fill_target = schema
        .layers
        .iter()
        .find(|l| l.id == target.layer)
        .and_then(|l| l.working_set.as_ref().map(|ws| (l, ws)))
        .zip(resolver.trained_window())
        .filter(|_| declares_working_set);
    let flexed = |gs: &GroupState| gs.working_set.is_none() && fill_target.is_none();
    let flex_layer_indices: Vec<usize> = surviving_layer_indices
        .iter()
        .copied()
        .filter(|&li| {
            group_states
                .iter()
                .any(|gs| gs.layer_idx == li && flexed(gs))
        })
        .collect();

    // **An ingest conversation compounding on the layers below it sees each at
    // its own window**, as a dialogue does — not all of them sliced from its
    // own. Its own group asks for nothing (it is kept whole), so its window
    // was the whole pie for every layer below: one layer below took all 8k, two
    // shared it, and a fourth layer up was left a sliver per layer, under the
    // size of one unit — the higher the layer, the less it compounded. The
    // budget is the windows of the layers it draws on, each capped at its own
    // below. An ingest target with no layer to draw on (`code_reading`, whose
    // `repo_map` below is a working set) keeps its own window, as before; a
    // layer below it reads self-local (one it does not compound on) adds
    // nothing to the budget.
    let compounded = |li: usize| {
        visible_layers[li].id != target.layer
            && group_states
                .iter()
                .any(|gs| gs.layer_idx == li && flexed(gs) && resolver.compounds_on(gs.schema.id))
    };
    let compounding =
        resolver.target_is_ingest_self() && flex_layer_indices.iter().any(|&li| compounded(li));
    let below_windows: usize = flex_layer_indices
        .iter()
        .filter(|&&li| compounded(li))
        .map(|&li| visible_layers[li].window)
        .sum();
    let turn_budget = if compounding {
        own_window.max(below_windows)
    } else {
        own_window
    };

    // Natural token consumption per group (unbounded selection result).
    let natural_tokens: HashMap<GroupId, usize> = group_states
        .iter()
        .filter(|gs| flexed(gs))
        .map(|gs| {
            // An ingest conversation's own group is kept whole whatever its
            // budget (the bounded pass below), so it asks the window for
            // nothing: priced at its size, a unit whose lineage is a whole
            // file's reading took nearly all of an 8k window and left the layers
            // it compounds on a sliver — the units of other files it should
            // draw on scored in the hundreds and were dropped for want of room.
            if gs.ingest_self {
                return (gs.schema.id, 0);
            }
            // Priced per EXCHANGE — what emitting the candidate actually costs —
            // so the flexbox natural cap matches the tokens the group emits.
            let tokens: usize = gs
                .selected
                .iter()
                .map(|(key, _)| exchange_token_count(resolver, &gs.members, *key))
                .sum();
            (gs.schema.id, tokens)
        })
        .collect();

    // Natural consumption per layer (sum of groups).
    let layer_natural: Vec<usize> = flex_layer_indices
        .iter()
        .map(|&li| {
            group_states
                .iter()
                .filter(|gs| gs.layer_idx == li && flexed(gs))
                .map(|gs| natural_tokens[&gs.schema.id])
                .sum()
        })
        .collect();

    // Layer-level flex with natural cap so freed budget redistributes. Concept
    // B: a layer's attention mass (the sum over its groups') scales its
    // priority within the declared rails — a tour-shaped probe lifts repo_map
    // above its static share, a code probe grows the scopes layer.
    let layer_items: Vec<FlexItem> = flex_layer_indices
        .iter()
        .enumerate()
        .map(|(slot, &li)| {
            let layer_mass: f32 = group_states
                .iter()
                .filter(|gs| gs.layer_idx == li && flexed(gs))
                .map(|gs| resolver.group_attention_mass(gs.schema.id))
                .sum();
            let mut item = FlexItem::from_budget_with_mass(
                &visible_layers[li].budget,
                turn_budget,
                layer_mass,
            );
            let mut nat = layer_natural[slot];
            if compounding && visible_layers[li].id != target.layer {
                nat = nat.min(visible_layers[li].window);
            }
            item.max_tokens = Some(item.max_tokens.map_or(nat, |m| m.min(nat)));
            item
        })
        .collect();
    let layer_budgets = flexbox_distribute(&layer_items, turn_budget);

    let mut final_selected: Vec<(GroupId, TurnKey)> = vec![];

    // Working-set groups emit what Step 4b picked, verbatim and untrimmed —
    // unless the dialogue is filled, which places them below.
    if fill_target.is_none() {
        if declares_working_set && !UNFILLED_REPORTED.swap(true, Ordering::Relaxed) {
            tracing::info!(
                layer = ?target.layer,
                "a dialogue with a working set is projected without a fill: the engine reports \
                 no trained window, so its members are placed whole"
            );
        }
        for gs in group_states.iter().filter(|gs| !flexed(gs)) {
            for (key, _) in &gs.selected {
                push_exchange(&mut final_selected, gs.schema.id, &gs.members, *key);
            }
        }
    }

    for (slot, &li) in flex_layer_indices.iter().enumerate() {
        let layer_budget = layer_budgets[slot];

        let layer_groups: Vec<&GroupState> = group_states
            .iter()
            .filter(|gs| gs.layer_idx == li && flexed(gs))
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
                    push_exchange(&mut final_selected, gs.schema.id, &gs.members, *key);
                }
                continue;
            }
            let group_budget = group_budgets[gi];
            let tc = |key: TurnKey| exchange_token_count(resolver, &gs.members, key);

            let selected_indices: Vec<TurnKey> = if gs.ingest_self {
                // An ingest conversation's own group is its lineage and its own
                // turns, and neither is the budget's to trim. Its lineage is
                // kept whole here (bounded by
                // `Substrate::INHERITED_CHAIN_TOKEN_CAP`, not by the window) —
                // unlike a dialogue's, which `select_conversation` scores from
                // `lineage_base` — and the own turns are the conversation itself — its
                // request and every page it has read since, bounded by its own
                // read rounds. Trimmed by score, both lost to a tie: the own
                // turns carry no belief yet and sort last by key, so a file
                // conversation answered its final turn without its own request
                // and summarised the priming chain's last document instead.
                let mut kept = gs.selected.clone();
                kept.sort_by_key(|(idx, _)| *idx);
                kept.into_iter().map(|(idx, _)| idx).collect()
            } else if gs.schema.is_belief_driven() {
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
                    Some(target.timeline),
                )
            };

            // A group the budget trimmed to nothing stays trimmed. Re-injecting
            // a declared default here restored a member the budget had just
            // rejected, which is the same defeat of the gate as the empty-
            // selection fallback above — the group is present not because
            // anything qualified but because something had to be.

            for idx in selected_indices {
                push_exchange(&mut final_selected, gs.schema.id, &gs.members, idx);
            }
        }
    }

    // ── Step 11b: The dialogue's fill ────────────────────────────────────────
    let mut fill: Vec<PieceReport> = Vec::new();
    let mut unpooled: Vec<Unpooled> = Vec::new();
    if let Some(((layer, ws), trained)) = fill_target {
        let slot_window = ws.slot_window(trained);
        if let Some(why) = &slot_window.expanded {
            if !SLOT_EXPANDED_WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    layer = %layer.name,
                    window = slot_window.window,
                    "{why}: the slot fills YaRN ×2"
                );
            }
        }
        // The room the fill has: the slot's window less the reserve, the
        // system prompt as emitted and what the turn has appended.
        let room = slot_window
            .window
            .saturating_sub(ws.reserve_tokens)
            .saturating_sub(
                segment_tokens(&system_prompt_segments, resolver).saturating_sub(files_tokens),
            )
            .saturating_sub(decode_pos.unwrap_or(0));
        let (recent, lineage_base) = group_states
            .iter()
            .find(|gs| gs.schema.id == target.group)
            .and_then(|gs| match gs.schema.selection {
                SelectionRule::Sequence {
                    recent,
                    lineage_base,
                    ..
                } => Some((recent, lineage_base)),
                _ => None,
            })
            .unwrap_or((0, LINEAGE_BASE));
        let exchanges_of = |keys: &[TurnKey]| -> Vec<Vec<TurnKey>> {
            let mut ex = resolver.group_exchanges(keys);
            ex.sort_by_key(|e| e.first().copied());
            ex
        };
        // The target's own turns and its lineage.
        let own_group = group_states.iter().find(|gs| gs.schema.id == target.group);
        let (own_keys, inherited): (Vec<TurnKey>, Vec<TurnKey>) = own_group
            .map(|gs| gs.selected.iter().map(|(k, _)| *k).collect::<Vec<_>>())
            .unwrap_or_default()
            .into_iter()
            .partition(|k| k.timeline == target.timeline);
        let mut ancestors: Vec<TimelineId> = Vec::new();
        for k in &inherited {
            if !ancestors.contains(&k.timeline) {
                ancestors.push(k.timeline);
            }
        }
        let lineage: Vec<(Vec<TurnKey>, f32)> = exchanges_of(&inherited)
            .into_iter()
            .map(|ex| {
                let rank = ex
                    .first()
                    .and_then(|k| ancestors.iter().position(|&t| t == k.timeline))
                    .unwrap_or(0) as f32;
                let s = ex
                    .iter()
                    .map(|k| resolver.turn_score(*k))
                    .fold(0.0, f32::max)
                    .max(lineage_base)
                    + rank * LINEAGE_RANK_STEP;
                (ex, s)
            })
            .collect();
        // The working set: each member conversation a piece, in the set's order.
        let mut working: Vec<WorkingPiece> = Vec::new();
        let mut working_groups: Vec<GroupId> = Vec::new();
        for gs in group_states.iter().filter(|gs| gs.working_set.is_some()) {
            let kind = share_kind(gs.working_set);
            let mut order: Vec<TimelineId> = Vec::new();
            for (k, _) in &gs.selected {
                if !order.contains(&k.timeline) {
                    order.push(k.timeline);
                }
            }
            for tl in order {
                let ks: Vec<TurnKey> = gs
                    .selected
                    .iter()
                    .filter(|(k, _)| k.timeline == tl)
                    .map(|(k, _)| *k)
                    .collect();
                working.push(WorkingPiece {
                    kind,
                    exchanges: exchanges_of(&ks),
                    momentum: resolver.working_set_standing(tl).momentum,
                });
                working_groups.push(gs.schema.id);
            }
        }
        // Every other group: each selected exchange whole, at its selection
        // score — on the dialogue's layer under its window, on another layer
        // under that layer's.
        let mut context: Vec<ContextExchange> = Vec::new();
        let mut context_of: Vec<GroupId> = Vec::new();
        for gs in group_states
            .iter()
            .filter(|gs| gs.working_set.is_none() && gs.schema.id != target.group)
        {
            let kind = if visible_layers[gs.layer_idx].id == target.layer {
                Kind::Dialogue
            } else {
                Kind::Layer(gs.layer_idx)
            };
            for (key, score) in &gs.selected {
                context.push(ContextExchange {
                    kind,
                    exchange: gs.members.get(key).cloned().unwrap_or_else(|| vec![*key]),
                    score: *score,
                });
                context_of.push(gs.schema.id);
            }
        }
        // Each kind's cap is its share of this projection's room
        // (`docs/hit_driven_projection.md` §7.2 step 4).
        // A kind the projection dial leaves out has no room at all.
        let shares = ws.fill_shares.caps(room);
        let cap = |kind: Kind, tokens: usize| if level.places(kind) { tokens } else { 0 };
        let readings = cap(Kind::Reading, shares.readings);
        let caps = Caps {
            readings,
            folders: cap(Kind::Folder, shares.folders),
            guides: cap(Kind::Guide, shares.guides),
            journals: cap(Kind::Journal, shares.journals),
            dialogue: shares.dialogue,
            layers: visible_layers
                .iter()
                .enumerate()
                .map(|(i, l)| cap(Kind::Layer(i), l.window))
                .collect(),
            reading_piece: reading_piece_cap(readings),
            minimum: minimum_cap(readings),
        };
        // What the slot holds from the boundary before, and whether this is
        // the first boundary after a new user message — which chooses whole,
        // with no allowance (§7.2 step 5).
        let boundary = resolver.working_set_boundary().unwrap_or_default();
        let allowance = (!boundary.opening).then(|| caps.allowance());
        let slot = DialogueSlot {
            room,
            caps,
            recent,
            lineage,
            working,
            own: exchanges_of(&own_keys),
            context,
            held: boundary.held,
            allowance,
        };
        let plan = dialogue_fill::plan(
            &slot,
            |k| resolver.turn_score(k),
            |k| resolver.turn_token_count(k),
            |k| resolver.turn_reads(k),
        );
        // What the fill weighed each working piece at — the scores it ranked
        // by, which the selection record does not show (`fill_report`) — and
        // what the scan scored that the pool does not hold. Both ride the
        // projection record.
        fill = piece_reports(
            &slot.working,
            &plan.working,
            |k| resolver.turn_score(k),
            |k| resolver.turn_token_count(k),
            |t| resolver.working_set_standing(t),
        );
        unpooled = resolver.working_set_unpooled(UNPOOLED_KEPT);
        note("projection fill pieces", || line(&fill, 10));
        // Each cap's use (`docs/hit_driven_projection.md` §12 item 26).
        let tokens_of = |keys: &[TurnKey]| -> usize {
            keys.iter().map(|k| resolver.turn_token_count(*k)).sum()
        };
        let mut used: HashMap<Kind, usize> = HashMap::new();
        for (piece, placed) in slot.working.iter().zip(&plan.working) {
            *used.entry(piece.kind).or_default() += tokens_of(placed);
        }
        let used_of = |k: Kind| used.get(&k).copied().unwrap_or(0);
        tracing::info!(
            target: "candle_conversation::projection::fill",
            room,
            readings = format!("{}/{}", used_of(Kind::Reading), slot.caps.readings),
            folders = format!("{}/{}", used_of(Kind::Folder), slot.caps.folders),
            guides = format!("{}/{}", used_of(Kind::Guide), slot.caps.guides),
            journals = format!("{}/{}", used_of(Kind::Journal), slot.caps.journals),
            dialogue = format!(
                "{}/{}",
                tokens_of(&plan.lineage) + tokens_of(&plan.own),
                slot.caps.dialogue
            ),
            pieces_offered = slot.working.len(),
            pieces_placed = plan.working.iter().filter(|p| !p.is_empty()).count(),
            over_room = plan.over_room,
            "fill: each cap's use"
        );
        let mut push = |gid: GroupId, key: TurnKey| final_selected.push((gid, key));
        for p in plan.lineage.into_iter().chain(plan.own) {
            push(target.group, p);
        }
        for (gid, placed) in working_groups.into_iter().zip(plan.working) {
            for p in placed {
                push(gid, p);
            }
        }
        // In each group's own selection order, which a summary-forest group
        // has already arranged with each summary above the turns it covers.
        for (gid, placed) in context_of.into_iter().zip(plan.context) {
            for p in placed {
                push(gid, p);
            }
        }
    }

    // ── Step 12: Emit ─────────────────────────────────────────────────────────
    // Groups are emitted in declaration order per layer, but sorted by
    // descending group score within each layer (ties: declaration order).
    // Each group's turns are gathered into one block first; a target with a
    // working set then has its blocks reordered (below).
    let mut blocks: Vec<(&GroupState, LayerId, Vec<TurnKey>)> = Vec::new();

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
            // the turns it refines); leave it untouched. So does a working-set
            // group: its members come in the order they joined the set, each
            // conversation's turns in order, so that a rebuild keeps the slot's
            // prefix up to the newest member (`working_set_pick::pick`).
            if !gs.score_density && gs.working_set.is_none() {
                group_turns.sort();
            }
            blocks.push((gs, layer_id_for_walk, group_turns));
        }
    }

    // **A dialogue's working set sits between its opening and its own
    // turns**: `[other layers][the inherited chain][working set][own turns]`
    // (`docs/zend_working_set.md` §4.1). The locks — the files the model was
    // just told it has — are the working set's tail, so they land next to the
    // conversation that asked for them. In layer order they sat behind every
    // other layer's groups and the whole inherited chain: measured
    // 2026-10-24, a locked paper ~130K tokens before the question, which the
    // model then could not find and called the tool's reply false. What now
    // sits after the working set changes rarely — the chain never, the
    // analysis layers only as locks change — so moving it costs no rebuilds.
    if declares_working_set {
        let (working_set, mut rest): (Vec<_>, Vec<_>) = blocks
            .into_iter()
            .partition(|(gs, _, _)| gs.working_set.is_some());
        let own_at = rest
            .iter()
            .position(|(gs, _, _)| gs.schema.id == target.group);
        let mut own_tail = None;
        if let Some(at) = own_at {
            let (gs, layer, keys) = rest.remove(at);
            let (lineage, own): (Vec<TurnKey>, Vec<TurnKey>) = keys
                .into_iter()
                .partition(|key| key.timeline != target.timeline);
            rest.push((gs, layer, lineage));
            own_tail = Some((gs, layer, own));
        }
        rest.extend(working_set);
        rest.extend(own_tail);
        blocks = rest;
    }

    // A dialogue holds an ingest unit's opening by its face — the unit's opening
    // calls under a user turn naming only the reading — never by the request
    // that commissioned it, which reads as the user's own
    // (`Substrate::face_of`). A unit being read projects its own opening.
    let faces = !resolver.target_is_ingest_self();
    let mut turns: Vec<ResolvedTurn> = vec![];
    for (gs, layer_id_for_walk, group_turns) in blocks {
        for selected in group_turns {
            let key = match faces {
                true => resolver.face_of(selected).unwrap_or(selected),
                false => selected,
            };
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
            // resolver doesn't track origins (e.g. test mocks). A face
            // reports the layer of the unit it stands for.
            let origin_layer = resolver.turn_origin(selected).unwrap_or(layer_id_for_walk);
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
        fill,
        unpooled,
        files_placed,
    }
}
