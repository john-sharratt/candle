//! Walks the shared system prompt and emits its sections for one projection.
//!
//! Sections emit in declaration order, and a collection emits the members its
//! selection kept. A section-tree node emits the pre-sealed variant of its
//! selected branch, so the K/V it carries matches the prefix it follows. A
//! collection's catalog summary is emitted outside its structural markers, and
//! it is emitted even when nothing was selected, because it is then the only
//! thing that names the members. Sections gated on a collection emit only when
//! that collection materialised.
//!
//! Every item takes the same positions on every projection
//! (`docs/placed_section_layout.md` §3.1c): a section is padded to its slot and
//! a gated one that does not emit is a hole of it, each tree node is padded to
//! its slot, a ranged collection fills its range and a summary with a range
//! fills that.

use super::*;
use crate::projection::{collection_of, node_slot, section_slot, summary_points};

/// Walk the shared system prompt's `items` in declaration order, emitting each
/// plain section verbatim or each collection's surviving subset (after applying
/// its selection rule), with section-tree branches resolved from `selection_state`.
///
/// A [`SectionSchema`] with `depends_on = Some(cid)` only emits if the
/// named collection materialised ≥ 1 section in this same emission pass.
#[allow(clippy::too_many_arguments)]
/// A catalog summary, then the notes the turn set on it that are sealed.
pub(super) fn emit_summary<R: ContentResolver>(
    out: &mut Vec<ProjectionSegment>,
    summary: SectionId,
    resolver: &R,
    selection_state: &SelectionState,
) {
    let notes = selection_state
        .summary_notes(summary)
        .iter()
        .copied()
        .filter(|&id| resolver.section_token_count(id) > 0);
    for id in std::iter::once(summary).chain(notes) {
        out.push(ProjectionSegment::Sealed(SealedKind::Section(
            ResolvedSection::new(id),
        )));
    }
}

pub(super) fn emit_system_prompt_items<R: ContentResolver>(
    sp: &SystemPromptSchema,
    resolver: &R,
    mode: ProjectionMode,
    selection_state: &SelectionState,
    prior: &PriorBelief,
    decode_pos: Option<usize>,
    scores: &mut super::super::SelectionScores,
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
    // marker (OUTSIDE it). A summary emits when the selection is a proper subset
    // — top-k/threshold dropped at least one member, so the summary names the
    // full set — and the summary section is actually sealed.
    //
    // **Including when the selection is EMPTY**, which is the case that matters
    // most: the summary is the only thing naming the collection's members, and
    // the turn where provenance selected none is precisely the turn the model
    // most needs to be told they exist. Requiring a non-empty selection deleted
    // the catalog listing on exactly that turn — the model was shown no tools and
    // answered from memory. It sits outside the structural markers by design, so
    // it stands alone perfectly well with no `<tools>` block after it.
    let mut pending_summaries: std::collections::HashMap<CollectionId, SectionId> =
        std::collections::HashMap::new();
    let mut record = |coll: &SectionCollection, selected: Vec<ProjectionSegment>| {
        if let Some(sum_id) = coll.summary_section {
            let partial = selected.len() < coll.sections.len();
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
                        let selected = select_tree_collection_segments(
                            tc,
                            active_key,
                            resolver,
                            &scoring,
                            selection_state,
                        );
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
    // Collections that are CONFIGURED — declare ≥1 member — which is a different
    // question from whether provenance picked one of them this projection, and
    // the one the `*_configured` gates ask. A tools dial set to `None` filters
    // every member out, so an empty membership really is "the facility is off";
    // a full membership with nothing selected is "the facility is on and nothing
    // scored", which must still tell the model the facility exists.
    let configured_collections: std::collections::HashSet<CollectionId> = sp
        .items
        .iter()
        .filter_map(|it| match it {
            SystemPromptItem::Collection(c) if !c.sections.is_empty() => Some(c.id),
            _ => None,
        })
        .collect();
    // Second pass: walk items in declaration order, applying the
    // `depends_on` predicate to Sections and using the cached
    // collection results for Collections. Every item takes its slot
    // (`docs/placed_section_layout.md` §3.1c): a section is padded to its slot
    // and a gated one that does not emit is a hole of it, each tree node is
    // padded to its slot, and a catalog summary with a range stands in it.
    let summary_slots = summary_points(&sp.items);
    for (at, item) in sp.items.iter().enumerate() {
        for &cid in summary_slots.get(&at).into_iter().flatten() {
            emit_summary_slot(
                &mut out,
                &sp.items,
                cid,
                pending_summaries.remove(&cid),
                resolver,
                selection_state,
            );
        }
        match item {
            SystemPromptItem::Section(s) => {
                let should_emit = match (
                    s.depends_on,
                    s.depends_on_absent,
                    s.depends_on_configured,
                    s.depends_on_unconfigured,
                ) {
                    // `depends_on`: emit only when the collection materialised ≥1.
                    (Some(cid), _, _, _) => non_empty_collections.contains(&cid),
                    // `depends_on_absent`: the inverse — emit only when it
                    // materialised zero (the no-tools variant).
                    (None, Some(cid), _, _) => !non_empty_collections.contains(&cid),
                    // `depends_on_configured` / `_unconfigured`: keyed on whether
                    // the collection HAS members, independent of this
                    // projection's selection.
                    (None, None, Some(cid), _) => configured_collections.contains(&cid),
                    (None, None, None, Some(cid)) => !configured_collections.contains(&cid),
                    (None, None, None, None) => true,
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
                                emit_summary(&mut out, sum_id, resolver, selection_state);
                            }
                        }
                    }
                    push_section_segment(&mut out, s);
                }
                // The section's slot: what it emitted and a hole of the rest,
                // or all of it a hole when it did not emit.
                let tokens = resolver.section_token_count(s.id);
                let slot = section_slot(s, tokens);
                let positions = if should_emit { slot - tokens } else { slot };
                if positions > 0 {
                    out.push(ProjectionSegment::Hole { positions });
                }
            }
            SystemPromptItem::Collection(coll) => {
                if let Some(selected) = collection_results.remove(&coll.id) {
                    // Fallback for a collection with no opening structural template
                    // to hang the summary on: emit it just before the members, as
                    // before. (When the collection IS wrapped — e.g. `<tools>` —
                    // the open template already drained and emitted it outside.)
                    if let Some(sum_id) = pending_summaries.remove(&coll.id) {
                        emit_summary(&mut out, sum_id, resolver, selection_state);
                    }
                    match &coll.range {
                        Some(range) => emit_in_range(&mut out, coll, range, selected, resolver),
                        None => out.extend(selected),
                    }
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
                    let node_start = out.len();
                    'node: {
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
                            break 'node;
                        }
                        // A prefix-transparent embedded collection emits its cached
                        // active-branch selection — the same drain path as a
                        // top-level collection (summary outside its markers).  When
                        // `deferred_projection` is set its selection is emitted by a
                        // placeholder node below (`inject_collection`), so it skips
                        // its own emission and leaves the cached results in place.
                        if let Some(tc) = &node.collection {
                            if tc.deferred_projection {
                                break 'node;
                            }
                            if let Some(selected) = collection_results.remove(&tc.collection.id) {
                                if let Some(sum_id) = pending_summaries.remove(&tc.collection.id) {
                                    emit_summary(&mut out, sum_id, resolver, selection_state);
                                }
                                out.extend(selected);
                            }
                            break 'node;
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
                            break 'node;
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
                        if let Some(v) =
                            option.variant_for(tree.pack(&selection, node.ancestor_dims))
                        {
                            out.push(ProjectionSegment::Sealed(SealedKind::Section(
                                ResolvedSection::new(v.id),
                            )));
                        }
                    }
                    // The node's slot: what it emitted and a hole of the rest.
                    let used = segment_tokens(&out[node_start..], resolver);
                    let slot = node_slot(node, |id| resolver.section_token_count(id));
                    if slot > used {
                        out.push(ProjectionSegment::Hole {
                            positions: slot - used,
                        });
                    }
                }
            }
        }
    }
    out
}

/// Emit collection `cid`'s catalog summary slot: its `summary` and the notes
/// the turn set on it, when it emits this projection, and a hole of the rest
/// of the collection's `summary_range` (`docs/placed_section_layout.md`
/// §3.1c). A summary past its range is emitted whole and warned of: everything
/// after it moves.
fn emit_summary_slot<R: ContentResolver>(
    out: &mut Vec<ProjectionSegment>,
    items: &[SystemPromptItem],
    cid: CollectionId,
    summary: Option<SectionId>,
    resolver: &R,
    selection_state: &SelectionState,
) {
    let Some(range) = collection_of(items, cid).and_then(|c| c.summary_range) else {
        return;
    };
    let mut slot = Vec::new();
    if let Some(summary) = summary {
        emit_summary(&mut slot, summary, resolver, selection_state);
    }
    let used = segment_tokens(&slot, resolver);
    let reserved = range.positions(resolver.trained_window());
    if used > reserved {
        tracing::warn!(
            collection = ?cid,
            used,
            reserved,
            "a catalog summary takes more than its range: everything after it moves"
        );
    }
    out.extend(slot);
    let hole = range.hole(resolver.trained_window(), used);
    if hole > 0 {
        out.push(ProjectionSegment::Hole { positions: hole });
    }
}

/// Emit a collection's `selected` members in its reserved range: what they do
/// not fill is a hole, ahead of them in a range filled from the top, after
/// them in one filled from the bottom, so the range always takes its full size
/// on an engine trained to the resolver's window (`docs/placed_section_layout.md`
/// §3.1).
///
/// A range filled from the bottom holds its members in the order they were
/// written, and a full one is a window (§3.7): the oldest go, their glue with
/// them, until the rest fit — the newest always placed, the survivors sliding
/// down by what left. Members past a range filled from the top are emitted
/// whole and the range runs over, which the fill is there to prevent, and is
/// warned of.
pub(super) fn emit_in_range<R: ContentResolver>(
    out: &mut Vec<ProjectionSegment>,
    coll: &SectionCollection,
    range: &ReservedRange,
    selected: Vec<ProjectionSegment>,
    resolver: &R,
) {
    let native = resolver.trained_window();
    let reserved = range.positions(native);
    let mut selected = selected;
    if range.fill == RangeFill::Bottom {
        let mut used = segment_tokens(&selected, resolver);
        while used > reserved && selected.len() > 1 {
            used -= segment_tokens(&selected[..1], resolver);
            selected.remove(0);
        }
    }
    let used = segment_tokens(&selected, resolver);
    if used > reserved {
        tracing::warn!(
            collection = %coll.name,
            used,
            reserved,
            "a range's members take more than the range: everything after it moves"
        );
    }
    let hole = ProjectionSegment::Hole {
        positions: range.hole(native, used),
    };
    match range.fill {
        RangeFill::Top => {
            out.push(hole);
            out.extend(selected);
        }
        RangeFill::Bottom => {
            out.extend(selected);
            out.push(hole);
        }
    }
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
///
/// **A section with no content emits nothing.** Ingest seals what a section's
/// content tokenises to and skips one with no content (`insert_section_with_prefix`
/// in `conversation.rs`), so such a section never has K/V. A
/// segment naming it is one no walk can place: the assembler counts it skipped,
/// the projection is never recorded as complete, and every reprojection rebuilds
/// the slot — re-sealing the whole active turn, ~330 ms at 128K. The same test
/// as ingest's, on the same text, so the two agree on which sections exist.
pub(super) fn push_section_segment(out: &mut Vec<ProjectionSegment>, s: &SectionSchema) {
    if s.content.is_empty() {
        return;
    }
    out.push(ProjectionSegment::Sealed(SealedKind::Section(
        ResolvedSection::new(s.id),
    )));
}

/// Emit the collection's `member_glue` (a live-prefilled structural token, e.g. a
/// newline) into `out` BEFORE the next member — but never before the first one.
/// The glue is independent of which members provenance selected: it is not baked
/// into any member's seal, so a dropped member never takes its separator with it.
pub(super) fn push_member_glue(out: &mut Vec<ProjectionSegment>, coll: &SectionCollection) {
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
