//! Score-density selection (§8.4) — the projection step-9 algorithm
//! that fills the layer's token budget with the highest-scoring subset
//! of the summary tree's nodes, then eliminates redundant ancestors and
//! fills coverage gaps until convergence.
//!
//! # The five steps
//!
//! 1. **Compute effective score per node** —
//!    `effective_score(node) = max(provenance_score, recency_score)`.
//! 2. **Greedy fit by score, descending** — add nodes to the selected
//!    set as long as budget allows.
//! 3. **Eliminate redundant ancestors** — bottom-up, drop any node
//!    whose entire subtree is already covered by selected descendants.
//! 4. **Fill coverage gaps largest first** — for each maximal run of
//!    uncovered Normal sub-leaves, add the smallest tree node whose
//!    subtree covers that run.
//! 5. **Multi-pass refill until convergence** — add the highest-score
//!    non-selected nodes that fit, then re-eliminate; stop when a full
//!    pass adds nothing.
//!
//! The output is a [`Selection`] holding the chronologically-ordered
//! selected nodes plus per-node origin tags (consumed by
//! [`SelectionDiagnostics`](super::SelectionDiagnostics)).

use ahash::{AHashMap, AHashSet};

use super::diagnostics::SelectionOrigin;
use super::recency::{recency_score, RecencyConfig};
use super::tree::{NodeId, SummaryTree, TurnKind};

/// Result of `select_dense`.  The `selected` list is in **chronological
/// order** (older content first, recent last) so it can be injected
/// into the slot directly; the `origins` align index-by-index with
/// `selected`.
#[derive(Debug, Clone, Default)]
pub struct Selection {
    pub selected: Vec<NodeId>,
    pub origins: Vec<SelectionOrigin>,
    pub effective_scores: AHashMap<NodeId, f32>,
    pub used_tokens: u32,
}

impl Selection {
    pub fn contains(&self, id: NodeId) -> bool {
        self.selected.contains(&id)
    }
}

/// Score-density selection over the entire summary tree.
///
/// `provenance_scores` is the per-node Q-agreement score produced by
/// the BDP scan; missing nodes default to `0.0`.  `recency_cfg`
/// controls the hard-anchor + decay (see [`RecencyConfig`]).  `budget`
/// is the layer's `window` in tokens.
pub fn select_dense(
    tree: &SummaryTree,
    provenance_scores: &AHashMap<NodeId, f32>,
    recency_cfg: RecencyConfig,
    budget: u32,
) -> Selection {
    let mut sel = Selection::default();
    if tree.is_empty() {
        return sel;
    }

    // Step 1 — effective score per node.
    let mut effective: AHashMap<NodeId, f32> = AHashMap::default();
    for id in tree.all_ids() {
        let prov = provenance_scores.get(&id).copied().unwrap_or(0.0);
        let rec = recency_score(id, tree.chrono_leaves(), recency_cfg);
        let eff = if prov >= rec { prov } else { rec };
        effective.insert(id, eff);
    }

    // For convenience: a list of (id, score) sorted by score descending,
    // with deterministic tie-breaking by id ascending (so the same
    // scores produce the same selection across runs).
    let mut ranked: Vec<(NodeId, f32)> = effective.iter().map(|(k, v)| (*k, *v)).collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    // Step 2 — greedy fit.  Build `selected_set` and origin map in
    // parallel; we re-order to chronological at the end.
    let mut selected_set: AHashSet<NodeId> = AHashSet::default();
    let mut origin_map: AHashMap<NodeId, SelectionOrigin> = AHashMap::default();
    let mut used: u32 = 0;
    let prov_threshold = |id: NodeId, prov_map: &AHashMap<NodeId, f32>| -> f32 {
        prov_map.get(&id).copied().unwrap_or(0.0)
    };
    for (id, eff_score) in &ranked {
        if !eff_score.is_finite() || *eff_score > 0.0 {
            // Skip; finite check below.  (Both INFINITY and positive
            // finite scores qualify; only 0.0 / NaN / negative don't.)
        }
        if !(eff_score.is_finite() && *eff_score > 0.0 || eff_score.is_infinite()) {
            continue;
        }
        let node = match tree.get(*id) {
            Some(n) => n,
            None => continue,
        };
        if used.saturating_add(node.tokens) > budget {
            continue;
        }
        used = used.saturating_add(node.tokens);
        selected_set.insert(*id);
        let rec = recency_score(*id, tree.chrono_leaves(), recency_cfg);
        let prov = prov_threshold(*id, provenance_scores);
        let origin = if rec.is_infinite() {
            SelectionOrigin::HardAnchor
        } else if rec > prov {
            SelectionOrigin::RecencyDecay
        } else {
            SelectionOrigin::ProvenanceScore
        };
        origin_map.insert(*id, origin);
    }

    // Build the parent map once — used by `covered` and the coverage
    // gap analysis.
    let parents = build_parent_map(tree);
    let normal_to_leaf = build_normal_to_leaf_map(tree);

    // Step 3 — eliminate redundant ancestors.
    eliminate_redundant(tree, &mut selected_set, &mut used);

    // Step 4 — fill coverage gaps largest first.
    let gaps = uncovered_ranges(tree, &selected_set, &parents);
    let mut gap_queue: Vec<Gap> = gaps;
    gap_queue.sort_by(|a, b| b.normals.len().cmp(&a.normals.len()));
    for gap in gap_queue {
        // Smallest covering node = LCA of the gap's Normal sub-leaves
        // (mapped through normal_to_leaf to their SoT leaves).
        let covering = lca_of_normals(tree, &gap.normals, &parents, &normal_to_leaf);
        let covering = match covering {
            Some(c) => c,
            None => continue,
        };
        if selected_set.contains(&covering) {
            continue;
        }
        let cover_tokens = tree.get(covering).map(|n| n.tokens).unwrap_or(0);
        if used.saturating_add(cover_tokens) > budget {
            // Per §8.4: covering-node size grows monotonically with gap
            // depth, so once one gap's cover is unaffordable, every
            // larger gap's cover is also unaffordable.  Break out.
            //
            // Caveat: our priority order is largest-first, so a smaller
            // later gap might have a cheaper cover.  Stay safe and
            // continue rather than break — the cost of an extra
            // iteration is O(gaps), negligible.
            continue;
        }
        used = used.saturating_add(cover_tokens);
        selected_set.insert(covering);
        origin_map.insert(covering, SelectionOrigin::CoverageFill);
    }

    // Step 5 — multi-pass refill until convergence.
    loop {
        let mut added_any = false;
        for (id, eff_score) in &ranked {
            if selected_set.contains(id) {
                continue;
            }
            if !(eff_score.is_finite() && *eff_score > 0.0 || eff_score.is_infinite()) {
                continue;
            }
            let node = match tree.get(*id) {
                Some(n) => n,
                None => continue,
            };
            if used.saturating_add(node.tokens) > budget {
                continue;
            }
            used = used.saturating_add(node.tokens);
            selected_set.insert(*id);
            origin_map.insert(*id, SelectionOrigin::Refill);
            added_any = true;
        }
        eliminate_redundant(tree, &mut selected_set, &mut used);
        if !added_any {
            break;
        }
    }

    // Assemble Selection in chronological order — walk chrono_normals
    // and chrono_leaves; place internal SoS nodes via post-order at
    // each Normal-cluster boundary so the result is readable.
    let ordered = chronological_order(tree, &selected_set);
    for id in ordered {
        let origin = origin_map.remove(&id).unwrap_or(SelectionOrigin::Refill);
        let eff = effective.get(&id).copied().unwrap_or(0.0);
        sel.selected.push(id);
        sel.origins.push(origin);
        sel.effective_scores.insert(id, eff);
    }
    sel.used_tokens = used;
    sel
}

/// Bottom-up redundancy elimination: drop any node from `selected`
/// whose every child is already covered by `selected`.
///
/// Applies to both `SummaryOfTurns` (covered iff every Normal child ∈
/// covered) and `SummaryOfSummaries` (covered iff every of the 2
/// summary children ∈ covered).  Normal turns have no children and are
/// never themselves redundant.
pub fn eliminate_redundant(
    tree: &SummaryTree,
    selected: &mut AHashSet<NodeId>,
    used: &mut u32,
) {
    let order = tree.post_order();
    for id in order {
        if !selected.contains(&id) {
            continue;
        }
        let node = match tree.get(id) {
            Some(n) => n,
            None => continue,
        };
        if !node.has_children() {
            continue;
        }
        if node.children.iter().all(|c| covered(tree, *c, selected)) {
            selected.remove(&id);
            *used = used.saturating_sub(node.tokens);
        }
    }
}

/// True iff `node` is itself in `selected`, or every one of its
/// children is `covered` (recursive — terminates at Normal sub-leaves).
pub fn covered(tree: &SummaryTree, node: NodeId, selected: &AHashSet<NodeId>) -> bool {
    if selected.contains(&node) {
        return true;
    }
    let n = match tree.get(node) {
        Some(n) => n,
        None => return false,
    };
    if !n.has_children() {
        return false;
    }
    n.children
        .iter()
        .all(|c| covered(tree, *c, selected))
}

/// Maximal contiguous runs of uncovered Normal sub-leaves.
#[derive(Debug, Clone)]
struct Gap {
    /// The uncovered Normal IDs in this gap, in chronological order.
    normals: Vec<NodeId>,
}

/// Walk `chrono_normals` and group consecutive uncovered Normals into
/// gaps.  A Normal is "uncovered" iff neither it nor any ancestor SoT
/// / SoS is in `selected`.
fn uncovered_ranges(
    tree: &SummaryTree,
    selected: &AHashSet<NodeId>,
    parents: &AHashMap<NodeId, NodeId>,
) -> Vec<Gap> {
    let mut gaps: Vec<Gap> = Vec::new();
    let mut current: Vec<NodeId> = Vec::new();
    for normal in tree.chrono_normals() {
        if normal_is_covered(*normal, selected, parents) {
            if !current.is_empty() {
                gaps.push(Gap {
                    normals: std::mem::take(&mut current),
                });
            }
        } else {
            current.push(*normal);
        }
    }
    if !current.is_empty() {
        gaps.push(Gap { normals: current });
    }
    gaps
}

/// True iff the Normal turn `id` or any of its ancestors in the binary
/// tree is in `selected`.
fn normal_is_covered(
    id: NodeId,
    selected: &AHashSet<NodeId>,
    parents: &AHashMap<NodeId, NodeId>,
) -> bool {
    if selected.contains(&id) {
        return true;
    }
    let mut cur = parents.get(&id).copied();
    while let Some(p) = cur {
        if selected.contains(&p) {
            return true;
        }
        cur = parents.get(&p).copied();
    }
    false
}

/// Lowest common ancestor in the binary tree of a list of Normal IDs.
/// Returns None for empty input.  Walks up from each Normal until all
/// converge.
fn lca_of_normals(
    _tree: &SummaryTree,
    normals: &[NodeId],
    parents: &AHashMap<NodeId, NodeId>,
    normal_to_leaf: &AHashMap<NodeId, NodeId>,
) -> Option<NodeId> {
    if normals.is_empty() {
        return None;
    }
    // Start each walker at the Normal's containing SoT leaf — Normals
    // themselves aren't in the binary AVL spine.
    let mut leaves: Vec<NodeId> = normals
        .iter()
        .filter_map(|n| normal_to_leaf.get(n).copied())
        .collect();
    leaves.sort();
    leaves.dedup();
    if leaves.is_empty() {
        return None;
    }
    if leaves.len() == 1 {
        // Single SoT leaf covers the whole gap.  If every Normal in the
        // gap is one of that leaf's children, the smallest covering
        // node is the leaf itself.  Otherwise (which shouldn't happen
        // given a single-leaf gap), fall back to the leaf.
        return Some(leaves[0]);
    }
    // For multi-leaf gaps: walk up from each leaf, collect ancestors,
    // pick the lowest common one.
    let ancestors_of = |start: NodeId| -> Vec<NodeId> {
        let mut chain = vec![start];
        let mut cur = parents.get(&start).copied();
        while let Some(p) = cur {
            chain.push(p);
            cur = parents.get(&p).copied();
        }
        chain
    };
    let first_chain = ancestors_of(leaves[0]);
    let first_set: AHashSet<NodeId> = first_chain.iter().copied().collect();
    // For each other leaf, walk up — the first ancestor that's in
    // first_set is a candidate LCA.  The lowest of those (highest
    // tree_height that's still common) is the answer.  Simpler: take
    // the intersection-by-depth and pick the deepest.
    let mut candidates: Vec<NodeId> = first_chain.clone();
    for leaf in &leaves[1..] {
        let chain = ancestors_of(*leaf);
        let chain_set: AHashSet<NodeId> = chain.iter().copied().collect();
        candidates.retain(|c| chain_set.contains(c));
        if candidates.is_empty() {
            // Shouldn't happen for a well-formed tree (root is common).
            break;
        }
        // No need to use first_set after the first iteration.
        let _ = &first_set;
    }
    // The deepest common ancestor is the last one in any walker's
    // chain that's in `candidates`.  Walker chains are leaf-first, so
    // the *first* element of `candidates` that appears in
    // `first_chain` (ordered leaf-first) is the deepest.
    for a in &first_chain {
        if candidates.contains(a) {
            return Some(*a);
        }
    }
    None
}

/// Build a child → parent map by walking the tree from the root.
fn build_parent_map(tree: &SummaryTree) -> AHashMap<NodeId, NodeId> {
    let mut map: AHashMap<NodeId, NodeId> = AHashMap::default();
    if let Some(root) = tree.root() {
        walk_parents(tree, root, &mut map);
    }
    map
}

fn walk_parents(tree: &SummaryTree, id: NodeId, out: &mut AHashMap<NodeId, NodeId>) {
    let node = match tree.get(id) {
        Some(n) => n,
        None => return,
    };
    for child in &node.children {
        out.insert(*child, id);
        walk_parents(tree, *child, out);
    }
}

/// Build a Normal → SoT-leaf-parent map.
fn build_normal_to_leaf_map(tree: &SummaryTree) -> AHashMap<NodeId, NodeId> {
    let mut map: AHashMap<NodeId, NodeId> = AHashMap::default();
    for leaf_id in tree.chrono_leaves() {
        if let Some(leaf) = tree.get(*leaf_id) {
            if leaf.kind == TurnKind::SummaryOfTurns {
                for normal_id in &leaf.children {
                    map.insert(*normal_id, *leaf_id);
                }
            }
        }
    }
    map
}

/// Reorder `selected_set` into chronological order suitable for slot
/// injection.  Normal turns are placed at their position in
/// `chrono_normals`; SoT leaves are placed at their position in
/// `chrono_leaves` (which equals the position of their first Normal
/// child); SoS internals are interleaved by post-order index of their
/// rightmost Normal-leaf descendant.
fn chronological_order(tree: &SummaryTree, selected_set: &AHashSet<NodeId>) -> Vec<NodeId> {
    // Assign each node a chronological key = position of its leftmost
    // Normal-descendant in chrono_normals.  Normal turns get their own
    // chrono index; SoT leaves get the index of their first Normal
    // child (or, if empty, the position in chrono_leaves); SoS gets
    // the min over its descendants.
    let normal_pos: AHashMap<NodeId, usize> = tree
        .chrono_normals()
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, i))
        .collect();

    let chrono_key = |id: NodeId, tree: &SummaryTree| -> (usize, u8) {
        // Return (leftmost normal pos, tie-break priority).  Tie-break:
        // for the same leftmost normal, place Normal first, then SoT,
        // then SoS — so that the slot reads small-to-large.
        let node = match tree.get(id) {
            Some(n) => n,
            None => return (usize::MAX, 3),
        };
        let kind_rank = match node.kind {
            TurnKind::Normal => 0,
            TurnKind::SummaryOfTurns => 1,
            TurnKind::SummaryOfSummaries => 2,
        };
        let pos = match node.kind {
            TurnKind::Normal => *normal_pos.get(&id).unwrap_or(&0),
            TurnKind::SummaryOfTurns => node
                .children
                .first()
                .and_then(|c| normal_pos.get(c).copied())
                .unwrap_or(usize::MAX / 2),
            TurnKind::SummaryOfSummaries => leftmost_normal_pos(tree, id, &normal_pos),
        };
        (pos, kind_rank)
    };

    let mut ordered: Vec<NodeId> = selected_set.iter().copied().collect();
    ordered.sort_by(|a, b| {
        let ka = chrono_key(*a, tree);
        let kb = chrono_key(*b, tree);
        ka.cmp(&kb)
    });
    ordered
}

fn leftmost_normal_pos(
    tree: &SummaryTree,
    id: NodeId,
    normal_pos: &AHashMap<NodeId, usize>,
) -> usize {
    let node = match tree.get(id) {
        Some(n) => n,
        None => return usize::MAX,
    };
    match node.kind {
        TurnKind::Normal => *normal_pos.get(&id).unwrap_or(&usize::MAX),
        TurnKind::SummaryOfTurns => node
            .children
            .first()
            .and_then(|c| normal_pos.get(c).copied())
            .unwrap_or(usize::MAX),
        TurnKind::SummaryOfSummaries => node
            .children
            .iter()
            .map(|c| leftmost_normal_pos(tree, *c, normal_pos))
            .min()
            .unwrap_or(usize::MAX),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::tree::Node;
    use super::*;

    /// Build a tree where every Normal turn is `tokens_per_normal`
    /// tokens and every SoT leaf has `normals_per_leaf` Normal sub-
    /// leaves, with `n_leaves` total binary leaves.  Returns the tree
    /// and a chrono-ordered list of Normal IDs.
    fn build_uniform_tree(
        n_leaves: u32,
        normals_per_leaf: u32,
        tokens_per_normal: u32,
        tokens_per_summary: u32,
    ) -> (SummaryTree, Vec<NodeId>) {
        let mut tree = SummaryTree::new();
        let mut all_normals = Vec::new();
        let mut next_normal_id = 1000u32;
        for leaf_id in 1..=n_leaves {
            // Seed the leaf with its Normal children up front so the
            // tree's chrono_normals stays in step with what we return.
            let mut normals = Vec::new();
            for _ in 0..normals_per_leaf {
                normals.push(NodeId(next_normal_id));
                next_normal_id += 1;
            }
            // Insert the SoT leaf with its normals.  But the
            // `insert_leaf_rightmost` API expects the normals to
            // already be present in the tree (or it'll skip
            // registering them).  Insert them as Normal nodes first.
            for n in &normals {
                tree.insert_node(Node::normal(*n, tokens_per_normal));
            }
            let leaf = Node::summary_of_turns(
                NodeId(leaf_id),
                normals.clone(),
                tokens_per_summary,
            );
            tree.insert_leaf_rightmost(leaf);
            all_normals.extend(normals);
        }
        (tree, all_normals)
    }

    #[test]
    fn empty_tree_returns_empty_selection() {
        let tree = SummaryTree::new();
        let scores: AHashMap<NodeId, f32> = AHashMap::default();
        let sel = select_dense(&tree, &scores, RecencyConfig::default(), 1000);
        assert!(sel.selected.is_empty());
        assert_eq!(sel.used_tokens, 0);
    }

    #[test]
    fn small_tree_anchor_only() {
        // 1 leaf, 1 normal, tiny budget — only the hard-anchored leaf
        // fits.
        let (tree, _) = build_uniform_tree(1, 1, 10, 20);
        let scores = AHashMap::default();
        let sel = select_dense(&tree, &scores, RecencyConfig::default(), 50);
        // Hard anchor includes the rightmost (only) leaf.
        assert!(sel.contains(NodeId(1)));
        assert!(sel.origins.contains(&SelectionOrigin::HardAnchor));
    }

    #[test]
    fn hard_anchor_three_leaves_always_included() {
        // 5 leaves, no provenance.  Last 3 are hard-anchored.
        let (tree, _) = build_uniform_tree(5, 1, 10, 20);
        let scores = AHashMap::default();
        let sel = select_dense(&tree, &scores, RecencyConfig::default(), 10_000);
        assert!(sel.contains(NodeId(3)));
        assert!(sel.contains(NodeId(4)));
        assert!(sel.contains(NodeId(5)));
    }

    #[test]
    fn high_provenance_old_turn_beats_decayed_recent() {
        // 10 leaves.  Plant a huge provenance score on leaf 1 (oldest).
        // It must end up in selected even though it's well outside the
        // hard anchor.
        let (tree, _) = build_uniform_tree(10, 1, 10, 20);
        let mut scores = AHashMap::default();
        scores.insert(NodeId(1), 100.0); // huge
        let sel = select_dense(&tree, &scores, RecencyConfig::default(), 10_000);
        assert!(sel.contains(NodeId(1)));
        assert_eq!(
            sel.origins
                [sel.selected.iter().position(|n| *n == NodeId(1)).unwrap()],
            SelectionOrigin::ProvenanceScore,
            "leaf 1 must come in via provenance, not recency",
        );
    }

    #[test]
    fn budget_starves_low_score_nodes() {
        // 10 leaves, budget for ~3 summary turns + some buffer.  Only
        // hard-anchor 3 + maybe 1 more fit.
        let (tree, _) = build_uniform_tree(10, 1, 10, 20);
        let scores = AHashMap::default();
        // Budget: tight — fit 3 SoT leaves (60 tokens) + 3 Normals
        // (30 tokens) + maybe an SoS or two.
        let sel = select_dense(&tree, &scores, RecencyConfig::default(), 100);
        assert!(sel.used_tokens <= 100);
        // The hard anchor 3 are nonnegotiable; they must all fit at
        // budget = 100 (3 leaves * 20 tokens = 60).
        assert!(sel.contains(NodeId(8)));
        assert!(sel.contains(NodeId(9)));
        assert!(sel.contains(NodeId(10)));
    }

    #[test]
    fn redundancy_eliminates_ancestor_when_all_children_selected() {
        // Build a tree, hand-curate `selected` with both children of
        // an internal node, then call eliminate_redundant directly.
        let (tree, _) = build_uniform_tree(2, 1, 10, 20);
        let mut sel: AHashSet<NodeId> = AHashSet::default();
        sel.insert(NodeId(1));
        sel.insert(NodeId(2));
        // The auto-generated SoS internal has id 2^31.
        let internal = NodeId(1u32 << 31);
        sel.insert(internal);
        // 20 (leaf 1) + 20 (leaf 2) + 20 (internal, DEFAULT_INTERNAL_TOKENS).
        let mut used = 60;
        eliminate_redundant(&tree, &mut sel, &mut used);
        assert!(!sel.contains(&internal));
        assert!(sel.contains(&NodeId(1)));
        assert!(sel.contains(&NodeId(2)));
        // The dropped internal returns its 20 tokens to the budget.
        assert_eq!(used, 40);
    }

    #[test]
    fn redundancy_keeps_ancestor_when_one_child_uncovered() {
        // SoS with two SoT leaves; only one leaf selected.  Ancestor
        // must stay if it's selected (it's the only way to cover the
        // other side).
        let (tree, _) = build_uniform_tree(2, 1, 10, 20);
        let mut sel: AHashSet<NodeId> = AHashSet::default();
        sel.insert(NodeId(1));
        let internal = NodeId(1u32 << 31);
        sel.insert(internal);
        let mut used = 40;
        eliminate_redundant(&tree, &mut sel, &mut used);
        assert!(sel.contains(&internal), "SoS must stay when right child uncovered");
        assert!(sel.contains(&NodeId(1)));
    }

    #[test]
    fn covered_is_subtree_semantics_not_ancestor_walk() {
        // `covered(node, sel)` from §8.4 means: every leaf descendant
        // of `node` is in `sel` (or `node` itself is).  It does NOT
        // walk upward.  This is the redundancy-elimination semantics.
        let (tree, _) = build_uniform_tree(2, 2, 10, 20);
        let internal = NodeId(1u32 << 31);

        // SoS in sel → trivially covered.
        let mut sel_internal = AHashSet::default();
        sel_internal.insert(internal);
        assert!(covered(&tree, internal, &sel_internal));

        // SoT leaf with NEITHER it nor its Normals in sel → NOT covered
        // (its only ancestor is in sel, but that's not what `covered`
        // asks).
        assert!(!covered(&tree, NodeId(1), &sel_internal));

        // Both SoT leaves in sel → SoS is covered (its 2 binary children
        // are both in sel).
        let mut sel_leaves = AHashSet::default();
        sel_leaves.insert(NodeId(1));
        sel_leaves.insert(NodeId(2));
        assert!(covered(&tree, internal, &sel_leaves));
        // ...and each leaf is covered (in sel).
        assert!(covered(&tree, NodeId(1), &sel_leaves));

        // All Normals of leaf 1 in sel → leaf 1 covered.
        let mut sel_norms = AHashSet::default();
        sel_norms.insert(NodeId(1000));
        sel_norms.insert(NodeId(1001));
        assert!(covered(&tree, NodeId(1), &sel_norms));
        // ...but leaf 2 is not, because its Normals aren't in sel.
        assert!(!covered(&tree, NodeId(2), &sel_norms));

        // Normal sub-leaf only covered if it itself is in sel (it has
        // no children).
        let mut sel_one = AHashSet::default();
        sel_one.insert(NodeId(1000));
        assert!(covered(&tree, NodeId(1000), &sel_one));
        assert!(!covered(&tree, NodeId(1001), &sel_one));

        // Empty selection covers nothing.
        let empty = AHashSet::default();
        assert!(!covered(&tree, NodeId(1), &empty));
        assert!(!covered(&tree, NodeId(1000), &empty));
    }

    #[test]
    fn coverage_gap_fill_keeps_every_normal_covered() {
        // 5 leaves, ample budget.  Every Normal in the timeline must
        // have either itself OR some ancestor in the selection.
        // "Coverage" here is the upward semantics (selected ancestor
        // exists), NOT `covered()` (which walks downward).
        let (tree, _) = build_uniform_tree(5, 1, 10, 20);
        let scores = AHashMap::default();
        let sel = select_dense(&tree, &scores, RecencyConfig::default(), 10_000);
        let parents = build_parent_map(&tree);
        let sel_set: AHashSet<NodeId> = sel.selected.iter().copied().collect();
        for normal in tree.chrono_normals() {
            assert!(
                normal_is_covered(*normal, &sel_set, &parents),
                "Normal {} must have some selected ancestor",
                normal
            );
        }
    }

    #[test]
    fn lca_of_single_leaf_normal_returns_that_leaf() {
        let (tree, normals) = build_uniform_tree(3, 2, 10, 20);
        let parents = build_parent_map(&tree);
        let map = build_normal_to_leaf_map(&tree);
        // First normal: should be in leaf 1.
        let lca = lca_of_normals(&tree, &[normals[0]], &parents, &map);
        assert_eq!(lca, Some(NodeId(1)));
    }

    #[test]
    fn lca_of_normals_across_leaves_finds_common_ancestor() {
        let (tree, normals) = build_uniform_tree(4, 1, 10, 20);
        let parents = build_parent_map(&tree);
        let map = build_normal_to_leaf_map(&tree);
        // Normals 0 and 3 span all 4 leaves → LCA = root.
        let lca = lca_of_normals(
            &tree,
            &[normals[0], normals[3]],
            &parents,
            &map,
        );
        assert_eq!(lca, tree.root());
    }

    #[test]
    fn chronological_order_oldest_first() {
        let (tree, _) = build_uniform_tree(4, 1, 10, 20);
        let scores = AHashMap::default();
        let sel = select_dense(&tree, &scores, RecencyConfig::default(), 10_000);
        // The leaves are in chronological order if present.
        let leaf_indices: Vec<usize> = sel
            .selected
            .iter()
            .filter_map(|id| match tree.get(*id).map(|n| n.kind) {
                Some(TurnKind::SummaryOfTurns) => Some(id.0 as usize),
                _ => None,
            })
            .collect();
        let mut sorted = leaf_indices.clone();
        sorted.sort();
        assert_eq!(leaf_indices, sorted, "leaves must come out chronologically");
    }

    #[test]
    fn deterministic_under_ties() {
        // Build a tree where many nodes have score 0.0 (recency
        // dominates) — the selection must be deterministic across
        // repeated calls.
        let (tree, _) = build_uniform_tree(8, 1, 10, 20);
        let scores = AHashMap::default();
        let cfg = RecencyConfig::default();
        let a = select_dense(&tree, &scores, cfg, 200);
        let b = select_dense(&tree, &scores, cfg, 200);
        assert_eq!(a.selected, b.selected);
        assert_eq!(a.origins, b.origins);
    }

    #[test]
    fn multi_pass_refill_converges() {
        // 16 leaves, budget tight enough that the algorithm has to
        // make 2+ passes.  Verify it terminates and produces a valid
        // result.
        let (tree, _) = build_uniform_tree(16, 2, 8, 20);
        let mut scores = AHashMap::default();
        // Plant a few high-score middle nodes.
        scores.insert(NodeId(5), 5.0);
        scores.insert(NodeId(8), 4.0);
        let sel = select_dense(&tree, &scores, RecencyConfig::default(), 200);
        assert!(sel.used_tokens <= 200);
        // Both planted high-score nodes should be in (budget is loose
        // enough).
        assert!(sel.contains(NodeId(5)));
        assert!(sel.contains(NodeId(8)));
        // Hard anchors must always be in.
        assert!(sel.contains(NodeId(14)));
        assert!(sel.contains(NodeId(15)));
        assert!(sel.contains(NodeId(16)));
    }

    #[test]
    fn at_saturation_root_only_pattern() {
        // Budget = exactly one summary turn.  The selection collapses
        // to the root (or the most recent hard anchor if that fits
        // first).  The "default coverage by root" property (§8.6).
        let (tree, _) = build_uniform_tree(8, 1, 8, 20);
        let scores = AHashMap::default();
        let sel = select_dense(&tree, &scores, RecencyConfig::default(), 20);
        assert!(sel.used_tokens <= 20);
        // Selected must be exactly 1 node.
        assert_eq!(sel.selected.len(), 1);
    }
}
