//! Budget-fitted conversation selection — the projection step-9
//! algorithm that chooses which summary-tree nodes fill a layer's token
//! window.
//!
//! # Top-down refinement
//!
//! The selector starts from the **coarsest full coverage** — the forest
//! peaks (roots) — and then *refines* the most-recent frontier node into
//! its children (finer summaries, and ultimately the raw turns
//! themselves) for as long as the extra tokens fit the budget.  Older
//! content is left coarse.
//!
//! ```text
//!   start:   [ ─────────── peak (SoS) ─────────── ]     coarse, covers all
//!   refine:  [ SoS ][ SoS ][ SoT ][ turn ][ turn ]      recent side opened
//!            └ older, still summarised ┘ └ recent, raw ┘
//! ```
//!
//! Two properties fall out of the shape, not out of careful tuning:
//!
//! * **It cannot livelock.**  Every node is pushed onto the frontier at
//!   most once and popped at most once; refining only ever pushes
//!   *children*.  The loop therefore drains in at most `tree.len()`
//!   iterations — there is no add/re-add cycle to spin on.
//! * **It cannot over-compress.**  Coverage begins whole and refinement
//!   only ever makes it *finer*, so a raw turn that fits the budget is
//!   always reached (never left standing as its lossy summary).  When the
//!   whole conversation fits the window every turn is shown raw and no
//!   summary is emitted at all.
//!
//! # The `floor` — hybrid of budget and rules
//!
//! The `budget` alone decides how far back turns stay raw, so under a tight
//! window even the newest turn can collapse into a summary.  The `floor`
//! parameter is the rule-based guarantee grafted on top: the newest `floor`
//! turns are refined to raw **unconditionally**, even when that pushes past
//! `budget`.  This is the group's `recent: N` — you never summarise the user's
//! most recent exchange.  `floor == 0` is pure budget-fit.
//!
//! The output is a [`Selection`] holding the chronologically-ordered
//! selected nodes plus per-node origin tags (consumed by
//! [`SelectionDiagnostics`](super::SelectionDiagnostics)).

use std::collections::BinaryHeap;

use ahash::{AHashMap, AHashSet};

use super::diagnostics::SelectionOrigin;
use super::tree::{NodeId, SummaryTree, TurnKind};

/// Result of [`select_budget_fit`].  The `selected` list is in
/// **chronological order** (older content first, recent last) so it can
/// be injected into the slot directly; the `origins` align index-by-index
/// with `selected`.
#[derive(Debug, Clone, Default)]
pub struct Selection {
    pub selected: Vec<NodeId>,
    pub origins: Vec<SelectionOrigin>,
    pub used_tokens: u32,
}

impl Selection {
    pub fn contains(&self, id: NodeId) -> bool {
        self.selected.contains(&id)
    }
}

/// Budget-fitted selection over the whole summary tree (see the module
/// doc for the algorithm).  `budget` is the layer's `window` in tokens;
/// `floor` is the group's `recent` — the newest `floor` turns are kept raw
/// unconditionally, even past `budget`.
pub fn select_budget_fit(tree: &SummaryTree, budget: u32, floor: usize) -> Selection {
    let mut sel = Selection::default();
    if tree.is_empty() {
        return sel;
    }

    // Chronological index of each Normal turn; used to rank frontier
    // nodes by recency (a node's key is its newest Normal descendant).
    let normal_pos: AHashMap<NodeId, usize> = tree
        .chrono_normals()
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, i))
        .collect();
    let n_normals = tree.chrono_normals().len();
    // Normal positions `>= floor_start` are the guaranteed-raw newest turns.
    let floor_start = n_normals.saturating_sub(floor);
    // A node must be opened regardless of budget when it still covers a
    // guaranteed-raw turn — i.e. its newest Normal descendant is in the floor.
    let covers_floor =
        |id: NodeId| floor > 0 && rightmost_normal_pos(tree, id, &normal_pos) >= floor_start;

    let mut selected: AHashSet<NodeId> = AHashSet::default();
    let mut used: u32 = 0;
    // Max-heap of `(recency_key, id.0)` over the current frontier — pop
    // returns the node whose newest content is most recent, with a
    // deterministic id tie-break.
    let mut frontier: BinaryHeap<(usize, u32)> = BinaryHeap::new();

    // Baseline: the coarsest full coverage is the set of forest ROOTS — every
    // parentless node.  `tree.peaks()` returns only the orphan SUMMARY roots
    // (it also drives forest construction, which must not fold bare turns into
    // an SoS), but the recent turns the async summariser hasn't absorbed yet are
    // orphan NORMALS with no covering summary.  They must seed the frontier too,
    // or the newest context silently vanishes.  Add newest-first, dropping the
    // oldest that don't fit; a root covering a floor turn is always added.
    let mut claimed: AHashSet<NodeId> = AHashSet::default();
    for id in tree.all_ids() {
        if let Some(n) = tree.get(id) {
            for c in &n.children {
                claimed.insert(*c);
            }
        }
    }
    let mut peaks: Vec<NodeId> = tree.all_ids().filter(|id| !claimed.contains(id)).collect();
    peaks.sort_by_key(|p| std::cmp::Reverse(rightmost_normal_pos(tree, *p, &normal_pos)));
    for peak in peaks {
        let tokens = tree.get(peak).map_or(0, |n| n.tokens);
        if used.saturating_add(tokens) > budget && !covers_floor(peak) {
            continue;
        }
        if selected.insert(peak) {
            used = used.saturating_add(tokens);
            frontier.push((rightmost_normal_pos(tree, peak, &normal_pos), peak.0));
        }
    }

    // Refine the most-recent frontier node into its children while the
    // extra tokens fit.  A Normal turn has no children — it's already as
    // fine as it gets, so it just leaves the frontier.  Floor nodes (which
    // sort first, being newest) refine unconditionally.
    while let Some((_, id_raw)) = frontier.pop() {
        let id = NodeId(id_raw);
        let node = match tree.get(id) {
            Some(n) => n,
            None => continue,
        };
        if !node.has_children() {
            continue;
        }
        let child_tokens: u32 = node
            .children
            .iter()
            .map(|c| tree.get(*c).map_or(0, |n| n.tokens))
            .sum();
        // Refining swaps this node for its children: -node.tokens, +child_tokens.
        let after = used
            .saturating_sub(node.tokens)
            .saturating_add(child_tokens);
        if after > budget && !covers_floor(id) {
            // Can't afford to open the most-recent unopened node, and it's not
            // required by the floor.  STOP rather than skip ahead to a cheaper
            // older node: that keeps the raw turns a contiguous *recent suffix*
            // (newest raw, older summarised) instead of a ragged mix, which is
            // what the panel and the decode context want.
            break;
        }
        used = after;
        selected.remove(&id);
        for c in &node.children {
            if selected.insert(*c) {
                frontier.push((rightmost_normal_pos(tree, *c, &normal_pos), c.0));
            }
        }
    }

    // Assemble in chronological order (summaries above the content they
    // cover) and tag each node's origin.  Floor turns are `HardAnchor`;
    // older raw turns `RecencyDecay`.
    let ordered = chronological_order(tree, &selected);
    for id in ordered {
        let origin = match tree.get(id).map(|n| n.kind) {
            Some(TurnKind::Normal) => {
                let pos = normal_pos.get(&id).copied().unwrap_or(0);
                if floor > 0 && pos >= floor_start {
                    SelectionOrigin::HardAnchor
                } else {
                    SelectionOrigin::RecencyDecay
                }
            }
            _ => SelectionOrigin::CoverageFill,
        };
        sel.selected.push(id);
        sel.origins.push(origin);
    }
    sel.used_tokens = used;
    sel
}

/// Reorder `selected_set` into chronological order suitable for slot
/// injection.  Normal turns are placed at their position in
/// `chrono_normals`; SoT leaves at the position of their first Normal
/// child; SoS internals at the min over their descendants.  For a tied
/// leftmost position the SUMMARY sorts first so it reads ABOVE the
/// content it covers.
fn chronological_order(tree: &SummaryTree, selected_set: &AHashSet<NodeId>) -> Vec<NodeId> {
    let normal_pos: AHashMap<NodeId, usize> = tree
        .chrono_normals()
        .iter()
        .enumerate()
        .map(|(i, n)| (*n, i))
        .collect();

    let chrono_key = |id: NodeId, tree: &SummaryTree| -> (usize, u8) {
        // (leftmost normal pos, tie-break priority).  Tie-break: for the
        // same leftmost normal, place the SUMMARY first (SoS, then SoT),
        // then the Normal turn — so a summary reads ABOVE the content it
        // covers.  This is what makes partial coverage correct.
        let node = match tree.get(id) {
            Some(n) => n,
            None => return (usize::MAX, 3),
        };
        let kind_rank = match node.kind {
            TurnKind::SummaryOfSummaries => 0,
            TurnKind::SummaryOfTurns => 1,
            TurnKind::Normal => 2,
        };
        (leftmost_normal_pos(tree, id, &normal_pos), kind_rank)
    };

    let mut ordered: Vec<NodeId> = selected_set.iter().copied().collect();
    ordered.sort_by(|a, b| {
        chrono_key(*a, tree)
            .cmp(&chrono_key(*b, tree))
            .then(a.cmp(b))
    });
    ordered
}

/// Chronological index of a node's *oldest* (leftmost) Normal descendant.
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

/// Chronological index of a node's *newest* (rightmost) Normal
/// descendant — the recency key that drives refinement order.
fn rightmost_normal_pos(
    tree: &SummaryTree,
    id: NodeId,
    normal_pos: &AHashMap<NodeId, usize>,
) -> usize {
    let node = match tree.get(id) {
        Some(n) => n,
        None => return 0,
    };
    match node.kind {
        TurnKind::Normal => *normal_pos.get(&id).unwrap_or(&0),
        _ => node
            .children
            .iter()
            .map(|c| rightmost_normal_pos(tree, *c, normal_pos))
            .max()
            .unwrap_or(0),
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::tree::{Node, MERGE_FANOUT};
    use super::*;

    /// Build a tree where every Normal turn is `tokens_per_normal`
    /// tokens and every SoT leaf has `normals_per_leaf` Normal sub-leaves,
    /// with `n_leaves` total binary leaves.  Returns the tree and a
    /// chrono-ordered list of Normal IDs.
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
            let mut normals = Vec::new();
            for _ in 0..normals_per_leaf {
                normals.push(NodeId(next_normal_id));
                next_normal_id += 1;
            }
            for n in &normals {
                tree.insert_node(Node::normal(*n, tokens_per_normal));
            }
            let leaf = Node::summary_of_turns(NodeId(leaf_id), normals.clone(), tokens_per_summary);
            tree.append_leaf(leaf);
            all_normals.extend(normals);
        }
        (tree, all_normals)
    }

    /// Every Normal turn must have itself or some ancestor selected.
    fn assert_full_coverage(tree: &SummaryTree, sel: &Selection) {
        let sel_set: AHashSet<NodeId> = sel.selected.iter().copied().collect();
        for normal in tree.chrono_normals() {
            let mut cur = Some(*normal);
            let mut covered = false;
            // Walk self → ancestors looking for a selected node.
            while let Some(id) = cur {
                if sel_set.contains(&id) {
                    covered = true;
                    break;
                }
                cur = tree.all_ids().find(|p| {
                    tree.get(*p)
                        .map(|n| n.children.contains(&id))
                        .unwrap_or(false)
                });
            }
            assert!(covered, "Normal {} has no selected ancestor", normal);
        }
    }

    #[test]
    fn empty_tree_returns_empty_selection() {
        let tree = SummaryTree::new();
        let sel = select_budget_fit(&tree, 1000, 0);
        assert!(sel.selected.is_empty());
        assert_eq!(sel.used_tokens, 0);
    }

    #[test]
    fn ample_budget_refines_everything_to_raw() {
        // With plenty of room every turn is shown raw and NO summary node
        // survives — refinement drills all the way down.
        let (tree, normals) = build_uniform_tree(5, 1, 10, 20);
        let sel = select_budget_fit(&tree, 10_000, 0);
        for n in &normals {
            assert!(
                sel.contains(*n),
                "normal {} must be raw under ample budget",
                n
            );
        }
        for id in &sel.selected {
            assert_eq!(
                tree.get(*id).map(|n| n.kind),
                Some(TurnKind::Normal),
                "no summary node should survive when everything fits",
            );
        }
        assert_full_coverage(&tree, &sel);
    }

    #[test]
    fn fitting_raw_never_replaced_by_summary() {
        // The regression: two turns, room for both raw — the OLDEST must
        // stay raw, never collapse into its lossy summary.
        let (tree, normals) = build_uniform_tree(2, 1, 10, 50);
        let sel = select_budget_fit(&tree, 10_000, 0);
        assert!(sel.contains(normals[0]), "oldest turn must remain raw");
        assert!(sel.contains(normals[1]));
        assert!(
            sel.selected
                .iter()
                .all(|id| tree.get(*id).map(|n| n.kind) == Some(TurnKind::Normal)),
            "no summary should be emitted when both turns fit raw",
        );
    }

    /// Total tokens of the forest peaks — the coarsest full-coverage baseline.
    fn peak_cost(tree: &SummaryTree) -> u32 {
        tree.peaks()
            .iter()
            .map(|p| tree.get(*p).map_or(0, |n| n.tokens))
            .sum()
    }

    #[test]
    fn raw_turns_are_a_recent_contiguous_suffix() {
        // Under any budget, the turns shown raw must be the most-recent
        // contiguous run — refinement always opens newest-first, so raw
        // turns never appear with a summarised turn wedged between them.
        let (tree, normals) = build_uniform_tree(6, 1, 10, 15);
        let base = peak_cost(&tree);
        for extra in [0u32, 10, 25, 45, 500] {
            let budget = base + extra;
            let sel = select_budget_fit(&tree, budget, 0);
            assert!(sel.used_tokens <= budget, "budget {budget} exceeded");
            let raw: Vec<usize> = normals
                .iter()
                .enumerate()
                .filter(|(_, n)| sel.contains(**n))
                .map(|(i, _)| i)
                .collect();
            if let Some(&first) = raw.first() {
                let expected: Vec<usize> = (first..normals.len()).collect();
                assert_eq!(
                    raw, expected,
                    "raw turns must be the most-recent contiguous suffix (budget {budget})",
                );
            }
        }
    }

    #[test]
    fn only_summaries_when_budget_covers_just_the_peaks() {
        // At exactly the baseline budget there's no room to open anything —
        // every selected node is a covering summary, none is raw, and the
        // whole timeline is still covered. A full run of leaves so one SoS peak
        // covers them all (refining it costs strictly more than the peak budget).
        let (tree, _) = build_uniform_tree(MERGE_FANOUT as u32, 1, 10, 15);
        let sel = select_budget_fit(&tree, peak_cost(&tree), 0);
        assert!(!sel.selected.is_empty());
        assert!(
            sel.selected
                .iter()
                .all(|id| tree.get(*id).map(|n| n.kind) != Some(TurnKind::Normal)),
            "at baseline budget nothing should be raw — only covering summaries",
        );
        assert_full_coverage(&tree, &sel);
    }

    #[test]
    fn coverage_whole_and_budget_respected_when_peaks_fit() {
        let (tree, _) = build_uniform_tree(8, 1, 10, 12);
        let base = peak_cost(&tree);
        for extra in [0u32, 20, 60, 10_000] {
            let budget = base + extra;
            let sel = select_budget_fit(&tree, budget, 0);
            assert!(sel.used_tokens <= budget, "budget {budget} exceeded");
            assert_full_coverage(&tree, &sel);
        }
    }

    #[test]
    fn deterministic_under_ties() {
        // No provenance, uniform tokens — recency drives everything.  The
        // selection must be identical across repeated calls.
        let (tree, _) = build_uniform_tree(8, 1, 10, 20);
        let a = select_budget_fit(&tree, 120, 0);
        let b = select_budget_fit(&tree, 120, 0);
        assert_eq!(a.selected, b.selected);
        assert_eq!(a.origins, b.origins);
    }

    #[test]
    fn newest_raw_turns_are_hard_anchored() {
        let (tree, normals) = build_uniform_tree(5, 1, 10, 20);
        let sel = select_budget_fit(&tree, 10_000, 3);
        let newest = *normals.last().unwrap();
        let pos = sel.selected.iter().position(|n| *n == newest).unwrap();
        assert_eq!(sel.origins[pos], SelectionOrigin::HardAnchor);
    }

    /// Reproduces the real substrate shape where the async summariser has
    /// absorbed only ONE older turn (an SoT peak) while the recent turns are
    /// still bare orphan normals with no covering summary. Those orphans must
    /// still be selected — before the roots fix they vanished, and the decode
    /// saw only the one summarised turn.
    fn tree_with_orphan_normals() -> (SummaryTree, Vec<NodeId>) {
        let mut tree = SummaryTree::new();
        // Four normal turns 1000..1003, all registered chronologically.
        let normals: Vec<NodeId> = (1000..1004).map(NodeId).collect();
        for n in &normals {
            tree.insert_node(Node::normal(*n, 30));
            tree.push_chrono_normal(*n);
        }
        // Only the SECOND turn (1001) has been summarised into an SoT leaf; the
        // other three are orphan normals with no parent.
        let leaf = Node::summary_of_turns(NodeId(1), vec![normals[1]], 12);
        tree.insert_node(leaf);
        tree.push_chrono_leaf(NodeId(1));
        (tree, normals)
    }

    #[test]
    fn orphan_normals_are_not_dropped() {
        let (tree, normals) = tree_with_orphan_normals();
        // Ample budget: every turn should be present as raw, including the three
        // orphan normals that `tree.peaks()` alone would never surface.
        let sel = select_budget_fit(&tree, 10_000, 0);
        for n in &normals {
            assert!(
                sel.contains(*n),
                "orphan/covered normal {n} must be selected"
            );
        }
        assert_full_coverage(&tree, &sel);
    }

    #[test]
    fn orphan_normals_kept_raw_by_floor_under_tight_budget() {
        // Even with a summaries-only budget, the floor keeps the newest orphan
        // normals raw — the recent context can't be dropped just because the
        // summariser hasn't caught up.
        let (tree, normals) = tree_with_orphan_normals();
        let sel = select_budget_fit(&tree, 1, 2);
        assert!(sel.contains(normals[3]), "newest orphan normal must be raw");
        assert!(
            sel.contains(normals[2]),
            "second-newest orphan normal must be raw"
        );
    }

    #[test]
    fn floor_guarantees_newest_raw_even_over_budget() {
        // The budget covers only the coarse summaries, but a floor of 3 forces
        // the newest three turns raw anyway — the rule wins over the budget for
        // recent context, and the older turns stay summarised. Two full-run SoS
        // peaks so the newest turns and `normals[0]` sit under different peaks.
        let n = (2 * MERGE_FANOUT) as u32;
        let (tree, normals) = build_uniform_tree(n, 1, 10, 15);
        let sel = select_budget_fit(&tree, peak_cost(&tree), 3);
        for i in (n as usize - 3)..n as usize {
            assert!(
                sel.contains(normals[i]),
                "floor turn {i} must be raw regardless of budget",
            );
        }
        assert!(
            !sel.contains(normals[0]),
            "old turn beyond the floor must stay summarised",
        );
        assert_full_coverage(&tree, &sel);
        // The floor is allowed to exceed the budget — that is the point.
    }

    #[test]
    fn floor_at_least_turn_count_keeps_everything_raw() {
        // A floor at least as large as the turn count keeps every turn raw even
        // under a summaries-only budget.
        let (tree, normals) = build_uniform_tree(4, 1, 10, 20);
        let sel = select_budget_fit(&tree, peak_cost(&tree), 4);
        for n in &normals {
            assert!(sel.contains(*n), "floor must keep {n} raw");
        }
    }

    #[test]
    fn chronological_order_oldest_first() {
        let (tree, _) = build_uniform_tree(4, 1, 10, 20);
        let sel = select_budget_fit(&tree, 10_000, 0);
        // Selected raw normals must come out in chronological order.
        let normal_ids: Vec<u32> = sel
            .selected
            .iter()
            .filter(|id| tree.get(**id).map(|n| n.kind) == Some(TurnKind::Normal))
            .map(|id| id.0)
            .collect();
        let mut sorted = normal_ids.clone();
        sorted.sort();
        assert_eq!(normal_ids, sorted, "normals must come out chronologically");
    }
}
