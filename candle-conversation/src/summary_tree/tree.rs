//! The `SummaryTree` — a three-kind tagged **append-only immutable forest**
//! (a ternary Merkle Mountain Range). Pure data structure: no substrate, no
//! scheduler. See `docs/immutable_summary_forest.md`.
//!
//! # Shape
//!
//! ```text
//!   peaks:   SoS(level h)        SoS(h')   SoT … (orphans = window entry points)
//!            ╱   │   ╲
//!         SoS  SoS  SoS          ← exactly MERGE_FANOUT children, all level h-1
//!          │    │    │
//!         SoT  SoT  SoT          ← one Normal child each (level 1)
//!          │    │    │
//!          ●    ●    ●           ← Normal content sub-leaves
//! ```
//!
//! # Insertion
//!
//! New `SummaryOfTurns` leaves are appended on the right (newest turn →
//! rightmost peak). After each append the **carry rule** fires: while the
//! last `MERGE_FANOUT` peaks share a level, they are merged into one parent
//! one level up. Nodes are never mutated once created — a node leaves the
//! peak set only when a later merge gives it a parent. There is no balancing,
//! no rotation, and no `dirty` state: the structure is a pure function of the
//! leaf sequence.

use ahash::{AHashMap, AHashSet};

/// Fan-out of an internal `SummaryOfSummaries` node: a merge combines exactly
/// this many same-level peaks into one parent. Ternary (3) halves the
/// internal-node count versus binary while keeping the peak window tight.
pub const MERGE_FANOUT: usize = 3;

/// Default token cost for a synthesised `SummaryOfSummaries` internal node.
/// The §6 probe overwrites this with the actual content size once the sealed
/// summary turn lands.
pub const DEFAULT_INTERNAL_TOKENS: u32 = 20;

/// Tag distinguishing the three node kinds in the summary forest.
///
/// Every persisted turn carries one of these tags so the cold-load path can
/// reconstruct the forest from a flat list of turn records.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnKind {
    /// Ordinary user / assistant content. Lives as a content sub-leaf under a
    /// [`SummaryOfTurns`](TurnKind::SummaryOfTurns) parent; not part of the
    /// forest spine.
    Normal,
    /// Forest leaf: a single summary turn over exactly one `Normal` turn.
    /// Level 1. Produced by the §6 probe over its Normal child.
    SummaryOfTurns,
    /// Internal forest node: a summary covering exactly [`MERGE_FANOUT`] child
    /// summary nodes (each `SummaryOfTurns` or another `SummaryOfSummaries`),
    /// all of the same level. Produced by the §6 probe.
    SummaryOfSummaries,
}

impl TurnKind {
    /// True when this kind participates in the forest spine (summary nodes —
    /// not Normal sub-leaves).
    pub fn is_summary(self) -> bool {
        matches!(
            self,
            TurnKind::SummaryOfTurns | TurnKind::SummaryOfSummaries
        )
    }
}

/// Abstract node identifier. In algorithm tests this is a plain `u32`; the
/// production integration maps it to the substrate's `TurnIndex`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NodeId(pub u32);

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// One forest node. Uniform shape across all three kinds so the score-density
/// selection (`select_dense`) can walk the forest without branching on kind.
#[derive(Debug, Clone)]
pub struct Node {
    pub id: NodeId,
    pub kind: TurnKind,
    /// For `SummaryOfSummaries`: exactly [`MERGE_FANOUT`] child summary IDs in
    /// chronological order.
    /// For `SummaryOfTurns`: the IDs of all Normal-turn sub-leaves under this
    /// leaf, in chronological order (one in production).
    /// For `Normal`: empty.
    pub children: Vec<NodeId>,
    /// Forest level. `SummaryOfTurns` is always 1; a `SummaryOfSummaries` over
    /// level-`h` children is level `h + 1`. `Normal` turns are not part of the
    /// spine and carry level 0.
    pub tree_height: u8,
    /// Token cost of this node's content for budget accounting.
    pub tokens: u32,
}

impl Node {
    /// Build a fresh `Normal` content sub-leaf.
    pub fn normal(id: NodeId, tokens: u32) -> Self {
        Self {
            id,
            kind: TurnKind::Normal,
            children: Vec::new(),
            tree_height: 0,
            tokens,
        }
    }

    /// Build a fresh `SummaryOfTurns` leaf. The caller supplies the Normal-turn
    /// children in chronological order.
    pub fn summary_of_turns(id: NodeId, normals: Vec<NodeId>, tokens: u32) -> Self {
        Self {
            id,
            kind: TurnKind::SummaryOfTurns,
            children: normals,
            tree_height: 1,
            tokens,
        }
    }

    /// Build a fresh `SummaryOfSummaries` internal node over `children` (exactly
    /// [`MERGE_FANOUT`] same-level summary nodes). `tree_height` is one above the
    /// children's level.
    pub fn summary_of_summaries(
        id: NodeId,
        children: Vec<NodeId>,
        tree_height: u8,
        tokens: u32,
    ) -> Self {
        Self {
            id,
            kind: TurnKind::SummaryOfSummaries,
            children,
            tree_height,
            tokens,
        }
    }

    /// True iff this node has children.
    pub fn has_children(&self) -> bool {
        !self.children.is_empty()
    }
}

/// Whether appending fired a carry: given the current peak **levels** in
/// chronological order, return `Some(start)` of the trailing run of
/// [`MERGE_FANOUT`] equal-level peaks to merge, else `None`. Shared by the pure
/// [`SummaryTree::append_leaf`] and the substrate-driven summariser so the
/// carry rule lives in exactly one place.
pub fn carry_triple(levels: &[u8]) -> Option<usize> {
    let n = levels.len();
    if n < MERGE_FANOUT {
        return None;
    }
    let start = n - MERGE_FANOUT;
    let lvl = levels[start];
    if levels[start..].iter().all(|&l| l == lvl) {
        Some(start)
    } else {
        None
    }
}

/// The summary forest. Owns every node; appends are the only mutation and they
/// never rewrite an existing node.
///
/// Storage: `AHashMap<NodeId, Node>`. The peak set (orphan summary nodes) is
/// derived on demand — a node is a peak iff no `SummaryOfSummaries` lists it as
/// a child.
#[derive(Debug, Clone, Default)]
pub struct SummaryTree {
    nodes: AHashMap<NodeId, Node>,
    /// All Normal-turn children in chronological order.
    chrono_normals: Vec<NodeId>,
    /// All `SummaryOfTurns` leaves in chronological order.
    chrono_leaves: Vec<NodeId>,
}

impl SummaryTree {
    /// Empty forest.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of nodes (all kinds).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Lookup a node by id.
    pub fn get(&self, id: NodeId) -> Option<&Node> {
        self.nodes.get(&id)
    }

    /// All node IDs (unspecified order).
    pub fn all_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.nodes.keys().copied()
    }

    /// All `SummaryOfTurns` leaves in chronological order.
    pub fn chrono_leaves(&self) -> &[NodeId] {
        &self.chrono_leaves
    }

    /// All Normal-turn sub-leaves in chronological order.
    pub fn chrono_normals(&self) -> &[NodeId] {
        &self.chrono_normals
    }

    /// The peak set — orphan summary nodes (no parent) in chronological order
    /// (oldest/leftmost-covering first). This is the window's set of coarse
    /// entry points.
    pub fn peaks(&self) -> Vec<NodeId> {
        let mut claimed: AHashSet<NodeId> = AHashSet::new();
        for node in self.nodes.values() {
            if node.kind == TurnKind::SummaryOfSummaries {
                for c in &node.children {
                    claimed.insert(*c);
                }
            }
        }
        let mut peaks: Vec<NodeId> = self
            .nodes
            .values()
            .filter(|n| n.kind.is_summary() && !claimed.contains(&n.id))
            .map(|n| n.id)
            .collect();
        peaks.sort_by_key(|id| self.leftmost_normal(*id));
        peaks
    }

    /// Levels of the peaks, aligned with [`Self::peaks`].
    pub fn peak_levels(&self) -> Vec<u8> {
        self.peaks()
            .iter()
            .map(|id| self.nodes.get(id).map(|n| n.tree_height).unwrap_or(0))
            .collect()
    }

    /// Smallest Normal-turn index covered by `id` — defines chronological order
    /// of peaks (and of any subtree).
    fn leftmost_normal(&self, id: NodeId) -> u32 {
        let node = match self.nodes.get(&id) {
            Some(n) => n,
            None => return id.0,
        };
        match node.kind {
            TurnKind::Normal => id.0,
            TurnKind::SummaryOfTurns => node.children.iter().map(|c| c.0).min().unwrap_or(id.0),
            TurnKind::SummaryOfSummaries => node
                .children
                .iter()
                .map(|c| self.leftmost_normal(*c))
                .min()
                .unwrap_or(id.0),
        }
    }

    /// Tallest peak level, or 0 when empty.
    pub fn height(&self) -> u8 {
        self.peak_levels().into_iter().max().unwrap_or(0)
    }

    /// Insert a freshly-built `Node` into the storage map. Does not touch
    /// `chrono_*`. Used by test scaffolding and the substrate-side rebuild path
    /// (which feeds persisted nodes back in without re-running the merge).
    pub fn insert_node(&mut self, node: Node) {
        self.nodes.insert(node.id, node);
    }

    /// Append a Normal sub-leaf to the chronological order. Used by
    /// substrate-side rebuild after `insert_node` has registered the node.
    pub fn push_chrono_normal(&mut self, id: NodeId) {
        self.chrono_normals.push(id);
    }

    /// Append a `SummaryOfTurns` leaf to the chronological order. Used by
    /// substrate-side rebuild.
    pub fn push_chrono_leaf(&mut self, id: NodeId) {
        self.chrono_leaves.push(id);
    }

    /// Append a Normal-turn sub-leaf to the rightmost (open) `SummaryOfTurns`
    /// leaf. Does not touch the forest spine.
    ///
    /// # Panics
    /// Panics if the forest has no `SummaryOfTurns` leaf yet.
    pub fn append_normal_to_open_cluster(&mut self, normal: Node) {
        assert_eq!(
            normal.kind,
            TurnKind::Normal,
            "append_normal_to_open_cluster called with non-Normal node"
        );
        let open_leaf = *self.chrono_leaves.last().expect(
            "forest must contain at least one SummaryOfTurns leaf before appending Normals",
        );
        let normal_id = normal.id;
        let normal_tokens = normal.tokens;
        self.nodes.insert(normal_id, normal);
        self.chrono_normals.push(normal_id);
        if let Some(leaf) = self.nodes.get_mut(&open_leaf) {
            leaf.children.push(normal_id);
            leaf.tokens = leaf.tokens.saturating_add(normal_tokens);
        }
    }

    /// Append a fresh `SummaryOfTurns` leaf on the right, then run the ternary
    /// carry: while the last [`MERGE_FANOUT`] peaks share a level, merge them
    /// into a fresh internal node one level up. Internal node IDs are
    /// auto-allocated (test/pure use; the summariser supplies real substrate
    /// turn indices instead). Returns the internal nodes created by this append,
    /// oldest-merge first.
    pub fn append_leaf(&mut self, leaf: Node) -> Vec<NodeId> {
        assert_eq!(
            leaf.kind,
            TurnKind::SummaryOfTurns,
            "append_leaf requires a SummaryOfTurns node"
        );
        let leaf_id = leaf.id;
        let normal_ids: Vec<NodeId> = leaf.children.clone();
        self.nodes.insert(leaf_id, leaf);
        for nid in &normal_ids {
            if self.nodes.contains_key(nid) && !self.chrono_normals.iter().any(|n| n == nid) {
                self.chrono_normals.push(*nid);
            }
        }
        self.chrono_leaves.push(leaf_id);

        let mut created = Vec::new();
        loop {
            let peaks = self.peaks();
            let levels: Vec<u8> = peaks.iter().map(|id| self.nodes[id].tree_height).collect();
            let Some(start) = carry_triple(&levels) else {
                break;
            };
            let children: Vec<NodeId> = peaks[start..].to_vec();
            let level = levels[start] + 1;
            let parent_id = self.fresh_internal_id();
            self.nodes.insert(
                parent_id,
                Node::summary_of_summaries(parent_id, children, level, DEFAULT_INTERNAL_TOKENS),
            );
            created.push(parent_id);
        }
        created
    }

    /// Allocate a fresh NodeId for a synthesised internal node (auto-allocator
    /// for pure/test use). Reserves IDs ≥ 2^31 so they cannot collide with
    /// caller-supplied leaf IDs.
    fn fresh_internal_id(&mut self) -> NodeId {
        let mut candidate = 1u32 << 31;
        while self.nodes.contains_key(&NodeId(candidate)) {
            candidate += 1;
        }
        NodeId(candidate)
    }

    /// Walk the forest in post-order across all peaks (chronological): Normal
    /// sub-leaves first, then `SummaryOfTurns`, then `SummaryOfSummaries`.
    pub fn post_order(&self) -> Vec<NodeId> {
        let mut out = Vec::with_capacity(self.nodes.len());
        for peak in self.peaks() {
            self.post_order_walk(peak, &mut out);
        }
        out
    }

    fn post_order_walk(&self, id: NodeId, out: &mut Vec<NodeId>) {
        let node = match self.nodes.get(&id) {
            Some(n) => n,
            None => return,
        };
        match node.kind {
            TurnKind::Normal => {
                out.push(id);
            }
            TurnKind::SummaryOfTurns => {
                for child in &node.children {
                    self.post_order_walk(*child, out);
                }
                out.push(id);
            }
            TurnKind::SummaryOfSummaries => {
                for child in &node.children {
                    self.post_order_walk(*child, out);
                }
                out.push(id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A SoT leaf with no Normal sub-children — the carry/peak structure only
    // depends on the summary spine, and leaving normals out keeps leaf ids
    // (which the SoS children lists reference) easy to assert against.
    fn leaf(id: u32, tokens: u32) -> Node {
        Node::summary_of_turns(NodeId(id), Vec::new(), tokens)
    }

    fn make_normal(id: u32, tokens: u32) -> Node {
        Node::normal(NodeId(id), tokens)
    }

    /// Sorted multiset of peak levels — the structural fingerprint of the
    /// forest after `N` appends. The carry rule makes this exactly the base-3
    /// digit expansion of `N` (digit `d` at level `h+1` ⇒ `d` peaks of level
    /// `h+1`).
    fn peak_level_histogram(t: &SummaryTree) -> Vec<u8> {
        let mut levels = t.peak_levels();
        levels.sort_unstable();
        levels
    }

    /// Expected peak levels for `n` leaves, derived from base-3 digits.
    fn expected_levels(n: u32) -> Vec<u8> {
        let mut levels = Vec::new();
        let mut n = n;
        let mut level = 1u8; // leaves are level 1
        while n > 0 {
            let digit = n % 3;
            for _ in 0..digit {
                levels.push(level);
            }
            n /= 3;
            level += 1;
        }
        levels.sort_unstable();
        levels
    }

    #[test]
    fn empty_forest() {
        let t = SummaryTree::new();
        assert!(t.is_empty());
        assert!(t.peaks().is_empty());
        assert_eq!(t.height(), 0);
    }

    #[test]
    fn first_leaf_is_lone_peak() {
        let mut t = SummaryTree::new();
        let created = t.append_leaf(leaf(1, 20));
        assert!(created.is_empty(), "first leaf creates no internal");
        assert_eq!(t.peaks(), vec![NodeId(1)]);
        assert_eq!(t.height(), 1);
    }

    #[test]
    fn three_leaves_carry_into_one_sos() {
        let mut t = SummaryTree::new();
        t.append_leaf(leaf(1, 20));
        t.append_leaf(leaf(2, 20));
        let created = t.append_leaf(leaf(3, 20));
        // Third leaf triggers a single carry: one level-2 SoS over [1,2,3].
        assert_eq!(created.len(), 1);
        let sos = created[0];
        let node = t.get(sos).unwrap();
        assert_eq!(node.kind, TurnKind::SummaryOfSummaries);
        assert_eq!(node.tree_height, 2);
        assert_eq!(node.children, vec![NodeId(1), NodeId(2), NodeId(3)]);
        assert_eq!(t.peaks(), vec![sos]);
    }

    #[test]
    fn nine_leaves_collapse_to_single_level3_peak() {
        let mut t = SummaryTree::new();
        for i in 1..=9u32 {
            t.append_leaf(leaf(i, 20));
        }
        // 9 = 3^2 → a single level-3 peak over three level-2 nodes.
        let peaks = t.peaks();
        assert_eq!(peaks.len(), 1);
        let root = t.get(peaks[0]).unwrap();
        assert_eq!(root.tree_height, 3);
        assert_eq!(root.children.len(), MERGE_FANOUT);
        for c in &root.children {
            let child = t.get(*c).unwrap();
            assert_eq!(child.tree_height, 2);
            assert_eq!(child.children.len(), MERGE_FANOUT);
        }
    }

    #[test]
    fn peak_levels_track_base3_digits() {
        let mut t = SummaryTree::new();
        for n in 1..=40u32 {
            t.append_leaf(leaf(n, 20));
            assert_eq!(
                peak_level_histogram(&t),
                expected_levels(n),
                "peak levels diverged from base-3 digits at n={n}"
            );
        }
    }

    #[test]
    fn every_sos_has_fanout_equal_level_children() {
        let mut t = SummaryTree::new();
        for n in 1..=40u32 {
            t.append_leaf(leaf(n, 20));
        }
        for node in t.all_ids().filter_map(|id| t.get(id)) {
            if node.kind != TurnKind::SummaryOfSummaries {
                continue;
            }
            assert_eq!(node.children.len(), MERGE_FANOUT);
            let levels: Vec<u8> = node
                .children
                .iter()
                .map(|c| t.get(*c).unwrap().tree_height)
                .collect();
            assert!(
                levels.iter().all(|&l| l == levels[0]),
                "SoS children must share a level"
            );
            assert_eq!(node.tree_height, levels[0] + 1);
        }
    }

    #[test]
    fn nodes_are_immutable_across_appends() {
        // Snapshot every node after each append; nodes that already existed
        // must never change their children or level.
        let mut t = SummaryTree::new();
        let mut snap: std::collections::HashMap<NodeId, (Vec<NodeId>, u8)> =
            std::collections::HashMap::new();
        for n in 1..=27u32 {
            t.append_leaf(leaf(n, 20));
            for id in t.all_ids() {
                let node = t.get(id).unwrap();
                let now = (node.children.clone(), node.tree_height);
                if let Some(prev) = snap.get(&id) {
                    assert_eq!(prev, &now, "node {id} mutated after creation");
                } else {
                    snap.insert(id, now);
                }
            }
        }
    }

    #[test]
    fn peaks_form_contiguous_cover() {
        let mut t = SummaryTree::new();
        for n in 1..=20u32 {
            t.append_leaf(leaf(n, 20));
        }
        // Each peak's covered leaves, concatenated left→right, must be exactly
        // leaves 1..=20 with no gaps or overlaps.
        let mut covered: Vec<u32> = Vec::new();
        for peak in t.peaks() {
            collect_leaves(&t, peak, &mut covered);
        }
        assert_eq!(covered, (1..=20).collect::<Vec<_>>());
    }

    fn collect_leaves(t: &SummaryTree, id: NodeId, out: &mut Vec<u32>) {
        let node = t.get(id).unwrap();
        match node.kind {
            TurnKind::SummaryOfTurns => out.push(id.0),
            TurnKind::SummaryOfSummaries => {
                for c in &node.children {
                    collect_leaves(t, *c, out);
                }
            }
            TurnKind::Normal => {}
        }
    }

    #[test]
    fn carry_triple_detects_trailing_run() {
        assert_eq!(carry_triple(&[]), None);
        assert_eq!(carry_triple(&[1, 1]), None);
        assert_eq!(carry_triple(&[2, 1, 1, 1]), Some(1));
        assert_eq!(carry_triple(&[1, 1, 2]), None);
        assert_eq!(carry_triple(&[2, 2, 2]), Some(0));
    }

    #[test]
    fn append_normal_attaches_under_open_cluster() {
        let mut t = SummaryTree::new();
        t.append_leaf(leaf(1, 20));
        t.append_normal_to_open_cluster(make_normal(100, 10));
        t.append_normal_to_open_cluster(make_normal(101, 10));
        let open = t.get(NodeId(1)).unwrap();
        assert_eq!(open.children, vec![NodeId(100), NodeId(101)]);
    }

    #[test]
    fn post_order_visits_children_before_peaks() {
        let mut t = SummaryTree::new();
        for n in 1..=3u32 {
            t.append_leaf(leaf(n, 20));
        }
        let order = t.post_order();
        // SoT leaves come before the SoS peak; the peak is last.
        let peak = t.peaks()[0];
        assert_eq!(*order.last().unwrap(), peak);
        let pos = |id: u32| order.iter().position(|n| *n == NodeId(id)).unwrap();
        assert!(pos(1) < pos(peak.0));
        assert!(pos(2) < pos(peak.0));
        assert!(pos(3) < pos(peak.0));
    }
}
