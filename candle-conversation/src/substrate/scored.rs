//! Holds the transient belief scores one reprojection produces, and the views
//! that pair a substrate with them for projection.
//!
//! Scores belong to one projection: the belief scan fills them, the emitter reads
//! them, and they are discarded, so a reload starts empty and the next scan
//! repopulates them. A turn the model's own selection kept reading is lifted by
//! those reads, so evidence the scan missed still counts. A bare substrate scores
//! every turn zero, which is why projection reads through the paired views. Where
//! a timeline has summary nodes, the score-density selector picks among them;
//! where it has none, the rule-based selector runs.

use super::*;

// ── Per-turn statistics ───────────────────────────────────────────────────────

/// Transient per-projection belief score cache.
///
/// **Not part of the persistent substrate state.** Built by the scheduler's
/// wide-Q belief scan during one reprojection, consumed by the projection
/// emitter during that same pass, then discarded. Conversation identity does not
/// include this — reload from log starts empty and the next scan repopulates it.
#[derive(Debug, Clone, Default)]
pub struct ProjectionScores {
    pub(super) turns: AHashMap<TurnKey, f32>,
    pub(super) sections: AHashMap<SectionId, f32>,
    /// Concept B attention mass per score competition: turn groups keyed by
    /// group id, collections keyed by collection id. Computed by the belief
    /// scan; consumed by the flexbox / member-budget modulation in `project`.
    pub(super) group_mass: AHashMap<u32, f32>,
    pub(super) collection_mass: AHashMap<u32, f32>,
    /// The order statistics of each turn group's raw scores this projection,
    /// keyed by group id — what the group's need is read from.
    pub(super) group_scan: AHashMap<u32, GroupScan>,
    /// Concept C: turns whose score was RAISED by locality drag (selected only
    /// through a neighbor) — stamped `SelectionOrigin::Locality` at selection.
    pub(super) locality_boosted: AHashSet<TurnKey>,
    /// Each lifted turn's reads: its lift less one, `gain · h`
    /// ([`Self::lift_turns`]).
    pub(super) reads: AHashMap<TurnKey, f32>,
}

impl ProjectionScores {
    /// An empty score cache — every lookup defaults to zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the belief score for one turn.
    pub fn set_turn(&mut self, timeline: TimelineId, index: TurnIndex, score: f32) {
        self.turns.insert(TurnKey::new(timeline, index), score);
    }

    /// Record the belief score for one system-prompt section.
    pub fn set_section(&mut self, section: SectionId, score: f32) {
        self.sections.insert(section, score);
    }

    /// Look up a turn's belief score. Zero when not scored this projection.
    pub fn turn(&self, timeline: TimelineId, index: TurnIndex) -> f32 {
        self.turns
            .get(&TurnKey::new(timeline, index))
            .copied()
            .unwrap_or(0.0)
    }

    /// How much the model's selection read a turn ([`Self::lift_turns`]).
    /// Zero when unread.
    pub fn reads(&self, timeline: TimelineId, index: TurnIndex) -> f32 {
        self.reads
            .get(&TurnKey::new(timeline, index))
            .copied()
            .unwrap_or(0.0)
    }

    /// Look up a section's belief score. Zero when not scored.
    pub fn section(&self, section: SectionId) -> f32 {
        self.sections.get(&section).copied().unwrap_or(0.0)
    }

    /// Number of scored turns.
    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    /// Lift each turn among `lifts` by its factor `1 + gain · h` — a turn the
    /// model's own selection keeps reading (`docs/hit_driven_projection.md`
    /// §7.1). A scored turn is multiplied, `score · (1 + gain · h)`. A turn this
    /// scan did not score, and that `credit` admits — the conversation's own
    /// turns and its lineage, which share the lineage's scale — is given what
    /// its reads alone are worth over a base of `base`, `base · gain · h`: a
    /// turn the model reads holds evidence the scan missed, and one read once
    /// barely registers. Any other unscored turn stays unscored.
    pub fn lift_turns(
        &mut self,
        lifts: impl IntoIterator<Item = (TurnKey, f32)>,
        base: f32,
        credit: impl Fn(&TurnKey) -> bool,
    ) {
        for (key, factor) in lifts {
            self.reads.insert(key, factor - 1.0);
            match self.turns.get_mut(&key) {
                Some(s) => *s *= factor,
                None if credit(&key) => {
                    let read = base * (factor - 1.0);
                    if read > 0.0 {
                        self.turns.insert(key, read);
                    }
                }
                None => {}
            }
        }
    }

    /// Every scored turn and its score, in no order.
    pub fn turn_scores(&self) -> impl Iterator<Item = (TurnKey, f32)> + '_ {
        self.turns.iter().map(|(k, s)| (*k, *s))
    }

    /// Number of scored sections.
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Record a turn group's Concept B attention mass.
    pub fn set_group_mass(&mut self, group: GroupId, mass: f32) {
        self.group_mass.insert(group.raw(), mass);
    }

    /// Record a collection's Concept B attention mass.
    pub fn set_collection_mass(&mut self, collection: u32, mass: f32) {
        self.collection_mass.insert(collection, mass);
    }

    /// A turn group's attention mass (0 when unscored this projection).
    pub fn group_mass(&self, group: GroupId) -> f32 {
        self.group_mass.get(&group.raw()).copied().unwrap_or(0.0)
    }

    /// Record the order statistics of a turn group's raw scores.
    pub fn set_group_scan(&mut self, group: GroupId, scan: GroupScan) {
        self.group_scan.insert(group.raw(), scan);
    }

    /// A turn group's raw-score statistics; `None` when it was not scanned.
    pub fn group_scan(&self, group: GroupId) -> Option<GroupScan> {
        self.group_scan.get(&group.raw()).copied()
    }

    /// A collection's attention mass (0 when unscored this projection).
    pub fn collection_mass(&self, collection: u32) -> f32 {
        self.collection_mass
            .get(&collection)
            .copied()
            .unwrap_or(0.0)
    }

    /// Mark a turn as locality-boosted (Concept C).
    pub fn mark_locality_boost(&mut self, timeline: TimelineId, index: TurnIndex) {
        self.locality_boosted.insert(TurnKey::new(timeline, index));
    }

    /// Whether a turn's score was raised by locality drag.
    pub fn is_locality_boosted(&self, timeline: TimelineId, index: TurnIndex) -> bool {
        self.locality_boosted
            .contains(&TurnKey::new(timeline, index))
    }

    /// Discard every score, leaving the cache empty.
    pub fn clear(&mut self) {
        self.turns.clear();
        self.sections.clear();
        self.group_mass.clear();
        self.collection_mass.clear();
        self.group_scan.clear();
        self.locality_boosted.clear();
    }

    /// Test helper: resolve `group`'s first registered timeline against
    /// `substrate`, then record `scores` for `(timeline, index)`.
    /// No-op when `group` has no registered timeline.
    ///
    /// Mirrors the old `Substrate::set_scores_for_test`, but operates on
    /// this transient cache rather than on substrate state.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn set_for_group_test(
        &mut self,
        substrate: &Substrate,
        group: GroupId,
        index: TurnIndex,
        score: f32,
    ) {
        if let Some(timeline) = substrate.timelines_for_group(group).next() {
            self.set_turn(timeline, index, score);
        }
    }
}

/// Pairing of a [`Substrate`] with a transient [`ProjectionScores`] cache.
///
/// Used by tests operating on a bare [`Substrate`] that want scoring
/// without going through a [`super::projection::resolver::Conversation`]
/// read guard. Production code does the same thing via
/// [`super::projection::resolver::Conversation::read_scored`].
pub struct ScoredSubstrate<'a> {
    pub(super) substrate: &'a Substrate,
    pub(super) scores: &'a ProjectionScores,
}

impl<'a> ScoredSubstrate<'a> {
    pub fn new(substrate: &'a Substrate, scores: &'a ProjectionScores) -> Self {
        Self { substrate, scores }
    }
}

/// Shared body of [`ContentResolver::summary_tree_select`] — the score-density
/// pick: build the timeline's summary forest, stamp each node with its
/// provenance turn score, and run [`select_dense`], which selects the most
/// relevant nodes (with recency anchoring/decay) that fit the window. Returns
/// the picked turns with their selection origins, each carrying its effective
/// provenance score for the diagnostics panel. `None` when the timeline has no
/// summary nodes, in which case the projection falls through to the rule-based
/// selector (which excludes summary turns). Used by the production
/// [`SubstrateRead`] resolver and the test-only [`ScoredSubstrate`].
pub(super) fn select_summary_tree(
    substrate: &Substrate,
    scores: &ProjectionScores,
    timeline: TimelineId,
    budget: u32,
) -> Option<Vec<(TurnIndex, SelectionOrigin, f32)>> {
    // No summary nodes yet → fall through to the rule-based path.
    if !substrate.has_summary_nodes(timeline) {
        return None;
    }
    let tree = substrate.build_summary_tree_in_memory(timeline);
    if tree.is_empty() {
        return None;
    }
    let mut node_scores: ahash::AHashMap<NodeId, f32> = ahash::AHashMap::default();
    for id in tree.all_ids() {
        let idx = TurnIndex(id.0);
        // A tree node without a backing substrate turn is an orphan (redo-log
        // TreeMetadata whose matching TurnDecl never landed). It can't be
        // elevated, so exclude it from the selection that flows into the
        // projection / elevate path.
        if substrate.turn(timeline, idx).is_none() {
            continue;
        }
        node_scores.insert(id, scores.turn(timeline, idx));
    }
    let cfg = RecencyConfig::default();
    let sel = select_dense(&tree, &node_scores, cfg, budget);
    // Convert (NodeId, SelectionOrigin) pairs back to TurnIndex, post-filtering
    // orphan NodeIds `select_dense` may have walked in via the tree shape.
    let out: Vec<_> = sel
        .selected
        .iter()
        .zip(sel.origins.iter())
        .filter_map(|(id, origin)| {
            let idx = TurnIndex(id.0);
            substrate.turn(timeline, idx)?;
            let eff = sel.effective_scores.get(id).copied().unwrap_or(0.0);
            Some((idx, *origin, eff))
        })
        .collect();
    Some(out)
}

impl<'a> std::ops::Deref for ScoredSubstrate<'a> {
    type Target = Substrate;
    fn deref(&self) -> &Substrate {
        self.substrate
    }
}

/// [`ContentResolver`] impl over a `(Substrate, ProjectionScores)` pair.
///
/// Enumerates every active timeline in a group, so a multi-conversation group is
/// scored and projected in full rather than collapsing to the first timeline.
/// Like the bare [`Substrate`] impl, it carries **no target** and applies **no
/// sibling-timeline masking** — a target-masked projection routes through
/// [`TargetedRead`]; this pairing is for structural / test callers.
impl<'a> ContentResolver for ScoredSubstrate<'a> {
    fn trained_window(&self) -> Option<usize> {
        self.substrate.trained_window()
    }

    fn group_turns(&self, group: GroupId) -> Vec<TurnKey> {
        self.substrate
            .active_timelines_for_group(group)
            .flat_map(|tl| {
                (0..Substrate::turn_count(self.substrate, tl))
                    .map(move |i| TurnKey::new(tl, TurnIndex(i)))
            })
            .collect()
    }

    fn any_layer_disabled(&self) -> bool {
        self.substrate.any_layer_disabled()
    }

    fn layer_disabled(&self, name: &str) -> bool {
        self.substrate.is_layer_disabled(name)
    }

    fn turn_token_count(&self, turn: TurnKey) -> usize {
        self.substrate
            .turn(turn.timeline, turn.index)
            .map_or(0, |e| e.content.token_count)
    }

    fn turn_score(&self, turn: TurnKey) -> f32 {
        if self.substrate.turn(turn.timeline, turn.index).is_none() {
            return 0.0;
        }
        self.scores.turn(turn.timeline, turn.index)
    }

    fn turn_reads(&self, turn: TurnKey) -> f32 {
        self.scores.reads(turn.timeline, turn.index)
    }

    fn group_attention_mass(&self, group: GroupId) -> f32 {
        self.scores.group_mass(group)
    }

    fn group_scan(&self, group: GroupId) -> Option<GroupScan> {
        self.scores.group_scan(group)
    }

    fn collection_attention_mass(&self, collection: u32) -> f32 {
        self.scores.collection_mass(collection)
    }

    fn turn_locality_boosted(&self, turn: TurnKey) -> bool {
        self.scores.is_locality_boosted(turn.timeline, turn.index)
    }

    fn seed_timelines(&self) -> HashSet<TimelineId> {
        self.substrate.seed_chain().into_iter().collect()
    }

    fn turn_origin(&self, turn: TurnKey) -> Option<LayerId> {
        let (layer, _) = self.substrate.timeline_target(turn.timeline)?;
        Some(layer)
    }

    fn face_of(&self, turn: TurnKey) -> Option<TurnKey> {
        self.substrate.face_of(turn)
    }

    fn unit_of(&self, turn: TurnKey) -> Option<TurnKey> {
        self.substrate.unit_of(turn)
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        self.substrate
            .sections
            .get(&section)
            .map_or(0, |e| e.token_count)
    }

    fn section_score(&self, section: SectionId) -> f32 {
        if !self.substrate.sections.contains_key(&section) {
            return 0.0;
        }
        self.scores.section(section)
    }

    fn summary_tree_select(
        &self,
        timeline: TimelineId,
        budget: u32,
    ) -> Option<Vec<(TurnIndex, SelectionOrigin, f32)>> {
        select_summary_tree(self.substrate, self.scores, timeline, budget)
    }

    fn pending_summary_len(&self, timeline: TimelineId) -> usize {
        self.substrate.pending_summary_len(timeline)
    }
}
