//! Resolves what a projection may select for one target, masking every
//! conversation that target should not see.
//!
//! For the target's group only its own timeline is visible, so one chat never sees
//! a sibling's history. A forked conversation's own history includes the lineage
//! it continues. An append-only ingest target is masked to its own scope, so a scope
//! summary is grounded only in its own turns. Exchanges use the same partition the
//! belief scan scores, so a rule ranks the unit provenance voted on.

use super::*;

// ── TargetedRead ──────────────────────────────────────────────────────────────

/// Target-aware [`ContentResolver`] wrapper around a [`SubstrateRead`].
///
/// For `target.group`: only `target.timeline` is visible; sibling timelines
/// are masked.  For other groups: the first-registered timeline is used
/// (Phase 3 simplification for groups with a single shared timeline).
/// Sections are workspace singletons and pass straight through.
pub struct TargetedRead<'a> {
    pub(super) read: SubstrateRead<'a>,
    pub(super) target: ProjectionTarget,
}

impl<'a> TargetedRead<'a> {
    pub fn new(read: SubstrateRead<'a>, target: ProjectionTarget) -> Self {
        Self { read, target }
    }

    /// Partition one timeline's turns into exchanges — the same partition the
    /// belief scan scores over, derived the same way.
    ///
    /// Couplings address the `Normal` subsequence: summary-forest nodes hold
    /// turn indices too, so a raw `i → i+1` walk would fuse a coupled call with
    /// a summary that happens to sit between it and its response. `over_normals`
    /// projects the couplings onto that subsequence, and the ranges map back
    /// through it.
    ///
    /// Returns `(exchanges, slot_of)` where `slot_of[idx]` names the exchange a
    /// turn belongs to; a summary node appears in neither.
    pub(super) fn exchange_partition(
        &self,
        timeline: TimelineId,
    ) -> (Vec<Vec<TurnIndex>>, HashMap<TurnIndex, usize>) {
        let total = Substrate::turn_count(&self.read, timeline);
        let normals: Vec<TurnIndex> = (0..total)
            .map(TurnIndex)
            .filter(|idx| {
                !self
                    .read
                    .tree_meta_of(timeline, *idx)
                    .is_some_and(|m| m.kind.is_summary())
            })
            .collect();
        let couplings = over_normals(&self.read.couplings_of(timeline), &normals);
        let parts: Vec<Vec<TurnIndex>> = exchanges(&couplings, normals.len())
            .into_iter()
            .map(|r| r.map(|pos| normals[pos]).collect())
            .collect();
        let mut slot_of = HashMap::new();
        for (slot, part) in parts.iter().enumerate() {
            for idx in part {
                slot_of.insert(*idx, slot);
            }
        }
        (parts, slot_of)
    }
}

impl<'a> std::ops::Deref for TargetedRead<'a> {
    type Target = Substrate;
    fn deref(&self) -> &Substrate {
        &self.read
    }
}

impl<'a> ContentResolver for TargetedRead<'a> {
    fn any_layer_disabled(&self) -> bool {
        self.read.any_layer_disabled()
    }

    fn layer_disabled(&self, name: &str) -> bool {
        self.read.is_layer_disabled(name)
    }

    /// Target group → only `target.timeline` (sibling conversations of the same
    /// shape stay masked, so a slot never sees another chat's history). Every
    /// other group → **all** its active conversations: `code_reading` declares
    /// one timeline per file, and collapsing to the first-registered one made
    /// every other file unreachable to both scoring and projection.
    fn group_turns(&self, group: GroupId) -> Vec<TurnKey> {
        let turns_of = |tl: TimelineId| {
            (0..Substrate::turn_count(&self.read, tl)).map(move |i| TurnKey::new(tl, TurnIndex(i)))
        };
        // Self-local: the target group is always masked to the target timeline
        // (a slot never sees a sibling chat); when the target is an append-only
        // ingest layer (code_reading/repo_map), EVERY group is masked the same way
        // so a scope summary is grounded only in its own scope — the multi-timeline
        // scan (cross-file retrieval) belongs to dialogue, not ingest generation.
        // The exception is a layer that compounds on those below it: their groups
        // are retrieved across conversations, its own stays masked
        // (`Substrate::is_self_local_group`).
        if group == self.target.group || self.read.is_self_local_group(self.target.layer, group) {
            // "My own history" spans the fork lineage: a conversation forked
            // from another continues it, so the ancestors' turns are its own
            // opening, oldest first. Each key keeps its own timeline, so
            // exchange partitioning, ordering and emission downstream are
            // already multi-timeline and need no change — and an ancestor
            // sitting warm or cold is elevated by the ordinary projection
            // working-set path, which is why nothing has to be pinned hot.
            //
            // The dialogue's working set is NOT here: it sits in its own
            // layers' groups (`working_set_members`), so this group holds the
            // lineage and the conversation's own turns only.
            return self
                .read
                .inherited_chain(self.target.timeline)
                .into_iter()
                .flat_map(turns_of)
                .collect();
        }
        // The same scope the belief scan applied: what was never scored for
        // this target is never selected for it either.
        // An append-only target's own lineage is never a candidate: a layer it
        // compounds on can hold it (a day's history is the day's dream log's
        // opening), and offered again it would be emitted twice and take the
        // window of the units it was there to draw on.
        let lineage: HashSet<TimelineId> = if self.read.is_append_only_layer(self.target.layer) {
            self.read
                .inherited_chain(self.target.timeline)
                .into_iter()
                .collect()
        } else {
            HashSet::new()
        };
        self.read
            .scoped_timelines_for_group(group, self.target.timeline)
            .filter(|tl| !lineage.contains(tl))
            .flat_map(turns_of)
            .collect()
    }

    /// Exchange closure over the substrate's `TurnCoupling` records — the same
    /// partition the belief scan scores over, so a rule ranks exactly the unit
    /// provenance voted on.
    fn group_exchanges(&self, keys: &[TurnKey]) -> Vec<Vec<TurnKey>> {
        let mut partitions: HashMap<TimelineId, (Vec<Vec<TurnIndex>>, HashMap<TurnIndex, usize>)> =
            HashMap::new();
        let mut seen: HashSet<(TimelineId, usize)> = HashSet::new();
        let mut out: Vec<Vec<TurnKey>> = Vec::new();
        for key in keys {
            let (parts, slot_of) = partitions
                .entry(key.timeline)
                .or_insert_with(|| self.exchange_partition(key.timeline));
            // A summary-forest node joins no exchange (couplings are written
            // over Normal turns only), so it stands as its own unit.
            let Some(&slot) = slot_of.get(&key.index) else {
                out.push(vec![*key]);
                continue;
            };
            if !seen.insert((key.timeline, slot)) {
                continue;
            }
            out.push(
                parts[slot]
                    .iter()
                    .map(|idx| TurnKey::new(key.timeline, *idx))
                    .collect(),
            );
        }
        out
    }

    fn turn_token_count(&self, turn: TurnKey) -> usize {
        self.read.turn_token_count_of(turn.timeline, turn.index)
    }

    fn trained_window(&self) -> Option<usize> {
        self.read.trained_window()
    }

    fn turn_score(&self, turn: TurnKey) -> f32 {
        self.read.turn_score_for_timeline(turn.timeline, turn.index)
    }

    fn turn_reads(&self, turn: TurnKey) -> f32 {
        self.read.scores_or_empty().reads(turn.timeline, turn.index)
    }

    fn turn_carries(&self, turn: TurnKey, tags: &[String]) -> bool {
        self.read
            .turn_tags(turn.timeline, turn.index)
            .iter()
            .any(|t| tags.contains(t))
    }

    fn group_attention_mass(&self, group: GroupId) -> f32 {
        self.read.scores_or_empty().group_mass(group)
    }

    fn group_scan(&self, group: GroupId) -> Option<GroupScan> {
        self.read.scores_or_empty().group_scan(group)
    }

    fn collection_attention_mass(&self, collection: u32) -> f32 {
        self.read.scores_or_empty().collection_mass(collection)
    }

    fn turn_locality_boosted(&self, turn: TurnKey) -> bool {
        self.read
            .scores_or_empty()
            .is_locality_boosted(turn.timeline, turn.index)
    }

    fn seed_timelines(&self) -> HashSet<TimelineId> {
        self.read.seed_chain().into_iter().collect()
    }

    fn turn_origin(&self, turn: TurnKey) -> Option<LayerId> {
        let (layer, _) = self.read.timeline_target(turn.timeline)?;
        Some(layer)
    }

    fn face_of(&self, turn: TurnKey) -> Option<TurnKey> {
        self.read.face_of(turn)
    }

    fn unit_of(&self, turn: TurnKey) -> Option<TurnKey> {
        self.read.unit_of(turn)
    }

    fn target_is_ingest_self(&self) -> bool {
        self.read.is_append_only_layer(self.target.layer)
    }

    fn compounds_on(&self, group: GroupId) -> bool {
        self.read.is_append_only_layer(self.target.layer)
            && !self.read.is_self_local_group(self.target.layer, group)
    }

    fn turn_kind(&self, turn: TurnKey) -> TurnKind {
        self.read
            .tree_meta_of(turn.timeline, turn.index)
            .map(|m| m.kind)
            .unwrap_or(TurnKind::Normal)
    }

    fn node_covers(&self, turn: TurnKey) -> Vec<TurnIndex> {
        // Walk the immutable forest downward from `turn`, collecting every
        // transitive child. `tree_meta_of` gives a node's direct children;
        // Normal leaves have none, so a raw turn yields an empty cover set.
        // A summary only ever covers turns on its own timeline.
        let mut out = Vec::new();
        let mut stack = vec![turn.index];
        while let Some(node) = stack.pop() {
            if let Some(meta) = self.read.tree_meta_of(turn.timeline, node) {
                for &child in &meta.children {
                    out.push(child);
                    stack.push(child);
                }
            }
        }
        out
    }

    fn turn_no_think(&self, timeline: TimelineId, index: TurnIndex) -> bool {
        self.read.turn_no_think(timeline, index)
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        ContentResolver::section_token_count(&self.read, section)
    }

    fn section_score(&self, section: SectionId) -> f32 {
        ContentResolver::section_score(&self.read, section)
    }

    fn summary_tree_select(
        &self,
        timeline: TimelineId,
        budget: u32,
    ) -> Option<Vec<(TurnIndex, SelectionOrigin, f32)>> {
        // Delegate to the inner scored read: score-density selection over the
        // timeline's summary forest (see `select_dense` in `select.rs`).
        self.read.summary_tree_select(timeline, budget)
    }

    /// The target's working-set members in `group`, in the working set's order.
    ///
    /// A member the scope no longer offers is left out, the same as every
    /// other selection from a scoped group; the next observation takes it out
    /// of the set.
    fn working_set_members(&self, group: GroupId) -> Option<WorkingSetMembers> {
        let target = self.target.timeline;
        let ws = self.read.working_set(target)?;
        let members = ws
            .members()
            .into_iter()
            .filter(|tl| self.read.placing_group(target, *tl) == Some(group))
            .filter(|tl| self.read.scope_offers(group, target, *tl))
            // A new question's first reading is scored without the seeds.
            .filter(|tl| !ws.holds_out(*tl))
            .map(|tl| (tl, ws.momentum_of(tl).unwrap_or(0.0)))
            .collect();
        Some(WorkingSetMembers { members })
    }

    /// What the target's slot holds and whether a new question opened this
    /// boundary, as its working set recorded them at the scan
    /// (`WorkingSet::boundary`).
    fn working_set_boundary(&self) -> Option<SlotBoundary> {
        let ws = self.read.working_set(self.target.timeline)?;
        Some(SlotBoundary {
            held: ws.held().clone(),
            opening: ws.opening(),
            question_pending: ws.question_pending(),
        })
    }

    fn working_set_standing(&self, timeline: TimelineId) -> Standing {
        let path = self
            .read
            .custom_of(timeline)
            .and_then(|m| m.get("path"))
            .filter(|p| !p.is_empty())
            .cloned();
        let Some(ws) = self.read.working_set(self.target.timeline) else {
            return Standing {
                path,
                ..Standing::default()
            };
        };
        Standing {
            momentum: ws.momentum_of(timeline).unwrap_or(0.0),
            fresh: ws.fresh_of(timeline),
            hits: ws.hits_of(timeline),
            strength: ws.score_of(timeline).unwrap_or(0.0),
            path,
        }
    }

    fn working_set_unpooled(&self, n: usize) -> Vec<Unpooled> {
        let Some(ws) = self.read.working_set(self.target.timeline) else {
            return Vec::new();
        };
        let mut scanned: Vec<(TimelineId, f32)> =
            ws.scanned().filter(|(t, _)| !ws.is_member(*t)).collect();
        scanned.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.raw().cmp(&b.0.raw())));
        scanned
            .into_iter()
            .take(n)
            .map(|(t, _)| Unpooled {
                timeline: t.raw(),
                standing: self.working_set_standing(t),
            })
            .collect()
    }

    fn timeline_turns(&self, timeline: TimelineId) -> Vec<TurnKey> {
        (0..Substrate::turn_count(&self.read, timeline))
            .map(|i| TurnKey::new(timeline, TurnIndex(i)))
            .collect()
    }
}
