//! Holds the timeline registry: which timelines exist, the groups and scopes they
//! belong to, the fork lineage that links them, and the working sets each
//! dialogue draws on.
//!
//! Registration is idempotent and replays the records that arrived before it, so
//! the registry reads the same whatever order the log is walked in. A
//! conversation's history is its ancestors' turns oldest-first, then its own,
//! capped so the nearest ancestors survive. A dangling parent reads as no
//! inheritance rather than as an error. A scoped group offers a target only the
//! conversations its scope names, and a conversation that cannot deliver whole
//! content is never placed in a working set.

use super::*;

impl Substrate {
    // ── Timeline registry ────────────────────────────────────────────────────

    pub fn register_timeline(&mut self, timeline: TimelineId, layer: LayerId, group: GroupId) {
        // Idempotent — `HashMap::insert` *replaces*, so calling it on
        // an already-registered timeline would wipe its `turns` map (a
        // real data-loss bug on substrate replay, which calls
        // `register_timeline` once per recovered TurnDecl).
        if self.timelines.contains_key(&timeline) {
            return;
        }
        self.timelines
            .insert(timeline, Arc::new(TimelineEntry::new(layer, group)));
        self.timelines_by_group
            .entry(group)
            .or_default()
            .push_back(timeline);
        // Apply every record that arrived before this timeline registered —
        // the sidebar listing depends on the conv_id landing on the entry.
        // Each goes through the same stamped apply it would have met on
        // arrival, so the order they were held in does not matter.
        for (child, id) in self.pending.release(timeline.raw()) {
            let raw = timeline.raw();
            match child {
                PendingChild::Label(meta) => self.apply_conv_meta(raw, &meta, id),
                PendingChild::State(state) => self.apply_conv_state(raw, state, id),
                PendingChild::TreeMeta(p) => self.apply_tree_metadata_payload(&p, id),
                PendingChild::DebugId(debug_id) => self.apply_debug_id_payload(
                    &DebugIdPayload {
                        timeline_id: raw,
                        debug_id,
                    },
                    id,
                ),
                PendingChild::Coupling(from_turn) => self.apply_turn_coupling(
                    &TurnCouplingPayload {
                        timeline_id: raw,
                        from_turn,
                    },
                    id,
                ),
            }
        }
    }

    pub fn mint_timeline(
        &mut self,
        layer: LayerId,
        group: GroupId,
        allocator: &TimelineAllocator,
    ) -> TimelineId {
        let id = allocator.next();
        self.register_timeline(id, layer, group);
        id
    }

    /// Every timeline this substrate holds, in no particular order.
    pub fn timeline_ids(&self) -> Vec<TimelineId> {
        self.timelines.keys().copied().collect()
    }

    /// Flip the diagnostic toggle on layer `name`; returns the new state
    /// (`true` = now excluded from projection assembly).
    pub fn toggle_layer(&mut self, name: &str) -> bool {
        if self.disabled_layers.remove(name).is_some() {
            false
        } else {
            self.disabled_layers.insert(name.to_string());
            true
        }
    }

    /// Whether layer `name` is toggled out of projection assembly.
    pub fn is_layer_disabled(&self, name: &str) -> bool {
        !self.disabled_layers.is_empty() && self.disabled_layers.contains(name)
    }

    /// Whether any layer is toggled out — the check a per-layer walk hoists.
    pub fn any_layer_disabled(&self) -> bool {
        !self.disabled_layers.is_empty()
    }

    /// Every layer currently toggled out.
    pub fn disabled_layers(&self) -> HashSet<String> {
        self.disabled_layers.iter().cloned().collect()
    }

    pub fn timeline_target(&self, timeline: TimelineId) -> Option<(LayerId, GroupId)> {
        self.timelines.get(&timeline).map(|e| (e.layer, e.group))
    }

    /// Borrow a timeline's full entry — projection target plus optional
    /// sidebar metadata. Used by the daemon's sidebar listing.
    pub fn timeline_entry(&self, timeline: TimelineId) -> Option<&TimelineEntry> {
        self.timelines.get(&timeline).map(|e| &**e)
    }

    /// Borrow a single turn by `(timeline, index)`: two map hops, since turns
    /// live inside their owning [`TimelineEntry`].
    pub(super) fn turn(&self, timeline: TimelineId, index: TurnIndex) -> Option<&TurnEntryData> {
        self.timelines
            .get(&timeline)
            .and_then(|t| t.turns.get(&index))
            .map(|t| &**t)
    }

    /// Mutable variant of [`Self::turn`]: this version's own copy of the
    /// timeline and the turn, so older versions keep theirs.
    pub(super) fn turn_mut(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<&mut TurnEntryData> {
        entry_mut(&mut self.timelines, &timeline).and_then(|t| ord_arc_mut(&mut t.turns, &index))
    }

    /// Every conversation registered in `group` — none for a drawing group,
    /// which holds none of its own ([`Self::active_timelines_for_group`] is
    /// what it selects from).
    pub fn timelines_for_group(&self, group: GroupId) -> impl Iterator<Item = TimelineId> + '_ {
        self.timelines_by_group
            .get(&group)
            .into_iter()
            .flat_map(|v| v.iter().copied())
    }

    /// Resolve which timeline a projected turn in `group` belongs to.
    ///
    /// The projection `target` pins the ACTIVE conversation: a turn in the
    /// target's own group resolves to `target.timeline` (the conversation being
    /// projected/decoded); every other group has a single registered timeline,
    /// taken in registration order.  A `None` target (e.g. a utility pass with no
    /// active conversation) falls back to that registration-order pick for all
    /// groups.
    ///
    /// This is the SINGLE source of truth for turn → timeline resolution.  It
    /// exists because open-coding `timelines_for_group(g).next()` WITHOUT the
    /// target-group special case resolves the wrong conversation's timeline as
    /// soon as more than one conversation shares a group — which silently dropped
    /// every turn of any non-first conversation on mid-decode reproject (the slot
    /// rebuilt with `turns=0`, i.e. the model lost its whole history).  Every call
    /// site — the SubmitTurn prefill, the reproject rebuild, and
    /// `inject_sealed_turn` — routes through here so they can never diverge again.
    pub fn resolve_turn_timeline(
        &self,
        target: Option<ProjectionTarget>,
        group: GroupId,
    ) -> Option<TimelineId> {
        match target {
            Some(t) if group == t.group => Some(t.timeline),
            _ => self.timelines_for_group(group).next(),
        }
    }

    /// Like [`Self::timelines_for_group`] but excludes timelines
    /// flagged `archived` or tombstoned.  Projection retrieval
    /// ([`ContentResolver`]) uses this so retired and user-archived
    /// conversations drop out of selection without their turns
    /// being physically deleted from the substrate. A drawing group's are
    /// those of the group it draws from ([`Self::set_group_draws`]).
    pub fn active_timelines_for_group(
        &self,
        group: GroupId,
    ) -> impl Iterator<Item = TimelineId> + '_ {
        self.timelines_for_group(self.holding_group(group))
            .filter(move |tl| {
                if self.tombstoned_timelines.contains(tl) {
                    return false;
                }
                self.timelines.get(tl).map(|e| !e.archived).unwrap_or(true)
            })
    }

    // ── fork lineage ────────────────────────────────────────────────────────

    /// Metadata key under which a fork records the timeline it came from.
    ///
    /// A plain `custom` entry rather than a record type of its own: the Label
    /// record carrying `custom` is already durable, already re-emitted by both
    /// the full-compaction and incremental-maintenance live sets, and already
    /// replayed into [`TimelineEntry`] on reload. A parent pointer kept here
    /// therefore needs no new wire format and cannot be silently dropped by a
    /// rewrite that has never heard of it — which is exactly how the in-memory
    /// `splice_source_timelines` pin failed.
    pub const FORKED_FROM_KEY: &'static str = "forked_from";

    /// The `forked_from` value that names no timeline but **the seed**: the
    /// priming chain's end as the member reading it holds it ([`Self::seed`]).
    ///
    /// A dialogue opens on the priming chain, and the chain is rebuilt every
    /// boot on every member — a changed anchor document is a new unit, and each
    /// member's process mints its own base conversation. A parent recorded as a
    /// timeline id named one of those, and it went stale: a conversation
    /// parented on a boot's base kept that boot's chain for good, and on any
    /// other member, which never held that base, inherited nothing at all. The
    /// alias is resolved where the lineage is read, so every conversation opens
    /// on the chain the member serving it holds now.
    pub const SEED_PARENT: &'static str = "seed";

    /// Metadata key under which a reading records the content key it was read
    /// under: a file's `v{N}:{path}@{blob}`, a folder's listing hash.
    pub const CONTENT_KEY: &'static str = "content_key";

    /// Metadata key holding a file conversation's workspace-relative path —
    /// what a scan that reads files by their kind (their extension) reads.
    pub const PATH_KEY: &'static str = "path";

    /// The prefix of a `forked_from` that names its parent by **content key**
    /// rather than timeline id — `unit:<key>`, resolved where the lineage is
    /// read to the conversation holding that key ([`Self::unit_holding`]).
    ///
    /// A priming-chain link names the link before it this way. Two members
    /// that read a changed link at once each mint a conversation for it, and
    /// each re-pointed the next link at its own; the pointer that replicated
    /// last won, and on the member whose copy was retired as the duplicate the
    /// next link's parent read as absent — every dialogue's seed cut there
    /// (2026-11-11). Named by key, both members write the same parent, and
    /// whichever copy survives satisfies it.
    pub const UNIT_PARENT: &'static str = "unit:";

    /// The `forked_from` value naming the unit read under `key` as a parent.
    pub fn unit_parent(key: &str) -> String {
        format!("{}{key}", Self::UNIT_PARENT)
    }

    /// Name `seed` as what [`Self::SEED_PARENT`] resolves to on this member;
    /// `None` while there is no chain.
    pub fn set_seed(&mut self, seed: Option<TimelineId>) {
        self.seed = seed;
    }

    /// What [`Self::SEED_PARENT`] resolves to on this member.
    pub fn seed(&self) -> Option<TimelineId> {
        self.seed
    }

    /// Every link of this member's seed, oldest first: the priming chain a
    /// dialogue opens on. Empty before a seed is named.
    pub fn seed_chain(&self) -> Vec<TimelineId> {
        self.seed
            .map(|seed| self.inherited_chain(seed))
            .unwrap_or_default()
    }

    /// Ceiling on the tokens an inherited chain may contribute *above* the
    /// target's own turns.
    ///
    /// The chain is walked child-upward and stops once the next ancestor would
    /// cross this, so what gets dropped is the FRONT — the oldest, most
    /// general documents — while the nearest, most specific ancestors survive.
    ///
    /// Sized inside a dialogue's window, which seats the lineage and the
    /// dialogue's own turns within the trained window's slot
    /// (`docs/hit_driven_projection.md` §4): the priming chain is ~40K.
    pub const INHERITED_CHAIN_TOKEN_CAP: usize = 64_000;

    /// The timeline `tl` was forked from, if it recorded one and that parent
    /// is still usable.
    ///
    /// A tombstoned or archived parent reads as absent rather than as an
    /// error: a chain link can be retired by its own layer's reconciliation
    /// long after a child recorded it, and a dangling pointer must degrade to
    /// "no inheritance" instead of failing the child. A parent recorded as
    /// [`Self::SEED_PARENT`] is this member's seed; one recorded as
    /// [`Self::UNIT_PARENT`] is the unit holding that content key.
    pub fn parent_timeline(&self, tl: TimelineId) -> Option<TimelineId> {
        let recorded = self.timelines.get(&tl)?.custom.get(Self::FORKED_FROM_KEY)?;
        let parent = match recorded.as_str() {
            Self::SEED_PARENT => self.seed?,
            unit if unit.starts_with(Self::UNIT_PARENT) => {
                self.unit_holding(&unit[Self::UNIT_PARENT.len()..])?
            }
            raw => TimelineId::from_raw(raw.parse().ok()?)?,
        };
        if self.tombstoned_timelines.contains(&parent) {
            return None;
        }
        if !self
            .timelines
            .get(&parent)
            .map(|e| !e.archived)
            .unwrap_or(false)
        {
            return None;
        }
        Some(parent)
    }

    /// Every conversation whose recorded parent is a **base** like `base`
    /// rather than [`Self::SEED_PARENT`]: a timeline on `base`'s own layer and
    /// group that holds no turn and no `conv_id` — a boot's base conversation,
    /// which every dialogue of that boot was parented on before the alias. The
    /// repair re-points them at the alias, so each opens on the chain this
    /// member holds now. `base` itself is among them when it names a base.
    pub fn parented_on_a_base(&self, base: TimelineId) -> Vec<TimelineId> {
        let Some(target) = self.timeline_target(base) else {
            return Vec::new();
        };
        let is_base = |tl: TimelineId| {
            self.timelines.get(&tl).is_some_and(|e| {
                (e.layer, e.group) == target && e.conv_id.is_none() && e.turns.is_empty()
            }) && !self.tombstoned_timelines.contains(&tl)
        };
        self.timelines
            .iter()
            .filter(|(_, e)| {
                e.custom
                    .get(Self::FORKED_FROM_KEY)
                    .and_then(|raw| raw.parse().ok())
                    .and_then(TimelineId::from_raw)
                    .is_some_and(is_base)
            })
            .map(|(tl, _)| *tl)
            .collect()
    }

    /// Whether `ancestor` is `tl` or lies anywhere on its `forked_from`
    /// lineage. Unlike [`Self::inherited_chain`] the walk is not capped by the
    /// token budget: it answers whether recording a parent would close a
    /// cycle, and an ancestor past the cap closes one just the same.
    pub fn lineage_contains(&self, tl: TimelineId, ancestor: TimelineId) -> bool {
        let mut seen: HashSet<TimelineId> = HashSet::from([tl]);
        let mut cursor = tl;
        loop {
            if cursor == ancestor {
                return true;
            }
            match self.parent_timeline(cursor) {
                Some(parent) if seen.insert(parent) => cursor = parent,
                _ => return false,
            }
        }
    }

    // ── working sets ────────────────────────────────────────────────────────

    /// Whether `timeline` can deliver its content into `target`'s projection.
    ///
    /// A pinned item is a promise made to the model — "already in context" —
    /// so a tombstoned, archived or empty conversation must read as a miss and
    /// send the call to a real read: being told it has a file it does not have
    /// is worse than reading the file twice. A conversation never carries
    /// itself.
    ///
    /// **Nor does one that is not whole** ([`Self::holds_whole`]). The
    /// projection skips an empty turn — a gap still to arrive from another
    /// member, or a retired turn — so such a conversation, locked, would tell
    /// the model a file is in context and place none of what is missing.
    ///
    /// A calibration ([`Self::CALIBRATES_KEY`]) never is: its questions stand
    /// in the gallery for the conversation they calibrate, which is the one
    /// delivered.
    pub(super) fn deliverable(&self, target: TimelineId, timeline: TimelineId) -> bool {
        timeline != target
            && !self.tombstoned_timelines.contains(&timeline)
            && self.timelines.get(&timeline).is_some_and(|e| !e.archived)
            && self.calibrates(timeline).is_none()
            && self.holds_whole(timeline)
            && Substrate::turn_count(self, timeline) > 0
    }

    /// `target`'s working set, when it has one.
    pub fn working_set(&self, target: TimelineId) -> Option<&WorkingSet> {
        self.working_sets.get(&target).map(|ws| &**ws)
    }

    /// `target`'s working set in this version, made if absent — this
    /// version's own copy, so older versions keep theirs.
    pub(super) fn working_set_mut(&mut self, target: TimelineId) -> &mut WorkingSet {
        Arc::make_mut(self.working_sets.entry(target).or_default())
    }

    /// Record which share of the working-set budget `group`'s conversations
    /// draw on. Idempotent.
    pub fn set_working_set_share(&mut self, group: GroupId, share: WorkingSetShare) {
        self.working_set_shares.insert(group, share);
    }

    /// Cap each of `types` at its share of a working set's budget, a
    /// conversation's type read from the subject its metadata names under
    /// `subject_key`.
    pub fn set_working_set_types(&mut self, types: TypeShares, subject_key: &str) {
        self.working_set_types = Some(Arc::new((types, subject_key.to_string())));
    }

    /// Record that `drawing` places units of `from`'s conversations: its
    /// candidates are `from`'s, as `drawing`'s own retrieval scope names them.
    /// Both are scoped, and the embedder gives each target scopes that do not
    /// overlap, so a unit is placed through one group or the other. Idempotent.
    pub fn set_group_draws(&mut self, drawing: GroupId, from: GroupId) {
        if self.drawing_groups.insert(drawing, from) != Some(from) {
            self.group_content_changed(drawing);
        }
    }

    /// The group whose registered conversations `group`'s candidates are: its
    /// source for a drawing group, itself otherwise.
    pub(super) fn holding_group(&self, group: GroupId) -> GroupId {
        self.drawing_groups.get(&group).copied().unwrap_or(group)
    }

    /// The group `timeline` is placed through in `target`'s projection: a
    /// drawing group of its own group whose scope offers it to `target`, or
    /// else the group it is registered in. `None` for an unknown timeline.
    pub fn placing_group(&self, target: TimelineId, timeline: TimelineId) -> Option<GroupId> {
        let (_, owner) = self.timeline_target(timeline)?;
        let drawn = self
            .drawing_groups
            .iter()
            .filter(|(_, from)| **from == owner)
            .map(|(drawing, _)| *drawing)
            .find(|drawing| self.scope_offers(*drawing, target, timeline));
        Some(drawn.unwrap_or(owner))
    }

    /// What admitting `timeline` into `target`'s working set needs: its
    /// recorded size and its share — or `None` when it cannot be a member at
    /// all: it cannot deliver ([`Self::deliverable`]), it is a link of
    /// `target`'s inherited chain (projected as the conversation's own opening
    /// already — admitted again it emitted twice: `AGENTS.md`'s 46K characters
    /// stood in both the working set and the chain), or the scope of its group
    /// no longer offers it to `target`.
    pub(super) fn working_set_candidate(
        &self,
        target: TimelineId,
        timeline: TimelineId,
    ) -> Option<Candidate> {
        let chain: HashSet<TimelineId> = self.inherited_chain(target).into_iter().collect();
        self.working_set_candidate_beside(target, &chain, timeline)
    }

    /// [`Self::working_set_candidate`], with `target`'s inherited chain
    /// walked once by the caller: one observe judges every scored
    /// conversation against the same chain, and walking it per candidate cost
    /// a `Vec`, a `HashSet` and the parent walk each time.
    pub(super) fn working_set_candidate_beside(
        &self,
        target: TimelineId,
        chain: &HashSet<TimelineId>,
        timeline: TimelineId,
    ) -> Option<Candidate> {
        if !self.deliverable(target, timeline) {
            return None;
        }
        if chain.contains(&timeline) {
            return None;
        }
        let group = self.placing_group(target, timeline);
        if group.is_some_and(|g| !self.scope_offers(g, target, timeline)) {
            return None;
        }
        let kind = self.working_set_types.as_ref().and_then(|types| {
            let (types, key) = &**types;
            types.classify(self.custom_of(timeline)?.get(key)?)
        });
        Some(Candidate {
            tokens: self.timeline_token_totals.get(&timeline).copied(),
            share: group.and_then(|g| self.working_set_shares.get(&g).copied()),
            kind,
        })
    }

    /// Seed `target`'s pool with `candidates` at `momentum`
    /// ([`WorkingSet::seed`]), under this card's pool size. Returns the
    /// candidates that did not enter.
    pub fn working_set_seed(
        &mut self,
        target: TimelineId,
        candidates: &[TimelineId],
        momentum: f32,
    ) -> Vec<TimelineId> {
        let mut refused = Vec::new();
        let mut admissible = Vec::new();
        for &timeline in candidates {
            match self.working_set_candidate(target, timeline) {
                Some(candidate) => admissible.push((timeline, candidate)),
                None => refused.push(timeline),
            }
        }
        let pool = self.pool_size;
        refused.extend(
            self.working_set_mut(target)
                .seed(&admissible, pool, momentum),
        );
        refused
    }

    /// Keep `reads` from entering `target`'s working set
    /// ([`WorkingSet::withhold`]).
    pub fn working_set_withhold(&mut self, target: TimelineId, reads: &[TimelineId]) {
        if !reads.is_empty() {
            self.working_set_mut(target).withhold(reads.iter().copied());
        }
    }

    /// A boundary of `target`'s slot, which holds `placed` now
    /// ([`WorkingSet::boundary`]). Nothing for a target with no working set.
    pub fn working_set_boundary(&mut self, target: TimelineId, placed: &[TurnKey]) {
        if let Some(ws) = arc_mut(&mut self.working_sets, &target) {
            ws.boundary(placed.iter().copied());
        }
    }

    /// A new question on `target`: its withheld conversations may enter again
    /// and its next boundary is the question's opening
    /// ([`WorkingSet::clear_withheld`]). Made if absent — a dialogue opens with
    /// its set cleared away, and its first question still opens.
    pub fn working_set_clear_withheld(&mut self, target: TimelineId) {
        self.working_set_mut(target).clear_withheld();
    }

    /// Fold one reprojection's per-piece fresh scores into `target`'s pool
    /// ([`WorkingSet::observe`]) under this card's pool size, after the
    /// stride's hit densities ([`WorkingSet::observe_hits`]) — so what the
    /// indexer read since the last reprojection lifts every judgment this one
    /// makes.
    pub fn working_set_observe(
        &mut self,
        target: TimelineId,
        fresh: &HashMap<TimelineId, f32>,
        hit_density: &HashMap<TimelineId, f32>,
        hits: &HitsConfig,
        beta: f32,
        min_momentum: f32,
    ) {
        let mut ws = self
            .working_sets
            .remove(&target)
            .map(Arc::unwrap_or_clone)
            .unwrap_or_default();
        ws.observe_hits(hit_density, hits);
        let chain: HashSet<TimelineId> = self.inherited_chain(target).into_iter().collect();
        ws.observe(fresh, beta, min_momentum, self.pool_size, &|timeline| {
            self.working_set_candidate_beside(target, &chain, timeline)
        });
        self.working_sets.insert(target, Arc::new(ws));
    }

    /// Drop `target`'s working set — the conversation is going away, or its
    /// set is about to be rebuilt from its own history.
    pub fn working_set_clear(&mut self, target: TimelineId) {
        self.working_sets.remove(&target);
    }

    // ── retrieval scope ─────────────────────────────────────────────────────

    /// Offer `group`'s conversations to a projection only as its target's
    /// scope names them. Idempotent.
    pub fn mark_group_scoped(&mut self, group: GroupId) {
        if self.scoped_groups.insert(group).is_none() {
            self.group_content_changed(group);
        }
    }

    /// Whether `group` offers only what a target's scope names.
    pub fn is_group_scoped(&self, group: GroupId) -> bool {
        self.scoped_groups.contains(&group)
    }

    /// Name the conversations the scoped `group` may offer `target`,
    /// replacing whatever it named before.
    pub fn set_retrieval_scope(
        &mut self,
        target: TimelineId,
        group: GroupId,
        allowed: Arc<HashSet<TimelineId>>,
    ) {
        let digest = scope_digest_of(&allowed);
        let before = self
            .scope_digests
            .entry(target)
            .or_default()
            .insert(group, digest);
        self.retrieval_scopes
            .entry(target)
            .or_default()
            .insert(group, allowed);
        // A scope naming the same conversations changes nothing a scan reads:
        // a target's scope is set again on every request.
        if before != Some(digest) {
            self.group_content_changed(group);
        }
    }

    /// Forget every scope `target` was given — it is going away.
    pub fn clear_retrieval_scope(&mut self, target: TimelineId) {
        self.retrieval_scopes.remove(&target);
        if let Some(scoped) = self.scope_digests.remove(&target) {
            for group in scoped.keys().copied().collect::<Vec<_>>() {
                self.group_content_changed(group);
            }
        }
    }

    /// What `group` offers `target`, as a digest two targets compare: `None`
    /// for a group that is not scoped — it offers every target the same — and
    /// otherwise the digest of the target's scope, [`EMPTY_SCOPE_DIGEST`] when
    /// it has none and the group offers it nothing.
    pub fn scope_digest(&self, group: GroupId, target: TimelineId) -> Option<u64> {
        self.scoped_groups.contains(&group).then(|| {
            self.scope_digests
                .get(&target)
                .and_then(|by_group| by_group.get(&group))
                .copied()
                .unwrap_or(EMPTY_SCOPE_DIGEST)
        })
    }

    /// [`Self::active_timelines_for_group`] as `target` may see it: all of
    /// them for a group that is not scoped, only those `target`'s scope names
    /// for one that is — and none when it names none.
    pub fn scoped_timelines_for_group(
        &self,
        group: GroupId,
        target: TimelineId,
    ) -> impl Iterator<Item = TimelineId> + '_ {
        let scope: Option<Option<&Arc<HashSet<TimelineId>>>> =
            self.scoped_groups.contains(&group).then(|| {
                self.retrieval_scopes
                    .get(&target)
                    .and_then(|by_group| by_group.get(&group))
            });
        self.active_timelines_for_group(group)
            .filter(move |tl| match scope {
                None => true,
                Some(None) => false,
                Some(Some(allowed)) => allowed.contains(tl),
            })
    }

    /// Whether `group` may offer `timeline` to `target` — the scope test
    /// [`Self::scoped_timelines_for_group`] applies, for one conversation.
    pub fn scope_offers(&self, group: GroupId, target: TimelineId, timeline: TimelineId) -> bool {
        if !self.scoped_groups.contains(&group) {
            return true;
        }
        self.retrieval_scopes
            .get(&target)
            .and_then(|by_group| by_group.get(&group))
            .is_some_and(|allowed| allowed.contains(&timeline))
    }

    /// `tl` together with the ancestors it inherits, **oldest first**.
    ///
    /// This is what "my own history" means once conversations fork from one
    /// another: the turns of every ancestor, in the order they were read, then
    /// the target's own. Oldest-first is what makes the concatenation
    /// chronological — and it is also `TurnKey`'s own ordering, since an
    /// ancestor is always minted before the child that forks from it.
    ///
    /// The target itself is always present, even when it alone exceeds the
    /// budget: a conversation never loses its own turns to an inherited one.
    pub fn inherited_chain(&self, tl: TimelineId) -> Vec<TimelineId> {
        let mut chain = vec![tl];
        let mut seen: HashSet<TimelineId> = HashSet::from([tl]);
        let mut budget = Self::INHERITED_CHAIN_TOKEN_CAP;
        let mut cursor = tl;
        while let Some(parent) = self.parent_timeline(cursor) {
            // A cycle is not reachable by construction (a fork's parent always
            // predates it), but the pointer is durable, user-visible metadata
            // and a walk that trusted it could hang the projection.
            if !seen.insert(parent) {
                tracing::warn!(
                    timeline = tl.raw(),
                    repeated = parent.raw(),
                    "inherited chain revisits a timeline — cycle in `forked_from`, truncating",
                );
                break;
            }
            let cost = self
                .timeline_token_totals
                .get(&parent)
                .copied()
                .unwrap_or(0);
            if cost > budget {
                tracing::warn!(
                    timeline = tl.raw(),
                    dropped_at = parent.raw(),
                    inherited = chain.len() - 1,
                    cost,
                    budget,
                    cap = Self::INHERITED_CHAIN_TOKEN_CAP,
                    "inherited chain hit its token cap — older ancestors dropped from the front",
                );
                break;
            }
            budget -= cost;
            chain.push(parent);
            cursor = parent;
        }
        chain.reverse();
        chain
    }
}
