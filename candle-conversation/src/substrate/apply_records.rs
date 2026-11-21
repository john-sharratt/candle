//! Folds each replayed or replicated record into the substrate's in-RAM state,
//! so the greatest stamp on every value wins whatever order the records arrive.
//!
//! Every value a record sets is held by the greatest stamp that set it, so two
//! members writing the same stream, label or snapshot converge on one answer in
//! any order. A declaration that ranks below the one held changes nothing, and a
//! record for a timeline not registered yet waits until that timeline's first
//! declaration arrives. A section generation a tombstone has ended is refused
//! wherever it arrives, so a late record cannot bring back dropped content.

use super::*;

impl Substrate {
    /// Install a stream declaration written as record `id`. The declaration
    /// with the greatest stamp is the stream's (§6.5); one ranked below the
    /// declaration held changes nothing.
    ///
    /// A section declaration starts a **generation** of the section, named by
    /// its writer and its `oseq` ([`SectionGeneration`]). One a tombstone has
    /// ended is refused, whenever it arrives; one that is installed drops
    /// whatever its stream still held of an earlier generation. An unstamped
    /// section declaration lifts an unstamped tombstone before it, as the log
    /// order of unstamped records always has.
    pub fn apply_stream_decl(&mut self, stream_id: StreamId, decl: StreamDecl, id: RecordIdentity) {
        // Turn decls register their timeline whether or not they win the
        // stream: registration is a set insert, so it holds in any order, and
        // the records held for the timeline until now are applied by it.
        if let StreamDecl::Turn(t) = &decl {
            if let (Some(timeline), Some(layer), Some(group)) = (
                TimelineId::from_raw(t.timeline_id),
                LayerId::from_raw(t.layer_id),
                GroupId::from_raw(t.group_id),
            ) {
                self.register_timeline(timeline, layer, group);
            }
        }
        let key = StreamKey::of(stream_id, &id);
        let turn_tags: Option<Vec<String>> = match &decl {
            StreamDecl::Turn(t) => Some(t.tags.clone()),
            _ => None,
        };
        let section = matches!(decl, StreamDecl::PromptSection(_));
        let generation = SectionGeneration {
            writer: key.writer,
            oseq: id.oseq,
        };
        if section && !id.is_none() && self.ended_sections.is_ended(stream_id, generation) {
            return;
        }
        // The turn the stream held before, whose timeline a redeclaration
        // takes it from.
        let left = self.stream_turn_timeline(key);
        // A memo lives on the physical stream it was decoded from, so a
        // declaration moving a turn's position leaves nothing stale behind.
        let Some(Declared { displaced, .. }) = self.streams.declare(key, decl, id) else {
            return;
        };
        if let Some(timeline) = self
            .streams
            .turn(key)
            .and_then(|t| TimelineId::from_raw(t.decl.timeline_id))
        {
            if displaced
                .iter()
                .any(|&old| self.holds_turn(timeline, TurnIndex(old)))
            {
                self.reload_due = true;
            }
            // Which stream sits at each of the timeline's positions may have
            // moved: its gallery reads them by position.
            self.gallery_changed(timeline);
            self.timeline_content_changed(timeline);
        }
        // A turn redeclared elsewhere, or as no turn, leaves its timeline.
        if let Some(old) = left.filter(|&old| Some(old) != self.stream_turn_timeline(key)) {
            self.gallery_changed(old);
            self.timeline_content_changed(old);
        }
        if section {
            if id.is_none() {
                self.ended_sections.lift_unstamped(stream_id);
            } else {
                self.streams
                    .prune(key, |held| held.epoch >= generation.oseq);
            }
        }
        // Only a turn is in a gather scope or among the dialogue; a stream
        // redeclared as anything else leaves both.
        match &turn_tags {
            Some(tags) => self.tagged_turns.set(key, tags),
            None => self.tagged_turns.remove(key),
        }
        self.gallery_epoch += 1;
    }

    /// Whether a value written as record `id` belongs on `key`'s stream: not
    /// when it carries, in its `epoch`, a section generation a tombstone ended
    /// or one older than the generation the stream is declared in. An
    /// unstamped record always does.
    pub(super) fn admits(&self, key: StreamKey, id: &RecordIdentity) -> bool {
        if id.is_none() {
            return true;
        }
        let generation = SectionGeneration {
            writer: key.writer,
            oseq: id.epoch,
        };
        if self.ended_sections.is_ended(key.stream_id, generation) {
            return false;
        }
        match self.streams.physical(key) {
            Some(StreamRuntime {
                decl: Some(StreamDecl::PromptSection(_)),
                ids,
                ..
            }) if !ids.decl.is_none() => id.epoch >= ids.decl.oseq,
            _ => true,
        }
    }

    /// Every stream declared as a turn carrying one of `tags`, in stream-key
    /// order — the members a collection tagged `tags` draws its gallery from.
    /// See the `tagged_turns` field.
    pub fn tagged_turn_streams(
        &self,
        tags: &[String],
    ) -> impl Iterator<Item = TurnStream<'_>> + '_ {
        self.tagged_turns
            .with_any(tags)
            .into_iter()
            .filter_map(|key| self.streams.turn(key))
    }

    /// Every stream declared as a turn with no gather tag, in stream-key order
    /// — the members an untagged collection draws its gallery from. See the
    /// `tagged_turns` field.
    pub fn dialogue_turn_streams(&self) -> impl Iterator<Item = TurnStream<'_>> + '_ {
        self.tagged_turns
            .dialogue()
            .filter_map(|key| self.streams.turn(key))
    }

    /// Every declared turn, each at its position in its timeline.
    pub fn all_turn_streams(&self) -> impl Iterator<Item = TurnStream<'_>> + '_ {
        self.streams
            .iter_physical()
            .filter_map(|(key, _)| self.streams.turn(key))
    }

    /// Record the location of chunk `idx` of `stream_id`, written as record
    /// `id`: the greatest stamp holds the index, and a copy of the record that
    /// holds it (a relocation keeps its identity) moves it.
    pub fn apply_chunk_loc(
        &mut self,
        stream_id: StreamId,
        idx: u64,
        loc: ChunkLoc,
        id: RecordIdentity,
    ) {
        let key = StreamKey::of(stream_id, &id);
        if !self.admits(key, &id) {
            return;
        }
        let entry = self.streams.entry(key);
        if take_keyed(&mut entry.ids.chunks, idx, id) {
            entry.chunks.insert(idx, loc);
        }
    }

    /// Record the location of `stream_id`'s `Tokens` record `id` — the greatest
    /// stamp holds it.
    pub fn apply_tokens_loc(&mut self, stream_id: StreamId, loc: RecordLoc, id: RecordIdentity) {
        let key = StreamKey::of(stream_id, &id);
        if !self.admits(key, &id) {
            return;
        }
        let entry = self.streams.entry(key);
        if take(&mut entry.ids.tokens, id) {
            entry.tokens = Some(loc);
        }
    }

    /// Record the location of a snapshot stream's recurrent-state record `id`
    /// — the greatest stamp is the stream's tail. The location it replaces was
    /// credited dead by the accounting layer when this one appended.
    ///
    /// Whether the record became the tail: `false` when the stream is retired
    /// (its timeline tombstoned or distilled) or a record that outranks it is
    /// already held. Only a record that became the tail is what
    /// [`Self::recurrent_snapshot_loc`] serves, so only that one may stand in
    /// for state a slot drops.
    pub fn apply_snapshot_loc(
        &mut self,
        stream_id: StreamId,
        loc: RecordLoc,
        id: RecordIdentity,
    ) -> bool {
        if self.retired_snapshots.contains(&stream_id) {
            return false;
        }
        if take(self.snapshot_ids.entry(stream_id).or_default(), id) {
            self.recurrent_snapshots.insert(stream_id, loc);
            return true;
        }
        false
    }

    /// A snapshot retraction, record `id`: when it outranks the stream's tail
    /// the tail is gone, and every snapshot it outranks that arrives later is
    /// refused (§6.5 — an empty payload is a value meaning *retracted*).
    pub fn retract_snapshot(&mut self, stream_id: StreamId, id: RecordIdentity) {
        if self.retired_snapshots.contains(&stream_id) {
            return;
        }
        if take(self.snapshot_ids.entry(stream_id).or_default(), id) {
            self.recurrent_snapshots.remove(&stream_id);
        }
    }

    /// The live snapshot for a snapshot stream, if one exists. The caller
    /// derives `stream_id` from the timeline it is resuming
    /// (`snapshot_stream_id`); payload validation — schedule hash, turn
    /// index vs the recovered tail — happens at materialization.
    pub fn recurrent_snapshot_loc(&self, stream_id: StreamId) -> Option<RecordLoc> {
        self.recurrent_snapshots.get(&stream_id).copied()
    }

    /// Forget a snapshot stream entirely, its tail and the record behind it —
    /// what a timeline tombstone or a distillation does to the conversation's
    /// recurrent state, and what a retraction that could not be written leaves.
    /// A retraction that was written goes through [`Self::retract_snapshot`]
    /// instead, which keeps its identity.
    pub fn drop_snapshot_loc(&mut self, stream_id: StreamId) {
        self.recurrent_snapshots.remove(&stream_id);
        self.snapshot_ids.remove(&stream_id);
    }

    /// Retire `timeline`'s snapshot stream for good — what its tombstone or
    /// its distillation does to the conversation's recurrent state. Its tail
    /// goes, and no snapshot of it is installed again, whenever it arrives.
    pub(super) fn retire_snapshot_stream(&mut self, timeline: TimelineId) {
        let stream_id = snapshot_stream_id(timeline.raw());
        if self.retired_snapshots.insert(stream_id).is_none() {
            // Named, so a slot that later finds its record gone can be traced
            // to what retired it from the log alone.
            tracing::info!(
                target: "candle_conversation::substrate::snapshots",
                timeline = timeline.raw(),
                had_tail = self.recurrent_snapshots.contains_key(&stream_id),
                "recurrent snapshot stream retired"
            );
        }
        self.drop_snapshot_loc(stream_id);
    }

    /// Every snapshot stream whose current record is a retraction, with the
    /// retraction's identity, by stream id. A retraction is a value (§6.5): a
    /// rewrite carries it, so a snapshot it outranks that arrives later is
    /// still refused.
    pub fn retracted_snapshots(&self) -> Vec<(StreamId, RecordIdentity)> {
        let mut out: Vec<(StreamId, RecordIdentity)> = self
            .snapshot_ids
            .iter()
            .filter(|(sid, _)| !self.recurrent_snapshots.contains_key(sid))
            .map(|(sid, id)| (*sid, *id))
            .collect();
        out.sort_unstable_by_key(|(sid, _)| sid.0);
        out
    }

    /// Every live snapshot `(stream id, location)` — the compactor's source
    /// for carrying the tails forward.
    pub fn recurrent_snapshot_entries(&self) -> impl Iterator<Item = (StreamId, RecordLoc)> + '_ {
        self.recurrent_snapshots.iter().map(|(k, v)| (*k, *v))
    }

    /// Record the location of a **prompt branch**'s checkpoint record `id` —
    /// the greatest stamp holds the branch, exactly as for a snapshot.
    pub fn apply_branch_checkpoint_loc(
        &mut self,
        stream_id: StreamId,
        loc: RecordLoc,
        id: RecordIdentity,
    ) {
        if take(self.branch_checkpoint_ids.entry(stream_id).or_default(), id) {
            self.branch_checkpoints.insert(stream_id, loc);
        }
    }

    /// The record behind a prompt branch's checkpoint.
    pub fn branch_checkpoint_id(&self, stream_id: StreamId) -> RecordIdentity {
        self.branch_checkpoint_ids
            .get(&stream_id)
            .copied()
            .unwrap_or_default()
    }

    /// The record behind a snapshot stream's tail.
    pub fn snapshot_id(&self, stream_id: StreamId) -> RecordIdentity {
        self.snapshot_ids
            .get(&stream_id)
            .copied()
            .unwrap_or_default()
    }

    /// The live checkpoint for a prompt branch, if one exists. The caller
    /// derives `stream_id` from the branch's content prefix.
    pub fn branch_checkpoint_loc(&self, stream_id: StreamId) -> Option<RecordLoc> {
        self.branch_checkpoints.get(&stream_id).copied()
    }

    /// Every live branch checkpoint `(stream id, location)`.
    pub fn branch_checkpoint_entries(&self) -> impl Iterator<Item = (StreamId, RecordLoc)> + '_ {
        self.branch_checkpoints.iter().map(|(k, v)| (*k, *v))
    }

    /// Record that the stream is durably committed through `through_index`,
    /// by `Commit` record `id`. The greatest index stated holds — a max, never
    /// an overwrite (§6.5) — and of the records stating it, the greatest stamp
    /// is the one a rewrite writes back.
    pub fn apply_commit_through(
        &mut self,
        stream_id: StreamId,
        through_index: u64,
        id: RecordIdentity,
    ) {
        let key = StreamKey::of(stream_id, &id);
        if !self.admits(key, &id) {
            return;
        }
        let entry = self.streams.entry(key);
        match entry.committed_through {
            Some(held) if through_index < held => {}
            Some(held) if through_index == held => {
                take(&mut entry.ids.commit, id);
            }
            _ => {
                entry.committed_through = Some(through_index);
                entry.ids.commit = id;
            }
        }
    }

    /// Record `stream_id`'s projection events, record `id` — the greatest
    /// stamp holds them.
    pub fn apply_projection_events(
        &mut self,
        stream_id: StreamId,
        payload: Vec<u8>,
        id: RecordIdentity,
    ) {
        let key = StreamKey::of(stream_id, &id);
        if !self.admits(key, &id) {
            return;
        }
        let entry = self.streams.entry(key);
        if !take(&mut entry.ids.projection_events, id) {
            return;
        }
        entry.projection_events = Some(payload.into());
        // The memo belonged to the blob just replaced. `entry` is this
        // version's own copy of the runtime, so older versions keep theirs.
        entry.events_digest = OnceLock::new();
        self.gallery_epoch += 1;
        // The turn's seams are read out of its events.
        self.stream_gallery_changed(key);
        self.stream_content_changed(key);
    }

    /// Record `stream_id`'s wide-Q signature window, record `id` — the greatest
    /// stamp holds it.
    pub fn apply_wide_q_sigs(&mut self, stream_id: StreamId, payload: Vec<u8>, id: RecordIdentity) {
        let key = StreamKey::of(stream_id, &id);
        if !self.admits(key, &id) {
            return;
        }
        let entry = self.streams.entry(key);
        if !take(&mut entry.ids.wide_q_sigs, id) {
            return;
        }
        let replaced = entry.wide_q_sigs.replace(payload.into()).is_some();
        self.sig_rewrites += u64::from(replaced);
        self.gallery_epoch += 1;
        self.stream_gallery_changed(key);
        self.stream_content_changed(key);
    }

    /// Record the QSA index page of a turn or a prompt section, record `id` —
    /// the greatest stamp holds it.
    pub fn apply_index_page(
        &mut self,
        stream_id: StreamId,
        payload: impl Into<Arc<[u8]>>,
        id: RecordIdentity,
    ) {
        let key = StreamKey::of(stream_id, &id);
        if !self.admits(key, &id) {
            return;
        }
        let entry = self.streams.entry(key);
        if take(&mut entry.ids.index_page, id) {
            entry.index_page = Some(IndexPage::Resident(Arc::new(StoredPages::new(payload))));
        }
    }

    /// A held page has become durable at `loc`: hold its location instead of
    /// its bytes. Only the page record `id` itself — a newer page that took the
    /// stream meanwhile is not the one written, and stays as it is.
    pub fn settle_index_page(&mut self, stream_id: StreamId, loc: RecordLoc, id: RecordIdentity) {
        let key = StreamKey::of(stream_id, &id);
        if self.streams.physical(key).is_none() {
            return;
        }
        let entry = self.streams.entry(key);
        if entry.ids.index_page == id && matches!(entry.index_page, Some(IndexPage::Resident(_))) {
            entry.index_page = Some(IndexPage::Stored(loc));
        }
    }

    /// [`Self::apply_index_page`] for a page left in the log at `loc` — what
    /// recovery and a relocation record ([`IndexPage::Stored`]).
    pub fn apply_index_page_loc(
        &mut self,
        stream_id: StreamId,
        loc: RecordLoc,
        id: RecordIdentity,
    ) {
        let key = StreamKey::of(stream_id, &id);
        if !self.admits(key, &id) {
            return;
        }
        let entry = self.streams.entry(key);
        if take(&mut entry.ids.index_page, id) {
            entry.index_page = Some(IndexPage::Stored(loc));
        }
    }

    // ── Per-timeline LWW payload applicators (decode + dispatch) ────────

    /// Apply `Label` record `id` — a timeline's `conv_id`, `label` and `custom`
    /// keys, any of which it may leave empty. **Each field keeps the value of
    /// the greatest record that set it** (§6.5), so two members renaming a
    /// conversation and tagging it at once both land, whatever order their
    /// records arrive in. A timeline not registered yet holds the record until
    /// its first turn declaration arrives.
    pub fn apply_conv_meta(&mut self, timeline_raw: u64, meta: &ConvMeta, id: RecordIdentity) {
        let Some(timeline) = TimelineId::from_raw(timeline_raw) else {
            return;
        };
        // The file this timeline calibrated before the record, which a
        // re-pointed calibration leaves.
        let calibrated = self.calibrates(timeline);
        let Some(entry) = entry_mut(&mut self.timelines, &timeline) else {
            self.pending
                .hold(timeline_raw, PendingChild::Label(meta.clone()), id);
            return;
        };
        let mut custom_changed = false;
        if !meta.conv_id.is_empty() {
            if take(&mut entry.ids.conv_id, id) {
                entry.conv_id = Some(meta.conv_id.clone());
            }
            Self::note_named(entry, id, &meta.conv_id, &mut self.conv_order_counter);
        }
        if !meta.label.is_empty() && take(&mut entry.ids.label, id) {
            entry.label = Some(meta.label.clone());
        }
        for (k, v) in &meta.custom {
            if take_keyed(&mut entry.ids.custom, k.clone(), id) {
                // A file's path names its kind and a calibration names its
                // unit: what an assembled group scan was built from.
                custom_changed = true;
                let before = entry.custom.insert(k.clone(), v.clone());
                if k == Self::CALIBRATES_KEY {
                    Self::index_calibration(
                        &mut self.calibrations_of,
                        timeline,
                        before.as_deref(),
                        v,
                    );
                }
                if k == Self::CONTENT_KEY {
                    Self::index_unit(&mut self.units_by_key, timeline, before.as_deref(), v);
                }
            }
        }
        if custom_changed {
            // Its own groups and its calibrated file's as it now stands, and
            // the file it calibrated before.
            self.timeline_content_changed(timeline);
            if let Some(unit) = calibrated.filter(|&u| Some(u) != self.calibrates(timeline)) {
                self.timeline_content_changed(unit);
            }
        }
    }

    /// Move `calibration` in `index` from the unit `before` named to the one
    /// `now` names.
    pub(super) fn index_calibration(
        index: &mut PMap<TimelineId, PVec<TimelineId>>,
        calibration: TimelineId,
        before: Option<&str>,
        now: &str,
    ) {
        let unit = |v: &str| v.parse().ok().and_then(TimelineId::from_raw);
        if let Some(old) = before.and_then(unit) {
            if let Some(list) = index.get_mut(&old) {
                list.retain(|&t| t != calibration);
            }
        }
        if let Some(new) = unit(now) {
            let list = index.entry(new).or_default();
            if let Err(at) = list.binary_search(&calibration) {
                list.insert(at, calibration);
            }
        }
    }

    /// Move `unit` in `index` from the content key `before` to `now`.
    pub(super) fn index_unit(
        index: &mut PMap<String, PVec<TimelineId>>,
        unit: TimelineId,
        before: Option<&str>,
        now: &str,
    ) {
        if let Some(old) = before {
            if let Some(list) = index.get_mut(old) {
                list.retain(|&t| t != unit);
            }
        }
        let list = index.entry(now.to_string()).or_default();
        if let Err(at) = list.binary_search(&unit) {
            list.insert(at, unit);
        }
    }

    /// The live, unarchived timeline read under the content key `key`, the
    /// oldest when more than one is: two members that read the same unit at
    /// once each hold one, and every member must resolve a parent named by
    /// that key to the same conversation, whichever of the two survives.
    pub fn unit_holding(&self, key: &str) -> Option<TimelineId> {
        self.units_by_key.get(key)?.iter().copied().find(|tl| {
            !self.tombstoned_timelines.contains(tl)
                && self.timelines.get(tl).is_some_and(|e| {
                    !e.archived && e.custom.get(Self::CONTENT_KEY).map(String::as_str) == Some(key)
                })
        })
    }

    /// A record `id` named `entry`'s `conv_id`: rank the conversation by the
    /// least record that did. A stamped record ranks by its clock reading,
    /// which every member reads the same; a conversation named only by records
    /// without an identity is counted in the order the walk met them, which
    /// for such a log is the order they were written.
    pub(super) fn note_named(
        entry: &mut TimelineEntry,
        id: RecordIdentity,
        conv_id: &str,
        counter: &mut u64,
    ) {
        match entry.ids.named {
            None => {
                entry.order = if id.is_none() {
                    *counter += 1;
                    *counter
                } else {
                    id.hlc
                };
            }
            Some(held) if !id.is_none() && (held.is_none() || id.stamp() < held.stamp()) => {
                entry.order = id.hlc;
            }
            Some(_) => return,
        }
        entry.ids.named = Some(id);
        entry.ids.named_conv_id = conv_id.to_string();
    }

    /// Apply `ConvState` record `id` — the whole state, so the greatest stamp
    /// holds all of it. A timeline not registered yet holds the record until
    /// its first turn declaration arrives.
    pub fn apply_conv_state(&mut self, timeline_raw: u64, state: ConvState, id: RecordIdentity) {
        let Some(timeline) = TimelineId::from_raw(timeline_raw) else {
            return;
        };
        let Some(entry) = entry_mut(&mut self.timelines, &timeline) else {
            self.pending
                .hold(timeline_raw, PendingChild::State(state), id);
            return;
        };
        if take(&mut entry.ids.state, id) {
            self.replace_conv_state(timeline, state);
        }
    }

    /// Install `state` as `timeline`'s whole conversation state.
    pub(super) fn replace_conv_state(&mut self, timeline: TimelineId, state: ConvState) {
        if let Some(entry) = entry_mut(&mut self.timelines, &timeline) {
            entry.archived = state.archived;
            entry.branches = state.branches.into_iter().collect();
            entry.active = state.active;
            self.conv_active_counter = self.conv_active_counter.max(state.active);
        }
    }

    /// Apply `TreeMetadata` record `id` — the turn's per-node forest meta
    /// (kind / children / level), the greatest stamp per turn. The peak set is
    /// derived, so there is no root marker to apply. A timeline not registered
    /// yet holds the record until it is.
    pub fn apply_tree_metadata_payload(
        &mut self,
        payload: &TreeMetadataPayload,
        id: RecordIdentity,
    ) {
        let Some(timeline) = TimelineId::from_raw(payload.timeline_id) else {
            return;
        };
        if !self.timelines.contains_key(&timeline) {
            self.pending.hold(
                payload.timeline_id,
                PendingChild::TreeMeta(payload.clone()),
                id,
            );
            return;
        }
        let kind = match payload.kind {
            0 => TurnKind::Normal,
            1 => TurnKind::SummaryOfTurns,
            2 => TurnKind::SummaryOfSummaries,
            _ => return,
        };
        let meta = TreeNodeMeta {
            kind,
            children: payload.children.iter().map(|c| TurnIndex(*c)).collect(),
            tree_height: payload.tree_height,
        };
        self.set_tree_meta(timeline, TurnIndex(payload.turn_index), meta, id);
    }

    /// Apply `DebugId` record `id` — the greatest stamp holds the timeline's
    /// resume key. A timeline not registered yet holds the record until it is.
    pub fn apply_debug_id_payload(&mut self, payload: &DebugIdPayload, id: RecordIdentity) {
        let Some(timeline) = TimelineId::from_raw(payload.timeline_id) else {
            return;
        };
        let Some(entry) = entry_mut(&mut self.timelines, &timeline) else {
            self.pending.hold(
                payload.timeline_id,
                PendingChild::DebugId(payload.debug_id.clone()),
                id,
            );
            return;
        };
        if take(&mut entry.ids.debug_id, id) {
            self.index_debug_id(timeline, payload.debug_id.clone());
        }
    }

    /// Apply custom-object record `id` — on replay and on the runtime write
    /// path. Keyed by the stable hash of its `key`, and the greatest stamp per
    /// key holds it.
    pub fn apply_custom_object(&mut self, obj: CustomObjectPayload, id: RecordIdentity) {
        let key = obj.stream_id();
        if take(self.custom_object_ids.entry(key).or_default(), id) {
            if !self.custom_object_keys.contains_key(&obj.key) {
                self.custom_object_keys.insert(obj.key.clone(), key);
            }
            self.install_chunk_exclusion(&obj);
            self.custom_objects.insert(key, Arc::new(obj));
            self.custom_objects_version += 1;
        }
    }
}
