//! Keeps each timeline's live conversation state: tool-round couplings, the
//! reasoning window of a request, summary-forest bookkeeping and the pending
//! summary queue.
//!
//! A tool round-trip is recorded as coupled turns, so the summary tree and the
//! selector read a call and its response as one exchange. A coupling replays as
//! a set insert, which commutes, and waits for its timeline if that is not
//! registered yet. A request's next turn keeps the reasoning of its newest rounds
//! only, and a finished request carries none into the next one. Summary-node
//! metadata is stamped per turn, and whether the forest is whole is computed from
//! its persisted nodes rather than stored as a flag.

use super::*;

impl Substrate {
    // ── tool-round-trip couplings ───────────────────────────────────────────

    /// Couple `from_turn` to the tool response that follows it, as
    /// `TurnCoupling` record `id`.
    ///
    /// Idempotent — replaying the same record, or coupling twice, adds nothing;
    /// of the records stating one coupling, the greatest stamp is the one a
    /// rewrite writes back. A coupling naming a timeline not registered yet is
    /// held until it is.
    pub fn couple_turn(&mut self, timeline: TimelineId, from_turn: u32, id: RecordIdentity) {
        self.apply_turn_coupling(
            &TurnCouplingPayload {
                timeline_id: timeline.raw(),
                from_turn,
            },
            id,
        );
    }

    /// Apply `TurnCoupling` record `id` — a set insert, which commutes.
    pub fn apply_turn_coupling(&mut self, payload: &TurnCouplingPayload, id: RecordIdentity) {
        let Some(timeline) = TimelineId::from_raw(payload.timeline_id) else {
            return;
        };
        let Some(tl) = entry_mut(&mut self.timelines, &timeline) else {
            self.pending.hold(
                payload.timeline_id,
                PendingChild::Coupling(payload.from_turn),
                id,
            );
            return;
        };
        tl.couplings.insert(payload.from_turn);
        take_keyed(&mut tl.ids.couplings, payload.from_turn, id);
    }

    /// The turns on `timeline` coupled to the tool response that follows them.
    /// Empty for a timeline that has never made a tool call — every turn is then
    /// its own exchange.
    /// The engine's trained window, as [`Self::set_trained_window`] last set
    /// it.
    pub fn trained_window(&self) -> Option<usize> {
        self.trained_window
    }

    /// Record the trained window of the engine this substrate serves.
    pub fn set_trained_window(&mut self, window: Option<usize>) {
        self.trained_window = window;
    }

    /// How large each dialogue's pool is on this substrate's card.
    pub fn pool_size(&self) -> PoolSize {
        self.pool_size
    }

    /// Record the pool size of the card this substrate serves.
    pub fn set_pool_size(&mut self, size: PoolSize) {
        self.pool_size = size;
    }

    pub fn couplings_of(&self, timeline: TimelineId) -> Couplings {
        self.timelines
            .get(&timeline)
            .map(|t| t.couplings.clone())
            .unwrap_or_default()
    }

    /// The exchanges of `timeline`'s `Normal` turns, each its turns in order:
    /// a head turn and every turn its tool round-trips coupled after it
    /// ([`exchanges`]). A summary node is in none.
    pub fn exchanges_of(&self, timeline: TimelineId) -> Vec<Vec<TurnIndex>> {
        let normals = self.normals_of(timeline);
        let couplings = over_normals(&self.couplings_of(timeline), &normals);
        exchanges(&couplings, normals.len())
            .into_iter()
            .map(|r| r.map(|pos| normals[pos]).collect())
            .collect()
    }

    /// The turns of `timeline` whose reasoning its next projection keeps: the
    /// newest [`REASONING_ROUNDS`] rounds of the request its next turn
    /// continues ([`continued_exchange`]), oldest first. Empty when the next turn
    /// is a new user message — a finished request's reasoning is not carried
    /// into the next one (`docs/thinking_span_projection.md` §6.3).
    pub fn reasoning_turns(&self, timeline: TimelineId) -> Vec<TurnIndex> {
        let request = self.request_turns(timeline);
        request[request.len().saturating_sub(REASONING_ROUNDS)..].to_vec()
    }

    /// The `Normal` turns of the request `timeline`'s next turn continues, in
    /// order: the request's own turn first, then each of its rounds. Empty
    /// when the next turn opens a request of its own. What the request's steps
    /// are read again from after a restart (`docs/request_steps.md` §2.5).
    pub fn request_turns(&self, timeline: TimelineId) -> Vec<TurnIndex> {
        let normals = self.normals_of(timeline);
        let couplings = over_normals(&self.couplings_of(timeline), &normals);
        let request = continued_exchange(&couplings, normals.len());
        normals[request].to_vec()
    }

    /// `timeline`'s `Normal` turns in order: every turn but its summary nodes.
    pub(super) fn normals_of(&self, timeline: TimelineId) -> Vec<TurnIndex> {
        (0..self.turn_count(timeline))
            .map(TurnIndex)
            .filter(|idx| {
                !self
                    .tree_meta_of(timeline, *idx)
                    .is_some_and(|m| m.kind.is_summary())
            })
            .collect()
    }

    // ── debug_id resume keys ────────────────────────────────────────────────

    /// Set the substrate-side resume key for a timeline, as `DebugId` record
    /// `id` — see [`Self::apply_debug_id_payload`]. Used by the test harness's
    /// `find_or_create(debug_id)` path (§10.4) and reconstructed on cold-load
    /// from the timeline's record stream.
    pub fn set_debug_id(
        &mut self,
        timeline: TimelineId,
        debug_id: impl Into<String>,
        id: RecordIdentity,
    ) {
        self.apply_debug_id_payload(
            &DebugIdPayload {
                timeline_id: timeline.raw(),
                debug_id: debug_id.into(),
            },
            id,
        );
    }

    /// Index `timeline`'s resume key. Replaces any previous mapping for the
    /// same key; clears the old mapping if the timeline already had a
    /// different debug_id.
    pub(super) fn index_debug_id(&mut self, timeline: TimelineId, id: String) {
        if let Some(tl) = entry_mut(&mut self.timelines, &timeline) {
            if let Some(old) = tl.debug_id.take() {
                if old != id {
                    self.timeline_by_debug_id.remove(&old);
                }
            }
            tl.debug_id = Some(id.clone());
            self.timeline_by_debug_id.insert(id, timeline);
        }
    }

    /// Look up a timeline by its `debug_id`.  O(1).
    pub fn lookup_by_debug_id(&self, id: &str) -> Option<TimelineId> {
        self.timeline_by_debug_id.get(id).copied()
    }

    /// Current `debug_id` for a timeline, if one has been set.
    pub fn debug_id_of(&self, timeline: TimelineId) -> Option<&str> {
        self.timelines
            .get(&timeline)
            .and_then(|tl| tl.debug_id.as_deref())
    }

    // ── Tree metadata read / write ───────────────────────────────────────────

    /// Read the parallel [`TreeNodeMeta`] for a turn.  Returns `None`
    /// when the timeline or the turn index is unknown.
    pub fn tree_meta_of(&self, timeline: TimelineId, idx: TurnIndex) -> Option<&TreeNodeMeta> {
        self.timelines
            .get(&timeline)
            .and_then(|tl| tl.tree_meta.get(&idx))
    }

    /// Set a turn's [`TreeNodeMeta`], as `TreeMetadata` record `id` — the
    /// greatest stamp per turn holds. The summariser thread calls this once, to
    /// promote a turn into the forest (Normal sub-leaf → SummaryOfTurns leaf,
    /// or to record a fresh SummaryOfSummaries internal). Nodes are immutable:
    /// an existing summary node is never rewritten.
    pub fn set_tree_meta(
        &mut self,
        timeline: TimelineId,
        idx: TurnIndex,
        meta: TreeNodeMeta,
        id: RecordIdentity,
    ) {
        if let Some(tl) = entry_mut(&mut self.timelines, &timeline) {
            if take_keyed(&mut tl.ids.tree_meta, idx, id) {
                tl.tree_meta.insert(idx, meta);
            }
        }
    }

    /// The peak set — orphan summary nodes (no parent), in chronological order
    /// (oldest/leftmost-covering first), each paired with its level.  These are
    /// the window's coarse entry points (`docs/archived/immutable_summary_forest.md`).
    pub fn peaks_of(&self, timeline: TimelineId) -> Vec<(TurnIndex, u8)> {
        let Some(tl) = self.timelines.get(&timeline) else {
            return Vec::new();
        };
        let mut claimed: std::collections::BTreeSet<TurnIndex> = std::collections::BTreeSet::new();
        for meta in tl.tree_meta.values() {
            if meta.kind == TurnKind::SummaryOfSummaries {
                for c in &meta.children {
                    claimed.insert(*c);
                }
            }
        }
        let mut peaks: Vec<(TurnIndex, u8)> = tl
            .tree_meta
            .iter()
            .filter(|(idx, meta)| meta.kind.is_summary() && !claimed.contains(*idx))
            .map(|(idx, meta)| (*idx, meta.tree_height))
            .collect();
        peaks.sort_by_key(|(idx, _)| leftmost_normal_in(&tl.tree_meta, *idx));
        peaks
    }

    /// Cheap guard: does this timeline have any summary nodes yet?
    pub fn has_summary_nodes(&self, timeline: TimelineId) -> bool {
        self.timelines
            .get(&timeline)
            .map(|tl| tl.tree_meta.values().any(|m| m.kind.is_summary()))
            .unwrap_or(false)
    }

    /// The next internal node to (re)build during reconciliation, expressed as
    /// the [`MERGE_FANOUT`] child indices it must cover — the lowest buildable
    /// node (all children already present) whose canonical `SummaryOfSummaries`
    /// is missing from the persisted forest.  `None` once the forest matches the
    /// canonical ternary shape for its leaves.
    ///
    /// Derived from the persisted state each call (the "dirty" bit is gone —
    /// staleness is computed, never stored).  See
    /// `docs/archived/immutable_summary_forest.md`.
    pub fn reconcile_next(&self, timeline: TimelineId) -> Option<Vec<TurnIndex>> {
        let tl = self.timelines.get(&timeline)?;
        let tm = &tl.tree_meta;
        // SoT leaves in chronological order (by their single Normal child).
        let mut leaves: Vec<(u32, TurnIndex)> = tm
            .iter()
            .filter(|(_, m)| m.kind == TurnKind::SummaryOfTurns)
            .map(|(idx, m)| (m.children.first().map(|c| c.0).unwrap_or(idx.0), *idx))
            .collect();
        leaves.sort_by_key(|(normal, _)| *normal);
        // Persisted SoS indexed by exact children signature (canonical nodes
        // have `MERGE_FANOUT` children; old binary nodes never match).
        let mut sos_by_children: std::collections::HashMap<Vec<TurnIndex>, TurnIndex> =
            std::collections::HashMap::new();
        for (idx, m) in tm {
            if m.kind == TurnKind::SummaryOfSummaries {
                sos_by_children.insert(m.children.clone(), *idx);
            }
        }
        // Replay the ternary carry with persisted nodes; the first carry whose
        // canonical SoS is absent is the lowest buildable rebuild.
        let mut peaks: Vec<(TurnIndex, u8)> = Vec::new();
        for (_, leaf_idx) in &leaves {
            peaks.push((*leaf_idx, 1));
            loop {
                let n = peaks.len();
                if n < MERGE_FANOUT {
                    break;
                }
                let lvl = peaks[n - 1].1;
                if !peaks[n - MERGE_FANOUT..].iter().all(|(_, l)| *l == lvl) {
                    break;
                }
                let children: Vec<TurnIndex> =
                    peaks[n - MERGE_FANOUT..].iter().map(|(i, _)| *i).collect();
                match sos_by_children.get(&children) {
                    Some(&sos_idx) => {
                        peaks.truncate(n - MERGE_FANOUT);
                        peaks.push((sos_idx, lvl + 1));
                    }
                    None => return Some(children),
                }
            }
        }
        None
    }

    /// Whether the summariser should run a reconcile pass for this timeline.
    pub fn needs_reconcile(&self, timeline: TimelineId) -> bool {
        self.timelines
            .get(&timeline)
            .map(|tl| tl.needs_reconcile)
            .unwrap_or(false)
    }

    /// Set/clear the reconcile hint.
    pub fn set_needs_reconcile(&mut self, timeline: TimelineId, v: bool) {
        if let Some(tl) = entry_mut(&mut self.timelines, &timeline) {
            tl.needs_reconcile = v;
        }
    }

    /// Prepare a freshly-loaded timeline for reconciliation: drop any
    /// non-canonical `SummaryOfSummaries` meta (e.g. binary nodes written by the
    /// superseded AVL code) so the ternary canonical nodes can be rebuilt, and
    /// arm the reconcile hint when the forest isn't already whole.
    pub fn mark_for_reconcile(&mut self, timeline: TimelineId) {
        let Some(tl) = entry_mut(&mut self.timelines, &timeline) else {
            return;
        };
        let noncanonical: Vec<TurnIndex> = tl
            .tree_meta
            .iter()
            .filter(|(_, m)| {
                m.kind == TurnKind::SummaryOfSummaries && m.children.len() != MERGE_FANOUT
            })
            .map(|(idx, _)| *idx)
            .collect();
        for idx in noncanonical {
            tl.tree_meta.remove(&idx);
        }
        // Arm reconcile when any leaf exists; `reconcile_next` returns `None`
        // immediately when the forest is already whole, so this is cheap.
        let has_leaf = tl
            .tree_meta
            .values()
            .any(|m| m.kind == TurnKind::SummaryOfTurns);
        tl.needs_reconcile = has_leaf;
    }

    // ── Pending + reconcile accessors (§6 backpressure metrics) ────────────

    /// FIFO pop: next pending turn for the summariser to absorb, or
    /// `None` if the queue is empty.
    pub fn pop_pending_summary(&mut self, timeline: TimelineId) -> Option<TurnIndex> {
        entry_mut(&mut self.timelines, &timeline)
            .and_then(|tl| tl.pending_summary_queue.pop_front())
    }

    /// Push a turn onto the pending-summary queue.  Used during
    /// cold-load reconstruction (§4) to re-enqueue orphan turns whose
    /// summary parent didn't survive a crash.
    ///
    /// Timelines in APPEND-ONLY ingest layers (repo_map, code_reading) never
    /// enqueue, regardless of their per-timeline `summarize` flag: ingest
    /// turns are background reference whose summaries the ingest pipeline
    /// itself owns, and letting them queue here storms the summariser with
    /// pointless decodes during a repo scan (competing with the ingest's own
    /// forwards for the GPU). Enforced at the mechanism so no ingest path can
    /// forget to clear the flag.
    pub fn push_pending_summary(&mut self, timeline: TimelineId, idx: TurnIndex) {
        // One map lookup: the entry carries its layer, so the per-timeline
        // flag and the append-only-layer refusal read off the same borrow
        // (`append_only_layers` is a disjoint field).
        if let Some(tl) = entry_mut(&mut self.timelines, &timeline) {
            if !tl.summarize || self.append_only_layers.contains(&tl.layer) {
                return;
            }
            tl.pending_summary_queue.push_back(idx);
        }
    }

    /// `pending_summary_queue.len()` — backpressure metric (§9).
    pub fn pending_summary_len(&self, timeline: TimelineId) -> usize {
        self.timelines
            .get(&timeline)
            .map(|tl| tl.pending_summary_queue.len())
            .unwrap_or(0)
    }

    /// Store the latest score-density [`SelectionDiagnostics`] for a
    /// timeline.  Called by the scheduler at projection time;
    /// last-write-wins across re-projections within a turn.  Pure
    /// test-harness instrumentation — production daemons can ignore.
    pub fn set_last_selection(&mut self, timeline: TimelineId, diag: SelectionDiagnostics) {
        if let Some(tl) = entry_mut(&mut self.timelines, &timeline) {
            tl.last_selection = Some(Arc::new(diag));
        }
    }

    /// Most recent [`SelectionDiagnostics`] for a timeline, or `None`
    /// if no projection has run yet (or the projection used the
    /// rule-based path).
    pub fn last_selection_of(&self, timeline: TimelineId) -> Option<&SelectionDiagnostics> {
        self.timelines
            .get(&timeline)
            .and_then(|tl| tl.last_selection.as_deref())
    }

    /// Chronologically-ordered `SummaryOfTurns` leaf turn indices for
    /// a timeline.  Walked by the projection's score-density selector
    /// (§8) to evaluate the right-edge recency anchor.  Empty when no
    /// summary tree exists yet.
    pub fn summary_leaves_chrono(&self, timeline: TimelineId) -> Vec<TurnIndex> {
        let tl = match self.timelines.get(&timeline) {
            Some(t) => t,
            None => return Vec::new(),
        };
        tl.tree_meta
            .iter()
            .filter(|(_, m)| m.kind == TurnKind::SummaryOfTurns)
            .map(|(idx, _)| *idx)
            .collect()
    }

    /// Chronologically-ordered `Normal` turn indices for a timeline.
    pub fn normal_turns_chrono(&self, timeline: TimelineId) -> Vec<TurnIndex> {
        let tl = match self.timelines.get(&timeline) {
            Some(t) => t,
            None => return Vec::new(),
        };
        tl.tree_meta
            .iter()
            .filter(|(_, m)| m.kind == TurnKind::Normal)
            .map(|(idx, _)| *idx)
            .collect()
    }

    /// Chronologically-ordered `SummaryOfSummaries` internal-node turn
    /// indices for a timeline.
    pub fn summary_internals_chrono(&self, timeline: TimelineId) -> Vec<TurnIndex> {
        let tl = match self.timelines.get(&timeline) {
            Some(t) => t,
            None => return Vec::new(),
        };
        tl.tree_meta
            .iter()
            .filter(|(_, m)| m.kind == TurnKind::SummaryOfSummaries)
            .map(|(idx, _)| *idx)
            .collect()
    }

    /// All registered timelines.
    pub fn all_timeline_ids(&self) -> impl Iterator<Item = TimelineId> + '_ {
        self.timelines.keys().copied()
    }

    /// Every registered timeline that is not tombstoned, named or not.
    ///
    /// [`Self::known_conversations`] lists only timelines that carry a
    /// `conv_id`; an ingested document is a conversation with none, so a caller
    /// retiring *everything* needs this set instead.
    pub fn live_timeline_ids(&self) -> Vec<TimelineId> {
        self.timelines
            .keys()
            .filter(|tl| !self.tombstoned_timelines.contains(tl))
            .copied()
            .collect()
    }

    // ── Per-stream runtime state (was Manifest.streams) ─────────────────

    /// Read the in-RAM runtime state for `stream_id` — chunk index +
    /// latest tokens location + committed-through watermark + decl. A turn's
    /// id names the turn at that position in its timeline, whoever wrote it
    /// (§6.6).
    pub fn stream_of(&self, stream_id: StreamId) -> Option<&StreamRuntime> {
        self.streams.get(stream_id)
    }

    /// One writer's stream, whatever position it holds.
    pub fn stream_at(&self, key: StreamKey) -> Option<&StreamRuntime> {
        self.streams.physical(key)
    }

    /// One writer's stream as the declared turn it is, at its position; `None`
    /// for a stream that is not a declared turn.
    pub fn turn_at(&self, key: StreamKey) -> Option<TurnStream<'_>> {
        self.streams.turn(key)
    }

    /// Where a declared turn's stream sits in its timeline — the turn index
    /// callers address it by. `None` for a stream that is not a declared turn.
    pub fn turn_position(&self, key: StreamKey) -> Option<TurnIndex> {
        self.streams.position(key).map(TurnIndex)
    }

    /// `timeline`'s declared turns in turn order, each at its position. A
    /// position no turn holds is a gap: a declaration lost in a crash.
    pub fn turn_streams_of(
        &self,
        timeline: u64,
    ) -> impl Iterator<Item = (TurnIndex, StreamKey, &StreamRuntime)> + '_ {
        self.streams.turns_of(timeline).filter_map(|(pos, key)| {
            self.streams
                .physical(key)
                .map(|rt| (TurnIndex(pos), key, rt))
        })
    }

    /// The version stamp of `timeline`'s first declared turn — what orders two
    /// members' ingests of one unit (`docs/replicated_substrate.md` §10.4): the
    /// lower is the unit. `None` for a timeline with no turn declared.
    pub fn first_turn_stamp(&self, timeline: u64) -> Option<VersionStamp> {
        self.turn_streams_of(timeline)
            .next()
            .map(|(_, _, rt)| rt.ids.decl.stamp())
    }

    /// Every timeline with a declared turn.
    pub fn timelines_with_turns(&self) -> impl Iterator<Item = u64> + '_ {
        self.streams.timelines()
    }

    /// Whether ANY indexed stream carries KV chunk records. `false` means this
    /// workspace never persists KV (e.g. `compression_policy = None`), so the
    /// startup integrity check must not treat chunk absence as damage.
    pub fn any_stream_has_chunks(&self) -> bool {
        self.streams
            .iter_physical()
            .any(|(_, s)| !s.chunks.is_empty())
    }

    /// The belief-gallery epoch: unchanged while nothing a collection's gallery
    /// is assembled from has changed, so an assembled gallery keyed on it can be
    /// reused. See the field.
    pub fn gallery_epoch(&self) -> u64 {
        self.gallery_epoch
    }

    /// The tombstone epoch: unchanged while no turn or timeline has been
    /// tombstoned. See the field.
    pub fn tombstone_epoch(&self) -> u64 {
        self.tombstone_epoch
    }

    /// `stream_id`'s stored wide-Q window, sized and fingerprinted but not
    /// decoded — it shares the stream's encoded buffer. `None` when the stream
    /// has no signature, an empty one, or one a scan must not read (see
    /// [`SigWindow::new`]).
    ///
    /// Nothing decoded is kept: a gallery resident on the device needs only the
    /// window's length and fingerprint, and a caller that needs the tokens
    /// decodes them for as long as it uses them ([`SigWindow::decode`]).
    pub fn wide_sig_window(&self, stream_id: StreamId) -> Option<SigWindow> {
        self.streams
            .get(stream_id)
            .and_then(|e| e.wide_q_sigs.clone())
            .and_then(SigWindow::new)
    }

    /// The hybrid-logical-clock reading of the record that holds
    /// `stream_id`'s signature — the same on every member — or `None` when it
    /// holds none (`docs/replicated_substrate.md` §6.2). What an incremental
    /// exclusion's cursor is kept in (`docs/incremental_chunk_exclusion.md`
    /// §4.2).
    pub fn wide_sig_hlc(&self, stream_id: StreamId) -> Option<u64> {
        self.streams
            .get(stream_id)
            .filter(|e| e.wide_q_sigs.is_some())
            .map(|e| e.ids.wide_q_sigs.hlc)
    }

    /// A turn's self-referencing sub-window seam offsets (the `start_token`s of
    /// its `self_reference` projection events), **sorted and deduped** — decoded
    /// from the events JSON blob once per session and memoized, so the belief
    /// scan reads them without re-parsing JSON for every gallery turn on every
    /// reprojection. Returns an empty (shared) list when the stream has no events
    /// or no self-referencing seams (the common case for a code-read scope, which
    /// scores as one whole-turn window). See [`Self::score_belief_groups`].
    pub fn decoded_seams(&self, stream_id: StreamId) -> Arc<Vec<usize>> {
        self.events_digest(stream_id).seams.clone()
    }

    /// The section `collection` was last seen selecting in this turn's
    /// projections — the latest of its events that selected one there — or
    /// `None` when none did. What labels an untagged turn in a collection's
    /// belief gallery, read from the memo rather than by decoding the turn's
    /// events JSON on every assembly.
    pub fn last_selected_in(&self, stream_id: StreamId, collection: &str) -> Option<String> {
        self.events_digest(stream_id)
            .selected
            .get(collection)
            .cloned()
    }

    /// The stream's decoded events, memoized on the stream itself
    /// ([`StreamRuntime::events_digest`]). The memo belongs to the blob it was
    /// decoded from: a write of the blob replaces the stream's runtime with an
    /// empty memo, so a reader of any version reads the digest of exactly the
    /// blob that version holds. Two readers racing on the first decode both
    /// decode and one value is kept — the values are identical.
    pub(super) fn events_digest(&self, stream_id: StreamId) -> Arc<EventsDigest> {
        let Some(stream) = self.streams.get(stream_id) else {
            return Arc::default();
        };
        stream
            .events_digest
            .get_or_init(|| {
                let events = stream
                    .projection_events
                    .as_deref()
                    .map(decode_events)
                    .unwrap_or_default();
                Arc::new(digest_events(&events))
            })
            .clone()
    }

    /// True iff `stream_id` names a stream the substrate holds a record of.
    pub fn has_stream(&self, stream_id: StreamId) -> bool {
        self.streams.get(stream_id).is_some()
    }

    /// Every stream id that names a stream ([`Self::all_streams`]).
    pub fn all_stream_ids(&self) -> impl Iterator<Item = StreamId> + '_ {
        self.streams.iter_logical().map(|(sid, _)| sid)
    }

    /// Iterate `(stream_id, &StreamRuntime)` pairs: every stream an id names,
    /// under that id — each declared turn under its position's, the way
    /// [`Self::stream_of`] reaches it.
    pub fn all_streams(&self) -> impl Iterator<Item = (StreamId, &StreamRuntime)> + '_ {
        self.streams.iter_logical()
    }

    /// Every writer's stream, each turn whatever position it holds — what a
    /// rewrite carries.
    pub fn all_physical_streams(&self) -> impl Iterator<Item = (StreamKey, &StreamRuntime)> + '_ {
        self.streams.iter_physical()
    }

    /// Total live chunk count across all streams.  Used by the
    /// compaction dead-ratio calculation.
    pub fn live_chunk_count(&self) -> usize {
        self.streams
            .iter_physical()
            .map(|(_, s)| s.chunks.len())
            .sum()
    }

    /// Streams that hold KV chunks but have NO `StreamDecl` and none coming —
    /// orphans to node `node` ([`StreamRuntime::is_orphan`]). Their turn cannot
    /// be reconstructed (reload rebuilds turns from their decl), so the chunks
    /// are unreachable and the turn is silently lost. Surfaced at reload so
    /// this loss is VISIBLE rather than silent, and reclaimed by compaction
    /// (see `collect_live_records`). A healthy substrate reports 0.
    pub fn orphan_chunk_stream_count(&self, node: u64) -> usize {
        self.streams
            .iter_physical()
            .filter(|(_, s)| s.is_orphan(node) && !s.chunks.is_empty())
            .count()
    }

    /// Every timeline's label as the `Label` records that hold it — one per
    /// winning identity, with the fields it won ([`label_records`]). Used by
    /// both rewrite paths to re-emit live `Label` records; the `ConvState` side
    /// is [`Self::live_conv_states`].
    ///
    /// In creation `order`, then timeline, then stamp: the output is a function
    /// of the substrate, so two compactions of one store write the same file. A
    /// timeline with no label fields writes nothing.
    pub fn live_label_records(&self) -> Vec<(u64, ConvMeta, RecordIdentity)> {
        let mut timelines: Vec<(&TimelineId, &TimelineEntry)> =
            self.timelines.iter().map(|(t, e)| (t, &**e)).collect();
        timelines.sort_by_key(|(tid, tl)| (tl.order, tid.raw()));
        timelines
            .into_iter()
            .flat_map(|(tid, tl)| {
                label_records(tl)
                    .into_iter()
                    .map(move |(meta, id)| (tid.raw(), meta, id))
            })
            .collect()
    }

    /// Emit one `TreeMetadataPayload` per tree node a `TreeMetadata` record
    /// set, with that record's identity. Used by compaction and maintenance to
    /// re-emit live `TreeMetadata` records.
    ///
    /// **Only those.** Every turn carries a default `Normal` node from the moment
    /// it is appended, implied by the turn and rebuilt with it on every load; no
    /// record ever stated it, so it has no identity. Emitting it anyway wrote one
    /// unstamped record per plain turn on every rewrite, and the next open read
    /// those as a store written before identities and rewrote the whole of it —
    /// 98 s of a restart on zend's store, three and a half minutes on npcd's.
    pub fn live_tree_metadata_payloads(&self) -> Vec<(TreeMetadataPayload, RecordIdentity)> {
        let mut out = Vec::new();
        for (tid, tl) in &self.timelines {
            for (idx, meta) in &tl.tree_meta {
                let Some(&id) = tl.ids.tree_meta.get(idx) else {
                    continue;
                };
                let kind = match meta.kind {
                    TurnKind::Normal => 0,
                    TurnKind::SummaryOfTurns => 1,
                    TurnKind::SummaryOfSummaries => 2,
                };
                out.push((
                    TreeMetadataPayload {
                        timeline_id: tid.raw(),
                        turn_index: idx.0,
                        kind,
                        tree_height: meta.tree_height,
                        children: meta.children.iter().map(|c| c.0).collect(),
                    },
                    id,
                ));
            }
        }
        out
    }

    /// Every record held for a timeline not registered yet, as a rewrite
    /// writes it back — see [`PendingChildren::records`].
    pub fn pending_records(&self) -> Vec<(u64, RecordType, Vec<u8>, RecordIdentity)> {
        self.pending.records()
    }

    /// Emit `(timeline_id, from_turn)` for every coupling on every timeline —
    /// the live `TurnCoupling` set, re-emitted by both rewrite paths.
    ///
    /// A coupling is the only durable record that a tool round-trip's two halves
    /// belong to one exchange. Nothing recomputes it: the fact is known once, in
    /// the window between the tools returning and the response turn being
    /// submitted, and never again. So a rewrite that fails to carry it forward
    /// does not degrade the grouping, it deletes it — the call turn and its
    /// response become independent exchanges, and provenance can select one
    /// without the other (a `<tool_call>` with no result, or a `<tool_response>`
    /// with no call).
    ///
    /// Sorted, because a rewrite's output must be a function of its input: an
    /// unordered walk of a per-timeline set produces a different file from the
    /// same store on every pass, which makes two generations impossible to diff.
    pub fn live_couplings(&self) -> Vec<(u64, u32, RecordIdentity)> {
        let mut out: Vec<(u64, u32, RecordIdentity)> = self
            .timelines
            .iter()
            .flat_map(|(tid, tl)| {
                tl.couplings.iter().map(move |&from| {
                    let id = tl.ids.couplings.get(&from).copied().unwrap_or_default();
                    (tid.raw(), from, id)
                })
            })
            .collect();
        out.sort_unstable_by_key(|(tl, from, _)| (*tl, *from));
        out
    }

    /// Emit `(timeline_id, debug_id, record)` for every timeline that has one
    /// set.  Used by compaction to re-emit live `DebugId` records.
    pub fn live_debug_ids(&self) -> Vec<(u64, String, RecordIdentity)> {
        self.timelines
            .iter()
            .filter_map(|(tid, tl)| {
                tl.debug_id
                    .as_ref()
                    .map(|d| (tid.raw(), d.clone(), tl.ids.debug_id))
            })
            .collect()
    }
}
