//! Moves each residence between the VRAM, host-RAM and disk tiers, and records
//! which copies a residence holds at any moment.
//!
//! A hot copy is dropped only where a warm copy already holds the same bytes, and
//! a working-set pin defers any drop that would take KV a live wave is reading.
//! An install lands only over the content it was made from, so a copy overtaken
//! by a kill or a replacement takes nothing. Once a cold copy is durable, the
//! resident tiers are freed and a later elevation pulls the residence back from
//! disk on demand. A pinned fresh seal is still migrated to warm, since that
//! migration is what swaps its native form for the quantized one.

use super::*;

impl Substrate {
    pub fn new() -> Self {
        Self::default()
    }

    /// The residence slab every version of this substrate shares.
    pub fn residences(&self) -> &Arc<ResidenceSlab> {
        &self.residences
    }

    /// The residence at `idx`.
    pub fn residence(&self, idx: ResidenceIndex) -> &Residence {
        self.residences.get(idx)
    }

    /// Allocate a fresh turn residence. Used by every turn insert path so each
    /// owning entry has a stable [`ResidenceIndex`] to address its KV state.
    ///
    /// The slot starts empty in every tier. Allocation happens inside the
    /// version edit that names the residence: an edit retried against a newer
    /// version names a fresh one, and the lost attempt's is discarded
    /// ([`residence::begin_attempt`]).
    pub(super) fn alloc_residence(
        &self,
        stream_id: StreamId,
        compression: Option<ConvCompression>,
    ) -> ResidenceIndex {
        self.residences.alloc(stream_id, compression, 0)
    }

    /// [`Self::alloc_residence`] for a prompt section.
    pub(super) fn alloc_section_residence(&self, stream_id: StreamId) -> ResidenceIndex {
        self.residences.alloc(stream_id, None, residence::SECTION)
    }

    /// Set (or clear) the per-conversation compression override for
    /// `timeline`. Called at conversation creation from the
    /// `SequenceConfig`. Must run before the timeline's first turn seals
    /// so each turn residence picks it up at alloc time.
    pub fn set_timeline_compression(
        &mut self,
        timeline: TimelineId,
        compression: Option<ConvCompression>,
    ) {
        match compression {
            Some(cc) => {
                self.timeline_compression.insert(timeline, cc);
            }
            None => {
                self.timeline_compression.remove(&timeline);
            }
        }
    }

    /// `timeline`'s per-conversation compression override, the one each of
    /// its turn residences is allocated with.
    pub fn timeline_compression(&self, timeline: TimelineId) -> Option<ConvCompression> {
        self.timeline_compression.get(&timeline).copied()
    }

    /// Mark whether `timeline`'s turns should be summarised. Set `false` for
    /// append-only utility/reference layers (repo_map, code_reading) so their
    /// turns never enter the summariser's pending queue. Must run after the
    /// timeline is registered and before its first turn seals.
    pub fn set_timeline_summarize(&mut self, timeline: TimelineId, summarize: bool) {
        if let Some(tl) = entry_mut(&mut self.timelines, &timeline) {
            tl.summarize = summarize;
        }
    }

    /// The chunks cut from `group`'s live belief scans; none until learned.
    pub fn chunk_exclusion(&self, group: GroupId) -> Arc<ChunkExclusion> {
        self.chunk_exclusion
            .get(&group)
            .map(Arc::clone)
            .unwrap_or_default()
    }

    /// Mark `layer` as an append-only ingest layer (code_reading, repo_map) — a
    /// projection targeting it is scored/selected self-local. Idempotent; called
    /// once per ingest layer at setup. See [`Self::append_only_layers`].
    pub fn mark_layer_append_only(&mut self, layer: LayerId) {
        self.append_only_layers.insert(layer);
    }

    /// Whether `layer` is an append-only ingest layer — read by the belief
    /// scoring/selection to decide self-local masking (all groups → target
    /// timeline). Dialogue layers are never marked, so cross-file retrieval there
    /// is untouched. See [`Self::mark_layer_append_only`].
    pub fn is_append_only_layer(&self, layer: LayerId) -> bool {
        self.append_only_layers.contains(&layer)
    }

    /// Let projections targeting the append-only `layer` score and select
    /// `groups` — groups of the layers below it — across conversations instead
    /// of self-local. See [`Self::ingest_sees`].
    /// An empty set clears the layer's: it reads every group self-local again.
    pub fn set_ingest_sees(&mut self, layer: LayerId, groups: HashSet<GroupId>) {
        if groups.is_empty() {
            self.ingest_sees.remove(&layer);
        } else {
            self.ingest_sees.insert(layer, Arc::new(groups));
        }
    }

    /// Whether a projection targeting `layer` reads `group` self-local: every
    /// group when the layer is append-only, but those of the layers it compounds
    /// on ([`Self::set_ingest_sees`]). Never true for a dialogue target.
    pub fn is_self_local_group(&self, layer: LayerId, group: GroupId) -> bool {
        self.append_only_layers.contains(&layer)
            && !self
                .ingest_sees
                .get(&layer)
                .is_some_and(|groups| groups.contains(&group))
    }

    /// Place `sealed` into the residence's hot tier, most recently used. Used
    /// by every code path that brings KV bytes into VRAM — fresh seal, cold→hot
    /// promotion, post-cold-load materialisation. A dead residence takes
    /// nothing. Turns and sections alike: a section's [`residence::SECTION`]
    /// flag is what keeps the recency-ordered evictions off it.
    ///
    /// Caller invariant: `sealed` is non-empty.
    pub(super) fn install_hot(
        &self,
        residence: ResidenceIndex,
        sealed: Vec<SealedSequence>,
    ) -> bool {
        self.residences.install_hot(residence, sealed)
    }

    /// The hot residences of turns, oldest first — what the recency-ordered
    /// evictions walk. Sections are not among them.
    pub(super) fn hot_turns_oldest_first(&self) -> Vec<ResidenceIndex> {
        let mut hot: Vec<(u64, ResidenceIndex)> = self
            .residences
            .hot_members()
            .into_iter()
            .filter_map(|idx| {
                let r = self.residences.try_get(idx)?;
                (r.has_hot() && !r.has_flag(residence::SECTION)).then(|| (r.touched(), idx))
            })
            .collect();
        hot.sort_unstable_by_key(|&(t, idx)| (t, idx.0));
        hot.into_iter().map(|(_, idx)| idx).collect()
    }

    /// The residences holding a warm copy, oldest first.
    pub(super) fn warm_oldest_first(&self) -> Vec<ResidenceIndex> {
        let mut warm: Vec<(u64, ResidenceIndex)> = self
            .residences
            .warm_members()
            .into_iter()
            .filter_map(|idx| {
                let r = self.residences.try_get(idx)?;
                r.has_warm().then(|| (r.touched(), idx))
            })
            .collect();
        warm.sort_unstable_by_key(|&(t, idx)| (t, idx.0));
        warm.into_iter().map(|(_, idx)| idx).collect()
    }

    // ── Tier transitions (persistence-thread API) ───────────────────────────

    /// Install RAM-resident sealed chunks as a residence's warm copy. Called
    /// by the persistence thread after `migrate_sealed_to_cpu` produces a CPU
    /// copy.
    ///
    /// Caller invariant: `warm` is non-empty.
    pub fn install_warm(&self, residence: ResidenceIndex, warm: Vec<SealedSequence>) {
        self.residences.install_warm(residence, warm);
    }

    /// Whether an automatic drop may take `residence`'s resident copies: no
    /// working set holds it and it is not a splice source.
    pub(super) fn may_shed(&self, residence: ResidenceIndex) -> bool {
        let r = self.residences.get(residence);
        !r.is_pinned() && !r.has_flag(SPLICE_SOURCE)
    }

    /// Install warm AND atomically drop hot — the production
    /// persistence-thread path. Once warm holds the compressed
    /// canonical copy of the turn there is no reason to keep the
    /// uncompressed hot reference around: it just pins VRAM the
    /// arena pool could reclaim. The next decode that touches this
    /// residence re-elevates warm→hot, landing quantized GPU chunks
    /// (bit-identical to what a daemon restart would reconstruct
    /// from cold) so the in-session attention path exercises the
    /// same compressed bytes that survive a restart — fidelity bugs
    /// surface immediately instead of being masked by the
    /// uncompressed hot copy until the next reload.
    ///
    /// Tests that need the "hot + warm coexist" intermediate state
    /// should call [`Self::install_warm`] directly.
    pub fn install_warm_and_evict_hot(&self, residence: ResidenceIndex, warm: Vec<SealedSequence>) {
        self.install_warm(residence, warm);
        // A working-set pin DEFERS the hot-drop. The scheduler is attending this
        // residence in the current wave (a cross-timeline belief pick that
        // `elevate_projection_working_set` just lifted), so it keeps hot+warm
        // and sheds hot on a later pass, once it leaves the working set (the
        // `EVICT_WHEN_COLD` flag stays set). A splice source keeps hot until its
        // fork's tombstone — see [`residence::SPLICE_SOURCE`]. The dropped copy
        // is freed on the device's thread, after any forward still reading it.
        if self.may_shed(residence) {
            self.residences.take_hot(residence);
        }
    }

    /// Install one hot→warm copy's result: `warm`, the CPU copy of the hot
    /// copy `copied` the persistence thread read. The hot copy is already in
    /// its stored formats — the gap quantized it — so it stays as it is.
    ///
    /// A residence flagged `evict_when_cold` drops hot now, so its VRAM
    /// returns at once; the warm→cold write then frees warm too
    /// ([`Self::install_cold`]), leaving it cold-only. **A flagged residence a
    /// live working set names keeps its hot copy** for the wave reading it;
    /// the flag stands, and [`Self::demote_idle_hot`] sheds the hot copy once
    /// the residence leaves the working set.
    ///
    /// **Only over the content it was made from.** When the residence no
    /// longer holds `copied`'s content — killed, or taken or replaced since,
    /// a gap's quantize among them — nothing is installed, and `false` says
    /// so. A KV compaction that relocated the copy meanwhile has not moved the
    /// content on ([`HotKv::lineage`]), so the install lands over it.
    pub fn install_copied(
        &self,
        residence: ResidenceIndex,
        copied: &Arc<HotKv>,
        warm: Vec<SealedSequence>,
    ) -> bool {
        let r = self.residences.get(residence);
        if r.hot().is_none_or(|now| now.lineage() != copied.lineage()) {
            return false;
        }
        if r.has_flag(EVICT_WHEN_COLD) && !r.is_pinned() {
            self.install_warm_and_evict_hot(residence, warm);
            return true;
        }
        self.residences.install_warm(residence, warm)
    }

    /// Install a residence's hot and warm copies together, hot first.
    ///
    /// Replaces — not adds — `residence.hot`. The previous hot sequences drop
    /// their Arc refs here; if no other holder remains, the underlying arena
    /// chunks return to the pool, freed on the device's thread.
    pub fn install_warm_and_hot(
        &self,
        residence: ResidenceIndex,
        hot: Vec<SealedSequence>,
        warm: Vec<SealedSequence>,
    ) {
        debug_assert!(
            !hot.is_empty(),
            "install_warm_and_hot called with empty hot"
        );
        // Hot first: a reader that finds warm and not hot would lift a copy the
        // install is about to land.
        self.residences.install_hot(residence, hot);
        self.residences.install_warm(residence, warm);
    }

    /// Install the cold-tier references for a residence — called by the
    /// persistence thread after writing the turn's chunks to the redo log.
    pub fn install_cold(&self, residence: ResidenceIndex, cold: Vec<StoredSequence>) {
        debug_assert!(!cold.is_empty(), "install_cold called with empty Vec");
        self.residences.install_cold(residence, cold);
        // Offload-as-we-go: the (quantized) bytes are now durable on disk, so
        // free EVERY resident tier and let `elevate_to_hot` pull the residence
        // back from cold (NVMe) on demand if a later projection re-selects it.
        //   - Sections are hot→cold (no warm): the `warm` take finds nothing.
        //   - Completed-ingest turns are hot→warm→cold: `hot` was already freed
        //     at warm-land, and taking `warm` here reclaims the RAM copy too —
        //     leaving the turn cold-only, off both the VRAM and RAM tiers.
        // A pinned residence keeps both (same reason as
        // `install_warm_and_evict_hot`); the flag stays set, so the idle demote
        // sheds them once it leaves the working set. The hot copy is freed on
        // the device's thread, never here, so a compaction relocating chunks is
        // never raced.
        if !self.residences.get(residence).has_flag(EVICT_WHEN_COLD) || !self.may_shed(residence) {
            return;
        }
        self.residences.take_hot(residence);
        self.residences.take_warm(residence);
    }

    /// Flag a section residence so the persistence thread frees its `hot` the
    /// moment a cold copy lands — see [`residence::EVICT_WHEN_COLD`]. Set by
    /// the scheduler's collection-member quantize drain.
    pub fn mark_section_evict_when_cold(&self, residence: ResidenceIndex) {
        self.residences.get(residence).set_flag(EVICT_WHEN_COLD);
    }

    /// Publish the scheduler's live working set as the eviction keep-set —
    /// resolving `keep_turns` / `keep_sections` (the union of every live slot's
    /// projection plus any in-flight elevate) to their residence indices and
    /// replacing the scheduler's previous publication wholesale. Called at
    /// elevate time, before any KV is attended, so every automatic drop honours
    /// it for the whole wave. Touches residences only: it commits no substrate
    /// version and never waits.
    ///
    /// Wholesale replace (not merge) is correct: the union is recomputed from
    /// current scheduler state each call, so a residence that has left every
    /// working set loses its pin and becomes evictable again. Mirrors the
    /// protected-set resolution in [`Self::demote_cold_ingest`].
    ///
    /// **Also the last-touch stamp point** (`docs/vram_governor_design.md` §8.1).
    /// The pin set *is* the active set, so the current epoch is written onto
    /// exactly the residences this wave will attend, in the same publication.
    /// Stamping here rather than at some later consumer is what makes the idle
    /// demote safe by construction: a residence the current wave touched carries
    /// the current epoch, so [`Self::demote_idle_hot`] cannot select it — the
    /// active set is excluded arithmetically, not by a filter that could drift
    /// out of sync with this one. The stamp never advances the clock: that
    /// belongs to the persistence pass that reads it.
    pub fn set_working_set_pins(&self, keep_turns: &[TurnKey], keep_sections: &[SectionId]) {
        let mut pins: HashSet<usize> =
            HashSet::with_capacity(keep_turns.len() + keep_sections.len());
        for &key in keep_turns {
            if let Some(e) = self.turn(key.timeline, key.index) {
                pins.insert(e.content.residence.0);
            }
        }
        for &sid in keep_sections {
            if let Some(e) = self.sections.get(&sid) {
                pins.insert(e.residence.0);
            }
        }
        self.residences.publish_pins(SCHEDULER_PINS, pins);
    }

    /// Release every working set's pins — the shutdown drain's final pass.
    pub fn release_working_set_pins(&self) {
        self.residences.release_all_pins();
    }

    /// Whether a working set holds `residence`.
    pub fn is_pinned(&self, residence: ResidenceIndex) -> bool {
        self.residences.get(residence).is_pinned()
    }

    /// Drop the hot copy of residences that no wave has attended for `grace`
    /// epochs and that already have a warm or a cold one
    /// (`docs/vram_governor_design.md` §8.1). Returns how many were shed.
    ///
    /// **Sets no flag.** An earlier version latched `evict_when_cold`, which was
    /// wrong twice over: that flag is a statement about a residence's *kind*
    /// (collection members and completed-ingest turns are offload-only by
    /// nature) and nothing in the tree ever clears it — every write is
    /// `= true`. Latching it on a live dialogue turn that merely went quiet made
    /// the residence permanently offload-only, so every later warm purge cost it
    /// its hot copy where an unflagged residence would have kept it. Dropping
    /// hot directly says exactly what is meant, is undone by the next elevate,
    /// and leaves no state to reset.
    ///
    /// This is the counterpart `elevate_to_hot` never had. Everything above it
    /// in the relief ladder reclaims *turns*; nothing demoted a section once it
    /// was lifted, nor an idle conversation's turns, so the hot set only grew
    /// and every shortfall was met by conceding expert-weight ground — which is
    /// one-way, because the give-back needs a floor move that is refused while
    /// any region above it is live.
    ///
    /// **Why it cannot take live KV.** Two independent barriers, the first
    /// structural: a residence the current wave attends was stamped with the
    /// current epoch by [`Self::set_working_set_pins`] and so fails the
    /// staleness test outright, and its pin excludes it again. The drop itself
    /// only takes the residence's reference — a forward still reading the copy
    /// holds its own, and the chunks are freed on the device's thread once the
    /// last reference is gone — and only ever from a residence whose warm copy
    /// already holds the same bytes.
    ///
    /// `PENDING_QUANTIZE` residences are skipped: their `hot` is the interim
    /// native form the scheduler still owes a quantize, and persisting that
    /// would diverge from the final Q form.
    ///
    /// **Also the sweep for an offload a pin deferred.** A residence flagged
    /// `EVICT_WHEN_COLD` has its resident tiers dropped the moment a lower tier
    /// lands — hot once warm holds it, warm too once cold does — unless a
    /// working set holds it then. One that was held is shed here once it is
    /// not, whatever its age: the flag already says it is not wanted resident.
    pub fn demote_idle_hot(&self, grace: u64) -> usize {
        // **The pass is the clock.** One tick per call, from the persistence
        // thread's between-forward pass, which wakes on every seal/flush trigger
        // and otherwise on its own 5 s tick. That makes an epoch track real work
        // — pressure ages residences sooner, which is the direction we want —
        // while the idle floor still retires a dormant conversation that
        // generates no traffic at all.
        let epoch = self.residences.tick_epoch();
        let mut shed = 0usize;
        // Sections as well as turns: nothing else demotes a section once
        // `elevate_to_hot` lifted it.
        for idx in self.residences.hot_members() {
            let Some(r) = self.residences.try_get(idx) else {
                continue;
            };
            // A lower copy is the safety condition AND the whole mechanism: the
            // copy the drop falls back on must already exist — warm, or cold for
            // a residence that never takes a warm copy (`pin_defers_warm`). A
            // residence with neither is simply left for a later pass, by which
            // time the migrate will have made one.
            let backed = r.has_warm() || r.has_cold();
            let offload = r.has_flag(EVICT_WHEN_COLD) && backed;
            let take = r.has_hot()
                && !r.is_pinned()
                && !r.has_flag(PENDING_QUANTIZE | SPLICE_SOURCE)
                && (offload || (backed && idle_for(r.last_used_epoch(), epoch) > grace));
            if !take || self.residences.take_hot(idx).is_none() {
                continue;
            }
            // A flagged residence whose cold copy has landed sheds warm as well —
            // what `install_cold` would have done had a pin not deferred it.
            if r.has_flag(EVICT_WHEN_COLD) && r.has_cold() {
                self.residences.take_warm(idx);
            }
            shed += 1;
        }
        shed
    }

    /// Rewrite every resident sequence's gids through a KV compaction's map.
    ///
    /// The substrate's half of a compaction sweep. A chunk's gid *is* its physical
    /// location, so a pass that relocates one must rewrite every holder of that
    /// identity inside the same window — and the residences are the largest holder
    /// outside the backing itself, invisible to any walk of the block tables.
    ///
    /// Both tiers are visited. `warm` gids are in the same `arena_idx` namespace as
    /// hot ones (the pool is one index space; `ArenaKey` only selects which pool the
    /// refcount table lives in), so leaving them out would be leaving real holders
    /// stale — and the map's own two-level lookup rejects anything from an untouched
    /// arena, so a host chunk simply misses.
    ///
    /// Each copy is replaced against the one it was rewritten from, and
    /// rewritten again from the newer one when an install landed in between, so
    /// no copy installed during the sweep keeps a stale gid.
    ///
    /// Returns how many residences were rewritten, for the pass's log line.
    pub fn rewrite_for_compaction(&self, sweep: &mut Sweep<'_>) -> candle::Result<usize> {
        let mut rewritten = 0usize;
        // The residences a sweep can find anything in — those with a hot copy
        // or a warm one on the device ([`ResidenceSlab::compaction_members`]);
        // walking every slot visited each warm-only and cold-only residence of
        // the corpus to find nothing, ~30 ms of every pass on BlackWorld.
        for idx in self.residences.compaction_members() {
            let Some(r) = self.residences.try_get(idx) else {
                continue;
            };
            let mut any = false;
            while let Some(hot) = r.hot() {
                // A copy in arenas this pass touched nothing of is passed by
                // without a chunk read: `rewrite_sealed` would find nothing.
                if !sweep.may_touch(hot.arenas()) {
                    break;
                }
                let Some(next) = rewrite_sealed(&hot, sweep)? else {
                    break;
                };
                if self.residences.relocate_hot(idx, &hot, next) || r.is_dead() {
                    any = true;
                    break;
                }
            }
            while let Some(warm) = r.warm() {
                let Some(next) = rewrite_sealed(&warm, sweep)? else {
                    break;
                };
                if self.residences.replace_warm(idx, &warm, next) {
                    any = true;
                    break;
                }
            }
            if any {
                rewritten += 1;
            }
        }
        Ok(rewritten)
    }

    /// Flag every turn residence of `timeline` for full eviction the moment its
    /// KV is durable on disk (see [`residence::EVICT_WHEN_COLD`]): the
    /// hot→warm install frees VRAM as warm lands, then `install_cold` frees the
    /// warm RAM copy as cold lands, leaving each turn cold-only on NVMe.
    ///
    /// Used to reclaim a **completed ingest** conversation (e.g. a code_read
    /// file) whose turns won't be attended again until retrieval pulls them
    /// back via `elevate_to_hot` — so their KV must not linger resident and
    /// accumulate through a large multi-file ingest. Returns the number of turn
    /// residences flagged.
    ///
    /// Reclaims VRAM **immediately** for any turn already warm-backed: it drops
    /// the hot copy on the spot (keeping warm — a safe demote, since the RAM copy
    /// survives and `elevate_to_hot` reloads on demand). This is the load-bearing
    /// case: a file usually completes AFTER its turns have already migrated
    /// hot→warm, so `install_warm_and_evict_hot` (which only fires DURING the
    /// migrate, for residences flagged beforehand) never ran for them — nothing
    /// else drops their hot until the async, possibly-lagging cold write lands.
    /// Without this, completed files' hot KV piles up and fills the card mid-ingest.
    ///
    /// The dropped copy is freed on the device's thread once the last forward
    /// reading it lets go, so the drop is safe from any thread.
    pub fn mark_timeline_evict_when_cold(&self, timeline: TimelineId) -> usize {
        let Some(entry) = self.timelines.get(&timeline) else {
            return 0;
        };
        let mut n = 0usize;
        for t in entry.turns.values() {
            let idx = t.content.residence;
            let r = self.residences.get(idx);
            r.set_flag(EVICT_WHEN_COLD);
            // Immediate hot-drop for already-warm turns (VRAM back now, warm kept)
            // — UNLESS a working set holds the residence (a concurrent slot
            // selected this just-completed file), in which case defer: the flag
            // stays set and the idle demote sheds it once it leaves the set.
            if r.has_hot() && r.has_warm() && self.may_shed(idx) {
                self.residences.take_hot(idx);
            }
            n += 1;
        }
        n
    }

    /// How many of `timeline`'s turn residences still hold a device (hot) copy.
    ///
    /// What a caller that evicted a timeline polls to learn when its KV has left
    /// the device: [`Self::mark_timeline_evict_when_cold`] drops the hot copy of
    /// every warm-backed turn at once, and the persistence thread drops the rest
    /// as each one's warm copy lands. Zero for an unknown timeline.
    pub fn timeline_hot_residences(&self, timeline: TimelineId) -> usize {
        self.timelines.get(&timeline).map_or(0, |entry| {
            entry
                .turns
                .values()
                .filter(|t| self.residences.get(t.content.residence).has_hot())
                .count()
        })
    }

    /// Flag a SINGLE turn's residence for full eviction once its KV is durable —
    /// see [`Self::mark_timeline_evict_when_cold`]. Set at splice time (while the
    /// turn is still hot, before the hot→warm migrate) so the migrate's
    /// `install_warm_and_evict_hot` drops its Q-format hot the moment warm lands,
    /// instead of installing a resident Q copy. Load-bearing for code_read: a
    /// scope turn adopted onto its file timeline is never re-attended until query
    /// time, and the file timeline is NOT in the scheduler's `ingest_timelines`,
    /// so the gentle demote never covers it — without this its migrated Q hot
    /// copy accumulates and climbs the card (`quant_live` leak).
    pub fn mark_turn_evict_when_cold(&self, timeline: TimelineId, idx: TurnIndex) {
        if let Some(r) = self.turn(timeline, idx).map(|t| t.content.residence) {
            self.residences.get(r).set_flag(EVICT_WHEN_COLD);
        }
    }

    /// Snapshot indices of hot-resident slots that lack a warm copy —
    /// the work list for the persistence thread's hot→warm phase. Each
    /// entry pairs the residence index with a clone of its hot bytes so
    /// the thread can drop the substrate read lock before doing the
    /// (slow) CUDA-side `migrate_sealed_to_cpu`.
    ///
    /// The clone of `Vec<SealedSequence>` is cheap — the inner
    /// `SealedChunk`s hold `Arc<ChunkGid>` references; clone bumps the
    /// arena refcount but doesn't copy KV bytes.
    pub fn snapshot_pending_warm(
        &self,
    ) -> Vec<(ResidenceIndex, Vec<SealedSequence>, Option<ConvCompression>)> {
        self.pending_warm()
            .map(|(idx, hot, cc)| (idx, hot.to_vec(), cc))
            .collect()
    }

    /// The distinct compression overrides among the residences
    /// [`Self::snapshot_pending_warm`] would return — the persistence thread's
    /// hot→warm groups, named without cloning a single sequence.
    pub fn pending_warm_policies(&self) -> Vec<Option<ConvCompression>> {
        let mut out: Vec<Option<ConvCompression>> = Vec::new();
        for (_, _, cc) in self.pending_warm() {
            if !out.contains(&cc) {
                out.push(cc);
            }
        }
        out
    }

    /// [`Self::snapshot_pending_warm`] restricted to one compression override:
    /// one hot→warm group, cloned when the persistence thread is about to
    /// migrate it and not before.
    pub fn snapshot_pending_warm_for(
        &self,
        policy: Option<ConvCompression>,
    ) -> Vec<(ResidenceIndex, Arc<HotKv>)> {
        self.pending_warm()
            .filter(|(_, _, cc)| *cc == policy)
            .map(|(idx, hot, _)| (idx, hot))
            .collect()
    }

    /// The oldest turn residence awaiting its warm copy whose hot copy still
    /// holds native chunks — a turn whose seal gap the ground could not cover
    /// — with its compression override, for the gap to quantize. A lossless
    /// conversation's turns stay native and are never named.
    pub fn native_pending_warm(
        &self,
    ) -> Option<(ResidenceIndex, Arc<HotKv>, Option<ConvCompression>)> {
        self.pending_warm().find(|(_, hot, cc)| {
            !cc.is_some_and(|c| c.lossless)
                && hot
                    .iter()
                    .any(|s| s.chunks.iter().any(|c| c.token_count > 0 && c.is_native()))
        })
    }

    /// Swap `quantized` in as `residence`'s hot copy — the gap's quantize of
    /// `read` — only while the residence still holds `read`'s content. A
    /// residence whose hot copy changed or went keeps what it has, and
    /// `quantized` is dropped. Answers whether it was swapped in.
    pub fn install_gap_quantized(
        &self,
        residence: ResidenceIndex,
        read: &Arc<HotKv>,
        quantized: Vec<SealedSequence>,
    ) -> bool {
        self.residences.replace_hot(residence, read, quantized)
    }

    /// Every turn's hot residence awaiting a warm copy, with its hot bytes and
    /// its compression override — what the hot→warm phase works through.
    /// Sections go hot→cold and never take a warm copy.
    pub(super) fn pending_warm(
        &self,
    ) -> impl Iterator<Item = (ResidenceIndex, Arc<HotKv>, Option<ConvCompression>)> + '_ {
        self.hot_turns_oldest_first().into_iter().filter_map(|idx| {
            let r = self.residences.get(idx);
            if r.has_warm() || r.has_cold() || self.pin_defers_warm(idx) {
                return None;
            }
            r.hot().map(|hot| (idx, hot, r.compression))
        })
    }

    /// Whether the working set's pin on `idx` defers its warm copy.
    ///
    /// **A residence with a cold copy never takes one at all** — one recalled
    /// hot from its cold record, or whose warm copy the RAM budget purged while
    /// it was hot. Its hot copy is the stored form, and the log already holds
    /// those bytes: a hot→warm copy of it is a device-to-pageable transfer on
    /// the primary stream that buys nothing the cold read does not. Measured on
    /// BlackWorld (2026-11-22), once recalls stopped being shed straight back to
    /// cold: 145–382 MiB copied per pass at ~0.7 GB/s, 100–540 ms of the stream
    /// a thinking block's decode shares. [`Self::pending_warm`] skips them, and
    /// the idle demote sheds their hot copy onto the cold one.
    ///
    /// **Deferred while pinned for one whose cold write is in flight.**
    /// Backing it up hot→warm while the in-flight decode attends it is
    /// redundant work that monopolises the persistence thread and stream
    /// against the elevation the decode is blocked on: the tier-migration
    /// livelock, the same working set churning every pass while the decode lands
    /// zero forwards. Its warm copy waits until it leaves the working set.
    ///
    /// **Never for a fresh seal.** Its hot copy is the native form the forward
    /// wrote, and the hot→warm migrate is also what swaps that for the quantized
    /// form (`install_warm_and_hot`). Deferred while pinned, a turn its own
    /// conversation keeps attending stayed native for the conversation's whole
    /// life: on BlackWorld (2026-10-27) one `code_read` of a 67-page file held
    /// 160k tokens at 23.6 KB/token — 3.6 GB — where its persisted turns read
    /// 8.1 KB/token. It is migrated once, as any seal is; it then has a warm
    /// copy, so it is never a candidate again.
    pub(super) fn pin_defers_warm(&self, idx: ResidenceIndex) -> bool {
        let r = self.residences.get(idx);
        r.is_pinned() && r.has_flag(COLD_PENDING)
    }

    /// Total VRAM byte footprint of hot residences that lack a warm copy — the
    /// hot→warm drain **backlog**. Cheap companion to [`Self::snapshot_pending_warm`]
    /// (which clones every hot `SealedSequence` for the migration): this only
    /// sums each slot's cached `byte_size`, so the persistence thread can stamp
    /// it every pass and the scheduler can poll it to drive ingest admission
    /// backpressure off a *leading* signal (drain deficit) rather than the
    /// lagging VRAM-pressure trip.
    pub fn pending_warm_bytes(&self) -> u64 {
        // Matches `snapshot_pending_warm`'s pin skip: a deferred residence is
        // NOT drainable this pass, so counting it would report a backlog the
        // drain intentionally never clears — the admission backpressure would
        // then throttle forever on a phantom deficit.
        self.pending_warm()
            .map(|(idx, _, _)| self.residences.get(idx).byte_size())
            .sum()
    }

    /// Number of residences holding a warm (host RAM) copy — the population the
    /// RAM purge draws its victims from. The persistence thread gates its purge
    /// on this being non-zero, so a workspace with nothing warm never pays the
    /// OS memory query.
    pub fn warm_resident_count(&self) -> usize {
        self.residences
            .warm_members()
            .into_iter()
            .filter(|&idx| self.residences.get(idx).has_warm())
            .count()
    }

    /// Total bytes held by warm (host RAM) copies — the byte size of the purge
    /// population. This is the one host-side quantity admission throttling can
    /// actually shrink, so the memory report needs it in bytes, not just as a
    /// residence count. O(warm residences), taken once per persistence pass.
    pub fn warm_resident_bytes(&self) -> u64 {
        self.residences
            .warm_members()
            .into_iter()
            .map(|idx| self.residences.get(idx))
            .filter(|r| r.has_warm())
            .map(|r| r.byte_size())
            .sum()
    }

    /// Host bytes held by the per-turn payloads the substrate keeps resident
    /// whole, rebuilt from the redo log at open: each stream's wide-Q
    /// signature window and its projection events. These sit outside the warm
    /// KV budget and grow with the corpus. O(streams), for the memory report.
    pub fn held_payload_bytes(&self) -> HeldPayloadBytes {
        let mut held = HeldPayloadBytes::default();
        for (_, stream) in self.streams.iter_physical() {
            held.wide_q_sigs += stream.wide_q_sigs.as_ref().map_or(0, |b| b.len() as u64);
            held.projection_events += stream
                .projection_events
                .as_ref()
                .map_or(0, |b| b.len() as u64);
        }
        held
    }

    /// Count of hot residences awaiting a warm copy — the cheap COUNT companion to
    /// [`Self::snapshot_pending_warm`] (which clones every pending `SealedSequence`
    /// for the migration). Mirrors that snapshot's filter exactly (skip
    /// already-warm, skip the pinned working set), so the shutdown drain can poll
    /// convergence without allocating the whole work list each pass.
    pub fn pending_warm_count(&self) -> usize {
        self.pending_warm().count()
    }

    /// The VRAM-resident KV by storage format: every residence holding a hot
    /// copy — turns and sections alike — counted in token-equivalents
    /// ([`FormatCensus`]).
    pub fn hot_format_census(&self) -> FormatCensus {
        let mut census = FormatCensus::default();
        for idx in self.residences.hot_members() {
            if let Some(hot) = self.residences.get(idx).hot() {
                census.add_layers(&hot);
            }
        }
        census
    }

    /// Diagnostic split of the hot-without-warm set into (drainable, pinned-skip):
    /// `(drainable_count, drainable_bytes, pinned_count, pinned_bytes)`. The
    /// persistence thread logs it each pass so the tier-migration livelock is
    /// legible: a large `pinned_*` with near-zero `drainable_*` is the working set
    /// correctly deferred (the fix engaged); a large `drainable_*` that never
    /// falls is a genuine drain that the migrate can't keep up with.
    pub fn warm_backlog_split(&self) -> (usize, u64, usize, u64) {
        let (mut dc, mut db, mut pc, mut pb) = (0usize, 0u64, 0usize, 0u64);
        for idx in self.hot_turns_oldest_first() {
            let r = self.residences.get(idx);
            if !r.has_hot() || r.has_warm() || r.has_cold() {
                continue;
            }
            if self.pin_defers_warm(idx) {
                pc += 1;
                pb += r.byte_size();
            } else {
                dc += 1;
                db += r.byte_size();
            }
        }
        (dc, db, pc, pb)
    }
}
