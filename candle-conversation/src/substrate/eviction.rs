//! Chooses which hot and warm KV copies the substrate drops to free VRAM or host
//! RAM, and keeps every live working set out of reach.
//!
//! A hot copy is dropped only where a warm copy already holds the same bytes, and
//! a residence a live slot's wave attends is never dropped. That protection is
//! stamped with the current epoch, so the staleness test excludes it
//! arithmetically rather than by a filter that could drift out of step. Warm RAM
//! is shed only where its data survives elsewhere, except for a transient
//! residence whose cold copy is never coming. Each purge is bounded per pass, so
//! an unreachable budget degrades to steady shedding instead of draining the
//! tier on every call.

use super::*;

impl Substrate {
    /// The hot KV the working-set pins hold, by the timeline whose turns they
    /// are, largest first and at most `top`: `(timeline, turns, bytes)`. With
    /// the total over every timeline and the pins on sections, which no
    /// timeline owns.
    ///
    /// What the pinned half of [`Self::warm_backlog_split`] is made of. That
    /// half is the hot KV no drain may move while a live slot's working set
    /// names it, so when it grows the question is whose working set — and a
    /// count cannot say.
    pub fn pinned_hot_by_timeline(&self, top: usize) -> PinnedHot {
        let mut by_timeline: Vec<(TimelineId, usize, u64)> = Vec::new();
        let mut turn_bytes = 0u64;
        for (&timeline, entry) in &self.timelines {
            let (mut turns, mut bytes) = (0usize, 0u64);
            for t in entry.turns.values() {
                let r = self.residences.get(t.content.residence);
                if r.has_hot() && r.is_pinned() {
                    turns += 1;
                    bytes += r.byte_size();
                }
            }
            if turns > 0 {
                turn_bytes += bytes;
                by_timeline.push((timeline, turns, bytes));
            }
        }
        by_timeline.sort_unstable_by(|a, b| b.2.cmp(&a.2).then(a.0.raw().cmp(&b.0.raw())));
        by_timeline.truncate(top);
        let pinned = self.residences.pinned();
        let pinned_bytes: u64 = pinned
            .iter()
            .map(|&idx| self.residences.get(idx))
            .filter(|r| r.has_hot())
            .map(|r| r.byte_size())
            .sum();
        PinnedHot {
            pins: pinned.len(),
            pinned_bytes,
            turn_bytes,
            by_timeline,
        }
    }

    /// Snapshot indices of warm-resident slots that lack a cold (on-disk) record
    /// — the work list for the persistence thread's warm→cold phase. Pairs each
    /// index with the slot's [`StreamId`] and a clone of its **warm** bytes (the
    /// authoritative stored form: in the quantize-on-evict path warm holds the
    /// compressed chunks while hot still holds the source floats, so the redo log
    /// must capture warm — see the in-body note). Walks the warm set, so hot
    /// residency is irrelevant here: a warm-only slot whose hot was evicted is
    /// STILL captured (this is why the shutdown drain reaches every warm turn).
    /// Skips slots whose cold copy already landed or is queued (`COLD_PENDING`);
    /// a working-set pin skips nothing ([`Self::cold_pending`]).
    pub fn snapshot_pending_cold(&self) -> Vec<(ResidenceIndex, StreamId, Vec<SealedSequence>)> {
        // We persist the **warm** tier, not hot. In the quantize-on-evict path
        // warm holds the compressed chunks and hot still holds the source
        // floats; reading warm is the only way the redo log captures the actual
        // stored form.
        self.warm_oldest_first()
            .into_iter()
            .filter_map(|idx| {
                let slot = self.residences.get(idx);
                if !self.cold_pending(idx) {
                    return None;
                }
                slot.warm()
                    .map(|warm| (idx, slot.stream_id, (*warm).clone()))
            })
            .collect()
    }

    /// Whether `idx` is a warm residence the warm→cold phase may write now.
    ///
    /// **A working-set pin never defers it.** A pin says a live slot reads the
    /// residence, which governs eviction, not durability: the write reads the
    /// warm copy out of its CPU arenas (`seal_to_chunk_images_cpu`), touches no
    /// device memory and no stream, and [`Self::install_cold`] keeps a pinned
    /// residence's hot and warm copies. Deferred while pinned, a conversation's
    /// own turns — pinned for as long as it is active — reached the redo log
    /// only once it went quiet: a kill on BlackWorld (2026-11-21) brought one
    /// back with its turns' tokens and no K/V, empty to the projection and
    /// never resumable.
    pub(super) fn cold_pending(&self, idx: ResidenceIndex) -> bool {
        let r = self.residences.get(idx);
        // Skip a turn whose cold copy already landed OR whose async cold write
        // is still queued on the writer — re-selecting the latter would
        // double-write the same KV. A transient (scratch-fork) residence never
        // cold-persists: its KV is spliced by reference onto a durable file
        // timeline, so a cold copy would only strand an orphan when the fork's
        // decl drops at tombstone.
        r.has_warm() && !r.has_cold() && !r.has_flag(COLD_PENDING | NO_COLD_PERSIST)
    }

    /// Count of warm residences awaiting a cold write — the cheap COUNT companion
    /// to [`Self::snapshot_pending_cold`] (which clones every pending warm
    /// `SealedSequence`). Mirrors that snapshot's filter exactly (skip the pinned
    /// working set, skip slots already cold or with a queued cold write), so the
    /// shutdown drain can poll convergence without cloning the work list.
    pub fn pending_cold_count(&self) -> usize {
        self.residences
            .warm_members()
            .into_iter()
            .filter(|&idx| self.cold_pending(idx))
            .count()
    }

    /// Flag a residence's warm→cold write as in flight on the off-thread writer,
    /// so [`Self::snapshot_pending_cold`] won't re-gather it before the write lands
    /// (which clears the flag via `install_cold`). Set at enqueue.
    pub fn mark_cold_pending(&self, residence: ResidenceIndex) {
        self.residences.get(residence).set_flag(COLD_PENDING);
    }

    /// Classify a batch of items (sections + turns) by what work
    /// `elevate_to_hot` will need to do to put each one in VRAM.
    ///
    /// One read-lock walk over the substrate; shares each residence's warm
    /// `Vec<SealedSequence>` and cold `Vec<StoredSequence>` — one refcount
    /// each — so the caller can drop the lock before running the slow CUDA
    /// work.
    pub fn snapshot_promotion_state(
        &self,
        sections: &[SectionId],
        turns: &[TurnKey],
    ) -> PromotionPlan {
        let mut plan = PromotionPlan::default();
        for &sid in sections {
            let Some(entry) = self.sections.get(&sid) else {
                plan.missing.push(PromotionItemKind::Section(sid));
                continue;
            };
            self.classify_one(PromotionItemKind::Section(sid), entry.residence, &mut plan);
        }
        for &key in turns {
            let Some(entry) = self.turn(key.timeline, key.index) else {
                plan.missing.push(PromotionItemKind::Turn(key));
                continue;
            };
            // Classify the assistant residence — the one carrying the
            // turn's K/V under today's seal path.  The user residence
            // is reserved for the Phase 5 `NewUserMessage` capture and
            self.classify_one(
                PromotionItemKind::Turn(key),
                entry.content.residence,
                &mut plan,
            );
        }
        plan
    }

    /// Helper for [`Self::snapshot_promotion_state`]: route one item
    /// into the right bucket of the plan based on its residence's tier
    /// occupancy.
    pub(super) fn classify_one(
        &self,
        kind: PromotionItemKind,
        residence: ResidenceIndex,
        plan: &mut PromotionPlan,
    ) {
        let slot = self.residences.get(residence);
        if slot.has_hot() {
            plan.already_hot.push(kind);
            return;
        }
        if let Some(warm) = slot.warm() {
            plan.warm_to_hot.push(WarmToHotEntry {
                kind,
                residence,
                warm,
            });
            return;
        }
        if let Some(cold) = slot.cold() {
            plan.cold_to_hot.push(ColdToHotEntry {
                kind,
                residence,
                cold,
                stream_id: slot.stream_id,
            });
            return;
        }
        // The item is tracked in the substrate but its residence has
        // no tier installed.  This is the expected state for ghost
        // summary turns appended via `append_with_blocks(0..0)` —
        // they exist as tree-meta anchors without any K/V to load.
        // Route to `tier_less`, not `missing`, so `elevate_to_hot`'s
        // WARN doesn't fire for the design-intended case.
        plan.tier_less.push(kind);
    }

    /// Bulk install of freshly-promoted bytes — both elevation legs.
    ///
    /// `recalls` lands residences pulled out of cold storage: hot only, since
    /// the cold record already makes them durable and the persistence thread
    /// produces the warm copy off this path. `lifts` lands residences lifted
    /// from the warm cache: hot only, warm is already in place.
    ///
    /// **A promotion lands only on an empty, live cell.** Two elevations of one
    /// residence built two copies; the second finds the first installed and
    /// drops its own, so copies never stack. A residence whose owner was
    /// tombstoned while its copy was being built takes nothing. Empty payloads
    /// are skipped: a phase that produced nothing usable leaves the residence
    /// as it was. Returns how many were installed.
    ///
    /// **A promotion ends a residence's offload.** `EVICT_WHEN_COLD` says the
    /// KV is not attended again until retrieval pulls it back, and a promotion
    /// is that retrieval. Left set, the flag sheds the recall's hot *and* warm
    /// copies the first pass after it leaves the working set — no idle grace,
    /// no RAM budget consulted — so a projection that re-selects it a boundary
    /// later reads it off the log again. Measured on BlackWorld (2026-11-22):
    /// 15–36 code_read turns, 120–236 MiB, recalled cold at every 64-token
    /// boundary of one thinking block, 75–180 ms of each rebuild. Cleared, the
    /// recall ages like any residence: its hot copy goes after
    /// [`IDLE_DEMOTE_GRACE_EPOCHS`] once warm holds it, and its warm copy goes
    /// to the RAM budget's LRU purge.
    pub fn install_promoted(&self, recalls: Vec<ColdRecall>, lifts: Vec<WarmLift>) -> usize {
        let mut installed = 0usize;
        let promoted = recalls
            .into_iter()
            .map(|r| (r.residence, r.hot))
            .chain(lifts.into_iter().map(|l| (l.residence, l.hot)));
        for (residence, hot) in promoted {
            if !hot.is_empty() && self.residences.install_hot_if_absent(residence, hot) {
                self.residences.get(residence).clear_flag(EVICT_WHEN_COLD);
                installed += 1;
            }
        }
        installed
    }

    /// The residence-slab index that a turn addresses: what a tier
    /// transition from outside the substrate's own write methods installs
    /// on (`install_warm` for a warm-on-request read, the tests' `install_cold`
    /// and `evict_hot_except`). Returns `None` when the turn isn't tracked.
    pub fn turn_residence(&self, timeline: TimelineId, index: TurnIndex) -> Option<ResidenceIndex> {
        self.turn(timeline, index).map(|e| e.content.residence)
    }

    /// Resolve a section's residence index in the substrate.
    ///
    /// Used by the scheduler's section-quantize drain to look up the
    /// hot slot it needs to swap, and by integration tests that walk
    /// the section map directly.
    pub fn section_residence(&self, section: SectionId) -> Option<ResidenceIndex> {
        self.sections.get(&section).map(|e| e.residence)
    }

    /// The persisted stream `section` is registered against — its content
    /// address's stream id. `None` when the section is not registered.
    pub fn section_stream(&self, section: SectionId) -> Option<StreamId> {
        self.sections
            .get(&section)
            .map(|e| self.residences.get(e.residence).stream_id)
    }

    /// Every section id currently registered in the substrate, in
    /// unspecified order.  Used by tooling that needs to walk the
    /// section map — integration tests, workspace diagnostics, the
    /// section-quantize regression test in `zend/tests/`.
    pub fn all_section_ids(&self) -> Vec<SectionId> {
        self.sections.keys().copied().collect()
    }

    /// Which tiers a turn residence currently occupies. Returns `None`
    /// when the turn isn't tracked. Load-bearing in production: the
    /// SubmitTurn handler uses this as a tier-agnostic existence
    /// check before adding a turn to the projection's elevate list,
    /// so cold-marker turns (post-restart, before any elevation has
    /// fired) still survive the filter and reach `elevate_to_hot`.
    pub fn turn_tier_state(&self, timeline: TimelineId, index: TurnIndex) -> Option<TierState> {
        let residence = self.turn(timeline, index).map(|e| e.content.residence)?;
        Some(self.tier_state(residence))
    }

    pub(super) fn tier_state(&self, residence: ResidenceIndex) -> TierState {
        let r = self.residences.get(residence);
        TierState {
            hot: r.has_hot(),
            warm: r.has_warm(),
            cold: r.has_cold(),
        }
    }

    /// Section counterpart of [`Self::turn_tier_state`].  Sections
    /// can now occupy any of the three tiers: hot when prefilled
    /// fresh or post-elevate, cold-marker when restored from the
    /// redo log on daemon reload (lazy lift on next projection),
    /// and warm transiently during a tier transition.  Returns
    /// `None` for unknown section ids.
    pub fn section_tier_state(&self, section: SectionId) -> Option<TierState> {
        let residence = self.sections.get(&section).map(|e| e.residence)?;
        Some(self.tier_state(residence))
    }

    /// Ceiling on warm bytes one [`Self::purge_warm_to_target`] call may free.
    ///
    /// The purge chases a *total free RAM* threshold, which a process holding a
    /// large non-pageable allocation can be structurally unable to reach — an
    /// 11 GB pinned MoE expert pool on a 32 GB box is never released, so no
    /// amount of warm shedding lifts free RAM over a 2 GiB bar. This bound turns
    /// that unreachable case from "drain the entire warm tier every pass, and
    /// pay a cold re-fetch on every subsequent read" into a slow trickle, while
    /// leaving the reachable case (transient warm growth during a bulk ingest)
    /// fully served — 1 GiB is several passes' worth of migration output.
    pub(super) const WARM_PURGE_MAX_BYTES_PER_PASS: u64 = 1024 * 1024 * 1024;

    /// Drop warm-tier residences from the LRU tail until the warm tier fits its
    /// host-RAM **budget** (`candle::vram::host_ram_budget` — total minus the OS
    /// buffer, the pinned pools, and the fully-reserved weights), leaving room
    /// for `incoming_bytes` about to be installed.
    ///
    /// Budget-based, not availability-based. The old threshold compared OS
    /// "available" against `max(2 GiB, 5%)` — a number the mmap'd weights push
    /// down as the page cache fills, so on a box whose model fills RAM the purge
    /// drained the ENTIRE warm tier every pass chasing a target no amount of
    /// purging could reach. Against the budget, a zero-budget machine correctly
    /// keeps warm transient-only, and a large-RAM machine finally gets to RETAIN
    /// its warm tier up to the computed ceiling instead of an arbitrary floor.
    /// The caller passes the budget — keeping the gauge/env reads at the
    /// orchestrator level so the substrate stays dependency-free and
    /// unit-testable.
    ///
    /// **Invariants:**
    /// - Only warm copies are taken; hot and cold are untouched. A purged
    ///   residence that's also hot stays hot; its next hot eviction will need a
    ///   fresh hot→warm DMA.
    /// - Recency-ordered: the least recently touched warm copy goes first.
    /// - Single batch: no intermediate work between victims.
    /// - **Bounded**: at most [`WARM_PURGE_MAX_BYTES_PER_PASS`] per call.
    ///
    /// The per-pass bound is what makes an UNREACHABLE target safe. The
    /// threshold counts total free RAM, but a large share of this process's
    /// footprint can be non-pageable and permanent — an 11 GB pinned MoE expert
    /// pool on a 32 GB box leaves free RAM structurally under a 2 GiB
    /// threshold no matter how much warm KV is shed. Unbounded, the loop then
    /// drains the ENTIRE warm tier every pass chasing a target it can never
    /// reach, and every subsequent read pays a cold re-fetch. Bounded, the same
    /// state degrades to steady incremental shedding: real relief when the
    /// target is reachable, a slow trickle when it is not.
    ///
    /// Returns a [`PurgeReport`] with `count` victims and `bytes`
    /// freed (sum of `residence.byte_size`).
    pub fn purge_warm_to_budget(&self, warm_budget: u64, incoming_bytes: u64) -> PurgeReport {
        // Room the incoming install needs comes out of the same budget.
        let target: u64 = warm_budget.saturating_sub(incoming_bytes);
        let mut warm_now: u64 = self.warm_resident_bytes();
        let mut freed_bytes: u64 = 0;
        let mut count: usize = 0;
        for idx in self.warm_oldest_first() {
            if warm_now <= target {
                break;
            }
            // Per-pass bound: an unreachable target must degrade to incremental
            // shedding, not a full drain of the warm tier on every pass.
            if freed_bytes >= Self::WARM_PURGE_MAX_BYTES_PER_PASS {
                break;
            }
            // Only drop warm when the data survives elsewhere — hot in VRAM or
            // cold on disk. A warm-only residence (its cold write hasn't landed,
            // or failed) is its turn's ONLY copy; dropping it would lose the K/V.
            // It is passed over, and a later pass reclaims it once a lower tier
            // lands.
            //
            // **Except when its cold copy is never coming.** `no_cold_persist`
            // says this K/V is disposable and will not be written to disk — a
            // code_read fork, a calibration exemplar whose signature is already
            // captured. For those, "wait for cold to land" waits forever, so the
            // guard turns a temporary skip into permanently unreclaimable host
            // RAM. Measured on a full calibration: warm climbed to 8.8 GiB and
            // could not be shed, past the level that had already taken the
            // machine down once.
            //
            // The pin check replaces the safety the copy rule was providing.
            // Warm-only means the residence is not on the GPU and so is not
            // being attended, but `elevate_to_hot` publishes pins before it
            // reads warm to build hot — and this purge is the one warm-dropping
            // path that does not otherwise consult them. A lift that read warm
            // before the take holds its own reference to it.
            let slot = self.residences.get(idx);
            if slot.has_warm() && !slot.has_cold() && !slot.has_hot() {
                let disposable = slot.has_flag(NO_COLD_PERSIST) && !slot.is_pinned();
                if !disposable {
                    continue;
                }
            }
            if self.residences.take_warm(idx).is_some() {
                let bytes = slot.byte_size();
                freed_bytes = freed_bytes.saturating_add(bytes);
                warm_now = warm_now.saturating_sub(bytes);
                count += 1;
                tracing::trace!(
                    target: "candle_conversation::persistence::tier",
                    residence = idx.0,
                    bytes,
                    "purged warm (RAM headroom)"
                );
            }
        }
        if count > 0 {
            tracing::trace!(
                target: "candle_conversation::persistence::tier",
                count,
                bytes = freed_bytes,
                warm_budget,
                incoming_bytes,
                "warm purge batch complete"
            );
        }
        PurgeReport {
            count,
            bytes: freed_bytes,
        }
    }

    /// Drop hot bytes from every warm-backed hot residence **except**
    /// the ones whose sections / turns are in the keep set. Steady-state
    /// working-set turnover: when a projection has produced a new
    /// working set, the items in `keep_sections` + `keep_turns` are
    /// about to be elevated (or are already hot), so evicting them
    /// would just waste a DMA round-trip.
    ///
    /// The keep-set membership is checked by **residence index** —
    /// resolved here under the same write lock from the substrate's
    /// own `sections` map and per-timeline turn lookups. Unresolvable
    /// items in the keep set are silently ignored (they can't match a
    /// real residence anyway).
    ///
    /// Invariants:
    /// - Only residences holding both a hot and a warm copy are eligible.
    /// - Sections are never taken: this only ever touches turn residences.
    /// - Returned [`EvictionReport`] sums the dropped residences.
    ///
    /// Emits one `tracing::debug!` per evicted residence and an
    /// aggregate `tracing::info!` at the end, both on the
    /// `candle_conversation::persistence::tier` target.
    pub fn evict_hot_except(
        &self,
        keep_sections: &[SectionId],
        keep_turns: &[TurnKey],
    ) -> EvictionReport {
        let keep = self.keep_residences(keep_sections, keep_turns);
        let (mut count, mut bytes) = (0usize, 0u64);
        for idx in self.hot_turns_oldest_first() {
            if keep.contains(&idx) || !self.evictable_hot(idx) {
                continue;
            }
            if self.residences.take_hot(idx).is_none() {
                continue;
            }
            let b = self.residences.get(idx).byte_size();
            tracing::debug!(
                target: "candle_conversation::persistence::tier",
                residence = idx.0,
                bytes = b,
                "evicted hot (working-set turnover)"
            );
            count += 1;
            bytes += b;
        }
        if count > 0 {
            tracing::info!(
                target: "candle_conversation::persistence::tier",
                count,
                bytes,
                keep_size = keep.len(),
                "evict_from_hot batch complete"
            );
        }
        EvictionReport { count, bytes }
    }

    /// Budget-aware hot eviction: evict the **least-recently-promoted** hot
    /// turns to warm, stopping as soon as
    /// `target_bytes` of VRAM has been freed. Unlike [`Self::evict_hot_except`]
    /// (which drops the entire non-selected working set every reproject), this
    /// keeps the recent working set resident and only frees what the incoming
    /// cold-load actually needs.
    ///
    /// The selection (`keep_sections` / `keep_turns`) and sections are always
    /// protected. Only items with both a hot and a warm copy are evicted, so
    /// eviction is hot→warm (the warm copy survives for a fast reload).
    ///
    /// **Every live working set is protected too**, not only the projection
    /// being elevated: a residence another slot's wave attends is in use.
    pub fn evict_hot_to_free(
        &self,
        keep_sections: &[SectionId],
        keep_turns: &[TurnKey],
        target_bytes: u64,
    ) -> EvictionReport {
        self.evict_hot_inner(keep_sections, keep_turns, target_bytes, true)
    }

    /// [`Self::evict_hot_to_free`] for a guest's drain, with no live working
    /// set protected — only `keep_*`.
    ///
    /// **Nothing forwards for the length of a drain.** A slot's next
    /// projection lifts what it selects warm→hot before that turn's first
    /// forward, so what is evicted here reloads from RAM then. Held, it was the
    /// draw: on WhiteWorld (2026-10-27 09:37) one dialogue's 16 GB working set
    /// left a draw 6.8 GB short with nothing in flight, and the draw failed.
    pub fn evict_hot_for_guest(
        &self,
        keep_sections: &[SectionId],
        keep_turns: &[TurnKey],
        target_bytes: u64,
    ) -> EvictionReport {
        self.evict_hot_inner(keep_sections, keep_turns, target_bytes, false)
    }

    pub(super) fn evict_hot_inner(
        &self,
        keep_sections: &[SectionId],
        keep_turns: &[TurnKey],
        target_bytes: u64,
        protect_live: bool,
    ) -> EvictionReport {
        if target_bytes == 0 {
            return EvictionReport { count: 0, bytes: 0 };
        }
        let mut keep = self.keep_residences(keep_sections, keep_turns);
        if protect_live {
            // Every live working set, not only the projection being elevated:
            // a residence another slot's wave attends is as much in use as this
            // one's.
            keep.extend(self.residences.pinned());
        }

        // Walk oldest→newest, taking victims until we've freed enough.
        // `byte_size` is the VRAM the hot copy holds.
        let mut freed: u64 = 0;
        let mut count = 0usize;
        for idx in self.hot_turns_oldest_first() {
            if freed >= target_bytes {
                break;
            }
            if keep.contains(&idx) || !self.evictable_hot(idx) {
                continue;
            }
            if self.residences.take_hot(idx).is_some() {
                freed += self.residences.get(idx).byte_size();
                count += 1;
            }
        }
        if count > 0 {
            tracing::info!(
                target: "candle_conversation::persistence::tier",
                count,
                bytes = freed,
                target_bytes,
                keep_size = keep.len(),
                "evict_hot_to_free (budget-aware) complete"
            );
        }
        EvictionReport {
            count,
            bytes: freed,
        }
    }

    /// [`Self::evict_hot_to_free`]'s view of this substrate's hot bytes,
    /// without evicting anything: what it would take, and what it protects
    /// and why. The first reason that applies is the one counted, in the order
    /// the eviction checks them.
    pub fn hot_census(&self, keep_sections: &[SectionId], keep_turns: &[TurnKey]) -> HotCensus {
        let mut census = HotCensus::default();
        let keep = self.keep_residences(keep_sections, keep_turns);
        let pinned: HashSet<ResidenceIndex> = self.residences.pinned().into_iter().collect();
        for e in self.sections.values() {
            let r = self.residences.get(e.residence);
            if r.has_hot() {
                census.sections += r.byte_size();
            }
        }
        for idx in self.hot_turns_oldest_first() {
            let r = self.residences.get(idx);
            let b = r.byte_size();
            if keep.contains(&idx) {
                census.attended += b;
            } else if pinned.contains(&idx) {
                census.pinned += b;
            } else if !r.has_warm() {
                census.unwarmed += b;
            } else if r.has_flag(SPLICE_SOURCE) {
                census.splice += b;
            } else {
                census.evictable += b;
            }
        }
        census
    }

    /// The residences of `keep_sections` and `keep_turns`. Items the substrate
    /// doesn't know are dropped — they can't match a live residence anyway.
    pub(super) fn keep_residences(
        &self,
        keep_sections: &[SectionId],
        keep_turns: &[TurnKey],
    ) -> HashSet<ResidenceIndex> {
        let mut keep = HashSet::with_capacity(keep_sections.len() + keep_turns.len());
        for &sid in keep_sections {
            if let Some(e) = self.sections.get(&sid) {
                keep.insert(e.residence);
            }
        }
        for &key in keep_turns {
            if let Some(e) = self.turn(key.timeline, key.index) {
                keep.insert(e.content.residence);
            }
        }
        keep
    }

    /// Whether a recency-ordered eviction may take `idx`'s hot copy: it holds a
    /// warm one to fall back on and is not a splice source.
    pub(super) fn evictable_hot(&self, idx: ResidenceIndex) -> bool {
        let r = self.residences.get(idx);
        r.has_hot() && r.has_warm() && !r.has_flag(SPLICE_SOURCE)
    }

    /// Drop the hot copy of each of `turns` that also holds a warm copy, keeping
    /// warm — the cheap inverse of a warm→hot lift (no migrate, since the warm
    /// bytes already exist). Unlike [`Self::evict_hot_to_free`], which evicts the
    /// oldest *non*-kept residences until a byte target is met, this targets an
    /// explicit set: used to release a *transient* working set (e.g. a summary
    /// pass's children, lifted only so the compressor could attend over them)
    /// back out of VRAM the instant it's no longer needed, so background
    /// elevation churn can't accumulate hot residency and exhaust the card.
    /// Turns that are cold-only or hot-without-warm are left untouched (dropping
    /// hot without a warm copy would lose their K/V). Returns the number demoted.
    pub fn demote_turns_to_warm(&self, turns: &[TurnKey]) -> usize {
        let mut demoted = 0;
        for &key in turns {
            let Some(residence) = self
                .turn(key.timeline, key.index)
                .map(|e| e.content.residence)
            else {
                continue;
            };
            if self.evictable_hot(residence) && self.residences.take_hot(residence).is_some() {
                demoted += 1;
            }
        }
        demoted
    }

    /// Gentle-early relief, LRU-smart: shed the **least-recently-active** ingest
    /// hot KV to warm, freeing at most `target_bytes` (relieve to the watermark,
    /// not everything), while PROTECTING the active working set so it never
    /// churns. Only ingest KV is touched — it re-elevates from warm at zero cost,
    /// unlike a live-chat turn.
    ///
    /// Two protections keep an actively-ingesting conversation resident:
    /// 1. `keep_turns` / `keep_sections` — the union working set of every live
    ///    slot (what in-flight prefills/decodes are attending RIGHT NOW).
    /// 2. `keep_recent` — the rolling window of the newest turns per ingest
    ///    timeline (what the NEXT scope's projection will re-gather). This must
    ///    cover the projection's gather width or the demote fights it.
    ///
    /// Everything else is walked least recently touched first and demoted until
    /// `target_bytes` is met — so a settled conversation's cold tail is shed
    /// before an active one's window.
    ///
    /// Safe from any thread: a taken copy is freed on the device's thread once
    /// the last forward reading it lets go ([`residence::drain_reclaimed`]).
    pub fn demote_cold_ingest(
        &self,
        ingest_timelines: &HashSet<TimelineId>,
        keep_turns: &[TurnKey],
        keep_sections: &[SectionId],
        keep_recent: usize,
        target_bytes: u64,
    ) -> EvictionReport {
        if target_bytes == 0 {
            return EvictionReport { count: 0, bytes: 0 };
        }
        // Protected residence: live working sets + each ingest timeline's rolling
        // window. Also collect which residence belongs to an ingest timeline —
        // only those are evictable here (zero reload cost).
        let mut protected = self.keep_residences(keep_sections, keep_turns);
        protected.extend(self.residences.pinned());
        let mut ingest_residence: HashSet<ResidenceIndex> = HashSet::new();
        // Driven by the ingest set, not by a scan of every registered timeline.
        // The old form walked the whole timeline map to pick out the ingest ones,
        // so the cost grew with the size of the substrate rather than with the
        // ingest working set — on a workspace with thousands of ingested units
        // that is a full map walk per pressured wave. Same set either way: a
        // timeline had to be both registered and in `ingest_timelines` to be
        // considered, which is exactly what the lookup below yields.
        for tl in ingest_timelines {
            let Some(entry) = self.timelines.get(tl) else {
                continue;
            };
            let n = entry.turns.len();
            let cutoff = n.saturating_sub(keep_recent);
            for (i, turn_data) in entry.turns.values().enumerate() {
                let r = turn_data.content.residence;
                ingest_residence.insert(r);
                if i >= cutoff {
                    protected.insert(r); // the will-be-regathered rolling window
                }
            }
        }
        // Walk oldest→newest, demoting warm-backed ingest KV that isn't
        // protected, until we've freed `target_bytes`.
        let mut freed: u64 = 0;
        let mut count = 0usize;
        for idx in self.hot_turns_oldest_first() {
            if freed >= target_bytes {
                break;
            }
            if !ingest_residence.contains(&idx)
                || protected.contains(&idx)
                || !self.evictable_hot(idx)
            {
                continue;
            }
            if self.residences.take_hot(idx).is_some() {
                freed += self.residences.get(idx).byte_size();
                count += 1;
            }
        }
        if count > 0 {
            tracing::info!(
                target: "candle_conversation::persistence::tier",
                count,
                bytes = freed,
                keep_recent,
                target_bytes,
                "demote_cold_ingest (gentle-early, LRU) complete"
            );
        }
        EvictionReport {
            count,
            bytes: freed,
        }
    }
}
