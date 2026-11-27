//! Frees KV ground under VRAM pressure by the cheapest rung that clears it, and
//! takes expert residency from the weight side only as the last resort.
//!
//! Relief releases empty arenas first, evicts resident galleries, packs the
//! pools, evacuates the oldest warm-backed turns, and only then concedes expert
//! residency, because each rung costs more than the one before it. Nothing here
//! quantizes a turn; that belongs to the persistence thread between forwards. A
//! caller that cannot proceed without the ground waits, bounded, for the
//! persistence thread to make turns evictable, and a paced caller wakes it and
//! goes on. The working sets of live slots are protected from eviction, and an
//! idle slot's working set is not counted as live.

use super::*;

/// Safety cap on the synchronous substrate-offload flush under pressure. The
/// pass migrates hot→warm *before* its cold-disk writes, so the warm copies
/// the eviction needs exist well before this fires — a timeout only clips the
/// tail of the cold-write wait (turns are already evictable) and guards against
/// a wedged persistence thread; it is not the expected path.
pub(super) const VRAM_OFFLOAD_FLUSH_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(5);

/// Whether relief's caller needs the ground before it can go on
/// ([`Scheduler::relieve_vram_pressure`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum ReliefNeed {
    /// It cannot proceed without it: KV loaded before attention reads it
    /// (elevate), a probe or guest that runs now. Relief may wait, bounded,
    /// for the persistence thread to make turns warm and evictable.
    Now,
    /// It waits a pass when the ground is not there: admission, a section
    /// ingest, a decode's growth. Relief never holds the loop for it.
    Paced,
}

impl Scheduler {
    /// Under VRAM pressure, shed until the free-region setpoint is met again,
    /// and report whether pressure **survived** the attempt.
    ///
    /// Cheapest first, each step run only if the one before it left pressure
    /// standing:
    ///
    ///  1. **Release empty arenas.** Under the reservation this is a free-list
    ///     push per region with no device work at all, so it is always worth
    ///     trying first — §3.8's "steal an empty region from any class".
    ///  2. **Evict resident galleries.** Belief-scan pages rebuild on demand
    ///     from the substrate blob, so dropping one costs only the rebuild.
    ///     They go before model KV for exactly that reason.
    ///  3. **Pack the pools.** A KV compaction pass: arenas held by a few live
    ///     chunks are emptied into lower ones and handed back. One budgeted
    ///     device pass, and nothing resident is lost — which the two rungs after
    ///     it cannot say.
    ///  4. **Evacuate.** Flush the pending hot→warm so just-sealed turns have a
    ///     warm copy — only warm-backed turns are evictable — then drop the hot
    ///     copies of the oldest ones. This is §3.8's evict-as-evacuation, and
    ///     it runs through the demotion path the tiering already owns.
    ///  5. **Take ground from the weight side**, which costs expert residency.
    ///
    /// Nothing here quantizes: a finished turn's float→quant rewrite belongs to
    /// the persistence thread's hot→warm migrate, never to the wave that called
    /// relief.
    ///
    /// This ordering used to be the VRAM governor's relief ladder, each step a
    /// numbered `Criticality` rung with the governor re-measuring driver
    /// headroom between them to decide whether to climb. The rungs are gone:
    /// against an exact free-region count there is nothing to re-measure and
    /// nothing to arbitrate, so the priority is expressed as call order.
    ///
    /// Returns `true` if pressure is **still** on afterwards — the caller's
    /// signal to narrow the admission window, which is §3.8's third and last
    /// response. `whence` tags the log line with the calling gate.
    pub(in super::super) fn relieve_vram_pressure(
        &mut self,
        whence: &str,
        phase: VramPhase,
        need: ReliefNeed,
    ) -> bool {
        let t = std::time::Instant::now();
        let Some(want) = self.relief_shortfall_bytes(phase) else {
            return false;
        };

        // **Hand back a finished forward's transient tier before recycling
        // anything.** The tier outlives the guards that used it — a forward's
        // outputs escape into its caller — so relief, which runs between
        // forwards, can find one still standing over ground its wave no longer
        // needs. Every rung below claims regions, so it goes back first.
        //
        // This is not tidiness. `region_ceiling` is `transient_base` while a tier
        // is placed: an *address*, fixed where the last forward put it. Move the
        // weight boundary and it does not follow. So **a placed tier makes the
        // ceiling deaf to the boundary** — the last rung concedes weight-side
        // ground and the rungs above it still cannot claim a region, because the
        // cap is pinned at wherever the tier was placed. That is the shape of the
        // section-prefill wedge: the weight side conceded down to its floor
        // across thousands of retries while the ceiling never moved off 293.
        //
        // Declines while a wave generation is live, which is the one case where
        // the tier is genuinely still in use.
        if let Device::Cuda(d) = self.session.device() {
            end_wave_transient(&d.cuda_stream());
        }

        let mut released = self.session.release_empty_arenas().unwrap_or(0);
        let mut gallery_freed = 0u64;
        let mut flushed = false;
        let mut evicted = crate::substrate::EvictionReport { count: 0, bytes: 0 };

        // Gallery eviction — **this cannot clear the pressure below it**, and is
        // not here to.
        //
        // `evict_past_keep` drops `PageRun`s, returning page slots to the gallery's
        // own arenas. A region goes back to the span only when the last page in
        // its arena goes, so `gallery_freed` counts pages freed, and
        // `region_stats().free` moves only by the arenas that emptied — the next
        // `vram_under_pressure_for` can still be true.
        //
        // Gallery growth is bounded by the arena itself — it gives up the turns
        // past its keep time at admission — so this is not the only limit, and it
        // must not fire merely because KV is tight: that would shed belief-scan
        // residency the next scan rebuilds from the substrate, every episode. It
        // applies the same rule the arena does (`evict_past_keep`, the keep time
        // `update_gallery_keep` derives): only turns not scanned often enough to
        // be worth their ground go, so a working set scanned at every
        // reprojection stays. A plain LRU here shed a live dialogue's working set on every
        // decode relief — 892 MiB a wave, `relieved=false` each time since
        // scattered pages return no region — and its next scan re-uploaded it,
        // a 1.6–2.0 s index rebuild per reprojection.
        if self.vram_under_pressure_for(phase) {
            if let Some(arena) = self.gallery_arena.as_ref() {
                gallery_freed = arena.evict_past_keep();
            }
        }

        // **Pack the pools before taking anything from anyone.** Ground the KV side
        // holds as air — arenas kept alive by a handful of live chunks — is ground
        // a compaction hands back without evicting a turn or an expert, and both
        // rungs below cost one of those. Without this rung relief reached for them
        // with the pools a third air: measured on Qwen3-30B-A3B, the weight zone
        // conceded 193, 240 and 290 MiB in two seconds while 287 arenas held what
        // 185 would, and the efficiency gate read 51–64% through it.
        //
        // Unpaced: the interval bounds the census on iterations with nothing at stake,
        // and here the alternative is an eviction.
        //
        // **No rung quantizes a turn.** A finished turn's float → quant rewrite is
        // the persistence thread's, as part of its hot→warm migrate, between this
        // thread's forwards. Run from here it landed inside whatever wave called
        // relief — 0.65–2.4 s inside a 128K decode window on Flash-Next — and
        // freed nothing for a turn the decode was still attending, whose block
        // table keeps the float chunks alive until the slot goes.
        let mut packed = 0usize;
        // Where relief spends its time, stage by stage, for its log line.
        let (mut pack_ms, mut evict_ms, mut flush_ms) = (0u64, 0u64, 0u64);
        let t_stage = std::time::Instant::now();
        if self.vram_under_pressure_for(phase) {
            packed = self.pack_until_settled(PackGoal::Ground);
        }

        pack_ms += t_stage.elapsed().as_millis() as u64;
        if self.vram_under_pressure_for(phase) {
            let t_evict = std::time::Instant::now();
            evicted = self.evict_cold_tail(want);
            evict_ms += t_evict.elapsed().as_millis() as u64;
            if evicted.bytes < want {
                // The blocking flush is only paid when the already-warm turns
                // were not enough: under sustained pressure there are usually
                // plenty of them. It waits for the shortfall, not the backlog —
                // answered once that much is warm and evictable.
                let short = want.saturating_sub(evicted.bytes);
                let t_flush = std::time::Instant::now();
                // **Only a caller that cannot proceed without the ground waits
                // for it.** The wait holds the loop, and every stream with it:
                // on blackworld (2026-10-18) a promote's relief spent 1,277 ms
                // in this flush, admission backpressure that a pass later would
                // have answered by itself. A paced caller wakes the persistence
                // thread and goes on; the warm copies land behind it, and the
                // backlog falling wakes the loop.
                flushed = match need {
                    ReliefNeed::Now => super::super::timed_wait(|| {
                        self.persist_triggers
                            .flush_at_least(short, VRAM_OFFLOAD_FLUSH_TIMEOUT)
                    }),
                    ReliefNeed::Paced => {
                        self.persist_triggers.fire();
                        false
                    }
                };
                flush_ms = t_flush.elapsed().as_millis() as u64;
                let t_evict = std::time::Instant::now();
                let more = self.evict_cold_tail(want.saturating_sub(evicted.bytes));
                evict_ms += t_evict.elapsed().as_millis() as u64;
                evicted.count += more.count;
                evicted.bytes += more.bytes;
            }
            released += self.session.release_empty_arenas().unwrap_or(0);
        }
        let t_stage = std::time::Instant::now();

        // **Pack again before the weight side pays.** The evictions just above left air
        // of their own, and a pass the first rung could not get in for has had the
        // flush's wait as well. Measured on the 30B probe before this rung: 1,513 ms of
        // relief that flushed, evicted two turns and then conceded 160 MiB of expert
        // residency with `arenas_packed=0`. Whenever eviction ran, for the reason the
        // first pack runs whenever compression did.
        if evicted.count > 0 || self.vram_under_pressure_for(phase) {
            packed += self.pack_until_settled(PackGoal::Ground);
        }
        pack_ms += t_stage.elapsed().as_millis() as u64;

        // **Last resort, and the only one that adds ground rather than
        // recycling it.** Everything above reclaims KV the engine already owns —
        // compress a turn, evict a cold tail, drop an empty arena — and all of
        // it is worth nothing against a workload with nothing reclaimable. A
        // base conversation's sections are permanent by design: they are not
        // turns, so there is no turn to compress and no tail to evict, and a
        // section prefill that outgrows its ground stalls with every relief
        // counter reading zero. That is exactly how it failed.
        //
        // The weight side is holding ground in that case, and the boundary is
        // meant to move. It could not: the give-back runs at the end of a
        // completed forward, and the wave that needs it never completes. Asking
        // here breaks that circle — this is between waves, which is where the
        // move is safe, and a refusal (a wave still open, or the zone already at
        // its floor) comes back as zero rather than as a wait.
        //
        // **`want` is the ask.** It is the shortfall this pass measured against
        // the setpoint, and passing it is the whole of the fix for the run that
        // died here: the boundary used to read an accumulated count of refused
        // claims instead, which said 4,436 regions on a pass whose own `want_mib`
        // was 448 — 28 regions. It conceded 5,752 MiB, evicted 1,598 experts, and
        // put the zone under its pinned working set, after which nothing ran.
        // The number was in this function the whole time; it just was not sent.
        let mut conceded = 0u64;
        let t_stage = std::time::Instant::now();
        if self.vram_under_pressure_for(phase) {
            conceded = self
                .model
                .request_kv_ground(want.div_ceil(region_bytes()) as usize);
        }
        let concede_ms = t_stage.elapsed().as_millis() as u64;

        let still = self.vram_under_pressure_for(phase);
        let acted =
            released > 0 || gallery_freed > 0 || packed > 0 || evicted.count > 0 || conceded > 0;
        if acted {
            relief_trace::note("sched", "relieve", want, evicted.bytes);
        }
        let (free, setpoint) = self.kv_region_state(phase).unwrap_or((0, 0));
        // INFO when the pass actually shed something — that is a real event.
        // TRACE otherwise: this runs from several gates every scheduler loop,
        // so a no-op pass at any lower level floods the log under a sustained
        // burst.
        macro_rules! emit {
            ($lvl:ident) => {
                tracing::$lvl!(
                    target: "candle_conversation::scheduler::timing",
                    whence,
                    want_mib = want / (1 << 20),
                    relief_ms = t.elapsed().as_millis() as u64,
                    // Where it went: both packing rungs, the evictions, the wait
                    // for the persistence thread, and the weight side's concession.
                    pack_ms,
                    evict_ms,
                    flush_ms,
                    concede_ms,
                    warm_flushed = flushed,
                    gallery_freed_mib = gallery_freed / (1 << 20),
                    arenas_packed = packed,
                    turns_evicted = evicted.count,
                    evicted_mib = evicted.bytes / (1 << 20),
                    arenas_released = released,
                    conceded_mib = conceded / (1 << 20),
                    free_regions = free,
                    setpoint_regions = setpoint,
                    relieved = !still,
                    "KV region relief"
                )
            };
        }
        if acted {
            emit!(info);
        } else {
            emit!(trace);
        }
        // **Relief that shed something is an admission opportunity too.** Unlike
        // a completion it can happen with nothing finishing at all, and a pass
        // gated only on completions would leave that ground unoffered until the
        // next slot ended — which on a stalled engine is never.
        if acted {
            self.settled_since_admit = true;
        }
        still
    }

    /// The relief ladder driven to `target` bytes of free KV ground rather than
    /// to the pressure setpoint — what a guest drain sheds toward. Returns the
    /// bytes freed.
    ///
    /// **A guest's need is not KV pressure.** [`Self::relieve_vram_pressure`]
    /// answers the KV side's own shortfall and returns at once when there is
    /// none, which on a quiet cache is always — so a drain that asked it for an
    /// image's 9 GB freed nothing, every time, and took all of its ground from
    /// the weight zone: experts to the floor for every draw, and a 1024² draw
    /// refused 240 MiB short beside ground the KV side held for turns nobody
    /// was attending. The rungs are the same, in the same order, and evict only
    /// what they always may — a turn with a warm copy, outside every live
    /// working set, which reloads from RAM when it is next reselected. The
    /// weight side's concession stays the claim's, for what this cannot free.
    pub(in super::super) fn relieve_toward(&mut self, target: u64) -> u64 {
        let free = |s: &Self| s.kv_regions().map_or(0, |r| r.free as u64 * region_bytes());
        let before = free(self);
        if before >= target {
            return 0;
        }
        if let Device::Cuda(d) = self.session.device() {
            end_wave_transient(&d.cuda_stream());
        }
        let _ = self.session.release_empty_arenas();
        self.pack_until_settled(PackGoal::Ground);
        let short = target.saturating_sub(free(self));
        if short > 0 {
            let evicted = self.evict_cold_tail(short);
            if evicted.bytes < short {
                let rest = short - evicted.bytes;
                super::super::timed_wait(|| {
                    self.persist_triggers
                        .flush_at_least(rest, VRAM_OFFLOAD_FLUSH_TIMEOUT)
                });
                self.evict_cold_tail(rest);
            }
            let _ = self.session.release_empty_arenas();
            self.pack_until_settled(PackGoal::Ground);
        }
        // **Dialogue working-set locks, last.** The rungs above protect every
        // dialogue's locked working set (`Substrate::evict_hot_to_free`), which
        // is right between forwards; a drain blocks every forward, so only a
        // slot with a turn on it still needs its own. The rest reload warm→hot
        // at their next projection, a few seconds at the link's rate against a
        // draw that otherwise cannot run — as the weight side's concession goes
        // below its serving floor for the drain alone.
        let short = target.saturating_sub(free(self));
        if short > 0 {
            let evicted = self.evict_idle_working_sets(short);
            if evicted.bytes > 0 {
                let _ = self.session.release_empty_arenas();
                self.pack_until_settled(PackGoal::Ground);
            }
        }
        let freed = free(self).saturating_sub(before);
        tracing::info!(
            target: "candle_conversation::scheduler::timing",
            target_mib = target >> 20,
            before_mib = before >> 20,
            freed_mib = freed >> 20,
            "relief toward a guest's ground"
        );
        freed
    }

    /// Bytes one relief pass should aim to free: enough to reach the setpoint
    /// plus [`RELIEF_OVERSHOOT_REGIONS`]. `None` when there is no pressure, so
    /// a relief call on a healthy cache costs one counter read.
    pub(super) fn relief_shortfall_bytes(&self, phase: VramPhase) -> Option<u64> {
        let stats = self.kv_regions()?;
        let free = stats.free + stats.blocked;
        let target = phase_target(phase);
        if free >= target.setpoint(stats.total) {
            return None;
        }
        let relieved = target.relieved(stats.total);
        Some((relieved.saturating_sub(free) as u64).saturating_mul(region_bytes()))
    }

    /// The union of the working sets of the slots that are not idle — a turn
    /// running, carved or queued on them ([`Self::slot_idle`]). What the
    /// working-set pins hold, and what relief eviction protects
    /// ([`Self::evict_cold_tail`]).
    ///
    /// **An idle slot's working set is not live.** An npcd conversation keeps
    /// its slot bound between turns, and while every bound slot's set counted,
    /// each publish re-pinned and re-stamped it: the idle demote never reached
    /// it and relief never might. On WhiteWorld (2026-10-27) one idle
    /// dialogue's set grew to 16 GB of hot K/V across turns and refused image
    /// draws with nothing running. Left out, it ages like any other unit — the
    /// idle demote's own grace spares the gap between quick turns — and its next
    /// projection lifts back from warm what it selects.
    pub(in super::super) fn live_working_sets(&self) -> (Vec<SectionId>, Vec<TurnKey>) {
        let mut keep_sections: Vec<SectionId> = Vec::new();
        let mut keep_turns: Vec<TurnKey> = Vec::new();
        for (&slot, st) in &self.slot_projection_state {
            if self.slot_idle(slot) {
                continue;
            }
            keep_sections.extend(st.working_set.sections.iter().copied());
            // The knowledge base's resident sections stand ready to enter the
            // range at the next step: hot, like what the slot places.
            keep_sections.extend(st.kb_resident.iter().map(|r| r.id));
            keep_turns.extend(st.working_set.turns.iter().copied());
        }
        (keep_sections, keep_turns)
    }

    /// What relief eviction protects: [`Self::live_working_sets`].
    pub(in super::super) fn relief_keep_list(&self) -> (Vec<SectionId>, Vec<TurnKey>) {
        self.live_working_sets()
    }

    /// Shed to warm the hot turns of every working set but the live ones
    /// ([`Self::live_working_sets`]), dialogue working-set locks included, up
    /// to `target_bytes`. **Only inside a guest's drain**, where nothing
    /// forwards until it ends.
    pub(super) fn evict_idle_working_sets(
        &mut self,
        target_bytes: u64,
    ) -> crate::substrate::EvictionReport {
        let (keep_sections, keep_turns) = self.live_working_sets();
        let mut report = crate::substrate::EvictionReport { count: 0, bytes: 0 };
        let mut remaining = target_bytes;
        for conv in self.distinct_substrates() {
            if remaining == 0 {
                break;
            }
            let r = conv
                .read()
                .evict_hot_for_guest(&keep_sections, &keep_turns, remaining);
            remaining = remaining.saturating_sub(r.bytes);
            report.count += r.count;
            report.bytes += r.bytes;
        }
        if report.count > 0 {
            tracing::info!(
                target: "candle_conversation::guest",
                count = report.count,
                evicted_mib = report.bytes >> 20,
                target_mib = target_bytes >> 20,
                "idle working sets shed to warm for a guest"
            );
        }
        report
    }

    /// The hot bytes across the resident conversations, by why relief
    /// eviction would or would not take them.
    pub(in super::super) fn hot_census(&self) -> crate::substrate::HotCensus {
        let (keep_sections, keep_turns) = self.relief_keep_list();
        let mut census = crate::substrate::HotCensus::default();
        for conv in self.distinct_substrates() {
            census += conv.read().hot_census(&keep_sections, &keep_turns);
        }
        census
    }

    /// Shed least-recently-used hot turn KV to the warm (RAM) tier across the
    /// resident conversations, freeing up to `target_bytes` of pool VRAM.
    /// Oldest-first and reversible (a reselected turn reloads from RAM). Only
    /// turns that already hold a warm copy are evictable, so callers should
    /// first [`PersistenceTriggers::flush_blocking`] to make the just-sealed
    /// turns qualify. The `target_bytes` budget caps total bytes freed, so a
    /// conversation reached via several slots is naturally not over-evicted
    /// (and `evict_hot_to_free` is per-conversation scoped — it can never touch
    /// a parallel conversation's selected working set).
    pub(super) fn evict_cold_tail(
        &mut self,
        target_bytes: u64,
    ) -> crate::substrate::EvictionReport {
        // Explicit protect-list: the union of every live slot's current
        // projection working set (the sealed turns/sections in-flight
        // prefills/decodes are attending over). Relief eviction must not drop the
        // hot copy of an in-scope turn — the block table still references its
        // chunks, so `hot = None` would free NO VRAM and only force a reload when
        // the turn is next reprojected. The reprojection path already protects its
        // incoming selection via the same keep-list; this extends that explicit
        // protection to the relief path. `evict_hot_to_free` resolves keys against
        // each conversation's own substrate, so passing the global union to every
        // conversation only ever protects that conversation's own attended turns
        // (a non-matching key is a no-op) — no per-conversation grouping needed.
        let (keep_sections, keep_turns) = self.relief_keep_list();

        let t = std::time::Instant::now();
        let mut report = crate::substrate::EvictionReport { count: 0, bytes: 0 };
        let mut remaining = target_bytes;
        for conv in self.distinct_substrates() {
            if remaining == 0 {
                break;
            }
            let r = conv
                .read()
                .evict_hot_to_free(&keep_sections, &keep_turns, remaining);
            remaining = remaining.saturating_sub(r.bytes);
            report.count += r.count;
            report.bytes += r.bytes;
        }
        // Feed the GUI's phase timeline here — the single chokepoint every relief
        // path (governor driver, footprint reclaim, compression-starvation
        // recovery) funnels through, so each eviction is counted exactly once
        // regardless of caller.
        self.wave_stats.add_evict(
            report.bytes,
            report.count as u64,
            t.elapsed().as_millis() as u64,
        );
        report
    }
}
