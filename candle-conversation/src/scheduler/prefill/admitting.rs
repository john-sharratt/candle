//! Admits queued prefills and section ingests into the wave, and holds the
//! rule that keeps the engine moving when the ground is short.
//!
//! Queued prefills are offered to the wave in priority order and admitted
//! while the rate model says a wider forward still makes the wave faster. The
//! rate is the decision rather than a byte fit: a wider forward streams the
//! same experts, so a narrow one is slower per row, not cheaper. A pass that
//! admits nothing with nothing in flight takes the queue head regardless, since
//! an engine that admits nothing never completes work and never frees the
//! ground it was refused for. Admission also closes while a guest waits for
//! room, so queued turns do not take the ground that guest is waiting for.

use super::*;

/// How long prefill throughput must be COMPLETELY silent (no forward
/// completing) under surviving VRAM pressure before the promote path halves
/// the admission window. Longer than any healthy forward (the widest
/// calibration forwards run ~7 s), so completions keep the width; a genuine
/// wedge still backs off, one halving per grace period. Device-OOM shrinks at
/// its own site instantly.
pub(super) const PROMOTE_STALL_GRACE: std::time::Duration = std::time::Duration::from_secs(15);

impl Scheduler {
    /// Bytes one 32-token KV block costs across the whole model, in the formats
    /// a **live** sequence actually occupies — the unit every admission cost is
    /// quoted in. See [`per_block_kv_bytes`].
    ///
    /// ACTIVE formats, not the configured sealed ones: a block only reaches
    /// `k_format`/`v_format` once its turn seals and quantizes, and admission is
    /// deciding whether a sequence fits while it is running. Pricing the sealed
    /// pair understated the working set by ~3.7x (192 B/block active vs 52 B
    /// sealed), so admission cleared batches whose real KV was several GiB and
    /// the allocator then refused them one arena at a time. See
    /// [`candle_nn::kv_cache::active_kv_formats`].
    ///
    /// # The seal's second copy is *not* charged here, and that was tried
    ///
    /// Admission is meant to account for persistence's quantize destinations,
    /// and the obvious reading — a block
    /// occupies its active slot *and* its sealed destination while the
    /// compressor copies between them, so charge both — was built and reverted.
    ///
    /// It is the wrong shape. The overlap lasts one copy; the charge lasts the
    /// block's whole life. Applying it here doubles the price of **every** block
    /// in every admission decision, in-flight accounting and decode reserve, so
    /// admission clears roughly half the work it should. On a live rebuild that
    /// showed as `(no forwards)` against a `64MiB` budget with a 14k-token
    /// backlog: nothing admitted, so nothing completed, so nothing freed, so the
    /// budget never recovered.
    ///
    /// A transient double-occupancy is a *reserve* — a fixed pool the compressor
    /// draws on — not a per-block tariff. Such a reserve is not built here.
    pub(in super::super) fn per_block_kv_bytes(&self) -> u64 {
        let (k, v) = self.session.active_kv_formats();
        per_block_kv_bytes(
            self.session.num_layers(),
            self.session.n_kv_head(),
            self.session.head_dim(),
            k,
            v,
        )
    }

    /// Admit queued prefills against what they do to the wave's throughput.
    ///
    /// A burst of small parallel scopes (code_read's worker count), a bulk
    /// collection ingest's per-section prefills, or a batch of calibration cases
    /// all arrive here. What coalesces into one ragged forward is whatever makes
    /// the wave *faster* carrying it: offers are taken in priority order
    /// ([`admit::order`]), priced against the state the eviction pass just
    /// settled ([`admit::cost`]), and admitted while the rate improves
    /// ([`admit::rate`]) without crossing the residency the engine defends
    /// ([`admit::gate`]).
    ///
    /// [`MAX_PREFILL_WIDTH`] is a backstop above this, not the
    /// control. The keep-one-alive rule below is what guarantees progress: a
    /// pass that admits nothing with nothing in flight takes the queue head
    /// regardless, so an oversized lone turn still runs and is bounded by the
    /// per-arena VRAM gate rather than by a refusal it can never clear.
    pub(in super::super) fn promote_new_prefills(&mut self) {
        if self.prefill_queue.is_empty()
            && self.pending_submits.is_empty()
            && self.section_queue.is_empty()
        {
            return;
        }
        // `in_flight` counts only work that can advance — prefills and section
        // ingests alike, see `running_members`.
        let in_flight = self.running_members();
        if in_flight >= MAX_PREFILL_WIDTH {
            return;
        }
        // **A guest waiting for room closes admission**, the queue head's forced
        // admission included. The ground it is short of is held by work in
        // flight, and every prefill admitted now would take what that work gives
        // back as it finishes: on WhiteWorld (2026-10-26) draws were refused
        // 96–928 MiB short four times while the Makers kept starting turns.
        // Deadlock-free because the wait is: a job waits only while something
        // is running (`EngineRoom::busy`), and for at most `ROOM_PATIENCE`, and
        // its drain sets `settled_since_admit` when it ends. One atomic load.
        //
        // **Every band, `Urgent` included.** Letting `Urgent` in unjudged while
        // a guest waited put work past the rate model on a card that was, by
        // construction, short of ground: on WhiteWorld (3907, 2026-10-26 22:05)
        // four content-check prompts went in that way during a wait, a wave
        // came out too wide for the partition and failed mid-sweep, and a
        // decode tripped a device-side assert fifteen seconds later.
        if self.guests.queue.waiting_for_room() {
            return;
        }

        // VRAM-pressure backpressure. Each admitted prefill pins its
        // conversation's KV in VRAM, so under pressure we shed hot KV to the
        // substrate rather than piling on more concurrent prefills; if that
        // doesn't clear it, leave the rest queued this pass.
        //
        // **Not over a prompt that outranks everything in flight.** The pile it
        // guards against is more of the same band beside the work running; a
        // higher band's prompt is not more of it but what that work is set aside
        // for (`pause_creep`), and only once admitted. Held here it waits for
        // the work it outranks to finish. On WhiteWorld (2026-10-27 08:16) an
        // image draw's four one-token Urgent content checks waited 7.5 minutes
        // behind one High dialogue's 155k-token creep: relief ran 80 times and
        // relieved nothing, since the ground was that creep's own working set.
        // It still goes to the fill below, which prices it like any other offer.
        // Each part of the pass is charged to the window's admission split, so
        // the allocation band says which part it went to.
        let t_relief = Instant::now();
        let held = in_flight > 0
            && self.vram_under_pressure()
            && self.relieve_vram_pressure("promote", VramPhase::Load, ReliefNeed::Paced)
            && !self.a_queued_prompt_outranks_the_work_in_flight();
        self.wave_stats
            .admission_split
            .part(AdmitPart::Relief, t_relief.elapsed());
        if held {
            // Pressure survived eviction — stop piling on this pass (the
            // `in_flight > 0` guard keeps ≥1 in flight). The budget halves
            // only on a genuine THROUGHPUT STALL, never on the mere presence
            // of nominal pressure: multiplicative decrease is failure
            // evidence, and a card whose steady state sits just under the
            // pressure band would otherwise pin every bulk-prefill phase at
            // the floor.
            //
            // Stall detection is time-aware because this branch runs many
            // times a second while `PREFILL_OK_TOKENS` advances only when a
            // forward completes (seconds apart for wide forwards): a stall is
            // real only when NO forward has completed for a full
            // [`PROMOTE_STALL_GRACE`]. Each elapsed grace period backs off one
            // halving and re-arms; a device-OOM still cuts instantly at its
            // own site.
            let ok = super::super::PREFILL_OK_TOKENS.load(std::sync::atomic::Ordering::Relaxed);
            if ok > self.promote_ok_tokens_seen {
                self.promote_ok_tokens_seen = ok;
                self.promote_last_progress = Some(std::time::Instant::now());
            }
            let stalled = self
                .promote_last_progress
                .is_some_and(|t| t.elapsed() >= PROMOTE_STALL_GRACE);
            if stalled {
                self.cut_admit_budget(ThrottleReason::ReliefSurvived);
                self.promote_last_progress = Some(std::time::Instant::now());
            }
            return;
        }

        // **The planner is consulted once per settling, not once per pass.**
        //
        // [`admit::fill`] refuses an unsettled ground itself (`Ground::settled`),
        // but the machinery that *reaches* that refusal is not free:
        // [`Self::take_planner`] arms the planner, and `AdmitPass::new` prices the
        // standing tier, reads the region census and computes the headroom and the
        // budget — every bit of it discarded the moment `fill` answers `skipped`.
        // This pass runs from the top of the wave loop AND from every decode step
        // (`Scheduler::mid_wave_admission`), so on an unsettled engine that setup,
        // not the fill, is the whole cost.
        //
        // Nothing new can fit that did not fit before unless something freed
        // ground, so the gate belongs in front of the pricing rather than behind
        // it. What opens the next pass is a sequence finishing — a decode
        // completing, a prefill promoting to decode, a section sealing, a slot
        // terminally freed — plus the two events that can offer ground with
        // nothing finishing at all: a turn arriving at an idle engine, which has
        // never been offered and would otherwise wait for a completion that is
        // never coming, and a relief pass that actually shed.
        if !self.settled_since_admit {
            // Skipping the pass must not skip the deadlock-freedom rule.
            if in_flight == 0 {
                self.force_queue_head("admission is closed until something settles");
            }
            return;
        }

        // **The decision is a rate, not a fit.**
        //
        // Admission used to ask whether an offer's bytes fitted the ground
        // standing free above the weight floor. That question has no answer in
        // tokens a second: a wave of 250 rows and a wave of 2,000 pay the *same*
        // expert copy — a prefill forward needs every expert, resident or
        // streamed — so the narrow one is not cheaper, it is slower per row. A
        // gate that admits by fit stops widening the moment the bytes run out,
        // which on this card measured ~470 tok/s against a modelled 1,210 at the
        // same residency.
        //
        // The bytes have not gone away: they are what the offer *costs*, read by
        // the model as the residency it dislodges, which is what makes a wide
        // wave stop being worth it. They are one term in a throughput comparison
        // now rather than the whole question.
        let Some(mut rate) = self.take_planner() else {
            // No weight side to trade rows against — a dense stack, or an expert
            // cache that has not published its gauges yet (which is every engine
            // before its first classify). There is no rate question to ask, so
            // the queue head goes in and the width backstop is the only bound.
            // Without this an engine would never take the first prefill that
            // makes the cache describe itself.
            //
            // **And it says so, because this branch admits exactly one.** It used to
            // return in silence, which makes it indistinguishable from a healthy engine
            // in every log the scheduler emits: no admission pass line, no refusal, just
            // a wave one row wide, forever. Measured on the 30B-A3B — twenty conversations
            // queued, `decode seqs avg=1.0 max=1`, 32 t/s against a batched ceiling of
            // 518 — and the only way to find it was to notice that the pass which should
            // have logged never did. A path that quietly costs an order of magnitude is
            // the one path that must not be quiet.
            if let Some(work) = self.pop_highest_band_head() {
                tracing::debug!(
                    target: "candle_conversation::scheduler::throttle",
                    queued = self.prefill_queue.len(),
                    in_flight,
                    "no weight plan: admitting ONE prefill unjudged. The rate model cannot \
                     run without the expert cache's gauges, so the wave stays as narrow as \
                     this path makes it",
                );
                self.begin_forced(work);
            }
            // Sections have no rate question either, and nothing to wait on: the
            // width backstop is their only bound here, as it is a prefill's.
            while self.running_members() < MAX_PREFILL_WIDTH {
                let Some(s) = self.pop_highest_band_section() else {
                    break;
                };
                if !self.begin_section_forced(s) {
                    break;
                }
            }
            return;
        };
        let t_setup = Instant::now();
        let mut pass = AdmitPass::new(self);
        let setup = t_setup.elapsed();
        let t_fill = Instant::now();
        let filled = admit::fill(&mut pass, &mut rate);
        let fill = t_fill.elapsed();
        let t_conclude = Instant::now();
        let unoffered = pass.conclude();
        let conclude = t_conclude.elapsed();
        self.wave_rate = Some(rate);
        let split = &mut self.wave_stats.admission_split;
        split.pass();
        split.part(AdmitPart::Setup, setup);
        split.part(AdmitPart::Fill, fill);
        split.part(AdmitPart::Conclude, conclude);
        // **Keep one prefill in flight, whatever the model says.**
        //
        // The engine's deadlock-freedom rule, and the one guarantee no
        // throughput judgement may override: an engine that admits nothing makes
        // no progress, and nothing completes, so nothing frees the ground the
        // refusal was about. The wave then refuses the same turn on the same
        // grounds forever.
        //
        // `fill` has its own version of this — it carries its first offer
        // unconditionally — but that waiver is keyed on the *wave* being empty,
        // and a wave is not empty while a decode is still stepping. So a turn
        // arriving behind a decode is judged on its merits, which is right, and
        // if it is refused while nothing is prefilling there is nothing left to
        // create the conditions for it to be admitted later.
        //
        // This is the rule the byte-fit planner carried as `MIN_PREFILL_WIDTH`,
        // restored after its absence wedged a live daemon: two turns queued,
        // nothing in flight, `stopped_on_weights` on every pass, and the loop
        // spinning in relief that could not help because the ground it wanted
        // was not the ground being refused.
        //
        // FIFO, not the cheapest that fits: under a budget stuck at its floor,
        // cheapest-first starves the expensive work permanently, and the
        // expensive work is never the cheapest.
        if filled.prefills == 0 && filled.sections == 0 && self.running_members() == 0 {
            self.force_queue_head("admission refused every offer");
        }
        // **A pass held on the floor over holes is held on ground that exists.**
        // Residency is measured to the arena frontier, so holes stranded below it
        // count for nothing until the frontier comes down into them — and relief,
        // which measures free regions, never sees the shortage. On WhiteWorld
        // (673f67a2a, heavy-read) residency read under the prefill floor in 329
        // of 333 windows while 181 regions (about 3 GB) of holes stood beneath the
        // frontier, and read turns waited six minutes between tool rounds. The
        // frontier holder's own pass is what turns those holes into ground.
        if filled.stopped_on_weights {
            let holes = super::super::interleave::stranded_hole_bytes() as usize;
            if holes
                >= super::super::compaction_stall::MIN_FREEABLE_ARENAS
                    * candle_nn::kv_cache::REGION_BYTES
            {
                let t_frontier = Instant::now();
                self.lower_frontier_for(holes);
                self.wave_stats
                    .admission_split
                    .part(AdmitPart::Frontier, t_frontier.elapsed());
            }
        }
        // The pass acted on the completions that opened it; the next one waits
        // for its own.
        self.settled_since_admit = false;
        if filled.prefills > 0
            || filled.sections > 0
            || filled.stopped_on_weights
            || filled.stopped_on_rate
            || !self.prefill_queue.is_empty()
            || !self.pending_submits.is_empty()
            || !self.section_queue.is_empty()
        {
            tracing::debug!(
                target: "candle_conversation::scheduler::throttle",
                admitted = filled.prefills,
                sections = filled.sections,
                in_flight,
                queued = self.prefill_queue.len(),
                // Submitted turns whose projection waits on admission — offered
                // alongside the queue, and most of what waits in a tool loop.
                pending = self.pending_submits.len(),
                queued_sections = self.section_queue.len(),
                // **The one field that distinguishes "refused" from "never
                // asked".** A pass that skips admits nothing and refuses
                // nothing, which reads identically to a pass that judged every
                // offer and turned it down — and the two call for opposite
                // fixes.
                skipped = filled.skipped,
                // Which of the two ends a pass stopped at is the signal worth
                // having: weights means the device is the bound and the producer
                // should hear about it, rate means the engine is working well and
                // the queue simply rides the next wave.
                stopped_on_weights = filled.stopped_on_weights,
                stopped_on_rate = filled.stopped_on_rate,
                // And the two ends that are not judgements: the forward's rows
                // already full with the cohort in flight, or the width
                // backstop. Work waiting behind either was never offered, so
                // without these a full forward read as admission refusing.
                stopped_on_rows = unoffered.rows,
                stopped_on_width = unoffered.width,
                "admission pass"
            );
        }
    }

    /// Admit the queue head whatever the rate model would have said — the
    /// engine's deadlock-freedom rule, and the one guarantee no throughput
    /// judgement may override.
    ///
    /// An engine that admits nothing makes no progress, so nothing completes, so
    /// nothing frees the ground the refusal was about, and the same turn is
    /// refused on the same grounds forever. This is the rule the byte-fit planner
    /// carried as `MIN_PREFILL_WIDTH`, restored after its absence wedged a live
    /// daemon: two turns queued, nothing in flight, `stopped_on_weights` on every
    /// pass, and the loop spinning in relief that could not help because the
    /// ground it wanted was not the ground being refused.
    ///
    /// FIFO, not the cheapest that fits: under a budget stuck at its floor,
    /// cheapest-first starves the expensive work permanently, and the expensive
    /// work is never the cheapest.
    ///
    /// Callers apply it only with no prefill that can advance
    /// ([`Self::running_prefills`] zero). `why` names the path
    /// that forced it, because a head admitted this way was never judged and a
    /// reader has to be able to tell that from an admission that was.
    ///
    /// The head is the first entry of the **highest band** queued: a dialogue
    /// turn queued behind ingest is the one forced in.
    pub(super) fn force_queue_head(&mut self, why: &'static str) {
        if let Some(work) = self.pop_highest_band_head() {
            tracing::debug!(
                target: "candle_conversation::scheduler::throttle",
                queued = self.prefill_queue.len() + 1,
                why,
                "nothing in flight; forcing the queue head so the engine makes progress",
            );
            self.begin_forced(work);
        } else if let Some(s) = self.pop_highest_band_section() {
            tracing::debug!(
                target: "candle_conversation::scheduler::throttle",
                queued_sections = self.section_queue.len() + 1,
                why,
                "nothing in flight; forcing the section queue's head so the engine makes progress",
            );
            self.begin_section_forced(s);
        }
    }

    /// Remove and return the oldest queued section of the highest band queued.
    pub(super) fn pop_highest_band_section(&mut self) -> Option<QueuedSection> {
        let top = self.section_queue.iter().map(|s| rank(s.band)).max()?;
        let at = self
            .section_queue
            .iter()
            .position(|s| rank(s.band) == top)?;
        self.section_queue.remove(at)
    }

    /// Remove and return the first queued prefill of the highest band queued,
    /// in queue order.
    ///
    /// A pending submission counts as queued: forced, its projection is put on
    /// the device first ([`Self::materialise_submit`]). Within a band, carved
    /// work goes ahead of submissions, which arrived after it.
    pub(super) fn pop_highest_band_head(&mut self) -> Option<PrefillWork> {
        loop {
            let queued = self
                .prefill_queue
                .iter()
                .map(|w| rank(self.decode_priority_or_high(w.sequence_id)))
                .max();
            let pending = self
                .pending_submits
                .iter()
                .map(|p| rank(self.decode_priority_or_high(p.parent())))
                .max();
            match (queued, pending) {
                (Some(q), p) if p.is_none_or(|p| q >= p) => {
                    let at = (0..self.prefill_queue.len()).find(|&i| {
                        rank(self.decode_priority_or_high(self.prefill_queue[i].sequence_id)) == q
                    })?;
                    return self.prefill_queue.remove(at);
                }
                (_, Some(p)) => {
                    let at = (0..self.pending_submits.len()).find(|&i| {
                        rank(self.decode_priority_or_high(self.pending_submits[i].parent())) == p
                    })?;
                    let pending = self.pending_submits.remove(at)?;
                    // A submission that could not be carved has told its caller;
                    // the next head is offered instead.
                    if let Some(work) = self.materialise_submit(pending) {
                        return Some(work);
                    }
                }
                _ => return None,
            }
        }
    }

    /// The planner, built on first use and taken for the duration of a pass.
    ///
    /// Taken rather than borrowed because the pass borrows the whole scheduler:
    /// [`admit::fill`] needs `&mut` on both the ground and the planner, and they
    /// live in the same struct. It goes back the moment the pass ends.
    ///
    /// `None` while the model cannot describe its weight side — see
    /// [`ManagedBatchedModel::weight_plan`].
    pub(in super::super) fn take_planner(&mut self) -> Option<admit::WaveRate> {
        if self.wave_rate.is_none() {
            // **The refusal names itself.** `WeightPlan::from_stats` declines a partial
            // gauge set outright — any of `moe_layers`, `total_experts`,
            // `expert_slot_bytes` or `zone_max_bytes` reading zero — and a bare `?` here
            // turns that into an absent planner with no record of which gauge was missing.
            // Since an absent planner collapses every wave to one row (see the caller),
            // the distinction between "no expert cache at all" and "the cache has not
            // described itself yet" is the difference between a dense model working as
            // designed and a routed model silently running an order of magnitude slow.
            // **A routed model with broken gauges is fatal, and a dense one is not.**
            //
            // These were one `None` until the conflation was measured: an absent plan
            // collapses admission to one prefill per pass (see the caller), which is
            // correct for a dense stack and an order of magnitude for a routed one. On the
            // 30B-A3B at 72 GiB it read `decode seqs avg=1.0 max=1` and 32 t/s against a
            // batched ceiling of 518, logged nothing, and inverted with card size — a card
            // too small to hold the checkpoint streamed experts, published gauges and
            // batched, while a card large enough held everything, published nothing, and
            // ran one row wide.
            //
            // So the broken case panics. It is not recoverable by degrading: the engine
            // would serve, slowly, with no signal distinguishable from a healthy narrow
            // workload, and the only way it was ever found was noticing that a log line
            // which should have appeared never did.
            let p = match self.model.weight_plan() {
                candle_transformers::models::expert_lre::WeightPlanning::Ready(p) => p,
                candle_transformers::models::expert_lre::WeightPlanning::Dense => {
                    tracing::debug!(
                        target: "candle_conversation::scheduler::throttle",
                        "dense stack: no expert residency to trade rows against, so the \
                         width backstop is the only bound",
                    );
                    return None;
                }
                candle_transformers::models::expert_lre::WeightPlanning::Incomplete { field } => {
                    let s = self.model.expert_stats();
                    panic!(
                        "the expert cache reports a routed model but its gauge set is \
                         incomplete: `{field}` is zero. The wave rate planner cannot be \
                         armed without it, and an unarmed planner silently narrows every \
                         wave to one row — so this refuses to run rather than serve at a \
                         fraction of the rate with no way to tell. Gauges: \
                         moe_layers={:?} total_experts={:?} slot_bytes={:?} \
                         zone_bytes={:?} zone_max_bytes={:?}",
                        s.as_ref().map(|s| s.moe_layers),
                        s.as_ref().map(|s| s.total_experts),
                        s.as_ref().map(|s| s.expert_slot_bytes),
                        s.as_ref().map(|s| s.zone_bytes),
                        s.as_ref().map(|s| s.zone_max_bytes),
                    );
                }
            };
            let geometry = admit::rate::ExpertGeometry {
                moe_layers: p.moe_layers,
                experts_per_layer: p.experts_per_layer,
                slot_bytes: p.slot_bytes,
            };
            // Measured on the device the experts actually stream over, so the
            // 4090 Mobile, the 3090 behind PCIe 3.0 and the Blackwell each seed
            // from their own bus rather than from a datasheet.
            let rate = match admit::WaveRate::measure(
                &self.device,
                geometry,
                admit::rate::RateModel::default(),
                admit::rate::DecodeModel::default(),
            ) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("wave rate planner: link probe failed: {e}");
                    return None;
                }
            };
            // What a previous run learned of this card, when it ran this model:
            // the figures it publishes are then priced from the start, for the
            // state this member is in, rather than seeded or kept stale.
            let mut rate = rate;
            let restored = super::super::phase_ring::kept_learned_model()
                .is_some_and(|text| rate.import_learned(&text));
            tracing::info!(
                target: "candle_conversation::scheduler::admission",
                moe_layers = p.moe_layers,
                experts_per_layer = p.experts_per_layer,
                slot_mib = p.slot_bytes >> 20,
                link_gbps = rate.link_bytes_per_s() / 1e9,
                restored,
                "wave rate planner armed",
            );
            self.wave_rate = Some(rate);
        }
        self.wave_rate.take()
    }

    /// Start a turn admission has taken: carve a queued turn-group case its view
    /// ([`Self::carve_group_case`]), fork a submitted turn's view its parent's
    /// state ([`Self::fork_view_state`]), claim the sequence's carried state
    /// (`claim_carried_state`), then put it in flight. `Err` hands the work back
    /// unstarted with why it could not be — the caller decides whether it waits
    /// or fails.
    ///
    /// **Everything a turn claims up front is claimed here**, between forwards,
    /// where the weight side can sell for it and a refusal costs the turn a wait.
    /// Claimed at the turn's first wave instead, the recurrent store would come
    /// after that wave's own K/V and after the weight side had just taken spare
    /// ground back, so a refusal would fail every member of the cohort — on
    /// Qwen3.8-Flash-Next's calibration, `no region for recurrent state after 34
    /// of 72 slots` with admission having priced the store a moment before.
    pub(in super::super) fn begin_admitted(
        &mut self,
        mut work: PrefillWork,
    ) -> Result<(), Box<(PrefillWork, ConversationError)>> {
        // Each step is charged to the window's admission split, so the
        // allocation band says which part of a begin it went to.
        let t = Instant::now();
        if let Some(case) = self.uncarved_case(&work) {
            if let Err(e) = self.carve_group_case(&mut work, case) {
                return Err(Box::new((work, e)));
            }
        }
        self.wave_stats
            .admission_split
            .part(AdmitPart::BeginCarve, t.elapsed());
        let t = Instant::now();
        let view = work.sequence_id;
        if let Some(parent) = self
            .turn_views
            .get(&view)
            .filter(|v| !v.state_forked)
            .map(|v| v.parent_id)
        {
            if let Err(e) = self.fork_view_state(parent, view) {
                return Err(Box::new((work, e)));
            }
            if let Some(v) = self.turn_views.get_mut(&view) {
                v.state_forked = true;
            }
        }
        self.wave_stats
            .admission_split
            .part(AdmitPart::BeginFork, t.elapsed());
        let t = Instant::now();
        let waited = quiesce_wait_ns();
        let split = ClaimSplit::now();
        let claimed = self.model.claim_carried_state(work.sequence_id.0);
        self.wave_stats
            .admission_split
            .part(AdmitPart::BeginClaim, t.elapsed());
        let [parked, install, build, recurrent, ple, seeds] = ClaimSplit::since(split);
        for (part, took) in [
            (AdmitPart::BeginClaimParked, parked),
            (AdmitPart::BeginClaimInstall, install),
            (AdmitPart::BeginClaimBuild, build),
            (AdmitPart::BeginClaimRecurrent, recurrent),
            (AdmitPart::BeginClaimPle, ple),
            (AdmitPart::BeginClaimSeeds, seeds),
        ] {
            self.wave_stats.admission_split.part(part, took);
        }
        self.wave_stats.admission_split.part(
            AdmitPart::BeginClaimQuiesce,
            Duration::from_nanos(quiesce_wait_ns() - waited),
        );
        if let Err(e) = claimed {
            return Err(Box::new((work, ConversationError::Model(e))));
        }
        let t = Instant::now();
        self.seal_turn_under_its_conversation(work.sequence_id);
        self.wave_stats
            .admission_split
            .part(AdmitPart::BeginSeal, t.elapsed());
        let t = Instant::now();
        self.begin_prefill(work);
        self.wave_stats
            .admission_split
            .part(AdmitPart::BeginStart, t.elapsed());
        Ok(())
    }

    /// Put a turn that cannot start into flight already failed, so it leaves
    /// the way any failed prefill does — its caller answered, its view and its
    /// group case settled — rather than being dropped unanswered.
    pub(super) fn begin_failed(&mut self, work: PrefillWork, why: ConversationError) {
        self.active_prefills.push(ActivePrefill {
            work,
            offset: 0,
            next_projection: 0,
            finished: None,
            error: Some(why),
            prefill_start: None,
            stops_every: None,
            awaiting: false,
        });
    }

    /// Start a turn taken outside the rate judgement — the keep-one-alive head,
    /// or the one a stack with no planner admits.
    ///
    /// A turn that cannot be started — its view not carved, its state not
    /// claimed — goes back to the head of the queue while something else is in
    /// flight to free ground for it. With nothing in flight nothing will: relief
    /// gets one attempt, and a turn still refused fails rather than being
    /// retried forever — a queued group case lost, any other turn answered with
    /// the refusal.
    pub(super) fn begin_forced(&mut self, work: PrefillWork) {
        let Err(refused) = self.begin_admitted(work) else {
            return;
        };
        let (work, _) = *refused;
        if self.running_members() > 0 {
            self.prefill_queue.push_front(work);
            return;
        }
        self.relieve_vram_pressure("claim", VramPhase::Load, ReliefNeed::Paced);
        let Err(refused) = self.begin_admitted(work) else {
            return;
        };
        let (work, why) = *refused;
        match self.uncarved_case(&work) {
            Some(case) => self.lose_queued_case(case, &why),
            None => self.begin_failed(work, why),
        }
    }

    /// Start a section taken outside the rate judgement, by
    /// [`Self::begin_forced`]'s rule: one whose state cannot be claimed goes
    /// back to the head of its queue while something is in flight, and with
    /// nothing in flight gets one relief attempt before it is answered with the
    /// refusal. `false` when it went back to the queue.
    pub(in super::super) fn begin_section_forced(&mut self, s: QueuedSection) -> bool {
        let Err(refused) = self.begin_section(s) else {
            return true;
        };
        let (s, _) = *refused;
        if self.running_members() > 0 {
            self.section_queue.push_front(s);
            return false;
        }
        self.relieve_vram_pressure("claim", VramPhase::Load, ReliefNeed::Paced);
        let Err(refused) = self.begin_section(s) else {
            return true;
        };
        let (s, why) = *refused;
        let _ = s.response_tx.send(Err(why));
        true
    }

    /// Put one admitted turn into flight.
    ///
    /// The tail of admission, and the whole of what admitting a prefill *does*:
    /// announce it to its caller and move it onto `active_prefills`. Separate
    /// from the pass that chooses it so the choosing can be replaced without
    /// touching what the choice means.
    pub(in super::super) fn begin_prefill(&mut self, work: PrefillWork) {
        let total = work.tokens.len();
        if total > 0 {
            self.prompt_mean.add(total);
        }
        let _ = work
            .event_tx
            .send(TurnEvent::Prefill(work.prefill_text.clone()));
        let _ = work.event_tx.send(TurnEvent::PrefillProgress {
            tokens_done: 0,
            tokens_total: total,
        });
        let error = if total == 0 {
            Some(ConversationError::Channel(
                "prefill received zero tokens".into(),
            ))
        } else {
            None
        };
        // No index cut here. Admission is not a unit boundary — it is the moment
        // work leaves the queue, which happens once per unit but says nothing
        // about where that unit's tokens start. The boundary was taken with the
        // unit's K/V anchor (`Scheduler::close_unit_boundary`), and a second cut
        // on this slot would close whatever the unit has already forwarded into
        // a page of its own.
        //
        // An opening on a filled slot stops every `reserve_tokens` for a
        // boundary (§7.4) — but not a calibration's staged or captured prefill,
        // whose projection points and recurrent arms are its own.
        // A request's first reading is read whole and stops only at its end
        // (`first_reading`); its second reading takes the strides.
        let stops_every = work
            .reprojection
            .as_ref()
            .filter(|_| work.projection_offsets.is_empty() && work.recurrent_spans.is_empty())
            .filter(|_| !work.rereads)
            .and_then(|p| p.opening_chunk())
            .filter(|&every| every < total);
        self.active_prefills.push(ActivePrefill {
            work,
            offset: 0,
            next_projection: 0,
            finished: None,
            error,
            prefill_start: None,
            stops_every,
            awaiting: false,
        });
    }

    /// Number of in-flight prefills that still have tokens left to process
    /// and have not errored.
    pub(in super::super) fn prefill_width(&self) -> usize {
        self.active_prefills
            .iter()
            .filter(|p| p.error.is_none() && p.offset < p.work.tokens.len())
            .count()
    }

    /// Whether a prompt a person is waiting on is prefilling, or queued to.
    /// A dialogue's tool results are that prompt between its decodes, so the
    /// `Low` hold beside its decode covers them too (`urgent_hold`).
    pub(in super::super) fn an_urgent_prompt_prefills(&self) -> bool {
        let urgent = |id: SequenceId| self.decode_priority_or_high(id) == DecodePriority::Urgent;
        self.active_prefills
            .iter()
            .filter(|p| p.error.is_none() && p.offset < p.work.tokens.len())
            .any(|p| urgent(p.work.sequence_id))
            || self.prefill_queue.iter().any(|w| urgent(w.sequence_id))
    }

    /// In-flight prefills that can advance: not errored. What admission's
    /// progress gates count.
    pub(in super::super) fn running_prefills(&self) -> usize {
        self.active_prefills
            .iter()
            .filter(|p| p.error.is_none())
            .count()
    }

    /// Admitted work that can still advance a row: running prefills and section
    /// ingests together — what [`MAX_PREFILL_WIDTH`] bounds and what the
    /// keep-one-alive rule asks about, since both are prefill rows in one wave.
    pub(in super::super) fn running_members(&self) -> usize {
        self.running_prefills() + self.section_ingest_width()
    }

    /// Number of in-flight section ingests with tokens remaining (not errored).
    pub(in super::super) fn section_ingest_width(&self) -> usize {
        self.active_section_ingests
            .iter()
            .filter(|s| s.error.is_none() && s.offset < s.tokens.len())
            .count()
    }
}
