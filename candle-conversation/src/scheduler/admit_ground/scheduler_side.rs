//! The scheduler's measured readings that admission works from: headroom, the
//! floors, the wave already in flight and the learning inputs for the planner.
//!
//! Headroom is measured rather than forecast, and the free list is not added to
//! it, because that would let admission spend the same region twice. The
//! spendable ground is how far residency stands above its floor, less a useful
//! forward's tier, so that tier always has room to be placed. Growth the running
//! decodes will claim, and the tier of a wave already in flight, are committed
//! ground that every residency reading is net of. The planner learns from a
//! forward only when it was wide enough to teach the copy rate, and from a
//! decode forward only when it was compute-bound.

use super::*;

impl Scheduler {
    /// Ground the KV side can claim without crossing into the weight zone,
    /// together with the zone's own position and range.
    ///
    /// Region-granular and **measured, not forecast**: the relief pass ahead of
    /// admission has already run, so a free region is one this process has
    /// claimed and not yet spent rather than one it hopes to recover.
    pub(in super::super) fn admit_headroom(&self, standing_tier: usize) -> Headroom {
        let zone = residency_now(standing_tier);
        // The next forward's tier, beyond what the standing ground already holds
        // (`interleave::reserve_for` nets that out of `zone`).
        let tier_reserve = self
            .min_forward_tier_bytes()
            .saturating_sub(standing_tier as u64);
        let room = Headroom {
            // **Deliberately zero here, and netted below.** The free list is not
            // headroom *beside* the zone, it is the same ground seen twice:
            // `effective = extent + non-live regions - reserve`, so claiming a
            // free region moves it from non-live to live and drops the zone by
            // exactly its size. Any formula that adds the two lets admission
            // spend one region twice — measured as 95 slots standing open on
            // 1.6 GiB that did not exist.
            free_kv: 0,
            zone,
            zone_min: interleave::optimal_weight_bytes().unwrap_or(0),
            zone_max: interleave::achievable_weight_now().unwrap_or(u64::MAX),
            zone_min_prefill: self.expert_zone_min_bytes(),
            forward_reserve: tier_reserve,
        };
        Headroom {
            // **The spendable ground is one subtraction: how far residency
            // stands above its floor**, less a useful forward's tier.
            //
            // The tier is the span's third tenant and takes whatever K/V leaves
            // at the frontier, so admission must stop short of the weight floor
            // by that much. Reserve nothing and K/V claims all the way down: the
            // tier reaches zero, no forward can be planned at all, nothing
            // completes, and no ground comes back for it to recover with.
            //
            // Self-correcting, which is the property worth having: as slots
            // finish and their regions return, the zone rises and this room
            // reopens by the same amount.
            free_kv: zone.saturating_sub(room.floor().saturating_add(tier_reserve)),
            ..room
        }
    }

    /// The smallest weight zone the expert cache can serve a forward from —
    /// its pinned set, a worst-case working layer and one slot for the load
    /// that triggered an eviction (`minimum_resident_slots`), as the cache
    /// publishes it. Zero for a dense stack: no expert residency to defend.
    ///
    /// **The prefill floor, derived rather than set.** It was a quarter of the
    /// achievable residency plus a fixed 512 MiB margin — a fraction tuned on
    /// one ingest, the same on every card and every load, and on the 3090 with
    /// two dozen conversations' state resident a line the zone could not stand
    /// above. What a prefill must never cross is the point where the cache can
    /// no longer serve, and the cache knows that point exactly: under it every
    /// slot can be pinned and no promotion lands. Above it, spending weight for
    /// rows is the trade the rate model judges.
    pub(in super::super) fn expert_zone_min_bytes(&self) -> u64 {
        self.model
            .expert_stats_shared()
            .and_then(|s| s.lock().ok().map(|s| s.zone_min_bytes as u64))
            .unwrap_or(0)
    }

    /// Tier bytes a forward worth running needs, held back from admission.
    ///
    /// **A useful forward's worth, not merely a placeable one.** Reserve nothing
    /// and the tier reaches zero — no forward at all, not a narrow one. Reserve
    /// one row and a forward can be placed but not filled: every wave carried
    /// one sequence while a third of the regions stood free. So it is the least
    /// advance the engine will hand a sequence, priced through the same planner
    /// that places the tier, which is what makes it follow the model's geometry
    /// instead of being a byte count to re-derive per card.
    pub(in super::super) fn min_forward_tier_bytes(&self) -> u64 {
        // The least advance is a few chunks, far below the rows a launch
        // pre-stages for, so it stages nothing.
        self.model
            .wave_tier_bytes(
                PREFILL_MIN_ADVANCE,
                1,
                &[],
                &[],
                &[],
                &self.decode_stage(),
                self.session.activation_dtype(),
            )
            .unwrap_or(0)
    }

    /// Whether spending `bytes` more of the weight side's residency leaves it at
    /// or above the decode floor admission defends — the same line
    /// [`Self::admit_budget_terms`] gives the rate model, read now. What a
    /// mid-decode rebuild is judged by: it adds no rows, so the only question is
    /// whether the decodes it would slow can still be carried.
    pub(in super::super) fn decode_floor_holds_after(&self, bytes: u64) -> bool {
        let standing = self.standing_ground_bytes();
        let room = self.admit_headroom(standing);
        let floor = room.floor().saturating_add(room.forward_reserve);
        residency_now(standing).saturating_sub(bytes) >= floor
    }

    /// The tier the wave already in flight reserves — the held creep group,
    /// its members' pre-stage at each one's own depth included.
    ///
    /// A fact this decision reads, not an output it feeds back into itself: the
    /// tier each *new* admission adds is charged separately, through
    /// `Cost::dislodged_bytes`, so between them the wave's tier is counted
    /// exactly once.
    pub(in super::super) fn standing_tier_bytes(&self) -> usize {
        self.composed_tier(&self.compose(&self.standing_wave())) as usize
    }

    /// Ground the standing wave holds against the weight side: its tier, and
    /// the K/V its running decodes will claim before a sweep's windows are
    /// done ([`super::sweep_forecast`]), in whole regions as the claims take
    /// it. What every residency reading is net of.
    ///
    /// **The decodes' growth is committed ground, not headroom.** A sweep's
    /// tier is placed per window on the ground left then, and the decodes
    /// beside it keep claiming as they generate; admission that spends to the
    /// floor's edge leaves the later windows that much short. Measured on the
    /// 3090 scratch daemon (2026-10-18): floor-sized offers took residency to
    /// within 27 MiB of the prefill floor, and the sweep's last window found
    /// its 5.0 GB tier two regions short — ten decodes' claims over its 30 s.
    /// The group budget already withholds this forecast; admission now reads
    /// the same one.
    ///
    /// A decode's stencil adds nothing here: its static runs ride the steps a
    /// piece at a time, each within the model's short span, so none claims its
    /// sequence's prefill scratch (`stencil_wave`).
    pub(in super::super) fn standing_ground_bytes(&self) -> usize {
        self.standing_tier_bytes()
            .saturating_add(self.decode_growth_bytes())
            .saturating_add(self.held_claims_outstanding())
    }

    /// What the held cohort will still claim as it sweeps: all of its own
    /// claims before its first window — the residual, its stores and scratch
    /// are claimed there — and after it the K/V its remaining layers write.
    ///
    /// Its fit priced them against the ground at formation; a turn admitted
    /// mid-sweep claims from that same ground at once. Unreserved, one admitted
    /// above the floor took a 144 MiB scratch under an 8,192-row cohort fitted
    /// with 57 MiB to spare, and its last window was refused four regions short
    /// (WhiteWorld, c3ebac8bf, 2026-10-19).
    pub(in super::super) fn held_claims_outstanding(&self) -> usize {
        let Some((own, kv)) = self.held_fit_claims else {
            return 0;
        };
        if self.wave_prefill_members.is_empty() {
            return 0;
        }
        let layers = self.session.num_layers().max(1);
        let cursor = self.wave_prefill_cursor.min(layers);
        if cursor == 0 {
            own as usize
        } else {
            (kv as usize).saturating_mul(layers - cursor) / layers
        }
    }

    /// The K/V the running decodes will claim over one sweep, in whole regions
    /// — see [`Self::standing_ground_bytes`].
    pub(in super::super) fn decode_growth_bytes(&self) -> usize {
        let growth = self
            .model
            .kv_bytes_per_row(self.session.activation_dtype())
            .map_or(0, |row| self.sweep_forecast.withheld_bytes(row));
        let region = candle_nn::kv_cache::REGION_BYTES;
        growth.div_ceil(region).saturating_mul(region)
    }

    /// The cohort the next forwards carry as a [`WaveDraft`]: the held creep
    /// group at each member's advance and depth, then **every prefill and
    /// section admitted that has not joined it yet**.
    ///
    /// Those are the next group's members as surely as the held ones — the
    /// group is formed from every prefill in flight — and an earlier pass
    /// charged their tier against the draft it composed. Starting the next pass
    /// from the held group alone forgets that charge: measured on the 3090
    /// scratch daemon (2026-10-18), five passes in two seconds each read
    /// ~9.3 GiB, admitted three turns down to ~4.6 GiB and stopped — sixteen
    /// turns whose cohort then formed at 8,192 rows on a 6.9 GB tier, and its
    /// sweep failed two to four regions short with every member lost.
    ///
    /// Each takes rows from what the ceiling has left, staged at its deepest
    /// chunk, exactly as its admission's offer was priced; past the ceiling
    /// it joins, adding a sequence and no rows.
    pub(in super::super) fn standing_wave(&self) -> WaveDraft {
        let mut wave = WaveDraft::default();
        let mut held = Vec::with_capacity(self.wave_prefill_members.len());
        for m in &self.wave_prefill_members {
            let (WaveMember::Prefill { seq_id, advance } | WaveMember::Section { seq_id, advance }) =
                m;
            wave.held_rows += *advance;
            wave.held_advances.push(*advance);
            wave.held_depths
                .push(self.session.sequence_offset(*seq_id).unwrap_or(0));
            wave.held_seqs.push(*seq_id);
            held.push(*seq_id);
        }
        let prefills = self
            .active_prefills
            .iter()
            .filter(|p| p.error.is_none() && p.finished.is_none())
            .map(|p| {
                (
                    p.work.sequence_id,
                    p.work.tokens.len().saturating_sub(p.offset),
                )
            });
        let sections = self
            .active_section_ingests
            .iter()
            .filter(|s| s.error.is_none())
            .map(|s| (s.sequence_id, s.tokens.len().saturating_sub(s.offset)));
        for (seq, remaining) in prefills.chain(sections) {
            if remaining > 0 && !held.contains(&seq.0) {
                let depth = self.session.sequence_offset(seq.0).unwrap_or(0);
                wave.joining.push((remaining, depth, seq.0));
            }
        }
        wave.width = self.rate_width.map(|w| w.rows);
        wave
    }

    /// `wave` composed the way the group forms: the held group at its fixed
    /// advances, then each joining member taking rows from what the ceiling
    /// has left, staged at its deepest chunk; past the ceiling it rides as a
    /// sequence with no rows.
    ///
    /// **The ceiling is the width the group will be composed at** — the group
    /// budget at the draft's width, the rule `prefill_group_budget` forms the
    /// group by — not the pass ceiling an offer is sized against. The group
    /// budget absorbs a total that passes the model's cap by no more than its
    /// slack, so a cohort can form wider than any ceiling an offer saw:
    /// measured on the 3090 scratch daemon (2026-10-18), a sweep at 8,066 rows
    /// whose last window's 7.26 GB tier was nine regions short. The stretch
    /// fires exactly when the model's cap binds, which is when ground is short.
    /// And a narrower width the pass closes at binds the other way: a member
    /// clipped to it is priced at the rows it will actually run.
    pub(in super::super) fn compose(&self, wave: &WaveDraft) -> Composed {
        let total = wave
            .held_rows
            .saturating_add(wave.joining.iter().map(|&(r, _, _)| r).sum::<usize>());
        let ceiling = self.prefill_group_budget_at(total, self.composition_target_at(wave.width));
        let mut out = Composed {
            rows: wave.held_rows,
            seqs: wave.held_advances.len() + wave.joining.len(),
            advances: wave.held_advances.clone(),
            depths: wave.held_depths.clone(),
            stage_seqs: wave.held_seqs.clone(),
        };
        for &(remaining, depth, seq) in &wave.joining {
            let rows = remaining.min(ceiling.saturating_sub(out.rows));
            if rows > 0 {
                out.advances.push(rows);
                out.stage_seqs.push(seq);
                out.depths
                    .push(deepest_chunk_offset(depth, remaining, rows));
                out.rows += rows;
            }
        }
        out
    }

    /// The tier a forward composed as `wave` places — the same planner the
    /// placement reads, with the running decodes the step carries beside it.
    pub(in super::super) fn composed_tier(&self, wave: &Composed) -> u64 {
        self.model
            .wave_tier_bytes(
                wave.rows,
                wave.seqs.max(1),
                &wave.advances,
                &wave.depths,
                &wave.stage_seqs,
                &self.decode_stage(),
                self.session.activation_dtype(),
            )
            .unwrap_or(0)
    }

    /// The wave's opening terms, from the same settled reading as the headroom.
    pub(in super::super) fn admit_budget_terms(
        &self,
        room: &Headroom,
        chunk_rows: usize,
    ) -> Budget {
        Budget {
            // One currency for the rate, the dislodge and the floor alike — see
            // [`residency_now`]. Reading any of the three in a different one is
            // what lets the model be shown the same ground twice.
            resident: room.zone,
            // The same line `headroom` nets out of the spendable ground: the
            // hold, the eviction margin that keeps the cache evictable, and a
            // useful forward's tier. The model enforces it as a hard refusal.
            prefill_floor: room.prefill_floor().saturating_add(room.forward_reserve),
            decode_floor: room.floor().saturating_add(room.forward_reserve),
            max_rows: chunk_rows,
            max_decodes: MAX_DECODE_WIDTH,
        }
    }

    /// The recurrent store **one** new sequence needs, or zero on a stack that
    /// carries none.
    ///
    /// **Priced from the model's geometry, never from what is resident.** Every
    /// store the model builds has the same shape, so it can say what one costs
    /// without a store existing to measure — which matters, because admission
    /// prices a claim precisely before it is made.
    ///
    /// This was a mean over the live stores, and a mean is the one thing it
    /// could not be: `recurrent_reserved_bytes` sums every store the process
    /// holds, parked conversations included, while the only count available to
    /// divide by is what the scheduler has *in flight* — and admission runs
    /// between forwards, where that is 0 or 1. The two range over different
    /// populations, so the quotient is not a per-sequence figure but roughly
    /// the whole engine's carried state, and it climbs as conversations go
    /// idle. It therefore peaked exactly when admission should have been
    /// cheapest. Measured on a 72 GB card: a 41-row turn priced at 4,450 MiB
    /// and refused as throughput-worse on fifteen consecutive passes, seven
    /// turns queued behind it and 20 GiB standing free above the floor.
    ///
    /// Being a constant, it is also immune to the per-offer decay that made the
    /// old figure read 160, 80, 53, 40 MiB down a single pass as newly admitted
    /// prefills joined the denominator before allocating anything. It is still
    /// read once at [`AdmitPass::open`], because nothing in a pass can change
    /// it and re-reading would only take the lock again.
    pub(in super::super) fn recurrent_cost(&self) -> u64 {
        self.model.recurrent_store_bytes() as u64
    }

    /// Fold one completed prefill forward into the planner.
    ///
    /// Narrow forwards are refused by the model itself
    /// ([`rate::WaveRate::min_learn_rows`]): a wave too narrow to route across
    /// the whole layer copies a fraction of the experts the observation divides
    /// by, and one 26-row forward read 119 GB/s on a 25 GB/s link. Passing them
    /// in anyway is correct — the refusal belongs with the model that knows the
    /// routing, not with the caller.
    pub(in super::super) fn observe_prefill_forward(&mut self, rows: usize, pairs: f64, us: u64) {
        if rows == 0 || us == 0 {
            return;
        }
        // Counted even when the model refuses to learn from it: the loop's overhead is
        // amortised across forwards that *ran*, not across forwards wide enough to teach
        // the copy rate.
        self.wave_forwards += 1;
        let resident = residency_now(self.standing_ground_bytes());
        if let Some(rate) = self.wave_rate.as_mut() {
            rate.observe_prefill_attending(rows, pairs, resident, us as f64 / 1e6);
        }
    }

    /// A co-batched decode forward, for the decode fit: its creep's rows
    /// through layers are the fit's third term.
    pub(in super::super) fn observe_decode_beside(
        &mut self,
        decodes: usize,
        us: u64,
        creep_layer_rows: usize,
        creep_layer_pairs: f64,
        positions: f64,
        rows: usize,
    ) {
        if decodes == 0 || us == 0 {
            return;
        }
        let resident = residency_now(self.standing_ground_bytes());
        if let Some(rate) = self.wave_rate.as_mut() {
            rate.observe_decode_beside(
                decodes,
                resident,
                us as f64 / 1e6,
                creep_layer_rows,
                creep_layer_pairs,
                positions,
                rows,
            );
        }
    }

    /// Fold one completed decode forward into the planner.
    ///
    /// A bus-bound step says nothing about the layer time, and the model drops
    /// it; only a compute-bound one teaches anything. `rows` is every row the
    /// step scored — a drafted decode's whole verify block — so the misses the
    /// fit learns are the ones the step routed.
    pub(in super::super) fn observe_decode_forward(
        &mut self,
        decodes: usize,
        us: u64,
        positions: f64,
        rows: usize,
    ) {
        if decodes == 0 || us == 0 {
            return;
        }
        // Same reason as the prefill funnel: the loop's overhead is shared by every
        // forward a wave ran, of either kind.
        self.wave_forwards += 1;
        let resident = residency_now(self.standing_ground_bytes());
        if let Some(rate) = self.wave_rate.as_mut() {
            rate.observe_decode_at(decodes, rows, resident, us as f64 / 1e6, positions);
        }
    }

    /// Fold the expert cache's own hit and miss counters into the planner's hit
    /// coefficient.
    ///
    /// **The one estimate that cannot be guessed.** The LRU zone and the Markov
    /// predictor together are worth far more than the resident fraction alone,
    /// and by how much depends on the workload's routing: measured at 37%
    /// resident the hit rate was 0.65, a coefficient of ~1.73 against a seed of
    /// 0.7 — a decode's copy priced two and a half times too dear, which the
    /// wave pays for by refusing decodes that would have fitted.
    ///
    /// **Read unsettled, once per decode forward.** Settling the expert
    /// pipeline (`expert_stats`) waits for the device to reach the forward's
    /// last MoE layer and for the pipeline thread to serve every layer queued
    /// behind it — on Qwen3.8-Flash-Next single-session decode (RTX PRO 5000)
    /// reading unsettled took a 4K step from 23.6 to 19.9 ms. The interval it
    /// folds needs no exact boundary: a tally that trails by the layers still
    /// queued counts them in the next interval instead.
    pub(in super::super) fn observe_expert_hit_rate(&mut self) {
        let Some((total_hits, total_misses)) = self.model.expert_hit_counts() else {
            return;
        };
        // **The interval, not the lifetime.** Nothing in the daemon calls
        // `reset_expert_stats` — only the bench harness does — so these are
        // cumulative totals for the life of the process. Folding a lifetime
        // average into an exponential estimate converges it to that average
        // within a few dozen forwards and then freezes it, after which the
        // coefficient can no longer answer the question it exists for: how this
        // residency converts into hits. The sample count goes on rising, so the
        // telemetry reads healthy while the estimate has stopped moving.
        let (hits, misses) = (
            total_hits.saturating_sub(self.expert_hits_seen),
            total_misses.saturating_sub(self.expert_misses_seen),
        );
        self.expert_hits_seen = total_hits;
        self.expert_misses_seen = total_misses;
        let routed = hits + misses;
        if routed == 0 {
            return;
        }
        // Smoothed by half an expert each way: an interval that found every
        // expert says the hit is near certain, not that it is certain at every
        // residency, which an exponent of zero would claim.
        let hit = (hits as f64 + 0.5) / (routed as f64 + 1.0);
        let resident = residency_now(self.standing_ground_bytes());
        if let Some(rate) = self.wave_rate.as_mut() {
            rate.observe_hit_rate(hit, resident);
        }
    }

    /// The layer priority a slot decodes at, defaulting to the protective
    /// `High` when its target is not resolvable yet.
    pub(in super::super) fn decode_priority_or_high(&self, id: SequenceId) -> DecodePriority {
        self.decode_layer_priority(id)
            .unwrap_or(DecodePriority::High)
    }

    /// Whether a slot in flight — prefilling, ingesting a section, or decoding —
    /// belongs to a band below `prio`. A section ingest is at its own band.
    pub(in super::super) fn lower_band_active(&self, prio: DecodePriority) -> bool {
        let below = |p: DecodePriority| rank(p) < rank(prio);
        let prefills = self
            .active_prefills
            .iter()
            .filter(|p| p.error.is_none() && p.finished.is_none())
            .any(|p| below(self.decode_priority_or_high(p.work.sequence_id)));
        let sections = self
            .active_section_ingests
            .iter()
            .any(|s| s.error.is_none() && below(s.band));
        let decodes = self
            .active_decodes
            .iter()
            .filter(|(_, s)| !s.finished)
            .any(|(&id, _)| below(self.decode_priority_or_high(id)));
        prefills || sections || decodes
    }

    /// Running decodes of `prio`'s band or above — the ones a `prio` prompt
    /// does not outrank, whose steps the residency it spends still slows.
    pub(in super::super) fn decodes_at_or_above(&self, prio: DecodePriority) -> usize {
        self.active_decodes
            .iter()
            .filter(|(_, s)| !s.finished)
            .filter(|(&id, _)| rank(self.decode_priority_or_high(id)) >= rank(prio))
            .count()
    }

    /// Whether a live prompt of `turn` tokens on `seq` at `prio` would take
    /// rows in the next group formed though the forward's rows are spent.
    ///
    /// Its group carries its own band and above (`form_wave_group`), so the
    /// work in flight below it is no competition — a creep of theirs pauses
    /// for it. What stands ahead of it: work above its band, which runs first;
    /// a paused creep of its band or above, which resumes before any group
    /// forms (`resume_creep`); and the prompts of its own band nearer their
    /// first token, which the group re-formed nearest-first fills first, each
    /// at most a pass. It takes rows when those leave at least a chunk of the
    /// pass budget — and when none of them carries another adapter, since the
    /// nearest sets the group's. Sections ride behind every prompt, in rows
    /// the group holds back for them (`gang::prompt_budget`), and take nothing
    /// ahead of it.
    pub(in super::super) fn runs_ahead(
        &self,
        prio: DecodePriority,
        turn: usize,
        seq: usize,
    ) -> bool {
        let paused_ahead = self
            .paused_creep
            .as_ref()
            .is_some_and(|p| rank(p.band) >= rank(prio));
        if paused_ahead || self.waiting_above(prio, |_| false) {
            return false;
        }
        // Of its own band, the prompts nearer their first token. A paused
        // creep's members are below its band here, so none of them is one.
        let adapter = self.session.sequence_adapter(seq);
        let mut rows_ahead = 0usize;
        for p in &self.active_prefills {
            let remaining = p.work.tokens.len().saturating_sub(p.offset);
            if p.error.is_some()
                || p.finished.is_some()
                || remaining == 0
                || remaining > turn
                || self.decode_priority_or_high(p.work.sequence_id) != prio
            {
                continue;
            }
            if self.session.sequence_adapter(p.work.sequence_id.0) != adapter {
                return false;
            }
            rows_ahead += remaining.min(self.max_prefill_pass_tokens);
        }
        rows_ahead + turn.min(CHUNK_SIZE) <= self.prefill_pass_budget()
    }

    /// Whether a live section of `turn` tokens at `prio` would take rows in the
    /// next group formed though the forward's rows are spent.
    ///
    /// What stands ahead of it is what stands ahead of a prompt: work above its
    /// band and a paused creep of its band or above. Of its own kind, the live
    /// sections already in flight, whose rows the group holds back first. It
    /// takes rows when those leave at least a chunk of the pass budget. Past
    /// that it would claim its store and advance by nothing, as a turn joining
    /// a full forward would ([`room_left`]).
    pub(in super::super) fn section_runs_ahead(&self, prio: DecodePriority, turn: usize) -> bool {
        let paused_ahead = self
            .paused_creep
            .as_ref()
            .is_some_and(|p| rank(p.band) >= rank(prio));
        if paused_ahead || self.waiting_above(prio, |_| false) {
            return false;
        }
        let rows_ahead: usize = self
            .active_section_ingests
            .iter()
            .filter(|s| s.error.is_none() && joins_a_full_forward(s.band, s.tokens.len()))
            .map(|s| s.tokens.len().saturating_sub(s.offset))
            .sum();
        rows_ahead + turn.min(CHUNK_SIZE) <= self.prefill_pass_budget()
    }
}
