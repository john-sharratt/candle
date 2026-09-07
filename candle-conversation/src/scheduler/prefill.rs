use super::admit;
use super::interleave;
use super::*;
use crate::projection::DecodePriority;

/// The engine as [`interleave::fill`] sees it: a cursor over both queues that
/// really claims what it takes.
///
/// Holds the scheduler mutably for the whole fill because every admission is a
/// device allocation and every check is a read of the partition that allocation
/// moved — there is no snapshot to work from, which is the entire point.
struct WaveFill<'a> {
    sched: &'a mut Scheduler,
    /// Decodes eligible this wave, in the order they are offered. Rotated so
    /// the sequences past last wave's cut are offered first this wave — a
    /// fixed order plus a cut is a fixed set of starved sequences.
    decode_order: Vec<SequenceId>,
    /// How far down `decode_order` the offers have reached, per priority band.
    /// One cursor cannot serve six passes: the bands are offered in turn and
    /// each has to resume where *it* left off, not where the previous band did.
    decode_cursor: [usize; 3],
    /// The decodes admitted, in order — **the set the wave runs**, handed to
    /// `batch_decode_step` so the sequences whose ground was claimed here are
    /// exactly the sequences that step.
    decodes_taken: Vec<SequenceId>,
    /// Eligible decodes the allocators refused this fill. Published as the
    /// engine's "full" signal: a producer reading a non-zero here stops
    /// opening conversations.
    decodes_refused: usize,
    /// Queue indices of the prefills admitted this fill, in admission order.
    /// A prefix of the band's FIFO order: nothing is passed over, so the plan
    /// removes exactly these.
    prefill_admitted: Vec<usize>,
    /// The next queue index to offer, per priority band — see
    /// [`Self::decode_cursor`].
    prefill_cursor: [usize; 3],
    /// The weight floor this fill defends — the `min` of the range admission
    /// works inside, and the point past which a claim costs resident experts.
    optimal: u64,
}

/// The widest the tier margin grows on repeated refusals, in regions — a
/// gigabyte, past which a refusal is not a rounding problem.
const TIER_MARGIN_CAP_REGIONS: usize = 64;

/// The fewest tokens a prefill advances in one wave, when it has that many
/// left. A dialogue prefill rides the wave in chunks — `[offset, offset +
/// advance)` per group, exactly as a section ingest does — so a turn of any
/// length is carried beside the decodes in pieces the transient tier can hold,
/// instead of needing the tier for its whole token set at once (≈1 MiB a row
/// on the 35B hybrid: a 2,855-token read turn is a 3 GiB tier, which no
/// partition on a 16 GiB card places beside twenty decodes). Below this the
/// chunk amortises its expert load over too few rows to be worth the wave, so
/// the fill holds the item until the tier has this much room for it.
pub(super) const PREFILL_MIN_ADVANCE: usize = 128;

/// The slots an idle-demotion pass takes, advancing the idle counters as it
/// goes: a slot the engine touched restarts its count, and a quiet slot demotes
/// on the pass that reaches [`Scheduler::IDLE_SLOT_DEMOTE_PASSES`].
///
/// **A slot holding no blocks is still demoted, and that is the whole point.**
/// It used to be skipped, on the reasoning that a slot with nothing on the
/// device has nothing to give back. That is false, and it made this pass inert
/// for the population it exists to serve: a slot's block table and the
/// substrate's *hot* copies of its sealed turns are two different holdings, and
/// `apply_projection` truncates the slot to zero blocks at every turn. So a
/// conversation between turns — exactly the case this sheds — reads
/// `tokens == 0` while its hot turn residences are untouched. Those are the
/// large half: `hot = None` freed 8–9 MiB a pass against 5,920 MiB of resident
/// K/V, and the census reported `idle_slots = 0` throughout, because every
/// waiting conversation had been classified as holding nothing.
///
/// Demoting resets the counter rather than leaving it above the threshold, so a
/// slot sheds once per quiet window instead of re-running the substrate walk on
/// every wave for the rest of its life.
///
/// Pure over `(slots, busy)` so the policy is testable without a session or a
/// substrate; `slots` carries each live slot with the tokens its block table
/// currently holds.
fn idle_slots_to_demote(
    counters: &mut HashMap<SequenceId, u32>,
    slots: &[(SequenceId, usize)],
    busy: &HashSet<SequenceId>,
) -> IdlePass {
    let mut pass = IdlePass::default();
    for (id, tokens) in slots.iter().copied() {
        if busy.contains(&id) {
            // Report only slots that were actually being aged. One busy
            // throughout has no quiet to interrupt and nothing to say about how
            // long a demotion lasts.
            if counters.remove(&id).is_some() {
                pass.woke.push(id);
            }
            continue;
        }
        // **Exactly on the crossing pass, then never again until touched.** A
        // `>=` would re-demote a slot that simply stays quiet on every
        // subsequent pass — with a grace of one, every pass — and each of those
        // costs a substrate write lock and a walk per conversation, contending
        // with the persistence thread to shed nothing (the first demote already
        // took it all). The counter keeps climbing, saturating, so the equality
        // holds only once per quiet period; a touch resets it and re-arms.
        let quiet = counters.entry(id).or_insert(0);
        *quiet = quiet.saturating_add(1);
        if *quiet == Scheduler::IDLE_SLOT_DEMOTE_PASSES {
            pass.demote.push((id, tokens));
        }
    }
    // Counters for slots that are gone (freed between passes) would otherwise
    // accumulate for the life of the process.
    let live: HashSet<SequenceId> = slots.iter().map(|(id, _)| *id).collect();
    counters.retain(|id, _| live.contains(id));
    pass
}

/// What one [`idle_slots_to_demote`] pass decided.
#[derive(Debug, Default, PartialEq, Eq)]
struct IdlePass {
    /// `(slot, tokens)` per demotion. `tokens` is what the caller truncates and
    /// may be zero — a slot between turns holds no blocks while its bytes sit
    /// in the substrate's hot copies.
    demote: Vec<(SequenceId, usize)>,
    /// Slots that were being aged and have gone busy again. Paired against the
    /// caller's demotion timestamps this measures how long a demotion actually
    /// lasts, which is the number that says whether the grace window earns its
    /// keep or is pure delay.
    woke: Vec<SequenceId>,
}

/// The widest wave the engine will compose, whatever the hit rate says: the
/// widest rung the batched-forwarding gates measure (`C10 × 64`), and the
/// backstop [`Scheduler::MAX_PREFILL_WIDTH`] is set to match. A model's own
/// row (`decode_width_target`) bounds the width on a streaming card; this
/// bounds it on a card that holds every expert, where the hit rate alone
/// would never stop widening.
pub(super) const WAVE_WIDTH_HARD_CAP: usize = 64;

impl Scheduler {
    /// A wave's transient tier was refused placement. Nothing ran and the KV
    /// side rolled back, so nothing has failed: the wave was composed too wide
    /// for the ground the placement found, and the next fill composes it
    /// narrower.
    ///
    /// Two things make that so. The margin the fill holds back from the tier
    /// budget doubles, so the same gap prices to fewer rows. And the prefills
    /// the refused wave carried go **back to the front of the queue**, in
    /// order — they were admitted but never started, so there is nothing to
    /// unwind but the admission itself — and are offered again against the
    /// wider margin. Measured before this: one wave priced 26 MiB over a
    /// 4,054 MiB gap failed 18 directories.
    ///
    /// **A refusal at the widest margin is final for the prefills that had
    /// started.** A started prefill (one with chunks already committed) cannot
    /// be requeued, so it rides the next group — and if the placement refuses
    /// that group too, and the next, the wave never advances: measured, 1,641
    /// refusals of one wave with nothing requeued, the margin at its cap from
    /// the third refusal on. When the margin is already at the cap and the
    /// refusal comes again, the started prefills are failed with the numbers
    /// and their sequences released, so the pipeline moves.
    pub(super) fn note_tier_refusal(&mut self, err: &candle::Error) {
        let at_cap = self.tier_margin_regions >= TIER_MARGIN_CAP_REGIONS;
        self.tier_margin_regions = (self.tier_margin_regions * 2).min(TIER_MARGIN_CAP_REGIONS);
        let mut unstarted = Vec::new();
        let mut i = 0;
        while i < self.active_prefills.len() {
            let p = &self.active_prefills[i];
            if p.offset == 0 && p.final_logits.is_none() && p.error.is_none() {
                unstarted.push(self.active_prefills.remove(i));
            } else {
                i += 1;
            }
        }
        let requeued = unstarted.len();
        // Back to the front, in their original order.
        for p in unstarted.into_iter().rev() {
            self.prefill_queue.push_front(p.work);
        }
        let mut failed = 0usize;
        if at_cap {
            let started: Vec<usize> = self
                .wave_prefill_members
                .iter()
                .filter_map(|m| match m {
                    WaveMember::Prefill { seq_id, .. } => Some(*seq_id),
                    WaveMember::Section { .. } => None,
                })
                .collect();
            for p in self.active_prefills.iter_mut() {
                if p.error.is_none()
                    && p.final_logits.is_none()
                    && p.offset > 0
                    && started.contains(&p.work.sequence_id.0)
                {
                    p.error = Some(ConversationError::Channel(format!(
                        "prefill of {} tokens ({} committed): the wave transient tier refused \
                         its next chunk at the widest margin ({} regions) — this partition \
                         cannot place it",
                        p.work.tokens.len(),
                        p.offset,
                        TIER_MARGIN_CAP_REGIONS,
                    )));
                    failed += 1;
                }
            }
        }
        self.prefill_head_blocked = false;
        tracing::warn!(
            target: "candle_conversation::scheduler::interleave",
            margin_regions = self.tier_margin_regions,
            requeued,
            failed,
            "wave transient tier refused placement — wave requeued, margin widened: {err}",
        );
    }
}

impl<'a> WaveFill<'a> {
    fn new(sched: &'a mut Scheduler, optimal: u64) -> Self {
        let mut decode_order = sched.decode_wave_candidates();
        // Start the offers just past the last sequence admitted last wave, so
        // whatever was refused at the tail goes first now. The list is sorted
        // by id (`decode_wave_candidates`), so the split point is the first id
        // above it.
        if let Some(last) = sched.last_decode_admitted {
            let split = decode_order.iter().position(|id| id.0 > last).unwrap_or(0);
            decode_order.rotate_left(split);
        }
        Self {
            sched,
            decode_order,
            decode_cursor: [0; 3],
            decodes_taken: Vec::new(),
            decodes_refused: 0,
            prefill_admitted: Vec::new(),
            prefill_cursor: [0; 3],
            optimal,
        }
    }

    /// The band a sequence belongs to, as an index into the per-band cursors.
    ///
    /// A slot whose target or timeline will not resolve is treated as `High` —
    /// the protective answer, matching `wave_prefill_layer_budget` and the
    /// decode batch's own ordering. Guessing `Low` for an unresolvable dialogue
    /// slot would put a person's token behind a repository scan.
    fn band_of(&self, seq: SequenceId) -> usize {
        band_index(
            self.sched
                .decode_layer_priority(seq)
                .unwrap_or(DecodePriority::High),
        )
    }

    /// Rows the decodes taken so far put at the head of the wave: a drafted
    /// decode rides as a verify block of `1 + draft` rows in the prefill slot.
    fn head_rows(&self) -> usize {
        let decodes = self.decodes_taken.len();
        decodes * (1 + self.sched.model.draft_budget(decodes))
    }

    /// Tier bytes a forward worth running needs, held back from admission.
    ///
    /// **The tier is the span's third tenant and takes whatever K/V leaves at
    /// the frontier**, so admission stops this far short of the weight floor.
    /// Every K/V reservation raises the arena frontier, and the tier lives in
    /// what is left between the frontier and the weight side.
    ///
    /// **It must be a useful forward's worth, not merely a placeable one.**
    /// Both failures are measured. Reserve nothing and K/V claims to the floor,
    /// the tier reaches zero and no forward can be planned at all: run CB, wave
    /// after wave of `(no forwards)` with 73 slots admitted. Reserve one row and
    /// a forward *can* be placed but not filled — run CG opened 96 slots on the
    /// freed budget, left the tier 176 MiB, and every forward carried
    /// `seqs avg=1.0`, one sequence at a time, with 335 of 468 regions standing
    /// free. A tier that can only place a forward is barely better than one that
    /// cannot.
    ///
    /// So it is [`PREFILL_MIN_ADVANCE`] rows — the least the engine will hand a
    /// sequence — priced through the same `tier_bytes` that places the tier.
    /// That is ~1.5 GiB on the 4090 and buys wide forwards: run CE carried it
    /// and reached 82 directories at 102 tok/s aggregate with a 150 median,
    /// where CG's one-row reserve produced single-sequence waves.
    ///
    /// Reading it through the planner is what makes it portable: it follows the
    /// model's geometry and the activation dtype rather than being a byte count
    /// to re-derive per card.
    fn min_forward_tier_bytes(&self) -> u64 {
        let dtype = self.sched.session.activation_dtype();
        let plan = WavePlan::new(self.sched.model.wave_geometry(dtype));
        plan.tier_bytes(PREFILL_MIN_ADVANCE) as u64
    }

    /// Prefill tokens the transient tier has room for beside `head_rows` rows
    /// already in the wave, when the next item would add `want` tokens.
    ///
    /// **The co-batched wave is bounded here, and nowhere else.** The engine's
    /// slab packer bounds a *pure* prefill wave; a wave carrying decode rows
    /// takes its prefill group whole, so a scheduler that admitted freely built
    /// waves whose tier came to 6.3 GiB against a 6.1 GiB gap — every one of
    /// them refused, every one a failed directory.
    ///
    /// **The tier's ground is bought here, at fill time, or not counted.** The
    /// gap between the arena frontier and the weight floor is what stands free
    /// and is small on its own — live arenas are scattered up to the floor. The
    /// rest of a tier's room is weight-side ground the zone can concede down to
    /// the hold point. Pricing that concession in and leaving the purchase to
    /// the placement was measured twice and refused twice (`needs 4,064 MiB
    /// against a 3,830 MiB gap`, the weight side "could not concede"): the
    /// placement runs inside the forward, where the boundary may not move. The
    /// fill runs between forwards, where it may, so when the head's tier would
    /// exceed the gap and the zone stands far enough above the hold to cover
    /// the shortfall, the fill asks the weight side for exactly that now
    /// (`request_kv_ground`) and prices against the gap it then measures. What
    /// the weight side does not concede is not a budget. The result is
    /// recorded on the session so the engine's slab packer prices against the
    /// same number and does not re-slice a group this fill composed.
    /// Publish this wave's transient-tier budget, so the engine's slab packer
    /// prices against the same ground the fill did.
    ///
    /// The frontier gap, less the margin a refusal widens. **Nothing is bought
    /// from the weight side.** The old fill asked the zone to concede ground
    /// when a wave's tier did not fit, and that ask is exactly the thing
    /// admission now exists to refuse: ground taken there is resident experts,
    /// and an engine that streams its experts is slower at everything. A wave
    /// whose tier will not fit the standing gap is a wave that should be
    /// narrower, which is `admit::gate`'s answer, not the zone's to pay for.
    fn publish_tier_budget(&mut self) {
        let margin = self.sched.tier_margin_regions * REGION_BYTES;
        // **Ground the weight side is owed comes off the top.** When residency
        // stands under its hold, the frontier gap is not the tier's to take: it
        // is where the weight side grows back, as fast as it is left free.
        // Publishing the whole gap is what crushed the weights on run BT — the
        // tier for a wide wave took its ground from the zone, which fell to
        // 1,417 MiB against a 4,775 hold, and the forward then failed outright
        // with `Expert cache full, cannot evict (all pinned)` because every
        // remaining slot was pinned by the wave needing it. 125 forwards died
        // that way.
        //
        // **The debt is measured against the effective zone, never the extent.**
        // The extent is where the boundary happens to rest, and nothing moves it
        // back on its own: the weight side grows only when the expert cache asks
        // for ground it cannot evict, so an extent that settles under the hold
        // stays there and the debt never clears. Run BW showed what that costs
        // within three directories — extent 4,240 MiB against a 4,772 hold, so a
        // 532 MiB debt was deducted on every wave, the tier reached exactly
        // zero, prefill forwards carried 2 to 4 sequences with 82 standing
        // admitted, and the effective zone was a wholly healthy 6,112 MiB the
        // entire time. Run BV died of the same arithmetic after 26 minutes at 13
        // tok/s. The effective zone charges every live region the instant it is
        // claimed, so it says whether residency is *actually* short — which is
        // the only condition under which the tier owes anything.
        let owed = interleave::effective_weight_zone_bytes()
            .map_or(0, |zone| self.optimal.saturating_sub(zone)) as usize;
        let budget = transient_headroom_bytes(0)
            .unwrap_or(0)
            .saturating_sub(margin)
            .saturating_sub(owed);
        self.sched.session.set_tier_budget_bytes(budget);
    }
}

/// Whether a prefill can contribute nothing more to a wave group.
///
/// **This is one predicate on purpose, and both callers must use it.**
/// `form_wave_group` declines to schedule a prefill this returns true for, and
/// `promote_finished_prefills_to_decodes` takes exactly those — so every prefill
/// is either advancing in a wave or being drained, and none can be both
/// unschedulable and unpromotable.
///
/// They used to disagree, and the gap between them stalled the engine outright.
/// The group skipped on `logits || consumed` while promotion took only
/// `logits && consumed`, so a prefill that had consumed all its tokens without
/// producing logits was skipped by the first and refused by the second: it could
/// never ride a forward, so it could never get the logits that would have let it
/// be promoted. It simply sat in `active_prefills` forever. Run CE ended that
/// way at 82 directories — four such slots, no rows to build a forward from,
/// `(no forwards)` wave after wave with 220 free regions and every resource
/// standing idle. A slot with logits but tokens left cannot advance either, for
/// the same reason, so it is drained here too rather than left in the same trap.
///
/// A slot whose logits are missing is not silently promoted: the drain reports
/// it as a failed turn and frees it, which is a bounded loss of one turn instead
/// of an unbounded loss of the engine.
fn prefill_done(has_logits: bool, offset: usize, tokens: usize) -> bool {
    has_logits || offset >= tokens
}

/// Whether a prefill has no more chunks to submit: its last chunk landed and
/// produced the logits the first decode step samples from. Such an entry stays
/// in `active_prefills` — holding its prefix — until it is drained.
fn prefill_finished(p: &ActivePrefill) -> bool {
    prefill_done(p.final_logits.is_some(), p.offset, p.work.tokens.len())
}

/// A priority as an index into the per-band cursors, highest first.
fn band_index(p: DecodePriority) -> usize {
    match p {
        DecodePriority::High => 0,
        DecodePriority::Normal => 1,
        DecodePriority::Low => 2,
    }
}

impl WaveFill<'_> {
    /// The next FIFO candidate in this band, as `(queue index, sequence, whole
    /// turn's tokens, tokens riding this wave)`. Does not consume it.
    ///
    /// **The two token counts are different and both matter.** The KV claim is
    /// for the *whole turn* — every chunk of it lands in this sequence's cache
    /// and is never given back until the turn seals — while only `advance`
    /// rides this forward and needs transient tier. Pricing the chunk and
    /// claiming the turn is what collapsed run BS: the gate authorised a
    /// quarter of what the allocator then took, the weight zone fell from 5,020
    /// to 1,417 MiB against a 4,774 hold, and the expert hit rate went to 0.257.
    fn peek_prefill(&self, band: usize) -> Option<(usize, SequenceId, usize, usize)> {
        let cap = self.sched.max_prefill_pass_tokens.max(1);
        let from = self.prefill_cursor[band];
        for idx in from..self.sched.prefill_queue.len() {
            let w = &self.sched.prefill_queue[idx];
            if self.band_of(w.sequence_id) != band {
                continue;
            }
            let whole = w.tokens.len();
            return Some((idx, w.sequence_id, whole, whole.min(cap)));
        }
        None
    }

    /// The next decode candidate in this band that has not already been taken.
    fn peek_decode(&self, band: usize) -> Option<SequenceId> {
        let from = self.decode_cursor[band];
        self.decode_order[from.min(self.decode_order.len())..]
            .iter()
            .copied()
            .find(|id| self.band_of(*id) == band && !self.decodes_taken.contains(id))
    }

    /// One 32-token block, in the formats a **live** sequence occupies.
    fn per_block_bytes(&self) -> u64 {
        let (k, v) = self.sched.session.active_kv_formats();
        admit::cost::per_block_kv_bytes(
            self.sched.session.num_layers(),
            self.sched.session.n_kv_head(),
            self.sched.session.head_dim(),
            k,
            v,
        )
    }

    /// What admitting `seq` would take: `claimed` tokens of K/V — the whole turn,
    /// which is what the allocator is asked for — and transient tier for the
    /// `advance` that rides this forward.
    fn price(&self, seq: SequenceId, claimed: usize, advance: usize) -> admit::Cost {
        let held = self.sched.session.sequence_offset(seq.0).unwrap_or(0);
        let kv = admit::cost::kv_bytes_for_advance(held, claimed, self.per_block_bytes());
        let recurrent = if self.sched.model.carries_recurrent_state()
            && !self.sched.model.recurrent_resident(seq.0)
        {
            self.sched.model.recurrent_store_bytes() as u64
        } else {
            0
        };
        // The rows this admission adds to the forward, priced through the same
        // planner that places the tier — so this is the tier's cost, not an
        // estimate of it.
        let rows = self.head_rows() + advance;
        let dtype = self.sched.session.activation_dtype();
        let plan = WavePlan::new(self.sched.model.wave_geometry(dtype));
        let activations = plan
            .tier_bytes(rows)
            .saturating_sub(plan.tier_bytes(self.head_rows())) as u64;
        admit::Cost {
            kv,
            recurrent,
            activations,
        }
    }
}

impl admit::Ground for WaveFill<'_> {
    fn active(&self) -> usize {
        self.sched.active_slots()
    }

    fn decodes_active(&self) -> usize {
        self.sched
            .active_decodes
            .values()
            .filter(|s| !s.finished)
            .count()
    }

    /// Defers to [`Scheduler::admission_due`], which the eviction pass reads
    /// too — the two must answer identically or relief and admission fall out of
    /// step. See that method for the three cases and what each one cost.
    fn settled(&self) -> bool {
        self.sched.admission_due()
    }

    fn headroom(&self) -> admit::Headroom {
        // **One measure, read once.** The *effective* zone is the residency the
        // weight side could reach: by its own identity
        // (`interleave::achievable_weight_bytes`) that is
        // `extent + non-live regions - reserve`, so it already accounts for
        // every free region and for every claim the instant it is made. The
        // boundary's *extent* is deliberately not consulted here — it lags,
        // moving only between forwards, and every attempt to combine the two
        // has ended up counting the same ground twice (see below).
        let zone = interleave::effective_weight_zone_bytes().unwrap_or(u64::MAX);
        // The KV side and the weight side share one elastic span: a claim that
        // runs out of free regions buys its ground from the weight zone on the
        // spot, and that is legitimate all the way down to the hold, which is
        // the line below which the model would start streaming. Pricing against
        // the free list alone refused admissions with gigabytes standing above
        // that line — measured on run BR's calibration: three sequences a
        // forward where every earlier run carried six, 979 tokens against
        // 1,958, and the phase aggregate down 29%.
        let room = admit::Headroom {
            free_kv: 0,
            zone,
            zone_min: self.optimal,
            zone_max: interleave::achievable_weight_now().unwrap_or(u64::MAX),
        };
        // **The spendable ground is one subtraction: how far residency stands
        // above its floor.**
        //
        // The free list is not separate headroom to be added to that. It is the
        // *same* ground: `effective = extent + non-live regions - reserve`, so
        // claiming a free region moves it from non-live to live and drops
        // `effective` by exactly its size. Spending the free list therefore
        // spends the weight side's residency byte for byte, and any formula that
        // adds the two lets admission spend the same ground twice.
        //
        // Both earlier shapes were that double-count wearing different hats.
        // `free_list + (effective - floor)` counted the free regions once inside
        // `effective` and once beside it — run BV stood 95 slots open on 1.6 GiB
        // that did not exist. Netting a debt off the free list instead
        // (`free_list - (floor - effective) + cedeable`) still treated the list
        // as spendable while it lasted, so the zone simply drained before the
        // debt term grew large enough to bite: run CA opened 40 slots and had
        // driven residency to 2,314 MiB against a 4,772 MiB hold by its first
        // directory, and run BZ to 2,617.
        //
        // Read as a single quantity it is obvious, and it is self-correcting: as
        // slots finish and their regions come back, `effective` rises and the
        // room reopens by the same amount. The floor is the hold *plus* an
        // eviction margin, because stopping at the hold itself leaves every
        // expert slot pinned and the forward fails outright rather than
        // degrading.
        //
        // Nothing is subtracted for work already in flight, and nothing should
        // be: an admission *reserves* its prompt and its decode lease through
        // `ensure_capacity`, so a live slot's ground is already live ground and
        // `effective` already reflects it. Charging it again here would refuse
        // admissions the device could afford.
        //
        // **The span holds three tenants, so the floor answers to two of them.**
        // It is `| persist | KV | wave transient tier | expert weights |`, and
        // K/V claiming down to the weight side's floor still leaves the tier
        // nothing: the tier sits between them, at the arena frontier, and takes
        // what K/V has not. With no tier there is no forward at all — not a
        // narrow one, none — so nothing completes, no ground comes back, and the
        // tier cannot recover. Run CB wedged exactly there: `tier=0MiB`, then
        // `(no forwards)` wave after wave with 73 slots admitted and every one
        // of them idle.
        //
        // **The reserve comes out of the elastic budget, not out of the current
        // frontier gap.** Those look interchangeable and are not: the gap is
        // `weight_floor - live_end()`, and the weight floor *moves* — K/V buys
        // ground from the weight side on the spot, down to the hold. So the gap
        // is smallest exactly when residency is healthiest, and gating on it
        // inverts the logic. Measured: run CK refused after two prefills a fill
        // with an 8,204 MiB weight zone and 3.7 GiB of residency headroom
        // standing unused, because the momentary gap was 64 MiB. One budget,
        // with both floors taken out of it, is the shape that holds.
        let tier_reserve = self.min_forward_tier_bytes();
        admit::Headroom {
            free_kv: zone.saturating_sub(room.floor().saturating_add(tier_reserve)),
            ..room
        }
    }

    fn peek(&mut self, kind: admit::Kind, prio: DecodePriority) -> Option<admit::Cost> {
        use admit::Kind;
        let band = band_index(prio);
        match kind {
            Kind::Prefill => {
                let (_, seq, whole, advance) = self.peek_prefill(band)?;
                // **The price is the prompt and the lease that follows it.**
                // Admitting a prefill commits to the whole slot — prefill, then
                // decode to the end of its lease — and none of that ground comes
                // back until it seals. `admit` reserves exactly this, so the
                // price is what the allocator is about to be handed rather than
                // a guess at it.
                let lease = Scheduler::DECODE_LEASE_TOKENS;
                Some(self.price(seq, whole.saturating_add(lease), advance))
            }
            Kind::Decode => {
                let seq = self.peek_decode(band)?;
                // **A decode step buys no K/V.** Its whole lease was reserved
                // through `ensure_capacity` when the slot was admitted, so the
                // blocks it writes into are blocks it already owns; charging it
                // again is the same ground counted twice, and this time it
                // charges the tenant that cannot pay. Run CC showed what that
                // costs: once the budget closed, every decode was refused, so
                // prefills promoted into decodes that never stepped, nothing
                // generated, and 44 slots sat admitted with 52 queued behind
                // them while the engine ran forwards that produced no tokens.
                //
                // What a step *does* cost is the tier for its row, which the
                // wave has to place whether or not the K/V is already there — so
                // that term stands and the K/V term is zero.
                let step = Scheduler::DECODE_CLAIM_TOKENS;
                Some(admit::Cost {
                    kv: 0,
                    ..self.price(seq, step, step)
                })
            }
        }
    }

    fn admit(&mut self, kind: admit::Kind, prio: DecodePriority) -> bool {
        use admit::Kind;
        let band = band_index(prio);
        match kind {
            Kind::Prefill => {
                let Some((idx, seq, whole, _advance)) = self.peek_prefill(band) else {
                    return false;
                };
                self.prefill_cursor[band] = idx + 1;
                // The whole turn **and its decode lease**, matching what `peek`
                // priced. `claim_kv` ensures capacity, so this reserves the
                // ground rather than predicting it: the arena for the generation
                // is taken here, at admission, and the slot cannot later grow
                // into ground the gate never authorised.
                let reserve = whole.saturating_add(Scheduler::DECODE_LEASE_TOKENS);
                if !self.sched.claim_kv(seq.0, reserve) || !self.sched.claim_recurrent(seq.0) {
                    return false;
                }
                self.prefill_admitted.push(idx);
                true
            }
            Kind::Decode => {
                let Some(seq) = self.peek_decode(band) else {
                    return false;
                };
                if let Some(at) = self.decode_order.iter().position(|id| *id == seq) {
                    self.decode_cursor[band] = at + 1;
                }
                if !self.sched.claim_kv(seq.0, Scheduler::DECODE_CLAIM_TOKENS)
                    || !self.sched.claim_recurrent(seq.0)
                {
                    self.decodes_refused += 1;
                    return false;
                }
                self.decodes_taken.push(seq);
                true
            }
        }
    }
}

use crate::persistence::thread::effective_turn_policy;
use crate::substrate::ConvCompression;
use crate::token_buffer::TokenBuffer;
use candle_nn::kv_cache::{
    end_wave_transient, is_device_oom, is_tier_refusal, transient_headroom_bytes, WavePlan,
    REGION_BYTES,
};
use candle_transformers::models::batched_inference::PendingGlue;
use std::collections::{HashMap, HashSet};

/// Free KV regions kept in hand before [`Scheduler::vram_under_pressure_for`]
/// calls it pressure, as a divisor of the reservation's KV side plus an absolute
/// floor in regions. This is §3.8's setpoint.
///
/// It replaced a band of *bytes* derived from the driver — headroom held against
/// a wide forward's transient activation peak. That quantity is no longer the KV
/// side's business: transients come from the reservation's other end (§3.6), and
/// what a seal pass needs is simply somewhere to put its chunks. So the setpoint
/// asks the only question that remains, and asks it of an exact counter: are
/// there enough free regions to absorb the work already admitted?
///
/// Scaled to the span rather than fixed, so the same numbers hold on a 3.6 GiB
/// KV side and on the workstation's. Step 6 tunes both terms against the
/// observed claim rate; the floors are what keeps a small card from setting a
/// setpoint of two regions and stalling mid-seal.
const LOAD_SETPOINT_DIVISOR: usize = 8;
const LOAD_SETPOINT_FLOOR_REGIONS: usize = 24;
/// Decode's setpoint is half of load's: a decode step advances one token per
/// sequence, so KV grows by ~one chunk per sequence per 32 steps — orders of
/// magnitude slower than a prefill's upload, and the whole point of unbounded
/// context is to leave KV resident rather than evict it defensively.
const DECODE_SETPOINT_DIVISOR: usize = 16;
const DECODE_SETPOINT_FLOOR_REGIONS: usize = 8;

/// What one [`Scheduler::compress_pending_turns`] pass achieved.
///
/// The two fields answer different questions and the caller needs both:
/// `compressed == 0` alone cannot distinguish "there was nothing pending" from
/// "the rung ran and the pool refused it ground", and those want opposite
/// responses — the first is a quiet pass, the second is the compress-to-free
/// rung failing at the moment compression is what would relieve the pressure.
#[derive(Default)]
pub(super) struct CompressPass {
    /// Turns whose hot copy was replaced by its quantized form.
    compressed: usize,
    /// The pass stopped early because a quantize destination could not be
    /// allocated, rather than because it ran out of work or hit its budget.
    refused: bool,
}

/// The region quantum in bytes.
fn region_bytes() -> u64 {
    candle_nn::kv_cache::REGION_BYTES as u64
}

/// The free-region setpoint for `phase`, in regions, given a KV side of
/// `total` regions. Pure — unit-tested in isolation.
fn setpoint_regions(phase: VramPhase, total: usize) -> usize {
    let (divisor, floor) = match phase {
        VramPhase::Load => (LOAD_SETPOINT_DIVISOR, LOAD_SETPOINT_FLOOR_REGIONS),
        VramPhase::Decode => (DECODE_SETPOINT_DIVISOR, DECODE_SETPOINT_FLOOR_REGIONS),
    };
    // Never ask for more than half the span: on a card too small to hold the
    // setpoint, demanding it would mean permanent pressure and an eviction pass
    // per wave that can never succeed.
    (total / divisor).max(floor).min(total / 2)
}

/// The phase a VRAM pressure decision is made in. Both phases read the same
/// exact counter — free regions — and differ only in how many they insist on:
///
/// - [`Load`](VramPhase::Load) — bringing KV into VRAM *before* attention
///   (prefill upload, section/scope ingest, warm→hot elevation). A wide ragged
///   forward claims regions fast, so the setpoint is wide enough that a seal
///   pass never finds the free list empty mid-wave.
/// - [`Decode`](VramPhase::Decode) — one token per sequence per step, so KV
///   grows slowly and predictably. A thin setpoint keeps the maximum KV
///   resident, which is the whole point of unbounded context.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum VramPhase {
    Load,
    Decode,
}

/// Regions the relief sequence frees past the setpoint, so a pass that just
/// clears pressure does not re-trip on the very next wave. Eviction is bulk and
/// coarse by nature — one turn's hot copy spans many chunks — so overshooting
/// deliberately is cheaper than nibbling every wave, which is what caused the
/// reload churn the old watermark ladder was built to damp.
const RELIEF_OVERSHOOT_REGIONS: usize = 8;

/// Capacity fraction (%) at which cold **ingest** KV starts demoting to the warm
/// (RAM) tier — gentle and early, well before the free-region setpoint is
/// approached at all. Ingest KV is zero-reload-cost (never
/// re-attended until query time; it re-elevates warm→hot on demand), so it is the
/// cheapest relief and sheds first.
const INGEST_DEMOTE_PCT: usize = 50;
/// Backlog (as a % of resident capacity) above which the wave loop blocks on a
/// device sync after its eviction callbacks — "heavy pressure". Draining the
/// primary stream lets the (now cross-layer-batched, short) hot→warm pass run
/// uncontended by ingest forwards, so it catches up instead of interleaving.
/// This is the *only* backpressure the hot→warm drain applies to ingest: the
/// gentle byte-setpoint throttle that used to act first was removed with the
/// admission budget it moved (nothing read that budget once the wave fill
/// replaced it).
const INGEST_SYNC_CEILING_PCT: usize = 25;
/// Slack the warm PIPELINE may hold above the standing budget: hot→warm output
/// that exists only while the drain moves it to cold. On a zero-budget machine
/// this is the only warm residency there ever is, and cutting admission for it
/// would recreate the ratchet-to-the-floor failure — the drain clears it in a
/// pass. The throttle fires only when `resident + pending` exceeds
/// `budget + slack`, i.e. when the drain is genuinely not keeping up.
pub(super) const WARM_PIPELINE_SLACK_BYTES: u64 = 1024 * 1024 * 1024;
/// Minimum spacing between OS memory probes for host-RAM backpressure —
/// `sysinfo` is a syscall, so the scheduler caches the reading between waves.
const HOST_RAM_PROBE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(1000);
/// Sealed ingest turns kept hot per timeline (the rolling window) before the
/// gentle-early demote sheds the rest to RAM.
///
/// **Must cover the ingest projection's gather width.** With the tool-round-trip
/// ingest, each scope's summary decode projects the `scopes` group (`top_k` turns)
/// — i.e. an actively-ingesting conversation RE-ATTENDS its own recent turns every
/// scope. If this window is narrower than that gather, the demote sheds turns the
/// very next projection re-elevates: a warm↔hot churn that stalls the decode batch.
/// The scopes group is `top_k: 4`, so a scope's projected working set is ~4 turns
/// (2 coupled turns × ~2 scopes); 8 keeps a couple of scopes of margin resident so
/// the active working set never leaves hot.
const INGEST_HOT_WINDOW: usize = 8;

/// Max float bytes the synchronous compress-to-free rung brings forward per relief
/// episode. Bounds the per-episode stall: a large accumulated backlog drains over
/// several episodes (plus the background persistence thread) instead of one
/// multi-second blocking compression of *everything* pending. This is a WORK/time
/// budget — compression cost scales with turns × chunks × layers (~model
/// dependent, not card capacity) — so it is an absolute size rather than a
/// fraction of the card.
const VRAM_COMPRESS_MAX: u64 = 1024 * 1024 * 1024;
/// The rung compresses `want × this` per episode (clamped to the max above), so
/// it overshoots the immediate shortfall a little and coasts rather than
/// re-tripping on the very next wave.
const VRAM_COMPRESS_HYSTERESIS: u64 = 4;
/// Safety cap on the synchronous substrate-offload flush under pressure. The
/// pass migrates hot→warm *before* its cold-disk writes, so the warm copies
/// the eviction needs exist well before this fires — a timeout only clips the
/// tail of the cold-write wait (turns are already evictable) and guards against
/// a wedged persistence thread; it is not the expected path.
const VRAM_OFFLOAD_FLUSH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

impl Scheduler {
    /// Promote up to `MAX_ACTIVE_PREFILLS - active_prefills.len()` newly
    /// submitted PrefillWorks from the FIFO queue into the in-flight
    /// `active_prefills` set. Emits the initial `Prefill` and
    /// `PrefillProgress(0, total)` events so callers see their submission
    /// was picked up.
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
    ///  3. **Compress to free.** Bring forward the float→quant the persistence
    ///     thread would do anyway. A shrink in place rather than a move: the
    ///     turn stays resident and attended-over, and only its float working
    ///     set goes. Cheaper than eviction, which has to be reloaded if the
    ///     turn is re-attended.
    ///  4. **Evacuate.** Flush the pending hot→warm so just-sealed turns have a
    ///     warm copy — only warm-backed turns are evictable — then drop the hot
    ///     copies of the oldest ones. This is §3.8's evict-as-evacuation, and
    ///     it runs through the demotion path the tiering already owns; there is
    ///     no GPU→GPU compaction behind it any more.
    ///
    /// This ordering used to be the VRAM governor's relief ladder, each step a
    /// numbered `Criticality` rung with the governor re-measuring driver
    /// headroom between them to decide whether to climb. The rungs are gone:
    /// against an exact free-region count there is nothing to re-measure and
    /// nothing to arbitrate, so the priority is expressed as call order
    /// (`docs/archived/arena_unification.md` §5).
    ///
    /// Returns `true` if pressure is **still** on afterwards — the caller's
    /// signal to narrow the admission window, which is §3.8's third and last
    /// response. `whence` tags the log line with the calling gate.
    pub(super) fn relieve_vram_pressure(&mut self, whence: &str, phase: VramPhase) -> bool {
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
        let mut compressed = 0usize;
        let mut compress_refused = false;
        let mut flushed = false;
        let mut evicted = crate::substrate::EvictionReport { count: 0, bytes: 0 };

        // Gallery eviction — **this cannot clear the pressure below it**, and is
        // not here to.
        //
        // `evict_lru` drops `PageRun`s, returning pages to the gallery's own
        // `PagePool`. The VRAM behind them is `GalleryArena`'s `storage.slabs`,
        // which is only ever appended to (`add_slab`) and never shrunk, and
        // those slabs come from the CUDA pool rather than the KV reservation.
        // So `region_stats().free` is unchanged by this call and the next
        // `vram_under_pressure_for` is still true — `gallery_freed` counts bytes
        // returned to a free list, not to the card.
        //
        // Gallery growth is bounded by the arena itself now — it evicts to its
        // own ceiling at admission — so this no longer has to be the only limit,
        // and it must not fire merely because KV is tight. It used to: the test
        // was KV pressure alone, which this call cannot clear, so every episode
        // shed belief-scan residency that the next scan rebuilt from the
        // substrate. Now it only runs when the arena is *itself* over its
        // ceiling, which is the one case where evicting is the right answer and
        // the bytes are genuinely reclaimable.
        if self.vram_under_pressure_for(phase) {
            if let Some(arena) = self.gallery_arena.as_ref() {
                let cap = arena.cap_bytes();
                let resident = arena.resident_bytes();
                if resident > cap {
                    gallery_freed = arena.evict_lru((resident - cap).max(want));
                }
            }
        }

        if self.vram_under_pressure_for(phase) {
            // Bound the batch so a large backlog drains over several episodes
            // rather than one multi-second blocking pass over everything
            // pending; the persistence thread is working the same queue.
            let budget = want
                .saturating_mul(VRAM_COMPRESS_HYSTERESIS)
                .min(VRAM_COMPRESS_MAX);
            let pass = self.compress_pending_turns(budget);
            compressed = pass.compressed;
            compress_refused = pass.refused;
            released += self.session.release_empty_arenas().unwrap_or(0);
        }

        if self.vram_under_pressure_for(phase) {
            evicted = self.evict_cold_tail(want);
            if evicted.bytes < want {
                // The blocking flush is only paid when the already-warm turns
                // were not enough: under sustained pressure there are usually
                // plenty of them, and this wait is measured in seconds.
                flushed = super::timed_wait(|| {
                    self.persist_trigger
                        .flush_blocking(VRAM_OFFLOAD_FLUSH_TIMEOUT)
                });
                let more = self.evict_cold_tail(want.saturating_sub(evicted.bytes));
                evicted.count += more.count;
                evicted.bytes += more.bytes;
            }
            released += self.session.release_empty_arenas().unwrap_or(0);
        }

        // **Last resort, and the only one that adds ground rather than
        // recycling it.** Everything above reclaims KV the engine already owns —
        // compress a turn, evict a cold tail, drop an empty arena — and all of
        // it is worth nothing against a workload with nothing reclaimable. A
        // base conversation's sections are not turns, so there is no turn to
        // compress and no tail to evict *on this ladder*, and a section prefill
        // that outgrows its ground stalls with every relief counter reading
        // zero. That is exactly how it failed.
        //
        // Sections are no longer permanently resident, though the rung that
        // reclaims them is not this one: `Substrate::demote_idle_hot` sheds a
        // dormant section on the persistence thread once a durable copy exists
        // (`docs/vram_governor_design.md` §8.1). It is a different cadence and
        // cannot be reached from inside a stalled prefill, so this rung still
        // has nothing to offer that workload.
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
        // **Relief does not buy weight-side ground.** The setpoint is a level
        // of free regions the KV side likes to keep, not a claim that failed,
        // and a claim that does run out buys exactly what it needs on the spot
        // (`request_kv_ground` from the claim path). Asking the weight side for
        // the setpoint shortfall here meant every decode-only wave under the
        // setpoint took 144–240 MiB from the experts — 260 concessions in 441
        // waves of one pool phase, the zone falling from 6.1 to 5.0 GiB and the
        // hit rate through the knee, with nothing having asked for a region.
        let still = self.vram_under_pressure_for(phase);
        let acted = released > 0 || gallery_freed > 0 || compressed > 0 || evicted.count > 0;
        if acted {
            relief_trace::note("sched", "relieve", want, evicted.bytes);
        }
        let (free, setpoint) = self.kv_region_state(phase).unwrap_or((0, 0));
        // INFO when the pass actually shed something — that is a real event.
        // DEBUG otherwise: this runs from several gates every scheduler loop,
        // so an unconditional INFO floods the log under a sustained burst.
        macro_rules! emit {
            ($lvl:ident) => {
                tracing::$lvl!(
                    target: "candle_conversation::scheduler::timing",
                    whence,
                    want_mib = want / (1 << 20),
                    relief_ms = t.elapsed().as_millis() as u64,
                    warm_flushed = flushed,
                    gallery_freed_mib = gallery_freed / (1 << 20),
                    turns_compressed = compressed,
                    compress_refused,
                    turns_evicted = evicted.count,
                    evicted_mib = evicted.bytes / (1 << 20),
                    arenas_released = released,
                    free_regions = free,
                    setpoint_regions = setpoint,
                    relieved = !still,
                    "KV region relief"
                )
            };
        }
        // A refused compression is not an action, but it *is* an event: the rung
        // that shrinks a resident turn in place was asked to run and could not
        // get the ground to run in. Left at DEBUG it reads as `turns_compressed=0`,
        // identical to a pass with nothing to compress — which is how the
        // feedback loop running backwards (compression is what relieves the
        // pressure that refuses it) stayed invisible through the whole wedge.
        if acted || compress_refused {
            emit!(info);
        } else {
            emit!(debug);
        }
        still
    }

    /// What the card can actually deliver to admission right now.
    ///
    /// Free reservation bytes plus reversibly-evictable KV, minus the hot KV the
    /// drain is skipping because it is pinned. The pinned discount is what keeps
    /// the forecast from reading its most optimistic exactly when the hot→warm
    /// drain has stalled: those bytes are counted as evictable but cannot be
    /// reclaimed at any price.
    ///
    /// The first term used to be a contest between three driver-derived
    /// estimates — governor headroom, the pool's reserved-but-free gap, and the
    /// allocator's own `init_free − pool_used − reserve` — clamped to whichever
    /// looked smallest, because each was wrong in a different regime. The worst
    /// was the reuse gap: admission once read 3045 MiB of it while `vram_free`
    /// was 0 and the pool held 15168 of 16375 MiB, admitted six prefills onto
    /// memory WDDM had already spilled, and the run aborted at ~3 tok/s. None of
    /// that survives the reservation. KV comes from regions that were claimed at
    /// startup, so what admission can spend is a count of the free ones, and no
    /// driver reading enters into it.
    ///
    /// Two corrections went with those estimates. One added what registered
    /// relievers claimed they could reversibly free; the other subtracted hot KV
    /// the drain was skipping because it was pinned, which the first had counted
    /// and could not actually reclaim. Both existed because the base number
    /// described *the card*. A free-region count describes what this process has
    /// claimed and not yet spent, so pinned KV is excluded by construction — it
    /// holds live regions — and evictable KV shows up as free regions the moment
    /// the relief pass ahead of admission actually evicts it. Measured, not
    /// forecast, which is why nothing has to be added back or discounted.
    pub(super) fn admit_budget_ceiling(&self) -> u64 {
        // The relief setpoint IS subtracted — those regions are the relief
        // pass's working room, not admission's to spend.
        let Some((free, setpoint)) = self.kv_region_state(VramPhase::Load) else {
            return 0;
        };
        (free.saturating_sub(setpoint) as u64).saturating_mul(region_bytes())
    }

    /// One decode step's worth of tokens — what a decode admission claims.
    const DECODE_CLAIM_TOKENS: usize = 1;

    /// Tokens of generation an admission reserves arena for, up front.
    ///
    /// **This is a lease, not a limit on the answer.** A turn that wants more
    /// runs to the end of its lease, seals what it has into the substrate — the
    /// seal compresses it, so the ground comes back smaller than it went out —
    /// and goes to the back of the queue to be resumed from its sealed prefix.
    /// So the engine's exposure to one conversation is bounded at its prompt
    /// plus this, however long the conversation turns out to be.
    ///
    /// **Reserving it is what makes admission exact.** `claim_kv` calls
    /// `ensure_capacity`, which allocates — so a lease priced here is ground
    /// actually taken, visible to `region_stats`, and not an estimate anyone
    /// can get wrong. Every pricing failure this engine has had came from
    /// predicting a cost instead of taking it: run BV charged a decode step one
    /// token, which opens a block once in thirty-two and is free the rest of the
    /// time, and stood 95 slots open on ground it had never reserved.
    ///
    /// 256 is eight 32-token blocks — wide enough to amortise the seal and the
    /// hot→warm migration that follows it (warm arenas are pageable, so that
    /// copy runs at about half PCIe bandwidth), and short enough that a slot's
    /// resident K/V stays bounded while it decodes. It is a token count rather
    /// than a byte figure, so it carries to the 3090 and the workstation
    /// unchanged: what a block costs follows the model's own geometry.
    pub(super) const DECODE_LEASE_TOKENS: usize = 256;

    /// Parked turns the engine will hold before it stops parking.
    ///
    /// **A park moves ground from the card to the host**, so an unbounded queue
    /// does not relieve pressure, it relocates it — and a daemon sitting on tens
    /// of gigabytes of host RAM with healthy VRAM is this engine's hardest
    /// failure to read. Past this the lease renews instead and the turn keeps
    /// decoding: holding more open turns is the lesser problem, and the turns
    /// already parked are the ones that need the engine back.
    ///
    /// It is a safety valve rather than a tuning knob. Parking should be rare —
    /// it fires only for turns that outrun a whole lease — so reaching this
    /// bound means resumes are not keeping up, which is worth the log line it
    /// produces.
    const PARKED_TURNS_MAX: usize = 32;

    /// Times a parked turn is retried before it is failed to its caller.
    ///
    /// A resume needs ground; the first refusals are ordinary backpressure. But
    /// a turn that can never be restored must not sit at the head of the queue
    /// forever holding a caller that is still waiting on its channel, so the
    /// retries are finite and the last one reports.
    const PARK_RESUME_ATTEMPTS: usize = 8;

    /// Ask the allocator for `add` tokens on `seq`, for real.
    ///
    /// Between forwards, which is where arena creation is allowed: a claim
    /// arriving with a wave's transient tier standing is carved out of ground
    /// that tier occupies. `promote_new_prefills` runs in the gap, so this is
    /// legal here and would not be inside the wave.
    ///
    /// A refusal is not an error — it is the allocator saying this queue has no
    /// more to give right now, which is exactly what the interleave's `false`
    /// means.
    /// Slots admitted and not yet finished — prefilling or decoding alike.
    pub(super) fn active_slots(&self) -> usize {
        self.active_prefills
            .iter()
            .filter(|p| p.error.is_none())
            .count()
            + self.active_decodes.values().filter(|s| !s.finished).count()
    }

    /// Whether an admission decision is due this wave.
    ///
    /// **One definition, two callers, and they must not drift.** The eviction
    /// pass exists to settle the free lists the fill then prices against, so a
    /// wave where one runs and the other does not is either a shed with nothing
    /// to admit or — worse — an admission priced against a device nobody
    /// cleaned, which reads as fuller than it is and refuses work the card could
    /// have taken. [`admit::Ground::settled`] defers to this rather than
    /// restating it.
    ///
    /// Due when nothing is running (there is no completion left to wait for),
    /// when a slot has been reaped since the last offer (ground came back), or
    /// when the last wave ran no forward at all. That last clause is what stops
    /// the fast path waiting on a completion that cannot come: without it run BS
    /// sat at `skipped=true active=32 queued=24` for 98 consecutive waves.
    pub(super) fn admission_due(&self) -> bool {
        self.active_slots() == 0
            || self.completions != self.admit_completions
            || !self.wave_ran_forward
    }

    /// Park every slot whose decode lease has run out.
    ///
    /// The lease is the ground admission actually reserved for this turn's
    /// generation, so reaching the end of it is where the slot gives that ground
    /// back: its K/V goes to the warm tier — compressed on the way, so it holds
    /// less parked than it did resident, and none of it on the card — the slot's
    /// blocks are released, and the turn joins the back of the queue.
    ///
    /// **A lease ending is a completion as far as admission is concerned**, and
    /// `completions` is bumped to say so: ground came back, so the next fill is
    /// due rather than taking the fast path.
    ///
    /// A snapshot or migration that fails leaves the turn decoding on a renewed
    /// lease instead. Losing a turn to a transient tier error would be a far
    /// worse failure than briefly holding more ground than the gate authorised,
    /// and the next expiry tries again.
    fn park_expired_leases(&mut self) {
        let expired: Vec<SequenceId> = self
            .active_decodes
            .iter()
            .filter(|(_, s)| s.lease_expired && !s.finished)
            .map(|(&id, _)| id)
            .collect();
        for slot in expired {
            // **Stop parking rather than park unboundedly.** A park relocates
            // ground from the card to the host; past the bound the queue is
            // already the problem, so the lease renews and the turn keeps
            // decoding instead.
            if self.parked.len() >= Self::PARKED_TURNS_MAX {
                if let Some(s) = self.active_decodes.get_mut(&slot) {
                    s.lease_expired = false;
                    s.lease_left = Self::DECODE_LEASE_TOKENS;
                }
                tracing::debug!(
                    target: "candle_conversation::scheduler::interleave",
                    parked = self.parked.len(),
                    "park queue full — lease renewed instead of parking",
                );
                continue;
            }
            let warm = self
                .session
                .snapshot_sequence_per_layer(slot.0)
                .and_then(|hot| self.session.sealed_to_cpu(&hot));
            let warm = match warm {
                Ok(w) => w,
                Err(e) => {
                    tracing::warn!(
                        target: "candle_conversation::scheduler::interleave",
                        slot = slot.0,
                        "lease park failed, turn keeps decoding: {e}",
                    );
                    if let Some(s) = self.active_decodes.get_mut(&slot) {
                        s.lease_expired = false;
                        s.lease_left = Self::DECODE_LEASE_TOKENS;
                    }
                    continue;
                }
            };
            let Some(mut state) = self.active_decodes.remove(&slot) else {
                continue;
            };
            state.lease_expired = false;
            state.lease_left = Self::DECODE_LEASE_TOKENS;
            // The hot snapshot is dropped by now, so this releases the regions
            // rather than merely unlinking them.
            if let Err(e) = self.session.truncate_sequence_to_blocks(slot.0, 0) {
                tracing::warn!(
                    target: "candle_conversation::scheduler::interleave",
                    slot = slot.0,
                    "parked slot did not release its blocks: {e}",
                );
            }
            tracing::debug!(
                target: "candle_conversation::scheduler::interleave",
                slot = slot.0,
                generated = state.generated_tokens.len(),
                parked_ahead = self.parked.len(),
                "decode lease spent — turn parked to warm",
            );
            self.parked.push_back(ParkedTurn {
                slot,
                warm,
                state,
                resume_failures: 0,
            });
            self.completions = self.completions.saturating_add(1);
        }
    }

    /// Bring the oldest parked turn back onto the slot it left, with a fresh
    /// lease.
    ///
    /// **One per pass, and ahead of new admissions.** A parked turn is work the
    /// engine already took on, and finishing what is open is what frees ground
    /// for what is not — the same reason the queue puts continuations before
    /// first turns. Taking only one keeps that from becoming a convoy that
    /// crowds out every fresh conversation.
    fn resume_parked(&mut self, optimal: u64) {
        if self.parked.is_empty() {
            return;
        }
        // **A resume claims ground, so it answers to the hold like anything
        // else.** It is the one path that brings K/V back onto the card without
        // going through `admit::fill`, and left ungated it walks residency under
        // the line the whole engine is built to defend — quietly, because no
        // gate reported a refusal. It is held to the hold rather than to
        // admission's full floor: this is work already taken on, and finishing
        // it is what frees ground, so it gets the benefit of the margin that a
        // *new* admission does not.
        let zone = interleave::effective_weight_zone_bytes().unwrap_or(u64::MAX);
        if zone <= optimal {
            return;
        }
        let Some(parked) = self.parked.pop_front() else {
            return;
        };
        let ParkedTurn {
            slot,
            warm,
            state,
            resume_failures,
        } = parked;
        let restored = self
            .session
            .sealed_to_gpu(&warm)
            .and_then(|hot| self.session.inject_sealed_at_tail(slot.0, &hot));
        match restored {
            Ok(_) => {
                tracing::debug!(
                    target: "candle_conversation::scheduler::interleave",
                    slot = slot.0,
                    generated = state.generated_tokens.len(),
                    "parked turn resumed on a fresh lease",
                );
                self.active_decodes.insert(slot, state);
            }
            Err(e) if resume_failures + 1 >= Self::PARK_RESUME_ATTEMPTS => {
                // **Report rather than hold.** The caller is still blocked on
                // this turn's channel; a turn that cannot be restored has to
                // fail visibly instead of sitting at the head of the queue with
                // nobody able to tell why nothing is happening.
                tracing::error!(
                    target: "candle_conversation::scheduler::interleave",
                    slot = slot.0,
                    attempts = resume_failures + 1,
                    "parked turn abandoned after repeated resume failures: {e}",
                );
                let _ = state
                    .event_tx
                    .send(TurnEvent::Error(ConversationError::Model(e)));
                let _ = self.session.truncate_sequence_to_blocks(slot.0, 0);
            }
            Err(e) => {
                // Warm is still the only copy, so the turn goes back at the
                // front rather than being dropped — the ground it needs may
                // simply not be there yet.
                tracing::warn!(
                    target: "candle_conversation::scheduler::interleave",
                    slot = slot.0,
                    attempt = resume_failures + 1,
                    "parked turn could not be resumed this wave: {e}",
                );
                self.parked.push_front(ParkedTurn {
                    slot,
                    warm,
                    state,
                    resume_failures: resume_failures + 1,
                });
            }
        }
    }

    fn claim_kv(&self, seq: usize, add: usize) -> bool {
        self.session.ensure_capacity(&[seq], add).is_ok()
    }

    /// Ask the model to make `seq`'s per-sequence state resident for the
    /// coming wave, for real.
    ///
    /// The other half of an admission: a decode row needs its recurrent state
    /// on the device as much as it needs its KV chunk, and on a model whose
    /// state is carved from the same reservation that is the claim that
    /// actually bounds width. A model with no such state answers yes for free.
    /// An error is treated as a refusal — the sequence waits a wave — and is
    /// logged, because "no room" comes back as `Ok(false)` and an `Err` here
    /// is something else.
    fn claim_recurrent(&self, seq: usize) -> bool {
        let offset = self.session.sequence_offset(seq).unwrap_or(0);
        match self.model.admit_recurrent(seq, offset) {
            Ok(admitted) => admitted,
            Err(e) => {
                tracing::debug!(
                    target: "candle_conversation::scheduler::interleave",
                    seq,
                    "recurrent admission errored — treated as a refusal: {e}",
                );
                false
            }
        }
    }

    /// Fold the expert cache's routing since the last fill into the running
    /// hit rate the admission reads. The cache's counters are cumulative;
    /// the difference is this interval's, and a fill that saw no routing
    /// (an idle engine) leaves the average where it was. Smoothed over a few
    /// fills so one wide wave's union does not close the gate by itself.
    fn observe_expert_hit_rate(&mut self) {
        let Some(stats) = self.model.expert_stats() else {
            return;
        };
        let hits = stats.expert_hits.saturating_sub(self.expert_hits_seen);
        let misses = stats.expert_misses.saturating_sub(self.expert_misses_seen);
        self.expert_hits_seen = stats.expert_hits;
        self.expert_misses_seen = stats.expert_misses;
        let routed = hits + misses;
        if routed == 0 {
            return;
        }
        let rate = hits as f64 / routed as f64;
        let smoothed = match self.expert_hit_rate {
            Some(prev) => prev * 0.7 + rate * 0.3,
            None => rate,
        };
        self.expert_hit_rate = Some(smoothed);
        // **The width follows the hit rate.** Under the knee the wave is
        // loading experts rather than using them, so it narrows by one; clear
        // above the knee (a tenth of headroom, so the two sides do not chase
        // each other) it widens back toward the model's ceiling. The width is
        // the one thing that bounds how many stores stand, so this is the whole
        // residency policy: there is no separate gate on a store.
        let knee = self.model.expert_hit_rate_knee();
        // **A resident cache is not bounded by the model's row.** The row is
        // the width the gate's curve peaked at on the card the model was
        // calibrated on — a streaming card. A card that holds every expert
        // reports nearly every routing a hit, and there the curve keeps rising
        // to widths the row never measured; capping it at the row would leave
        // a 72 GB card running ten sequences. So while the smoothed rate stands
        // at or above the model's resident mark, the ceiling is the engine's
        // hard cap instead; when the rate falls back under it, the width comes
        // down one step at a time toward the row.
        let ceiling = if smoothed >= self.model.resident_hit_rate() {
            WAVE_WIDTH_HARD_CAP
        } else {
            self.model.decode_width_target()
        };
        let before = self.wave_width;
        // One step per wave at most: the fill runs more than once a wave, and
        // a step per fill walked the width from eight to two inside one
        // calibration dip and back, the producer's marks following it down.
        // Five seconds between steps: the hit rate answers a width change only
        // after the stores and K/V that width brings have arrived, a few waves
        // later. Two seconds walked the width from 15 to 5 and back inside a
        // pool's first ten minutes, overshooting on both sides of the knee.
        let due = self
            .last_width_adjust
            .is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(5));
        // The dead band above the knee is a twentieth: a tenth left the width
        // parked at four for minutes with the rate at 0.55–0.58, a queue of
        // waiting conversations and the prefill row idle — the rate had
        // recovered from its dip, the width had not.
        // The width never narrows below the point the gate's curve peaks at
        // (four on this card): below it the wave is under-using a zone that
        // is, by construction, wide open, and a run that narrowed to two on a
        // thirty-second dip sat there with twenty-five conversations waiting
        // and the rate back in the dead band.
        let narrowest = 4.min(ceiling);
        if due && (smoothed < knee || self.wave_width > ceiling) {
            self.wave_width = self.wave_width.saturating_sub(1).max(narrowest);
        } else if due && smoothed > knee + 0.05 {
            self.wave_width = (self.wave_width + 1).min(ceiling);
        }
        if self.wave_width != before {
            self.last_width_adjust = Some(std::time::Instant::now());
        }
        if self.wave_width != before {
            tracing::debug!(
                target: "candle_conversation::scheduler::interleave",
                hit_rate = (smoothed * 1000.0).round() / 1000.0,
                width = self.wave_width,
                "wave width follows the expert hit rate",
            );
        }
    }

    pub(super) fn promote_new_prefills(&mut self) {
        // **The tier margin decays as fast as it grew.** A refusal doubles it
        // (`note_tier_refusal`), and nothing brought it back: after one bad
        // minute it stood at its 1 GiB cap for the rest of the run, so every
        // fill priced the tier a gigabyte under the gap, bought that gigabyte
        // from the weight side, placed a tier that did not use it, and the
        // weight side grew back into it between forwards — 764 concessions
        // and 90,017 expert slots evicted in one 45-minute run, every one a
        // reload. One region back per fill: a refusal still costs a wave and
        // widens the margin, and a run that places its tiers reaches the base
        // margin within a minute.
        self.tier_margin_regions = self.tier_margin_regions.saturating_sub(1).max(4);
        self.observe_expert_hit_rate();
        // **Eviction runs before the measurement — always, not only when a
        // setpoint calls it pressure.** Admission prices against the free lists
        // this pass leaves behind, so a pass that is skipped hands `admit::cost`
        // a device that looks fuller than it is, and the gate refuses work the
        // card could have taken. The condition is therefore the same one the
        // fill itself uses: shed whenever an admission decision is due.
        //
        // It runs on completions rather than on every wave because a wave where
        // nothing finished has nothing new to shed and nothing new to admit —
        // that is the fast path, and it is the only reason this is not literally
        // per-wave.
        //
        // **What it sheds is still bounded, deliberately.** The rungs below
        // reclaim against a measured shortfall; they are not asked to strip the
        // device to bare weights. The last rung moves the elastic boundary, and
        // an over-large ask there is what killed a run outright — 5,752 MiB
        // conceded, 1,598 experts evicted, the zone left under its own pinned
        // working set, nothing ran afterwards. Removing the gate changes *when*
        // relief happens; it must not change how hard it pulls.
        //
        // **Surviving pressure does not close admission.** It used to — `room`
        // went to zero while the free-region count sat under the setpoint —
        // and with the relief pass no longer buying weight-side ground that
        // was a KV side that never grew: 13 free of 316 regions, 22 queued, 39
        // of 60 fills admitting no prefill, the hit rate at 0.61 and the width
        // at 8 with nothing to fill it. A claim that runs out of regions buys
        // its ground from the weight side on the spot, and the width follows
        // the hit rate that purchase moves; that is the bound, not this pass.
        // A spent lease hands ground back, so park before anything measures the
        // device — and before the fill picks this wave's decode set, so a turn
        // at the end of its lease does not ride one more forward.
        self.park_expired_leases();
        if self.admission_due() {
            // **Eviction is gated on pressure, not on the admission cadence —
            // and that is a correctness bound, not a policy preference.**
            //
            // The relief ladder compresses sealed turns, replacing a turn's hot
            // copy with its quantized form. Live slots *borrow those very
            // chunks*, Arc-shared, from the projection that assembled them — so
            // compressing a turn a live slot is reading rewrites that slot's
            // slice layout underneath it. A forward caches its position map
            // across layers, and when the layout moves between two of them the
            // map's `(slice_idx, in_blk)` entries address the wrong slices;
            // `prefill_utils` catches it and refuses to launch rather than
            // sending the kernel through a garbage pointer.
            //
            // Running it every wave a slot completed made that constant instead
            // of rare, and the count is unambiguous: `slice layout changed
            // mid-forward` is **0** across runs BP and BV, which gated on
            // pressure, and 243 / 168 / 189 / 273 across CE, CF, CH and CL,
            // which did not. Run CL lost 102 of 354 directories to it and the
            // pass aborted.
            //
            // Running relief *more* is only safe once compression can tell which
            // turns a live slot is borrowing. Until then the setpoint is what
            // keeps the two off each other, and a per-wave shed is a correctness
            // regression dressed as a throughput idea.
            if self.vram_under_pressure() {
                self.relieve_vram_pressure("wave", VramPhase::Load);
            }
            // **Hand back the K/V of conversations that are between turns.**
            // Here, in the shed slot, for two reasons. It is the eviction the
            // compactions below want to run after — chunks have to die before
            // packing has anything to gain — and the fill measures afterwards,
            // so the ground this frees is ground admission can actually spend
            // this pass rather than next.
            //
            // Calling it here is also what makes the admission pass the clock
            // for `IDLE_SLOT_DEMOTE_PASSES`. It ran once per wave, which paced
            // demotion by a quantity that lengthens under load — stretching the
            // grace exactly when ground is scarcest.
            self.demote_idle_slots();
            // Continuations before first turns: a parked turn is already-admitted
            // work, and finishing it is what frees ground for what is queued.
            let hold = interleave::optimal_weight_bytes().unwrap_or(0);
            self.resume_parked(hold);
        }

        // **Continuations before first turns.** A turn on a sequence that
        // already holds K/V is a conversation part-way through its work — an
        // ingest chain between a tool call and its result, a dialogue mid-reply
        // — and it holds that K/V (and, on a recurrent model, its recurrent
        // store) until it finishes. A first turn on an empty sequence holds nothing
        // yet. FIFO across the two starves the former behind the latter: with
        // 77 queued, a chain's next turn waited behind 76 fresh openings while
        // the engine carried twenty decodes for sixteen minutes and no
        // directory completed, because no chain ever reached its next turn.
        // Finishing what is open is also what frees ground for what is not.
        // Stable within each class, so the order a caller submitted in is kept.
        {
            let (mut continuing, mut fresh): (Vec<PrefillWork>, Vec<PrefillWork>) = self
                .prefill_queue
                .drain(..)
                .partition(|w| self.session.sequence_offset(w.sequence_id.0).unwrap_or(0) > 0);
            self.prefill_queue.extend(continuing.drain(..));
            self.prefill_queue.extend(fresh.drain(..));
        }

        // **The fill.** Rows are offered in priority order (`admit::order`),
        // priced against the state the eviction pass just settled
        // (`admit::cost`), and admitted while the price does not reach the
        // weight zone (`admit::gate`). A slot admitted here is held to
        // completion — prefill through its chunks, then the same slot decoding
        // to EOS — and its reaping is what frees the next admission.
        let optimal = interleave::optimal_weight_bytes().unwrap_or(0);
        let (filled, decodes, refused, admitted) = {
            let mut ground = WaveFill::new(self, optimal);
            ground.publish_tier_budget();
            let filled = admit::fill(&mut ground);
            (
                filled,
                ground.decodes_taken,
                ground.decodes_refused,
                ground.prefill_admitted,
            )
        };
        // The offer has been made against this generation of completions; the
        // next pass is a fast path until another slot is reaped — or until a
        // wave runs no forward at all, which clears the flag below and forces a
        // re-offer rather than waiting on a completion that cannot come.
        self.admit_completions = self.completions;
        self.wave_ran_forward = false;
        if self.prefill_queue.is_empty() {
            // No head to be blocked by.
            self.prefill_head_blocked = false;
        }
        self.last_decode_admitted = decodes.last().map(|id| id.0);
        // Traced only when there was something to fill: this runs on every
        // scheduler iteration, and an idle engine would otherwise write
        // ten empty lines a second.
        if !decodes.is_empty()
            || refused > 0
            || filled.prefills > 0
            || !self.prefill_queue.is_empty()
        {
            // **Which band the work actually resolved to.** The band order is
            // only the order it appears to be if the layer lookup succeeds: a
            // slot whose target will not resolve is treated as `High`, and a
            // whole ingest queue reading `High` would put decode back in front
            // of prefill and make the ordering a silent no-op. `unresolved`
            // counts exactly that failure, so it is visible in the run rather
            // than inferred afterwards from a rate that did not move.
            let band_of = |sid: SequenceId| self.decode_layer_priority(sid);
            let mut q_hi = 0usize;
            let mut q_lo = 0usize;
            let mut q_unresolved = 0usize;
            for w in &self.prefill_queue {
                match band_of(w.sequence_id) {
                    Some(DecodePriority::High) => q_hi += 1,
                    Some(_) => q_lo += 1,
                    None => q_unresolved += 1,
                }
            }
            let d_unresolved = decodes.iter().filter(|id| band_of(**id).is_none()).count();
            tracing::debug!(
                target: "candle_conversation::scheduler::interleave",
                optimal_mib = optimal >> 20,
                weights_mib = interleave::weight_zone_bytes().unwrap_or(0) >> 20,
                effective_mib = interleave::effective_weight_zone_bytes().unwrap_or(0) >> 20,
                decodes = decodes.len(),
                decodes_refused = refused,
                prefills = filled.prefills,
                stopped_on_weights = filled.stopped_on_weights,
                skipped = filled.skipped,
                active = admit::Ground::active(&WaveFill::new(self, optimal)),
                expert_hit_rate = self.expert_hit_rate.map_or(-1.0, |r| (r * 1000.0).round() / 1000.0),
                wave_width = self.wave_width,
                queued = self.prefill_queue.len(),
                queued_high = q_hi,
                queued_background = q_lo,
                queued_unresolved = q_unresolved,
                decodes_unresolved = d_unresolved,
                "wave fill",
            );
        }
        self.wave_decode_set = Some(decodes);
        self.wave_decode_starved = refused;
        // The decodes are never yielded now: nothing passes over the queue head
        // for a cheaper item, so there is no head to clear the wave for.
        self.wave_decode_yielded = false;
        if self.prefill_queue.is_empty() {
            return;
        }

        // Remove by descending index so earlier positions stay valid as we take
        // them out of the queue.
        let mut take = admitted;
        take.sort_unstable_by(|a, b| b.cmp(a));
        let mut admitted: Vec<PrefillWork> = take
            .into_iter()
            .filter_map(|i| self.prefill_queue.remove(i))
            .collect();
        // …then restore submission order among the admitted set.
        admitted.reverse();

        for mut work in admitted {
            let total = work.tokens.len();
            // Once per turn, not once per admission — a wave the tier refused
            // puts its unstarted prefills back at the head of the queue and
            // they arrive here again (see `note_tier_refusal` and
            // `PrefillWork::announced`).
            if !work.announced {
                work.announced = true;
                let _ = work
                    .event_tx
                    .send(TurnEvent::Prefill(work.prefill_text.clone()));
                let _ = work.event_tx.send(TurnEvent::PrefillProgress {
                    tokens_done: 0,
                    tokens_total: total,
                });
            }
            let error = if total == 0 {
                Some(ConversationError::Channel(
                    "prefill received zero tokens".into(),
                ))
            } else {
                None
            };
            // No index cut here. Admission is not a unit boundary — it is the
            // moment work leaves the queue, which happens once per unit but says
            // nothing about where that unit's tokens start. The boundary was
            // taken with the unit's K/V anchor (`Scheduler::close_unit_boundary`),
            // and a second cut on this slot would close whatever the unit has
            // already forwarded into a page of its own.
            self.active_prefills.push(ActivePrefill {
                work,
                offset: 0,
                next_projection: 0,
                final_logits: None,
                error,
                prefill_start: None,
            });
        }
    }

    /// Free KV regions right now, and the setpoint for `phase` — the two
    /// numbers every pressure and admission decision is made from.
    ///
    /// "Free" includes regions a standing transient tier has blocked
    /// (`stats.blocked`): every decision made from this pair concerns work
    /// scheduled for a *later* forward, and that forward's phase 0 releases the
    /// tier before any of its claims run. Counting only the tier-capped free
    /// count made every wave's own scratch read as KV pressure from the
    /// scheduler's seat, shedding sequences to relieve ground that was never
    /// occupied.
    ///
    /// `None` before the reservation exists, which the callers read as "no
    /// pressure, nothing to spend": there is no KV on the device yet to be
    /// under pressure about.
    fn kv_region_state(&self, phase: VramPhase) -> Option<(usize, usize)> {
        let stats = self.kv_regions()?;
        Some((
            stats.free + stats.blocked,
            setpoint_regions(phase, stats.total),
        ))
    }

    /// The KV side's region counters, or `None` before the reservation exists.
    fn kv_regions(&self) -> Option<candle_nn::kv_cache::RegionStats> {
        let candle::DeviceLocation::Cuda { gpu_id } = self.device.location() else {
            return None;
        };
        candle_nn::kv_cache::region_stats(gpu_id)
    }

    /// The card's resident capacity C (bytes) — the balloon-measured limit
    /// below which our footprint stays resident (no WDDM paging). Falls back to
    /// the driver's physical total until the balloon has measured C
    /// (`capacity()` is 0 then), so a threshold scaled by it is never a
    /// spurious zero at startup. `None` when unavailable.
    ///
    /// Only the host-side warm-tier thresholds still scale by C; the KV side's
    /// own pressure is counted in regions, not measured against the card.
    fn resident_capacity(&self) -> Option<usize> {
        self.session
            .vram_governor()
            .map(|g| g.capacity() as usize)
            .filter(|&c| c > 0)
            .or_else(|| self.session.vram_free_total().map(|(_, total)| total))
    }

    /// True when the KV side has fewer free regions than the setpoint — the
    /// signal to shed, and failing that to stop admitting.
    ///
    /// This used to be three gates in disjunction: a byte budget derived from
    /// `init_free − pool_used − reserve`, a driver-free floor qualified by how
    /// much the CUDA pool could still absorb by reuse, and a footprint gate on
    /// `pool_reserved` versus a compaction ceiling. Each existed because the
    /// other two were wrong in some regime, and the footprint gate needed a
    /// cooldown and a futility latch on top because a fragmented gap the engine
    /// kept reusing would otherwise report pressure on every scheduler loop.
    ///
    /// None of it survives the reservation. KV comes from regions claimed at
    /// startup, so the question "is there room?" has one exact answer that no
    /// driver reading enters into, and it cannot disagree with itself.
    ///
    /// Phase-independent default (`Load`, the wider setpoint). Prefer
    /// [`vram_under_pressure_for`](Self::vram_under_pressure_for) at call sites
    /// that know their phase.
    pub(super) fn vram_under_pressure(&self) -> bool {
        self.vram_under_pressure_for(VramPhase::Load)
    }

    /// Phase-aware pressure signal — see [`VramPhase`] for why the setpoint
    /// differs by phase.
    pub(super) fn vram_under_pressure_for(&self, phase: VramPhase) -> bool {
        self.kv_region_state(phase)
            .is_some_and(|(free, setpoint)| free < setpoint)
    }

    /// Bytes one relief pass should aim to free: enough to reach the setpoint
    /// plus [`RELIEF_OVERSHOOT_REGIONS`]. `None` when there is no pressure, so
    /// a relief call on a healthy cache costs one counter read.
    fn relief_shortfall_bytes(&self, phase: VramPhase) -> Option<u64> {
        let (free, setpoint) = self.kv_region_state(phase)?;
        if free >= setpoint {
            return None;
        }
        let target = setpoint.saturating_add(RELIEF_OVERSHOOT_REGIONS);
        Some((target.saturating_sub(free) as u64).saturating_mul(region_bytes()))
    }

    /// Admission passes a live slot must go entirely untouched before its
    /// device K/V is demoted.
    ///
    /// **One, and the clock is the admission pass rather than the wave.** The
    /// signal this rides on — `busy`, which spans every phase the engine knows
    /// about including the prefill queue, parked turns and views — already means
    /// "no outstanding work anywhere". The only thing a grace window adds is
    /// cover for the gap between a turn sealing and its successor being queued:
    /// a tool round trip, where the latency is external and the engine has no
    /// predicate for it. One pass is enough for a result already in flight.
    ///
    /// It was four *waves*, justified as matching "a chain between its decode
    /// and its tool result". That inverted the units — a tool round trip is
    /// wall-clock and does not know what a wave is, while a wave is 2–4 s here
    /// and a fraction of that on a card holding its experts, so the protection
    /// varied per machine while the thing protected did not. It also ran
    /// backwards under load: waves lengthen when the engine is busy, stretching
    /// the grace exactly when ground is scarcest.
    ///
    /// Counting admission passes fixes both. A pass happens when something
    /// completed, so a slot ages toward demotion fast under churn and slowly in
    /// a quiet engine — and the shed lands in the same pass that is about to
    /// measure and spend the ground it frees.
    ///
    /// **Short on purpose, because the reload is the mechanism and not the
    /// cost.** A warm→hot lift allocates fresh chunks through the leftmost-
    /// biased pool, so it re-places that KV at the bottom of the span *densely*
    /// — the only thing in the design that repacks the inside of an arena, which
    /// neither compactor can do (they relocate a container and preserve its
    /// holes). Holding a slot back declines that relocation. The two sides are
    /// not comparable: a lift is bounded PCIe on the copy stream, paid once,
    /// while a watermark that never falls narrows `weight_floor − live_end()`
    /// permanently — measured here as the weight zone driven from 10,398 to
    /// 1,417 MiB with the hit rate through the knee.
    const IDLE_SLOT_DEMOTE_PASSES: u32 = 1;

    /// **Give back the device K/V of live conversations that are between
    /// turns.** The one tenant nothing could shed, and the ceiling every
    /// configuration of the feeder eventually hit: a pool holding thirty-odd
    /// open chains had 375 of 400 KV regions live for six decodes, the weight
    /// zone under its floor and the hit rate through the knee, with the engine
    /// working on a handful of sequences and the rest simply *stored*.
    ///
    /// **Why this is safe, and why it costs nothing to rebuild.** A turn's
    /// projection (`apply_projection`) already truncates its slot to zero
    /// blocks and rebuilds the whole prefix from the substrate — every turn,
    /// on every slot. So a slot's device K/V between turns is not state: it is
    /// a cache of what the substrate holds, and the next turn rebuilds it
    /// whether or not this pass drops it. What the pass changes is *when* the
    /// ground comes back: at the moment the conversation stops using it,
    /// rather than when its next turn happens to arrive.
    ///
    /// Dropping the slot's block table is only half of it — the substrate's
    /// hot copy holds the same chunks, which is why `hot = None` alone freed
    /// 8–9 MiB a pass against 5,920 MiB of ingest K/V (§4.11.5). So the pass
    /// drops both: the block table here, and the hot copies through
    /// `evict_hot_to_free` with a keep-list built from the slots it did *not*
    /// demote — a conversation reached by a live fork keeps everything that
    /// fork attends. Only turns that already have a warm copy are evicted, so
    /// this is hot→warm and the reload is a PCIe copy, never a recompute.
    ///
    /// The recurrent state is untouched: the store's own idle lag parks it,
    /// and a demoted slot is not in any wave, so nothing calls the
    /// `offset == 0` reset that would otherwise wipe it. Its next turn
    /// re-fills the slot before the prefill is admitted, so the store comes
    /// back seeded from the parked copy exactly as it does today.
    ///
    /// Returns the number of slots demoted.
    pub(super) fn demote_idle_slots(&mut self) -> usize {
        // **The pass is the clock** — see [`Self::IDLE_SLOT_DEMOTE_PASSES`].
        // Ticked here rather than at the call site so a caller cannot advance
        // the counter without also running the sweep it paces.
        self.admission_passes = self.admission_passes.wrapping_add(1);
        if self.slot_conversations.is_empty() {
            return 0;
        }
        // Everything the engine is touching, in any phase. A slot named here
        // is not idle whatever its counter says, and its counter restarts.
        let mut busy: HashSet<SequenceId> = HashSet::new();
        busy.extend(self.active_decodes.keys().copied());
        busy.extend(self.active_prefills.iter().map(|p| p.work.sequence_id));
        busy.extend(self.active_section_ingests.iter().map(|s| s.sequence_id));
        busy.extend(self.prefill_queue.iter().map(|w| w.sequence_id));
        busy.extend(self.pending_reprojections.iter().copied());
        busy.extend(self.deferred_glue_fires.iter().map(|p| p.parent_id));
        // A parked turn holds no K/V on the card, but its slot is still its own
        // and it is coming back — reaping it as idle would lose the turn.
        busy.extend(self.parked.iter().map(|p| p.slot));
        busy.extend(self.ephemeral_slots.iter().copied());
        for (view, st) in &self.turn_views {
            busy.insert(*view);
            busy.insert(st.parent_id);
        }
        for m in &self.wave_prefill_members {
            let seq_id = match m {
                WaveMember::Prefill { seq_id, .. } | WaveMember::Section { seq_id, .. } => *seq_id,
            };
            busy.insert(SequenceId(seq_id));
        }

        let slots: Vec<(SequenceId, usize)> = self
            .slot_conversations
            .keys()
            .map(|id| (*id, self.session.sequence_offset(id.0).unwrap_or(0)))
            .collect();

        // **Where the K/V actually is, once a wave.** The pass demoting
        // nothing is not evidence that nothing is idle — it is equally
        // consistent with every slot holding K/V being busy, and the two ask
        // for opposite fixes. So the pass reports the split it sees: tokens
        // held by the slots it may take, by the slots the engine is working
        // on, and by slots that hold nothing at all.
        {
            // **Busy first, then blocks.** Classifying on `tokens == 0` before
            // asking whether the engine is touching the slot is what made this
            // census lie: a conversation between turns holds no blocks (every
            // turn's projection truncates the slot) and was counted `empty`, so
            // `idle_slots` read 0 while thousands of MiB of hot turn residence
            // sat behind those very slots. Idle is now about whether the engine
            // is working on the slot; `idle_empty_slots` is the subset holding
            // no blocks, which is a statement about where the bytes are, not
            // about whether there are any.
            let (mut idle_tok, mut busy_tok, mut idle_n, mut busy_n, mut idle_empty_n) =
                (0usize, 0usize, 0usize, 0usize, 0usize);
            for (id, tokens) in slots.iter().copied() {
                if busy.contains(&id) {
                    busy_n += 1;
                    busy_tok += tokens;
                } else {
                    idle_n += 1;
                    idle_tok += tokens;
                    if tokens == 0 {
                        idle_empty_n += 1;
                    }
                }
            }
            tracing::debug!(
                target: "candle_conversation::persistence::tier",
                slots = slots.len(),
                idle_slots = idle_n,
                idle_tokens = idle_tok,
                busy_slots = busy_n,
                busy_tokens = busy_tok,
                idle_empty_slots = idle_empty_n,
                decodes = self.active_decodes.len(),
                prefills = self.active_prefills.len(),
                queued = self.prefill_queue.len(),
                views = self.turn_views.len(),
                "idle demote: slot K/V census",
            );
        }

        let pass = idle_slots_to_demote(&mut self.slot_idle_passes, &slots, &busy);

        // **How long a demotion actually lasted.** The grace window exists to
        // cover one gap — a turn sealing before its successor is queued, i.e. a
        // tool round trip, whose latency is external and has no predicate. Only
        // the distribution of these gaps says whether covering it is worth
        // declining a relocation: clustered inside a pass or two and the window
        // is preventing real thrash, long-tailed and it is pure delay and the
        // grace should go to zero. Reported here rather than inferred later,
        // because the pairing is only knowable at the moment a slot wakes.
        for id in &pass.woke {
            if let Some(at) = self.slot_demoted_at.remove(id) {
                tracing::debug!(
                    target: "candle_conversation::persistence::tier",
                    slot = id.0,
                    // `wrapping_sub`, matching the wrapping counter: a saturating
                    // one would read 0 across the wrap instead of the true gap.
                    out_passes = self.admission_passes.wrapping_sub(at.0),
                    out_ms = at.1.elapsed().as_millis() as u64,
                    "idle demote: a demoted slot was re-admitted",
                );
            }
        }

        let mut demoted: Vec<SequenceId> = Vec::new();
        let mut tokens_released = 0usize;
        // Slots that still held blocks, so the log can separate the two
        // populations this pass now covers: a slot mid-conversation giving its
        // block table back, and one already truncated whose bytes are entirely
        // in the substrate's hot copies. Only the second was ever the large one.
        let mut slots_with_blocks = 0usize;
        for (id, offset) in pass.demote {
            // A slot already at zero blocks has nothing to truncate, but its
            // conversation's hot turn residences are the bytes this pass is
            // actually after — so it goes on the demoted list either way.
            if offset > 0 {
                if let Err(e) = self.session.truncate_sequence_to_blocks(id.0, 0) {
                    tracing::warn!(
                        target: "candle_conversation::persistence::tier",
                        slot = id.0,
                        "idle demote: could not release the slot's blocks: {e}",
                    );
                    continue;
                }
                slots_with_blocks += 1;
            }
            tokens_released += offset;
            self.slot_tokens.remove(&id);
            if let Some(st) = self.slot_projection_state.get_mut(&id) {
                // The working set is the relief path's protect-list; with the
                // blocks gone it protects nothing but the hot copies this pass
                // exists to free. The glue islands and the in-flight user
                // capture are device K/V of a turn that is over.
                st.working_set.sections.clear();
                st.working_set.turns.clear();
                st.glue_islands.clear();
                st.pending_user_part = None;
            }
            self.slot_demoted_at
                .insert(id, (self.admission_passes, std::time::Instant::now()));
            demoted.push(id);
        }
        if demoted.is_empty() {
            return 0;
        }

        // The keep-list is every slot this pass left standing — a conversation
        // reached by a live fork keeps what that fork attends.
        let demoted_set: HashSet<SequenceId> = demoted.iter().copied().collect();
        let mut keep_sections: Vec<SectionId> = Vec::new();
        let mut keep_turns: Vec<TurnKey> = Vec::new();
        for (id, st) in &self.slot_projection_state {
            if demoted_set.contains(id) {
                continue;
            }
            keep_sections.extend(st.working_set.sections.iter().copied());
            keep_turns.extend(st.working_set.turns.iter().copied());
        }
        // **The span before, so the line can show what the pass actually
        // bought.** Freed bytes and released arenas say what left; only
        // `live` — the watermark the weight zone and the wave tier both grow
        // into — says whether the span got tighter, which is the entire purpose.
        // An eviction that frees megabytes without moving `live` means the
        // survivors are scattered, and that is a compaction problem, not an
        // eviction one. Reading it here rather than inferring it later is the
        // difference between the two being distinguishable in a log.
        let live_before = self.kv_regions().map(|s| s.live).unwrap_or(0);
        let mut freed = crate::substrate::EvictionReport { count: 0, bytes: 0 };
        for id in &demoted {
            let Some(conv) = self.slot_conversations.get(id).cloned() else {
                continue;
            };
            let r = conv
                .write()
                .evict_hot_to_free(&keep_sections, &keep_turns, u64::MAX);
            freed.count += r.count;
            freed.bytes += r.bytes;
        }
        let arenas = self.session.release_empty_arenas().unwrap_or(0);
        let (live_after, regions_free) = self
            .kv_regions()
            .map(|s| (s.live, s.free))
            .unwrap_or((0, 0));
        tracing::debug!(
            target: "candle_conversation::persistence::tier",
            slots = demoted.len(),
            slots_without_blocks = demoted.len() - slots_with_blocks,
            tokens_released,
            residences_evicted = freed.count,
            freed_mib = freed.bytes / (1 << 20),
            arenas_released = arenas,
            live_before,
            live_after,
            regions_free,
            "idle demote: gave back the K/V of conversations between turns",
        );
        demoted.len()
    }

    /// Shed least-recently-used hot turn KV to the warm (RAM) tier across the
    /// resident conversations, freeing up to `target_bytes` of pool VRAM.
    /// Oldest-first and reversible (a reselected turn reloads from RAM). Only
    /// turns that already hold a warm copy are evictable, so callers should
    /// first [`PersistenceTrigger::flush_blocking`] to make the just-sealed
    /// turns qualify. The `target_bytes` budget caps total bytes freed, so a
    /// conversation reached via several slots is naturally not over-evicted
    /// (and `evict_hot_to_free` is per-conversation scoped — it can never touch
    /// a parallel conversation's selected working set).
    fn evict_cold_tail(&mut self, target_bytes: u64) -> crate::substrate::EvictionReport {
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
        let mut keep_sections: Vec<SectionId> = Vec::new();
        let mut keep_turns: Vec<TurnKey> = Vec::new();
        for st in self.slot_projection_state.values() {
            keep_sections.extend(st.working_set.sections.iter().copied());
            keep_turns.extend(st.working_set.turns.iter().copied());
        }

        let t = std::time::Instant::now();
        let mut report = crate::substrate::EvictionReport { count: 0, bytes: 0 };
        let mut remaining = target_bytes;
        let convs: Vec<Conversation> = self.slot_conversations.values().cloned().collect();
        for conv in convs {
            if remaining == 0 {
                break;
            }
            let r = conv
                .write()
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

    /// Gentle-early ingest relief, run per-wave and long before the setpoint is
    /// approached. Once the KV side is more than [`ingest_demote_pct`] occupied
    /// (~50 % of its regions), shed the sealed, warm-backed KV of append-only
    /// ingest timelines down to a small rolling hot window
    /// ([`ingest_hot_window`]).
    ///
    /// Zero reload cost: ingest KV is never re-attended until query time, when
    /// it re-elevates warm→hot on demand. So it is the cheapest thing to shed
    /// and it sheds first, which is what keeps a bulk repo ingest from pinning
    /// a whole corpus hot until real pressure forces a much more expensive
    /// eviction of turns that are actually being attended.
    ///
    /// The watermark used to be `pool_used` against a fraction of the card.
    /// That reading no longer describes KV at all — the pool holds the model,
    /// the expert cache and a few scratches, so it sits at a high, flat
    /// fraction of C forever and the gate would fire on every wave regardless
    /// of how much ingest is resident. Occupancy of the KV span is the same
    /// question asked of the right counter.
    pub(super) fn demote_cold_ingest_if_pressured(&mut self) {
        if self.ingest_timelines.is_empty() {
            return;
        }
        let Some(stats) = self.kv_regions() else {
            return;
        };
        // Multiply before dividing. The same expression read `capacity / 100 *
        // pct` when `capacity` was bytes (~1.6e10), where the truncation was
        // invisible; `stats.total` is a region *count* in the hundreds, so
        // dividing first quantises the watermark to whole percent-of-100 steps
        // — and on any span below 100 regions it truncates to **zero**, which
        // the `live <= watermark` early-return below can never satisfy. That
        // turns the gentle-early rung into an unconditional full demote of the
        // ingest tail on every wave.
        let watermark = stats.total * INGEST_DEMOTE_PCT / 100;
        if stats.live <= watermark {
            return;
        }
        let used = stats.live.saturating_mul(region_bytes() as usize);
        let watermark = watermark.saturating_mul(region_bytes() as usize);
        let window = INGEST_HOT_WINDOW;
        // Relieve back to the watermark, no further: `target` bounds the LRU walk
        // so the demote sheds the least-recently-active ingest tail just enough to
        // clear the pressure, never the whole hot working set.
        let target_bytes = used.saturating_sub(watermark) as u64;
        // 1. Shed whatever is already warm-backed — free, no migration.
        let t_demote = std::time::Instant::now();
        let report = self.demote_ingest_once(window, target_bytes);
        // Feed the GUI's phase timeline: the gentle-rung ingest demotion.
        self.wave_stats.add_evict(
            report.bytes,
            report.count as u64,
            t_demote.elapsed().as_millis() as u64,
        );
        // 2. If `used` is still over the watermark, the demote is **warm-starved**:
        //    warm-copy production (the async persistence pass) lags the ingest seal
        //    rate, so the cold backlog is hot-without-warm and not yet demotable.
        //    NUDGE the persistence thread to run its hot→warm drain (non-blocking),
        //    and let the *next* wave's step 1 shed the freshly-warmed backlog. We
        //    deliberately do NOT `flush_blocking` here: this runs per-wave on the
        //    scheduler thread, and under sustained pressure the persist thread is
        //    already mid-pass — a blocking wait would stall the scheduler for the
        //    full timeout while draining nothing sooner. A `fire()` is a no-op when
        //    a pass is already queued, so it never adds latency.
        //    The test is whether step 1 *could* shed what it needed to, which is
        //    `report.bytes` against `target_bytes` — not the CUDA pool. This read
        //    the pool's `used`, which since KV moved to the reservation holds the
        //    model, the expert cache and the scratches: ~6.5 GiB against a
        //    region-derived watermark of ~2.4 GiB, so it was true on every wave
        //    and `nudged` recorded nothing. It is the same trap the doc comment
        //    above this function describes for the other gate.
        let nudged = if report.bytes < target_bytes {
            self.persist_trigger.fire();
            true
        } else {
            false
        };
        if report.count > 0 {
            // Freed hot arenas → release, so their regions return to the free
            // list where the pressure signal can see them.
            let _ = self.session.release_empty_arenas();
            tracing::debug!(
                target: "candle_conversation::scheduler::vram_relief",
                used_mib = used / (1 << 20),
                watermark_mib = watermark / (1 << 20),
                ingest_timelines = self.ingest_timelines.len(),
                turns = report.count,
                freed_mib = report.bytes / (1 << 20),
                window,
                nudged,
                "cold-ingest demote (gentle-early)"
            );
        }
    }

    /// Size the ingest admission window to the **hot→warm drain backlog** — the
    /// leading backpressure signal that keeps `used` off the warm-starved climb
    /// (see the pool-footprint dashboard). The persistence thread publishes its
    /// live backlog via [`PersistenceTrigger::pending_warm_bytes`]; when it
    /// exceeds the target the drain is behind the seal rate, so narrow the AIMD
    /// window (fewer concurrent scopes → lower seal rate → drain catches up);
    /// when it falls below half the target, reopen. `vram_under_pressure` stays
    /// the hard floor beneath this (its per-admission shrinks still fire on a
    /// true VRAM spike). No-op when nothing is ingesting — chat keeps the
    /// per-iteration AIMD recovery in the run loop. Runs at the ~2 s wave
    /// cadence, matching how often the backlog signal refreshes.
    /// Refresh the cached `sysinfo` reading at most once per
    /// [`HOST_RAM_PROBE_INTERVAL`] — never a per-wave syscall — and return the
    /// cached `(available, total)`. `(0, 0)` until the first probe.
    pub(super) fn host_ram_reading(&mut self) -> (u64, u64) {
        let stale = self
            .host_ram_probe
            .map(|(t, _, _)| t.elapsed() >= HOST_RAM_PROBE_INTERVAL)
            .unwrap_or(true);
        if stale {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            self.host_ram_probe = Some((
                std::time::Instant::now(),
                sys.available_memory(),
                sys.total_memory(),
            ));
        }
        self.host_ram_probe
            .map(|(_, a, t)| (a, t))
            .unwrap_or((0, 0))
    }

    /// Whether the warm KV tier has outgrown its host-RAM budget PLUS the drain
    /// pipeline's slack — the condition under which slowing admission actually
    /// helps (less sealing → less hot→warm output). This replaced the absolute
    /// available-RAM floor, which our own resident weights held permanently
    /// true on any machine whose model fills RAM: an untestable condition that
    /// ratcheted the setpoint to the floor against structure, not pressure.
    pub(super) fn warm_over_budget(&mut self) -> bool {
        let (_, total) = self.host_ram_reading();
        if total == 0 {
            return false;
        }
        let budget = candle::vram::host_ram_budget(total);
        let usage = self
            .persist_trigger
            .warm_resident_bytes()
            .saturating_add(self.persist_trigger.pending_warm_bytes());
        usage
            > budget
                .kv_warm_budget_bytes
                .saturating_add(WARM_PIPELINE_SLACK_BYTES)
    }

    /// Under **heavy** hot→warm backlog, block the wave loop on a device sync so
    /// ingest stops racing ahead of the drain. This runs *after* the per-wave
    /// eviction callbacks, so it also drains the primary stream: the persist
    /// pass — now a handful of cross-layer-batched kernel launches — runs
    /// uncontended by ingest forwards instead of interleaving with them on the
    /// shared stream (the contention that inflates each pass on WDDM). A sync
    /// only *adds* ordering, so there is no KV-before-copy hazard. No-op unless
    /// ingesting and the backlog is over [`ingest_sync_ceiling_pct`].
    pub(super) fn sync_if_backlog_critical(&mut self) {
        // **Not gated on there being an ingest.** The backlog this answers is
        // hot KV waiting for a warm copy, and every sealed turn produces some —
        // `demote_idle_slots` can only shed a turn that already has one, so a
        // dialogue-only workload that outruns the drain stalls the eviction that
        // gives regions back, with nothing pushing back on the seals causing it.
        // The ingest guard closed the one backpressure valve for exactly the
        // population the idle demote now sheds.
        let Some(capacity) = self.resident_capacity() else {
            return;
        };
        let ceiling = capacity / 100 * INGEST_SYNC_CEILING_PCT;
        let backlog = self.persist_trigger.pending_warm_bytes() as usize;
        if backlog <= ceiling {
            return;
        }
        let t = std::time::Instant::now();
        super::timed_synchronize(&self.device);
        tracing::debug!(
            target: "candle_conversation::scheduler::vram_relief",
            backlog_mib = backlog / (1 << 20),
            ceiling_mib = ceiling / (1 << 20),
            stall_ms = t.elapsed().as_millis() as u64,
            "heavy-backlog device sync (de-contend drain)"
        );
    }

    /// One pass of LRU-smart cold-ingest demotion across every live conversation,
    /// freeing at most `target_bytes` total (the `remaining` budget threads across
    /// conversations, so the walk stops the moment the watermark is cleared).
    /// `demote_cold_ingest` self-filters to the timelines each conversation owns (a
    /// non-matching id is a no-op), walks that conversation's `hot_lru` oldest-first
    /// so the least-recently-active tail sheds before an active window, and is
    /// idempotent (already-demoted turns have `hot = None` and are skipped). The
    /// global working-set protect-list is passed to every conversation but only
    /// ever matches that conversation's own attended turns — mirrors
    /// [`Self::evict_cold_tail`].
    fn demote_ingest_once(
        &mut self,
        window: usize,
        target_bytes: u64,
    ) -> crate::substrate::EvictionReport {
        // Protect the active working set of every live slot (what in-flight
        // prefills/decodes are attending) — the same union `evict_cold_tail`
        // builds, so an actively-ingesting conversation's gathered turns are never
        // demoted out from under the next projection.
        let mut keep_sections: Vec<SectionId> = Vec::new();
        let mut keep_turns: Vec<TurnKey> = Vec::new();
        for st in self.slot_projection_state.values() {
            keep_sections.extend(st.working_set.sections.iter().copied());
            keep_turns.extend(st.working_set.turns.iter().copied());
        }
        let mut report = crate::substrate::EvictionReport { count: 0, bytes: 0 };
        let mut remaining = target_bytes;
        let convs: Vec<Conversation> = self.slot_conversations.values().cloned().collect();
        for conv in convs {
            if remaining == 0 {
                break;
            }
            let r = conv.write().demote_cold_ingest(
                &self.ingest_timelines,
                &keep_turns,
                &keep_sections,
                window,
                remaining,
            );
            remaining = remaining.saturating_sub(r.bytes);
            report.count += r.count;
            report.bytes += r.bytes;
        }
        report
    }

    /// Compress-to-free: bring forward the quantization of completed, still-
    /// float turns under VRAM pressure. Mirrors the persistence thread's
    /// hot→warm quantize (same [`quantize_sealed_in_place`], same per-
    /// [`ConvCompression`] policy grouping) but installs **only** the quantized
    /// hot — it does not write the warm (RAM) copy.
    ///
    /// This is a deliberate division of labor: the pass runs on the scheduler
    /// thread to reclaim float VRAM *now* — for a turn NOT currently attended,
    /// the source float arenas free the instant the old hot `Arc`s drop under
    /// the write lock (the substrate held the only reference). For a turn the
    /// active decode IS attending over, the block-table GID clones keep the
    /// float chunks alive until the next reprojection rebuilds the table from the
    /// new quant `hot` — so its float reclaim lands one reproject later, still
    /// safe (a live forward never reads freed memory). Meanwhile the persistence
    /// thread still owns the warm/cold DtoH writes on its own tick (the
    /// compressed turns remain in `snapshot_pending_warm`, warm-absent, so it
    /// still picks them up and lands their bytes).
    ///
    /// A net shrink, not a move: the turn stays resident and attended-over, so
    /// there is no reload or hit-rate cost, and no *extra* quality loss — these
    /// turns get quantized on seal regardless; pressure only pulls it earlier.
    /// Turns whose hot is already quant (a prior pass, or persistence, beat us
    /// to them) are skipped via [`sealed_has_compressible_chunk`] so an undrained
    /// warm backlog doesn't re-walk finished turns.
    ///
    /// [`sealed_has_compressible_chunk`]: candle_nn::kv_cache::ChunkedKvBacking::sealed_has_compressible_chunk
    /// Bring forward the quantization of up to `budget_bytes` of completed float
    /// turns (estimated by their float footprint), oldest-conversation-first.
    /// **Bounded** so a large accumulated backlog is drained over several relief
    /// episodes — a few seconds each — rather than one multi-second blocking
    /// compression of *everything* pending (a 697-turn / 23 GiB / 66 s stall was
    /// the symptom). The background persistence thread drains the rest.
    fn compress_pending_turns(&mut self, budget_bytes: u64) -> CompressPass {
        // Need an engine-wide turn policy to compress against; without one turns
        // stay native float (lossless capture) and there is nothing to bring
        // forward.
        let base = match self.session.compression_policy() {
            Some(p) => p,
            None => return CompressPass::default(),
        };
        let n_layers = self.session.num_layers();
        let device = self.session.device().clone();
        let copy_stream = match &device {
            Device::Cuda(d) => d.cuda_stream(),
            _ => return CompressPass::default(),
        };
        // Bound `backings`' immutable borrow of `self.session` to a disjoint
        // field from `self.elevate_pinned_scratch` (the `&mut` below), exactly
        // like `quantize_section_batch`.
        let backings = self.session.backings();

        let convs: Vec<Conversation> = self.slot_conversations.values().cloned().collect();
        let mut compressed = 0usize;
        let mut refused = false;
        // Estimated float bytes queued for compression so far — the bound.
        let mut collected: u64 = 0;
        'convs: for conv in convs {
            if collected >= budget_bytes {
                break; // Budget met — the rest drains next episode / in the background.
            }
            // Snapshot still-float turns (hot present, warm absent) grouped by
            // their per-conversation compression override — as the persistence
            // thread does — under a brief read lock, filtered to those whose hot
            // is still GPU-float so an undrained warm backlog can't make us
            // re-walk already-quant turns. Stop collecting once the byte budget is
            // reached so a big backlog doesn't compress all at once.
            let groups: HashMap<
                Option<ConvCompression>,
                Vec<(ResidenceIndex, Vec<SealedSequence>)>,
            > = {
                let view = conv.read();
                let mut g: HashMap<_, Vec<_>> = HashMap::new();
                for (idx, hot, cc) in view.snapshot_pending_warm() {
                    if hot.len() != n_layers {
                        continue;
                    }
                    // Layer 0 is representative: a turn's layers seal and
                    // compress together, so if layer 0 is still float, all are.
                    if !backings[0].sealed_has_compressible_chunk(&hot[0]) {
                        continue;
                    }
                    collected += sealed_total_bytes(&hot);
                    g.entry(cc).or_default().push((idx, hot));
                    if collected >= budget_bytes {
                        break;
                    }
                }
                g
            };

            for (cc, group) in groups {
                let policy = match effective_turn_policy(Some(&base), cc) {
                    Some(p) => p,
                    None => continue, // lossless capture: nothing to bring forward
                };
                // Per-residence quantized hot accumulator, one SealedSequence per
                // layer, filled positionally across the per-layer batched launches
                // (`quantize_sealed_in_place` returns one output per input in order).
                let mut q_per: Vec<Vec<SealedSequence>> = (0..group.len())
                    .map(|_| Vec::with_capacity(n_layers))
                    .collect();
                let mut ok = vec![true; group.len()];
                for layer in 0..n_layers {
                    let inputs: Vec<&SealedSequence> =
                        group.iter().map(|(_, hot)| &hot[layer]).collect();
                    match quantize_sealed_in_place(
                        &backings[layer],
                        &inputs,
                        &policy,
                        &device,
                        &copy_stream,
                        &mut self.elevate_pinned_scratch,
                    ) {
                        Ok(out) => {
                            for (slot, qi) in out.into_iter().enumerate() {
                                q_per[slot].push(qi);
                            }
                        }
                        Err(e) if is_device_oom(&e) => {
                            // **The pool refused a quantize destination.** This
                            // rung cannot fix that: the ground it needs comes
                            // from the rungs below (evict a cold tail) or from
                            // the boundary (`request_kv_ground`), and both of
                            // them run after this returns. Every remaining group
                            // would be refused for the same reason, so stop the
                            // pass rather than burn a kernel launch per group
                            // rediscovering it.
                            //
                            // Reported, not retried and not waited on. Waiting
                            // here would deadlock: this runs on the scheduler
                            // thread, and the scheduler thread is what would
                            // release the ground — both the rung below and the
                            // next wave's `end_wave_transient` are further down
                            // this same call stack's future.
                            tracing::debug!(
                                "compress_pending_turns: layer {layer} was refused a quantize \
                                 destination, stopping the pass: {e}"
                            );
                            refused = true;
                            ok.fill(false);
                            break;
                        }
                        Err(e) => {
                            tracing::warn!(
                                "compress_pending_turns: layer {layer} quantize failed: {e} (last CUDA kernel: {})",
                                candle::last_cuda_kernel_launch()
                            );
                            ok.fill(false);
                            break;
                        }
                    }
                }
                // Device-wide sync before the swap: the quantize kernels leave the
                // new Q-arenas' K/V writes in flight (including V work that can
                // retire on a stream a primary-stream-only sync misses — the
                // multi-turn V-duplication window), and the very next reproject on
                // THIS thread reads them. Mirrors the persistence thread's
                // post-batch `device.synchronize()`.
                let sync_failed = if let Err(e) = device.synchronize() {
                    tracing::warn!(
                        "compress_pending_turns: device sync failed: {e:?} — skipping this group's installs"
                    );
                    true
                } else {
                    false
                };
                // Leaving **after** the sync, not at the refusal. The layers that
                // quantized before it left kernels in flight writing into
                // `q_per`'s destination arenas, and dropping those handles
                // returns their regions to the pool — so an early exit would
                // hand a region back while a kernel was still writing into it.
                // `ok` is all-false for this group, so the swap below is a no-op
                // for it either way.
                //
                // **Before the sync's own bail-out**, because that one only skips
                // a group: leaving the refusal check behind it means a failed sync
                // resumes the pass and launches quantizes for every remaining
                // group, each of which the pool refuses for the same reason the
                // first one was refused.
                if refused {
                    break 'convs;
                }
                if sync_failed {
                    continue;
                }
                // Atomic swap under one write lock: replace each residence's hot
                // with its quantized form. Dropping the old (float) hot `Vec`s
                // after the lock releases returns the source float chunks' arena
                // slots to the pool — the VRAM this rung exists to reclaim. Warm
                // stays untouched: the persistence thread still owes the DtoH.
                {
                    let mut view = conv.write();
                    for (i, (residence, _float)) in group.into_iter().enumerate() {
                        if !ok[i] || q_per[i].len() != n_layers {
                            continue;
                        }
                        view.replace_section_hot(residence, std::mem::take(&mut q_per[i]));
                        compressed += 1;
                    }
                }
            }
        }
        if compressed > 0 {
            // Wake the persistence thread so it lands the warm/cold copies of the
            // turns we just compressed without waiting for its 5 s tick.
            self.persist_trigger.fire();
        }
        CompressPass {
            compressed,
            refused,
        }
    }

    /// Continuous-fair-wave prefill throttle: how many transformer layers a
    /// background prefill/glue cohort advances **per wave**
    /// (`docs/continuous_fair_waves.md`).
    ///
    /// `budget = ceil(N / R)`, where `R` is the decode-to-prefill airtime ratio
    /// of the interactive work to protect:
    /// - **No foreground decode active** → `R = 1` → `budget = N`: the prefill
    ///   clears every layer in one wave (nothing to shield → full speed).
    /// - **Decode active** → `R` = the max `decode_priority` ratio over the active
    ///   foreground decodes (default `High` when a layer can't be resolved) → the
    ///   prefill creeps `~N/R` layers per wave while decode keeps its experts hot.
    pub(super) fn wave_prefill_layer_budget(&self) -> usize {
        let n = self.model.num_layers().max(1);
        if self.foreground_decode_width() == 0 {
            return n;
        }
        let ratio = self
            .active_decodes
            .keys()
            .filter_map(|sid| self.decode_layer_priority(*sid))
            .map(|p| p.ratio())
            .max()
            .unwrap_or_else(|| crate::projection::DecodePriority::High.ratio());
        n.div_ceil(ratio.max(1) as usize).max(1)
    }

    /// Resolve the `decode_priority` of a decode slot's target layer, or `None`
    /// when the slot's target/timeline isn't resolvable (the caller then defaults
    /// to the protective `High`).
    pub(super) fn decode_layer_priority(
        &self,
        sid: SequenceId,
    ) -> Option<crate::projection::DecodePriority> {
        // A decode runs on a VIEW sequence, but the projection target (which
        // carries the layer's decode_priority) is pinned on the view's PARENT
        // slot. Resolve view → parent first, falling back to the sid itself for a
        // slot that decodes directly (no view).
        let slot = self
            .turn_views
            .get(&sid)
            .map(|v| v.parent_id)
            .unwrap_or(sid);
        let target = self.slot_targets.get(&slot)?;
        let builder = self.timeline_projections.get(&target.timeline)?;
        builder
            .schema()
            .layers
            .iter()
            .find(|l| l.id == target.layer)
            .map(|l| l.decode_priority)
    }

    /// Number of in-flight prefills that still have tokens left to process
    /// and have not errored.
    pub(super) fn prefill_width(&self) -> usize {
        self.active_prefills
            .iter()
            .filter(|p| p.error.is_none() && p.offset < p.work.tokens.len())
            .count()
    }

    /// Number of in-flight section ingests with tokens remaining (not errored).
    pub(super) fn section_ingest_width(&self) -> usize {
        self.active_section_ingests
            .iter()
            .filter(|s| s.error.is_none() && s.offset < s.tokens.len())
            .count()
    }

    /// Build one ragged section-ingest chunk: for each active section, its next
    /// `min(remaining, cap)` tokens, packed until the per-forward token budget.
    /// Returns `(seq_ids, inputs, group_idxs, advances)`, or `None` when nothing
    /// is pending. Shared by the standalone pass and the co-batched decode wave.
    ///
    /// Ragged batch: each section advances by its OWN min(remaining, cap). The
    /// varlen forward packs the heterogeneous lengths flat, so one near-finished
    /// section no longer collapses the whole wave to the batch minimum — the bug
    /// that dragged a 93-wide tool-catalog ingest down to ~1 token/seq/forward.
    ///
    /// Bound the TOTAL tokens to the same per-forward budget a normal prefill
    /// targets (`max_prefill_pass_tokens`). Without this the whole active set
    /// coalesces into one forward: the 93-section tool catalog (~21k tokens)
    /// packed into a single pass whose transient activation spiked VRAM to the
    /// card ceiling and paged. Sections beyond the budget ride the next chunk —
    /// the wave loop pumps until every section seals — so throughput is unchanged
    /// (each forward still fills to the expert-amortization target) while the peak
    /// stays bounded. At least one section is always admitted so the wave makes
    /// progress.
    #[allow(clippy::type_complexity)]
    pub(super) fn build_section_batch(
        &mut self,
    ) -> Option<(Vec<usize>, Vec<Tensor>, Vec<usize>, Vec<usize>)> {
        // Sections already creeping inside the wave group are excluded — their
        // offset isn't advanced until that group's head, so picking them here would
        // ingest the same chunk twice.
        let in_flight = self.wave_group_section_seqs();
        let active: Vec<usize> = (0..self.active_section_ingests.len())
            .filter(|&i| {
                let s = &self.active_section_ingests[i];
                s.error.is_none()
                    && s.offset < s.tokens.len()
                    && !in_flight.contains(&s.sequence_id.0)
            })
            .collect();
        if active.is_empty() {
            return None;
        }
        let cap = self.max_prefill_pass_tokens;
        let mut seq_ids: Vec<usize> = Vec::with_capacity(active.len());
        let mut inputs: Vec<Tensor> = Vec::with_capacity(active.len());
        let mut group_idxs: Vec<usize> = Vec::with_capacity(active.len());
        let mut advances: Vec<usize> = Vec::with_capacity(active.len());
        let mut batch_tokens = 0usize;
        for &i in &active {
            let s = &mut self.active_section_ingests[i];
            let off = s.offset;
            let advance = (s.tokens.len() - off).min(cap);
            // Stop packing once this forward has reached the per-forward budget
            // (but never emit an empty forward).
            if !seq_ids.is_empty() && batch_tokens + advance > cap {
                break;
            }
            let tokens = &s.tokens[off..off + advance];
            match Tensor::new(tokens, &self.device).and_then(|t| t.unsqueeze(0)) {
                Ok(t) => {
                    seq_ids.push(s.sequence_id.0);
                    inputs.push(t);
                    group_idxs.push(i);
                    advances.push(advance);
                    batch_tokens += advance;
                }
                Err(e) => {
                    s.error = Some(ConversationError::Model(e));
                }
            }
        }
        if seq_ids.is_empty() {
            return None;
        }
        Some((seq_ids, inputs, group_idxs, advances))
    }

    /// Commit one section-ingest chunk after its forward (standalone or
    /// co-batched): advance each section by its own `advance`, record its slot
    /// tokens, and bump its offset. Section logits are never used (no decode).
    pub(super) fn complete_section_chunk(&mut self, group_idxs: &[usize], advances: &[usize]) {
        for (&i, &advance) in group_idxs.iter().zip(advances.iter()) {
            let s = &mut self.active_section_ingests[i];
            if let Err(e) = self.session.advance_sequence(s.sequence_id.0, advance) {
                s.error = Some(ConversationError::Model(e));
                continue;
            }
            let seq_id = s.sequence_id;
            let off = s.offset;
            let chunk_tokens = s.tokens[off..off + advance].to_vec();
            super::Scheduler::record_slot_tokens(&mut self.slot_tokens, seq_id, &chunk_tokens);
            s.offset += advance;
        }
    }

    /// Drain completed or errored section ingest entries. Errored entries send
    /// `Err`; finished entries call `finalize_section_ingest` (seal + write)
    /// and send the `SealResult`.
    pub(super) fn finalize_done_section_ingests(&mut self) {
        let mut i = 0;
        while i < self.active_section_ingests.len() {
            let done = {
                let s = &self.active_section_ingests[i];
                s.error.is_some() || s.offset >= s.tokens.len()
            };
            if !done {
                i += 1;
                continue;
            }
            let s = self.active_section_ingests.swap_remove(i);
            if let Some(e) = s.error {
                let _ = s.response_tx.send(Err(e));
                continue;
            }
            let result = self.finalize_section_ingest(
                s.sequence_id,
                s.section_id,
                s.seal_block_from,
                std::sync::Arc::new(s.tokens.to_vec()),
                s.address,
                s.debug_name,
                s.in_collection,
            );
            let _ = s.response_tx.send(result);
            // swap_remove pulled the last element into i; don't increment.
        }
    }

    /// Clear the in-flight continuous-fair-wave prefill group (residual, cursor,
    /// members) so the next wave forms a fresh one.
    pub(super) fn reset_wave_prefill(&mut self) {
        self.wave_prefill_residual = None;
        self.wave_prefill_cursor = 0;
        self.wave_prefill_members.clear();
    }

    /// Set of section-ingest `seq_id`s currently in flight in the wave group, so
    /// the standalone section pass and a fresh group formation don't double-admit
    /// a chunk that is already creeping (its offset isn't advanced until the head).
    pub(super) fn wave_group_section_seqs(&self) -> std::collections::HashSet<usize> {
        self.wave_prefill_members
            .iter()
            .filter_map(|m| match m {
                WaveMember::Section { seq_id, .. } => Some(*seq_id),
                WaveMember::Prefill { .. } => None,
            })
            .collect()
    }

    /// Form a FRESH wave group into `wave_prefill_members`: the ready dialogue
    /// prefills, each advancing the chunk of its tokens the wave has room for,
    /// plus — when `include_sections` and at least one prefill is present —
    /// section chunks bounded by the per-forward token cap. Section chunks join
    /// only alongside a cohort (so they co-batch a creep that is happening anyway);
    /// with no cohort the caller uses the faster full-sweep section path instead.
    /// Members are ordered prefills-then-sections and this order is then fixed for
    /// the group's life (the held residual depends on a stable input order).
    ///
    /// `prefill_rows` is what the wave's transient tier holds beside the rows
    /// already at its head — the fill priced its admissions against the same
    /// tier budget, one chunk per item, so this hands out that room in admission
    /// order: each prefill takes the smaller of its remaining tokens, the
    /// per-forward chunk, and the rows still unassigned. A prefill that would
    /// get less than its least chunk ([`PREFILL_MIN_ADVANCE`], or all it has
    /// left) waits for the next group rather than riding this one for a handful
    /// of rows — the decodes at the head run meanwhile and free the ground.
    /// When the wave has no head at all (`alone`), the first prefill takes its
    /// least chunk whatever the rows say: nothing else in the wave can make the
    /// room, so the placement is the judge of that chunk, and a refusal it keeps
    /// giving ends in [`Self::note_tier_refusal`] rather than in a wave that
    /// never advances.
    fn form_wave_group(&mut self, include_sections: bool, prefill_rows: usize, alone: bool) {
        let cap = self.max_prefill_pass_tokens.max(1);
        let mut rows_left = prefill_rows;
        // ── One adapter per wave ─────────────────────────────────────────────
        //
        // Same rule the decode cohort follows, applied where the prefill group
        // is formed: the projections run once over every row, so a group carries
        // one adapter or none. The first ready prefill sets it and the rest wait
        // for a group of their own — they are still active, so nothing is lost,
        // and successive groups drain each adapter's queue in turn.
        //
        // Sections are filtered by the same key below rather than after the
        // fact: a section chunk is an ordinary row of this forward, and an
        // unadapted ingest riding an adapted group would be prefilled through
        // the wrong projections and its KV written that way permanently.
        let group_adapter: Option<String> = self
            .active_prefills
            .iter()
            .find(|p| {
                p.error.is_none()
                    && !prefill_done(p.final_logits.is_some(), p.offset, p.work.tokens.len())
            })
            .and_then(|p| self.session.sequence_adapter(p.work.sequence_id.0))
            .map(|s| s.to_owned());
        let mut members: Vec<WaveMember> = Vec::new();
        for i in 0..self.active_prefills.len() {
            let p = &self.active_prefills[i];
            // The same predicate the drain uses — see `prefill_done`. A slot
            // this skips must be one the drain will take, or it is stranded.
            if p.error.is_some()
                || prefill_done(p.final_logits.is_some(), p.offset, p.work.tokens.len())
            {
                continue;
            }
            if self.session.sequence_adapter(p.work.sequence_id.0) != group_adapter.as_deref() {
                continue;
            }
            let remaining = p.work.tokens.len() - p.offset;
            let least = remaining.min(PREFILL_MIN_ADVANCE);
            if rows_left < least {
                if members.is_empty() && alone {
                    members.push(WaveMember::Prefill {
                        seq_id: p.work.sequence_id.0,
                        advance: least,
                    });
                }
                break;
            }
            let advance = remaining.min(cap).min(rows_left);
            members.push(WaveMember::Prefill {
                seq_id: p.work.sequence_id.0,
                advance,
            });
            rows_left -= advance;
        }
        if include_sections && !members.is_empty() {
            let mut sec_tokens = 0usize;
            for i in 0..self.active_section_ingests.len() {
                let s = &self.active_section_ingests[i];
                if s.error.is_some() || s.offset >= s.tokens.len() {
                    continue;
                }
                if self.session.sequence_adapter(s.sequence_id.0) != group_adapter.as_deref() {
                    continue;
                }
                let remaining = s.tokens.len() - s.offset;
                // **A section row costs the tier exactly what a dialogue row
                // costs.** Both put their tokens through the same forward and
                // both write K/V into the same wave transient tier, so they
                // draw from one budget: sections that took their rows from a
                // second, private `cap` composed waves the tier had not been
                // priced for, and the placement refused them whole — the fill
                // above having already stopped at the row it could afford.
                let least = remaining.min(PREFILL_MIN_ADVANCE);
                if rows_left < least {
                    break;
                }
                let advance = remaining.min(cap).min(rows_left);
                // Bound the section contribution to the per-forward token budget
                // (at least one always admitted); the rest ride the next group.
                if sec_tokens > 0 && sec_tokens + advance > cap {
                    break;
                }
                members.push(WaveMember::Section {
                    seq_id: s.sequence_id.0,
                    advance,
                });
                sec_tokens += advance;
                rows_left -= advance;
            }
        }
        self.wave_prefill_members = members;
    }

    /// Resume the held wave group: rebuild each member's `(seq_id, input tensor)`
    /// from its live backing (both kinds: the stable `[offset, offset+advance)`
    /// chunk of their tokens), dropping members that errored/completed.
    /// Returns the kept members (aligned with `seq_ids`/`inputs`) plus the
    /// `active_prefills` positions of the prefill members (for OOM/error routing).
    #[allow(clippy::type_complexity)]
    fn build_wave_group_inputs(
        &mut self,
    ) -> (Vec<WaveMember>, Vec<usize>, Vec<Tensor>, Vec<usize>) {
        let members = self.wave_prefill_members.clone();
        let mut kept: Vec<WaveMember> = Vec::with_capacity(members.len());
        let mut seq_ids: Vec<usize> = Vec::with_capacity(members.len());
        let mut inputs: Vec<Tensor> = Vec::with_capacity(members.len());
        let mut prefill_gidxs: Vec<usize> = Vec::new();
        for m in members {
            match m {
                WaveMember::Prefill { seq_id, advance } => {
                    let Some(i) = self
                        .active_prefills
                        .iter()
                        .position(|p| p.work.sequence_id.0 == seq_id)
                    else {
                        continue;
                    };
                    if self.active_prefills[i].error.is_some()
                        || self.active_prefills[i].final_logits.is_some()
                    {
                        continue;
                    }
                    if self.active_prefills[i].prefill_start.is_none() {
                        self.active_prefills[i].prefill_start = Some(Instant::now());
                    }
                    let off = self.active_prefills[i].offset;
                    let end = (off + advance).min(self.active_prefills[i].work.tokens.len());
                    if end <= off {
                        continue;
                    }
                    let toks: Vec<u32> = self.active_prefills[i].work.tokens[off..end].to_vec();
                    match Tensor::new(toks.as_slice(), &self.device).and_then(|t| t.unsqueeze(0)) {
                        Ok(t) => {
                            kept.push(m);
                            seq_ids.push(seq_id);
                            inputs.push(t);
                            prefill_gidxs.push(i);
                        }
                        Err(e) => self.active_prefills[i].error = Some(ConversationError::Model(e)),
                    }
                }
                WaveMember::Section { seq_id, advance } => {
                    let Some(i) = self
                        .active_section_ingests
                        .iter()
                        .position(|s| s.sequence_id.0 == seq_id)
                    else {
                        continue;
                    };
                    if self.active_section_ingests[i].error.is_some() {
                        continue;
                    }
                    let off = self.active_section_ingests[i].offset;
                    let end = (off + advance).min(self.active_section_ingests[i].tokens.len());
                    let toks: Vec<u32> = self.active_section_ingests[i].tokens[off..end].to_vec();
                    match Tensor::new(toks.as_slice(), &self.device).and_then(|t| t.unsqueeze(0)) {
                        Ok(t) => {
                            kept.push(m);
                            seq_ids.push(seq_id);
                            inputs.push(t);
                        }
                        Err(e) => {
                            self.active_section_ingests[i].error = Some(ConversationError::Model(e))
                        }
                    }
                }
            }
        }
        (kept, seq_ids, inputs, prefill_gidxs)
    }

    /// Finish a wave group that reached the final layer: `members`/`member_logits`
    /// are aligned in caller order. Prefill members commit their chunk — advance
    /// the sequence, record the slot tokens, emit progress — and the member whose
    /// chunk reaches the end of its tokens also emits its staged projections and
    /// records `final_logits` for promotion to decode; one with tokens left stays
    /// active and rides the next group. Section members advance their chunk +
    /// record slot tokens (sealed later by `finalize_done_section_ingests`).
    /// Clears the group.
    fn complete_wave_group(&mut self, members: &[WaveMember], member_logits: &[Tensor]) {
        for (k, m) in members.iter().enumerate() {
            match *m {
                WaveMember::Prefill {
                    seq_id: sid,
                    advance,
                } => {
                    let Some(i) = self
                        .active_prefills
                        .iter()
                        .position(|p| p.work.sequence_id.0 == sid)
                    else {
                        continue;
                    };
                    let total = self.active_prefills[i].work.tokens.len();
                    let seq_id = self.active_prefills[i].work.sequence_id;
                    let off = self.active_prefills[i].offset;
                    let end = (off + advance).min(total);
                    if let Err(e) = self.session.advance_sequence(seq_id.0, end - off) {
                        self.active_prefills[i].error = Some(ConversationError::Model(e));
                        continue;
                    }
                    let chunk_tokens: Vec<u32> =
                        self.active_prefills[i].work.tokens[off..end].to_vec();
                    super::Scheduler::record_slot_tokens(
                        &mut self.slot_tokens,
                        seq_id,
                        &chunk_tokens,
                    );
                    self.active_prefills[i].offset = end;
                    let _ =
                        self.active_prefills[i]
                            .work
                            .event_tx
                            .send(TurnEvent::PrefillProgress {
                                tokens_done: end,
                                tokens_total: total,
                            });
                    if end < total {
                        // More chunks to come: the logits of a mid-turn chunk
                        // are not the turn's first-token distribution.
                        continue;
                    }
                    // Staged calibration prefill: every segment's pinned
                    // projection is emitted here, at the completion of the
                    // final chunk, in segment order.
                    if let Some(comp) = self.active_prefills[i].work.staged_composition.clone() {
                        let gen_start = self.active_prefills[i].work.assistant_content_start;
                        let offs = self.active_prefills[i].work.projection_offsets.clone();
                        for seg in 0..offs.len() {
                            let prev_off = if seg == 0 { gen_start } else { offs[seg - 1] };
                            let mut ev = comp.clone();
                            ev.start_token = prev_off.saturating_sub(gen_start);
                            let _ = self.active_prefills[i]
                                .work
                                .event_tx
                                .send(TurnEvent::Projection(ev));
                        }
                        self.active_prefills[i].next_projection = offs.len();
                    }
                    if let Some(l) = member_logits.get(k) {
                        // DEEP-copy the final-logits row at capture. `Tensor::clone`
                        // is shallow (shared storage), and this tensor is HELD until
                        // the once-per-wave `promote_finished_prefills_to_decodes`
                        // samples the turn's FIRST token from it — up to a whole
                        // decode quantum later. The wave's forward path reuses its
                        // output buffers, so by promotion time the shared storage
                        // holds a LATER step's logits for some other slot: the first
                        // token gets sampled from a foreign distribution, and a
                        // greedy summary anchors on it and coherently continues in
                        // whatever language that row suggests (the stored CJK drift,
                        // 0.007%→0.135% at 42553ca3, amplified later by longer
                        // quanta). A real copy makes the captured row immutable —
                        // one ~vocab-sized row per completed prefill, negligible.
                        let owned = l.copy().unwrap_or_else(|_| l.clone());
                        self.active_prefills[i].final_logits = Some(owned);
                    }
                }
                WaveMember::Section {
                    seq_id: sid,
                    advance,
                } => {
                    let Some(i) = self
                        .active_section_ingests
                        .iter()
                        .position(|s| s.sequence_id.0 == sid)
                    else {
                        continue;
                    };
                    if let Err(e) = self.session.advance_sequence(sid, advance) {
                        self.active_section_ingests[i].error = Some(ConversationError::Model(e));
                        continue;
                    }
                    let seq_id = self.active_section_ingests[i].sequence_id;
                    let off = self.active_section_ingests[i].offset;
                    let end = (off + advance).min(self.active_section_ingests[i].tokens.len());
                    let chunk_tokens = self.active_section_ingests[i].tokens[off..end].to_vec();
                    super::Scheduler::record_slot_tokens(
                        &mut self.slot_tokens,
                        seq_id,
                        &chunk_tokens,
                    );
                    self.active_section_ingests[i].offset = end;
                }
            }
        }
        self.reset_wave_prefill();
    }

    /// Route a wave-group forward failure. On device-OOM, requeue the prefill
    /// members' scope prefills ([`Self::handle_prefill_oom`]) — section members and
    /// dialogue turns just retry next wave once the group is dropped. On any other
    /// error, surface it on each member's backing entry. Always resets the group.
    fn fail_wave_group(
        &mut self,
        members: &[WaveMember],
        prefill_gidxs: &[usize],
        err: &candle::Error,
    ) {
        if candle_nn::kv_cache::is_device_oom(err) {
            self.handle_prefill_oom(prefill_gidxs, err);
        } else if is_tier_refusal(err) {
            // The wave never ran: too wide for the placement, not a failure of
            // anything in it. Requeue and compose it narrower.
            self.note_tier_refusal(err);
        } else {
            let msg = format!("wave group forward failed: {err}");
            for m in members {
                match *m {
                    WaveMember::Prefill { seq_id, .. } => {
                        if let Some(i) = self
                            .active_prefills
                            .iter()
                            .position(|p| p.work.sequence_id.0 == seq_id)
                        {
                            self.active_prefills[i].error =
                                Some(ConversationError::Channel(msg.clone()));
                        }
                    }
                    WaveMember::Section { seq_id, .. } => {
                        if let Some(i) = self
                            .active_section_ingests
                            .iter()
                            .position(|s| s.sequence_id.0 == seq_id)
                        {
                            self.active_section_ingests[i].error =
                                Some(ConversationError::Channel(msg.clone()));
                        }
                    }
                }
            }
        }
        self.reset_wave_prefill();
    }

    /// Consume this wave's deferred gap-fill plans into a co-batchable glue group
    /// `(parent slot ids, glue-token input tensors, per-slot scatter descriptors)`.
    ///
    /// Deferred glue is ingest / compression gap-fill — a pure K/V scatter whose
    /// content prefills through a *separate* unit later (`apply_segments`), so it
    /// has no same-wave, same-slot consumer and can ride the wave as a full-sweep
    /// member alongside decode rather than a separate drain forward. `mem::take`
    /// consumes it once; later decode steps this wave see an empty queue. Returns
    /// `None` when nothing was deferred (or every plan was empty).
    fn take_wave_glue(&mut self) -> Option<(Vec<usize>, Vec<Tensor>, Vec<PendingGlue>)> {
        if self.deferred_glue_fires.is_empty() {
            return None;
        }
        let plans = std::mem::take(&mut self.deferred_glue_fires);
        let mut ids: Vec<usize> = Vec::with_capacity(plans.len());
        let mut inputs: Vec<Tensor> = Vec::with_capacity(plans.len());
        let mut pending: Vec<PendingGlue> = Vec::with_capacity(plans.len());
        for p in &plans {
            if p.glue_tokens.is_empty() {
                continue;
            }
            let input = match Tensor::new(p.glue_tokens.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
            {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!("wave glue input build failed: {e}");
                    continue;
                }
            };
            ids.push(p.parent_id.0);
            inputs.push(input);
            pending.push(PendingGlue {
                write_slice: p.glue_write_slice.clone(),
                write_in_blk: p.glue_write_in_blk.clone(),
                fwd_ahead: p.fwd_ahead.clone(),
            });
        }
        if ids.is_empty() {
            None
        } else {
            Some((ids, inputs, pending))
        }
    }

    /// Reconcile each slot's logical offset with its physical backing length —
    /// the wave-boundary invariant every member must satisfy: the varlen
    /// metadata (`cu_seqlens` / `kv_lens`, built from `session.offset`) and the
    /// slot headers (built from the live block table) describe the SAME slot,
    /// and the attention kernels resolve every `[0, kv_len)` position through
    /// the table. Any divergence sends the kernel past the slot's staged state
    /// into neighboring uploads (garbage slice indices → wild record pointers
    /// → CUDA_ERROR_ILLEGAL_ADDRESS, or silent cross-slot attention reads).
    ///
    /// Two producers, one per direction:
    /// - backing > offset: the co-batched glue scatter reserved gap chunk
    ///   space the unified wave didn't reflect in the slot's logical offset.
    ///   Left as-is, the NEXT prefill computes its write region from the
    ///   stale, shorter offset and clobbers the occupied `[offset, backing)`
    ///   span. Advance the offset up to the backing. (Previously a hard
    ///   assert that aborted the whole wave — the crash root at 42553ca3.)
    /// - offset > backing: a projection injected FEWER tokens than the
    ///   planner counted (`select-promote` drops sections it cannot lift to
    ///   hot under VRAM pressure), leaving the offset counting KV that never
    ///   landed. Clamp the offset down to the backing — positions are
    ///   slot-relative (slice ropes), so the clamped value is also the
    ///   correct RoPE base for the new tokens.
    fn reconcile_wave_offsets(&mut self, ids: &[usize]) -> candle::Result<()> {
        for &id in ids {
            let session_off = self.session.sequence_offset(id).unwrap_or(0);
            // Physical ground truth: the token count the live block table
            // actually covers (the same walk the slot-header build performs).
            // NOT `current_seq_len` — that is the write cursor and reads 0 for
            // freshly injected slots whose tables already hold sealed tokens.
            let backing_len = self
                .session
                .sequence_backing_tokens(id)
                .unwrap_or(session_off);
            if backing_len > session_off {
                self.session
                    .advance_sequence(id, backing_len - session_off)
                    .map_err(|e| candle::Error::Msg(format!("reconcile_wave_offsets: {e}")))?;
                tracing::debug!(
                    slot = id,
                    from = session_off,
                    to = backing_len,
                    "slot offset reconciled up to backing length"
                );
            } else if backing_len < session_off {
                self.session
                    .set_sequence_offset(id, backing_len)
                    .map_err(|e| candle::Error::Msg(format!("reconcile_wave_offsets: {e}")))?;
                tracing::warn!(
                    slot = id,
                    offset = session_off,
                    backing = backing_len,
                    "slot offset AHEAD of backing — clamped down (projection dropped \
                     sections it could not lift; kv metadata must describe the \
                     physical backing)"
                );
            }
        }
        Ok(())
    }

    /// Concatenate the present residual parts along the token dim (1) in the given
    /// caller order, skipping `None` parts. Returns `None` when all are absent.
    fn cat_caller_residual(parts: &[Option<&Tensor>]) -> candle::Result<Option<Tensor>> {
        let present: Vec<&Tensor> = parts.iter().filter_map(|p| *p).collect();
        match present.len() {
            0 => Ok(None),
            1 => Ok(Some(present[0].clone())),
            _ => Ok(Some(Tensor::cat(&present, 1)?)),
        }
    }

    /// The unified continuous-fair-wave step (`docs/continuous_fair_waves.md`): ONE
    /// forward folding every class of work through the shared grouped GEMM so one
    /// expert load per layer serves them all — the whole point on the streaming box.
    ///
    /// Two kinds of member co-batch here:
    /// - **Full-sweep** — decode (1 token/seq) and glue (deferred gap-fill scatter).
    ///   Both traverse all `N` layers every wave.
    /// - **Creep** — the wave group: dialogue prefills plus section-ingest chunks.
    ///   The group shares the GEMM only in `[cursor, win_end)`, its inter-layer
    ///   residual held across waves so the full-sweep members overtake it.
    ///
    /// So the sweep splits into up to THREE segments — `[0, cursor)` and
    /// `[win_end, N)` carry only the full-sweep members, `[cursor, win_end)` adds
    /// the creep. `forward_wave` returns the residual in CALLER order
    /// `[decode | creep | glue]`, so the segment boundaries split it by contiguous
    /// group: the creep is held WHOLE, the full-sweep members `[decode | glue]`
    /// continue. At the head, per-sequence logits are `[decode | creep]` (glue
    /// logits, if present, trail and are discarded): prefills promote, sections seal.
    ///
    /// With no creep group, all members are full-sweep: one `[0, N)` forward folding
    /// decode + a standalone section chunk + glue. The glue is a side effect only —
    /// its logits discarded and it must not advance its slot (asserted after).
    ///
    /// Called per decode step; the cohort/section/glue fold in on the first step
    /// (`wave_cohort_advanced` / `wave_section_advanced` guards, `take_wave_glue`
    /// drains once), the rest are plain decode.
    pub(super) fn decode_forward_cobatched(
        &mut self,
        decode_seqs: &[usize],
        decode_inputs: &[Tensor],
        verify_seqs: &[usize],
        verify_inputs: &[Tensor],
    ) -> candle::Result<Vec<Tensor>> {
        let n = self.model.num_layers().max(1);
        let n_dec = decode_seqs.len();
        let none_seqs: [usize; 0] = [];
        let none_inputs: [Tensor; 0] = [];
        // A speculative step's verify blocks. They are multi-token, so they take
        // the PREFILL slot rather than the decode slot — but they are
        // **full-sweep** members like decode, not creep: their logits are read
        // by the accept walk in this same step, so a block held mid-sweep would
        // stall the decode it exists to accelerate. They therefore ride the
        // prefill slot in EVERY segment, and the creep joins them only inside
        // its window. Empty on an ordinary decode wave, which collapses all of
        // this back to what it was.
        let verify_tok: usize = verify_inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(0))
            .sum();
        // Rows ahead of the creep in caller order, and the logits prefix the
        // caller gets back: `[decode | verify]`.
        let head_rows = n_dec + verify_tok;
        // Wave-step wall-clock, shared across the co-batched classes so the prefill
        // and section throughput panels reflect the CONCURRENT rate (they ride
        // decode's sweep in one forward) rather than reading zero.
        let t_wave = Instant::now();

        // Fold this wave's deferred glue in as a full-sweep member co-batched with
        // decode (see `take_wave_glue`). A slot that decodes this wave is never
        // also a glue member — `take_active_decode_batch` excludes slots with a
        // pending deferred glue fire (they reproject this wave and resume decode
        // next), so the two groups are disjoint and the assembled context list
        // never lists a slot twice.
        let glue = self.take_wave_glue();
        let (glue_seqs, glue_inputs): (&[usize], &[Tensor]) = match &glue {
            Some((ids, ins, _)) => (ids.as_slice(), ins.as_slice()),
            None => (&none_seqs, &none_inputs),
        };
        let glue_pending: Option<&Vec<PendingGlue>> = glue.as_ref().map(|(_, _, p)| p);
        let has_glue = !glue_seqs.is_empty();
        let glue_tok: usize = glue_inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(0))
            .sum();
        // A "full-sweep" wave carries decode and/or glue across all N layers; it
        // drives segments 1 and 3. With neither, only the creep runs (seg 2).
        let has_fullsweep = n_dec > 0 || has_glue;

        let budget = self.wave_prefill_layer_budget();
        // Not `let`: a residual/group mismatch below restarts the creep at layer 0
        // in place (see the check after `creep_tok`), which moves both.
        let mut cursor = self.wave_prefill_cursor;
        let mut win_end = (cursor + budget).min(n);

        // Form/resume the creep group (dialogue prefills + section chunks) unless it
        // was already advanced this wave. A fresh group folds section chunks in to
        // co-batch the creep (`form_wave_group(true)`), unless the standalone
        // section pass already ran this wave (no decode present).
        let (members, seq_ids, inputs, prefill_gidxs) = if !self.wave_cohort_advanced {
            if cursor == 0 && self.wave_prefill_residual.is_none() {
                // The prefill rows the tier holds beside this wave's head, at
                // the budget the fill set on the session for exactly this wave.
                let prefill_rows = self.model.prefill_width_cap(
                    self.session.activation_dtype(),
                    head_rows,
                    self.session.tier_budget_bytes(),
                );
                self.form_wave_group(!self.wave_section_advanced, prefill_rows, head_rows == 0);
            }
            self.build_wave_group_inputs()
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        // No creep group → one full-sweep [0, N) forward folding decode + a
        // standalone section chunk (if pending) + glue. All full-sweep, no residual
        // to hold; logits `[decode | section]` split at n_dec (glue logits, if any,
        // trail and are discarded).
        if seq_ids.is_empty() {
            let section = if !self.wave_cohort_advanced && !self.wave_section_advanced {
                if self.vram_under_pressure() {
                    self.relieve_vram_pressure("section", VramPhase::Load);
                }
                self.build_section_batch()
            } else {
                None
            };
            let (sec_seqs, sec_inputs, sec_gidx, sec_adv) = match section {
                Some((s, i, g, a)) => (s, i, g, a),
                None => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            };
            if sec_seqs.is_empty() && !has_fullsweep && verify_seqs.is_empty() {
                // Nothing to run: no creep, no section, no decode, no glue, no
                // verify blocks.
                return Ok(Vec::new());
            }
            if !sec_seqs.is_empty() {
                self.wave_section_advanced = true;
            }
            if let Some(p) = glue_pending {
                self.session.set_pending_glue(p.clone());
            }
            // Verify blocks lead the prefill slot so `[decode | verify]` stays
            // the logits prefix regardless of what else joined.
            let pre_seqs: Vec<usize> = verify_seqs.iter().chain(&sec_seqs).copied().collect();
            let pre_inputs: Vec<Tensor> =
                verify_inputs.iter().chain(&sec_inputs).cloned().collect();
            let out = self.model.forward_wave(
                &mut self.session,
                decode_seqs,
                decode_inputs,
                &pre_seqs,
                &pre_inputs,
                glue_seqs,
                glue_inputs,
                0,
                n,
                None,
            )?;
            if has_glue {
                self.reconcile_wave_offsets(glue_seqs)?;
            }
            let logits = out.logits_owned()?;
            let d = head_rows.min(logits.len());
            let dec_logits = logits[..d].to_vec();
            if !sec_gidx.is_empty() {
                // Attended-KV summed before `complete_section_chunk` advances the
                // sequences. One record per co-batched section chunk.
                let sec_kv: usize = sec_seqs
                    .iter()
                    .map(|&sid| self.session.sequence_offset(sid).unwrap_or(0))
                    .sum();
                self.wave_stats.record_section(
                    sec_seqs.len(),
                    sec_adv.iter().sum(),
                    sec_kv,
                    t_wave.elapsed().as_millis() as u64,
                );
                super::PREFILL_OK_TOKENS.fetch_add(
                    sec_adv.iter().sum::<usize>() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                self.complete_section_chunk(&sec_gidx, &sec_adv);
            }
            return Ok(dec_logits);
        }

        // Creep group present. Full-sweep members (decode + glue) ride all N layers;
        // the creep rides only [cursor, win_end), its residual held WHOLE between
        // waves. The residual crosses `forward_wave` in caller order
        // `[decode | creep | glue]`, split by contiguous group at the boundaries.
        self.wave_cohort_advanced = true;
        let creep_tok: usize = inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(0))
            .sum();

        // The held residual is a slice of a PREVIOUS wave's activations, sized by
        // that wave's creep membership. `build_wave_group_inputs` rebuilds the
        // group each wave and silently drops members that errored or completed, so
        // a mid-creep drop leaves a residual wider than the tokens it is about to
        // be paired with — the rows would then be attributed to the wrong members
        // for the remaining layers, and wrong activations are exactly what makes
        // the sampler emit token 0 forever.
        //
        // Recover rather than abort: the creep re-forms from layer 0 next wave.
        // That is idempotent — a prefill member re-feeds its whole token block
        // (`work.tokens[..]`, never a chunk) and its slot offset is only advanced
        // at completion, so re-running `[0, cursor)` rewrites the same KV at the
        // same positions. Deliberately NOT an assert: a hard assert on this path
        // is what aborted the whole wave at 42553ca3 (see `reconcile_wave_offsets`),
        // and a panic on the scheduler thread takes the daemon with it.
        if cursor > 0 {
            if let Some(res) = self.wave_prefill_residual.as_ref() {
                let held = res.dims().get(1).copied().unwrap_or(0);
                if held != creep_tok {
                    tracing::error!(
                        held_residual_tokens = held,
                        creep_tokens = creep_tok,
                        cursor,
                        members = members.len(),
                        "wave creep membership changed mid-sweep — the held residual \
                         no longer matches the group. Restarting the creep from \
                         layer 0; the affected prefill re-runs the layers it had \
                         already done.",
                    );
                    // Restart IN PLACE rather than re-entering: the wave's deferred
                    // glue was already drained by `take_wave_glue` above, so a
                    // recursive call would find an empty queue and silently drop it.
                    // Dropping the residual and rewinding the cursor gives the same
                    // fresh start — the group already rebuilt this wave is the
                    // consistent one, and seg1 is skipped once `cursor == 0`.
                    //
                    // Idempotent: a prefill member re-feeds its whole token block
                    // (`work.tokens[..]`, never a chunk) and its slot offset only
                    // advances at completion, so re-running `[0, cursor)` rewrites
                    // the same K/V at the same positions.
                    self.wave_prefill_residual = None;
                    self.wave_prefill_cursor = 0;
                    cursor = 0;
                    win_end = budget.min(n);
                }
            }
        }

        // Segment 1 — full-sweep members only over [0, cursor). Runs when there is
        // any full-sweep member (decode or glue) and cursor > 0; the creep resumes
        // from its held residual at `cursor`. Yields caller order `[decode | glue]`.
        let seg1_res: Option<Tensor> = if cursor > 0 && (has_fullsweep || !verify_seqs.is_empty()) {
            if let Some(p) = glue_pending {
                self.session.set_pending_glue(p.clone());
            }
            self.model
                .forward_wave(
                    &mut self.session,
                    decode_seqs,
                    decode_inputs,
                    verify_seqs,
                    verify_inputs,
                    glue_seqs,
                    glue_inputs,
                    0,
                    cursor,
                    None,
                )?
                .into_residual()
        } else {
            None
        };
        // Split seg1's `[decode | verify | glue]` so the creep residual inserts
        // between them for seg2's `[decode | verify | creep | glue]` order.
        let (seg1_dec, seg1_glue): (Option<Tensor>, Option<Tensor>) = match &seg1_res {
            Some(r) => {
                let dec = if head_rows > 0 {
                    Some(r.narrow(1, 0, head_rows)?)
                } else {
                    None
                };
                let g = if glue_tok > 0 {
                    Some(r.narrow(1, head_rows, glue_tok)?)
                } else {
                    None
                };
                (dec, g)
            }
            None => (None, None),
        };

        // Segment 2 — full-sweep members + creep over [cursor, win_end). Input
        // residual caller order `[decode | creep | glue]`; at cursor 0 all embed
        // fresh (None).
        let pf_res = self.wave_prefill_residual.take();
        let seg2_in =
            Self::cat_caller_residual(&[seg1_dec.as_ref(), pf_res.as_ref(), seg1_glue.as_ref()])?;
        if let Some(p) = glue_pending {
            self.session.set_pending_glue(p.clone());
        }
        // Time seg2 alone (the co-batch the creep actually rides) — seg1 is
        // decode+glue over [0, cursor), which the creep did NOT ride, so charging
        // its wall-clock to the prefill channel would understate the prefill rate.
        let t_seg2 = Instant::now();
        // Verify blocks lead the prefill slot, the creep follows: `[decode |
        // verify]` then stays the contiguous head that every segment shares and
        // that the caller reads its logits from.
        let mid_seqs: Vec<usize> = verify_seqs.iter().chain(&seq_ids).copied().collect();
        let mid_inputs: Vec<Tensor> = verify_inputs.iter().chain(&inputs).cloned().collect();
        let seg2 = match self.model.forward_wave(
            &mut self.session,
            decode_seqs,
            decode_inputs,
            &mid_seqs,
            &mid_inputs,
            glue_seqs,
            glue_inputs,
            cursor,
            win_end,
            seg2_in,
        ) {
            Ok(s) => s,
            Err(e) => {
                // Drop the creep group cleanly (requeue scope prefills on OOM) so a
                // fresh one forms next wave, then surface the error.
                self.fail_wave_group(&members, &prefill_gidxs, &e);
                return Err(e);
            }
        };

        // Record the co-batched creep throughput — prefill and section members are
        // tallied into their own channels, sharing seg2's wall-clock (the forward
        // they rode concurrently with decode) so the dashboard shows their CONCURRENT
        // rate instead of reading zero. One record per wave; KV is summed now, before
        // `complete_wave_group` advances the sequences at the head.
        {
            let ms = t_seg2.elapsed().as_millis() as u64;
            let (mut pf_seqs, mut pf_tok, mut pf_kv) = (0usize, 0usize, 0usize);
            let (mut sc_seqs, mut sc_tok, mut sc_kv) = (0usize, 0usize, 0usize);
            for (m, inp) in members.iter().zip(inputs.iter()) {
                let tok = inp.dims().get(1).copied().unwrap_or(0);
                match m {
                    WaveMember::Prefill { seq_id, .. } => {
                        pf_seqs += 1;
                        pf_tok += tok;
                        pf_kv += self.session.sequence_offset(*seq_id).unwrap_or(0);
                    }
                    WaveMember::Section { seq_id, .. } => {
                        sc_seqs += 1;
                        sc_tok += tok;
                        sc_kv += self.session.sequence_offset(*seq_id).unwrap_or(0);
                    }
                }
            }
            // A forward ran, so the admitted set is advancing — the admission
            // fast path may safely wait for a completion (`settled`).
            self.wave_ran_forward = true;
            if pf_seqs > 0 {
                self.wave_stats.record(true, pf_seqs, pf_tok, pf_kv, ms);
            }
            if sc_seqs > 0 {
                self.wave_stats.record_section(sc_seqs, sc_tok, sc_kv, ms);
            }
            // Every completed wave forward is OOM-free prefill throughput —
            // the progress signal the stall-grace gate and evidence reopen
            // read. Without this, pump-driven phases (scope ingest, section
            // creep) look stalled to the admission regulator even at full
            // throughput, because only the drain-path prefills tick it.
            if pf_tok + sc_tok > 0 {
                super::PREFILL_OK_TOKENS.fetch_add(
                    (pf_tok + sc_tok) as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        }

        if win_end >= n {
            // Head reached: per-sequence logits, caller order `[decode | creep |
            // glue]`. Decode first; creep members next (promote/seal); glue logits,
            // if present, trail and are discarded.
            if has_glue {
                self.reconcile_wave_offsets(glue_seqs)?;
            }
            let logits = seg2.logits_owned()?;
            let d = head_rows.min(logits.len());
            let creep_end = (d + members.len()).min(logits.len());
            let dec_logits = logits[..d].to_vec();
            let member_logits = logits[d..creep_end].to_vec();
            self.complete_wave_group(&members, &member_logits);
            return Ok(dec_logits);
        }

        // Paused: split seg2's `[decode | creep | glue]` residual. Hold the creep
        // whole; continue the full-sweep members `[decode | glue]` into seg3.
        let res = seg2
            .into_residual()
            .ok_or_else(|| candle::Error::Msg("co-batch wave: missing residual".into()))?;
        let dec_part = if head_rows > 0 {
            Some(res.narrow(1, 0, head_rows)?)
        } else {
            None
        };
        let creep_part = res.narrow(1, head_rows, creep_tok)?;
        let glue_part = if glue_tok > 0 {
            Some(res.narrow(1, head_rows + creep_tok, glue_tok)?)
        } else {
            None
        };
        self.wave_prefill_residual = Some(creep_part);
        self.wave_prefill_cursor = win_end;
        self.wave_prefill_members = members;

        // Segment 3 — full-sweep members only over [win_end, N). Input caller order
        // `[decode | verify | glue]`. Skipped when there is no full-sweep member
        // (the creep paused at win_end, nothing else to sweep).
        if !has_fullsweep && verify_seqs.is_empty() {
            return Ok(Vec::new());
        }
        let seg3_in = Self::cat_caller_residual(&[dec_part.as_ref(), glue_part.as_ref()])?;
        if let Some(p) = glue_pending {
            self.session.set_pending_glue(p.clone());
        }
        let seg3 = self.model.forward_wave(
            &mut self.session,
            decode_seqs,
            decode_inputs,
            verify_seqs,
            verify_inputs,
            glue_seqs,
            glue_inputs,
            win_end,
            n,
            seg3_in,
        )?;
        if has_glue {
            self.reconcile_wave_offsets(glue_seqs)?;
        }
        let mut logits = seg3.logits_owned()?;
        logits.truncate(head_rows);
        Ok(logits)
    }

    /// Handle a device-OOM from the ragged prefill forward: the batch was too
    /// wide for the card. Cut the admission budget (so subsequent waves admit
    /// less) and surface the error on each in-batch prefill's caller channel.
    ///
    /// The hardest evidence the controller gets — a forward that actually failed
    /// — so it acts immediately here rather than waiting for the setpoint loop.
    ///
    /// `group_idxs` are the `active_prefills` positions that were in this forward;
    /// they're still valid because nothing mutates `active_prefills` between the
    /// forward returning and this call.
    fn handle_prefill_oom(&mut self, group_idxs: &[usize], err: &candle::Error) {
        let in_batch: HashSet<usize> = group_idxs.iter().copied().collect();
        let msg = format!("batched prefill forward failed: {err}");
        for (i, p) in self.active_prefills.iter_mut().enumerate() {
            if in_batch.contains(&i) {
                p.error = Some(ConversationError::Channel(msg.clone()));
            }
        }
    }

    /// Drain finished or errored entries from `active_prefills`. Errored
    /// entries emit `TurnEvent::Error`; finished entries are passed to
    /// `finalise_prefill` (which samples the first token and inserts into
    /// `active_decodes`).
    pub(super) fn promote_finished_prefills_to_decodes(&mut self) {
        // Use swap_remove for efficiency; iterate from the back.
        // **Unconditional: a slot is held from admission to completion.** A
        // finished prefill is not asking for a decode slot — it *is* the slot,
        // changing phase, and its ground was claimed when it was admitted. So
        // there is nothing to gate and nothing to wait for.
        //
        // Gating this on a free decode slot is what produced the wedge the
        // admission rewrite removed: finished turns queued here holding their
        // whole materialised prefixes (19 of them, 30,710 tokens of K/V against
        // 22 free regions of 395), while the decode side sat at its cap and
        // could not drain them, and the prefill side was refused the rows that
        // would have started anything new. Bounding what may be *admitted* is
        // `admit::gate`'s job; by the time work reaches here it has already
        // been paid for.
        let mut i = 0;
        while i < self.active_prefills.len() {
            let done = {
                let p = &self.active_prefills[i];
                p.error.is_some() || prefill_finished(p)
            };
            if !done {
                i += 1;
                continue;
            }
            let p = self.active_prefills.swap_remove(i);
            let ActivePrefill {
                work,
                offset,
                next_projection: _,
                final_logits,
                error,
                prefill_start,
            } = p;
            // **A drain that loses tokens says so.** `prefill_done` treats a
            // slot holding final logits as finished whatever its offset reads,
            // because such a slot can never advance — `form_wave_group` will not
            // schedule it, so leaving it undrained strands it and the engine
            // with it. But finalising it seals the turn short, and content
            // vanishing quietly is exactly the kind of failure that gets
            // diagnosed as the model being wrong months later.
            if error.is_none() && final_logits.is_some() && offset < work.tokens.len() {
                tracing::warn!(
                    target: "candle_conversation::scheduler::interleave",
                    seq_id = work.sequence_id.0,
                    unread = work.tokens.len() - offset,
                    total = work.tokens.len(),
                    "prefill drained holding logits with tokens unread — the turn \
                     seals short by that many",
                );
            }
            // A compression-turn re-prefill carries no decode and reports to the
            // summariser, not a caller. Seal it directly off the wave (snapshot
            // the role-coherent K/V + record the turn) instead of running
            // `finalise_prefill`.
            if let SealAction::CompressionTurn { job_id } = &work.seal_action {
                let job_id = *job_id;
                let slot = work.sequence_id;
                match error {
                    Some(e) => {
                        if let Some(p) = self.pending_compression_seals.remove(&job_id) {
                            let _ = p
                                .response_tx
                                .send(Err(crate::summary_tree::ProbeError::Soft(format!(
                                    "SubmitSummaryProbe: reproject prefill: {e}"
                                ))));
                        }
                        self.free_summary_slot(slot);
                    }
                    None => {
                        let t = std::time::Instant::now();
                        self.complete_compression_turn(slot, job_id);
                        crate::scheduler::run::note_promote_split(
                            crate::scheduler::run::PromoteStep::Compression,
                            t.elapsed().as_micros() as u64,
                        );
                    }
                }
                continue;
            }
            if let Some(e) = error {
                let _ = work.event_tx.send(TurnEvent::Error(e));
                continue;
            }
            let logits = match final_logits {
                Some(l) => l,
                None => {
                    let _ = work
                        .event_tx
                        .send(TurnEvent::Error(ConversationError::Channel(
                            "prefill produced no final logits".into(),
                        )));
                    continue;
                }
            };
            let prefill_ms = prefill_start
                .map(|s| s.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let turn_start = work.submitted_at;
            let token_count = work.tokens.len();
            let t_fin = std::time::Instant::now();
            self.finalise_prefill(work, logits, prefill_ms, turn_start, token_count);
            crate::scheduler::run::note_promote_split(
                crate::scheduler::run::PromoteStep::Finalise,
                t_fin.elapsed().as_micros() as u64,
            );
            // swap_remove pulled the last element into i; don't increment.
        }
    }

    /// Post-forward path shared by both single and batched prefill: sample
    /// the first token, emit it, and either transition to decode or close
    /// the turn out immediately on EOS / max_decode_tokens == 0.
    fn finalise_prefill(
        &mut self,
        work: PrefillWork,
        logits: Tensor,
        prefill_ms: f64,
        turn_start: Instant,
        token_count: usize,
    ) {
        // Total KV position after this prefill.
        let context_depth = self
            .session
            .sequence_offset(work.sequence_id.0)
            .unwrap_or(token_count);

        // Decode-start line: the effective sampling config this conversation turn
        // will decode under. Confirms empirically whether a turn is stochastic
        // (temp>0 + top_k/top_p) or greedy (temp≈0 → argmax), and at what context
        // depth. Enable with
        // `RUST_LOG=candle_conversation::scheduler::decode=debug`.
        tracing::trace!(
            target: "candle_conversation::scheduler::decode",
            seq = work.sequence_id.0,
            context_depth,
            prefill_tokens = token_count,
            max_decode_tokens = work.max_decode_tokens,
            temperature = work.sampling.temperature,
            top_k = work.sampling.top_k,
            top_p = work.sampling.top_p,
            repeat_penalty = work.sampling.repeat_penalty,
            segment_temp_boost = work.sampling.segment_temp_boost,
            dry = work.sampling.dry.is_some(),
            greedy = work.sampling.temperature <= 0.01,
            seed = work.sampling.seed,
            "conversation decode start",
        );

        let mut sampling_state = self
            .sampling_states
            .remove(&work.sequence_id)
            .expect("sampling state must exist for active sequence");
        sampling_state.end_turn(work.sampling.cross_turn_window);
        sampling_state.record_context_tokens(&work.tokens, self.sampler.max_recent_len());

        // Send prefill progress: complete (single-prefill path needs this;
        // batched path already streams progress per-chunk, but a final
        // tokens_done==tokens_total event is always benign).
        let _ = work.event_tx.send(TurnEvent::PrefillProgress {
            tokens_done: token_count,
            tokens_total: token_count,
        });

        // ── a turn that BEGINS inside a grammar ──────────────────────────────
        //
        // `triggers` cannot express this. Both registry checks run on *sampled*
        // tokens — this function's first-token check and the decode loop's
        // per-token one — so a turn whose grammar is entered on a token the
        // caller prefilled would never arm at all: the prefill goes into K/V
        // without passing the sampler. That is not a missed optimisation, it is
        // a grammar that silently does not apply, and the decode then imitates
        // the shape it was seeded with while nothing enforces it.
        //
        // So the tree is armed here, before anything is sampled. Its opening
        // scaffold was already appended to `work.tokens` when the turn was
        // assembled (`Conversation::submit_turn`), which is why the walk starts
        // by replaying it: the driver has to sit at the same node the K/V does.
        // What is left is the first genuine choice the grammar leaves open, and
        // the first sampled token of the turn is taken under its mask.
        let mut turn_driver = work.turn_grammar.clone().map(StencilDriver::new);
        let mut sampling = work.sampling.clone();
        if let Some(driver) = turn_driver.as_mut() {
            let (scaffold, action) = driver.opening();
            match &action {
                StepMask::Branch(set) => {
                    sampling.stencil = set.tokens().iter().map(|&t| t as i32).collect();
                }
                // A tree whose opening is free text or empty constrains nothing
                // here; the decode loop picks it up from the next step.
                StepMask::Free { .. } | StepMask::Done | StepMask::Prefill(_) => {}
            }
            tracing::debug!(
                target: "candle_conversation::stencil",
                seq_id = work.sequence_id.0,
                tree = driver.tree().label(),
                scaffold = scaffold.len(),
                masked = matches!(action, StepMask::Branch(_)),
                "turn grammar armed at the prefill boundary",
            );
        }

        let first_token = match self.sample_single(&logits, &sampling, &mut sampling_state) {
            Ok(t) => t,
            Err(e) => {
                self.sampling_states
                    .insert(work.sequence_id, sampling_state);
                let _ = work.event_tx.send(TurnEvent::Error(e));
                return;
            }
        };

        // Detect think-mode entry: the model opens its OWN `<think>` as the first
        // decoded token, or the assistant prefill leaves one open. "Leaves open"
        // is not "contains" — see [`prefill_leaves_think_open`].
        let initial_inside_think_block = {
            let tid = work.sampling.segment_open_token_id;
            if tid >= 0 {
                let tok = tid as u32;
                // **Only the ASSISTANT lead can leave a block open**, so the scan
                // starts at `assistant_content_start`. The markers are ordinary
                // vocabulary ids, and the tokenizer emits them for the literal
                // text too — so a `<tool_response>` carrying source that merely
                // MENTIONS `<think>` puts the open id in the USER half of the
                // grid. This repo's own `dialect.rs` does exactly that, and the
                // code-reading ingest feeds it back in. Scanning the whole grid
                // would arm `in_segment` off that quoted text before the turn had
                // decoded anything, and under `ThinkMode::Off` (hard cap of one)
                // the second decoded token would be rewritten to `</think>`.
                let assistant_lead = work
                    .tokens
                    .get(work.assistant_content_start as usize..)
                    .unwrap_or(&[]);
                let close = u32::try_from(work.sampling.segment_close_token_id).ok();
                let prefill_has_think =
                    prefill_leaves_think_open(assistant_lead.iter(), tok, close);
                // The block opens either way: the common case is the model
                // sampling its OWN `<think>` as the first token; the rarer case is
                // a caller-supplied assistant prefill that already opens one.  In
                // BOTH cases the sampler's `in_segment` must flip — it gates the
                // reflection-marker suppression, the thinking temperature boost,
                // and the `</think>` EOT ramp (all keyed off `segment_len`, which
                // only advances while `in_segment`).  (DRY is no longer gated
                // here — it has its own `dry_span_len`/`dry_suppressed` scope,
                // reset at `<think>`/`</think>` via `enter_segment`/`exit_segment`.)
                // Flipping it only for the prefilled case left the sampler's flag
                // stuck false for a model-opened block, silently disabling every
                // one of those controls for its whole duration even though the
                // health flag (`inside_think_block`) correctly tracked it.
                let opens_think = prefill_has_think || first_token == tok;
                if opens_think && !sampling_state.in_segment {
                    sampling_state.enter_segment();
                }
                opens_think
            } else {
                false
            }
        };

        self.sampling_states
            .insert(work.sequence_id, sampling_state);

        // Per-token trace for the prefill-emitted first token.  Enable
        // with `RUST_LOG=candle_conversation::scheduler::sampling=trace`.
        // This is the canonical "what did the model say first?" diag —
        // an early-EOS bug very often shows up as the first sampled
        // token already being EOS, meaning the model's K/V context is
        // pushing logits onto the EOS column straight out of prefill.
        if tracing::enabled!(
            target: "candle_conversation::scheduler::sampling",
            tracing::Level::TRACE,
        ) {
            let decoded = self
                .tokenizer
                .decode(&[first_token], false)
                .unwrap_or_else(|_| "<?>".to_string());
            let first_token_is_eos = self.is_eos(first_token);
            tracing::trace!(
                target: "candle_conversation::scheduler::sampling",
                seq_id = work.sequence_id.0,
                step = 0,
                token_id = first_token,
                is_eos = first_token_is_eos,
                decoded = %decoded,
                "sampled token (prefill first)",
            );
            if first_token_is_eos {
                tracing::debug!(
                    target: "candle_conversation::scheduler::sampling",
                    seq_id = work.sequence_id.0,
                    token_id = first_token,
                    "EOS fired on the very first sampled token — model is \
                     producing EOS immediately after prefill; check K/V \
                     context coherence",
                );
            }
        }

        let sampling_temperature = work.sampling.temperature;

        if self.is_eos(first_token) || work.max_decode_tokens == 0 {
            // View sequences (SubmitTurn path): the prefill already wrote KV
            // blocks that must be finalized onto the parent and sealed into
            // the substrate.  Insert as a finished DecodeState so
            // cleanup_finished runs finalize_view + perform_seal_and_write.
            //
            // Non-view sequences (raw RULER / summarisation): no parent to
            // finalize and seal=None is correct — use the fast path.
            if self.turn_views.contains_key(&work.sequence_id) {
                // Through `push_generated` like every other token, so the
                // reasoning boundary is seen even on this no-decode path.
                let mut state = DecodeState {
                    event_tx: work.event_tx,
                    generated_tokens: TokenBuffer::default(),
                    think_close_at: None,
                    lease_left: Scheduler::DECODE_LEASE_TOKENS,
                    lease_expired: false,
                    forwarded_generated: 0,
                    pending_page_cut: false,
                    pending_page_cut_after: None,
                    max_tokens: work.max_decode_tokens,
                    sampling_config: work.sampling,
                    seal_action: work.seal_action,
                    post_decode_tokens: work.post_decode_tokens,
                    belief: work.belief,
                    prefill_tokens: work.tokens,
                    user_text: work.user_text,
                    tags: work.tags,
                    user_content_start: work.user_content_start,
                    user_content_end: work.user_content_end,
                    assistant_content_start: work.assistant_content_start,
                    no_think: work.no_think,
                    prefill_assistant_text: work.prefill_assistant_text,
                    finished: true,
                    decode_start: Instant::now(),
                    decode_busy_us: 0,
                    prefill_ms,
                    prefill_token_count: context_depth,
                    turn_start,
                    health: {
                        let mut hs = crate::decode_health::DecodeHealthState::new(
                            self.health_config.repetition_window,
                            self.health_config.health_log_capacity,
                        );
                        hs.apply_baseline_config(
                            self.health_config.entropy_baseline_window,
                            self.health_config.entropy_trend_relative_factor,
                            self.health_config.entropy_trend_absolute_min_nats,
                        );
                        hs.inside_think_block = initial_inside_think_block;
                        hs.skip_entropy_checks = sampling_temperature <= 0.01;
                        hs
                    },
                    reprojection: work.reprojection,
                    non_punct_since_reproject: 0,
                    last_projection_end: 0,
                    in_tool_call: false,
                    free_tool_calls_from_penalties: work.free_tool_calls_from_penalties,
                    triggers: work.triggers,
                    stencil: None,
                    pending_mask: None,
                };
                // The turn's first token opens a page at the prefill/decode
                // boundary, so the reasoning starts one of its own.
                state.push_committed(first_token, self.think_close, &self.page_break_tokens);
                // No speculative rewind can be in flight — this turn decodes
                // nothing — so the cut is taken at once.
                super::flush_page_cut(self.model.as_ref(), work.sequence_id, &mut state);
                self.active_decodes.insert(work.sequence_id, state);
            } else {
                self.finish_immediately(
                    work.sequence_id,
                    first_token,
                    &work.event_tx,
                    prefill_ms,
                    turn_start,
                    context_depth,
                );
            }
            return;
        }

        let _ = work.event_tx.send(TurnEvent::Token(first_token));

        // An armed turn grammar already owns this token — it was sampled under
        // the mask above, so it is fed back rather than tested against the
        // registry. A turn cannot be in both states: beginning inside a tree and
        // entering one on this token are the same slot.
        let stencil = match turn_driver {
            Some(mut driver) => {
                let bytes = self
                    .tokenizer
                    .decode(&[first_token], false)
                    .unwrap_or_default();
                driver.accept(first_token, bytes.as_bytes());
                Some(driver)
            }
            // The first sampled token can itself be a stencil trigger — e.g. the
            // model emits `<tool_call>` as its very first response token. The
            // decode-loop trigger check runs only on tokens sampled in
            // `batch_decode_step`, never this one, so check it here too —
            // otherwise steering silently never engages for those calls.
            None => {
                let d = work.triggers.driver_for(first_token);
                if let Some(d) = &d {
                    tracing::debug!(
                        target: "candle_conversation::stencil",
                        seq_id = work.sequence_id.0,
                        tree = d.tree().label(),
                        trigger = first_token,
                        "stencil steering started (trigger on the first decoded token)",
                    );
                }
                d
            }
        };
        // A first-token `<tool_call>` trigger enters the call immediately, so the
        // in-call state must be set HERE — the decode loop's `is_tool_open` scan
        // (which normally sets it) only sees tokens sampled in `batch_decode_step`,
        // never this one. Without it the in-call reprojection freeze never engages
        // for these turns and cadence/punctuation triggers re-orient the selection
        // mid-call. The early first-reprojection push below still fires once — it
        // is this turn's lock-in reprojection, exactly like the one `is_tool_open`
        // fires before freezing.
        let first_token_opens_call = stencil
            .as_ref()
            .is_some_and(|d| d.tree().label() == super::TOOL_CALL_TREE_LABEL);
        // Captured before `work.reprojection` moves into the DecodeState: the
        // early first-reprojection below fires only for turns whose target
        // layer runs belief-driven selection — a plain-prompt layer (the
        // titler's single-section schema) gains nothing from the extra swap.
        let wants_early_reprojection = work
            .reprojection
            .as_ref()
            .is_some_and(|p| p.has_belief_collections());

        // Through `push_generated` like every other token, so a `</think>` the
        // prefill's own logits produced still fixes the reasoning boundary.
        let mut state = DecodeState {
            event_tx: work.event_tx,
            generated_tokens: TokenBuffer::default(),
            think_close_at: None,
            lease_left: Scheduler::DECODE_LEASE_TOKENS,
            lease_expired: false,
            forwarded_generated: 0,
            pending_page_cut: false,
            pending_page_cut_after: None,
            max_tokens: work.max_decode_tokens,
            sampling_config: work.sampling,
            seal_action: work.seal_action,
            post_decode_tokens: work.post_decode_tokens,
            belief: work.belief,
            prefill_tokens: work.tokens,
            user_text: work.user_text,
            tags: work.tags,
            user_content_start: work.user_content_start,
            user_content_end: work.user_content_end,
            assistant_content_start: work.assistant_content_start,
            no_think: work.no_think,
            prefill_assistant_text: work.prefill_assistant_text,
            finished: false,
            decode_start: Instant::now(),
            decode_busy_us: 0,
            prefill_ms,
            prefill_token_count: context_depth,
            turn_start,
            health: {
                let mut hs = crate::decode_health::DecodeHealthState::new(
                    self.health_config.repetition_window,
                    self.health_config.health_log_capacity,
                );
                hs.apply_baseline_config(
                    self.health_config.entropy_baseline_window,
                    self.health_config.entropy_trend_relative_factor,
                    self.health_config.entropy_trend_absolute_min_nats,
                );
                hs.inside_think_block = initial_inside_think_block;
                hs.skip_entropy_checks = sampling_temperature <= 0.01;
                hs
            },
            reprojection: work.reprojection,
            non_punct_since_reproject: 0,
            last_projection_end: 0,
            in_tool_call: first_token_opens_call,
            free_tool_calls_from_penalties: work.free_tool_calls_from_penalties,
            triggers: work.triggers,
            stencil,
            pending_mask: None,
        };
        // The turn's first token opens a page at the prefill/decode boundary, so
        // the reasoning starts one of its own.
        state.push_committed(first_token, self.think_close, &self.page_break_tokens);
        // No speculative rewind can be in flight on this path — the turn has not
        // decoded yet — so the cut is taken at once.
        super::flush_page_cut(self.model.as_ref(), work.sequence_id, &mut state);
        self.active_decodes.insert(work.sequence_id, state);
        // Fire the turn's FIRST reprojection immediately (drained right after
        // the next decode step, ~token 1). The prefill just wrote the user
        // query's wide-Q into R16, so the belief scan can score it and
        // materialize the right sections BEFORE the model's plan forms in the
        // early <think> tokens — waiting for the 64-token cadence lets a
        // wrong-tool prefix anchor the reasoning first (the submit-time
        // projection only carries the PREVIOUS turn's belief; it cannot see
        // this turn's query). For a first-token tool call this is the turn's
        // lock-in reprojection: `in_tool_call` is already set above, so the
        // call body stays frozen afterwards.
        if wants_early_reprojection {
            Self::queue_reprojection(&mut self.pending_reprojections, work.sequence_id);
        }
    }

    /// Forward `tokens` on `sequence_id`, splitting the pass at the turn's
    /// reasoning boundary if it falls inside them.
    ///
    /// **A forward must not carry tokens from both sides of `</think>`.** A
    /// span's index rows are pooled by the forward that carries it, so a block
    /// pooled across the boundary cannot be un-pooled at seal time and the
    /// reasoning would not occupy whole pages. Plain decode never straddles —
    /// one token per sequence per wave — but a static run does: a think-steer
    /// tree suppresses the model's own `</think>` and injects the closing tag as
    /// a run, which can carry tokens after it.
    ///
    /// Splitting here rather than at the two call sites is what makes the
    /// invariant structural: this is the one function that forwards an arbitrary
    /// token span for a single sequence, so a future third caller inherits it.
    pub(super) fn run_prefill(
        &mut self,
        sequence_id: SequenceId,
        tokens: &[u32],
    ) -> Result<Tensor, ConversationError> {
        // **Every break token in the span, not just the first.** A prefilled
        // assistant head carries `<think>` and `</think>` in one pass, and a
        // multi-turn prefill carries a turn closer as well — splitting once
        // would leave the later markers pooled across their own boundaries,
        // which is not correctable afterwards.
        let mut rest = tokens;
        let mut last_logits = None;
        while let Some(at) = self.reasoning_split(rest) {
            let (head, tail) = rest.split_at(at);
            last_logits = Some(self.run_prefill_span(sequence_id, head)?);
            match self.model.close_positional_page(sequence_id.0) {
                // The head's own token ids when it is short. A surplus page is
                // identified by what is IN it, and the pages that do not belong
                // to any turn are consistently 5 and 7 tokens wide — small
                // enough to name outright rather than infer from their width.
                Ok(closed) => tracing::info!(
                    target: "candle_conversation::scheduler::unit_boundary",
                    seq_id = sequence_id.0,
                    site = "prefill-break-token",
                    closed,
                    at,
                    span = rest.len(),
                    head = ?(head.len() <= 16).then_some(head),
                    break_token = ?head.last(),
                    "index: closed a page mid-prefill at a break token"
                ),
                Err(e) => tracing::warn!(
                    seq_id = sequence_id.0,
                    "closing the index page at a prefilled break token failed ({e}); the \
                     region it bounds will not occupy whole pages and cannot be windowed \
                     out of a later projection"
                ),
            }
            rest = tail;
        }
        if rest.is_empty() {
            // Every token was consumed by a split, so the last head's logits are
            // the span's — `reasoning_split` never returns a split at the end,
            // so this is only reachable for an empty input.
            return match last_logits {
                Some(l) => Ok(l),
                None => self.run_prefill_span(sequence_id, rest),
            };
        }
        self.run_prefill_span(sequence_id, rest)
    }

    /// Where to cut `tokens` so the reasoning boundary lands on a page edge:
    /// one past this turn's first `</think>`, or `None` when the span carries no
    /// boundary that needs one.
    ///
    /// `None` when the marker is absent, when it is the last token (nothing
    /// follows it in this pass, so the next forward is already the edge), or
    /// when the turn has recorded a close already — `think_close_at` holds the
    /// first only, and a later `</think>` in the answer body must not move a
    /// boundary that is fixed.
    /// **Reads the token stream, not the decode state.** The previous version
    /// asked `active_decodes` for the turn's `DecodeState` — which does not exist
    /// yet while the turn is prefilling, because it is built from the prefill's
    /// own logits afterwards. So for the case this exists to serve, a
    /// `<think>…</think>` block baked into the prompt, the lookup returned `None`
    /// and the pass was never split: 0 splits across 822 turns.
    ///
    /// One past the first break token, and `None` when that is the end of the
    /// span — there is nothing on the far side to separate.
    fn reasoning_split(&self, tokens: &[u32]) -> Option<usize> {
        let at = tokens
            .iter()
            .position(|t| self.page_break_tokens.contains(t))?
            + 1;
        (at < tokens.len()).then_some(at)
    }

    fn run_prefill_span(
        &mut self,
        sequence_id: SequenceId,
        tokens: &[u32],
    ) -> Result<Tensor, ConversationError> {
        // Chunked prefill: split large prompts into bounded chunks to keep
        // intermediate activation buffers from growing unboundedly.
        let logits = if tokens.len() > self.max_prefill_pass_tokens {
            let mut last_logits: Option<Tensor> = None;
            for chunk in tokens.chunks(self.max_prefill_pass_tokens) {
                let input = Tensor::new(chunk, &self.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(ConversationError::Model)?;
                let nl = self.model.num_layers();
                let logits_vec = self
                    .model
                    .forward_wave(
                        &mut self.session,
                        &[],
                        &[],
                        &[sequence_id.0],
                        &[input],
                        &[],
                        &[],
                        0,
                        nl,
                        None,
                    )
                    .and_then(|s| s.logits_owned())
                    .map_err(ConversationError::Model)?;
                self.session
                    .advance_sequence(sequence_id.0, chunk.len())
                    .map_err(ConversationError::Model)?;
                super::Scheduler::record_slot_tokens(&mut self.slot_tokens, sequence_id, chunk);
                last_logits = logits_vec.into_iter().next();
            }
            last_logits.ok_or_else(|| {
                ConversationError::Channel("no logits returned from chunked prefill".into())
            })?
        } else {
            let input = Tensor::new(tokens, &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(ConversationError::Model)?;

            let logits_vec = self
                .model
                .forward_wave(
                    &mut self.session,
                    &[],
                    &[],
                    &[sequence_id.0],
                    &[input],
                    &[],
                    &[],
                    0,
                    self.model.num_layers().max(1),
                    None,
                )
                .and_then(|s| s.logits_owned())
                .map_err(ConversationError::Model)?;

            self.session
                .advance_sequence(sequence_id.0, tokens.len())
                .map_err(ConversationError::Model)?;

            // Mirror these tokens into the slot's diagnostic log so the
            // turn-complete dump can reconstruct the exact context the
            // kernel saw (compiled out without the `context-dump` feature).
            super::Scheduler::record_slot_tokens(&mut self.slot_tokens, sequence_id, tokens);

            logits_vec.into_iter().next().ok_or_else(|| {
                ConversationError::Channel("no logits returned from prefill".into())
            })?
        };

        // Single exit for every prefill path: the forward wrote KV without the
        // decode kernel's self-increment, so refresh the cached decode
        // slot-state's writer slice with the advanced tail length. Without
        // this, a mid-decode injection (a stencil static run, a think-steer
        // continuation) is INVISIBLE to the following decode steps — the
        // kernel attends the tail chunk at its stale pre-prefill length and
        // the model decodes as if the injected tokens were never written.
        // No-op for slots that haven't decoded yet.
        self.session
            .refresh_decode_slot_state(sequence_id.0)
            .map_err(ConversationError::Model)?;

        Ok(logits)
    }
}

/// Whether an assistant prefill leaves a think block **open** at the point
/// decode takes over.
///
/// **Containing `<think>` is not the question.** Qwen3.5 suppresses reasoning by
/// prefilling an already-closed block, `<think>\n\n</think>\n\n`, so the open
/// marker is in every suppressed turn's prefill by construction — and the
/// sampler's segment state only ever sees *sampled* tokens, so the prefilled
/// close never reaches it. Asked "is `<think>` among the last few tokens", the
/// check answered yes on exactly the turns where thinking had been turned off,
/// and the sampler then decoded the whole answer believing it was inside a
/// think block: the ban on `</think>` outside a block lifted, and any
/// `force_segment_close_after` fired a closer into the prose. Measured on an
/// unstencilled reflection turn: `The belt is</think>`, nine tokens, the answer
/// ended by a forced close of a block that had closed before it began.
///
/// The most recent marker decides, which is the rule that holds whatever the
/// prefill is: a closed block, a closed block followed by a tool-call opener, a
/// bare `<think>` for a turn that is meant to reason, or an earlier turn's
/// markers further back in the buffer. The scan stops at the first marker from
/// the end, so it costs a handful of comparisons on every real prefill.
/// **The caller decides what is scanned, and it hands over the ASSISTANT lead
/// only.** The markers are ordinary vocabulary ids and the tokenizer emits them
/// for literal text, so a `<tool_response>` carrying source that merely mentions
/// `<think>` puts the open id in the USER half of the grid — this repo's own
/// `dialect.rs` does exactly that, and the code-reading ingest feeds it back.
/// Scanning a whole prefill would arm `in_segment` off that quoted text before
/// the turn had decoded anything; under `ThinkMode::Off`, where the hard cap is
/// one token, the second decoded token then gets rewritten to `</think>`. The
/// slice at `assistant_content_start` is what prevents it, and
/// `only_the_assistant_lead_is_scanned` holds the boundary.
fn prefill_leaves_think_open<'a>(
    tokens: impl DoubleEndedIterator<Item = &'a u32>,
    open: u32,
    close: Option<u32>,
) -> bool {
    tokens
        .rev()
        .find(|&&t| t == open || Some(t) == close)
        .is_some_and(|&t| t == open)
}

/// The slice boundary the helper above relies on, which main's own module does
/// not exercise: a prefill whose USER half quotes `<think>` while its assistant
/// lead closes its block.
#[cfg(test)]
mod think_prefill_slice_tests {
    use super::prefill_leaves_think_open;

    const OPEN: u32 = 248068;
    const CLOSE: u32 = 248069;

    /// **User content that merely QUOTES `<think>` must not arm the flag.**
    #[test]
    fn only_the_assistant_lead_is_scanned() {
        // `[user … <think> … ] [assistant lead: closed block]`
        let grid = [9u32, OPEN, 9, 9, OPEN, 3, CLOSE, 4];
        let assistant_content_start = 4;
        assert!(
            !prefill_leaves_think_open(grid[assistant_content_start..].iter(), OPEN, Some(CLOSE)),
            "the assistant lead closed its block; quoted user text must not override that"
        );
        // What the slice prevents: handed the user half, the very same scan does
        // arm — so the boundary is doing the work, not the scan.
        assert!(prefill_leaves_think_open(
            grid[..2].iter(),
            OPEN,
            Some(CLOSE)
        ));
    }

    /// A deep opener in the assistant lead still arms. The fixed 5-token tail
    /// window this replaced would have missed it entirely.
    #[test]
    fn a_deep_opener_in_the_lead_still_arms() {
        let mut deep = vec![OPEN];
        deep.extend(std::iter::repeat_n(7u32, 40));
        assert!(prefill_leaves_think_open(deep.iter(), OPEN, Some(CLOSE)));
    }
}

#[cfg(test)]
mod idle_demote_tests {
    use super::{idle_slots_to_demote, Scheduler, SequenceId};
    use std::collections::{HashMap, HashSet};

    const PASSES: u32 = Scheduler::IDLE_SLOT_DEMOTE_PASSES;

    fn slot(n: usize, tokens: usize) -> (SequenceId, usize) {
        (SequenceId(n), tokens)
    }

    /// Run every quiet pass *before* the threshold, asserting none of them
    /// demote. At the current one-pass grace there are none to run; the helper
    /// exists so each test reads the same whatever the grace is set to, rather
    /// than spelling a range that is empty for this value of it.
    fn quiet_below_threshold(
        counters: &mut HashMap<SequenceId, u32>,
        slots: &[(SequenceId, usize)],
        busy: &HashSet<SequenceId>,
    ) {
        let below = PASSES.saturating_sub(1);
        for pass in 0..below {
            assert!(
                idle_slots_to_demote(counters, slots, busy)
                    .demote
                    .is_empty(),
                "pass {pass} demoted before the threshold",
            );
        }
    }

    /// A quiet slot is demoted on the pass that reaches the threshold, and not
    /// before — one admission pass of cover for a tool result already in
    /// flight, and no more, because holding a slot back declines the
    /// relocation its reload would perform.
    #[test]
    fn a_quiet_slot_demotes_on_the_threshold_pass_and_not_before() {
        let mut counters = HashMap::new();
        let slots = [slot(1, 4_000)];
        let busy = HashSet::new();
        quiet_below_threshold(&mut counters, &slots, &busy);
        assert_eq!(
            idle_slots_to_demote(&mut counters, &slots, &busy).demote,
            vec![(SequenceId(1), 4_000)],
            "the threshold pass demotes, and reports the tokens it gives back",
        );
    }

    /// **A demoted slot is not demoted again until something touches it.**
    ///
    /// With a grace of one pass, a `>=` threshold would re-fire on every
    /// subsequent quiet pass. Each re-fire takes a substrate write lock and
    /// walks a conversation that has already given everything back, contending
    /// with the persistence thread to shed nothing — per idle conversation, per
    /// admission. Firing on the equality means once per quiet period.
    #[test]
    fn a_demoted_slot_is_not_demoted_again_until_it_is_touched() {
        let mut counters = HashMap::new();
        let slots = [slot(1, 4_000)];
        let quiet = HashSet::new();
        let busy: HashSet<SequenceId> = [SequenceId(1)].into_iter().collect();

        quiet_below_threshold(&mut counters, &slots, &quiet);
        assert_eq!(
            idle_slots_to_demote(&mut counters, &slots, &quiet).demote,
            vec![(SequenceId(1), 4_000)],
        );
        for pass in 0..8 {
            assert!(
                idle_slots_to_demote(&mut counters, &slots, &quiet)
                    .demote
                    .is_empty(),
                "pass {pass} after the demote fired again on a slot with nothing left",
            );
        }

        // A touch re-arms it, and reports the wake so the caller can measure
        // how long the demotion lasted.
        let woken = idle_slots_to_demote(&mut counters, &slots, &busy);
        assert_eq!(woken.woke, vec![SequenceId(1)], "the wake is reported");
        assert!(woken.demote.is_empty());
        quiet_below_threshold(&mut counters, &slots, &quiet);
        assert_eq!(
            idle_slots_to_demote(&mut counters, &slots, &quiet).demote,
            vec![(SequenceId(1), 4_000)],
            "after a touch the slot serves a fresh quiet window and sheds again",
        );
    }

    /// A slot that was never being aged reports no wake — there is no quiet to
    /// interrupt, so pairing it against a demotion timestamp would be noise.
    #[test]
    fn a_slot_that_was_never_idle_reports_no_wake() {
        let mut counters = HashMap::new();
        let slots = [slot(1, 4_000)];
        let busy: HashSet<SequenceId> = [SequenceId(1)].into_iter().collect();
        assert!(idle_slots_to_demote(&mut counters, &slots, &busy)
            .woke
            .is_empty());
    }

    /// Any touch restarts the count: a slot that goes busy short of the
    /// threshold starts again from zero when it falls quiet.
    #[test]
    fn a_touch_restarts_the_count() {
        let mut counters = HashMap::new();
        let slots = [slot(1, 4_000)];
        let quiet = HashSet::new();
        let busy: HashSet<SequenceId> = [SequenceId(1)].into_iter().collect();
        quiet_below_threshold(&mut counters, &slots, &quiet);
        assert!(
            idle_slots_to_demote(&mut counters, &slots, &busy)
                .demote
                .is_empty(),
            "a busy slot is never demoted",
        );
        quiet_below_threshold(&mut counters, &slots, &quiet);
        assert_eq!(
            idle_slots_to_demote(&mut counters, &slots, &quiet).demote,
            vec![(SequenceId(1), 4_000)],
        );
    }

    /// **A slot holding no blocks still demotes** — this is the population the
    /// pass exists for, not an exception to it.
    ///
    /// A conversation between turns reads `tokens == 0`, because every turn's
    /// projection truncates its slot to zero blocks and rebuilds the prefix
    /// from the substrate. Its bytes are in the substrate's *hot* copies of its
    /// sealed turns, which only the caller's `evict_hot_to_free` reaches — and
    /// the caller only reaches slots this function returns. Skipping them made
    /// the whole pass inert against 5,920 MiB of resident K/V while the census
    /// reported `idle_slots = 0`.
    #[test]
    fn a_slot_holding_no_blocks_still_demotes_for_its_hot_turns() {
        let mut counters = HashMap::new();
        let slots = [slot(1, 0)];
        let busy = HashSet::new();
        quiet_below_threshold(&mut counters, &slots, &busy);
        assert_eq!(
            idle_slots_to_demote(&mut counters, &slots, &busy).demote,
            vec![(SequenceId(1), 0)],
            "a zero-block slot must reach the caller, which sheds its hot turns",
        );
    }

    /// Counters do not outlive their slots — a freed id is dropped from the
    /// map, so a recycled id starts from zero rather than inheriting the
    /// previous tenant's quiet.
    #[test]
    fn counters_do_not_outlive_their_slots() {
        let mut counters = HashMap::new();
        let busy = HashSet::new();
        let both = [slot(1, 4_000), slot(2, 4_000)];
        // Enough quiet passes that both slots are past the threshold and both
        // carry a counter, whatever the grace is set to.
        for _ in 0..PASSES {
            let _ = idle_slots_to_demote(&mut counters, &both, &busy);
        }
        assert_eq!(counters.len(), 2);

        // Slot 2 is freed; the next pass sees only slot 1, and slot 2's counter
        // must not survive it.
        let one = [slot(1, 4_000)];
        let _ = idle_slots_to_demote(&mut counters, &one, &busy);
        assert_eq!(counters.len(), 1, "the freed slot's counter was dropped");
        assert!(!counters.contains_key(&SequenceId(2)));

        // The recycled id serves its own quiet rather than inheriting slot 2's,
        // so it sheds on its own threshold pass and not on the first one it
        // appears in.
        let recycled = [slot(1, 4_000), slot(2, 4_000)];
        let mut fired = None;
        for pass in 1..=PASSES {
            if idle_slots_to_demote(&mut counters, &recycled, &busy)
                .demote
                .iter()
                .any(|(id, _)| *id == SequenceId(2))
            {
                fired = Some(pass);
                break;
            }
        }
        assert_eq!(
            fired,
            Some(PASSES),
            "the recycled id must serve a full quiet window of its own",
        );
    }

    /// Several quiet slots demote in the same pass — the pool's whole idle
    /// cohort gives its ground back together, which is the point.
    #[test]
    fn every_quiet_slot_demotes_in_the_same_pass() {
        let mut counters = HashMap::new();
        let slots = [slot(1, 100), slot(2, 200), slot(3, 300)];
        let busy: HashSet<SequenceId> = [SequenceId(2)].into_iter().collect();
        quiet_below_threshold(&mut counters, &slots, &busy);
        assert_eq!(
            idle_slots_to_demote(&mut counters, &slots, &busy).demote,
            vec![(SequenceId(1), 100), (SequenceId(3), 300)],
            "the busy slot is skipped, the quiet ones go together",
        );
    }
}

#[cfg(test)]
mod setpoint_tests {
    use super::{setpoint_regions, VramPhase};

    /// The setpoint scales with the span so the same constants hold on this
    /// card's 226-region KV side and on the workstation's, and decode always
    /// insists on less than load — KV grows a chunk per sequence per 32 steps
    /// there, so evicting defensively would just cost reloads.
    #[test]
    fn the_setpoint_scales_with_the_span_and_decode_asks_for_less() {
        let load = setpoint_regions(VramPhase::Load, 800);
        let decode = setpoint_regions(VramPhase::Decode, 800);
        assert_eq!(load, 100, "load is span/8 once the span clears the floor");
        assert_eq!(decode, 50, "decode is span/16");
        assert!(decode < load);
    }

    /// On a span too small for the floors, the setpoint stops at half the span.
    /// Asking for more would mean permanent pressure: every wave would run a
    /// relief pass that cannot possibly reach a setpoint the card can't hold.
    #[test]
    fn a_small_span_clamps_to_half_rather_than_demanding_the_floor() {
        assert_eq!(setpoint_regions(VramPhase::Load, 32), 16);
        assert_eq!(setpoint_regions(VramPhase::Decode, 8), 4);
        assert_eq!(
            setpoint_regions(VramPhase::Load, 0),
            0,
            "no span, no demand"
        );
    }
}

#[cfg(test)]
mod prefill_done_tests {
    use super::prefill_done;

    /// The whole grid of (logits, tokens consumed), stated explicitly.
    ///
    /// `true` means the wave group skips it *and* the drain takes it; `false`
    /// means it rides a forward. What matters is that one function answers for
    /// both, so no state can be skipped by the group and refused by the drain —
    /// which is what stalled run CE at 82 directories with every resource free.
    /// The cell that did it is the third row.
    #[test]
    fn the_done_grid_is_what_the_engine_relies_on() {
        let cases = [
            (false, 0usize, 10usize, false), // fresh — runs
            (false, 9, 10, false),           // nearly through — runs
            (false, 10, 10, true),           // consumed, no logits — CE's stall
            (false, 12, 10, true),           // past the end — drained
            (true, 3, 10, true),             // holding logits — cannot advance
            (true, 10, 10, true),            // the ordinary finish
        ];
        for (has_logits, offset, tokens, want) in cases {
            assert_eq!(
                prefill_done(has_logits, offset, tokens),
                want,
                "logits={has_logits} offset={offset}/{tokens}",
            );
        }
    }

    /// The cell that stalled CE: all tokens consumed, no logits. It cannot ride
    /// a forward (nothing left to advance), so it must be drained.
    #[test]
    fn a_consumed_prefill_without_logits_is_drained() {
        assert!(prefill_done(false, 10, 10));
    }

    /// A slot with logits cannot advance either, whatever its offset says.
    #[test]
    fn a_prefill_holding_logits_is_drained_whatever_its_offset() {
        assert!(prefill_done(true, 3, 10));
        assert!(prefill_done(true, 10, 10));
    }

    /// And a prefill with real work left is left alone to do it.
    #[test]
    fn a_prefill_with_tokens_left_and_no_logits_still_runs() {
        assert!(!prefill_done(false, 0, 10));
        assert!(!prefill_done(false, 9, 10));
    }
}

/// The decode lease's shape, checked when the crate is built rather than when a
/// test is run — a violation is a build error, which is where an invariant on a
/// constant belongs.
///
/// * **Non-zero**, or a slot parks before generating anything and the queue
///   cycles without progress.
/// * **A whole number of blocks.** Admission reserves the lease through
///   `ensure_capacity`, which allocates in blocks, so a lease ending mid-block
///   makes the allocator round up and the engine hold ground the gate never
///   priced. Small drift per slot — but it is *per slot*, and every failure this
///   engine has had was accounting drift multiplied by concurrency.
/// * **Longer than the step that spends it**, for the same reason as the first.
/// The park queue's bounds, checked at build time for the same reason as the
/// lease's: a zero bound would silently disable parking, and a zero retry count
/// would fail every parked turn on its first refusal.
const _: () = {
    assert!(Scheduler::PARKED_TURNS_MAX > 0);
    assert!(Scheduler::PARK_RESUME_ATTEMPTS > 0);
};

const _: () = {
    assert!(Scheduler::DECODE_LEASE_TOKENS > 0);
    assert!(Scheduler::DECODE_LEASE_TOKENS.is_multiple_of(candle_nn::kv_cache::CHUNK_SIZE));
    assert!(Scheduler::DECODE_LEASE_TOKENS > Scheduler::DECODE_CLAIM_TOKENS);
};

#[cfg(test)]
mod warm_budget_tests {
    use super::WARM_PIPELINE_SLACK_BYTES;

    /// The slack exists so a zero-budget machine's transient drain traffic never
    /// reads as over-budget — tonight's healthy pipeline peaked ~0.7 GiB.
    #[test]
    fn default_slack_clears_a_healthy_drain_pipeline() {
        let slack = WARM_PIPELINE_SLACK_BYTES;
        assert!(slack >= 768 * 1024 * 1024, "slack {slack} too small");
    }
}

#[cfg(test)]
mod prefill_think_tests {
    use super::prefill_leaves_think_open;

    const OPEN: u32 = 89;
    const CLOSE: u32 = 90;
    const NL: u32 = 10;
    const TOOL_CALL: u32 = 91;
    const WORD: u32 = 5;

    fn open(tokens: &[u32]) -> bool {
        prefill_leaves_think_open(tokens.iter(), OPEN, Some(CLOSE))
    }

    /// **The suppression prefill is a closed block, and it must read as one.**
    ///
    /// The regression this exists for: Qwen3.5's `<think>\n\n</think>\n\n`
    /// contains the open marker, and the check that asked "does it contain one"
    /// put every think-suppressed turn's sampler inside a block it would never
    /// see close.
    #[test]
    fn a_prefilled_closed_block_leaves_nothing_open() {
        assert!(!open(&[WORD, OPEN, NL, CLOSE, NL]));
    }

    /// The acting turn's prefill: the closed block, then straight into the call.
    #[test]
    fn a_closed_block_followed_by_a_call_opener_leaves_nothing_open() {
        assert!(!open(&[WORD, OPEN, NL, CLOSE, NL, TOOL_CALL]));
    }

    /// A turn that is meant to reason prefills a bare `<think>`, and it must.
    #[test]
    fn a_bare_open_marker_leaves_the_block_open() {
        assert!(open(&[WORD, NL, OPEN]));
    }

    #[test]
    fn a_prefill_with_no_markers_leaves_nothing_open() {
        assert!(!open(&[WORD, NL, WORD]));
        assert!(!open(&[]));
    }

    /// The most recent marker decides, so an earlier turn's closed block
    /// further back in the buffer cannot mask a fresh open, and cannot fake one.
    #[test]
    fn only_the_most_recent_marker_counts() {
        assert!(open(&[OPEN, WORD, CLOSE, WORD, NL, OPEN]));
        assert!(!open(&[OPEN, WORD, CLOSE, WORD, OPEN, NL, CLOSE, NL]));
    }

    /// A vocabulary with no close token cannot close a block, so an open marker
    /// anywhere behind the boundary leaves it open.
    #[test]
    fn without_a_close_token_an_open_marker_stays_open() {
        assert!(prefill_leaves_think_open(
            [WORD, OPEN, NL, CLOSE].iter(),
            OPEN,
            None
        ));
    }
}
