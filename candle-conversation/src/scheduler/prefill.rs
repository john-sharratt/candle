use super::admit;
use super::admit::{Ground, Order};
use super::interleave;
use super::*;
use crate::projection::DecodePriority;
use crate::recorded_reply::replayed_step;
use std::time::Duration;

/// The engine as [`interleave::fill`] sees it: a cursor over both queues that
/// really claims what it takes.
///
/// Holds the scheduler mutably for the whole fill because every admission is a
/// device allocation and every check is a read of the partition that allocation
/// moved — there is no snapshot to work from, which is the entire point.
pub(super) struct WaveFill<'a> {
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
    /// Forward rows this fill admitted, beyond the head it admitted into.
    ///
    /// What the tier must be reserved for: `publish_tier_budget` publishes the
    /// tier of the wave that was *composed*, and the weight side leaves exactly
    /// that standing. Counted here rather than recovered from the queues
    /// afterwards because a prefill's admitted rows are its `advance`, not its
    /// whole turn, and only the fill knows which.
    admitted_rows: usize,
}

/// Consecutive placement refusals after which the refused wave's **started**
/// prefills are failed. Each refusal drops the wave and re-forms it against the
/// fresh gap, so a wave refused this many times running is one the partition
/// cannot hold at any width the group former can reach: a lone least chunk
/// with the weight side at its floor and nothing else in flight to finish and
/// free ground. Failing the chunk releases its ground so the pipeline moves.
pub(super) const TIER_REFUSALS_BEFORE_FAIL: usize = 8;

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
    /// for the ground the placement found, and the next wave is composed
    /// narrower.
    ///
    /// **The wave is dropped, held group and all.** A creep group lives across
    /// waves — its members, its layer cursor and its held residual — and the
    /// wave builder re-forms a group only when none is held. So a refusal that
    /// left the group standing was refused again at exactly the same width,
    /// whatever the budget said: the requeued prefills were re-admitted by the
    /// next fill, `build_wave_group_inputs` found them under the same ids, and
    /// the same rows went back to the same placement. Run 11 wedged there the
    /// moment ingest began — 199,704 refusals of one 320 MiB wave against a 213
    /// MiB gap, 157 waves with no forward, seven decodes never stepping, zero
    /// directories — while the margin the fill held back doubled to its cap and
    /// bounded nothing, because the group it was meant to narrow was never
    /// re-formed. Dropping the group is what "compose it narrower" requires:
    /// the next build reads the gap as it stands and forms a group to it, and
    /// the layers the creep had done are redone from zero, which is idempotent
    /// (a member re-feeds its whole chunk and commits its offset only at the
    /// head).
    ///
    /// The prefills the refused wave carried and had not started go **back to
    /// the front of the queue**, in order — they were admitted but never
    /// started, so there is nothing to unwind but the admission itself.
    /// Measured before this: one wave priced 26 MiB over a 4,054 MiB gap
    /// failed 18 directories.
    ///
    /// **A run of refusals is final for the prefills that had started.** A
    /// started prefill (one with chunks already committed) cannot be requeued,
    /// so it rides the next group — and if the placement refuses that group
    /// too, and the next, the wave never advances: measured, 1,641 refusals of
    /// one wave with nothing requeued. With the group re-formed against the
    /// fresh gap on every refusal, [`TIER_REFUSALS_BEFORE_FAIL`] refusals
    /// running mean the partition cannot hold even the least chunk, and the
    /// started prefills are failed with the numbers and their sequences
    /// released, so the pipeline moves. A placed forward ends the run
    /// (`note_wave_placed`).
    ///
    /// Nothing is bought here. The fill buys the least wave into the budget on
    /// the next pass (`WaveFill::publish_tier_budget`), and that pass runs every
    /// iteration; a second buyer at the refusal was the same purchase made
    /// twice from two places.
    pub(super) fn note_tier_refusal(&mut self, err: &candle::Error) {
        self.tier_refusal_streak = self.tier_refusal_streak.saturating_add(1);
        let final_for_started = self.tier_refusal_streak >= TIER_REFUSALS_BEFORE_FAIL;
        let started: Vec<usize> = self
            .wave_prefill_members
            .iter()
            .filter_map(|m| match m {
                WaveMember::Prefill { seq_id, .. } => Some(*seq_id),
                WaveMember::Section { .. } => None,
            })
            .collect();
        let held_rows = self.held_creep_rows();
        self.reset_wave_prefill();
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
        // **A requeued turn gives its store back.** `claim_recurrent` placed a
        // 160 MiB store when the turn was admitted, on the rule its own comment
        // states: "a slot waiting in the queue holds nothing". Putting the turn
        // back without releasing breaks that — the store sits with a queued turn
        // until it is admitted again, holding exactly the ground whose scarcity
        // caused the refusal.
        //
        // Safe because these are *unstarted*: `offset == 0`, no logits, so the
        // store holds only what the seed put there — the timeline snapshot, the
        // branch checkpoint, or the parent's copy — and `claim_recurrent`
        // re-materialises it from that same seed on re-admission. This is the
        // reasoning `rematerialise_demoted_prefill` already relies on: a turn
        // that has not run has advanced nothing.
        //
        // Measured on run 34: 10 refusals requeued 14 turns, and the holdings
        // census then showed 7 queued slots holding a store with `tokens=0` and
        // `kv_bytes=0` — ~1.1 GiB the expert zone could not grow into, which is
        // why it plateaued at 6,116 MiB instead of recovering.
        let mut released = 0usize;
        for p in &unstarted {
            let seq = p.work.sequence_id;
            if !self.model.recurrent_resident(seq.0) {
                continue;
            }
            match self.model.release_sequence(seq.0) {
                Ok(_) => released += 1,
                Err(e) => tracing::warn!(
                    target: "candle_conversation::scheduler::interleave",
                    slot = seq.0,
                    "requeued turn kept its recurrent store: {e}",
                ),
            }
        }
        // Back to the front, in their original order.
        for p in unstarted.into_iter().rev() {
            self.prefill_queue.push_front(p.work);
        }
        let mut failed = 0usize;
        if final_for_started {
            for p in self.active_prefills.iter_mut() {
                if p.error.is_none()
                    && p.final_logits.is_none()
                    && p.offset > 0
                    && started.contains(&p.work.sequence_id.0)
                {
                    p.error = Some(ConversationError::Channel(format!(
                        "prefill of {} tokens ({} committed): the wave transient tier refused \
                         its next chunk {} waves running — this partition cannot place it",
                        p.work.tokens.len(),
                        p.offset,
                        self.tier_refusal_streak,
                    )));
                    failed += 1;
                }
            }
        }
        self.prefill_head_blocked = false;
        let gap = transient_headroom_bytes(0).unwrap_or(0);
        tracing::warn!(
            target: "candle_conversation::scheduler::interleave",
            streak = self.tier_refusal_streak,
            requeued,
            failed,
            dropped_creep_rows = held_rows,
            gap_mib = gap >> 20,
            least_mib = self.min_forward_tier_bytes() >> 20,
            stores_released = released,
            "wave transient tier refused placement — wave dropped and requeued: {err}",
        );
        self.dump_on_tier_refusal(err);
    }

    /// How often a tier refusal may emit the full state dump.
    ///
    /// **A refusal arrives in bursts, so the dump has to be rate limited.** The
    /// refused wave is dropped and re-formed against the fresh gap, so a
    /// partition that cannot hold the wave refuses again on the next pass and
    /// the next — run BS produced 1,641 refusals of one wave and run 11 produced
    /// 199,704. One report is ~20 KB, so an ungated dump would write gigabytes
    /// and bury the very lines it exists to preserve.
    ///
    /// Long enough that a burst yields one dump, short enough that a refusal
    /// arriving minutes later in a different phase gets its own.
    const TIER_REFUSAL_DUMP_COOLDOWN: Duration = Duration::from_secs(30);

    /// Emit the whole instrumented state at a tier refusal, at most once per
    /// [`Self::TIER_REFUSAL_DUMP_COOLDOWN`].
    ///
    /// **The refusal is the one event whose cause is never in the refusal.** Its
    /// message carries the span geometry and nothing else, so "this wave is
    /// wider than the ground its admissions bought" cannot say which admissions,
    /// what the planner projected when it took them, what is holding the ground
    /// they wanted, or whether the wave was bandwidth-, latency- or
    /// compute-bound at the time. By the time the question is asked the run has
    /// moved on and the state is gone — which is exactly what happened to run
    /// 29's startup refusal, leaving nothing to trace.
    ///
    /// So this refreshes the report and writes it whole: the planner's learned
    /// constants and budget, the holdings census, the latency split, and the
    /// span's own accounting. The same structure `/v1/memory` serves, so a dump
    /// and a live query are read the same way.
    fn dump_on_tier_refusal(&mut self, err: &candle::Error) {
        let due = self
            .last_tier_refusal_dump
            .is_none_or(|t| t.elapsed() >= Self::TIER_REFUSAL_DUMP_COOLDOWN);
        if !due {
            return;
        }
        self.last_tier_refusal_dump = Some(Instant::now());
        // Refresh first: the published report is up to a wave old, and the
        // interesting difference is precisely what this wave did.
        self.publish_memory_report();
        let Some((report, _)) = memory_report::latest() else {
            tracing::debug!(
                target: "candle_conversation::scheduler::interleave",
                "tier refusal dump: no memory report is available in this build",
            );
            return;
        };
        match serde_json::to_string(&report) {
            Ok(json) => tracing::debug!(
                target: "candle_conversation::scheduler::interleave",
                streak = self.tier_refusal_streak,
                refusal = %err,
                "tier refusal state dump {json}",
            ),
            Err(e) => tracing::debug!(
                target: "candle_conversation::scheduler::interleave",
                "tier refusal dump could not be serialised: {e}",
            ),
        }
    }

    /// A wave's transient tier was placed and its forward ran: the refusal
    /// streak [`Self::note_tier_refusal`] counts is over.
    pub(super) fn note_wave_placed(&mut self) {
        self.tier_refusal_streak = 0;
    }

    /// Rows of the creep group held between waves — the prefill and section
    /// chunks whose residual is standing and which ride the next wave whole,
    /// whatever else the fill admits into it. Zero when no group is held, in
    /// which case the next build forms a fresh group.
    ///
    /// The fill counts these at the head of the wave it prices
    /// (`WaveFill::head_rows`): the wave's tier is one quantity sized to every
    /// row it carries, so an admission that priced only its own rows beside
    /// the decodes let its claims eat the gap the held rows needed. Run 11: a
    /// 300-row creep was standing, a fill admitted one prefill whose checkpoint
    /// install took eight regions off the gap, and the placement refused the
    /// held wave by four.
    pub(super) fn held_creep_rows(&self) -> usize {
        if self.wave_prefill_cursor == 0 && self.wave_prefill_residual.is_none() {
            return 0;
        }
        self.wave_prefill_members
            .iter()
            .map(|m| match *m {
                WaveMember::Prefill { advance, .. } | WaveMember::Section { advance, .. } => {
                    advance
                }
            })
            .sum()
    }

    /// Sequences in that creep group, which is what the head scores it at —
    /// one row of logits each, however many tokens the member advances.
    pub(super) fn held_creep_seqs(&self) -> usize {
        if self.wave_prefill_cursor == 0 && self.wave_prefill_residual.is_none() {
            return 0;
        }
        self.wave_prefill_members.len()
    }
}

/// Rows the fill reserves tier for: the wave it actually composed, floored at a
/// forward worth running.
///
/// **The reservation is what the weight side leaves standing** — everything
/// else in the frontier gap is reclaimed — so this figure becomes the width
/// ceiling of every later wave. Publishing the *minimum* made that a fixed
/// point: 176 MiB reserved, gap reclaimed to 176 + margin, priced at 145 rows,
/// 145 rows composed, 176 MiB reserved again, for four hours.
///
/// The floor stays because a fill that admitted nothing must still leave room
/// for a forward — a tier of zero runs no wave at all, narrow or otherwise
/// (run CB, `(no forwards)` with 73 slots admitted).
///
/// Pure, so the fixed point is testable without a device.
fn published_tier_rows(head_rows: usize, admitted_rows: usize, next_chunk: usize) -> usize {
    head_rows
        .saturating_add(admitted_rows)
        .max(head_rows.saturating_add(next_chunk))
}

/// The same wave, in the three units the plan prices from.
///
/// The rows [`published_tier_rows`] adds are all **prefill** rows — an
/// admission's advance, or the floor chunk standing in for one — so they widen
/// the prefill chain and add one scored row per sequence, not per token. Pure
/// for the same reason its row-count twin is.
fn published_tier_width(
    head: WaveWidth,
    admitted_rows: usize,
    admitted_seqs: usize,
    next_chunk: usize,
) -> WaveWidth {
    let extra_rows = published_tier_rows(0, admitted_rows, next_chunk);
    // Which of the two the `max` took decides how many sequences those rows
    // belong to: the admissions' own count, or the single sequence the floor
    // chunk is reserved for.
    let extra_seqs = if extra_rows == 0 {
        0
    } else if admitted_rows >= next_chunk {
        admitted_seqs
    } else {
        1
    };
    WaveWidth {
        prefill_rows: head.prefill_rows.saturating_add(extra_rows),
        decode_rows: head.decode_rows,
        scored_rows: head.scored_rows.saturating_add(extra_seqs),
        ..head
    }
}

/// Regions the fill buys so the least wave fits the tier **budget**: what
/// `need` (the least wave's tier plus the margin) lacks of `gap`, bounded by
/// `affordable` — how far the weight zone stands above the hold the gate
/// defends. Pure, so the two bounds are testable without a device.
pub(super) fn least_wave_purchase_regions(need: usize, gap: usize, affordable: u64) -> usize {
    let short = need.saturating_sub(gap).div_ceil(REGION_BYTES);
    let cap = (affordable / REGION_BYTES as u64) as usize;
    short.min(cap)
}

impl<'a> WaveFill<'a> {
    pub(super) fn new(sched: &'a mut Scheduler, optimal: u64) -> Self {
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
            admitted_rows: 0,
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

    /// Whether the first item the fill would offer does not fit the ground the
    /// K/V side holds above the hold — the signal the eviction pass runs on.
    ///
    /// Walks the bands in offer order and judges the first prefill or section
    /// found against the same headroom the fill prices with. Decodes are
    /// skipped: they are continuations whose ground was reserved at admission,
    /// and the fill steps them whatever the budget says. Nothing queued means
    /// nothing is short.
    fn head_needs_ground(&mut self) -> bool {
        let room = self.headroom();
        let mut order = Order::new();
        while let Some((prio, kind)) = order.next() {
            if kind == admit::Kind::Decode {
                continue;
            }
            if let Some(cost) = self.peek(kind, prio) {
                return cost.total() > room.free_kv;
            }
        }
        false
    }

    /// Rows `n` decodes put at the head of the wave: a drafted decode rides as
    /// a verify block of `1 + draft` rows in the prefill slot, and the draft is
    /// the model's ladder for that width.
    fn decode_rows(&self, n: usize) -> usize {
        n * (1 + self.sched.model.draft_budget(n))
    }

    /// Rows the next wave already carries before this fill adds anything: the
    /// decodes taken so far, and the creep group held from the last wave
    /// ([`Scheduler::held_creep_rows`]). The tier is one quantity sized to
    /// every row of the wave, so every admission is priced as an increment over
    /// this, and every purchase guards the whole of it.
    pub(super) fn head_rows(&self) -> usize {
        self.head_width().rows()
    }

    /// The same head, in the three units the plan prices from.
    ///
    /// A row count cannot say what the tier costs, because the phases these
    /// rows widen are not the same ones a prefill's rows widen. The decode
    /// group's rows price the decode chain and the head; a creep member's price
    /// the prefill chain but score only one row of logits however many tokens
    /// it advances.
    ///
    /// Every decode row is scored, speculative blocks included: a verify block
    /// scores all of its rows, because each is a prediction to compare a
    /// proposal against.
    /// The tier the wave **already in flight** reserves — its decodes and its
    /// held creep, before this fill admits anything.
    ///
    /// What `interleave::effective_weight_zone_bytes` raises the floor's
    /// reserve by, so the residency every admission is judged against reflects
    /// the tier the standing wave has actually taken rather than the flat 912
    /// MiB `MIN_ELASTIC_RESERVE` assumes. The tier each *new* admission adds is
    /// charged separately, through `admit::Cost::dislodged_bytes`.
    ///
    /// Deliberately the head and not the wave-so-far: the head is committed
    /// work, so it is a fact this decision reads, not an output it feeds back
    /// into itself.
    pub(super) fn standing_tier_bytes(&self) -> usize {
        let dtype = self.sched.session.activation_dtype();
        WavePlan::new(self.sched.model.wave_geometry(dtype)).tier_bytes(self.head_width())
    }

    pub(super) fn head_width(&self) -> WaveWidth {
        let decodes = self.decode_rows(self.decodes_taken.len());
        WaveWidth {
            prefill_rows: self.sched.held_creep_rows(),
            decode_rows: decodes,
            scored_rows: decodes + self.sched.held_creep_seqs(),
            // The scheduler composes waves, never replays: a verify replay is
            // the speculative driver's, priced where it stages.
            ..WaveWidth::default()
        }
    }

    /// The tier of the wave `cost` would be admitted into — the head as it
    /// stands plus this admission's rows — which is what the frontier gap must
    /// hold once the admission's claims have landed. `cost.activations` is the
    /// increment the gate charges; the placement sees the sum.
    pub(super) fn wave_tier_after(&self, cost: &admit::Cost) -> u64 {
        let dtype = self.sched.session.activation_dtype();
        let plan = WavePlan::new(self.sched.model.wave_geometry(dtype));
        (plan.tier_bytes(self.head_width()) as u64).saturating_add(cost.activations)
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
    /// Run CE carried it and reached 82 directories at 102 tok/s aggregate with
    /// a 150 median, where CG's one-row reserve produced single-sequence waves.
    ///
    /// **On the 4090 with the 35B that is 176–192 MiB**, measured from the
    /// fill's own `least_mib`. This said "~1.5 GiB" for a while, which is out by
    /// an order of magnitude and is the kind of stale figure that gets reasoned
    /// from rather than checked: the number matters because `WaveFill::headroom`
    /// subtracts it from every admission's room, so believing it was a gigabyte
    /// and a half makes the floor look unreachable when it is 176 MiB away.
    ///
    /// Reading it through the planner is what makes it portable: it follows the
    /// model's geometry and the activation dtype rather than being a byte count
    /// to re-derive per card.
    fn min_forward_tier_bytes(&self) -> u64 {
        self.sched.min_forward_tier_bytes()
    }

    /// Publish this wave's transient-tier budget, so the engine's slab packer
    /// prices against the same ground the fill did.
    ///
    /// **The co-batched wave is bounded here, and nowhere else.** The engine's
    /// slab packer bounds a *pure* prefill wave; a wave carrying decode rows
    /// takes its prefill group whole, so a scheduler that admitted freely built
    /// waves whose tier came to 6.3 GiB against a 6.1 GiB gap — every one of
    /// them refused, every one a failed directory.
    ///
    /// The budget is the frontier gap as it stands after the fill, less the
    /// margin a refusal widens. Each admission priced its rows' tier
    /// (`admit::Cost::activations`) and bought what the gap lacked
    /// (`Scheduler::buy_kv_ground`); the placement inside the forward, where
    /// the boundary may not move, then finds its ground already there. A wave
    /// whose tier will not fit is a wave that should be narrower, which is
    /// `admit::gate`'s answer, not the placement's to pay for.
    ///
    /// **One purchase closes the pass: the least wave the admitted set can run
    /// must fit the budget.** A claim is region-granular where its price is not —
    /// a section's 12 MiB of K/V opens a fresh arena in every layer that has no
    /// room in its current one, each a whole region off the top of the free
    /// list — so the gap an admission measured and bought for is a few regions
    /// narrower once its claims have landed. The wave then asks for its least
    /// chunk regardless ([`PREFILL_MIN_ADVANCE`] rows, so a wave with nothing
    /// else in it makes progress) and the placement refuses by those few
    /// regions: run 6 lost thirteen waves in two minutes, each 1–2 regions
    /// short with a gap of 250–400 MiB. So the fill, still between forwards and
    /// still the one buyer, asks for what the gap lacks of that least wave's
    /// tier **plus the margin the budget holds back** — the group former reads
    /// the budget, not the gap, and a purchase that stopped at the gap left the
    /// least chunk exactly one margin short of joining any wave with a decode in
    /// it (run 11: `budget=141..190 MiB` against a 192 MiB least wave, eight
    /// prefills admitted and unstarted while seven decodes stepped). The
    /// purchase is bounded by how far the zone stands above the hold, the same
    /// line the gate defends: a zone at its hold buys nothing, the wave carries
    /// its decodes alone, and what they finish makes the room.
    ///
    /// **The published figure is the wave admission actually composed, not the
    /// least one it could get away with.**
    ///
    /// This is what the weight side leaves standing: `spare_regions` reclaims
    /// the whole frontier gap *except* `least_tier_bytes`. So whatever is
    /// published here is, within a wave or two, the entire room the tier will
    /// ever have — and publishing one minimum forward made that a self-inflicted
    /// ceiling. The loop measured: the fill published `tier_bytes(head + 128)` =
    /// 176 MiB, the weight side reclaimed the gap down to 176 + margin = ~251
    /// MiB, `prefill_width_cap` priced that at 145 rows, the next fill composed
    /// 145 rows and published the same 176 MiB again. The gap sat at 251 MiB for
    /// four hours with 1,534 MiB of residency standing free, every offer refused
    /// `Cap { max_tokens: 145 }`, because nothing in the engine ever asked for a
    /// wider tier than the narrowest one that works.
    ///
    /// So the reservation follows the admission decision: the rows the fill
    /// admitted, on top of the head it was admitting into. Admission chooses
    /// width against residency and the rate it buys ([`super::admit`]); the tier
    /// then reserves for that width, rather than the width being dictated by
    /// last wave's reservation. Bounded exactly as before — the purchase below
    /// cannot take the zone under the floor — so a wave is still only as wide as
    /// the weight side can afford, which is the trade the rate model exists to
    /// judge.
    ///
    /// Never less than one least chunk, so a wave that admitted nothing can
    /// still place a forward (run CB: `tier=0MiB`, `(no forwards)` wave after
    /// wave with 73 slots admitted).
    pub(super) fn publish_tier_budget(&mut self) {
        let margin = TIER_MARGIN_REGIONS * REGION_BYTES;
        if self.sched.active_slots() > 0 || !self.decodes_taken.is_empty() {
            let dtype = self.sched.session.activation_dtype();
            let plan = WavePlan::new(self.sched.model.wave_geometry(dtype));
            let next_chunk = if self.sched.held_creep_rows() > 0 {
                0
            } else {
                PREFILL_MIN_ADVANCE
            };
            // The wave as composed: the head, plus every row this fill admitted,
            // and never narrower than a forward worth running.
            let width = published_tier_width(
                self.head_width(),
                self.admitted_rows,
                self.prefill_admitted.len(),
                next_chunk,
            );
            let least = plan.tier_bytes(width);
            // The same figure is what the weight side's growth leaves standing
            // in the gap, so the two sides agree on what the next wave needs.
            if let candle::DeviceLocation::Cuda { gpu_id } = self.sched.device.location() {
                set_least_tier_bytes(gpu_id, least);
            }
            let gap = transient_headroom_bytes(0).unwrap_or(0);
            let room = self.headroom();
            let affordable = room.zone.saturating_sub(room.floor());
            let short = least_wave_purchase_regions(least + margin, gap, affordable);
            if short > 0 {
                let conceded = self.sched.model.request_kv_ground(short);
                tracing::debug!(
                    target: "candle_conversation::scheduler::interleave",
                    least_mib = least >> 20,
                    margin_mib = margin >> 20,
                    gap_mib = gap >> 20,
                    affordable_mib = affordable >> 20,
                    short_regions = short,
                    conceded_mib = conceded >> 20,
                    "admission bought the budget the least placeable wave lacked",
                );
            }
        }
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
        let (gap, owed, budget) = self.sched.tier_budget_now();
        self.sched.session.set_tier_budget_bytes(budget);
        if let Some(stats) = self.sched.kv_regions() {
            tracing::debug!(
                target: "candle_conversation::scheduler::interleave",
                gap_mib = gap >> 20,
                budget_mib = budget >> 20,
                margin_mib = margin >> 20,
                owed_mib = owed >> 20,
                head_rows = self.head_rows(),
                live = stats.live,
                free = stats.free,
                blocked = stats.blocked,
                total = stats.total,
                "tier budget published",
            );
        }
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

/// Proposals a decode drafted this wave, from the verify rows it rode as.
///
/// A speculative step puts `1 + draft` rows through the prefill slot per
/// decoding sequence; the planner prices a decode's routed experts off that
/// width. An ordinary decode drafts nothing.
fn draft_of(decodes: usize, verify_rows: usize) -> usize {
    verify_rows.checked_div(decodes).unwrap_or(0)
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
    /// turn's tokens, the least chunk that rides a wave)`. Does not consume it.
    ///
    /// **The two token counts are different and both matter.** The KV claim is
    /// for the *whole turn* — every chunk of it lands in this sequence's cache
    /// and is never given back until the turn seals — while the tier is priced
    /// for the **least** chunk only ([`PREFILL_MIN_ADVANCE`]). Pricing the
    /// chunk and claiming the turn is what collapsed run BS: the gate
    /// authorised a quarter of what the allocator then took, the weight zone
    /// fell from 5,020 to 1,417 MiB against a 4,774 hold, and the expert hit
    /// rate went to 0.257.
    ///
    /// **Why the least chunk and not the whole chunk.** The tier is transient
    /// and the wave packs it to whatever gap stands free, so a wider chunk
    /// costs nothing the K/V side does not already hold. Pricing the whole
    /// chunk made admission *buy* that width from the weight side: on run 6 a
    /// 1,575-token turn priced a 1.0–1.8 GiB tier per admission, the fill
    /// bought it, and the zone went from 10,398 to 5,898 MiB for prefill batch
    /// width — resident experts traded for a wider forward. The least chunk is
    /// what the wave needs to make progress; the rest it takes only if free.
    pub(super) fn peek_prefill(&self, band: usize) -> Option<(usize, SequenceId, usize, usize)> {
        let from = self.prefill_cursor[band];
        let live_decodes = self
            .sched
            .active_decodes
            .values()
            .filter(|s| !s.finished)
            .count();
        for idx in from..self.sched.prefill_queue.len() {
            let w = &self.sched.prefill_queue[idx];
            if self.band_of(w.sequence_id) != band {
                continue;
            }
            // **A turn the decode side refused is not a candidate until a decode
            // finishes.** This is not the FIFO exception the module header
            // forbids — that rule is about never passing over an item because it
            // is *expensive*, and this item is not eligible at all. Offering it
            // costs a full re-prefill of its prompt to re-ask a question only a
            // departing decode can answer differently.
            if w.held_until_decodes_below
                .is_some_and(|n| live_decodes >= n)
            {
                continue;
            }
            let whole = w.tokens.len();
            return Some((idx, w.sequence_id, whole, whole.min(PREFILL_MIN_ADVANCE)));
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
        self.sched.per_block_kv_bytes()
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
        // One more sequence, scored once however many tokens it advances.
        let head = self.head_width();
        let after = WaveWidth {
            prefill_rows: head.prefill_rows + advance,
            scored_rows: head.scored_rows + 1,
            ..head
        };
        let dtype = self.sched.session.activation_dtype();
        let plan = WavePlan::new(self.sched.model.wave_geometry(dtype));
        let activations = plan.tier_bytes(after).saturating_sub(plan.tier_bytes(head)) as u64;
        admit::Cost {
            kv,
            recurrent,
            activations,
            // The rows the forward actually gains — the same `advance` the tier
            // was priced for. What the throughput model earns its copy back on.
            rows: advance,
            // The price is the same whether or not a decode follows; the caller
            // that knows which turn this is sets the fact (`WaveFill::peek`).
            decodes_after: false,
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
        let zone =
            interleave::effective_weight_zone_bytes(self.standing_tier_bytes()).unwrap_or(u64::MAX);
        // The KV side and the weight side share one elastic span: an admission
        // whose price exceeds the free regions buys the rest from the weight
        // zone (`Scheduler::buy_kv_ground`), and that is legitimate all the way
        // down to the hold, which is the line below which the model would start
        // streaming. Pricing against the free list alone refused admissions
        // with gigabytes standing above that line — measured on run BR's
        // calibration: three sequences a forward where every earlier run
        // carried six, 979 tokens against 1,958, and the phase aggregate down
        // 29%.
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
        // `weight_floor - live_end()`, and the weight floor *moves* — admission
        // buys ground from the weight side as it admits, down to the hold. So the gap
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

    /// Resident weights, in the **effective zone's** currency — the residency
    /// the weight side could hold, not the bytes the expert cache happens to
    /// have loaded into it.
    ///
    /// Those are different numbers and the choice matters. The cache's own
    /// `resident_weight_bytes` is what the copy term physically measures, and
    /// it is the wrong input here: it does **not** fall when a claim takes
    /// ground the weight side was about to fill, so admission would spend the
    /// whole free list before the model noticed anything had been dislodged,
    /// and only then discover the floor. The effective zone falls by exactly
    /// one region per region claimed (`interleave::achievable_weight_bytes`),
    /// which is the same identity the floor is defended in — so the rate, the
    /// dislodge and the floor are all read in one currency and the model cannot
    /// be shown ground twice. Runs BV and CA are what mixing the two costs.
    fn resident_weights(&self) -> u64 {
        interleave::effective_weight_zone_bytes(self.standing_tier_bytes()).unwrap_or(u64::MAX)
    }

    /// The creep group held from the last wave: rows the next forward carries
    /// whatever this fill admits.
    fn standing_rows(&self) -> usize {
        self.sched.held_creep_rows()
    }

    fn budget(&self) -> admit::Budget {
        let room = self.headroom();
        let tier_reserve = self.min_forward_tier_bytes();
        // **The widest wave the partition will actually compose**, which is the
        // only width worth judging a rate against. The model would widen until
        // residency stopped paying; the transient tier is the other bound, and
        // it is not optional — a wave whose tier cannot stand in the frontier
        // gap is refused placement and runs nothing at all, however good its
        // projected rate was.
        //
        // **Bounded by what the partition could hold, not by what it currently
        // does.** The gap as it stands is the wrong bound because the gap is
        // *this decision's own output*: `publish_tier_budget` reserves the tier
        // for the wave admission composed, and the weight side reclaims the
        // rest. Capping admission at the standing gap closes that into a loop
        // with only one fixed point — the narrowest wave that works. Measured:
        // 145 rows, held for four hours, with 1,534 MiB of residency free and
        // every offer refused `Cap { max_tokens: 145 }`.
        //
        // So the bound is the gap **plus what the weight side could concede
        // before reaching the floor**, which is what `buy_kv_ground` and the
        // purchase below will actually buy. Residency is still the real limit —
        // the floor is hard, and the rate model refuses a width that dislodges
        // more than it repays. What changes is that the tier follows the
        // decision instead of dictating it.
        let gap = transient_headroom_bytes(0).unwrap_or(0) as u64;
        let placeable = gap
            .saturating_add(room.free_kv)
            .saturating_add(tier_reserve);
        let dtype = self.sched.session.activation_dtype();
        admit::Budget {
            resident: self.resident_weights(),
            // The same line `headroom` nets out of the spendable ground: the
            // hold, the eviction margin that keeps the cache evictable, and a
            // useful forward's tier. The model enforces it as a hard refusal.
            floor: room.floor().saturating_add(tier_reserve),
            // **Every prefill row the wave may carry, the standing ones
            // included.** `prefill_width_cap` answers "rows available *beside*
            // the head", and the head already counts the held creep — but the
            // fill charges that creep to the wave as prefill rows, so a cap
            // that had also subtracted them would bind one creep group early,
            // every wave, and tighten exactly when a group is being carried.
            max_rows: self.sched.held_creep_rows().saturating_add(
                self.sched.model.prefill_width_cap(
                    dtype,
                    self.head_width(),
                    placeable.min(usize::MAX as u64) as usize,
                ),
            ),
            // The width the expert hit rate has walked to. It bounds the
            // decodes a wave carries, as it did before; inside it the rate
            // model decides whether another one is worth carrying.
            max_decodes: self.sched.wave_width,
        }
    }

    fn peek(&mut self, kind: admit::Kind, prio: DecodePriority) -> Option<admit::Cost> {
        use admit::Kind;
        let band = band_index(prio);
        match kind {
            Kind::Prefill => {
                let (idx, seq, whole, advance) = self.peek_prefill(band)?;
                // **The price is the prompt, and only the prompt.** Admitting a
                // prefill commits to prefilling this turn; the decode that
                // follows buys its own ground one lease at a time through
                // `renew_expired_leases`, starting with its first. So the price
                // is what the allocator is about to be handed for the prefill
                // rather than a guess at the whole turn's lifetime.
                //
                // Bundling the first lease in here priced the prefill correctly
                // and then handed it a decode seat no gate had approved: the
                // boundary was the one lease boundary that never asked, so
                // concurrency grew by however many prefills were admitted.
                //
                // What the turn's decode budget carries is not a price but a
                // *fact* — whether a decode follows at all — so admission can
                // judge this offer by the decode model too (`Cost::decodes_after`).
                // The price below is still the prompt and only the prompt.
                let decodes_after = self.sched.prefill_queue[idx].max_decode_tokens > 0;
                Some(admit::Cost {
                    decodes_after,
                    ..self.price(seq, whole, advance)
                })
            }
            Kind::Decode => {
                let seq = self.peek_decode(band)?;
                // **A decode step buys no K/V.** Its lease was reserved through
                // `ensure_capacity` at the lease boundary that granted it — the
                // renewal in `renew_expired_leases`, which either buys the next
                // lease or seals the turn short — so the blocks it writes into are
                // blocks it already owns; charging it
                // again is the same ground counted twice, and this time it
                // charges the tenant that cannot pay. Run CC showed what that
                // costs: once the budget closed, every decode was refused, so
                // prefills promoted into decodes that never stepped, nothing
                // generated, and 44 slots sat admitted with 52 queued behind
                // them while the engine ran forwards that produced no tokens.
                //
                // What a step *does* cost is the tier for its rows, which the
                // wave has to place whether or not the K/V is already there — so
                // that term stands and the K/V term is zero.
                //
                // **Its rows, not one row.** A drafted decode rides as a verify
                // block of `1 + draft` rows, and the draft is the model's ladder
                // for the width the wave will have — on the 35B, sixteen decodes
                // came to ~740 rows and a 1,056 MiB tier. Priced at one row
                // each, no admission bought that tier, the placement was four
                // regions short on every wave, and run 6 sat at `(no forwards)`
                // for a quarter of an hour with sixteen decodes admitted.
                let step = Scheduler::DECODE_CLAIM_TOKENS;
                let taken = self.decodes_taken.len();
                let head = self.head_width();
                // Every row of the block is a decode row and every one is
                // scored — a verify block compares a proposal per row.
                let decodes_after = self.decode_rows(taken + 1);
                let added = decodes_after.saturating_sub(head.decode_rows);
                let after = WaveWidth {
                    decode_rows: decodes_after,
                    scored_rows: head.scored_rows + added,
                    ..head
                };
                let dtype = self.sched.session.activation_dtype();
                let plan = WavePlan::new(self.sched.model.wave_geometry(dtype));
                let activations =
                    plan.tier_bytes(after).saturating_sub(plan.tier_bytes(head)) as u64;
                // The verify block this decode rides as, `1 + draft` rows — the
                // same width the tier above was priced for, and what the
                // throughput model reads its routed expert count from.
                let rows = self
                    .decode_rows(taken + 1)
                    .saturating_sub(self.decode_rows(taken));
                Some(admit::Cost {
                    kv: 0,
                    activations,
                    rows,
                    ..self.price(seq, step, step)
                })
            }
            Kind::Section => {
                // One band: sections carry no priority of their own.
                if prio != DecodePriority::Low {
                    return None;
                }
                let s = self.sched.section_queue.front()?;
                // The whole section's K/V on an empty scratch slot, a store, and
                // the tier for its least chunk — the same three terms as a
                // prefill, without a lease: nothing decodes.
                let whole = s.tokens.len();
                Some(self.price(s.sequence_id, whole, whole.min(PREFILL_MIN_ADVANCE)))
            }
        }
    }

    fn admit(&mut self, kind: admit::Kind, prio: DecodePriority, cost: admit::Cost) -> bool {
        use admit::Kind;
        let band = band_index(prio);
        match kind {
            Kind::Prefill => {
                let Some((idx, seq, whole, _advance)) = self.peek_prefill(band) else {
                    return false;
                };
                self.prefill_cursor[band] = idx + 1;
                // The whole turn, matching what `peek` priced. The ground is
                // bought first and then claimed: `claim_kv` ensures capacity, so
                // this reserves the ground rather than predicting it. The
                // generation's own ground is claimed a lease at a time at the
                // lease boundary, so the slot cannot later grow into ground the
                // gate never authorised.
                let reserve = whole;
                let wave_tier = self.wave_tier_after(&cost);
                self.sched.buy_kv_ground(&cost, wave_tier);
                if !self.sched.claim_kv(seq.0, reserve) || !self.sched.claim_recurrent(seq.0) {
                    return false;
                }
                self.prefill_admitted.push(idx);
                self.admitted_rows = self.admitted_rows.saturating_add(cost.rows);
                true
            }
            Kind::Decode => {
                let Some(seq) = self.peek_decode(band) else {
                    return false;
                };
                if let Some(at) = self.decode_order.iter().position(|id| *id == seq) {
                    self.decode_cursor[band] = at + 1;
                }
                // A step's K/V was reserved with its lease; what it buys is the
                // tier for its rows, which `peek` priced.
                let wave_tier = self.wave_tier_after(&cost);
                self.sched.buy_kv_ground(&cost, wave_tier);
                if !self.sched.claim_kv(seq.0, Scheduler::DECODE_CLAIM_TOKENS)
                    || !self.sched.claim_recurrent(seq.0)
                {
                    self.decodes_refused += 1;
                    return false;
                }
                self.decodes_taken.push(seq);
                true
            }
            Kind::Section => {
                let Some(pending) = self.sched.section_queue.pop_front() else {
                    return false;
                };
                let wave_tier = self.wave_tier_after(&cost);
                self.sched.buy_kv_ground(&cost, wave_tier);
                let seal_block_from = match self.sched.prepare_section_ingest(
                    pending.sequence_id,
                    pending.section_id,
                    &pending.prefix_section_ids,
                    &pending.tokens,
                ) {
                    Ok(from) => from,
                    // A section whose setup fails is answered and consumed —
                    // the band moves on to the next one rather than stopping
                    // on an item no later pass could set up either.
                    Err(e) => {
                        let _ = pending.response_tx.send(Err(e));
                        return true;
                    }
                };
                if !self
                    .sched
                    .claim_kv(pending.sequence_id.0, pending.tokens.len())
                    || !self.sched.claim_recurrent(pending.sequence_id.0)
                {
                    // Refused: back to the head, FIFO, and the band stops. The
                    // setup is idempotent — it truncates the slot before it
                    // injects — so the next pass repeats it.
                    self.sched.section_queue.push_front(pending);
                    return false;
                }
                let PendingSectionIngest {
                    sequence_id,
                    section_id,
                    tokens,
                    address,
                    debug_name,
                    in_collection,
                    response_tx,
                    ..
                } = pending;
                self.sched.active_section_ingests.push(ActiveSectionIngest {
                    sequence_id,
                    section_id,
                    tokens,
                    offset: 0,
                    seal_block_from,
                    address,
                    debug_name,
                    in_collection,
                    response_tx,
                    error: None,
                });
                self.admitted_rows = self.admitted_rows.saturating_add(cost.rows);
                true
            }
        }
    }
}

use crate::token_buffer::TokenBuffer;
use candle_nn::kv_cache::{
    is_tier_refusal, kv_ground_shortfall, least_tier_bytes, set_least_tier_bytes,
    transient_headroom_bytes, WavePlan, WaveWidth, REGION_BYTES, TIER_MARGIN_REGIONS,
};
use candle_transformers::models::batched_inference::PendingGlue;
use std::collections::{HashMap, HashSet};

/// The region quantum in bytes.
fn region_bytes() -> u64 {
    candle_nn::kv_cache::REGION_BYTES as u64
}

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
impl Scheduler {
    /// The K/V side's free ground right now, in bytes — what the memory report
    /// publishes as admission's ceiling.
    ///
    /// A count of the regions this process has claimed and not yet spent; no
    /// driver reading enters into it. The fill itself does not price against
    /// this figure — it prices against the effective weight zone
    /// (`WaveFill::headroom`), which already counts every free region — so this
    /// is telemetry, exact rather than forecast.
    pub(super) fn admit_budget_ceiling(&self) -> u64 {
        self.kv_regions().map_or(0, |s| {
            ((s.free + s.blocked) as u64).saturating_mul(region_bytes())
        })
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
            // A section in flight holds its K/V, its store and its tier rows
            // until it seals, exactly as a prefill does — so it is active, and
            // rule 1's "nothing running" is false while one runs.
            + self.section_ingest_width()
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

    /// Claim the next chunk of K/V for every slot whose decode lease has run out.
    ///
    /// **The lease is a chunked ground reservation, not a decision point.** A
    /// turn decoding to EOS generates an unbounded number of tokens, so
    /// admission cannot reserve the whole generation without massively
    /// over-reserving for turns that stop early. [`Self::DECODE_LEASE_TOKENS`]
    /// at a time is how a slot gets ground without predicting its length, and it
    /// is what licenses `Kind::Decode`'s `kv: 0` price — a step buys nothing
    /// because its lease already did.
    ///
    /// **Nothing is re-judged here.** The slot was decided once, at the
    /// prefill→decode boundary, where the rate model was asked whether the wave
    /// goes faster carrying this decode. Past that point the engine owes the
    /// turn the forwards that finish it: a slot is held from admission to
    /// completion, which is the invariant [`super::admit`] is built on.
    ///
    /// A refused claim seals the turn short rather than parking it — see the
    /// body for why that trade is the right one.
    pub(super) fn renew_expired_leases(&mut self) {
        let expired: Vec<SequenceId> = self
            .active_decodes
            .iter()
            .filter(|(_, s)| s.lease_expired && !s.finished)
            .map(|(&id, _)| id)
            .collect();
        if expired.is_empty() {
            return;
        }
        // **Seal the spent lease's chunks before pricing the next one.** A
        // lease is exactly `DECODE_LEASE_TOKENS` and that is a whole number of
        // chunks, so a decode arriving here has just completed eight of them in
        // the active formats and will never write into them again. Sealing
        // first is not merely tidy: it hands back the widest size classes
        // *before* `buy_kv_ground` below asks what the next lease costs, so the
        // renewal is priced against the ground this just freed rather than
        // competing with it. The tail chunk stays writable — the seal opens a
        // fresh one — so the decode continues without noticing.
        self.seal_completed_chunks(&expired);
        for slot in expired {
            let held = self.session.sequence_offset(slot.0).unwrap_or(0);
            let lease_kv = admit::cost::kv_bytes_for_advance(
                held,
                Self::DECODE_LEASE_TOKENS,
                self.per_block_kv_bytes(),
            );
            // **A running decode is never re-judged.** Its slot was decided
            // once, at the prefill→decode boundary, and the engine owes it the
            // forwards that finish it. Re-gating a continuation is what wedged
            // 40/40 waves in runs CB/CD, and parking one is what produced the
            // churn this replaced: 4–8 parks per directory at ~445 ms each,
            // every one resumed inside 40 ms because the park tested the floor
            // while the resume tested the hold. So the only question here is
            // whether the ground for the next lease can be had.
            //
            // **And that question is asked with no floor of its own.** The
            // `may_admit` check that used to bound this went with the gate, and
            // `buy_kv_ground` delegates to `request_kv_ground`, which defends the
            // expert cache's *survival* floor (~1,408 MiB) rather than the
            // throughput floor admission reasons about (~5,387 MiB) — so a burst
            // of renewals can concede far more weight ground than any refusal
            // would have allowed, and logs nothing, because a concession is not
            // an error. It is worst at startup, where `decode_side_can_carry`
            // has no history to judge from.
            //
            // Left deliberately unbounded. The alternative is refusing a running
            // decode its next lease, and a continuation that cannot write is
            // sealed mid-turn — user-visible output lost to protect a throughput
            // figure. Re-gating continuations is also what wedged 40/40 waves in
            // runs CB/CD. So the cost is paid here on purpose; what would fix it
            // properly is `request_kv_ground` learning the throughput floor, not
            // a test at this call site.
            let least_tier = self.min_forward_tier_bytes();
            self.buy_kv_ground(
                &admit::Cost {
                    kv: lease_kv,
                    recurrent: 0,
                    activations: 0,
                    // A renewal buys ground, not width: the turn was already
                    // riding the wave and its rows are unchanged.
                    rows: 0,
                    // A lease renewal IS the decode; there is no prefill here to
                    // judge on its behalf.
                    decodes_after: false,
                },
                least_tier,
            );
            if self.claim_kv(slot.0, Self::DECODE_LEASE_TOKENS) {
                self.renew_lease(slot);
                continue;
            }
            // **A refused renewal is the one case parking used to cover.** The
            // allocator has nothing to give, so this turn cannot write another
            // token — carrying on would put K/V into ground nobody claimed,
            // which the partition does not police and which surfaces as a wrong
            // number many layers later.
            //
            // So the turn is sealed where it stands. The tokens it produced are
            // real and its record is written; what it loses is the tail it had
            // not generated yet, and that is reported rather than hidden. This
            // should be rare by construction — the boundary gate is what stops
            // the over-admission that creates the shortage — so a run that logs
            // this often is telling us the boundary decision is too permissive.
            let generated = self
                .active_decodes
                .get(&slot)
                .map(|s| s.generated_tokens.len())
                .unwrap_or(0);
            if let Some(state) = self.active_decodes.get_mut(&slot) {
                state.lease_expired = false;
                state.finished = true;
            }
            tracing::warn!(
                target: "candle_conversation::scheduler::interleave",
                slot = slot.0,
                held,
                generated,
                lease_kv_mib = lease_kv >> 20,
                "decode lease could not be renewed — no ground for the next \
                 chunk; the turn is sealed short at the tokens it has",
            );
            self.completions = self.completions.saturating_add(1);
        }
    }

    /// Give `slot` a fresh lease in place: it keeps decoding.
    fn renew_lease(&mut self, slot: SequenceId) {
        if let Some(s) = self.active_decodes.get_mut(&slot) {
            s.lease_expired = false;
            s.lease_left = Self::DECODE_LEASE_TOKENS;
        }
    }

    fn claim_kv(&self, seq: usize, add: usize) -> bool {
        self.session.ensure_capacity(&[seq], add).is_ok()
    }

    /// The ground the next wave's transient tier may be sized against, **as it
    /// stands right now**: `(gap, owed, budget)` in bytes.
    ///
    /// The gap is `weight_floor − live_end`, the frontier the tier is placed
    /// against. Off it come the fixed margin for what moves between the build
    /// and the placement ([`TIER_MARGIN_REGIONS`]) and what the weight side is
    /// owed — when residency stands under its hold the gap is not the tier's to
    /// take, it is where the weight side grows back. The debt is measured against the
    /// effective zone, never the extent: the extent settles under the hold and
    /// nothing moves it back on its own, so a debt read from it never clears
    /// (run BW: a 532 MiB debt deducted every wave against a wholly healthy
    /// 6,112 MiB effective zone, the tier at zero, 82 slots admitted and 2–4
    /// sequences a forward).
    ///
    /// **Read when the wave is built, not when the fill ran.** The two are a
    /// wave quantum apart, and the section seals and the persistence thread's
    /// deferred arena creation run in between; each is a region claim, and with
    /// the scattered free regions spent those come off the top of the gap. A
    /// wave sized against the fill's figure was then two regions wide of the
    /// gap it found — 73 refused placements in twenty minutes of run 7 — while
    /// the same wave sized against the figure at build time simply packs two
    /// regions narrower.
    /// The tier this scheduler last published, or `0` off CUDA and before the
    /// first fill — which reads as "no wave in flight" and leaves the floor's
    /// constant binding.
    pub(super) fn published_tier_bytes(&self) -> usize {
        match self.device.location() {
            candle::DeviceLocation::Cuda { gpu_id } => least_tier_bytes(gpu_id),
            _ => 0,
        }
    }

    pub(super) fn tier_budget_now(&self) -> (usize, usize, usize) {
        let margin = TIER_MARGIN_REGIONS * REGION_BYTES;
        let optimal = interleave::optimal_weight_bytes().unwrap_or(0);
        // **Zero, deliberately.** This computes the tier budget, so raising the
        // reserve by the tier here would make the budget a function of itself —
        // the loop with one fixed point that this function's own note is about.
        // What a *new* admission's tier costs is charged where it belongs, in
        // `admit::Cost::dislodged_bytes`.
        let owed = interleave::effective_weight_zone_bytes(0)
            .map_or(0, |zone| optimal.saturating_sub(zone)) as usize;
        let gap = transient_headroom_bytes(0).unwrap_or(0);
        (gap, owed, gap.saturating_sub(margin).saturating_sub(owed))
    }

    /// Hold this wave's transient tier against the weight side's growth.
    ///
    /// The weight side takes spare K/V ground at phase 0 of every forward —
    /// after the scheduler has sized the wave against the frontier gap and
    /// before the tier is placed in it. Its growth term leaves standing
    /// whatever the pool's `least_tier_bytes` says (`spare_regions`), which the
    /// fill sets to the least useful forward. A wave packed wider than that,
    /// into ground that was free when it was composed, then lost it to the
    /// growth: run 9, `tier budget published gap=882`, `weight side took free
    /// KV regions gained=69 spare=8` four hundred microseconds later, and the
    /// 816 MiB tier refused by four regions. So the wave, once composed, is
    /// what the growth must leave — `width` is every row it carries, in the
    /// units each phase is priced from.
    fn hold_wave_tier(&self, width: WaveWidth) {
        let dtype = self.session.activation_dtype();
        let plan = WavePlan::new(self.model.wave_geometry(dtype));
        let tier = plan
            .tier_bytes(width)
            .max(self.min_forward_tier_bytes() as usize);
        if let candle::DeviceLocation::Cuda { gpu_id } = self.device.location() {
            set_least_tier_bytes(gpu_id, tier);
        }
    }

    /// The transient tier of the least forward worth running —
    /// [`PREFILL_MIN_ADVANCE`] rows, priced through the same planner that
    /// places the tier, so it follows the model's geometry and the activation
    /// dtype rather than being a byte count re-derived per card.
    pub(super) fn min_forward_tier_bytes(&self) -> u64 {
        let dtype = self.session.activation_dtype();
        let plan = WavePlan::new(self.model.wave_geometry(dtype));
        // The least forward is one sequence advancing `PREFILL_MIN_ADVANCE`
        // tokens, scored once.
        plan.tier_bytes(WaveWidth::prefill(PREFILL_MIN_ADVANCE, 1)) as u64
    }

    /// One 32-token K/V block across the model, in the formats a **live**
    /// sequence occupies — see `admit::cost` for the 3.7x that distinction is
    /// worth.
    pub(super) fn per_block_kv_bytes(&self) -> u64 {
        self.session.live_kv_block_bytes()
    }

    /// Buy from the weight side whatever of `cost` the K/V side does not hold
    /// free — **the one place the weight boundary is asked to move toward K/V.**
    ///
    /// The gate has already said this price stays above the residency the
    /// engine defends (`admit::gate`), so the purchase is bounded by the same
    /// accounting that admitted the item; nothing else in the engine buys.
    /// Runs between forwards, which is the only moment the boundary may move
    /// (`set_weight_floor` refuses while a wave generation is open).
    ///
    /// K/V blocks and a recurrent store take regions from anywhere on the free
    /// list; the wave transient tier stands only in the gap between the arena
    /// frontier and the weight floor, and the claims eat into that gap once the
    /// scattered free regions are spent — see [`kv_ground_shortfall`].
    ///
    /// **`wave_tier` is the tier of the whole wave this admission joins**, not
    /// the increment `cost.activations` charges the gate: the tier is one
    /// quantity sized to every row the wave carries, and the gap has to hold
    /// all of it after this admission's claims land. Guarding only the
    /// increment let each admission eat the gap down to its own few rows while
    /// a held creep needed the rest — run 11's refusals began on exactly that
    /// fill. Never less than a useful forward's (`min_forward_tier_bytes`), so
    /// the wave this admits into can always be placed.
    ///
    /// Answers the bytes conceded. A weight side at its own floor concedes
    /// less than asked, and the claims that follow then refuse — which is the
    /// gate's `stopped_on_weights`, arriving from the allocator rather than the
    /// arithmetic.
    pub(super) fn buy_kv_ground(&self, cost: &admit::Cost, wave_tier: u64) -> u64 {
        if cost.total() == 0 {
            return 0;
        }
        let Some(stats) = self.kv_regions() else {
            return 0;
        };
        let candle::DeviceLocation::Cuda { gpu_id } = self.device.location() else {
            return 0;
        };
        let regions = |bytes: u64| bytes.div_ceil(region_bytes()) as usize;
        // Regions a standing tier blocks count as free: phase 0 of the forward
        // this admits for releases that tier before any of its claims run.
        let free = stats.free + stats.blocked;
        let gap = regions(transient_headroom_bytes(gpu_id).unwrap_or(0) as u64);
        let claims = regions(cost.kv.saturating_add(cost.recurrent));
        let tier = regions(wave_tier.max(self.min_forward_tier_bytes()));
        let short = kv_ground_shortfall(claims, tier, free, gap);
        if short == 0 {
            return 0;
        }
        let conceded = self.model.request_kv_ground(short);
        tracing::debug!(
            target: "candle_conversation::scheduler::interleave",
            kv_mib = cost.kv >> 20,
            recurrent_mib = cost.recurrent >> 20,
            tier_mib = cost.activations >> 20,
            wave_tier_mib = wave_tier >> 20,
            free_regions = free,
            gap_regions = gap,
            short_regions = short,
            conceded_mib = conceded >> 20,
            "admission bought weight-side ground",
        );
        conceded
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
    pub(super) fn claim_recurrent(&mut self, seq: usize) -> bool {
        // **A slot's state is materialised here, not at open.** This is the
        // first moment the slot is actually going to run, so it is the first
        // moment holding a store is justified; a slot waiting in the queue
        // holds nothing, and a fill that cannot place one more store refuses
        // here with `Ok(false)` — a real refusal, where before it found the
        // store already claimed at conversation open and had nothing left to
        // decide. What the state is — the timeline's snapshot, the prompt
        // branch's checkpoint, a live parent's copy, or zeros — was recorded
        // at open (`RecurrentSeed`) and is resolved by `materialise_recurrent`.
        //
        // A view takes its parent's state by MOVE, not copy: a view is a linear
        // continuation of its parent — what it advances IS what the parent's
        // state becomes — and `finalize_view` moves it back. The parent is
        // materialised first because it usually carries no store between turns;
        // `move_recurrent` is tolerant of one that carries none, and the view
        // then starts from the sequence-start value through `admit_recurrent`'s
        // vacant arm, which is right.
        //
        // **"Usually" is exact, and the exception cost a run.** A seal evicts
        // the parent's store only on the branch that persisted a snapshot for
        // it. A transient timeline takes `Ok(None)` — deliberately no snapshot,
        // because nothing will ever resume it — and its parent therefore keeps
        // the device copy after its turn seals. That is correct while the slot
        // still holds the K/V the state was advanced over; it becomes 160 MiB
        // of unreachable ground the moment a demote pass truncates the slot,
        // which is why `release_slot_ground` takes the two together.
        let view_id = SequenceId(seq);
        if !self.model.recurrent_resident(seq) {
            match self.turn_views.get(&view_id).map(|st| st.parent_id) {
                Some(parent) => {
                    self.materialise_recurrent(parent);
                    // Read before the move: afterwards the parent holds none, and
                    // the coverage warning below has to say whether the parent was
                    // already short.
                    let parent_coverage = self.model.positional_coverage(parent.0);
                    if let Err(e) = self.model.move_recurrent(parent.0, seq) {
                        tracing::warn!(
                            target: "candle_conversation::scheduler::interleave",
                            seq,
                            parent = parent.0,
                            "recurrent state could not be moved onto the view — refused: {e}",
                        );
                        return false;
                    }
                    FORK_RECURRENT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    // The view borrows the parent's K/V blocks and now holds its
                    // per-position state, so the two have to agree at this moment.
                    // A view that starts short stays short for its whole life and
                    // says nothing until a selection past the identity threshold
                    // refuses — so the mismatch is named here, where the parent
                    // that caused it is still in hand. The parent's coverage
                    // decides where to look: a view short of a complete parent is
                    // a move that dropped rows; a view short of a parent that was
                    // already short is the parent's prefix having arrived
                    // unindexed.
                    if let Some(cov) = self.model.positional_coverage(seq) {
                        let parent_tokens = self.session.sequence_offset(parent.0).unwrap_or(0);
                        if cov < parent_tokens {
                            tracing::warn!(
                                parent = parent.0,
                                view = seq,
                                parent_tokens,
                                parent_coverage = ?parent_coverage,
                                view_coverage = cov,
                                "claim_recurrent: the view borrows {} token(s) of its \
                                 parent's history that its index does not cover",
                                parent_tokens - cov
                            );
                        }
                    }
                }
                None => self.materialise_recurrent(view_id),
            }
        }
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

    /// Fold one **complete** forward into the wave planner's estimates.
    ///
    /// A wave that swept every layer copied every non-resident expert once, so
    /// its wall-clock less its rows' compute is the copy — and the copy over
    /// the bytes is the effective rate the next wave is composed against. A
    /// decode wave that ran compute-bound teaches the layer time the same way;
    /// one that ran on the bus teaches nothing about it, and the planner
    /// declines it rather than learning a layer time from a queue.
    ///
    /// **Only complete sweeps.** The creep group rides a *window* of layers,
    /// `[cursor, win_end)`, held across waves — so `prefill_rows` is the creep's
    /// rows scaled by the fraction of the model it actually crossed, and a call
    /// that returned with the creep paused and no full-sweep member (no layer
    /// swept end to end) never reaches here. Feeding a partial sweep in whole
    /// would subtract compute the forward never did and read the copy as faster
    /// than the bus can go.
    pub(super) fn observe_wave_rate(
        &mut self,
        prefill_rows: usize,
        decodes: usize,
        draft: usize,
        elapsed: std::time::Duration,
    ) {
        // **What the residency actually was while that wave ran**, which is the
        // span less the live regions and less the tier the wave held. Teaching
        // the model the figure with a flat 912 MiB tier term credits the run
        // with residency it did not have, and every rate learned from it is a
        // rate at the wrong operating point. An observation, not a decision, so
        // reading the published tier here is a record of fact.
        let Some(resident) = interleave::effective_weight_zone_bytes(self.published_tier_bytes())
        else {
            return;
        };
        let secs = elapsed.as_secs_f64();
        self.wave_rate.observe_prefill(prefill_rows, resident, secs);
        if decodes > 0 {
            self.wave_rate
                .observe_decode(decodes, draft, resident, secs);
            // **What the promotion decision prices a decode at.** The draft
            // width is a property of the forward — how many verify rows the
            // speculation put through the slot — so it is not knowable at the
            // boundary, where the turn has not stepped yet. The last wave's
            // observed width is the honest stand-in: it is what this workload
            // actually drafts, measured, rather than a constant or a zero that
            // would under-price every speculative decode and admit more of them
            // than the model would allow.
            self.last_observed_draft = draft;
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
        // **The wave planner prices every decode's copy off this.** It is the
        // one figure in the decode model that cannot be derived — how much more
        // the LRU zone and the Markov prefetch are worth than the resident
        // fraction alone depends on what this workload routes to — and the
        // counters above are already measuring it. Fed as the raw interval rate
        // rather than the smoothed one: the planner does its own dampening, and
        // stacking two averages would make it answer a width change a dozen
        // waves after the residency that caused it.
        // The residency that hit rate was achieved at, same as above.
        if let Some(resident) = interleave::effective_weight_zone_bytes(self.published_tier_bytes())
        {
            self.wave_rate.observe_hit_rate(rate, resident);
        }
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
        self.observe_expert_hit_rate();
        // **Eviction runs at admission, before the measurement, and only when an
        // admission decision is due.** Admission prices against the free lists
        // this pass leaves behind, so a shed that ran anywhere else would hand
        // `admit::cost` a device that looks fuller than it is. It runs on
        // completions rather than on every wave because a wave where nothing
        // finished has nothing new to shed and nothing new to admit — that is
        // the fast path.
        //
        // A spent lease hands ground back, so park before anything measures the
        // device — and before the fill picks this wave's decode set, so a turn
        // at the end of its lease does not ride one more forward.
        self.renew_expired_leases();
        if self.admission_due() {
            // **Shed for a reason, and there are two.** The head of the queue
            // does not fit the ground the K/V side holds above the hold — then
            // the K/V of conversations between turns is handed back, and the
            // fill measures afterwards, so what this frees is ground admission
            // spends this pass rather than next. Or the engine is idle — nothing
            // running, nothing queued — and the weight side should have its
            // ground back so residency climbs to what the card can hold.
            //
            // Shedding on every due pass, as this did, demoted the K/V of a
            // conversation between two turns of one chain and lifted it straight
            // back for the next turn 241–650 ms later: 1,611 such round trips in
            // one run, each a hot→warm→hot migration and a claim under
            // pressure, for ground nobody had asked for. The head is judged
            // against the same headroom the fill prices with; decodes never
            // ask, because they are continuations whose ground was reserved at
            // admission and the fill steps them regardless.
            //
            // Calling it here is also what makes the admission pass the clock
            // for `IDLE_SLOT_DEMOTE_PASSES`. It ran once per wave, which paced
            // demotion by a quantity that lengthens under load — stretching the
            // grace exactly when ground is scarcest.
            let optimal = interleave::optimal_weight_bytes().unwrap_or(0);
            let head_short = WaveFill::new(self, optimal).head_needs_ground();
            let idle = self.active_slots() == 0
                && self.prefill_queue.is_empty()
                && self.section_queue.is_empty();
            if head_short || idle {
                self.demote_idle_slots();
                // **Then take back what the queue is sitting on.** The pass
                // above can only touch slots nothing has claimed; the ground
                // that actually matters on an ingest workload is held by turns
                // that are submitted and waiting, which it is forbidden to
                // reach. This one takes exactly the part of that the substrate
                // already has a copy of — see `demote_unadmitted_slots` for why
                // the rest is not ours to take.
                self.demote_unadmitted_slots();
            }
            self.take_census(head_short);

            // **Pack the span at every admission, not only under pressure.**
            // Recurrent stores relocate leftward here, so the live set is pulled
            // to the bottom of the span and `live_end` follows it down — and the
            // weight zone and the wave tier both grow into
            // `weight_floor − live_end()`, so what this frees, they get.
            //
            // Waiting until the zone is starved means packing a span that is
            // already full, which is when there is least free ground to move
            // into and most live data to move. Packing continuously keeps the
            // extent tight so the pressure does not arrive, and each pass is
            // cheap because the previous one left little to do. It declines
            // rather than fails when there is nothing to move, and refuses to
            // buy ground to compact with.
            if let Err(e) = self.model.compact_span() {
                tracing::debug!("span compaction skipped: {e}");
            }
            // **Nothing that moves chunk bytes runs here.** The defrag and the
            // arena compaction both used to, and both now live on the
            // persistence thread beside the hot→warm migrate — the one other
            // thing that reads those bytes through captured addresses. Sharing
            // a thread with it is what makes them exclusive; running here left
            // a mover free to relocate a chunk while the migrate's kernel was
            // dereferencing a pointer to it, which is silent and decodes as
            // nonsense a turn later. See `persistence::thread`'s mover block.
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
        // (`admit::cost`), and admitted while they make the wave *faster*
        // (`admit::rate`) without crossing the residency the engine defends
        // (`admit::gate`). A slot admitted here is held to completion — prefill
        // through its chunks, then the same slot decoding to EOS — and its
        // reaping is what frees the next admission.
        //
        // The planner rides the pass as a value and is written back with what
        // it learned. It is a few hundred bytes of counters and estimates, and
        // the fill borrows the scheduler whole — a second borrow would have to
        // be threaded through every `Ground` method to save the copy.
        let optimal = interleave::optimal_weight_bytes().unwrap_or(0);
        let mut rate = self.wave_rate.clone();
        let (filled, decodes, refused, admitted) = {
            let mut ground = WaveFill::new(self, optimal);
            let filled = admit::fill(&mut ground, &mut rate);
            // After the fill: its purchases widened the frontier gap, and the
            // wave the engine now builds is packed against the gap as it stands.
            ground.publish_tier_budget();
            (
                filled,
                ground.decodes_taken,
                ground.decodes_refused,
                ground.prefill_admitted,
            )
        };
        self.wave_rate = rate;
        // Kept for the next pass's census: a fill that stopped on the tier's
        // width cap is throttled even though the head fits.
        self.last_fill_stopped_on_weights = filled.stopped_on_weights;
        self.last_fill_stopped_on_rate = filled.stopped_on_rate;
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
            || filled.sections > 0
            || !self.prefill_queue.is_empty()
            || !self.section_queue.is_empty()
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
                effective_mib = interleave::effective_weight_zone_bytes(0).unwrap_or(0) >> 20,
                decodes = decodes.len(),
                decodes_refused = refused,
                prefills = filled.prefills,
                sections = filled.sections,
                queued_sections = self.section_queue.len(),
                stopped_on_weights = filled.stopped_on_weights,
                stopped_on_rate = filled.stopped_on_rate,
                projected_tok_per_s = (self.wave_rate.rate(
                    self.wave_rate.tokens(),
                    self.wave_rate.resident_now(),
                ) as u64),
                wave_rows = self.wave_rate.tokens(),
                copy_gb_per_s = (self.wave_rate.effective_bytes_per_s() / 1e9 * 100.0).round() / 100.0,
                layer_ms = (self.wave_rate.layer_secs() * 1e5).round() / 100.0,
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
            // **A demoted turn is put back before it runs.** Its block tables
            // were given up while it waited; the projection it carries rebuilds
            // them from the substrate. Done here rather than inside the fill
            // because it is a projection apply and a view carve — real work,
            // and the fill is arithmetic and claims.
            if work.demoted {
                if let Err(e) = self.rematerialise_demoted_prefill(&mut work) {
                    tracing::warn!(
                        target: "candle_conversation::scheduler::interleave",
                        slot = work.sequence_id.0,
                        "a demoted turn could not be rebuilt; failing it rather than \
                         prefilling against an empty slot: {e}",
                    );
                    let _ = work.event_tx.send(TurnEvent::Error(e));
                    continue;
                }
            }
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

    /// Rebuild a turn whose block tables were given back while it waited.
    ///
    /// The inverse of the truncation in [`Self::demote_unadmitted_slots`], and
    /// the same sequence `reproject_view_complete` performs for a live decode:
    /// re-apply the stored projection to the parent, then carve a fresh view
    /// from the rebuilt parent. Carving mints a **new** `SequenceId` — that is
    /// how the session's view API works — so the turn's own id moves with it and
    /// the scheduler's maps follow.
    ///
    /// The turn is failed rather than run if this cannot be done: prefilling
    /// against an empty slot produces a fluent answer about nothing, which is
    /// the one outcome worse than losing the turn.
    fn rematerialise_demoted_prefill(
        &mut self,
        work: &mut PrefillWork,
    ) -> Result<(), ConversationError> {
        let old_view = work.sequence_id;
        let Some(state) = self.turn_views.get(&old_view).cloned() else {
            return Err(ConversationError::Channel(format!(
                "rematerialise: slot {old_view} has no view state to rebuild from"
            )));
        };
        let parent_id = state.parent_id;
        let t = std::time::Instant::now();

        // **Give the old view back before carving its replacement.** Carving
        // mints a new sequence, so the one being replaced has to be released or
        // every rebuild leaks a slot — and, far worse, the 160 MiB recurrent
        // store bound to it, which nothing can then reach. Measured when this
        // was missing: 37 rebuilds left 49 stores standing against 13 live
        // slots, 7.8 GB of orphaned state, the weight zone at 137 MiB, and the
        // engine deadlocked with nothing running and 96 turns queued.
        //
        // The store **moves** to the parent rather than being dropped: a view
        // is a linear continuation of its parent, so what it advanced is what
        // the parent's state becomes, and `claim_recurrent` gives it back to
        // the new view at admission. Move before free, or there is nothing left
        // to read — the same ordering `reproject_view_complete` documents.
        //
        // A turn that has not run has advanced nothing, so a failure to move is
        // not fatal here: the state is the parent's own and the new view will
        // take it either way. The free and release are what must happen.
        if let Err(e) = self.model.move_recurrent(old_view.0, parent_id.0) {
            tracing::debug!(
                target: "candle_conversation::scheduler::interleave",
                view = old_view.0,
                parent = parent_id.0,
                "rematerialise: no recurrent state to move back: {e}",
            );
        }
        let _ = self.session.free_sequence(old_view.0);
        let _ = self.model.release_sequence(old_view.0);

        // The parent's prefix, from the substrate, exactly as submit built it —
        // lifted back into VRAM first, as submit lifted it. The wait that gave
        // this turn's blocks back is exactly the window in which the turns it
        // selected get demoted to RAM, and a unit that is not hot cannot be
        // injected: `apply_projection` refuses it rather than build a context
        // that silently lacks it.
        if let Some(conversation) = self.slot_conversations.get(&parent_id).cloned() {
            let (sections, turns) = projection_assembler::projection_working_set(&work.projection);
            self.elevate_projection_working_set(&conversation, &sections, &turns, "rematerialise");
        }
        self.apply_projection(parent_id, BlockCount(0), &work.projection)?;

        // A fresh view over every block the rebuilt parent now holds.
        let parent_blocks = self.session.sequence_block_count(parent_id.0).unwrap_or(0);
        let ranges: Vec<BlockRange> = if parent_blocks == 0 {
            Vec::new()
        } else {
            vec![BlockRange::new(0, parent_blocks)]
        };
        let (new_view, borrowed) = self.create_view(parent_id, &ranges)?;

        // Re-key: the turn is the same turn, on a new slot.
        self.turn_views.remove(&old_view);
        self.slot_tokens.remove(&new_view);
        self.turn_views.insert(
            new_view,
            ViewState {
                parent_id,
                original_borrowed: borrowed,
                turn_start_parent_blocks: borrowed.0,
                question_tokens: state.question_tokens,
            },
        );
        work.sequence_id = new_view;
        work.demoted = false;

        // **The admission's claims were made against the slot just freed.**
        //
        // `admit` reserves this turn's ground with `claim_kv`/`claim_recurrent`
        // keyed on `work.sequence_id` — and for a demoted turn that is
        // `old_view`, because the rebuild runs *after* the fill has judged and
        // claimed. Both are released above with the view, so without this the
        // turn prefills into a slot holding no K/V reservation at all: it grows
        // into ground admission never bought, which is the elastic boundary's
        // whole failure mode. The store is worse — `move_recurrent` handed it to
        // the parent expecting `claim_recurrent` to hand it back at admission,
        // and admission has already been and gone, so it stays on the parent
        // where no reclamation pass can reach it.
        //
        // Re-issuing here rather than reordering the fill keeps the fill what
        // its own doc says it is — arithmetic and claims, no projection work —
        // and the price is already paid: `buy_kv_ground` ran against the same
        // figures, so this is re-keying a reservation, not taking a second one.
        let reserve = work.tokens.len();
        if !self.claim_kv(new_view.0, reserve) || !self.claim_recurrent(new_view.0) {
            return Err(ConversationError::Channel(format!(
                "rematerialise: slot {new_view} could not retake the ground admission \
                 bought for {old_view} (reserve {reserve} tokens)"
            )));
        }

        tracing::debug!(
            target: "candle_conversation::scheduler::interleave",
            old_view = old_view.0,
            new_view = new_view.0,
            parent = parent_id.0,
            parent_blocks,
            segments = work.projection.len(),
            rebuild_ms = t.elapsed().as_millis() as u64,
            "demoted turn rebuilt for admission",
        );
        Ok(())
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

    /// Give back the K/V of conversations that are **queued but not admitted**.
    ///
    /// # The population
    ///
    /// A turn submitted to the engine materialises its projection onto a slot
    /// and carves a view **at submit time**, long before admission chooses to
    /// run it. With a directory scan queueing ninety turns at once, ninety
    /// slots are holding K/V for work that has not started and may not start
    /// for minutes — and [`Self::demote_idle_slots`] cannot touch any of them,
    /// because a queued prefill and a live view both put a slot in its busy
    /// set. Measured: ~96 of 121 slots held that way, ~3.8 GB of K/V, while the
    /// weight zone sat 600 MiB above its hold for four hours with eviction
    /// running on every pass and finding nothing it was allowed to take. The
    /// backlog was holding the residency that draining the backlog needed.
    ///
    /// # What it takes, and what it must not
    ///
    /// **Only what the substrate already has.** `evict_hot_to_free` drops a
    /// residence only when it holds *both* a hot and a warm copy, so this can
    /// never lose K/V whose migration is still in flight — which is the whole
    /// hazard, the hot→warm pass being asynchronous. What it drops reloads as a
    /// PCIe copy, never a recompute.
    ///
    /// **It truncates the block tables too**, which is where the ground
    /// actually is. Measured on this workload: every holder reported
    /// `hot = 0 MiB` — the substrate keeps no VRAM copy once a turn is sealed —
    /// while 178 waiting slots referenced 8.7 GB of arena chunks. Evicting only
    /// the substrate-hot half therefore freed nothing at all, ever.
    ///
    /// That is safe **only because the turn carries the projection it was built
    /// from** ([`PrefillWork::projection`]): every segment is substrate-pinned
    /// or a generated template, so `rematerialise_demoted_prefill` puts the slot
    /// back exactly as it was when the fill admits it. A turn with no stored
    /// projection is never demoted — there would be nothing to rebuild from.
    ///
    /// **Both the view and its parent go, or neither does.** A view borrows its
    /// parent's chunks as `Arc` clones, so truncating one table drops references
    /// the other still holds and frees not one byte. That is why this is done as
    /// a pair, and why the earlier substrate-only pass could never have worked
    /// even if the hot tier had been populated.
    ///
    /// The keep-list is every running slot's working set, so a conversation
    /// reached by a live fork keeps everything that fork attends.
    ///
    /// Returns what was freed.
    pub(super) fn demote_unadmitted_slots(&mut self) -> crate::substrate::EvictionReport {
        use super::holdings::{running_slots, waiting_only_slots};

        let mut freed = crate::substrate::EvictionReport { count: 0, bytes: 0 };
        let census = self.census();
        let waiting = waiting_only_slots(&census);
        if waiting.is_empty() {
            return freed;
        }
        let running = running_slots(&census);

        // Everything any *running* slot attends is protected, whichever
        // conversation holds it.
        //
        // **The two sides are keyed differently, and matching them directly
        // matches nothing.** `running_slots` reports the slots doing work, which
        // are *views*; `slot_projection_state` is keyed by the **parent** a view
        // was carved from — as `evict_finished_prefill` shows when it clears the
        // working set through `parent`. Intersecting them left the keep-list
        // permanently empty, so the sweep below ran as
        // `evict_hot_to_free(&[], &[], …)` and dropped the hot copies of turns
        // that were actively decoding: the working set is a *protect*-list, and
        // an empty one protects nothing.
        //
        // A parent is attended exactly when one of its views is, so the running
        // set is widened to the owners before it is asked.
        let mut running_owners: HashSet<SequenceId> = running.iter().copied().collect();
        for id in &running {
            if let Some(v) = self.turn_views.get(id) {
                running_owners.insert(v.parent_id);
            }
        }
        let mut keep_sections: Vec<SectionId> = Vec::new();
        let mut keep_turns: Vec<TurnKey> = Vec::new();
        for (id, st) in &self.slot_projection_state {
            if running_owners.contains(id) {
                keep_sections.extend(st.working_set.sections.iter().copied());
                keep_turns.extend(st.working_set.turns.iter().copied());
            }
        }

        let live_before = self.kv_regions().map(|s| s.live).unwrap_or(0);
        let t = std::time::Instant::now();
        let mut touched = 0usize;

        // **Which queued turns may be given back.** Only a prefill still in the
        // queue, carrying the projection it was built from — that list is what
        // rebuilds it, and a turn without one (a resume, a compression
        // re-prefill, a section) has nothing to rebuild from and is left alone.
        let mut pairs: Vec<(SequenceId, SequenceId)> = Vec::new();
        let waiting_set: HashSet<SequenceId> = waiting.iter().copied().collect();
        for (i, w) in self.prefill_queue.iter().enumerate() {
            if w.demoted || w.projection.is_empty() || !waiting_set.contains(&w.sequence_id) {
                continue;
            }
            // The view and its parent go together or not at all: a view holds
            // its parent's chunks as `Arc` clones, so truncating one table drops
            // references the other still holds and frees nothing.
            let Some(parent) = self.turn_views.get(&w.sequence_id).map(|s| s.parent_id) else {
                continue;
            };
            if !waiting_set.contains(&parent) {
                continue;
            }
            pairs.push((w.sequence_id, parent));
            let _ = i;
        }

        let mut stores_released = 0usize;
        let store_bytes = self.model.recurrent_store_bytes() as u64;
        for (view, parent) in &pairs {
            // Truncating to zero blocks releases this slot's chunk handles; the
            // arena ground returns once the last holder — the other half of the
            // pair — has let go too. The recurrent store goes with the blocks:
            // see `release_slot_ground` for why the two are one operation.
            for slot in [view, parent] {
                if self.release_slot_ground(*slot) {
                    stores_released += 1;
                }
            }
            // The working set was a protect-list for ground that is now gone.
            if let Some(st) = self.slot_projection_state.get_mut(parent) {
                st.working_set.sections.clear();
                st.working_set.turns.clear();
                st.glue_islands.clear();
                st.pending_user_part = None;
            }
            self.slot_tokens.remove(view);
            touched += 1;
        }

        // Mark the work so admission rebuilds it before the prefill runs.
        let demoted: HashSet<SequenceId> = pairs.iter().map(|(v, _)| *v).collect();
        for w in self.prefill_queue.iter_mut() {
            if demoted.contains(&w.sequence_id) {
                w.demoted = true;
            }
        }

        // And the substrate's own hot copies — **for every waiting conversation,
        // not only the pairs whose block tables were demotable.**
        //
        // This used to run over `pairs` alone, on the reasoning that "on this
        // workload they do not [hold hot copies] — a sealed turn's VRAM copy is
        // already gone". That was true when it was written and stops being true
        // as an ingest deepens: measured on run 36, 180 waiting slots held
        // 9,306 MiB of hot copies of which **2,939 MiB was already evictable**,
        // while this pass freed 2 MiB because it found one demotable pair out of
        // 181 waiting slots.
        //
        // The two things are independent and were wrongly coupled. Truncating a
        // block table needs the view *and* its parent to be waiting and the turn
        // to carry a projection to rebuild from — a narrow population. Dropping a
        // hot copy needs only that no running slot attends it, which is exactly
        // what `keep_sections`/`keep_turns` encode, and `evict_hot_to_free`
        // refuses anything whose warm copy has not landed
        // (`EvictionReport::not_durable`). So the keep-list, not the pair rule,
        // is what makes this safe.
        let mut evicted_convs = 0usize;
        // Keyed on the **conversation**, not the slot. The guard exists because
        // one conversation can back several waiting slots and its hot copy
        // should be swept once; keying on the slot made every insert unique, so
        // the guard never fired, the sweep ran once per slot, and
        // `convs_evicted` counted slots — the very figure the log line below
        // uses to judge whether the keep-list is over-protecting.
        // **Counted per slot, and named for it.** This carried a `seen` guard
        // meant to charge one conversation once, keyed on the slot — and slots
        // are unique per entry, so it never fired and the figure was slots all
        // along. `projection::Conversation` is a resolver handle with no id to
        // key on, so rather than plumb one through for a log line the guard is
        // gone and the metric says what it counts. Sweeping the same
        // conversation from two of its waiting slots is idempotent:
        // `evict_hot_to_free` returns `bytes == 0` the second time, so only the
        // first is tallied.
        for slot in &waiting {
            let Some(conv) = self.slot_conversations.get(slot).cloned() else {
                continue;
            };
            let r = conv
                .write()
                .evict_hot_to_free(&keep_sections, &keep_turns, u64::MAX);
            if r.bytes > 0 {
                evicted_convs += 1;
            }
            freed.count += r.count;
            freed.bytes += r.bytes;
        }

        let arenas = self.session.release_empty_arenas().unwrap_or(0);
        let live_after = self.kv_regions().map(|s| s.live).unwrap_or(0);
        self.wave_stats.add_evict(
            freed.bytes,
            freed.count as u64,
            t.elapsed().as_millis() as u64,
        );
        if touched > 0 {
            tracing::debug!(
                target: "candle_conversation::persistence::tier",
                waiting_slots = waiting.len(),
                pairs_demoted = touched,
                stores_released,
                store_mib = (stores_released as u64 * store_bytes) >> 20,
                residences_evicted = freed.count,
                hot_freed_mib = freed.bytes >> 20,
                // Waiting slots the hot-copy sweep actually took something from,
                // against `waiting_slots` offered. A run where this stays near
                // zero while `held_mib` is large means the keep-list is
                // protecting more than the running set really attends.
                slots_evicted = evicted_convs,
                held_mib = (census.waiting_only.kv_bytes + census.waiting_only.hot_bytes) >> 20,
                arenas_released = arenas,
                live_before,
                live_after,
                regions_freed = live_before.saturating_sub(live_after),
                "unadmitted demote: gave back the block tables of queued turns",
            );
        }
        freed
    }

    /// Take and publish the holdings census when it would say something.
    ///
    /// **Which of the eleven busy-set sources is holding the span** is the
    /// question every hard diagnosis of this engine has turned on, and it is
    /// not recoverable after the fact — a run that ends wedged leaves a log
    /// full of `busy_slots=96` and no way to learn what the 96 were.
    ///
    /// Taken whenever the engine is **throttled**, not only when the head does
    /// not fit. Those are different states and the difference is the whole
    /// point: a fill can stop on the tier's width cap with gigabytes of
    /// residency standing free, in which case the head fits perfectly and the
    /// engine is still admitting nothing. Keying the census on the head alone
    /// made it silent through exactly that — 1,534 MiB of room, every offer
    /// refused `Cap { max_tokens: 145 }`, and no census taken.
    ///
    /// Costs nothing on a healthy run: no throttle, no census.
    pub(super) fn take_census(&mut self, head_short: bool) {
        let throttled =
            head_short || self.last_fill_stopped_on_weights || self.last_fill_stopped_on_rate;
        if !throttled {
            return;
        }
        let census = self.census();
        // **Orphans, not idle stores.** The census counts a store as idle when
        // no holder is running, which conflates a turn waiting to decode — its
        // state legitimately held, and the bulk of the figure — with a store
        // nothing can reach. Reading the conflated number as waste is how
        // 1,120 MiB of live turns got mistaken for a leak. This one is only the
        // unreachable kind, so a non-zero value is a defect and not occupancy.
        let orphans = self.orphaned_store_slots();
        let orphan_mib = (orphans.len() as u64 * self.model.recurrent_store_bytes() as u64) >> 20;
        tracing::debug!(
            target: "candle_conversation::scheduler::holdings",
            head_short,
            stopped_on_weights = self.last_fill_stopped_on_weights,
            stopped_on_rate = self.last_fill_stopped_on_rate,
            orphan_stores = orphans.len(),
            orphan_mib,
            "{}", census.summary(),
        );
        if !orphans.is_empty() {
            tracing::warn!(
                target: "candle_conversation::scheduler::holdings",
                orphan_stores = orphans.len(),
                orphan_mib,
                slots = ?orphans,
                "recurrent stores no live work can account for — ground that \
                 nothing will read and nothing will free",
            );
        }
        super::holdings::publish(census);
    }

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
        busy.extend(self.section_queue.iter().map(|s| s.sequence_id));
        busy.extend(self.prefill_queue.iter().map(|w| w.sequence_id));
        busy.extend(self.pending_reprojections.iter().copied());
        busy.extend(self.deferred_glue_fires.iter().map(|p| p.parent_id));
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
        // Stores handed back, which is the larger half of a demotion — see the
        // release in the loop below.
        let mut stores_released = 0usize;
        // Read **before** the releases: `recurrent_store_bytes` is the max over
        // the stores that currently exist, so asking after the loop reports the
        // map the loop just emptied — which logged `store_mib=0` against a real
        // release of 160 MiB.
        let store_bytes = self.model.recurrent_store_bytes() as u64;
        // Slots that still held blocks, so the log can separate the two
        // populations this pass now covers: a slot mid-conversation giving its
        // block table back, and one already truncated whose bytes are entirely
        // in the substrate's hot copies. Only the second was ever the large one.
        let mut slots_with_blocks = 0usize;
        for (id, offset) in pass.demote {
            // A slot already at zero blocks has nothing to truncate, but its
            // conversation's hot turn residences are the bytes this pass is
            // actually after — so it goes on the demoted list either way. The
            // store goes with the blocks in one call; `busy` holds every view
            // and every view's parent, so a slot reaching here has no live turn
            // to cut the state out from under.
            if offset > 0 {
                slots_with_blocks += 1;
            }
            if self.release_slot_ground(id) {
                stores_released += 1;
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
        let t_evict = std::time::Instant::now();
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
        // The dashboard's eviction band: this is the engine's one eviction pass.
        self.wave_stats.add_evict(
            freed.bytes,
            freed.count as u64,
            t_evict.elapsed().as_millis() as u64,
        );
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
            stores_released,
            store_mib = (stores_released as u64 * store_bytes) >> 20,
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
    /// targets ([`Self::prefill_pass_budget`]). Without this the whole active set
    /// coalesces into one forward: the 93-section tool catalog (~21k tokens)
    /// packed into a single pass whose transient activation spiked VRAM to the
    /// card ceiling and paged. Sections beyond the budget ride the next chunk —
    /// the wave loop pumps until every section seals — so throughput is unchanged
    /// (each forward still fills to the expert-amortization target) while the peak
    /// stays bounded.
    ///
    /// **And bound the rows to the tier the fill left this wave**, exactly as
    /// `form_wave_group` does for a dialogue cohort. The tier is placed in the
    /// gap between the arena frontier and the weight floor, and nothing buys
    /// ground at placement; a batch packed to the token cap alone asked for a
    /// 160 MiB tier against a 112 MiB gap on run 5 and failed every wave for
    /// the rest of the run with forty-two sections admitted and idle. `head_rows`
    /// is what the wave already carries ahead of the sections. The first
    /// section always gets its least chunk whatever the rows say, so a wave with
    /// nothing else in it makes progress and the placement is the judge of that
    /// chunk.
    #[allow(clippy::type_complexity)]
    pub(super) fn build_section_batch(
        &mut self,
        head_rows: usize,
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
        let cap = self.prefill_pass_budget();
        // The head is decode and verify rows, every one of them scored.
        let mut rows_left = self.model.prefill_width_cap(
            self.session.activation_dtype(),
            WaveWidth::decode(head_rows),
            self.session.tier_budget_bytes(),
        );
        let mut seq_ids: Vec<usize> = Vec::with_capacity(active.len());
        let mut inputs: Vec<Tensor> = Vec::with_capacity(active.len());
        let mut group_idxs: Vec<usize> = Vec::with_capacity(active.len());
        let mut advances: Vec<usize> = Vec::with_capacity(active.len());
        let mut batch_tokens = 0usize;
        for &i in &active {
            let s = &mut self.active_section_ingests[i];
            let off = s.offset;
            let remaining = s.tokens.len() - off;
            let least = remaining.min(PREFILL_MIN_ADVANCE);
            let advance = if seq_ids.is_empty() {
                remaining.min(cap).min(rows_left.max(least))
            } else {
                if rows_left < least {
                    break;
                }
                remaining.min(cap).min(rows_left)
            };
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
                    rows_left = rows_left.saturating_sub(advance);
                }
                Err(e) => {
                    s.error = Some(ConversationError::Model(e));
                }
            }
        }
        if seq_ids.is_empty() {
            return None;
        }
        tracing::debug!(
            target: "candle_conversation::scheduler::interleave",
            sections = seq_ids.len(),
            rows = batch_tokens,
            rows_left,
            head_rows,
            budget_mib = self.session.tier_budget_bytes() >> 20,
            "section batch formed",
        );
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
            // A section leaving the active set is a completion as far as
            // admission is concerned: its scratch slot's ground comes back when
            // the caller frees it, and the next fill is due rather than taking
            // the fast path.
            self.completions = self.completions.saturating_add(1);
            if let Some(e) = s.error {
                let _ = s.response_tx.send(Err(e));
                continue;
            }
            let result = self.finalize_section_ingest(
                s.sequence_id,
                s.section_id,
                s.seal_block_from,
                Arc::new(s.tokens.to_vec()),
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
    /// section chunks bounded by the per-forward pass budget
    /// ([`Self::prefill_pass_budget`]). Section chunks join
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
        // The per-forward pass budget: the configured target, bounded by what
        // the model can run in one forward. A 15.2k-token Cline turn entered
        // whole once asked for a 13.2 GiB transient tier on a 16 GB card; in
        // chunks of this size it creeps instead.
        let cap = self.prefill_pass_budget();
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

    /// The `[decode | verify]` prefix of a wave's logits — every row of it, or
    /// an error naming what is missing.
    ///
    /// **A short head is a broken contract, not a small answer.** The wave
    /// promised `head_rows` scored rows: one per decode row and one per verify
    /// row, named to the head through `set_verify_row_seqs` before the forward
    /// opened. Clamping the prefix to whatever came back (`head_rows.min(len)`)
    /// turned that into a shorter vector and handed it on, so the shortfall
    /// surfaced downstream as `split_block_rows` reporting a row count that
    /// matched nothing — a symptom carrying none of the composition that
    /// produced it, and no clue whether the missing rows were decode or verify.
    ///
    /// More rows than `head_rows` is ordinary: the creep's and glue's rows
    /// follow the head, and the caller slices them off separately.
    ///
    /// **A caller must settle its own co-batched members before returning this
    /// error.** The rows after the head are positioned by `head_rows`, so a
    /// short head misplaces every creep and section row behind it too; those
    /// members are failed, never completed from misaligned logits and never left
    /// half-advanced.
    fn head_logits(
        logits: &[Tensor],
        head_rows: usize,
        n_dec: usize,
        verify_tok: usize,
        segment: &str,
    ) -> candle::Result<Vec<Tensor>> {
        if logits.len() < head_rows {
            candle::bail!(
                "co-batch wave ({segment}): the head scored {} rows for a wave whose head is \
                 {head_rows} ({n_dec} decode + {verify_tok} verify) — every named row is \
                 scored or the verify split downstream cannot be trusted",
                logits.len(),
            );
        }
        Ok(logits[..head_rows].to_vec())
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
        // **The tier budget is read here, as the wave is built.** The fill
        // published one a quantum ago; the section seals and the persistence
        // thread's arena creation have claimed since, and a wave sized against
        // the stale figure is refused by exactly what they took. The session
        // carries the fresh figure so the engine's own slab packer prices
        // against the same ground this wave was composed for.
        let (_, _, tier_budget) = self.tier_budget_now();
        self.session.set_tier_budget_bytes(tier_budget);
        let (members, seq_ids, inputs, prefill_gidxs) = if !self.wave_cohort_advanced {
            if cursor == 0 && self.wave_prefill_residual.is_none() {
                // The prefill rows the tier holds beside this wave's head.
                let prefill_rows = self.model.prefill_width_cap(
                    self.session.activation_dtype(),
                    WaveWidth::decode(head_rows),
                    tier_budget,
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
                self.build_section_batch(head_rows)
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
            let sec_tok: usize = sec_inputs
                .iter()
                .map(|t| t.dims().get(1).copied().unwrap_or(0))
                .sum();
            // Decode and verify rows are all scored; each section scores one
            // row however many tokens it advances, and glue scores none — it
            // only scattered K/V.
            self.hold_wave_tier(WaveWidth {
                prefill_rows: sec_tok + glue_tok,
                decode_rows: head_rows,
                scored_rows: head_rows + sec_seqs.len(),
                ..WaveWidth::default()
            });
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
            self.note_wave_placed();
            if has_glue {
                self.reconcile_wave_offsets(glue_seqs)?;
            }
            let logits = out.logits_owned()?;
            // **The section is settled before the head is checked.** It was
            // written by the forward that just ran — every row's K/V, whichever
            // rows the head went on to score — and it completes from its
            // advances, not its logits. So its progress is real even when the
            // head comes back short, and checking the head first would return
            // with the section flagged advanced but never completed.
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
            // A full `[0, N)` sweep: every member crossed every layer, so the
            // rows are counted whole.
            self.observe_wave_rate(
                sec_adv.iter().sum(),
                n_dec,
                draft_of(n_dec, verify_tok),
                t_wave.elapsed(),
            );
            return Self::head_logits(&logits, head_rows, n_dec, verify_tok, "one-shot");
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

        // Every segment of this wave places a tier for the rows it carries; the
        // widest — head, creep and glue together — is what the growth must
        // leave standing across all of them.
        self.hold_wave_tier(WaveWidth {
            prefill_rows: creep_tok + glue_tok,
            decode_rows: head_rows,
            // Every decode and verify row is scored; each creep member scores
            // one row however many tokens it advances, and glue scores none.
            scored_rows: head_rows + inputs.len(),
            ..WaveWidth::default()
        });

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
            self.note_wave_placed();
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
            let dec_logits = match Self::head_logits(&logits, head_rows, n_dec, verify_tok, "seg2")
            {
                Ok(d) => d,
                Err(e) => {
                    // The creep's rows sit behind the head, so a short head
                    // misplaces them too. Drop the group exactly as a failed
                    // seg2 forward does, rather than complete it from
                    // misaligned logits or leave it with its residual taken,
                    // its cursor above zero and nothing to resume from.
                    self.fail_wave_group(&members, &prefill_gidxs, &e);
                    return Err(e);
                }
            };
            // The creep's own rows follow the head. They are still clamped: a
            // creep member that scored nothing is a member the wave paused
            // rather than a contract broken, and `complete_wave_group` reads
            // only as many as came back.
            let creep_end = (head_rows + members.len()).min(logits.len());
            let member_logits = logits[head_rows..creep_end].to_vec();
            self.complete_wave_group(&members, &member_logits);
            // The full-sweep members crossed every layer; the creep crossed
            // `[cursor, N)` of them, so its rows count in that proportion.
            self.observe_wave_rate(
                creep_tok * n.saturating_sub(cursor) / n,
                n_dec,
                draft_of(n_dec, verify_tok),
                t_wave.elapsed(),
            );
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
        let seg3_logits = seg3.logits_owned()?;
        let logits = Self::head_logits(&seg3_logits, head_rows, n_dec, verify_tok, "seg3")?;
        // Segments 1–3 together are one `[0, N)` sweep for the full-sweep
        // members; the creep crossed `[cursor, win_end)`.
        self.observe_wave_rate(
            creep_tok * win_end.saturating_sub(cursor) / n,
            n_dec,
            draft_of(n_dec, verify_tok),
            t_wave.elapsed(),
        );
        Ok(logits)
    }

    /// Handle a device-OOM from the ragged prefill forward: the batch was too
    /// wide for the card. Cut the admission budget (so subsequent waves admit
    /// less) and surface the error on each in-batch prefill's caller channel.
    ///
    /// The hardest evidence the controller gets — a forward that actually failed
    /// — so it acts immediately here rather than waiting for the next fill.
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
            // A turn that will never decode never finalizes its view, so the
            // view is released here or not at all.
            if let Some(e) = error {
                let _ = work.event_tx.send(TurnEvent::Error(e));
                // Reclaim the carved view, or the sequence wedges forever: the
                // view was registered in `turn_views` before the prefill ran and
                // never reached `active_decodes`, so a dangling one makes the
                // parent's next `SubmitTurn` wind-down refuse with `TurnInFlight`
                // for good. Every prefill-error path drains through here.
                self.discard_turn_view(work.sequence_id);
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
                    self.discard_turn_view(work.sequence_id);
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

    /// Whether the decode side can carry one more turn — the promotion
    /// decision, asked of the same model every other admission answers to.
    ///
    /// **The offer is a counterfactual.** The slot is already resident: its K/V
    /// was claimed when the prefill was admitted and its store placed by
    /// `claim_recurrent`. Offering it as it stands would ask the model whether
    /// to admit something already present, double-counting its ground, and could
    /// only answer yes. So residency-before is reconstructed by adding the
    /// slot's own bytes back, and residency-after is what actually stands —
    /// exactly the shape `admit::fill` hands over for a fresh admission.
    ///
    /// **A turn with nothing to rebuild from is always carried.** An evicted
    /// turn is replayed from [`PrefillWork::projection`]; a resume, a
    /// compression re-prefill and a section carry none, and evicting one would
    /// lose it outright. Keeping it is the only safe answer, and it is a real
    /// exception rather than a corner case.
    fn decode_side_can_carry(&mut self, work: &PrefillWork) -> bool {
        if work.projection.is_empty() {
            return true;
        }
        let slot = work.sequence_id;
        let held_tokens = self.session.sequence_offset(slot.0).unwrap_or(0);
        let held = admit::Cost {
            kv: admit::cost::kv_bytes_for_advance(0, held_tokens, self.per_block_kv_bytes()),
            recurrent: if self.model.recurrent_resident(slot.0) {
                self.model.recurrent_store_bytes() as u64
            } else {
                0
            },
            activations: 0,
            // Rows are the wave's to price when it composes; what is being
            // judged here is whether to carry the turn at all.
            rows: 0,
            // This IS the promotion question, asked of a turn whose prefill is
            // already done — only `claimed_bytes` is read from this cost.
            decodes_after: false,
        }
        .claimed_bytes();

        // Residency as it actually stands, the standing wave's tier included —
        // this is a rate judgement, and the tier is ground the weight side does
        // not have while the wave holds it.
        let after = interleave::effective_weight_zone_bytes(self.published_tier_bytes())
            .unwrap_or(u64::MAX);
        let before = after.saturating_add(held);
        let decodes = self.active_decodes.values().filter(|s| !s.finished).count();
        let draft = self.last_observed_draft;
        match self.wave_rate.judge_promotion(draft, before, after) {
            admit::rate::Admit::Admitted { .. } => true,
            admit::rate::Admit::Refused(refusal) => {
                tracing::debug!(
                    target: "candle_conversation::scheduler::interleave",
                    slot = slot.0,
                    held_tokens,
                    held_mib = held >> 20,
                    resident_mib = after >> 20,
                    decodes,
                    draft,
                    refusal = ?refusal,
                    "prefill refused promotion to decode — evicting the turn",
                );
                false
            }
        }
    }

    /// Set a refused turn down: give its ground back and put it in the queue to
    /// be prefilled again when the decode side has room.
    ///
    /// **Both the view and its parent, or nothing is freed.** A view borrows its
    /// parent's chunks as `Arc` clones, so truncating one table drops references
    /// the other still holds and returns not one byte — the same pairing
    /// [`Self::demote_unadmitted_slots`] documents.
    ///
    /// **And the store, which is the larger half.** One store is 160 MiB against
    /// a whole turn's K/V in the tens; releasing it is what actually moves the
    /// weight boundary. `rematerialise_demoted_prefill` *moves* the store to the
    /// parent rather than dropping it, which is right for a turn that is coming
    /// straight back — but a turn being set down for an unknown time must hand
    /// the ground over, and its state is re-derived by the replay.
    fn evict_finished_prefill(&mut self, mut work: PrefillWork) {
        let slot = work.sequence_id;
        let parent = self.turn_views.get(&slot).map(|v| v.parent_id);
        for s in [Some(slot), parent].into_iter().flatten() {
            self.release_slot_ground(s);
        }
        // The working set was a protect-list for ground that is now gone.
        if let Some(p) = parent {
            if let Some(st) = self.slot_projection_state.get_mut(&p) {
                st.working_set.sections.clear();
                st.working_set.turns.clear();
                st.glue_islands.clear();
                st.pending_user_part = None;
            }
        }
        self.slot_tokens.remove(&slot);
        // Marked so admission rebuilds it before the prefill runs.
        work.demoted = true;
        // **Held until the decode side is genuinely narrower.** The refusal was
        // about carrying another turn at this width, so the turn waits until
        // fewer decodes are running — the one event that changes the answer.
        // See the field's docs for why the general completion counter was too
        // loose to serve.
        //
        // **A hold of zero is a wedge, not a wait.** `peek_prefill` skips while
        // `live_decodes >= n`, so `Some(0)` is "wait until fewer than zero
        // decodes are running" — never true, and the turn is never offered
        // again while its caller blocks on a handle that will not complete.
        // Reachable whenever the decode set has already drained by the time the
        // refusal lands, which is exactly the `Refusal::Floor` case at the
        // prefill→decode boundary. With nothing running there is no departing
        // decode to wait for, so the turn is simply re-queued.
        let live = self.active_decodes.values().filter(|s| !s.finished).count();
        work.held_until_decodes_below = (live > 0).then_some(live);
        self.prefill_queue.push_back(work);
        // Ground came back, so the next fill is due rather than taking the fast
        // path.
        self.completions = self.completions.saturating_add(1);
    }

    /// Post-forward path shared by both single and batched prefill: sample
    /// the first token, emit it, and either transition to decode or close
    /// the turn out immediately on EOS / max_decode_tokens == 0.
    fn finalise_prefill(
        &mut self,
        mut work: PrefillWork,
        logits: Tensor,
        prefill_ms: f64,
        turn_start: Instant,
        token_count: usize,
    ) {
        // **Seal what the prefill just wrote, before a single decode token
        // lands on top of it.** This is the largest single block of active-
        // format K/V the turn will ever hold — the whole prompt, in R16/F16 at
        // roughly 3.7× its quantized size — and until now it stayed that way
        // until the turn ended. Sealing here hands the widest size classes back
        // at the one moment the turn is guaranteed to be between writers, and
        // the decode that follows opens a fresh chunk on top.
        self.seal_completed_chunks(&[work.sequence_id]);

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

        let sampled = match self.sample_single(&logits, &sampling, &mut sampling_state) {
            Ok(t) => t,
            Err(e) => {
                self.sampling_states
                    .insert(work.sequence_id, sampling_state);
                let _ = work.event_tx.send(TurnEvent::Error(e));
                return;
            }
        };
        // A replayed turn opens with its recording's first id, whatever the
        // prefill's logits chose.
        let first_token = match (work.recorded_reply.as_deref(), self.eos_tokens.first()) {
            (Some(reply), Some(&eos)) => replayed_step(reply, 0, eos),
            _ => sampled,
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
            // The first token ended the turn: an end-of-sequence, or a budget of
            // zero decoded tokens.
            let finish = if self.is_eos(first_token) {
                FinishReason::Stop
            } else {
                FinishReason::Length
            };
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
                    finish,
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
                    recorded_reply: work.recorded_reply.map(Replay::new),
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
                    finish,
                );
            }
            return;
        }

        // ── The one decision the decode side gets ───────────────────────────
        //
        // **Prefill and decode want opposite widths, and this is the only place
        // they can differ.** Prefill amortises one all-expert copy over as many
        // rows as it can get, so it wants many slots; decode pays per concurrent
        // turn in routed experts, so it wants few. Without a decision here the
        // prefill population simply *becomes* the decode population — the width
        // chosen for one end of the curve is imposed on the other, which is what
        // ran decode concurrency to 23 while the weight zone sat on its floor.
        //
        // The turn is offered to the same model every admission answers to, at
        // the one moment the answer is both knowable and free to act on: the
        // prefill is complete, and **no decode work has been done yet.** The
        // first token was sampled from the prefill forward's own final row, so
        // turning the turn away here costs nothing already computed for decode.
        //
        // Refused means evicted — the turn is set down and re-prefilled later,
        // when the decode side has room for it. It is not parked: parking is
        // reversible and so needs the state kept somewhere, which cost 190 MB of
        // substrate per directory and churned 4–8 times per directory at ~445 ms
        // a park, every one resumed within 40 ms.
        //
        // Decided *before* the token is emitted, because an evicted turn will
        // sample its own first token again when it re-prefills; emitting here
        // and then evicting would deliver it twice.
        if !self.decode_side_can_carry(&work) {
            self.evict_finished_prefill(work);
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
                // A once-trigger (the think block) is spent by firing, so the
                // rest of the turn decodes that token as text.
                if let Some(rest) = work.triggers.after_firing(first_token) {
                    work.triggers = Arc::new(rest);
                }
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
            // **A finished prefill starts its decode with a spent lease.**
            // The prefill→decode transition is a lease boundary like every
            // other, so it is decided by the same code: `park_expired_leases`
            // runs at the head of the next wave, before the fill picks its
            // decode set, and either buys this turn its first lease through
            // the gate or parks it on the prefix the prefill just sealed.
            //
            // Handing the first lease out here instead is what let decode
            // concurrency grow without a decision: the gate was not
            // consulted until `DECODE_LEASE_TOKENS` had already been
            // generated, so every admitted prefill became a resident decode
            // and the count was a residue of the prefill rate rather than
            // anything chosen.
            //
            // What that costs is a correlation rather than a controlled
            // result, so it is recorded as one. Over a single run the
            // concurrency, the expert zone and the hit rate moved together:
            // `active` 1.0 → 13.6 while `zone` fell 10,026 → 5,879 MiB and
            // the hit rate 0.864 → 0.644. The mechanism that would explain
            // it — more turns in flight routing more distinct experts per
            // layer, so the working set outgrows the zone — is the one the
            // width half of `lease_verdict` acts on, and capping
            // concurrency is the experiment that would confirm it.
            lease_left: 0,
            lease_expired: true,
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
            finish: FinishReason::Stop,
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
            recorded_reply: work.recorded_reply.map(Replay::new),
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
        let pass = self.prefill_pass_budget();
        let logits = if tokens.len() > pass {
            let mut last_logits: Option<Tensor> = None;
            for chunk in tokens.chunks(pass) {
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

        // Nothing to bring up to date here: the prefill's own commit
        // (`KvCache::commit_written_tokens`, the one place every write outside
        // the decode kernel goes through) marked the cached decode slot buffer,
        // and the next sync that reads it re-serialises its writer region.
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
mod published_tier_tests {
    use super::{published_tier_rows, PREFILL_MIN_ADVANCE};

    /// **The reservation follows the wave that was composed.** Publishing the
    /// minimum instead is a fixed point: the weight side reclaims the gap to
    /// whatever is published, the next wave is priced from that gap, and it can
    /// never be wider than the last. Measured at 145 rows for four hours with
    /// 1,534 MiB of residency standing free.
    #[test]
    fn the_published_width_is_the_wave_that_was_admitted() {
        // A fill that admitted eight 128-row chunks onto a 200-row head asks
        // for all of it, not for one chunk.
        assert_eq!(
            published_tier_rows(200, 8 * 128, PREFILL_MIN_ADVANCE),
            200 + 1_024,
        );
        // And the width it asks for grows with what it admits, which is the
        // property the old figure did not have.
        let widths: Vec<usize> = [1usize, 4, 16]
            .iter()
            .map(|n| published_tier_rows(200, n * 128, PREFILL_MIN_ADVANCE))
            .collect();
        assert!(widths.windows(2).all(|w| w[1] > w[0]), "{widths:?}");
    }

    /// **A fill that admitted nothing still leaves room for a forward.** A tier
    /// of zero runs no wave at all — run CB sat at `(no forwards)` with 73 slots
    /// admitted — so the floor is one least chunk above the head.
    #[test]
    fn a_fill_that_admitted_nothing_still_reserves_a_useful_forward() {
        assert_eq!(
            published_tier_rows(200, 0, PREFILL_MIN_ADVANCE),
            200 + PREFILL_MIN_ADVANCE,
        );
        assert_eq!(
            published_tier_rows(0, 0, PREFILL_MIN_ADVANCE),
            PREFILL_MIN_ADVANCE
        );
        // A held creep takes no new member, so its next chunk is zero — and the
        // head it is already carrying is what must be placed.
        assert_eq!(published_tier_rows(300, 0, 0), 300);
    }

    /// Admitted rows below the floor do not shrink the reservation under it.
    #[test]
    fn a_narrow_admission_never_reserves_less_than_the_floor() {
        assert_eq!(
            published_tier_rows(100, 8, PREFILL_MIN_ADVANCE),
            100 + PREFILL_MIN_ADVANCE,
        );
    }
}

#[cfg(test)]
mod least_wave_purchase_tests {
    use super::least_wave_purchase_regions;
    use candle_nn::kv_cache::REGION_BYTES;

    /// The purchase is what the *budget* lacks — the least wave plus the
    /// margin — so a gap that already holds the least wave but not the margin
    /// still buys the margin's worth. Stopping at the gap is what left the
    /// least chunk one margin short of every wave with a decode in it.
    #[test]
    fn the_margin_is_bought_along_with_the_least_wave() {
        let least = 11 * REGION_BYTES;
        let margin = 4 * REGION_BYTES;
        assert_eq!(
            least_wave_purchase_regions(least + margin, least, u64::MAX),
            4,
            "gap holds the least wave exactly: the margin is what is missing"
        );
        assert_eq!(
            least_wave_purchase_regions(least + margin, least + margin, u64::MAX),
            0,
            "budget already holds the least wave: nothing to buy"
        );
        assert_eq!(
            least_wave_purchase_regions(least + margin, least + margin + 1, u64::MAX),
            0
        );
    }

    /// A fraction of a region short buys a whole region — the boundary moves in
    /// regions and the placement rounds the tier up to one.
    #[test]
    fn a_partial_region_short_buys_a_whole_one() {
        assert_eq!(
            least_wave_purchase_regions(3 * REGION_BYTES + 1, 3 * REGION_BYTES, u64::MAX),
            1
        );
    }

    /// The hold bounds the purchase: a zone standing this far above the hold
    /// gives up at most that, and a zone at or under it gives nothing — the
    /// wave then runs its decodes alone and what they finish makes the room.
    #[test]
    fn the_purchase_stops_at_the_hold() {
        let need = 20 * REGION_BYTES;
        assert_eq!(
            least_wave_purchase_regions(need, 0, 3 * REGION_BYTES as u64),
            3,
            "three regions above the hold: three bought of twenty wanted"
        );
        assert_eq!(
            least_wave_purchase_regions(need, 0, 0),
            0,
            "at the hold: nothing"
        );
        assert_eq!(
            least_wave_purchase_regions(need, 0, REGION_BYTES as u64 - 1),
            0,
            "less than a region above the hold is not a region to sell"
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
mod wave_chunk_tests {
    use super::PREFILL_MIN_ADVANCE;
    use std::sync::Arc;
    use std::time::Instant;

    use super::super::tests::make_test_scheduler;
    use super::super::*;

    /// A dialogue prefill carrying `tokens` and nothing else.
    pub(super) fn dialogue_prefill(seq: SequenceId, tokens: Vec<u32>) -> ActivePrefill {
        let (event_tx, _event_rx) = flume::unbounded();
        ActivePrefill {
            work: PrefillWork {
                sequence_id: seq,
                tokens: TokenBuffer::from(tokens),
                prefill_text: String::new(),
                user_text: String::new(),
                tags: Vec::new(),
                user_content_start: 0,
                user_content_end: 0,
                assistant_content_start: 0,
                no_think: false,
                prefill_assistant_text: String::new(),
                event_tx,
                max_decode_tokens: 0,
                sampling: SamplingConfig::compression(),
                submitted_at: Instant::now(),
                reprojection: None,
                belief: PriorBelief::default(),
                seal_action: SealAction::None,
                post_decode_tokens: TokenBuffer::default(),
                projection_offsets: Vec::new(),
                staged_composition: None,
                triggers: Arc::new(TriggerRegistry::new()),
                turn_grammar: None,
                free_tool_calls_from_penalties: false,
                projection: Vec::new(),
                demoted: false,
                held_until_decodes_below: None,
                announced: false,
                recorded_reply: None,
            },
            offset: 0,
            next_projection: 0,
            final_logits: None,
            error: None,
            prefill_start: None,
        }
    }

    /// **A dialogue prefill enters the wave group in pass-sized chunks.** It
    /// used to enter whole: a 15.2k-token Cline turn needed a 13.2 GiB transient
    /// tier on a 16 GB card, more than all the ground below the weight floor,
    /// and failed identically on every retry.
    #[test]
    fn a_long_dialogue_prefill_creeps_in_pass_sized_chunks() {
        let (mut scheduler, _tx) = make_test_scheduler();
        // A tier with room for far more than one pass: the target binds.
        scheduler.session.set_tier_budget_bytes(1 << 40);
        let cap = scheduler.prefill_pass_budget();
        let seq = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler
            .active_prefills
            .push(dialogue_prefill(seq, (0..15_000u32).collect()));

        // A tier with room for far more than one pass: the pass budget binds.
        scheduler.form_wave_group(false, usize::MAX, true);
        let (members, _, inputs, _) = scheduler.build_wave_group_inputs();
        assert_eq!(members.len(), 1);
        assert_eq!(
            inputs[0].dims(),
            &[1, cap],
            "the first chunk is one pass wide"
        );

        // A later group resumes at the committed offset, never from token 0.
        scheduler.reset_wave_prefill();
        scheduler.active_prefills[0].offset = 14_800;
        scheduler.form_wave_group(false, usize::MAX, true);
        let (_, _, inputs, _) = scheduler.build_wave_group_inputs();
        let rows = inputs[0].to_vec2::<u32>().expect("u32 rows");
        assert_eq!(rows[0].len(), 200, "the tail chunk is what remains");
        assert_eq!(rows[0][0], 14_800, "the chunk starts at the offset");
    }

    /// Prefills share the rows the tier holds: small ones pack together, and the
    /// next takes the rows still unassigned as a chunk of its own rather than
    /// waiting for a group that can take it whole.
    #[test]
    fn dialogue_prefills_share_the_rows_the_tier_holds() {
        let (mut scheduler, _tx) = make_test_scheduler();
        scheduler.session.set_tier_budget_bytes(1 << 40);
        let cap = scheduler.prefill_pass_budget();
        assert!(
            cap > 200 + PREFILL_MIN_ADVANCE,
            "the test needs room for a least chunk after the two short turns"
        );
        let a = SequenceId(scheduler.session.create_sequence().expect("create"));
        let b = SequenceId(scheduler.session.create_sequence().expect("create"));
        let c = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler
            .active_prefills
            .push(dialogue_prefill(a, vec![1; 100]));
        scheduler
            .active_prefills
            .push(dialogue_prefill(b, vec![1; 100]));
        scheduler
            .active_prefills
            .push(dialogue_prefill(c, vec![1; cap]));

        scheduler.form_wave_group(false, cap, false);
        let members: Vec<(usize, usize)> = scheduler
            .wave_prefill_members
            .iter()
            .map(|m| match *m {
                WaveMember::Prefill { seq_id, advance } => (seq_id, advance),
                WaveMember::Section { seq_id, advance } => (seq_id, advance),
            })
            .collect();
        assert_eq!(
            members,
            vec![(a.0, 100), (b.0, 100), (c.0, cap - 200)],
            "the third takes the rows the first two left"
        );
    }
}

#[cfg(test)]
mod turn_view_release_tests {
    use super::super::tests::{make_test_scheduler, register_turn_view};
    use super::super::*;
    use super::wave_chunk_tests::dialogue_prefill;

    /// **An errored turn prefill releases its view.** The error reached the
    /// caller and the view stayed registered with the parent's prefix borrowed
    /// — nothing but a completed decode or a reprojection ever released it, and
    /// an errored turn has neither.
    #[test]
    fn an_errored_turn_prefill_releases_its_view() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let parent = SequenceId(scheduler.session.create_sequence().expect("create"));
        let view = register_turn_view(&mut scheduler, parent);
        let mut prefill = dialogue_prefill(view, vec![1; 8]);
        prefill.error = Some(ConversationError::Channel("the wave failed".into()));
        scheduler.active_prefills.push(prefill);

        scheduler.promote_finished_prefills_to_decodes();

        assert!(scheduler.turn_views.is_empty(), "the view is unregistered");
        assert!(
            scheduler.session.sequence_offset(view.0).is_none(),
            "the view's slot is released"
        );
        assert!(
            scheduler.session.sequence_offset(parent.0).is_some(),
            "the parent is untouched"
        );
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

#[cfg(test)]
mod head_logits_tests {
    use super::Scheduler;
    use candle::{Device, Tensor};

    /// One scored row, as the head hands them back.
    fn row(v: f32) -> Tensor {
        Tensor::from_vec(vec![v], 1, &Device::Cpu).expect("row")
    }

    /// The ordinary shape: the head's rows are the whole answer.
    #[test]
    fn an_exact_head_is_returned_whole() {
        let logits: Vec<Tensor> = (0..4).map(|i| row(i as f32)).collect();
        let head = Scheduler::head_logits(&logits, 4, 1, 3, "test").expect("exact");
        assert_eq!(head.len(), 4);
    }

    /// **More rows than the head is ordinary, not an error.** The creep's and
    /// glue's rows follow it and the caller slices them off separately, so the
    /// check is a floor on the head and never a ceiling on the wave.
    #[test]
    fn rows_beyond_the_head_are_left_for_the_caller() {
        let logits: Vec<Tensor> = (0..9).map(|i| row(i as f32)).collect();
        let head = Scheduler::head_logits(&logits, 4, 1, 3, "test").expect("prefix");
        assert_eq!(head.len(), 4, "only the head, with the creep left behind");
    }

    /// **A short head is an error naming both halves.** This is the case the
    /// old `head_rows.min(len)` clamp turned into a shorter vector: the verify
    /// split downstream then reported a row count matching neither cohort, with
    /// nothing in it to say which side was short or which segment produced it.
    #[test]
    fn a_short_head_names_what_is_missing() {
        let logits: Vec<Tensor> = (0..9).map(|i| row(i as f32)).collect();
        let err = Scheduler::head_logits(&logits, 18, 0, 18, "seg3")
            .expect_err("nine rows cannot answer for eighteen")
            .to_string();
        assert!(err.contains("seg3"), "names the segment: {err}");
        assert!(err.contains('9'), "names what was scored: {err}");
        assert!(err.contains("18"), "names what was promised: {err}");
    }

    /// A wave with no head at all asks for nothing and gets nothing.
    #[test]
    fn an_empty_head_is_satisfied_by_any_wave() {
        let logits: Vec<Tensor> = (0..3).map(|i| row(i as f32)).collect();
        assert!(Scheduler::head_logits(&logits, 0, 0, 0, "test")
            .expect("empty")
            .is_empty());
        assert!(Scheduler::head_logits(&[], 0, 0, 0, "test")
            .expect("empty on empty")
            .is_empty());
    }
}
