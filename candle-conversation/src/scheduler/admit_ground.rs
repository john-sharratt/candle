//! The scheduler as [`admit::Ground`] — one admission pass over a settled
//! device.
//!
//! [`admit::fill`] is the policy: whose turn it is, what the offer costs, and
//! whether the wave goes faster carrying it. This file is the other half, read
//! off the live scheduler.
//!
//! # What this pass offers, and what it does not
//!
//! **Prefills only.** The other two bands have nothing to offer and say so:
//!
//! * A **decode** is never admitted here because nothing queues one — a decode
//!   exists because a prefill finished and promoted, which is one slot changing
//!   phase rather than a new claim. [`admit::fill`] already charges every active
//!   decode to the wave before the first offer, so they are priced into every
//!   judgement without being offered.
//! * A **section** is pushed straight into flight when its request is drained,
//!   so by the time a pass runs it is already resident and there is nothing left
//!   to decide. Giving sections a queue of their own is a real improvement —
//!   they are first admissions exactly as prefills are, and the module header
//!   records a minute of them taking the weight zone from 10,398 MiB to its hold
//!   — but it is a change to how `IngestSection` is handled, not to admission,
//!   and it is not made here.
//!
//! # Why a pass object rather than `impl Ground for Scheduler`
//!
//! `fill` walks its bands and offers each FIFO until it stops paying, so `peek`
//! has to be able to say "nothing more from this band" without consuming
//! anything. That needs a cursor, and one that belongs to the pass: carried on
//! the scheduler it would survive between passes and the next would resume part
//! way down a queue that has since changed.
//!
//! The pass also caches the two figures `fill` reads once — the headroom and the
//! budget — so every offer in one pass is judged against the same settled
//! moment. Residency is the deliberate exception, re-read per offer for the
//! reason [`admit::Ground::resident_weights`] gives: it moves under the fill's
//! feet from places admission cannot see.

use super::admission::prefill_cost_bytes;
use super::admit::{Budget, Cost, Ground, Headroom, Kind};
use super::interleave;
use super::{Scheduler, SequenceId};
use crate::projection::DecodePriority;

/// The least advance the engine hands a sequence, and so the width a forward
/// worth running is priced at. 128 rows.
pub(super) const PREFILL_MIN_ADVANCE: usize = 128;

/// Residency as admission measures it.
///
/// **Neither what is resident nor what the zone reads.** Both are wrong, in
/// opposite directions, and this engine has now been wedged by each:
///
/// * *Occupancy* (`resident_vram_bytes`, occupied slots × slot size) is what the
///   cache holds, and it is compared against figures that are capacities. It sits
///   below them on a warming cache, so every offer is refused on a healthy zone
///   and nothing an admission does moves it up. Measured: two turns queued,
///   nothing in flight, `stopped_on_weights` on every pass.
/// * *The zone's extent* fails the other way. It moves only when the boundary
///   moves, and the boundary moves only between forwards — so a store or a KV
///   chunk taken from the free list does not change it at all. The zone read the
///   same after twenty-five stores as before the first, every one of them passing
///   a hold that watched a number which had not moved; the boundary caught up
///   later and the zone fell 2 GiB under the mark in one step.
///
/// What a hold compares against is [`interleave::effective_weight_zone_bytes`]:
/// the residency the weight side could hold *right now*, derived from the span
/// identity (`total × region + weight = span`) against the **live** region count,
/// so each claim is charged the instant it is made rather than whenever the
/// boundary next catches up. `standing_tier` is the wave already committed — a
/// fact, not a choice — and the tier each new admission adds is charged
/// separately as `admit::Cost::dislodged_bytes`, so between them the wave's tier
/// is counted exactly once.
///
/// `u64::MAX` when there is no reservation to read (a CPU device, a unit test):
/// nothing to defend, so admission is bounded only by what the allocators give.
fn residency_now(standing_tier: usize) -> u64 {
    interleave::effective_weight_zone_bytes(standing_tier).unwrap_or(u64::MAX)
}

/// A priority's slot in the per-band cursors. Three bands, in the order
/// `admit::order::BANDS` offers them.
fn band_index(prio: DecodePriority) -> usize {
    match prio {
        DecodePriority::High => 0,
        DecodePriority::Normal => 1,
        DecodePriority::Low => 2,
    }
}

/// One admission pass: the engine, plus how far each prefill band has been
/// offered.
pub(super) struct AdmitPass<'a> {
    sched: &'a mut Scheduler,
    headroom: Headroom,
    budget: Budget,
    /// Bytes one 32-token KV block costs, in the formats a live sequence holds.
    per_block: u64,
    /// The store one more sequence costs, read at the open. A constant of the
    /// model's geometry — see [`Scheduler::recurrent_cost`].
    recurrent: u64,
    /// Residency this pass has already committed, summed over what it admitted.
    ///
    /// **Without it the floor is not defended in aggregate.** `fill` re-reads
    /// `resident_weights()` before every offer and tests
    /// `before - this_one_offer` against the floor — which is the right question
    /// only if `before` moves as the pass admits. It does not: nothing in
    /// `admit` claims a region (the K/V is claimed per chunk by the forward, not
    /// here), so the live region count the zone is derived from stands still for
    /// the whole pass. Ten turns that each individually clear a floor 4 GiB
    /// below the zone would all be admitted, committing the engine to ten times
    /// the ground it checked for once.
    ///
    /// Subtracting what has been taken so far makes each offer face the wave as
    /// it now stands, which is what the model is written to be asked.
    committed: u64,
    /// Rows a prefill chunk may carry into this wave.
    chunk_rows: usize,
    /// Rows this pass has admitted, so each offer is sized to the room that is
    /// still free rather than to the whole of [`Self::chunk_rows`]. See `peek`.
    rows_taken: usize,
    /// The tier the wave already in flight reserves — see
    /// [`Scheduler::standing_tier_bytes`]. Read once: it describes committed
    /// work, so it cannot move under the pass.
    standing_tier: usize,
    /// How far down `prefill_queue` this pass has offered, **per priority
    /// band**.
    ///
    /// One cursor cannot serve the bands: they are offered in turn and each has
    /// to resume where *it* left off, not where the previous band did. Shared,
    /// the `Low` band would start from wherever `High` had walked to and skip
    /// every `Low` candidate ahead of it.
    prefill_cursor: [usize; 3],
}

impl<'a> AdmitPass<'a> {
    pub(super) fn new(sched: &'a mut Scheduler) -> Self {
        let per_block = sched.per_block_kv_bytes();
        let recurrent = sched.recurrent_cost();
        let chunk_rows = sched.prefill_pass_budget();
        let standing_tier = sched.standing_tier_bytes();
        let headroom = sched.admit_headroom(standing_tier);
        let budget = sched.admit_budget_terms(&headroom, chunk_rows);
        Self {
            sched,
            headroom,
            budget,
            per_block,
            recurrent,
            committed: 0,
            rows_taken: 0,
            chunk_rows,
            standing_tier,
            prefill_cursor: [0; 3],
        }
    }

    /// The tier a chunk of `rows` would need, or zero when the model cannot
    /// price one.
    fn tier_for(&self, rows: usize) -> u64 {
        self.sched
            .model
            .wave_tier_bytes(rows, 1, self.sched.session.activation_dtype())
            .unwrap_or(0)
    }
}

impl Ground for AdmitPass<'_> {
    fn active(&self) -> usize {
        self.sched.prefill_width() + self.sched.section_ingest_width() + self.sched.decode_width()
    }

    fn decodes_active(&self) -> usize {
        self.sched.decode_width()
    }

    fn settled(&self) -> bool {
        self.sched.settled_since_admit
    }

    fn headroom(&self) -> Headroom {
        self.headroom
    }

    fn budget(&self) -> Budget {
        self.budget
    }

    /// **Re-read per offer, in the effective zone's currency.**
    ///
    /// The cache's own resident bytes are the wrong input: they do not fall when
    /// a claim takes ground the weight side was about to fill, so admission
    /// would spend the whole free list before the model noticed anything had
    /// been dislodged, and only then discover the floor. The effective zone
    /// falls by exactly one region per region claimed, which is the identity the
    /// floor is defended in — so the rate, the dislodge and the floor are read
    /// in one currency.
    fn resident_weights(&self) -> u64 {
        residency_now(self.standing_tier).saturating_sub(self.committed)
    }

    fn standing_rows(&self) -> usize {
        self.sched.standing_rows()
    }

    fn peek(&mut self, kind: Kind, prio: DecodePriority) -> Option<Cost> {
        // See the module header: neither band has anything to offer.
        if kind != Kind::Prefill {
            return None;
        }
        // **The width backstop bounds the pass, not just its entry.** It is the
        // dumb ceiling beneath the rate model — there so an error in the cost
        // model costs throughput rather than the daemon — and checking it only
        // before the pass leaves it useless: one fill starting empty could take
        // as many prefills as the row budget allows, which is the case the
        // ceiling exists for.
        if self.sched.active_prefills.len() >= Scheduler::MAX_PREFILL_WIDTH {
            return None;
        }
        // **Walk past what this band is not being offered.** A band is asked for
        // its own priority's next candidate, and the queue interleaves them: on
        // the shipped configuration only the dialogue layer sets `High` and every
        // other layer takes `Low`, so a dialogue turn behind queued background
        // work is the ordinary case. Stopping at the head because it belongs to
        // another band gives that turn no precedence at all — it would be served
        // by FIFO position alone, and `admit::order`'s whole point is that at
        // `High` a person is blocked on the next token.
        //
        // The cursor is the pass's and this band's, so a band resumes where *it*
        // left off rather than where the previous band did, and nothing is
        // consumed by looking.
        let band = band_index(prio);
        while let Some(w) = self.sched.prefill_queue.get(self.prefill_cursor[band]) {
            if self.sched.decode_priority_or_high(w.sequence_id) == prio {
                break;
            }
            self.prefill_cursor[band] += 1;
        }
        let w = self.sched.prefill_queue.get(self.prefill_cursor[band])?;
        // **Offer what is LEFT of the wave's rows, not the whole cap.**
        //
        // Clipping to `chunk_rows` made every offer the size of the entire budget,
        // so the first turn long enough to fill it took the wave alone and the next
        // one was refused `Cap` for exceeding a budget it was never shown. Measured
        // on this daemon at `--max-depth 3`: a 2,646-row head admitted, the next
        // turn offering its full 3,000 against a 4,880 cap, refused — the wave
        // running 2,646 of 4,880 with sequences queued and every wave
        // `seqs max=1`, at a fraction of the batched rate.
        //
        // Offered the remainder, a turn takes the room that is actually free and
        // the wave fills with as many turns as it takes. A turn clipped short is
        // not turned away, it advances by that much and is offered again next wave
        // — which is how a prefill longer than one forward already progresses.
        //
        // Standing rows count: they are in the forward this fill is composing, and
        // the rate model's own cap check counts them, so a clip that ignored them
        // would offer ground that is already spoken for.
        let room = self
            .chunk_rows
            .saturating_sub(self.sched.standing_rows().saturating_add(self.rows_taken));
        if room == 0 {
            return None;
        }
        let rows = w.tokens.len().min(room);
        Some(Cost {
            // The whole turn's K/V, not this chunk's: admitting the turn commits
            // the engine to feeding all of it, and a cost that priced only the
            // first chunk would admit a queue of turns whose tails cannot fit.
            kv: prefill_cost_bytes(w.tokens.len(), self.per_block),
            recurrent: self.recurrent,
            activations: self.tier_for(rows),
            rows,
            // The turn decodes when this prefill finishes, which is what puts
            // the decode model's judgement in play beside the prefill one.
            decodes_after: true,
        })
    }

    fn admit(&mut self, kind: Kind, prio: DecodePriority, cost: Cost) -> bool {
        if kind != Kind::Prefill {
            return false;
        }
        // Spend the rows this offer took, so the next offer in this pass is sized
        // to what is left — see `peek`.
        self.rows_taken = self.rows_taken.saturating_add(cost.rows);
        let band = band_index(prio);
        // **Nothing is purchased here, and that is not an omission.**
        //
        // A turn's K/V is claimed incrementally, chunk by chunk, by the admit
        // phase of each forward it rides (`admit_wave_kv`) — so the free pool
        // holds one wave's worth at a time, never a whole turn's, and that is
        // the steady state rather than a shortage. Demanding the turn's full
        // claim up front is therefore a test the engine can never pass: measured
        // on this daemon, an offer wanting 5,632 MiB against 576 MiB free, asking
        // the weight side for the difference and being conceded 0 — every pass,
        // so nothing was ever admitted and the queue only drained through the
        // keep-one-alive rule.
        //
        // The cost still does its job: it is what the rate model weighs, because
        // the residency this turn will dislodge as it runs is exactly the trade
        // being judged. What does not follow is that admission must hold that
        // ground before the first row runs.
        // Removing at the cursor shifts every later entry down by one, so the
        // *other* bands' cursors, if they sit past this index, now point one
        // place too far. Walk them back rather than leaving them to skip a
        // candidate each time another band takes one.
        let at = self.prefill_cursor[band];
        for (i, c) in self.prefill_cursor.iter_mut().enumerate() {
            if i != band && *c > at {
                *c -= 1;
            }
        }
        let Some(work) = self.sched.prefill_queue.remove(at) else {
            return false;
        };
        // What the weight side loses to this admission, carried so the offers
        // behind it face the wave as it now stands — see [`Self::committed`].
        self.committed = self.committed.saturating_add(cost.dislodged_bytes());
        self.sched.begin_prefill(work);
        true
    }
}

impl Scheduler {
    /// Ground the KV side can claim without crossing into the weight zone,
    /// together with the zone's own position and range.
    ///
    /// Region-granular and **measured, not forecast**: the relief pass ahead of
    /// admission has already run, so a free region is one this process has
    /// claimed and not yet spent rather than one it hopes to recover.
    pub(super) fn admit_headroom(&self, standing_tier: usize) -> Headroom {
        let zone = residency_now(standing_tier);
        let tier_reserve = self.min_forward_tier_bytes();
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
            zone_min_prefill: interleave::prefill_weight_bytes().unwrap_or(0),
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

    /// Tier bytes a forward worth running needs, held back from admission.
    ///
    /// **A useful forward's worth, not merely a placeable one.** Reserve nothing
    /// and the tier reaches zero — no forward at all, not a narrow one. Reserve
    /// one row and a forward can be placed but not filled: every wave carried
    /// one sequence while a third of the regions stood free. So it is the least
    /// advance the engine will hand a sequence, priced through the same planner
    /// that places the tier, which is what makes it follow the model's geometry
    /// instead of being a byte count to re-derive per card.
    pub(super) fn min_forward_tier_bytes(&self) -> u64 {
        self.model
            .wave_tier_bytes(PREFILL_MIN_ADVANCE, 1, self.session.activation_dtype())
            .unwrap_or(0)
    }

    /// The tier the wave already in flight reserves — the held creep group.
    ///
    /// A fact this decision reads, not an output it feeds back into itself: the
    /// tier each *new* admission adds is charged separately, through
    /// `Cost::dislodged_bytes`, so between them the wave's tier is counted
    /// exactly once.
    pub(super) fn standing_tier_bytes(&self) -> usize {
        let rows = self.standing_rows();
        let seqs = self.wave_prefill_members.len().max(1);
        self.model
            .wave_tier_bytes(rows, seqs, self.session.activation_dtype())
            .unwrap_or(0) as usize
    }

    /// The wave's opening terms, from the same settled reading as the headroom.
    pub(super) fn admit_budget_terms(&self, room: &Headroom, chunk_rows: usize) -> Budget {
        Budget {
            // One currency for the rate, the dislodge and the floor alike — see
            // [`residency_now`]. Reading any of the three in a different one is
            // what lets the model be shown the same ground twice.
            resident: room.zone,
            // The same line `headroom` nets out of the spendable ground: the
            // hold, the eviction margin that keeps the cache evictable, and a
            // useful forward's tier. The model enforces it as a hard refusal.
            prefill_floor: room
                .prefill_floor()
                .saturating_add(self.min_forward_tier_bytes()),
            decode_floor: room.floor().saturating_add(self.min_forward_tier_bytes()),
            max_rows: chunk_rows,
            max_decodes: Self::MAX_DECODE_WIDTH,
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
    pub(super) fn recurrent_cost(&self) -> u64 {
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
    pub(super) fn observe_prefill_forward(&mut self, rows: usize, us: u64) {
        if rows == 0 || us == 0 {
            return;
        }
        // Counted even when the model refuses to learn from it: the loop's overhead is
        // amortised across forwards that *ran*, not across forwards wide enough to teach
        // the copy rate.
        self.wave_forwards += 1;
        let resident = residency_now(self.standing_tier_bytes());
        if let Some(rate) = self.wave_rate.as_mut() {
            rate.observe_prefill(rows, resident, us as f64 / 1e6);
        }
    }

    /// Fold one completed decode forward into the planner.
    ///
    /// A bus-bound step says nothing about the layer time, and the model drops
    /// it; only a compute-bound one teaches anything. `draft: 0` prices the
    /// unspeculated row — a verify block was priced by the speculative driver
    /// when it was staged.
    pub(super) fn observe_decode_forward(&mut self, decodes: usize, us: u64) {
        if decodes == 0 || us == 0 {
            return;
        }
        // Same reason as the prefill funnel: the loop's overhead is shared by every
        // forward a wave ran, of either kind.
        self.wave_forwards += 1;
        let resident = residency_now(self.standing_tier_bytes());
        if let Some(rate) = self.wave_rate.as_mut() {
            rate.observe_decode(decodes, 0, resident, us as f64 / 1e6);
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
    pub(super) fn observe_expert_hit_rate(&mut self) {
        let Some(stats) = self.model.expert_stats() else {
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
            stats.expert_hits.saturating_sub(self.expert_hits_seen),
            stats.expert_misses.saturating_sub(self.expert_misses_seen),
        );
        self.expert_hits_seen = stats.expert_hits;
        self.expert_misses_seen = stats.expert_misses;
        let routed = hits + misses;
        if routed == 0 {
            return;
        }
        let hit = hits as f64 / routed as f64;
        let resident = residency_now(self.standing_tier_bytes());
        if let Some(rate) = self.wave_rate.as_mut() {
            rate.observe_hit_rate(hit, resident);
        }
    }

    /// The layer priority a slot decodes at, defaulting to the protective
    /// `High` when its target is not resolvable yet.
    pub(super) fn decode_priority_or_high(&self, id: SequenceId) -> DecodePriority {
        self.decode_layer_priority(id)
            .unwrap_or(DecodePriority::High)
    }
}
