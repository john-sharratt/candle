//! Admission: who runs next, what it costs, and whether the wave is better off
//! carrying it.
//!
//! # Status: this is the admission path
//!
//! [`fill`] is what admits a wave — `super::prefill::promote_new_prefills`
//! calls it, and `super::admit_ground::AdmitPass` is the engine's answer to
//! [`Ground`]. The byte-fit planner it replaced is gone rather than kept
//! alongside: there is one admission path in this crate at any time.
//!
//! What it offers is prefills. Decodes reach a wave as continuations, charged
//! to it before the first offer and never re-judged; sections are still pushed
//! straight into flight when their request is drained, which is the next thing
//! to bring inside the gate. See `super::admit_ground` for both.
//!
//! ## Two things checked and NOT defects, so they are not re-raised
//!
//! **A busy engine with no standing prefill rows charges
//! `Admission::Prefill { tokens: 0 }`**, which leaves `WaveRate::tokens` at zero
//! and so skips `judge_prefill`'s gain comparison. That was recorded here as a
//! defect and is not one. What the charge exists to deny is the *head waiver*,
//! and it denies it by marking the wave non-empty rather than by the token
//! count — so the first offer is still floor-checked, and
//! [`rate::WaveRate::decode_would_carry`] still judges the turn it will become,
//! because the continuations were charged. The comparison that is skipped has
//! nothing to compare against: a wave carrying no prefill rows has a prefill
//! rate of zero, against which every positive rate is an unbounded gain.
//! Pinned by `a_busy_engine_with_no_standing_rows_is_still_judged_by_the_decode_model`.
//!
//! **`starting_decodes` is latched before the band loop.** Admissions land in
//! the fill's taken set and never in the engine's active set, so
//! [`Ground::decodes_active`] cannot change under the loop — see the comment at
//! its binding.
//!
//! Four concerns, four files, no overlap:
//!
//! * [`order`] — whose turn it is. Pure policy; touches no device.
//! * [`cost`] — what one admission takes. Arithmetic over settled state.
//! * [`rate`] — what the wave's throughput does if it joins. The decision.
//! * [`gate`] — the two lines no answer to that may cross.
//!
//! [`fill`] is the loop that asks them in that order. [`pass_budget`] bounds
//! the tokens any one prefill forward carries, whoever was admitted.
//!
//! # The decision is a rate, not a fit
//!
//! Admission used to ask whether an offer's bytes fitted in the ground standing
//! free above the weight floor. That question has no answer in tokens a second:
//! a wave of 250 rows and a wave of 2,000 rows pay the *same* expert copy — a
//! prefill forward needs every expert, resident or streamed — so the narrow one
//! is not cheaper, it is simply slower per row. A gate that admits by fit stops
//! widening the moment the bytes run out, which on this card was ~470 tok/s
//! against a modelled 1,210 at the same residency.
//!
//! So the fill asks [`rate::WaveRate`] instead: **does the wave go faster with
//! this in it?** Bytes have not gone away — they are what the offer *costs*,
//! and the model reads them as the residency the admission dislodges, which is
//! what makes a wide wave stop being worth it. What changed is that they are
//! now one term in a throughput comparison rather than the whole question.
//!
//! # The shape, and why it is this shape
//!
//! **A slot is held from admission to completion.** A prefill runs through its
//! chunks; the same slot then decodes to EOS; then it is sealed and evicted,
//! and that is what frees the next admission. Nothing waits for a *different*
//! kind of slot on the way — the finished-prefill queue that produced the
//! measured wedges (19 turns holding 30,710 tokens of K/V; the prefill side at
//! 8.1% of admitted rows with directories queued) does not exist here, because
//! a prefill becoming a decode is one slot changing phase rather than one slot
//! asking for another.
//!
//! **Admission opportunities are created only by completions**, so the whole
//! pass is skipped on a wave where nothing finished — see [`Ground::settled`].
//! The wave simply runs the set it already has.
//!
//! **Eviction runs before the measurement, not after.** The caller sheds
//! everything it can that is neither weights nor an active slot, and only then
//! reads [`Ground::headroom`]. So [`cost`] is never estimating against a
//! hypothetical: the free lists it prices against are the ones eviction just
//! produced.

pub(crate) mod cost;
pub(crate) mod gate;
pub(crate) mod order;
pub(crate) mod pass_budget;
pub mod rate;

pub(crate) use cost::Cost;
pub(crate) use gate::Headroom;
pub(crate) use order::{Kind, Order};
pub(crate) use rate::{Admission, WaveRate};

use crate::projection::DecodePriority;

/// The wave's opening terms, read once per fill from the settled device.
///
/// Everything [`rate::WaveRate::reset`] needs, gathered in one place so the
/// four figures are taken from the same moment — a floor read after a purchase
/// that a residency read preceded would let a wave admit into ground that had
/// already moved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Budget {
    /// Resident weights as the wave opens.
    pub resident: u64,
    /// The residency a **prefill** admission may not take them under — lower than
    /// [`Self::decode_floor`], because a prefill's copy amortises over its rows
    /// while a decode's is charged per layer per step. See
    /// `WaveRate::prefill_floor`.
    pub prefill_floor: u64,
    /// The residency a **decode** admission may not take them under.
    pub decode_floor: u64,
    /// The widest wave whose transient tier the partition could place — the
    /// gap as it stands plus what the weight side could concede for it. A rate
    /// the tier cannot hold is not a rate the engine can run.
    pub max_rows: usize,
    /// The most decodes a wave will carry.
    pub max_decodes: usize,
}

/// The engine, as the fill needs to see it.
pub(crate) trait Ground {
    /// Slots admitted and not yet finished — prefilling or decoding alike.
    fn active(&self) -> usize;

    /// Active slots currently in their decode phase.
    fn decodes_active(&self) -> usize;

    /// Whether anything finished since the last admission pass.
    ///
    /// `false` is the fast path: no slot freed, so nothing new can fit that did
    /// not fit before, and the wave runs what it holds. Admission is O(slots
    /// completed), not O(waves).
    fn settled(&self) -> bool;

    /// The device after the eviction pass — see the module header.
    fn headroom(&self) -> Headroom;

    /// The wave's opening terms, from that same settled reading.
    fn budget(&self) -> Budget;

    /// Resident weights **right now**, re-read before every offer.
    ///
    /// Not carried forward from the last admission: residency moves inside a
    /// fill from places admission cannot see. The expert cache grows back into
    /// spare K/V ground at phase 0 of every forward, evicts slots mid-forward,
    /// and concedes ground to any claim that runs the K/V side out. A wave
    /// judged against a figure from three offers ago is judged against a card
    /// that has since moved.
    fn resident_weights(&self) -> u64;

    /// Rows the next wave already carries before this fill adds anything: the
    /// creep group held from the last wave, which rides the next one whole.
    ///
    /// Charged to the wave at the open, so the offers that follow are judged
    /// against what the forward will actually run — and so a wave that is
    /// already carrying rows does not also take a head unconditionally.
    fn standing_rows(&self) -> usize;

    /// What admitting this band's next FIFO candidate would take, or `None`
    /// when the band has nothing left to offer.
    ///
    /// Strictly FIFO: an item that does not fit is **not** passed over for one
    /// that does. Cheapest-first starves the expensive work permanently, and
    /// the expensive work is never the cheapest.
    fn peek(&mut self, kind: Kind, prio: DecodePriority) -> Option<Cost>;

    /// Admit the candidate [`Self::peek`] just priced at `cost`, buying and
    /// claiming its ground. `false` when the allocators refused it after all.
    ///
    /// **This is the one place the weight boundary is asked to move toward
    /// K/V.** The gate has already said the price stays above the residency the
    /// engine defends, so what the K/V side does not hold free of that price is
    /// bought from the weight side here, before the wave — never by a claim
    /// that runs out mid-wave, never by the tier's placement, never by a
    /// forward's own arithmetic. Every one of those bought outside the gate's
    /// accounting, and between them took the zone to its floor with nothing
    /// admitted.
    fn admit(&mut self, kind: Kind, prio: DecodePriority, cost: Cost) -> bool;
}

/// What one fill pass took.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct Filled {
    pub decodes: usize,
    pub prefills: usize,
    pub sections: usize,
    /// The pass ran the fast path: nothing had finished, so nothing was offered.
    pub skipped: bool,
    /// An admission was refused because it would have reached the weight zone —
    /// the floor, the tier's placeable width, or the decode cap. The producer's
    /// backpressure signal: the device is the bound.
    pub stopped_on_weights: bool,
    /// An admission was refused because the wave was already going as fast as
    /// it is going to go — another row would not pay for the residency it
    /// dislodges. **Not** backpressure: the engine is working well and the
    /// queue behind it simply rides the next wave.
    pub stopped_on_rate: bool,
}

/// Log one refused offer with everything that produced the answer.
///
/// **A refusal is the decision worth seeing, and it is the one the summary line
/// cannot carry.** The per-fill line says a band stopped; it cannot say which
/// item, what it would have cost, what the model projected, or which of six
/// rules said no — and those are exactly the questions asked of a run that
/// admitted nothing for an hour. One line per refusal, at debug, and refusals
/// are rare on a healthy engine by construction: the band stops at the first.
fn log_refusal(
    kind: Kind,
    prio: DecodePriority,
    cost: &Cost,
    before: u64,
    after: u64,
    budget: &Budget,
    refusal: rate::Refusal,
) {
    tracing::debug!(
        target: "candle_conversation::scheduler::admission",
        kind = ?kind,
        prio = ?prio,
        rows = cost.rows,
        kv_mib = cost.kv >> 20,
        recurrent_mib = cost.recurrent >> 20,
        tier_mib = cost.activations >> 20,
        claimed_mib = cost.claimed_bytes() >> 20,
        // What the judgement was actually made on — the claim plus the tier.
        // Logged beside `claimed_mib` because the two differ by exactly the
        // term that used to be missing, and a run that refuses unexpectedly
        // wants to see which of them moved.
        dislodged_mib = cost.dislodged_bytes() >> 20,
        resident_before_mib = before >> 20,
        resident_after_mib = after >> 20,
        // Both floors, because a refusal is read against the one its kind stands
        // on and the gap between them is the prefill's extra room.
        prefill_floor_mib = budget.prefill_floor >> 20,
        decode_floor_mib = budget.decode_floor >> 20,
        room_mib = before.saturating_sub(budget.prefill_floor) >> 20,
        max_rows = budget.max_rows,
        max_decodes = budget.max_decodes,
        refusal = ?refusal,
        "offer refused",
    );
}

impl Filled {
    /// Record which kind of refusal ended a band.
    fn note(&mut self, refusal: rate::Refusal) {
        use rate::Refusal;
        match refusal {
            Refusal::Worse { .. } | Refusal::Saturated { .. } => self.stopped_on_rate = true,
            Refusal::Floor { .. } | Refusal::Cap { .. } | Refusal::DecodeCap { .. } => {
                self.stopped_on_weights = true
            }
            // The wave latched full on an earlier refusal, which was itself
            // recorded when it happened. Nothing new to say.
            Refusal::Full => {}
        }
    }
}

/// Offer the wave's rows, in order, to whatever makes the wave faster.
///
/// `rate` is the engine's one planner, carried across fills because what it has
/// learned — the effective copy rate, the decode layer time — is a property of
/// the machine, not of this wave.
pub(crate) fn fill<G: Ground>(ground: &mut G, rate: &mut WaveRate) -> Filled {
    let mut out = Filled::default();
    if !ground.settled() {
        out.skipped = true;
        return out;
    }
    // Measured once, after eviction. `zone` is spent down as stores are placed,
    // because the decode-start rule reads it; the rate model re-reads residency
    // per offer instead (`Ground::resident_weights`), since that is the figure
    // the whole decision turns on and it moves under the fill's feet.
    let mut room = ground.headroom();
    let budget = ground.budget();
    rate.reset(
        budget.resident,
        budget.prefill_floor,
        budget.decode_floor,
        budget.max_rows,
        budget.max_decodes,
    );
    // **What the wave already carries, charged before anything is offered.**
    //
    // Two things ride the next wave whatever this fill decides: the creep group
    // held from the last one, and — through the continuation rule below — every
    // decode the engine is already running. Charging them first is what makes
    // the offers behind them judged against the forward that will actually run.
    //
    // It is also what scopes the head waiver. The model takes its first offer
    // unconditionally so that a wave carries *something* and a slot too large
    // to ever fit cannot block the queue behind it forever — the rule that
    // makes this design deadlock-free. That is meant for a wave with nothing to
    // run, not for every fill: an engine with sixty slots in flight would
    // otherwise take one free admission per pass, each of them allowed under
    // the floor. So a busy engine charges the wave even when the creep is
    // empty, and only a genuinely idle one gets the waiver.
    let standing = ground.standing_rows();
    if standing > 0 || ground.active() > 0 {
        rate.charge(Admission::Prefill { tokens: standing }, budget.resident);
    }
    // **The decodes the engine is already running, charged before any offer.**
    //
    // A continuation is not an admission — its ground was reserved when its slot
    // was admitted, and the engine owes it the forward that finishes it — but it
    // is unquestionably part of the wave every offer below is judged against,
    // and the decode model is the half that prices residency *per layer*.
    //
    // Charging them here rather than when the decode band is reached is what
    // makes the second judge real for a background workload.
    // [`rate::WaveRate::decode_would_carry`] skips its gain comparison entirely
    // while `decodes == 0`, and for an ingest every conversation is `Low`, so
    // the band order is Low Section, Low Prefill, Low Decode: every prefill was
    // judged at `decodes == 0`, and the question "can the decode this turn
    // becomes be carried at that residency" collapsed to a bare floor check.
    // That is the mechanism `docs/wave_feeder.md` §4.11.12 measured as decode
    // cost per directory going 4.3 s to 12.3 s — the cheap phase spending the
    // residency the expensive one was about to need.
    //
    // The set is the engine's **committed** decodes, not the subset this fill
    // goes on to admit, because that is what the next forward carries whatever
    // admission decides. `draft: 0` prices the unspeculated row: a continuation's
    // verify block was priced by the speculative driver when it was staged, and
    // re-pricing it here would charge this wave for the drafter's rows twice.
    for _ in 0..ground.decodes_active() {
        rate.charge(Admission::Decode { draft: 0 }, budget.resident);
    }
    // **Which decodes this pass treats as admissions.** Only the first, and
    // only when nothing is decoding — that one starts the expert cache warming
    // and is genuinely new. Read once: `Ground::admit` moves a decode into this
    // fill's taken set, never into the engine's active set, so this cannot
    // change under the loop and re-reading it per offer would only invite the
    // belief that it might.
    let starting_decodes = ground.decodes_active() == 0 && ground.active() > 0;
    let mut order = Order::new();
    while let Some((prio, kind)) = order.next() {
        // The band, FIFO, until it runs dry or the wave stops paying. A refusal
        // ends this band and not the pass: a later band's work is a different
        // size and may still be worth carrying.
        while let Some(cost) = ground.peek(kind, prio) {
            let total = cost.total();
            // The truth as of this offer, and what this offer would leave of
            // it — its region claim **and its tier**, which the weight side
            // loses alike. See `Cost::dislodged_bytes`: the tier is transient
            // per forward but published per wave, and the growth term is
            // bounded by it, so a wave that widens holds that ground against
            // the weight side for as long as it stays that wide.
            let before = ground.resident_weights();
            let after = before.saturating_sub(cost.dislodged_bytes());
            let allowed = match kind {
                // A decode when none is running is the one case with its own
                // rule — it keeps the expert cache warm. It is also the only
                // decode that is a genuine admission rather than a
                // continuation, so it is the only one the rate model judges.
                Kind::Decode if starting_decodes => {
                    if !gate::may_start_decode(0, total, &room) {
                        // The zone is too low to keep a decode's working set
                        // alive — a weight condition, and the producer's to
                        // hear about.
                        out.stopped_on_weights = true;
                        tracing::debug!(
                            target: "candle_conversation::scheduler::admission",
                            zone_mib = room.zone >> 20,
                            midpoint_mib = room.midpoint() >> 20,
                            cost_mib = total >> 20,
                            "first decode held back to keep the expert cache warm",
                        );
                        false
                    } else {
                        match rate.try_admit(decode_of(&cost), before, after) {
                            rate::Admit::Admitted { .. } => true,
                            rate::Admit::Refused(r) => {
                                out.note(r);
                                log_refusal(kind, prio, &cost, before, after, &budget, r);
                                false
                            }
                        }
                    }
                }
                // **Stepping a decode is not an admission.** [`Ground::peek`]
                // only ever offers decodes that are already active, and their
                // ground was reserved when the slot was admitted — a slot is
                // held from admission to completion, so the engine owes them the
                // forwards that finish them. Gating them again refuses work that
                // has already been paid for, and does it precisely when the
                // device is tightest: once the budget closes, prefills promote
                // into decodes that can never be stepped, no rows reach the wave
                // at all, and nothing completes to reopen the budget. Runs CB
                // and CD both died there — 34 of 40 waves running no forward
                // with fifty-odd slots admitted and the queue backing up.
                Kind::Decode => true,
                // **The wave's rows are judged on what they do to its rate.**
                // A prefill chunk earns `cost.rows` of forward width against
                // the copy every prefill forward pays whatever its width, and
                // costs whatever residency its K/V and store dislodge; it joins
                // while that trade is winning.
                //
                // The head waiver — a wave carries at least one thing, so a
                // slot too large to ever fit still runs rather than blocking
                // the queue behind it forever — lives in the model, keyed on
                // the wave being **empty** rather than on the engine being
                // idle. That is the same rule read from the wave's side, and it
                // is why the standing creep is charged above: without it a wave
                // already carrying rows would take a head as well, and the
                // waiver would fire on every fill instead of on the ones with
                // nothing to run. (Run CJ is what firing it too often costs: 83
                // queued, 56 prefills admitted in a single fill, the weight zone
                // from 8,180 MiB to 1,417 with every region live.)
                //
                // A section is a first admission exactly as a prefill is: fresh
                // K/V on a scratch slot, a store, a tier row — so it is judged
                // the same way and counted the same way. Sections used to enter
                // the wave without passing here at all, and one minute of them
                // took the weight zone from 10,398 MiB to its hold.
                Kind::Prefill | Kind::Section => {
                    // **A turn that will decode is judged by both models, here.**
                    //
                    // The prefill model prices a copy paid per forward, so it
                    // amortises across the chunk and keeps saying yes down to the
                    // floor. The decode that this turn becomes pays its copy per
                    // layer, against `1 - hit(resident)` — so the residency this
                    // admission spends is charged again on every step of a decode
                    // no gate had yet asked about. Asking both questions at the
                    // one moment a refusal is still cheap is what stops the cheap
                    // phase spending what the expensive one needs.
                    //
                    // It buys nothing extra: the decode still takes its ground a
                    // lease at a time. The head waiver is applied explicitly
                    // because the second question is outside the model's own.
                    let decode_ok = if cost.decodes_after && !rate.carries_nothing() {
                        // The decode that follows steps one row at a time; a
                        // drafted turn is wider, and would only be refused more
                        // readily than this asks for.
                        match rate.decode_would_carry(0, after) {
                            Ok(_) => true,
                            Err(r) => {
                                out.note(r);
                                log_refusal(kind, prio, &cost, before, after, &budget, r);
                                false
                            }
                        }
                    } else {
                        true
                    };
                    // An offer of no rows joins a forward whose rows are already
                    // spent: it shares the forwards to come rather than widening
                    // this one, so there is no prefill gain to judge — only the
                    // floor, and the decode question asked above.
                    decode_ok
                        && match if cost.rows == 0 {
                            rate.try_join(after)
                        } else {
                            rate.try_admit(Admission::Prefill { tokens: cost.rows }, before, after)
                        } {
                            rate::Admit::Admitted { .. } => true,
                            rate::Admit::Refused(r) => {
                                out.note(r);
                                log_refusal(kind, prio, &cost, before, after, &budget, r);
                                false
                            }
                        }
                }
            };
            if !allowed {
                break;
            }
            if !ground.admit(kind, prio, cost) {
                // The allocators refused after the model said yes. The ground
                // is genuinely gone, whatever the arithmetic made of it.
                out.stopped_on_weights = true;
                break;
            }
            // A continuation is NOT charged here. It was charged at the top of
            // the fill, with every other decode the engine is already
            // committed to, so that the prefill bands ahead of the decode bands
            // are judged against a wave that includes them. Charging it again
            // on admission would count the same row twice — once as standing
            // load and once as this fill's — and the double count lands on
            // `routed_per_layer`, which is what the decode model's hit rate is
            // computed from.
            room.zone = room.zone.saturating_sub(cost.recurrent);
            match kind {
                Kind::Decode => out.decodes += 1,
                Kind::Prefill => out.prefills += 1,
                Kind::Section => out.sections += 1,
            }
        }
    }
    out
}

/// One decode offer as the rate model reads it: a verify block of `1 + draft`
/// rows, so a plain decode is one row and no draft.
fn decode_of(cost: &Cost) -> Admission {
    Admission::Decode {
        draft: cost.rows.saturating_sub(1),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use candle_nn::kv_cache::REGION_BYTES;
    use rate::{DecodeModel, ExpertGeometry, RateModel};

    /// One region of ground, the granularity a claim actually takes.
    const REGION: u64 = REGION_BYTES as u64;
    /// A floor with room above it for the tests to spend.
    const FLOOR: u64 = 5 << 30;
    /// Rows a prefill or section chunk puts in the forward.
    const ROWS: usize = 128;

    /// The engine's planner as these tests use it: the 35B's expert geometry on
    /// the 4090 Mobile's link, with no minimum gain — so an offer is judged on
    /// the floor and on whether it makes the wave slower, and the band policy
    /// under test is not confounded by saturation.
    fn planner() -> WaveRate {
        WaveRate::with_link_rate(
            25e9,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel::default(),
        )
        .with_min_gain(0.0)
    }

    /// A fake device: a dislodge per queued item, a residency it comes out of,
    /// and a record of the order things were taken in.
    struct Fake {
        decodes: Vec<u64>,
        prefills: Vec<u64>,
        sections: Vec<u64>,
        /// Resident weights, spent down by what each admission claims.
        resident: u64,
        standing: usize,
        active: usize,
        decodes_active: usize,
        settled: bool,
        zone: u64,
        taken: Vec<Kind>,
        decode_prio: DecodePriority,
        prefill_prio: DecodePriority,
        /// Every prefill after the first offers no rows, as a turn joining a
        /// forward whose rows are spent does.
        join_after_head: bool,
    }

    impl Fake {
        /// `decodes` and `prefills` carry the K/V each item claims; `resident`
        /// is the weight side it comes out of.
        fn new(decodes: Vec<u64>, prefills: Vec<u64>, resident: u64) -> Self {
            Self {
                decodes,
                prefills,
                sections: Vec::new(),
                resident,
                standing: 0,
                active: 0,
                decodes_active: 0,
                settled: true,
                zone: 9 << 30,
                taken: Vec::new(),
                decode_prio: DecodePriority::Low,
                prefill_prio: DecodePriority::Low,
                join_after_head: false,
            }
        }
        fn queue(&mut self, kind: Kind) -> &mut Vec<u64> {
            match kind {
                Kind::Decode => &mut self.decodes,
                Kind::Prefill => &mut self.prefills,
                Kind::Section => &mut self.sections,
            }
        }
        fn wants(&self, kind: Kind) -> DecodePriority {
            match kind {
                Kind::Decode => self.decode_prio,
                Kind::Prefill => self.prefill_prio,
                // Sections carry no priority of their own: one band, `Low`.
                Kind::Section => DecodePriority::Low,
            }
        }
    }

    impl Ground for Fake {
        fn active(&self) -> usize {
            self.active
        }
        fn decodes_active(&self) -> usize {
            self.decodes_active
        }
        fn settled(&self) -> bool {
            self.settled
        }
        fn headroom(&self) -> Headroom {
            Headroom {
                free_kv: self.resident.saturating_sub(FLOOR),
                zone: self.zone,
                zone_min: 4 << 30,
                zone_max: 10 << 30,
                zone_min_prefill: 2 << 30,
            }
        }
        fn budget(&self) -> Budget {
            Budget {
                resident: self.resident,
                // One floor for both kinds here: these cases are about the rate
                // model's gain arithmetic and its caps, not about the prefill /
                // decode split, and giving them the same number keeps each
                // assertion meaning exactly what it meant before the split.
                prefill_floor: FLOOR,
                decode_floor: FLOOR,
                max_rows: 8_192,
                max_decodes: 64,
            }
        }
        fn resident_weights(&self) -> u64 {
            self.resident
        }
        fn standing_rows(&self) -> usize {
            self.standing
        }
        fn peek(&mut self, kind: Kind, prio: DecodePriority) -> Option<Cost> {
            if prio != self.wants(kind) {
                return None;
            }
            // A decode claims nothing — its lease was bought at admission — and
            // rides as one row; a prefill or section claims its K/V and carries
            // a chunk's rows.
            let rows = match kind {
                Kind::Decode => 1,
                Kind::Prefill if self.join_after_head && !self.taken.is_empty() => 0,
                _ => ROWS,
            };
            self.queue(kind).first().map(|&kv| Cost {
                kv,
                rows,
                ..Default::default()
            })
        }
        fn admit(&mut self, kind: Kind, prio: DecodePriority, cost: Cost) -> bool {
            if prio != self.wants(kind) {
                return false;
            }
            if self.queue(kind).is_empty() {
                return false;
            }
            let priced = self.queue(kind).remove(0);
            assert_eq!(cost.kv, priced, "admit is handed the price peek quoted");
            // The engine's own residency moves by what the weight side loses,
            // which is the claim *and* the tier — see `Cost::dislodged_bytes`.
            self.resident = self.resident.saturating_sub(cost.dislodged_bytes());
            self.active += 1;
            if kind == Kind::Decode {
                self.decodes_active += 1;
            }
            self.taken.push(kind);
            true
        }
    }

    /// **An offer of no rows is a join, judged by the floor, not by the prefill
    /// gain.** Every prefill after the head offers zero rows — the forward's rows
    /// are spent, the way `AdmitPass::peek` offers a turn once they are. Each
    /// dislodges a region; all are taken while residency stays above the floor,
    /// and the fill stops on the weights at the first that would cross it.
    #[test]
    fn prefills_offering_no_rows_join_until_the_floor() {
        let mut f = Fake::new(Vec::new(), vec![REGION; 6], FLOOR + 3 * REGION + REGION / 2);
        f.join_after_head = true;
        let got = fill_once(&mut f);
        // Residency starts three and a half regions above the floor: the head
        // leaves 2.5, the joins 1.5 and 0.5, and the next would land under it.
        assert_eq!(got.prefills, 3, "the head and two joins above the floor");
        assert!(
            got.stopped_on_weights,
            "the third join would cross the floor"
        );
        assert!(
            !got.stopped_on_rate,
            "no join is judged by the prefill gain"
        );
    }

    /// Run one fill with a fresh planner.
    fn fill_once(f: &mut Fake) -> Filled {
        fill(f, &mut planner())
    }

    /// Nothing finished, so nothing can newly fit: the pass costs nothing.
    #[test]
    fn an_unsettled_wave_takes_the_fast_path() {
        let mut f = Fake::new(vec![1; 4], vec![1; 4], FLOOR + 8 * REGION);
        f.settled = false;
        let got = fill_once(&mut f);
        assert_eq!(
            got,
            Filled {
                skipped: true,
                ..Default::default()
            },
        );
        assert!(f.taken.is_empty(), "nothing offered, nothing taken");
    }

    /// The fast path returns before anything is charged, so a skipped pass
    /// leaves the planner exactly as it found it — it is not a pass that
    /// admitted nothing, it is a pass that asked nothing.
    #[test]
    fn a_skipped_fill_charges_nothing() {
        let mut f = Fake::new(vec![0; 2], vec![REGION; 2], FLOOR + 4 * REGION);
        f.settled = false;
        f.active = 2;
        f.decodes_active = 2;
        let mut rate = planner();
        let got = fill(&mut f, &mut rate);
        assert!(got.skipped);
        assert_eq!(rate.decodes(), 0, "no charge on a pass that made no offer");
        assert_eq!(rate.tokens(), 0);
    }

    /// **The decodes the engine is already running are charged before any
    /// prefill is judged.**
    ///
    /// The decode queue is empty here, so nothing is ever *offered* as a
    /// decode — which means anything the planner knows about decodes can only
    /// have come from the charge at the top of the fill.
    ///
    /// Before that charge moved there, a background workload's prefill bands
    /// were judged at `decodes == 0` (every ingest conversation is `Low`, so
    /// the order is Low Section, Low Prefill, Low Decode), and
    /// `WaveRate::decode_would_carry` skips its gain comparison entirely while
    /// that counter is zero — leaving the second judge as a bare floor check on
    /// exactly the workload it was built for.
    #[test]
    fn standing_decodes_are_charged_before_the_prefill_bands() {
        let mut f = Fake::new(Vec::new(), vec![REGION; 2], FLOOR + 4 * REGION);
        f.active = 3;
        f.decodes_active = 3;
        let mut rate = planner();
        let got = fill(&mut f, &mut rate);
        assert_eq!(
            got.decodes, 0,
            "no decode was offered, so none was admitted"
        );
        assert_eq!(
            rate.decodes(),
            3,
            "the three the engine is already running were charged at the open",
        );
    }

    /// A continuation is charged **once**, at the open, and not again when it
    /// is admitted. The double count landed on `routed_per_layer`, which is
    /// what the decode model's hit rate is computed from — so it made a wave
    /// look like it routed to twice the experts it does.
    #[test]
    fn an_admitted_continuation_is_not_charged_twice() {
        let mut f = Fake::new(vec![0; 2], Vec::new(), FLOOR + 4 * REGION);
        f.active = 2;
        f.decodes_active = 2;
        let mut rate = planner();
        let got = fill(&mut f, &mut rate);
        assert_eq!(got.decodes, 2, "both continuations step");
        assert_eq!(
            rate.decodes(),
            2,
            "charged at the open, not again on admission",
        );
    }

    /// **The fast path must never be the only thing standing between the engine
    /// and a deadlock.**
    ///
    /// Skipping admission until something completes assumes the admitted set is
    /// advancing towards completing. A caller whose wave ran no forward has no
    /// such assurance, and must report itself unsettled — otherwise nothing
    /// finishes, so the skip never clears, so nothing is admitted, forever.
    /// Measured on run BS: `skipped=true active=32 queued=24` for 98
    /// consecutive waves with no forward at all, the weight zone pinned at
    /// 1,417 MiB against a 4,774 hold.
    ///
    /// This asserts the contract from the fill's side — a ground that reports
    /// `settled` gets offered work, whatever else is true of it — so the rule
    /// cannot be lost by a later change to how `settled` is computed.
    #[test]
    fn a_settled_ground_is_always_offered_work() {
        let mut f = Fake::new(vec![1; 2], vec![1; 2], FLOOR + 8 * REGION);
        f.active = 32; // busy, and nothing has completed
        f.decodes_active = 1;
        f.settled = true; // but the engine ran no forward, so: re-offer
        let got = fill_once(&mut f);
        assert!(!got.skipped);
        assert!(
            got.prefills + got.decodes > 0,
            "a settled ground with queued work must be able to admit",
        );
    }

    /// **The head runs even when it does not fit**, so a slot too large for the
    /// card cannot block the queue behind it forever. The waiver is the rate
    /// model's — an empty wave takes its first offer whatever it costs — and it
    /// applies here because nothing is running.
    #[test]
    fn an_empty_engine_admits_the_head_whatever_it_costs() {
        let mut f = Fake::new(Vec::new(), vec![64 << 30, REGION], FLOOR);
        let got = fill_once(&mut f);
        assert_eq!(got.prefills, 1, "the head lands");
        assert_eq!(f.active, 1);
        assert_eq!(f.resident, 0, "and it took the weights under the floor");
        assert!(got.stopped_on_weights, "and the next one does not");
        assert_eq!(f.prefills.len(), 1, "which waits, FIFO");
    }

    /// **A busy engine gets no waiver.** The rule is "a wave carries something",
    /// and a wave that will step the decodes already running carries something
    /// — so the prefills offered ahead of them are judged on the floor like any
    /// other admission, rather than one riding free on every pass.
    #[test]
    fn a_busy_engine_takes_no_free_head() {
        let mut f = Fake::new(Vec::new(), vec![REGION; 4], FLOOR);
        f.active = 8;
        f.decodes_active = 3;
        let got = fill_once(&mut f);
        assert_eq!(got.prefills, 0, "at the floor, nothing is free");
        assert!(got.stopped_on_weights);
        assert_eq!(f.resident, FLOOR, "the floor held");
    }

    /// **A busy engine with no standing rows is still judged by both models.**
    ///
    /// The charge is `Prefill { tokens: 0 }` there, which leaves
    /// `WaveRate::tokens` at zero and so skips `judge_prefill`'s gain
    /// comparison — the thing this module's header recorded as a defect. It is
    /// not one, and this pins why: what the charge exists to deny is the head
    /// waiver, and it denies it by marking the wave non-empty, not by the token
    /// count. So the first offer is still floor-checked, and the decode model
    /// still judges the turn it will become.
    ///
    /// The comparison that is skipped has nothing to compare against: a wave
    /// carrying no prefill rows has a prefill rate of zero, so every positive
    /// rate is an unbounded gain. The judgement that matters on such a wave is
    /// the decode one, and that is live — the continuations were charged, so
    /// `decode_would_carry` has a `decodes > 0` to weigh against.
    #[test]
    fn a_busy_engine_with_no_standing_rows_is_still_judged_by_the_decode_model() {
        // A planner whose decode side has been measured — below
        // `MIN_DECODE_SAMPLES` the second judge abstains by design. The default
        // minimum gain, not the zero the other tests use: a bus-bound decode
        // wave's rate is *flat* in the number of decodes (the copy grows with
        // the routed set, so the step grows with it), and flat is refused as
        // `Saturated` rather than as `Worse`. A short layer time is what puts the
        // wave on the bus: a 20 ms step is long enough to be learned as compute
        // (it outlasts even a fully resident decode's copy), and short enough
        // that at the residency below the copy a wider wave streams outruns the
        // layer it hides under — so the decode model is what refuses.
        let mut rate = WaveRate::with_link_rate(
            25e9,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel::default(),
        );
        for _ in 0..WaveRate::MIN_DECODE_SAMPLES {
            rate.observe_decode(1, 0, u64::MAX, 20e-3);
        }
        assert!(rate.decode_samples() >= WaveRate::MIN_DECODE_SAMPLES);

        // Room to spare, so nothing here is a floor refusal: the decode model
        // is the only thing that can say no.
        struct Decoding(Fake);
        impl Ground for Decoding {
            fn active(&self) -> usize {
                self.0.active()
            }
            fn decodes_active(&self) -> usize {
                self.0.decodes_active()
            }
            fn settled(&self) -> bool {
                self.0.settled()
            }
            fn headroom(&self) -> Headroom {
                self.0.headroom()
            }
            fn budget(&self) -> Budget {
                self.0.budget()
            }
            fn resident_weights(&self) -> u64 {
                self.0.resident_weights()
            }
            fn standing_rows(&self) -> usize {
                self.0.standing_rows()
            }
            fn peek(&mut self, kind: Kind, prio: DecodePriority) -> Option<Cost> {
                // The turn this prefill becomes goes on to decode, which is
                // what puts the second judge in play.
                self.0.peek(kind, prio).map(|c| Cost {
                    decodes_after: true,
                    ..c
                })
            }
            fn admit(&mut self, kind: Kind, prio: DecodePriority, cost: Cost) -> bool {
                self.0.admit(kind, prio, cost)
            }
        }

        // Residency far above the floor, so nothing here can be a weight
        // refusal: whatever stops the band is a rate judgement. And below the
        // experts' total, so a decode streams what is not resident — with every
        // expert on the card a decode copies nothing, its rate climbs with every
        // row, and the decode model has nothing to refuse.
        let mut f = Fake::new(Vec::new(), vec![REGION; 8], 12 << 30);
        f.active = 4;
        f.decodes_active = 4;
        f.standing = 0;
        let mut g = Decoding(f);
        let got = fill(&mut g, &mut rate);
        assert!(
            got.stopped_on_rate,
            "the decode model must be what stops the band: {got:?}"
        );
        assert!(
            !got.stopped_on_weights,
            "with the floor nowhere near, no weight rule may fire: {got:?}"
        );
    }

    /// **A held creep group is charged before anything is offered.** It rides
    /// the next wave whole, so the offers behind it are judged against a wave
    /// that already carries its rows — and it too denies the head waiver.
    #[test]
    fn a_standing_creep_group_is_charged_and_denies_the_head_waiver() {
        let mut f = Fake::new(Vec::new(), vec![REGION; 4], FLOOR);
        f.standing = 300; // a creep group held from the last wave
        let got = fill_once(&mut f);
        assert_eq!(got.prefills, 0, "the wave is not empty, so nothing is free");
        assert!(got.stopped_on_weights);
    }

    /// **…and only the head**, even when the caller cannot see the admission
    /// yet.
    ///
    /// A real `Ground` does not move a prefill into its active set until the
    /// fill returns, so `active()` reads zero for the whole pass. A waiver
    /// keyed on that alone would fire for every queued item: run CJ drained to
    /// zero with 83 queued, took 56 prefills in one fill and drove the weight
    /// zone from 8,180 MiB to 1,417. This `Fake` reproduces that blindness
    /// deliberately — `admit` leaves `active` untouched — so the waiver has to
    /// come from the *wave*, which the model tracks itself.
    #[test]
    fn a_ground_that_cannot_see_its_own_admissions_still_takes_only_the_head() {
        struct Blind(Fake);
        impl Ground for Blind {
            fn active(&self) -> usize {
                0 // never observes what this pass admitted
            }
            fn decodes_active(&self) -> usize {
                // Nothing is running, which is the case this test is about: the
                // waiver fires, and the only thing that may stop the ten behind
                // the head is the model's own count. The decode-start rule is
                // out regardless, because `starting_decodes` also requires
                // `active() > 0` and this ground reports zero.
                0
            }
            fn settled(&self) -> bool {
                true
            }
            fn headroom(&self) -> Headroom {
                self.0.headroom()
            }
            fn budget(&self) -> Budget {
                self.0.budget()
            }
            fn resident_weights(&self) -> u64 {
                self.0.resident_weights()
            }
            fn standing_rows(&self) -> usize {
                self.0.standing_rows()
            }
            fn peek(&mut self, k: Kind, p: DecodePriority) -> Option<Cost> {
                self.0.peek(k, p)
            }
            fn admit(&mut self, k: Kind, p: DecodePriority, c: Cost) -> bool {
                self.0.admit(k, p, c)
            }
        }
        // Ten queued at the floor: without the wave's own count every one lands.
        let mut b = Blind(Fake::new(Vec::new(), vec![REGION; 10], FLOOR));
        let got = fill(&mut b, &mut planner());
        assert_eq!(got.prefills, 1, "the head, and nothing behind it");
        assert!(got.stopped_on_weights);
    }

    /// Background work offers prefill before decode, and the budget binds on the
    /// prefills — the side that is actually asking for new ground. The decodes
    /// behind them step regardless, because they are continuations rather than
    /// admissions; see `an_active_decode_steps_even_with_no_budget_left`.
    #[test]
    fn background_prefill_precedes_background_decode_and_the_budget_binds() {
        let mut f = Fake::new(vec![0; 3], vec![REGION; 3], FLOOR + 2 * REGION);
        f.active = 1; // something already running, so there is no head waiver
        f.decodes_active = 1; // and the decode-start rule does not apply
        let got = fill_once(&mut f);
        assert_eq!(got.prefills, 2, "two regions above the floor, two prefills");
        assert_eq!(f.resident, FLOOR, "exactly to the floor");
        assert!(got.stopped_on_weights, "the third did not");
        assert_eq!(got.decodes, 3, "and the running decodes are unaffected");
        assert_eq!(
            f.taken,
            vec![
                Kind::Prefill,
                Kind::Prefill,
                Kind::Decode,
                Kind::Decode,
                Kind::Decode
            ],
            "prefill leads the band, decode follows",
        );
    }

    /// **A section is an admission, gated and counted like a prefill.** It is
    /// offered ahead of background prefill, it spends the budget the prefills
    /// behind it then see, and a section that does not fit stops the section
    /// band without stopping the prefill band behind it.
    #[test]
    fn sections_are_gated_like_prefills_and_offered_ahead_of_background_prefill() {
        let mut f = Fake::new(Vec::new(), vec![REGION; 2], FLOOR + 2 * REGION);
        f.sections = vec![REGION, REGION, REGION];
        f.active = 1; // something running, so there is no head waiver
        f.decodes_active = 1;
        let got = fill_once(&mut f);
        assert_eq!(got.sections, 2, "two regions above the floor, two sections");
        assert_eq!(
            got.prefills, 0,
            "and the prefills behind them found none left"
        );
        assert!(got.stopped_on_weights);
        assert_eq!(f.taken, vec![Kind::Section, Kind::Section]);
        assert_eq!(f.sections.len(), 1, "the third section waits, FIFO");

        // With nothing running, the head — a section — is admitted regardless
        // of what it costs, and only the head: the wave is no longer empty.
        let mut f = Fake::new(Vec::new(), vec![REGION], FLOOR);
        f.sections = vec![64 << 30, 64 << 30];
        let got = fill_once(&mut f);
        assert_eq!((got.sections, got.prefills), (1, 0));
        assert!(got.stopped_on_weights);
    }

    /// A person waiting is served before any background work.
    #[test]
    fn an_interactive_decode_precedes_background_prefill() {
        let mut f = Fake::new(vec![0; 2], vec![REGION; 2], FLOOR + 4 * REGION);
        f.active = 1;
        f.decodes_active = 1;
        f.decode_prio = DecodePriority::High;
        let got = fill_once(&mut f);
        assert_eq!((got.decodes, got.prefills), (2, 2));
        assert_eq!(
            f.taken,
            vec![Kind::Decode, Kind::Decode, Kind::Prefill, Kind::Prefill],
        );
    }

    /// With no decode running, one is started only while the zone stays in its
    /// upper half — the margin that keeps the expert cache warm.
    #[test]
    fn the_first_decode_is_held_to_the_midpoint() {
        let mut f = Fake::new(vec![0], Vec::new(), FLOOR + 4 * REGION);
        f.active = 2;
        f.decodes_active = 0;
        f.zone = 9 << 30; // clear of the 7 GiB midpoint
        assert_eq!(fill_once(&mut f).decodes, 1);

        let mut f = Fake::new(vec![0], Vec::new(), FLOOR + 4 * REGION);
        f.active = 2;
        f.decodes_active = 0;
        f.zone = 6 << 30; // under it
        let got = fill_once(&mut f);
        assert_eq!(got.decodes, 0, "not while the zone is low");
        assert!(got.stopped_on_weights, "and the producer hears why");
    }

    /// **An active decode steps whatever the budget says.** Its ground was
    /// reserved when the slot was admitted and a slot is held to completion, so
    /// refusing it here refuses work already paid for — and refuses it exactly
    /// when the device is tightest, which is when the engine most needs those
    /// turns to finish and hand their ground back. Runs CB and CD wedged there:
    /// 40 of 40 waves running no forward at all, fifty-odd slots admitted, the
    /// queue backing up and nothing able to complete.
    #[test]
    fn an_active_decode_steps_even_with_no_budget_left() {
        let mut f = Fake::new(vec![0; 3], vec![REGION; 2], FLOOR);
        f.active = 8;
        f.decodes_active = 3; // already running, so the decode-start rule is out
        let got = fill_once(&mut f);
        assert_eq!(got.decodes, 3, "every active decode is stepped");
        assert_eq!(got.prefills, 0, "while new work still waits for room");
        assert!(got.stopped_on_weights, "the prefill side did stop");
        assert_eq!(f.resident, FLOOR, "and the floor held");
    }

    /// **A latched wave still steps its decodes.** The rate model closes the
    /// wave to further *admissions* on its first refusal; a continuation is not
    /// an admission, so the decodes behind a refused prefill band ride anyway.
    /// This is the CB/CD wedge in one assertion: refuse them here and prefills
    /// promote into decodes that never step, nothing completes, and no ground
    /// ever comes back to reopen the wave.
    #[test]
    fn a_wave_closed_to_admissions_still_steps_its_continuations() {
        let mut f = Fake::new(vec![0; 5], vec![64 << 30; 2], FLOOR);
        f.active = 9;
        f.decodes_active = 5;
        let got = fill_once(&mut f);
        assert_eq!(got.prefills, 0, "the wave closed to new work");
        assert_eq!(
            got.decodes, 5,
            "and stepped every decode it was already running"
        );
        assert_eq!(f.taken, vec![Kind::Decode; 5]);
    }

    /// **A wave that has stopped paying is not a device that has run out**, and
    /// the two are reported apart: one is the queue's turn to wait, the other
    /// is backpressure the producer must act on.
    #[test]
    fn a_saturated_wave_and_a_starved_one_report_differently() {
        // Saturated: residency to spare, but the wave is as fast as it gets for
        // what each row spends. Each offer dislodges a region the forward then
        // copies — the spend the minimum gain exists to weigh; an offer that
        // dislodged nothing would cost only its compute, which the queued prompt
        // pays whenever it runs, and is never refused as saturated.
        let mut f = Fake::new(Vec::new(), vec![REGION; 64], FLOOR + 512 * REGION);
        f.active = 1;
        f.decodes_active = 1;
        let got = fill(&mut f, &mut planner().with_min_gain(0.05));
        assert!(got.prefills > 0, "it widened first");
        assert!(got.stopped_on_rate, "then stopped paying");
        assert!(!got.stopped_on_weights, "with the floor nowhere near");
        assert!(f.resident > FLOOR + 500 * REGION);

        // Starved: the floor is what stopped it.
        let mut f = Fake::new(Vec::new(), vec![REGION; 64], FLOOR + 2 * REGION);
        f.active = 1;
        f.decodes_active = 1;
        let got = fill_once(&mut f);
        assert_eq!(got.prefills, 2);
        assert!(got.stopped_on_weights);
        assert!(!got.stopped_on_rate);
    }

    /// **A `Ground` whose `admit` claims nothing must still defend the floor
    /// across a whole pass.**
    ///
    /// Every other test here uses a `Fake` that subtracts the dislodge on
    /// admit, so residency falls as the pass proceeds and the floor bites on its
    /// own. The production ground does not: its K/V is claimed per chunk by the
    /// forward, not by admission, so the live region count the zone is derived
    /// from stands still for the whole pass. Read naively, `before` is then
    /// identical for the first offer and the tenth, each one is measured against
    /// a floor it clears alone, and a pass admits ten times the ground it
    /// checked for once.
    ///
    /// So the production ground carries the running total itself. This pins the
    /// property that makes that necessary: with residency frozen, the fill must
    /// still stop.
    #[test]
    fn a_ground_that_claims_nothing_still_stops_at_the_floor() {
        /// Residency that never moves, with the pass's own commitment netted off
        /// — `AdmitPass::committed` in miniature.
        struct Frozen {
            inner: Fake,
            resident: u64,
            committed: u64,
        }
        impl Ground for Frozen {
            fn active(&self) -> usize {
                self.inner.active()
            }
            fn decodes_active(&self) -> usize {
                self.inner.decodes_active()
            }
            fn settled(&self) -> bool {
                self.inner.settled()
            }
            fn headroom(&self) -> Headroom {
                self.inner.headroom()
            }
            fn budget(&self) -> Budget {
                Budget {
                    resident: self.resident,
                    ..self.inner.budget()
                }
            }
            fn resident_weights(&self) -> u64 {
                self.resident.saturating_sub(self.committed)
            }
            fn standing_rows(&self) -> usize {
                self.inner.standing_rows()
            }
            fn peek(&mut self, kind: Kind, prio: DecodePriority) -> Option<Cost> {
                self.inner.peek(kind, prio)
            }
            fn admit(&mut self, kind: Kind, prio: DecodePriority, cost: Cost) -> bool {
                // The production shape: the engine's own residency is untouched
                // by admitting. Only the pass's tally moves.
                if !self.inner.admit(kind, prio, cost) {
                    return false;
                }
                self.committed = self.committed.saturating_add(cost.dislodged_bytes());
                true
            }
        }

        // Eight region-sized turns against four regions of room above the floor.
        let resident = FLOOR + 4 * REGION;
        let mut g = Frozen {
            inner: Fake::new(Vec::new(), vec![REGION; 8], resident),
            resident,
            committed: 0,
        };
        g.inner.active = 1;
        g.inner.decodes_active = 1;
        let got = fill(&mut g, &mut planner());

        assert!(
            got.prefills <= 4,
            "a frozen ground must not admit past the floor: {got:?}",
        );
        assert!(
            got.stopped_on_weights,
            "and must stop because of it, not run out of queue: {got:?}",
        );
    }

    /// **A deep queue on a roomy card fills the wave, not one turn per pass.**
    ///
    /// The shape a live daemon produced when the integration went in: a dozen
    /// conversations arriving at once, nothing in flight, and a card with room
    /// to spare. Filling the wave is the entire point — a pass that takes one
    /// item and leaves eleven queued is the byte-fit planner's behaviour, which
    /// is what this replaced.
    #[test]
    fn a_deep_queue_on_a_roomy_card_fills_the_wave() {
        let mut f = Fake::new(Vec::new(), vec![REGION; 12], FLOOR + 400 * REGION);
        f.prefill_prio = DecodePriority::High;
        let got = fill_once(&mut f);
        assert!(
            got.prefills > 1,
            "one admission per pass leaves the queue standing: {got:?}"
        );
        assert!(
            !got.skipped,
            "a settled engine must offer its queue: {got:?}"
        );
    }

    /// **FIFO, not cheapest-first.** An item the budget cannot take ends its
    /// band; the cheap items behind it do not jump the queue.
    #[test]
    fn a_band_stops_at_its_head_rather_than_passing_it_over() {
        let mut f = Fake::new(
            Vec::new(),
            vec![64 << 30, REGION, REGION],
            FLOOR + 2 * REGION,
        );
        f.active = 1;
        f.decodes_active = 1;
        let got = fill_once(&mut f);
        assert_eq!(got.prefills, 0, "the head did not fit, so nothing did");
        assert!(got.stopped_on_weights);
        assert_eq!(f.prefills.len(), 3, "and nothing was consumed");
    }
}
