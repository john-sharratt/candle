//! Admission: who runs next, what each admission costs, and whether the wave is
//! better off carrying it.
//!
//! The fill admits prefills and sections. Decodes reach the wave as
//! continuations, are charged to it before the first offer, and are never judged
//! again. The decision is a throughput comparison rather than a byte fit: a
//! prefill forward pays the same expert copy at any width, so a narrow wave is
//! slower per row and not cheaper, and an offer is taken only while it makes the
//! wave faster. Admission is skipped entirely on a wave where nothing finished,
//! because only a finished slot opens room.
//!
//! - `cost`: what one admission takes from the ground.
//! - `gate`: the two limits no answer to an offer may cross.
//! - `ground`: the trait the fill is run against.
//! - `order`: whose turn it is.
//! - `outcome`: how a pass ended and why an offer was refused.
//! - `pass_budget`: the tokens one prefill forward may carry.
//! - `tests_fill`: tests of the fill against a fake device.
//! - `tests_policy`: tests of the admission order and the waiver.

pub(crate) mod cost;
pub(crate) mod gate;
pub(crate) mod order;
pub(crate) mod pass_budget;
pub mod rate;

pub(crate) use cost::Cost;
pub(crate) use gate::Headroom;
pub(crate) use order::{Kind, Order};
pub(crate) use rate::{Admission, WaveRate};

use std::time::{Duration, Instant};

use crate::projection::DecodePriority;

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
        loop {
            let t_peek = Instant::now();
            let offered = ground.peek(kind, prio);
            ground.charge(FillPart::Peek, t_peek.elapsed());
            let Some(cost) = offered else {
                break;
            };
            let total = cost.total();
            // The truth as of this offer, and what this offer would leave of
            // it — its region claim **and its tier**, which the weight side
            // loses alike. See `Cost::dislodged_bytes`: the tier is transient
            // per forward but published per wave, and the growth term is
            // bounded by it, so a wave that widens holds that ground against
            // the weight side for as long as it stays that wide.
            let t_residency = Instant::now();
            let before = ground.resident_weights();
            ground.charge(FillPart::Residency, t_residency.elapsed());
            let after = before.saturating_sub(cost.dislodged_bytes());
            let t_judge = Instant::now();
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
                                log_refusal(rate, kind, prio, &cost, before, after, &budget, r);
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
                // **A band the running work yields to is not held behind it.**
                // The floors defend the residency the running slots decode at,
                // and slots of a lower band are what spent it; they hold it until
                // they finish. A turn they yield to is judged on the row cap
                // alone (`WaveRate::try_outranking`), and a cap refusal re-offers
                // it as a join like a rate refusal does.
                //
                // **Outranking orders the queue; it does not exempt the trade.**
                // The decodes of its own band and above are not the lower band's
                // and did not take the residency it spends, and they step slower
                // for every byte of it — so an outranking turn is traded against
                // them, as any turn is against the decodes running. Judged on the
                // cap alone, every `High` tool result in a heavy-read was admitted
                // while a `Low` pass ran: on BlackWorld (e00cb4300, 2026-10-20)
                // all 46 admissions of four minutes outranked, none was refused,
                // and the expert zone stood 6.4 GB under the residency the
                // decodes were held to while their step went 98 → 614 ms.
                // **Only a turn somebody waits on spends the decodes' hold.**
                // The exemption above exists for a person: their tool result is
                // not to wait for background work to drain. A band nobody waits
                // on token by token that merely outranks lower work — a
                // calibration sweep over a `Low` ingest — has no such claim, and
                // judged on the prefill floor alone it took every byte the
                // decodes stand on: on BlackWorld (b5d60732, 2026-11-08) every
                // one of 69 admissions a pass was outranking calibration, none was
                // refused, and residency fell to 4.1 GB against a 29.7 GB hold
                // while the expert hit rate went from 99% to 61%. Such a turn is
                // refused where it would leave residency under the decode floor;
                // the trades and the floors judge it as before above it.
                Kind::Prefill
                    if ground.outranks_active(prio)
                        && breaks_the_hold(prio, &cost, after, budget.decode_floor) =>
                {
                    let r = rate::Refusal::Floor {
                        resident_after: after,
                        floor: budget.decode_floor,
                    };
                    out.note(r);
                    log_refusal(rate, kind, prio, &cost, before, after, &budget, r);
                    false
                }
                Kind::Prefill if ground.outranks_active(prio) => {
                    let peers = ground.decodes_at_or_above(prio);
                    let verdict = if cost.rows > 0 && peers > 0 {
                        rate.try_widen_trading(
                            cost.rows,
                            before,
                            after,
                            ground.prefill_weight(prio),
                            peers as f64,
                        )
                    } else if peers > 0 {
                        // Its rows refused, it joins only at no cost to them.
                        rate.try_join_trading(before, after, peers as f64, cost.decodes_after)
                    } else {
                        rate.try_outranking(cost.rows, after)
                    };
                    match verdict {
                        rate::Admit::Admitted { .. } => true,
                        rate::Admit::Refused(
                            r @ (rate::Refusal::Cap { .. }
                            | rate::Refusal::Saturated { .. }
                            | rate::Refusal::Worse { .. }),
                        ) if cost.rows > 0 && ground.close_rows() => {
                            log_refusal(rate, kind, prio, &cost, before, after, &budget, r);
                            ground.charge(FillPart::Judge, t_judge.elapsed());
                            continue;
                        }
                        rate::Admit::Refused(r) => {
                            // A join running ahead of a full forward's members
                            // ends its band alone (`Cost::ahead`).
                            if !cost.ahead {
                                rate.latch_full();
                            }
                            out.note(r);
                            log_refusal(rate, kind, prio, &cost, before, after, &budget, r);
                            false
                        }
                    }
                }
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
                    // A turn that widens the forward and then decodes is judged on
                    // one trade, both phases at once (`WaveRate::try_widen_trading`);
                    // the separate decode question is for the joins.
                    // **A section beside running decodes trades too.** It decodes
                    // nothing itself, but its rows dislodge the residency the
                    // running decodes step on exactly as a turn's do; judged on
                    // the prefill rate alone it paid none of their loss. On
                    // WhiteWorld (c3ebac8bf, 2026-10-19) a 6,968-row file ingest
                    // went in beside seven decodes while the dialogue prompts
                    // queued behind it were refused, and held the ground for a
                    // minute and a half: ten openings waited 132 s for it.
                    let trading = cost.rows > 0
                        && match kind {
                            Kind::Prefill => cost.decodes_after,
                            Kind::Section => ground.decodes_active() > 0,
                            Kind::Decode => false,
                        };
                    // An offer of no rows joins a forward whose rows are already
                    // spent: it shares the forwards to come rather than widening
                    // this one, so there is no prefill gain to judge — only the
                    // floor and, beside running decodes, their rate: with the
                    // join's own decode in the wave when it goes on to decode
                    // (`WaveRate::try_join_trading`). The decode that follows
                    // steps one row at a time; a drafted turn is wider, and would
                    // only be refused more readily than this asks for.
                    //
                    // A prefill whose rows the rate model turns down for the rate
                    // alone has not been turned away: the forward's rows are spent,
                    // and the ground re-offers the same turn as a join on the next
                    // peek (`Ground::close_rows`, `WaveRate::try_widen`).
                    match if cost.rows == 0 {
                        rate.try_join_trading(
                            before,
                            after,
                            ground.decodes_active() as f64,
                            cost.decodes_after,
                        )
                    } else if trading {
                        rate.try_widen_trading(
                            cost.rows,
                            before,
                            after,
                            ground.prefill_weight(prio),
                            ground.decodes_active() as f64,
                        )
                    } else if kind == Kind::Prefill {
                        rate.try_widen(cost.rows, before, after)
                    } else {
                        rate.try_admit(Admission::Prefill { tokens: cost.rows }, before, after)
                    } {
                        rate::Admit::Admitted { .. } => true,
                        rate::Admit::Refused(
                            r @ (rate::Refusal::Saturated { .. } | rate::Refusal::Worse { .. }),
                        ) if kind == Kind::Prefill && !cost.ahead && ground.close_rows() => {
                            log_refusal(rate, kind, prio, &cost, before, after, &budget, r);
                            ground.charge(FillPart::Judge, t_judge.elapsed());
                            continue;
                        }
                        rate::Admit::Refused(r) => {
                            // `try_widen` leaves a rate refusal unlatched for
                            // the join above; with no join to make, it ends
                            // the wave as `try_admit`'s does — unless it is a
                            // join running ahead of a full forward's members,
                            // which ends its band alone (`Cost::ahead`).
                            if !cost.ahead {
                                rate.latch_full();
                            }
                            out.note(r);
                            log_refusal(rate, kind, prio, &cost, before, after, &budget, r);
                            false
                        }
                    }
                }
            };
            ground.charge(FillPart::Judge, t_judge.elapsed());
            if !allowed {
                break;
            }
            if kind != Kind::Decode {
                rate.note_admitted(ground.outranks_active(prio), after, budget.prefill_floor);
                // The refusal's twin: a pass that admits more than its readings
                // allow is read from these, not from the pass total.
                tracing::debug!(
                    target: "candle_conversation::scheduler::admission",
                    kind = ?kind,
                    prio = ?prio,
                    outranking = ground.outranks_active(prio),
                    rows = cost.rows,
                    tier_mib = cost.activations >> 20,
                    dislodged_mib = cost.dislodged_bytes() >> 20,
                    hole_credit_mib = cost.hole_credit >> 20,
                    resident_before_mib = before >> 20,
                    resident_after_mib = after >> 20,
                    prefill_floor_mib = budget.prefill_floor >> 20,
                    "offer admitted",
                );
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

/// Whether an outranking prefill of `prio` costing `cost` would leave the
/// weight residency at `after`, under the decode floor the running decodes are
/// held to, without a person waiting on it: a band that is not
/// [`DecodePriority::is_live`] and dislodges something. A live band may spend
/// the hold down to the prefill floor; a claim landing wholly in holes
/// dislodges nothing and breaks no hold.
fn breaks_the_hold(prio: DecodePriority, cost: &Cost, after: u64, decode_floor: u64) -> bool {
    !prio.is_live() && cost.dislodged_bytes() > 0 && after < decode_floor
}

/// One decode offer as the rate model reads it: a verify block of `1 + draft`
/// rows, so a plain decode is one row and no draft.
fn decode_of(cost: &Cost) -> Admission {
    Admission::Decode {
        draft: cost.rows.saturating_sub(1),
    }
}

mod ground;
mod outcome;
#[cfg(test)]
mod tests_fill;
#[cfg(test)]
mod tests_policy;

pub(crate) use ground::*;
pub(crate) use outcome::*;
