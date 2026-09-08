//! Admission: who runs next, what it costs, and whether the device can take it.
//!
//! Three concerns, three files, no overlap:
//!
//! * [`order`] — whose turn it is. Pure policy; touches no device.
//! * [`cost`] — what one admission takes. Arithmetic over settled state.
//! * [`gate`] — whether it may proceed. Four rules, no fifth.
//!
//! [`fill`] is the loop that asks them in that order.
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

pub(crate) use cost::Cost;
pub(crate) use gate::Headroom;
pub(crate) use order::{Kind, Order};

use crate::projection::DecodePriority;

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
    /// An admission was refused because it would have reached the weight zone.
    pub stopped_on_weights: bool,
}

/// Offer the wave's rows, in order, to whatever the device can take.
pub(crate) fn fill<G: Ground>(ground: &mut G) -> Filled {
    let mut out = Filled::default();
    if !ground.settled() {
        out.skipped = true;
        return out;
    }
    // Measured once, after eviction — then spent down as admissions take from
    // it. Re-reading the device per admission would be a query per item for a
    // figure this pass is itself moving; charging every item against the
    // *opening* figure would let a whole band through on one item's worth of
    // room, which is what the arithmetic is here to prevent.
    let mut room = ground.headroom();
    let mut order = Order::new();
    while let Some((prio, kind)) = order.next() {
        // The band, FIFO, until it runs dry or the device says no. A refusal
        // ends this band and not the pass: a later band's work is a different
        // size and may still fit.
        while let Some(cost) = ground.peek(kind, prio) {
            let total = cost.total();
            let allowed = match kind {
                // A decode when none is running is the one case with its own
                // rule — it keeps the expert cache warm.
                Kind::Decode if ground.decodes_active() == 0 && ground.active() > 0 => {
                    gate::may_start_decode(0, total, &room)
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
                // **Count what this pass has already taken.** A prefill admitted
                // here does not reach `active_prefills` until the fill returns —
                // it sits in `prefill_admitted` — so `Ground::active` reads the
                // same value all pass. Rule 1 waives the budget when nothing is
                // running, and without this it waives it for *every* item in the
                // queue rather than for the head: run CJ drained to zero with 83
                // queued, admitted **56 prefills in a single fill**, and put the
                // weight zone from 8,180 MiB to 1,417 with every region live.
                // The rule is "admit the next one regardless", and the next one
                // is one.
                //
                // A section is a first admission exactly as a prefill is: fresh
                // K/V on a scratch slot, a store, a tier row — so it is gated the
                // same way and counted the same way. Sections used to enter the
                // wave without passing here at all, and one minute of them took
                // the weight zone from 10,398 MiB to its hold.
                Kind::Prefill | Kind::Section => gate::may_admit(
                    ground.active() + out.prefills + out.sections,
                    total,
                    &room,
                ),
            };
            if !allowed {
                out.stopped_on_weights = true;
                break;
            }
            if !ground.admit(kind, prio, cost) {
                break;
            }
            room.free_kv = room.free_kv.saturating_sub(total);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake device: fixed per-item costs, a fixed budget, and a record of the
    /// order things were taken in.
    struct Fake {
        decodes: Vec<u64>,
        prefills: Vec<u64>,
        sections: Vec<u64>,
        free: u64,
        active: usize,
        decodes_active: usize,
        settled: bool,
        zone: u64,
        taken: Vec<Kind>,
        decode_prio: DecodePriority,
        prefill_prio: DecodePriority,
    }

    impl Fake {
        fn new(decodes: Vec<u64>, prefills: Vec<u64>, free: u64) -> Self {
            Self {
                decodes,
                prefills,
                sections: Vec::new(),
                free,
                active: 0,
                decodes_active: 0,
                settled: true,
                zone: 9 << 30,
                taken: Vec::new(),
                decode_prio: DecodePriority::Low,
                prefill_prio: DecodePriority::Low,
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
                free_kv: self.free,
                zone: self.zone,
                zone_min: 4 << 30,
                zone_max: 10 << 30,
            }
        }
        fn peek(&mut self, kind: Kind, prio: DecodePriority) -> Option<Cost> {
            if prio != self.wants(kind) {
                return None;
            }
            self.queue(kind).first().map(|&kv| Cost {
                kv,
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
            self.free = self.free.saturating_sub(priced);
            self.active += 1;
            if kind == Kind::Decode {
                self.decodes_active += 1;
            }
            self.taken.push(kind);
            true
        }
    }

    /// Nothing finished, so nothing can newly fit: the pass costs nothing.
    #[test]
    fn an_unsettled_wave_takes_the_fast_path() {
        let mut f = Fake::new(vec![1; 4], vec![1; 4], u64::MAX);
        f.settled = false;
        let got = fill(&mut f);
        assert_eq!(
            got,
            Filled {
                skipped: true,
                ..Default::default()
            },
        );
        assert!(f.taken.is_empty(), "nothing offered, nothing taken");
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
        let mut f = Fake::new(vec![1; 2], vec![1; 2], u64::MAX);
        f.active = 32; // busy, and nothing has completed
        f.decodes_active = 1;
        f.settled = true; // but the engine ran no forward, so: re-offer
        let got = fill(&mut f);
        assert!(!got.skipped);
        assert!(
            got.prefills + got.decodes > 0,
            "a settled ground with queued work must be able to admit",
        );
    }

    /// **The head runs even when it does not fit**, so a slot too large for the
    /// card cannot block the queue behind it forever.
    #[test]
    fn an_empty_engine_admits_the_head_whatever_it_costs() {
        let mut f = Fake::new(Vec::new(), vec![u64::MAX, 1], 0);
        let got = fill(&mut f);
        assert_eq!(got.prefills, 1, "the head lands");
        assert_eq!(f.active, 1);
        assert!(got.stopped_on_weights, "and the next one does not");
    }

    /// **…and only the head**, even when the caller cannot see the admission
    /// yet.
    ///
    /// A real `Ground` does not move a prefill into its active set until the
    /// fill returns, so `active()` reads zero for the whole pass. If the
    /// budget-waiver keyed on that alone it would waive for every queued item:
    /// run CJ drained to zero with 83 queued, took 56 prefills in one fill and
    /// drove the weight zone from 8,180 MiB to 1,417. This `Fake` reproduces
    /// that blindness deliberately — `admit` leaves `active` untouched — so the
    /// count the gate sees has to come from the pass itself.
    #[test]
    fn a_ground_that_cannot_see_its_own_admissions_still_takes_only_the_head() {
        struct Blind(Fake);
        impl Ground for Blind {
            fn active(&self) -> usize {
                0 // never observes what this pass admitted
            }
            fn decodes_active(&self) -> usize {
                1 // a decode is running, so the decode-start rule is out
            }
            fn settled(&self) -> bool {
                true
            }
            fn headroom(&self) -> Headroom {
                self.0.headroom()
            }
            fn peek(&mut self, k: Kind, p: DecodePriority) -> Option<Cost> {
                self.0.peek(k, p)
            }
            fn admit(&mut self, k: Kind, p: DecodePriority, c: Cost) -> bool {
                self.0.admit(k, p, c)
            }
        }
        // Ten queued, no room at all: without the pass count every one lands.
        let mut b = Blind(Fake::new(Vec::new(), vec![1_000; 10], 0));
        let got = fill(&mut b);
        assert_eq!(got.prefills, 1, "the head, and nothing behind it");
        assert!(got.stopped_on_weights);
    }

    /// Background work offers prefill before decode, and the budget binds on the
    /// prefills — the side that is actually asking for new ground. The decodes
    /// behind them step regardless, because they are continuations rather than
    /// admissions; see `an_active_decode_steps_even_with_no_budget_left`.
    #[test]
    fn background_prefill_precedes_background_decode_and_the_budget_binds() {
        let mut f = Fake::new(vec![10; 3], vec![10; 3], 25);
        f.active = 1; // something already running, so rule 3 applies
        f.decodes_active = 1; // and the decode-start rule does not
        let got = fill(&mut f);
        assert_eq!(got.prefills, 2, "two fit in 25 bytes of room");
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
        let mut f = Fake::new(Vec::new(), vec![10; 2], 25);
        f.sections = vec![10, 10, 10];
        f.active = 1; // something running, so the budget binds
        f.decodes_active = 1;
        let got = fill(&mut f);
        assert_eq!(got.sections, 2, "two sections fit in 25 bytes of room");
        assert_eq!(got.prefills, 0, "and the prefills behind them found none left");
        assert!(got.stopped_on_weights);
        assert_eq!(f.taken, vec![Kind::Section, Kind::Section]);
        assert_eq!(f.sections.len(), 1, "the third section waits, FIFO");

        // With nothing running, the head — a section — is admitted regardless
        // of budget, and only the head: the pass counts its own sections.
        let mut f = Fake::new(Vec::new(), vec![1], 0);
        f.sections = vec![1_000, 1_000];
        let got = fill(&mut f);
        assert_eq!((got.sections, got.prefills), (1, 0));
        assert!(got.stopped_on_weights);
    }

    /// A person waiting is served before any background work.
    #[test]
    fn an_interactive_decode_precedes_background_prefill() {
        let mut f = Fake::new(vec![1; 2], vec![1; 2], 1_000);
        f.active = 1;
        f.decodes_active = 1;
        f.decode_prio = DecodePriority::High;
        let got = fill(&mut f);
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
        let mut f = Fake::new(vec![1], Vec::new(), 1_000);
        f.active = 2;
        f.decodes_active = 0;
        f.zone = 9 << 30; // clear of the 7 GiB midpoint
        assert_eq!(fill(&mut f).decodes, 1);

        let mut f = Fake::new(vec![1], Vec::new(), 1_000);
        f.active = 2;
        f.decodes_active = 0;
        f.zone = 6 << 30; // under it
        assert_eq!(fill(&mut f).decodes, 0, "not while the zone is low");
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
        let mut f = Fake::new(vec![10; 3], vec![10; 2], 0);
        f.active = 8;
        f.decodes_active = 3; // already running, so the decode-start rule is out
        let got = fill(&mut f);
        assert_eq!(got.decodes, 3, "every active decode is stepped");
        assert_eq!(got.prefills, 0, "while new work still waits for room");
        assert!(got.stopped_on_weights, "the prefill side did stop");
    }

    /// **FIFO, not cheapest-first.** An item the budget cannot take ends its
    /// band; the cheap items behind it do not jump the queue.
    #[test]
    fn a_band_stops_at_its_head_rather_than_passing_it_over() {
        let mut f = Fake::new(Vec::new(), vec![100, 1, 1], 50);
        f.active = 1;
        f.decodes_active = 1;
        let got = fill(&mut f);
        assert_eq!(got.prefills, 0, "the head did not fit, so nothing did");
        assert!(got.stopped_on_weights);
        assert_eq!(f.prefills.len(), 3, "and nothing was consumed");
    }
}
