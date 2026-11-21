//! Tests of who is offered admission, in what order, and when a wave stops
//! taking work.
//!
//! A ground that reports itself settled is always offered work, because skipping
//! admission until something finishes is only safe while the admitted set is
//! advancing. An empty engine admits its head whatever it costs, so an oversized
//! slot cannot block the queue, and a busy engine gets no such waiver. Active
//! decodes step whatever the budget says, and a wave closed to admissions still
//! steps its continuations, since refusing work already paid for is what wedged
//! the engine.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use rate::{DecodeModel, ExpertGeometry, RateModel};

    use super::super::tests_fill::tests::{fill_once, planner, Fake, FLOOR, REGION};

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

    /// **Admission sways with what is waiting.** A turn that widens the forward
    /// and then decodes is judged on one trade (`WaveRate::try_widen_trading`):
    /// the prefill throughput it earns, weighted by the prompts waiting, against
    /// the decode throughput it costs, weighted by the decodes running. So the
    /// same queue on the same ground is taken more sparingly beside many
    /// decodes than beside few, and more freely the more — or the longer — the
    /// prompts have waited.
    #[test]
    fn admission_sways_between_decode_and_prefill_with_what_is_waiting() {
        // A planner whose decode side has been measured — below
        // `MIN_DECODE_SAMPLES` the decode side abstains by design. A 20 ms layer
        // puts the wave on the bus at this residency, so a dislodge slows every
        // running decode and the trade has a real decode side.
        let learned = || {
            let mut rate = WaveRate::with_link_rate(
                25e9,
                ExpertGeometry::QWEN36_35B_A3B,
                RateModel::default(),
                DecodeModel::default(),
            );
            for _ in 0..WaveRate::MIN_DECODE_SAMPLES {
                rate.observe_decode(1, 0, u64::MAX, 20e-3);
            }
            rate
        };
        struct Waiting(Fake, f64);
        impl Ground for Waiting {
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
                // Every turn goes on to decode, which is what puts the decode
                // side of the trade in play.
                self.0.peek(kind, prio).map(|c| Cost {
                    decodes_after: true,
                    ..c
                })
            }
            fn admit(&mut self, kind: Kind, prio: DecodePriority, cost: Cost) -> bool {
                self.0.admit(kind, prio, cost)
            }
            fn outranks_active(&self, prio: DecodePriority) -> bool {
                self.0.outranks_active(prio)
            }
            fn prefill_weight(&self, _prio: DecodePriority) -> f64 {
                self.1
            }
        }
        // Residency far above the floor and below the experts' total, so every
        // dislodge is streamed by the decodes and no floor rule fires.
        let taken = |decodes: usize, waiting: f64| {
            let mut f = Fake::new(Vec::new(), vec![32 * REGION; 32], 12 << 30);
            f.active = decodes;
            f.decodes_active = decodes;
            let mut g = Waiting(f, waiting);
            let mut rate = learned();
            fill(&mut g, &mut rate).prefills
        };
        let beside_many = taken(24, 1.0);
        let beside_few = taken(1, 1.0);
        let many_waiting = taken(24, 24.0);
        assert!(
            beside_many < beside_few,
            "many decodes keep the weights: {beside_many} taken beside 24, {beside_few} beside 1"
        );
        assert!(
            many_waiting > beside_many,
            "prompts waiting pull the balance back: {many_waiting} with 24 waiting, \
             {beside_many} with one"
        );
    }

    /// **A section beside running decodes pays their loss as a prompt does.** It
    /// decodes nothing itself, but its rows dislodge the residency the decodes
    /// step on, so many running decodes hold sections back as they hold prompts
    /// back — judged on the prefill rate alone, they went in regardless.
    #[test]
    fn a_section_beside_running_decodes_is_traded_against_them() {
        let learned = || {
            let mut rate = WaveRate::with_link_rate(
                25e9,
                ExpertGeometry::QWEN36_35B_A3B,
                RateModel::default(),
                DecodeModel::default(),
            );
            for _ in 0..WaveRate::MIN_DECODE_SAMPLES {
                rate.observe_decode(1, 0, u64::MAX, 20e-3);
            }
            rate
        };
        let taken = |decodes: usize| {
            let mut f = Fake::new(Vec::new(), Vec::new(), 12 << 30);
            f.sections = vec![32 * REGION; 32];
            f.active = decodes;
            f.decodes_active = decodes;
            let mut rate = learned();
            fill(&mut f, &mut rate).sections
        };
        let beside_many = taken(24);
        let beside_one = taken(1);
        assert!(
            beside_many < beside_one,
            "many decodes keep the weights from sections too: {beside_many} beside 24, \
             {beside_one} beside 1"
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
            fn outranks_active(&self, p: DecodePriority) -> bool {
                self.0.outranks_active(p)
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
            fn outranks_active(&self, prio: DecodePriority) -> bool {
                self.inner.outranks_active(prio)
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
