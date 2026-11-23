//! Tests of the fill loop against a fake device, covering joins, outranking,
//! the decode floor and the decodes already charged to the wave.
//!
//! The fake device spends residency as each admission takes it, so the tests
//! judge the policy and not the engine. A refused join that runs ahead ends only
//! its own band. A turn that outranks the running work is admitted under the
//! prefill floor, and is admitted under the decode floor only when somebody is
//! waiting on it. The continuations are charged once at the top of the fill,
//! since charging them again would count the same row twice and skew the decode
//! model's hit rate.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use candle_nn::kv_cache::REGION_BYTES;

    use rate::{DecodeModel, ExpertGeometry, RateModel};

    use crate::scheduler::band::rank;

    /// One region of ground, the granularity a claim actually takes.
    pub(in super::super) const REGION: u64 = REGION_BYTES as u64;

    /// A floor with room above it for the tests to spend.
    pub(in super::super) const FLOOR: u64 = 5 << 30;

    /// Rows a prefill or section chunk puts in the forward.
    pub(in super::super) const ROWS: usize = 128;

    /// The engine's planner as these tests use it: the 35B's expert geometry on
    /// the 4090 Mobile's link, with no minimum gain — so an offer is judged on
    /// the floor and on whether it makes the wave slower, and the band policy
    /// under test is not confounded by saturation.
    pub(in super::super) fn planner() -> WaveRate {
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
    pub(in super::super) struct Fake {
        pub(in super::super) decodes: Vec<u64>,
        pub(in super::super) prefills: Vec<u64>,
        pub(in super::super) sections: Vec<u64>,
        /// Resident weights, spent down by what each admission claims.
        pub(in super::super) resident: u64,
        pub(in super::super) standing: usize,
        pub(in super::super) active: usize,
        pub(in super::super) decodes_active: usize,
        pub(in super::super) settled: bool,
        pub(in super::super) zone: u64,
        pub(in super::super) taken: Vec<Kind>,
        pub(in super::super) decode_prio: DecodePriority,
        pub(in super::super) prefill_prio: DecodePriority,
        /// Every prefill after the first offers no rows, as a turn joining a
        /// forward whose rows are spent does.
        pub(in super::super) join_after_head: bool,
        /// Whether `close_rows` is honoured — `AdmitPass`'s behaviour — and
        /// whether it has been called: every prefill offered after it is a join.
        pub(in super::super) closes_rows: bool,
        pub(in super::super) rows_closed: bool,
        /// The band of the slots already running, when one is set.
        pub(in super::super) running_band: Option<DecodePriority>,
        /// The prefill floor, when a case needs it apart from the decode floor.
        pub(in super::super) prefill_floor: u64,
        /// Every prefill is offered as a join — a turn that goes on to decode —
        /// and, with `reports_ahead`, as one running ahead of the members of a
        /// full forward (`Cost::ahead`).
        pub(in super::super) ahead: bool,
        pub(in super::super) reports_ahead: bool,
        /// The card streams its experts ([`Budget::streams_experts`]).
        pub(in super::super) streams: bool,
    }

    impl Fake {
        /// `decodes` and `prefills` carry the K/V each item claims; `resident`
        /// is the weight side it comes out of.
        pub(in super::super) fn new(decodes: Vec<u64>, prefills: Vec<u64>, resident: u64) -> Self {
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
                closes_rows: false,
                rows_closed: false,
                running_band: None,
                prefill_floor: FLOOR,
                ahead: false,
                reports_ahead: false,
                streams: false,
            }
        }
        pub(in super::super) fn queue(&mut self, kind: Kind) -> &mut Vec<u64> {
            match kind {
                Kind::Decode => &mut self.decodes,
                Kind::Prefill => &mut self.prefills,
                Kind::Section => &mut self.sections,
            }
        }
        pub(in super::super) fn wants(&self, kind: Kind) -> DecodePriority {
            match kind {
                Kind::Decode => self.decode_prio,
                Kind::Prefill => self.prefill_prio,
                // This fake's sections are all background ones.
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
                forward_reserve: 0,
            }
        }
        fn budget(&self) -> Budget {
            Budget {
                resident: self.resident,
                // One floor for both kinds here: these cases are about the rate
                // model's gain arithmetic and its caps, not about the prefill /
                // decode split, and giving them the same number keeps each
                // assertion meaning exactly what it meant before the split.
                prefill_floor: self.prefill_floor,
                decode_floor: FLOOR,
                max_rows: 8_192,
                max_decodes: 64,
                streams_experts: self.streams,
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
                Kind::Prefill if self.rows_closed || self.ahead => 0,
                _ => ROWS,
            };
            // A turn joining ahead goes on to decode, as a turn does.
            let decodes_after = kind == Kind::Prefill && self.ahead;
            let ahead = kind == Kind::Prefill && self.reports_ahead;
            self.queue(kind).first().map(|&kv| Cost {
                kv,
                rows,
                decodes_after,
                ahead,
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
        fn close_rows(&mut self) -> bool {
            self.rows_closed = self.closes_rows;
            self.closes_rows
        }
        fn outranks_active(&self, prio: DecodePriority) -> bool {
            self.running_band.is_some_and(|b| rank(b) < rank(prio))
        }
    }

    /// **A join that runs ahead, refused on the trade, ends its band and
    /// nothing more**: the bands after it are still offered. Any other join
    /// refused on the trade ends the wave. (A refusal on the floor latches the
    /// wave inside the rate model, for every admission alike.)
    #[test]
    fn a_refused_join_running_ahead_leaves_the_bands_after_it_offered() {
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
        let section_after_refused_join = |reported: bool| {
            // Beside many decodes, one more does not pay the gain the planner
            // asks for; the section after it claims nothing.
            let mut f = Fake::new(Vec::new(), vec![REGION], 12 << 30);
            f.sections = vec![0];
            f.active = 24;
            f.decodes_active = 24;
            f.prefill_prio = DecodePriority::High;
            f.ahead = true;
            f.reports_ahead = reported;
            let got = fill(&mut f, &mut learned().with_min_gain(0.5));
            assert!(got.stopped_on_rate, "the join was refused on the trade");
            f.taken
        };
        assert_eq!(
            section_after_refused_join(true),
            vec![Kind::Section],
            "the join was refused alone"
        );
        assert!(
            section_after_refused_join(false).is_empty(),
            "an ordinary refused join ends the wave"
        );
    }

    /// **A dialogue turn is not held behind the background work it outranks —
    /// down to the prefill floor, and no further.** Residency already stands
    /// under the decode floor — the running `Low` slots spent it — so a `Low`
    /// turn is refused there, while a `High` one is admitted past it. Under the
    /// prefill floor the committed wave no longer fits, and the `High` turn
    /// waits as well: admitted there, it breaks its own cohort's sweep.
    #[test]
    fn a_turn_outranking_the_running_slots_is_admitted_under_the_floor() {
        let under = FLOOR - 4 * REGION;
        let mut low = Fake::new(Vec::new(), vec![REGION], under);
        low.active = 14;
        low.running_band = Some(DecodePriority::Low);
        let got = fill_once(&mut low);
        assert_eq!(got.prefills, 0, "a turn of the running band waits");
        assert!(got.stopped_on_weights);

        let mut high = Fake::new(Vec::new(), vec![REGION], under);
        high.active = 14;
        high.running_band = Some(DecodePriority::Low);
        high.prefill_prio = DecodePriority::High;
        high.prefill_floor = FLOOR - 8 * REGION;
        let got = fill_once(&mut high);
        assert_eq!(got.prefills, 1, "the band they yield to is admitted");
        assert!(!got.stopped_on_weights);

        let mut broke = Fake::new(Vec::new(), vec![REGION], under);
        broke.active = 14;
        broke.running_band = Some(DecodePriority::Low);
        broke.prefill_prio = DecodePriority::High;
        let got = fill_once(&mut broke);
        assert_eq!(got.prefills, 0, "under the prefill floor it waits too");
        assert!(got.stopped_on_weights);

        let mut peer = Fake::new(Vec::new(), vec![REGION], under);
        peer.active = 14;
        peer.running_band = Some(DecodePriority::High);
        peer.prefill_prio = DecodePriority::High;
        let got = fill_once(&mut peer);
        assert_eq!(got.prefills, 0, "among equals the floor still holds");
    }

    /// **Only a turn somebody waits on spends the decodes' hold.** A `Normal`
    /// turn outranking running `Low` work — a calibration sweep over an ingest
    /// — is refused where it would leave residency under the decode floor,
    /// though the prefill floor is far below; above the decode floor it is
    /// admitted as before. A `High` turn in the same place is admitted past the
    /// decode floor (`a_turn_outranking_the_running_slots_is_admitted_under_the_floor`).
    #[test]
    fn an_outranking_turn_nobody_waits_on_keeps_the_decodes_hold() {
        let outranking = |prio: DecodePriority, resident: u64| {
            let mut f = Fake::new(Vec::new(), vec![REGION], resident);
            f.active = 14;
            f.running_band = Some(DecodePriority::Low);
            f.prefill_prio = prio;
            f.prefill_floor = FLOOR - 8 * REGION;
            let got = fill_once(&mut f);
            (got.prefills, got.stopped_on_weights)
        };
        let under = FLOOR - 4 * REGION;
        assert_eq!(
            outranking(DecodePriority::Normal, under),
            (0, true),
            "under the hold, a turn nobody waits on waits"
        );
        assert_eq!(
            outranking(DecodePriority::Normal, FLOOR + 4 * REGION),
            (1, false),
            "above the hold it is admitted"
        );
        assert_eq!(
            outranking(DecodePriority::Normal, FLOOR + REGION / 2),
            (0, true),
            "a claim that would carry residency across the hold waits"
        );
        assert_eq!(
            outranking(DecodePriority::Urgent, under),
            (1, false),
            "a person's turn spends the hold down to the prefill floor"
        );
    }

    /// The hold is broken only by a band nobody waits on, only under the
    /// decode floor, and only by a claim that dislodges weights.
    #[test]
    fn the_hold_is_broken_only_by_a_dislodging_claim_nobody_waits_on() {
        let claim = Cost {
            kv: REGION,
            ..Default::default()
        };
        let in_holes = Cost {
            kv: REGION,
            hole_credit: REGION,
            ..Default::default()
        };
        let budget = |streams: bool| Budget {
            decode_floor: FLOOR,
            streams_experts: streams,
            ..Default::default()
        };
        let breaks = |p: DecodePriority, c: &Cost, after: u64| {
            breaks_the_hold(Kind::Prefill, p, true, c, after, &budget(false), false)
        };
        use DecodePriority::{High, Low, Normal, Urgent};
        assert!(breaks(Normal, &claim, FLOOR - 1));
        assert!(breaks(Low, &claim, FLOOR - 1));
        assert!(!breaks(Normal, &claim, FLOOR));
        assert!(!breaks(High, &claim, 0));
        assert!(!breaks(Urgent, &claim, 0));
        assert!(!breaks(Normal, &in_holes, 0));
        // Not outranking, on a card that holds its experts: the trade decides.
        assert!(!breaks_the_hold(
            Kind::Prefill,
            Low,
            false,
            &claim,
            0,
            &budget(false),
            false
        ));
        // On a card that streams them, every prefill and section is held —
        // a person's turn included — and the head of an idle wave is carried.
        for kind in [Kind::Prefill, Kind::Section] {
            for p in [Low, Normal, High, Urgent] {
                assert!(breaks_the_hold(
                    kind,
                    p,
                    false,
                    &claim,
                    FLOOR - 1,
                    &budget(true),
                    false
                ));
                assert!(!breaks_the_hold(
                    kind,
                    p,
                    false,
                    &claim,
                    FLOOR,
                    &budget(true),
                    false
                ));
                assert!(!breaks_the_hold(
                    kind,
                    p,
                    false,
                    &claim,
                    0,
                    &budget(true),
                    true
                ));
                assert!(!breaks_the_hold(
                    kind,
                    p,
                    false,
                    &in_holes,
                    0,
                    &budget(true),
                    false
                ));
            }
        }
    }

    /// **On a card that streams its experts, a turn somebody waits on keeps
    /// the decodes' hold too.** The same `High` turn the card that holds its
    /// experts admits past the decode floor
    /// (`a_turn_outranking_the_running_slots_is_admitted_under_the_floor`) is
    /// refused there, and so is a plain prefill the trade alone would take;
    /// above the floor both are admitted as before.
    #[test]
    fn a_streaming_card_holds_every_prefill_to_the_decode_floor() {
        let offer = |streams: bool, outranking: bool, resident: u64| {
            let mut f = Fake::new(Vec::new(), vec![REGION], resident);
            f.active = 14;
            f.streams = streams;
            f.prefill_prio = DecodePriority::High;
            f.prefill_floor = FLOOR - 8 * REGION;
            if outranking {
                f.running_band = Some(DecodePriority::Low);
            }
            let got = fill_once(&mut f);
            (got.prefills, got.stopped_on_weights)
        };
        let under = FLOOR - 4 * REGION;
        assert_eq!(
            offer(false, true, under),
            (1, false),
            "holds its experts: spends the hold"
        );
        assert_eq!(offer(true, true, under), (0, true), "streams them: held");
        assert_eq!(
            offer(true, false, under),
            (0, true),
            "a plain prefill is held too"
        );
        assert_eq!(
            offer(true, false, FLOOR + 4 * REGION),
            (1, false),
            "above the hold it is admitted"
        );
    }

    /// **The head of an idle wave is carried on a streaming card.** With
    /// nothing running and nothing standing, the first offer runs whatever it
    /// costs, so a turn too large to ever fit cannot block the queue; the one
    /// behind it is held.
    #[test]
    fn a_streaming_cards_idle_wave_still_carries_its_head() {
        let mut f = Fake::new(Vec::new(), vec![REGION, REGION], FLOOR - 4 * REGION);
        f.streams = true;
        f.prefill_prio = DecodePriority::High;
        f.prefill_floor = FLOOR - 8 * REGION;
        let got = fill_once(&mut f);
        assert_eq!(got.prefills, 1, "the head runs, the next is held");
        assert!(got.stopped_on_weights);
    }

    /// **Outranking orders the queue; it does not exempt the trade.** A `High`
    /// turn ahead of running `Low` work is still traded against the decodes it
    /// does not outrank, which its residency slows; with none running there
    /// is nothing to trade, and it is judged on the cap and the prefill floor
    /// as before.
    #[test]
    fn an_outranking_turn_is_traded_against_the_decodes_it_does_not_outrank() {
        let trades = |decodes: usize| {
            let mut f = Fake::new(Vec::new(), vec![REGION], FLOOR + 8 * REGION);
            f.active = 14;
            f.decodes_active = decodes;
            f.running_band = Some(DecodePriority::Low);
            f.prefill_prio = DecodePriority::High;
            f.prefill_floor = FLOOR - 8 * REGION;
            let mut rate = planner();
            fill(&mut f, &mut rate);
            rate.take_trades().judged
        };
        assert_eq!(trades(0), 0, "no decode to slow: nothing traded");
        assert_eq!(trades(4), 1, "beside decodes it does not outrank, traded");
    }

    /// **A turn whose rows do not pay joins the cohort instead of waiting for
    /// the next one.** The saturated wave of
    /// `a_saturated_wave_and_a_starved_one_report_differently`, on a ground
    /// that re-offers as joins: the rate model still stops the forward's rows
    /// where they stop paying, and every turn behind that joins — the floor is
    /// nowhere near — so the whole queue is admitted as one cohort.
    #[test]
    fn a_saturated_offer_comes_back_as_a_join() {
        let mut f = Fake::new(Vec::new(), vec![REGION; 64], FLOOR + 512 * REGION);
        f.active = 1;
        f.decodes_active = 1;
        f.closes_rows = true;
        let mut rate = planner().with_min_gain(0.05);
        let got = fill(&mut f, &mut rate);
        assert!(f.rows_closed, "the rows stopped paying");
        let widened = rate.tokens() / ROWS;
        assert!(
            widened > 0 && widened < 64,
            "the forward took {widened} turns' rows"
        );
        assert_eq!(got.prefills, 64, "and every turn behind them joined");
        assert!(!got.stopped_on_rate, "nothing was turned away on the rate");
        assert!(!got.stopped_on_weights);
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
    pub(in super::super) fn fill_once(f: &mut Fake) -> Filled {
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
}
