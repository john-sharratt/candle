//! Checks the joins, promotions and outranking offers that judge a turn at the
//! boundary between prefill and decode.
//!
//! A joining turn that goes on to decode is priced with that decode in the wave, and
//! a burst of joins stops where the next decode no longer pays, because each admitted
//! join is charged before the next is judged. A promotion is judged even after the
//! wave has latched full, and a refused promotion does not latch the wave. An
//! unlearned layer time admits rather than refusing on a seed. The floors that still
//! bind are checked as well.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{
        decode, offer, offer_at, planner, planner_fitted, planner_judging, prefill, promote,
        share_of_experts, CAP, DECODE_CAP, DECODE_DISLODGE, FLOOR_16GB, GIB, LINK_4090_MOBILE, MIB,
        RESIDENT_RUN15,
    };

    /// **A join that goes on to decode is priced with its decode in the wave.**
    /// Nine tenths of the experts resident and three decodes running: the
    /// store a fourth turn claims slows every step a little, and the fourth row
    /// more than repays it — admitted, and counted in. The same claim from a
    /// join that never decodes buys the running decodes nothing, and is
    /// refused as `Worse`: what a decoding join was, before its own row was
    /// priced.
    #[test]
    fn a_decoding_join_is_admitted_where_its_row_repays_the_dislodge() {
        let resident = share_of_experts(0.9);
        let store = 288 * MIB;
        let mut p = planner_fitted(0.25e-3, 0.05e-3, resident, 3);
        assert!(matches!(
            p.try_join_trading(resident, resident - store, 3.0, true),
            Admit::Admitted { .. }
        ));
        assert_eq!(p.decodes(), 4, "the admitted join's decode is charged");
        assert_eq!(p.resident_now(), resident - store);

        let mut p = planner_fitted(0.25e-3, 0.05e-3, resident, 3);
        assert!(matches!(
            p.try_join_trading(resident, resident - store, 3.0, false),
            Admit::Refused(Refusal::Worse { .. })
        ));
        assert_eq!(p.decodes(), 3, "a refused join charges nothing");
    }

    /// **Where the dislodge is a large share of the zone, the row does not
    /// repay it.** A fifth of the experts resident and eight decodes streaming
    /// most of what they route to — a small card's working point — and a turn
    /// whose claim is a twentieth of the model: every step slows by more than
    /// a ninth row earns, and the join is refused.
    #[test]
    fn a_decoding_join_is_refused_where_the_dislodge_outweighs_its_row() {
        let resident = share_of_experts(0.2);
        let claim = share_of_experts(0.05);
        let mut p = planner_fitted(0.25e-3, 0.05e-3, resident, 8);
        assert!(matches!(
            p.try_join_trading(resident, resident - claim, 8.0, true),
            Admit::Refused(Refusal::Worse { .. } | Refusal::Saturated { .. })
        ));
        assert_eq!(p.decodes(), 8);
    }

    /// **A burst of joins is a run of marginal decodes, not of first ones.**
    /// Each admitted join is charged, so the next is priced against the wave
    /// with it decoding: the burst widens the wave while the next row pays and
    /// stops at the rate, well above the floor — rather than every join
    /// comparing one more decode against the same three until the floor.
    #[test]
    fn a_burst_of_decoding_joins_stops_where_the_next_row_stops_paying() {
        let mut resident = share_of_experts(0.9);
        let store = 288 * MIB;
        let mut p = planner_fitted(0.25e-3, 0.05e-3, resident, 3);
        let mut admitted = 0;
        let refusal = loop {
            match p.try_join_trading(resident, resident - store, p.decodes() as f64, true) {
                Admit::Admitted { .. } => {
                    admitted += 1;
                    resident -= store;
                }
                Admit::Refused(r) => break r,
            }
        };
        assert!(admitted >= 2, "a resident card widens the wave: {admitted}");
        assert_eq!(p.decodes(), 3 + admitted);
        assert!(
            matches!(refusal, Refusal::Worse { .. } | Refusal::Saturated { .. }),
            "stopped by the rate, not the floor: {refusal:?}"
        );
        assert!(resident > GIB, "stopped above the floor");
        // What stopped it is the joins before it: the same claim, priced as a
        // fourth decode beside the three that were running, still pays.
        let fresh = planner_fitted(0.25e-3, 0.05e-3, resident, 3);
        assert!(
            fresh
                .decode_would_carry(0, resident, resident - store)
                .is_ok(),
            "priced against the first three alone, the refused join would pass"
        );
    }

    /// **Searched from a hint, the same answer as from nothing.** Every
    /// threshold of a prefix that pays, every hint in and around the range,
    /// and none: the gallop finds what the bisection finds, and a hint on the
    /// answer costs at most two probes.
    #[test]
    fn a_hinted_search_finds_what_bisection_finds() {
        for lo in 0..3usize {
            for hi in lo..12usize {
                // `edge` is the first index that does not pay.
                for edge in lo..=hi + 1 {
                    let expect = (edge > lo).then(|| edge - 1);
                    for hint in std::iter::once(None).chain((0..=hi + 2).map(Some)) {
                        let mut probes = 0;
                        let got = super::largest_paying(lo, hi, hint, |i| {
                            probes += 1;
                            i < edge
                        });
                        assert_eq!(got, expect, "lo {lo} hi {hi} edge {edge} hint {hint:?}");
                        if hint == expect && expect.is_some() {
                            assert!(probes <= 2, "a hint on the answer took {probes} probes");
                        }
                    }
                }
            }
        }
    }

    /// **A turn joining a spent forward adds no rows and is judged by the floor
    /// alone.** The wave's rows are full at the cap; the join keeps them there,
    /// lowers residency by what it dislodges, and is admitted while that stays
    /// above the prefill floor — where `try_admit` would have read the same rows
    /// at lower residency as `Worse`.
    #[test]
    fn a_join_adds_no_rows_and_is_refused_only_under_the_floor() {
        let floor = 4 * GIB;
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(8 * GIB, floor, floor, CAP, DECODE_CAP);
        assert!(matches!(
            offer(&mut p, prefill(CAP), 0),
            Admit::Admitted { .. }
        ));
        let rows = p.tokens();
        assert!(matches!(
            p.try_admit(prefill(0), 8 * GIB, 7 * GIB),
            Admit::Refused(Refusal::Worse { .. })
        ));

        let mut p = planner(LINK_4090_MOBILE);
        p.reset(8 * GIB, floor, floor, CAP, DECODE_CAP);
        let _ = offer(&mut p, prefill(CAP), 0);
        assert!(matches!(p.try_join(5 * GIB), Admit::Admitted { .. }));
        assert_eq!(p.tokens(), rows, "a join adds no rows");
        assert_eq!(p.resident_now(), 5 * GIB);
        assert!(matches!(
            p.try_join(floor - 1),
            Admit::Refused(Refusal::Floor { resident_after, floor: f })
                if resident_after == floor - 1 && f == floor
        ));
        assert!(
            matches!(p.try_join(6 * GIB), Admit::Refused(Refusal::Full)),
            "a refused join latches the wave like any refusal"
        );
    }

    /// **A band the running work yields to is admitted under the decode floor,
    /// and stopped by the row cap and the prefill floor.** The wave already
    /// carries rows, so the head waiver is spent. The decode floor is the
    /// residency the lower band spent; the prefill floor is the line below
    /// which the committed wave no longer fits, and nobody may cross it.
    #[test]
    fn an_outranking_prefill_passes_the_decode_floor_and_stops_at_the_prefill_floor() {
        let (prefill_floor, decode_floor) = (2 * GIB, 4 * GIB);
        let between = 3 * GIB;
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(8 * GIB, prefill_floor, decode_floor, CAP, DECODE_CAP);
        let _ = offer(&mut p, prefill(CAP - 2), 0);

        assert!(matches!(
            p.try_outranking(2, between),
            Admit::Admitted { .. }
        ));
        assert_eq!(p.tokens(), CAP, "its rows join the wave's");
        assert_eq!(p.resident_now(), between);
        assert!(matches!(
            p.try_outranking(1, between),
            Admit::Refused(Refusal::Cap { max_tokens }) if max_tokens == CAP
        ));
        assert!(
            matches!(p.try_outranking(0, between), Admit::Admitted { .. }),
            "the cap leaves the wave open for the same turn as a join"
        );
        assert_eq!(p.tokens(), CAP, "a join adds no rows");

        assert!(matches!(
            p.try_outranking(0, prefill_floor - GIB),
            Admit::Refused(Refusal::Floor { floor, .. }) if floor == prefill_floor
        ));
        assert!(
            matches!(p.try_outranking(0, between), Admit::Refused(Refusal::Full)),
            "a floor refusal latches, as it does for every admission"
        );
    }

    /// **A latched wave must not decide a promotion.** `try_admit` refuses
    /// everything once any offer has been refused, which is right while
    /// composing a wave and wrong at the boundary: a finished prefill would be
    /// turned away because some earlier, unrelated offer did not fit, and the
    /// turn's own merits would never be read.
    #[test]
    fn a_promotion_is_judged_even_after_the_wave_latched_full() {
        let mut p = planner_judging(20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        // The first offer after a reset takes the head bypass, so admit one to
        // clear it, then latch the wave on something that cannot fit.
        assert!(matches!(
            offer(&mut p, decode(2), DECODE_DISLODGE),
            Admit::Admitted { .. }
        ));
        let before = p.resident_now();
        let latched = p.try_admit(decode(2), before, FLOOR_16GB.saturating_sub(1));
        assert!(matches!(latched, Admit::Refused(_)), "{latched:?}");
        assert!(p.is_full(), "the wave is latched");

        // The boundary still gets a real answer.
        assert!(
            matches!(promote(&mut p, 2, DECODE_DISLODGE), Admit::Admitted { .. }),
            "a promotion is judged on its own cost, not the wave's latch"
        );
    }

    /// **A refused promotion turns away one turn, not the queue behind it.**
    /// Latching here would make the first refusal at the boundary evict every
    /// turn that finished prefilling after it, whatever the decode side could
    /// actually carry.
    #[test]
    fn a_refused_promotion_does_not_latch_the_wave() {
        let mut p = planner_judging(20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        let after = p.resident_now();
        let refused = p.judge_promotion(2, after.saturating_add(DECODE_DISLODGE), FLOOR_16GB - 1);
        assert!(
            matches!(refused, Admit::Refused(Refusal::Floor { .. })),
            "{refused:?}"
        );
        assert!(!p.is_full(), "the wave carries on for the turns behind it");
        assert!(
            matches!(promote(&mut p, 2, DECODE_DISLODGE), Admit::Admitted { .. }),
            "the next boundary decision is still made on its merits"
        );
    }

    /// **While the layer time is unlearned the boundary admits, whatever the
    /// numbers say.** The seed is ~26x optimistic, so `T_step` is dominated by
    /// the copy term and every promotion prices as ruinous — run 32 evicted 36
    /// turns that way and run 33 another 10, all `Worse` at three or four
    /// decodes, and all of them wrong once the estimate caught up.
    #[test]
    fn an_unlearned_layer_time_never_refuses_a_promotion() {
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(
            p.decode_samples() < WaveRate::MIN_DECODE_SAMPLES,
            "the estimate has not been learned"
        );
        // Even an offer that would breach the floor is carried while the model
        // has no measured basis to refuse on.
        let after = p.resident_now();
        assert!(matches!(
            p.judge_promotion(2, after.saturating_add(DECODE_DISLODGE), FLOOR_16GB - 1),
            Admit::Admitted { .. }
        ));
    }

    /// **Under the floor already, the floor is not what refuses.** Committed
    /// work holds residency there, and refusing the turn returns no weight; the
    /// marginal question decides instead — here, with no decode in the wave to
    /// compare against, the head rule carries it.
    #[test]
    fn a_floor_already_crossed_leaves_the_decode_to_the_gain() {
        let mut p = planner_judging(20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        let under = FLOOR_16GB - GIB;
        assert!(
            p.decode_would_carry(0, under, under - GIB / 4).is_ok(),
            "the turn did not cross the floor, so the floor does not refuse it"
        );
    }

    /// **The question commits nothing.** Asked and refused, or asked again, it
    /// leaves the wave as it found it; what a join admits on its answer is
    /// charged by the join (`a_burst_of_decoding_joins_stops_where_the_next_row_stops_paying`).
    #[test]
    fn asking_whether_a_decode_fits_changes_no_counter() {
        let mut p = planner_judging(20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        let decodes = p.decodes();
        let resident = p.resident_now();
        for _ in 0..4 {
            let _ = p.decode_would_carry(0, RESIDENT_RUN15, RESIDENT_RUN15);
            let _ = p.decode_would_carry(0, RESIDENT_RUN15, FLOOR_16GB - 1);
        }
        assert_eq!(p.decodes(), decodes, "no decode was counted in");
        assert_eq!(p.resident_now(), resident, "no residency was spent");
        assert!(!p.is_full(), "a question never latches the wave");
    }

    /// **The unlearned window applies here too, and it has to.** Refusing on a
    /// seed measured ~26x optimistic would turn away the prefills whose decode
    /// forwards the estimate is learned from — the engine would never leave the
    /// window it was refusing on.
    #[test]
    fn an_unlearned_layer_time_never_refuses_a_prefill_on_its_decode() {
        let mut p = planner(LINK_4090_MOBILE);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        assert!(p.decode_samples() < WaveRate::MIN_DECODE_SAMPLES);
        assert!(
            p.decode_would_carry(0, RESIDENT_RUN15, FLOOR_16GB - 1)
                .is_ok(),
            "while unlearned the decode question cannot refuse"
        );
    }

    /// And once it *is* learned, the same offer is refused — so the bypass is a
    /// window, not a permanent hole.
    #[test]
    fn a_learned_layer_time_restores_the_refusal() {
        let mut p = planner_judging(20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        let after = p.resident_now();
        assert!(matches!(
            p.judge_promotion(2, after.saturating_add(DECODE_DISLODGE), FLOOR_16GB - 1),
            Admit::Refused(Refusal::Floor { .. })
        ));
    }

    /// The promotion that would take residency under the floor is refused —
    /// the same hard line every other admission answers to.
    #[test]
    fn a_promotion_under_the_floor_is_refused() {
        let mut p = planner_judging(20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        let after = p.resident_now();
        match p.judge_promotion(2, after, FLOOR_16GB - 1) {
            Admit::Refused(Refusal::Floor { floor, .. }) => assert_eq!(floor, FLOOR_16GB),
            other => panic!("{other:?}"),
        }
    }

    /// **No decode floor: between the old floors both kinds land.**
    ///
    /// A decode floor above the prefill's used to refuse a decode on ground a
    /// prefill could take, as a fixed fraction of achievable residency standing
    /// in for the question the decode model answers — whether the wave decodes
    /// faster with it at the residency it leaves. Ground above the prefill
    /// floor, where the cache can still serve, now takes a decode whose rate
    /// climbs, and under the prefill floor a decode is refused on it as a
    /// prefill is.
    #[test]
    fn ground_above_the_prefill_floor_takes_a_prefill_and_a_decode() {
        let prefill_floor = FLOOR_16GB / 2;
        let between = FLOOR_16GB - MIB;

        let mut p = planner_judging(20e-3);
        p.reset(RESIDENT_RUN15, prefill_floor, FLOOR_16GB, CAP, DECODE_CAP);
        // Not the head — the head is carried whatever it costs, which would mask
        // the floor entirely.
        let _ = offer(&mut p, prefill(1), 0);
        match offer_at(&mut p, prefill(1), RESIDENT_RUN15, between) {
            Admit::Admitted { .. } => {}
            other => panic!("a prefill may stand above its floor, got {other:?}"),
        }

        let mut d = planner_judging(20e-3);
        d.reset(RESIDENT_RUN15, prefill_floor, FLOOR_16GB, CAP, DECODE_CAP);
        let _ = offer(&mut d, decode(0), 0);
        match offer_at(&mut d, decode(0), RESIDENT_RUN15, between) {
            Admit::Admitted { .. } => {}
            other => panic!("a decode that pays may stand above the prefill floor, got {other:?}"),
        }
        match offer_at(&mut d, decode(0), between, prefill_floor - 1) {
            Admit::Refused(Refusal::Floor { floor, .. }) => assert_eq!(
                floor, prefill_floor,
                "under it, a decode is refused on the prefill floor",
            ),
            other => panic!("a decode may not go under the prefill floor, got {other:?}"),
        }
    }

    /// The prefill floor is still a floor: under it, a prefill is refused too, and
    /// the refusal names the prefill floor rather than the decode's.
    #[test]
    fn under_the_prefill_floor_a_prefill_is_refused_and_names_it() {
        let prefill_floor = FLOOR_16GB / 2;
        let mut p = planner_judging(20e-3);
        p.reset(RESIDENT_RUN15, prefill_floor, FLOOR_16GB, CAP, DECODE_CAP);
        let _ = offer(&mut p, prefill(1), 0);
        match offer_at(&mut p, prefill(1), RESIDENT_RUN15, prefill_floor - 1) {
            Admit::Refused(Refusal::Floor { floor, .. }) => assert_eq!(floor, prefill_floor),
            other => panic!("{other:?}"),
        }
    }
}
