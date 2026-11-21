//! Checks the decode model's residency, layer time and admission against the decode
//! waves the engine has measured.
//!
//! Each test reads one piece of the decode model: the streamed fraction at a
//! residency, the bus limit past which copies stop hiding under the layer, and how
//! dislodged weights refuse decodes that a fixed-residency wave would take. A decode
//! step grows with the experts its decodes miss at the rate a miss costs, and a burst
//! of decodes is never priced below one more.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{
        approx, decode, fill_with, offer, planner, planner_with_layer, BW_RUN15, CAP, DECODE_CAP,
        DECODE_DISLODGE, FLOOR_16GB, GIB, LAYERS, LINK_3090, LINK_4090_MOBILE, LINK_BLACKWELL, MIB,
        RESIDENT_RUN15, SLOT,
    };

    use super::super::tests_trading::tests::{traded, CHUNK, FULL, LEAST};

    // ── the decode model: what residency does to the copy ──────────────────

    /// The streamed fraction is what the zone and the predictor miss: nothing
    /// with everything resident, everything with nothing resident, and in
    /// between the power curve through the measured hit — run 15's 39% resident
    /// streams 72% of what a decode routes to.
    #[test]
    fn weight_pressure_raises_the_streamed_fraction() {
        let p = planner(LINK_4090_MOBILE);
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        assert!(approx(p.streamed_fraction(total), 0.0, 1e-12));
        assert!(approx(p.streamed_fraction(0), 1.0, 1e-12));
        assert!(approx(p.resident_fraction(RESIDENT_RUN15), 0.3933, 1e-3));
        assert!(approx(p.streamed_fraction(RESIDENT_RUN15), 0.7247, 1e-3));
        assert!(approx(p.streamed_fraction(total / 2), 0.6164, 1e-3));
        assert!(
            p.streamed_fraction(RESIDENT_RUN15 - GIB) > p.streamed_fraction(RESIDENT_RUN15),
            "less resident, more streamed"
        );
        assert!(
            approx(p.resident_fraction(2 * total), 1.0, 1e-12),
            "capped at the whole model"
        );
    }

    /// The seed: one step at 30 tok/s spread over 41 MoE layers is 0.81 ms a
    /// layer. At run 15's residency a plain decode's eight experts stream 72%
    /// of 15.5 MB in 0.56 ms at the 80% seed of the 4090's link — it fits — and
    /// a drafted decode's 24 do not.
    #[test]
    fn the_layer_time_is_seeded_from_thirty_tokens_a_second() {
        let p = planner(LINK_4090_MOBILE);
        assert!(approx(p.layer_secs(), 1.0 / (30.0 * LAYERS), 1e-12));
        assert_eq!(p.experts_per_decode(0), 8);
        assert_eq!(p.experts_per_decode(2), 24);
        let streamed = p.streamed_fraction(RESIDENT_RUN15);
        assert!(approx(
            p.decode_layer_bytes(8, RESIDENT_RUN15),
            8.0 * SLOT * streamed,
            1e-12
        ));
        assert!(approx(
            p.decode_copy_secs(8, RESIDENT_RUN15),
            8.0 * SLOT * streamed / 20e9,
            1e-9
        ));
        assert!(approx(
            p.decode_copy_secs(8, RESIDENT_RUN15),
            0.5604e-3,
            1e-3
        ));
        assert!(
            p.decode_copy_secs(8, RESIDENT_RUN15) < p.layer_secs(),
            "a plain decode hides under the seed"
        );
        assert!(
            p.decode_copy_secs(24, RESIDENT_RUN15) > p.layer_secs(),
            "a drafted decode does not"
        );
    }

    /// A step is the longer of the layer and its copy over every layer, and
    /// the decode rate is the decodes over that: under the layer the rate grows
    /// with each decode, past it the bus holds the rate flat — until the
    /// decodes between them route to every expert of the layer, after which
    /// another decode adds no copy and the rate climbs again.
    #[test]
    fn the_decode_rate_grows_until_the_bus_holds_it_flat() {
        let p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        let r = RESIDENT_RUN15;
        // 2.20 ms of copy a drafted decode a layer: nine hide under 20 ms.
        assert!(approx(p.decode_copy_secs(24, r), 2.1977e-3, 1e-3));
        assert!(approx(p.decode_step_secs(24 * 9, r), LAYERS * 20e-3, 1e-12));
        assert!(approx(
            p.decode_step_secs(24 * 10, r),
            LAYERS * 10.0 * p.decode_copy_secs(24, r),
            1e-12
        ));
        assert!(p.decode_rate(9, 24 * 9, r) > p.decode_rate(8, 24 * 8, r));
        assert_eq!(p.decode_rate(0, 0, r), 0.0);
        assert_eq!(p.decode_bus_limit(2, r), 9);
        // Plain decodes: 27 hide, the 28th and 29th are both on the bus with
        // the layer unsaturated, and their rates are equal.
        assert_eq!(p.decode_bus_limit(0, r), 27);
        assert!(
            approx(
                p.decode_rate(29, 8 * 29, r),
                p.decode_rate(28, 8 * 28, r),
                1e-9
            ),
            "flat on the bus"
        );
        // Eleven drafted decodes route to 264 ≥ 256 experts: the twelfth adds
        // no copy, and the rate climbs with it.
        assert!(approx(
            p.decode_step_secs(24 * 12, r),
            p.decode_step_secs(24 * 11, r),
            1e-12
        ));
        assert!(p.decode_rate(12, 24 * 12, r) > p.decode_rate(11, 24 * 11, r));
    }

    /// Decodes are admitted while the decode rate improves. At a learned 20 ms
    /// layer and run 15's bus, with each decode's store dislodging 176 MiB:
    /// eight hide under the layer, the ninth puts the wave on the bus but still
    /// lifts the rate, and the tenth — its own copy plus the miss rate every
    /// decode's dislodge raised — lowers it, and is refused as worse.
    #[test]
    fn decodes_stop_when_another_would_lower_the_decode_rate() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        let (admitted, refusal) = fill_with(&mut p, decode(2), DECODE_DISLODGE);
        assert_eq!(admitted, 9);
        assert_eq!(p.routed_per_layer(), 216);
        assert_eq!(p.resident_now(), RESIDENT_RUN15 - 9 * DECODE_DISLODGE);
        match refusal {
            Refusal::Worse { before, after } => {
                assert!(approx(
                    before,
                    p.decode_rate(9, 216, p.resident_now()),
                    1e-9
                ));
                assert!(after < before);
                assert!(approx(before, 10.046, 1e-3), "{before}");
                assert!(approx(after, 9.946, 1e-3), "{after}");
            }
            other => panic!("expected worse, got {other:?}"),
        }
        assert!(p.is_full());
        assert!(
            p.resident_now() > FLOOR_16GB,
            "stopped on the rate, not the floor"
        );
    }

    /// At fixed residency — decodes that dislodge nothing — the bus limit ends
    /// the wave as saturation: once the copy sets the step, another decode
    /// stretches it in exact proportion and the rate does not move. Plain
    /// decodes at an 18 ms layer: 24 hide, the 25th puts the wave on the bus
    /// and still lifts the rate by 2%, the 26th lifts it by nothing.
    #[test]
    fn at_fixed_residency_the_bus_limit_saturates_the_decode_rate() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 18e-3);
        assert_eq!(p.decode_bus_limit(0, RESIDENT_RUN15), 24);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        let (admitted, refusal) = fill_with(&mut p, decode(0), 0);
        assert_eq!(admitted, 25);
        assert!(
            matches!(refusal, Refusal::Saturated { gain } if gain.abs() < 1e-9),
            "{refusal:?}"
        );
        assert_eq!(p.resident_now(), RESIDENT_RUN15);
    }

    /// Drafted decodes saturate the layer before the bus stops paying at this
    /// residency and layer time: eleven route to every expert, and from there
    /// each decode adds no copy, so at fixed residency only the cap ends the
    /// wave.
    #[test]
    fn drafted_decodes_that_saturate_the_layer_run_to_the_cap_at_fixed_residency() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, 40);
        let (admitted, refusal) = fill_with(&mut p, decode(2), 0);
        assert_eq!(admitted, 40);
        assert!(
            matches!(refusal, Refusal::DecodeCap { max_decodes: 40 }),
            "{refusal:?}"
        );
        assert_eq!(p.routed_per_layer(), 960);
    }

    /// The refinement itself: the same decodes on the same card, once with
    /// nothing dislodged and once with half a GiB a decode. The dislodge lowers
    /// the hit rate for every decode already in the wave, so the copy outruns
    /// the layer and the rate turns down where the fixed-residency wave — at
    /// 74% resident, its copy hidden under the layer — runs to the cap: sixty-
    /// four admitted against eight.
    #[test]
    fn dislodged_weights_refuse_decodes_a_fixed_residency_wave_would_take() {
        let resident = 14 * GIB;
        let floor = 6 * GIB;
        let fixed = {
            let mut p = planner_with_layer(LINK_3090, 9.6e9, 20e-3);
            p.reset(resident, floor, floor, CAP, DECODE_CAP);
            let (n, refusal) = fill_with(&mut p, decode(2), 0);
            assert!(matches!(refusal, Refusal::DecodeCap { .. }), "{refusal:?}");
            n
        };
        let dislodging = {
            let mut p = planner_with_layer(LINK_3090, 9.6e9, 20e-3);
            p.reset(resident, floor, floor, CAP, DECODE_CAP);
            let (n, refusal) = fill_with(&mut p, decode(2), 512 * MIB);
            assert!(
                matches!(refusal, Refusal::Worse { before, after } if after < before),
                "{refusal:?}"
            );
            assert!(
                p.resident_now() >= floor + 4 * GIB,
                "the floor was nowhere near"
            );
            n
        };
        assert_eq!(fixed, DECODE_CAP);
        assert_eq!(dislodging, 8);
    }

    /// A decode's dislodge is judged against the floor like a prefill's: with
    /// the layer time generous, the floor is what ends a decode wave that keeps
    /// taking stores.
    #[test]
    fn a_decode_that_would_dislodge_under_the_floor_is_refused_on_the_floor() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 40e-3);
        // Room for exactly three stores above the floor.
        p.reset(
            FLOOR_16GB + 3 * DECODE_DISLODGE,
            FLOOR_16GB,
            FLOOR_16GB,
            CAP,
            DECODE_CAP,
        );
        let (admitted, refusal) = fill_with(&mut p, decode(0), DECODE_DISLODGE);
        assert_eq!(admitted, 3);
        assert!(matches!(refusal, Refusal::Floor { .. }), "{refusal:?}");
        assert_eq!(p.resident_now(), FLOOR_16GB);
    }

    /// The admitted projection is the decode rate the wave would run at,
    /// climbing with each decode under the layer.
    #[test]
    fn a_decode_admission_reports_the_projected_decode_rate() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, DECODE_CAP);
        let mut last = 0.0;
        for n in 1..=8 {
            match offer(&mut p, decode(2), DECODE_DISLODGE) {
                Admit::Admitted { projected } => {
                    assert!(projected > last, "decode {n}: {projected}");
                    assert!(approx(
                        projected,
                        p.decode_rate(n, 24 * n, p.resident_now()),
                        1e-9
                    ));
                    assert!(
                        approx(projected, n as f64 / (LAYERS * 20e-3), 1e-9),
                        "under the layer"
                    );
                    last = projected;
                }
                other => panic!("decode {n}: {other:?}"),
            }
        }
    }

    /// Speculation is what a decode pays in: at run 15's residency and a 20 ms
    /// layer, 27 plain decodes hide where 9 drafted ones or 3 wide-drafted ones
    /// do.
    #[test]
    fn plain_decodes_fit_three_times_as_many_as_drafted_ones() {
        let p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        assert_eq!(p.decode_bus_limit(0, RESIDENT_RUN15), 27);
        assert_eq!(p.decode_bus_limit(2, RESIDENT_RUN15), 9);
        assert_eq!(
            p.decode_bus_limit(7, RESIDENT_RUN15),
            3,
            "64 experts a decode"
        );
    }

    /// Once the admitted decodes route to every expert of a layer, the copy is
    /// the whole layer's streamed share whatever else joins, so the limit is
    /// that copy against the layer time — and when it fits, only the cap ends
    /// the wave.
    #[test]
    fn a_saturated_layer_costs_the_same_however_many_decodes_join() {
        // The whole layer at run 15's residency: 256 experts, 72% streamed, at
        // 15.3 GB/s is 23.4 ms; give the layer 40.
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 40e-3);
        assert!(approx(
            p.decode_copy_secs(256, RESIDENT_RUN15),
            23.44e-3,
            1e-3
        ));
        assert_eq!(
            p.decode_bus_limit(7, RESIDENT_RUN15),
            usize::MAX,
            "nothing but the cap bounds it"
        );
        p.reset(RESIDENT_RUN15, FLOOR_16GB, FLOOR_16GB, CAP, 12);
        let (admitted, refusal) = fill_with(&mut p, decode(7), 8 * MIB);
        assert_eq!(admitted, 12);
        assert!(
            matches!(refusal, Refusal::DecodeCap { max_decodes: 12 }),
            "{refusal:?}"
        );
        // 12 × 64 = 768 routed, but a layer has 256: the copy is capped there.
        assert_eq!(p.routed_per_layer(), 768);
        let r = p.resident_now();
        assert!(approx(
            p.decode_layer_bytes(768, r),
            p.decode_layer_bytes(256, r),
            1e-12
        ));
    }

    /// The slower bus hides fewer decodes for the same layer time; the faster
    /// one more. Same formula, different card.
    #[test]
    fn the_bus_sets_how_many_decodes_a_layer_can_hide() {
        // 3090 at its 9.6 GB/s seed with 12 GiB resident: 24 experts, 47%
        // streamed, 2.25 ms a decode — eight fit in 20 ms.
        let p = planner_with_layer(LINK_3090, 9.6e9, 20e-3);
        assert!(approx(p.decode_copy_secs(24, 12 * GIB), 2.253e-3, 1e-3));
        assert_eq!(p.decode_bus_limit(2, 12 * GIB), 8);
        // Blackwell at its 48 GB/s seed with 19 GB resident: 2.5% streamed;
        // eleven drafted decodes route to every expert and the whole layer's
        // share is 0.26 ms, so the cap alone bounds the wave.
        let p = planner_with_layer(LINK_BLACKWELL, 48e9, 20e-3);
        assert!(approx(
            p.decode_copy_secs(256, 19_000 * MIB),
            0.2583e-3,
            1e-3
        ));
        assert_eq!(p.decode_bus_limit(2, 19_000 * MIB), usize::MAX);
    }

    /// A fully resident card holds every expert a decode routes to, so it
    /// streams none and the bus never limits it, where the same layer at run
    /// 15's residency streams 72% and hides nine drafted decodes.
    #[test]
    fn a_fully_resident_card_streams_nothing() {
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        let p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        assert_eq!(p.decode_copy_secs(24, total), 0.0);
        assert_eq!(p.decode_copy_secs(256, total), 0.0);
        assert_eq!(p.decode_bus_limit(2, total), usize::MAX);
        assert_eq!(p.decode_bus_limit(2, RESIDENT_RUN15), 9);
        // Plain decodes: 32 route to every expert, and at full residency 65
        // would hide, so the layer saturates first there too; at run 15's
        // residency 27 hide, short of saturation, and the bus is the limit.
        assert_eq!(p.decode_bus_limit(0, total), usize::MAX);
        assert_eq!(p.decode_bus_limit(0, RESIDENT_RUN15), 27);
    }

    /// **A step grows with the experts its decodes miss, at what a miss costs
    /// here.** Steps whose per-layer time is a fixed 1.4 ms plus 0.12 ms per
    /// missed expert — a cost the bus model alone does not see, the CPU pool's —
    /// taught at widths 1 to 24 decodes, price a step sixteen decodes wider
    /// above one decode wider, by the learned slope; taught at one width, the
    /// step stands as the old model priced it.
    #[test]
    fn a_decode_step_grows_with_the_experts_it_misses() {
        let mut rate = WaveRate::with_link_rate(
            LINK_3090,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel::default(),
        );
        let resident = RESIDENT_RUN15;
        let per = rate.experts_per_decode(0);
        let layers = ExpertGeometry::QWEN36_35B_A3B.moe_layers as f64;
        let (a, b) = (1.4e-3, 0.12e-3);
        for _ in 0..4 {
            for d in 1..=24usize {
                let m = rate.decode_misses(d * per, resident);
                rate.observe_decode(d, 0, resident, layers * (a + b * m));
            }
        }
        let one = rate.decode_step_secs(per, resident);
        let wide = rate.decode_step_secs(17 * per, resident);
        // The narrowest step sits at the edge of the misses the fit saw, and
        // that edge is drawn toward their mean as the fit forgets in wall time
        // — by however many microseconds the steps took to fold in. Priced a
        // hair past it by the link, so it is held to a part in a thousand, not
        // to the fit's exactness.
        let want = |d: usize| layers * (a + b * rate.decode_misses(d * per, resident));
        assert!(
            (one - want(1)).abs() < 1e-3 * want(1),
            "{one} against {}",
            want(1)
        );
        assert!(
            (wide - want(17)).abs() < 1e-6 * want(17),
            "{wide} against {}",
            want(17)
        );
        assert!(
            wide > one * 2.0,
            "sixteen more decodes step slower: {wide} vs {one}"
        );
        // One width only: nothing to separate, and the old model stands.
        let mut flat = WaveRate::with_link_rate(
            LINK_3090,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel::default(),
        );
        for _ in 0..40 {
            flat.observe_decode(4, 0, resident, 0.2);
        }
        assert!(flat.decode_fit.solve().is_none());
    }

    /// **A burst's step grows with the burst.** The step at `PLACEMENT_WIDE`
    /// decodes more re-trades the cohort against all of them and routes all
    /// their experts, so it is never below the one-more step, and more placed
    /// at once never step faster than fewer.
    #[test]
    fn a_burst_s_step_is_never_below_one_more() {
        let (rate, weights_at) = traded();
        let wide = crate::scheduler::phase_ring::PLACEMENT_WIDE;
        let one = rate.placed_decode_step_ms(2, FULL, LEAST, CHUNK, 1.0, &weights_at);
        let at =
            |more| rate.placed_decode_step_ms_with(2, more, FULL, LEAST, CHUNK, 1.0, &weights_at);
        assert!((at(1) - one).abs() < 1e-12, "one more is the one-more step");
        let mut last = one;
        for more in [2, 4, 8, wide] {
            let s = at(more);
            assert!(s >= last - 1e-12, "{more} more stepped {s} ms after {last}");
            last = s;
        }
    }
}
