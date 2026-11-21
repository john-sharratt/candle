//! Checks that each learned estimate converges, is dampened, and ignores the
//! forwards that cannot teach it.
//!
//! The copy rate converges from its seed, and one outlier moves it by the dampening
//! alone. A large forward faster than the probe corrects the link, while a small one
//! that reads impossibly fast is clamped to it. Hit exponents and layer times are
//! checked the same way, and a fully resident card still learns the per-forward cost,
//! so width remains worth buying.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{
        approx, planner, planner_learned, planner_with_layer, BW_RUN15, GIB, LAYERS, LINK_3090,
        LINK_4090_MOBILE, MIB, RESIDENT_RUN15,
    };

    // ── learning: the copy rate ────────────────────────────────────────────

    /// From an 80% seed of a 25 GB/s link, forwards at run 15's rate bring the
    /// estimate to within 5% in a couple of dozen observations, and each step
    /// moves it by exactly `alpha` of the remaining error.
    #[test]
    fn the_copy_rate_converges_from_the_seed_to_the_observed_rate() {
        let mut p = planner(LINK_4090_MOBILE);
        assert!(
            approx(p.effective_bytes_per_s(), 20e9, 1e-9),
            "seeded at 80% of the link"
        );
        let bytes = p.non_resident_bytes(RESIDENT_RUN15) as f64;
        let secs = bytes / BW_RUN15 + 474.0 * RateModel::SEED_COMPUTE_SECS_PER_TOKEN;
        let before = p.effective_bytes_per_s();
        let observed = p
            .observe_prefill(474, RESIDENT_RUN15, secs)
            .expect("an observation");
        assert!(approx(observed, BW_RUN15, 1e-6));
        let expected = before + WaveRate::DEFAULT_ALPHA * (BW_RUN15 - before);
        assert!(
            approx(p.effective_bytes_per_s(), expected, 1e-9),
            "one dampened step"
        );
        for _ in 0..24 {
            p.observe_prefill(474, RESIDENT_RUN15, secs);
        }
        assert!(approx(p.effective_bytes_per_s(), BW_RUN15, 0.05));
        assert_eq!(p.samples(), 25);
    }

    /// A single outlier moves the estimate by at most `alpha` of its distance:
    /// a stalled forward cannot swing the next wave's plan.
    #[test]
    fn one_outlier_is_dampened() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        let bytes = p.non_resident_bytes(RESIDENT_RUN15) as f64;
        let stalled = 10.0 * bytes / BW_RUN15;
        let before = p.effective_bytes_per_s();
        let observed = p
            .observe_prefill(474, RESIDENT_RUN15, stalled)
            .expect("an observation");
        assert!(observed < before / 5.0);
        let after = p.effective_bytes_per_s();
        assert!(after < before);
        assert!(
            approx(
                before - after,
                WaveRate::DEFAULT_ALPHA * (before - observed),
                1e-9
            ),
            "moved by alpha of the way, no more"
        );
    }

    /// **A big copy that beats the link corrects the link.** The probe reads a
    /// contended figure at startup; a forward that moved 12 GB at 15 GB/s is
    /// better evidence about the same bus than one 256 MiB transfer taken while
    /// the expert cache was staging over it.
    #[test]
    fn a_large_forward_faster_than_the_probe_corrects_the_probe() {
        // A probe that under-read the 4090's link by half.
        let mut p = planner(12.03e9).with_alpha(1.0);
        let bytes = p.non_resident_bytes(RESIDENT_RUN15);
        assert!(
            bytes as usize > WaveRate::PROBE_BYTES,
            "a copy worth believing"
        );
        let rows = p.min_learn_rows();
        let compute = rows as f64 * RateModel::SEED_COMPUTE_SECS_PER_TOKEN;
        let observed = p
            .observe_prefill(rows, RESIDENT_RUN15, bytes as f64 / 15.0e9 + compute)
            .expect("an observation");
        assert!(approx(observed, 15.0e9, 1e-9), "not clamped to 12.03");
        assert!(
            approx(p.link_bytes_per_s(), 15.0e9, 1e-9),
            "the link moved up"
        );
        assert!(approx(p.effective_bytes_per_s(), 15.0e9, 1e-9));
        // And it never moves back down on a slower forward.
        p.observe_prefill(rows, RESIDENT_RUN15, bytes as f64 / 5.0e9 + compute);
        assert!(
            approx(p.link_bytes_per_s(), 15.0e9, 1e-9),
            "the link is a high-water mark"
        );
    }

    /// **One fast forward does not become the new ceiling.** The link can only
    /// ratchet upward — nothing below it moves it back — so a lone outlier that
    /// latched would raise the ceiling permanently and let the copy rate follow
    /// it. It moves by `alpha`, so evidence has to be sustained.
    #[test]
    fn a_single_fast_outlier_barely_moves_the_link() {
        let mut p = planner(12.0e9);
        let bytes = p.non_resident_bytes(RESIDENT_RUN15);
        let rows = p.min_learn_rows();
        let compute = rows as f64 * RateModel::SEED_COMPUTE_SECS_PER_TOKEN;
        // A forward implying 60 GB/s — five times the measured link.
        p.observe_prefill(rows, RESIDENT_RUN15, bytes as f64 / 60.0e9 + compute);
        let after_one = p.link_bytes_per_s();
        let expected = 12.0e9 + WaveRate::DEFAULT_ALPHA * (60.0e9 - 12.0e9);
        assert!(approx(after_one, expected, 1e-9), "{after_one}");
        assert!(after_one < 20.0e9, "one outlier stays near the measurement");
        // Sustained, it does get there.
        for _ in 0..60 {
            p.observe_prefill(rows, RESIDENT_RUN15, bytes as f64 / 60.0e9 + compute);
        }
        assert!(
            approx(p.link_bytes_per_s(), 60.0e9, 0.02),
            "sustained evidence wins"
        );
    }

    /// A *small* copy that reads impossibly fast is still clamped: with few
    /// bytes to divide, the rate is mostly the error in the compute model, and
    /// that is not evidence about the bus.
    #[test]
    fn a_small_observation_faster_than_the_link_is_clamped_to_it() {
        let mut p = planner(LINK_4090_MOBILE).with_alpha(1.0);
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        // 64 MiB short of full residency: a quarter of the probe's transfer.
        let nearly = total - 64 * MIB;
        let bytes = p.non_resident_bytes(nearly);
        assert!((bytes as usize) < WaveRate::PROBE_BYTES);
        // Wide enough to have routed across the layer, and long enough to have
        // paid its own compute — otherwise the copy reads as negative.
        let rows = p.min_learn_rows();
        let observed = p
            .observe_prefill(
                rows,
                nearly,
                bytes as f64 / 200e9 + rows as f64 * RateModel::SEED_COMPUTE_SECS_PER_TOKEN,
            )
            .expect("an observation");
        assert!(approx(observed, LINK_4090_MOBILE, 1e-9));
        assert!(
            approx(p.link_bytes_per_s(), LINK_4090_MOBILE, 1e-9),
            "the link held"
        );
        assert!(approx(p.effective_bytes_per_s(), LINK_4090_MOBILE, 1e-9));
    }

    /// Forwards that cannot teach the copy rate leave the estimate alone: a
    /// fully resident card, a forward shorter than its own compute, a
    /// non-finite or zero duration, and a duration so long the implied rate is
    /// a stall rather than a bus.
    #[test]
    fn prefill_forwards_that_teach_nothing_are_ignored() {
        let mut p = planner(LINK_4090_MOBILE);
        let bw = p.effective_bytes_per_s();
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        // **A forward too narrow to route across the layer.** The 26-row
        // section forwards a run opens with read as 119 GB/s on a 25 GB/s link
        // before this was guarded — the model divides by the whole expert set
        // and a narrow wave touches a fraction of it.
        assert_eq!(p.min_learn_rows(), 126, "8 of 256, to a 2% untouched share");
        assert_eq!(
            p.observe_prefill(26, RESIDENT_RUN15, 0.05),
            None,
            "too narrow to have needed every expert"
        );
        assert_eq!(
            p.observe_prefill(125, RESIDENT_RUN15, 0.9),
            None,
            "one row short"
        );
        assert_eq!(
            p.observe_prefill(474, total, 0.5),
            None,
            "fully resident: nothing copied"
        );
        assert_eq!(
            p.observe_prefill(1_000, RESIDENT_RUN15, 0.3),
            None,
            "shorter than its compute"
        );
        assert_eq!(p.observe_prefill(474, RESIDENT_RUN15, f64::NAN), None);
        assert_eq!(p.observe_prefill(474, RESIDENT_RUN15, 0.0), None);
        assert_eq!(p.observe_prefill(474, RESIDENT_RUN15, -1.0), None);
        assert_eq!(
            p.observe_prefill(474, RESIDENT_RUN15, 3_600.0),
            None,
            "a stall, not a rate"
        );
        assert_eq!(p.effective_bytes_per_s(), bw);
        assert_eq!(p.samples(), 0);
    }

    // ── learning: the layer time ───────────────────────────────────────────

    /// A compute-bound decode wave — its per-layer time above the modelled
    /// copy — teaches that per-layer time as the layer time, dampened by
    /// `alpha`, and converges on it.
    #[test]
    fn the_layer_time_is_learned_from_compute_bound_decode_forwards() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        let seed = p.layer_secs();
        // Eight drafted decodes at run 15's residency: 17.6 ms of copy a layer,
        // under a true 20 ms layer.
        let true_layer = 20e-3;
        assert!(p.decode_copy_secs(192, RESIDENT_RUN15) < true_layer);
        let observed = p
            .observe_decode(8, 2, RESIDENT_RUN15, true_layer * LAYERS)
            .expect("an observation");
        assert!(approx(observed, true_layer, 1e-9));
        let expected = seed + WaveRate::DEFAULT_ALPHA * (true_layer - seed);
        assert!(approx(p.layer_secs(), expected, 1e-9));
        for _ in 0..30 {
            p.observe_decode(8, 2, RESIDENT_RUN15, true_layer * LAYERS);
        }
        assert!(approx(p.layer_secs(), true_layer, 0.01));
        assert_eq!(p.decode_samples(), 31);
    }

    /// Run 12's decode waves: seven drafted decodes stepping in ~760 ms at run
    /// 15's residency. The model puts their copy at 15.4 ms a layer against
    /// 18.5 ms a layer of step — compute-bound, so the layer time is 18.5 ms —
    /// and says eight such decodes would have hidden under it.
    #[test]
    fn the_measured_decode_waves_teach_an_eighteen_millisecond_layer() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_alpha(1.0);
        assert!(approx(
            p.decode_copy_secs(168, RESIDENT_RUN15),
            15.38e-3,
            1e-3
        ));
        let observed = p
            .observe_decode(7, 2, RESIDENT_RUN15, 0.76)
            .expect("an observation");
        assert!(approx(observed, 0.76 / LAYERS, 1e-9));
        assert!(approx(p.layer_secs(), 18.54e-3, 1e-3));
        assert_eq!(p.decode_bus_limit(2, RESIDENT_RUN15), 8);
    }

    /// A bus-bound decode wave — its per-layer time at or under the modelled
    /// copy — says nothing about the layer, and leaves the estimate alone.
    #[test]
    fn a_bus_bound_decode_forward_teaches_nothing_about_the_layer() {
        let mut p = planner_with_layer(LINK_4090_MOBILE, BW_RUN15, 20e-3);
        // Twelve drafted decodes: 26.4 ms of copy a layer; a 400 ms step is
        // 9.8 ms a layer, under it.
        assert!(p.decode_copy_secs(288, RESIDENT_RUN15) > 0.4 / LAYERS);
        assert_eq!(p.observe_decode(12, 2, RESIDENT_RUN15, 0.4), None);
        assert!(approx(p.layer_secs(), 20e-3, 1e-9));
        // Exactly the copy is still bus-bound.
        let copy_step = p.decode_copy_secs(288, RESIDENT_RUN15) * LAYERS;
        assert_eq!(p.observe_decode(12, 2, RESIDENT_RUN15, copy_step), None);
        assert_eq!(p.decode_samples(), 1, "only the fixture's own observation");
    }

    /// Residency is part of the observation: the same seven decodes in the
    /// same 760 ms read as compute-bound at run 15's residency and as bus-bound
    /// with a third of the zone gone, where the modelled copy is longer than
    /// the step.
    #[test]
    fn the_observed_residency_decides_whether_the_forward_teaches() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_alpha(1.0);
        let starved = 2 * GIB;
        assert!(p.decode_copy_secs(168, starved) > 0.76 / LAYERS);
        assert_eq!(p.observe_decode(7, 2, starved, 0.76), None);
        assert!(p.observe_decode(7, 2, RESIDENT_RUN15, 0.76).is_some());
    }

    /// Decode forwards that cannot teach are ignored: no decodes, a non-finite
    /// or zero duration.
    #[test]
    fn decode_forwards_that_teach_nothing_are_ignored() {
        let mut p = planner(LINK_4090_MOBILE);
        let layer = p.layer_secs();
        assert_eq!(p.observe_decode(0, 2, RESIDENT_RUN15, 0.5), None);
        assert_eq!(p.observe_decode(8, 2, RESIDENT_RUN15, f64::INFINITY), None);
        assert_eq!(p.observe_decode(8, 2, RESIDENT_RUN15, 0.0), None);
        assert_eq!(p.observe_decode(8, 2, RESIDENT_RUN15, -0.5), None);
        assert_eq!(p.layer_secs(), layer);
        assert_eq!(p.decode_samples(), 0);
    }

    /// A learned bus changes the decode limit through the same copy figure the
    /// prefill side learned: the layer time is untouched, the copy is not.
    #[test]
    fn a_faster_learned_bus_hides_more_decodes_under_the_same_layer() {
        let slow = planner_with_layer(LINK_4090_MOBILE, 10e9, 16e-3);
        let fast = planner_with_layer(LINK_4090_MOBILE, 20e9, 16e-3);
        assert_eq!(slow.layer_secs(), fast.layer_secs());
        // 3.36 ms a drafted decode at 10 GB/s, 1.68 ms at 20: four hide in
        // 16 ms against nine.
        assert_eq!(slow.decode_bus_limit(2, RESIDENT_RUN15), 4);
        assert_eq!(fast.decode_bus_limit(2, RESIDENT_RUN15), 9);
    }

    // ── constructor guards and the probe ───────────────────────────────────

    #[test]
    fn the_seed_is_eighty_percent_of_the_link() {
        let p = planner(LINK_3090);
        assert_eq!(p.link_bytes_per_s(), LINK_3090);
        assert!(approx(p.link_fraction(), WaveRate::SEED_FRACTION, 1e-12));
        assert!(approx(p.effective_bytes_per_s(), 9.6e9, 1e-9));
    }

    #[test]
    #[should_panic(expected = "positive finite")]
    fn a_zero_link_rate_is_refused() {
        let _ = planner(0.0);
    }

    #[test]
    #[should_panic(expected = "alpha must be in")]
    fn an_alpha_above_one_is_refused() {
        let _ = planner(LINK_4090_MOBILE).with_alpha(1.5);
    }

    #[test]
    #[should_panic(expected = "decode step rate")]
    fn a_zero_decode_seed_rate_is_refused() {
        let _ = WaveRate::with_link_rate(
            LINK_4090_MOBILE,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel {
                initial_tok_per_s: 0.0,
                ..DecodeModel::default()
            },
        );
    }

    /// An exponent below 1 is ordinary — a cache holding a third of the model
    /// hits far more than a third of the time — and the hit reaches certainty
    /// only with the whole model resident.
    #[test]
    fn a_hit_exponent_below_one_beats_the_share_and_never_saturates_early() {
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        let p = WaveRate::with_link_rate(
            LINK_4090_MOBILE,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel {
                hit_exponent: 0.4616,
                ..DecodeModel::default()
            },
        );
        // Run 15's residency: 39.3% resident to the 0.46 is a 0.65 hit.
        assert!(approx(p.hit_fraction(RESIDENT_RUN15), 0.65, 2e-3));
        assert!(approx(p.hit_fraction(total), 1.0, 1e-12));
        assert!(approx(p.streamed_fraction(total), 0.0, 1e-12));
        assert_eq!(p.hit_fraction(0), 0.0);
        for share in [0.5, 0.8, 0.95, 0.99] {
            let r = (total as f64 * share) as u64;
            assert!(
                p.streamed_fraction(r) > 0.0,
                "a card short of the whole model still misses at {share}",
            );
        }
    }

    #[test]
    #[should_panic(expected = "hit exponent")]
    fn a_non_positive_hit_exponent_is_refused() {
        let _ = WaveRate::with_link_rate(
            LINK_4090_MOBILE,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel {
                hit_exponent: 0.0,
                ..DecodeModel::default()
            },
        );
    }

    /// **Learned at one residency, priced at another, it does not read every
    /// expert as found.** WhiteWorld (a94d106e0, 2026-10-20) measured a 0.47
    /// hit; the line through the origin learned there crossed 1 at the
    /// residency a placed cohort leaves, and the step published at every width
    /// was the layer time alone. The power keeps misses on both sides, and
    /// a wider placement still steps slower.
    #[test]
    fn a_hit_learned_low_still_misses_where_it_is_priced_higher() {
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        let mut p = planner(LINK_4090_MOBILE).with_alpha(1.0);
        let learned_at = (total as f64 * 0.30) as u64;
        p.observe_hit_rate(0.47, learned_at).expect("an interval");
        assert!(approx(p.hit_fraction(learned_at), 0.47, 1e-9));
        let priced_at = (total as f64 * 0.70) as u64;
        let streamed = p.streamed_fraction(priced_at);
        assert!(
            streamed > 0.1 && streamed < 0.53,
            "more resident, fewer misses, but not none: {streamed}"
        );
        let one = p.decode_copy_secs(8, priced_at);
        let wide = p.decode_copy_secs(8 * 17, priced_at);
        assert!(
            wide > one * 4.0,
            "seventeen decodes route more: {one} → {wide}"
        );
    }

    /// **A small dislodge near full residency costs what the misses it adds
    /// cost.** BlackWorld measured 0.82 at about 94% resident; the coefficient
    /// read 1.000 there, so 352 MiB dislodged priced every decode 31% slower.
    /// Through the measured point, the streamed share moves by the share of
    /// residency moved, scaled by the exponent.
    #[test]
    fn a_hit_learned_near_full_residency_prices_a_small_dislodge_small() {
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        let mut p = planner(LINK_4090_MOBILE).with_alpha(1.0);
        let at = (total as f64 * 0.94) as u64;
        p.observe_hit_rate(0.82, at).expect("an interval");
        let before = p.streamed_fraction(at);
        // BlackWorld's 352 MiB, as a share of its model: about 0.75%.
        let after = p.streamed_fraction(at - (total as f64 * 0.0075) as u64);
        assert!(approx(before, 0.18, 1e-9), "{before}");
        assert!(
            after > before && (after - before) / before < 0.15,
            "a 352 MiB dislodge moves the misses a little: {before} → {after}"
        );
    }

    /// The learning width is the model's own premise, in rows: the point where
    /// a wave's routing has covered all but 2% of a layer, so that dividing the
    /// forward's copy time by *every* expert's bytes is honest.
    #[test]
    fn the_learning_width_follows_the_routing_geometry() {
        let p = planner(LINK_4090_MOBILE);
        assert_eq!(p.min_learn_rows(), 126, "8 of 256 → 32 · ln 50");
        // A wider top-k covers the layer sooner; a bigger layer takes longer.
        let wide = WaveRate::with_link_rate(
            LINK_4090_MOBILE,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel {
                experts_per_token: 32,
                ..DecodeModel::default()
            },
        );
        assert_eq!(
            wide.min_learn_rows(),
            32,
            "four times the top-k, a quarter the rows"
        );
        // And a forward at that width does teach, where one row short does not.
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15).with_alpha(1.0);
        let rows = p.min_learn_rows();
        let secs = p.non_resident_bytes(RESIDENT_RUN15) as f64 / 15.0e9
            + rows as f64 * RateModel::SEED_COMPUTE_SECS_PER_TOKEN;
        assert_eq!(p.observe_prefill(rows - 1, RESIDENT_RUN15, secs), None);
        assert!(p.observe_prefill(rows, RESIDENT_RUN15, secs).is_some());
    }

    /// **The exponent is learned from the cache's own counters.** Run 15
    /// reported a 0.65 hit rate at 7,611 MiB resident of a 20.29 GB model —
    /// 39.3% — so the exponent is 0.46, and the seed of 1 (the cache hitting
    /// its share) was pricing every decode's copy 1.7 times too dear.
    #[test]
    fn the_hit_exponent_is_learned_from_the_caches_counters() {
        let deployed = || {
            WaveRate::with_link_rate(
                LINK_4090_MOBILE,
                ExpertGeometry::QWEN36_35B_A3B,
                RateModel::default(),
                DecodeModel::default(),
            )
        };
        let mut p = deployed().with_alpha(1.0);
        assert!(approx(p.hit_exponent(), 1.0, 1e-12), "seeded at the share");
        let observed = p
            .observe_hit_rate(0.65, RESIDENT_RUN15)
            .expect("a routing interval");
        assert!(approx(observed, 0.4616, 2e-3), "{observed}");
        assert!(approx(p.hit_exponent(), observed, 1e-12));
        assert_eq!(p.hit_samples(), 1);
        // And the copy it prices is the one the counters measured.
        let streamed = p.streamed_fraction(RESIDENT_RUN15);
        assert!(approx(streamed, 0.35, 1e-2), "{streamed}");
        let seeded = deployed().streamed_fraction(RESIDENT_RUN15);
        assert!(
            approx(seeded / streamed, 1.73, 0.02),
            "the seed's copy was over by 1.7x"
        );
    }

    /// Intervals that cannot say anything about the exponent leave it alone: a
    /// card holding none of the model or all of it, and a hit rate at either
    /// end or not a fraction.
    #[test]
    fn hit_intervals_that_teach_nothing_are_ignored() {
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        let mut p = planner(LINK_4090_MOBILE);
        let seed = p.hit_exponent();
        assert_eq!(p.observe_hit_rate(0.5, 0), None, "nothing resident");
        assert_eq!(p.observe_hit_rate(0.5, total), None, "everything resident");
        assert_eq!(p.observe_hit_rate(1.0, RESIDENT_RUN15), None);
        assert_eq!(p.observe_hit_rate(0.0, RESIDENT_RUN15), None);
        assert_eq!(p.observe_hit_rate(1.5, RESIDENT_RUN15), None);
        assert_eq!(p.observe_hit_rate(-0.1, RESIDENT_RUN15), None);
        assert_eq!(p.observe_hit_rate(f64::NAN, RESIDENT_RUN15), None);
        assert_eq!(p.hit_exponent(), seed);
        assert_eq!(p.hit_samples(), 0);
    }

    /// **Residency still multiplies, which is the point of the refinement.**
    /// Whatever the exponent has learned, dislodging weights raises the
    /// streamed share of every decode already in the wave.
    #[test]
    fn a_learned_exponent_still_answers_to_weight_pressure() {
        let mut p = planner(LINK_4090_MOBILE).with_alpha(1.0);
        p.observe_hit_rate(0.65, RESIDENT_RUN15);
        let before = p.streamed_fraction(RESIDENT_RUN15);
        let after = p.streamed_fraction(RESIDENT_RUN15 - 2 * GIB);
        assert!(
            after > before,
            "less resident, more streamed: {before} → {after}"
        );
        assert!(p.streamed_fraction(0) > p.streamed_fraction(RESIDENT_RUN15));
    }

    /// **A fully resident card still learns that width is worth buying.**
    ///
    /// The regression this exists for: with every expert resident the copy term is
    /// identically zero, so `forward_secs` was `W·c` alone and `rate = W/(W·c) = 1/c` —
    /// flat in width. A flat rate shows zero gain at every width, `judge_gain` refuses the
    /// second offer as `Saturated`, and the wave stays one row wide however much is
    /// queued. Measured on the 30B-A3B at 72 GiB: nine turns queued, one admitted,
    /// `decode seqs avg=1.0 max=1`, 32 t/s against a batched ceiling of 518 — and worse on
    /// a bigger card, since residency is what zeroes the copy.
    ///
    /// Feeding forwards that copy nothing, at varied widths, the fit must recover the
    /// per-forward cost from the intercept and the rate must then rise with width.
    #[test]
    fn a_fully_resident_card_learns_the_per_forward_cost() {
        let all_resident = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        let fixed = 0.040;
        let c = RateModel::SEED_COMPUTE_SECS_PER_TOKEN;
        // The default alpha, deliberately: at `alpha = 1.0` the decay keeps nothing, so
        // each observation replaces the running sums and the fit holds exactly one
        // equation — which can never separate two costs however many forwards it sees.
        let mut p = planner(LINK_4090_MOBILE);
        assert_eq!(
            p.non_resident_bytes(all_resident),
            0,
            "the case under test is a card that copies nothing",
        );
        // Widths spread so the intercept is separable; `T = fixed + W·c` exactly.
        for rows in [200usize, 400, 600, 800, 1000, 1200, 1400, 1600] {
            p.observe_prefill(rows, all_resident, fixed + rows as f64 * c);
        }
        assert!(
            approx(p.fixed_secs(), fixed, 1e-6),
            "the per-forward cost must come out of the intercept, got {}",
            p.fixed_secs(),
        );

        // And the consequence: the rate now rises with width, so a second row shows a
        // real gain instead of zero.
        let narrow = p.rate(100, all_resident);
        let wide = p.rate(1000, all_resident);
        assert!(
            wide > narrow * 1.5,
            "width must buy throughput on a resident card: {narrow} -> {wide}",
        );
    }

    /// The link probe needs a device; on the CPU it says so rather than
    /// inventing a number.
    #[test]
    fn the_probe_refuses_a_cpu_device() {
        let err = WaveRate::measure(
            &Device::Cpu,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel::default(),
            DecodeModel::default(),
        )
        .expect_err("no link to measure on the CPU");
        assert!(err.to_string().contains("CUDA device"));
    }
}
