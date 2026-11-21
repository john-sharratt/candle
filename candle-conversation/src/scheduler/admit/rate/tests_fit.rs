//! Checks that the cost fits recover their costs from forwards, and that a kept model
//! prices as it did.
//!
//! The prefill fit must separate width, copy and depth from the forwards it is given,
//! and the decode fit must separate the decode's own cost from the creep riding it,
//! counting the distinct experts a layer fetches rather than the picks. A far miss is
//! priced by the link, not by a slope learned on nearby widths. A kept model restored
//! into a fresh one prices the same steps, and one learned on another geometry is
//! refused.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use super::super::tests_support::tests::{
        approx, planner, planner_learned, BW_RUN15, LAYERS, LINK_4090_MOBILE, MIB, RESIDENT_RUN15,
    };

    /// **Rows share experts, and a layer fetches each once.** One decode's
    /// top-8 are eight distinct experts; twenty-four decodes' 192 picks are
    /// about 136 of a 256-expert layer; and however many rows route, never
    /// more than the layer holds.
    #[test]
    fn the_distinct_experts_a_layer_serves_are_fewer_than_its_picks() {
        let p = planner(LINK_4090_MOBILE);
        assert!(
            approx(p.distinct_routed(8), 8.0, 1e-9),
            "{}",
            p.distinct_routed(8)
        );
        assert!(
            approx(p.distinct_routed(192), 136.5, 1e-2),
            "{}",
            p.distinct_routed(192)
        );
        assert!(p.distinct_routed(100_000) <= 256.0);
        assert!(p.distinct_routed(200) > p.distinct_routed(192));
        assert_eq!(p.distinct_routed(0), 0.0);
    }

    /// **The creep that rode a step is the fit's to separate, not to absorb.**
    /// A layer costs its fixed part, a cost per missed expert and a cost per
    /// creep row; taught from co-batched steps alone — a heavy-read, with no
    /// decode-only step to learn from — the fit recovers all three, and the
    /// decode step it prices carries none of the creep's.
    ///
    /// **And at whatever depth it attends.** The creep's cost at 120–190K
    /// context is its attention, which rows alone do not carry: with the creep
    /// at two depths the pairs column takes the difference, where a rows-only
    /// fit would have folded the deeper creep's attention into `a` and `b`.
    #[test]
    fn a_decode_fit_taught_beside_a_creep_prices_the_decode_alone() {
        let (a, b, c, d) = (0.25e-3, 0.05e-3, 2e-6, 4e-14);
        let mut fit = DecodeFit::default();
        for round in 0..80 {
            let m = 4.0 + (round % 7) as f64 * 6.0;
            let r = 100.0 + (round % 5) as f64 * 900.0;
            // Two depths: a creep at 20K, and one at 180K.
            let depth = if round % 3 == 0 { 180_000.0 } else { 20_000.0 };
            let p = r * depth;
            fit.observe(m, 0.0, r, p, a + b * m + c * r + d * p, 0.05);
        }
        let (fa, fb, fc, fd) = fit.solve_all().expect("the steps varied in every column");
        assert!(approx(fa, a, 1e-6), "a {fa}");
        assert!(approx(fb, b, 1e-6), "b {fb}");
        assert!(approx(fc, c, 1e-5), "c {fc}");
        assert!(approx(fd, d, 1e-5), "d {fd}");
        // At one depth rows and pairs move together: one creep column carries
        // it, and the decode's coefficients are still the decode's.
        let mut one_depth = DecodeFit::default();
        for round in 0..80 {
            let m = 4.0 + (round % 7) as f64 * 6.0;
            let r = 100.0 + (round % 5) as f64 * 900.0;
            let p = r * 60_000.0;
            one_depth.observe(m, 0.0, r, p, a + b * m + c * r + d * p, 0.05);
        }
        let (oa, ob, _, _) = one_depth.solve_all().expect("one creep column separates");
        assert!(approx(oa, a, 1e-5) && approx(ob, b, 1e-5), "a {oa} b {ob}");
        // Decode-only steps alone: the two-term fit, no creep to find.
        let mut alone = DecodeFit::default();
        for round in 0..60 {
            let m = 4.0 + (round % 7) as f64 * 6.0;
            alone.observe(m, 0.0, 0.0, 0.0, a + b * m, 0.05);
        }
        let (aa, ab, ac, ad) = alone.solve_all().expect("the widths varied");
        assert!(approx(aa, a, 1e-6) && approx(ab, b, 1e-6) && ac == 0.0 && ad == 0.0);
    }

    /// **A slope where it was taught, the link's beyond.** Taught on steps a
    /// miss or so apart, a fit's slope says nothing of a step twenty misses
    /// away: there the step moves from the nearest it saw by what a miss costs
    /// the link, so a slope the data barely set cannot price a far width.
    #[test]
    fn a_fit_prices_far_misses_by_the_link_not_its_slope() {
        let mut p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        let (a, b) = (0.25e-3, 2e-3);
        for round in 0..60 {
            let m = 10.0 + (round % 3) as f64 * 0.5;
            p.decode_fit.observe(m, 0.0, 0.0, 0.0, a + b * m, 0.05);
        }
        let (lo, hi) = p.decode_fit.miss_support();
        assert!(lo > 9.0 && hi < 12.0, "taught at 10–11: {lo}..{hi}");
        let per_miss = ExpertGeometry::QWEN36_35B_A3B.slot_bytes as f64 / p.effective_bytes_per_s();
        // Find a width and residency whose misses land far above the band.
        let resident = 0;
        let routed = 256;
        let m = p.decode_misses(routed, resident);
        assert!(m > hi + 50.0, "{m}");
        let priced = p.decode_step_secs(routed, resident) / LAYERS;
        let expect = a + b * hi + per_miss * (m - hi);
        assert!(
            approx(priced, expect, 1e-6),
            "priced {priced} against {expect}"
        );
        assert!(priced < a + b * m, "the slope would have priced it dearer");
    }

    /// **What a run learned, a restart starts from.** A model taught its card
    /// and kept as text prices the same steps once read back into a fresh
    /// one — and one learned on another model's geometry is refused, leaving
    /// the fresh model as it was.
    #[test]
    fn a_learned_model_kept_and_restored_prices_as_it_did() {
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        let mut taught = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        for round in 0..40 {
            for decodes in 1..=6usize {
                let share = 0.3 + 0.1 * ((round + decodes) % 5) as f64;
                let resident = (total as f64 * share) as u64;
                let m = taught.distinct_routed(decodes * 8) * taught.streamed_fraction(resident);
                let q = decodes as f64 * (20_000.0 + 10_000.0 * (round % 4) as f64);
                let secs = (0.25e-3 + 0.05e-3 * m + 2e-9 * q) * LAYERS;
                taught.observe_decode_at(decodes, decodes, resident, secs, q);
            }
        }
        let resident = (total as f64 * 0.6) as u64;
        let kept = taught.export_learned();
        let mut restored = planner(LINK_4090_MOBILE);
        assert!(restored.import_learned(&kept));
        assert!(approx(
            restored.decode_step_secs(24 * 8, resident),
            taught.decode_step_secs(24 * 8, resident),
            1e-9
        ));
        assert!(approx(
            restored.rate(4_096, resident),
            taught.rate(4_096, resident),
            1e-9
        ));
        assert_eq!(restored.decode_samples(), taught.decode_samples());

        // Restored on a start whose probe read the link at half, the copy rate
        // the forwards learned stands, and the link is raised to it.
        let mut slow = planner(LINK_4090_MOBILE / 2.0);
        assert!(slow.import_learned(&kept));
        assert!(approx(
            slow.effective_bytes_per_s(),
            taught.effective_bytes_per_s(),
            1e-9
        ));
        assert!(slow.link_bytes_per_s() >= taught.effective_bytes_per_s());

        let mut other = WaveRate::with_link_rate(
            LINK_4090_MOBILE,
            ExpertGeometry {
                moe_layers: 7,
                ..ExpertGeometry::QWEN36_35B_A3B
            },
            RateModel::default(),
            DecodeModel::default(),
        );
        let before = other.export_learned();
        assert!(!other.import_learned(&kept), "another geometry's learning");
        assert_eq!(other.export_learned(), before, "left as it was");
    }

    /// **The decodes' own attention is the fit's to price, not its fixed
    /// term's.** Each decode row attends its whole context; with steps at
    /// deepening contexts the positions column takes that cost, and the
    /// fixed part stays what it is — where without it `a` climbed 0.60 → 1.27
    /// ms a layer through a heavy-read as its conversations deepened
    /// (BlackWorld, f20da86f0, 2026-10-20).
    #[test]
    fn a_decode_fit_prices_its_rows_attention_apart_from_its_fixed_cost() {
        let (a, b, e) = (0.6e-3, 0.15e-3, 3e-9);
        let mut fit = DecodeFit::default();
        for round in 0..80 {
            let m = 4.0 + (round % 7) as f64 * 6.0;
            let decodes = 8.0 + (round % 4) as f64 * 4.0;
            let context = 20_000.0 + (round % 9) as f64 * 20_000.0;
            let q = decodes * context;
            fit.observe(m, q, 0.0, 0.0, a + b * m + e * q, 0.05);
        }
        let x = fit.answer().expect("the depths varied");
        assert!(approx(x[0], a, 1e-6), "a {}", x[0]);
        assert!(approx(x[1], b, 1e-6), "b {}", x[1]);
        let depth = fit.depth().expect("fitted");
        assert!(approx(depth, e, 1e-5), "e {depth}");
    }

    /// **A fit prices twenty-four decodes at what their distinct experts cost.**
    /// The card's per-layer time is its fixed part plus a cost per distinct
    /// expert it fetched; taught at one to thirty-two decodes over a spread of
    /// residencies, the fit's step at twenty-four — whose 192 picks share down
    /// to about 136 experts — lands on that truth, where counting the picks
    /// would have read 40% more misses at the width a placement prices. (Far
    /// outside the widths it was taught at, the link prices instead:
    /// `a_fit_prices_far_misses_by_the_link_not_its_slope`.)
    #[test]
    fn a_decode_fit_extrapolates_by_distinct_experts_not_picks() {
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        let mut p = planner(LINK_4090_MOBILE);
        let (a, b) = (0.25e-3, 0.05e-3);
        let truth = |p: &WaveRate, decodes: usize, resident: u64| {
            let m = p.distinct_routed(decodes * 8) * p.streamed_fraction(resident);
            (a + b * m) * LAYERS
        };
        for round in 0..40 {
            for decodes in (1..=32usize).step_by(3) {
                let share = 0.3 + 0.1 * ((round + decodes) % 5) as f64;
                let resident = (total as f64 * share) as u64;
                let secs = truth(&p, decodes, resident);
                p.observe_decode(decodes, 0, resident, secs);
            }
        }
        let resident = (total as f64 * 0.6) as u64;
        let priced = p.decode_step_secs(24 * 8, resident);
        let real = truth(&p, 24, resident);
        assert!(approx(priced, real, 0.05), "priced {priced} against {real}");
    }

    // ── geometry ───────────────────────────────────────────────────────────

    #[test]
    fn the_35b_carries_twenty_gigabytes_of_experts() {
        let g = ExpertGeometry::QWEN36_35B_A3B;
        assert_eq!(g.total_bytes(), 41 * 256 * 1_933_312);
        assert!(approx(g.total_bytes() as f64, 20.29e9, 0.01));
    }

    // ── the prefill model reproduces the measured run ──────────────────────

    /// Run 15: 7.5–7.7 GiB resident, 474 tokens, 1,004 ms. With the copy rate
    /// the run implied the model lands on the measured forward.
    #[test]
    fn the_model_reproduces_run_15s_forward() {
        let p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        let secs = p.forward_secs(474, RESIDENT_RUN15);
        assert!(
            approx(secs, 1.004, 0.02),
            "projected {secs:.3} s vs 1.004 s measured"
        );
        assert!(approx(p.rate(474, RESIDENT_RUN15), 472.0, 0.02));
    }

    /// **A second forward from the same run, independently.** The fixture's
    /// 15.3 GB/s was derived from one 474-token forward; run 15's ingest phase
    /// logged many more, and a different one — 567 rows in 1,068 ms at 7,467 MiB
    /// of effective residency — implies 15.0 GB/s through the same arithmetic.
    /// Two forwards of different widths and residencies agreeing to 2% is what
    /// makes the copy rate a property of the machine rather than a fit to one
    /// measurement, and it is the check that would fail first if the model's
    /// shape were wrong.
    #[test]
    fn a_second_measured_forward_implies_the_same_copy_rate() {
        let mut p = planner(LINK_4090_MOBILE).with_alpha(1.0);
        let resident = 7_467 * MIB;
        let observed = p
            .observe_prefill(567, resident, 1.068)
            .expect("an observation");
        assert!(approx(observed, 15.0e9, 0.02), "{observed}");
        assert!(
            approx(observed, BW_RUN15, 0.03),
            "within 3% of the fixture's rate: {observed} vs {BW_RUN15}",
        );
        // And the model then reproduces the forward it learned from.
        assert!(approx(p.forward_secs(567, resident), 1.068, 1e-9));
        assert!(approx(p.rate(567, resident), 531.0, 0.01));
    }

    /// The width dependence run 15 measured at fixed weights: 0.42 ms a token,
    /// and the copy the rest.
    #[test]
    fn the_per_token_slope_matches_the_measured_widths() {
        let p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        let slope =
            (p.forward_secs(1_000, RESIDENT_RUN15) - p.forward_secs(0, RESIDENT_RUN15)) / 1_000.0;
        assert!(approx(slope, RateModel::SEED_COMPUTE_SECS_PER_TOKEN, 1e-9));
        let copy = p.non_resident_bytes(RESIDENT_RUN15) as f64 / BW_RUN15;
        assert!(approx(p.forward_secs(0, RESIDENT_RUN15), copy, 1e-9));
    }

    /// Width is what buys throughput: two exemplars to eight is ~2.4× the rate
    /// for ~70% more forward time, because compute starts to count at 1,900
    /// tokens while the copy stays ~0.9 s.
    #[test]
    fn widening_the_wave_multiplies_the_rate() {
        let p = planner_learned(LINK_4090_MOBILE, BW_RUN15);
        let r = RESIDENT_RUN15;
        let two = p.rate(474, r);
        let eight = p.rate(1_896, r - 1_422 * MIB);
        assert!(
            eight / two > 2.2 && eight / two < 2.6,
            "ratio {}",
            eight / two
        );
        let time_ratio = p.forward_secs(1_896, r - 1_422 * MIB) / p.forward_secs(474, r);
        assert!(
            time_ratio > 1.5 && time_ratio < 1.8,
            "time ratio {time_ratio}"
        );
    }

    /// **Forwards at different depths separate attention from compute.** The
    /// four costs of `T = f + X/bw + W·c + P·p` come back from forwards that
    /// varied in width, residency and depth — heavy-read's 114K-position
    /// prompts priced as deep, not as the shallow rows `c` alone describes.
    #[test]
    fn forwards_at_different_depths_separate_attention_from_compute() {
        let (f, bw, c, p) = (0.05, 12e9, 0.3e-3, 2e-11);
        let mut fit = CostFit::default();
        let mut i = 0u64;
        for w in [256.0f64, 1_024.0, 4_096.0, 8_192.0] {
            for x in [0.0f64, 4e9, 12e9, 20e9] {
                for ctx in [0.0f64, 16e3, 64e3, 120e3] {
                    i += 1;
                    let pairs = w * (ctx + w / 2.0);
                    let t = f + x / bw + w * c + pairs * p;
                    fit.observe(x, w, pairs, t, 0.01);
                }
            }
        }
        assert!(i >= CostFit::MIN_SAMPLES);
        let sol = fit.solve(25e9).expect("all four vary");
        assert!(approx(sol.compute_secs_per_token, c, 1e-6), "{sol:?}");
        assert!(approx(sol.fixed_secs, f, 1e-6), "{sol:?}");
        assert!(approx(sol.bw.unwrap(), bw, 1e-6), "{sol:?}");
        assert!(
            approx(sol.attend_secs_per_pair.unwrap(), p, 1e-6),
            "{sol:?}"
        );
    }

    /// **All at one depth, attention stays in `c`.** Pairs that are a fixed
    /// multiple of the rows say nothing apart from width, so the fit drops the
    /// column and `c` carries the depth it saw — the model as it was.
    #[test]
    fn forwards_at_one_depth_leave_attention_in_compute() {
        let (f, bw, c, p, ctx) = (0.05, 12e9, 0.3e-3, 2e-11, 50e3);
        let mut fit = CostFit::default();
        for w in [256.0f64, 1_024.0, 4_096.0, 8_192.0] {
            for x in [0.0f64, 4e9, 12e9, 20e9] {
                let pairs = w * ctx;
                fit.observe(x, w, pairs, f + x / bw + w * c + pairs * p, 0.01);
            }
        }
        let sol = fit.solve(25e9).expect("width and residency vary");
        assert_eq!(sol.attend_secs_per_pair, None, "{sol:?}");
        assert!(
            approx(sol.compute_secs_per_token, c + p * ctx, 1e-6),
            "{sol:?}"
        );
    }

    /// A caller that names no depth is priced at the depth the model learned
    /// from; one that names a deeper cohort pays for it.
    #[test]
    fn a_deeper_cohort_is_priced_deeper() {
        let mut p = WaveRate::with_link_rate(
            LINK_4090_MOBILE,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel {
                attend_secs_per_pair: 2e-11,
                ..RateModel::default()
            },
            DecodeModel::default(),
        );
        let resident = RESIDENT_RUN15;
        // Rows learned from at about 10K positions.
        for _ in 0..200 {
            p.observe_prefill_attending(2_048, 2_048.0 * 10e3, resident, 1.0);
        }
        let ctx = p.context_per_row();
        assert!(ctx > 9e3 && ctx < 11e3, "learned depth {ctx}");
        let named = p.forward_secs_attending(2_048, 2_048.0 * ctx, resident);
        assert!(approx(p.forward_secs(2_048, resident), named, 1e-9));
        let deep = p.forward_secs_attending(2_048, 2_048.0 * 114e3, resident);
        assert!(deep > named, "a 114K cohort costs more than a 10K one");
    }

    /// **The shares say which cost grew.** Rows that come to attend deeper
    /// put more of a forward in attention and less in the copy, and the three
    /// shares leave the fixed costs the rest.
    #[test]
    fn a_deepening_context_shows_in_the_attention_share() {
        let mut p = WaveRate::with_link_rate(
            LINK_4090_MOBILE,
            ExpertGeometry::QWEN36_35B_A3B,
            RateModel {
                attend_secs_per_pair: 2e-11,
                ..RateModel::default()
            },
            DecodeModel::default(),
        );
        let resident = RESIDENT_RUN15;
        for _ in 0..200 {
            p.observe_prefill_attending(2_048, 2_048.0 * 10e3, resident, 1.0);
        }
        let (copy0, compute0, attend0) = p.forward_shares(2_048);
        assert!(
            copy0 + compute0 + attend0 < 1.0 + 1e-9,
            "{copy0} {compute0} {attend0}"
        );
        for _ in 0..400 {
            p.observe_prefill_attending(2_048, 2_048.0 * 100e3, resident, 1.0);
        }
        let (copy1, _, attend1) = p.forward_shares(2_048);
        assert!(attend1 > attend0, "attention {attend0} → {attend1}");
        assert!(copy1 < copy0, "copy {copy0} → {copy1}");
    }

    /// With residency held, the rate climbs toward the compute ceiling and
    /// never crosses it; a fixed per-forward cost is what it amortises.
    #[test]
    fn the_rate_saturates_at_the_compute_ceiling() {
        let model = RateModel {
            fixed_secs: 0.2,
            ..RateModel::default()
        };
        let p = WaveRate::with_link_rate(
            LINK_4090_MOBILE,
            ExpertGeometry::QWEN36_35B_A3B,
            model,
            DecodeModel::default(),
        );
        let total = ExpertGeometry::QWEN36_35B_A3B.total_bytes();
        let asym = p.asymptotic_rate();
        assert!(approx(
            asym,
            1.0 / RateModel::SEED_COMPUTE_SECS_PER_TOKEN,
            1e-12
        ));
        let mut last = 0.0;
        for w in [128usize, 512, 2_048, 8_192, 16_384] {
            let rate = p.rate(w, total);
            assert!(rate > last, "monotone in width");
            assert!(rate < asym, "under the ceiling at {w}");
            last = rate;
        }
        assert!(
            approx(last, asym, 0.05),
            "16k tokens is within 5% of the ceiling"
        );
    }
}
