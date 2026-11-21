//! Tests the block statistics against hand-computed values: amax, jitter, quantile
//! spread, sink threshold and the K threshold formula.
//!
//! The tests check that the tie jitter makes zero blocks distinct and strictly
//! positive without affecting real amax values, that the K threshold collapses to
//! the geometric mean when relevance is flat, and that it clamps to its bounds at
//! the extremes. The quantile test uses known relevance values, and the sink tests
//! check that an all-zero score gives the lower bound and an aligned outlier gives
//! the upper one.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    // These fixtures fill and compare fixed-size K/V/assignment arrays by index
    // on both sides, mirroring the kernel's own indexing so a divergence reads
    // as an index. Iterator rewrites would hide exactly what is being asserted.
    #![allow(clippy::needless_range_loop)]

    use super::*;

    pub(in super::super) fn make_block(values: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        let mut buf = vec![0.0f32; HEAD_DIM * CHUNK_SIZE];
        for d in 0..HEAD_DIM {
            for t in 0..CHUNK_SIZE {
                buf[d * CHUNK_SIZE + t] = values(d, t);
            }
        }
        buf
    }

    pub(in super::super) fn deterministic_uniform(seed: u64, idx: usize) -> f32 {
        let mut x = seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(idx as u64);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58476D1CE4E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D049BB133111EB);
        x ^= x >> 31;
        let mantissa = (x & 0x7FFF_FFFF) as f32 / 2147483647.0;
        mantissa * 2.0 - 1.0
    }

    /// Box-Muller from two uniforms → ~ N(0, 1).
    pub(in super::super) fn deterministic_gauss(seed: u64, idx: usize) -> f32 {
        let u1 = (deterministic_uniform(seed, idx * 2) + 1.0) * 0.5;
        let u2 = (deterministic_uniform(seed, idx * 2 + 1) + 1.0) * 0.5;
        let u1 = u1.clamp(1e-7, 1.0 - 1e-7);
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let th = 2.0 * std::f32::consts::PI * u2;
        r * th.cos()
    }

    #[test]
    fn amax_tie_jitter_breaks_zero_block_ties() {
        // Pure zero K and V — every block's true amax is 0; only the
        // jitter survives. The sort must then produce a non-monotonic
        // order; without jitter it would be the bitonic network's
        // tie-break which biases toward original-index order.
        let k = vec![0.0f32; HEAD_DIM * CHUNK_SIZE];
        let v = vec![0.0f32; HEAD_DIM * CHUNK_SIZE];
        let q = vec![0u16; HEAD_DIM * CHUNK_SIZE];
        let stats = per_block_amax_qrel(&k, &v, &q);
        // Every block must be strictly positive (jitter is in 2^-24..2^-23)
        // and pairwise distinct, so sort_amax_desc gives a unique permutation.
        let mut seen = std::collections::HashSet::new();
        for d in 0..HEAD_DIM {
            assert!(stats.amax_v[d] > 0.0, "jitter zero at d={d}");
            assert!(
                stats.amax_v[d] < 1e-6,
                "jitter too large at d={d}: {}",
                stats.amax_v[d]
            );
            assert!(
                seen.insert(stats.amax_v[d].to_bits()),
                "duplicate jitter at d={d}"
            );
        }
        let order = sort_amax_desc(&stats.amax_v);
        // The sorted indices must not be the identity permutation (which
        // is the failure mode the jitter exists to prevent).
        let identity_count = (0..HEAD_DIM).filter(|&i| order[i] as usize == i).count();
        assert!(
            identity_count < HEAD_DIM / 2,
            "post-jitter sort still near-identity: {identity_count}/{HEAD_DIM} fixed points"
        );
    }

    #[test]
    fn amax_tie_jitter_negligible_vs_real_amax() {
        // Jitter must be small enough that it doesn't perturb real amax
        // values meaningfully — anything ≥ 1e-4 magnitude should win
        // unchanged. Confirms the format-search path is unaffected by
        // the tie-breaker on real data.
        for b in 0..HEAD_DIM {
            assert!(amax_tie_jitter(b) < 1e-6);
            assert!(amax_tie_jitter(b) > 0.0);
        }
    }

    #[test]
    fn per_block_amax_known_values() {
        let k = make_block(|d, t| (d as f32 + 1.0) * t as f32);
        let v = make_block(|d, t| -(d as f32 + 1.0) * t as f32);
        let q = vec![0u16; HEAD_DIM * CHUNK_SIZE];
        let stats = per_block_amax_qrel(&k, &v, &q);
        for d in 0..HEAD_DIM {
            let expected = (d as f32 + 1.0) * 31.0;
            assert!(
                (stats.amax_k[d] - expected).abs() < 1e-4,
                "K amax mismatch at d={d}: got {}, want {}",
                stats.amax_k[d],
                expected
            );
            assert!(
                (stats.amax_v[d] - expected).abs() < 1e-4,
                "V amax mismatch at d={d}: got {}, want {}",
                stats.amax_v[d],
                expected
            );
        }
        let expected_head = 128.0 * 31.0;
        assert!((stats.k_head_amax - expected_head).abs() < 1e-3);
        assert!((stats.v_head_amax - expected_head).abs() < 1e-3);
    }

    #[test]
    fn sort_desc_with_tiebreak() {
        let mut a = [0.0f32; HEAD_DIM];
        for d in 0..HEAD_DIM {
            a[d] = (d / 2) as f32;
        }
        let idx = sort_amax_desc(&a);
        assert_eq!(idx[0], 126);
        assert_eq!(idx[1], 127);
        assert_eq!(idx[2], 124);
        assert_eq!(idx[3], 125);
        assert_eq!(idx[HEAD_DIM - 2], 0);
        assert_eq!(idx[HEAD_DIM - 1], 1);
    }

    #[test]
    fn slot_stats_quantiles() {
        let mut a = [0.0f32; HEAD_DIM];
        for d in 0..HEAD_DIM {
            a[d] = (128 - d) as f32;
        }
        let idx = sort_amax_desc(&a);
        let alive = [true; HEAD_DIM];
        let compact = compact_alive(&idx, &alive);
        let s = slot_stats(&a, &compact);
        assert_eq!(s.slot_amax, 128.0);
        assert_eq!(s.safe_p95, 122.0);
        assert_eq!(s.safe_p80, 103.0);
        assert_eq!(s.safe_p25, 33.0);
        assert!((s.slot_mean - 64.5).abs() < 1e-3);
    }

    #[test]
    fn preferred_range_idx_table() {
        let amax = 10.0;
        let p95 = 8.0;
        let p80 = 5.0;
        let mean = 4.0;
        let p25 = 2.0;
        assert_eq!(preferred_range_outer(0, amax, p95, p80, mean, p25), 1.0);
        assert!((preferred_range_outer(1, amax, p95, p80, mean, p25) - 0.1).abs() < 1e-7);
        assert!((preferred_range_outer(2, amax, p95, p80, mean, p25) - 0.125).abs() < 1e-7);
        assert!((preferred_range_outer(3, amax, p95, p80, mean, p25) - 0.2).abs() < 1e-7);
        assert!((preferred_range_outer(4, amax, p95, p80, mean, p25) - 0.25).abs() < 1e-7);
        assert!((preferred_range_outer(5, amax, p95, p80, mean, p25) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn k_threshold_scaled_matches_kernel_formula() {
        let lo = 0.05f32;
        let hi = 0.01f32;
        let base = (lo * hi).sqrt();

        // q_spread tiny → collapses to safe_spread=1e-8, multiplier ≈ exp(huge)
        // or exp(-huge); clamp pins it. Pick q_med = q_relevance → z=0 →
        // multiplier=1 → scaled=base. Geom-mean of lo/hi.
        let r = k_threshold_scaled(lo, hi, 0.05, 0.05, 1.0e-12);
        assert!(
            (r - base).abs() < 1e-7,
            "tiny-spread, q_med=qrel: got {r}, want {base}"
        );

        // z = 0 → multiplier = 1 → result = base
        let r = k_threshold_scaled(lo, hi, 0.1, 0.1, 0.05);
        assert!((r - base).abs() < 1e-7, "z=0: got {r}, want {base}");

        // q_med large vs qrel small: z = (small − large) / spread = strongly
        // negative → multiplier = exp(+big) → scaled huge → clamp to lo.
        let r = k_threshold_scaled(lo, hi, 0.001, 0.5, 0.05);
        assert!((r - lo).abs() < 1e-7, "z<<0: got {r}, want lo={lo}");

        // q_med small vs qrel large: z >> 0 → multiplier = exp(-big) ≈ 0 →
        // scaled ≈ 0 → clamp to hi.
        let r = k_threshold_scaled(lo, hi, 0.5, 0.001, 0.05);
        assert!((r - hi).abs() < 1e-7, "z>>0: got {r}, want hi={hi}");

        // Hand-computed mid case: lo=0.05, hi=0.01, base=sqrt(5e-4)≈0.022360680.
        // qrel=0.06, q_med=0.05, spread=0.02 → z=0.5 → mult=exp(-0.5)≈0.6065307.
        // scaled = 0.022360680 * 0.6065307 ≈ 0.013561242, in [hi=0.01, lo=0.05].
        let r = k_threshold_scaled(0.05, 0.01, 0.06, 0.05, 0.02);
        let expected_base = (0.05f32 * 0.01).sqrt();
        let expected = expected_base * (-0.5f32).exp();
        assert!(
            (r - expected).abs() < 1e-6,
            "mid case: got {r}, want {expected}"
        );
    }

    #[test]
    fn q_relevance_quantiles_matches_kernel_histogram() {
        // Synthesize 128 blocks with known qrel distribution. We need:
        //  - k_block: non-zero per (d, t) so k2_sum > 0.
        //  - q_block: per-dim Q value chosen so qrel_d takes a known value.
        // qrel = Σ q² k² / Σ k² ⇒ when k is constant across t, qrel = mean(q²).
        // Use k = 1.0 everywhere; q[d, t] = sqrt(d / 127) * sign(t-15.5).
        // Then qrel[d] = mean(q²) = d / 127, evenly spaced 0..1.
        let mut k = vec![0.0f32; HEAD_DIM * CHUNK_SIZE];
        let mut q = vec![0u16; HEAD_DIM * CHUNK_SIZE];
        for d in 0..HEAD_DIM {
            let target = d as f32 / 127.0;
            let qv = target.sqrt();
            for t in 0..CHUNK_SIZE {
                k[d * CHUNK_SIZE + t] = 1.0;
                let sign = if t < 16 { 1.0 } else { -1.0 };
                let q_actual = qv * sign;
                q[d * CHUNK_SIZE + t] = f16::from_f32(q_actual).to_bits();
            }
        }
        let (median, spread) = q_relevance_quantiles(&k, &q);
        // Ground truth median ≈ 0.5; q1 ≈ 0.25; q3 ≈ 0.75; iqr ≈ 0.5;
        // spread ≈ 0.5 * 2 * 1.4427 ≈ 1.4427. Bin resolution = 1/63 ≈ 0.016.
        // f16 quantization adds a few % on top — slack 0.05 captures both.
        assert!((median - 0.5).abs() < 0.05, "median {median} not near 0.5");
        let expected_spread = 0.5f32 * 2.0 * std::f32::consts::LOG2_E;
        assert!(
            (spread - expected_spread).abs() < 0.1,
            "spread {spread} not near {expected_spread}"
        );
    }

    #[test]
    fn sink_threshold_zero_k_collapses_to_lo() {
        let k = vec![0.0f32; HEAD_DIM * CHUNK_SIZE];
        let q_mean = [0.0f32; HEAD_DIM];
        let stats = sink_threshold(&k, &q_mean, 0.05, 0.005);
        // sink_score all zero → mu=0, sigma=0, safe_sigma=1e-8, z=0,
        // tanh(0)=0, w_max=0 → v_thr_eff = lo.
        assert_eq!(stats.sink_weight_max, 0.0);
        let expected_eff = 0.05f32;
        let expected_sq = expected_eff * expected_eff;
        assert!(
            (stats.v_thr_sq - expected_sq).abs() < 1e-9,
            "v_thr_sq {} not near {} (= lo²)",
            stats.v_thr_sq,
            expected_sq
        );
    }

    #[test]
    fn sink_threshold_aligned_k_spikes_threshold() {
        // q_mean is a unit-ish vector along dim 0; K has one token (t=5)
        // with a huge value along dim 0 — that token's sink_score will
        // be a strong outlier vs the other 31 tokens (which dot to 0).
        let mut k = vec![0.0f32; HEAD_DIM * CHUNK_SIZE];
        let spike_t = 5;
        // Dim 0, token `spike_t` — the row-major offset is just the token index.
        k[spike_t] = 100.0;
        let mut q_mean = [0.0f32; HEAD_DIM];
        q_mean[0] = 1.0;
        let lo = 0.05f32;
        let hi = 0.005f32;
        let stats = sink_threshold(&k, &q_mean, lo, hi);
        // 31 tokens at sink_score=0, one at ~100/sqrt(128). mu = score/32,
        // sigma = sqrt(score² * 31/32²); z for the spike = (score - mu)/sigma
        // = (31/32 * score) / (sqrt(31)/32 * score) = sqrt(31) ≈ 5.57.
        // tanh(5.57) ≈ 0.9999... → w_max ≈ 1.0 → v_thr_eff ≈ hi.
        assert!(
            stats.sink_weight_max > 0.99,
            "sink_weight_max {} should be near 1.0 for aligned spike",
            stats.sink_weight_max
        );
        let expected_eff = lo + stats.sink_weight_max * (hi - lo);
        let expected_sq = expected_eff * expected_eff;
        assert!(
            (stats.v_thr_sq - expected_sq).abs() < 1e-9,
            "v_thr_sq {} not the lerp",
            stats.v_thr_sq
        );
        // v_thr_eff should have lerped much closer to hi than to lo.
        assert!(
            expected_eff < (lo + hi) * 0.5,
            "v_thr_eff {expected_eff} did not lerp toward hi"
        );
    }
}
