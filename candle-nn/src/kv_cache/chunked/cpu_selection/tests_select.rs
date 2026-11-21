//! Tests the palette selection against the outcomes the search must produce, from
//! synthetic blocks with known amax, variation and sink structure.
//!
//! A slot claims the top blocks by amax when every block passes its format, and
//! the fallback still fills the quota when no format passes. A block with real
//! variation is never stored as its mean, while a flat block still takes Q0, which
//! is the one format offered flat blocks alone. Flatness is measured against one
//! INT8 step of the head scale, not against the block's own spread. A sink element
//! cannot stretch the head scale beyond the cap, and when a sink is present the
//! capped scale is what lets the slots that have a choice move to a higher-quality
//! format rather than accept the lossier one.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    // These fixtures fill and compare fixed-size K/V/assignment arrays by index
    // on both sides, mirroring the kernel's own indexing so a divergence reads
    // as an index. Iterator rewrites would hide exactly what is being asserted.
    #![allow(clippy::needless_range_loop)]

    use super::*;

    use crate::kv_cache::chunked::sampled_selection::params::{
        production_adaptive_candidates, PRODUCTION_K_QREL_HIGH_THRESHOLDS,
        PRODUCTION_K_QREL_LOW_THRESHOLDS, PRODUCTION_V_QREL_HIGH_THRESHOLDS,
        PRODUCTION_V_QREL_LOW_THRESHOLDS,
    };

    use super::super::tests_stats::tests::{deterministic_gauss, deterministic_uniform};

    use super::super::tests_roundtrip::tests::quant_from_kv;

    #[test]
    fn select_palette4_smoke() {
        let n_chunks = 2;
        let n_kv_head = 2;
        let total = n_chunks * n_kv_head * HEAD_DIM * CHUNK_SIZE;
        let mut k = vec![0.0f32; total];
        let mut v = vec![0.0f32; total];
        for i in 0..total {
            k[i] = deterministic_uniform(0xA17EBA1ABCDEF000, i);
            v[i] = deterministic_uniform(0xB17EBA1A12345600, i);
        }
        let q = vec![0u16; total];
        let k_cands = [QuantFormat::Q4_0, QuantFormat::Q8_0];
        let v_cands = [QuantFormat::Q4_0, QuantFormat::Q8_0];
        let input = SelectionInput {
            k_data: &k,
            v_data: &v,
            q_data: &q,
            k_candidates: &k_cands,
            v_candidates: &v_cands,
            k_threshold_hi: 0.01,
            k_threshold_lo: 0.05,
            v_threshold_hi: 0.005,
            v_threshold_lo: 0.05,
            n_chunks,
            n_kv_head,
            head_dim: HEAD_DIM,
        };
        let out = select_palette4(input);
        assert_eq!(out.heads.len(), n_chunks * n_kv_head);
        for (head_id, h) in out.heads.iter().enumerate() {
            for d in 0..HEAD_DIM {
                assert!(h.k_assignments[d] < N_PALETTE as u8);
                assert!(h.v_assignments[d] < N_PALETTE as u8);
            }
            let mut k_counts = [0usize; N_PALETTE];
            let mut v_counts = [0usize; N_PALETTE];
            for d in 0..HEAD_DIM {
                k_counts[h.k_assignments[d] as usize] += 1;
                v_counts[h.v_assignments[d] as usize] += 1;
            }
            for s in 0..N_PALETTE {
                assert_eq!(k_counts[s], SLOT_QUOTA);
                assert_eq!(v_counts[s], SLOT_QUOTA);
            }
            let base = head_id * HEAD_DIM * CHUNK_SIZE;
            let k_slice = &k[base..base + HEAD_DIM * CHUNK_SIZE];
            let v_slice = &v[base..base + HEAD_DIM * CHUNK_SIZE];
            let expected_k = k_slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let expected_v = v_slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert!((h.k_head_amax - expected_k).abs() < 1e-5);
            assert!((h.v_head_amax - expected_v).abs() < 1e-5);
        }
    }

    #[test]
    fn select_palette4_clustering_resists() {
        let n_chunks = 1;
        let n_kv_head = 1;
        let total = HEAD_DIM * CHUNK_SIZE;
        let mut v = vec![0.0f32; total];
        for d in 0..HEAD_DIM {
            let mut mixed = 0u32;
            let mut x = d as u32;
            for _ in 0..7 {
                mixed = (mixed << 1) | (x & 1);
                x >>= 1;
            }
            let mag = (mixed as f32 + 1.0) / 128.0;
            for t in 0..CHUNK_SIZE {
                let phase = if t % 2 == 0 { 1.0 } else { -1.0 };
                v[d * CHUNK_SIZE + t] =
                    mag * phase * (1.0 + 0.1 * deterministic_uniform(0xC0FFEE, d * 32 + t));
            }
        }
        let mut k = vec![0.0f32; total];
        for i in 0..total {
            k[i] = 0.01 * deterministic_uniform(0xDEAD, i);
        }
        let q = vec![0u16; total];
        let cands = [QuantFormat::Q4_0, QuantFormat::Q8_0];
        let input = SelectionInput {
            k_data: &k,
            v_data: &v,
            q_data: &q,
            k_candidates: &cands,
            v_candidates: &cands,
            k_threshold_hi: 0.01,
            k_threshold_lo: 0.05,
            v_threshold_hi: 0.005,
            v_threshold_lo: 0.05,
            n_chunks,
            n_kv_head,
            head_dim: HEAD_DIM,
        };
        let out = select_palette4(input);
        let v_assign = &out.heads[0].v_assignments;
        let mut transitions = 0usize;
        for d in 1..HEAD_DIM {
            if v_assign[d] != v_assign[d - 1] {
                transitions += 1;
            }
        }
        assert!(
            transitions > 60,
            "V assignments too clustered: {transitions} transitions"
        );
    }

    /// Synth a single-head V dataset where all 128 blocks pass the low-BPE
    /// candidate at outer=1.0. With smooth small-amplitude data (well below
    /// the V threshold scaled by head_amax), Q8_0 trivially passes the quota
    /// for every block in slot 0.
    #[test]
    fn process_side_quota_fulfilled_in_priority_order() {
        let total = HEAD_DIM * CHUNK_SIZE;
        let mut v = vec![0.0f32; total];
        // Smooth small amplitude — Q8_0 reconstructs ~1e-3 error, MSE
        // normalized by head_amax² is tiny.
        for d in 0..HEAD_DIM {
            for t in 0..CHUNK_SIZE {
                v[d * CHUNK_SIZE + t] = 0.5 * ((d as f32 * 0.03 + t as f32 * 0.1).sin())
                    + 0.1 * deterministic_uniform(0x1234, d * 32 + t);
            }
        }
        let stats = per_block_amax_qrel(&v, &v, &vec![0u16; total]);
        let idx = sort_amax_desc(&stats.amax_v);
        let kthresh = [0.05f32; HEAD_DIM];
        // Generous v_thr_sq so all blocks pass.
        let v_thr_sq = 1.0f32;
        let cands = [QuantFormat::Q8_0];
        let (slot_fmt, _slot_scale, assigns) = process_side(
            &v,
            &stats.amax_v,
            &idx,
            &cands,
            stats.v_head_amax,
            false,
            &kthresh,
            v_thr_sq,
        );
        assert_eq!(slot_fmt[0], QuantFormat::Q8_0);
        // Slot 0 should claim the top-32 amax blocks, i.e. idx[0..32].
        let top32: std::collections::HashSet<u16> = idx.iter().take(32).copied().collect();
        for d in 0..HEAD_DIM {
            if top32.contains(&(d as u16)) {
                assert_eq!(
                    assigns[d], 0,
                    "dim {d} (top-32 amax) should be in slot 0 but is {}",
                    assigns[d]
                );
            }
        }
    }

    /// Construct a side where no candidate passes the threshold for even
    /// the slot quota — the fallback path must engage and still fill 32
    /// blocks. We use an extreme-low threshold and small candidate set.
    #[test]
    fn process_side_fallback_when_no_format_fits() {
        let total = HEAD_DIM * CHUNK_SIZE;
        let mut v = vec![0.0f32; total];
        for i in 0..total {
            v[i] = deterministic_gauss(0xCAFE, i) * 2.0;
        }
        let stats = per_block_amax_qrel(&v, &v, &vec![0u16; total]);
        let idx = sort_amax_desc(&stats.amax_v);
        let kthresh = [0.0f32; HEAD_DIM];
        // Effectively-zero V threshold so no block can pass.
        let v_thr_sq = 1.0e-20f32;
        let cands = [QuantFormat::Q4_0, QuantFormat::Q8_0];
        let (slot_fmt, _, assigns) = process_side(
            &v,
            &stats.amax_v,
            &idx,
            &cands,
            stats.v_head_amax,
            false,
            &kthresh,
            v_thr_sq,
        );
        // All four slots should still have valid (one of the candidate)
        // formats from fallback selection.
        for s in 0..N_PALETTE {
            assert!(
                slot_fmt[s] == QuantFormat::Q4_0 || slot_fmt[s] == QuantFormat::Q8_0,
                "slot {s} fmt {:?} not in candidate set",
                slot_fmt[s]
            );
        }
        let mut counts = [0usize; N_PALETTE];
        for d in 0..HEAD_DIM {
            assert!(assigns[d] < N_PALETTE as u8);
            counts[assigns[d] as usize] += 1;
        }
        for s in 0..N_PALETTE {
            assert_eq!(counts[s], SLOT_QUOTA, "slot {s} did not get 32 blocks");
        }
    }

    #[test]
    fn v_threshold_squared_comparison() {
        // Build a V-side scenario where the metric (mse/head_amax²) for a
        // specific block straddles a chosen v_thr_sq. We control orig vs
        // recon directly by handcrafting a tiny test of the inner predicate.
        // mean_squared_error * inv_head_amax_sq <= v_thr_sq is the V check.
        let head_amax = 1.0f32;
        let inv_head_amax_sq = 1.0;
        let mut orig = [0.0f32; CHUNK_SIZE];
        let mut recon_just_above = [0.0f32; CHUNK_SIZE];
        let mut recon_just_below = [0.0f32; CHUNK_SIZE];
        for i in 0..CHUNK_SIZE {
            orig[i] = 0.5;
            recon_just_above[i] = 0.5 + 0.11;
            recon_just_below[i] = 0.5 + 0.09;
        }
        let mse_above = mean_squared_error(&orig, &recon_just_above);
        let mse_below = mean_squared_error(&orig, &recon_just_below);
        // v_thr_eff = 0.1; v_thr_sq = 0.01. mse_below = 0.0081 (< 0.01),
        // mse_above = 0.0121 (> 0.01).
        let v_thr_sq = 0.01f32;
        assert!(
            mse_above * inv_head_amax_sq > v_thr_sq,
            "above case: mse {mse_above} should exceed v_thr_sq {v_thr_sq}"
        );
        assert!(
            mse_below * inv_head_amax_sq <= v_thr_sq,
            "below case: mse {mse_below} should be <= v_thr_sq {v_thr_sq}"
        );
        let _ = head_amax;
    }

    #[test]
    fn select_palette4_against_real_c5_thresholds() {
        let n_chunks = 1;
        let n_kv_head = 1;
        let total = HEAD_DIM * CHUNK_SIZE;
        let mut k = vec![0.0f32; total];
        let mut v = vec![0.0f32; total];
        for i in 0..total {
            k[i] = deterministic_gauss(0xC5A11, i) * 1.5;
            v[i] = deterministic_gauss(0xC5A12, i) * 0.5;
        }
        let mut q = vec![0u16; total];
        for i in 0..total {
            let qv = deterministic_uniform(0xC5A13, i) * 0.1;
            q[i] = f16::from_f32(qv).to_bits();
        }
        let (k_cands_kv, v_cands_kv) = production_adaptive_candidates(5);
        let k_cands: Vec<QuantFormat> = k_cands_kv.iter().map(|f| quant_from_kv(*f)).collect();
        let v_cands: Vec<QuantFormat> = v_cands_kv.iter().map(|f| quant_from_kv(*f)).collect();
        let input = SelectionInput {
            k_data: &k,
            v_data: &v,
            q_data: &q,
            k_candidates: &k_cands,
            v_candidates: &v_cands,
            k_threshold_hi: PRODUCTION_K_QREL_HIGH_THRESHOLDS[5],
            k_threshold_lo: PRODUCTION_K_QREL_LOW_THRESHOLDS[5],
            v_threshold_hi: PRODUCTION_V_QREL_HIGH_THRESHOLDS[5],
            v_threshold_lo: PRODUCTION_V_QREL_LOW_THRESHOLDS[5],
            n_chunks,
            n_kv_head,
            head_dim: HEAD_DIM,
        };
        let out = select_palette4(input);
        assert_eq!(out.heads.len(), 1);
        let h = &out.heads[0];
        let mut k_counts = [0usize; N_PALETTE];
        let mut v_counts = [0usize; N_PALETTE];
        for d in 0..HEAD_DIM {
            assert!(h.k_assignments[d] < N_PALETTE as u8);
            assert!(h.v_assignments[d] < N_PALETTE as u8);
            k_counts[h.k_assignments[d] as usize] += 1;
            v_counts[h.v_assignments[d] as usize] += 1;
        }
        for s in 0..N_PALETTE {
            assert_eq!(k_counts[s], SLOT_QUOTA, "K slot {s} count");
            assert_eq!(v_counts[s], SLOT_QUOTA, "V slot {s} count");
        }
        let expected_k = k.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let expected_v = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!((h.k_head_amax - expected_k).abs() < 1e-5);
        assert!((h.v_head_amax - expected_v).abs() < 1e-5);
        for s in 0..N_PALETTE {
            assert!(
                k_cands.contains(&h.k_pal_format[s]),
                "K slot {s} fmt {:?} not in C5 K candidates",
                h.k_pal_format[s]
            );
            assert!(
                v_cands.contains(&h.v_pal_format[s]),
                "V slot {s} fmt {:?} not in C5 V candidates",
                h.v_pal_format[s]
            );
            assert!(h.k_pal_scale[s].is_finite() && h.k_pal_scale[s] > 0.0);
            assert!(h.v_pal_scale[s].is_finite() && h.v_pal_scale[s] > 0.0);
        }
    }

    /// One chunk, one head, no Q: `fill(d, t, seed)` gives K and V at (dim d,
    /// token t); the seed differs between the two sides.
    pub(in super::super) fn one_head_selection(
        fill: impl Fn(usize, usize, u64) -> f32,
        cands: &[QuantFormat],
        lo: f32,
        hi: f32,
    ) -> PerHeadSelection {
        let total = HEAD_DIM * CHUNK_SIZE;
        let k: Vec<f32> = (0..total)
            .map(|i| fill(i / CHUNK_SIZE, i % CHUNK_SIZE, 0xA1))
            .collect();
        let v: Vec<f32> = (0..total)
            .map(|i| fill(i / CHUNK_SIZE, i % CHUNK_SIZE, 0xB2))
            .collect();
        let q = vec![0u16; total];
        let out = select_palette4(SelectionInput {
            k_data: &k,
            v_data: &v,
            q_data: &q,
            k_candidates: cands,
            v_candidates: cands,
            k_threshold_hi: hi,
            k_threshold_lo: lo,
            v_threshold_hi: hi,
            v_threshold_lo: lo,
            n_chunks: 1,
            n_kv_head: 1,
            head_dim: HEAD_DIM,
        });
        out.heads.into_iter().next().expect("one head")
    }

    /// **A block with real variation is never stored as its mean.** Dim 0 swings
    /// ±100 and sets the head's range; every other dim varies with σ = 2 across
    /// its tokens. At a lenient threshold the error metric alone lets Q0 take
    /// those blocks — the variation is small against the head's amax — and Q0
    /// keeps none of it.
    #[test]
    fn a_block_with_real_variation_is_never_stored_as_its_mean() {
        let sel = one_head_selection(
            |d, t, seed| {
                if d == 0 {
                    if t % 2 == 0 {
                        100.0
                    } else {
                        -100.0
                    }
                } else {
                    2.0 * deterministic_gauss(seed, d * CHUNK_SIZE + t)
                }
            },
            &[QuantFormat::Q0, QuantFormat::Q4_0, QuantFormat::Q8_0],
            0.3,
            0.3,
        );
        for s in 0..N_PALETTE {
            assert_ne!(sel.k_pal_format[s], QuantFormat::Q0, "K slot {s}");
            assert_ne!(sel.v_pal_format[s], QuantFormat::Q0, "V slot {s}");
        }
    }

    /// **Flat blocks still compress to Q0.** Every dim is constant across its
    /// tokens to within 1e-5, far below one INT8 step of the head scale: there is
    /// no variation to retain, so the retention test must not stand in the way.
    #[test]
    fn a_flat_block_still_takes_q0() {
        let sel = one_head_selection(
            |d, t, seed| {
                let level = (d as f32 / 64.0 - 1.0) * 0.9 + 0.05;
                level + 1.0e-5 * deterministic_gauss(seed, d * CHUNK_SIZE + t)
            },
            &[QuantFormat::Q0, QuantFormat::Q4_0],
            0.3,
            0.3,
        );
        for s in 0..N_PALETTE {
            assert_eq!(sel.k_pal_format[s], QuantFormat::Q0, "K slot {s}");
            assert_eq!(sel.v_pal_format[s], QuantFormat::Q0, "V slot {s}");
        }
    }

    /// Only Q0, which keeps nothing but a block's mean, is offered flat blocks
    /// alone. Every other format keeps at least a few levels per block.
    #[test]
    fn only_q0_is_offered_flat_blocks_alone() {
        assert!(flat_only(QuantFormat::Q0));
        for fmt in [
            QuantFormat::Q0_V,
            QuantFormat::Q0_X,
            QuantFormat::Q0_M2,
            QuantFormat::Q0_M4,
            QuantFormat::Q1_S,
            QuantFormat::Q1_A,
            QuantFormat::Q2_S,
            QuantFormat::Q2_A,
            QuantFormat::Q3_0,
            QuantFormat::Q4_0,
            QuantFormat::Q8_0,
        ] {
            assert!(!flat_only(fmt), "{fmt:?}");
        }
    }

    /// Flatness on raw numbers, against one INT8 step of the head scale: at a
    /// head scale of 127 a step is 1.0, so a standard deviation of 0.5 is flat and
    /// one of 2.0 is not — wherever the block's mean sits.
    #[test]
    fn flatness_is_judged_against_one_int8_step_of_the_head_scale() {
        let wobble = |sd: f32, mean: f32| -> [f32; CHUNK_SIZE] {
            std::array::from_fn(|t| mean + if t % 2 == 0 { sd } else { -sd })
        };
        assert!(block_is_flat(&wobble(0.5, 0.0), 127.0));
        assert!(block_is_flat(&wobble(0.5, 80.0), 127.0));
        assert!(!block_is_flat(&wobble(2.0, 0.0), 127.0));
        assert!(!block_is_flat(&wobble(2.0, 80.0), 127.0));
        assert!(block_is_flat(&[7.0; CHUNK_SIZE], 127.0));
        // The same wobble is flat against a coarser head scale.
        assert!(block_is_flat(&wobble(2.0, 0.0), 1270.0));
    }

    /// **A sink token cannot stretch the head scale.** One element at 1000
    /// against a bulk of σ = 1 sets the amax a thousand times above the bulk;
    /// the scale the metrics divide by is capped at 8× the p95 instead.
    #[test]
    fn a_sink_token_cannot_stretch_the_head_scale() {
        let total = HEAD_DIM * CHUNK_SIZE;
        let mut data: Vec<f32> = (0..total).map(|i| deterministic_gauss(0x51, i)).collect();
        data[0] = 1000.0;
        let amax = data.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let p95 = head_p95(&data, amax);
        // The bulk's own p95 is ~2; the percentile answers the upper edge of its
        // bin, which at sqrt spacing over a 1000 range is ~2.3.
        assert!((2.0..3.0).contains(&p95), "p95 {p95}");
        let scale = head_scale(amax, p95);
        assert_eq!(scale, HEAD_SCALE_P95_CAP * p95);
        assert!(scale < 25.0, "head scale {scale} against an amax of {amax}");
    }

    /// A head with no outlier elements keeps its amax exactly: the cap only ever
    /// binds when something has stretched the range.
    #[test]
    fn a_head_without_a_sink_keeps_its_amax() {
        let total = HEAD_DIM * CHUNK_SIZE;
        let data: Vec<f32> = (0..total).map(|i| deterministic_uniform(0x52, i)).collect();
        let amax = data.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let p95 = head_p95(&data, amax);
        assert_eq!(head_scale(amax, p95), amax);
    }

    /// **The capped scale reaches the selection.** The same K and V, with one
    /// sink element at 1000: against the raw amax Q4_0's error is 1e-4 of the
    /// scale and passes a 2e-3 threshold; against the capped scale it is
    /// several times the threshold, so the slots that have a choice move up to
    /// Q8_0. The sink element's own block fits no format at that scale and is
    /// left to the last slot, which then takes the lowest-error fallback — a tie
    /// between the two here — so the last slot is not asserted.
    #[test]
    fn a_sink_token_does_not_loosen_every_threshold_in_its_chunk() {
        let sel = one_head_selection(
            |d, t, seed| {
                if d == 0 && t == 0 {
                    1000.0
                } else {
                    deterministic_gauss(seed, d * CHUNK_SIZE + t)
                }
            },
            &[QuantFormat::Q4_0, QuantFormat::Q8_0],
            0.002,
            0.002,
        );
        for s in 0..N_PALETTE - 1 {
            assert_eq!(sel.k_pal_format[s], QuantFormat::Q8_0, "K slot {s}");
            assert_eq!(sel.v_pal_format[s], QuantFormat::Q8_0, "V slot {s}");
        }
    }
}
