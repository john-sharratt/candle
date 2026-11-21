//! Tests for the gated delta rule: the write-then-read identity, the effect of
//! the decay and the write strength, and that the chunked form equals the
//! sequential one.
//!
//! Each test checks one property of the rule against a value that can be worked
//! out by hand or against the sequential reference. The chunked comparison runs
//! at widths that tile the sequence evenly, leave a tail, exceed it, and reduce to
//! one, all from a nonzero entering state so that the read from earlier chunks is
//! exercised. A separate case uses decays large enough to overflow an unclamped
//! exponent within a single chunk, a range the random fixtures never reached.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use candle::Device;

    pub(in super::super) fn dev() -> Device {
        Device::Cpu
    }

    /// Deterministic pseudo-random fill (no external RNG dependency; tests
    /// must be reproducible byte-for-byte).
    pub(in super::super) fn lcg_tensor(shape: &[usize], seed: u64, dev: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let vals: Vec<f32> = (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Map the top bits to (-0.5, 0.5).
                ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
            })
            .collect();
        Tensor::from_vec(vals, shape, dev).unwrap()
    }

    pub(in super::super) fn assert_close(a: &Tensor, b: &Tensor, tol: f32, what: &str) {
        let d = a
            .sub(b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(d <= tol, "{what}: max abs diff {d} > {tol}");
    }

    #[test]
    fn a_written_value_reads_back_along_its_key() {
        // One head, no decay (g = 0), β = 1, unit key: the delta rule stores v
        // exactly, and reading with q = k returns it exactly.
        let dev = dev();
        let d = 4usize;
        let state = Tensor::zeros((1, d, d), DType::F32, &dev).unwrap();
        let k = Tensor::from_vec(vec![1f32, 0., 0., 0.], (1, d), &dev).unwrap();
        let v = Tensor::from_vec(vec![0.25f32, -1.5, 3., 0.5], (1, d), &dev).unwrap();
        let g = Tensor::zeros((1,), DType::F32, &dev).unwrap();
        let beta = Tensor::ones((1,), DType::F32, &dev).unwrap();
        let (o, s) = delta_step(&state, &k, &k, &v, &g, &beta).unwrap();
        assert_close(&o, &v, 1e-6, "read-back");
        // Writing the SAME (k, v) again is a no-op: the state already
        // predicts v for k, so the correction is zero.
        let (_, s2) = delta_step(&s, &k, &k, &v, &g, &beta).unwrap();
        assert_close(&s, &s2, 1e-6, "idempotent rewrite");
    }

    #[test]
    fn beta_interpolates_and_decay_forgets() {
        let dev = dev();
        let d = 4usize;
        let state = Tensor::zeros((1, d, d), DType::F32, &dev).unwrap();
        let k = Tensor::from_vec(vec![0f32, 1., 0., 0.], (1, d), &dev).unwrap();
        let v = Tensor::from_vec(vec![2f32, 2., 2., 2.], (1, d), &dev).unwrap();
        let g0 = Tensor::zeros((1,), DType::F32, &dev).unwrap();
        let half = Tensor::full(0.5f32, (1,), &dev).unwrap();
        // β = 0.5 writes half the correction.
        let (o, s) = delta_step(&state, &k, &k, &v, &g0, &half).unwrap();
        let expect = v.affine(0.5, 0.).unwrap();
        assert_close(&o, &expect, 1e-6, "half write");
        // A strongly negative gate wipes the state before the next write:
        // reading k after decay-only (β = 0 write of an orthogonal key) sees ~0.
        let g_forget = Tensor::full(-40f32, (1,), &dev).unwrap();
        let beta0 = Tensor::zeros((1,), DType::F32, &dev).unwrap();
        let k_other = Tensor::from_vec(vec![1f32, 0., 0., 0.], (1, d), &dev).unwrap();
        let (_, s_forgot) = delta_step(&s, &k_other, &k_other, &v, &g_forget, &beta0).unwrap();
        let norm = s_forgot
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(norm < 1e-6, "state survived a -40 log-decay: {norm}");
    }

    #[test]
    fn segmented_recurrence_equals_one_shot() {
        // THE sealing/resume property: running [0..a) then [a..T) from the
        // carried state must equal one run over [0..T).
        let dev = dev();
        let (t, h, d) = (7usize, 3usize, 5usize);
        let q = l2_norm(&lcg_tensor(&[t, h, d], 1, &dev), 1e-6).unwrap();
        let k = l2_norm(&lcg_tensor(&[t, h, d], 2, &dev), 1e-6).unwrap();
        let v = lcg_tensor(&[t, h, d], 3, &dev);
        // Log-decays in a realistic range: −softplus-ish magnitudes.
        let g = lcg_tensor(&[t, h], 4, &dev).abs().unwrap().neg().unwrap();
        let beta = candle_nn::ops::sigmoid(&lcg_tensor(&[t, h], 5, &dev)).unwrap();
        let s0 = Tensor::zeros((h, d, d), DType::F32, &dev).unwrap();

        let (o_full, s_full) = delta_recurrence(s0.clone(), &q, &k, &v, &g, &beta).unwrap();

        let a = 3usize;
        let slice = |x: &Tensor, from: usize, len: usize| x.narrow(0, from, len).unwrap();
        let (o1, s_mid) = delta_recurrence(
            s0,
            &slice(&q, 0, a),
            &slice(&k, 0, a),
            &slice(&v, 0, a),
            &slice(&g, 0, a),
            &slice(&beta, 0, a),
        )
        .unwrap();
        let (o2, s_end) = delta_recurrence(
            s_mid,
            &slice(&q, a, t - a),
            &slice(&k, a, t - a),
            &slice(&v, a, t - a),
            &slice(&g, a, t - a),
            &slice(&beta, a, t - a),
        )
        .unwrap();
        let o_seg = Tensor::cat(&[o1, o2], 0).unwrap();
        assert_close(&o_full, &o_seg, 1e-5, "segmented outputs");
        assert_close(&s_full, &s_end, 1e-5, "segmented final state");
    }

    /// The chunked prefill form must match the sequential rule exactly —
    /// outputs and final state — across chunk widths that tile the sequence
    /// evenly, leave a short tail, exceed it, and degenerate to 1 (where the
    /// chunked algebra IS the sequential rule). Nonzero entering state, so the
    /// inter-chunk read path is exercised from the first token.
    #[test]
    fn chunked_matches_sequential_for_all_chunk_widths() {
        let dev = dev();
        let (t, h, d) = (13usize, 3usize, 5usize);
        let q = l2_norm(&lcg_tensor(&[t, h, d], 61, &dev), 1e-6).unwrap();
        let k = l2_norm(&lcg_tensor(&[t, h, d], 62, &dev), 1e-6).unwrap();
        let v = lcg_tensor(&[t, h, d], 63, &dev);
        let g = lcg_tensor(&[t, h], 64, &dev).abs().unwrap().neg().unwrap();
        let beta = candle_nn::ops::sigmoid(&lcg_tensor(&[t, h], 65, &dev)).unwrap();
        // Nonzero entering state: seed it with a couple of sequential steps.
        let s_seed = Tensor::zeros((h, d, d), DType::F32, &dev).unwrap();
        let pre_q = l2_norm(&lcg_tensor(&[2, h, d], 66, &dev), 1e-6).unwrap();
        let pre_k = l2_norm(&lcg_tensor(&[2, h, d], 67, &dev), 1e-6).unwrap();
        let pre_v = lcg_tensor(&[2, h, d], 68, &dev);
        let pre_g = lcg_tensor(&[2, h], 69, &dev).abs().unwrap().neg().unwrap();
        let pre_b = candle_nn::ops::sigmoid(&lcg_tensor(&[2, h], 70, &dev)).unwrap();
        let (_, s0) = delta_recurrence(s_seed, &pre_q, &pre_k, &pre_v, &pre_g, &pre_b).unwrap();

        let (o_ref, s_ref) = delta_recurrence(s0.clone(), &q, &k, &v, &g, &beta).unwrap();

        for chunk in [1usize, 4, 5, 13, 64] {
            let mut s_ch = s0.copy().unwrap();
            let o_ch = delta_chunked(&mut s_ch, &q, &k, &v, &g, &beta, chunk).unwrap();
            assert_close(&o_ref, &o_ch, 3e-5, &format!("chunk={chunk} outputs"));
            assert_close(&s_ref, &s_ch, 3e-5, &format!("chunk={chunk} state"));
        }
    }

    /// Strong decays must not poison the chunk.
    ///
    /// The decay matrix is `exp(G[i] − G[j])` over **all** `(i, j)`, masked to
    /// the lower triangle afterwards. The discarded half's exponent grows
    /// positive with the distance between tokens, so once a chunk accumulates
    /// enough decay it overflows to `+inf`, the mask turns that into `NaN`, and
    /// `unit_lower_inverse` spreads it over the whole chunk.
    ///
    /// It is content-dependent, which is why the width sweep above — random
    /// decays around −0.5 over 13 tokens — never saw it, while real prompt
    /// embeddings produced a non-finite logit row at token 23 of one prompt and
    /// 27 of another. Here the decays are large enough that an unclamped
    /// exponent exceeds the f32 range within one chunk, and the sequential rule
    /// (which computes one step at a time and cannot overflow this way) is the
    /// reference for what the answer should be.
    #[test]
    fn strong_decay_stays_finite_and_matches_sequential() {
        let dev = dev();
        let (t, h, d) = (48usize, 2usize, 4usize);
        let q = l2_norm(&lcg_tensor(&[t, h, d], 81, &dev), 1e-6).unwrap();
        let k = l2_norm(&lcg_tensor(&[t, h, d], 82, &dev), 1e-6).unwrap();
        let v = lcg_tensor(&[t, h, d], 83, &dev);
        let beta = candle_nn::ops::sigmoid(&lcg_tensor(&[t, h], 84, &dev)).unwrap();
        // ~−4 per token: 48 tokens accumulate ≈ −190, so the upper triangle's
        // mirror exponent passes f32's ~88 overflow point well inside one
        // 64-wide chunk. The published models reach the same place by depth
        // rather than by magnitude.
        let g = lcg_tensor(&[t, h], 85, &dev)
            .abs()
            .unwrap()
            .affine(1.0, 3.5)
            .unwrap()
            .neg()
            .unwrap();

        let s0 = Tensor::zeros((h, d, d), DType::F32, &dev).unwrap();
        let (o_ref, s_ref) = delta_recurrence(s0.clone(), &q, &k, &v, &g, &beta).unwrap();

        let finite = |x: &Tensor, what: &str| {
            let m = x
                .abs()
                .unwrap()
                .flatten_all()
                .unwrap()
                .max(0)
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(m.is_finite(), "{what} went non-finite under strong decay");
        };
        for chunk in [8usize, 16, 64] {
            let mut s_ch = s0.copy().unwrap();
            let o_ch = delta_chunked(&mut s_ch, &q, &k, &v, &g, &beta, chunk).unwrap();
            finite(&o_ch, &format!("chunk={chunk} output"));
            finite(&s_ch, &format!("chunk={chunk} state"));
            assert_close(&o_ref, &o_ch, 3e-5, &format!("chunk={chunk} outputs"));
            assert_close(&s_ref, &s_ch, 3e-5, &format!("chunk={chunk} state"));
        }
    }

    #[test]
    fn unit_lower_inverse_inverts() {
        let dev = dev();
        let (h, c) = (2usize, 6usize);
        let strict = lower_tri_mask(c, true, &dev).unwrap();
        let a = lcg_tensor(&[h, c, c], 71, &dev)
            .broadcast_mul(&strict.unsqueeze(0).unwrap())
            .unwrap();
        let t_inv = unit_lower_inverse(&a).unwrap();
        // (I + A) · T = I.
        let eye_vals: Vec<f32> = (0..c * c)
            .map(|i| if i % (c + 1) == 0 { 1f32 } else { 0f32 })
            .collect();
        let eye = Tensor::from_vec(eye_vals, (1, c, c), &dev)
            .unwrap()
            .expand((h, c, c))
            .unwrap();
        let prod = eye.add(&a).unwrap().matmul(&t_inv).unwrap();
        assert_close(&prod, &eye.contiguous().unwrap(), 1e-5, "inverse");
    }
}
