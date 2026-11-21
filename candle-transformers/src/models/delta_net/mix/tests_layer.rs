//! Tests for the DeltaNet layer's arithmetic: the GQA broadcast, the causal conv,
//! the norm's epsilon floor, and the closed-form single-token output.
//!
//! The GQA test compares whole head maps, because the tiled and blocked
//! conventions coincide whenever the head counts match, and a test on such a
//! model would pass either one. The read-scale test checks a closed form whose
//! output moves measurably if the scale is dropped, so the fixture is shown to be
//! sensitive to the thing it guards. The segmented test checks that the layer
//! carries its state across a boundary with no drift, which is the property that
//! sealing and resume depend on.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use candle::Device;

    use super::super::tests_recurrence::tests::{assert_close, dev, lcg_tensor};

    /// **V head `j` must read K head `j % h_k`.**
    ///
    /// ggml broadcasts the DeltaNet GQA with `ggml_repeat`, which tiles rather
    /// than blocks. The blocked reading (`j / group`) is the other natural way
    /// to write it, produces the same shapes, and is the identity whenever
    /// `h_v == h_k` — which is true of the 0.8B the reference was first
    /// validated on, so nothing caught it there. On the 9B (16 K heads, 32 V)
    /// it cost the model its ability to follow an instruction while leaving it
    /// fluent, which is the hardest possible symptom to attribute.
    ///
    /// Checked by construction rather than through a forward: give each K head
    /// a distinguishable constant and assert which head each V head received.
    #[test]
    fn delta_net_gqa_broadcast_tiles_rather_than_blocks() {
        let dev = dev();
        let (t, h_k, group, d) = (3usize, 4usize, 2usize, 5usize);
        let h_v = h_k * group;

        // K head i is filled with the value `i`.
        let vals: Vec<f32> = (0..t * h_k * d)
            .map(|idx| ((idx / d) % h_k) as f32)
            .collect();
        let x = Tensor::from_vec(vals, (t, h_k, d), &dev).unwrap();

        // The broadcast under test, written exactly as `delta_net_mix` writes it.
        let repeated = x
            .unsqueeze(1)
            .unwrap()
            .expand((t, group, h_k, d))
            .unwrap()
            .reshape((t, h_v, d))
            .unwrap();

        let got: Vec<f32> = repeated.flatten_all().unwrap().to_vec1().unwrap();
        for tt in 0..t {
            for j in 0..h_v {
                let want = (j % h_k) as f32;
                let at = (tt * h_v + j) * d;
                assert_eq!(
                    got[at],
                    want,
                    "V head {j} read K head {} (t={tt}); ggml tiles, so it must \
                     read {}",
                    got[at] as usize,
                    j % h_k
                );
            }
        }
        // And the fixture must actually distinguish the two conventions —
        // they coincide entirely when `h_v == h_k`, and even on a real
        // broadcast they agree on individual heads, so compare the whole maps.
        let tiled: Vec<usize> = (0..h_v).map(|j| j % h_k).collect();
        let blocked: Vec<usize> = (0..h_v).map(|j| j / group).collect();
        assert_ne!(
            tiled, blocked,
            "fixture does not distinguish tiled from blocked, so it would pass \
             under either"
        );
    }

    #[test]
    fn causal_conv_matches_hand_computation_and_carries_tail() {
        let dev = dev();
        // 1 channel, kernel [1, 2, 3], input [10, 20, 30, 40].
        let x = Tensor::from_vec(vec![10f32, 20., 30., 40.], (1, 4), &dev).unwrap();
        let kern = Tensor::from_vec(vec![1f32, 2., 3.], (1, 3), &dev).unwrap();
        let tail0 = Tensor::zeros((1, 2), DType::F32, &dev).unwrap();
        let (y, tail) = causal_conv1d(&x, &kern, &tail0).unwrap();
        // y[t] = 1·x[t−2] + 2·x[t−1] + 3·x[t]  (zeros left of the start)
        let expect = Tensor::from_vec(vec![30f32, 80., 140., 200.], (1, 4), &dev).unwrap();
        assert_close(&y, &expect, 1e-6, "conv values");
        let expect_tail = Tensor::from_vec(vec![30f32, 40.], (1, 2), &dev).unwrap();
        assert_close(&tail, &expect_tail, 1e-6, "conv tail");

        // Split invocation equals one-shot — same property as the recurrence.
        let (y1, tmid) = causal_conv1d(&x.narrow(1, 0, 2).unwrap(), &kern, &tail0).unwrap();
        let (y2, tend) = causal_conv1d(&x.narrow(1, 2, 2).unwrap(), &kern, &tmid).unwrap();
        let y_seg = Tensor::cat(&[y1, y2], 1).unwrap();
        assert_close(&y, &y_seg, 1e-6, "segmented conv outputs");
        assert_close(&tail, &tend, 1e-6, "segmented conv tail");
    }

    #[test]
    fn l2_norm_produces_unit_rows_and_respects_eps_floor() {
        let dev = dev();
        let x = lcg_tensor(&[4, 8], 7, &dev);
        let n = l2_norm(&x, 1e-6).unwrap();
        let sums = n.sqr().unwrap().sum(candle::D::Minus1).unwrap();
        let ones = Tensor::ones((4,), DType::F32, &dev).unwrap();
        assert_close(&sums, &ones, 1e-5, "unit rows");
        // A zero row divides by sqrt(eps), not by zero.
        let z = Tensor::zeros((1, 8), DType::F32, &dev).unwrap();
        let nz = l2_norm(&z, 1e-6).unwrap();
        let m = nz.abs().unwrap().flatten_all().unwrap().max(0).unwrap();
        assert_eq!(m.to_scalar::<f32>().unwrap(), 0.0);
    }

    pub(in super::super) fn tiny_dims() -> DeltaNetDims {
        DeltaNetDims {
            head_dim: 4,
            n_k_heads: 2,
            n_v_heads: 4,
            conv_kernel: 3,
        }
    }

    pub(in super::super) fn tiny_weights(
        dims: &DeltaNetDims,
        hidden: usize,
        dev: &Device,
    ) -> DeltaNetWeights {
        let scale = |t: Tensor| t.affine(0.2, 0.).unwrap();
        DeltaNetWeights {
            wqkv: scale(lcg_tensor(&[dims.conv_dim(), hidden], 11, dev)),
            wz: scale(lcg_tensor(&[dims.value_dim(), hidden], 12, dev)),
            w_beta: scale(lcg_tensor(&[dims.n_v_heads, hidden], 13, dev)),
            w_alpha: scale(lcg_tensor(&[dims.n_v_heads, hidden], 14, dev)),
            dt_bias: scale(lcg_tensor(&[dims.n_v_heads], 15, dev)),
            // a = −exp(A_log): strictly negative.
            a: lcg_tensor(&[dims.n_v_heads], 16, dev)
                .abs()
                .unwrap()
                .affine(-1.0, -0.05)
                .unwrap(),
            conv: scale(lcg_tensor(&[dims.conv_dim(), dims.conv_kernel], 17, dev)),
            norm: lcg_tensor(&[dims.head_dim], 18, dev)
                .affine(0.3, 1.0)
                .unwrap(),
            w_out: scale(lcg_tensor(&[hidden, dims.value_dim()], 19, dev)),
        }
    }

    #[test]
    fn layer_forward_segments_equal_one_shot() {
        // The whole layer — projections, conv tail, recurrence state, gating —
        // must carry across a segment boundary with zero drift. This is the
        // sealing/resume contract at layer granularity.
        let dev = dev();
        let dims = tiny_dims();
        let hidden = 6usize;
        let w = tiny_weights(&dims, hidden, &dev);
        let t = 9usize;
        let x = lcg_tensor(&[t, hidden], 21, &dev);

        let mut s_full = DeltaNetState::zeros(&dims, &dev).unwrap();
        let y_full =
            delta_net_layer_forward(&x, &w, &dims, &mut s_full, 1e-6, ZGate::Silu).unwrap();

        // The same tokens in two calls, through one state buffer — which is
        // the property sealing and resume rest on, and now also the property
        // that the buffer is advanced rather than replaced.
        let a = 4usize;
        let mut s_seg = DeltaNetState::zeros(&dims, &dev).unwrap();
        let y1 = delta_net_layer_forward(
            &x.narrow(0, 0, a).unwrap(),
            &w,
            &dims,
            &mut s_seg,
            1e-6,
            ZGate::Silu,
        )
        .unwrap();
        let y2 = delta_net_layer_forward(
            &x.narrow(0, a, t - a).unwrap(),
            &w,
            &dims,
            &mut s_seg,
            1e-6,
            ZGate::Silu,
        )
        .unwrap();
        let y_seg = Tensor::cat(&[y1, y2], 0).unwrap();
        assert_close(&y_full, &y_seg, 1e-5, "layer segmented outputs");
        assert_close(&s_full.s, &s_seg.s, 1e-5, "layer segmented state");
        assert_close(
            &s_full.conv_tail,
            &s_seg.conv_tail,
            1e-6,
            "layer segmented conv tail",
        );
        // Sanity: the output is finite and non-degenerate.
        let m = y_full
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(m.is_finite() && m > 0.0);
    }

    /// The `1/√d_k` read scale is load-bearing and must not be "optimised
    /// away" as cancelled by the gated RMSNorm downstream.
    ///
    /// It looks removable: `o = S q` is linear in `q` and RMSNorm is
    /// scale-invariant. It is not, because that norm has an epsilon floor —
    /// `rms(o/√d) = o/√(mean(o²) + d·ε)` — so dropping the scale silently
    /// shrinks epsilon by `d`. On the real 0.8B checkpoint that moved the
    /// argmax and turned fluent output into word salad while every
    /// structural test still passed.
    ///
    /// From a zero entering state a single token collapses the recurrence to
    /// `o = β·(k·q̂)·v` with `q̂ = q/√d`, so the whole layer has a closed form
    /// that this test spells out independently. Dropping the read scale
    /// changes `o` by `√d`, which survives the gated norm through its
    /// epsilon term and fails the comparison.
    #[test]
    fn single_token_layer_matches_the_closed_form_with_the_read_scale() {
        let dev = dev();
        let dims = tiny_dims();
        let hidden = 6usize;
        let (d, h_k, h_v) = (dims.head_dim, dims.n_k_heads, dims.n_v_heads);
        let (key_dim, val_dim) = (dims.key_dim(), dims.value_dim());
        let w = tiny_weights(&dims, hidden, &dev);
        let x = lcg_tensor(&[1, hidden], 77, &dev);
        let eps = 1e-6f64;

        let mut s = DeltaNetState::zeros(&dims, &dev).unwrap();
        let got = delta_net_layer_forward(&x, &w, &dims, &mut s, eps, ZGate::Silu).unwrap();

        // The conv tail is zero at sequence start, so only the newest kernel
        // tap contributes and the conv reduces to a per-channel scale.
        let qkv = x.matmul(&w.wqkv.t().unwrap()).unwrap();
        let tap = w
            .conv
            .narrow(1, dims.conv_kernel - 1, 1)
            .unwrap()
            .t()
            .unwrap();
        let conv = silu(&qkv.broadcast_mul(&tap).unwrap()).unwrap();

        let head = |t: &Tensor, off: usize, n: usize, heads: usize| {
            t.narrow(1, off, n).unwrap().reshape((1, heads, d)).unwrap()
        };
        let q = l2_norm(&head(&conv, 0, key_dim, h_k), eps).unwrap();
        let k = l2_norm(&head(&conv, key_dim, key_dim, h_k), eps).unwrap();
        let v = head(&conv, 2 * key_dim, val_dim, h_v);

        // The GQA broadcast, tiled as ggml does it — see
        // `delta_net_gqa_broadcast_tiles_rather_than_blocks`. Restated here to
        // keep the closed form independent of `delta_net_mix`'s *arithmetic*,
        // but it has to agree with it on this convention or the two are
        // describing different models.
        let group = h_v / h_k;
        let rep = |t: &Tensor| {
            t.unsqueeze(1)
                .unwrap()
                .expand((1, group, h_k, d))
                .unwrap()
                .reshape((1, h_v, d))
                .unwrap()
        };
        let (q, k) = (rep(&q), rep(&k));

        // o = β · (k · q/√d) · v — the g gate cannot appear, since it only
        // decays a state that is still zero.
        let q_hat = (&q * (1.0 / (d as f64).sqrt())).unwrap();
        let kq = k.mul(&q_hat).unwrap().sum(candle::D::Minus1).unwrap();
        let beta = candle_nn::ops::sigmoid(&x.matmul(&w.w_beta.t().unwrap()).unwrap()).unwrap();
        let o = v
            .broadcast_mul(&beta.mul(&kq).unwrap().unsqueeze(2).unwrap())
            .unwrap();

        let z = x
            .matmul(&w.wz.t().unwrap())
            .unwrap()
            .reshape((1, h_v, d))
            .unwrap();
        let want = rms_norm_per_head(&o, &w.norm, eps)
            .unwrap()
            .mul(&silu(&z).unwrap())
            .unwrap()
            .reshape((1, val_dim))
            .unwrap()
            .matmul(&w.w_out.t().unwrap())
            .unwrap();

        assert_close(&got, &want, 1e-6, "single-token layer closed form");

        // Guard the guard: the fixture must actually be sensitive to the
        // scale, or the assertion above proves nothing.
        let unscaled = v
            .broadcast_mul(
                &beta
                    .mul(&k.mul(&q).unwrap().sum(candle::D::Minus1).unwrap())
                    .unwrap()
                    .unsqueeze(2)
                    .unwrap(),
            )
            .unwrap();
        let drift = rms_norm_per_head(&o, &w.norm, eps)
            .unwrap()
            .sub(&rms_norm_per_head(&unscaled, &w.norm, eps).unwrap())
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            drift > 1e-6,
            "gated norm is insensitive to the read scale in this fixture \
             ({drift}) — it cannot detect the scale going missing"
        );
    }

    #[test]
    fn softplus_is_stable_at_extremes() {
        let dev = dev();
        let x = Tensor::from_vec(vec![-100f32, -1., 0., 1., 100.], (5,), &dev).unwrap();
        let y = softplus(&x).unwrap().to_vec1::<f32>().unwrap();
        assert!(y[0].abs() < 1e-6, "softplus(-100) ≈ 0, got {}", y[0]);
        assert!((y[2] - std::f32::consts::LN_2).abs() < 1e-6);
        assert!(
            (y[4] - 100.0).abs() < 1e-4,
            "softplus(100) ≈ 100, got {}",
            y[4]
        );
        assert!((y[1] - 0.313_261_7).abs() < 1e-5);
        assert!((y[3] - 1.313_261_7).abs() < 1e-5);
    }
}
