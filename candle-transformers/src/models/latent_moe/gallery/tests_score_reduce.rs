//! Tests for the fused Indexer score reduction against exact arithmetic and its
//! degenerate inputs.
//!
//! The fused reduction must be at least as faithful to exact f64 arithmetic as the
//! eager chain it replaces, over the decode and prefill shapes, packed and strided,
//! masked and unmasked. Its result must not depend on batch width. With zero heads
//! it must write zeros rather than leave uninitialised memory in the output, and
//! counts are read through their own stride, so a column of a wider tensor is never
//! taken for its neighbour.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod score_reduce_tests {
    use super::indexer_score_reduce;
    use candle::{DType, Device, Result, Tensor};

    /// The fused Indexer score reduction against the eager chain it replaces.
    ///
    /// Bit-equality is NOT the bar, and the reason is measured rather than
    /// assumed: candle reduces the head axis with a tree while the kernel walks
    /// it sequentially, so the summation orders differ by construction (3 ULP at
    /// H = 64) and matching would mean replicating candle's reduction shape.
    /// Since they cannot agree, the kernel is held to being at least as faithful
    /// to exact arithmetic as the eager chain — the same bar the compressor pool
    /// uses, and the one that matters here because this feeds an argsort where a
    /// few ULP can reorder near-tied candidates. The kernel accumulates EXACTLY
    /// in f32 (two-product by FMA, two-sum compensation), so it should win
    /// outright — measured 0 relative error on every shape below. Note the bar
    /// is not decoration: plain Kahan compensation measured WORSE than the eager
    /// chain here and failed this assert, because `relu` leaves every term
    /// non-negative while `w` is signed, so the running sum cancels toward zero
    /// while the products stay large. Do not weaken the summation to match a
    /// tolerance; the tolerance is the eager chain's own error.
    ///
    /// Covers both selectors' shapes: the decode wave's `[sessions, heads,
    /// shortlist]` and prefill's `[tokens, heads, corpus]`, masked and unmasked,
    /// including a row masked to ZERO valid columns and a row fully valid.
    /// Half the cases feed STRIDED scores — the pool kernel shipped with a
    /// packed-input assumption that only the model gate caught, so this covers
    /// that axis up front.
    #[test]
    #[ignore]
    fn indexer_score_reduce_matches_eager() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for &(b, h, m) in &[
            (16usize, 64usize, 128usize),
            (1, 64, 1024),
            (5, 32, 37),
            (128, 16, 512),
            (3, 1, 64),
        ] {
            let base = Tensor::randn(0f32, 1.0, (b, h, m), &dev)?;
            let w = Tensor::randn(0f32, 1.0, (b, h), &dev)?;
            // Strided on half the shapes: a channel-narrow of a wider block.
            let strided = m >= 128;
            let scores = if strided {
                let pad = Tensor::randn(0f32, 1.0, (b, h, m), &dev)?;
                Tensor::cat(&[&base, &pad], 2)?.narrow(2, 0, m)?
            } else {
                base
            };
            assert_eq!(
                scores.is_contiguous(),
                !strided,
                "strided setup ({b},{h},{m})"
            );

            // Counts spanning the interesting boundaries: none valid, all valid,
            // and a ragged middle.
            let cv: Vec<u32> = (0..b).map(|i| ((i * 7) % (m + 1)) as u32).collect();
            let counts = Tensor::from_vec(cv, b, &dev)?;

            for use_mask in [false, true] {
                let cnt = if use_mask { Some(&counts) } else { None };
                let got = indexer_score_reduce(&scores, &w, cnt)?;

                let want = {
                    let e = scores.relu()?.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)?;
                    match cnt {
                        None => e,
                        Some(c) => {
                            let col = Tensor::arange(0u32, m as u32, &dev)?.reshape((1, m))?;
                            let mask = col
                                .broadcast_lt(&c.reshape((b, 1))?)?
                                .to_dtype(DType::F32)?
                                .affine(1e30, -1e30)?;
                            e.broadcast_add(&mask)?
                        }
                    }
                };

                // The same expression in exact f64 on the host — the third
                // opinion that says which device path owns the divergence.
                let exact = {
                    let sc: Vec<f32> = scores.flatten_all()?.to_vec1()?;
                    let wv: Vec<f32> = w.flatten_all()?.to_vec1()?;
                    let cv: Vec<u32> = counts.flatten_all()?.to_vec1()?;
                    let mut o = vec![0f32; b * m];
                    for bi in 0..b {
                        for j in 0..m {
                            let mut s = 0f64;
                            for hi in 0..h {
                                let v = sc[(bi * h + hi) * m + j];
                                s += (v.max(0.0) as f64) * (wv[bi * h + hi] as f64);
                            }
                            let pad = use_mask && (j as u32) >= cv[bi];
                            o[bi * m + j] = s as f32 + if pad { -1e30 } else { 0.0 };
                        }
                    }
                    o
                };

                let a = want.flatten_all()?.to_vec1::<f32>()?;
                let x = got.flatten_all()?.to_vec1::<f32>()?;
                assert_eq!(a.len(), x.len(), "shape ({b},{h},{m}) mask={use_mask}");
                let rel = |p: f32, q: f32| {
                    if p == q {
                        0.0
                    } else {
                        (p - q).abs() / p.abs().max(q.abs()).max(1e-30)
                    }
                };
                let (mut k_err, mut e_err) = (0f32, 0f32);
                for ((p, q), z) in a.iter().zip(&x).zip(&exact) {
                    k_err = k_err.max(rel(*q, *z));
                    e_err = e_err.max(rel(*p, *z));
                }
                println!(
                    "[score b={b} h={h} m={m} mask={use_mask}] kernel-vs-exact \
                     {k_err:.3e}  eager-vs-exact {e_err:.3e}"
                );
                assert!(
                    k_err <= e_err.max(1e-7),
                    "kernel drifts from exact arithmetic MORE than the eager chain \
                     it replaces ({b},{h},{m}) mask={use_mask}: kernel {k_err:.3e} \
                     > eager {e_err:.3e}"
                );
            }
        }
        Ok(())
    }

    /// Isolation bench at the shapes both selectors actually run: the decode
    /// wave's `[sessions, idx_heads, shortlist]` and prefill's
    /// `[tokens, idx_heads, corpus]`.
    ///
    ///   cargo test -p candle-transformers --features cuda --release --lib \
    ///     bench_indexer_score -- --ignored --nocapture
    ///
    /// The `--launch-skip` is REQUIRED and shape-dependent: this bench ramps the
    /// device for 100 launches and then runs 8 warm-up + 5 x 40 timed launches
    /// per shape, so profiling from launch 0 captures the ramp (which runs
    /// unmasked) rather than any shape the bench reports. 850 lands in the
    /// prefill shape, the last of the five:
    ///
    ///   ncu --kernel-name indexer_score_reduce_kernel \
    ///     --launch-skip 850 --launch-count 1 \
    ///     --section SpeedOfLight --section ComputeWorkloadAnalysis \
    ///     target/release/deps/candle_transformers-<hash>.exe \
    ///     bench_indexer_score --ignored --nocapture
    ///
    /// ComputeWorkloadAnalysis is not optional here — it names the saturated
    /// PIPE. SpeedOfLight alone said "compute-bound" while the actual finding
    /// was FP64 at 84.7% on a part that runs FP64 at 1/64 rate.
    #[test]
    #[ignore]
    fn bench_indexer_score() -> Result<()> {
        let dev = Device::new_cuda(0)?;

        // Ramp the device BEFORE the first timed shape, so module load and the
        // idle-clock climb land outside the measurement.
        {
            let s = Tensor::randn(0f32, 1.0, (512usize, 64usize, 512usize), &dev)?;
            let wv = Tensor::randn(0f32, 1.0, (512usize, 64usize), &dev)?;
            for _ in 0..100 {
                let _ = indexer_score_reduce(&s, &wv, None)?;
            }
            dev.synchronize()?;
        }

        // Timing is MIN-of-rounds, not a single mean, because this box is WDDM:
        // some round takes a ~60 us scheduling stall that has nothing to do with
        // the kernel, and a mean folds it into the result. It is identifiable as
        // interference rather than cost because it lands on ONE shape per run,
        // a different shape each run, and inflates the eager and fused sides
        // together (one run put decode x1 at 113.68 us fused against 108.71 us
        // eager, where that shape steady-states at 11 vs 70). The minimum over
        // several rounds is the shape's real cost; the stall shows up as spread
        // between rounds, not as the reported number.
        let best = |f: &mut dyn FnMut() -> Result<()>| -> Result<f64> {
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                dev.synchronize()?;
                let t = std::time::Instant::now();
                for _ in 0..40 {
                    f()?;
                }
                dev.synchronize()?;
                best = best.min(t.elapsed().as_secs_f64() * 1e6 / 40.0);
            }
            Ok(best)
        };

        for &(b, h, m, tag) in &[
            (1usize, 64usize, 128usize, "decode x1"),
            (8, 64, 128, "decode x8"),
            (16, 64, 128, "decode x16"),
            (16, 64, 1024, "decode deep"),
            (512, 64, 512, "prefill"),
        ] {
            let scores = Tensor::randn(0f32, 1.0, (b, h, m), &dev)?;
            let w = Tensor::randn(0f32, 1.0, (b, h), &dev)?;
            let cv: Vec<u32> = (0..b).map(|i| ((i * 7) % (m + 1)) as u32).collect();
            let counts = Tensor::from_vec(cv, b, &dev)?;

            for _ in 0..8 {
                let _ = indexer_score_reduce(&scores, &w, Some(&counts))?;
            }
            let fused = best(&mut || {
                indexer_score_reduce(&scores, &w, Some(&counts))?;
                Ok(())
            })?;

            let eager = |s: &Tensor| -> Result<Tensor> {
                let e = s.relu()?.broadcast_mul(&w.unsqueeze(2)?)?.sum(1)?;
                let col = Tensor::arange(0u32, m as u32, &dev)?.reshape((1, m))?;
                let mask = col
                    .broadcast_lt(&counts.reshape((b, 1))?)?
                    .to_dtype(DType::F32)?
                    .affine(1e30, -1e30)?;
                e.broadcast_add(&mask)
            };
            for _ in 0..8 {
                let _ = eager(&scores)?;
            }
            let eg = best(&mut || {
                eager(&scores)?;
                Ok(())
            })?;

            // Bytes the kernel must move: scores once, w once, out once.
            let bytes = (b * h * m + b * h + b * m) * 4;
            println!(
                "[score {tag:>12}] b={b:<4} h={h:<3} m={m:<5} fused {fused:7.2} us  \
                 eager {eg:8.2} us  {:5.2}x   {:6.2} GB/s",
                eg / fused,
                bytes as f64 / (fused * 1e-6) / 1e9
            );
        }
        Ok(())
    }

    /// A row's reduced score must not depend on how many other rows shared the
    /// launch — the same batch-width invariance the compressor pool needs, and
    /// for the same reason: a decode wave reduces however many sessions selected
    /// this step, so if width perturbed the result a session's top-k would
    /// depend on who it was batched with.
    #[test]
    #[ignore]
    fn indexer_score_reduce_is_batch_width_invariant() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let (b, h, m) = (24usize, 64usize, 256usize);
        let scores = Tensor::randn(0f32, 1.0, (b, h, m), &dev)?;
        let w = Tensor::randn(0f32, 1.0, (b, h), &dev)?;
        let cv: Vec<u32> = (0..b).map(|i| ((i * 11) % (m + 1)) as u32).collect();
        let counts = Tensor::from_vec(cv, b, &dev)?;
        let batched = indexer_score_reduce(&scores, &w, Some(&counts))?;
        for i in 0..b {
            let alone = indexer_score_reduce(
                &scores.narrow(0, i, 1)?,
                &w.narrow(0, i, 1)?,
                Some(&counts.narrow(0, i, 1)?),
            )?;
            let p = batched.narrow(0, i, 1)?.flatten_all()?.to_vec1::<f32>()?;
            let q = alone.flatten_all()?.to_vec1::<f32>()?;
            for (j, (x, y)) in p.iter().zip(&q).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "row {i} elem {j} changed with batch width: {x} vs {y}"
                );
            }
        }
        Ok(())
    }

    /// The two degenerate inputs the fused path could get wrong SILENTLY.
    ///
    /// `h == 0`: the reduction over zero heads is 0, and the CUDA path has to
    /// WRITE that. `out` is allocated uninitialised (hot-path invariant 6), so a
    /// launcher that bails on an empty head axis hands whatever was in that VRAM
    /// straight to the argsort — no error, and a different wrong answer each
    /// run. The eager path returns zeros, so the two must agree.
    ///
    /// Strided `counts`: every other input is read through its own stride, so
    /// counts must be too. A column of a `[b, 2]` tensor has stride 2, and
    /// reading it as packed silently takes the NEIGHBOURING column as a
    /// visibility bound — masking real entries out of the top-k, or letting
    /// padding in, with nothing raised.
    #[test]
    #[ignore]
    fn degenerate_head_axis_and_strided_counts() -> Result<()> {
        let dev = Device::new_cuda(0)?;

        // h == 0, masked and unmasked.
        let (b, m) = (3usize, 16usize);
        let scores = Tensor::zeros((b, 0, m), DType::F32, &dev)?;
        let w = Tensor::zeros((b, 0), DType::F32, &dev)?;
        let cv: Vec<u32> = vec![0, 7, m as u32];
        let counts = Tensor::from_vec(cv.clone(), b, &dev)?;
        for cnt in [None, Some(&counts)] {
            let got = indexer_score_reduce(&scores, &w, cnt)?;
            assert_eq!(got.dims(), [b, m], "h=0 shape");
            let v = got.flatten_all()?.to_vec1::<f32>()?;
            for (i, x) in v.iter().enumerate() {
                let (row, col) = (i / m, i % m);
                let want = match cnt {
                    Some(_) if (col as u32) >= cv[row] => -1e30f32,
                    _ => 0f32,
                };
                assert_eq!(
                    x.to_bits(),
                    want.to_bits(),
                    "h=0 row {row} col {col}: got {x}, want {want} \
                     (uninitialised output?)"
                );
            }
        }

        // Strided counts: a column of a [b, 2] block, stride 2, against the
        // same values passed packed.
        let (b, h, m) = (6usize, 8usize, 32usize);
        let scores = Tensor::randn(0f32, 1.0, (b, h, m), &dev)?;
        let w = Tensor::randn(0f32, 1.0, (b, h), &dev)?;
        let want_v: Vec<u32> = (0..b).map(|i| ((i * 5) % (m + 1)) as u32).collect();
        // Interleave with a decoy column that would produce a DIFFERENT mask if
        // the kernel read this as packed.
        let mut inter: Vec<u32> = Vec::with_capacity(b * 2);
        for (i, &c) in want_v.iter().enumerate() {
            inter.push(c);
            inter.push(((i * 3 + 1) % (m + 1)) as u32);
        }
        let strided = Tensor::from_vec(inter, (b, 2), &dev)?
            .narrow(1, 0, 1)?
            .squeeze(1)?;
        assert_eq!(strided.dims(), [b], "strided counts shape");
        assert!(!strided.is_contiguous(), "counts should be strided");
        let packed = Tensor::from_vec(want_v, b, &dev)?;

        let a = indexer_score_reduce(&scores, &w, Some(&strided))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let e = indexer_score_reduce(&scores, &w, Some(&packed))?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (i, (x, y)) in a.iter().zip(&e).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "strided counts elem {i} ({}, {}) diverged: {x} vs {y}",
                i / m,
                i % m
            );
        }
        Ok(())
    }
}
