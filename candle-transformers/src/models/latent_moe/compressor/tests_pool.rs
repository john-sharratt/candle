//! Tests for the pool: its scalar references, the fused kernel against the eager
//! chain and exact arithmetic, and the close of a partial group.
//!
//! The non-overlapping pool is checked against a scalar softmax average, and the
//! seam fold against a whole-group fold. The fused kernel must be at least as
//! faithful to exact f32 arithmetic as the eager chain it replaces, over the
//! shapes the model pools, and a fully masked row must contribute exactly zero
//! weight. The pooled result must not depend on batch width. A trailing partial
//! group is checked against hand-computed references, and closing an empty buffer
//! returns nothing.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    #![allow(clippy::needless_range_loop)]

    use super::*;

    use candle::{Device, IndexOp};

    pub(in super::super) fn lin(x: &Tensor, w: &Tensor) -> Result<Tensor> {
        x.broadcast_matmul(&w.t()?)
    }

    /// [`pool_group`] over one already-dense `[G, P, d]` pair — the shape the
    /// pool's own gates are written against.
    pub(in super::super) fn pool_dense(kv: &Tensor, score: &Tensor) -> Result<Tensor> {
        let (g, _, d) = kv.dims3()?;
        let gp = GroupPool::dense(kv.clone(), score.clone(), vec![0u32; g]);
        let scope = desc::scope(kv.device())?;
        pool_group(&[&gp], d, &scope)
    }

    /// Non-overlapping pooling (`ratio != 4`) equals a scalar softmax-weighted average
    /// over each group of `ratio` consecutive tokens.
    #[test]
    fn nonoverlap_pool_matches_scalar() -> Result<()> {
        let dev = Device::Cpu;
        let (dim, d, ratio) = (6usize, 4usize, 3usize);
        let s = 7; // groups = 2, cutoff = 6, trailing token dropped
        let x = Tensor::randn(0f32, 1.0, (1, s, dim), &dev)?;
        let wkv = Tensor::randn(0f32, 1.0, (d, dim), &dev)?;
        let wgate = Tensor::randn(0f32, 1.0, (d, dim), &dev)?;
        let ape = Tensor::randn(0f32, 1.0, (ratio, d), &dev)?;
        let norm = Tensor::ones(d, DType::F32, &dev)?;
        let c = Compressor::new(
            wkv.clone(),
            wgate.clone(),
            ape.clone(),
            norm,
            ratio,
            d,
            2,
            1e-6,
        );
        let got = c.pool(&x)?.unwrap(); // [1, 2, d]

        // Scalar reference.
        let kv = lin(&x, &wkv)?.i(0)?.to_vec2::<f32>()?; // [s, d]
        let sc = lin(&x, &wgate)?.i(0)?.to_vec2::<f32>()?;
        let apev = ape.to_vec2::<f32>()?;
        let got = got.i(0)?.to_vec2::<f32>()?;
        let groups = s / ratio;
        for g in 0..groups {
            for chan in 0..d {
                // softmax over the ratio rows of (score + ape) for this channel.
                let mut logits = vec![0f32; ratio];
                for t in 0..ratio {
                    logits[t] = sc[g * ratio + t][chan] + apev[t][chan];
                }
                let m = logits.iter().cloned().fold(f32::MIN, f32::max);
                let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
                let z: f32 = exps.iter().sum();
                let mut acc = 0f32;
                for t in 0..ratio {
                    acc += exps[t] / z * kv[g * ratio + t][chan];
                }
                assert!(
                    (got[g][chan] - acc).abs() < 1e-4,
                    "g{g} c{chan}: {} vs {acc}",
                    got[g][chan]
                );
            }
        }
        Ok(())
    }

    /// §C/§E compression-seam monoid: a group's pooling rows folded WHOLE and
    /// folded in two fragments then LSE-merged both equal the single-shot
    /// softmax pool — the "two-turn seam-straddling group reconstructs ==
    /// single-shot forward of the concatenation" property. Also checks the
    /// monoid identity law. Pure `(m,l,acc)` arithmetic, model-independent.
    #[test]
    fn group_partial_seam_fold_matches_whole() -> Result<()> {
        let dev = Device::Cpu;
        let (r, d) = (6usize, 8usize);
        let scores = Tensor::randn(0f32, 1.0, (r, d), &dev)?;
        let kvs = Tensor::randn(0f32, 1.0, (r, d), &dev)?;

        // Single-shot reference: softmax over the r rows (per channel) · kv.
        let w = softmax(&scores, 0)?; // [r, d]
        let reference = w.broadcast_mul(&kvs)?.sum(0)?; // [d]
        let ref_v = reference.to_vec1::<f32>()?;

        let close = |a: &Tensor, msg: &str| -> Result<()> {
            let av = a.to_vec1::<f32>()?;
            for c in 0..d {
                assert!(
                    (av[c] - ref_v[c]).abs() < 1e-5,
                    "{msg} channel {c}: {} vs {}",
                    av[c],
                    ref_v[c]
                );
            }
            Ok(())
        };

        // Whole fold == reference.
        let whole = GroupPartial::identity(d, &dev)?
            .fold(&scores, &kvs)?
            .finalize()?;
        close(&whole, "whole fold")?;

        // Cut the group at every interior seam point; each split + merge ==
        // reference (order-independent, so both merge orders too).
        for cut in 1..r {
            let s1 = scores.narrow(0, 0, cut)?;
            let k1 = kvs.narrow(0, 0, cut)?;
            let s2 = scores.narrow(0, cut, r - cut)?;
            let k2 = kvs.narrow(0, cut, r - cut)?;
            let p1 = GroupPartial::identity(d, &dev)?.fold(&s1, &k1)?;
            let p2 = GroupPartial::identity(d, &dev)?.fold(&s2, &k2)?;
            close(
                &p1.merge(&p2)?.finalize()?,
                &format!("seam cut {cut} (p1⊕p2)"),
            )?;
            close(
                &p2.merge(&p1)?.finalize()?,
                &format!("seam cut {cut} (p2⊕p1)"),
            )?;
        }

        // Identity law: id ⊕ p == p ⊕ id == p.
        let p = GroupPartial::identity(d, &dev)?.fold(&scores, &kvs)?;
        let id = GroupPartial::identity(d, &dev)?;
        close(&id.merge(&p)?.finalize()?, "id ⊕ p")?;
        close(&p.merge(&id)?.finalize()?, "p ⊕ id")?;
        Ok(())
    }

    /// Overlapping pooling (`ratio == 4`): group 0 has no previous group, so its `-inf`
    /// prev-half is fully masked and the entry equals a pool over just the current 4 rows.
    #[test]
    fn overlap_group0_ignores_prev() -> Result<()> {
        let dev = Device::Cpu;
        let (dim, d, ratio) = (6usize, 4usize, 4usize);
        let s = 8; // groups = 2
        let x = Tensor::randn(0f32, 1.0, (1, s, dim), &dev)?;
        let wkv = Tensor::randn(0f32, 1.0, (2 * d, dim), &dev)?;
        let wgate = Tensor::randn(0f32, 1.0, (2 * d, dim), &dev)?;
        let ape = Tensor::randn(0f32, 1.0, (ratio, 2 * d), &dev)?;
        let norm = Tensor::ones(d, DType::F32, &dev)?;
        let c = Compressor::new(
            wkv.clone(),
            wgate.clone(),
            ape.clone(),
            norm,
            ratio,
            d,
            2,
            1e-6,
        );
        let got = c.pool(&x)?.unwrap().i((0, 0))?.to_vec1::<f32>()?; // group 0 entry

        // Scalar: group 0 pools the current 4 rows using the SECOND-half projection dims.
        let kv = lin(&x, &wkv)?.i(0)?.to_vec2::<f32>()?;
        let sc = lin(&x, &wgate)?.i(0)?.to_vec2::<f32>()?;
        let apev = ape.to_vec2::<f32>()?;
        for chan in 0..d {
            let mut logits = vec![0f32; ratio];
            for t in 0..ratio {
                // second-half dims are [d, 2d); ape added over the full 2d then sliced.
                logits[t] = sc[t][d + chan] + apev[t][d + chan];
            }
            let m = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            let mut acc = 0f32;
            for t in 0..ratio {
                acc += exps[t] / z * kv[t][d + chan];
            }
            assert!(
                (got[chan] - acc).abs() < 1e-4,
                "c{chan}: {} vs {acc}",
                got[chan]
            );
        }
        Ok(())
    }

    /// A group's pooled entry must not depend on how many OTHER groups were in
    /// the launch — pooling alone and pooling beside 31 others must give the
    /// same bytes.
    ///
    /// This is the property that keeps decode reproducible across session mixes:
    /// a wave pools however many sessions happened to close a group that step,
    /// so if batch width perturbed the result, the same session would get
    /// different corpus entries depending on who it was batched with, and MoE
    /// top-k would turn that into different tokens.
    /// `pool_batched_across_seqs_matches_per_seq` holds the same property for the
    /// eager path but runs on the CPU, so it never exercised the kernel; the
    /// kernel earns it structurally (one block per group per channel tile, no
    /// cross-group communication) and this pins that down.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn pool_kernel_is_batch_width_invariant() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for &(g, p, d) in &[(32usize, 8usize, 512usize), (7, 128, 512), (5, 8, 128)] {
            let kv = Tensor::randn(0f32, 1.0, (g, p, d), &dev)?;
            let score = Tensor::randn(0f32, 2.0, (g, p, d), &dev)?;
            let batched = pool_dense(&kv, &score)?;
            for i in 0..g {
                let alone = pool_dense(&kv.narrow(0, i, 1)?, &score.narrow(0, i, 1)?)?;
                let a = batched.narrow(0, i, 1)?.flatten_all()?.to_vec1::<f32>()?;
                let b = alone.flatten_all()?.to_vec1::<f32>()?;
                for (j, (x, y)) in a.iter().zip(&b).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "group {i} elem {j} changed with batch width \
                         (g={g}, p={p}, d={d}): batched {x} vs alone {y}"
                    );
                }
            }
        }
        Ok(())
    }

    /// Isolation bench for the fused pool, at the two shapes the model actually
    /// runs it at: the decode wave's per-layer pool (a handful of groups — one
    /// per session that closed a group this step) and a prefill fleet's pool
    /// (hundreds of groups at once). The decode shape is the interesting one,
    /// because a grid of one block per group leaves most of the GPU idle there,
    /// and that is invisible at the prefill shape.
    ///
    ///   cargo test -p candle-transformers --features cuda --release --lib \
    ///     bench_compressor_pool -- --ignored --nocapture
    ///   ncu --set full --kernel-name compressor_pool_kernel --launch-count 4 \
    ///     -o pool_ncu target/release/deps/candle_transformers-<hash>.exe \
    ///     bench_compressor_pool --ignored --nocapture
    /// Time `f` as the MINIMUM of five rounds. This box lands an intermittent
    /// ~60 us WDDM stall on one random shape per run; a single timed mean folds
    /// it in and makes an unrelated shape look like a regression. Applies to
    /// every microbench in this repo.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn best_us(
        dev: &Device,
        iters: usize,
        f: &mut dyn FnMut() -> Result<()>,
    ) -> Result<f64> {
        for _ in 0..8 {
            f()?;
        }
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            dev.synchronize()?;
            let t = std::time::Instant::now();
            for _ in 0..iters {
                f()?;
            }
            dev.synchronize()?;
            best = best.min(t.elapsed().as_secs_f64() * 1e6 / iters as f64);
        }
        Ok(best)
    }

    /// `n_pool` pools of one group each, described the way the DECODE streamer
    /// describes them: `p` separate one-row segments, because each row arrived
    /// on its own step from its own batched projection. This is the shape the
    /// wave actually pools at width — the dense `[G, P, d]` block is what
    /// prefill produces — and it is the one that stresses the descriptor table.
    #[cfg(feature = "cuda")]
    pub(in super::super) fn streaming_pools(
        n_pool: usize,
        p: usize,
        d: usize,
        dev: &Device,
    ) -> Result<(Vec<GroupPool>, Vec<Tensor>)> {
        let mut keep: Vec<Tensor> = Vec::new();
        let mut pools = Vec::with_capacity(n_pool);
        for _ in 0..n_pool {
            let mut segs = Vec::with_capacity(p);
            for _ in 0..p {
                let kv = Tensor::randn(0f32, 1.0, (1, 1, d), dev)?;
                let sc = Tensor::randn(0f32, 1.0, (1, 1, d), dev)?;
                keep.push(kv.clone());
                keep.push(sc.clone());
                segs.push(PoolSegs {
                    kv,
                    score: sc,
                    ape: None,
                });
            }
            pools.push(GroupPool {
                spans: vec![PoolSpan { segs }],
                positions: vec![0],
            });
        }
        Ok((pools, keep))
    }

    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn bench_compressor_pool() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        println!("--- dense segments (one wide segment per group: the prefill shape) ---");
        for &(g, p, d, tag) in &[
            (1usize, 8usize, 512usize, "decode ×1"),
            (8, 8, 512, "decode ×8"),
            (16, 8, 512, "decode ×16"),
            (16, 8, 128, "decode ×16 idx"),
            (256, 8, 512, "prefill fleet"),
            (16, 128, 512, "hca"),
        ] {
            let kv = Tensor::randn(0f32, 1.0, (g, p, d), &dev)?;
            let score = Tensor::randn(0f32, 1.0, (g, p, d), &dev)?;
            let fused = best_us(&dev, 200, &mut || pool_dense(&kv, &score).map(|_| ()))?;
            let eager = best_us(&dev, 200, &mut || {
                let w = softmax(&score, 1)?;
                kv.broadcast_mul(&w)?.sum(1).map(|_| ())
            })?;
            println!(
                "[pool {tag:>14}] g={g:<4} p={p:<4} d={d:<4}  fused {fused:7.2} us  \
                 eager {eager:7.2} us  {:.2}x",
                eager / fused
            );
        }

        // Where the wall actually goes. `ncu` puts the kernel at ~5 us on the
        // decode shape while the wall is ~15, so the majority is NOT the kernel:
        // it is the per-call device allocations and the descriptor upload. Print
        // them separately so an optimisation attempt aims at the right one.
        println!("--- per-call overhead (everything the launch is wrapped in) ---");
        {
            let words = 2 * 16 + 7 * 16 * 8; // 16 streaming pools, 8 segments each
            let upload = best_us(&dev, 200, &mut || {
                Tensor::from_vec(vec![0i64; words], words, &dev).map(|_| ())
            })?;
            let alloc = best_us(&dev, 200, &mut || {
                Tensor::empty((16, 512), DType::F32, &dev).map(|_| ())
            })?;
            println!(
                "[pool       overhead] desc upload ({words} i64) {upload:7.2} us   \
                 out alloc [16,512] {alloc:7.2} us"
            );
        }

        println!("--- one-row segments (the decode streamer's shape) ---");
        for &(n, p, d, tag) in &[
            (1usize, 8usize, 512usize, "stream ×1"),
            (8, 8, 512, "stream ×8"),
            (16, 8, 512, "stream ×16"),
            (16, 8, 128, "stream ×16 idx"),
            (64, 8, 512, "stream ×64"),
        ] {
            let (pools, _keep) = streaming_pools(n, p, d, &dev)?;
            let refs: Vec<&GroupPool> = pools.iter().collect();
            // Both staging paths, side by side: the wave's bump arena (the table
            // read in place over PCIe, no copy) against the fallback upload.
            let gen = desc::scope(&dev)?;
            let arena = best_us(&dev, 200, &mut || pool_group(&refs, d, &gen).map(|_| ()))?;
            println!(
                "[pool {tag:>14}] pools={n:<4} segs/group={p:<4} d={d:<4}  arena {arena:7.2} us"
            );
        }
        Ok(())
    }

    /// The fused pool kernel versus the eager chain it replaces
    /// (`softmax(score, 1)` → `broadcast_mul` → `sum(1)`), with the same
    /// expression in plain host f32 as the tiebreaker.
    ///
    /// Bit-equality with the eager chain is deliberately NOT the bar, because
    /// measurement says the eager chain is the inaccurate side: it tracks exact
    /// arithmetic ~3 orders of magnitude worse than the fused kernel does (the
    /// printed table shows it). Under `--use_fast_math` it pays `__expf` and an
    /// approximate division across five kernels, each rounding an intermediate
    /// out to memory and reading it back, while the fused pass keeps them in
    /// registers. So the assertion is that the kernel is at least as faithful to
    /// exact arithmetic as what it replaces — a strictly stronger property than
    /// reproducing the eager chain's drift, and the one that matters given MoE
    /// top-k turns small numerical differences into different tokens.
    ///
    /// Sweeps the shapes the model actually pools: `P` = 8 (overlapping CSA,
    /// 2·ratio), `P` = 128 (HCA, non-overlap at ratio 128) and `P` = 5 (a partial
    /// group closed at a turn seal), against both channel widths — 512 for the
    /// attention compressor, 128 for the indexer. Scores include a fully `-inf`
    /// row, the group-0 overlap pad, which must contribute exactly zero weight.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn pool_kernel_matches_eager() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        for &(g, p, d) in &[
            (3usize, 8usize, 512usize),
            (1, 8, 512),
            (5, 128, 512),
            (4, 8, 128),
            (2, 5, 128),
            (17, 8, 512),
        ] {
            let kv = Tensor::randn(0f32, 1.0, (g, p, d), &dev)?;
            let score = Tensor::randn(0f32, 2.0, (g, p, d), &dev)?;
            // Group 0's overlap pad: the first half of the pooling axis is
            // -inf, fully masked. `expf(-inf - m)` must land on exactly 0.
            let half = p / 2;
            let masked = Tensor::cat(
                &[
                    &Tensor::full(f32::NEG_INFINITY, (1, half, d), &dev)?,
                    &score.narrow(0, 0, 1)?.narrow(1, half, p - half)?,
                ],
                1,
            )?;
            let score = Tensor::cat(&[&masked, &score.narrow(0, 1, g - 1)?], 0)?;

            // Half the shapes go through as NON-CONTIGUOUS views — the channel
            // narrow the overlap split produces, and a dim-0 slice of a wider
            // block, which is what `assemble_groups_batched` hands over. The
            // first version of this kernel assumed packed inputs and the model
            // gate caught it, not this one; these views are that gap closed.
            let strided = d >= 128;
            let (kv, score) = if strided {
                let pad = Tensor::randn(0f32, 1.0, (g, p, d), &dev)?;
                (
                    Tensor::cat(&[&kv, &pad], D::Minus1)?.narrow(D::Minus1, 0, d)?,
                    Tensor::cat(&[&score, &pad], D::Minus1)?.narrow(D::Minus1, 0, d)?,
                )
            } else {
                (kv, score)
            };
            assert_eq!(
                kv.is_contiguous(),
                !strided,
                "the strided cases must actually be strided (g={g}, p={p}, d={d})"
            );

            let got = pool_dense(&kv, &score)?;
            let w = softmax(&score, 1)?;
            let want = kv.broadcast_mul(&w)?.sum(1)?;
            // The same expression on the HOST, in plain IEEE f32 with no
            // fast-math and no FMA contraction — the third opinion that says
            // which of the two device paths the seam belongs to.
            let cpu = {
                let kc = kv.to_device(&Device::Cpu)?;
                let sc = score.to_device(&Device::Cpu)?;
                let wc = softmax(&sc, 1)?;
                kc.broadcast_mul(&wc)?.sum(1)?
            };

            let a = want.flatten_all()?.to_vec1::<f32>()?;
            let b = got.flatten_all()?.to_vec1::<f32>()?;
            let c = cpu.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(a.len(), b.len(), "shape (g={g}, p={p}, d={d})");
            let rel = |x: f32, y: f32| (x - y).abs() / x.abs().max(y.abs()).max(1e-30);
            let (mut k_vs_e, mut k_vs_c, mut e_vs_c) = (0f32, 0f32, 0f32);
            for ((x, y), z) in a.iter().zip(&b).zip(&c) {
                k_vs_e = k_vs_e.max(rel(*x, *y));
                k_vs_c = k_vs_c.max(rel(*y, *z));
                e_vs_c = e_vs_c.max(rel(*x, *z));
            }
            println!(
                "[pool g={g} p={p} d={d}] kernel-vs-eager {k_vs_e:.3e}  \
                 kernel-vs-cpu {k_vs_c:.3e}  eager-vs-cpu {e_vs_c:.3e}"
            );
            // The kernel must be at least as faithful to exact arithmetic as the
            // eager device chain it replaces — that is the property that matters,
            // and it is stronger than agreeing with the eager chain's own drift.
            assert!(
                k_vs_c <= e_vs_c.max(1e-6),
                "kernel drifts from exact arithmetic MORE than the eager chain it \
                 replaces (g={g}, p={p}, d={d}): kernel-vs-cpu {k_vs_c:.3e} > \
                 eager-vs-cpu {e_vs_c:.3e}"
            );
        }
        Ok(())
    }

    /// `close` finalizes a trailing partial group of `n < ratio` buffered rows
    /// (non-overlapping compressor): the pooled entry is the softmax over
    /// exactly those `n` rows of `(score + ape[:n])` per channel, then RMSNorm —
    /// hand-computed scalar reference. Also checks that closing an empty buffer
    /// (right after a group emitted) yields `None`.
    #[test]
    fn close_pools_trailing_partial_nonoverlap() -> Result<()> {
        let dev = Device::Cpu;
        let (dim, d, ratio, rd) = (6usize, 4usize, 3usize, 2usize);
        let wkv = Tensor::randn(0f32, 1.0, (d, dim), &dev)?;
        let wgate = Tensor::randn(0f32, 1.0, (d, dim), &dev)?;
        let ape = Tensor::randn(0f32, 1.0, (ratio, d), &dev)?;
        let norm = Tensor::randn(0f32, 1.0, d, &dev)?;
        let c = Compressor::new(
            wkv.clone(),
            wgate.clone(),
            ape.clone(),
            norm.clone(),
            ratio,
            d,
            rd,
            1e-6,
        );

        // Exactly `ratio` rows: a group emits and the buffer empties, so a close
        // immediately after must return None (boundary fell on the seal).
        let x_full = Tensor::randn(0f32, 1.0, (1, ratio, dim), &dev)?;
        let mut inc0 = c.incremental();
        let mut emitted = 0;
        for t in 0..ratio {
            if inc0.push_raw(&x_full.i((0, t))?)?.is_some() {
                emitted += 1;
            }
        }
        assert_eq!(emitted, 1);
        assert!(inc0.close()?.is_none(), "empty buffer must close to None");

        // Two buffered rows (< ratio=3): no group emitted, then close.
        let n = 2usize;
        let x = Tensor::randn(0f32, 1.0, (1, n, dim), &dev)?;
        let mut inc = c.incremental();
        for t in 0..n {
            assert!(inc.push_raw(&x.i((0, t))?)?.is_none());
        }
        let (entry, pos) = inc.close()?.expect("partial rows must close to an entry");
        assert_eq!(pos, 0, "first group starts at position 0");
        let got = entry.reshape(d)?.to_vec1::<f32>()?;

        // Scalar reference: per-channel softmax over the n rows of (score+ape[:n]),
        // weighted sum of kv, then RMSNorm·norm_w.
        let kv = lin(&x, &wkv)?.i(0)?.to_vec2::<f32>()?; // [n, d]
        let sc = lin(&x, &wgate)?.i(0)?.to_vec2::<f32>()?;
        let apev = ape.to_vec2::<f32>()?;
        let normv = norm.to_vec1::<f32>()?;
        let mut pooled = vec![0f32; d];
        for (chan, p) in pooled.iter_mut().enumerate() {
            let logits: Vec<f32> = (0..n).map(|t| sc[t][chan] + apev[t][chan]).collect();
            let m = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            *p = (0..n).map(|t| exps[t] / z * kv[t][chan]).sum();
        }
        let ms: f32 = pooled.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for (chan, &pv) in pooled.iter().enumerate() {
            let expect = pv * inv * normv[chan];
            assert!(
                (got[chan] - expect).abs() < 1e-4,
                "c{chan}: {} vs {expect}",
                got[chan]
            );
        }
        Ok(())
    }

    /// `close` on the overlapping (`ratio == 4`) compressor after one complete
    /// group: the partial pool includes the previous group's first-half rows
    /// (already `+ape`) alongside the buffered second-half rows — hand-computed
    /// scalar reference, then RMSNorm.
    #[test]
    fn close_pools_trailing_partial_overlap() -> Result<()> {
        let dev = Device::Cpu;
        let (dim, d, ratio, rd) = (6usize, 4usize, 4usize, 2usize);
        let wkv = Tensor::randn(0f32, 1.0, (2 * d, dim), &dev)?;
        let wgate = Tensor::randn(0f32, 1.0, (2 * d, dim), &dev)?;
        let ape = Tensor::randn(0f32, 1.0, (ratio, 2 * d), &dev)?;
        let norm = Tensor::randn(0f32, 1.0, d, &dev)?;
        let c = Compressor::new(
            wkv.clone(),
            wgate.clone(),
            ape.clone(),
            norm.clone(),
            ratio,
            d,
            rd,
            1e-6,
        );

        // One complete group (4 rows) then a partial (2 rows), then close.
        let n = 2usize;
        let total = ratio + n; // 6
        let x = Tensor::randn(0f32, 1.0, (1, total, dim), &dev)?;
        let mut inc = c.incremental();
        let mut emitted = 0;
        for t in 0..total {
            if inc.push_raw(&x.i((0, t))?)?.is_some() {
                emitted += 1;
            }
        }
        assert_eq!(
            emitted, 1,
            "group 0 emits, the trailing 2 rows stay buffered"
        );
        let (entry, pos) = inc.close()?.expect("partial closes to an entry");
        assert_eq!(pos, ratio as u32, "group 1 starts at position ratio");
        let got = entry.reshape(d)?.to_vec1::<f32>()?;

        // Reference: pool = prev group-0 first-half rows (t=0..ratio, dims [0,d),
        // score+ape[t]) ‖ curr partial rows (t=ratio..ratio+n, dims [d,2d),
        // score+ape[within]). Softmax over the r+n rows per channel, then RMSNorm.
        let kv = lin(&x, &wkv)?.i(0)?.to_vec2::<f32>()?; // [total, 2d]
        let sc = lin(&x, &wgate)?.i(0)?.to_vec2::<f32>()?;
        let apev = ape.to_vec2::<f32>()?;
        let normv = norm.to_vec1::<f32>()?;
        let mut pooled = vec![0f32; d];
        for (chan, p) in pooled.iter_mut().enumerate() {
            let mut logits: Vec<f32> = Vec::new();
            let mut kvs: Vec<f32> = Vec::new();
            for t in 0..ratio {
                logits.push(sc[t][chan] + apev[t][chan]);
                kvs.push(kv[t][chan]);
            }
            for j in 0..n {
                let t = ratio + j;
                logits.push(sc[t][d + chan] + apev[j][d + chan]);
                kvs.push(kv[t][d + chan]);
            }
            let m = logits.iter().cloned().fold(f32::MIN, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&v| (v - m).exp()).collect();
            let z: f32 = exps.iter().sum();
            *p = exps.iter().zip(&kvs).map(|(&e, &k)| e / z * k).sum();
        }
        let ms: f32 = pooled.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let inv = 1.0 / (ms + 1e-6).sqrt();
        for (chan, &pv) in pooled.iter().enumerate() {
            let expect = pv * inv * normv[chan];
            assert!(
                (got[chan] - expect).abs() < 1e-4,
                "c{chan}: {} vs {expect}",
                got[chan]
            );
        }
        Ok(())
    }

    #[test]
    fn forward_shape_and_finite() -> Result<()> {
        let dev = Device::Cpu;
        let (dim, d, ratio, rd) = (8usize, 6usize, 4usize, 4usize);
        let rope = RotaryCache::new(rd, 160000.0, 64, 16.0, 32.0, 1.0, &dev)?;
        let x = Tensor::randn(0f32, 1.0, (2, 20, dim), &dev)?;
        let c = Compressor::new(
            Tensor::randn(0f32, 1.0, (2 * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (2 * d, dim), &dev)?,
            Tensor::randn(0f32, 1.0, (ratio, 2 * d), &dev)?,
            Tensor::ones(d, DType::F32, &dev)?,
            ratio,
            d,
            rd,
            1e-6,
        );
        let out = c.forward(&x, &rope)?.unwrap();
        assert_eq!(out.dims(), &[2, 5, d]); // 20/4 = 5 groups
        assert!(out
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite()));
        Ok(())
    }
}
