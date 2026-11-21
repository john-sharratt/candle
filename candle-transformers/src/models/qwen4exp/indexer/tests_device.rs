//! Tests the device path against the CPU oracle, and the rotation kernels
//! against the host's arithmetic.
//!
//! Selections must agree with the oracle row for row, since the gaps at the cut
//! are wider than rounding. Appends are tested in ragged waves so the open block
//! straddles a wave boundary. The lookup is compared bit for bit with the host
//! table at the positions where its split turns over. Rows on different rungs
//! must not bleed into one another in a shared launch.

use super::*;

// Pinned ranges are lists of ranges, and a list of one is the common case.
#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
pub(super) mod tests {
    use super::*;

    use crate::models::qwen35::attention::RopeTables;

    use crate::models::qwen4exp::qsa::{qsa_selection_mask, IndexState};

    use crate::models::qwen4exp::qsa_select::{entry_block, entry_cells};

    use crate::models::rope_schedule::plain_inv_freq;

    use super::super::tests_support::tests::{cuda, lcg};

    /// The whole device path — project, pool, norm, rope, score, select —
    /// against the CPU oracle's `qsa_selection_mask` on the same weights.
    ///
    /// The two compute their scores through different GEMMs, so in principle a
    /// rank could flip where two blocks' scores agree to within f32 rounding.
    /// The scores here are continuous and the gaps at the cut are orders of
    /// magnitude wider than that, so the selections are required to agree
    /// EXACTLY, row for row — if this ever becomes flaky the right answer is
    /// to say so, not to widen it into a test that no longer checks anything.
    #[test]
    fn device_selection_matches_the_cpu_oracle() -> Result<()> {
        let Some(device) = cuda() else { return Ok(()) };
        let cfg = IndexerConfig {
            n_heads: 2,
            head_dim: 16,
            top_k: 8,
            strata: Strata::WHOLE,
        };
        let (ratio, hidden, eps) = (4usize, 12usize, 1e-6);
        let t = 64usize;

        let w = |dev: &Device| -> Result<IndexerWeights> {
            Ok(IndexerWeights {
                q_proj: Tensor::from_vec(
                    lcg(cfg.n_heads * cfg.head_dim * hidden, 0x61, 0.6),
                    (cfg.n_heads * cfg.head_dim, hidden),
                    dev,
                )?,
                k_proj: Tensor::from_vec(
                    lcg(cfg.head_dim * hidden, 0x62, 0.6),
                    (cfg.head_dim, hidden),
                    dev,
                )?,
                q_norm: Tensor::from_vec(
                    lcg(cfg.head_dim, 0x63, 0.2)
                        .into_iter()
                        .map(|v| v + 1.0)
                        .collect::<Vec<f32>>(),
                    (cfg.head_dim,),
                    dev,
                )?,
                k_norm: Tensor::from_vec(
                    lcg(cfg.head_dim, 0x64, 0.2)
                        .into_iter()
                        .map(|v| v + 1.0)
                        .collect::<Vec<f32>>(),
                    (cfg.head_dim,),
                    dev,
                )?,
            })
        };
        let x_host = lcg(t * hidden, 0x65, 1.0);
        let x_gpu = Tensor::from_vec(x_host.clone(), (t, hidden), &device)?;
        let x_cpu = Tensor::from_vec(x_host, (t, hidden), &Device::Cpu)?;
        let rope_gpu = FactoredRope::new(&plain_inv_freq(8, 1e6), &device)?;
        let rope_cpu = RopeTables::new(8, 1e6, 512, &Device::Cpu)?;

        // Device: append the segment's keys, then select its rows.
        let mut cache = IndexCache::new(cfg.head_dim, &device)?;
        cache.ensure_capacity(t, ratio)?;
        let w_gpu = w(&device)?;
        let k_all = project_keys(&x_gpu, &w_gpu)?;
        // Appended in RAGGED waves, not one span. None of 7, 5, 20 is a
        // multiple of `ratio`, so every wave but the last leaves an open block
        // and the next wave's first key straddles the carried rows — which is
        // the case the batched append exists to handle and the case a
        // single-span test cannot reach. The oracle is unchanged: the same 64
        // tokens must produce the same selection however they arrived.
        let mut at = 0usize;
        for rows in [7usize, 5, 20, 32] {
            let mut work = [AppendSpan {
                cache: &mut cache,
                start: at,
                rows,
            }];
            append_wave(&mut work, &k_all, &w_gpu, ratio, eps)?;
            at += rows;
            assert_eq!(cache.len(ratio), at, "cache length after {at} tokens");
        }
        assert_eq!(at, t);
        let qpos: Vec<usize> = (0..t).collect();
        let q_all = project_queries(
            &x_gpu,
            &w_gpu,
            &cfg,
            &rope_gpu,
            &qpos,
            RowRungs::Uniform(0),
            eps,
            None,
        )?;
        let widest = t.div_ceil(ratio).max(1);
        let mut table =
            SelectionTable::new(t, ratio, cfg.top_k, widest, &cfg.strata, &device, None)?;
        let scores = Tensor::empty((t, widest), DType::F32, &device)?;
        let cand = cache.score_rows(
            &q_all, &qpos, &cfg, ratio, &rope_gpu, 0, &scores, widest, 0, None,
        )?;
        let tail: Vec<u32> = qpos.iter().map(|&p| cache.tail_len(p, ratio)).collect();
        let prompt = vec![0u32; t];
        let pins = vec![PinRow::EMPTY; t];
        table.fill_rows(
            &scores, &cand, &qpos, &tail, &prompt, &pins, ratio, cfg.top_k, 0,
        )?;

        // Oracle: the additive mask, from the same weights on the CPU.
        let w_cpu = w(&Device::Cpu)?;
        let mut state = IndexState::empty();
        let mask = qsa_selection_mask(&x_cpu, &w_cpu, &mut state, &rope_cpu, ratio, &cfg, eps)?
            .expect("64 tokens is past the 11-cell budget");
        let mask: Vec<Vec<f32>> = mask.to_vec2()?;

        for (r, mrow) in mask.iter().enumerate() {
            let want: Vec<usize> = mrow
                .iter()
                .enumerate()
                .filter(|(_, &v)| v == 0.0)
                .map(|(j, _)| j)
                .collect();
            match table.row_to_host(r)? {
                None => {
                    assert_eq!(want, (0..=r).collect::<Vec<_>>(), "row {r} dense mismatch");
                }
                Some(entries) => {
                    let mut got: Vec<usize> = entries
                        .iter()
                        .flat_map(|&e| {
                            let b = entry_block(e);
                            (0..entry_cells(e)).map(move |c| b * ratio + c)
                        })
                        .collect();
                    got.sort_unstable();
                    assert_eq!(got, want, "row {r} disagrees with the oracle's selection");
                }
            }
        }
        Ok(())
    }

    /// **The device lookup is the host mirror, bit for bit.** A unit pair
    /// `(1, 0)` rotates to exactly `(cos, sin)` — every product is by 1 or 0 —
    /// so the rotated row is the kernel's lookup itself, compared against
    /// `table::lookup` at the positions where the factored split turns over:
    /// both ends of `LO`, every `HI` boundary sampled across the reach, and the
    /// last position. For both rotary widths the fleet runs.
    #[test]
    fn device_lookup_is_the_host_mirror_bit_for_bit() -> Result<()> {
        use crate::models::rope_schedule::ROPE_REACH;
        let Some(device) = cuda() else { return Ok(()) };
        let mut pos: Vec<usize> = vec![0, 1, 1023, 1024, 1025, ROPE_REACH - 1];
        for k in (1..2048).step_by(97) {
            pos.extend([(k << 10) - 1, k << 10, (k << 10) + 1]);
        }
        for (pairs, theta) in [(32usize, 1e7f32), (64, 1e6)] {
            let rope = FactoredRope::new(&plain_inv_freq(2 * pairs, theta), &device)?;
            let d = 2 * pairs;
            let mut host = vec![0f32; pos.len() * d];
            for r in 0..pos.len() {
                host[r * d..r * d + pairs].fill(1.0);
            }
            let src = Tensor::from_vec(host, (pos.len(), d), &device)?;
            let got = rotate_rows(
                RowSource::Dense(&src),
                &rope,
                1,
                RowPositions::PerGroup(&pos),
                RowRungs::Uniform(0),
                RotSide::Key,
                None,
            )?
            .to_vec2::<f32>()?;
            for (r, &p) in pos.iter().enumerate() {
                for k in 0..pairs {
                    let (c, s) = rope.rungs().cos_sin(0, p, k);
                    assert_eq!(
                        (got[r][k].to_bits(), got[r][k + pairs].to_bits()),
                        (c.to_bits(), s.to_bits()),
                        "P {pairs}, position {p}, pair {k}: device ({}, {}) against host ({c}, {s})",
                        got[r][k],
                        got[r][k + pairs]
                    );
                }
            }
        }
        Ok(())
    }

    /// **Normed rotation is the eager norm then the rotation, to rounding.** The
    /// queries' fused launch against `rms_norm_last` followed by the dense
    /// rotation, at the indexer's head width on two rungs; and the fused launch
    /// repeats bit for bit.
    #[test]
    fn normed_rotation_is_the_eager_norm_then_the_rotation() -> Result<()> {
        use super::super::super::qsa::rms_norm_last;
        use crate::models::rope_schedule::{RopeRungs, RopeSchedule, Rung};
        let Some(device) = cuda() else { return Ok(()) };
        let schedule = RopeSchedule::yarn(
            64,
            1e6,
            4096,
            vec![
                Rung {
                    ceiling: 4096,
                    factor: 1.0,
                },
                Rung {
                    ceiling: 16384,
                    factor: 4.0,
                },
            ],
            true,
        )?;
        let rungs = RopeRungs::new(&schedule, &device)?;
        let rope = FactoredRope::over(&rungs, &device)?;
        let (heads, d) = (4usize, 128usize);
        let pos = [3usize, 900, 5000, 12_000, 17];
        let row_rung = [0u32, 0, 1, 1, 0];
        let n = pos.len() * heads;
        let src = Tensor::from_vec(lcg(n * d, 0x6B, 4.0), (n, d), &device)?;
        let weight = Tensor::from_vec(lcg(d, 0x6C, 1.0), (d,), &device)?;
        let eps = 1e-6;
        let normed = || {
            rotate_rows(
                RowSource::Normed {
                    rows: &src,
                    weight: &weight,
                    eps,
                },
                &rope,
                heads,
                RowPositions::PerGroup(&pos),
                RowRungs::PerGroup(&row_rung),
                RotSide::Query,
                None,
            )
        };
        let got = normed()?;
        let eager = rms_norm_last(&src, &weight, eps)?;
        let want = rotate_rows(
            RowSource::Dense(&eager),
            &rope,
            heads,
            RowPositions::PerGroup(&pos),
            RowRungs::PerGroup(&row_rung),
            RotSide::Query,
            None,
        )?;
        let diff = (&got - &want)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let scale = want.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        assert!(
            diff <= 1e-5 * scale.max(1.0),
            "fused norm+rotate off the eager chain by {diff} at scale {scale}"
        );
        assert_eq!(
            got.flatten_all()?.to_vec1::<f32>()?,
            normed()?.flatten_all()?.to_vec1::<f32>()?,
            "the fused launch differs between two runs"
        );
        Ok(())
    }

    /// **Rows on different rungs share one rotation launch without bleed.** A
    /// wave's queries sit on their sequences' rungs; rotated together, each row
    /// is bit-for-bit what it is rotated alone at its own rung, and matches the
    /// host mirror of that rung's table with that rung's `m²` — while a key
    /// rotation at the same rung takes no scale.
    #[test]
    fn rows_on_different_rungs_share_a_launch_without_bleed() -> Result<()> {
        use crate::models::rope_schedule::{RopeRungs, RopeSchedule, Rung};
        let Some(device) = cuda() else { return Ok(()) };
        let schedule = RopeSchedule::yarn(
            8,
            1e6,
            64,
            vec![
                Rung {
                    ceiling: 64,
                    factor: 1.0,
                },
                Rung {
                    ceiling: 128,
                    factor: 2.0,
                },
                Rung {
                    ceiling: 256,
                    factor: 4.0,
                },
            ],
            true,
        )?;
        let rungs = RopeRungs::new(&schedule, &device)?;
        let rope = FactoredRope::over(&rungs, &device)?;
        let (heads, d) = (2usize, 16usize);
        let pos = [3usize, 90, 200, 17, 130, 250];
        let row_rung = [0u32, 1, 2, 0, 2, 1];
        let n = pos.len() * heads;
        let host = lcg(n * d, 0x5A, 1.0);
        let src = Tensor::from_vec(host.clone(), (n, d), &device)?;
        let mixed = rotate_rows(
            RowSource::Dense(&src),
            &rope,
            heads,
            RowPositions::PerGroup(&pos),
            RowRungs::PerGroup(&row_rung),
            RotSide::Query,
            None,
        )?
        .to_vec2::<f32>()?;
        for (g, (&p, &r)) in pos.iter().zip(&row_rung).enumerate() {
            // Its own tensor: the launch takes a whole operand, never a view.
            let one = Tensor::from_vec(
                host[g * heads * d..(g + 1) * heads * d].to_vec(),
                (heads, d),
                &device,
            )?;
            let alone = rotate_rows(
                RowSource::Dense(&one),
                &rope,
                heads,
                RowPositions::PerGroup(&[p]),
                RowRungs::Uniform(r),
                RotSide::Query,
                None,
            )?
            .to_vec2::<f32>()?;
            let key = rotate_rows(
                RowSource::Dense(&one),
                &rope,
                heads,
                RowPositions::PerGroup(&[p]),
                RowRungs::Uniform(r),
                RotSide::Key,
                None,
            )?
            .to_vec2::<f32>()?;
            let m2 = rungs.q_scale(r);
            for h in 0..heads {
                let row = g * heads + h;
                assert_eq!(mixed[row], alone[h], "row {row} bled across rungs");
                let s = &host[row * d..(row + 1) * d];
                for k in 0..4 {
                    let (c, sn) = rungs.cos_sin(r, p, k);
                    let lo = s[k] * c - s[k + 4] * sn;
                    let hi = s[k + 4] * c + s[k] * sn;
                    for (got, want, what) in [
                        (mixed[row][k], lo * m2, "query lo"),
                        (mixed[row][k + 4], hi * m2, "query hi"),
                        (key[h][k], lo, "key lo"),
                        (key[h][k + 4], hi, "key hi"),
                    ] {
                        assert!(
                            (got - want).abs() <= 1e-5,
                            "row {row} pair {k} rung {r}: {what} {got} against {want}"
                        );
                    }
                }
                assert_eq!(
                    &mixed[row][8..],
                    &s[8..],
                    "row {row}: pass-through channels"
                );
            }
        }
        assert!(rotate_rows(
            RowSource::Dense(&src),
            &rope,
            heads,
            RowPositions::PerGroup(&pos),
            RowRungs::Uniform(3),
            RotSide::Key,
            None,
        )
        .is_err());
        Ok(())
    }

    /// `rows_per_pos == 0` would divide by zero inside the kernel, so the
    /// guard there refuses to launch — but that guard also means `dst` never
    /// gets a single byte written to it. This needs no GPU: the check runs
    /// before `rotate_rows` ever asks whether `src`'s device is CUDA.
    #[test]
    fn rows_per_pos_zero_is_refused_before_dst_is_left_uninitialised() -> Result<()> {
        let device = Device::Cpu;
        let rope = FactoredRope::new(&plain_inv_freq(4, 1e6), &device)?;
        let src = Tensor::from_vec(vec![0f32; 8], (1, 8), &device)?;
        assert!(rotate_rows(
            RowSource::Dense(&src),
            &rope,
            0,
            RowPositions::Affine { base: 0, step: 1 },
            RowRungs::Uniform(0),
            RotSide::Key,
            None,
        )
        .is_err());
        Ok(())
    }

    /// The tail's route turns on the cell count at 2²⁵, however the cells
    /// split between rows and tail blocks.
    #[test]
    fn tail_route_turns_on_cells() {
        assert_eq!(GEMM_TAIL_MIN_CELLS, 33_554_432);
        assert_eq!(TailRoute::for_span(1, 0), TailRoute::Paged);
        assert_eq!(TailRoute::for_span(4096, 8191), TailRoute::Paged);
        assert_eq!(TailRoute::for_span(127, 262_144), TailRoute::Paged);
        assert_eq!(TailRoute::for_span(4096, 8192), TailRoute::Gemm);
        assert_eq!(TailRoute::for_span(128, 262_144), TailRoute::Gemm);
        assert_eq!(TailRoute::for_span(2048, 32_768), TailRoute::Gemm);
        assert_eq!(TailRoute::for_span(usize::MAX, 2), TailRoute::Gemm);
    }
}
