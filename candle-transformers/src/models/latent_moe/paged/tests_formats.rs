//! Gates the kernel's dequantisation of every band format against the CPU
//! mirror, one format at a time and then all of them combined.
//!
//! Each format is checked bit-exact, or within one bf16 code where the dequant
//! divide runs under fast-math, so a failure names the format responsible. The
//! stage dump is compared stage by stage, which locates the first diverging
//! quantity. A window-staging test checks the kernel's dequantised values
//! element by element against the values the codec stored.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod tests {
    // Host mirrors of the latent kernels. Every one of these is a transcription:
    // the kernel's own loop bounds, its float literals to full width, and its
    // `fminf`/`fmaxf` NaN behaviour. Restyling them into iterator form or
    // trimming digits would break the line-by-line diff against the `.cu` that
    // makes a mismatch findable.
    #![allow(
        clippy::assign_op_pattern,
        clippy::excessive_precision,
        clippy::needless_range_loop,
        clippy::neg_cmp_op_on_partial_ord,
        clippy::type_complexity
    )]

    use super::*;

    use candle::Device;

    use super::super::tests_geometry::tests::H;

    use super::super::tests_oracle::tests::{
        bf16_round_host, check_within_one_code, gen_inputs, mirror_fmt, rope_vec, run_kernel_fmt,
        Case,
    };

    /// Stage-by-stage divergence localizer: runs the tiny live-rope case with
    /// the kernel's stage dump and compares each staged quantity against the
    /// mirror — the first diverging stage names the bug.
    #[test]
    #[ignore]
    fn mirror_stage_dump_rope() -> Result<()> {
        let case = Case {
            n_win: 1,
            window_size: 128,
            g_total: 0,
            ratio: 4,
            selected: vec![],
            num_splits: 1,
            zero_rope: false,
        };
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let inp = gen_inputs(&case, 42);
        let dev = Device::new_cuda(0)?;
        let slots = SyntheticSlots::build(&dev, std::slice::from_ref(&inp.window))?;
        let qf: Vec<f32> = inp.q.iter().flat_map(|h| h.iter().copied()).collect();
        let q = Tensor::from_vec(qf, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let kv_new =
            Tensor::from_vec(inp.kv_new.to_vec(), (1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let comp = Tensor::zeros((1, HEAD_DIM), DType::F32, &dev)?;
        let comp_pos = Tensor::zeros(1, DType::U32, &dev)?;
        let comp_idx = Tensor::full(u32::MAX, (1, 1), &dev)?;
        let comp_cnt = Tensor::zeros(1, DType::U32, &dev)?;
        let sinks = Tensor::from_vec(inp.sinks.clone(), H, &dev)?;
        let freqs = Tensor::from_vec(inp.freqs.clone(), ROPE_DIM / 2, &dev)?;
        let rope_tab = build_rope_table(&freqs)?;
        let ws = LatentWorkspace::build(&dev)?;
        let dbg = Tensor::zeros(DBG_LEN, DType::F32, &dev)?;
        let _ = paged_latent_decode(
            &q,
            &slots.headers,
            &kv_new,
            &CorpusCache::build(&comp, &comp_pos)?,
            &comp_idx,
            &comp_cnt,
            &Tensor::from_vec(vec![case.n_win as u32], 1, &dev)?,
            &sinks,
            &rope_tab,
            &ws,
            softmax_scale,
            case.window_size,
            1,
            Some(&dbg),
        )?;
        let d = dbg.to_vec1::<f32>()?;

        // Mirror stage quantities for tile 0 (keys: window token pos 0 + writer
        // pos 1) and heads 0..16.
        let stored: Vec<[f32; HEAD_DIM]> = inp
            .window
            .iter()
            .map(|w| std::array::from_fn(|dd| e4m3_to_f32(f32_to_e4m3(w[dd]))))
            .chain(std::iter::once(std::array::from_fn(|dd| {
                e4m3_to_f32(f32_to_e4m3(inp.kv_new[dd]))
            })))
            .collect();
        let q_pos = case.n_win as u32;

        // Q stage.
        for h in 0..16 {
            let mut v = inp.q[h];
            rope_vec(&mut v, q_pos, &inp.freqs);
            let (qi, qs) = quant_bands(&v);
            for p in 0..N_BANDS {
                let got = d[DBG_SCALEQ + h * N_BANDS + p];
                assert_eq!(
                    got.to_bits(),
                    qs[p].to_bits(),
                    "scaleQ[{h}][{p}]: device {got} vs mirror {}",
                    qs[p]
                );
            }
            for dd in 0..HEAD_DIM {
                let got = d[DBG_SQ + h * HEAD_DIM + dd] as i32 as i8;
                assert_eq!(
                    got, qi[dd],
                    "sQ[{h}][{dd}]: device {got} vs mirror {}",
                    qi[dd]
                );
            }
        }

        // K stage (tile 0: both keys). int8 quant from the roped f32; the
        // staged bf16 is the PV read.
        for t in 0..2usize {
            let mut l = stored[t];
            rope_vec(&mut l, t as u32, &inp.freqs);
            let mut staged = l;
            for x in staged.iter_mut() {
                *x = half::bf16::from_f32(*x).to_f32();
            }
            let (ki, ks) = quant_bands(&l);
            for dd in 0..HEAD_DIM {
                let got = d[DBG_KVF + t * HEAD_DIM + dd];
                assert_eq!(
                    got.to_bits(),
                    staged[dd].to_bits(),
                    "kv_f[{t}][{dd}]: device {got} vs mirror {}",
                    staged[dd]
                );
            }
            for p in 0..N_BANDS {
                let got = d[DBG_SCALEK + t * N_BANDS + p];
                assert_eq!(
                    got.to_bits(),
                    ks[p].to_bits(),
                    "scaleK[{t}][{p}]: device {got} vs mirror {}",
                    ks[p]
                );
            }
            for dd in 0..HEAD_DIM {
                let got = d[DBG_SK + t * HEAD_DIM + dd] as i32 as i8;
                assert_eq!(
                    got, ki[dd],
                    "sK[{t}][{dd}]: device {got} vs mirror {}",
                    ki[dd]
                );
            }
        }
        Ok(())
    }

    /// Adaptive per-band window formats: mixed Q8_0/Q4_0/FP8 bands with
    /// non-unit outer scales across the sealed chunks (writer stays FP8), the
    /// kernel dispatching on the KvHead format tags, gated BIT-EXACT against
    /// the mirror (whose stored values come from the same codec that authored
    /// the arena bytes). Live RoPE + split-KV so the full pipeline runs over
    /// the dispatched reads.
    #[test]
    #[ignore]
    fn mirror_bit_exact_mixed_band_formats() -> Result<()> {
        let case = Case {
            n_win: 80,
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 2,
            zero_rope: false,
        };
        // Chunks: [0..32) [32..64) sealed — mixed formats; [64..80)+incoming
        // is the writer — FP8 (the scatters write FP8 only).
        let q8 = |outer| BandSpec { fmt: 7, outer };
        let q4 = |outer| BandSpec { fmt: 15, outer };
        let t8 = |a: [BandSpec; 8]| -> [BandSpec; N_BANDS] { std::array::from_fn(|i| a[i % 8]) };
        let fmts = vec![
            t8([
                q8(1.5),
                q4(1.0),
                BandSpec::default(),
                q8(1.0),
                q8(1.5),
                q4(1.0),
                BandSpec::default(),
                q8(1.0),
            ]),
            t8([
                q4(2.0),
                q8(1.0),
                q8(0.5),
                q4(1.0),
                q4(2.0),
                q8(1.0),
                q8(0.5),
                q4(1.0),
            ]),
            [BandSpec::default(); N_BANDS],
        ];
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let inp = gen_inputs(&case, 0xA5A5);
        let kernel = run_kernel_fmt(&case, &inp, softmax_scale, Some(&fmts))?;
        let mirror_v: Vec<f32> = mirror_fmt(&case, &inp, softmax_scale, Some(&fmts))
            .into_iter()
            .flat_map(|h| h.into_iter())
            .collect();
        let mirror_v = bf16_round_host(&mirror_v);
        // Attention-OUTPUT gate: ≤1 bf16 code per cell (the fast-math dequant
        // `/outer` + int8-QK hot seam), ≥99.9% exactly equal.
        let n_bad = check_within_one_code(&kernel, &mirror_v);
        assert!(
            n_bad * 1000 <= kernel.len(),
            "mixed-format: {n_bad}/{} cells off (> 0.1%)",
            kernel.len()
        );
        Ok(())
    }

    /// EVERY band format under the real attention kernel: seven sealed
    /// chunks covering the full K-ladder union — the f16-scale family at
    /// assorted outers plus the INT8-scale and Q0-family formats at
    /// searched-style outers — decoded through `load_band_elem`'s per-band
    /// dispatch, gated BIT-EXACT against the mirror. Arena bytes are
    /// codec-authored on both sides, so any mismatch is a decode op-order
    /// divergence between `dequant_element_inline` and the codec's arms.
    #[test]
    #[ignore]
    fn mirror_bit_exact_all_band_formats() -> Result<()> {
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        // Phase 1 — ISOLATION: one sealed chunk per (fmt, outer), all four
        // bands in that format, so a divergence names its format. Values are
        // fp8-exact in roughly ±3.75; 0.25-0.5 outers play the searched
        // 1/amax role for formats whose outer IS the quantization scale.
        let combos: [(u8, f32); 21] = [
            (7, 0.5),
            (8, 1.0),
            (8, 1.5),
            (10, 1.5),
            (15, 2.0),
            (16, 1.0),
            (18, 1.0),
            (18, 2.0),
            (19, 1.0),
            (19, 0.5),
            (20, 1.0),
            (20, 1.5),
            (25, 0.25),
            (26, 0.25),
            (27, 0.25),
            (28, 0.25),
            (29, 0.25),
            (30, 0.25),
            (31, 0.25),
            (32, 0.25),
            (33, 0.25),
        ];
        let iso_case = Case {
            n_win: 64,
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 1,
            zero_rope: false,
        };
        for (i, &(fmt, outer)) in combos.iter().enumerate() {
            // Chunks: [0..32) and [32..64) sealed in the probed format; the
            // incoming token's fresh chunk is the FP8 writer.
            let fmts = vec![
                [BandSpec { fmt, outer }; N_BANDS],
                [BandSpec { fmt, outer }; N_BANDS],
                [BandSpec::default(); N_BANDS],
            ];
            let inp = gen_inputs(&iso_case, 0xBEE0 + i as u64);
            let kernel = run_kernel_fmt(&iso_case, &inp, softmax_scale, Some(&fmts))?;
            let mirror_v: Vec<f32> = mirror_fmt(&iso_case, &inp, softmax_scale, Some(&fmts))
                .into_iter()
                .flat_map(|h| h.into_iter())
                .collect();
            let mirror_v = bf16_round_host(&mirror_v);
            // The int8 window staging is bit-exact — kernel `__fdiv_rn` /
            // `__frcp_rn` match the mirror's IEEE ops even under
            // `--use_fast_math`. The only residual is the dequant `/outer`
            // divide in the block converters, which the attention kernel runs
            // under fast-math (an ~1-ulp approximate reciprocal on the hot
            // decode path — precise division there measured ~8% slower and is
            // not worth it): a handful of cells/32768 land one bf16 code off
            // at a .5 boundary. ≥99.9% exactly equal, every cell ≤1 code.
            let n_bad = check_within_one_code(&kernel, &mirror_v);
            assert!(
                n_bad * 1000 <= kernel.len(),
                "fmt {fmt} outer {outer}: {n_bad}/{} cells off (> 0.1%)",
                kernel.len()
            );
        }

        // Phase 2 — the COMBINED case: seven sealed chunks mixing the whole
        // ladder union, decoded in one window.
        let case = Case {
            n_win: 240,
            window_size: 256,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 2,
            zero_rope: false,
        };
        let b = |fmt, outer| BandSpec { fmt, outer };
        // Tile an 8-format pattern across the 16 bands (bands 8-15 repeat 0-7).
        let t8 = |a: [BandSpec; 8]| -> [BandSpec; N_BANDS] { std::array::from_fn(|i| a[i % 8]) };
        let fmts = vec![
            t8([
                b(8, 1.0),
                b(18, 1.0),
                b(19, 1.0),
                b(20, 1.0),
                b(8, 1.0),
                b(18, 1.0),
                b(19, 1.0),
                b(20, 1.0),
            ]),
            t8([
                b(25, 0.25),
                b(26, 0.25),
                b(27, 0.25),
                b(29, 0.25),
                b(25, 0.25),
                b(26, 0.25),
                b(27, 0.25),
                b(29, 0.25),
            ]),
            t8([
                b(28, 0.25),
                b(30, 0.25),
                b(31, 0.25),
                b(32, 0.25),
                b(28, 0.25),
                b(30, 0.25),
                b(31, 0.25),
                b(32, 0.25),
            ]),
            t8([
                b(33, 0.25),
                b(25, 0.5),
                b(27, 0.5),
                b(32, 0.5),
                b(28, 0.3),
                b(26, 0.3),
                b(30, 0.5),
                b(31, 0.5),
            ]),
            t8([
                b(10, 1.5),
                b(16, 1.0),
                b(15, 2.0),
                b(7, 0.5),
                b(19, 0.5),
                b(20, 1.5),
                b(18, 2.0),
                b(8, 1.5),
            ]),
            t8([
                b(19, 0.5),
                b(20, 1.5),
                b(18, 2.0),
                b(8, 1.5),
                b(10, 1.5),
                b(16, 1.0),
                b(15, 2.0),
                b(7, 0.5),
            ]),
            t8([
                b(28, 0.3),
                b(26, 0.3),
                b(30, 0.5),
                b(31, 0.5),
                b(33, 0.25),
                b(25, 0.5),
                b(27, 0.5),
                b(32, 0.5),
            ]),
            [BandSpec::default(); N_BANDS],
        ];
        let inp = gen_inputs(&case, 0xBEEF);
        let kernel = run_kernel_fmt(&case, &inp, softmax_scale, Some(&fmts))?;
        let mirror_v: Vec<f32> = mirror_fmt(&case, &inp, softmax_scale, Some(&fmts))
            .into_iter()
            .flat_map(|h| h.into_iter())
            .collect();
        let mirror_v = bf16_round_host(&mirror_v);
        // ≤1-code / ≥99.9%, as the isolation loop (residual dequant seam).
        let n_bad = check_within_one_code(&kernel, &mirror_v);
        assert!(
            n_bad * 1000 <= kernel.len(),
            "all-format: {n_bad}/{} cells off (> 0.1%)",
            kernel.len()
        );
        Ok(())
    }

    /// Stage-level probe: the kernel's dequantized window staging (`kv_f`
    /// from the stage dump) must match the codec's stored values ELEMENT BY
    /// ELEMENT on the non-rope dims — pinning any format divergence to the
    /// dequant itself rather than the downstream int8/attention pipeline.
    #[test]
    #[ignore]
    fn latent_window_stage_matches_codec() -> Result<()> {
        let case = Case {
            n_win: 64,
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 1,
            zero_rope: false,
        };
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let dev = Device::new_cuda(0)?;
        // Seeds match the all-format isolation loop (0xBEE0 + combo index)
        // so a staging divergence reproduces the exact failing configuration.
        for &(fmt, outer, seed) in &[
            (27u8, 0.25f32, 0xBEEEu64),
            (25, 0.25, 0xBEEC),
            (33, 0.25, 0xBEF4),
            (20, 1.5, 0xBEEB),
        ] {
            let spec = BandSpec { fmt, outer };
            let specs = vec![
                [spec; N_BANDS],
                [spec; N_BANDS],
                [BandSpec::default(); N_BANDS],
            ];
            let inp = gen_inputs(&case, seed);
            let slots = SyntheticSlots::build_mapped(
                &dev,
                std::slice::from_ref(&inp.window),
                &[0usize],
                std::slice::from_ref(&specs),
                std::slice::from_ref(&vec![identity_pal_map(); 3]),
            )?;
            let qf: Vec<f32> = inp.q.iter().flat_map(|h| h.iter().copied()).collect();
            let q = Tensor::from_vec(qf, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let kv_new = Tensor::from_vec(inp.kv_new.to_vec(), (1, HEAD_DIM), &dev)?
                .to_dtype(DType::BF16)?;
            let compf: Vec<f32> = inp.comp.iter().flat_map(|c| c.iter().copied()).collect();
            let comp = Tensor::from_vec(compf, (case.g_total, HEAD_DIM), &dev)?;
            let comp_pos = Tensor::from_vec(inp.comp_pos.clone(), case.g_total, &dev)?;
            let idx: Vec<u32> = case.selected.iter().map(|&g| g as u32).collect();
            let comp_idx = Tensor::from_vec(idx, (1, case.selected.len()), &dev)?;
            let comp_cnt = Tensor::from_vec(vec![case.selected.len() as u32], 1, &dev)?;
            let sinks = Tensor::from_vec(inp.sinks.clone(), H, &dev)?;
            let freqs = Tensor::from_vec(inp.freqs.clone(), ROPE_DIM / 2, &dev)?;
            let rope_tab = build_rope_table(&freqs)?;
            let ws = LatentWorkspace::build(&dev)?;
            let dbg = Tensor::zeros(DBG_LEN, DType::F32, &dev)?;
            let _ = paged_latent_decode(
                &q,
                &slots.headers,
                &kv_new,
                &CorpusCache::build(&comp, &comp_pos)?,
                &comp_idx,
                &comp_cnt,
                &Tensor::from_vec(vec![case.n_win as u32], 1, &dev)?,
                &sinks,
                &rope_tab,
                &ws,
                softmax_scale,
                case.window_size,
                case.num_splits,
                Some(&dbg),
            )?;
            let d = dbg.to_vec1::<f32>()?;
            // The dump covers the first 16 window tokens (chunk 0). Expected
            // stored values from the codec, identity layout.
            let toks: Vec<[f32; HEAD_DIM]> = inp.window[..CHUNK].to_vec();
            let (_, dec) = {
                let mut full = vec![[0f32; HEAD_DIM]; CHUNK];
                for p in 0..N_BANDS {
                    let band_toks: Vec<[f32; SUB_DIM]> = toks
                        .iter()
                        .map(|tok| std::array::from_fn(|r| tok[p * SUB_DIM + r]))
                        .collect();
                    let (_, bdec) = band_chunk_roundtrip(spec, &band_toks)?;
                    for (t, row) in bdec.iter().enumerate() {
                        for (r, &v) in row.iter().enumerate() {
                            full[t][p * SUB_DIM + r] = v;
                        }
                    }
                }
                ((), full)
            };
            // Pre-rope stored values (non-rope dims are rope-invariant).
            let mut bad = 0usize;
            let mut first = None;
            for t in 0..8 {
                for dim in 0..(HEAD_DIM - ROPE_DIM) {
                    let got = d[DBG_KVF + t * HEAD_DIM + dim];
                    let want = half::bf16::from_f32(dec[t][dim]).to_f32();
                    if got.to_bits() != want.to_bits() {
                        bad += 1;
                        if first.is_none() {
                            first = Some((t, dim, got, want));
                        }
                    }
                }
            }
            assert_eq!(
                bad, 0,
                "fmt {fmt} outer {outer}: {bad} staged elements differ; first {first:?}"
            );
            // Int8 staging from the roped f32 REGISTERS (kernel requants from
            // full precision, before the bf16 store). scaleK / sK inherit the
            // dequant `/outer` fast-math seam (a ≤1-ulp value flip can move a
            // band's max, so `mx/127` and the int8 requant shift by one step):
            // scaleK within 2 ulp, sK codes within ±1, and only on a handful
            // of cells (≥99.9% exact). The `kv_f` bf16 store above is strict —
            // bf16 rounding absorbs the ulp.
            let (mut bad_s, mut bad_q) = (0usize, 0usize);
            for t in 0..8 {
                let mut l = dec[t];
                rope_vec(&mut l, t as u32, &inp.freqs);
                let (ki, ks) = quant_bands(&l);
                for p in 0..N_BANDS {
                    let got = d[DBG_SCALEK + t * N_BANDS + p];
                    let ulp = (got.to_bits() as i64 - ks[p].to_bits() as i64).abs();
                    if ulp > 2 {
                        bad_s += 1;
                    }
                }
                for dim in 0..HEAD_DIM {
                    let got = d[DBG_SK + t * HEAD_DIM + dim] as i32;
                    if (got - ki[dim] as i32).abs() > 1 {
                        bad_q += 1;
                    }
                }
            }
            assert!(
                bad_s == 0 && bad_q == 0,
                "fmt {fmt} outer {outer}: {bad_s} scaleK (>2ulp) + {bad_q} sK (>±1) diverge"
            );
        }
        Ok(())
    }

    /// Reference FP8 window recovery: the nope span stored as **FP8 E4M3 with a
    /// per-64 ue8m0 (power-of-two) scale** must round-trip through the kernel's
    /// symmetric `store_band_elem` (`E4M3(v·outer)`) → `load_band_elem`
    /// (`decode / outer`) exactly, and the rope tail through BF16. This pins the
    /// symmetric-scale contract the reference relies on: a non-unit per-band
    /// `outer` in the KvHead `k_scale` slot recovers the source. The scale is a
    /// pure power of two (as ue8m0 requires) and the inputs are FP8-exact, so
    /// the recovery is BIT-exact after the bf16 kv_f staging store.
    #[test]
    #[ignore]
    fn latent_window_fp8_scale_round_trip() -> Result<()> {
        let case = Case {
            n_win: 40, // chunk 0 sealed (32 tok), chunk 1 the writer (8 tok)
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 1,
            zero_rope: false,
        };
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let dev = Device::new_cuda(0)?;
        // Sealed chunk 0: FP8 nope+rope at a per-64 power-of-two scale (ue8m0);
        // the writer chunk (last) stays the unit-scale FP8 default that the
        // fused scatter targets. `outer` is the decoder-divide scale = 1/scale.
        for &outer in &[0.5f32, 2.0f32] {
            let sealed = BandSpec { fmt: 34, outer };
            let specs = vec![[sealed; N_BANDS], [BandSpec::default(); N_BANDS]];
            let inp = gen_inputs(&case, 0xF80 + outer.to_bits() as u64);
            let slots = SyntheticSlots::build_mapped(
                &dev,
                std::slice::from_ref(&inp.window),
                &[0usize],
                std::slice::from_ref(&specs),
                std::slice::from_ref(&vec![identity_pal_map(); 2]),
            )?;
            let qf: Vec<f32> = inp.q.iter().flat_map(|h| h.iter().copied()).collect();
            let q = Tensor::from_vec(qf, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let kv_new = Tensor::from_vec(inp.kv_new.to_vec(), (1, HEAD_DIM), &dev)?
                .to_dtype(DType::BF16)?;
            let compf: Vec<f32> = inp.comp.iter().flat_map(|c| c.iter().copied()).collect();
            let comp = Tensor::from_vec(compf, (case.g_total, HEAD_DIM), &dev)?;
            let comp_pos = Tensor::from_vec(inp.comp_pos.clone(), case.g_total, &dev)?;
            let idx: Vec<u32> = case.selected.iter().map(|&g| g as u32).collect();
            let comp_idx = Tensor::from_vec(idx, (1, case.selected.len()), &dev)?;
            let comp_cnt = Tensor::from_vec(vec![case.selected.len() as u32], 1, &dev)?;
            let sinks = Tensor::from_vec(inp.sinks.clone(), H, &dev)?;
            let freqs = Tensor::from_vec(inp.freqs.clone(), ROPE_DIM / 2, &dev)?;
            let rope_tab = build_rope_table(&freqs)?;
            let ws = LatentWorkspace::build(&dev)?;
            let dbg = Tensor::zeros(DBG_LEN, DType::F32, &dev)?;
            let _ = paged_latent_decode(
                &q,
                &slots.headers,
                &kv_new,
                &CorpusCache::build(&comp, &comp_pos)?,
                &comp_idx,
                &comp_cnt,
                &Tensor::from_vec(vec![case.n_win as u32], 1, &dev)?,
                &sinks,
                &rope_tab,
                &ws,
                softmax_scale,
                case.window_size,
                case.num_splits,
                Some(&dbg),
            )?;
            let d = dbg.to_vec1::<f32>()?;
            // Codec recovery of chunk 0's stored values (the mirror the kernel
            // dequant must match). The dump's kv_f holds bf16(load_band_elem).
            let toks: Vec<[f32; HEAD_DIM]> = inp.window[..CHUNK].to_vec();
            let mut dec = vec![[0f32; HEAD_DIM]; CHUNK];
            for p in 0..N_BANDS {
                let band_toks: Vec<[f32; SUB_DIM]> = toks
                    .iter()
                    .map(|tok| std::array::from_fn(|r| tok[p * SUB_DIM + r]))
                    .collect();
                let (_, bdec) = band_chunk_roundtrip(sealed, &band_toks)?;
                for (t, row) in bdec.iter().enumerate() {
                    for (r, &v) in row.iter().enumerate() {
                        dec[t][p * SUB_DIM + r] = v;
                    }
                }
            }
            // Non-rope dims are rope-invariant: assert the FP8-with-scale
            // recovery is bit-exact through the bf16 kv_f store.
            for t in 0..8 {
                for dim in 0..(HEAD_DIM - ROPE_DIM) {
                    let got = d[DBG_KVF + t * HEAD_DIM + dim];
                    let want = half::bf16::from_f32(dec[t][dim]).to_f32();
                    assert_eq!(
                        got.to_bits(),
                        want.to_bits(),
                        "outer {outer}: kv_f[t{t} d{dim}] {got} != {want}"
                    );
                    // Power-of-two ue8m0 scale over FP8-exact inputs: exact.
                    assert_eq!(dec[t][dim].to_bits(), inp.window[t][dim].to_bits());
                }
            }
        }
        Ok(())
    }

    /// Stage-dump section offsets — mirror of latent_decode_kernel's DBG_*
    /// layout, derived from N_BANDS so the read offsets track the band count.
    /// (scaleQ/sQ are always 16-head-sized; scaleK/sK/kv_f are KEYS_TILE-sized.)
    pub(in super::super) const DBG_KEYS: usize = 8;

    // KEYS_TILE
    pub(in super::super) const DBG_SCALEQ: usize = 0;

    pub(in super::super) const DBG_SQ: usize = DBG_SCALEQ + 16 * N_BANDS;

    pub(in super::super) const DBG_SCALEK: usize = DBG_SQ + 16 * HEAD_DIM;

    pub(in super::super) const DBG_SK: usize = DBG_SCALEK + DBG_KEYS * N_BANDS;

    pub(in super::super) const DBG_KVF: usize = DBG_SK + DBG_KEYS * HEAD_DIM;

    pub(in super::super) const DBG_LOGITS: usize = DBG_KVF + DBG_KEYS * HEAD_DIM;

    pub(in super::super) const DBG_LEN: usize = DBG_LOGITS + 16 * DBG_KEYS;
}
