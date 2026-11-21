//! Kernel tests for the prefill conv, the chunked scan and verify-width spans,
//! against the sequential rule.
//!
//! The scan is checked against the CPU reference at the real head width. The
//! triangular solve is checked against the explicit inverse it replaced, since
//! the two differ in code though not in mathematics. Spans batched in one launch
//! must equal each span alone, bit for bit. A sequence segmented across calls
//! must match one call over the whole of it, within numerical closeness, which is
//! the property that turn sealing and forks rely on.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::mix::{causal_conv1d, delta_recurrence, l2_norm};

    use super::*;

    use candle::{DType, Device};

    use super::super::tests_support::tests::{
        conv_epilogue_ref, f32_bits, lcg_tensor, max_diff, span_table, FusedCase, TestSpan,
    };

    #[path = "../../tests/prefill_golden.rs"]
    pub(in super::super) mod prefill_golden;

    /// The kernel must match the sequential CPU reference token by token,
    /// including the carried state, at realistic dims (d = 128, the published
    /// family's DeltaNet head width).
    /// **The solve must equal the explicit inverse it replaced.**
    ///
    /// `solve_unit_lower` is the chunked scan's hot path and it swaps a
    /// recursive block inversion followed by two matmuls for one `trsm`. The two
    /// are the same mathematics and *not* the same code, and the swap crosses a
    /// row-major/column-major boundary where a wrong `side` or `uplo` produces a
    /// plausible matrix rather than an error — so it is pinned against the
    /// reference form directly.
    ///
    /// A width that is not a multiple of anything convenient (37) and a
    /// right-hand side wider than the matrix, so a transposed dimension cannot
    /// go unnoticed.
    #[test]
    fn solve_matches_the_explicit_inverse() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        use super::super::super::mix::unit_lower_inverse_for_test;

        let (h, c, n) = (3usize, 37usize, 53usize);
        // Strictly lower triangular, with entries of a realistic size — the
        // scan's `A` is `(βk·k)·D` with `β ≤ 1` and `D ≤ 1`.
        let mut vals = vec![0f32; h * c * c];
        let raw: Vec<f32> = lcg_tensor(&[h, c, c], 21, &Device::Cpu)
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for head in 0..h {
            for i in 0..c {
                for j in 0..i {
                    vals[(head * c + i) * c + j] = raw[(head * c + i) * c + j];
                }
            }
        }
        let a = Tensor::from_vec(vals, (h, c, c), &gpu).unwrap();
        let rhs = lcg_tensor(&[h, c, n], 22, &gpu);

        let want = unit_lower_inverse_for_test(&a)
            .unwrap()
            .matmul(&rhs)
            .unwrap();
        let got = solve_unit_lower(&a, rhs.clone()).unwrap();

        let d = max_diff(&got, &want);
        let scale = want
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        println!("solve vs explicit inverse: max abs {d:.3e} over scale {scale:.3e}");
        assert!(
            d / scale.max(1e-6) < 1e-4,
            "triangular solve disagrees with the explicit inverse: {d} over {scale}"
        );
    }

    /// **The chunked scan, on the device, against the sequential rule.**
    ///
    /// `delta_net`'s own scan tests all run on the CPU, where
    /// [`super::super::mix::solve_pseudo_values`] takes the explicit-inverse
    /// fallback — so none of them touches the `trsm` path, and the first thing
    /// that path got wrong (a strided `narrow` handed to `matmul`) reached a
    /// gate run instead of a unit test. This is that coverage: the same
    /// property those tests assert, evaluated where the solve actually runs.
    #[test]
    fn chunked_scan_matches_sequential_on_cuda() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        use super::super::super::mix::{delta_chunked, delta_recurrence};

        // Two chunks plus a ragged tail at width 8, so the tail path and the
        // chunk-to-chunk state carry both run.
        let (t, h, d) = (21usize, 3usize, 8usize);
        let q = l2_norm(&lcg_tensor(&[t, h, d], 31, &gpu), 1e-6).unwrap();
        let k = l2_norm(&lcg_tensor(&[t, h, d], 32, &gpu), 1e-6).unwrap();
        let v = lcg_tensor(&[t, h, d], 33, &gpu);
        let g = lcg_tensor(&[t, h], 34, &gpu).abs().unwrap().neg().unwrap();
        let beta = candle_nn::ops::sigmoid(&lcg_tensor(&[t, h], 35, &gpu)).unwrap();
        let s0 = Tensor::zeros((h, d, d), DType::F32, &gpu).unwrap();

        let (o_ref, s_ref) = delta_recurrence(s0.copy().unwrap(), &q, &k, &v, &g, &beta).unwrap();
        for chunk in [8usize, 21, 64] {
            let mut s = s0.copy().unwrap();
            let o = delta_chunked(&mut s, &q, &k, &v, &g, &beta, chunk).unwrap();
            assert!(
                max_diff(&o, &o_ref) < 3e-4,
                "chunk={chunk} outputs diverged: {}",
                max_diff(&o, &o_ref)
            );
            assert!(
                max_diff(&s, &s_ref) < 3e-4,
                "chunk={chunk} state diverged: {}",
                max_diff(&s, &s_ref)
            );
        }
    }

    /// The token-parallel prefill conv against [`causal_conv1d`] + the
    /// op-form epilogue, including a carried tail from a previous segment, a
    /// segment shorter than the tail, and the advanced tail after each call.
    #[test]
    fn conv_prefill_matches_the_segment_reference() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        let (c, kw, qkc) = (512usize, 4usize, 256usize);
        let eps = 1e-6f64;
        let kern = lcg_tensor(&[c, kw], 41, &cpu);
        let kern_g = kern.to_device(&gpu).unwrap().contiguous().unwrap();

        // Segments 9, then 2 (shorter than kw−1, so the new tail mixes old
        // tail and fresh tokens), then 5 — one running CPU oracle throughout.
        let mut tail_cpu = Tensor::zeros((c, kw - 1), DType::F32, &cpu).unwrap();
        // Two buffers, exchanged after each segment — the store's live state and
        // prefill scratch, in the smallest form that still exercises the
        // kernel's contract that the entering and advanced tails are separate
        // allocations.
        let mut tail_gpu = Tensor::zeros((c, kw - 1), DType::F32, &gpu).unwrap();
        let mut tail_gpu_out = Tensor::zeros((c, kw - 1), DType::F32, &gpu).unwrap();
        for (seg, seed) in [(9usize, 42u64), (2, 43), (5, 44)] {
            let x_cm = lcg_tensor(&[c, seg], seed, &cpu); // [C, seg]
            let (y_raw, tail_next) = causal_conv1d(&x_cm, &kern, &tail_cpu).unwrap();
            let y_ref = conv_epilogue_ref(&y_raw, qkc, eps);
            tail_cpu = tail_next;

            // Kernel input is token-major; the output lands in the caller's
            // whole-wave buffer at the span's rows (here: the whole of it).
            let x_tok = x_cm
                .t()
                .unwrap()
                .contiguous()
                .unwrap()
                .to_device(&gpu)
                .unwrap();
            let conved = Tensor::zeros((seg, c), DType::F32, &gpu).unwrap();
            let tbl = span_table(&[TestSpan {
                tail: &tail_gpu,
                tail_out: &tail_gpu_out,
                state: &tail_gpu,
                state_out: &tail_gpu_out,
                start: 0,
                len: seg,
            }]);
            delta_net_conv_prefill(&x_tok, &kern_g, &tbl, qkc, eps as f32, &conved).unwrap();
            std::mem::swap(&mut tail_gpu, &mut tail_gpu_out);

            let y_cm = conved
                .t()
                .unwrap()
                .contiguous()
                .unwrap()
                .to_device(&cpu)
                .unwrap();
            let yd = max_diff(&y_ref, &y_cm);
            let td = max_diff(&tail_cpu, &tail_gpu.to_device(&cpu).unwrap());
            assert!(yd <= 1e-5, "segment {seg}: conv outputs diverged: {yd}");
            assert!(td <= 1e-6, "segment {seg}: advanced tail diverged: {td}");
        }
    }

    /// **The fused prefill scan against the sequential rule**, at the real
    /// head width (d = 128) with several chunks, a ragged tail, and h_k < h_v
    /// so the in-kernel GQA tiling is exercised — output and carried state
    /// both, everything (gates, scale, tiling) derived in-kernel.
    #[test]
    fn prefill_scan_matches_the_sequential_reference() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        // 2 full chunks + a 22-token tail at the kernel's width of 64.
        let (t, h_k, h_v, d) = (150usize, 2usize, 4usize, 128usize);
        let case = FusedCase::build(t, h_k, h_v, 51, &cpu, &gpu);
        let s0 = lcg_tensor(&[h_v, d, d], 56, &cpu);

        let (o_ref, s_ref) = delta_recurrence(
            s0.copy().unwrap(),
            &case.q_ref,
            &case.k_ref,
            &case.v_ref,
            &case.g_ref,
            &case.b_ref,
        )
        .unwrap();

        let s_gpu = s0.to_device(&gpu).unwrap().contiguous().unwrap();
        let o = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
        let tbl = span_table(&[TestSpan {
            tail: &s_gpu,
            tail_out: &s_gpu,
            state: &s_gpu,
            state_out: &s_gpu,
            start: 0,
            len: t,
        }]);
        delta_net_prefill_scan(&case.fused(&o), &tbl).unwrap();

        let o_gpu = o.reshape((t, h_v, d)).unwrap().to_device(&cpu).unwrap();
        let od = max_diff(&o_gpu, &o_ref);
        let sd = max_diff(&s_gpu.to_device(&cpu).unwrap(), &s_ref);
        println!("prefill scan vs sequential: o {od:.3e}, state {sd:.3e}");
        assert!(od < 3e-4, "outputs diverged from the sequential rule: {od}");
        assert!(sd < 3e-4, "state diverged from the sequential rule: {sd}");
    }

    /// A verify wave's spans — one to five tokens each, in one launch — match the
    /// sequential rule, output and state. These are the lengths where the intra
    /// pass sizes its row buffers to the span rather than the chunk, and where the
    /// A/kq grid's 4-wide tiles read rows past `c_len`: one- and two-token spans
    /// are the ones a buffer cut to the bare length would overrun.
    #[test]
    fn verify_width_spans_match_the_sequential_reference() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        let (h_k, h_v, d) = (2usize, 4usize, 128usize);
        for lens in [&[1usize][..], &[2, 1], &[3, 5, 1, 2]] {
            let t: usize = lens.iter().sum();
            let case = FusedCase::build(t, h_k, h_v, 61 + t as u64, &cpu, &gpu);
            let states: Vec<Tensor> = (0..lens.len())
                .map(|i| lcg_tensor(&[h_v, d, d], 700 + i as u64, &cpu))
                .collect();
            let gpu_states: Vec<Tensor> = states
                .iter()
                .map(|s| s.to_device(&gpu).unwrap().contiguous().unwrap())
                .collect();
            let mut start = 0usize;
            let rows: Vec<TestSpan<'_>> = lens
                .iter()
                .zip(&gpu_states)
                .map(|(&len, s)| {
                    let row = TestSpan {
                        tail: s,
                        tail_out: s,
                        state: s,
                        state_out: s,
                        start,
                        len,
                    };
                    start += len;
                    row
                })
                .collect();
            let o = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
            delta_net_prefill_scan(&case.fused(&o), &span_table(&rows)).unwrap();
            let o_gpu = o.reshape((t, h_v, d)).unwrap().to_device(&cpu).unwrap();

            let mut start = 0usize;
            for (i, &len) in lens.iter().enumerate() {
                let span = |x: &Tensor| x.narrow(0, start, len).unwrap().contiguous().unwrap();
                let (o_ref, s_ref) = delta_recurrence(
                    states[i].copy().unwrap(),
                    &span(&case.q_ref),
                    &span(&case.k_ref),
                    &span(&case.v_ref),
                    &span(&case.g_ref),
                    &span(&case.b_ref),
                )
                .unwrap();
                let od = max_diff(&span(&o_gpu), &o_ref);
                let sd = max_diff(&gpu_states[i].to_device(&cpu).unwrap(), &s_ref);
                assert!(od < 3e-4, "{lens:?} span {i}: output diverged by {od}");
                assert!(sd < 3e-4, "{lens:?} span {i}: state diverged by {sd}");
                start += len;
            }
        }
    }

    /// **Several sequences in one launch must equal each of them alone.**
    ///
    /// This is what the span table exists for, and the property it can break:
    /// the spans share a launch but not a state, so an index that reaches past
    /// its own span — the transients are one wave-wide allocation, and the
    /// packed buffers are rebased per span — silently mixes one sequence's rows
    /// into another's recurrence. Bit-identity is the assertion, not closeness:
    /// batching changes which blocks run together, never the arithmetic any one
    /// of them does, so a span's rows and its advanced state must come back the
    /// same to the bit as when it ran by itself.
    ///
    /// The spans are deliberately UNEQUAL in length (and neither a multiple of
    /// the 64-token chunk), because the launch is a rectangle over the widest of
    /// them: an equal-length case would never exercise the surplus-block guard.
    #[test]
    fn spans_batched_equal_spans_alone() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        let (h_k, h_v, d) = (2usize, 4usize, 128usize);
        let lens = [70usize, 33, 129];
        let t: usize = lens.iter().sum();
        let case = FusedCase::build(t, h_k, h_v, 77, &cpu, &gpu);

        let starts: Vec<usize> = lens
            .iter()
            .scan(0usize, |acc, &l| {
                let s = *acc;
                *acc += l;
                Some(s)
            })
            .collect();

        // Entering states that differ per span, so a span reading the wrong
        // one cannot coincidentally agree.
        let states: Vec<Tensor> = (0..lens.len())
            .map(|i| {
                lcg_tensor(&[h_v, d, d], 900 + i as u64, &cpu)
                    .to_device(&gpu)
                    .unwrap()
            })
            .collect();

        // Together: one table, one launch pair.
        let batched_states: Vec<Tensor> = states.iter().map(|s| s.copy().unwrap()).collect();
        let o_batched = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
        let rows: Vec<TestSpan<'_>> = (0..lens.len())
            .map(|i| TestSpan {
                tail: &batched_states[i],
                tail_out: &batched_states[i],
                state: &batched_states[i],
                state_out: &batched_states[i],
                start: starts[i],
                len: lens[i],
            })
            .collect();
        delta_net_prefill_scan(&case.fused(&o_batched), &span_table(&rows)).unwrap();

        // Alone: one table per span, into a separate output buffer.
        let o_alone = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
        let alone_states: Vec<Tensor> = states.iter().map(|s| s.copy().unwrap()).collect();
        for i in 0..lens.len() {
            let tbl = span_table(&[TestSpan {
                tail: &alone_states[i],
                tail_out: &alone_states[i],
                state: &alone_states[i],
                state_out: &alone_states[i],
                start: starts[i],
                len: lens[i],
            }]);
            delta_net_prefill_scan(&case.fused(&o_alone), &tbl).unwrap();
        }

        let b = o_batched.to_device(&cpu).unwrap().to_vec2::<f32>().unwrap();
        let a = o_alone.to_device(&cpu).unwrap().to_vec2::<f32>().unwrap();
        for (r, (br, ar)) in b.iter().zip(a.iter()).enumerate() {
            assert_eq!(
                br, ar,
                "row {r} differs between the batched and lone launches"
            );
        }
        for i in 0..lens.len() {
            let bs = batched_states[i]
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap();
            let as_ = alone_states[i]
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap();
            assert_eq!(
                bs.to_vec1::<f32>().unwrap(),
                as_.to_vec1::<f32>().unwrap(),
                "span {i}'s advanced state differs between the batched and lone launches"
            );
        }
    }

    /// The conv half of the same property: every span's tail advanced in one
    /// launch must equal each advanced alone.
    #[test]
    fn conv_spans_batched_equal_spans_alone() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        let (c, kw, qkc, eps) = (768usize, 4usize, 512usize, 1e-6f64);
        let lens = [5usize, 17, 2];
        let t: usize = lens.iter().sum();
        let starts: Vec<usize> = lens
            .iter()
            .scan(0usize, |acc, &l| {
                let s = *acc;
                *acc += l;
                Some(s)
            })
            .collect();

        let kern = lcg_tensor(&[c, kw], 51, &cpu).to_device(&gpu).unwrap();
        let qkv = lcg_tensor(&[t, c], 52, &cpu).to_device(&gpu).unwrap();
        let tails: Vec<Tensor> = (0..lens.len())
            .map(|i| {
                lcg_tensor(&[c, kw - 1], 60 + i as u64, &cpu)
                    .to_device(&gpu)
                    .unwrap()
            })
            .collect();

        let run = |outs: &[Tensor], batched: bool| -> Tensor {
            let conved = Tensor::zeros((t, c), DType::F32, &gpu).unwrap();
            let mk = |i: usize| TestSpan {
                tail: &tails[i],
                tail_out: &outs[i],
                state: &tails[i],
                state_out: &outs[i],
                start: starts[i],
                len: lens[i],
            };
            if batched {
                let rows: Vec<TestSpan<'_>> = (0..lens.len()).map(mk).collect();
                delta_net_conv_prefill(&qkv, &kern, &span_table(&rows), qkc, eps as f32, &conved)
                    .unwrap();
            } else {
                for i in 0..lens.len() {
                    delta_net_conv_prefill(
                        &qkv,
                        &kern,
                        &span_table(&[mk(i)]),
                        qkc,
                        eps as f32,
                        &conved,
                    )
                    .unwrap();
                }
            }
            conved
        };

        let outs_b: Vec<Tensor> = (0..lens.len())
            .map(|_| Tensor::zeros((c, kw - 1), DType::F32, &gpu).unwrap())
            .collect();
        let outs_a: Vec<Tensor> = (0..lens.len())
            .map(|_| Tensor::zeros((c, kw - 1), DType::F32, &gpu).unwrap())
            .collect();
        let cb = run(&outs_b, true).to_device(&cpu).unwrap();
        let ca = run(&outs_a, false).to_device(&cpu).unwrap();
        assert_eq!(
            cb.to_vec2::<f32>().unwrap(),
            ca.to_vec2::<f32>().unwrap(),
            "conv output differs between the batched and lone launches"
        );
        for i in 0..lens.len() {
            assert_eq!(
                outs_b[i].to_device(&cpu).unwrap().to_vec2::<f32>().unwrap(),
                outs_a[i].to_device(&cpu).unwrap().to_vec2::<f32>().unwrap(),
                "span {i}'s advanced conv tail differs"
            );
        }
    }

    /// Segmenting a sequence across calls must equal one call over the whole
    /// of it — the property turn sealing, forks and resume rest on, evaluated
    /// through the kernel (segment boundaries fall mid-chunk on purpose, and
    /// the segments are spans of one wave, exactly as the mixer issues them).
    #[test]
    fn prefill_scan_carries_state_across_segments() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        let (t, h_k, h_v, d) = (130usize, 2usize, 4usize, 128usize);
        let case = FusedCase::build(t, h_k, h_v, 61, &cpu, &gpu);

        let s_one = Tensor::zeros((h_v, d, d), DType::F32, &gpu).unwrap();
        let o_one = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
        let one = |s: &Tensor, start: usize, len: usize| {
            span_table(&[TestSpan {
                tail: s,
                tail_out: s,
                state: s,
                state_out: s,
                start,
                len,
            }])
        };
        delta_net_prefill_scan(&case.fused(&o_one), &one(&s_one, 0, t)).unwrap();

        // 70 + 60: neither boundary is a multiple of the 64-token chunk. Two
        // CALLS, not two spans of one table: the segments are the same sequence
        // and the second must enter with the state the first left, where spans
        // sharing a launch are independent and all enter with their own.
        let s_seg = Tensor::zeros((h_v, d, d), DType::F32, &gpu).unwrap();
        let o_seg = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
        let fused = case.fused(&o_seg);
        delta_net_prefill_scan(&fused, &one(&s_seg, 0, 70)).unwrap();
        delta_net_prefill_scan(&fused, &one(&s_seg, 70, 60)).unwrap();

        // The chunk grouping differs (the second call re-chunks from its own
        // origin), so this is numerical closeness, not bitwise identity.
        let od = max_diff(&o_one, &o_seg);
        let sd = max_diff(&s_one, &s_seg);
        println!("prefill scan segmented vs one-shot: o {od:.3e}, state {sd:.3e}");
        assert!(od < 3e-4, "segmented outputs diverged: {od}");
        assert!(sd < 3e-4, "segmented state diverged: {sd}");
    }
}
