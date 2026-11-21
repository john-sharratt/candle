//! Kernel tests for a wave whose spans take different kernels: verify blocks
//! through the short-span path and a prefill chunk through the conv and scan pair.
//!
//! Each span's rows, state and tail must match what it produces alone, bit for
//! bit, which shows that each span took its own path and read none of another's
//! rows. The norm gate's quantised operand is checked against the bytes of a
//! standalone quantise. The mixer's component timing is an ignored benchmark.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use candle::{DType, Device};

    use super::super::tests_support::tests::{
        f32_bits, lcg_tensor, max_diff, span_table, tile_bytes, FusedCase, TestSpan,
    };

    /// **Each span of a mixed wave takes its own kernel** (§10, test 6): a wave
    /// carrying two verify blocks beside a prefill chunk runs the blocks through
    /// the short-span kernel into their timelines and the chunk through the
    /// conv + scan pair into its scratch — and every span's output rows, state
    /// and conv tail are, bit for bit, what it produces running alone.
    ///
    /// The short spans' states are checked untouched before their fold, which
    /// is what says they took the timeline path at all; the long span's live
    /// state likewise, because its advance belongs in the scratch.
    #[test]
    fn a_mixed_wave_runs_each_span_through_its_own_kernel() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        use super::super::super::mix::{
            delta_net_mix_spans, finish_solo, solo_write, DeltaNetConstants, DeltaNetOut,
            DeltaNetProjections, SpanWrite,
        };
        let dims = DeltaNetDims {
            head_dim: 128,
            n_k_heads: 1,
            n_v_heads: 2,
            conv_kernel: 4,
        };
        let lens = [5usize, TIMELINE_ROWS + 12, 3];
        let total: usize = lens.iter().sum();
        let starts = [0usize, lens[0], lens[0] + lens[1]];
        let p = DeltaNetProjections {
            qkv: lcg_tensor(&[total, dims.conv_dim()], 11, &gpu),
            z: lcg_tensor(&[total, dims.value_dim()], 12, &gpu),
            beta_lin: lcg_tensor(&[total, dims.n_v_heads], 13, &gpu),
            alpha_lin: lcg_tensor(&[total, dims.n_v_heads], 14, &gpu),
        };
        let dt_bias = lcg_tensor(&[dims.n_v_heads], 15, &gpu);
        let a = lcg_tensor(&[dims.n_v_heads], 16, &gpu)
            .abs()
            .unwrap()
            .affine(1.0, 0.1)
            .unwrap()
            .neg()
            .unwrap();
        let conv = lcg_tensor(&[dims.conv_dim(), dims.conv_kernel], 17, &gpu);
        let norm = lcg_tensor(&[dims.head_dim], 18, &gpu).abs().unwrap();
        let c = DeltaNetConstants {
            dt_bias: &dt_bias,
            a: &a,
            conv: &conv,
            norm: &norm,
        };
        let entering = |i: usize| DeltaNetState {
            s: lcg_tensor(&[2, 128, 128], 40 + i as u64, &gpu),
            conv_tail: lcg_tensor(&[dims.conv_dim(), 3], 60 + i as u64, &gpu),
        };

        // Each span alone, over its own rows of the same projections.
        let mut alone_o = Vec::new();
        let mut alone_states = Vec::new();
        for i in 0..lens.len() {
            let rows = |x: &Tensor| {
                x.narrow(0, starts[i], lens[i])
                    .unwrap()
                    .contiguous()
                    .unwrap()
            };
            let ps = DeltaNetProjections {
                qkv: rows(&p.qkv),
                z: rows(&p.z),
                beta_lin: rows(&p.beta_lin),
                alpha_lin: rows(&p.alpha_lin),
            };
            let mut st = entering(i);
            let write = solo_write(&st, lens[i], &dims).unwrap();
            let mut seqs = [DeltaNetSeq {
                start: 0,
                len: lens[i],
                state: &mut st,
                write,
                hook: SpanHook::None,
            }];
            let o = delta_net_mix_spans(&ps, &c, &dims, &mut seqs, 1e-6, None, ZGate::Silu, None)
                .unwrap();
            let [s] = seqs;
            finish_solo(s.state, &s.write, s.len, &dims).unwrap();
            alone_o.push(o.to_device(&Device::Cpu).unwrap());
            alone_states.push(st);
        }

        // Together: one call, a timeline for each block and — as a store hands
        // a prefilling sequence — a scratch apart from the live state for the
        // chunk.
        let mut states: Vec<DeltaNetState> = (0..lens.len()).map(entering).collect();
        let scratch = DeltaNetOut {
            s: Tensor::zeros((2, 128, 128), DType::F32, &gpu).unwrap(),
            conv_tail: Tensor::zeros((dims.conv_dim(), 3), DType::F32, &gpu).unwrap(),
        };
        let writes: Vec<SpanWrite> = states
            .iter()
            .zip(lens)
            .map(|(s, l)| match l > TIMELINE_ROWS {
                true => SpanWrite::State(scratch.clone()),
                false => solo_write(s, l, &dims).unwrap(),
            })
            .collect();
        assert!(writes[0].timeline().is_some() && writes[2].timeline().is_some());
        assert!(writes[1].state().is_some());
        let mixed = {
            let mut seqs: Vec<DeltaNetSeq<'_>> = states
                .iter_mut()
                .zip(writes.iter())
                .enumerate()
                .map(|(i, (state, write))| DeltaNetSeq {
                    start: starts[i],
                    len: lens[i],
                    state,
                    write: write.clone(),
                    hook: SpanHook::None,
                })
                .collect();
            delta_net_mix_spans(&p, &c, &dims, &mut seqs, 1e-6, None, ZGate::Silu, None).unwrap()
        };
        for (i, st) in states.iter().enumerate() {
            assert_eq!(
                f32_bits(&st.s),
                f32_bits(&entering(i).s),
                "span {i}: the wave wrote the live state it read"
            );
        }
        // The blocks fold whole; the chunk's advance is the scratch, as a
        // commit would make it.
        for i in [0, 2] {
            finish_solo(&mut states[i], &writes[i], lens[i], &dims).unwrap();
        }
        states[1] = DeltaNetState {
            s: scratch.s.clone(),
            conv_tail: scratch.conv_tail.clone(),
        };
        let mixed = mixed.to_device(&Device::Cpu).unwrap();
        for i in 0..lens.len() {
            assert_eq!(
                f32_bits(&mixed.narrow(0, starts[i], lens[i]).unwrap()),
                f32_bits(&alone_o[i]),
                "span {i}: output rows"
            );
            assert_eq!(
                f32_bits(&states[i].s),
                f32_bits(&alone_states[i].s),
                "span {i}: state"
            );
            assert_eq!(
                f32_bits(&states[i].conv_tail),
                f32_bits(&alone_states[i].conv_tail),
                "span {i}: conv tail"
            );
        }
    }

    /// Component-level timing of the fused mixer kernels at the 9B's real
    /// geometry — the harness the end-to-end `dn:mix` span cannot be: it
    /// splits the per-layer cost into conv / intra / state / norm_gate (and
    /// the decode step), with no `profile_sync` overhead in the numbers.
    ///
    /// `#[ignore]` because it is a benchmark, not an assertion — it prints.
    /// Run alone on an idle GPU; this machine's run-to-run band is ~2×, so
    /// compare components within one run, not absolutes across runs.
    #[test]
    #[ignore = "GPU timing benchmark — prints per-kernel µs, asserts nothing"]
    fn fused_mixer_kernel_timing() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        // The 9B mixer geometry: 16 K heads, 32 V heads, d = 128,
        // conv_dim = (2·16 + 32)·128 = 8192, a 649-token gate prompt.
        let (t, h_k, h_v, d, kw) = (649usize, 16usize, 32usize, 128usize, 4usize);
        let conv_dim = (2 * h_k + h_v) * d;
        let case = FusedCase::build(t, h_k, h_v, 81, &cpu, &gpu);
        let qkv = lcg_tensor(&[t, conv_dim], 82, &gpu);
        let kern = lcg_tensor(&[conv_dim, kw], 83, &gpu);
        let tail = Tensor::zeros((conv_dim, kw - 1), DType::F32, &gpu).unwrap();
        let state = Tensor::zeros((h_v, d, d), DType::F32, &gpu).unwrap();
        let o = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
        let z = lcg_tensor(&[t, h_v * d], 84, &gpu);
        let gain = lcg_tensor(&[d], 85, &gpu);
        let fused = case.fused(&o);
        let qk_channels = 2 * h_k * d;

        let time = |label: &str, f: &mut dyn FnMut()| {
            for _ in 0..5 {
                f();
            }
            gpu.synchronize().unwrap();
            let reps = 50;
            let start = std::time::Instant::now();
            for _ in 0..reps {
                f();
            }
            gpu.synchronize().unwrap();
            let us = start.elapsed().as_secs_f64() * 1e6 / reps as f64;
            eprintln!("  {label:<26} {us:9.1} µs/call");
            us
        };

        eprintln!("--- fused mixer kernels, 9B geometry, T={t} ---");
        let conved = Tensor::zeros((t, conv_dim), DType::F32, &gpu).unwrap();
        let tail_out = Tensor::zeros((conv_dim, kw - 1), DType::F32, &gpu).unwrap();
        let whole = span_table(&[TestSpan {
            tail: &tail,
            tail_out: &tail_out,
            state: &state,
            state_out: &state,
            start: 0,
            len: t,
        }]);
        let c_us = time("conv_prefill (+epilogue)", &mut || {
            delta_net_conv_prefill(&qkv, &kern, &whole, qk_channels, 1e-6, &conved).unwrap();
        });
        let s_us = time("scan (intra + state)", &mut || {
            delta_net_prefill_scan(&fused, &whole).unwrap();
        });
        let g_us = time("norm_gate", &mut || {
            let _ = delta_net_norm_gate(&o, &z, &gain, d, 1e-6, super::super::super::ZGate::Silu)
                .unwrap();
        });
        eprintln!(
            "  {:<26} {:9.1} µs/layer (T={t} prefill)",
            "TOTAL",
            c_us + s_us + g_us
        );
        // Decode: batched over 4 sequences (the gate's multi-context shape),
        // one launch pair regardless of the count.
        let n_dec = 4usize;
        let dims = DeltaNetDims {
            head_dim: d,
            n_k_heads: h_k,
            n_v_heads: h_v,
            conv_kernel: kw,
        };
        let lay = TimelineLayout::of(&dims);
        let dec_states: Vec<DeltaNetState> = (0..n_dec)
            .map(|_| DeltaNetState::zeros(&dims, &gpu).unwrap())
            .collect();
        let dec_timelines: Vec<Tensor> = (0..n_dec).map(|_| lay.alloc(&gpu).unwrap()).collect();
        let mut ptrs: Vec<i64> = Vec::new();
        for st in &dec_states {
            ptrs.push(f32_ptr(&st.conv_tail, "tail").unwrap() as i64);
        }
        for st in &dec_states {
            ptrs.push(f32_ptr(&st.s, "state").unwrap() as i64);
        }
        for tl in &dec_timelines {
            ptrs.push(f32_ptr(tl, "timeline").unwrap() as i64);
        }
        let dec_table = DeltaNetLayerTable {
            ptrs: Tensor::from_vec(ptrs, (3, n_dec), &gpu).unwrap(),
            rows: Tensor::from_vec((0..n_dec as u32).collect::<Vec<_>>(), n_dec, &gpu).unwrap(),
        };
        let d_us = time("decode step (batch of 4)", &mut || {
            delta_net_decode_batch(&fused, &dec_table).unwrap();
        });
        let dc = time("conv_decode (batch of 4)", &mut || {
            delta_net_conv_decode(&qkv, &kern, &dec_table, &conved, qk_channels, 1e-6).unwrap();
        });
        let jobs: Vec<(&DeltaNetState, &Tensor, usize)> = dec_states
            .iter()
            .zip(&dec_timelines)
            .map(|(s, tl)| (s, tl, 1))
            .collect();
        let df = time("fold (4 × 1 row)", &mut || {
            delta_net_fold_states(&jobs, &dims).unwrap();
        });
        let jobs8: Vec<(&DeltaNetState, &Tensor, usize)> = dec_states
            .iter()
            .zip(&dec_timelines)
            .map(|(s, tl)| (s, tl, TIMELINE_ROWS))
            .collect();
        time("fold (4 × 8 rows)", &mut || {
            delta_net_fold_states(&jobs8, &dims).unwrap();
        });
        eprintln!(
            "  {:<26} {:9.1} µs/layer (4-seq decode)",
            "TOTAL",
            d_us + dc + df
        );
    }

    /// The fused epilogue against the op forms it replaced: per-head RMS norm
    /// with the gain, gated by the z-gate — BOTH instantiations (SiLU for the
    /// Qwen3.5 lineage, sigmoid for qwen4exp), because a kernel reused at a
    /// new configuration extends its harness to cover it (§0.4 rule 4).
    #[test]
    fn norm_gate_matches_the_op_epilogue() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        let (t, h_v, d) = (37usize, 4usize, 128usize);
        let eps = 1e-6f64;
        let o = lcg_tensor(&[t, h_v * d], 71, &cpu);
        let z = lcg_tensor(&[t, h_v * d], 72, &cpu);
        let gain = lcg_tensor(&[d], 73, &cpu);

        // The op-form reference, exactly as the fallback computes it.
        let o3 = o.reshape((t, h_v, d)).unwrap();
        let z3 = z.reshape((t, h_v, d)).unwrap();
        let ms = o3.sqr().unwrap().mean_keepdim(candle::D::Minus1).unwrap();
        let denom = (ms + eps).unwrap().sqrt().unwrap();
        let normed = o3
            .broadcast_div(&denom)
            .unwrap()
            .broadcast_mul(&gain)
            .unwrap();
        let to = |x: &Tensor| x.to_device(&gpu).unwrap().contiguous().unwrap();

        for zgate in [
            super::super::super::ZGate::Silu,
            super::super::super::ZGate::Sigmoid,
        ] {
            let sig_z = candle_nn::ops::sigmoid(&z3).unwrap();
            let gate = match zgate {
                super::super::super::ZGate::Silu => z3.broadcast_mul(&sig_z).unwrap(),
                super::super::super::ZGate::Sigmoid => sig_z,
            };
            let want = normed.mul(&gate).unwrap().reshape((t, h_v * d)).unwrap();
            let got = delta_net_norm_gate(&to(&o), &to(&z), &to(&gain), d, eps as f32, zgate)
                .unwrap()
                .to_device(&cpu)
                .unwrap();
            let diff = max_diff(&got, &want);
            assert!(
                diff <= 2e-5,
                "epilogue diverged from the op forms under {zgate:?}: {diff}"
            );
        }
    }

    /// The operand the epilogue emits is the standalone quantize of the output
    /// the F32 epilogue stores, byte for byte — under both gates and both Σx
    /// conventions, at a decode width, a verify width and a prefill-like one,
    /// at the released 48 heads (48 tiles a row, whole super-blocks of 8) and at
    /// 5 heads, whose 5-tile rows leave every super-block but the first partial.
    #[test]
    fn the_norm_gate_emits_the_quantize_bytes() {
        use candle::quantized::cuda::{to_dynamic, DynamicTensor};
        use candle::quantized::Int8Mode;
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let Device::Cuda(dev) = &gpu else {
            unreachable!("built as CUDA")
        };
        let cpu = Device::Cpu;
        let d = 128usize;
        let mode = Int8Mode::auto(&gpu);
        for h_v in [48usize, 5] {
            for t in [1usize, 5, 37] {
                let seed = 81 + (h_v * 100 + t) as u64;
                let o = lcg_tensor(&[t, h_v * d], seed, &cpu)
                    .to_device(&gpu)
                    .unwrap();
                let z = lcg_tensor(&[t, h_v * d], seed + 1, &cpu)
                    .to_device(&gpu)
                    .unwrap();
                let gain = lcg_tensor(&[d], 83, &cpu).to_device(&gpu).unwrap();
                for zgate in [ZGate::Silu, ZGate::Sigmoid] {
                    for ss in [SumScale::Raw, SumScale::ByAmax] {
                        let plain = delta_net_norm_gate(&o, &z, &gain, d, 1e-6, zgate).unwrap();
                        let op = delta_net_norm_gate_q8(&o, &z, &gain, d, 1e-6, zgate, ss).unwrap();
                        let acts = to_dynamic(&plain, mode, dev, ss).unwrap();
                        let DynamicTensor::Int8(want) = acts.as_dynamic() else {
                            panic!("an int8 mode quantizes")
                        };
                        assert_eq!(
                            tile_bytes(&op, dev),
                            tile_bytes(want, dev),
                            "h_v={h_v} t={t} {zgate:?} {ss:?}: operand bytes"
                        );
                    }
                }
            }
        }
    }
}
