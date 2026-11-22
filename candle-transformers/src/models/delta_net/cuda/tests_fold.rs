//! Kernel tests that the fold of a short span's timeline matches the three-launch
//! scan bit for bit, at every prefix.
//!
//! Each span length from 2 to 8 is folded at every kept prefix and compared with
//! the conv and scan pair run over those rows alone. The state, the tail and the
//! outputs must match bit for bit. A replay takes the captured span's state and
//! none of its outputs, and a span longer than the short-span limit is refused.

use super::*;

#[cfg(test)]
pub(super) mod tests {

    use super::*;

    use candle::{DType, Device};

    use super::super::tests_support::tests::{
        f32_bits, lcg_tensor, max_diff, short_table, span_table, ShortCase, TestSpan,
    };

    /// **The fold of a short span's timeline is the three-launch scan over that
    /// many rows, to the bit, at every prefix** (`docs/deltanet_timeline.md`
    /// §10, test 1).
    ///
    /// A short span reads its state and writes a timeline; the fold of its
    /// first `m` rows must be exactly the state and conv tail the conv + scan
    /// pair writes for a span of those `m` rows alone — the arithmetic a
    /// speculative rewind is locked to. One wave carries a span of every short
    /// length, 2..=8, from random entering states and tails; for every span and
    /// every `m`, the pair runs over its first `m` rows into poisoned buffers,
    /// and the fold runs over a copy of the entering state. `S`, the conv tail
    /// and the `o` rows of the prefix must match bit for bit — the `o` rows
    /// because a row's output reads only the rows before it, whatever the span's
    /// length.
    #[test]
    fn a_short_span_timeline_folds_to_the_three_launches_bit_for_bit() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let d = ShortCase::D;
        let eps = 1e-6f32;
        let lens: Vec<usize> = (2..=DELTA_NET_SHORT_SPAN_ROWS).collect();
        let t: usize = lens.iter().sum();
        let seed = 300u64;
        let c = ShortCase::build(t, seed, &gpu);
        let (h_v, conv_dim) = (c.h_v, c.conv_dim);
        let dims = DeltaNetDims {
            head_dim: d,
            n_k_heads: c.h_k,
            n_v_heads: h_v,
            conv_kernel: DELTA_NET_SHORT_SPAN_CONV,
        };
        let lay = TimelineLayout::of(&dims);
        let starts: Vec<usize> = lens
            .iter()
            .scan(0usize, |acc, &l| {
                let s = *acc;
                *acc += l;
                Some(s)
            })
            .collect();
        let entering: Vec<Tensor> = (0..lens.len())
            .map(|i| lcg_tensor(&[h_v, d, d], seed + 10 + i as u64, &gpu))
            .collect();
        let tails: Vec<Tensor> = (0..lens.len())
            .map(|i| lcg_tensor(&[conv_dim, 3], seed + 30 + i as u64, &gpu))
            .collect();
        let poison = |shape: &[usize], s: u64| lcg_tensor(shape, s, &gpu).affine(1e3, 0.0).unwrap();

        // The short path: every span in one launch, each into its own timeline.
        // The states and tails it reads are the entering ones — it must leave
        // them as they were.
        let states: Vec<Tensor> = entering.iter().map(|s| s.copy().unwrap()).collect();
        let timelines: Vec<Tensor> = (0..lens.len())
            .map(|i| poison(&[lay.elems()], seed + 50 + i as u64))
            .collect();
        let o_short = poison(&[t, h_v * d], seed + 70);
        let geometry = Tensor::zeros((t, conv_dim), DType::F32, &gpu).unwrap();
        let rows: Vec<_> = (0..lens.len())
            .map(|i| (&tails[i], &states[i], &timelines[i], starts[i], lens[i]))
            .collect();
        delta_net_short_span_scan(
            &c.fused(&geometry, &o_short),
            &c.qkv,
            &c.kern,
            &short_table(&rows),
            eps,
        )
        .unwrap();
        for i in 0..lens.len() {
            assert_eq!(
                f32_bits(&states[i]),
                f32_bits(&entering[i]),
                "span {i}: S was written"
            );
        }
        let o_short = o_short.to_device(&Device::Cpu).unwrap();

        for (i, &len) in lens.iter().enumerate() {
            for m in 1..=len {
                // The oracle: the pair over the span's first `m` rows alone.
                let state_out = poison(&[h_v, d, d], seed + 1000 + m as u64);
                let tail_out = poison(&[conv_dim, 3], seed + 2000 + m as u64);
                let o_three = poison(&[t, h_v * d], seed + 3000 + m as u64);
                let conved = Tensor::zeros((t, conv_dim), DType::F32, &gpu).unwrap();
                let tbl = span_table(&[TestSpan {
                    tail: &tails[i],
                    tail_out: &tail_out,
                    state: &entering[i],
                    state_out: &state_out,
                    start: starts[i],
                    len: m,
                }]);
                delta_net_conv_prefill(&c.qkv, &c.kern, &tbl, 2 * c.h_k * d, eps, &conved).unwrap();
                delta_net_prefill_scan(&c.fused(&conved, &o_three), &tbl).unwrap();

                // The fold of the first `m` rows into a copy of the entering state.
                let folded = DeltaNetState {
                    s: entering[i].copy().unwrap(),
                    conv_tail: tails[i].copy().unwrap(),
                };
                delta_net_fold_states(&[(&folded, &timelines[i], m)], &dims).unwrap();

                assert_eq!(
                    f32_bits(&folded.s),
                    f32_bits(&state_out),
                    "span of {len}, m = {m}: S"
                );
                assert_eq!(
                    f32_bits(&folded.conv_tail),
                    f32_bits(&tail_out),
                    "span of {len}, m = {m}: conv tail"
                );
                let o_three = o_three.to_device(&Device::Cpu).unwrap();
                assert_eq!(
                    f32_bits(&o_three.narrow(0, starts[i], m).unwrap()),
                    f32_bits(&o_short.narrow(0, starts[i], m).unwrap()),
                    "span of {len}, m = {m}: o"
                );
            }
        }
    }

    /// **The tensor-op fold is the kernel fold** (`docs/deltanet_timeline.md`
    /// §10, test 3b): from the timelines the short-span kernel writes, for every
    /// span length 2..=8 and every kept prefix, `fold_by_ops` lands the state the
    /// fold kernel lands within the reference's parity tolerance, and the conv
    /// tail exactly — the tail is a copy of raw inputs, with no arithmetic to
    /// differ in.
    #[test]
    fn the_tensor_op_fold_matches_the_kernel_fold() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        use super::super::super::timeline::fold_by_ops;
        let d = ShortCase::D;
        let lens: Vec<usize> = (2..=DELTA_NET_SHORT_SPAN_ROWS).collect();
        let t: usize = lens.iter().sum();
        let seed = 500u64;
        let c = ShortCase::build(t, seed, &gpu);
        let (h_v, conv_dim) = (c.h_v, c.conv_dim);
        let dims = DeltaNetDims {
            head_dim: d,
            n_k_heads: c.h_k,
            n_v_heads: h_v,
            conv_kernel: DELTA_NET_SHORT_SPAN_CONV,
        };
        let lay = TimelineLayout::of(&dims);
        let starts: Vec<usize> = lens
            .iter()
            .scan(0usize, |acc, &l| {
                let s = *acc;
                *acc += l;
                Some(s)
            })
            .collect();
        let entering: Vec<Tensor> = (0..lens.len())
            .map(|i| lcg_tensor(&[h_v, d, d], seed + 10 + i as u64, &gpu))
            .collect();
        let tails: Vec<Tensor> = (0..lens.len())
            .map(|i| lcg_tensor(&[conv_dim, 3], seed + 30 + i as u64, &gpu))
            .collect();
        let timelines: Vec<Tensor> = (0..lens.len()).map(|_| lay.alloc(&gpu).unwrap()).collect();
        let o = Tensor::zeros((t, h_v * d), DType::F32, &gpu).unwrap();
        let geometry = Tensor::zeros((t, conv_dim), DType::F32, &gpu).unwrap();
        let rows: Vec<_> = (0..lens.len())
            .map(|i| (&tails[i], &entering[i], &timelines[i], starts[i], lens[i]))
            .collect();
        delta_net_short_span_scan(
            &c.fused(&geometry, &o),
            &c.qkv,
            &c.kern,
            &short_table(&rows),
            1e-6,
        )
        .unwrap();

        for (i, &len) in lens.iter().enumerate() {
            for m in 1..=len {
                let kernel = DeltaNetState {
                    s: entering[i].copy().unwrap(),
                    conv_tail: tails[i].copy().unwrap(),
                };
                delta_net_fold_states(&[(&kernel, &timelines[i], m)], &dims).unwrap();
                let mut ops = DeltaNetState {
                    s: entering[i].copy().unwrap(),
                    conv_tail: tails[i].copy().unwrap(),
                };
                fold_by_ops(&mut ops, &timelines[i], m, &dims).unwrap();
                let sd = max_diff(&kernel.s, &ops.s);
                assert!(sd < 1e-4, "span of {len}, m = {m}: S differs by {sd}");
                assert_eq!(
                    f32_bits(&kernel.conv_tail),
                    f32_bits(&ops.conv_tail),
                    "span of {len}, m = {m}: conv tail"
                );
            }
        }
    }

    /// **A replay is the captured span's state, and nothing else of it**
    /// (`docs/deltanet_timeline.md` §14.3). Span A runs with its rows
    /// captured; span B — other rows, another length — runs from the same
    /// entering state replaying A's rows. B's advanced state is A's, bit for
    /// bit, while B's outputs and conv tail are B's own, bit for bit what B
    /// produces with no hook at all.
    #[test]
    fn a_replay_takes_the_captured_span_s_state_and_leaves_its_own_outputs() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        use super::super::super::mix::{
            delta_net_mix_spans, DeltaNetConstants, DeltaNetOut, DeltaNetProjections, SpanWrite,
        };
        use super::super::super::span_rows::SpanRows;
        let dims = DeltaNetDims {
            head_dim: 128,
            n_k_heads: 1,
            n_v_heads: 2,
            conv_kernel: 4,
        };
        let proj = |len: usize, seed: u64| DeltaNetProjections {
            qkv: lcg_tensor(&[len, dims.conv_dim()], seed, &gpu),
            z: lcg_tensor(&[len, dims.value_dim()], seed + 1, &gpu),
            beta_lin: lcg_tensor(&[len, dims.n_v_heads], seed + 2, &gpu),
            alpha_lin: lcg_tensor(&[len, dims.n_v_heads], seed + 3, &gpu),
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
        let entering = || DeltaNetState {
            s: lcg_tensor(&[2, 128, 128], 40, &gpu),
            conv_tail: lcg_tensor(&[dims.conv_dim(), 3], 60, &gpu),
        };
        let scratch = || DeltaNetOut {
            s: Tensor::zeros((2, 128, 128), DType::F32, &gpu).unwrap(),
            conv_tail: Tensor::zeros((dims.conv_dim(), 3), DType::F32, &gpu).unwrap(),
        };
        // One long span through the fused mixer, into a scratch apart from
        // its entering state, as a store hands a prefilling sequence.
        let run = |p: &DeltaNetProjections<'static>, len: usize, hook: SpanHook| {
            let mut st = entering();
            let out = scratch();
            let mut seqs = [DeltaNetSeq {
                start: 0,
                len,
                state: &mut st,
                write: SpanWrite::State(out.clone()),
                hook,
            }];
            let o = delta_net_mix_spans(p, &c, &dims, &mut seqs, 1e-6, None, ZGate::Sigmoid, None)
                .unwrap();
            (o.to_device(&Device::Cpu).unwrap(), out)
        };

        let (la, lb) = (TIMELINE_ROWS + 12, TIMELINE_ROWS + 7);
        let (pa, pb) = (proj(la, 11), proj(lb, 21));
        let captured = SpanRows::alloc(la, dims.conv_dim(), dims.n_v_heads, &gpu).unwrap();
        let (_, a_out) = run(&pa, la, SpanHook::Capture(captured.clone()));
        let (b_alone, b_own) = run(&pb, lb, SpanHook::None);
        let (b_replayed, b_out) = run(&pb, lb, SpanHook::Replay(captured));

        assert_eq!(
            f32_bits(&b_out.s),
            f32_bits(&a_out.s),
            "the replayed state is A's"
        );
        assert_ne!(
            f32_bits(&b_own.s),
            f32_bits(&a_out.s),
            "B's own state differs from A's"
        );
        assert_eq!(
            f32_bits(&b_replayed),
            f32_bits(&b_alone),
            "B's outputs are its own"
        );
        assert_eq!(
            f32_bits(&b_out.conv_tail),
            f32_bits(&b_own.conv_tail),
            "B's conv tail is its own"
        );
    }

    /// A span past [`DELTA_NET_SHORT_SPAN_ROWS`] is refused by name, before
    /// anything launches — the pair serves it.
    #[test]
    fn a_nine_row_span_is_refused_by_the_short_span_launch() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let t = DELTA_NET_SHORT_SPAN_ROWS + 1;
        let c = ShortCase::build(t, 77, &gpu);
        let d = ShortCase::D;
        let state = Tensor::zeros((c.h_v, d, d), DType::F32, &gpu).unwrap();
        let tail = Tensor::zeros((c.conv_dim, 3), DType::F32, &gpu).unwrap();
        let timeline = Tensor::zeros(16, DType::F32, &gpu).unwrap();
        let o = Tensor::zeros((t, c.h_v * d), DType::F32, &gpu).unwrap();
        let conved = Tensor::zeros((t, c.conv_dim), DType::F32, &gpu).unwrap();
        let tbl = short_table(&[(&tail, &state, &timeline, 0, t)]);
        let err = delta_net_short_span_scan(&c.fused(&conved, &o), &c.qkv, &c.kern, &tbl, 1e-6)
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("the short-span scan takes spans of at most 8 rows, got one of 9"),
            "{err}"
        );
        assert_eq!(
            f32_bits(&o).iter().filter(|&&b| b != 0).count(),
            0,
            "a refused launch wrote output"
        );
    }
}
