//! Tests that batching sequences into one mixer call gives the same result as
//! running each one alone, and that spans are validated before anything runs.
//!
//! Two sequences of different lengths cover both the timeline write and the state
//! write. Each must match its own solo run in outputs and in the state it carries,
//! so a shared state or an off-by-one row fails the test. Folding a prefix of a
//! timeline must equal having run only that prefix, which speculative rewind
//! depends on. Spans that leave a gap, overlap, or stop short are refused rather
//! than mixed.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use candle::Device;

    use super::super::tests_recurrence::tests::{assert_close, dev, lcg_tensor};

    /// Geometry for the span tests: small, but with a real conv kernel width
    /// and more than one head, so a wrong slice cannot coincide with a right one.
    pub(in super::super) fn span_dims() -> DeltaNetDims {
        DeltaNetDims {
            head_dim: 4,
            n_k_heads: 2,
            n_v_heads: 2,
            conv_kernel: 3,
        }
    }

    /// Projections over `t` rows, deterministic.
    pub(in super::super) fn span_projections(
        t: usize,
        dims: &DeltaNetDims,
        dev: &Device,
    ) -> DeltaNetProjections<'static> {
        DeltaNetProjections {
            qkv: lcg_tensor(&[t, dims.conv_dim()], 91, dev),
            z: lcg_tensor(&[t, dims.value_dim()], 92, dev),
            beta_lin: lcg_tensor(&[t, dims.n_v_heads], 93, dev),
            alpha_lin: lcg_tensor(&[t, dims.n_v_heads], 94, dev),
        }
    }

    pub(in super::super) fn span_constants(
        dims: &DeltaNetDims,
        dev: &Device,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let dt_bias = lcg_tensor(&[dims.n_v_heads], 95, dev);
        // `a` is `−exp(A_log)` in the checkpoint, so strictly negative.
        let a = lcg_tensor(&[dims.n_v_heads], 96, dev)
            .abs()
            .unwrap()
            .neg()
            .unwrap();
        let conv = lcg_tensor(&[dims.conv_dim(), dims.conv_kernel], 97, dev);
        let norm = lcg_tensor(&[dims.head_dim], 98, dev).abs().unwrap();
        (dt_bias, a, conv, norm)
    }

    /// **Batching several sequences into one call must equal running each
    /// alone.**
    ///
    /// This is the property the whole staged mixer rests on. Everything except
    /// the conv tail and the recurrence is row-wise, which is what lets the
    /// projections and the thirty-odd elementwise ops run once for a whole wave
    /// — but only if the two carried steps still see exactly their own rows and
    /// their own state. A span off by one row, a state shared between
    /// sequences, or a concatenation in the wrong order all leak one
    /// conversation into another, and the symptom is a plausible wrong answer
    /// rather than a crash.
    ///
    /// Two sequences of *different* lengths, so an off-by-one cannot alias —
    /// one long enough to write a state, one short enough to write a timeline.
    #[test]
    fn spans_equal_running_each_sequence_alone() {
        let dev = dev();
        let dims = span_dims();
        let (len_a, len_b) = (TIMELINE_ROWS + 3, 3usize);
        let total = len_a + len_b;
        let eps = 1e-6;

        let p = span_projections(total, &dims, &dev);
        let (dt_bias, a, conv, norm) = span_constants(&dims, &dev);
        let c = DeltaNetConstants {
            dt_bias: &dt_bias,
            a: &a,
            conv: &conv,
            norm: &norm,
        };

        // Each sequence alone, from its own fresh state, over its own rows.
        let mut wants = Vec::new();
        let mut solo_states = Vec::new();
        for (start, len) in [(0usize, len_a), (len_a, len_b)] {
            let rows = |x: &Tensor| x.narrow(0, start, len).unwrap().contiguous().unwrap();
            let ps = DeltaNetProjections {
                qkv: rows(&p.qkv),
                z: rows(&p.z),
                beta_lin: rows(&p.beta_lin),
                alpha_lin: rows(&p.alpha_lin),
            };
            let mut st = DeltaNetState::zeros(&dims, &dev).unwrap();
            wants.push(delta_net_mix(&ps, &c, &dims, &mut st, eps, ZGate::Silu).unwrap());
            solo_states.push(st);
        }
        let want = Tensor::cat(&wants, 0).unwrap();

        // Both together, one call, one state each.
        let mut st_a = DeltaNetState::zeros(&dims, &dev).unwrap();
        let mut st_b = DeltaNetState::zeros(&dims, &dev).unwrap();
        let got = {
            let write_a = solo_write(&st_a, len_a, &dims).unwrap();
            let write_b = solo_write(&st_b, len_b, &dims).unwrap();
            let mut seqs = [
                DeltaNetSeq {
                    start: 0,
                    len: len_a,
                    state: &mut st_a,
                    write: write_a,
                    hook: SpanHook::None,
                },
                DeltaNetSeq {
                    start: len_a,
                    len: len_b,
                    state: &mut st_b,
                    write: write_b,
                    hook: SpanHook::None,
                },
            ];
            let mixed = delta_net_mix_spans(&p, &c, &dims, &mut seqs, eps, None, ZGate::Silu, None)
                .unwrap();
            for s in seqs.iter_mut() {
                finish_solo(s.state, &s.write, s.len, &dims).unwrap();
            }
            mixed
        };

        assert_close(&got, &want, 1e-5, "batched vs solo activations");
        // And the carried state: the activations could match while the state
        // diverged, and the next token would be wrong instead of this one.
        for (i, (batched, solo)) in [&st_a, &st_b].iter().zip(&solo_states).enumerate() {
            assert_close(&batched.s, &solo.s, 1e-5, &format!("sequence {i} state"));
            assert_close(
                &batched.conv_tail,
                &solo.conv_tail,
                1e-5,
                &format!("sequence {i} conv tail"),
            );
        }
        // The fixture must actually distinguish the sequences — two that ended
        // in the same state would pass under a shared-state bug.
        let cross = st_a
            .s
            .sub(&st_b.s)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(cross > 1e-6, "the two sequences ended in identical state");
    }

    /// **Folding the first `m` rows of a block's timeline equals having only
    /// ever run those `m` rows.**
    ///
    /// This is the property speculative decode rests on, and it is the reason
    /// the hybrid can rewind at all: `S` is a running sum with no suffix to
    /// subtract, so a partial accept is a choice of how many of the block's
    /// rows to fold into the state the block entered with — which nothing
    /// wrote, because a short span writes only its timeline.
    ///
    /// The oracle is a clean run of the same `m` rows into a fresh copy of the
    /// entering state. If the two disagree, a rejected draft has left its mark
    /// on the recurrence, and the model would answer as though it had read
    /// tokens the KV no longer holds. Run for every `m`, so the prefix
    /// stability of `G`, the solve and `v_new` is held at every row.
    #[test]
    fn folding_a_prefix_equals_running_only_that_prefix() {
        let dev = dev();
        let dims = span_dims();
        let block = 6usize;
        let p = span_projections(block, &dims, &dev);
        let (dt_bias, a, conv, norm) = span_constants(&dims, &dev);
        let c = DeltaNetConstants {
            dt_bias: &dt_bias,
            a: &a,
            conv: &conv,
            norm: &norm,
        };

        // A non-trivial entering state, so a fold that silently started from
        // zeros could not pass.
        let entering = DeltaNetState {
            s: lcg_tensor(&[dims.n_v_heads, dims.head_dim, dims.head_dim], 77, &dev),
            conv_tail: lcg_tensor(&[dims.conv_dim(), dims.conv_kernel - 1], 78, &dev),
        };

        // The wave: the whole block, reading `entering` and writing a timeline.
        let mut read = entering.snapshot().unwrap();
        let tl = TimelineLayout::of(&dims).alloc(&dev).unwrap();
        {
            let mut seqs = [DeltaNetSeq {
                start: 0,
                len: block,
                state: &mut read,
                write: SpanWrite::Timeline(tl.clone()),
                hook: SpanHook::None,
            }];
            delta_net_mix_spans(&p, &c, &dims, &mut seqs, 1e-6, None, ZGate::Silu, None).unwrap();
        }
        assert_close(
            &read.s,
            &entering.s,
            0.0,
            "a short span left its state as it was",
        );

        for kept in 1..=block {
            let mut folded = entering.snapshot().unwrap();
            fold_by_ops(&mut folded, &tl, kept, &dims).unwrap();

            // The oracle: the same `kept` rows, nothing else, same entering state.
            let mut want = entering.snapshot().unwrap();
            let po = DeltaNetProjections {
                qkv: p.qkv.narrow(0, 0, kept).unwrap().contiguous().unwrap(),
                z: p.z.narrow(0, 0, kept).unwrap().contiguous().unwrap(),
                beta_lin: p.beta_lin.narrow(0, 0, kept).unwrap().contiguous().unwrap(),
                alpha_lin: p
                    .alpha_lin
                    .narrow(0, 0, kept)
                    .unwrap()
                    .contiguous()
                    .unwrap(),
            };
            delta_net_mix(&po, &c, &dims, &mut want, 1e-6, ZGate::Silu).unwrap();

            assert_close(&folded.s, &want.s, 1e-5, &format!("state folded at {kept}"));
            assert_close(
                &folded.conv_tail,
                &want.conv_tail,
                0.0,
                &format!("conv tail folded at {kept}"),
            );
        }

        // And the whole block really does advance further than a prefix, so
        // the assertions above are not comparing a rewind to a no-op.
        let mut whole = entering.snapshot().unwrap();
        fold_by_ops(&mut whole, &tl, block, &dims).unwrap();
        let mut one = entering.snapshot().unwrap();
        fold_by_ops(&mut one, &tl, 1, &dims).unwrap();
        let gap = whole
            .s
            .sub(&one.s)
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
            gap > 1e-5,
            "the {block}-row block ended in the same state as its first row — the \
             fixture cannot tell a rewind from a no-op"
        );
    }

    /// A span's write must fit its length — a short span handed a state, or a
    /// long one a timeline, is refused before anything runs.
    #[test]
    fn a_span_write_that_does_not_fit_its_length_is_refused() {
        let dev = dev();
        let dims = span_dims();
        let (dt_bias, a, conv, norm) = span_constants(&dims, &dev);
        let c = DeltaNetConstants {
            dt_bias: &dt_bias,
            a: &a,
            conv: &conv,
            norm: &norm,
        };
        let mut st = DeltaNetState::zeros(&dims, &dev).unwrap();
        let short = span_projections(3, &dims, &dev);
        let out = st.solo_out().unwrap();
        let mut seqs = [DeltaNetSeq {
            start: 0,
            len: 3,
            state: &mut st,
            write: SpanWrite::State(out),
            hook: SpanHook::None,
        }];
        let err = delta_net_mix_spans(&short, &c, &dims, &mut seqs, 1e-6, None, ZGate::Silu, None)
            .unwrap_err();
        assert!(err.to_string().contains("timeline"), "{err}");

        let long = span_projections(TIMELINE_ROWS + 1, &dims, &dev);
        let tl = TimelineLayout::of(&dims).alloc(&dev).unwrap();
        let mut seqs = [DeltaNetSeq {
            start: 0,
            len: TIMELINE_ROWS + 1,
            state: &mut st,
            write: SpanWrite::Timeline(tl),
            hook: SpanHook::None,
        }];
        let err = delta_net_mix_spans(&long, &c, &dims, &mut seqs, 1e-6, None, ZGate::Silu, None)
            .unwrap_err();
        assert!(err.to_string().contains("state"), "{err}");
    }

    /// Spans that do not tile the buffer must be refused, not silently mixed.
    ///
    /// Every one of these reads some sequence's rows into another's recurrence.
    /// The caller derives spans from the wave's packing, so a mismatch here is a
    /// packing bug — and it has to surface as one rather than as a model that
    /// answers slightly wrong.
    #[test]
    fn spans_must_tile_the_buffer() {
        let dev = dev();
        let dims = span_dims();
        let total = 8usize;
        let p = span_projections(total, &dims, &dev);
        let (dt_bias, a, conv, norm) = span_constants(&dims, &dev);
        let c = DeltaNetConstants {
            dt_bias: &dt_bias,
            a: &a,
            conv: &conv,
            norm: &norm,
        };
        let mut st_a = DeltaNetState::zeros(&dims, &dev).unwrap();
        let mut st_b = DeltaNetState::zeros(&dims, &dev).unwrap();

        let run = |spans: [(usize, usize); 2], sa: &mut DeltaNetState, sb: &mut DeltaNetState| {
            let write_a = solo_write(sa, spans[0].1, &dims).unwrap();
            let write_b = solo_write(sb, spans[1].1, &dims).unwrap();
            let mut seqs = [
                DeltaNetSeq {
                    start: spans[0].0,
                    len: spans[0].1,
                    state: sa,
                    write: write_a,
                    hook: SpanHook::None,
                },
                DeltaNetSeq {
                    start: spans[1].0,
                    len: spans[1].1,
                    state: sb,
                    write: write_b,
                    hook: SpanHook::None,
                },
            ];
            delta_net_mix_spans(&p, &c, &dims, &mut seqs, 1e-6, None, ZGate::Silu, None).map(|_| ())
        };

        // A gap: rows 5..6 belong to nobody.
        let err = run([(0, 5), (6, 2)], &mut st_a, &mut st_b).unwrap_err();
        assert!(err.to_string().contains("tile"), "{err}");
        // An overlap: row 4 is fed to both sequences.
        let err = run([(0, 5), (4, 4)], &mut st_a, &mut st_b).unwrap_err();
        assert!(err.to_string().contains("tile"), "{err}");
        // Short: the last rows of the buffer are never mixed.
        let err = run([(0, 5), (5, 2)], &mut st_a, &mut st_b).unwrap_err();
        assert!(err.to_string().contains("cover"), "{err}");
        // The tiling one is accepted, so the checks above are rejecting the
        // defect rather than the shape of the call.
        assert!(run([(0, 5), (5, 3)], &mut st_a, &mut st_b).is_ok());
    }
}
