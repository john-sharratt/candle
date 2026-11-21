//! Kernel tests for the batched decode step and the batched decode conv, against
//! sequential CPU references.
//!
//! Two sequences are stepped with their rows interleaved in one wave. Each step
//! is one launch over both, and each result is compared with an independent CPU
//! reference. The conv test covers both epilogue regimes and the block alignment,
//! and shifts each tail in place across a chained decode.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::super::super::mix::{causal_conv1d, delta_recurrence};

    use super::*;

    use candle::{DType, Device};

    use super::super::tests_support::tests::{conv_epilogue_ref, lcg_tensor, max_diff, FusedCase};

    #[test]
    fn decode_step_matches_reference_across_a_sequence() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        // TWO decode sequences stepped together through the batched kernel —
        // interleaved wave rows (seq A on even rows, B on odd), separate
        // in-place states — against two independent sequential references.
        // h_k < h_v so the in-kernel tiled GQA mapping is exercised too.
        let (steps, h_k, h_v, d) = (5usize, 2usize, 4usize, 128usize);
        let t_wave = 2 * steps;
        let case = FusedCase::build(t_wave, h_k, h_v, 61, &cpu, &gpu);

        // CPU oracles: each sequence's rows of the shared wave buffers.
        let seq_rows = |first: usize| -> Vec<usize> { (0..steps).map(|i| 2 * i + first).collect() };
        let gather = |x: &Tensor, rows: &[usize]| -> Tensor {
            let idx = Tensor::from_vec(
                rows.iter().map(|&r| r as u32).collect::<Vec<_>>(),
                rows.len(),
                &cpu,
            )
            .unwrap();
            x.index_select(&idx, 0).unwrap()
        };
        let mut o_refs = Vec::new();
        let mut s_refs = Vec::new();
        for first in [0usize, 1] {
            let rows = seq_rows(first);
            let s0 = Tensor::zeros((h_v, d, d), DType::F32, &cpu).unwrap();
            let (o_r, s_r) = delta_recurrence(
                s0,
                &gather(&case.q_ref, &rows),
                &gather(&case.k_ref, &rows),
                &gather(&case.v_ref, &rows),
                &gather(&case.g_ref, &rows),
                &gather(&case.b_ref, &rows),
            )
            .unwrap();
            o_refs.push(o_r);
            s_refs.push(s_r);
        }

        // GPU: one batched launch per step over BOTH sequences, then one fold
        // of each sequence's one-row timeline — the decode step reads its state
        // and writes only the timeline.
        let dims = DeltaNetDims {
            head_dim: d,
            n_k_heads: h_k,
            n_v_heads: h_v,
            conv_kernel: 4,
        };
        let lay = TimelineLayout::of(&dims);
        let states = [
            DeltaNetState::zeros(&dims, &gpu).unwrap(),
            DeltaNetState::zeros(&dims, &gpu).unwrap(),
        ];
        let timelines = [lay.alloc(&gpu).unwrap(), lay.alloc(&gpu).unwrap()];
        let o = Tensor::zeros((t_wave, h_v * d), DType::F32, &gpu).unwrap();
        let fused = case.fused(&o);
        for i in 0..steps {
            let ptrs: Vec<i64> = vec![
                f32_ptr(&states[0].conv_tail, "tail").unwrap() as i64,
                f32_ptr(&states[1].conv_tail, "tail").unwrap() as i64,
                f32_ptr(&states[0].s, "state").unwrap() as i64,
                f32_ptr(&states[1].s, "state").unwrap() as i64,
                f32_ptr(&timelines[0], "timeline").unwrap() as i64,
                f32_ptr(&timelines[1], "timeline").unwrap() as i64,
            ];
            let table = DeltaNetLayerTable {
                ptrs: Tensor::from_vec(ptrs, (3, 2), &gpu).unwrap(),
                rows: Tensor::from_vec(vec![2 * i as u32, 2 * i as u32 + 1], 2, &gpu).unwrap(),
            };
            delta_net_decode_batch(&fused, &table).unwrap();
            delta_net_fold_states(
                &[
                    (&states[0], &timelines[0], 1),
                    (&states[1], &timelines[1], 1),
                ],
                &dims,
            )
            .unwrap();
        }

        let o_cpu = o
            .reshape((t_wave, h_v, d))
            .unwrap()
            .to_device(&cpu)
            .unwrap();
        for (first, (o_ref, s_ref)) in o_refs.iter().zip(s_refs.iter()).enumerate() {
            let o_gpu = gather(&o_cpu, &seq_rows(first));
            let s_gpu = states[first].s.to_device(&cpu).unwrap();
            let od = max_diff(o_ref, &o_gpu);
            let sd = max_diff(s_ref, &s_gpu);
            assert!(
                od <= 2e-5,
                "seq {first}: outputs diverged from reference: {od}"
            );
            assert!(
                sd <= 2e-5,
                "seq {first}: state diverged from reference: {sd}"
            );
        }
    }

    /// The batched decode conv vs the CPU conv + epilogue, column by column,
    /// with the carried tail shifted in place across a chained decode. Two
    /// sequences stepped together on interleaved wave rows, so the pointer
    /// table's scattering is exercised; 512 channels = 2 normed Q|K heads +
    /// 2 V heads, so both epilogue regimes and the block alignment are too.
    #[test]
    fn conv_step_matches_reference() {
        let Ok(gpu) = Device::new_cuda(0) else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let cpu = Device::Cpu;
        let (c, kw, steps, qkc) = (512usize, 4usize, 6usize, 256usize);
        let eps = 1e-6f64;
        let kern = lcg_tensor(&[c, kw], 12, &cpu);
        let kern_g = kern.to_device(&gpu).unwrap().contiguous().unwrap();

        // Two sequences with different token streams.
        let xa = lcg_tensor(&[c, steps], 11, &cpu);
        let xb = lcg_tensor(&[c, steps], 13, &cpu);

        // CPU oracles: whole segments in one call each, then the epilogue.
        let mut y_refs = Vec::new();
        let mut tail_refs = Vec::new();
        for x in [&xa, &xb] {
            let tail0 = Tensor::zeros((c, kw - 1), DType::F32, &cpu).unwrap();
            let (y_raw, tail_ref) = causal_conv1d(x, &kern, &tail0).unwrap();
            y_refs.push(conv_epilogue_ref(&y_raw, qkc, eps));
            tail_refs.push(tail_ref);
        }

        // GPU: one batched launch per step over both sequences, interleaved
        // wave rows (A on even, B on odd). The conv reads each tail and writes
        // the row's raw input into its timeline; the fold shifts it in. 512
        // channels with 256 of them Q|K is h_k 1, h_v 2 at d 128. The timeline's
        // S fields stay zero (no decode step runs), so the fold leaves S as it is.
        let dims = DeltaNetDims {
            head_dim: 128,
            n_k_heads: 1,
            n_v_heads: 2,
            conv_kernel: kw,
        };
        assert_eq!(dims.conv_dim(), c);
        let lay = TimelineLayout::of(&dims);
        let states = [
            DeltaNetState::zeros(&dims, &gpu).unwrap(),
            DeltaNetState::zeros(&dims, &gpu).unwrap(),
        ];
        let timelines = [lay.alloc(&gpu).unwrap(), lay.alloc(&gpu).unwrap()];
        let t_wave = 2 * steps;
        // The wave's raw QKV rows: row 2i = A's token i, row 2i+1 = B's.
        let mut xw = Vec::with_capacity(t_wave * c);
        let (va, vb) = (
            xa.t()
                .unwrap()
                .contiguous()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            xb.t()
                .unwrap()
                .contiguous()
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
        for i in 0..steps {
            xw.extend_from_slice(&va[i * c..(i + 1) * c]);
            xw.extend_from_slice(&vb[i * c..(i + 1) * c]);
        }
        let qkv = Tensor::from_vec(xw, (t_wave, c), &gpu).unwrap();
        let conved = Tensor::zeros((t_wave, c), DType::F32, &gpu).unwrap();
        for i in 0..steps {
            // The state row is unused by the conv kernel.
            let ptrs: Vec<i64> = vec![
                f32_ptr(&states[0].conv_tail, "tail").unwrap() as i64,
                f32_ptr(&states[1].conv_tail, "tail").unwrap() as i64,
                f32_ptr(&states[0].s, "state").unwrap() as i64,
                f32_ptr(&states[1].s, "state").unwrap() as i64,
                f32_ptr(&timelines[0], "timeline").unwrap() as i64,
                f32_ptr(&timelines[1], "timeline").unwrap() as i64,
            ];
            let table = DeltaNetLayerTable {
                ptrs: Tensor::from_vec(ptrs, (3, 2), &gpu).unwrap(),
                rows: Tensor::from_vec(vec![2 * i as u32, 2 * i as u32 + 1], 2, &gpu).unwrap(),
            };
            delta_net_conv_decode(&qkv, &kern_g, &table, &conved, qkc, eps as f32).unwrap();
            delta_net_fold_states(
                &[
                    (&states[0], &timelines[0], 1),
                    (&states[1], &timelines[1], 1),
                ],
                &dims,
            )
            .unwrap();
        }

        let y_cpu = conved.to_device(&cpu).unwrap();
        for (s, (y_ref, tail_ref)) in y_refs.iter().zip(tail_refs.iter()).enumerate() {
            // Gather this sequence's rows back to channel-major [C, steps].
            let idx = Tensor::from_vec(
                (0..steps).map(|i| (2 * i + s) as u32).collect::<Vec<_>>(),
                steps,
                &cpu,
            )
            .unwrap();
            let y_gpu = y_cpu
                .index_select(&idx, 0)
                .unwrap()
                .t()
                .unwrap()
                .contiguous()
                .unwrap();
            let tail_gpu = states[s].conv_tail.to_device(&cpu).unwrap();
            let yd = max_diff(y_ref, &y_gpu);
            let td = max_diff(tail_ref, &tail_gpu);
            assert!(yd <= 1e-5, "seq {s}: conv outputs diverged: {yd}");
            assert_eq!(
                td, 0.0,
                "seq {s}: the folded conv tail is not the raw inputs"
            );
        }
    }
}
