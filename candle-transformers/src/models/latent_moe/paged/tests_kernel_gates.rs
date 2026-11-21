//! Checks the kernel against its CPU mirror and float reference across RoPE,
//! split-KV and sliding-window cases, and checks the arena round trip.
//!
//! Most gates are bit-exact, which proves every arithmetic path except the
//! fast-math dequant seam, where a handful of cells may differ by one bf16
//! code. The device exponential and RoPE trigonometry are checked against the
//! mirror over the ranges the kernel uses. Arena-backed and synthetic slots must
//! produce identical output, and the live-buffer and snapshot decode paths must
//! agree step after step across chunk boundaries.

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

    use crate::models::rope_schedule::yarn_freqs;

    use candle::Device;

    use super::super::tests_geometry::tests::H;

    use super::super::tests_oracle::tests::{
        bf16_exact, check_within_one_code, fp8_exact, gen_inputs, max_abs_diff, mirror_sincos,
        run_case, run_kernel, table_sincos, Case,
    };

    /// Oracle (a1): zero-rope, splits=1 — every arithmetic path except trig,
    /// gated BIT-EXACT (the mirror reproduces the kernel to the bf16 bit).
    #[test]
    #[ignore]
    fn mirror_bit_exact_zero_rope() -> Result<()> {
        let case = Case {
            n_win: 50,
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 1,
            zero_rope: true,
        };
        let (kernel, mirror_v, _) = run_case(&case, 42)?;
        let mismatches: Vec<(usize, usize, f32, f32)> = kernel
            .iter()
            .zip(&mirror_v)
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .map(|(i, (a, b))| (i / HEAD_DIM, i % HEAD_DIM, *a, *b))
            .collect();
        assert!(
            mismatches.is_empty(),
            "bit mismatches: {}/{} (max |Δ| = {}); first 8 (head, dim, kernel, mirror): {:?}",
            mismatches.len(),
            kernel.len(),
            max_abs_diff(&kernel, &mirror_v),
            &mismatches[..mismatches.len().min(8)]
        );
        Ok(())
    }

    /// The device `ds_exp` and the CPU mirror replica must agree bit-for-bit
    /// across the softmax input range — the foundation every bit-exact gate
    /// stands on.
    #[test]
    #[ignore]
    fn ds_exp_device_matches_mirror() -> Result<()> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        let dev = Device::new_cuda(0)?;
        let cuda = match &dev {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = cuda.cuda_stream();
        // Dense sweep of the softmax-relevant range plus edge values.
        let mut xs: Vec<f32> = (0..200_000).map(|i| -50.0 + i as f32 * 0.00025).collect();
        xs.extend_from_slice(&[0.0, -0.0, -1e38, -88.0, -87.999, f32::MIN_POSITIVE]);
        let n = xs.len();
        let input = Tensor::from_vec(xs.clone(), n, &dev)?;
        let out = Tensor::zeros(n, DType::F32, &dev)?;
        {
            let (si, _) = input.storage_and_layout();
            let (so, _) = out.storage_and_layout();
            let (ip, _g1) = match &*si {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.device_ptr(&stream),
                _ => unreachable!(),
            };
            let (op, _g2) = match &*so {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.device_ptr(&stream),
                _ => unreachable!(),
            };
            unsafe {
                candle_kernels::paged_latent::run_latent_exp_probe(
                    ip as *const f32,
                    op as *mut f32,
                    n as i32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                );
            }
            dev.synchronize()?;
        }
        let got = out.to_vec1::<f32>()?;
        let mismatches: Vec<(f32, f32, f32)> = xs
            .iter()
            .zip(&got)
            .filter(|(&x, &g)| g.to_bits() != ds_exp_mirror(x).to_bits())
            .map(|(&x, &g)| (x, g, ds_exp_mirror(x)))
            .take(5)
            .collect();
        assert!(
            mismatches.is_empty(),
            "ds_exp device≠mirror; first (x, device, mirror): {mismatches:?}"
        );
        Ok(())
    }

    /// Step-2 gate (a): window latents written through the PRODUCTION chunked
    /// backing (single-latent mode) read back exactly as the host round-trip of
    /// the reference two-region format — the raw-byte write→read contract. The
    /// nope span `[0, 448)` is FP8 E4M3, the rope tail `[448, 512)` is BF16, so
    /// the oracle rounds each dim through its region's dtype.
    #[test]
    #[ignore]
    fn arena_write_read_round_trip() -> Result<()> {
        use candle_nn::kv_cache::{ChunkedKvBacking, KvFormat};
        let dev = Device::new_cuda(0)?;
        let backing = ChunkedKvBacking::new_with_format_adaptive(
            1,
            1,
            HEAD_DIM,
            KvFormat::Float(DType::F8E4M3),
            KvFormat::Float(DType::F8E4M3),
            &dev,
            256,
            None,
        )?;
        backing.set_single_latent(true);

        let mut s = 7u64;
        let n = 50usize;
        let vals: Vec<f32> = (0..n * HEAD_DIM).map(|_| fp8_exact(&mut s)).collect();
        let latent = Tensor::from_vec(vals.clone(), (1, 1, n, HEAD_DIM), &dev)?;
        // The production calling convention: KvCache over the chunked backing,
        // write then commit the length.
        let mut cache = candle_nn::kv_cache::KvCache::new(2, 256);
        cache.set_chunked_backing(&backing, 0, None)?;
        cache.chunked_write_kv(0, &latent, &latent)?;
        cache.set_current_seq_len(n)?;

        let (rk, rv) = cache.chunked_read_kv(0, n)?;
        let got_k = rk.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let got_v = rv.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        for (i, &v) in vals.iter().enumerate() {
            let d = i % HEAD_DIM;
            // Two-region store: nope dims round through FP8 E4M3, rope dims
            // through BF16.
            let expect = if d < NOPE_DIM {
                e4m3_to_f32(f32_to_e4m3(v))
            } else {
                half::bf16::from_f32(v).to_f32()
            };
            assert_eq!(
                got_k[i].to_bits(),
                expect.to_bits(),
                "K[{i}] (dim {d}): {} vs {expect}",
                got_k[i]
            );
            // K≡V: the V read aliases the K bytes.
            assert_eq!(
                got_v[i].to_bits(),
                expect.to_bits(),
                "V[{i}] alias (dim {d})"
            );
        }
        Ok(())
    }

    /// Step-2 gate (b): the decode kernel over PRODUCTION-built slot tables
    /// (backing + `build_decode_metadata`) produces bit-identical output to
    /// the same case over the hand-built `SyntheticSlots`.
    #[test]
    #[ignore]
    fn arena_backed_matches_synthetic() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, BatchedInferenceSession};
        use candle_nn::kv_cache::{ChunkedKvBacking, KvFormat};

        let case = Case {
            n_win: 50,
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 1,
            zero_rope: false,
        };
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let inp = gen_inputs(&case, 42);

        // Synthetic-slot run (the rung-2 baseline).
        let synth = run_kernel(&case, &inp, softmax_scale)?;

        // Production-backing run over the same inputs.
        let dev = Device::new_cuda(0)?;
        let backing = ChunkedKvBacking::new_with_format_adaptive(
            1,
            1,
            HEAD_DIM,
            KvFormat::Float(DType::F8E4M3),
            KvFormat::Float(DType::F8E4M3),
            &dev,
            256,
            None,
        )?;
        backing.set_single_latent(true);
        let cfg = BatchedConfig {
            k_format: KvFormat::Float(DType::F8E4M3),
            v_format: KvFormat::Float(DType::F8E4M3),
            initial_seq_len: 256,
            ..Default::default()
        };
        let mut session =
            BatchedInferenceSession::new_with_backings(vec![backing.clone()], cfg, &dev);
        let seq = session.create_sequence()?;

        let flat: Vec<f32> = inp.window.iter().flat_map(|w| w.iter().copied()).collect();
        let latent = Tensor::from_vec(flat, (1, 1, case.n_win, HEAD_DIM), &dev)?;
        let mut cache = candle_nn::kv_cache::KvCache::new(2, 256);
        cache.set_chunked_backing(&backing, seq, None)?;
        cache.chunked_write_kv(0, &latent, &latent)?;
        cache.set_current_seq_len(case.n_win)?;
        session.set_sequence_offset(seq, case.n_win)?;

        let generation = session.begin_stager_generation();
        let (headers, _stride) = session.build_decode_metadata(&[seq], &generation)?;
        let headers = headers.expect("decode metadata headers");

        let qf: Vec<f32> = inp.q.iter().flat_map(|h| h.iter().copied()).collect();
        let q = Tensor::from_vec(qf, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let kv_new =
            Tensor::from_vec(inp.kv_new.to_vec(), (1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let compf: Vec<f32> = inp.comp.iter().flat_map(|c| c.iter().copied()).collect();
        let comp = Tensor::from_vec(compf, (case.g_total, HEAD_DIM), &dev)?;
        let comp_pos = Tensor::from_vec(inp.comp_pos.clone(), case.g_total, &dev)?;
        let max_sel = case.selected.len();
        let idx: Vec<u32> = case.selected.iter().map(|&g| g as u32).collect();
        let comp_idx = Tensor::from_vec(idx, (1, max_sel), &dev)?;
        let comp_cnt = Tensor::from_vec(vec![case.selected.len() as u32], 1, &dev)?;
        let sinks = Tensor::from_vec(inp.sinks.clone(), H, &dev)?;
        let freqs = Tensor::from_vec(inp.freqs.clone(), ROPE_DIM / 2, &dev)?;
        let rope_tab = build_rope_table(&freqs)?;
        let ws = LatentWorkspace::build(&dev)?;

        let out = paged_latent_decode_raw(
            &q,
            headers.dev_ptr(),
            &kv_new,
            &CorpusCache::build(&comp, &comp_pos)?,
            &comp_idx,
            &comp_cnt,
            &Tensor::from_vec(vec![case.n_win as u32], 1, &dev)?,
            &sinks,
            &rope_tab,
            softmax_scale,
            case.window_size,
            case.num_splits,
            1, // single-step audit against the live buffer: commit…
            1, // …and fused-scatter the one slot's token
            &ws,
            None,
        )?;
        drop(generation);
        // Kernel emits F32; round to bf16 to match the bf16-reproduced `synth`.
        let arena_out = out
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let mismatches = synth
            .iter()
            .zip(&arena_out)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches,
            0,
            "arena-backed vs synthetic: {mismatches}/{} bit mismatches (max |Δ| = {})",
            synth.len(),
            max_abs_diff(&synth, &arena_out)
        );
        Ok(())
    }

    /// Correctness gate for the decode-metadata optimization: a PURE-DECODE
    /// wave may serialize its slot state through the cheap LIVE persistent buffer
    /// (`build_decode_metadata_at` `snapshot_slices=false` + on-device
    /// `commit_write_len=true`) — the path Qwen/Llama already use — instead of
    /// the per-step host SNAPSHOT copy (`snapshot_slices=true` + host-patched
    /// write-len). This runs two independent sessions through IDENTICAL decode
    /// steps, one on each path, and asserts their per-step outputs are
    /// BIT-IDENTICAL across ≥3 chunk boundaries (positions 32/64/96). The live
    /// buffer's on-device write-len self-increment must track the snapshot's
    /// offset-derived write-len exactly, step after step, or the window walk
    /// diverges after a boundary.
    #[test]
    #[ignore]
    fn decode_live_buffer_matches_snapshot_multistep() -> Result<()> {
        use crate::models::batched_inference::{BatchedConfig, BatchedInferenceSession};
        use candle_nn::kv_cache::{ChunkedKvBacking, KvCache, KvFormat};

        let dev = Device::new_cuda(0)?;
        let n_win = 30usize; // window starts inside chunk 0
        let steps = 72usize; // new tokens land at 30..101 → crosses 32, 64, 96
        let window_size = 128usize;
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let mut s = 7u64;

        // Deterministic tokens (FP8-exact, the window store format) + queries.
        let toks: Vec<[f32; HEAD_DIM]> = (0..n_win + steps)
            .map(|_| std::array::from_fn(|_| fp8_exact(&mut s)))
            .collect();
        let qs: Vec<[f32; HEAD_DIM]> = (0..steps * H)
            .map(|_| std::array::from_fn(|_| bf16_exact(&mut s)))
            .collect();
        let sinks = Tensor::from_vec((0..H).map(|_| bf16_exact(&mut s) * 0.5).collect(), H, &dev)?;
        let freqs_v: Vec<f32> = yarn_freqs(ROPE_DIM, 10000.0, 0, 1.0, 32.0, 1.0)
            .into_iter()
            .map(|f| f as f32)
            .collect();
        let rope_tab = build_rope_table(&Tensor::from_vec(freqs_v, ROPE_DIM / 2, &dev)?)?;
        let ws = LatentWorkspace::build(&dev)?;

        // Two independent sessions, each with the initial `n_win` window written.
        let mk = || -> Result<(BatchedInferenceSession, usize)> {
            let backing = ChunkedKvBacking::new_with_format_adaptive(
                1,
                1,
                HEAD_DIM,
                KvFormat::Float(DType::F8E4M3),
                KvFormat::Float(DType::F8E4M3),
                &dev,
                256,
                None,
            )?;
            backing.set_single_latent(true);
            let cfg = BatchedConfig {
                k_format: KvFormat::Float(DType::F8E4M3),
                v_format: KvFormat::Float(DType::F8E4M3),
                initial_seq_len: 256,
                ..Default::default()
            };
            let mut session =
                BatchedInferenceSession::new_with_backings(vec![backing.clone()], cfg, &dev);
            let seq = session.create_sequence()?;
            let flat: Vec<f32> = toks[..n_win]
                .iter()
                .flat_map(|w| w.iter().copied())
                .collect();
            let latent = Tensor::from_vec(flat, (1, 1, n_win, HEAD_DIM), &dev)?;
            let mut cache = KvCache::new(2, 256);
            cache.set_chunked_backing(&backing, seq, None)?;
            cache.chunked_write_kv(0, &latent, &latent)?;
            cache.set_current_seq_len(n_win)?;
            session.set_sequence_offset(seq, n_win)?;
            Ok((session, seq))
        };
        let (mut sess_live, seq_l) = mk()?;
        let (mut sess_snap, seq_s) = mk()?;

        // Window-only decode (empty corpus) isolates the slot-buffer path.
        let comp = Tensor::zeros((1, HEAD_DIM), DType::F32, &dev)?;
        let comp_pos = Tensor::zeros(1, DType::U32, &dev)?;
        let comp_idx = Tensor::full(u32::MAX, (1, 1), &dev)?;
        let comp_cnt = Tensor::zeros(1, DType::U32, &dev)?;

        // One decode step on a session at `pos`, over the given snapshot/commit
        // mode; returns the flattened output bits.
        let step = |session: &mut BatchedInferenceSession,
                    seq: usize,
                    pos: usize,
                    t: usize,
                    snapshot: bool|
         -> Result<Vec<f32>> {
            session.set_sequence_offset(seq, pos)?;
            // Commit the host chunk usage to the pre-write prefix (the wave's
            // per-step `set_len`), so the metadata's write chunk/len is derived
            // from a consistent slot state (no eviction here → resident == pos).
            for backing in session.backings() {
                backing.set_len(seq, pos);
            }
            let kv_new =
                Tensor::from_vec(toks[pos].to_vec(), (1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let qf: Vec<f32> = qs[t * H..(t + 1) * H]
                .iter()
                .flat_map(|h| h.iter().copied())
                .collect();
            let q = Tensor::from_vec(qf, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let q_pos = Tensor::from_vec(vec![pos as u32], 1, &dev)?;
            let generation = session.begin_stager_generation();
            let snap_seqs: &[usize] = if snapshot {
                std::slice::from_ref(&seq)
            } else {
                &[]
            };
            let (headers, _stride) = session.build_decode_metadata_at(
                0..session.num_layers(),
                &[seq],
                &generation,
                &[],
                &[],
                snap_seqs,
            )?;
            let headers = headers.expect("decode metadata headers");
            let out = paged_latent_decode_raw(
                &q,
                headers.dev_ptr(),
                &kv_new,
                &CorpusCache::build(&comp, &comp_pos)?,
                &comp_idx,
                &comp_cnt,
                &q_pos,
                &sinks,
                &rope_tab,
                softmax_scale,
                window_size,
                1,
                // Live buffer commits on-device; snapshot patches host-side.
                if snapshot { 0 } else { 1 },
                1, // fused scatter writes the token in both modes
                &ws,
                None,
            )?;
            drop(generation);
            out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
        };

        for t in 0..steps {
            let pos = n_win + t;
            let live = step(&mut sess_live, seq_l, pos, t, false)?;
            let snap = step(&mut sess_snap, seq_s, pos, t, true)?;
            let mism = live
                .iter()
                .zip(&snap)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                mism,
                0,
                "step {t} (pos {pos}): {mism}/{} bit mismatches live-vs-snapshot (max |Δ| {})",
                live.len(),
                max_abs_diff(&live, &snap)
            );
        }
        Ok(())
    }

    /// The kernel's RoPE trig and the CPU mirror must agree bit-for-bit over
    /// the (position, frequency) grid the tests exercise.
    #[test]
    #[ignore]
    fn sincos_device_matches_mirror() -> Result<()> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;
        let dev = Device::new_cuda(0)?;
        let cuda = match &dev {
            Device::Cuda(d) => d.clone(),
            _ => unreachable!(),
        };
        let stream = cuda.cuda_stream();
        let freqs: Vec<f32> = yarn_freqs(ROPE_DIM, 10000.0, 0, 1.0, 32.0, 1.0)
            .into_iter()
            .map(|f| f as f32)
            .collect();
        let mut pos_v: Vec<i32> = Vec::new();
        let mut freq_v: Vec<f32> = Vec::new();
        for pos in (0..200).chain([1_000, 100_000, 1_000_000]) {
            for &f in &freqs {
                pos_v.push(pos);
                freq_v.push(f);
            }
        }
        let n = pos_v.len();
        let pos_t = Tensor::from_vec(pos_v.iter().map(|&p| p as u32).collect::<Vec<_>>(), n, &dev)?;
        let freq_t = Tensor::from_vec(freq_v.clone(), n, &dev)?;
        let out = Tensor::zeros(2 * n, DType::F32, &dev)?;
        {
            let (sp, _) = pos_t.storage_and_layout();
            let (sf, _) = freq_t.storage_and_layout();
            let (so, _) = out.storage_and_layout();
            let (pp, _g1) = match &*sp {
                Storage::Cuda(c) => c.as_cuda_slice::<u32>()?.device_ptr(&stream),
                _ => unreachable!(),
            };
            let (fp, _g2) = match &*sf {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.device_ptr(&stream),
                _ => unreachable!(),
            };
            let (op, _g3) = match &*so {
                Storage::Cuda(c) => c.as_cuda_slice::<f32>()?.device_ptr(&stream),
                _ => unreachable!(),
            };
            unsafe {
                candle_kernels::paged_latent::run_latent_sincos_probe(
                    pp as *const i32,
                    fp as *const f32,
                    op as *mut f32,
                    n as i32,
                    stream.cu_stream() as *mut core::ffi::c_void,
                );
            }
            dev.synchronize()?;
        }
        let got = out.to_vec1::<f32>()?;
        let mismatches: Vec<(i32, f32, (f32, f32), (f32, f32))> = (0..n)
            .filter_map(|i| {
                let (ms, mc) = mirror_sincos(pos_v[i] as u32, freq_v[i]);
                let (gs, gc) = (got[2 * i], got[2 * i + 1]);
                if gs.to_bits() != ms.to_bits() || gc.to_bits() != mc.to_bits() {
                    Some((pos_v[i], freq_v[i], (gs, gc), (ms, mc)))
                } else {
                    None
                }
            })
            .take(5)
            .collect();
        assert!(
            mismatches.is_empty(),
            "sincos device≠mirror; first (pos, freq, device(s,c), mirror(s,c)): {mismatches:?}"
        );
        Ok(())
    }

    /// The device-built factored table must be bit-identical to the mirror's
    /// entry recipe (`mirror_sincos` of the split positions), and the mirror's
    /// `table_sincos` combination must round-trip a spot-check of full
    /// positions through the table layout the kernel reads.
    #[test]
    #[ignore]
    fn rope_table_device_matches_mirror() -> Result<()> {
        let dev = Device::new_cuda(0)?;
        let freqs_v: Vec<f32> = yarn_freqs(ROPE_DIM, 10000.0, 0, 1.0, 32.0, 1.0)
            .into_iter()
            .map(|f| f as f32)
            .collect();
        let freqs = Tensor::from_vec(freqs_v.clone(), ROPE_DIM / 2, &dev)?;
        let tab = build_rope_table(&freqs)?;
        dev.synchronize()?;
        let got = tab.to_vec1::<f32>()?;
        assert_eq!(got.len(), ROPE_TAB_LEN);
        let nf = ROPE_DIM / 2;
        let mut mismatches = 0usize;
        for row in 0..(ROPE_HI_DIM + ROPE_LO_DIM) {
            let pos = if row < ROPE_HI_DIM {
                (row as u32) << 10
            } else {
                (row - ROPE_HI_DIM) as u32
            };
            for (j, &f) in freqs_v.iter().enumerate() {
                let (ms, mc) = mirror_sincos(pos, f);
                let i = row * nf + j;
                if got[2 * i].to_bits() != ms.to_bits() || got[2 * i + 1].to_bits() != mc.to_bits()
                {
                    mismatches += 1;
                }
            }
        }
        assert_eq!(mismatches, 0, "device table ≠ mirror entries");
        // Spot-check the combined lookup at full positions across the range
        // (window/query token positions and sparse group starts alike).
        for &pos in &[0u32, 1, 127, 1023, 1024, 4096, 65_537, 999_999, 1_048_575] {
            let hi = (pos >> 10) as usize;
            let lo = (pos & 1023) as usize;
            for (j, &f) in freqs_v.iter().enumerate() {
                let ih = hi * nf + j;
                let il = (ROPE_HI_DIM + lo) * nf + j;
                let (sh, ch) = (got[2 * ih], got[2 * ih + 1]);
                let (sl, cl) = (got[2 * il], got[2 * il + 1]);
                let s = sh * cl + ch * sl;
                let c = ch * cl - sh * sl;
                let (ts, tc) = table_sincos(pos, f);
                assert_eq!(
                    (s.to_bits(), c.to_bits()),
                    (ts.to_bits(), tc.to_bits()),
                    "combined lookup ≠ table_sincos at pos {pos} freq {f}"
                );
            }
        }
        Ok(())
    }

    /// Bisection probe: grows the key count from 1 (writer only) upward to
    /// localize any mirror divergence to the smallest failing configuration.
    #[test]
    #[ignore]
    fn mirror_probe_minimal() -> Result<()> {
        for (n_win, zero_rope) in [
            (0usize, true),
            (1, true),
            (3, true),
            (8, true),
            (9, true),
            (1, false),
            (3, false),
            (9, false),
        ] {
            let case = Case {
                n_win,
                window_size: 128,
                g_total: 0,
                ratio: 4,
                selected: vec![],
                num_splits: 1,
                zero_rope,
            };
            let (kernel, mirror_v, _) = run_case(&case, 42)?;
            // Attention-OUTPUT gate: ≤1 bf16 code per cell (the fast-math hot
            // path seam), ≥99.9% exactly equal.
            let n_bad = check_within_one_code(&kernel, &mirror_v);
            assert!(
                n_bad * 1000 <= kernel.len(),
                "n_win={n_win} zero_rope={zero_rope}: {n_bad}/{} cells off (> 0.1%)",
                kernel.len()
            );
        }
        Ok(())
    }

    /// Oracle (a1) under split-KV (3 splits + one empty-split config): the
    /// combine's split merge must stay bit-exact.
    #[test]
    #[ignore]
    fn mirror_bit_exact_zero_rope_splits() -> Result<()> {
        for num_splits in [3usize, 5] {
            let case = Case {
                n_win: 50,
                window_size: 128,
                g_total: 10,
                ratio: 4,
                selected: vec![0, 2, 5, 8],
                num_splits,
                zero_rope: true,
            };
            let (kernel, mirror_v, _) = run_case(&case, 7)?;
            let n_mismatch = kernel
                .iter()
                .zip(&mirror_v)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            assert_eq!(
                n_mismatch,
                0,
                "splits={num_splits}: bit mismatches {n_mismatch} (max |Δ| = {})",
                max_abs_diff(&kernel, &mirror_v)
            );
        }
        Ok(())
    }

    /// Oracle (a1) with a tight sliding window (the position clamp) and no
    /// compressed entries (pure-SWA layer shape).
    #[test]
    #[ignore]
    fn mirror_bit_exact_sliding_window_swa() -> Result<()> {
        let case = Case {
            n_win: 50,
            window_size: 16,
            g_total: 0,
            ratio: 4,
            selected: vec![],
            num_splits: 1,
            zero_rope: true,
        };
        let (kernel, mirror_v, _) = run_case(&case, 3)?;
        // Attention-OUTPUT gate: ≤1 bf16 code per cell (fast-math hot seam),
        // ≥99.9% exactly equal.
        let n_bad = check_within_one_code(&kernel, &mirror_v);
        assert!(
            n_bad * 1000 <= kernel.len(),
            "swa: {n_bad}/{} cells off (> 0.1%)",
            kernel.len()
        );
        Ok(())
    }

    /// Oracle (a2): live RoPE, gated **bit-exact** — the kernel's trig is the
    /// mirrorable `rope_angle`/`ds_sincos` pair (f64 reduction + plain-f32
    /// minimax polynomials), so RoPE introduces no unmirrorable operation.
    #[test]
    #[ignore]
    fn mirror_bit_exact_with_rope() -> Result<()> {
        let case = Case {
            n_win: 50,
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 2,
            zero_rope: false,
        };
        let (kernel, mirror_v, _) = run_case(&case, 42)?;
        let mismatches: Vec<(usize, usize, f32, f32)> = kernel
            .iter()
            .zip(&mirror_v)
            .enumerate()
            .filter(|(_, (a, b))| a.to_bits() != b.to_bits())
            .map(|(i, (a, b))| (i / HEAD_DIM, i % HEAD_DIM, *a, *b))
            .collect();
        assert!(
            mismatches.is_empty(),
            "rope bit mismatches: {}/{}; first 8: {:?}",
            mismatches.len(),
            kernel.len(),
            &mismatches[..mismatches.len().min(8)]
        );
        Ok(())
    }

    /// Oracle (a2) at splits=1 — the exact configuration the tolerance gate
    /// runs, so a kernel↔mirror divergence there is separable from a genuine
    /// quantization-quality regression. Also reports the pairwise deltas.
    #[test]
    #[ignore]
    fn mirror_bit_exact_with_rope_splits1() -> Result<()> {
        let case = Case {
            n_win: 50,
            window_size: 128,
            g_total: 10,
            ratio: 4,
            selected: vec![1, 3, 7],
            num_splits: 1,
            zero_rope: false,
        };
        let (kernel, mirror_v, reference) = run_case(&case, 42)?;
        eprintln!(
            "Δ(kernel,mirror)={} Δ(mirror,ref)={} Δ(kernel,ref)={}",
            max_abs_diff(&kernel, &mirror_v),
            max_abs_diff(&mirror_v, &reference),
            max_abs_diff(&kernel, &reference)
        );
        let nope_max = kernel
            .iter()
            .zip(&reference)
            .enumerate()
            .filter(|(i, _)| i % HEAD_DIM < NOPE_DIM)
            .map(|(_, (a, b))| (a - b).abs())
            .fold(0.0f32, f32::max);
        let rope_max = kernel
            .iter()
            .zip(&reference)
            .enumerate()
            .filter(|(i, _)| i % HEAD_DIM >= NOPE_DIM)
            .map(|(_, (a, b))| (a - b).abs())
            .fold(0.0f32, f32::max);
        eprintln!("kernel-vs-ref: nope_max={nope_max} rope_max={rope_max}");
        let mismatches = kernel
            .iter()
            .zip(&mirror_v)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(mismatches, 0, "kernel≠mirror at splits=1 live rope");
        Ok(())
    }

    /// Oracle (b): the float sink-softmax reference at int8-scale tolerance —
    /// the model-quality gate.
    #[test]
    #[ignore]
    fn float_reference_tolerance() -> Result<()> {
        for (seed, zero_rope) in [(42u64, false), (11, true)] {
            let case = Case {
                n_win: 50,
                window_size: 128,
                g_total: 10,
                ratio: 4,
                selected: vec![1, 3, 7],
                num_splits: 1,
                zero_rope,
            };
            let (kernel, _, reference) = run_case(&case, seed)?;
            let scale = reference.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let d = max_abs_diff(&kernel, &reference);
            assert!(
                d < 0.03 * scale.max(1.0),
                "zero_rope={zero_rope}: |Δ| = {d} vs scale {scale}"
            );
        }
        Ok(())
    }
}
