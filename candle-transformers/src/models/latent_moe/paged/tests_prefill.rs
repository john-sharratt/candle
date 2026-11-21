//! Gates the prefill kernel against the decode oracle, for settled and
//! fresh-source queries, across chunk boundaries and mixed formats.
//!
//! Prefill rows over a settled slot must be bit-identical to decode steps, and
//! repeated launches must return the same bits, since the accumulator is
//! fixed-point for exactly that reason. Fresh-source rows and rows that cross a
//! chunk boundary are anchored to an f64 reference instead, because the tile
//! partition reassociates the softmax. Mixed-format slots are held to the same
//! tolerance as FP8 prefill, and a glue scatter must be visible to attention
//! exactly as an authored window is.

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
        bf16_exact, fp8_exact, max_abs_diff, rope_vec, run_kernel_fmt, table_sincos, Case, Inputs,
    };

    /// Rung-2 prefill gate: row `i` of the prefill kernel over a settled slot
    /// holding tokens `[0..n)` must be BIT-IDENTICAL to the decode kernel run
    /// with window `[0..i)` + incoming token `i` — the prefill inherits the
    /// decode's whole proven oracle chain (mirror, float reference, arena
    /// equivalence) row by row.
    /// **The prefill attention kernel must return the same bits twice.**
    ///
    /// Its QK band-collapse sums one contribution per band across NPAL warps
    /// through shared-memory atomics, in an order the hardware does not define
    /// — so a float accumulator there is non-deterministic *by construction*,
    /// not by accident. The accumulator is int32 fixed-point for exactly this
    /// reason, and this is the gate that holds it that way.
    ///
    /// Kernel-level on purpose. Judging this through the model's token streams
    /// cannot separate "prefill wrote different KV" from "decode computed
    /// differently", and mirror-vs-host gates compare one launch against a
    /// reference rather than a launch against itself — neither can see a kernel
    /// that is merely *inconsistent with itself*. This runs the identical launch
    /// repeatedly and compares the outputs bitwise.
    #[test]
    #[ignore]
    fn prefill_kernel_is_bitwise_repeatable() -> Result<()> {
        const RUNS: usize = 8;
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let n = 80usize;
        let window_size = 128usize;
        let mut s = 7u64;
        let dev = Device::new_cuda(0)?;

        let tokens: Vec<[f32; HEAD_DIM]> = (0..n)
            .map(|_| std::array::from_fn(|_| fp8_exact(&mut s)))
            .collect();
        let qs: Vec<[f32; HEAD_DIM]> = (0..n * H)
            .map(|_| std::array::from_fn(|_| bf16_exact(&mut s)))
            .collect();
        let sinks_v: Vec<f32> = (0..H).map(|_| bf16_exact(&mut s) * 0.5).collect();
        let freqs_v: Vec<f32> = yarn_freqs(ROPE_DIM, 10000.0, 0, 1.0, 32.0, 1.0)
            .into_iter()
            .map(|f| f as f32)
            .collect();
        let sinks = Tensor::from_vec(sinks_v, H, &dev)?;
        let freqs = Tensor::from_vec(freqs_v, ROPE_DIM / 2, &dev)?;
        let rope_tab = build_rope_table(&freqs)?;
        let ws = LatentWorkspace::build(&dev)?;
        let slots = SyntheticSlots::build(&dev, std::slice::from_ref(&tokens.to_vec()))?;
        let qf: Vec<f32> = qs.iter().flat_map(|h| h.iter().copied()).collect();
        let q_all = Tensor::from_vec(qf, (n, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let q_pos = Tensor::from_vec((0..n as u32).collect::<Vec<_>>(), n, &dev)?;
        let comp = Tensor::zeros((1, HEAD_DIM), DType::F32, &dev)?;
        let comp_pos = Tensor::zeros(1, DType::U32, &dev)?;
        let comp_idx = Tensor::full(u32::MAX, (n, 1), &dev)?;
        let comp_cnt = Tensor::zeros(n, DType::U32, &dev)?;

        // The corpus cache is built ONCE, outside the loop: rebuilding it per
        // run would fold the quantize pre-pass's own behaviour into the result
        // and leave "which half is it" unanswered.
        let cache = CorpusCache::build(&comp, &comp_pos)?;

        let mut base: Option<Vec<f32>> = None;
        let mut dirty = 0usize;
        for r in 0..RUNS {
            let out = paged_latent_prefill(
                &q_all,
                &slots.headers,
                &q_pos,
                None,
                &cache,
                &comp_idx,
                &comp_cnt,
                &sinks,
                &rope_tab,
                &ws,
                softmax_scale,
                window_size,
                1,
                fp8_store_tag(),
                &desc::scope(&dev)?,
            )?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            match &base {
                None => base = Some(out),
                Some(b) => {
                    let n_diff = b
                        .iter()
                        .zip(&out)
                        .filter(|(x, y)| x.to_bits() != y.to_bits())
                        .count();
                    let worst = b
                        .iter()
                        .zip(&out)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0f32, f32::max);
                    if n_diff > 0 {
                        dirty += 1;
                        eprintln!(
                            "[prefill-kernel] run0 vs run{r}: {n_diff}/{} differ, max|Δ|={worst:.3e}",
                            b.len()
                        );
                    }
                }
            }
        }
        assert_eq!(
            dirty,
            0,
            "the prefill attention kernel returned different bits for identical inputs in \
             {dirty}/{} repeats",
            RUNS - 1
        );
        Ok(())
    }

    #[test]
    #[ignore]
    fn prefill_rows_equal_decode_steps() -> Result<()> {
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        // n=80 spans THREE 32-token chunks (0..31 | 32..63 | 64..79), so the
        // batched prefill's arena slice-walk crosses ≥2 chunk boundaries — the
        // Phase-3 prerequisite that the multi-query launch reproduces per-token
        // decode across chunk transitions, not just within a single chunk.
        let n = 80usize;
        let window_size = 128usize;
        let mut s = 42u64;
        let dev = Device::new_cuda(0)?;

        let tokens: Vec<[f32; HEAD_DIM]> = (0..n)
            .map(|_| std::array::from_fn(|_| fp8_exact(&mut s)))
            .collect();
        let qs: Vec<[f32; HEAD_DIM]> = (0..n * H)
            .map(|_| std::array::from_fn(|_| bf16_exact(&mut s)))
            .collect();
        let sinks_v: Vec<f32> = (0..H).map(|_| bf16_exact(&mut s) * 0.5).collect();
        let freqs_v: Vec<f32> = yarn_freqs(ROPE_DIM, 10000.0, 0, 1.0, 32.0, 1.0)
            .into_iter()
            .map(|f| f as f32)
            .collect();
        let sinks = Tensor::from_vec(sinks_v.clone(), H, &dev)?;
        let freqs = Tensor::from_vec(freqs_v.clone(), ROPE_DIM / 2, &dev)?;
        let rope_tab = build_rope_table(&freqs)?;
        let ws = LatentWorkspace::build(&dev)?;

        // Prefill: one settled slot holding all n tokens.
        let slots = SyntheticSlots::build(&dev, std::slice::from_ref(&tokens.to_vec()))?;
        let qf: Vec<f32> = qs.iter().flat_map(|h| h.iter().copied()).collect();
        let q_all = Tensor::from_vec(qf, (n, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let q_pos = Tensor::from_vec((0..n as u32).collect::<Vec<_>>(), n, &dev)?;
        let comp = Tensor::zeros((1, HEAD_DIM), DType::F32, &dev)?;
        let comp_pos = Tensor::zeros(1, DType::U32, &dev)?;
        let comp_idx = Tensor::full(u32::MAX, (n, 1), &dev)?;
        let comp_cnt = Tensor::zeros(n, DType::U32, &dev)?;
        let prefill = paged_latent_prefill(
            &q_all,
            &slots.headers,
            &q_pos,
            None,
            &CorpusCache::build(&comp, &comp_pos)?,
            &comp_idx,
            &comp_cnt,
            &sinks,
            &rope_tab,
            &ws,
            softmax_scale,
            window_size,
            1,
            fp8_store_tag(),
            &desc::scope(&dev)?,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

        // Fresh-source equivalence: arena holds only [0..k); rows [k..n) come
        // in as this-layer fresh latents. Queries [k..n) must be BIT-IDENTICAL
        // to the settled-slot run (the kernel FP8-round-trips fresh keys so the
        // bits match what the arena would return). split_at=40 puts the arena
        // prefix across the 0/1 chunk boundary and the fresh rows across the
        // 1/2 boundary, so both sources are exercised over multiple chunks.
        let split_at = 40usize;
        let fslots =
            SyntheticSlots::build(&dev, std::slice::from_ref(&tokens[..split_at].to_vec()))?;
        let fresh_vals: Vec<f32> = tokens[split_at..]
            .iter()
            .flat_map(|t| t.iter().copied())
            .collect();
        let kv_new =
            Tensor::from_vec(fresh_vals, (n - split_at, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let qf_tail: Vec<f32> = qs[split_at * H..]
            .iter()
            .flat_map(|h| h.iter().copied())
            .collect();
        let q_tail =
            Tensor::from_vec(qf_tail, (n - split_at, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let pos_tail = Tensor::from_vec(
            (split_at as u32..n as u32).collect::<Vec<_>>(),
            n - split_at,
            &dev,
        )?;
        let idx_tail = Tensor::full(u32::MAX, (n - split_at, 1), &dev)?;
        let cnt_tail = Tensor::zeros(n - split_at, DType::U32, &dev)?;
        let fresh_out = paged_latent_prefill(
            &q_tail,
            &fslots.headers,
            &pos_tail,
            Some((&kv_new, split_at)),
            &CorpusCache::build(&comp, &comp_pos)?,
            &idx_tail,
            &cnt_tail,
            &sinks,
            &rope_tab,
            &ws,
            softmax_scale,
            window_size,
            1,
            fp8_store_tag(),
            &desc::scope(&dev)?,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
        // Key VALUES are bit-identical (the kernel FP8-round-trips fresh keys),
        // but the 8-key TILE PARTITION shifts (the arena's 20-token chunk pads
        // its last tile; fresh rows tile separately), so the tile-batched
        // softmax accumulates in a different association order — the benign
        // few-ulp variance any split-KV repartition has. Gate: anchor BOTH
        // runs against an f64 sink-softmax reference over the same
        // FP8-round-tripped keys — the fresh path must sit no farther from
        // truth than the settled path does.
        let settled_tail = &prefill[split_at * H * HEAD_DIM..];
        let stored: Vec<[f32; HEAD_DIM]> = tokens
            .iter()
            .map(|t| std::array::from_fn(|d| e4m3_to_f32(f32_to_e4m3(t[d]))))
            .collect();
        let mut ref_tail: Vec<f32> = Vec::with_capacity((n - split_at) * H * HEAD_DIM);
        for i in split_at..n {
            let mut keys: Vec<[f32; HEAD_DIM]> = Vec::with_capacity(i + 1);
            for (t, tok) in stored.iter().enumerate().take(i + 1) {
                let mut l = *tok;
                rope_vec(&mut l, t as u32, &freqs_v);
                keys.push(l);
            }
            for h in 0..H {
                let mut qv = qs[i * H + h];
                rope_vec(&mut qv, i as u32, &freqs_v);
                let mut logits: Vec<f64> = keys
                    .iter()
                    .map(|k| {
                        let mut acc = 0.0f64;
                        for d in 0..HEAD_DIM {
                            acc += qv[d] as f64 * k[d] as f64;
                        }
                        acc * softmax_scale as f64
                    })
                    .collect();
                logits.push(sinks_v[h] as f64);
                let m = logits.iter().cloned().fold(f64::MIN, f64::max);
                let exps: Vec<f64> = logits.iter().map(|&v| (v - m).exp()).collect();
                let z: f64 = exps.iter().sum();
                let mut val = [0.0f64; HEAD_DIM];
                for (t, key) in keys.iter().enumerate() {
                    let p = exps[t] / z;
                    for d in 0..HEAD_DIM {
                        val[d] += p * key[d] as f64;
                    }
                }
                for k in 0..ROPE_DIM / 2 {
                    let (s, c) = table_sincos(i as u32, freqs_v[k]);
                    let d = NOPE_DIM + 2 * k;
                    let (x0, x1) = (val[d], val[d + 1]);
                    val[d] = x0 * c as f64 + x1 * s as f64;
                    val[d + 1] = x1 * c as f64 - x0 * s as f64;
                }
                ref_tail.extend(val.iter().map(|&v| v as f32));
            }
        }
        let d_settled = max_abs_diff(settled_tail, &ref_tail);
        let d_fresh = max_abs_diff(&fresh_out, &ref_tail);
        eprintln!(
            "[fresh] settled-vs-ref {d_settled:.6}, fresh-vs-ref {d_fresh:.6}, \
             settled-vs-fresh {:.6}",
            max_abs_diff(settled_tail, &fresh_out)
        );
        assert!(
            d_fresh <= d_settled * 1.5 + 1e-3,
            "fresh path drifts beyond the settled path's own truth distance: \
             fresh {d_fresh} vs settled {d_settled}"
        );

        // Decode oracle per row (spot rows — each is a full kernel launch).
        // Rows 31/32/33 straddle the 0/1 chunk boundary and 63/64/65 straddle
        // the 1/2 boundary; 79 is the final row of chunk 2.
        for &i in &[0usize, 1, 17, 31, 32, 33, 49, 63, 64, 65, 79] {
            let window: Vec<[f32; HEAD_DIM]> = tokens[..i].to_vec();
            let dslots = SyntheticSlots::build(&dev, std::slice::from_ref(&window))?;
            let qi: Vec<f32> = qs[i * H..(i + 1) * H]
                .iter()
                .flat_map(|h| h.iter().copied())
                .collect();
            let q_one = Tensor::from_vec(qi, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let kv_new =
                Tensor::from_vec(tokens[i].to_vec(), (1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let d_idx = Tensor::full(u32::MAX, (1, 1), &dev)?;
            let d_cnt = Tensor::zeros(1, DType::U32, &dev)?;
            let dec = paged_latent_decode(
                &q_one,
                &dslots.headers,
                &kv_new,
                &CorpusCache::build(&comp, &comp_pos)?,
                &d_idx,
                &d_cnt,
                &Tensor::from_vec(vec![i as u32], 1, &dev)?,
                &sinks,
                &rope_tab,
                &ws,
                softmax_scale,
                window_size,
                1,
                None,
            )?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            // Prefill runs int8 tensor-core PV; decode runs scalar bf16 PV.
            // Both are valid approximations of the same attention, so they
            // agree at int8-PV tolerance (~0.4-1% of scale), not bitwise.
            let row = &prefill[i * H * HEAD_DIM..(i + 1) * H * HEAD_DIM];
            let scale = dec.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
            let d = max_abs_diff(row, &dec);
            assert!(
                d < 0.03 * scale,
                "row {i}: prefill(int8 PV) vs decode(scalar PV) |Δ|={d} ≥ {}",
                0.03 * scale
            );
        }
        Ok(())
    }

    /// Chunked-prefill gate: launches whose query count crosses the workspace
    /// chunk boundary (`q_chunk = WORKSPACE_CAP / H` at splits=1) must stay
    /// correct on BOTH sides of the boundary — settled rows bit-identical to
    /// the decode oracle, fresh-source rows anchored to an f64 reference no
    /// looser than the settled rows are. Guards the chunk pointer arithmetic
    /// (q/out/q_pos/comp_idx/comp_cnt advances) and the launch-invariant
    /// new-key positions (`new_meta.base + j`, never q_pos-derived — a
    /// chunk-shifted q_pos read here corrupts every fresh key past the
    /// boundary and reads out of bounds).
    #[test]
    #[ignore]
    fn prefill_chunked_rows_match_decode() -> Result<()> {
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let n = 600usize;
        let split_at = 40usize; // fresh call: 560 rows → chunks [0,512) | [512,560)
        let window_size = 128usize;
        let q_chunk = WORKSPACE_CAP / H; // splits = 1
        assert!(
            n > q_chunk && n - split_at > q_chunk,
            "both calls must chunk"
        );
        let mut s = 7u64;
        let dev = Device::new_cuda(0)?;

        let tokens: Vec<[f32; HEAD_DIM]> = (0..n)
            .map(|_| std::array::from_fn(|_| fp8_exact(&mut s)))
            .collect();
        let qs: Vec<[f32; HEAD_DIM]> = (0..n * H)
            .map(|_| std::array::from_fn(|_| bf16_exact(&mut s)))
            .collect();
        let sinks_v: Vec<f32> = (0..H).map(|_| bf16_exact(&mut s) * 0.5).collect();
        let freqs_v: Vec<f32> = yarn_freqs(ROPE_DIM, 10000.0, 0, 1.0, 32.0, 1.0)
            .into_iter()
            .map(|f| f as f32)
            .collect();
        let sinks = Tensor::from_vec(sinks_v.clone(), H, &dev)?;
        let freqs = Tensor::from_vec(freqs_v.clone(), ROPE_DIM / 2, &dev)?;
        let rope_tab = build_rope_table(&freqs)?;
        let ws = LatentWorkspace::build(&dev)?;
        let comp = Tensor::zeros((1, HEAD_DIM), DType::F32, &dev)?;
        let comp_pos = Tensor::zeros(1, DType::U32, &dev)?;

        // Settled full prefill: n rows over a slot holding all n tokens.
        let slots = SyntheticSlots::build(&dev, std::slice::from_ref(&tokens.to_vec()))?;
        let qf: Vec<f32> = qs.iter().flat_map(|h| h.iter().copied()).collect();
        let q_all = Tensor::from_vec(qf, (n, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let q_pos = Tensor::from_vec((0..n as u32).collect::<Vec<_>>(), n, &dev)?;
        let comp_idx = Tensor::full(u32::MAX, (n, 1), &dev)?;
        let comp_cnt = Tensor::zeros(n, DType::U32, &dev)?;
        let prefill = paged_latent_prefill(
            &q_all,
            &slots.headers,
            &q_pos,
            None,
            &CorpusCache::build(&comp, &comp_pos)?,
            &comp_idx,
            &comp_cnt,
            &sinks,
            &rope_tab,
            &ws,
            softmax_scale,
            window_size,
            1,
            fp8_store_tag(),
            &desc::scope(&dev)?,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

        // Settled rows straddling the launch-chunk boundary are BIT-IDENTICAL
        // to the decode oracle (inherits the decode's proven oracle chain).
        for &i in &[q_chunk - 1, q_chunk, n - 1] {
            let window: Vec<[f32; HEAD_DIM]> = tokens[..i].to_vec();
            let dslots = SyntheticSlots::build(&dev, std::slice::from_ref(&window))?;
            let qi: Vec<f32> = qs[i * H..(i + 1) * H]
                .iter()
                .flat_map(|h| h.iter().copied())
                .collect();
            let q_one = Tensor::from_vec(qi, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let kv_new =
                Tensor::from_vec(tokens[i].to_vec(), (1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let d_idx = Tensor::full(u32::MAX, (1, 1), &dev)?;
            let d_cnt = Tensor::zeros(1, DType::U32, &dev)?;
            let dec = paged_latent_decode(
                &q_one,
                &dslots.headers,
                &kv_new,
                &CorpusCache::build(&comp, &comp_pos)?,
                &d_idx,
                &d_cnt,
                &Tensor::from_vec(vec![i as u32], 1, &dev)?,
                &sinks,
                &rope_tab,
                &ws,
                softmax_scale,
                window_size,
                1,
                None,
            )?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
            // int8 tensor-core PV (prefill) vs scalar bf16 PV (decode): agree
            // at tolerance, not bitwise (see prefill_rows_equal_decode_steps).
            let row = &prefill[i * H * HEAD_DIM..(i + 1) * H * HEAD_DIM];
            let scale = dec.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
            let d = max_abs_diff(row, &dec);
            assert!(
                d < 0.03 * scale,
                "settled row {i}: prefill(int8 PV) vs decode |Δ|={d} ≥ {}",
                0.03 * scale
            );
        }

        // Fresh-source call: arena holds [0..split_at), rows [split_at..n)
        // arrive as fresh latents; 560 queries chunk at fresh-call row 512
        // (absolute row split_at + 512). Spot rows bracket that boundary.
        let fslots =
            SyntheticSlots::build(&dev, std::slice::from_ref(&tokens[..split_at].to_vec()))?;
        let fresh_vals: Vec<f32> = tokens[split_at..]
            .iter()
            .flat_map(|t| t.iter().copied())
            .collect();
        let kv_new =
            Tensor::from_vec(fresh_vals, (n - split_at, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let qf_tail: Vec<f32> = qs[split_at * H..]
            .iter()
            .flat_map(|h| h.iter().copied())
            .collect();
        let q_tail =
            Tensor::from_vec(qf_tail, (n - split_at, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let pos_tail = Tensor::from_vec(
            (split_at as u32..n as u32).collect::<Vec<_>>(),
            n - split_at,
            &dev,
        )?;
        let idx_tail = Tensor::full(u32::MAX, (n - split_at, 1), &dev)?;
        let cnt_tail = Tensor::zeros(n - split_at, DType::U32, &dev)?;
        let fresh_out = paged_latent_prefill(
            &q_tail,
            &fslots.headers,
            &pos_tail,
            Some((&kv_new, split_at)),
            &CorpusCache::build(&comp, &comp_pos)?,
            &idx_tail,
            &cnt_tail,
            &sinks,
            &rope_tab,
            &ws,
            softmax_scale,
            window_size,
            1,
            fp8_store_tag(),
            &desc::scope(&dev)?,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

        // f64 sink-softmax anchor at the spot rows (window-clamped): the fresh
        // path must sit no farther from truth than the settled path (which is
        // bit-exact to decode) — tile-partition reassociation is the only
        // allowed variance.
        let stored: Vec<[f32; HEAD_DIM]> = tokens
            .iter()
            .map(|t| std::array::from_fn(|d| e4m3_to_f32(f32_to_e4m3(t[d]))))
            .collect();
        let boundary = split_at + q_chunk;
        for &i in &[boundary - 2, boundary - 1, boundary, boundary + 1, n - 1] {
            let lo = (i + 1).saturating_sub(window_size);
            let mut keys: Vec<[f32; HEAD_DIM]> = Vec::with_capacity(i + 1 - lo);
            for t in lo..=i {
                let mut l = stored[t];
                rope_vec(&mut l, t as u32, &freqs_v);
                keys.push(l);
            }
            let mut ref_row: Vec<f32> = Vec::with_capacity(H * HEAD_DIM);
            for h in 0..H {
                let mut qv = qs[i * H + h];
                rope_vec(&mut qv, i as u32, &freqs_v);
                let mut logits: Vec<f64> = keys
                    .iter()
                    .map(|k| {
                        let mut acc = 0.0f64;
                        for d in 0..HEAD_DIM {
                            acc += qv[d] as f64 * k[d] as f64;
                        }
                        acc * softmax_scale as f64
                    })
                    .collect();
                logits.push(sinks_v[h] as f64);
                let m = logits.iter().cloned().fold(f64::MIN, f64::max);
                let exps: Vec<f64> = logits.iter().map(|&v| (v - m).exp()).collect();
                let z: f64 = exps.iter().sum();
                let mut val = [0.0f64; HEAD_DIM];
                for (t, key) in keys.iter().enumerate() {
                    let p = exps[t] / z;
                    for d in 0..HEAD_DIM {
                        val[d] += p * key[d] as f64;
                    }
                }
                for k in 0..ROPE_DIM / 2 {
                    let (sn, cs) = table_sincos(i as u32, freqs_v[k]);
                    let d = NOPE_DIM + 2 * k;
                    let (x0, x1) = (val[d], val[d + 1]);
                    val[d] = x0 * cs as f64 + x1 * sn as f64;
                    val[d + 1] = x1 * cs as f64 - x0 * sn as f64;
                }
                ref_row.extend(val.iter().map(|&v| v as f32));
            }
            let settled_row = &prefill[i * H * HEAD_DIM..(i + 1) * H * HEAD_DIM];
            let fresh_row =
                &fresh_out[(i - split_at) * H * HEAD_DIM..(i - split_at + 1) * H * HEAD_DIM];
            let d_settled = max_abs_diff(settled_row, &ref_row);
            let d_fresh = max_abs_diff(fresh_row, &ref_row);
            eprintln!("[chunk] row {i}: settled-vs-ref {d_settled:.6} fresh-vs-ref {d_fresh:.6}");
            assert!(
                d_fresh <= d_settled * 1.5 + 1e-3,
                "row {i}: fresh path drifts beyond the settled path's truth distance: \
                 fresh {d_fresh} vs settled {d_settled}"
            );
        }
        Ok(())
    }

    /// The PREFILL kernel's adaptive window dispatch, gated through the decode
    /// oracle chain: prefill rows over a settled slot whose SEALED chunks are
    /// quant-authored must match per-row decode over slots authored with the
    /// SAME sealed specs, within the established int8-PV-vs-scalar-PV envelope
    /// (`d < 0.03·scale`, as the FP8 prefill≡decode gates). Only writer-chunk
    /// rows (64..80) compare — there both views' chunk roles coincide (chunks
    /// 0-1 sealed quant, chunk 2 the FP8 writer), so both kernels read
    /// byte-identical arenas and the residual is purely the PV algorithm.
    #[test]
    #[ignore]
    fn prefill_mixed_rows_equal_decode_steps() -> Result<()> {
        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let n = 80usize;
        let window_size = 128usize;
        let mut s = 7u64;
        let dev = Device::new_cuda(0)?;

        let tokens: Vec<[f32; HEAD_DIM]> = (0..n)
            .map(|_| std::array::from_fn(|_| fp8_exact(&mut s)))
            .collect();
        let qs: Vec<[f32; HEAD_DIM]> = (0..n * H)
            .map(|_| std::array::from_fn(|_| bf16_exact(&mut s)))
            .collect();
        let sinks_v: Vec<f32> = (0..H).map(|_| bf16_exact(&mut s) * 0.5).collect();
        let freqs_v: Vec<f32> = yarn_freqs(ROPE_DIM, 10000.0, 0, 1.0, 32.0, 1.0)
            .into_iter()
            .map(|f| f as f32)
            .collect();

        let q8 = |outer| BandSpec { fmt: 7, outer };
        let q4 = |outer| BandSpec { fmt: 15, outer };
        let t8 = |a: [BandSpec; 8]| -> [BandSpec; N_BANDS] { std::array::from_fn(|i| a[i % 8]) };
        let sealed = [
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
        ];
        let specs = vec![sealed[0], sealed[1], [BandSpec::default(); N_BANDS]];

        // Prefill over the mixed settled slot.
        let slots = SyntheticSlots::build_mixed(
            &dev,
            std::slice::from_ref(&tokens.to_vec()),
            &[0usize],
            std::slice::from_ref(&specs),
        )?;
        let sinks = Tensor::from_vec(sinks_v.clone(), H, &dev)?;
        let freqs = Tensor::from_vec(freqs_v.clone(), ROPE_DIM / 2, &dev)?;
        let rope_tab = build_rope_table(&freqs)?;
        let ws = LatentWorkspace::build(&dev)?;
        let qf: Vec<f32> = qs.iter().flat_map(|h| h.iter().copied()).collect();
        let q_all = Tensor::from_vec(qf, (n, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
        let q_pos = Tensor::from_vec((0..n as u32).collect::<Vec<_>>(), n, &dev)?;
        let comp = Tensor::zeros((1, HEAD_DIM), DType::F32, &dev)?;
        let comp_pos = Tensor::zeros(1, DType::U32, &dev)?;
        let comp_idx = Tensor::full(u32::MAX, (n, 1), &dev)?;
        let comp_cnt = Tensor::zeros(n, DType::U32, &dev)?;
        let prefill = paged_latent_prefill(
            &q_all,
            &slots.headers,
            &q_pos,
            None,
            &CorpusCache::build(&comp, &comp_pos)?,
            &comp_idx,
            &comp_cnt,
            &sinks,
            &rope_tab,
            &ws,
            softmax_scale,
            window_size,
            1,
            fp8_store_tag(),
            &desc::scope(&dev)?,
        )?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;

        // Per-row decode over slots with the SAME sealed specs (row i's view:
        // chunks 0-1 sealed, chunk 2 = writer — identical bytes to the
        // prefill slot for every token < i).
        for i in 64..n {
            let case = Case {
                n_win: i,
                window_size,
                g_total: 0,
                ratio: 4,
                selected: vec![],
                num_splits: 1,
                zero_rope: false,
            };
            let inp = Inputs {
                window: tokens[..i].to_vec(),
                kv_new: tokens[i],
                q: qs[i * H..(i + 1) * H].to_vec(),
                comp: vec![],
                comp_pos: vec![],
                sinks: sinks_v.clone(),
                freqs: freqs_v.clone(),
            };
            let dec = run_kernel_fmt(&case, &inp, softmax_scale, Some(&specs))?;
            let row = &prefill[i * H * HEAD_DIM..(i + 1) * H * HEAD_DIM];
            // Same tolerance as the FP8 prefill≡decode gates: the prefill's
            // int8 tensor-core PV vs the decode's scalar PV is a small bounded
            // envelope; the mixed formats must not widen it.
            let scale = dec.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-6);
            let d = max_abs_diff(row, &dec);
            assert!(
                d < 0.03 * scale,
                "mixed row {i}: prefill(int8 PV) vs decode |Δ|={d} ≥ {}",
                0.03 * scale
            );
        }
        Ok(())
    }

    /// Glue-scatter readback gate: a latent written into its reserved gap slot
    /// by `paged_latent_glue_scatter` must be BIT-IDENTICAL, through the
    /// decode kernel, to the same latent authored host-side into the arena.
    /// This is the read-after-scatter-through-attention proof the wave path's
    /// phase-A glue relies on: scatter targets, band pointer math, FP8
    /// conversion, and window visibility of the scattered token.
    #[test]
    #[ignore]
    fn glue_scatter_equals_authored_window() -> Result<()> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::Storage;

        let softmax_scale = (HEAD_DIM as f64).powf(-0.5) as f32;
        let n = 40usize; // 2 chunks: [0..32) settled + [32..40) writer
        let glue_slots = [3usize, 35]; // one per chunk, mid-chunk targets
        let mut s = 7u64;
        let dev = Device::new_cuda(0)?;

        let tokens: Vec<[f32; HEAD_DIM]> = (0..n)
            .map(|_| std::array::from_fn(|_| fp8_exact(&mut s)))
            .collect();
        let q: Vec<[f32; HEAD_DIM]> = (0..H)
            .map(|_| std::array::from_fn(|_| bf16_exact(&mut s)))
            .collect();
        let kv_new: [f32; HEAD_DIM] = std::array::from_fn(|_| fp8_exact(&mut s));
        let sinks_v: Vec<f32> = (0..H).map(|_| bf16_exact(&mut s) * 0.5).collect();
        let freqs_v: Vec<f32> = yarn_freqs(ROPE_DIM, 10000.0, 0, 1.0, 32.0, 1.0)
            .into_iter()
            .map(|f| f as f32)
            .collect();

        let run = |window: &Vec<[f32; HEAD_DIM]>, scatter: bool| -> Result<Vec<f32>> {
            let slots = SyntheticSlots::build(&dev, std::slice::from_ref(window))?;
            if scatter {
                let cuda = match &dev {
                    Device::Cuda(c) => c.clone(),
                    _ => unreachable!(),
                };
                let stream = cuda.cuda_stream();
                let headers_addr = {
                    let (storage, _) = slots.headers.storage_and_layout();
                    match &*storage {
                        Storage::Cuda(c) => {
                            let slice = c.as_cuda_slice::<u8>()?;
                            let (p, _g) = slice.device_ptr(&stream);
                            p
                        }
                        _ => unreachable!(),
                    }
                };
                let mut lat = Vec::with_capacity(glue_slots.len() * HEAD_DIM);
                for &g in &glue_slots {
                    lat.extend_from_slice(&tokens[g]);
                }
                let kv = Tensor::from_vec(lat, (glue_slots.len(), HEAD_DIM), &dev)?
                    .to_dtype(DType::BF16)?;
                let slices: Vec<u32> = glue_slots.iter().map(|&g| (g / CHUNK) as u32).collect();
                let in_blk: Vec<u32> = glue_slots.iter().map(|&g| (g % CHUNK) as u32).collect();
                paged_latent_glue_scatter(
                    &[GlueRun {
                        kv,
                        headers_ptr: headers_addr,
                        slices,
                        in_blk,
                    }],
                    &desc::scope(&dev)?,
                )?;
            }
            let qf: Vec<f32> = q.iter().flat_map(|h| h.iter().copied()).collect();
            let qt = Tensor::from_vec(qf, (1, H, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let kvt =
                Tensor::from_vec(kv_new.to_vec(), (1, HEAD_DIM), &dev)?.to_dtype(DType::BF16)?;
            let comp = Tensor::zeros((1, HEAD_DIM), DType::F32, &dev)?;
            let comp_pos = Tensor::zeros(1, DType::U32, &dev)?;
            let comp_idx = Tensor::from_vec(vec![u32::MAX], (1, 1), &dev)?;
            let comp_cnt = Tensor::from_vec(vec![0u32], 1, &dev)?;
            let sinks = Tensor::from_vec(sinks_v.clone(), H, &dev)?;
            let freqs = Tensor::from_vec(freqs_v.clone(), ROPE_DIM / 2, &dev)?;
            let rope_tab = build_rope_table(&freqs)?;
            let ws = LatentWorkspace::build(&dev)?;
            let out = paged_latent_decode(
                &qt,
                &slots.headers,
                &kvt,
                &CorpusCache::build(&comp, &comp_pos)?,
                &comp_idx,
                &comp_cnt,
                &Tensor::from_vec(vec![window.len() as u32], 1, &dev)?,
                &sinks,
                &rope_tab,
                &ws,
                softmax_scale,
                128,
                2,
                None,
            )?;
            // Kernel emits F32; round to bf16 to match the bf16 mirror.
            out.to_dtype(DType::BF16)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()
        };

        // Path A: every token authored host-side.
        let authored = run(&tokens, false)?;
        // Path B: the glue slots authored as ZERO, then filled by the scatter
        // kernel from the true latents.
        let mut gapped = tokens.clone();
        for &g in &glue_slots {
            gapped[g] = [0.0; HEAD_DIM];
        }
        let scattered = run(&gapped, true)?;

        let mismatches = authored
            .iter()
            .zip(&scattered)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            mismatches,
            0,
            "scattered glue diverges from authored window: {mismatches}/{} values",
            authored.len()
        );
        // Sanity: the gap actually mattered (the zeroed run differs), so the
        // scatter path — not luck — produced the agreement.
        let unfilled = run(&gapped, false)?;
        assert_ne!(
            authored, unfilled,
            "test degenerate: zeroed glue slots did not change the output"
        );
        Ok(())
    }
}
