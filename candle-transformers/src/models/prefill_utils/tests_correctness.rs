//! Checks the paged prefill kernel against a reference attention, with and
//! without a prefix, with RoPE, and across grouped-query shapes.
//!
//! The reference rotates the inputs and runs plain attention, so in-kernel RoPE
//! and the prefix path are each compared with an independent answer. Further
//! tests append chunks between two prefills on one slot, and give a later chunk a
//! shuffled palette map, where a stale routing would show as a permutation of
//! the values.

#[cfg(all(test, feature = "cuda"))]
pub(super) mod tests {
    #![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

    /// Serialize the GPU tests: the split-KV launcher's grow-on-demand
    /// partial pool is a function-local static sized for the production
    /// single-scheduler-thread model — concurrent test launches interleave
    /// its free/realloc. So does the region pool every `KvCache` here draws
    /// from, which is why the lock is crate-wide rather than module-local.
    use crate::models::gpu_test_lock::gpu_serial;

    use candle::{DType, Device, Result, Tensor};

    use candle_nn::kv_cache::KvCache;

    use super::super::tests_support::tests::{
        apply_rope_to_vec, make_test_rope, make_zero_rope, max_abs_error, mean_abs_error,
        paged_prefill_uniform, reference_attention, run_paged_prefill, run_paged_prefill_with_rope,
        run_prefill_with_rope, test_inv_freq,
    };

    // Prefill correctness: no prefix (offset=0)

    #[test]
    fn correctness_prefill_no_prefix_bf16() -> Result<()> {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        for &(n_head, n_kv_head, head_dim, seq_len, label) in &[
            (8, 8, 64, 4, "MHA hd64 short"),
            (8, 8, 64, 32, "MHA hd64 medium"),
            (8, 8, 64, 128, "MHA hd64 long"),
            (8, 8, 128, 4, "MHA hd128 short"),
            (8, 8, 128, 64, "MHA hd128 long"),
            (32, 8, 64, 4, "GQA 32/8 hd64 short"),
            (32, 8, 64, 32, "GQA 32/8 hd64 medium"),
            (32, 8, 64, 128, "GQA 32/8 hd64 long"),
            (32, 8, 128, 4, "GQA 32/8 hd128 short"),
            (32, 8, 128, 64, "GQA 32/8 hd128 long"),
            (40, 8, 64, 4, "GQA 40/8 hd64 short"), // num_groups=5, tricky for WARPS_TC
            (40, 8, 64, 32, "GQA 40/8 hd64 medium"),
            (40, 8, 128, 4, "GQA 40/8 hd128 short"),
            (14, 2, 64, 4, "GQA 14/2 hd64 short"), // Qwen2-style
            (14, 2, 64, 32, "GQA 14/2 hd64 medium"),
            (14, 2, 128, 4, "GQA 14/2 hd128 short"),
            (8, 1, 64, 4, "MQA hd64 short"),
            (8, 1, 64, 32, "MQA hd64 medium"),
            (8, 1, 128, 4, "MQA hd128 short"),
        ] {
            let b_sz = 1;
            let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            let paged_out = run_paged_prefill(
                &q, &k, &v, b_sz, seq_len, n_head, n_kv_head, head_dim, 0, dtype,
            )?;

            // Reference: for no-prefix, full KV == the input K/V
            let ref_out = reference_attention(&q, &k, &v, n_head, n_kv_head, head_dim, 0)?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            // BF16 tolerance: mean ~1e-2, max ~5e-2
            assert!(
                mae < 0.05,
                "[{label}] BF16 prefill mean error too large: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] BF16 prefill max error too large: {max_err}"
            );
            println!("[{label}] BF16 prefill OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    #[test]
    fn correctness_prefill_no_prefix_f16() -> Result<()> {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0)?;
        let dtype = DType::F16;

        for &(n_head, n_kv_head, head_dim, seq_len, label) in &[
            (8, 8, 64, 4, "MHA hd64 short"),
            (8, 8, 64, 32, "MHA hd64 medium"),
            (32, 8, 64, 4, "GQA 32/8 hd64 short"),
            (32, 8, 64, 32, "GQA 32/8 hd64 medium"),
            (32, 8, 128, 4, "GQA 32/8 hd128 short"),
            (40, 8, 64, 4, "GQA 40/8 hd64 short"),
            (14, 2, 64, 4, "GQA 14/2 hd64 short"),
            (8, 1, 64, 4, "MQA hd64 short"),
        ] {
            let b_sz = 1;
            let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            let paged_out = run_paged_prefill(
                &q, &k, &v, b_sz, seq_len, n_head, n_kv_head, head_dim, 0, dtype,
            )?;
            let ref_out = reference_attention(&q, &k, &v, n_head, n_kv_head, head_dim, 0)?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            // F16 tolerance — slightly tighter since F16 has more mantissa bits than BF16
            assert!(
                mae < 0.05,
                "[{label}] F16 prefill mean error too large: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] F16 prefill max error too large: {max_err}"
            );
            println!("[{label}] F16 prefill OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    // Prefill correctness: in-kernel RoPE, both pairings, real table

    /// The kernel applies RoPE internally from the frequency-indexed cos/sin
    /// table; the reference applies the same rotation on the host
    /// (`rotary_emb::rope` for the half-split pairing, `rope_i` for the
    /// interleaved LLaMA pairing) and then runs plain attention. Both pairings
    /// share one table — the pairing only changes which dims form a pair.
    /// The identity-RoPE tests above cover shapes/GQA; this one covers the
    /// rotation itself, which they zero out.
    #[test]
    fn correctness_prefill_rope_pairings() -> Result<()> {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0)?;
        let dtype = DType::F16;

        for &(n_head, n_kv_head, head_dim, seq_len, interleaved, label) in &[
            (8, 8, 64, 32, false, "half-split hd64"),
            (32, 8, 128, 64, false, "half-split GQA hd128"),
            (8, 8, 64, 32, true, "interleaved hd64"),
            (32, 8, 128, 64, true, "interleaved GQA hd128 (llama2 shape)"),
            (32, 8, 128, 4, true, "interleaved short"),
        ] {
            let b_sz = 1;
            let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            let rope = make_test_rope(head_dim, &device)?;
            let paged_out = run_paged_prefill_with_rope(
                &q,
                &k,
                &v,
                b_sz,
                seq_len,
                n_head,
                n_kv_head,
                head_dim,
                0,
                dtype,
                &rope,
                interleaved,
            )?;

            // Host reference: same inv_freq, same positions (0..seq_len),
            // rotation applied to full-precision Q/K before plain attention.
            let half = head_dim / 2;
            let inv_freq = test_inv_freq(head_dim);
            let mut cos_v = vec![0f32; seq_len * half];
            let mut sin_v = vec![0f32; seq_len * half];
            for t in 0..seq_len {
                for i in 0..half {
                    let angle = t as f64 * inv_freq[i] as f64;
                    cos_v[t * half + i] = angle.cos() as f32;
                    sin_v[t * half + i] = angle.sin() as f32;
                }
            }
            let cos = Tensor::from_vec(cos_v, (seq_len, half), &device)?;
            let sin = Tensor::from_vec(sin_v, (seq_len, half), &device)?;
            let q32 = q.to_dtype(DType::F32)?.contiguous()?;
            let k32 = k.to_dtype(DType::F32)?.contiguous()?;
            let (q_r, k_r) = if interleaved {
                (
                    candle_nn::rotary_emb::rope_i(&q32, &cos, &sin)?,
                    candle_nn::rotary_emb::rope_i(&k32, &cos, &sin)?,
                )
            } else {
                (
                    candle_nn::rotary_emb::rope(&q32, &cos, &sin)?,
                    candle_nn::rotary_emb::rope(&k32, &cos, &sin)?,
                )
            };
            let ref_out = reference_attention(&q_r, &k_r, &v, n_head, n_kv_head, head_dim, 0)?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;
            assert!(mae < 0.05, "[{label}] rope prefill mean error: {mae}");
            assert!(max_err < 0.2, "[{label}] rope prefill max error: {max_err}");
            println!("[{label}] rope prefill OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    // Prefill correctness: with prefix (offset > 0, multi-turn)

    #[test]
    fn correctness_prefill_with_prefix_bf16() -> Result<()> {
        let _gpu = gpu_serial();
        use candle_nn::kv_cache::ChunkedKvBacking;

        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        for &(n_head, n_kv_head, head_dim, prefix_len, new_len, label) in &[
            (8, 8, 64, 10, 4, "MHA hd64 prefix=10"),
            (8, 8, 64, 32, 8, "MHA hd64 prefix=32"),
            (32, 8, 64, 10, 4, "GQA 32/8 hd64 prefix=10"),
            (32, 8, 64, 64, 16, "GQA 32/8 hd64 prefix=64"),
            (32, 8, 128, 10, 4, "GQA 32/8 hd128 prefix=10"),
            (40, 8, 64, 10, 4, "GQA 40/8 hd64 prefix=10"),
            (14, 2, 64, 10, 4, "GQA 14/2 hd64 prefix=10"),
            (8, 1, 64, 10, 4, "MQA hd64 prefix=10"),
        ] {
            let b_sz = 1;
            let total_kv = prefix_len + new_len;

            // Create K/V for prefix (already in cache) and new segment
            let prefix_k =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, prefix_len, head_dim), &device)?
                    .to_dtype(dtype)?;
            let prefix_v =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, prefix_len, head_dim), &device)?
                    .to_dtype(dtype)?;
            let new_q = Tensor::randn(0f32, 1f32, (1, n_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let new_k = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let new_v = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            // Set up chunked KV cache and write prefix data
            let backing =
                ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, &device, total_kv)?;
            backing.ensure_for_offset(0, 0, total_kv)?;
            backing.write_contiguous(0, 0, &prefix_k, &prefix_v)?;

            let mut cache0 = KvCache::new(2, 4096);
            cache0.force_dtype(dtype);
            cache0.set_chunked_backing(&backing, 0, None)?;
            cache0.set_current_seq_len(prefix_len)?;

            let offsets = [prefix_len];
            let mut caches: [&mut KvCache; 1] = [&mut cache0];
            let generation = backing.begin_stager_generation_required();

            let out = paged_prefill_uniform(
                &mut caches,
                &offsets,
                &new_q,
                &new_k,
                &new_v,
                b_sz,
                new_len,
                n_head,
                n_kv_head,
                head_dim,
                None,
                &make_zero_rope(head_dim, &device)?,
                false, // rope_interleaved
                &generation,
            )?;
            assert_eq!(out.len(), 1);
            let paged_out = &out[0];

            // Reference: full KV = prefix + new
            let full_k = Tensor::cat(&[&prefix_k, &new_k], 2)?;
            let full_v = Tensor::cat(&[&prefix_v, &new_v], 2)?;
            let ref_out = reference_attention(
                &new_q, &full_k, &full_v, n_head, n_kv_head, head_dim, prefix_len,
            )?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            assert!(
                mae < 0.05,
                "[{label}] BF16 prefill-with-prefix mean error too large: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] BF16 prefill-with-prefix max error too large: {max_err}"
            );
            println!("[{label}] BF16 prefill-with-prefix OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    /// Two prefills on ONE slot, with a chunk boundary crossed between them.
    ///
    /// The header build points the kernel at the backing's persistent
    /// slot-state buffer rather than serialising a private copy, which is only
    /// sound while every mutation of the chunk table forces that buffer to be
    /// re-serialised. The single-launch tests cannot see this: they build
    /// headers once against a buffer that was empty, so every one of them takes
    /// the rebuild path and none exercises reuse at all.
    ///
    /// Here the first prefill leaves the slot's tail chunk part-filled, the
    /// capacity ensure for the second appends chunks, and the second prefill
    /// then reads a prefix that spans the appended ones. If the append does not
    /// invalidate, the second launch walks a slice array that stops short of
    /// the tokens the position map indexes — so the second output is checked
    /// against the reference over the WHOLE prefix, not merely the tail.
    #[test]
    fn correctness_prefill_reuses_slot_state_across_a_chunk_append() -> Result<()> {
        let _gpu = gpu_serial();
        use candle_nn::kv_cache::ChunkedKvBacking;

        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let (n_head, n_kv_head, head_dim) = (8, 8, 64);
        // 40 fills chunk 0 and part of chunk 1; the second segment tops chunk 1
        // up and appends two more, so `push_chunk` fires between the builds.
        let (len1, len2) = (40usize, 64usize);
        let b_sz = 1;
        let total_kv = len1 + len2;

        let mk = |n: usize, heads: usize| -> Result<Tensor> {
            Tensor::randn(0f32, 1f32, (1, heads, n, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()
        };
        let (q1, k1, v1) = (
            mk(len1, n_head)?,
            mk(len1, n_kv_head)?,
            mk(len1, n_kv_head)?,
        );
        let (q2, k2, v2) = (
            mk(len2, n_head)?,
            mk(len2, n_kv_head)?,
            mk(len2, n_kv_head)?,
        );

        let backing = ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, &device, total_kv)?;
        let mut cache0 = KvCache::new(2, 4096);
        cache0.force_dtype(dtype);
        cache0.set_chunked_backing(&backing, 0, None)?;
        cache0.set_current_seq_len(0)?;
        let rope = make_zero_rope(head_dim, &device)?;

        let mut run =
            |offset: usize, len: usize, q: &Tensor, k: &Tensor, v: &Tensor| -> Result<Tensor> {
                let offsets = [offset];
                let mut caches: [&mut KvCache; 1] = [&mut cache0];
                KvCache::ensure_chunked_capacity_batch(&mut caches, &offsets, len)?;
                let generation = backing.begin_stager_generation_required();
                let out = paged_prefill_uniform(
                    &mut caches,
                    &offsets,
                    q,
                    k,
                    v,
                    b_sz,
                    len,
                    n_head,
                    n_kv_head,
                    head_dim,
                    None,
                    &rope,
                    false,
                    &generation,
                )?;
                assert_eq!(out.len(), 1);
                Ok(out[0].clone())
            };

        let out1 = run(0, len1, &q1, &k1, &v1)?;
        let after_first = backing.sequence_block_count(0).unwrap_or(0);
        let out2 = run(len1, len2, &q2, &k2, &v2)?;
        let after_second = backing.sequence_block_count(0).unwrap_or(0);
        // The fixture only tests what it is for if the second capacity ensure
        // actually appended: a backing that pre-allocated every chunk would run
        // both launches over one unchanging table and prove nothing.
        assert!(
            after_second > after_first,
            "fixture no longer crosses a chunk boundary: {after_first} chunks after the \
             first prefill, {after_second} after the second"
        );

        let ref1 = reference_attention(&q1, &k1, &v1, n_head, n_kv_head, head_dim, 0)?;
        let full_k = Tensor::cat(&[&k1, &k2], 2)?;
        let full_v = Tensor::cat(&[&v1, &v2], 2)?;
        let ref2 = reference_attention(&q2, &full_k, &full_v, n_head, n_kv_head, head_dim, len1)?;

        for (label, got, want) in [
            ("first prefill", &out1, &ref1),
            ("second prefill across the append", &out2, &ref2),
        ] {
            let got_f32 = got.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&got_f32, want)?;
            let max_err = max_abs_error(&got_f32, want)?;
            assert!(mae < 0.05, "[{label}] mean error too large: {mae}");
            assert!(max_err < 0.2, "[{label}] max error too large: {max_err}");
            println!("[{label}] OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    /// In-kernel RoPE over a CACHED prefix, both pairings. The arena stores
    /// UNROTATED K and the kernel ropes each cached column at its absolute
    /// position on read — a path the no-prefix pairing test cannot reach. The
    /// reference rotates prefix K at positions `0..prefix_len` and the new
    /// segment at `prefix_len..`, then runs plain attention over the
    /// concatenation. LLaMA's multi-turn flow (system turn, then user turn
    /// over the sealed prefix) runs exactly this shape with the interleaved
    /// pairing.
    #[test]
    fn correctness_prefill_rope_pairings_with_prefix() -> Result<()> {
        let _gpu = gpu_serial();
        use candle_nn::kv_cache::ChunkedKvBacking;

        let device = Device::new_cuda(0)?;
        let dtype = DType::F16;

        for &(n_head, n_kv_head, head_dim, prefix_len, new_len, interleaved, label) in &[
            (8, 8, 64, 32, 8, false, "half-split hd64 prefix=32"),
            (32, 8, 128, 40, 12, false, "half-split GQA hd128 prefix=40"),
            (8, 8, 64, 32, 8, true, "interleaved hd64 prefix=32"),
            (32, 8, 128, 40, 12, true, "interleaved GQA hd128 prefix=40"),
            (
                32,
                8,
                128,
                130,
                24,
                true,
                "interleaved multi-chunk prefix=130",
            ),
        ] {
            let b_sz = 1;
            let total_kv = prefix_len + new_len;

            let prefix_k =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, prefix_len, head_dim), &device)?
                    .to_dtype(dtype)?;
            let prefix_v =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, prefix_len, head_dim), &device)?
                    .to_dtype(dtype)?;
            let new_q = Tensor::randn(0f32, 1f32, (1, n_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let new_k = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let new_v = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            // Arena holds the prefix UNROTATED — the kernel ropes it on read.
            let backing =
                ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, &device, total_kv)?;
            backing.ensure_for_offset(0, 0, total_kv)?;
            backing.write_contiguous(0, 0, &prefix_k, &prefix_v)?;

            let mut cache0 = KvCache::new(2, 4096);
            cache0.force_dtype(dtype);
            cache0.set_chunked_backing(&backing, 0, None)?;
            cache0.set_current_seq_len(prefix_len)?;

            let offsets = [prefix_len];
            let mut caches: [&mut KvCache; 1] = [&mut cache0];
            let generation = backing.begin_stager_generation_required();

            let rope = make_test_rope(head_dim, &device)?;
            let out = paged_prefill_uniform(
                &mut caches,
                &offsets,
                &new_q,
                &new_k,
                &new_v,
                b_sz,
                new_len,
                n_head,
                n_kv_head,
                head_dim,
                None,
                &rope,
                interleaved,
                &generation,
            )?;
            assert_eq!(out.len(), 1);
            let paged_out = &out[0];

            // Reference: rotate prefix + new K at their absolute positions,
            // Q at prefix_len.., then plain attention over the concatenation.
            let half = head_dim / 2;
            let inv_freq = test_inv_freq(head_dim);
            let table = |lo: usize, hi: usize| -> Result<(Tensor, Tensor)> {
                let n = hi - lo;
                let mut cos_v = vec![0f32; n * half];
                let mut sin_v = vec![0f32; n * half];
                for (row, pos) in (lo..hi).enumerate() {
                    for i in 0..half {
                        let angle = pos as f64 * inv_freq[i] as f64;
                        cos_v[row * half + i] = angle.cos() as f32;
                        sin_v[row * half + i] = angle.sin() as f32;
                    }
                }
                Ok((
                    Tensor::from_vec(cos_v, (n, half), &device)?,
                    Tensor::from_vec(sin_v, (n, half), &device)?,
                ))
            };
            let rot = |x: &Tensor, lo: usize, hi: usize| -> Result<Tensor> {
                let (cos, sin) = table(lo, hi)?;
                let x32 = x.to_dtype(DType::F32)?.contiguous()?;
                if interleaved {
                    candle_nn::rotary_emb::rope_i(&x32, &cos, &sin)
                } else {
                    candle_nn::rotary_emb::rope(&x32, &cos, &sin)
                }
            };
            let pk_r = rot(&prefix_k, 0, prefix_len)?;
            let nk_r = rot(&new_k, prefix_len, total_kv)?;
            let q_r = rot(&new_q, prefix_len, total_kv)?;
            let full_k = Tensor::cat(&[&pk_r, &nk_r], 2)?;
            let full_v = Tensor::cat(
                &[
                    &prefix_v.to_dtype(DType::F32)?,
                    &new_v.to_dtype(DType::F32)?,
                ],
                2,
            )?;
            let ref_out = reference_attention(
                &q_r, &full_k, &full_v, n_head, n_kv_head, head_dim, prefix_len,
            )?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;
            assert!(
                mae < 0.05,
                "[{label}] rope prefix-prefill mean error: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] rope prefix-prefill max error: {max_err}"
            );
            println!("[{label}] rope prefix-prefill OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    /// Regression: prefix palette ROUTING across mixed per-slice maps.
    ///
    /// The prefix is a sealed partial chunk 0 (`p` tokens, identity palette
    /// map) followed by a full chunk 1 whose palette map is SHUFFLED
    /// (dim `d` → palette `d % 4`, rank `d / 4`) with its data permuted to
    /// match, so a correct reader reconstructs the same logical values.
    /// Each slice-aligned tile must be routed with ITS OWN slice's map:
    /// routing chunk 1's positions through chunk 0's map (a stale cached
    /// table) gathers a dim-permutation of the true values — order-1
    /// garbage on random data. Float chunks keep the check
    /// quantization-free, so the tolerance is tight fp16 noise and any
    /// routing regression fails hard. The partial chunk 0 also exercises
    /// the gap walk ahead of the shuffled map.
    #[test]
    fn correctness_prefill_straddle_shuffled_pal_map() -> Result<()> {
        let _gpu = gpu_serial();
        use candle_nn::kv_cache::ChunkedKvBacking;

        let device = Device::new_cuda(0)?;
        let dtype = DType::F16;
        const CHUNK: usize = 32;
        let (n_head, n_kv_head, head_dim) = (32usize, 8usize, 128usize);
        let p = 20usize; // partial chunk 0 → a gap before every later slice
        let new_len = 8usize;
        let prefix_len = p + CHUNK;
        let b_sz = 1;

        // Logical prefix K/V and the new segment.
        let k0 = Tensor::randn(0f32, 1f32, (1, n_kv_head, p, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v0 = Tensor::randn(0f32, 1f32, (1, n_kv_head, p, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let k1 = Tensor::randn(0f32, 1f32, (1, n_kv_head, CHUNK, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v1 = Tensor::randn(0f32, 1f32, (1, n_kv_head, CHUNK, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let new_q =
            Tensor::randn(0f32, 1f32, (1, n_head, new_len, head_dim), &device)?.to_dtype(dtype)?;
        let new_k = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let new_v = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;

        // Shuffled map: dim d → palette d % 4 (balanced: 32 dims per palette,
        // maximally different from identity's d / 32). The write path places
        // logical dim d' at (palette d'/32, rank d'%32) — identity placement —
        // so for logical dim d to land where the shuffled map expects it
        // (palette d%4, rank d/4), permute the head_dim axis by
        // d ↦ (d % 4) * 32 + d / 4 before writing.
        let mut inv = vec![0u32; head_dim];
        for d in 0..head_dim {
            inv[(d % 4) * 32 + d / 4] = d as u32;
        }
        let inv_t = Tensor::from_vec(inv, head_dim, &device)?;
        let k1p = k1.index_select(&inv_t, 3)?.contiguous()?;
        let v1p = v1.index_select(&inv_t, 3)?.contiguous()?;

        // Packed 2-bit shuffled map, repeated per head.
        let mut shuf = vec![0u8; head_dim / 4];
        for d in 0..head_dim {
            shuf[d / 4] |= ((d % 4) as u8) << ((d % 4) * 2);
        }
        let pal_all: Vec<u8> = (0..n_kv_head).flat_map(|_| shuf.iter().copied()).collect();

        // Backing: chunk 0 = sealed partial (identity map), chunk 1 = full
        // (shuffled map + permuted data), chunk 2 = empty writer.
        let backing = ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, &device, 3 * CHUNK)?;
        backing.ensure_for_offset(0, 0, 2 * CHUNK + new_len)?;
        backing.write_contiguous(0, 0, &k0, &v0)?;
        backing.write_contiguous(0, CHUNK, &k1p, &v1p)?;
        backing.set_block_window(0, 0, 0, p as u32)?;
        backing.set_block_window(0, 1, 0, CHUNK as u32)?;
        backing.test_set_block_palette(0, 1, pal_all.clone(), pal_all, Vec::new(), Vec::new())?;
        backing.test_set_writer_start(0, 2)?;

        let mut cache0 = KvCache::new(2, 4096);
        cache0.force_dtype(dtype);
        cache0.set_chunked_backing(&backing, 0, None)?;
        cache0.set_current_seq_len(prefix_len)?;

        let offsets = [prefix_len];
        let mut caches: [&mut KvCache; 1] = [&mut cache0];
        let generation = backing.begin_stager_generation_required();

        let out = paged_prefill_uniform(
            &mut caches,
            &offsets,
            &new_q,
            &new_k,
            &new_v,
            b_sz,
            new_len,
            n_head,
            n_kv_head,
            head_dim,
            None,
            &make_zero_rope(head_dim, &device)?,
            false, // rope_interleaved
            &generation,
        )?;
        assert_eq!(out.len(), 1);
        let paged_out = &out[0];

        // Reference attends the LOGICAL prefix (un-permuted) + new segment.
        let full_k = Tensor::cat(&[&k0, &k1, &new_k], 2)?;
        let full_v = Tensor::cat(&[&v0, &v1, &new_v], 2)?;
        let ref_out = reference_attention(
            &new_q, &full_k, &full_v, n_head, n_kv_head, head_dim, prefix_len,
        )?;

        let paged_f32 = paged_out.to_dtype(DType::F32)?;
        let mae = mean_abs_error(&paged_f32, &ref_out)?;
        let max_err = max_abs_error(&paged_f32, &ref_out)?;
        assert!(
            mae < 0.05,
            "straddle shuffled-pal-map prefill mean error too large: {mae} — \
             the prefix tile loader routed a straddle slice through the wrong \
             palette table"
        );
        assert!(
            max_err < 0.2,
            "straddle shuffled-pal-map prefill max error too large: {max_err}"
        );
        println!("straddle shuffled-pal-map prefill OK: mae={mae:.4e} max_err={max_err:.4e}");
        Ok(())
    }

    #[test]
    fn correctness_prefill_with_prefix_f16() -> Result<()> {
        let _gpu = gpu_serial();
        use candle_nn::kv_cache::ChunkedKvBacking;

        let device = Device::new_cuda(0)?;
        let dtype = DType::F16;

        for &(n_head, n_kv_head, head_dim, prefix_len, new_len, label) in &[
            (8, 8, 64, 10, 4, "MHA hd64 prefix=10"),
            (32, 8, 64, 10, 4, "GQA 32/8 hd64 prefix=10"),
            (32, 8, 128, 10, 4, "GQA 32/8 hd128 prefix=10"),
            (40, 8, 64, 10, 4, "GQA 40/8 hd64 prefix=10"),
            (14, 2, 64, 10, 4, "GQA 14/2 hd64 prefix=10"),
        ] {
            let b_sz = 1;
            let total_kv = prefix_len + new_len;

            let prefix_k =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, prefix_len, head_dim), &device)?
                    .to_dtype(dtype)?;
            let prefix_v =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, prefix_len, head_dim), &device)?
                    .to_dtype(dtype)?;
            let new_q = Tensor::randn(0f32, 1f32, (1, n_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let new_k = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let new_v = Tensor::randn(0f32, 1f32, (1, n_kv_head, new_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            let backing =
                ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, &device, total_kv)?;
            backing.ensure_for_offset(0, 0, total_kv)?;
            backing.write_contiguous(0, 0, &prefix_k, &prefix_v)?;

            let mut cache0 = KvCache::new(2, 4096);
            cache0.force_dtype(dtype);
            cache0.set_chunked_backing(&backing, 0, None)?;
            cache0.set_current_seq_len(prefix_len)?;

            let offsets = [prefix_len];
            let mut caches: [&mut KvCache; 1] = [&mut cache0];
            let generation = backing.begin_stager_generation_required();

            let out = paged_prefill_uniform(
                &mut caches,
                &offsets,
                &new_q,
                &new_k,
                &new_v,
                b_sz,
                new_len,
                n_head,
                n_kv_head,
                head_dim,
                None,
                &make_zero_rope(head_dim, &device)?,
                false, // rope_interleaved
                &generation,
            )?;
            let paged_out = &out[0];

            let full_k = Tensor::cat(&[&prefix_k, &new_k], 2)?;
            let full_v = Tensor::cat(&[&prefix_v, &new_v], 2)?;
            let ref_out = reference_attention(
                &new_q, &full_k, &full_v, n_head, n_kv_head, head_dim, prefix_len,
            )?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            assert!(
                mae < 0.05,
                "[{label}] F16 prefill-with-prefix mean error too large: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] F16 prefill-with-prefix max error too large: {max_err}"
            );
            println!("[{label}] F16 prefill-with-prefix OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    // Decode correctness: prefill then single-token decode

    // GQA head mapping regression test (validates the kv_group_end fix)

    /// This test specifically targets the GQA head mapping bug where WARPS_TC
    /// could overflow into the next KV group. It exercises configurations where
    /// num_groups is not a multiple of WARPS_TC (e.g., 40/8 = 5 groups).
    #[test]
    fn correctness_prefill_gqa_head_mapping_regression() -> Result<()> {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        // These configurations are specifically chosen to trigger the GQA overflow bug:
        // - 40/8: num_groups=5, with WARPS_TC=2 → head_blocks_per_kv=3, last block has 1 warp active
        // - 28/4: num_groups=7, tricky for WARPS_TC alignment
        // - 48/8: num_groups=6, with WARPS_TC=2 → head_blocks_per_kv=3, all warps active
        // - 56/8: num_groups=7, same as 28/4 pattern
        // - 12/4: num_groups=3
        for &(n_head, n_kv_head, head_dim, seq_len, label) in &[
            (40, 8, 64, 4, "40/8 hd64 short"),
            (40, 8, 64, 32, "40/8 hd64 medium"),
            (40, 8, 64, 128, "40/8 hd64 long"),
            (40, 8, 128, 4, "40/8 hd128 short"),
            (40, 8, 128, 32, "40/8 hd128 medium"),
            (28, 4, 64, 4, "28/4 hd64 short"),
            (28, 4, 64, 32, "28/4 hd64 medium"),
            (48, 8, 64, 4, "48/8 hd64 short"),
            (48, 8, 64, 32, "48/8 hd64 medium"),
            (56, 8, 64, 4, "56/8 hd64 short"),
            (56, 8, 128, 4, "56/8 hd128 short"),
            (12, 4, 64, 4, "12/4 hd64 short"),
            (12, 4, 64, 32, "12/4 hd64 medium"),
        ] {
            let b_sz = 1;
            let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?;
            let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;
            let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
                .to_dtype(dtype)?
                .contiguous()?;

            let paged_out = run_paged_prefill(
                &q, &k, &v, b_sz, seq_len, n_head, n_kv_head, head_dim, 0, dtype,
            )?;
            let ref_out = reference_attention(&q, &k, &v, n_head, n_kv_head, head_dim, 0)?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;

            assert!(
                mae < 0.05,
                "[{label}] GQA regression: mean error too large: {mae} (may indicate head mapping overflow)"
            );
            assert!(
                max_err < 0.2,
                "[{label}] GQA regression: max error too large: {max_err} (may indicate head mapping overflow)"
            );
            println!("[{label}] GQA regression OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    /// Diagnostic test that prints per-head errors to help identify specific
    /// GQA group cross-contamination patterns.
    #[test]
    fn correctness_prefill_diagnostic_per_head() -> Result<()> {
        let _gpu = gpu_serial();
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;

        // Use a config known to be tricky: 40 heads / 8 KV heads = 5 groups
        let n_head = 40;
        let n_kv_head = 8;
        let head_dim = 64;
        let seq_len = 8;
        let b_sz = 1;

        let q = Tensor::randn(0f32, 1f32, (b_sz, n_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 1f32, (b_sz, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;

        let paged_out = run_paged_prefill(
            &q, &k, &v, b_sz, seq_len, n_head, n_kv_head, head_dim, 0, dtype,
        )?;
        let ref_out = reference_attention(&q, &k, &v, n_head, n_kv_head, head_dim, 0)?;

        let paged_f32 = paged_out.to_dtype(DType::F32)?;
        let num_groups = n_head / n_kv_head;

        println!("\n=== Per-head error for {n_head}/{n_kv_head} (groups={num_groups}) ===");
        for h in 0..n_head {
            let kv_group = h / num_groups;
            let paged_h = paged_f32.narrow(1, h, 1)?;
            let ref_h = ref_out.narrow(1, h, 1)?;
            let mae = mean_abs_error(&paged_h, &ref_h)?;
            let marker = if mae > 0.05 { " *** HIGH ***" } else { "" };
            println!("  head {h:2} (kv_group {kv_group}): mae={mae:.4e}{marker}");
        }

        let overall_mae = mean_abs_error(&paged_f32, &ref_out)?;
        assert!(
            overall_mae < 0.05,
            "Per-head diagnostic: overall mean error too large: {overall_mae}"
        );
        Ok(())
    }

    #[test]
    fn in_kernel_rope_matches_the_reference() -> candle::Result<()> {
        let _gpu = gpu_serial();
        // Verifies: reference_attention(rotated_Q, rotated_K, V) ≈ paged_kernel(unrotated_Q, unrotated_K, V)
        // Tolerance < 0.05 for BF16.
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let (n_head, n_kv_head, head_dim, seq_len) = (4, 4, 64, 8);

        let q =
            Tensor::randn(0f32, 1f32, (1, n_head, seq_len, head_dim), &device)?.to_dtype(dtype)?;
        let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;
        let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, seq_len, head_dim), &device)?
            .to_dtype(dtype)?
            .contiguous()?;

        // Manually rotate [1, n_head, seq_len, head_dim]: token t -> pos t
        let apply_rope_4d = |t: &Tensor| -> candle::Result<Tensor> {
            let (b, h, s, d) = t.dims4()?;
            // Reshape to [b*h, s, d] and back so we can use to_vec3
            let flat = t.reshape((b * h, s, d))?;
            let data = flat.to_dtype(DType::F32)?.to_vec3::<f32>()?;
            let mut out: Vec<f32> = Vec::with_capacity(b * h * s * d);
            for bh in 0..(b * h) {
                for si in 0..s {
                    out.extend_from_slice(&apply_rope_to_vec(&data[bh][si], d, si));
                }
            }
            Tensor::from_vec(out, (b * h, s, d), t.device())?
                .reshape((b, h, s, d))?
                .to_dtype(t.dtype())
        };

        // Pre-rotated versions
        let q_rotated = apply_rope_4d(&q)?.contiguous()?;
        let k_rotated = apply_rope_4d(&k)?.contiguous()?;

        // Branch A (reference): reference_attention with pre-rotated Q/K
        let ref_out =
            reference_attention(&q_rotated, &k_rotated, &v, n_head, n_kv_head, head_dim, 0)?;

        // Branch B: un-rotated Q, K — the kernel rotates each token at its own position.
        let out_kernel_rope =
            run_prefill_with_rope(&q, &k, &v, seq_len, n_head, n_kv_head, head_dim, dtype)?;

        let mae = mean_abs_error(
            &ref_out.to_dtype(DType::F32)?,
            &out_kernel_rope.to_dtype(DType::F32)?,
        )?;
        println!(
            "in_kernel_rope_matches_the_reference: mae(reference(rotated) vs kernel)={mae:.4e}"
        );
        // BF16 rounding: allow up to 0.05 (two roundings: manual rotation + kernel)
        assert!(
            mae < 0.05,
            "fused RoPE prefill functional mismatch: mae={mae}"
        );
        Ok(())
    }
}
