//! Tests of the prefill metadata, the rung check on non-paged rows and the float
//! prefill fallback against a reference.
//!
//! The head-dimension-256 fallback is checked, over a short sequence and one
//! that crosses a chunk boundary, against plain rotated causal attention, and the
//! arena must hold the unrotated keys. Ragged batches must give an exclusive
//! prefix sum of query lengths for their packed layout, and a uniform batch must
//! match the ragged case with equal lengths. The non-paged rows are checked to
//! answer only from an unscaled first rung.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use crate::models::rope_schedule::{plain_inv_freq, DeclaredScaling, RopeSchedule, Rung};

    /// The non-paged paths answer a row only on an unscaled rung 0: past the
    /// first ceiling they refuse, and so does a first rung carrying a
    /// temperature (a file's static YaRN), which they could not apply.
    #[test]
    fn non_paged_rows_stay_on_an_unscaled_first_rung() {
        let progressive = RopeSchedule::yarn(
            64,
            1e7,
            100,
            vec![
                Rung {
                    ceiling: 100,
                    factor: 1.0,
                },
                Rung {
                    ceiling: 200,
                    factor: 2.0,
                },
            ],
            true,
        )
        .unwrap();
        let rope = RopeRungs::new(&progressive, &Device::Cpu).unwrap();
        // Offsets 0 and 98 writing two rows reach 2 and 100: rung 0.
        assert!(refuse_rung_past_zero(&rope, &[0, 98], 2).is_ok());
        // Offset 99 writing two rows reaches 101: rung 1.
        assert!(refuse_rung_past_zero(&rope, &[0, 99], 2).is_err());

        let declared = DeclaredScaling::Yarn {
            factor: 4.0,
            original: 32,
        };
        let static_yarn = declared.schedule(64, 1e7, 1_000).unwrap();
        let rope = RopeRungs::new(&static_yarn, &Device::Cpu).unwrap();
        assert!(rope.q_scale(0) > 1.0);
        assert!(refuse_rung_past_zero(&rope, &[0], 1).is_err());

        let plain = RopeSchedule::stated(plain_inv_freq(64, 1e7), 1_000).unwrap();
        let rope = RopeRungs::new(&plain, &Device::Cpu).unwrap();
        assert!(refuse_rung_past_zero(&rope, &[0, 500], 1).is_ok());
    }

    #[test]
    fn prefill_meta_ragged_builds_varlen_layout() {
        let dev = Device::Cpu;
        // Three sequences with different prefix offsets and different new-token
        // (query) lengths — the ragged case.
        let offsets = [0usize, 5, 100];
        let q_lens = [3usize, 7, 2];
        let m = BatchedPrefillMeta::new_ragged(&offsets, &q_lens, &dev, None).unwrap();

        // cu_seqlens_q is the exclusive prefix sum of q_lens → packed Q has
        // 3+7+2 = 12 rows.
        assert_eq!(m.cu_seqlens_q.to_vec1::<u32>().unwrap(), vec![0, 3, 10, 12]);
        // q_lens passthrough.
        assert_eq!(m.q_lens.to_vec1::<u32>().unwrap(), vec![3, 7, 2]);
        // kv_lens[i] = offset[i] + q_len[i].
        assert_eq!(m.kv_lens.to_vec1::<u32>().unwrap(), vec![3, 12, 102]);
    }

    #[test]
    fn prefill_meta_uniform_matches_ragged_special_case() {
        let dev = Device::Cpu;
        let offsets = [0usize, 0, 0];
        let seq_len = 4usize;
        let u = BatchedPrefillMeta::new(&offsets, seq_len, &dev).unwrap();

        // Uniform is exactly the ragged case with equal q_lens.
        assert_eq!(u.cu_seqlens_q.to_vec1::<u32>().unwrap(), vec![0, 4, 8, 12]);
        assert_eq!(u.q_lens.to_vec1::<u32>().unwrap(), vec![4, 4, 4]);
        assert_eq!(u.kv_lens.to_vec1::<u32>().unwrap(), vec![4, 4, 4]);
    }

    /// The head_dim-256 float prefill fallback (Qwen3.5/3.8 attention shape):
    /// varlen sequences through the real chunked cache must match plain
    /// rotated causal attention, and — the paged-cache interop invariant —
    /// the arena must hold the UNROTATED keys.
    #[cfg(feature = "cuda")]
    #[test]
    fn float_fallback_prefill_hd256_matches_reference() -> Result<()> {
        use crate::models::rope_schedule::RopeSchedule;
        use candle::DType;
        use candle_nn::kv_cache::ChunkedKvBacking;

        let device = Device::new_cuda(0)?;
        let (n_head, n_kv_head, head_dim) = (16usize, 2usize, 256usize);
        let n_rep = n_head / n_kv_head;
        // One short sequence and one that crosses a chunk boundary.
        let q_lens = [7usize, 40usize];
        let offsets = [0usize, 0usize];
        let total_q: usize = q_lens.iter().sum();

        let backing = ChunkedKvBacking::new(2, n_kv_head, head_dim, DType::BF16, &device, 64)?;
        let mut caches: Vec<KvCache> = (0..2)
            .map(|seq| {
                let mut c = KvCache::new(2, 64);
                c.force_dtype(DType::BF16);
                c.set_chunked_backing(&backing, seq, None)?;
                Ok(c)
            })
            .collect::<Result<_>>()?;
        let mut cache_refs: Vec<&mut KvCache> = caches.iter_mut().collect();

        let q = Tensor::randn(0f32, 1f32, (total_q, n_head, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let k = Tensor::randn(0f32, 1f32, (total_q, n_kv_head, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let v = Tensor::randn(0f32, 1f32, (total_q, n_kv_head, head_dim), &device)?
            .to_dtype(DType::BF16)?;
        let inv_freq: Vec<f32> = (0..head_dim / 2)
            .map(|i| 1f32 / 1e6f32.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        let rope = RopeRungs::new(&RopeSchedule::stated(inv_freq.clone(), 1 << 20)?, &device)?;

        let q_live = q.clone();
        let k_live = k.clone();
        let v_live = v.clone();
        let out = paged_prefill_float_fallback(
            &mut cache_refs,
            &offsets,
            &q_live,
            &k_live,
            &v_live,
            &q_lens,
            n_head,
            n_kv_head,
            head_dim,
            &rope,
            false,
        )?;
        assert_eq!(out.dims(), &[total_q, n_head, head_dim]);

        // Reference: per sequence, rotate q/k at absolute positions — from
        // f64 angles, not from the table under test — and run plain causal
        // attention over bf16-rounded values.
        let half = head_dim / 2;
        let rot_ref = |x: &Tensor, len: usize| -> Result<Tensor> {
            let mut cos = Vec::with_capacity(len * half);
            let mut sin = Vec::with_capacity(len * half);
            for pos in 0..len {
                for w in &inv_freq {
                    let a = pos as f64 * *w as f64;
                    cos.push(a.cos() as f32);
                    sin.push(a.sin() as f32);
                }
            }
            let cos = Tensor::from_vec(cos, (len, half), &device)?.to_dtype(DType::BF16)?;
            let sin = Tensor::from_vec(sin, (len, half), &device)?.to_dtype(DType::BF16)?;
            candle_nn::rotary_emb::rope(&x.contiguous()?, &cos, &sin)
        };
        let mut row_start = 0usize;
        for &len in &q_lens {
            let take = |x: &Tensor| -> Result<Tensor> {
                x.narrow(0, row_start, len)?
                    .transpose(0, 1)?
                    .unsqueeze(0)?
                    .contiguous()
            };
            let q_r = rot_ref(&take(&q)?, len)?.to_dtype(DType::F32)?;
            let k_r = rot_ref(&take(&k)?, len)?.to_dtype(DType::F32)?;
            let v_s = take(&v)?.to_dtype(DType::F32)?;
            let k_rep = repeat_kv(k_r, n_rep)?;
            let v_rep = repeat_kv(v_s, n_rep)?;
            let expect = standard_attention_prefill(&q_r, &k_rep, &v_rep, head_dim)?;
            let got = out
                .narrow(0, row_start, len)?
                .transpose(0, 1)?
                .unsqueeze(0)?
                .to_dtype(DType::F32)?;
            let diff = got
                .sub(&expect)?
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_vec0::<f32>()?;
            assert!(
                diff < 5e-2,
                "seq len {len}: fallback diverged from reference (max abs {diff})"
            );
            row_start += len;
        }

        // Interop invariant: the arena holds the UNROTATED keys.
        let (k_stored, _v_stored) = cache_refs[1].chunked_read_kv(0, q_lens[1])?;
        let k_expect = k
            .narrow(0, q_lens[0], q_lens[1])?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;
        let sd = k_stored
            .to_dtype(DType::F32)?
            .sub(&k_expect.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        assert!(
            sd < 1e-6,
            "arena keys must be stored unrotated (max abs diff {sd})"
        );
        Ok(())
    }
}
