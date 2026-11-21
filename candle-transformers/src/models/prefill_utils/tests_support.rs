//! Provides the reference attention, RoPE frequencies and launch shims that the
//! prefill kernel tests share.
//!
//! The references are computed in float with an explicit causal mask, so a kernel
//! result is compared against an answer that does not use the kernel's own code.
//! Shims map the uniform test shapes onto the flat, ragged calling convention. The
//! helpers stand in for the wave's chunk claim, which the prefill entry no longer
//! makes itself.

use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod tests {
    #![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

    use super::paged_prefill_batched as paged_prefill_flat;

    use candle_nn::kv_cache::ChunkedKvBacking;

    use crate::models::rope_schedule::{RopeRungs, RopeSchedule};

    use candle::quantized::pinned_staging::{Generation, PinnedStager};

    use candle::{DType, Device, Result, Tensor};

    use candle_nn::kv_cache::KvCache;

    /// Test shim mapping the legacy uniform calling convention (q/k/v as
    /// `[b_sz, n_head, seq_len, head_dim]`, returning per-sequence
    /// `[1, n_head, seq_len, head_dim]`) onto the production flat/ragged
    /// [`paged_prefill_flat`]. Lets the kernel-correctness tests keep their
    /// original shapes/assertions while the production API is flat-packed.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn paged_prefill_uniform(
        caches: &mut [&mut KvCache],
        offsets: &[usize],
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        b_sz: usize,
        seq_len: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        _prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
        rope: &RopeRungs,
        rope_interleaved: bool,
        generation: &Generation,
    ) -> Result<Vec<Tensor>> {
        let total_q = b_sz * seq_len;
        let q_flat = q
            .transpose(1, 2)?
            .contiguous()?
            .reshape((total_q, n_head, head_dim))?;
        let k_flat = k
            .transpose(1, 2)?
            .contiguous()?
            .reshape((total_q, n_kv_head, head_dim))?;
        let v_flat = v
            .transpose(1, 2)?
            .contiguous()?
            .reshape((total_q, n_kv_head, head_dim))?;
        let q_lens = vec![seq_len; b_sz];
        let out = paged_prefill_flat(
            None,
            caches,
            offsets,
            &q_flat,
            &k_flat,
            &v_flat,
            b_sz,
            &q_lens,
            n_head,
            n_kv_head,
            head_dim,
            None,
            rope,
            rope_interleaved,
            generation,
            &std::cell::RefCell::new(None),
            None,
        )?;
        // Flat [total_q, n_head, head_dim] -> per-seq [1, n_head, seq_len, head_dim].
        let mut per_seq = Vec::with_capacity(b_sz);
        for i in 0..b_sz {
            per_seq.push(
                out.narrow(0, i * seq_len, seq_len)?
                    .reshape((1, seq_len, n_head, head_dim))?
                    .transpose(1, 2)?
                    .contiguous()?,
            );
        }
        Ok(per_seq)
    }

    /// One rung of the standard θ = 10000 frequencies, for the in-kernel RoPE
    /// tests.
    pub(in super::super) fn make_test_rope(head_dim: usize, device: &Device) -> Result<RopeRungs> {
        RopeRungs::new(
            &RopeSchedule::stated(test_inv_freq(head_dim), usize::MAX)?,
            device,
        )
    }

    /// The frequencies [`make_test_rope`] rotates with, for a reference built
    /// from exact angles.
    pub(in super::super) fn test_inv_freq(head_dim: usize) -> Vec<f32> {
        (0..head_dim / 2)
            .map(|i| 1.0f32 / 10000.0f32.powf(2.0 * i as f32 / head_dim as f32))
            .collect()
    }

    /// Zero frequencies for correctness tests: RoPE rotation becomes identity
    /// so the paged-kernel output can be compared directly to a no-RoPE
    /// reference.
    pub(in super::super) fn make_zero_rope(head_dim: usize, device: &Device) -> Result<RopeRungs> {
        RopeRungs::new(
            &RopeSchedule::stated(vec![0.0; head_dim / 2], usize::MAX)?,
            device,
        )
    }

    // NOTE: there are intentionally no "fp8 paged prefill" tests. Prefill arenas
    // are always float (see chunked/io.rs: "Quantization only happens via
    // reconcile_sealed after chunks are complete"), and the prefill compute dtype
    // is the cache's reported dtype — F16/BF16 only (cuda_fwd rejects anything
    // else). FP8 is a post-seal KV *storage* format, exercised on read by the
    // fp8 paged-decode tests (decode_utils.rs: test_fp8_hd{64,128}_paged_decode),
    // not a prefill compute dtype. Forcing a cache to F8E4M3 before prefill is a
    // configuration production never uses.

    // Correctness comparison tests: paged kernels vs reference matmul attention

    /// Gold-standard causal attention via matmul.
    ///
    /// q: (1, n_head, seq_len, head_dim)
    /// k: (1, n_kv_head, kv_len, head_dim)  — full KV including prefix
    /// v: (1, n_kv_head, kv_len, head_dim)
    /// offset: number of prefix tokens (causal mask allows attending to prefix + up to current pos)
    ///
    /// Returns: (1, n_head, seq_len, head_dim) in F32
    pub(in super::super) fn reference_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let seq_len = q.dim(2)?;
        let kv_len = k.dim(2)?;

        // Convert everything to F32 for reference precision
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        // Repeat KV heads for GQA/MQA
        let num_groups = n_head / n_kv_head;
        let k = crate::utils::repeat_kv(k, num_groups)?;
        let v = crate::utils::repeat_kv(v, num_groups)?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;

        // Causal mask: position i in the query (at absolute position offset+i)
        // can attend to KV positions 0..=offset+i
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..kv_len).map(move |j| {
                    if j > offset + i {
                        f32::NEG_INFINITY
                    } else {
                        0.0f32
                    }
                })
            })
            .collect();
        let mask = Tensor::from_vec(mask, (1, 1, seq_len, kv_len), q.device())?;
        let att = att.broadcast_add(&mask)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?;
        Ok(out)
    }

    /// Reference decode attention: single query token against full KV cache.
    ///
    /// q: (batch_size, n_head, head_dim)
    /// k: (batch_size, n_kv_head, kv_len, head_dim)
    /// v: (batch_size, n_kv_head, kv_len, head_dim)
    ///
    /// Returns: (batch_size, n_head, head_dim) in F32
    #[allow(dead_code)]
    pub(in super::super) fn reference_decode_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let _batch_size = q.dim(0)?;
        let _kv_len = k.dim(2)?;

        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;

        let num_groups = n_head / n_kv_head;
        let k = crate::utils::repeat_kv(k, num_groups)?;
        let v = crate::utils::repeat_kv(v, num_groups)?;

        // q: (batch, n_head, head_dim) -> (batch, n_head, 1, head_dim)
        let q = q.unsqueeze(2)?;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;

        // No causal mask needed for decode — all KV positions are valid
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?; // (batch, n_head, 1, head_dim)
        let out = out.squeeze(2)?; // (batch, n_head, head_dim)
        Ok(out)
    }

    /// Compute max absolute error between two tensors (both converted to F32).
    pub(in super::super) fn max_abs_error(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?;
        let diff = (a - b)?.abs()?;
        diff.max(0)?.to_vec0::<f32>()
    }

    /// Compute mean absolute error between two tensors (both converted to F32).
    pub(in super::super) fn mean_abs_error(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?;
        let diff = (a - b)?.abs()?;
        diff.mean_all()?.to_vec0::<f32>()
    }

    /// Helper: run paged prefill and return output in (1, n_head, seq_len, head_dim) shape.
    /// Identity RoPE (zero table) — the shape/GQA correctness tests compare
    /// against a rope-free reference.
    pub(in super::super) fn run_paged_prefill(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        b_sz: usize,
        seq_len: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        offset: usize,
        dtype: DType,
    ) -> Result<Tensor> {
        let rope = make_zero_rope(head_dim, q.device())?;
        run_paged_prefill_with_rope(
            q, k, v, b_sz, seq_len, n_head, n_kv_head, head_dim, offset, dtype, &rope, false,
        )
    }

    /// [`run_paged_prefill`] with a caller-supplied RoPE table and pairing —
    /// the harness for the in-kernel RoPE correctness tests.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn run_paged_prefill_with_rope(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        b_sz: usize,
        seq_len: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        offset: usize,
        dtype: DType,
        rope: &RopeRungs,
        rope_interleaved: bool,
    ) -> Result<Tensor> {
        // Stand in for the scheduler and for `wave_admit`: the prefill entry
        // neither creates a backing nor claims chunks of its own any more, so a
        // test driving the kernel directly has to arrive with both, exactly as
        // the real caller does.
        let backing = ChunkedKvBacking::new(
            b_sz,
            n_kv_head,
            head_dim,
            dtype,
            q.device(),
            offset + seq_len,
        )?;
        let mut cache0 = KvCache::new(2, 4096);
        cache0.force_dtype(dtype);
        cache0.set_chunked_backing(&backing, 0, None)?;
        cache0.set_current_seq_len(offset)?;
        let offsets = [offset];
        let mut caches: [&mut KvCache; 1] = [&mut cache0];
        KvCache::ensure_chunked_capacity_batch(&mut caches, &offsets, seq_len)?;
        let generation = PinnedStager::new(q.device().as_cuda_device()?).begin_generation();

        let out = paged_prefill_uniform(
            &mut caches,
            &offsets,
            q,
            k,
            v,
            b_sz,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            None,
            rope,
            rope_interleaved,
            &generation,
        )?;
        assert_eq!(out.len(), 1);
        Ok(out[0].clone())
    }

    // In-kernel RoPE

    /// Helper: run a single-batch prefill with the θ = 10000 rung and return
    /// the output tensor.
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn run_prefill_with_rope(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        dtype: DType,
    ) -> candle::Result<Tensor> {
        use candle_nn::kv_cache::ChunkedKvBacking;
        let device = q.device();
        let b_sz = 1;

        let backing = ChunkedKvBacking::new(b_sz, n_kv_head, head_dim, dtype, device, seq_len)?;
        backing.ensure_for_offset(0, 0, seq_len)?;

        let mut cache0 = KvCache::new(2, 4096);
        cache0.force_dtype(dtype);
        cache0.set_chunked_backing(&backing, 0, None)?;

        let offsets = [0usize];
        let mut caches: [&mut KvCache; 1] = [&mut cache0];
        // Stand in for `wave_admit`: the prefill entry no longer claims its own
        // chunks, so the caller must have made the write region before it runs.
        KvCache::ensure_chunked_capacity_batch(&mut caches, &offsets, seq_len)?;
        let generation = backing.begin_stager_generation_required();

        let out = paged_prefill_uniform(
            &mut caches,
            &offsets,
            q,
            k,
            v,
            b_sz,
            seq_len,
            n_head,
            n_kv_head,
            head_dim,
            None,
            &make_test_rope(head_dim, device)?,
            false, // rope_interleaved
            &generation,
        )?;
        Ok(out.into_iter().next().unwrap())
    }

    // Reference RoPE rotation: apply to f32 slice of HEAD_DIM values.
    // Pairs: dim d with dim d+half_dim.
    pub(in super::super) fn apply_rope_to_vec(x: &[f32], head_dim: usize, pos: usize) -> Vec<f32> {
        let mut out = x.to_vec();
        let half = head_dim / 2;
        for d in 0..half {
            let theta = pos as f32 * 10000.0f32.powf(-2.0 * d as f32 / head_dim as f32);
            let (sin_v, cos_v) = theta.sin_cos();
            let x_lo = x[d];
            let x_hi = x[d + half];
            out[d] = x_lo * cos_v - x_hi * sin_v;
            out[d + half] = x_lo * sin_v + x_hi * cos_v;
        }
        out
    }
}
