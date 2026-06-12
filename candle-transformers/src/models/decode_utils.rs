use candle::{Device, Result, Tensor};

/// Build a device tensor of per-sequence offsets (u32) for a batched decode step.
pub fn offsets_to_u32_tensor(offsets: &[usize], device: &Device) -> Result<Tensor> {
    let offsets_u32: Vec<u32> = offsets.iter().map(|&o| o as u32).collect();
    Tensor::from_vec(offsets_u32, offsets.len(), device)
}

/// Build a device tensor of per-sequence cache lengths (u32), computed as offset + 1.
pub fn cache_lens_to_u32_tensor(offsets: &[usize], device: &Device) -> Result<Tensor> {
    let cache_lens_u32: Vec<u32> = offsets.iter().map(|&o| (o + 1) as u32).collect();
    Tensor::from_vec(cache_lens_u32, offsets.len(), device)
}

/// Build a device tensor of per-block canonical RoPE start positions (i32) for chunked decode.
///
/// Shape: `[batch_size * max_blocks]`, row-major. Block B gets position
/// B * chunk_size.  K is stored un-rotated; the decode kernel applies RoPE
/// at this position + within + rope_offsets.
///
/// Stored as `u32` (Candle doesn't support `i32`); the raw bits are reinterpreted
/// as `i32` at the FFI boundary since i32 and u32 have identical layout.
pub fn chunk_rope_positions_to_i32_tensor(
    positions: &[i32],
    batch_size: usize,
    max_blocks: usize,
    device: &Device,
) -> Result<Tensor> {
    assert_eq!(positions.len(), batch_size * max_blocks);
    // Reinterpret i32 -> u32 for Candle tensor storage (same bit pattern).
    let as_u32: Vec<u32> = positions.iter().map(|&s| s as u32).collect();
    Tensor::from_vec(as_u32, (batch_size, max_blocks), device)
}

/// Gather per-batch RoPE (cos, sin) rows by `offsets_t` and reshape to (B, 1, D).
///
/// Expects `cos_all` and `sin_all` shaped like (max_seq, D).
pub fn gather_rope_cos_sin(
    cos_all: &Tensor,
    sin_all: &Tensor,
    offsets_t: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let b = offsets_t.dim(0)?;

    let mut cos = cos_all.index_select(offsets_t, 0)?;
    let mut sin = sin_all.index_select(offsets_t, 0)?;

    if !cos.is_contiguous() {
        cos = cos.contiguous()?;
    }
    if !sin.is_contiguous() {
        sin = sin.contiguous()?;
    }

    let cos = cos.reshape((b, 1, cos.dim(1)?))?;
    let sin = sin.reshape((b, 1, sin.dim(1)?))?;
    Ok((cos, sin))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_rope_positions_basic() {
        let positions = vec![0i32, 0, 100, 0, 100, 0]; // 2 batch, 3 blocks
        let t = chunk_rope_positions_to_i32_tensor(&positions, 2, 3, &Device::Cpu).unwrap();
        // Stored as u32, check shape
        assert_eq!(t.dims(), &[2, 3]);
        let vals = t.to_vec2::<u32>().unwrap();
        // 100 as i32 -> 100 as u32 (positive values are identical)
        assert_eq!(vals[0], vec![0, 0, 100]);
        assert_eq!(vals[1], vec![0, 100, 0]);
    }

    #[test]
    fn test_chunk_rope_positions_negative_value_bitcast() {
        // Verify negative i32 is correctly bitcast to u32
        let positions = vec![-1i32];
        let t = chunk_rope_positions_to_i32_tensor(&positions, 1, 1, &Device::Cpu).unwrap();
        let vals = t.to_vec2::<u32>().unwrap();
        assert_eq!(vals[0][0], u32::MAX); // -1i32 as u32 = 0xFFFFFFFF
    }

    #[test]
    fn test_chunk_rope_positions_all_zero() {
        let positions = vec![0i32; 6];
        let t = chunk_rope_positions_to_i32_tensor(&positions, 2, 3, &Device::Cpu).unwrap();
        let flat: Vec<u32> = t.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_offsets_basic() {
        let t = offsets_to_u32_tensor(&[0, 5, 100], &Device::Cpu).unwrap();
        let vals = t.to_vec1::<u32>().unwrap();
        assert_eq!(vals, vec![0, 5, 100]);
    }

    #[test]
    fn test_cache_lens_basic() {
        let t = cache_lens_to_u32_tensor(&[0, 5, 100], &Device::Cpu).unwrap();
        let vals = t.to_vec1::<u32>().unwrap();
        assert_eq!(vals, vec![1, 6, 101]);
    }

    /// Realistic canonical positions: 3 batches × 4 blocks where
    /// positions[b][i] = i * chunk_size for allocated blocks, 0 otherwise.
    #[test]
    fn test_chunk_rope_positions_canonical_layout() {
        let chunk_size = 64i32;
        let batch_size = 3;
        let max_blocks = 4;
        // batch 0: 3 blocks allocated, batch 1: 1 block, batch 2: 4 blocks
        let mut positions = vec![0i32; batch_size * max_blocks];
        // batch 0: blocks 0,1,2
        for i in 0..3 {
            positions[0 * max_blocks + i] = (i as i32) * chunk_size;
        }
        // batch 1: block 0 only
        positions[1 * max_blocks + 0] = 0;
        // batch 2: all 4 blocks
        for i in 0..4 {
            positions[2 * max_blocks + i] = (i as i32) * chunk_size;
        }

        let t =
            chunk_rope_positions_to_i32_tensor(&positions, batch_size, max_blocks, &Device::Cpu)
                .unwrap();
        assert_eq!(t.dims(), &[3, 4]);
        let vals = t.to_vec2::<u32>().unwrap();

        // Batch 0: [0, 64, 128, 0]
        assert_eq!(vals[0], vec![0, 64, 128, 0]);
        // Batch 1: [0, 0, 0, 0]
        assert_eq!(vals[1], vec![0, 0, 0, 0]);
        // Batch 2: [0, 64, 128, 192]
        assert_eq!(vals[2], vec![0, 64, 128, 192]);
    }

    /// Verify that the tensor shape assertion holds: positions.len() must
    /// equal batch_size * max_blocks.
    #[test]
    #[should_panic]
    fn test_chunk_rope_positions_shape_mismatch_panics() {
        // 5 elements but batch_size=2, max_blocks=3 expects 6
        let positions = vec![0i32; 5];
        let _ = chunk_rope_positions_to_i32_tensor(&positions, 2, 3, &Device::Cpu).unwrap();
    }
}

// ============================================================================
// CUDA-backed decode kernel correctness tests
//
// These 12 tests mirror the 12 prefill tests in prefill_utils.rs:
//   1.  paged_decode_bf16_smoke
//   2.  test_fp8_hd128_paged_decode
//   3.  test_fp8_hd64_paged_decode
//   4.  correctness_decode_no_history_bf16
//   5.  correctness_decode_no_history_f16
//   6.  correctness_decode_with_history_bf16
//   7.  correctness_decode_with_history_f16
//   8.  correctness_decode_gqa_head_mapping_regression
//   9.  correctness_decode_diagnostic_per_head
//   10. rope_offset_decode_none_succeeds
//   11. rope_offset_decode_real_vs_zero_differs
//   12. rope_offset_decode_functional
// ============================================================================
#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use crate::models::prefill_utils::{
        compute_rope_cs, paged_decode_attn_with_backend, DecodeBackend,
    };
    use candle::quantized::pinned_staging::PinnedStager;
    use candle::{DType, Device, Result, Tensor};
    use candle_nn::kv_cache::ChunkedKvBacking;

    // ------------------------------------------------------------------
    // RoPE table helpers (same theta=10000 convention as prefill_utils)
    // ------------------------------------------------------------------

    fn make_test_inv_freq(head_dim: usize, device: &Device) -> Result<Tensor> {
        let half = head_dim / 2;
        let v: Vec<f32> = (0..half)
            .map(|i| 1.0f32 / 10000.0f32.powf(2.0 * i as f32 / head_dim as f32))
            .collect();
        Tensor::from_vec(v, (half,), device)
    }

    fn make_zero_inv_freq(head_dim: usize, device: &Device) -> Result<Tensor> {
        Tensor::zeros((head_dim / 2,), DType::F32, device)
    }

    fn make_test_rope_cs(head_dim: usize, max_blocks: usize, device: &Device) -> Result<Tensor> {
        compute_rope_cs(&make_test_inv_freq(head_dim, device)?, max_blocks, head_dim, device)
    }

    fn make_zero_rope_cs(head_dim: usize, max_blocks: usize, device: &Device) -> Result<Tensor> {
        compute_rope_cs(&make_zero_inv_freq(head_dim, device)?, max_blocks, head_dim, device)
    }

    // ------------------------------------------------------------------
    // Error metrics
    // ------------------------------------------------------------------

    fn max_abs_error(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?;
        (a - b)?.abs()?.max(0)?.to_vec0::<f32>()
    }

    fn mean_abs_error(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a = a.to_dtype(DType::F32)?.flatten_all()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?;
        (a - b)?.abs()?.mean_all()?.to_vec0::<f32>()
    }

    // ------------------------------------------------------------------
    // Reference decode attention (gold-standard matmul)
    //
    // q:   (batch, n_head,    head_dim)
    // k/v: (batch, n_kv_head, kv_len, head_dim)
    // Returns: (batch, n_head, head_dim) in F32
    // ------------------------------------------------------------------

    fn reference_decode_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let q = q.to_dtype(DType::F32)?;
        let k = k.to_dtype(DType::F32)?;
        let v = v.to_dtype(DType::F32)?;
        let num_groups = n_head / n_kv_head;
        let k = crate::utils::repeat_kv(k, num_groups)?;
        let v = crate::utils::repeat_kv(v, num_groups)?;
        let q = q.unsqueeze(2)?; // (batch, n_head, 1, head_dim)
        let scale = 1.0 / (head_dim as f64).sqrt();
        let att = (q.matmul(&k.t()?)? * scale)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        let out = att.matmul(&v)?.squeeze(2)?;
        Ok(out)
    }

    // ------------------------------------------------------------------
    // Manual RoPE rotation (for functional test)
    // ------------------------------------------------------------------

    fn apply_rope_to_vec(x: &[f32], head_dim: usize, pos: usize) -> Vec<f32> {
        let mut out = x.to_vec();
        let half = head_dim / 2;
        for d in 0..half {
            let theta = pos as f32 * 10000.0f32.powf(-2.0 * d as f32 / head_dim as f32);
            let (sin_v, cos_v) = theta.sin_cos();
            let (x_lo, x_hi) = (x[d], x[d + half]);
            out[d] = x_lo * cos_v - x_hi * sin_v;
            out[d + half] = x_lo * sin_v + x_hi * cos_v;
        }
        out
    }

    // t: (batch, n_head, head_dim)
    fn apply_rope_3d(t: &Tensor, pos: usize) -> Result<Tensor> {
        let (b, h, d) = t.dims3()?;
        let data = t.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        let mut out = Vec::with_capacity(b * h * d);
        for bi in 0..b {
            for hi in 0..h {
                out.extend_from_slice(&apply_rope_to_vec(&data[bi][hi], d, pos));
            }
        }
        Tensor::from_vec(out, (b, h, d), t.device())?.to_dtype(t.dtype())
    }

    // ------------------------------------------------------------------
    // Core helper: run one paged decode step
    //
    // history: optional (k, v) of shape (1, n_kv_head, history_len, head_dim)
    //          — pass None for the first decode step (empty cache).
    // k_new / v_new: (1, n_kv_head, head_dim) — the new KV token.
    // q:             (1, n_head,    head_dim) — the query.
    //
    // arena_dtype controls both arena storage and kernel selection:
    //   F16            → f16 kernel;  q/k_new/v_new must be F16.
    //   BF16 | F8E4M3  → bf16 kernel; q/k_new/v_new must be BF16.
    // ------------------------------------------------------------------

    fn run_paged_decode_be(
        backend: DecodeBackend,
        history: Option<(&Tensor, &Tensor)>,
        k_new: &Tensor,
        v_new: &Tensor,
        q: &Tensor,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        arena_dtype: DType,
        rope_cs: &Tensor,
    ) -> Result<Tensor> {
        let device = q.device();
        let history_len = history
            .map(|(k, _)| k.dim(2))
            .transpose()?
            .unwrap_or(0);
        let seq_offset = history_len;

        let backing =
            ChunkedKvBacking::new(1, n_kv_head, head_dim, arena_dtype, device, history_len + 1)?;
        if let Some((hk, hv)) = history {
            backing.write_contiguous(0, 0, hk, hv)?;
            // write_contiguous lays down the history *data* but leaves the
            // cum_token `usage` metadata at 0 (production bumps it per decode
            // step via the kernel self-increment + set_len). Without this, the
            // writer-chunk resolution (decode_write_chunk_idx / before_wi) is
            // only correct while history_len < CHUNK_SIZE; for chunk-spanning
            // history it would pick the full chunk 0 as the writer and compute
            // write_len = history_len (>= CHUNK_SIZE). set_len mirrors prefill,
            // marking the history tokens used so the writer lands in a fresh
            // chunk at the right slot.
            backing.set_len(0, history_len);
        }
        backing.ensure_for_offset(0, seq_offset, 1)?;

        let arena_info = backing.resolve_arena_info()?;
        let (ptrs, _) = backing.sync_decode_gpu_chunks(&[(0, seq_offset)], &arena_info)?;
        let (ptr, n_slices, write_slice) = ptrs[0];

        // Stage the 16-byte SlotHeader to GPU.
        //
        // IMPORTANT: `gen` must stay alive until after `paged_decode_attn` returns.
        // GpuBuf arena buffers point into the stager's pinned GPU-mapped arena;
        // Generation::drop syncs the stream then frees that arena. Dropping `gen`
        // before the kernel runs causes CUDA_ERROR_ILLEGAL_ADDRESS.
        let stager = PinnedStager::new(device.as_cuda_device()?);
        let gen = stager.begin_generation();
        let mut hdr = [0u8; 16];
        hdr[..4].copy_from_slice(&n_slices.to_le_bytes());
        hdr[4..8].copy_from_slice(&write_slice.to_le_bytes());
        hdr[8..16].copy_from_slice(&ptr.to_le_bytes());
        let mut pinned = gen.alloc(16)?;
        pinned.copy_from_slice(&hdr);
        let _gpu_buf = gen.submit(pinned)?;
        let headers_ptr = _gpu_buf.dev_ptr();

        let softmax_scale = 1.0f32 / (head_dim as f32).sqrt();
        // Compute dtype must match the kernel dispatched by arena_dtype.
        let compute_dtype = if arena_dtype == DType::F16 { DType::F16 } else { DType::BF16 };
        let q_c = q.to_dtype(compute_dtype)?;
        // k_new/v_new are passed as compute_dtype (the kernel receives BF16/F16 new tokens
        // and writes them to the arena with any necessary conversion internally).
        let k_c = k_new.to_dtype(compute_dtype)?.contiguous()?;
        let v_c = v_new.to_dtype(compute_dtype)?.contiguous()?;

        let result = paged_decode_attn_with_backend(
            &q_c,
            headers_ptr,
            arena_dtype,
            n_head,
            n_kv_head,
            head_dim,
            softmax_scale,
            &k_c,
            &v_c,
            rope_cs,
            false, // rope_interleaved
            backend,
        )?;

        // `gen` drops here — Generation::drop syncs the stream then frees the
        // pinned arena, which is now safe because the kernel has completed.
        drop(gen);
        Ok(result)
    }

    // Thin Int8-default wrapper so the many smoke / rope-offset callers stay
    // unchanged; the correctness A/B tests call run_paged_decode_be directly.
    #[allow(clippy::too_many_arguments)]
    fn run_paged_decode(
        history: Option<(&Tensor, &Tensor)>,
        k_new: &Tensor,
        v_new: &Tensor,
        q: &Tensor,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        arena_dtype: DType,
        rope_cs: &Tensor,
    ) -> Result<Tensor> {
        run_paged_decode_be(
            DecodeBackend::Int8,
            history,
            k_new,
            v_new,
            q,
            n_head,
            n_kv_head,
            head_dim,
            arena_dtype,
            rope_cs,
        )
    }

    // Run an INT8 decode against a *substrate-seal gap*: a sealed partial chunk 0
    // (`p` tokens, usage `p`, physical slots 0..p) followed by a fresh writer
    // chunk 1 (`w` tokens, physical slots 0..w) holding logical positions p..p+w.
    // Chunk 0's slots p..32 are an addressing gap. A gap-blind chunk_div(logical)
    // = logical/32 aliases the writer's tokens back into chunk 0's empty tail and
    // drops them; a gap-aware per-slice walk reaches them at chunk 1's true
    // physical position. Zero RoPE (matches reference_decode_attention).
    #[allow(clippy::too_many_arguments)]
    fn run_paged_decode_gap(
        seal0: (&Tensor, &Tensor),
        writer1: (&Tensor, &Tensor),
        k_new: &Tensor,
        v_new: &Tensor,
        q: &Tensor,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        arena_dtype: DType,
    ) -> Result<Tensor> {
        const CHUNK_SIZE: usize = 32;
        let device = q.device();
        let (hk0, hv0) = seal0;
        let (hk1, hv1) = writer1;
        let p = hk0.dim(2)?;
        let w = hk1.dim(2)?;
        assert!(
            p < CHUNK_SIZE && p + w < CHUNK_SIZE,
            "gap test needs p < 32 and p+w < 32"
        );
        let seq_offset = p + w;

        let backing = ChunkedKvBacking::new(
            1,
            n_kv_head,
            head_dim,
            arena_dtype,
            device,
            CHUNK_SIZE + w + 1,
        )?;
        // Allocate two blocks, then lay chunk 0's data at logical 0 and chunk 1's
        // at logical 32 (physical chunk 1, slot 0).
        backing.ensure_for_offset(0, CHUNK_SIZE, 1)?;
        backing.write_contiguous(0, 0, hk0, hv0)?;
        backing.write_contiguous(0, CHUNK_SIZE, hk1, hv1)?;
        // chunk 0: sealed partial (off 0, usage p); chunk 1: writer (off 0, usage
        // w). Advancing the writer start past chunk 0 marks it sealed, so the
        // writer-chunk resolution computes rope_base = p for chunk 1 and the gap
        // opens between chunk 0's p-th slot and chunk 1's slot 0.
        backing.set_block_window(0, 0, 0, p as u32)?;
        backing.set_block_window(0, 1, 0, w as u32)?;
        backing.test_set_writer_start(0, 1)?;

        let arena_info = backing.resolve_arena_info()?;
        let (ptrs, _) = backing.sync_decode_gpu_chunks(&[(0, seq_offset)], &arena_info)?;
        let (ptr, n_slices, write_slice) = ptrs[0];

        let stager = PinnedStager::new(device.as_cuda_device()?);
        let gen = stager.begin_generation();
        let mut hdr = [0u8; 16];
        hdr[..4].copy_from_slice(&n_slices.to_le_bytes());
        hdr[4..8].copy_from_slice(&write_slice.to_le_bytes());
        hdr[8..16].copy_from_slice(&ptr.to_le_bytes());
        let mut pinned = gen.alloc(16)?;
        pinned.copy_from_slice(&hdr);
        let _gpu_buf = gen.submit(pinned)?;
        let headers_ptr = _gpu_buf.dev_ptr();

        let softmax_scale = 1.0f32 / (head_dim as f32).sqrt();
        let compute_dtype = if arena_dtype == DType::F16 {
            DType::F16
        } else {
            DType::BF16
        };
        let q_c = q.to_dtype(compute_dtype)?;
        let k_c = k_new.to_dtype(compute_dtype)?.contiguous()?;
        let v_c = v_new.to_dtype(compute_dtype)?.contiguous()?;
        let rope_cs = make_zero_rope_cs(head_dim, 16, device)?;

        let result = paged_decode_attn_with_backend(
            &q_c,
            headers_ptr,
            arena_dtype,
            n_head,
            n_kv_head,
            head_dim,
            softmax_scale,
            &k_c,
            &v_c,
            &rope_cs,
            false,
            DecodeBackend::Int8,
        )?;
        drop(gen);
        Ok(result)
    }

    // Decode correctness across a substrate-seal gap. The BMMA path (hd128,
    // hpg<=8) is already gap-aware and is the passing control; the stripe (hd64)
    // and warp=head (hpg>8) paths must match it once gap-fixed.
    #[test]
    fn correctness_decode_seal_gap() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        // Small sealed chunk (p) + large writer (w): a gap-blind chunk_div drops
        // the writer's w+1 tokens, attending only the p sealed ones — a large,
        // unambiguous error well above the INT8 noise floor. Covers the BMMA
        // (hd128, hpg<=8) and stripe (hd64, hpg<=8) paths; the warp=head wide
        // path (hpg>8) is covered by the ignored test below.
        for &(n_head, n_kv_head, head_dim, p, w, label) in &[
            (32, 8, 128, 6, 22, "BMMA hd128 gap"),
            (8, 8, 64, 6, 22, "stripe hd64 mha gap"),
            (32, 8, 64, 6, 22, "stripe hd64 gqa gap"),
        ] {
            let mk = |len: usize| -> Result<(Tensor, Tensor)> {
                let k = Tensor::randn(0f32, 1f32, (1, n_kv_head, len, head_dim), &device)?
                    .to_dtype(dtype)?;
                let v = Tensor::randn(0f32, 1f32, (1, n_kv_head, len, head_dim), &device)?
                    .to_dtype(dtype)?;
                Ok((k, v))
            };
            let (hk0, hv0) = mk(p)?;
            let (hk1, hv1) = mk(w)?;
            let k_new =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
            let v_new =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
            let q = Tensor::randn(0f32, 1f32, (1, n_head, head_dim), &device)?.to_dtype(dtype)?;

            let int8 = run_paged_decode_gap(
                (&hk0, &hv0),
                (&hk1, &hv1),
                &k_new,
                &v_new,
                &q,
                n_head,
                n_kv_head,
                head_dim,
                dtype,
            )?
            .to_dtype(DType::F32)?;

            // Reference attends the logical sequence: chunk0 (p) + chunk1 (w) + new.
            let full_k = Tensor::cat(&[&hk0, &hk1, &k_new.unsqueeze(2)?], 2)?;
            let full_v = Tensor::cat(&[&hv0, &hv1, &v_new.unsqueeze(2)?], 2)?;
            let ref_out =
                reference_decode_attention(&q, &full_k, &full_v, n_head, n_kv_head, head_dim)?;

            let mae = mean_abs_error(&int8, &ref_out)?;
            let max_err = max_abs_error(&int8, &ref_out)?;
            assert!(
                mae < 0.15,
                "[{label}] seal-gap int8 vs reference mae too large: {mae}"
            );
            assert!(
                max_err < 1.0,
                "[{label}] seal-gap int8 vs reference max error too large: {max_err}"
            );
            println!("[{label}] seal-gap decode OK: mae={mae:.3e} max_err={max_err:.3e}");
        }
        Ok(())
    }

    // Seal-gap handling for the warp=head wide path (heads_per_group > 8, WARPS=16).
    // Ignored: that kernel tiles 16 contiguous logical tokens into one 32-token
    // chunk (chunk_div(tile_base)), an assumption a sealed partial chunk breaks.
    // Making it gap-aware needs the per-slice retiling the stripe/BMMA paths use
    // (each tile entirely within one slice). The wide path is non-production —
    // no target model has hpg>8 — so this is tracked, not blocking. Un-ignore
    // once the warp=head kernel iterates per slice.
    #[test]
    #[ignore = "warp=head wide-path (hpg>8) seal-gap retiling pending; non-production"]
    fn correctness_decode_seal_gap_warp_head_wide() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let (n_head, n_kv_head, head_dim, p, w) = (16, 1, 64, 6, 22);
        let mk = |len: usize| -> Result<(Tensor, Tensor)> {
            let k =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, len, head_dim), &device)?.to_dtype(dtype)?;
            let v =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, len, head_dim), &device)?.to_dtype(dtype)?;
            Ok((k, v))
        };
        let (hk0, hv0) = mk(p)?;
        let (hk1, hv1) = mk(w)?;
        let k_new = Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
        let v_new = Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
        let q = Tensor::randn(0f32, 1f32, (1, n_head, head_dim), &device)?.to_dtype(dtype)?;

        let int8 = run_paged_decode_gap(
            (&hk0, &hv0),
            (&hk1, &hv1),
            &k_new,
            &v_new,
            &q,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
        )?
        .to_dtype(DType::F32)?;
        let full_k = Tensor::cat(&[&hk0, &hk1, &k_new.unsqueeze(2)?], 2)?;
        let full_v = Tensor::cat(&[&hv0, &hv1, &v_new.unsqueeze(2)?], 2)?;
        let ref_out =
            reference_decode_attention(&q, &full_k, &full_v, n_head, n_kv_head, head_dim)?;
        let mae = mean_abs_error(&int8, &ref_out)?;
        assert!(mae < 0.15, "warp=head wide seal-gap mae too large: {mae}");
        Ok(())
    }

    // Decode correctness check: the INT8 decode kernel must match the
    // gold-standard pure-tensor attention (reference_decode_attention) within
    // INT8 quantization error. The reference is a plain matmul, so it cannot
    // carry kernel bugs — any kernel defect (wrong GQA mapping, broken MMA
    // fragment layout, chunk-gap miscount) pushes the error far past quant
    // noise. Tolerances are calibrated to measured INT8 Q/K quantization error
    // (~0.08 mae on hd64), not to FP precision. Callers pass a zero RoPE table
    // (matches the reference, which applies no RoPE).
    #[allow(clippy::too_many_arguments)]
    fn assert_decode_ab(
        history: Option<(&Tensor, &Tensor)>,
        k_new: &Tensor,
        v_new: &Tensor,
        q: &Tensor,
        n_head: usize,
        n_kv_head: usize,
        head_dim: usize,
        dtype: DType,
        rope_cs: &Tensor,
        label: &str,
    ) -> Result<()> {
        let int8 = run_paged_decode_be(
            DecodeBackend::Int8, history, k_new, v_new, q, n_head, n_kv_head, head_dim, dtype,
            rope_cs,
        )?
        .to_dtype(DType::F32)?;

        let full_k = match history {
            Some((hk, _)) => Tensor::cat(&[hk, &k_new.unsqueeze(2)?], 2)?,
            None => k_new.unsqueeze(2)?,
        };
        let full_v = match history {
            Some((_, hv)) => Tensor::cat(&[hv, &v_new.unsqueeze(2)?], 2)?,
            None => v_new.unsqueeze(2)?,
        };
        let ref_out = reference_decode_attention(q, &full_k, &full_v, n_head, n_kv_head, head_dim)?;

        let i8_mae = mean_abs_error(&int8, &ref_out)?;
        let i8_max = max_abs_error(&int8, &ref_out)?;
        // mae is the precision gate (INT8 Q/K quantization ~0.08 on hd64). max is
        // a single-element-outlier garbage-detector: a real kernel defect (wrong
        // head, broken MMA layout, chunk miscount) blows max past ~2, while
        // legitimate INT8 rounding tails reach ~0.75 on hd64 with odd group counts.
        assert!(
            i8_mae < 0.15,
            "[{label}] int8 vs reference mae too large: {i8_mae}"
        );
        assert!(
            i8_max < 1.0,
            "[{label}] int8 vs reference max error too large: {i8_max}"
        );
        println!("[{label}] decode A/B OK: i8_mae={i8_mae:.3e} i8_max={i8_max:.3e}");
        Ok(())
    }

    // ======================================================================
    // Test 1: Basic BF16 smoke — shape and finite output
    // ======================================================================

    #[test]
    fn paged_decode_bf16_smoke() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let (n_head, n_kv_head, head_dim) = (8, 8, 64);

        let q =
            Tensor::randn(0f32, 1f32, (1, n_head, head_dim), &device)?.to_dtype(dtype)?;
        let k_new =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
        let v_new =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;

        let out = run_paged_decode(
            None,
            &k_new,
            &v_new,
            &q,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
            &make_zero_rope_cs(head_dim, 16, &device)?,
        )?;
        assert_eq!(out.dims(), &[1, n_head, head_dim]);
        let max_abs = out
            .to_dtype(DType::F32)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        assert!(max_abs.is_finite(), "smoke: output not finite: {max_abs}");
        Ok(())
    }

    // ======================================================================
    // Tests 2-3: FP8 arena smoke — kernel must not crash and output finite
    // ======================================================================

    #[test]
    fn test_fp8_hd128_paged_decode() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let (n_head, n_kv_head, head_dim) = (32, 8, 128);
        let q =
            Tensor::randn(0f32, 1f32, (1, n_head, head_dim), &device)?.to_dtype(DType::BF16)?;
        let k_new =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(DType::BF16)?;
        let v_new =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(DType::BF16)?;
        // Small history so the arena actually contains data in F8E4M3 format.
        let hk =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, 8, head_dim), &device)?
                .to_dtype(DType::BF16)?;
        let hv = hk.zeros_like()?;

        let out = run_paged_decode(
            Some((&hk, &hv)),
            &k_new,
            &v_new,
            &q,
            n_head,
            n_kv_head,
            head_dim,
            DType::F8E4M3,
            &make_zero_rope_cs(head_dim, 16, &device)?,
        )?;
        assert_eq!(out.dims(), &[1, n_head, head_dim]);
        let max_abs = out
            .to_dtype(DType::F32)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        assert!(max_abs.is_finite(), "fp8 hd128: not finite: {max_abs}");
        assert!(max_abs < 100.0, "fp8 hd128: values too large: {max_abs}");
        Ok(())
    }

    #[test]
    fn test_fp8_hd64_paged_decode() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let (n_head, n_kv_head, head_dim) = (32, 8, 64);
        let q =
            Tensor::randn(0f32, 1f32, (1, n_head, head_dim), &device)?.to_dtype(DType::BF16)?;
        let k_new =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(DType::BF16)?;
        let v_new =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(DType::BF16)?;
        let hk =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, 8, head_dim), &device)?
                .to_dtype(DType::BF16)?;
        let hv = hk.zeros_like()?;

        let out = run_paged_decode(
            Some((&hk, &hv)),
            &k_new,
            &v_new,
            &q,
            n_head,
            n_kv_head,
            head_dim,
            DType::F8E4M3,
            &make_zero_rope_cs(head_dim, 16, &device)?,
        )?;
        assert_eq!(out.dims(), &[1, n_head, head_dim]);
        let max_abs = out
            .to_dtype(DType::F32)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        assert!(max_abs.is_finite(), "fp8 hd64: not finite: {max_abs}");
        assert!(max_abs < 100.0, "fp8 hd64: values too large: {max_abs}");
        Ok(())
    }

    // ======================================================================
    // Tests 4-5: Correctness — no prior history (first decode step)
    //
    // With a single KV token the reference degenerates to v_new, but the test
    // still verifies shape, dtype, GQA expansion, and the BF16/F16 dispatch.
    // ======================================================================

    #[test]
    fn correctness_decode_no_history_bf16() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        for &(n_head, n_kv_head, head_dim, label) in &[
            (8, 8, 64, "MHA hd64"),
            (8, 8, 128, "MHA hd128"),
            (32, 8, 64, "GQA 32/8 hd64"),
            (32, 8, 128, "GQA 32/8 hd128"),
            (40, 8, 64, "GQA 40/8 hd64"),
            (40, 8, 128, "GQA 40/8 hd128"),
            (14, 2, 64, "GQA 14/2 hd64"),
            (14, 2, 128, "GQA 14/2 hd128"),
            (8, 1, 64, "MQA hd64"),
            (8, 1, 128, "MQA hd128"),
        ] {
            let q =
                Tensor::randn(0f32, 1f32, (1, n_head, head_dim), &device)?.to_dtype(dtype)?;
            let k_new =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
            let v_new =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;

            let paged_out = run_paged_decode(
                None,
                &k_new,
                &v_new,
                &q,
                n_head,
                n_kv_head,
                head_dim,
                dtype,
                &make_zero_rope_cs(head_dim, 16, &device)?,
            )?;

            // Reference: single KV token — attend over k_new/v_new only.
            let full_k = k_new.unsqueeze(2)?; // (1, n_kv_head, 1, head_dim)
            let full_v = v_new.unsqueeze(2)?;
            let ref_out =
                reference_decode_attention(&q, &full_k, &full_v, n_head, n_kv_head, head_dim)?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;
            assert!(
                mae < 0.05,
                "[{label}] BF16 decode no-history mean error too large: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] BF16 decode no-history max error too large: {max_err}"
            );
            println!("[{label}] BF16 decode no-history OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    #[test]
    fn correctness_decode_no_history_f16() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::F16;
        for &(n_head, n_kv_head, head_dim, label) in &[
            (8, 8, 64, "MHA hd64"),
            (32, 8, 64, "GQA 32/8 hd64"),
            (32, 8, 128, "GQA 32/8 hd128"),
            (40, 8, 64, "GQA 40/8 hd64"),
            (14, 2, 64, "GQA 14/2 hd64"),
            (8, 1, 64, "MQA hd64"),
        ] {
            let q =
                Tensor::randn(0f32, 1f32, (1, n_head, head_dim), &device)?.to_dtype(dtype)?;
            let k_new =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
            let v_new =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;

            let paged_out = run_paged_decode(
                None,
                &k_new,
                &v_new,
                &q,
                n_head,
                n_kv_head,
                head_dim,
                dtype,
                &make_zero_rope_cs(head_dim, 16, &device)?,
            )?;

            let full_k = k_new.unsqueeze(2)?;
            let full_v = v_new.unsqueeze(2)?;
            let ref_out =
                reference_decode_attention(&q, &full_k, &full_v, n_head, n_kv_head, head_dim)?;

            let paged_f32 = paged_out.to_dtype(DType::F32)?;
            let mae = mean_abs_error(&paged_f32, &ref_out)?;
            let max_err = max_abs_error(&paged_f32, &ref_out)?;
            assert!(
                mae < 0.05,
                "[{label}] F16 decode no-history mean error too large: {mae}"
            );
            assert!(
                max_err < 0.2,
                "[{label}] F16 decode no-history max error too large: {max_err}"
            );
            println!("[{label}] F16 decode no-history OK: mae={mae:.4e} max_err={max_err:.4e}");
        }
        Ok(())
    }

    // ======================================================================
    // Tests 6-7: Correctness — with prior history (multi-turn decode)
    //
    // These are the primary accuracy tests: they verify the kernel correctly
    // reads the chunked arena and computes attention over history + new token.
    // ======================================================================

    #[test]
    fn correctness_decode_with_history_bf16() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        for &(n_head, n_kv_head, head_dim, history_len, label) in &[
            (8, 8, 64, 10, "MHA hd64 hist=10"),
            (8, 8, 64, 32, "MHA hd64 hist=32"),
            (8, 8, 128, 10, "MHA hd128 hist=10"),
            (32, 8, 64, 10, "GQA 32/8 hd64 hist=10"),
            (32, 8, 64, 64, "GQA 32/8 hd64 hist=64"),
            (32, 8, 128, 10, "GQA 32/8 hd128 hist=10"),
            (40, 8, 64, 10, "GQA 40/8 hd64 hist=10"),
            (14, 2, 64, 10, "GQA 14/2 hd64 hist=10"),
            (8, 1, 64, 10, "MQA hd64 hist=10"),
            (8, 1, 128, 10, "MQA hd128 hist=10"),
            // head_dim=256. hpg<=8 takes the full-perf stripe path; the 16/1
            // config (hpg=16>8) takes the wide warp=head path, which runs
            // single-stage at hd256 to fit shared memory.
            (8, 8, 256, 10, "MHA hd256 hist=10 (stripe)"),
            (32, 8, 256, 10, "GQA 32/8 hd256 hist=10 (stripe)"),
            (16, 1, 256, 10, "GQA 16/1 hd256 hist=10 (wide)"),
        ] {
            let hk = Tensor::randn(
                0f32,
                1f32,
                (1, n_kv_head, history_len, head_dim),
                &device,
            )?
            .to_dtype(dtype)?;
            let hv = Tensor::randn(
                0f32,
                1f32,
                (1, n_kv_head, history_len, head_dim),
                &device,
            )?
            .to_dtype(dtype)?;
            let k_new =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
            let v_new =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
            let q =
                Tensor::randn(0f32, 1f32, (1, n_head, head_dim), &device)?.to_dtype(dtype)?;

            assert_decode_ab(
                Some((&hk, &hv)),
                &k_new,
                &v_new,
                &q,
                n_head,
                n_kv_head,
                head_dim,
                dtype,
                &make_zero_rope_cs(head_dim, 16, &device)?,
                label,
            )?;
        }
        Ok(())
    }

    #[test]
    fn correctness_decode_with_history_f16() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::F16;
        for &(n_head, n_kv_head, head_dim, history_len, label) in &[
            (8, 8, 64, 10, "MHA hd64 hist=10"),
            (32, 8, 64, 10, "GQA 32/8 hd64 hist=10"),
            (32, 8, 128, 10, "GQA 32/8 hd128 hist=10"),
            (40, 8, 64, 10, "GQA 40/8 hd64 hist=10"),
            (14, 2, 64, 10, "GQA 14/2 hd64 hist=10"),
            (8, 1, 64, 10, "MQA hd64 hist=10"),
        ] {
            let hk = Tensor::randn(
                0f32,
                1f32,
                (1, n_kv_head, history_len, head_dim),
                &device,
            )?
            .to_dtype(dtype)?;
            let hv = Tensor::randn(
                0f32,
                1f32,
                (1, n_kv_head, history_len, head_dim),
                &device,
            )?
            .to_dtype(dtype)?;
            let k_new =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
            let v_new =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
            let q =
                Tensor::randn(0f32, 1f32, (1, n_head, head_dim), &device)?.to_dtype(dtype)?;

            assert_decode_ab(
                Some((&hk, &hv)),
                &k_new,
                &v_new,
                &q,
                n_head,
                n_kv_head,
                head_dim,
                dtype,
                &make_zero_rope_cs(head_dim, 16, &device)?,
                label,
            )?;
        }
        Ok(())
    }

    // ======================================================================
    // Test 8: GQA head-mapping regression
    //
    // Same tricky num_groups-is-odd configurations as the prefill regression
    // test; validates that the decode kernel's GQA grouping arithmetic is
    // correct for non-power-of-2 group sizes.
    // ======================================================================

    #[test]
    fn correctness_decode_gqa_head_mapping_regression() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let history_len = 10usize;
        for &(n_head, n_kv_head, head_dim, label) in &[
            (40, 8, 64, "40/8 hd64"),   // num_groups=5
            (40, 8, 128, "40/8 hd128"),
            (28, 4, 64, "28/4 hd64"),   // num_groups=7
            (48, 8, 64, "48/8 hd64"),   // num_groups=6
            (56, 8, 64, "56/8 hd64"),   // num_groups=7
            (56, 8, 128, "56/8 hd128"),
            (12, 4, 64, "12/4 hd64"),   // num_groups=3
        ] {
            let hk = Tensor::randn(
                0f32,
                1f32,
                (1, n_kv_head, history_len, head_dim),
                &device,
            )?
            .to_dtype(dtype)?;
            let hv = Tensor::randn(
                0f32,
                1f32,
                (1, n_kv_head, history_len, head_dim),
                &device,
            )?
            .to_dtype(dtype)?;
            let k_new =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
            let v_new =
                Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
            let q =
                Tensor::randn(0f32, 1f32, (1, n_head, head_dim), &device)?.to_dtype(dtype)?;

            assert_decode_ab(
                Some((&hk, &hv)),
                &k_new,
                &v_new,
                &q,
                n_head,
                n_kv_head,
                head_dim,
                dtype,
                &make_zero_rope_cs(head_dim, 16, &device)?,
                label,
            )?;
        }
        Ok(())
    }

    // ======================================================================
    // Test 9: Per-head diagnostic
    //
    // Uses the same 40/8 config that exposed the GQA overflow bug in prefill.
    // Prints per-head MAE so regressions can be attributed to specific heads.
    // ======================================================================

    #[test]
    fn correctness_decode_diagnostic_per_head() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let (n_head, n_kv_head, head_dim, history_len) = (40, 8, 64, 10);
        let num_groups = n_head / n_kv_head;

        let hk = Tensor::randn(0f32, 1f32, (1, n_kv_head, history_len, head_dim), &device)?
            .to_dtype(dtype)?;
        let hv = Tensor::randn(0f32, 1f32, (1, n_kv_head, history_len, head_dim), &device)?
            .to_dtype(dtype)?;
        let k_new =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
        let v_new =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
        let q =
            Tensor::randn(0f32, 1f32, (1, n_head, head_dim), &device)?.to_dtype(dtype)?;

        let rope_cs = make_zero_rope_cs(head_dim, 16, &device)?;
        let int8 = run_paged_decode_be(
            DecodeBackend::Int8,
            Some((&hk, &hv)),
            &k_new,
            &v_new,
            &q,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
            &rope_cs,
        )?
        .to_dtype(DType::F32)?;

        let full_k = Tensor::cat(&[&hk, &k_new.unsqueeze(2)?], 2)?;
        let full_v = Tensor::cat(&[&hv, &v_new.unsqueeze(2)?], 2)?;
        let ref_out =
            reference_decode_attention(&q, &full_k, &full_v, n_head, n_kv_head, head_dim)?;

        // Per-head int8-vs-reference divergence: a GQA head-mapping bug shows up
        // as one kv_group's heads diverging far beyond int8 quantization noise.
        println!("\n=== Per-head int8-vs-reference decode error for {n_head}/{n_kv_head} (groups={num_groups}) ===");
        for h in 0..n_head {
            let kv_group = h / num_groups;
            let int8_h = int8.narrow(1, h, 1)?;
            let ref_h = ref_out.narrow(1, h, 1)?;
            let mae = mean_abs_error(&int8_h, &ref_h)?;
            let marker = if mae > 0.15 { " *** HIGH ***" } else { "" };
            println!("  head {h:2} (kv_group {kv_group}): mae={mae:.4e}{marker}");
        }

        let overall = mean_abs_error(&int8, &ref_out)?;
        assert!(
            overall < 0.15,
            "per-head diagnostic: overall int8-vs-reference mean error too large: {overall}"
        );
        Ok(())
    }

    // ======================================================================
    // Tests 10-12: RoPE plumbing
    // ======================================================================

    #[test]
    fn rope_offset_decode_none_succeeds() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let (n_head, n_kv_head, head_dim) = (8, 8, 64);

        let q =
            Tensor::randn(0f32, 1f32, (1, n_head, head_dim), &device)?.to_dtype(dtype)?;
        let k_new =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
        let v_new =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;

        let out = run_paged_decode(
            None,
            &k_new,
            &v_new,
            &q,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
            &make_zero_rope_cs(head_dim, 16, &device)?,
        )?;
        assert_eq!(out.dims(), &[1, n_head, head_dim]);
        let max_abs = out
            .to_dtype(DType::F32)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_vec0::<f32>()?;
        assert!(
            max_abs.is_finite(),
            "rope=zero decode: output not finite: {max_abs}"
        );
        println!("rope_offset_decode_none_succeeds OK: max_abs={max_abs:.4e}");
        Ok(())
    }

    /// With a multi-token history, using real RoPE vs zero RoPE produces
    /// different outputs because Q and K_new are rotated differently,
    /// changing the relative attention weights against the stored (unrotated)
    /// history keys.
    #[test]
    fn rope_offset_decode_real_vs_zero_differs() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let (n_head, n_kv_head, head_dim, history_len) = (8, 8, 64, 16);

        let hk = Tensor::randn(0f32, 1f32, (1, n_kv_head, history_len, head_dim), &device)?
            .to_dtype(dtype)?;
        let hv = Tensor::randn(0f32, 1f32, (1, n_kv_head, history_len, head_dim), &device)?
            .to_dtype(dtype)?;
        let k_new =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
        let v_new =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
        let q =
            Tensor::randn(0f32, 1f32, (1, n_head, head_dim), &device)?.to_dtype(dtype)?;

        let out_zero = run_paged_decode(
            Some((&hk, &hv)),
            &k_new,
            &v_new,
            &q,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
            &make_zero_rope_cs(head_dim, 16, &device)?,
        )?;

        let out_real = run_paged_decode(
            Some((&hk, &hv)),
            &k_new,
            &v_new,
            &q,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
            &make_test_rope_cs(head_dim, 16, &device)?,
        )?;

        let zero_f32 = out_zero.to_dtype(DType::F32)?;
        let real_f32 = out_real.to_dtype(DType::F32)?;
        assert!(
            zero_f32
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_vec0::<f32>()?
                .is_finite()
        );
        assert!(
            real_f32
                .abs()?
                .flatten_all()?
                .max(0)?
                .to_vec0::<f32>()?
                .is_finite()
        );

        let mae = mean_abs_error(&zero_f32, &real_f32)?;
        println!("rope_offset_decode_real_vs_zero_differs: mae={mae:.4e}");
        assert!(
            mae > 1e-4,
            "zero-rope and real-rope decode outputs are identical — RoPE is not applied: mae={mae:.4e}"
        );
        Ok(())
    }

    /// Verify: paged_decode(un-rotated q/k, rope=identity at pos 0)
    ///      ≈  reference_decode(manually rotated q/k at pos 0)
    ///
    /// Position 0 gives cos=1, sin=0 (identity rotation), so both branches
    /// see the same effective Q and K — a tight tolerance is expected.
    #[test]
    fn rope_offset_decode_functional() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let dtype = DType::BF16;
        let (n_head, n_kv_head, head_dim) = (4, 4, 64);

        let q =
            Tensor::randn(0f32, 1f32, (1, n_head, head_dim), &device)?.to_dtype(dtype)?;
        let k_new =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;
        let v_new =
            Tensor::randn(0f32, 1f32, (1, n_kv_head, head_dim), &device)?.to_dtype(dtype)?;

        // Branch A: kernel with zero rope (identity) at first decode step (pos=0).
        let out_kernel = run_paged_decode(
            None,
            &k_new,
            &v_new,
            &q,
            n_head,
            n_kv_head,
            head_dim,
            dtype,
            &make_zero_rope_cs(head_dim, 16, &device)?,
        )?;

        // Branch B: manually rotate Q and K at pos=0 (identity), then reference decode.
        // cos(0)=1, sin(0)=0 → rotation is a no-op, so this must match Branch A.
        let q_rot = apply_rope_3d(&q, 0)?;
        let k_rot = apply_rope_3d(&k_new, 0)?;
        let full_k = k_rot.unsqueeze(2)?;
        let full_v = v_new.unsqueeze(2)?;
        let out_ref =
            reference_decode_attention(&q_rot, &full_k, &full_v, n_head, n_kv_head, head_dim)?;

        let mae = mean_abs_error(
            &out_kernel.to_dtype(DType::F32)?,
            &out_ref.to_dtype(DType::F32)?,
        )?;
        println!("rope_offset_decode_functional: mae(kernel+zero_rope vs manual_rot@pos0)={mae:.4e}");
        assert!(
            mae < 0.05,
            "fused-RoPE decode functional mismatch: mae={mae}"
        );
        Ok(())
    }
}
