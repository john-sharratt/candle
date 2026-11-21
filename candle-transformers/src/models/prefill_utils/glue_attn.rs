//! Runs the reprojection glue forward, in which glue queries are written into
//! gaps reserved in the slot and attend over the whole slot.
//!
//! Each glue query's keys and values are scattered into its reserved gap. The
//! gaps already count toward the slot length, so the forward adds nothing to it.
//! Positions come from each chunk's rope base, the convention decode reads. The
//! output is returned at the arena's compute type and is not reconciled with the
//! caller's activation type, because the projection that consumes it converts
//! inside the store it was already making.

use super::*;

/// CPU stub: the paged-glue forward is a CUDA-only kernel path.
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn paged_glue_attn(
    // As `paged_prefill_batched`: concrete so `None` infers.
    _wave: Option<&()>,
    _caches: &mut [&mut KvCache],
    _offsets: &[usize],
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _b_sz: usize,
    _q_lens: &[usize],
    _n_head: usize,
    _n_kv_head: usize,
    _head_dim: usize,
    _prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
    _glue_write_slice: &Tensor,
    _glue_write_in_blk: &Tensor,
    _fwd_ahead: &Tensor,
    _rope: &RopeRungs,
    _rope_interleaved: bool,
    _generation: &Generation,
    _shared_pm: &std::cell::RefCell<Option<SharedPm>>,
) -> Result<Tensor> {
    candle::bail!("paged-glue requires the cuda feature")
}

/// FFI signature shared by the FP16/BF16 glue-kernel entry points.
#[cfg(feature = "cuda")]
pub(super) type GlueFfi = unsafe extern "C" fn(
    *const c_void, // q
    *const u8,     // headers
    *mut c_void,   // o
    i32,           // batch
    i32,           // max_glue
    i32,           // total_q (Σ q_lens — sizes the split-KV partial pool)
    i32,           // max_kv (max kv_lens[b] — sizes the split-KV grid)
    i32,           // n_q_head
    i32,           // n_kv_head
    i32,           // head_dim
    f32,           // softmax_scale
    *const c_void, // k_new
    *const c_void, // v_new
    RopeRungsFfi,  // rungs
    i32,           // rope_interleaved
    *const u32,    // cu_seqlens_q
    *const u32,    // q_lens
    *const u32,    // kv_lens
    *const u32,    // glue_write_slice
    *const u32,    // glue_write_in_blk
    *const u32,    // fwd_ahead
    *mut c_void,   // stream
);

/// Op for the paged-glue reprojection forward. Mirrors [`PagedPrefillInt8`]
/// but launches the decode-derivative glue kernel: each slot's `G` glue queries
/// attend its quantized sealed prefix (streamed-once dequant) plus earlier glue,
/// write their own K/V into the writer chunks, and mask by TRUE sequence
/// position via `col_actual_pos`. The output is `q`-shaped.
#[cfg(feature = "cuda")]
pub(super) struct PagedGlueChunks<'k> {
    pub(super) softmax_scale: f32,
    pub(super) cu_seqlens_q: Tensor,
    pub(super) q_lens: Tensor,
    pub(super) kv_lens: Tensor,
    pub(super) k_new: LiveTensor<'k>,
    pub(super) v_new: LiveTensor<'k>,
    /// Flat `[Σ q_lens]` U32 — gap chunk slice index per glue row (scatter target).
    pub(super) glue_write_slice: Tensor,
    /// Flat `[Σ q_lens]` U32 — in-block offset per glue row.
    pub(super) glue_write_in_blk: Tensor,
    /// Flat `[Σ q_lens]` U32 — forward bridge window per glue row (`0` == causal).
    pub(super) fwd_ahead: Tensor,
    pub(super) headers_ptr: u64,
    pub(super) batch_size: usize,
    pub(super) max_glue: usize,
    /// Max `kv_lens[b]` over the batch — sizes the kernel's split-KV grid.
    pub(super) max_kv: usize,
    pub(super) n_head: usize,
    pub(super) n_kv_head: usize,
    pub(super) head_dim: usize,
    pub(super) compute_dtype: DType,
    pub(super) rungs: RopeRungsFfi,
    pub(super) rope_interleaved: bool,
}

#[cfg(feature = "cuda")]
impl<'k> PagedGlueChunks<'k> {
    pub(super) fn cuda_fwd_t<'w, T: candle::cuda_backend::CudaDType + DeviceRepr>(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        ffi: GlueFfi,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        let (total_q, n_head, head_dim) = q_l.shape().dims3()?;
        if n_head != self.n_head || head_dim != self.head_dim {
            candle::bail!(
                "paged-glue: q shape mismatch got {:?} expected (total_q, {}, {})",
                q_l.shape(),
                self.n_head,
                self.head_dim
            )
        }
        let dev = q.device();
        let stream = dev.cuda_stream();
        let raw_stream = stream.cu_stream() as *mut c_void;

        // U32 metadata: extract the raw device address. Each source tensor is a
        // field of `self`, so its storage outlives this call; the device_ptr
        // guard is only needed for stream ordering, which the launch preserves.
        let u32_ptr = |t: &Tensor| -> Result<u64> {
            let (s, l) = t.storage_and_layout();
            let sl = match &*s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
                _ => candle::bail!("paged-glue: expected cuda u32 tensor"),
            }
            .slice(l.start_offset()..);
            let (p, _g) = sl.device_ptr(&stream);
            Ok(p)
        };
        let cu_ptr = u32_ptr(&self.cu_seqlens_q)?;
        let ql_ptr = u32_ptr(&self.q_lens)?;
        let kv_ptr = u32_ptr(&self.kv_lens)?;
        let gws_ptr = u32_ptr(&self.glue_write_slice)?;
        let gwi_ptr = u32_ptr(&self.glue_write_in_blk)?;
        let fa_ptr = u32_ptr(&self.fwd_ahead)?;

        // Typed Q/K/V (compute dtype).
        let q_slice = q.as_cuda_slice::<T>()?.slice(q_l.start_offset()..);
        let (q_ptr, _qg) = q_slice.device_ptr(&stream);
        let (k_s, k_l) = self.k_new.storage_and_layout();
        let k_slice = match &*k_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("paged-glue: k_new must be a cuda tensor"),
        }
        .slice(k_l.start_offset()..);
        let (k_ptr, _kg) = k_slice.device_ptr(&stream);
        let (v_s, v_l) = self.v_new.storage_and_layout();
        let v_slice = match &*v_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("paged-glue: v_new must be a cuda tensor"),
        }
        .slice(v_l.start_offset()..);
        let (v_ptr, _vg) = v_slice.device_ptr(&stream);

        let elem_count = q_l.shape().elem_count();
        let dst = KernelOutput::<T>::new(dev, elem_count, wave)?;

        unsafe {
            let (dst_ptr, _dg) = dst.device_ptr(&stream);
            candle::set_kernel_breadcrumb("run_paged_glue", file!(), line!());
            ffi(
                q_ptr as *const c_void,
                self.headers_ptr as *const u8,
                dst_ptr as *mut c_void,
                self.batch_size as i32,
                self.max_glue as i32,
                total_q as i32,
                self.max_kv as i32,
                self.n_head as i32,
                self.n_kv_head as i32,
                self.head_dim as i32,
                self.softmax_scale,
                k_ptr as *const c_void,
                v_ptr as *const c_void,
                self.rungs,
                self.rope_interleaved as i32,
                cu_ptr as *const u32,
                ql_ptr as *const u32,
                kv_ptr as *const u32,
                gws_ptr as *const u32,
                gwi_ptr as *const u32,
                fa_ptr as *const u32,
                raw_stream,
            );
        }

        Ok(dst.into_tensor(dev.clone(), q_l.shape().clone()))
    }
}

/// As [`PagedPrefillInt8::run`]: inherent so the wave-scoped output keeps `'w`.
#[cfg(feature = "cuda")]
impl<'k> PagedGlueChunks<'k> {
    pub(super) fn run<'w>(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        if q.dtype() != self.compute_dtype {
            candle::bail!(
                "paged-glue-chunks: expected {:?} Q, got {:?}",
                self.compute_dtype,
                q.dtype()
            );
        }
        match self.compute_dtype {
            DType::F16 => self.cuda_fwd_t::<f16>(q, q_l, run_paged_glue_fp16, wave),
            DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l, run_paged_glue_bf16, wave),
            dt => candle::bail!("paged-glue-chunks: unsupported compute dtype {:?}", dt),
        }
    }
}

/// Reprojection glue forward over chunked KV. Each slot's `q_lens[i]` glue
/// queries are reserved IN PLACE as gap chunks at their logical positions; this
/// forward scatters each query's K/V into its gap (`glue_write_slice/in_blk`)
/// and computes its attention over the whole slot. Every column's sequence
/// position comes from its chunk `rope_base` (`slice_rope`) — the same
/// convention decode reads — so there is no `col_actual_pos`. Each glue token
/// attends backward over everything and forward up to `fwd_ahead[t]` tokens (its
/// bridge window). HD128 only — other head dims stay on the plain prefill path.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn paged_glue_attn<'w>(
    wave: Option<&'w WaveGeneration>,
    caches: &mut [&mut KvCache],
    offsets: &[usize],
    q: &LiveTensor<'_>,
    k: &LiveTensor<'_>,
    v: &LiveTensor<'_>,
    b_sz: usize,
    q_lens: &[usize],
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    prefill_meta: Option<(&Tensor, &Tensor, &Tensor)>,
    glue_write_slice: &Tensor,
    glue_write_in_blk: &Tensor,
    fwd_ahead: &Tensor,
    rope: &RopeRungs,
    rope_interleaved: bool,
    generation: &Generation,
    shared_pm: &std::cell::RefCell<Option<SharedPm>>,
) -> Result<LiveTensor<'w>> {
    if head_dim != 128 {
        candle::bail!("paged-glue requires head_dim==128 (got {head_dim})");
    }
    let total_q: usize = q_lens.iter().sum();
    let max_glue = q_lens.iter().copied().max().unwrap_or(0);
    let _ = offsets; // slot geometry is read from the cache, not the caller offset

    // Glue fires over an already-sealed prefix, so the caches must be chunked.
    let use_chunks = caches
        .first()
        .and_then(|c| c.k_cache().chunked_arena_chunks())
        .is_some();
    if !use_chunks {
        candle::bail!("paged-glue requires chunked caches");
    }

    // The gaps are already reserved chunks (the slot length counts them) — no
    // capacity allocation here. `kv_len` is the slot's actual length, read from
    // the cache so it is independent of how the caller accounts the offset.
    let kv_lens_host: Vec<usize> = caches.iter().map(|c| c.current_seq_len()).collect();

    let (compute_dtype, _max_blocks) = {
        let first = caches
            .first()
            .ok_or_else(|| candle::Error::Msg("expected non-empty caches".into()))?;
        let k_compute = first.k_cache().dtype();
        let v_compute = first.v_cache().dtype();
        if k_compute != v_compute {
            candle::bail!(
                "paged-glue: K/V compute dtype mismatch: K={k_compute:?} V={v_compute:?}"
            );
        }
        (k_compute, first.k_cache().chunked_max_blocks())
    };

    let q_packed = if q.is_contiguous() {
        q.clone()
    } else {
        q.contiguous()?
    };
    let k_packed = if k.is_contiguous() {
        k.clone()
    } else {
        k.contiguous()?
    };
    let v_packed = if v.is_contiguous() {
        v.clone()
    } else {
        v.contiguous()?
    };

    let device = q.device();
    let _ = total_q;
    let (cu_seqlens_q, q_lens_dev) = if let Some((cu, ql, _kv)) = prefill_meta {
        (cu.clone(), ql.clone())
    } else {
        let mut cu = Vec::with_capacity(b_sz + 1);
        cu.push(0u32);
        let mut acc = 0u32;
        for &l in q_lens {
            acc += l as u32;
            cu.push(acc);
        }
        let cu_seqlens_q = Tensor::from_vec(cu, b_sz + 1, device)?;
        let q_lens_dev = Tensor::from_vec(
            q_lens.iter().map(|&l| l as u32).collect::<Vec<_>>(),
            b_sz,
            device,
        )?;
        (cu_seqlens_q, q_lens_dev)
    };
    // `kv_len` is the slot's actual length (the reserved gaps ARE the glue tokens,
    // already counted) — independent of how the caller accounts the offset.
    let kv_lens = Tensor::from_vec(
        kv_lens_host.iter().map(|&l| l as u32).collect::<Vec<_>>(),
        b_sz,
        device,
    )?;

    let g_hdr = gpu_span("glue:hdr_meta", device);
    // The gaps are real chunks already (no trailing write region), so the slot
    // headers cover exactly `[0, kv_len)`; pass zero glue so build_slot_headers
    // does not extend a write region. `shared_pm` is the forward's glue-group
    // position-map cache: the map is layer-invariant (chunk boundaries are the
    // same in every layer), so the first layer builds + uploads it and the
    // other 47 reuse the device buffer, skipping the host build and the PCIe
    // copy that otherwise dominate this span.
    let zero_q = vec![0usize; b_sz];
    let header_upload = build_slot_headers(
        caches,
        &zero_q,
        generation,
        shared_pm,
        &kv_lens_host,
        rope,
        device,
        wave,
    )?;
    g_hdr.end();

    let g_kernel = gpu_span("glue:kernel", device);
    let softmax_scale = 1f32 / (head_dim as f32).sqrt();
    let op = PagedGlueChunks {
        softmax_scale,
        cu_seqlens_q,
        q_lens: q_lens_dev,
        kv_lens,
        k_new: k_packed.to_dtype(compute_dtype)?,
        v_new: v_packed.to_dtype(compute_dtype)?,
        // Per-token gap scatter target + forward bridge window, from the caller.
        glue_write_slice: glue_write_slice.clone(),
        glue_write_in_blk: glue_write_in_blk.clone(),
        fwd_ahead: fwd_ahead.clone(),
        headers_ptr: header_upload.headers_ptr,
        batch_size: b_sz,
        max_glue,
        max_kv: kv_lens_host.iter().copied().max().unwrap_or(0),
        n_head,
        n_kv_head,
        head_dim,
        compute_dtype,
        rungs: rope.ffi()?,
        rope_interleaved,
    };
    let q_compute = q_packed.to_dtype(compute_dtype)?;
    let out = with_q_storage(&q_compute, |s, l| op.run(s, l, wave))?;
    g_kernel.end();

    // NO advance here: the glue tokens are reserved IN PLACE as gap chunks whose
    // `usage` is already counted in each slot's length (`reserve_glue_gap`). This
    // forward only SCATTERS their K/V into those gaps — it does not append a
    // trailing region. Advancing by `add` again (the old trailing-glue design)
    // would double-count the slot length and, because `writer_start` points past
    // the last chunk, `set_len`'s `writer_start.min(n-1)` fallback would inflate
    // the final gap chunk's usage — desyncing the slot and overflowing the next
    // prefill's write region. Nothing is primed for the decode that follows,
    // for the reason `paged_prefill_batched_impl` gives.
    drop(header_upload);
    // Returned at the ARENA's compute dtype, which is not always the caller's
    // activation dtype — and deliberately NOT reconciled here.
    //
    // Nothing downstream asserts the two are equal. The context's only consumers
    // are the attention gate and the out-projection: the gate belongs to the
    // Qwen3-Next/Qwen3.5 lineage, which is `head_dim` 256 and so is refused by
    // this path's 128-only route gate — it is always `None` here — and
    // `output_projection` names the residual's width as its store type, so the
    // conversion happens inside a GEMM that was running anyway.
    //
    // A `to_dtype` here would therefore be a full-tensor pass that converts
    // nothing in every reachable configuration, and in the one configuration it
    // could convert (an F32 reference session over a half-typed arena) it would
    // duplicate what the projection's store already does — hot-path invariant 1b.
    // If a gated `head_dim` 128 model is ever added, `apply_attention_gate`'s
    // `expect_dtype` is the right place for it to fail: loudly, at the boundary
    // that introduced the mismatch, rather than absorbed by a cast here.
    Ok(out)
}
