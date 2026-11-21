//! Launches the INT8 prefix-attention prefill kernel over packed prompts, and
//! states which head dimensions and activation types it serves.
//!
//! The kernel is instantiated for a fixed set of head dimensions and for
//! half-precision activations only. Other widths and f32 sessions take a float
//! fallback that is correct at any width but materialises the score matrix. The
//! output is written into the wave's transient buffer, which is why the launch is
//! an inherent method: a generic op would erase the wave's lifetime from the
//! result. Q, the new K and V and the output share one type, so no cast stands in
//! front of the launch.

use super::*;

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub(super) struct PagedPrefillInt8<'k> {
    /// Longest per-sequence q_len in the batch — sizes the kernel's
    /// query-tile grid exactly.
    pub(super) max_q_len: usize,
    /// The launch's cut, and its carve — the pre-staged K/V, the carry and the
    /// resume table — beside the output, when it stages anything.
    pub(super) cut: PrefillLaunchCut,
    pub(super) softmax_scale: f32,
    pub(super) cu_seqlens_q: Tensor,
    pub(super) q_lens: Tensor,
    pub(super) kv_lens: Tensor,
    pub(super) k_packed: LiveTensor<'k>,
    pub(super) v_packed: LiveTensor<'k>,
    /// Raw GPU virtual address of `SlotHeader[batch_size]`.
    pub(super) headers_ptr: u64,
    pub(super) batch_size: usize,
    pub(super) n_head: usize,
    pub(super) n_kv_head: usize,
    pub(super) head_dim: usize,
    /// Compute dtype for K/V/Q — F16 or BF16. Pre-resolved from K and V arena formats.
    pub(super) compute_dtype: DType,
    /// The model's RoPE rungs (`rope_schedule::RopeRungs`): each sequence
    /// rotates by the rung its `SlotHeader` names.
    pub(super) rungs: RopeRungsFfi,
    /// RoPE pairing style: false=non-interleaved half-split (Qwen/GPT2), true=interleaved adjacent-pairs (Llama).
    pub(super) rope_interleaved: bool,
    /// QSA: the block-sparse selection this layer reads through, one row per
    /// packed query (`cu_seqlens_q[b] + token`). `None` is a full causal read.
    pub(super) qsa: Option<QsaSelection>,
}

#[cfg(feature = "cuda")]
impl<'k> PagedPrefillInt8<'k> {
    pub(super) fn cuda_fwd_t<
        'w,
        Q: candle::cuda_backend::CudaDType + DeviceRepr, // Query type
        KV: candle::cuda_backend::CudaDType + DeviceRepr, // KV cache type
        O: candle::cuda_backend::CudaDType + DeviceRepr, // Output type
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        let (total_q, n_head, head_dim) = q_l.shape().dims3()?;
        if n_head != self.n_head || head_dim != self.head_dim {
            candle::bail!(
                "paged-prefill-int8: q shape mismatch got {:?} expected (total_q, {}, {})",
                q_l.shape(),
                self.n_head,
                self.head_dim
            )
        }

        let dev = q.device();
        let stream = dev.cuda_stream();

        let (cu_seqlens_q_s, cu_seqlens_q_l) = self.cu_seqlens_q.storage_and_layout();
        let (q_lens_s, q_lens_l) = self.q_lens.storage_and_layout();
        let (kv_lens_s, kv_lens_l) = self.kv_lens.storage_and_layout();
        let (k_packed_s, k_packed_l) = self.k_packed.storage_and_layout();
        let (v_packed_s, v_packed_l) = self.v_packed.storage_and_layout();

        if cu_seqlens_q_l.shape().dims1()? != self.batch_size + 1 {
            candle::bail!(
                "paged-prefill-int8: cu_seqlens_q must have len batch+1 ({})",
                self.batch_size + 1
            )
        }
        if q_lens_l.shape().dims1()? != self.batch_size {
            candle::bail!(
                "paged-prefill-int8: q_lens must have len batch ({})",
                self.batch_size
            )
        }
        if kv_lens_l.shape().dims1()? != self.batch_size {
            candle::bail!(
                "paged-prefill-int8: kv_lens must have len batch ({})",
                self.batch_size
            )
        }
        if self.cu_seqlens_q.dtype() != DType::U32
            || self.q_lens.dtype() != DType::U32
            || self.kv_lens.dtype() != DType::U32
        {
            candle::bail!("paged-prefill-int8: cu_seqlens_q/q_lens/kv_lens must be U32")
        }

        let cu_seqlens_q = match &*cu_seqlens_q_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("paged-prefill-int8: cu_seqlens_q must be a cuda tensor"),
        }
        .slice(cu_seqlens_q_l.start_offset()..);

        let q_lens = match &*q_lens_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("paged-prefill-int8: q_lens must be a cuda tensor"),
        }
        .slice(q_lens_l.start_offset()..);

        let kv_lens = match &*kv_lens_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("paged-prefill-int8: kv_lens must be a cuda tensor"),
        }
        .slice(kv_lens_l.start_offset()..);

        let k_packed = match &*k_packed_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
            _ => candle::bail!("paged-prefill-int8: k_packed must be a cuda tensor"),
        }
        .slice(k_packed_l.start_offset()..);

        let v_packed = match &*v_packed_s {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
            _ => candle::bail!("paged-prefill-int8: v_packed must be a cuda tensor"),
        }
        .slice(v_packed_l.start_offset()..);

        let elem_count = q_l.shape().elem_count();
        let dst = KernelOutput::<O>::new(dev, elem_count, wave)?;
        // Carved after the context, as the plan walks the attention chain
        // (`WaveBuffer::PrefillKvStage`) — one buffer of stage, carry and
        // resume table at the layout's offsets — and never zeroed: the
        // pre-staging pass writes every stage row the attention kernel reads,
        // and a carrying block writes its carry rows and resume entry in the
        // launch before the one that reads them (invariant 6).
        let layout = self.cut.layout;
        let carve = if layout.stage_positions > 0 {
            Some(KernelOutput::<u8>::new(dev, layout.bytes(), wave)?)
        } else {
            None
        };

        // Compute q_dtype code from Q's actual dtype
        // q_dtype codes: 0=F32, 1=F16, 2=BF16, 3=F8E4M3
        let q_dtype_code: i32 = match q.dtype() {
            candle::DType::F16 => 1,
            candle::DType::BF16 => 2,
            dt => candle::bail!(
                "paged-prefill: unsupported Q dtype {:?} (only F16/BF16 supported)",
                dt
            ),
        };

        match self.head_dim {
            64 | 128 | 256 => {}
            hd => {
                candle::bail!("paged-prefill: unsupported head_dim {hd} (must be 64, 128, or 256)")
            }
        }

        unsafe {
            let q = q.as_cuda_slice::<Q>()?;
            let q = q.slice(q_l.start_offset()..);
            let (q_ptr, _guard) = q.device_ptr(&stream);
            let (dst_ptr, _guard) = dst.device_ptr(&stream);
            let (k_ptr, _guard) = k_packed.device_ptr(&stream);
            let (v_ptr, _guard) = v_packed.device_ptr(&stream);
            let (cu_ptr, _guard) = cu_seqlens_q.device_ptr(&stream);
            let (q_lens_ptr, _guard) = q_lens.device_ptr(&stream);
            let (kv_lens_ptr, _guard) = kv_lens.device_ptr(&stream);
            let (stage_ptr, _guard) = match &carve {
                Some(s) => s.device_ptr(&stream),
                None => (0, None),
            };
            let (carry_ptr, resume_ptr) = if layout.carry_tokens > 0 && stage_ptr != 0 {
                (
                    stage_ptr + layout.carry_offset() as u64,
                    stage_ptr + layout.resume_offset() as u64,
                )
            } else {
                (0, 0)
            };
            // The kernel applies fused RoPE to Q (in smem) and to new K tokens
            // (k_pos >= prefix_len), each at its own position under its
            // sequence's rung, before computing attention scores and writing
            // the un-rotated K/V to the arena.
            let headers_ptr = self.headers_ptr as *const u8;

            let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
            candle::set_kernel_breadcrumb("run_paged_prefill_int8", file!(), line!());
            QsaSelection::with_kernel_args(
                self.qsa.as_ref(),
                &stream,
                |sel_e, sel_c, sel_pages, sel_win, sel_stride, sel_ratio| {
                    // A bulk launch's shape, which is what its cost follows:
                    // how many key chunks each row group re-stages and walks,
                    // and how wide a row's selection is.
                    if total_q >= PREFILL_KV_STAGE_MIN_Q_LEN {
                        note("bulk prefill attention launch", || {
                            format!(
                                "seqs={} rows={total_q} max_q={} max_kv={} groups={} \
                                 chunks={} group_blocks={} stage_positions={} \
                                 carry_tokens={} selection_stride={sel_stride} paged={}",
                                self.batch_size,
                                self.max_q_len,
                                self.cut.max_kv_len,
                                self.cut.group_chunks.len(),
                                self.cut.group_chunks.iter().sum::<u32>(),
                                layout.group_blocks,
                                layout.stage_positions,
                                layout.carry_tokens,
                                !sel_pages.is_null(),
                            )
                        });
                    }
                    run_paged_prefill_int8(
                        q_ptr as *const core::ffi::c_void,
                        k_ptr as *const core::ffi::c_void,
                        v_ptr as *const core::ffi::c_void,
                        headers_ptr,
                        cu_ptr as *const u32,
                        q_lens_ptr as *const u32,
                        kv_lens_ptr as *const u32,
                        dst_ptr as *mut core::ffi::c_void,
                        total_q as i32,
                        self.batch_size as i32,
                        self.n_head as i32,
                        self.n_kv_head as i32,
                        self.head_dim as i32,
                        self.max_q_len as i32,
                        self.softmax_scale,
                        q_dtype_code,
                        self.rungs,
                        self.rope_interleaved as i32,
                        raw_stream,
                        sel_e,
                        sel_c,
                        sel_pages,
                        sel_win,
                        sel_stride,
                        sel_ratio,
                        stage_ptr as *mut u8,
                        layout.stage_bytes() as i64,
                        layout.stage_positions as i64,
                        PREFILL_KV_STAGE_MIN_Q_LEN as i32,
                        self.cut.max_kv_len as i32,
                        self.cut.group_chunks.as_ptr(),
                        self.cut.group_chunks.len() as i32,
                        layout.group_blocks as i32,
                        layout.chunk_positions as i32,
                        carry_ptr as *mut f32,
                        resume_ptr as *mut u32,
                    )
                },
            )?;
        }

        Ok(dst.into_tensor(dev.clone(), q_l.shape().clone()))
    }
}

/// Dispatch on the compute dtype and run the kernel.
///
/// An inherent method rather than a `CustomOp1` impl: this kernel writes its
/// output into the wave's transient half, and `CustomOp1::cuda_fwd` returns
/// `(CudaStorage, Shape)` with nowhere to carry `'w`. Going through that trait
/// would erase the bound and hand back a `Tensor` claiming to own memory the
/// wave is about to reclaim.
#[cfg(feature = "cuda")]
impl<'k> PagedPrefillInt8<'k> {
    pub(super) fn run<'w>(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        if q.dtype() != self.compute_dtype {
            candle::bail!(
                "paged-prefill-int8: expected {:?} Q, got {:?}",
                self.compute_dtype,
                q.dtype()
            );
        }
        match self.compute_dtype {
            candle::DType::F16 => self.cuda_fwd_t::<f16, f16, f16>(q, q_l, wave),
            candle::DType::BF16 => self.cuda_fwd_t::<bf16, bf16, bf16>(q, q_l, wave),
            dt => candle::bail!("paged-prefill-int8: unsupported compute dtype {:?}", dt),
        }
    }
}

/// Whether the INT8 prefix-attention prefill kernel is instantiated at this
/// head dim.
///
/// The set is fixed by the dispatchers in `paged_prefill_int8_{fp16,bf16}.cu`:
/// 64 and 128 run at 4 blocks/SM on a 2-way output-dim split, 256 at 2
/// blocks/SM on a 4-way split (its staging slabs are ~41 KB, too wide for the
/// 25.6 KB union arena the higher occupancy needs). Head dims outside the set
/// take the float fallback, which is correct at any width but materializes
/// the score matrix per sequence.
pub(crate) fn int8_prefill_head_dim(head_dim: usize) -> bool {
    matches!(head_dim, 64 | 128 | 256)
}

/// Whether the INT8 prefix-attention prefill kernel can serve a session whose
/// activations are `act_dtype`.
///
/// The kernel is instantiated over F16 and BF16 only, and it emits the context
/// in the dtype it computed in. A reference-mode session carrying F32
/// activations would therefore get a BF16 context back, and the first RMSNorm
/// downstream would meet an F32 weight against BF16 activations — the arena's
/// `collapse_compute` maps F32/F64 to BF16 for the kernel's operands, but
/// nothing maps the result back, and casting it per layer is a full-tensor pass
/// on the hot path for a mode that exists to be exact rather than fast.
///
/// So an F32 session takes the float fallback, which computes at the session's
/// own dtype throughout. This is the same kind of boundary as
/// [`int8_prefill_head_dim`]: what the kernel is built for, not a toggle.
pub(crate) fn int8_prefill_act_dtype(act_dtype: DType) -> bool {
    matches!(act_dtype, DType::F16 | DType::BF16)
}

/// Whether the fused q8a128-emitting decode combine serves `head_dim`.
///
/// The set is fixed by the `LAUNCH_Q8` dispatchers in
/// `paged_decode_api_{fp16,bf16}.cu` and by the combine kernel's tile
/// arithmetic (`HEAD_DIM % 128 == 0` — a block is `head_dim/128` whole
/// q8a1024 tiles). The `.cu` switches return without launching for any other
/// width and rely on the Rust side bailing first, so **every** Rust-side
/// check of this set must go through this predicate: a widened literal in
/// one place but not the dispatcher would route `want_q8` to a launcher
/// that silently emits nothing, and `o_proj` would consume uninitialized
/// bytes. Same boundary discipline as [`int8_prefill_head_dim`].
pub(crate) fn paged_decode_q8_head_dim(head_dim: usize) -> bool {
    matches!(head_dim, 128 | 256)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
/// Paged prefill attention over chunked KV arenas (no KV materialization),
/// through the INT8 prefix-attention kernel.
///
/// `q` has shape `(total_q, n_head, head_dim)`.
/// `headers_ptr` is the raw GPU address of `SlotHeader[batch_size]`, reusing the
/// same persistent slot-payload representation as decode.
/// `compute_dtype` is the pre-resolved F16 or BF16 dtype for Q/K/V (derived from arena formats).
/// `cut` is how the launch is cut and what it carves ([`PrefillLaunchCut`]).
pub(super) fn paged_prefill_attn_varlen_chunks<'w>(
    wave: Option<&'w WaveGeneration>,
    q: &LiveTensor<'_>,
    cu_seqlens_q: &Tensor,
    q_lens: &Tensor,
    kv_lens: &Tensor,
    k_packed: &LiveTensor<'_>,
    v_packed: &LiveTensor<'_>,
    headers_ptr: u64,
    compute_dtype: DType,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    softmax_scale: f32,
    rope: &RopeRungs,
    rope_interleaved: bool,
    max_q_len: usize,
    cut: PrefillLaunchCut,
    qsa: Option<&QsaSelection>,
) -> Result<LiveTensor<'w>> {
    // The kernel handles both RoPE pairings (half-split in-thread; interleaved
    // via a lane^1 partner shuffle — see `i8_apply_rope`).
    if !int8_prefill_head_dim(head_dim) {
        candle::bail!("paged-prefill-int8 supports head_dim 64, 128 or 256 (got {head_dim})")
    }

    // Validated, not converted (invariant 1b). `project_qkv` already emits all
    // three at the KV arena's width (`attention_operand_dtype`), which is what
    // this kernel is instantiated on — Q, the new K/V and the output are one
    // template type. A reference session whose arena is F32 never arrives here:
    // `int8_prefill_act_dtype` sends it to the float fallback. So the casts that
    // stood here converted nothing, and would have hidden a producer emitting
    // the wrong width behind three full-tensor passes per layer, per wave.
    expect_dtype(q, compute_dtype, "paged-prefill-int8: q")?;
    expect_dtype(k_packed, compute_dtype, "paged-prefill-int8: k")?;
    expect_dtype(v_packed, compute_dtype, "paged-prefill-int8: v")?;
    let (q, k_packed, v_packed) = (q.clone(), k_packed.clone(), v_packed.clone());

    let (_total_q, q_n_head, q_head_dim) = q.dims3()?;
    if q_n_head != n_head || q_head_dim != head_dim {
        candle::bail!("paged-prefill-int8: q shape mismatch {:?}", q.dims())
    }

    let (kp_total_q, kp_n_kv, kp_hd) = k_packed.dims3()?;
    if kp_total_q != _total_q || kp_n_kv != n_kv_head || kp_hd != head_dim {
        candle::bail!(
            "paged-prefill-int8: k_packed shape mismatch {:?}",
            k_packed.dims()
        )
    }
    let (vp_total_q, vp_n_kv, vp_hd) = v_packed.dims3()?;
    if vp_total_q != _total_q || vp_n_kv != n_kv_head || vp_hd != head_dim {
        candle::bail!(
            "paged-prefill-int8: v_packed shape mismatch {:?}",
            v_packed.dims()
        )
    }

    let batch_size = cu_seqlens_q.dim(0)?.saturating_sub(1);

    let op = PagedPrefillInt8 {
        max_q_len,
        cut,
        softmax_scale,
        cu_seqlens_q: cu_seqlens_q.clone(),
        q_lens: q_lens.clone(),
        kv_lens: kv_lens.clone(),
        k_packed: k_packed.clone(),
        v_packed: v_packed.clone(),
        headers_ptr,
        batch_size,
        n_head,
        n_kv_head,
        head_dim,
        compute_dtype,
        rungs: rope.ffi()?,
        rope_interleaved,
        // The kernel indexes the selection by packed query row, so it must
        // carry exactly one row per query in the batch — a short table would
        // read another sequence's selection and answer with no fault.
        qsa: match qsa {
            None => None,
            Some(s) if s.rows() == _total_q => Some(s.clone()),
            Some(s) => candle::bail!(
                "paged-prefill-int8: QSA selection has {} rows for {_total_q} packed queries",
                s.rows()
            ),
        },
    };
    with_q_storage(&q, |s, l| op.run(s, l, wave))
}

/// Pull `q`'s CUDA storage out and hand it to a kernel's `run`.
///
/// What `Tensor::apply_op1` used to do for these four kernels, minus the part
/// that erased the lifetime. The closure's `'w` is the caller's, so the result
/// is bounded by the caller's wave guard rather than by what `CustomOp1` can
/// express, which is nothing.
#[cfg(feature = "cuda")]
pub(super) fn with_q_storage<'w, F>(q: &LiveTensor<'_>, run: F) -> Result<LiveTensor<'w>>
where
    F: FnOnce(&candle::CudaStorage, &Layout) -> Result<LiveTensor<'w>>,
{
    let (storage, layout) = q.storage_and_layout();
    let candle::Storage::Cuda(cuda) = &*storage else {
        candle::bail!(
            "paged attention: q must be a CUDA tensor, got {:?}",
            q.device()
        )
    };
    run(cuda, layout)
}
