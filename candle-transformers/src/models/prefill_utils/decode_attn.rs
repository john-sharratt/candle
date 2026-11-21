//! Runs paged decode attention over a slot pool, and can emit the context as the
//! quantised operand the output projection reads.
//!
//! The kernel reads each slot's header, including its RoPE rung, and advances the
//! write length itself after scattering, so no per-step offsets are passed. A
//! selection must carry one row per active slot, because a short table would
//! give a slot another slot's selection with no fault, so a mismatch is refused.
//! The quantised variant writes the context in the layout the int8 output
//! projection consumes, folds the output gate in, and reads a strided gate in
//! place rather than copying it.

use super::*;

// Decode kernel — persistent slot buffer edition

/// Paged decode attention using persistent slot buffers.
///
/// Takes the slot pool's `SlotHeader[n_active]` instead of the old per-step
/// chunk_meta / head_gids / kv_lens / per_head_table; each header names its
/// slot's RoPE rung in `rope`.
/// The kernel self-increments ws.len after scatter, so no write_offsets needed.
///
/// Runs the production INT8 split-KV / warp-stripe / batched-M decode kernel
/// (`run_paged_decode_*`) for head_dim 64/128/256.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn<'w>(
    wave: Option<&'w WaveGeneration>,
    q: &LiveTensor<'_>,
    headers_ptr: u64,
    arena_dtype: DType,
    n_q_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    softmax_scale: f32,
    k_new: &LiveTensor<'_>,
    v_new: &LiveTensor<'_>,
    rope: &RopeRungs,
    rope_interleaved: bool,
    qsa: Option<&QsaSelection>,
) -> Result<LiveTensor<'w>> {
    let num_active_slots = q.dim(0)?;
    let k_new = k_new.contiguous()?;
    let v_new = v_new.contiguous()?;
    let qsa = validated_decode_qsa(qsa, num_active_slots)?;
    let op = PagedDecode {
        headers_ptr,
        arena_dtype,
        n_q_head,
        n_kv_head,
        head_dim,
        softmax_scale,
        k_new,
        v_new,
        rungs: rope.ffi()?,
        rope_interleaved,
        num_active_slots,
        emit_q8: false,
        gate: None,
        gate_slot_stride: 0,
        qsa,
    };
    with_q_storage(q, |s, l| op.run(s, l, wave))
}

/// The decode path indexes the selection by SLOT, so it must carry exactly one
/// row per active slot. A mismatch would silently read another slot's
/// selection — a wrong answer with no fault — so it is refused here.
#[cfg(feature = "cuda")]
pub(super) fn validated_decode_qsa(
    qsa: Option<&QsaSelection>,
    num_active_slots: usize,
) -> Result<Option<QsaSelection>> {
    match qsa {
        None => Ok(None),
        Some(s) if s.rows() == num_active_slots => Ok(Some(s.clone())),
        Some(s) => candle::bail!(
            "paged-decode: QSA selection has {} rows for {num_active_slots} active slots",
            s.rows()
        ),
    }
}

/// B2 decode: like [`paged_decode_attn`] but the combine kernel emits the attention context
/// directly as q8a1024 blocks (head_dim 128 or 256). Returns a flat `[q8_bytes]` U8 tensor; the
/// caller wraps it as a `Q8a128Operand` (`rows = num_active_slots`, `cols = n_q_head·head_dim`)
/// and feeds `o_proj` via the int8 path — no FP store + standalone quantize.
///
/// `gate` (same dtype as `q`, `num_active_slots × n_q_head·head_dim` elements, model dim
/// order) folds the output gate `sigmoid(g) ⊙ ctx` into the same emit — the gated lineages'
/// post-attention gate with no extra launches and no FP context round-trip.
#[allow(clippy::too_many_arguments)]
#[cfg(feature = "cuda")]
pub fn paged_decode_attn_q8<'w>(
    wave: Option<&'w WaveGeneration>,
    q: &LiveTensor<'_>,
    headers_ptr: u64,
    arena_dtype: DType,
    n_q_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    softmax_scale: f32,
    k_new: &LiveTensor<'_>,
    v_new: &LiveTensor<'_>,
    rope: &RopeRungs,
    rope_interleaved: bool,
    gate: Option<&LiveTensor<'_>>,
    qsa: Option<&QsaSelection>,
) -> Result<LiveTensor<'w>> {
    let num_active_slots = q.dim(0)?;
    let k_new = k_new.contiguous()?;
    let v_new = v_new.contiguous()?;
    let row_width = n_q_head * head_dim;
    let (gate, gate_slot_stride) = match gate {
        Some(g) => {
            let want = num_active_slots * row_width;
            if g.elem_count() != want {
                candle::bail!(
                    "paged-decode q8: gate has {} elements against a {want}-element context",
                    g.elem_count()
                );
            }
            // Read in place — no copy, no layout normalization. The nominal
            // producer is the fused [q|gate] projection's narrow: rows sit the
            // projection's row width apart with a unit inner stride, and the
            // combine kernel takes that slot stride directly. Anything else must
            // arrive contiguous.
            let stride = if g.is_contiguous() {
                row_width as i64
            } else {
                let (dims, strides) = (g.dims(), g.stride());
                if dims.len() == 2
                    && dims[0] == num_active_slots
                    && dims[1] == row_width
                    && strides[1] == 1
                {
                    strides[0] as i64
                } else {
                    candle::bail!(
                        "paged-decode q8: gate must be contiguous or a \
                         [slots, n_head·head_dim] row-strided view with a unit inner \
                         stride; got dims {dims:?} strides {strides:?}"
                    );
                }
            };
            (Some(g.clone()), stride)
        }
        None => (None, 0),
    };
    let op = PagedDecode {
        headers_ptr,
        arena_dtype,
        n_q_head,
        n_kv_head,
        head_dim,
        softmax_scale,
        k_new,
        v_new,
        rungs: rope.ffi()?,
        rope_interleaved,
        num_active_slots,
        emit_q8: true,
        gate,
        gate_slot_stride,
        qsa: validated_decode_qsa(qsa, num_active_slots)?,
    };
    with_q_storage(q, |s, l| op.run(s, l, wave))
}

#[cfg(feature = "cuda")]
pub(super) struct PagedDecode<'k> {
    pub(super) headers_ptr: u64, // raw GPU virtual address of SlotHeader[num_active_slots]
    pub(super) arena_dtype: DType,
    pub(super) n_q_head: usize,
    pub(super) n_kv_head: usize,
    pub(super) head_dim: usize,
    pub(super) softmax_scale: f32,
    pub(super) k_new: LiveTensor<'k>,
    pub(super) v_new: LiveTensor<'k>,
    pub(super) rungs: RopeRungsFfi,
    pub(super) rope_interleaved: bool,
    pub(super) num_active_slots: usize,
    /// B2: when true the combine kernel emits the attention context as q8a1024 blocks (head_dim
    /// 128 or 256) instead of an FP tensor, so `o_proj` consumes it via the int8 path with no
    /// standalone quantize. The op then returns a flat `[q8_bytes]` U8 tensor of the operand bytes.
    pub(super) emit_q8: bool,
    /// Output gate folded into the q8 emit (`sigmoid(g) ⊙ ctx`):
    /// `[num_active_slots × n_q_head·head_dim]` in the context's dim order, same dtype as `q`.
    /// Read in place — a slot's row may sit `gate_slot_stride` elements from the previous
    /// slot's, so the fused `[q | gate]` projection's narrow feeds the kernel with no copy.
    pub(super) gate: Option<LiveTensor<'k>>,
    /// Elements between consecutive slots' gate rows (`n_q_head·head_dim` when contiguous).
    pub(super) gate_slot_stride: i64,
    /// QSA: the block-sparse selection this layer reads through, one row per
    /// slot. `None` is a full causal read — every model but Qwen3.8-Flash-Next's
    /// full-attention layers.
    pub(super) qsa: Option<QsaSelection>,
}

#[cfg(feature = "cuda")]
impl<'k> PagedDecode<'k> {
    /// q8a1024 byte size of the emitted context: `[num_active_slots × (n_q_head·head_dim)]`.
    pub(super) fn q8_byte_size(&self) -> usize {
        let cols = self.n_q_head * self.head_dim;
        let total_tiles = self.num_active_slots * (cols / 128);
        total_tiles.div_ceil(8) * 1152
    }

    pub(super) fn cuda_fwd_typed<
        'w,
        Q: candle::cuda_backend::CudaDType + DeviceRepr + 'static,
        KV: candle::cuda_backend::CudaDType + DeviceRepr,
        O: candle::cuda_backend::CudaDType + DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        ffi_fn: unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const u8,
            *mut core::ffi::c_void,
            i32,
            i32,
            i32,
            i32,
            f32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            RopeRungsFfi,
            i32,
            *mut core::ffi::c_void,
            *const u32,
            *const u32,
            *const u32,
            *const u32,
            i32,
            i32,
        ) -> i32,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        let dev = q.device().clone();
        let stream = dev.cuda_stream();

        let out_elem = self.num_active_slots * self.n_q_head * self.head_dim;
        let dst = KernelOutput::<O>::new(&dev, out_elem, wave)?;

        {
            let q_slice = q.as_cuda_slice::<Q>()?.slice(q_l.start_offset()..);
            let (q_ptr, _q_g) = q_slice.device_ptr(&stream);

            let headers_ptr = self.headers_ptr as *const u8;

            let (k_s, k_l) = self.k_new.storage_and_layout();
            let k_slice = match &*k_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
                _ => candle::bail!("paged-decode-v2: k_new must be CUDA"),
            }
            .slice(k_l.start_offset()..);
            let (k_ptr, _k_g) = k_slice.device_ptr(&stream);

            let (v_s, v_l) = self.v_new.storage_and_layout();
            let v_slice = match &*v_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
                _ => candle::bail!("paged-decode-v2: v_new must be CUDA"),
            }
            .slice(v_l.start_offset()..);
            let (v_ptr, _v_g) = v_slice.device_ptr(&stream);

            let (dst_ptr, _dst_g) = dst.device_ptr(&stream);

            // Pass the device's dedicated stream so that both the decode
            // kernel and the follow-up commit_decode_write_len_kernel run on
            // the same stream as all GpuChunksGuard::memcpy_htod calls.
            // Passing null here would use the default CUDA stream, which
            // can race with the non-blocking dedicated stream.
            candle::set_kernel_breadcrumb("run_paged_decode", file!(), line!());
            let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
            let status = QsaSelection::with_kernel_args(
                self.qsa.as_ref(),
                &stream,
                |sel_e, sel_c, sel_pages, sel_win, sel_stride, sel_ratio| unsafe {
                    ffi_fn(
                        q_ptr as *const core::ffi::c_void,
                        headers_ptr,
                        dst_ptr as *mut core::ffi::c_void,
                        self.num_active_slots as i32,
                        self.n_q_head as i32,
                        self.n_kv_head as i32,
                        self.head_dim as i32,
                        self.softmax_scale,
                        k_ptr as *const core::ffi::c_void,
                        v_ptr as *const core::ffi::c_void,
                        self.rungs,
                        self.rope_interleaved as i32,
                        raw_stream,
                        sel_e,
                        sel_c,
                        sel_pages,
                        sel_win,
                        sel_stride,
                        sel_ratio,
                    )
                },
            )?;
            match status {
                0 => {}
                1 => candle::bail!(
                    "paged-decode: the split-KV partial pool could not be allocated \
                     (VRAM exhausted) — the requested split/stripe launch needs it, \
                     so nothing was launched and this wave must fail rather than \
                     read an unwritten context"
                ),
                code => candle::bail!(
                    "paged-decode: routing fault {code} — head_dim {} with {} query heads \
                     over {} KV heads reached a kernel route that is not compiled in, so \
                     nothing was launched",
                    self.head_dim,
                    self.n_q_head,
                    self.n_kv_head
                ),
            }
        } // all guards dropped here, dst no longer borrowed

        let out_shape = Shape::from_dims(&[self.num_active_slots, self.n_q_head, self.head_dim]);
        Ok(dst.into_tensor(dev, out_shape))
    }

    /// B2 q8a1024-emitting variant: allocates the q8a1024 byte buffer and passes it as the
    /// kernel's `q8_out`, returning a flat `[q8_bytes]` U8 storage (no FP context store).
    /// The optional output gate rides along as a nullable pointer, `Q`-typed like the queries.
    pub(super) fn cuda_fwd_typed_q8<
        'w,
        Q: candle::cuda_backend::CudaDType + DeviceRepr + 'static,
        KV: candle::cuda_backend::CudaDType + DeviceRepr,
    >(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        ffi_fn: unsafe extern "C" fn(
            *const core::ffi::c_void,
            *const u8,
            *mut core::ffi::c_void,
            *const core::ffi::c_void,
            i64,
            i32,
            i32,
            i32,
            i32,
            f32,
            *const core::ffi::c_void,
            *const core::ffi::c_void,
            RopeRungsFfi,
            i32,
            *mut core::ffi::c_void,
            *const u32,
            *const u32,
            *const u32,
            *const u32,
            i32,
            i32,
        ) -> i32,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        let dev = q.device().clone();
        let stream = dev.cuda_stream();

        let q8_bytes = self.q8_byte_size();
        let dst = KernelOutput::<u8>::new(&dev, q8_bytes, wave)?;

        {
            let q_slice = q.as_cuda_slice::<Q>()?.slice(q_l.start_offset()..);
            let (q_ptr, _q_g) = q_slice.device_ptr(&stream);

            let headers_ptr = self.headers_ptr as *const u8;

            let (k_s, k_l) = self.k_new.storage_and_layout();
            let k_slice = match &*k_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
                _ => candle::bail!("paged-decode-v2: k_new must be CUDA"),
            }
            .slice(k_l.start_offset()..);
            let (k_ptr, _k_g) = k_slice.device_ptr(&stream);

            let (v_s, v_l) = self.v_new.storage_and_layout();
            let v_slice = match &*v_s {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<KV>()?,
                _ => candle::bail!("paged-decode-v2: v_new must be CUDA"),
            }
            .slice(v_l.start_offset()..);
            let (v_ptr, _v_g) = v_slice.device_ptr(&stream);

            // Nullable gate: resolve the device pointer while holding the storage
            // guard across the launch, exactly like the other operands.
            let gate_bind = self.gate.as_ref().map(|g| g.storage_and_layout());
            let gate_sl = match gate_bind.as_ref() {
                Some((g_s, g_l)) => Some(
                    match &**g_s {
                        candle::Storage::Cuda(c) => c.as_cuda_slice::<Q>()?,
                        _ => candle::bail!("paged-decode q8: gate must be CUDA"),
                    }
                    .slice(g_l.start_offset()..),
                ),
                None => None,
            };
            let (gate_ptr, _gate_g) = match gate_sl.as_ref() {
                Some(s) => {
                    let (p, g) = s.device_ptr(&stream);
                    (p as *const core::ffi::c_void, Some(g))
                }
                None => (core::ptr::null(), None),
            };

            let (dst_ptr, _dst_g) = dst.device_ptr(&stream);

            candle::set_kernel_breadcrumb("run_paged_decode_q8", file!(), line!());
            let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
            let status = QsaSelection::with_kernel_args(
                self.qsa.as_ref(),
                &stream,
                |sel_e, sel_c, sel_pages, sel_win, sel_stride, sel_ratio| unsafe {
                    ffi_fn(
                        q_ptr as *const core::ffi::c_void,
                        headers_ptr,
                        dst_ptr as *mut core::ffi::c_void,
                        gate_ptr,
                        self.gate_slot_stride,
                        self.num_active_slots as i32,
                        self.n_q_head as i32,
                        self.n_kv_head as i32,
                        self.head_dim as i32,
                        self.softmax_scale,
                        k_ptr as *const core::ffi::c_void,
                        v_ptr as *const core::ffi::c_void,
                        self.rungs,
                        self.rope_interleaved as i32,
                        raw_stream,
                        sel_e,
                        sel_c,
                        sel_pages,
                        sel_win,
                        sel_stride,
                        sel_ratio,
                    )
                },
            )?;
            match status {
                0 => {}
                1 => candle::bail!(
                    "paged-decode q8: the split-KV partial pool could not be allocated \
                     (VRAM exhausted) — every q8 emit routes through partials + combine, \
                     so nothing was launched and this wave must fail rather than consume \
                     an uninitialized context"
                ),
                code => candle::bail!(
                    "paged-decode q8: routing fault {code} — head_dim {} with {} query heads \
                     over {} KV heads reached a kernel route that is not compiled in, so \
                     nothing was launched",
                    self.head_dim,
                    self.n_q_head,
                    self.n_kv_head
                ),
            }
        } // all guards dropped here, dst no longer borrowed

        Ok(dst.into_tensor(dev, Shape::from_dims(&[q8_bytes])))
    }
}

/// As [`PagedPrefillInt8::run`]: inherent so the wave-scoped output keeps `'w`.
#[cfg(feature = "cuda")]
impl<'k> PagedDecode<'k> {
    pub(super) fn run<'w>(
        &self,
        q: &candle::CudaStorage,
        q_l: &Layout,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<LiveTensor<'w>> {
        match self.head_dim {
            64 | 128 | 256 => {}
            hd => {
                candle::bail!("paged-decode: unsupported head_dim {hd} (must be 64, 128, or 256)")
            }
        }

        // B2: emit the attention context as q8a1024 (head_dim 128 or 256 — whole
        // q8a128 tiles per head) so o_proj runs int8.
        if self.emit_q8 {
            use candle_kernels::paged_decode::{
                run_paged_decode_bf16_q8, run_paged_decode_fp16_q8,
            };
            if !paged_decode_q8_head_dim(self.head_dim) {
                candle::bail!(
                    "paged-decode q8: q8a1024 emit requires head_dim 128 or 256, got {}",
                    self.head_dim
                );
            }
            return match self.arena_dtype {
                DType::F16 => {
                    self.cuda_fwd_typed_q8::<f16, f16>(q, q_l, run_paged_decode_fp16_q8, wave)
                }
                DType::BF16 | DType::F8E4M3 => {
                    self.cuda_fwd_typed_q8::<bf16, bf16>(q, q_l, run_paged_decode_bf16_q8, wave)
                }
                dt => candle::bail!("paged-decode q8: unsupported arena dtype {:?}", dt),
            };
        }

        use candle_kernels::paged_decode::{run_paged_decode_bf16, run_paged_decode_fp16};
        match self.arena_dtype {
            DType::F16 => self.cuda_fwd_typed::<f16, f16, f16>(q, q_l, run_paged_decode_fp16, wave),
            DType::BF16 | DType::F8E4M3 => {
                self.cuda_fwd_typed::<bf16, bf16, bf16>(q, q_l, run_paged_decode_bf16, wave)
            }
            dt => candle::bail!(
                "paged-decode: unsupported arena dtype {:?} (only F16/BF16 supported)",
                dt
            ),
        }
    }
}
