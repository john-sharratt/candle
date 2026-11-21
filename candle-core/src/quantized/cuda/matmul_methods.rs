//! Chooses the kernel for each quantized matrix product, and refuses the weight
//! formats that cannot take it correctly.
//!
//! The vector kernel serves batches up to eight rows and MMQ serves larger ones,
//! with the boundary measured against the two tile shapes. Matmul kernels read
//! past a weight's last row into a padding tail, so a storage must reserve that
//! tail, and a lease is accepted only when its reserved extent covers it. Q8_K has
//! no MMQ form, and Q8_1 weights give wrong results there, so both stay on the
//! vector kernel. AWQ is not a weight format in this engine, and a matmul over it
//! is refused rather than left to return NaN.

use super::*;

impl QCudaStorage {
    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        // Both matmul kernels below gate on
        // `data.len / type_size * block_size >= ncols * nrows` and then address
        // `pad(ncols, MATRIX_ROW_PADDING)` columns, so a storage sized to its
        // payload alone passes the guard and reads up to 512 elements past its
        // end. An owned allocation always reserves that tail
        // ([`padded_storage_bytes`]); a lease reserves it only if whoever it
        // borrows from left room, and a lease that did not would read into the
        // next slot's bytes or past the region at the last one.
        //
        // **The invariant is the reserved extent, not the backing.** This used
        // to refuse every lease, `Backing::Lease` standing in for "has no
        // padding" — true of the expert slots, which are cut to the payload, and
        // false of a layer-streaming slot, which reserves the tail precisely so
        // the GGML path can run over it at `Int8Mode::Off` (where there is no KO
        // twin to take the int8 path instead). Checking the extent admits the
        // second without admitting anything the first would have let through.
        let want = padded_storage_bytes(self_shape.elem_count(), self.dtype);
        if self.data.inner.len() < want {
            crate::bail!(
                "QMatMul over {} B of storage for a {} B payload: the matmul kernels address \
                 {MATRIX_ROW_PADDING} elements past the end of every row, so this reads \
                 outside the allocation. An owned QTensor reserves the tail; a leased view \
                 must be given a slot that leaves room for it.",
                self.data.inner.len(),
                want,
            )
        }
        // **AWQ is not a matmul weight format here.** It survives as a KV arena
        // format (`ArenaFormatTag::QAWQ`), where its own dequantize/quantize
        // kernels serve the cache, and nothing in this engine loads an AWQ
        // weight — the GGUF loader does not produce one. Its matmul path was
        // never correct: the CUDA `block_q_awq_g64` declares 64 elements with a
        // single scale/zero where the Rust `BlockQAWQ_G64` has 128 with
        // `scales[2]`/`zeros[2]`, so the two describe different formats and the
        // kernel returned NaN at every batch size and activation dtype. Refused
        // here rather than left to answer with NaN.
        if matches!(self.dtype, GgmlDType::QAWQ | GgmlDType::QAWQ_G64) {
            crate::bail!(
                "QMatMul over a {:?} weight: AWQ is a KV arena format in this engine, \
                 not a matmul weight format, and its matmul kernels were removed \
                 because their block layout never matched the Rust one. Dequantize the \
                 weight, or load it in a format the matmul kernels are built for.",
                self.dtype
            )
        }
        // Vec kernel through 8 rows, MMQ beyond. Both sides of the boundary are
        // measured, not assumed — the curve lives in candle-transformers
        // `quantized_matmul::tests::qmatmul_row_count_curve`, re-run it before
        // moving this number. On a Q6_K [12288, 4096] (the 9B's gate/up): vec
        // 47–113 µs over 1–8 rows; MMQ's batch-narrow (mmq_x = 16) tile takes
        // 9–32 rows at 107–200 µs, and its wide tile the rest. The narrow tile
        // is what makes 8 the right boundary — against the wide-tile-only MMQ
        // of old, 9 rows cost 2.4× what 8 did and this threshold had to sit at
        // 16 with the vec path double-launching to cover the gap.
        //
        // The count that lands in 9..=16 in production is a speculative VERIFY
        // wave: `sessions × (draft + 1)` rows — 12 at four sessions, budget 2 —
        // where the decode wave beside it carries `sessions` rows and stays vec.
        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            8
        };
        // Formats MMQ cannot serve, whatever the batch — the vec kernel is the
        // only correct path for them.
        //
        // * **Q8_K**: `QI8_K > WARP_SIZE`, so the tile never had an MMQ form.
        // * **Q8_1**: it has one, and it is wrong. Measured against a
        //   dequantized reference at K=2048, Q8_1 weights agree to 0.0074 for
        //   every batch the vec kernel takes (1..=8) and then return garbage the
        //   moment MMQ picks it up — max_diff 1.7e5 at batch 16 and 2.6e5 at 64
        //   and 128, with HALF the output elements zero, identically for BF16,
        //   F16, F32 and F8E4M3 activations. Q8_1 is the ACTIVATION format these
        //   kernels quantize into, and its block carries two scales where the
        //   MMQ tile loader expects one; nothing in a GGUF ships Q8_1 weights,
        //   which is why a broken tile went unnoticed. Routed here rather than
        //   left to compute silently wrong answers for anyone who quantizes a
        //   weight to it. `quantized::q8_1::cuda_tests` is the gate.
        let force_vec = matches!(self.dtype, GgmlDType::Q8_K | GgmlDType::Q8_1);
        let use_vec_kernel = force_vec
            || match layout.shape().dims() {
                [b, m, _k] => b * m <= max_bm,
                [b, _k] => *b <= max_bm,
                _ => false,
            };

        let output_type = storage.dtype();

        let (mut storage, shape) = if use_vec_kernel {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }?;

        if storage.dtype() != output_type {
            storage = storage.to_dtype(layout, output_type)?;
        }
        Ok((storage, shape))
    }

    pub fn fwd_via_gemx(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        self.matmul_gemx(self_shape, storage, layout)
    }
}

impl QCudaStorage {
    /// Use the new quantized matmul kernel that takes F16/BF16 activations directly.
    /// Outputs BF16. For unsupported input dtypes, converts to BF16 before and back after.
    /// K/128 blocks have embedded scales, so no external scales needed.
    pub(super) fn matmul_gemx(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let qtype = dtype_to_qtype(self.dtype)? as i32;
        let stream = self.device.cuda_stream();
        let (weight_ptr, _weight_guard) = self.data.inner.device_ptr(&stream);
        dense_qmatmul_float(
            weight_ptr,
            qtype,
            self.data.len,
            nrows,
            ncols,
            rhs,
            rhs_l,
            &self.device,
        )
    }

    pub(super) fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let inherit = rhs.backing;
        let rhs = rhs.as_cuda_slice::<f32>()?;
        let rhs = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt())?,
        };
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            dequantize_mul_mat_vec(&self.data, &rhs, self.dtype, ncols, nrows, self.device())?
        } else {
            mul_mat_vec_via_q8_1(
                &self.data,
                &rhs,
                self.dtype,
                ncols,
                nrows,
                b_size,
                self.device(),
                inherit,
            )?
        };
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }

    pub(super) fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            let data_f32 = self.dequantize(n * k)?;
            let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
            storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?
        } else {
            let inherit = storage.backing;
            let storage = storage.as_cuda_slice::<f32>()?;
            let storage = match layout.contiguous_offsets() {
                Some((o1, o2)) => storage.slice(o1..o2),
                None => Err(crate::Error::RequiresContiguous {
                    op: "quantized-matmul",
                }
                .bt())?,
            };
            mul_mat_via_q8_1(
                &self.data,
                &storage,
                self.dtype,
                /* x_rows */ n,
                /* x_cols */ k,
                /* y_rows */ k,
                /* y_cols */ b * m,
                self.device(),
                inherit,
            )?
        };
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }

    /// Repack quantized weights to GEMX format (K/128 with embedded scales).
    ///
    /// This removes scale data from the weights (scales should be extracted
    /// separately via extract_scales before calling this) and reorders the
    /// quant bytes for optimal tensor core access patterns.
    ///
    /// # Returns
    /// A new QCudaStorage with the repacked data (smaller since scales removed)
    pub fn repack_gemx(&self, shape: &Shape) -> Result<Self> {
        let (nrows, ncols) = shape.dims2()?;
        let qtype = dtype_to_qtype(self.dtype)? as i32;

        // Check if this qtype supports GEMX repacking
        let supported = unsafe { is_gemx_supported(qtype) };
        if supported == 0 {
            crate::bail!("GEMX repacking not supported for dtype {:?}", self.dtype);
        }

        // Get expected output size
        let new_size = unsafe { get_repacked_size_bytes(nrows as i32, ncols as i32, qtype) };
        if new_size < 0 {
            crate::bail!("Failed to get repacked size for dtype {:?}", self.dtype);
        }
        let new_size = new_size as usize;

        // Allocate new buffer for repacked data
        let dst_data = self.device.alloc_zeros::<u8>(new_size)?;

        // Launch repacking kernel (src -> dst)
        // Scope the guards so they're dropped before we move dst_data
        {
            let stream = self.device.cuda_stream();
            let (src_ptr, _src_guard) = self.data.inner.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst_data.device_ptr(&stream);
            // Ordered on `stream`: the repacked buffer is only read behind it,
            // and the source's release queues behind it there too.
            let result = unsafe {
                run_repack_gemx(
                    src_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    nrows as i32,
                    ncols as i32,
                    qtype,
                    stream.cu_stream() as *mut std::ffi::c_void,
                )
            };

            if result < 0 {
                crate::bail!("repack_gemx kernel failed");
            }
        }

        Ok(Self {
            data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
                inner: dst_data,
                len: new_size,
            }),
            dtype: self.dtype,
            device: self.device.clone(),
            backing: Backing::Owned,
        })
    }

    /// Re-quantize this (compact) weight `[nrows × ncols]` into the lane-major KO format
    /// `ko_dtype` for the q8a128 int8 matmul: dequantize to f32, then per-128 affine KO
    /// quantize. Both stages run on the GPU and the dequantized f32 is fed straight into
    /// `run_quantize_ko` without ever leaving VRAM. The int8-mode counterpart of
    /// [`Self::repack_gemx`] used by `QMatMul::repack_for_optimization`.
    pub fn repack_ko(&self, shape: &Shape, ko_dtype: GgmlDType) -> Result<Self> {
        self.repack_ko_into(shape, ko_dtype, None)
    }

    /// [`Self::repack_ko`], writing the twin into memory the caller already owns.
    ///
    /// `dst` is `(address, origin)` for a destination of exactly
    /// [`ko_repacked_bytes`], or `None` to allocate from the CUDA pool as
    /// before. It is what puts the model's weights inside the device
    /// reservation: the twin is the buffer that *stays* — the compact source is
    /// dropped the moment this returns — so placing the twin is what moves the
    /// resident set, and placing the source instead would fill a block with no
    /// free path with buffers that die seconds later.
    ///
    /// Genuinely optional rather than a second code path: a CPU test or a
    /// process with no reservation has no address to give, and the pool is the
    /// right answer there.
    ///
    /// # Safety
    ///
    /// When `Some`, the address must name at least [`ko_repacked_bytes`] bytes
    /// of device memory that outlives the returned storage and is not aliased.
    /// The storage is stamped `Lease`, so dropping it frees nothing.
    pub fn repack_ko_into(
        &self,
        shape: &Shape,
        ko_dtype: GgmlDType,
        dst: Option<(u64, LeaseOrigin)>,
    ) -> Result<Self> {
        let (nrows, ncols) = shape.dims2()?;
        // The KO chunk layout packs 8 rows × 128 K per chunk, but the q8a128 matmul kernel that
        // reads the result tiles N in blocks of 32 — so require `nrows % 32` (not just 8), matching
        // the matmul (`ko_tileable`). Callers test that predicate themselves and route a sub-tile
        // weight to the dense path; reaching this bail means a caller did not, so it is a real
        // error and must not be read as "not KO-tileable".
        if !ko_tileable(nrows, ncols) {
            crate::bail!(
                "repack_ko: shape [{nrows}, {ncols}] must have nrows % 32 == 0 and ncols % 128 == 0"
            );
        }
        // MXFP4 → MXFP4_KO is an EXACT byte permutation (nibbles + per-sub E8M0 copied
        // verbatim, per-row dm baked) — never a dequant/requant, which would re-derive
        // E8M0 scales lossily and needs a quantize kernel that deliberately does not
        // exist (`run_quantize_ko` has no MXFP4 arm). Load-time only: pull the native
        // bytes to the host, reorder with the same routine the engine's prepare path
        // uses, upload the chunk tensor.
        if ko_dtype == GgmlDType::MXFP4_KO {
            if self.dtype != GgmlDType::MXFP4 {
                crate::bail!(
                    "repack_ko(MXFP4_KO): source must be MXFP4, got {:?}",
                    self.dtype
                );
            }
            let need = nrows * (ncols / 32) * GgmlDType::MXFP4.type_size();
            if self.data.len < need {
                crate::bail!(
                    "repack_ko(MXFP4_KO): storage holds {} bytes, need {need}",
                    self.data.len
                );
            }
            let native = self
                .device
                .memcpy_dtov(&self.data.inner.slice(..need))
                .map_err(crate::Error::wrap)?;
            let ko =
                crate::quantized::ko_quant::mxfp4_native_to_ko_gpu_chunk(&native, nrows, ncols);
            // The host reorder decides its own length; the caller sized `dst`
            // from `ko_repacked_bytes`. They agree by construction — both are
            // chunks × `MXFP4_KO_GPU_CHUNK_BYTES` — and this is what keeps them
            // agreeing, because a divergence writes past the destination and
            // corrupts the weight placed after it.
            let want = ko_repacked_bytes(shape, ko_dtype)?;
            if ko.len() != want {
                crate::bail!(
                    "repack_ko(MXFP4_KO): the reorder produced {} B but the destination \
                     was sized for {want} B",
                    ko.len(),
                );
            }
            let (mut out, backing) = self.dest_slice(ko.len(), dst)?;
            self.device
                .memcpy_htod(&ko, &mut out.slice_mut(..ko.len()))?;
            return Ok(Self {
                data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
                    inner: out,
                    len: ko.len(),
                }),
                dtype: ko_dtype,
                device: self.device.clone(),
                backing,
            });
        }
        let qtype = dtype_to_qtype(ko_dtype)? as i32;
        // **The same function the caller sized `dst` with.** These two numbers
        // must agree exactly: the caller carves a destination of
        // `ko_repacked_bytes` and this writes `bytes` into it, through a raw
        // pointer with no length check. Computing the product inline here — as
        // this did — put two copies of that arithmetic in different files, where
        // one could be corrected and the other left, and the failure would be a
        // silent write past the end of one weight into the next.
        let bytes = ko_repacked_bytes(shape, ko_dtype)?;
        // **The f32 intermediate is a bounded device band, not a whole-tensor buffer.**
        //
        // The repack is dequantize-then-requantize, and composing the two kernels through a
        // buffer means that buffer is the whole tensor in f32 — for the 27B's `[248320, 5120]`
        // head, 4,850 MiB to produce a 993 MiB twin. Held on the card it was the single
        // largest claim in the process, and worse than its size suggests: the span is created
        // once, at a size reduced by exactly this figure, and a `cuMemAddressReserve` cannot be
        // grown afterwards. So a buffer that lived for *one tensor during load* cost a third of
        // the card **for the life of the process** — the weight zone and the KV side are carved
        // from the span and could never reach that ground again.
        //
        // **The buffer is eliminated rather than relocated: the repack runs a row band at a
        // time**, so the intermediate is [`REPACK_BAND_BYTES`] regardless of tensor size —
        // 4,850 MiB becomes ~48. A band's source blocks and its f32 are both contiguous (every
        // row is a whole number of blocks), so the band needs no gather; and its KO output is
        // `k_blocks` contiguous runs, because a chunk's index is `k_blk · row_groups + g` with
        // `g` innermost. Those runs are copied into place device-to-device, a few dozen per
        // band, which is why this needs no change to the quantize kernel at all.
        //
        // Moving the intermediate to host memory instead — `cuMemHostAlloc(DEVICEMAP)`, so the
        // kernels read and write it across PCIe — was tried first and does not work at the
        // sizes that matter. A kernel dereferencing a host-mapped buffer faults once the
        // allocation is large: measured here, 64 MiB completes and 256 MiB fails with
        // `CUDA_ERROR_UNKNOWN` at the next synchronise, and banding the *launches* does not
        // help because the fault tracks the allocation's size, not the launch's. This is the
        // hazard `alloc_host_mapped` already warns about for `cuMemHostRegister`. Banding is
        // also simply better: the band stays in VRAM, so the repack costs no PCIe at all.
        //
        // It could go further still — one KO chunk needs 8 rows × 128 K of f32, 4 KiB and
        // register-resident, so a kernel fusing the dequant into `quantize_ko` would need no
        // intermediate whatever and would read only the quantized source. That wants a fused
        // kernel per source dtype. The bound above is what made the buffer stop being
        // structural, which is the part that was costing a third of the card.
        let bs = self.dtype.block_size();
        let ts = self.dtype.type_size();
        let row_groups = nrows / 8;
        let k_blocks = ncols / 128;
        let chunk_bytes = crate::quantized::ko_quant::ko_chunk_bytes(ko_dtype);
        let band_rows = repack_band_rows(ncols, row_groups);
        if !ncols.is_multiple_of(bs) {
            crate::bail!(
                "repack_ko: {ncols} columns is not a whole number of {:?} blocks ({bs})",
                self.dtype
            );
        }

        // The bands die with this call, so a load serves them from its scratch at the span's
        // top (`load_scratch`); the destination, which stays, is allocated outside the scope.
        let (mut f32_band, mut ko_band) = load_scratch::transient(|| -> Result<_> {
            Ok(unsafe {
                (
                    self.device.alloc::<f32>(band_rows * ncols)?,
                    self.device
                        .alloc::<u8>(band_rows / 8 * k_blocks * chunk_bytes)?,
                )
            })
        })?;
        let (out, backing) = self.dest_slice(bytes, dst)?;
        // **The destination is wrapped before the first fallible step, not after.** `out` may
        // be a `Lease` into the dense block, and a bare `CudaSlice` dropped on an error path
        // hands that address to `cuMemFreeAsync` — the reservation's own memory, freed by a
        // caller that does not own it. `QCudaStorage::drop` is what knows the difference (it
        // `leak`s a lease and frees only an `Owned`), so the fix is to be a `QCudaStorage`
        // before anything can fail rather than to remember a `mem::forget` at each `?`.
        //
        // This was safe by accident until the repack was banded: the old path had no fallible
        // step between `dest_slice` and the constructor. The band loop below has two.
        let mut me = Self {
            data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
                inner: out,
                len: bytes,
            }),
            dtype: ko_dtype,
            device: self.device.clone(),
            backing,
        };
        let stream = self.device.cuda_stream();

        let mut r0 = 0usize;
        while r0 < nrows {
            let rows = band_rows.min(nrows - r0);
            // Source bytes for rows [r0, r0+rows): contiguous, since every row is a whole
            // number of blocks.
            {
                let (src_ptr, _sg) = self.data.inner.device_ptr(&stream);
                let (fp, _fg) = f32_band.device_ptr_mut(&stream);
                dequantize_f32_into(
                    src_ptr + (r0 * ncols / bs * ts) as u64,
                    self.dtype,
                    rows * ncols,
                    fp,
                    &stream,
                )?;
            }
            {
                let (fp, _fg) = f32_band.device_ptr(&stream);
                let (kp, _kg) = ko_band.device_ptr_mut(&stream);
                unsafe {
                    run_quantize_ko(
                        fp as *const f32,
                        kp as *mut std::ffi::c_void,
                        rows as i32,
                        ncols as i32,
                        qtype,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }
            // Scatter the band's chunks to their global positions: one contiguous run per
            // k-block, because `g` is the inner index of `k_blk · row_groups + g`.
            let band_groups = rows / 8;
            let g0 = r0 / 8;
            let run = band_groups * chunk_bytes;
            for k in 0..k_blocks {
                let dst_off = (k * row_groups + g0) * chunk_bytes;
                let src_off = k * band_groups * chunk_bytes;
                debug_assert!(
                    dst_off + run <= bytes,
                    "band scatter passes the destination"
                );
                let src_view = ko_band.slice(src_off..src_off + run);
                let mut dst_view = me.data.inner.slice_mut(dst_off..dst_off + run);
                stream
                    .memcpy_dtod(&src_view, &mut dst_view)
                    .map_err(crate::Error::wrap)?;
            }
            r0 += rows;
        }
        // No synchronise here. Every step above is stream-ordered, and the two band buffers
        // free on the same stream when they drop, so the copies cannot outlive their source.
        // The earlier host-mapped draft did need one — the host allocation freed on the CPU's
        // timeline — and leaving it behind would have cost a full pipeline stall per repacked
        // tensor, which on a pack build is once per projection per layer.
        Ok(me)
    }

    /// A `bytes`-long destination: the caller's, or a fresh pool allocation.
    ///
    /// The one place the two answers are produced, so a repack path cannot end
    /// up wrapping a leased address as `Owned` — which would hand the
    /// reservation's memory to `cuMemFreeAsync` on drop.
    pub(super) fn dest_slice(
        &self,
        bytes: usize,
        dst: Option<(u64, LeaseOrigin)>,
    ) -> Result<(CudaSlice<u8>, Backing)> {
        match dst {
            // SAFETY: the caller's contract on `repack_ko_into` — at least
            // `bytes` of live, un-aliased device memory outliving the storage.
            Some((ptr, origin)) => Ok((
                unsafe {
                    self.device
                        .cuda_stream()
                        .upgrade_device_ptr::<u8>(ptr, bytes)
                },
                Backing::Lease(origin),
            )),
            // SAFETY: filled by the copy or the kernel the caller runs next.
            None => Ok((unsafe { self.device.alloc::<u8>(bytes)? }, Backing::Owned)),
        }
    }

    /// Get the size in bytes after GEMX repacking, without actually repacking.
    pub fn repacked_size(&self, shape: &Shape) -> Result<usize> {
        let (nrows, ncols) = shape.dims2()?;
        let qtype = dtype_to_qtype(self.dtype)? as i32;

        let size = unsafe { get_repacked_size_bytes(nrows as i32, ncols as i32, qtype) };
        if size < 0 {
            crate::bail!("GEMX repacking not supported for dtype {:?}", self.dtype);
        }
        Ok(size as usize)
    }

    /// Check if this storage's dtype supports GEMX repacking.
    pub fn supports_gemx_repacking(&self) -> bool {
        gemx_repacking_supported(self.dtype)
    }
}
