//! The forward passes of a quantized matmul, one for each shape the model's
//! layers present, each returning a result stored at the width its consumer asks
//! for.
//!
//! The fused forms take a pre-quantized activation and several weights in one
//! launch, and every column they produce is the one the single-weight form
//! computes, bit for bit. The int8 form stores its F32 accumulator at the
//! requested width instead of casting it afterwards, which saves a second buffer.
//! The GEMX form is an entry point for the benchmark, not a path the model code
//! calls. Non-CUDA builds return an error, since these kernels exist only on CUDA.

use super::*;

impl QMatMul {
    /// Matmul against a GEMX-repacked weight (see [`QTensor::repack_gemx`]),
    /// whose K/128 blocks carry embedded scales so no external scale tensor is
    /// needed.
    ///
    /// This is the measurement entry point to `CudaStorage::fwd_via_gemx`, not
    /// an inference path: the engine reaches the FP kernel through
    /// [`Self::forward_dynamic`], and nothing in the model code calls this.
    /// It exists so `candle-core/examples/quantized_matmul_benchmark.rs` can
    /// time the kernel and check its numerics per quant format from outside
    /// the crate, where the storage-level op is not reachable.
    #[allow(unused_variables)]
    pub fn forward_via_gemx<'w>(&self, xs: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
        match self {
            Self::QTensor(t) => match &t.storage {
                #[cfg(feature = "cuda")]
                QStorage::Cuda(cuda_storage) => {
                    let storage = xs.storage_and_layout().0;
                    let layout = xs.layout();
                    let cuda_xs = match &*storage {
                        Storage::Cuda(s) => s,
                        _ => crate::bail!("expected CUDA storage for quantized matmul"),
                    };
                    let (out_storage, out_shape) =
                        cuda_storage.fwd_via_gemx(&t.shape, cuda_xs, layout)?;
                    Ok(crate::tensor::from_storage(
                        Storage::Cuda(out_storage),
                        out_shape,
                        crate::op::BackpropOp::none(),
                        false,
                    ))
                }
                #[cfg(not(feature = "cuda"))]
                QStorage::Cuda(_) => crate::bail!("CUDA support not compiled"),
                _ => xs.apply_op1_no_bwd(t.as_ref()),
            },
            // Dequantized weights: plain matmul. `matmul` records a graph edge
            // and so returns its operand's lifetime; the copy is what makes the
            // result owned.
            Self::Tensor(w) => {
                let xs = &xs.to_owned_tensor()?;
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
            Self::TensorF16(w) => {
                let xs = &xs.to_owned_tensor()?;
                let in_dtype = xs.dtype();
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
            }
        }
    }

    /// Matmul over an already-built [`cuda::DynamicTensor`] activation — the keystone the fused
    /// producers feed: an `Int8` operand carries **pre-quantized** q8a128 (emitted directly by a
    /// fused RMSNorm/SwiGLU/attention epilogue, so no standalone quantize launch), a `Float`
    /// operand is a plain FP tensor. Dispatches to [`cuda::dense_qmatmul`], which forks the int8
    /// KO tensor-core path vs the FP path and enforces the KO⇔int8 pairing against this weight.
    /// Returns a result of shape `[lead.., N]` stored at `out_dtype`: the int8 kernel converts its
    /// F32 accumulator on the store, so there is no cast — and no second buffer — afterwards.
    #[cfg(feature = "cuda")]
    pub fn forward_dynamic<'w>(
        &self,
        input: cuda::DynamicTensor<'_, 'w>,
        out_dtype: crate::DType,
    ) -> Result<LiveTensor<'w>> {
        let t = match self {
            Self::QTensor(t) => t,
            _ => crate::bail!("forward_dynamic requires a QTensor weight"),
        };
        let cs = match &t.storage {
            QStorage::Cuda(cs) => cs,
            _ => crate::bail!("forward_dynamic requires CUDA storage"),
        };
        let device = cs.device().clone();
        // Weight is row-major [N, K]; nrows = N is the matmul's output width.
        let nrows = t.shape().dims()[0];
        let wptr = cs.data_ptr();
        let wlen = cs.storage_size_in_bytes();
        let wdtype = t.dtype();
        cuda::dense_qmatmul(input, wptr, wdtype, nrows, wlen, out_dtype, &device)
    }

    /// [`Self::forward_dynamic`] with the int8 path's launch forced to `plan` rather than chosen
    /// by `q8a128_dense_plan` — what the projection bench sweeps to tune that rule. Every plan
    /// produces the same bits. Production calls [`Self::forward_dynamic`].
    #[cfg(feature = "cuda")]
    pub fn forward_dynamic_plan<'w>(
        &self,
        input: cuda::DynamicTensor<'_, 'w>,
        out_dtype: crate::DType,
        plan: DensePlan,
    ) -> Result<LiveTensor<'w>> {
        let t = match self {
            Self::QTensor(t) => t,
            _ => crate::bail!("forward_dynamic_plan requires a QTensor weight"),
        };
        let cs = match &t.storage {
            QStorage::Cuda(cs) => cs,
            _ => crate::bail!("forward_dynamic_plan requires CUDA storage"),
        };
        let device = cs.device().clone();
        cuda::dense_qmatmul_with_plan(
            input,
            cs.data_ptr(),
            t.dtype(),
            t.shape().dims()[0],
            cs.storage_size_in_bytes(),
            out_dtype,
            Some(plan),
            &device,
        )
    }

    /// `silu(proj[:, 0..cols]) · Wᵀ` as F32, the activation quantized to q8a128 by the
    /// kernel's own tile loader under `sum_scale` — one launch where a SiLU producer and this
    /// matmul took two, with the same bits. `proj` is `[M, width ≥ cols]` F32 read through its
    /// row stride. The weight must be the Q8_KO twin on CUDA. See
    /// [`cuda::q8a128_dense_matmul_silu`].
    #[cfg(feature = "cuda")]
    pub fn forward_silu_f32<'w>(
        &self,
        proj: &LiveTensor<'w>,
        cols: usize,
        sum_scale: SumScale,
    ) -> Result<LiveTensor<'w>> {
        let t = match self {
            Self::QTensor(t) => t,
            _ => crate::bail!("forward_silu_f32 requires a QTensor weight"),
        };
        let cs = match &t.storage {
            QStorage::Cuda(cs) => cs,
            _ => crate::bail!("forward_silu_f32 requires CUDA storage"),
        };
        if t.shape().dims()[1] != cols {
            crate::bail!(
                "forward_silu_f32: a {:?} weight against {cols} activation columns",
                t.shape().dims()
            );
        }
        let device = cs.device().clone();
        cuda::q8a128_dense_matmul_silu(
            proj,
            cols,
            cs.data_ptr(),
            t.dtype(),
            t.shape().dims()[0],
            sum_scale,
            &device,
        )
    }

    /// Fused q/k/v projection in ONE launch: the shared q8a128 activation `op` × the separate KO
    /// weights `q/k/v` (any KO formats — no weight concatenation needed), returning the concatenated
    /// `[lead.., Nq+Nk+Nv]` output cast to `out_dtype`. Float-identical to three separate
    /// [`forward_dynamic`] calls but with full GPU occupancy. Each weight must be a KO QTensor on
    /// CUDA (the int8 twin). See [`cuda::qkv_segmented_matmul`].
    #[cfg(feature = "cuda")]
    pub fn qkv_segmented<'w>(
        op: &cuda::Q8a128Operand<'w>,
        weights: &[&QMatMul],
        out_dtype: crate::DType,
    ) -> Result<LiveTensor<'w>> {
        let mut segs = Vec::with_capacity(weights.len());
        let mut device = None;
        for w in weights {
            let t = match w {
                Self::QTensor(t) => t,
                _ => crate::bail!("qkv_segmented requires KO QTensor weights"),
            };
            let cs = match &t.storage {
                QStorage::Cuda(cs) => cs,
                _ => crate::bail!("qkv_segmented requires CUDA storage"),
            };
            device = Some(cs.device().clone());
            segs.push((cs.data_ptr(), t.dtype(), t.shape().dims()[0]));
        }
        let device = device.ok_or_else(|| crate::Error::Msg("qkv_segmented: no weights".into()))?;
        cuda::qkv_segmented_matmul(op, &segs, out_dtype, &device)
    }

    /// Several KO weights that read the same q8a128 operand `op`, in ONE launch: `[lead.., ΣNᵢ]`,
    /// weight `i`'s columns after those before it, stored at `out_dtype`. Every column is the
    /// one that weight's own [`forward_dynamic`](Self::forward_dynamic) computes, bit for bit —
    /// see [`cuda::dense_qmatmul_stacked`] for the launch it picks. Each weight must be a KO
    /// QTensor on CUDA with a multiple of 32 rows.
    #[cfg(feature = "cuda")]
    pub fn forward_stacked<'w>(
        op: &cuda::Q8a128Operand<'w>,
        weights: &[&QMatMul],
        out_dtype: crate::DType,
    ) -> Result<LiveTensor<'w>> {
        let mut segs = Vec::with_capacity(weights.len());
        let mut device = None;
        for w in weights {
            let t = match w {
                Self::QTensor(t) => t,
                _ => crate::bail!("forward_stacked requires KO QTensor weights"),
            };
            let cs = match &t.storage {
                QStorage::Cuda(cs) => cs,
                _ => crate::bail!("forward_stacked requires CUDA storage"),
            };
            if t.shape().dims()[1] != op.cols {
                crate::bail!(
                    "forward_stacked: a {:?} weight against a {}-wide operand",
                    t.shape().dims(),
                    op.cols
                );
            }
            device = Some(cs.device().clone());
            segs.push((cs.data_ptr(), t.dtype(), t.shape().dims()[0]));
        }
        let device =
            device.ok_or_else(|| crate::Error::Msg("forward_stacked: no weights".into()))?;
        cuda::dense_qmatmul_stacked(op, &segs, out_dtype, &device)
    }

    /// Int8 tensor-core matmul: q8a128 activations × KO weights. The weight must already be the
    /// KO twin QTensor on CUDA (produced by [`QMatMul::repack_for_optimization`] with an int8
    /// `mode`) — the twin choice was baked in at repack time, so here `mode` only selects the
    /// activation form (q8a128 for any non-`Off` mode). Quantizes the activation here (the
    /// **unfused** path, one standalone launch) then runs [`QMatMul::forward_dynamic`]; the fused
    /// producers bypass this by emitting q8a128 themselves and calling `forward_dynamic` directly.
    ///
    /// `out_dtype` is the width the result is **stored** at, not a cast applied after: the MMA
    /// accumulates in F32 registers and converts on the store, so asking for F32 here is the same
    /// number the kernel already had, and asking for a narrow type is bit-identical to the F32
    /// kernel followed by a cast, minus the cast. A consumer that wants F32 — the DeltaNet
    /// projections, whose recurrence carries state in F32 — must say so here rather than widen
    /// afterwards, which would round through the narrow type and back for nothing.
    ///
    /// Non-CUDA build: the q8a128 × KO int8 matmul is a CUDA tensor-core path — always
    /// errors (the `dummy_cuda` convention).
    #[cfg(not(feature = "cuda"))]
    pub fn forward_via_int8<'w>(
        &self,
        _xs: &LiveTensor<'w>,
        _mode: Int8Mode,
        _out_dtype: crate::DType,
        _sum_scale: SumScale,
    ) -> Result<LiveTensor<'w>> {
        crate::bail!("forward_via_int8 requires the cuda feature")
    }

    #[cfg(feature = "cuda")]
    pub fn forward_via_int8<'w>(
        &self,
        xs: &LiveTensor<'w>,
        mode: Int8Mode,
        out_dtype: crate::DType,
        // This quantizes the activation itself, so the Σx convention is its
        // choice to make and not one it can read off an operand. Named rather
        // than defaulted: a wrong value here is a wrong number, not an error.
        sum_scale: SumScale,
    ) -> Result<LiveTensor<'w>> {
        let device = match self {
            Self::QTensor(t) => match &t.storage {
                QStorage::Cuda(cs) => cs.device().clone(),
                _ => crate::bail!("forward_via_int8 requires CUDA storage"),
            },
            _ => crate::bail!("forward_via_int8 requires a KO QTensor weight"),
        };
        let acts = cuda::to_dynamic(xs, mode, &device, sum_scale)?;
        self.forward_dynamic(acts.as_dynamic(), out_dtype)
    }
}

impl crate::CustomOp1 for QTensor {
    fn name(&self) -> &'static str {
        "qmatmul"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        let (n, k) = self.shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().context("empty dst_shape")?;
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self.shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        #[allow(clippy::infallible_destructuring_match)]
        let self_storage = match &self.storage {
            QStorage::Cpu(storage) => storage,
            QStorage::Metal(_) | QStorage::Cuda(_) => crate::bail!("Invalid storage"),
        };
        let slice = storage.as_slice::<f32>()?;
        let slice = &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];
        self_storage.matmul_t((dst_shape.elem_count() / n, k, n), slice, &mut dst_storage)?;
        Ok((crate::CpuStorage::F32(dst_storage), dst_shape))
    }

    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Metal(metal) => metal,
            _ => unreachable!("Cannot call metal matmul on non metal QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }

    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Cuda(cuda) => cuda,
            _ => unreachable!("Cannot call cuda matmul on non cuda QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }
}
