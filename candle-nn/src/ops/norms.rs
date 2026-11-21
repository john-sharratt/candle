//! Provides RMS and layer normalisation, with a root-arena form that places a whole
//! chain of operations in the arena the caller names.
//!
//! The output of a normalisation is allocated in the memory arena of its input by
//! default. When a caller names a root arena, that choice wins, so a chain of
//! operations starting from a residual stream that lives on another phase's span
//! lands in the caller's phase. Downstream operations inherit the arena from their
//! input, so naming the root once is enough. The slow forms compose the same
//! arithmetic from tensor operations and serve as the reference the fused kernels are
//! checked against.

use super::*;

#[derive(Debug, Clone)]
pub(super) struct RmsNorm {
    pub(super) eps: f32,
    /// The arena to allocate the output from, over the operand's own.
    ///
    /// Every op downstream of a wave-backed value inherits its arena from that
    /// value, so a chain only needs to be told where it lives once — at its
    /// head, whose operand is the residual stream. That stream may be on the
    /// pool or on the forward's span, and either way the chain belongs to the
    /// phase the root names. `None` is the ordinary case and means "inherit or
    /// fall back to the pool", which is what plain [`rms_norm`] does.
    ///
    /// Only the CUDA path reads it — there are no wave arenas to name without a
    /// device — but the field stays so both builds construct the same struct.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    pub(super) root: Option<WaveTicket>,
}

impl candle::CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "rms-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        let eps = self.eps;
        fn inner<
            T: candle::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let sum2 = src
                        .iter()
                        .map(|&v| {
                            let v = v.as_();
                            v * v
                        })
                        .sum::<f32>();
                    let m = (sum2 / dim_m1 as f32 + eps).sqrt();
                    let m = T::from_f32(m).unwrap_or_else(T::nan);
                    for ((d, s), alpha) in dst.iter_mut().zip(src.iter()).zip(alpha) {
                        *d = *s / m * *alpha
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2) {
            (C::BF16(s1), C::BF16(s2)) => inner::<half::bf16>(s1, l1, s2, l2, eps),
            (C::F16(s1), C::F16(s2)) => inner::<half::f16>(s1, l1, s2, l2, eps),
            (C::F32(s1), C::F32(s2)) => inner::<f32>(s1, l1, s2, l2, eps),
            _ => candle::bail!("unsupported dtype for rmsnorm {:?}", s1.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::{kernels, CudaStorageSlice};

        let dev = s1.device();
        let stream = dev.cuda_stream();

        let (src_o1, src_o2) = match l1.contiguous_offsets() {
            None => candle::bail!("input has to be contiguous"),
            Some(offsets) => offsets,
        };
        let (alpha_o1, alpha_o2) = match l2.contiguous_offsets() {
            None => candle::bail!("alpha has to be contiguous"),
            Some(offsets) => offsets,
        };

        let el = l1.shape().elem_count();
        let dims = l1.shape().dims();
        let dim_m1 = dims[dims.len() - 1];
        let n_cols = dim_m1 as i32;
        let n_rows = (el / dim_m1) as i32;

        // Get dtype for FFI dispatcher
        let dtype = match &s1.slice {
            CudaStorageSlice::F32(_) => kernels::simple::reduce::FloatDType::F32 as i32,
            CudaStorageSlice::F64(_) => kernels::simple::reduce::FloatDType::F64 as i32,
            CudaStorageSlice::F16(_) => kernels::simple::reduce::FloatDType::F16 as i32,
            CudaStorageSlice::BF16(_) => kernels::simple::reduce::FloatDType::BF16 as i32,
            CudaStorageSlice::F8E4M3(_) => kernels::simple::reduce::FloatDType::F8E4M3 as i32,
            _ => candle::bail!("rmsnorm not supported for dtype {:?}", s1.dtype()),
        };

        // Where the output goes. A `root` names the phase the caller is in — the
        // head of a layer chain, whose input is the residual stream — and wins:
        // the residual may itself be on the *forward's* span, and a layer's norm
        // that inherited it would put the whole layer chain there instead of on
        // the layer's own phase. Everything downstream follows from this one
        // output without another mention of the wave. Without a root the output
        // is allocated beside its input, which is what makes the `'w` on the
        // result true rather than merely permitted. Declared before the dispatch
        // macro because a `macro_rules!` body resolves free identifiers at its
        // definition site, not its call.
        let inherit = match self.root {
            Some(t) => candle::cuda_backend::Backing::from_ticket(Some(t)),
            None => s1.backing,
        };

        // Assigned by the macro to whatever `alloc_inheriting` resolved.

        let out_backing;

        macro_rules! rmsnorm_impl {
            ($src_slice:expr, $alpha_slice:expr, $dtype_variant:ident, $rust_type:ty) => {{
                let src = $src_slice.slice(src_o1..src_o2);
                let alpha = $alpha_slice.slice(alpha_o1..alpha_o2);
                let (dst, resolved_backing) = unsafe {
                    candle::cuda_backend::alloc_inheriting::<$rust_type>(dev, el, inherit)?
                };
                out_backing = resolved_backing;
                {
                    let (src_ptr, _src_guard) = src.device_ptr(&stream);
                    let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
                    let (alpha_ptr, _alpha_guard) = alpha.device_ptr(&stream);
                    #[cfg(feature = "cuda")]
                    candle::set_kernel_breadcrumb("run_rmsnorm_op", file!(), line!());
                    unsafe {
                        kernels::simple::reduce::run_rmsnorm_op(
                            dtype,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            alpha_ptr as *const std::ffi::c_void,
                            n_rows,
                            n_cols,
                            self.eps,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                }
                CudaStorageSlice::$dtype_variant(dst)
            }};
        }

        let slice = match (&s1.slice, &s2.slice) {
            (CudaStorageSlice::F32(src), CudaStorageSlice::F32(alpha)) => {
                rmsnorm_impl!(src, alpha, F32, f32)
            }
            (CudaStorageSlice::F16(src), CudaStorageSlice::F16(alpha)) => {
                rmsnorm_impl!(src, alpha, F16, half::f16)
            }
            (CudaStorageSlice::BF16(src), CudaStorageSlice::BF16(alpha)) => {
                rmsnorm_impl!(src, alpha, BF16, half::bf16)
            }
            (CudaStorageSlice::F8E4M3(src), CudaStorageSlice::F8E4M3(alpha)) => {
                rmsnorm_impl!(src, alpha, F8E4M3, float8::F8E4M3)
            }
            _ => candle::bail!("rmsnorm: dtype mismatch between input and alpha"),
        };

        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
            backing: out_backing,
            anchor: None,
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &Layout,
        s2: &candle::MetalStorage,
        l2: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = s1.device();
        let command_buffer = device.command_buffer()?;
        let kernels = device.kernels();
        let name = match (s1.dtype(), s2.dtype()) {
            (DType::F32, DType::F32) => "rmsnorm_f32",
            (DType::F16, DType::F16) => "rmsnorm_f16",
            (DType::BF16, DType::BF16) => "rmsnorm_bf16",
            (dt1, dt2) => candle::bail!("rmsnorm is not implemented for {dt1:?} {dt2:?}"),
        };

        if !(l1.is_contiguous() && l2.is_contiguous()) {
            candle::bail!("Non contiguous rmsnorm is not implemented");
        }

        let last_dim = l1.dims()[l1.shape().rank() - 1];
        let elem_count = l1.shape().elem_count();
        let output = device.new_buffer(elem_count, s1.dtype(), "rmsnorm")?;
        candle_metal_kernels::call_rms_norm(
            device.metal_device(),
            &command_buffer,
            kernels,
            name,
            elem_count,
            last_dim,
            self.eps,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, s1.dtype());
        Ok((newstorage, l1.shape().clone()))
    }
}

pub fn rms_norm_slow(x: &Tensor, alpha: &Tensor, eps: f32) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed.to_dtype(x_dtype)?.broadcast_mul(alpha)
}

pub fn rms_norm<'w>(xs: &LiveTensor<'w>, alpha: &Tensor, eps: f32) -> Result<LiveTensor<'w>> {
    rms_norm_rooted(xs, alpha, eps, None)
}

/// [`rms_norm`], allocating its output from `root`.
///
/// **The seed of a wave-scoped chain.** Operand provenance carries an arena from
/// a value to everything computed from it, so a chain of forty ops needs to be
/// told where it lives exactly once — and it cannot inherit that from its own
/// input, which is the residual stream and crosses layers (on the pool, or on
/// the forward's span). This is where a layer says it: normalise the residual
/// into the phase's span, and the rest of the layer lands there by
/// construction.
///
/// `root` wins over the arena `xs` carries; with `None` the output inherits
/// it.
pub fn rms_norm_rooted<'w>(
    xs: &LiveTensor<'w>,
    alpha: &Tensor,
    eps: f32,
    root: Option<WaveTicket>,
) -> Result<LiveTensor<'w>> {
    let hidden_size_xs = xs.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    if hidden_size_xs != hidden_size_alpha {
        candle::bail!(
            "shape mismatch in rms-norm {:?} {:?}",
            xs.shape(),
            alpha.shape()
        )
    }
    xs.apply_op2_no_bwd(alpha, &RmsNorm { eps, root })
}

#[derive(Debug, Clone)]
pub(super) struct LayerNorm {
    pub(super) eps: f32,
}

impl candle::CustomOp3 for LayerNorm {
    fn name(&self) -> &'static str {
        "layer-norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        let eps = self.eps;
        fn inner<
            T: candle::WithDType
                + num_traits::Float
                + num_traits::AsPrimitive<f32>
                + num_traits::FromPrimitive,
        >(
            src: &[T],
            layout: &Layout,
            alpha: &[T],
            alpha_layout: &Layout,
            beta: &[T],
            beta_layout: &Layout,
            eps: f32,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let alpha = match alpha_layout.contiguous_offsets() {
                None => candle::bail!("alpha has to be contiguous"),
                Some((o1, o2)) => &alpha[o1..o2],
            };
            let beta = match beta_layout.contiguous_offsets() {
                None => candle::bail!("beta has to be contiguous"),
                Some((o1, o2)) => &beta[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let mut sum = 0f32;
                    let mut sum2 = 0f32;
                    for v in src {
                        let v = v.as_();
                        sum += v;
                        sum2 += v * v;
                    }
                    let mean = sum / dim_m1 as f32;
                    let var = sum2 / dim_m1 as f32 - mean * mean;
                    let inv_std = (var + eps).sqrt().recip();
                    for ((d, s), (alpha, beta)) in
                        dst.iter_mut().zip(src.iter()).zip(alpha.iter().zip(beta))
                    {
                        let alpha = alpha.as_();
                        let beta = beta.as_();
                        let d_ = (s.as_() - mean) * inv_std * alpha + beta;
                        *d = T::from_f32(d_).unwrap_or_else(T::nan);
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        use CpuStorage as C;
        match (s1, s2, s3) {
            (C::BF16(s1), C::BF16(s2), C::BF16(s3)) => {
                inner::<half::bf16>(s1, l1, s2, l2, s3, l3, eps)
            }
            (C::F16(s1), C::F16(s2), C::F16(s3)) => inner::<half::f16>(s1, l1, s2, l2, s3, l3, eps),
            (C::F32(s1), C::F32(s2), C::F32(s3)) => inner::<f32>(s1, l1, s2, l2, s3, l3, eps),
            _ => candle::bail!("unsupported dtype for rmsnorm {:?}", s1.dtype()),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle::CudaStorage,
        l1: &Layout,
        s2: &candle::CudaStorage,
        l2: &Layout,
        s3: &candle::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::{kernels, CudaStorageSlice};

        let dev = s1.device();
        let stream = dev.cuda_stream();

        let (src_o1, src_o2) = match l1.contiguous_offsets() {
            None => candle::bail!("input has to be contiguous"),
            Some(offsets) => offsets,
        };
        let (alpha_o1, alpha_o2) = match l2.contiguous_offsets() {
            None => candle::bail!("alpha has to be contiguous"),
            Some(offsets) => offsets,
        };
        let (beta_o1, beta_o2) = match l3.contiguous_offsets() {
            None => candle::bail!("beta has to be contiguous"),
            Some(offsets) => offsets,
        };

        let el = l1.shape().elem_count();
        let dims = l1.shape().dims();
        let dim_m1 = dims[dims.len() - 1];
        let n_cols = dim_m1 as i32;
        let n_rows = (el / dim_m1) as i32;

        // Get dtype for FFI dispatcher
        let dtype = match &s1.slice {
            CudaStorageSlice::F32(_) => kernels::simple::reduce::FloatDType::F32 as i32,
            CudaStorageSlice::F64(_) => kernels::simple::reduce::FloatDType::F64 as i32,
            CudaStorageSlice::F16(_) => kernels::simple::reduce::FloatDType::F16 as i32,
            CudaStorageSlice::BF16(_) => kernels::simple::reduce::FloatDType::BF16 as i32,
            CudaStorageSlice::F8E4M3(_) => kernels::simple::reduce::FloatDType::F8E4M3 as i32,
            _ => candle::bail!("layernorm not supported for dtype {:?}", s1.dtype()),
        };

        // The operand's arena: this op's output is allocated beside its input,

        // which is what makes the `'w` on the result true rather than merely

        // permitted. Declared before the dispatch macro because a `macro_rules!`

        // body resolves free identifiers at its definition site, not its call.

        let inherit = s1.backing;

        // Assigned by the macro to whatever `alloc_inheriting` resolved.

        let out_backing;

        macro_rules! layernorm_impl {
            ($src_slice:expr, $alpha_slice:expr, $beta_slice:expr, $dtype_variant:ident, $rust_type:ty) => {{
                let src = $src_slice.slice(src_o1..src_o2);
                let alpha = $alpha_slice.slice(alpha_o1..alpha_o2);
                let beta = $beta_slice.slice(beta_o1..beta_o2);
                let (dst, resolved_backing) = unsafe {
                    candle::cuda_backend::alloc_inheriting::<$rust_type>(dev, el, inherit)?
                };
                out_backing = resolved_backing;
                {
                    let (src_ptr, _src_guard) = src.device_ptr(&stream);
                    let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
                    let (alpha_ptr, _alpha_guard) = alpha.device_ptr(&stream);
                    let (beta_ptr, _beta_guard) = beta.device_ptr(&stream);
                    #[cfg(feature = "cuda")]
                    candle::set_kernel_breadcrumb("run_layernorm_op", file!(), line!());
                    unsafe {
                        kernels::simple::reduce::run_layernorm_op(
                            dtype,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            alpha_ptr as *const std::ffi::c_void,
                            beta_ptr as *const std::ffi::c_void,
                            n_rows,
                            n_cols,
                            self.eps,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                }
                CudaStorageSlice::$dtype_variant(dst)
            }};
        }

        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (
                CudaStorageSlice::F32(src),
                CudaStorageSlice::F32(alpha),
                CudaStorageSlice::F32(beta),
            ) => layernorm_impl!(src, alpha, beta, F32, f32),
            (
                CudaStorageSlice::F16(src),
                CudaStorageSlice::F16(alpha),
                CudaStorageSlice::F16(beta),
            ) => layernorm_impl!(src, alpha, beta, F16, half::f16),
            (
                CudaStorageSlice::BF16(src),
                CudaStorageSlice::BF16(alpha),
                CudaStorageSlice::BF16(beta),
            ) => layernorm_impl!(src, alpha, beta, BF16, half::bf16),
            (
                CudaStorageSlice::F8E4M3(src),
                CudaStorageSlice::F8E4M3(alpha),
                CudaStorageSlice::F8E4M3(beta),
            ) => layernorm_impl!(src, alpha, beta, F8E4M3, float8::F8E4M3),
            _ => candle::bail!("layernorm: dtype mismatch between input, alpha, and beta"),
        };

        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
            backing: out_backing,
            anchor: None,
        };
        Ok((dst, l1.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        s1: &candle::MetalStorage,
        l1: &Layout,
        s2: &candle::MetalStorage,
        l2: &Layout,
        s3: &candle::MetalStorage,
        l3: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = s1.device();
        let command_buffer = device.command_buffer()?;
        let kernels = device.kernels();
        let name = match (s1.dtype(), s2.dtype(), s3.dtype()) {
            (DType::F32, DType::F32, DType::F32) => "layernorm_f32",
            (DType::F16, DType::F16, DType::F16) => "layernorm_f16",
            (DType::BF16, DType::BF16, DType::BF16) => "layernorm_bf16",
            (dt1, dt2, dt3) => {
                candle::bail!("layernorm is not implemented for {dt1:?} {dt2:?} {dt3:?}")
            }
        };

        if !(l1.is_contiguous() && l2.is_contiguous() && l3.is_contiguous()) {
            candle::bail!("Non contiguous layernorm is not implemented");
        }

        let last_dim = l1.dims()[l1.shape().rank() - 1];
        let elem_count = l1.shape().elem_count();
        let output = device.new_buffer(elem_count, s1.dtype(), "layernorm")?;
        candle_metal_kernels::call_layer_norm(
            device.metal_device(),
            &command_buffer,
            kernels,
            name,
            elem_count,
            last_dim,
            self.eps,
            s1.buffer(),
            l1.start_offset() * s1.dtype().size_in_bytes(),
            s2.buffer(),
            l2.start_offset() * s2.dtype().size_in_bytes(),
            s3.buffer(),
            l3.start_offset() * s3.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, s1.dtype());
        Ok((newstorage, l1.shape().clone()))
    }
}

pub fn layer_norm_slow(x: &Tensor, alpha: &Tensor, beta: &Tensor, eps: f32) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let internal_dtype = match x_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        d => d,
    };
    let hidden_size = x.dim(D::Minus1)?;
    let x = x.to_dtype(internal_dtype)?;
    let x = {
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        x.broadcast_sub(&mean_x)?
    };
    let norm_x = (x.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    let x_normed = x.broadcast_div(&(norm_x + eps as f64)?.sqrt()?)?;
    x_normed
        .to_dtype(x_dtype)?
        .broadcast_mul(alpha)?
        .broadcast_add(beta)
}

pub fn layer_norm<'w>(
    xs: &LiveTensor<'w>,
    alpha: &Tensor,
    beta: &Tensor,
    eps: f32,
) -> Result<LiveTensor<'w>> {
    let hidden_size_xs = xs.dim(D::Minus1)?;
    let hidden_size_alpha = alpha.dims1()?;
    let hidden_size_beta = beta.dims1()?;
    if hidden_size_xs != hidden_size_alpha || hidden_size_xs != hidden_size_beta {
        candle::bail!(
            "shape mismatch in layer-norm src: {:?} alpha: {:?} beta: {:?}",
            xs.shape(),
            alpha.shape(),
            beta.shape()
        )
    }
    xs.apply_op3_no_bwd(alpha, beta, &LayerNorm { eps })
}
