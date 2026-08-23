//! Tensor ops.
//!

use candle::wave_provenance::WaveTicket;
use candle::{CpuStorage, DType, Layout, LiveTensor, Module, Result, Shape, Tensor, D};
use rayon::prelude::*;

/// Applies the softmax function to the input tensor, rescaling the element so that elements on
/// a slice of fixed index on dimension `dim` are between 0 and 1 and sum to 1.
///
/// ```rust
/// use candle::{Tensor, Device, test_utils::to_vec2_round};
/// let a = Tensor::new(&[[0f32, 1., 0., 1.], [-2., 2., 3., -3.]], &Device::Cpu)?;
/// let a = candle_nn::ops::softmax(&a, 1)?;
/// assert_eq!(
///     to_vec2_round(&a, 4)?,
///     &[
///         [0.1345, 0.3655, 0.1345, 0.3655],
///         [0.0049, 0.2671, 0.7262, 0.0018]
///     ]);
/// # Ok::<(), candle::Error>(())
/// ```
pub fn softmax<D: candle::shape::Dim>(xs: &Tensor, dim: D) -> Result<Tensor> {
    let dim = dim.to_index(xs.shape(), "softmax")?;
    let max = xs.max_keepdim(dim)?;
    let diff = xs.broadcast_sub(&max)?;
    let num = diff.exp()?;
    let den = num.sum_keepdim(dim)?;
    num.broadcast_div(&den)
}

pub fn log_softmax<D: candle::shape::Dim>(xs: &Tensor, d: D) -> Result<Tensor> {
    let d = d.to_index(xs.shape(), "log-softmax")?;
    let max = xs.max_keepdim(d)?;
    let diff = xs.broadcast_sub(&max)?;
    let sum_exp = diff.exp()?.sum_keepdim(d)?;
    let log_sm = diff.broadcast_sub(&sum_exp.log()?)?;
    Ok(log_sm)
}

pub fn silu<'w>(xs: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    xs.silu()
}

pub fn swiglu<'w>(xs: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    let xs = xs.chunk(2, D::Minus1)?;
    &xs[0].silu()? * &xs[1]
}

struct Sigmoid;

impl candle::CustomOp1 for Sigmoid {
    fn name(&self) -> &'static str {
        "sigmoid"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        use candle::backend::BackendStorage;

        fn fwd<T: num_traits::Float>(v: T) -> T {
            (v.neg().exp() + T::one()).recip()
        }

        // FIXME: using `candle::map_dtype` causes compilation errors.
        let storage = match storage {
            CpuStorage::BF16(slice) => {
                CpuStorage::BF16(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F16(slice) => {
                CpuStorage::F16(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F32(slice) => {
                CpuStorage::F32(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            CpuStorage::F64(slice) => {
                CpuStorage::F64(candle::cpu_backend::unary_map(slice, layout, fwd))
            }
            _ => Err(candle::Error::UnsupportedDTypeForOp(
                storage.dtype(),
                self.name(),
            ))?,
        };
        Ok((storage, layout.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::{kernels, CudaStorageSlice};

        let dev = storage.device();
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let start_offset = layout.start_offset();
        let stream = dev.cuda_stream();

        // Get dtype for FFI dispatcher
        let dtype = match &storage.slice {
            CudaStorageSlice::F32(_) => kernels::simple::unary::UnaryDType::F32 as i32,
            CudaStorageSlice::F64(_) => kernels::simple::unary::UnaryDType::F64 as i32,
            CudaStorageSlice::F16(_) => kernels::simple::unary::UnaryDType::F16 as i32,
            CudaStorageSlice::BF16(_) => kernels::simple::unary::UnaryDType::BF16 as i32,
            CudaStorageSlice::F8E4M3(_) => kernels::simple::unary::UnaryDType::F8E4M3 as i32,
            _ => candle::bail!("sigmoid not supported for dtype {:?}", storage.dtype()),
        };

        // Prepare dims/strides info for non-contiguous tensors
        let info: Option<candle::cuda_backend::cudarc::driver::CudaSlice<usize>> =
            if layout.is_contiguous() {
                None
            } else {
                Some(dev.memcpy_stod(&[dims, layout.stride()].concat())?)
            };
        let info_ptr = match &info {
            Some(s) => {
                let (ptr, _guard) = s.device_ptr(&stream);
                ptr as *const usize
            }
            None => std::ptr::null(),
        };

        // Macro to handle each dtype case with proper scoping for guards
        // The operand's arena: this op's output is allocated beside its input,
        // which is what makes the `'w` on the result true rather than merely
        // permitted. Declared before the dispatch macro because a `macro_rules!`
        // body resolves free identifiers at its definition site, not its call.
        let inherit = storage.backing;
        // Assigned by the macro to whatever `alloc_inheriting` resolved.
        let out_backing;
        macro_rules! sigmoid_impl {
            ($src_slice:expr, $dtype_variant:ident, $rust_type:ty) => {{
                let src = $src_slice.slice(start_offset..);
                let (out, resolved_backing) = unsafe {
                    candle::cuda_backend::alloc_inheriting::<$rust_type>(dev, el_count, inherit)?
                };
                out_backing = resolved_backing;
                {
                    let (src_ptr, _src_guard) = src.device_ptr(&stream);
                    let (out_ptr, _out_guard) = out.device_ptr(&stream);
                    let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));
                    #[cfg(feature = "cuda")]
                    candle::set_kernel_breadcrumb("run_unary_op(sigmoid)", file!(), line!());
                    unsafe {
                        kernels::simple::unary::run_unary_op(
                            kernels::simple::unary::UnaryOp::Sigmoid as i32,
                            dtype,
                            el_count,
                            dims.len(),
                            info_ptr,
                            src_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                        );
                    }
                }
                CudaStorageSlice::$dtype_variant(out)
            }};
        }

        let slice = match &storage.slice {
            CudaStorageSlice::F32(s) => sigmoid_impl!(s, F32, f32),
            CudaStorageSlice::F64(s) => sigmoid_impl!(s, F64, f64),
            CudaStorageSlice::F16(s) => sigmoid_impl!(s, F16, half::f16),
            CudaStorageSlice::BF16(s) => sigmoid_impl!(s, BF16, half::bf16),
            CudaStorageSlice::F8E4M3(s) => sigmoid_impl!(s, F8E4M3, float8::F8E4M3),
            _ => candle::bail!("sigmoid not supported for dtype {:?}", storage.dtype()),
        };

        let dst = candle::CudaStorage {
            slice,
            device: dev.clone(),
            backing: out_backing,
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::MetalError;
        let device = storage.device();
        let dtype = storage.dtype();
        let shape = layout.shape();
        let el_count = shape.elem_count();
        let buffer = device.new_buffer(el_count, dtype, "sigmoid")?;
        let command_buffer = device.command_buffer()?;
        command_buffer.set_label("sigmoid");
        let src = candle_metal_kernels::BufferOffset {
            buffer: storage.buffer(),
            offset_in_bytes: layout.start_offset() * storage.dtype().size_in_bytes(),
        };

        match (el_count % 2, dtype, layout.is_contiguous()) {
            (0, DType::BF16 | DType::F16, true) => {
                use candle_metal_kernels::unary::contiguous_tiled;
                let kernel_name = match dtype {
                    DType::F16 => contiguous_tiled::sigmoid::HALF,
                    DType::F32 => contiguous_tiled::sigmoid::FLOAT,
                    DType::BF16 => contiguous_tiled::sigmoid::BFLOAT,
                    dtype => {
                        candle::bail!(
                            "Metal contiguous_tiled unary sigmoid {dtype:?} not implemented"
                        )
                    }
                };
                candle_metal_kernels::call_unary_contiguous_tiled(
                    device.metal_device(),
                    &command_buffer,
                    device.kernels(),
                    kernel_name,
                    el_count,
                    src,
                    &buffer,
                )
                .map_err(MetalError::from)?;
            }
            (_, _, true) => {
                use candle_metal_kernels::unary::contiguous;
                let kernel_name = match dtype {
                    DType::F16 => contiguous::sigmoid::HALF,
                    DType::F32 => contiguous::sigmoid::FLOAT,
                    DType::BF16 => contiguous::sigmoid::BFLOAT,
                    dtype => {
                        candle::bail!("Metal contiguous unary sigmoid {dtype:?} not implemented")
                    }
                };
                candle_metal_kernels::call_unary_contiguous(
                    device.metal_device(),
                    &command_buffer,
                    device.kernels(),
                    kernel_name,
                    el_count,
                    src,
                    &buffer,
                )
                .map_err(MetalError::from)?;
            }
            (_, _, false) => {
                use candle_metal_kernels::unary::strided;
                let kernel_name = match dtype {
                    DType::F16 => strided::sigmoid::HALF,
                    DType::F32 => strided::sigmoid::FLOAT,
                    DType::BF16 => strided::sigmoid::BFLOAT,
                    dtype => {
                        candle::bail!("Metal strided unary sigmoid {dtype:?} not implemented")
                    }
                };
                let dst = candle_metal_kernels::BufferOffset::zero_offset(&buffer);
                candle_metal_kernels::call_unary_strided(
                    device.metal_device(),
                    &command_buffer,
                    device.kernels(),
                    kernel_name,
                    layout.dims(),
                    src,
                    layout.stride(),
                    dst,
                )
                .map_err(MetalError::from)?;
            }
        }

        let new_storage = candle::MetalStorage::new(buffer, device.clone(), el_count, dtype);
        Ok((new_storage, layout.shape().clone()))
    }

    fn bwd(&self, _arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // d/dx sigmoid(x) = (1 - sigmoid(x)) * sigmoid(x)
        let d_dx_sigmoid = res.ones_like()?.sub(res)?.mul(res)?;
        Ok(Some(grad_res.mul(&d_dx_sigmoid)?))
    }
}

pub fn sigmoid<'w>(xs: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    xs.apply_op1(Sigmoid)
}

pub fn hard_sigmoid<'w>(xs: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    // TODO: Should we have a specialized op for this?
    ((xs + 3.0)? / 6.0)?.clamp(0f32, 1f32)
}

pub fn mish<'w>(xs: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    xs * (1.0 + xs.exp()?)?.log()?.tanh()
}

pub fn leaky_relu<'w>(xs: &LiveTensor<'w>, negative_slope: f64) -> Result<LiveTensor<'w>> {
    let zeros = xs.zeros_like()?;
    xs.maximum(&zeros)? + xs.minimum(&zeros)? * negative_slope
}

pub fn selu(xs: &Tensor, alpha: f32, gamma: f32) -> Result<Tensor> {
    let is_pos = xs.gt(0f32)?;
    let alpha_t = Tensor::full(alpha, xs.dims(), xs.device())?;
    let neg = xs.exp()?.mul(&alpha_t)?.sub(&alpha_t)?;
    let selu = is_pos.where_cond(xs, &neg)?;
    let gamma_t = Tensor::full(gamma, xs.dims(), xs.device())?;
    selu.broadcast_mul(&gamma_t)
}

pub fn dropout(xs: &Tensor, drop_p: f32) -> Result<Tensor> {
    // This implementation is inefficient as it stores the full mask for the backward pass.
    // Instead we could just store the seed and have a specialized kernel that would both
    // generate the random mask and apply it.
    // Another easier optimization would be to be able to generate boolean mask using just a bit of
    // entropy per element rather than generating a full float per element.
    if !(0. ..1.).contains(&drop_p) {
        candle::bail!("dropout probability has to be in [0, 1), got {drop_p}")
    }
    let rand = Tensor::rand(0f32, 1f32, xs.shape(), xs.device())?;
    let scale = 1.0 / (1.0 - drop_p as f64);
    let drop_p = Tensor::new(drop_p, xs.device())?.broadcast_as(xs.shape())?;
    let mask = (rand.ge(&drop_p)?.to_dtype(xs.dtype())? * scale)?;
    xs * mask
}

#[derive(Clone, Debug)]
pub struct Dropout {
    drop_p: f32,
}

impl Dropout {
    pub fn new(drop_p: f32) -> Dropout {
        Self { drop_p }
    }

    pub fn forward(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        if train {
            dropout(xs, self.drop_p)
        } else {
            Ok(xs.clone())
        }
    }
}

impl candle::ModuleT for Dropout {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor> {
        self.forward(xs, train)
    }
}

struct SoftmaxLastDim;

impl candle::CustomOp1 for SoftmaxLastDim {
    fn name(&self) -> &'static str {
        "softmax-last-dim"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        fn softmax<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            layout: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match layout.contiguous_offsets() {
                None => candle::bail!("input has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let el_count = layout.shape().elem_count();
            let dims = layout.shape().dims();
            let dim_m1 = dims[dims.len() - 1];
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(dim_m1)
                .zip(dst.par_chunks_mut(dim_m1))
                .for_each(|(src, dst)| {
                    let mut max = T::neg_infinity();
                    unsafe { T::vec_reduce_max(src.as_ptr(), &mut max, dim_m1) };
                    for (s, d) in src.iter().zip(dst.iter_mut()) {
                        *d = (*s - max).exp();
                    }
                    let mut sum_exp = T::zero();
                    unsafe { T::vec_reduce_sum(dst.as_ptr(), &mut sum_exp, dim_m1) };
                    for d in dst.iter_mut() {
                        *d /= sum_exp
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, Shape::from_dims(dims)))
        }

        match storage {
            CpuStorage::BF16(slice) => softmax::<half::bf16>(slice, layout),
            CpuStorage::F16(slice) => softmax::<half::f16>(slice, layout),
            CpuStorage::F32(slice) => softmax::<f32>(slice, layout),
            CpuStorage::F64(slice) => softmax::<f64>(slice, layout),
            _ => candle::bail!("unsupported dtype for softmax {:?}", storage),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        storage: &candle::CudaStorage,
        layout: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::{kernels, CudaStorageSlice};

        let dev = storage.device();
        let stream = dev.cuda_stream();

        let (o1, o2) = match layout.contiguous_offsets() {
            None => candle::bail!("input has to be contiguous"),
            Some(offsets) => offsets,
        };
        let el = layout.shape().elem_count();
        let dims = layout.shape().dims();
        let dim_m1 = dims[dims.len() - 1];
        let n_cols = dim_m1 as i32;
        let n_rows = (el / dim_m1) as i32;

        // Get dtype for FFI dispatcher
        let dtype = match &storage.slice {
            CudaStorageSlice::F32(_) => kernels::simple::reduce::FloatDType::F32 as i32,
            CudaStorageSlice::F64(_) => kernels::simple::reduce::FloatDType::F64 as i32,
            CudaStorageSlice::F16(_) => kernels::simple::reduce::FloatDType::F16 as i32,
            CudaStorageSlice::BF16(_) => kernels::simple::reduce::FloatDType::BF16 as i32,
            CudaStorageSlice::F8E4M3(_) => kernels::simple::reduce::FloatDType::F8E4M3 as i32,
            _ => candle::bail!("softmax not supported for dtype {:?}", storage.dtype()),
        };

        // The operand's arena: this op's output is allocated beside its input,

        // which is what makes the `'w` on the result true rather than merely

        // permitted. Declared before the dispatch macro because a `macro_rules!`

        // body resolves free identifiers at its definition site, not its call.

        let inherit = storage.backing;

        // Assigned by the macro to whatever `alloc_inheriting` resolved.

        let out_backing;

        macro_rules! softmax_impl {
            ($src_slice:expr, $dtype_variant:ident, $rust_type:ty) => {{
                let src = $src_slice.slice(o1..o2);
                let (dst, resolved_backing) = unsafe {
                    candle::cuda_backend::alloc_inheriting::<$rust_type>(dev, el, inherit)?
                };
                out_backing = resolved_backing;
                {
                    let (src_ptr, _src_guard) = src.device_ptr(&stream);
                    let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
                    #[cfg(feature = "cuda")]
                    candle::set_kernel_breadcrumb("run_softmax_op", file!(), line!());
                    unsafe {
                        kernels::simple::reduce::run_softmax_op(
                            dtype,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            n_rows,
                            n_cols,
                        );
                    }
                }
                CudaStorageSlice::$dtype_variant(dst)
            }};
        }

        let slice = match &storage.slice {
            CudaStorageSlice::F32(src) => softmax_impl!(src, F32, f32),
            CudaStorageSlice::F64(src) => softmax_impl!(src, F64, f64),
            CudaStorageSlice::F16(src) => softmax_impl!(src, F16, half::f16),
            CudaStorageSlice::BF16(src) => softmax_impl!(src, BF16, half::bf16),
            CudaStorageSlice::F8E4M3(src) => softmax_impl!(src, F8E4M3, float8::F8E4M3),
            _ => candle::bail!("softmax not supported for dtype {:?}", storage.dtype()),
        };

        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
            backing: out_backing,
        };
        Ok((dst, layout.shape().clone()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = storage.device();
        let command_buffer = device.command_buffer()?;
        let kernels = device.kernels();
        let name = match storage.dtype() {
            DType::F32 => "softmax_f32",
            DType::F16 => "softmax_f16",
            DType::BF16 => "softmax_bf16",
            dtype => candle::bail!("softmax-last-dim is not implemented for {dtype:?}"),
        };

        let n = layout.stride().len();
        if !(layout.is_contiguous() && layout.stride()[n - 1] == 1) {
            candle::bail!("Non contiguous softmax-last-dim is not implemented");
        }

        let last_dim = layout.dims()[layout.shape().rank() - 1];
        let elem_count = layout.shape().elem_count();
        let output = device.new_buffer(elem_count, storage.dtype(), "softmax")?;
        candle_metal_kernels::call_last_softmax(
            device.metal_device(),
            &command_buffer,
            kernels,
            name,
            elem_count,
            last_dim,
            storage.buffer(),
            layout.start_offset() * storage.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let newstorage =
            candle::MetalStorage::new(output, device.clone(), elem_count, storage.dtype());
        Ok((newstorage, layout.shape().clone()))
    }
}

pub fn softmax_last_dim<'w>(xs: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    xs.apply_op1_no_bwd(&SoftmaxLastDim)
}

#[derive(Debug, Clone)]
struct RmsNorm {
    eps: f32,
    /// The arena to allocate the output from when the *operand* names none.
    ///
    /// Every op downstream of a wave-backed value inherits its arena from that
    /// value, so a chain only needs to be told where it lives once — at its
    /// head, whose operand is the residual stream and therefore pool-backed.
    /// `None` is the ordinary case and means "inherit or fall back to the pool",
    /// which is what plain [`rms_norm`] does.
    ///
    /// Only the CUDA path reads it — there are no wave arenas to name without a
    /// device — but the field stays so both builds construct the same struct.
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    root: Option<WaveTicket>,
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

        // The operand's arena: this op's output is allocated beside its input,
        // which is what makes the `'w` on the result true rather than merely
        // permitted. When the operand names no arena — the head of a chain,
        // whose input is the residual stream and so pool-backed — `root` says
        // where the chain lives instead, and everything downstream follows from
        // this one output without another mention of the wave. Declared before
        // the dispatch macro because a `macro_rules!` body resolves free
        // identifiers at its definition site, not its call.
        let inherit = match s1.backing.inherit_ticket() {
            Some(_) => s1.backing,
            None => candle::cuda_backend::Backing::from_ticket(self.root),
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

/// [`rms_norm`], allocating its output from `root` when `xs` names no arena.
///
/// **The seed of a wave-scoped chain.** Operand provenance carries an arena from
/// a value to everything computed from it, so a chain of forty ops needs to be
/// told where it lives exactly once — and it cannot inherit that from its own
/// input, which is the residual stream and crosses layers on the pool. This is
/// where a layer says it: normalise the residual into the wave's span, and the
/// rest of the layer lands there by construction.
///
/// `root` is ignored when `xs` already carries an arena — a value that came from
/// a wave belongs to *that* wave, and a caller cannot re-home it by asking.
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

// =============================================================================
// Fused SiLU-Mul: out = silu(gate) * up
// =============================================================================
// Eliminates 1 kernel launch + 1 intermediate allocation per call.
// This is the core SwiGLU activation used in MoE expert FFNs.

#[derive(Debug, Clone)]
struct SiluMul;

impl candle::CustomOp2 for SiluMul {
    fn name(&self) -> &'static str {
        "fused-silu-mul"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        fn inner<T: candle::WithDType + num_traits::Float>(
            gate: &[T],
            gate_layout: &Layout,
            up: &[T],
            up_layout: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let gate = match gate_layout.contiguous_offsets() {
                None => candle::bail!("fused-silu-mul: gate must be contiguous"),
                Some((o1, o2)) => &gate[o1..o2],
            };
            let up = match up_layout.contiguous_offsets() {
                None => candle::bail!("fused-silu-mul: up must be contiguous"),
                Some((o1, o2)) => &up[o1..o2],
            };
            let one = T::from(1.0f64).unwrap();
            let dst: Vec<T> = gate
                .par_iter()
                .zip(up.par_iter())
                .map(|(&g, &u)| {
                    // silu(g) * u = g / (1 + exp(-g)) * u
                    let silu_g = g / (one + (-g).exp());
                    silu_g * u
                })
                .collect();
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, gate_layout.shape().clone()))
        }

        use CpuStorage as C;
        match (s1, s2) {
            (C::BF16(s1), C::BF16(s2)) => inner::<half::bf16>(s1, l1, s2, l2),
            (C::F16(s1), C::F16(s2)) => inner::<half::f16>(s1, l1, s2, l2),
            (C::F32(s1), C::F32(s2)) => inner::<f32>(s1, l1, s2, l2),
            (C::F64(s1), C::F64(s2)) => inner::<f64>(s1, l1, s2, l2),
            _ => candle::bail!("fused-silu-mul: unsupported dtype combination"),
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

        let el = l1.shape().elem_count();
        if el != l2.shape().elem_count() {
            candle::bail!(
                "fused-silu-mul: shape mismatch {:?} vs {:?}",
                l1.shape(),
                l2.shape()
            );
        }

        // Both inputs must be contiguous (they're fresh GEMM outputs).
        let (gate_o1, gate_o2) = match l1.contiguous_offsets() {
            None => candle::bail!("fused-silu-mul: gate must be contiguous"),
            Some(offsets) => offsets,
        };
        let (up_o1, up_o2) = match l2.contiguous_offsets() {
            None => candle::bail!("fused-silu-mul: up must be contiguous"),
            Some(offsets) => offsets,
        };

        let dtype = match &s1.slice {
            CudaStorageSlice::F32(_) => {
                kernels::simple::fused_silu_mul::FusedSiluMulDType::F32 as i32
            }
            CudaStorageSlice::F16(_) => {
                kernels::simple::fused_silu_mul::FusedSiluMulDType::F16 as i32
            }
            CudaStorageSlice::BF16(_) => {
                kernels::simple::fused_silu_mul::FusedSiluMulDType::BF16 as i32
            }
            CudaStorageSlice::F8E4M3(_) => {
                kernels::simple::fused_silu_mul::FusedSiluMulDType::F8E4M3 as i32
            }
            _ => candle::bail!("fused-silu-mul: unsupported dtype {:?}", s1.dtype()),
        };

        // The operand's arena: this op's output is allocated beside its input,

        // which is what makes the `'w` on the result true rather than merely

        // permitted. Declared before the dispatch macro because a `macro_rules!`

        // body resolves free identifiers at its definition site, not its call.

        let inherit = s1.backing;

        // Assigned by the macro to whatever `alloc_inheriting` resolved.

        let out_backing;

        macro_rules! silu_mul_impl {
            ($gate_slice:expr, $up_slice:expr, $dtype_variant:ident, $rust_type:ty) => {{
                let gate = $gate_slice.slice(gate_o1..gate_o2);
                let up = $up_slice.slice(up_o1..up_o2);
                let (dst, resolved_backing) = unsafe {
                    candle::cuda_backend::alloc_inheriting::<$rust_type>(dev, el, inherit)?
                };
                out_backing = resolved_backing;
                {
                    let (gate_ptr, _g_guard) = gate.device_ptr(&stream);
                    let (up_ptr, _u_guard) = up.device_ptr(&stream);
                    let (dst_ptr, _d_guard) = dst.device_ptr(&stream);
                    #[cfg(feature = "cuda")]
                    candle::set_kernel_breadcrumb("run_fused_silu_mul", file!(), line!());
                    unsafe {
                        kernels::simple::fused_silu_mul::run_fused_silu_mul(
                            dtype,
                            el,
                            0,                // num_dims = 0 → contiguous
                            std::ptr::null(), // dims_and_strides = null → contiguous
                            gate_ptr as *const std::ffi::c_void,
                            up_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                        );
                    }
                }
                CudaStorageSlice::$dtype_variant(dst)
            }};
        }

        let slice = match (&s1.slice, &s2.slice) {
            (CudaStorageSlice::F32(gate), CudaStorageSlice::F32(up)) => {
                silu_mul_impl!(gate, up, F32, f32)
            }
            (CudaStorageSlice::F16(gate), CudaStorageSlice::F16(up)) => {
                silu_mul_impl!(gate, up, F16, half::f16)
            }
            (CudaStorageSlice::BF16(gate), CudaStorageSlice::BF16(up)) => {
                silu_mul_impl!(gate, up, BF16, half::bf16)
            }
            (CudaStorageSlice::F8E4M3(gate), CudaStorageSlice::F8E4M3(up)) => {
                silu_mul_impl!(gate, up, F8E4M3, float8::F8E4M3)
            }
            _ => candle::bail!("fused-silu-mul: dtype mismatch between gate and up"),
        };

        let dst = candle::cuda_backend::CudaStorage {
            slice,
            device: dev.clone(),
            backing: out_backing,
        };
        Ok((dst, l1.shape().clone()))
    }
}

/// Fused SiLU-Mul activation: `silu(gate) * up`.
///
/// This is the SwiGLU activation pattern used in MoE expert FFNs.
/// On CUDA, this runs as a single fused kernel instead of separate
/// `silu()` + `mul()` calls, saving 1 kernel launch and 1 intermediate
/// tensor allocation per invocation.
pub fn silu_mul<'w>(gate: &LiveTensor<'w>, up: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    gate.apply_op2_no_bwd(up, &SiluMul)
}

#[derive(Debug, Clone)]
struct LayerNorm {
    eps: f32,
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

// https://pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html
pub fn pixel_shuffle(xs: &Tensor, upscale_factor: usize) -> Result<Tensor> {
    let (b_size, c, h, w) = xs.dims4()?;
    let out_c = c / upscale_factor / upscale_factor;
    xs.reshape((b_size, out_c, upscale_factor, upscale_factor, h, w))?
        .permute((0, 1, 4, 2, 5, 3))?
        .reshape((b_size, out_c, h * upscale_factor, w * upscale_factor))
}

pub fn pixel_unshuffle(xs: &Tensor, downscale_factor: usize) -> Result<Tensor> {
    let (b_size, c, h, w) = xs.dims4()?;
    let out_c = c * downscale_factor * downscale_factor;
    xs.reshape((
        b_size,
        c,
        h / downscale_factor,
        downscale_factor,
        w / downscale_factor,
        downscale_factor,
    ))?
    .permute((0, 1, 3, 5, 2, 4))?
    .reshape((b_size, out_c, h / downscale_factor, w / downscale_factor))
}

// https://pytorch.org/docs/stable/generated/torch.nn.ReplicationPad2d.html
pub fn replication_pad2d(xs: &Tensor, pad: usize) -> Result<Tensor> {
    match pad {
        0 => Ok(xs.clone()),
        1 => {
            let (_b_size, _c, h, w) = xs.dims4()?;
            let (first, last) = (xs.narrow(3, 0, 1)?, xs.narrow(3, w - 1, 1)?);
            let xs = Tensor::cat(&[&first, xs, &last], 3)?;
            let (first, last) = (xs.narrow(2, 0, 1)?, xs.narrow(2, h - 1, 1)?);
            Tensor::cat(&[&first, &xs, &last], 2)
        }
        n => candle::bail!("replication-pad with a size of {n} is not supported"),
    }
}

#[derive(Clone, Debug)]
pub struct Identity;

impl Identity {
    pub fn new() -> Identity {
        Self
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self
    }
}

impl Module for Identity {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.clone())
    }
}

#[allow(dead_code)]
struct Sdpa {
    scale: f32,
    softcapping: f32,
}

impl candle::CustomOp3 for Sdpa {
    fn name(&self) -> &'static str {
        "metal-sdpa"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("SDPA has no cpu impl")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        q: &candle::MetalStorage,
        q_l: &Layout,
        k: &candle::MetalStorage,
        k_l: &Layout,
        v: &candle::MetalStorage,
        v_l: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        use candle_metal_kernels::SdpaDType;

        let device = q.device();

        let out_dims = vec![q_l.dim(0)?, q_l.dim(1)?, q_l.dim(2)?, v_l.dim(3)?];
        let elem_count: usize = out_dims.iter().product();

        let output = device.new_buffer(elem_count, q.dtype(), "sdpa_o")?;

        // q,k must have matching emb dim
        if q_l.dim(D::Minus1)? != k_l.dim(D::Minus1)? {
            candle::bail!("`q` and `k` last dims must match");
        }

        // k,v must have matching n kv heads
        if v_l.dim(D::Minus(3))? != k_l.dim(D::Minus(3))? {
            candle::bail!("`k` and `v` head dims must match");
        }

        // n_heads % n_kv_heads == 0; n_heads >= 1, n_kv_heads >= 1.
        if q_l.dim(D::Minus(3))? % k_l.dim(D::Minus(3))? != 0 {
            candle::bail!("query `n_heads` must be a multiple of `n_kv_heads`");
        }

        let k_head = k_l.dim(D::Minus1)?;
        let q_head = q_l.dim(D::Minus1)?;
        let q_seq = q_l.dim(2)?;

        let mut implementation_supports_use_case = q_head == k_head;
        let supported_head_dim =
            q_head == 32 || q_head == 64 || q_head == 96 || q_head == 128 || q_head == 256;

        const SDPA_FULL_THRESHOLD: usize = 2;

        let supports_sdpa_full =
            q_seq >= SDPA_FULL_THRESHOLD && supported_head_dim && q_head == k_head;
        let supports_sdpa_vector = q_seq == 1 && supported_head_dim;

        implementation_supports_use_case &= supports_sdpa_full || supports_sdpa_vector;

        if !supported_head_dim {
            candle::bail!(
                "Meta SDPA does not support q head dim {q_head}: q dims {:?}, k dims {:?}, v dims {:?}.",
                q_l.dims(),
                k_l.dims(),
                v_l.dims()
            );
        }
        if !implementation_supports_use_case {
            candle::bail!(
                "Meta SDPA does not support q dims {:?}, k dims {:?}, v dims {:?}.",
                q_l.dims(),
                k_l.dims(),
                v_l.dims()
            );
        }

        for t in [k.dtype(), v.dtype()] {
            if q.dtype() != t {
                candle::bail!("all q, k, v dtypes must match.");
            }
        }

        let itype = match q.dtype() {
            DType::BF16 => SdpaDType::BF16,
            DType::F16 => SdpaDType::F16,
            DType::F32 => SdpaDType::F32,
            other => candle::bail!("unsupported sdpa type {other:?}"),
        };

        let command_buffer = q.device().command_buffer()?;
        if supports_sdpa_vector {
            // Route to the 2 pass fused attention if the k seqlen is large.
            // https://github.com/ml-explore/mlx/pull/1597
            const TWO_PASS_K_THRESHOLD: usize = 1024;
            if k_l.dim(2)? >= TWO_PASS_K_THRESHOLD {
                let mut intermediate_shape = [
                    &out_dims[0..out_dims.len() - 2],
                    &[candle_metal_kernels::SDPA_2PASS_BLOCKS],
                    &[out_dims[out_dims.len() - 1]],
                ]
                .concat();
                let intermediate = device.new_buffer(
                    intermediate_shape.iter().product::<usize>(),
                    DType::F32,
                    "sdpa_2pass_intermediate",
                )?;
                let _ = intermediate_shape.pop().unwrap();
                let sums = device.new_buffer(
                    intermediate_shape.iter().product::<usize>(),
                    DType::F32,
                    "sdpa_2pass_sums",
                )?;
                let maxs = device.new_buffer(
                    intermediate_shape.iter().product::<usize>(),
                    DType::F32,
                    "sdpa_2pass_maxs",
                )?;

                command_buffer.set_label("vector_attention");
                candle_metal_kernels::call_sdpa_vector_2pass(
                    q.device().device(),
                    &command_buffer,
                    q.device().kernels(),
                    q_l.start_offset(),
                    q_l.dims(),
                    q.buffer(),
                    k_l.start_offset(),
                    k_l.dims(),
                    k_l.stride(),
                    k.buffer(),
                    v_l.start_offset(),
                    v_l.stride(),
                    v.buffer(),
                    &output,
                    &intermediate,
                    &sums,
                    &maxs,
                    self.scale,
                    self.softcapping,
                    itype,
                )
                .map_err(candle::Error::wrap)?;
            } else {
                command_buffer.set_label("vector_attention");
                candle_metal_kernels::call_sdpa_vector(
                    q.device().device(),
                    &command_buffer,
                    q.device().kernels(),
                    q_l.start_offset(),
                    q_l.dims(),
                    q.buffer(),
                    k_l.start_offset(),
                    k_l.dims(),
                    k_l.stride(),
                    k.buffer(),
                    v_l.start_offset(),
                    v_l.stride(),
                    v.buffer(),
                    &output,
                    self.scale,
                    self.softcapping,
                    itype,
                )
                .map_err(candle::Error::wrap)?;
            }
        } else if supports_sdpa_full {
            if q_l.dim(2)? != k_l.dim(2)? {
                candle::bail!(
                    "query and key sequence length must be equal if using full metal sdpa"
                )
            }

            command_buffer.set_label("full_attention");
            candle_metal_kernels::call_sdpa_full(
                q.device().device(),
                &command_buffer,
                q.device().kernels(),
                q_l.start_offset(),
                q_l.dims(),
                q.buffer(),
                k_l.start_offset(),
                k.buffer(),
                v_l.start_offset(),
                v.buffer(),
                &output,
                self.scale,
                self.softcapping,
                itype,
            )
            .map_err(candle::Error::wrap)?;
        } else {
            candle::bail!("must be vector or full sdpa kernel");
        }

        let newstorage = candle::MetalStorage::new(output, device.clone(), elem_count, q.dtype());
        Ok((newstorage, Shape::from_dims(&out_dims)))
    }
}

/// Scaled dot product attention with a fused kernel.
///
/// Computes softmax(qk^T*scale)v.
///
/// **Inputs shapes:**
/// - `q`: (bs, qhead, seq, hidden)
/// - `k`: (bs, kv_head, kv_seq, hidden)
/// - `k`: (bs, kv_head, kv_seq, v_hidden)
/// - `scale` is applied before softmax.
/// - If `softcapping` != 1.0:
///      - Computation is: softmax(tanh(qk^T*scale/cap)*cap)v
///
/// **Output shape:** (bs, qhead, seq, v_hidden)
///
/// **Supported head dims:** 32, 64, 96, 128, 256.
///
/// ## On Metal:
/// - If `seq` == 1:
///     - Use a vectorized kernel
///     - Supports `seq` != `kv_seq` (cross attn. support)
///     - Supports GQA when `qhead` is a multiple of `kv_head`
/// - Otherwise:
///     - Use an alternate kernel
///     - Requires `seq` == `kv_seq`
///     - GQA is not supported (requires `qhead` == `kv_head`)
pub fn sdpa<'w>(
    q: &LiveTensor<'w>,
    k: &LiveTensor<'w>,
    v: &LiveTensor<'w>,
    scale: f32,
    softcapping: f32,
) -> Result<LiveTensor<'w>> {
    q.apply_op3_no_bwd(k, v, &Sdpa { scale, softcapping })
}
