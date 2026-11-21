//! Provides softmax and log-softmax over a chosen dimension, with a fused form for
//! the last dimension.
//!
//! The general form applies the softmax along the requested dimension and returns
//! values between zero and one that sum to one along it. The last-dimension form is a
//! custom op whose output is allocated beside its input, which is the common case for
//! attention scores.

use super::*;

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

pub(super) struct SoftmaxLastDim;

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
                            stream.cu_stream() as *mut std::ffi::c_void,
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
            anchor: None,
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
