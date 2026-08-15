//! Rotary Embeddings
//!
use candle::{CpuStorage, Layout, LiveTensor, Result, Shape, Tensor, D};
use rayon::prelude::*;

/// Interleaved variant of rotary embeddings.
/// The x0 and x1 value are interleaved on the n_embd (= head_dim) dimension.
/// The resulting y0 and y1 are also interleaved with:
///   y0 = x0*cos - x1*sin
///   y1 = x0*sin + x1*cos
#[derive(Debug, Clone)]
struct RotaryEmbI;

impl candle::CustomOp3 for RotaryEmbI {
    fn name(&self) -> &'static str {
        "rotary-emb-int"
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
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * d)
                .zip(dst.par_chunks_mut(t * d))
                .enumerate()
                .for_each(|(bh_i, (src, dst))| {
                    for i_over_2 in 0..t * d / 2 {
                        let i = 2 * i_over_2;
                        let rope_i = if unbatched_rope {
                            let b_i = bh_i / h;
                            i_over_2 + b_i * t * d / 2
                        } else {
                            i_over_2
                        };
                        dst[i] = src[i] * cos[rope_i] - src[i + 1] * sin[rope_i];
                        dst[i + 1] = src[i] * sin[rope_i] + src[i + 1] * cos[rope_i];
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, h, t, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
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
            None => candle::bail!("src input has to be contiguous"),
            Some(offsets) => offsets,
        };
        let (cos_o1, cos_o2) = match l2.contiguous_offsets() {
            None => candle::bail!("cos input has to be contiguous"),
            Some(offsets) => offsets,
        };
        let (sin_o1, sin_o2) = match l3.contiguous_offsets() {
            None => candle::bail!("sin input has to be contiguous"),
            Some(offsets) => offsets,
        };

        let (b, h, t, d) = l1.shape().dims4()?;
        let stride_b = if l2.dims().len() == 3 && l3.dims().len() == 3 {
            (h * t * d) as u32
        } else {
            0u32
        };
        let el = b * h * t * d;
        let bh = (b * h) as u32;
        let td = (t * d) as u32;

        // Get dtype for FFI dispatcher
        let dtype = match &s1.slice {
            CudaStorageSlice::F32(_) => kernels::simple::reduce::FloatDType::F32 as i32,
            CudaStorageSlice::F64(_) => kernels::simple::reduce::FloatDType::F64 as i32,
            CudaStorageSlice::F16(_) => kernels::simple::reduce::FloatDType::F16 as i32,
            CudaStorageSlice::BF16(_) => kernels::simple::reduce::FloatDType::BF16 as i32,
            CudaStorageSlice::F8E4M3(_) => kernels::simple::reduce::FloatDType::F8E4M3 as i32,
            _ => candle::bail!("rope_i not supported for dtype {:?}", s1.dtype()),
        };

        // The operand's arena: this op's output is allocated beside its input,

        // which is what makes the `'w` on the result true rather than merely

        // permitted. Declared before the dispatch macro because a `macro_rules!`

        // body resolves free identifiers at its definition site, not its call.

        let inherit = s1.backing;

        // Assigned by the macro to whatever `alloc_inheriting` resolved.

        let out_backing;

        macro_rules! rope_i_impl {
            ($src_slice:expr, $cos_slice:expr, $sin_slice:expr, $dtype_variant:ident, $rust_type:ty) => {{
                let src = $src_slice.slice(src_o1..src_o2);
                let cos = $cos_slice.slice(cos_o1..cos_o2);
                let sin = $sin_slice.slice(sin_o1..sin_o2);
                let (dst, resolved_backing) = unsafe {
                    candle::cuda_backend::alloc_inheriting::<$rust_type>(dev, el, inherit)?
                };
                out_backing = resolved_backing;
                {
                    let (src_ptr, _src_guard) = src.device_ptr(&stream);
                    let (cos_ptr, _cos_guard) = cos.device_ptr(&stream);
                    let (sin_ptr, _sin_guard) = sin.device_ptr(&stream);
                    let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
                    #[cfg(feature = "cuda")]
                    candle::set_kernel_breadcrumb("run_rope_i_op", file!(), line!());
                    unsafe {
                        kernels::simple::reduce::run_rope_i_op(
                            dtype,
                            src_ptr as *const std::ffi::c_void,
                            cos_ptr as *const std::ffi::c_void,
                            sin_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            bh,
                            td,
                            stride_b,
                        );
                    }
                }
                CudaStorageSlice::$dtype_variant(dst)
            }};
        }

        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (
                CudaStorageSlice::BF16(src),
                CudaStorageSlice::BF16(cos),
                CudaStorageSlice::BF16(sin),
            ) => rope_i_impl!(src, cos, sin, BF16, half::bf16),
            (
                CudaStorageSlice::F16(src),
                CudaStorageSlice::F16(cos),
                CudaStorageSlice::F16(sin),
            ) => rope_i_impl!(src, cos, sin, F16, half::f16),
            (
                CudaStorageSlice::F32(src),
                CudaStorageSlice::F32(cos),
                CudaStorageSlice::F32(sin),
            ) => rope_i_impl!(src, cos, sin, F32, f32),
            (
                CudaStorageSlice::F64(src),
                CudaStorageSlice::F64(cos),
                CudaStorageSlice::F64(sin),
            ) => rope_i_impl!(src, cos, sin, F64, f64),
            (
                CudaStorageSlice::F8E4M3(src),
                CudaStorageSlice::F8E4M3(cos),
                CudaStorageSlice::F8E4M3(sin),
            ) => rope_i_impl!(src, cos, sin, F8E4M3, float8::F8E4M3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
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
        src: &candle::MetalStorage,
        l_src: &Layout,
        cos: &candle::MetalStorage,
        l_cos: &Layout,
        sin: &candle::MetalStorage,
        l_sin: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = src.device();
        let command_buffer = device.command_buffer()?;
        let kernels = device.kernels();
        if cos.dtype() != src.dtype() || sin.dtype() != src.dtype() {
            candle::bail!(
                "dtype mismatch in rope-i {:?} {:?} {:?}",
                src.dtype(),
                cos.dtype(),
                sin.dtype()
            )
        }
        let name = match src.dtype() {
            candle::DType::F32 => "rope_i_f32",
            candle::DType::F16 => "rope_i_f16",
            candle::DType::BF16 => "rope_i_bf16",
            dtype => candle::bail!("rope-i is not implemented for {dtype:?}"),
        };
        let (b, h, t, d) = l_src.shape().dims4()?;
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            h * t * d
        } else {
            0usize
        };
        let el = b * h * t * d;
        let output = device.new_buffer(el, src.dtype(), "rope-i")?;
        candle_metal_kernels::call_rope_i(
            device.metal_device(),
            &command_buffer,
            kernels,
            name,
            b * h,
            t * d,
            stride_b,
            src.buffer(),
            l_src.start_offset() * src.dtype().size_in_bytes(),
            cos.buffer(),
            l_cos.start_offset() * cos.dtype().size_in_bytes(),
            sin.buffer(),
            l_sin.start_offset() * sin.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let out = candle::MetalStorage::new(output, device.clone(), el, src.dtype());
        Ok((out, l_src.shape().clone()))
    }
}

fn rope_check_cs(cs: &Tensor, b_sz: usize) -> Result<(usize, usize)> {
    match *cs.dims() {
        [t, d] => Ok((t, d)),
        [b, t, d] => {
            if b != b_sz {
                candle::bail!("inconsistent batch size in rope {b_sz} {cs:?}",)
            }
            Ok((t, d))
        }
        _ => candle::bail!("cos/sin has to be 2D or 3D in rope {b_sz} {cs:?}"),
    }
}

pub fn rope_i<'w>(xs: &LiveTensor<'w>, cos: &Tensor, sin: &Tensor) -> Result<LiveTensor<'w>> {
    let (b_sz, _n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmbI)
}

pub fn rope_i_slow(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
    let cos = cos
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?;
    let sin = sin
        .narrow(0, 0, seq_len)?
        .reshape((seq_len, n_embd / 2, 1))?;
    let cos = cos.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
    let sin = sin.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
    let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
    let x0 = x.narrow(D::Minus1, 0, 1)?;
    let x1 = x.narrow(D::Minus1, 1, 1)?;
    let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
    let rope = Tensor::cat(&[y0, y1], D::Minus1)?;
    let rope = rope.flatten_from(D::Minus2)?;
    Ok(rope)
}

/// Contiguous variant of rope embeddings.
#[derive(Debug, Clone)]
struct RotaryEmb;

impl candle::CustomOp3 for RotaryEmb {
    fn name(&self) -> &'static str {
        "rotary-emb"
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
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, h, t, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * d)
                .zip(dst.par_chunks_mut(t * d))
                .enumerate()
                .for_each(|(bh_i, (src, dst))| {
                    for i_t in 0..t {
                        for i_d in 0..d / 2 {
                            let i1 = i_t * d + i_d;
                            let i2 = i1 + d / 2;
                            let i_cs = i_t * (d / 2) + i_d;
                            let i_cs = if unbatched_rope {
                                let b_i = bh_i / h;
                                i_cs + b_i * t * d / 2
                            } else {
                                i_cs
                            };
                            dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                            dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                        }
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, h, t, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
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
            None => candle::bail!("src input has to be contiguous"),
            Some(offsets) => offsets,
        };
        let (cos_o1, cos_o2) = match l2.contiguous_offsets() {
            None => candle::bail!("cos input has to be contiguous"),
            Some(offsets) => offsets,
        };
        let (sin_o1, sin_o2) = match l3.contiguous_offsets() {
            None => candle::bail!("sin input has to be contiguous"),
            Some(offsets) => offsets,
        };

        let (b, h, t, d) = l1.shape().dims4()?;
        let stride_b = if l2.dims().len() == 3 && l3.dims().len() == 3 {
            (h * t * d) as u32
        } else {
            0u32
        };
        let el = b * h * t * d;
        let bh = (b * h) as u32;
        let td = (t * d) as u32;
        let d_val = d as u32;

        // Get dtype for FFI dispatcher
        let dtype = match &s1.slice {
            CudaStorageSlice::F32(_) => kernels::simple::reduce::FloatDType::F32 as i32,
            CudaStorageSlice::F64(_) => kernels::simple::reduce::FloatDType::F64 as i32,
            CudaStorageSlice::F16(_) => kernels::simple::reduce::FloatDType::F16 as i32,
            CudaStorageSlice::BF16(_) => kernels::simple::reduce::FloatDType::BF16 as i32,
            CudaStorageSlice::F8E4M3(_) => kernels::simple::reduce::FloatDType::F8E4M3 as i32,
            _ => candle::bail!("rope not supported for dtype {:?}", s1.dtype()),
        };

        // The operand's arena: this op's output is allocated beside its input,

        // which is what makes the `'w` on the result true rather than merely

        // permitted. Declared before the dispatch macro because a `macro_rules!`

        // body resolves free identifiers at its definition site, not its call.

        let inherit = s1.backing;

        // Assigned by the macro to whatever `alloc_inheriting` resolved.

        let out_backing;

        macro_rules! rope_impl {
            ($src_slice:expr, $cos_slice:expr, $sin_slice:expr, $dtype_variant:ident, $rust_type:ty) => {{
                let src = $src_slice.slice(src_o1..src_o2);
                let cos = $cos_slice.slice(cos_o1..cos_o2);
                let sin = $sin_slice.slice(sin_o1..sin_o2);
                let (dst, resolved_backing) = unsafe {
                    candle::cuda_backend::alloc_inheriting::<$rust_type>(dev, el, inherit)?
                };
                out_backing = resolved_backing;
                {
                    let (src_ptr, _src_guard) = src.device_ptr(&stream);
                    let (cos_ptr, _cos_guard) = cos.device_ptr(&stream);
                    let (sin_ptr, _sin_guard) = sin.device_ptr(&stream);
                    let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
                    #[cfg(feature = "cuda")]
                    candle::set_kernel_breadcrumb("run_rope_op", file!(), line!());
                    unsafe {
                        kernels::simple::reduce::run_rope_op(
                            dtype,
                            src_ptr as *const std::ffi::c_void,
                            cos_ptr as *const std::ffi::c_void,
                            sin_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            bh,
                            td,
                            d_val,
                            stride_b,
                        );
                    }
                }
                CudaStorageSlice::$dtype_variant(dst)
            }};
        }

        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (
                CudaStorageSlice::BF16(src),
                CudaStorageSlice::BF16(cos),
                CudaStorageSlice::BF16(sin),
            ) => rope_impl!(src, cos, sin, BF16, half::bf16),
            (
                CudaStorageSlice::F16(src),
                CudaStorageSlice::F16(cos),
                CudaStorageSlice::F16(sin),
            ) => rope_impl!(src, cos, sin, F16, half::f16),
            (
                CudaStorageSlice::F32(src),
                CudaStorageSlice::F32(cos),
                CudaStorageSlice::F32(sin),
            ) => rope_impl!(src, cos, sin, F32, f32),
            (
                CudaStorageSlice::F64(src),
                CudaStorageSlice::F64(cos),
                CudaStorageSlice::F64(sin),
            ) => rope_impl!(src, cos, sin, F64, f64),
            (
                CudaStorageSlice::F8E4M3(src),
                CudaStorageSlice::F8E4M3(cos),
                CudaStorageSlice::F8E4M3(sin),
            ) => rope_impl!(src, cos, sin, F8E4M3, float8::F8E4M3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
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
        src: &candle::MetalStorage,
        l_src: &Layout,
        cos: &candle::MetalStorage,
        l_cos: &Layout,
        sin: &candle::MetalStorage,
        l_sin: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = src.device();
        let command_buffer = device.command_buffer()?;
        let kernels = device.kernels();
        if cos.dtype() != src.dtype() || sin.dtype() != src.dtype() {
            candle::bail!(
                "dtype mismatch in rope {:?} {:?} {:?}",
                src.dtype(),
                cos.dtype(),
                sin.dtype()
            )
        }
        let name = match src.dtype() {
            candle::DType::F32 => "rope_f32",
            candle::DType::F16 => "rope_f16",
            candle::DType::BF16 => "rope_bf16",
            dtype => candle::bail!("rope is not implemented for {dtype:?}"),
        };
        let (b, h, t, d) = l_src.shape().dims4()?;
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            h * t * d
        } else {
            0usize
        };
        let el = b * h * t * d;
        let output = device.new_buffer(el, src.dtype(), "rope-i")?;
        candle_metal_kernels::call_rope(
            device.metal_device(),
            &command_buffer,
            kernels,
            name,
            b * h,
            t * d,
            d,
            stride_b,
            src.buffer(),
            l_src.start_offset() * src.dtype().size_in_bytes(),
            cos.buffer(),
            l_cos.start_offset() * cos.dtype().size_in_bytes(),
            sin.buffer(),
            l_sin.start_offset() * sin.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let out = candle::MetalStorage::new(output, device.clone(), el, src.dtype());
        Ok((out, l_src.shape().clone()))
    }
}

pub fn rope<'w>(xs: &LiveTensor<'w>, cos: &Tensor, sin: &Tensor) -> Result<LiveTensor<'w>> {
    let (b_sz, _n_head, seq_len, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmb)
}

fn rotate_half(xs: &Tensor) -> Result<Tensor> {
    let last_dim = xs.dim(D::Minus1)?;
    let xs1 = xs.narrow(D::Minus1, 0, last_dim / 2)?;
    let xs2 = xs.narrow(D::Minus1, last_dim / 2, last_dim - last_dim / 2)?;
    Tensor::cat(&[&xs2.neg()?, &xs1], D::Minus1)
}

pub fn rope_slow(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b_sz, _h, seq_len, _n_embd) = x.dims4()?;
    let cos = Tensor::cat(&[cos, cos], D::Minus1)?;
    let sin = Tensor::cat(&[sin, sin], D::Minus1)?;
    let cos = cos.narrow(0, 0, seq_len)?;
    let sin = sin.narrow(0, 0, seq_len)?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
    x.broadcast_mul(&cos)? + rotate_half(x)?.broadcast_mul(&sin)?
}

/// T (seqlen)/H (num-heads)/D (head-dim) contiguous variant of rope embeddings.
#[derive(Debug, Clone)]
struct RotaryEmbThd;

impl candle::CustomOp3 for RotaryEmbThd {
    fn name(&self) -> &'static str {
        "rotary-emb"
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
        fn inner<T: candle::WithDType + num_traits::Float>(
            src: &[T],
            l_src: &Layout,
            cos: &[T],
            l_cos: &Layout,
            sin: &[T],
            l_sin: &Layout,
        ) -> Result<(CpuStorage, Shape)> {
            let src = match l_src.contiguous_offsets() {
                None => candle::bail!("input src has to be contiguous"),
                Some((o1, o2)) => &src[o1..o2],
            };
            let cos = match l_cos.contiguous_offsets() {
                None => candle::bail!("input cos has to be contiguous"),
                Some((o1, o2)) => &cos[o1..o2],
            };
            let sin = match l_sin.contiguous_offsets() {
                None => candle::bail!("input sin has to be contiguous"),
                Some((o1, o2)) => &sin[o1..o2],
            };
            let (b, t, h, d) = l_src.shape().dims4()?;
            let unbatched_rope = l_cos.dims().len() == 3 && l_sin.dims().len() == 3;
            let el_count = b * h * t * d;
            let mut dst = vec![T::zero(); el_count];
            src.par_chunks(t * h * d)
                .zip(dst.par_chunks_mut(t * h * d))
                .enumerate()
                .for_each(|(b_i, (src, dst))| {
                    for i_t in 0..t {
                        for i_d in 0..d / 2 {
                            let i_cs = i_t * (d / 2) + i_d;
                            let i_cs = if unbatched_rope {
                                i_cs + b_i * t * d / 2
                            } else {
                                i_cs
                            };
                            for i_h in 0..h {
                                let i1 = i_t * h * d + i_h * d + i_d;
                                let i2 = i1 + d / 2;
                                dst[i1] = src[i1] * cos[i_cs] - src[i2] * sin[i_cs];
                                dst[i2] = src[i1] * sin[i_cs] + src[i2] * cos[i_cs];
                            }
                        }
                    }
                });
            let storage = candle::WithDType::to_cpu_storage_owned(dst);
            Ok((storage, (b, t, h, d).into()))
        }

        use candle::backend::BackendStorage;
        use CpuStorage::{BF16, F16, F32, F64};
        match (s1, s2, s3) {
            (BF16(s1), BF16(s2), BF16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F16(s1), F16(s2), F16(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F32(s1), F32(s2), F32(s3)) => inner(s1, l1, s2, l2, s3, l3),
            (F64(s1), F64(s2), F64(s3)) => inner(s1, l1, s2, l2, s3, l3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
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
            None => candle::bail!("src input has to be contiguous"),
            Some(offsets) => offsets,
        };
        let (cos_o1, cos_o2) = match l2.contiguous_offsets() {
            None => candle::bail!("cos input has to be contiguous"),
            Some(offsets) => offsets,
        };
        let (sin_o1, sin_o2) = match l3.contiguous_offsets() {
            None => candle::bail!("sin input has to be contiguous"),
            Some(offsets) => offsets,
        };

        let (b, t, h, d) = l1.shape().dims4()?;
        let stride_b = if l2.dims().len() == 3 && l3.dims().len() == 3 {
            (h * t * d) as u32
        } else {
            0u32
        };
        let el = b * h * t * d;

        // Get dtype for FFI dispatcher
        let dtype = match &s1.slice {
            CudaStorageSlice::F32(_) => kernels::simple::reduce::FloatDType::F32 as i32,
            CudaStorageSlice::F64(_) => kernels::simple::reduce::FloatDType::F64 as i32,
            CudaStorageSlice::F16(_) => kernels::simple::reduce::FloatDType::F16 as i32,
            CudaStorageSlice::BF16(_) => kernels::simple::reduce::FloatDType::BF16 as i32,
            CudaStorageSlice::F8E4M3(_) => kernels::simple::reduce::FloatDType::F8E4M3 as i32,
            _ => candle::bail!("rope_thd not supported for dtype {:?}", s1.dtype()),
        };

        // The operand's arena: this op's output is allocated beside its input,

        // which is what makes the `'w` on the result true rather than merely

        // permitted. Declared before the dispatch macro because a `macro_rules!`

        // body resolves free identifiers at its definition site, not its call.

        let inherit = s1.backing;

        // Assigned by the macro to whatever `alloc_inheriting` resolved.

        let out_backing;

        macro_rules! rope_thd_impl {
            ($src_slice:expr, $cos_slice:expr, $sin_slice:expr, $dtype_variant:ident, $rust_type:ty) => {{
                let src = $src_slice.slice(src_o1..src_o2);
                let cos = $cos_slice.slice(cos_o1..cos_o2);
                let sin = $sin_slice.slice(sin_o1..sin_o2);
                let (dst, resolved_backing) = unsafe {
                    candle::cuda_backend::alloc_inheriting::<$rust_type>(dev, el, inherit)?
                };
                out_backing = resolved_backing;
                {
                    let (src_ptr, _src_guard) = src.device_ptr(&stream);
                    let (cos_ptr, _cos_guard) = cos.device_ptr(&stream);
                    let (sin_ptr, _sin_guard) = sin.device_ptr(&stream);
                    let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
                    #[cfg(feature = "cuda")]
                    candle::set_kernel_breadcrumb("run_rope_thd_op", file!(), line!());
                    unsafe {
                        kernels::simple::reduce::run_rope_thd_op(
                            dtype,
                            src_ptr as *const std::ffi::c_void,
                            cos_ptr as *const std::ffi::c_void,
                            sin_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            b as u32,
                            t as u32,
                            h as u32,
                            d as u32,
                            stride_b,
                        );
                    }
                }
                CudaStorageSlice::$dtype_variant(dst)
            }};
        }

        let slice = match (&s1.slice, &s2.slice, &s3.slice) {
            (
                CudaStorageSlice::BF16(src),
                CudaStorageSlice::BF16(cos),
                CudaStorageSlice::BF16(sin),
            ) => rope_thd_impl!(src, cos, sin, BF16, half::bf16),
            (
                CudaStorageSlice::F16(src),
                CudaStorageSlice::F16(cos),
                CudaStorageSlice::F16(sin),
            ) => rope_thd_impl!(src, cos, sin, F16, half::f16),
            (
                CudaStorageSlice::F32(src),
                CudaStorageSlice::F32(cos),
                CudaStorageSlice::F32(sin),
            ) => rope_thd_impl!(src, cos, sin, F32, f32),
            (
                CudaStorageSlice::F64(src),
                CudaStorageSlice::F64(cos),
                CudaStorageSlice::F64(sin),
            ) => rope_thd_impl!(src, cos, sin, F64, f64),
            (
                CudaStorageSlice::F8E4M3(src),
                CudaStorageSlice::F8E4M3(cos),
                CudaStorageSlice::F8E4M3(sin),
            ) => rope_thd_impl!(src, cos, sin, F8E4M3, float8::F8E4M3),
            _ => candle::bail!(
                "unsupported dtype for rope {:?} {:?} {:?}",
                s1.dtype(),
                s2.dtype(),
                s3.dtype()
            ),
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
        src: &candle::MetalStorage,
        l_src: &Layout,
        cos: &candle::MetalStorage,
        l_cos: &Layout,
        sin: &candle::MetalStorage,
        l_sin: &Layout,
    ) -> Result<(candle::MetalStorage, Shape)> {
        use candle::backend::BackendStorage;
        let device = src.device();
        let command_buffer = device.command_buffer()?;
        let kernels = device.kernels();
        if cos.dtype() != src.dtype() || sin.dtype() != src.dtype() {
            candle::bail!(
                "dtype mismatch in rope {:?} {:?} {:?}",
                src.dtype(),
                cos.dtype(),
                sin.dtype()
            )
        }
        let name = match src.dtype() {
            candle::DType::F32 => "rope_thd_f32",
            candle::DType::F16 => "rope_thd_f16",
            candle::DType::BF16 => "rope_thd_bf16",
            dtype => candle::bail!("rope_thd is not implemented for {dtype:?}"),
        };
        let (b, t, h, d) = l_src.shape().dims4()?;
        let stride_b = if l_cos.dims().len() == 3 && l_sin.dims().len() == 3 {
            h * t * d
        } else {
            0usize
        };
        let el = b * h * t * d;
        let output = device.new_buffer(el, src.dtype(), "rope-thd")?;
        candle_metal_kernels::call_rope_thd(
            device.metal_device(),
            &command_buffer,
            kernels,
            name,
            b,
            t,
            h,
            d,
            stride_b,
            src.buffer(),
            l_src.start_offset() * src.dtype().size_in_bytes(),
            cos.buffer(),
            l_cos.start_offset() * cos.dtype().size_in_bytes(),
            sin.buffer(),
            l_sin.start_offset() * sin.dtype().size_in_bytes(),
            &output,
        )
        .map_err(candle::Error::wrap)?;
        let out = candle::MetalStorage::new(output, device.clone(), el, src.dtype());
        Ok((out, l_src.shape().clone()))
    }
}

pub fn rope_thd<'w>(xs: &LiveTensor<'w>, cos: &Tensor, sin: &Tensor) -> Result<LiveTensor<'w>> {
    let (b_sz, seq_len, _n_head, n_embd) = xs.dims4()?;
    let (cos_seq_len, cos_n_embd) = rope_check_cs(cos, b_sz)?;
    let (sin_seq_len, sin_n_embd) = rope_check_cs(sin, b_sz)?;
    if cos_n_embd * 2 != n_embd
        || sin_n_embd * 2 != n_embd
        || seq_len > cos_seq_len
        || seq_len > sin_seq_len
    {
        candle::bail!(
            "inconsistent last dim size in rope {:?} {:?} {:?}",
            xs.shape(),
            cos.shape(),
            sin.shape()
        )
    }
    if !xs.is_contiguous() {
        candle::bail!("xs has to be contiguous in rope")
    }
    if !cos.is_contiguous() {
        candle::bail!("cos has to be contiguous in rope")
    }
    if !sin.is_contiguous() {
        candle::bail!("sin has to be contiguous in rope")
    }
    xs.apply_op3_no_bwd(cos, sin, &RotaryEmbThd)
}
