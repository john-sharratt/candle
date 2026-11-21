//! Provides the elementwise activation functions, including the fused SiLU-multiply
//! used by SwiGLU feed-forward blocks.
//!
//! The fused SiLU-multiply computes `silu(gate) * up` in one kernel, which saves a
//! launch and an intermediate tensor per call. Its inputs are required to be
//! contiguous, since they are fresh matmul outputs. The sigmoid is a custom op whose
//! output is allocated beside its input, so the output stays in the same memory arena
//! as the value it was computed from. The remaining activations are composed from
//! tensor operations.

use super::*;

pub fn silu<'w>(xs: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    xs.silu()
}

pub fn swiglu<'w>(xs: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
    let xs = xs.chunk(2, D::Minus1)?;
    &xs[0].silu()? * &xs[1]
}

pub(super) struct Sigmoid;

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
                            stream.cu_stream() as *mut std::ffi::c_void,
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

// Fused SiLU-Mul: out = silu(gate) * up
// Eliminates 1 kernel launch + 1 intermediate allocation per call.
// This is the core SwiGLU activation used in MoE expert FFNs.

#[derive(Debug, Clone)]
pub(super) struct SiluMul;

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
                            stream.cu_stream() as *mut std::ffi::c_void,
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
            anchor: None,
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
