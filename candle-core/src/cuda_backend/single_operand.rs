//! Runs the single-operand kernels (clone, affine, unary, reductions and casts),
//! each allocating its output through the provenance of its input.
//!
//! An output is allocated from the arena its operand came from, so a result
//! derived from a wave-backed input stays in that wave, and a fresh copy never
//! inherits the source's lease. A launch with no elements is skipped, since a
//! zero grid is an invalid configuration. A matrix product with a zero
//! dimension never reaches cuBLAS, which rejects it: with an empty inner
//! dimension the result is materialised as zeros, and with no output elements
//! nothing is computed. Non-contiguous inputs are passed with their dims and
//! strides and read in place.

use super::*;

pub(super) struct Clone;

impl Map1 for Clone {
    fn f<T: DeviceRepr>(
        &self,
        s: &CudaSlice<T>,
        _: &CudaDevice,
        _: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        // `CudaSlice::try_clone` allocates its destination, so it reaches the
        // driver without passing through `CudaDevice::alloc`.
        forbidden_alloc::record("CudaSlice::try_clone", s.len() * std::mem::size_of::<T>());
        // `try_clone` allocates its own destination from the pool, so the
        // copy owns its memory regardless of where the source came from.
        let _ = origin;
        Ok((s.try_clone().w()?, Backing::Owned))
    }
}

/// Execute affine transformation via direct FFI call (no PTX JIT)
/// `inherit` is the operand's arena; the resolved backing comes back alongside
/// the slice so the caller can stamp it rather than claiming ownership of what
/// may be a view into a wave span.
pub(super) fn run_affine_ffi(
    src: &CudaStorageSlice,
    dev: &CudaDevice,
    layout: &Layout,
    mul: f64,
    add: f64,
    inherit: Backing,
) -> Result<(CudaStorageSlice, Backing)> {
    let shape = layout.shape();
    let dims = shape.dims();
    let el = shape.elem_count();
    let start_offset = layout.start_offset();
    let stream = dev.cuda_stream();

    // Prepare dims/strides info for non-contiguous tensors
    let info: Option<Arc<Uploaded<usize>>> = if layout.is_contiguous() {
        None
    } else {
        Some(dev.info_table(&[dims, layout.stride()].concat())?)
    };
    let info_ptr = match &info {
        Some(s) => {
            let (ptr, _guard) = s.device_ptr(&stream);
            ptr as *const usize
        }
        None => std::ptr::null(),
    };

    // Get dtype for dispatcher
    let dtype = match src {
        CudaStorageSlice::F32(_) => DType::F32,
        CudaStorageSlice::F64(_) => DType::F64,
        CudaStorageSlice::F16(_) => DType::F16,
        CudaStorageSlice::BF16(_) => DType::BF16,
        CudaStorageSlice::F8E4M3(_) => DType::F8E4M3,
        CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        CudaStorageSlice::U8(_) => DType::U8,
        CudaStorageSlice::U32(_) => DType::U32,
        CudaStorageSlice::I64(_) => DType::I64,
    };
    let dtype_i32 = dtype_to_affine_dtype(dtype);

    // Assigned by the macro to whatever `alloc_inheriting` resolved.
    let out_backing;

    // Execute based on dtype - allocate output and call FFI
    macro_rules! affine_impl {
        ($slice:expr, $dtype_variant:ident) => {{
            let src_slice = $slice.slice(start_offset..);
            // SAFETY: Allocated memory will be initialized by the kernel
            let (out, resolved) = unsafe { alloc_inheriting(dev, el, inherit)? };
            out_backing = resolved;
            {
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (out_ptr, _out_guard) = out.device_ptr(&stream);
                // Keep info alive for the kernel call
                let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));
                cuda_breadcrumb!("run_affine");
                unsafe {
                    kernels::simple::affine::run_affine(
                        dtype_i32,
                        el,
                        dims.len(),
                        info_ptr,
                        src_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        mul,
                        add,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }
            // Guards dropped, safe to move out
            CudaStorageSlice::$dtype_variant(out)
        }};
    }

    let out = match src {
        CudaStorageSlice::F32(s) => affine_impl!(s, F32),
        CudaStorageSlice::F64(s) => affine_impl!(s, F64),
        CudaStorageSlice::F16(s) => affine_impl!(s, F16),
        CudaStorageSlice::BF16(s) => affine_impl!(s, BF16),
        CudaStorageSlice::F8E4M3(s) => affine_impl!(s, F8E4M3),
        CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        CudaStorageSlice::U8(s) => affine_impl!(s, U8),
        CudaStorageSlice::U32(s) => affine_impl!(s, U32),
        CudaStorageSlice::I64(s) => affine_impl!(s, I64),
    };

    Ok((out, out_backing))
}

/// Execute parametric unary operation (Elu/Powf) via direct FFI call
pub(super) fn run_unary_param_ffi(
    src: &CudaStorageSlice,
    dev: &CudaDevice,
    layout: &Layout,
    op: i32,
    param: f64,
) -> Result<CudaStorageSlice> {
    let shape = layout.shape();
    let dims = shape.dims();
    let el = shape.elem_count();
    let start_offset = layout.start_offset();
    let stream = dev.cuda_stream();

    // Prepare dims/strides info for non-contiguous tensors
    let info: Option<Arc<Uploaded<usize>>> = if layout.is_contiguous() {
        None
    } else {
        Some(dev.info_table(&[dims, layout.stride()].concat())?)
    };
    let info_ptr = match &info {
        Some(s) => {
            let (ptr, _guard) = s.device_ptr(&stream);
            ptr as *const usize
        }
        None => std::ptr::null(),
    };

    // Get dtype for dispatcher
    let dtype = match src {
        CudaStorageSlice::F32(_) => DType::F32,
        CudaStorageSlice::F64(_) => DType::F64,
        CudaStorageSlice::F16(_) => DType::F16,
        CudaStorageSlice::BF16(_) => DType::BF16,
        CudaStorageSlice::F8E4M3(_) => DType::F8E4M3,
        CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        CudaStorageSlice::U8(_) => DType::U8,
        CudaStorageSlice::U32(_) => DType::U32,
        CudaStorageSlice::I64(_) => DType::I64,
    };
    let dtype_i32 = dtype_to_unary_dtype(dtype);

    // Execute based on dtype - allocate output and call FFI
    macro_rules! unary_param_impl {
        ($slice:expr, $dtype_variant:ident) => {{
            let src_slice = $slice.slice(start_offset..);
            // SAFETY: Allocated memory will be initialized by the kernel
            let out = unsafe { dev.alloc(el)? };
            {
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (out_ptr, _out_guard) = out.device_ptr(&stream);
                // Keep info alive for the kernel call
                let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));
                cuda_breadcrumb!("run_unary_param_op");
                unsafe {
                    kernels::simple::unary::run_unary_param_op(
                        op,
                        dtype_i32,
                        param as f32,
                        el,
                        dims.len(),
                        info_ptr,
                        src_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }
            // Guards dropped, safe to move out
            CudaStorageSlice::$dtype_variant(out)
        }};
    }

    let out = match src {
        CudaStorageSlice::F32(s) => unary_param_impl!(s, F32),
        CudaStorageSlice::F64(s) => unary_param_impl!(s, F64),
        CudaStorageSlice::F16(s) => unary_param_impl!(s, F16),
        CudaStorageSlice::BF16(s) => unary_param_impl!(s, BF16),
        CudaStorageSlice::F8E4M3(s) => unary_param_impl!(s, F8E4M3),
        CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        _ => crate::bail!("Parametric unary ops only support float types"),
    };

    Ok(out)
}

pub(super) struct FastReduce<'a>(pub(super) &'a [usize], pub(super) ReduceOp);

impl Map1Any for FastReduce<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits, W: Fn(CudaSlice<T>) -> S>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
        wrap: W,
        origin: Backing,
    ) -> Result<OutS> {
        cuda_breadcrumb!("run_fast_reduce");
        let src_stride = layout.stride();
        let src_dims = layout.shape().dims();
        let src_el: usize = src_dims.iter().product();
        // Source dims and strides with the sum dims at the end.
        let mut dims = vec![];
        let mut stride = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !self.0.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(src_stride[dim_idx]);
            }
        }
        for &dim_idx in self.0.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(src_stride[dim_idx]);
        }
        let (check_empty, return_index) = match self.1 {
            ReduceOp::Sum => (false, false),
            ReduceOp::Min => (true, false),
            ReduceOp::Max => (true, false),
            ReduceOp::ArgMin => (true, true),
            ReduceOp::ArgMax => (true, true),
        };
        if check_empty && layout.shape().elem_count() == 0 {
            Err(crate::Error::EmptyTensor { op: "reduce" }.bt())?
        }
        // **An empty sum is settled here, not launched.** Over no output
        // elements there is nothing to write, and over an empty axis each sum
        // is zero; either way the kernel's own sizing divides by a count of
        // none, and a zero-block grid is a launch error left pending on the
        // thread.
        if src_el == 0 {
            // SAFETY: zeroed below before it is read; empty, it is never read.
            let (mut out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
            if dst_el > 0 {
                dev.cuda_stream().memset_zeros(&mut out).w()?;
            }
            return Ok((wrap(out), out_backing));
        }
        let el_to_sum_per_block = src_el / dst_el;

        // Get dtype for FFI dispatcher
        let dtype = T::DTYPE;
        let dtype_i32 = dtype_to_fast_reduce_dtype(dtype);

        let stream = dev.cuda_stream();
        let ds = dev.info_table(&[dims.as_slice(), stride.as_slice()].concat())?;
        let src = &src.slice(layout.start_offset()..);

        if return_index {
            use kernels::simple::reduce::FastArgReduceOp;
            let op = match self.1 {
                ReduceOp::ArgMin => FastArgReduceOp::ArgMin as i32,
                ReduceOp::ArgMax => FastArgReduceOp::ArgMax as i32,
                _ => unreachable!(),
            };
            // SAFETY: filled in by the follow up kernel.
            let (out, out_backing) = unsafe { alloc_inheriting::<u32>(dev, dst_el, origin)? };
            {
                let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
                let (src_ptr, _src_guard) = src.device_ptr(&stream);
                let (out_ptr, _out_guard) = out.device_ptr(&stream);

                unsafe {
                    kernels::simple::reduce::run_fast_arg_reduce_op(
                        op,
                        dtype_i32,
                        src_el,
                        el_to_sum_per_block,
                        src_dims.len(),
                        ds_ptr as *const usize,
                        src_ptr as *const std::ffi::c_void,
                        out_ptr as *mut u32,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }
            Ok((S::U32(out), out_backing))
        } else {
            use kernels::simple::reduce::FastReduceOp;
            let op = match self.1 {
                ReduceOp::Sum => FastReduceOp::Sum as i32,
                ReduceOp::Min => FastReduceOp::Min as i32,
                ReduceOp::Max => FastReduceOp::Max as i32,
                _ => unreachable!(),
            };
            // SAFETY: filled in by the follow up kernel.
            let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
            {
                let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
                let (src_ptr, _src_guard) = src.device_ptr(&stream);
                let (out_ptr, _out_guard) = out.device_ptr(&stream);

                unsafe {
                    kernels::simple::reduce::run_fast_reduce_op(
                        op,
                        dtype_i32,
                        src_el,
                        el_to_sum_per_block,
                        src_dims.len(),
                        ds_ptr as *const usize,
                        src_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }
            Ok((wrap(out), out_backing))
        }
    }
}

impl<U: UnaryOpT> Map1 for U {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        layout: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let start_offset = layout.start_offset();
        let stream = dev.cuda_stream();
        cuda_breadcrumb!(U::KERNEL);

        // Try to use FFI dispatcher first
        if let Some(op) = kernel_name_to_unary_op(U::KERNEL) {
            let dtype = T::DTYPE;
            let dtype_i32 = dtype_to_unary_dtype(dtype);

            // Prepare dims/strides info for non-contiguous tensors
            let info: Option<Arc<Uploaded<usize>>> = if layout.is_contiguous() {
                None
            } else {
                Some(dev.info_table(&[dims, layout.stride()].concat())?)
            };

            let src_slice = &src.slice(start_offset..);
            // SAFETY: Allocated memory will be initialized by the kernel
            let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, el_count, origin)? };
            {
                let info_ptr = match &info {
                    Some(s) => {
                        let (ptr, _guard) = s.device_ptr(&stream);
                        ptr as *const usize
                    }
                    None => std::ptr::null(),
                };
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (out_ptr, _out_guard) = out.device_ptr(&stream);

                // Keep info alive for the kernel call
                let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));

                // No elements, no launch: a zero grid is an invalid configuration
                // left pending on the thread for an unrelated caller to find.
                if el_count > 0 {
                    unsafe {
                        kernels::simple::unary::run_unary_op(
                            op,
                            dtype_i32,
                            el_count,
                            dims.len(),
                            info_ptr,
                            src_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                }
            }
            return Ok((out, out_backing));
        }

        // All unary operations should be handled by FFI dispatcher above
        Err(CudaError::InternalError(format!(
            "Unrecognized unary kernel '{}' - all operations should use FFI dispatcher",
            U::KERNEL
        )))
        .w()
    }
}

impl BackendStorage for CudaStorage {
    type Device = CudaDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let (slice, out_backing) = Clone.map(self, self.device(), layout)?;
        let device = self.device.clone();
        // A copy into fresh memory: nothing of the source's lease comes with it.
        Ok(Self {
            slice,
            device,
            backing: out_backing,
            anchor: None,
        })
    }

    fn dtype(&self) -> DType {
        match self.slice {
            CudaStorageSlice::U8(_) => DType::U8,
            CudaStorageSlice::U32(_) => DType::U32,
            CudaStorageSlice::I64(_) => DType::I64,
            CudaStorageSlice::BF16(_) => DType::BF16,
            CudaStorageSlice::F16(_) => DType::F16,
            CudaStorageSlice::F32(_) => DType::F32,
            CudaStorageSlice::F64(_) => DType::F64,
            CudaStorageSlice::F8E4M3(_) => DType::F8E4M3,
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        }
    }

    fn device(&self) -> &CudaDevice {
        &self.device
    }

    fn const_set(&mut self, s: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        use crate::scalar::Scalar;
        use kernels::simple::fill::FillDType;

        let dev = &self.device;
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        let src_o = layout.start_offset();
        let stream = dev.cuda_stream();

        // Prepare dims/strides info for non-contiguous tensors
        let info: Option<Arc<Uploaded<usize>>> = if layout.is_contiguous() {
            None
        } else {
            Some(dev.info_table(&[dims, layout.stride()].concat())?)
        };
        let info_ptr = match &info {
            Some(s) => {
                let (ptr, _guard) = s.device_ptr(&stream);
                ptr as *const usize
            }
            None => std::ptr::null(),
        };

        // Convert scalar to bits
        let value_bits = match s {
            Scalar::U8(v) => v as u64,
            Scalar::U32(v) => v as u64,
            Scalar::I64(v) => v as u64,
            Scalar::F32(v) => v.to_bits() as u64,
            Scalar::F64(v) => v.to_bits(),
            Scalar::F16(v) => v.to_bits() as u64,
            Scalar::BF16(v) => v.to_bits() as u64,
            Scalar::F8E4M3(v) => v.to_bits() as u64,
        };

        // Get dtype and output pointer
        let (dtype_i32, out_ptr): (i32, u64) = match &mut self.slice {
            S::U8(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::U8 as i32, ptr)
            }
            S::U32(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::U32 as i32, ptr)
            }
            S::I64(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::I64 as i32, ptr)
            }
            S::BF16(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::BF16 as i32, ptr)
            }
            S::F16(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::F16 as i32, ptr)
            }
            S::F32(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::F32 as i32, ptr)
            }
            S::F64(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::F64 as i32, ptr)
            }
            S::F8E4M3(slice) => {
                let (ptr, _) = slice.slice(src_o..).device_ptr(&stream);
                (FillDType::F8E4M3 as i32, ptr)
            }
            S::Moved => S::unreachable_moved(),
        };

        // Keep info alive for the kernel call
        let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));

        unsafe {
            kernels::simple::fill::run_const_set_op(
                dtype_i32,
                el_count,
                dims.len(),
                info_ptr,
                value_bits,
                out_ptr as *mut std::ffi::c_void,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
        Ok(())
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        use kernels::simple::cast::CastDType;

        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let dev = self.device();
        let start_o = layout.start_offset();
        let stream = dev.cuda_stream();

        // Convert candle DType to CastDType
        let dtype_to_cast = |dt: DType| -> i32 {
            match dt {
                DType::F32 => CastDType::F32 as i32,
                DType::F64 => CastDType::F64 as i32,
                DType::U8 => CastDType::U8 as i32,
                DType::U32 => CastDType::U32 as i32,
                DType::I64 => CastDType::I64 as i32,
                DType::F16 => CastDType::F16 as i32,
                DType::BF16 => CastDType::BF16 as i32,
                DType::F8E4M3 => CastDType::F8E4M3 as i32,
            }
        };

        let src_dtype = self.dtype();
        let src_dtype_i32 = dtype_to_cast(src_dtype);
        let dst_dtype_i32 = dtype_to_cast(dtype);

        // Prepare dims/strides info for non-contiguous tensors
        let info: Option<Arc<Uploaded<usize>>> = if layout.is_contiguous() {
            None
        } else {
            Some(dev.info_table(&[dims, layout.stride()].concat())?)
        };

        // Use a helper macro to reduce repetition and properly scope the guards
        // Set by `cast_impl!` to whatever `alloc_inheriting` resolved, so the
        // result storage is stamped with the backing of the buffer it actually
        // got rather than an assumed one. Left uninitialised so the compiler
        // checks every arm assigns it — an arm that allocated without recording
        // its backing is exactly the drift this pairing exists to prevent.
        let out_backing;
        macro_rules! cast_impl {
            ($inp_slice:expr, $out_ty:ty, $wrapper:path) => {{
                let inp = $inp_slice.slice(start_o..);
                // A cast reading a wave-backed operand writes into the same
                // generation: the result is consumed within the phase that
                // produced its input, which is why inheriting is right here and
                // not merely permissible.
                let (out, backing) = unsafe { alloc_inheriting::<$out_ty>(dev, el, self.backing)? };
                out_backing = backing;
                {
                    let info_ptr = match &info {
                        Some(s) => {
                            let (ptr, _guard) = s.device_ptr(&stream);
                            ptr as *const usize
                        }
                        None => std::ptr::null(),
                    };
                    let (inp_ptr, _inp_guard) = inp.device_ptr(&stream);
                    let (out_ptr, _out_guard) = out.device_ptr(&stream);
                    // Keep info alive
                    let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));
                    // An empty tensor casts to an empty tensor without a launch:
                    // a zero grid is an invalid configuration, left pending on the
                    // thread for an unrelated caller to find.
                    if el > 0 {
                        unsafe {
                            kernels::simple::cast::run_cast(
                                src_dtype_i32,
                                dst_dtype_i32,
                                el,
                                dims.len(),
                                info_ptr,
                                inp_ptr as *const std::ffi::c_void,
                                out_ptr as *mut std::ffi::c_void,
                                stream.cu_stream() as *mut std::ffi::c_void,
                            );
                        }
                    }
                }
                $wrapper(out)
            }};
        }

        let slice = match (&self.slice, dtype) {
            (CudaStorageSlice::U8(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::U8(inp), DType::U32) => cast_impl!(inp, u32, CudaStorageSlice::U32),
            (CudaStorageSlice::U8(inp), DType::I64) => cast_impl!(inp, i64, CudaStorageSlice::I64),
            (CudaStorageSlice::U8(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::U8(inp), DType::F16) => cast_impl!(inp, f16, CudaStorageSlice::F16),
            (CudaStorageSlice::U8(inp), DType::F32) => cast_impl!(inp, f32, CudaStorageSlice::F32),
            (CudaStorageSlice::U8(inp), DType::F64) => cast_impl!(inp, f64, CudaStorageSlice::F64),
            (CudaStorageSlice::U8(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }

            (CudaStorageSlice::U32(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::U32(inp), DType::U32) => cast_impl!(inp, u32, CudaStorageSlice::U32),
            (CudaStorageSlice::U32(inp), DType::I64) => cast_impl!(inp, i64, CudaStorageSlice::I64),
            (CudaStorageSlice::U32(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::U32(inp), DType::F16) => cast_impl!(inp, f16, CudaStorageSlice::F16),
            (CudaStorageSlice::U32(inp), DType::F32) => cast_impl!(inp, f32, CudaStorageSlice::F32),
            (CudaStorageSlice::U32(inp), DType::F64) => cast_impl!(inp, f64, CudaStorageSlice::F64),
            (CudaStorageSlice::U32(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }

            (CudaStorageSlice::I64(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::I64(inp), DType::U32) => cast_impl!(inp, u32, CudaStorageSlice::U32),
            (CudaStorageSlice::I64(inp), DType::I64) => cast_impl!(inp, i64, CudaStorageSlice::I64),
            (CudaStorageSlice::I64(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::I64(inp), DType::F16) => cast_impl!(inp, f16, CudaStorageSlice::F16),
            (CudaStorageSlice::I64(inp), DType::F32) => cast_impl!(inp, f32, CudaStorageSlice::F32),
            (CudaStorageSlice::I64(inp), DType::F64) => cast_impl!(inp, f64, CudaStorageSlice::F64),
            (CudaStorageSlice::I64(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }

            (CudaStorageSlice::BF16(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::BF16(inp), DType::U32) => {
                cast_impl!(inp, u32, CudaStorageSlice::U32)
            }
            (CudaStorageSlice::BF16(inp), DType::I64) => {
                cast_impl!(inp, i64, CudaStorageSlice::I64)
            }
            (CudaStorageSlice::BF16(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::BF16(inp), DType::F16) => {
                cast_impl!(inp, f16, CudaStorageSlice::F16)
            }
            (CudaStorageSlice::BF16(inp), DType::F32) => {
                cast_impl!(inp, f32, CudaStorageSlice::F32)
            }
            (CudaStorageSlice::BF16(inp), DType::F64) => {
                cast_impl!(inp, f64, CudaStorageSlice::F64)
            }
            (CudaStorageSlice::BF16(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }

            (CudaStorageSlice::F16(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::F16(inp), DType::U32) => cast_impl!(inp, u32, CudaStorageSlice::U32),
            (CudaStorageSlice::F16(inp), DType::I64) => cast_impl!(inp, i64, CudaStorageSlice::I64),
            (CudaStorageSlice::F16(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::F16(inp), DType::F16) => cast_impl!(inp, f16, CudaStorageSlice::F16),
            (CudaStorageSlice::F16(inp), DType::F32) => cast_impl!(inp, f32, CudaStorageSlice::F32),
            (CudaStorageSlice::F16(inp), DType::F64) => cast_impl!(inp, f64, CudaStorageSlice::F64),
            (CudaStorageSlice::F16(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }

            (CudaStorageSlice::F32(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::F32(inp), DType::U32) => cast_impl!(inp, u32, CudaStorageSlice::U32),
            (CudaStorageSlice::F32(inp), DType::I64) => cast_impl!(inp, i64, CudaStorageSlice::I64),
            (CudaStorageSlice::F32(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::F32(inp), DType::F16) => cast_impl!(inp, f16, CudaStorageSlice::F16),
            (CudaStorageSlice::F32(inp), DType::F32) => cast_impl!(inp, f32, CudaStorageSlice::F32),
            (CudaStorageSlice::F32(inp), DType::F64) => cast_impl!(inp, f64, CudaStorageSlice::F64),
            (CudaStorageSlice::F32(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }

            (CudaStorageSlice::F64(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::F64(inp), DType::U32) => cast_impl!(inp, u32, CudaStorageSlice::U32),
            (CudaStorageSlice::F64(inp), DType::I64) => cast_impl!(inp, i64, CudaStorageSlice::I64),
            (CudaStorageSlice::F64(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::F64(inp), DType::F16) => cast_impl!(inp, f16, CudaStorageSlice::F16),
            (CudaStorageSlice::F64(inp), DType::F32) => cast_impl!(inp, f32, CudaStorageSlice::F32),
            (CudaStorageSlice::F64(inp), DType::F64) => cast_impl!(inp, f64, CudaStorageSlice::F64),
            (CudaStorageSlice::F64(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }

            (CudaStorageSlice::F8E4M3(inp), DType::U8) => cast_impl!(inp, u8, CudaStorageSlice::U8),
            (CudaStorageSlice::F8E4M3(inp), DType::U32) => {
                cast_impl!(inp, u32, CudaStorageSlice::U32)
            }
            (CudaStorageSlice::F8E4M3(inp), DType::I64) => {
                cast_impl!(inp, i64, CudaStorageSlice::I64)
            }
            (CudaStorageSlice::F8E4M3(inp), DType::BF16) => {
                cast_impl!(inp, bf16, CudaStorageSlice::BF16)
            }
            (CudaStorageSlice::F8E4M3(inp), DType::F16) => {
                cast_impl!(inp, f16, CudaStorageSlice::F16)
            }
            (CudaStorageSlice::F8E4M3(inp), DType::F32) => {
                cast_impl!(inp, f32, CudaStorageSlice::F32)
            }
            (CudaStorageSlice::F8E4M3(inp), DType::F64) => {
                cast_impl!(inp, f64, CudaStorageSlice::F64)
            }
            (CudaStorageSlice::F8E4M3(inp), DType::F8E4M3) => {
                cast_impl!(inp, F8E4M3, CudaStorageSlice::F8E4M3)
            }
            (CudaStorageSlice::Moved, _) => CudaStorageSlice::unreachable_moved(),
        };
        Ok(Self {
            slice,
            device: dev.clone(),
            backing: out_backing,
            anchor: None,
        })
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device().clone();
        let (slice, backing) =
            run_affine_ffi(&self.slice, &device, layout, mul, add, self.backing)?;
        Ok(Self {
            slice,
            device,
            backing,
            anchor: None,
        })
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        use kernels::simple::unary::UnaryParamOp;
        let device = self.device().clone();
        let slice =
            run_unary_param_ffi(&self.slice, &device, layout, UnaryParamOp::Powf as i32, e)?;
        Ok(Self {
            slice,
            device,
            backing: Backing::Owned,
            anchor: None,
        })
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        use kernels::simple::unary::UnaryParamOp;
        let device = self.device().clone();
        let slice = run_unary_param_ffi(
            &self.slice,
            &device,
            layout,
            UnaryParamOp::Elu as i32,
            alpha,
        )?;
        Ok(Self {
            slice,
            device,
            backing: Backing::Owned,
            anchor: None,
        })
    }

    fn sub_at_indices(&self, layout: &Layout, indices: &[u32], value: f32) -> Result<Self> {
        // Delegate to the inherent method
        self.sub_at_indices(layout, indices, value)
    }

    fn div_at_indices(&self, layout: &Layout, indices: &[u32], value: f32) -> Result<Self> {
        // Delegate to the inherent method
        self.div_at_indices(layout, indices, value)
    }

    fn reduce_op(&self, op: ReduceOp, layout: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = FastReduce(sum_dims, op).map(self, &device, layout)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
            anchor: None,
        })
    }

    fn cmp(&self, op: CmpOp, rhs: &Self, lhs_l: &Layout, rhs_l: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = Cmp(op).map(self, lhs_l, rhs, rhs_l, &device)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
            anchor: None,
        })
    }

    fn unary_impl<U: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = U::V.map(self, &device, layout)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
            anchor: None,
        })
    }

    fn binary_impl<B: BinaryOpT>(
        &self,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = B::V.map(self, lhs_l, rhs, rhs_l, &device)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
            anchor: None,
        })
    }

    fn binary_inplace_impl(
        &mut self,
        op: crate::op::BinaryInplaceOp,
        rhs: &Self,
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<()> {
        use cudarc::driver::DevicePtr;

        // lhs must be contiguous for in-place operations
        if !lhs_l.is_contiguous() {
            return Err(CudaError::InternalError(
                "in-place binary op requires contiguous lhs tensor".to_string(),
            ))
            .w();
        }

        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let lhs_start = lhs_l.start_offset();
        let rhs_start = rhs_l.start_offset();
        let device = &self.device;
        let stream = device.cuda_stream();

        let dtype = self.dtype();
        let dtype_i32 = dtype_to_binary_dtype(dtype);
        let op_i32 = binary_inplace_op_to_ffi(op);

        // Check for min/max on integer types (not supported)
        let is_integer = matches!(dtype, DType::U8 | DType::U32 | DType::I64);
        if is_integer
            && matches!(
                op,
                crate::op::BinaryInplaceOp::Min | crate::op::BinaryInplaceOp::Max
            )
        {
            return Err(CudaError::InternalError(format!(
                "min/max in-place ops not supported for integer dtype {:?}",
                dtype
            )))
            .w();
        }

        // Prepare dims and strides info for non-contiguous rhs
        // Note: lhs is guaranteed contiguous, but rhs may not be
        let info: Option<Arc<Uploaded<usize>>> = if rhs_l.is_contiguous() {
            None
        } else {
            // Only need dims and rhs_strides since lhs is contiguous
            Some(device.info_table(&[dims, lhs_l.stride(), rhs_l.stride()].concat())?)
        };

        // Get lhs slice with offset for mutable access
        let lhs_ptr = self.slice.device_ptr_mut(&stream)?;
        let lhs_ptr_offset = match dtype {
            DType::F32 => unsafe { (lhs_ptr as *mut f32).add(lhs_start) as *mut std::ffi::c_void },
            DType::F64 => unsafe { (lhs_ptr as *mut f64).add(lhs_start) as *mut std::ffi::c_void },
            DType::U8 => unsafe { (lhs_ptr as *mut u8).add(lhs_start) as *mut std::ffi::c_void },
            DType::U32 => unsafe { (lhs_ptr as *mut u32).add(lhs_start) as *mut std::ffi::c_void },
            DType::I64 => unsafe { (lhs_ptr as *mut i64).add(lhs_start) as *mut std::ffi::c_void },
            DType::F16 => unsafe {
                (lhs_ptr as *mut half::f16).add(lhs_start) as *mut std::ffi::c_void
            },
            DType::BF16 => unsafe {
                (lhs_ptr as *mut half::bf16).add(lhs_start) as *mut std::ffi::c_void
            },
            DType::F8E4M3 => unsafe {
                (lhs_ptr as *mut float8::F8E4M3).add(lhs_start) as *mut std::ffi::c_void
            },
        };

        // Get rhs base pointer and offset it
        // We get the base pointer and manually offset it to avoid lifetime issues
        let rhs_dtype = rhs.dtype();
        let (rhs_base_ptr, _rhs_guard): (*const std::ffi::c_void, _) = match &rhs.slice {
            CudaStorageSlice::F32(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::F64(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::U8(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::U32(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::I64(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::F16(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::BF16(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::F8E4M3(s) => {
                let (ptr, guard) = s.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, guard)
            }
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        };

        // Offset rhs pointer
        let rhs_ptr = match rhs_dtype {
            DType::F32 => unsafe {
                (rhs_base_ptr as *const f32).add(rhs_start) as *const std::ffi::c_void
            },
            DType::F64 => unsafe {
                (rhs_base_ptr as *const f64).add(rhs_start) as *const std::ffi::c_void
            },
            DType::U8 => unsafe {
                (rhs_base_ptr as *const u8).add(rhs_start) as *const std::ffi::c_void
            },
            DType::U32 => unsafe {
                (rhs_base_ptr as *const u32).add(rhs_start) as *const std::ffi::c_void
            },
            DType::I64 => unsafe {
                (rhs_base_ptr as *const i64).add(rhs_start) as *const std::ffi::c_void
            },
            DType::F16 => unsafe {
                (rhs_base_ptr as *const half::f16).add(rhs_start) as *const std::ffi::c_void
            },
            DType::BF16 => unsafe {
                (rhs_base_ptr as *const half::bf16).add(rhs_start) as *const std::ffi::c_void
            },
            DType::F8E4M3 => unsafe {
                (rhs_base_ptr as *const float8::F8E4M3).add(rhs_start) as *const std::ffi::c_void
            },
        };

        {
            let info_ptr = match &info {
                Some(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    ptr as *const usize
                }
                None => std::ptr::null(),
            };

            // Keep info guard alive for the kernel call
            let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));

            // Call the FFI dispatcher - guards are still alive here
            unsafe {
                kernels::simple::binary::run_binary_inplace_op(
                    op_i32,
                    dtype_i32,
                    elem_count,
                    dims.len(),
                    info_ptr,
                    lhs_ptr_offset,
                    rhs_ptr,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }

        Ok(())
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        // A readback must see everything recorded before it.
        let _eager = self.device.pause_capture()?;
        match &self.slice {
            CudaStorageSlice::U8(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::U8(cpu_storage))
            }
            CudaStorageSlice::U32(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::U32(cpu_storage))
            }
            CudaStorageSlice::I64(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::I64(cpu_storage))
            }
            CudaStorageSlice::BF16(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::BF16(cpu_storage))
            }
            CudaStorageSlice::F16(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::F16(cpu_storage))
            }
            CudaStorageSlice::F32(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::F32(cpu_storage))
            }
            CudaStorageSlice::F64(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::F64(cpu_storage))
            }
            CudaStorageSlice::F8E4M3(slice) => {
                let cpu_storage = slice.stream().memcpy_dtov(slice).w()?;
                Ok(CpuStorage::F8E4M3(cpu_storage))
            }
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        }
    }

    fn where_cond(
        &self,
        layout: &Layout,
        t: &Self,
        t_l: &Layout,
        f: &Self,
        f_l: &Layout,
    ) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = WhereCond(self, layout).map(t, t_l, f, f_l, &device)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
            anchor: None,
        })
    }

    #[cfg(not(feature = "cudnn"))]
    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        const USE_IM2COL_CONV1D: bool = true;

        let device = self.device().clone();
        if !USE_IM2COL_CONV1D {
            let (slice, out_backing) = Conv1D(params).map(self, l, kernel, kernel_l, &device)?;
            return Ok(Self {
                slice,
                device,
                backing: out_backing,
                anchor: None,
            });
        }

        let (col, col_backing) = Im2Col1D {
            l_k: params.k_size,
            stride: params.stride,
            dilation: params.dilation,
            padding: params.padding,
        }
        .map(self, &device, l)?;
        let col = Self {
            slice: col,
            device,
            backing: col_backing,
            anchor: None,
        };
        let l_out = params.l_out();
        let b = params.b_size;
        let n = params.c_out;
        let k = params.k_size * params.c_in;
        let m = l_out;
        let col_l = Layout::contiguous((b * m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l =
                Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
            col.matmul(kernel, (1, b * m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            let mut kernel_c = unsafe {
                self.device()
                    .alloc_uninit(kernel_l.shape(), kernel.dtype())?
            };
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l = Layout::contiguous((n, k)).transpose(0, 1)?;
            col.matmul(&kernel_c, (1, b * m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, l_out, n)).transpose(1, 2)?;
        let mut res_t = unsafe { self.device().alloc_uninit(res_l.shape(), res.dtype())? };
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    #[cfg(feature = "cudnn")]
    fn conv1d(
        &self,
        inp_l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        let device = self.device().clone();
        if !kernel_l.is_contiguous() {
            let (slice, out_backing) =
                Conv1D(params).map(self, inp_l, kernel, kernel_l, &device)?;
            return Ok(Self {
                slice,
                device,
                backing: out_backing,
                anchor: None,
            });
        }
        let l_out = params.l_out();
        let dst_el = params.c_out * l_out * params.b_size;
        let slice = match (&self.slice, &kernel.slice) {
            (S::U8(inp), S::U8(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<u8>(dst_el)? };
                crate::cudnn::launch_conv1d::<u8, u8>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::U8(out)
            }
            (S::BF16(inp), S::BF16(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<bf16>(dst_el)? };
                // Only PSEUDO_BFLOAT16_CONFIG is supported in cudnn, there is no "true bfloat16"
                // version.
                // https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html#id88
                crate::cudnn::launch_conv1d::<bf16, f32>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::BF16(out)
            }
            (S::F16(inp), S::F16(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<f16>(dst_el)? };
                crate::cudnn::launch_conv1d::<f16, f16>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::F16(out)
            }
            (S::F32(inp), S::F32(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<f32>(dst_el)? };
                crate::cudnn::launch_conv1d::<f32, f32>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::F32(out)
            }
            (S::F64(inp), S::F64(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let mut out = unsafe { device.alloc::<f64>(dst_el)? };
                crate::cudnn::launch_conv1d::<f64, f64>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                S::F64(out)
            }
            (S::U32(_), S::U32(_)) => Err(CudaError::InternalError(
                "conv1d does not support u32".to_string(),
            ))?,
            (S::I64(_), S::I64(_)) => Err(CudaError::InternalError(
                "conv1d does not support i64".to_string(),
            ))?,
            _ => Err(CudaError::InternalError(
                "dtype mismatch in conv1d".to_string(),
            ))?,
        };
        Ok(Self {
            slice,
            device,
            backing: Backing::Owned,
            anchor: None,
        })
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        const USE_COL2IM_CONV1D_TR: bool = true;

        let device = self.device().clone();
        let can_use_col2im = kernel_l.is_contiguous()
            && params.dilation == 1
            && params.padding == 0
            && params.output_padding == 0;
        let slice = if USE_COL2IM_CONV1D_TR && can_use_col2im {
            let (b_size, c_in, l_in) = l.shape().dims3()?;
            let (c_in2, c_out, k_size) = kernel_l.shape().dims3()?;
            if !kernel_l.is_contiguous() {
                crate::bail!(
                    "convtr1d: the second argument (kernel) has to be contiguous {kernel_l:?}"
                )
            }
            if c_in != c_in2 {
                crate::bail!(
                    "convtr1d: shape mismatch on c_in {:?} {:?}",
                    l.shape(),
                    kernel_l.shape()
                )
            }
            let col = {
                // This merges the last two dimensions of the kernel together.
                let kernel_l_mm = Layout::new(
                    (b_size, c_in, k_size * c_out).into(),
                    vec![0, k_size * c_out, 1],
                    kernel_l.start_offset(),
                );
                self.matmul(
                    kernel,
                    (
                        b_size,
                        /* m */ l_in,
                        /* n */ c_out * k_size,
                        /* k */ c_in,
                    ),
                    &l.transpose(1, 2)?,
                    &kernel_l_mm,
                )?
            };
            let col_l = Layout::contiguous((b_size, l_in, c_out, k_size));
            Col2Im1D {
                stride: params.stride,
            }
            .map(&col, &device, &col_l)?
        } else {
            ConvTranspose1D(params).map(self, l, kernel, kernel_l, &device)?
        };
        Ok(Self {
            slice: slice.0,
            device,
            backing: slice.1,
            anchor: None,
        })
    }

    #[cfg(not(feature = "cudnn"))]
    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        const USE_IM2COL_CONV2D: bool = true;

        let device = self.device().clone();
        if !USE_IM2COL_CONV2D {
            let (slice, out_backing) = Conv2D(params).map(self, l, kernel, kernel_l, &device)?;
            return Ok(Self {
                slice,
                device,
                backing: out_backing,
                anchor: None,
            });
        }

        let (col, col_backing) = Im2Col {
            h_k: params.k_h,
            w_k: params.k_w,
            stride: params.stride,
            dilation: params.dilation,
            padding: params.padding,
        }
        .map(self, &device, l)?;
        let col = Self {
            slice: col,
            device,
            backing: col_backing,
            anchor: None,
        };
        let h_out = params.out_h();
        let w_out = params.out_w();
        let b = params.b_size;
        let n = params.c_out;
        let k = params.k_h * params.k_w * params.c_in;
        let m = h_out * w_out;
        let col_l = Layout::contiguous((b * m, k));
        let res = if kernel_l.is_contiguous() {
            let kernel_l =
                Layout::contiguous_with_offset((n, k), kernel_l.start_offset()).transpose(0, 1)?;
            col.matmul(kernel, (1, b * m, n, k), &col_l, &kernel_l)?
        } else {
            // Make the kernel contiguous if not already the case.
            //
            // **And then use it.** This branch built `kernel_c` and handed the
            // original `kernel` to the matmul, so the copy was computed and
            // thrown away while a non-contiguous tensor was read under a layout
            // that claims to be contiguous. The offset comes from the copy's own
            // layout, which starts at zero, rather than from the source's.
            let mut kernel_c = unsafe {
                self.device()
                    .alloc_uninit(kernel_l.shape(), kernel.dtype())?
            };
            kernel.copy_strided_src(&mut kernel_c, 0, kernel_l)?;
            let kernel_l = Layout::contiguous((n, k)).transpose(0, 1)?;
            col.matmul(&kernel_c, (1, b * m, n, k), &col_l, &kernel_l)?
        };
        let res_l = Layout::contiguous((b, h_out, w_out, n))
            .transpose(1, 2)?
            .transpose(1, 3)?;
        // **The convolution's output, in the arena its input came from.**
        //
        // Everything above this line already inherits — `Im2Col::map` carries
        // the operand's backing and the matmul takes its lhs's — so a plain
        // `alloc_uninit` here was the one pool allocation in the chain, and it
        // is the largest: this is the conv's whole result, at the layer's full
        // resolution. A convolutional decoder is nothing but this line, several
        // hundred times, which is why an image guest could seed its stage and
        // still watch the pool drain.
        let mut res_t = unsafe {
            self.device().alloc_uninit_from(
                res_l.shape(),
                res.dtype(),
                res.backing.inherit_ticket(),
            )?
        };
        res.copy_strided_src(&mut res_t, 0, &res_l)?;
        Ok(res_t)
    }

    #[cfg(feature = "cudnn")]
    fn conv2d(
        &self,
        inp_l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        let device = self.device().clone();
        if !kernel_l.is_contiguous() {
            let (slice, out_backing) =
                Conv2D(params).map(self, inp_l, kernel, kernel_l, &device)?;
            return Ok(Self {
                slice,
                device,
                backing: out_backing,
                anchor: None,
            });
        }
        let (out_w, out_h) = (params.out_w(), params.out_h());
        let dst_el = params.c_out * out_w * out_h * params.b_size;
        // **The convolution's output, in the arena its input came from.**
        //
        // The im2col path below already routes every allocation in its chain;
        // this one — the path a cudnn build actually takes — allocated all five
        // of its arms straight from the pool and returned `Backing::Owned`, so
        // a conv wrote its result outside the arena *and* told everything
        // downstream it had no provenance. In a convolutional decoder that is
        // the whole model: the guest's activations arena went almost unasked
        // while the pool drained, and the largest single site in a traced draw
        // was this line at 1 GiB across twelve calls.
        //
        // Inheritance is from `self` — the input activation, not the kernel —
        // for the same reason a matmul takes its lhs: the weight is resident
        // ground and its ticket would name the wrong lifetime.
        let inherit = self.backing;
        let (slice, backing) = match (&self.slice, &kernel.slice) {
            (S::U8(inp), S::U8(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let (mut out, backing) =
                    unsafe { alloc_inheriting::<u8>(&device, dst_el, inherit)? };
                crate::cudnn::launch_conv2d::<u8, u8>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                (S::U8(out), backing)
            }
            (S::BF16(inp), S::BF16(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let (mut out, backing) =
                    unsafe { alloc_inheriting::<bf16>(&device, dst_el, inherit)? };
                // Only PSEUDO_BFLOAT16_CONFIG is supported in cudnn, there is no "true bfloat16"
                // version.
                // https://docs.nvidia.com/deeplearning/cudnn/latest/api/cudnn-cnn-library.html#id88
                crate::cudnn::launch_conv2d::<bf16, f32>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                (S::BF16(out), backing)
            }
            (S::F16(inp), S::F16(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let (mut out, backing) =
                    unsafe { alloc_inheriting::<f16>(&device, dst_el, inherit)? };
                crate::cudnn::launch_conv2d::<f16, f16>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                (S::F16(out), backing)
            }
            (S::F32(inp), S::F32(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let (mut out, backing) =
                    unsafe { alloc_inheriting::<f32>(&device, dst_el, inherit)? };
                crate::cudnn::launch_conv2d::<f32, f32>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                (S::F32(out), backing)
            }
            (S::F64(inp), S::F64(k)) => {
                let inp = &inp.slice(inp_l.start_offset()..);
                let k = &k.slice(kernel_l.start_offset()..);
                let (mut out, backing) =
                    unsafe { alloc_inheriting::<f64>(&device, dst_el, inherit)? };
                crate::cudnn::launch_conv2d::<f64, f64>(inp, inp_l, k, &mut out, params, &device)
                    .map_err(crate::Error::wrap)?;
                (S::F64(out), backing)
            }
            (S::U32(_), S::U32(_)) => Err(CudaError::InternalError(
                "conv2d does not support u32".to_string(),
            ))?,
            (S::I64(_), S::I64(_)) => Err(CudaError::InternalError(
                "conv2d does not support i64".to_string(),
            ))?,
            _ => Err(CudaError::InternalError(
                "dtype mismatch in conv2d".to_string(),
            ))?,
        };
        Ok(Self {
            slice,
            device,
            backing,
            anchor: None,
        })
    }

    fn conv_transpose2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) =
            ConvTranspose2D(params).map(self, l, kernel, kernel_l, &device)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
            anchor: None,
        })
    }

    fn avg_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = Pool2D {
            w_k: k.0,
            h_k: k.1,
            w_stride: stride.0,
            h_stride: stride.1,
            op: PoolOp::Avg,
        }
        .map(self, &device, l)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
            anchor: None,
        })
    }

    fn max_pool2d(&self, l: &Layout, k: (usize, usize), stride: (usize, usize)) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = Pool2D {
            w_k: k.0,
            h_k: k.1,
            w_stride: stride.0,
            h_stride: stride.1,
            op: PoolOp::Max,
        }
        .map(self, &device, l)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
            anchor: None,
        })
    }

    fn upsample_nearest1d(&self, _: &Layout, _out_sz: usize) -> Result<Self> {
        crate::bail!("upsample-nearest1d is not supported on cuda")
    }

    fn upsample_nearest2d(&self, l: &Layout, out_w: usize, out_h: usize) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = UpsampleNearest2D(out_w, out_h).map(self, &device, l)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
            anchor: None,
        })
    }

    fn index_select(&self, ids: &Self, l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = IndexSelect(ids, ids_l, dim).map(self, &device, l)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
            anchor: None,
        })
    }
    fn gather(&self, l: &Layout, ids: &Self, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device().clone();
        let (slice, out_backing) = Gather(ids, ids_l, dim).map(self, &device, l)?;
        Ok(Self {
            slice,
            device,
            backing: out_backing,
            anchor: None,
        })
    }
    fn scatter_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let device = self.device().clone();
        Scatter(ids, ids_l, dim).map(&mut self.slice, l, &src.slice, src_l, &device)
    }
    fn scatter_add_set(
        &mut self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<()> {
        let device = self.device().clone();
        ScatterAdd(ids, ids_l, dim).map(&mut self.slice, l, &src.slice, src_l, &device)
    }
    fn index_add(
        &self,
        l: &Layout,
        ids: &Self,
        ids_l: &Layout,
        src: &Self,
        src_l: &Layout,
        dim: usize,
    ) -> Result<Self> {
        let device = self.device().clone();
        let mut acc = unsafe { device.alloc_uninit(l.shape(), self.dtype())? };
        self.copy_strided_src(&mut acc, 0, l)?;
        IndexAdd(ids, ids_l, dim).map(&mut acc.slice, l, &src.slice, src_l, &device)?;
        Ok(acc)
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        let elem_count = b * m * n;
        let dev = &self.device;

        // **A degenerate GEMM never reaches cuBLAS.**
        //
        // With any of `b`, `m`, `n` zero there is no output element to compute,
        // and with `k == 0` every output is the empty sum. cuBLAS handles
        // neither: it derives its grid from the problem shape, so a zero
        // dimension launches `(0,1,1)` and the runtime rejects it with
        // `cudaErrorInvalidConfiguration`. compute-sanitizer counted **41** such
        // launches in one short run, alongside the `cudaGetLastError` that then
        // picks the sticky error up somewhere unrelated — a real error planted
        // in the runtime's state by a call that had no work to do.
        //
        // These shapes are reachable in ordinary operation, not just in tests: a
        // wave with no rows for a quantum (the deferred-glue drain runs
        // `decode_forward_cobatched(&[], &[], &[], &[])`) and a grouped GEMM
        // whose group came out empty both arrive here with a zero extent.
        //
        // The allocation still happens — `alloc_zeros` for `elem_count`, which
        // is a 0-byte request in the common case and still shows up in the
        // forbidden-allocation report as its own call site. What is avoided is
        // the cuBLAS call, and that is the whole point; a 0-byte allocation is
        // cheap and wrong to elide, because the result must be a real
        // `CudaStorage` for the caller to go on using.
        //
        // The result is pool-backed (`Backing::Owned`) rather than inheriting
        // the activation's arena the way the live path below does. That is
        // honest rather than convenient — these bytes genuinely come from the
        // pool — and it costs nothing, because a zero-extent result has no
        // elements for a later `empty_beside` to inherit a span from.
        if elem_count == 0 || k == 0 {
            // `k == 0` with real output elements is an empty SUM, which is zero
            // — the value cuBLAS itself would have written with `beta = 0`. It
            // has to be materialised rather than left uninitialised, so this is
            // one of the few places `alloc_zeros` is right (AGENTS.md invariant
            // 6 excepts buffers read before they are written, and every element
            // here is read without any kernel writing it).
            let slice = match (&self.slice, &rhs.slice) {
                (CudaStorageSlice::BF16(_), CudaStorageSlice::BF16(_)) => {
                    CudaStorageSlice::BF16(dev.alloc_zeros::<bf16>(elem_count)?)
                }
                (CudaStorageSlice::F16(_), CudaStorageSlice::F16(_)) => {
                    CudaStorageSlice::F16(dev.alloc_zeros::<f16>(elem_count)?)
                }
                (CudaStorageSlice::F32(_), CudaStorageSlice::F32(_)) => {
                    CudaStorageSlice::F32(dev.alloc_zeros::<f32>(elem_count)?)
                }
                (CudaStorageSlice::F64(_), CudaStorageSlice::F64(_)) => {
                    CudaStorageSlice::F64(dev.alloc_zeros::<f64>(elem_count)?)
                }
                _ => {
                    return Err(
                        CudaError::InternalError("dtype mismatch in matmul op".to_string()).into(),
                    )
                }
            };
            return Ok(Self {
                slice,
                device: dev.clone(),
                backing: Backing::Owned,
                anchor: None,
            });
        }

        // The activation's arena, not the weight's: `self` is the left operand,
        // which for every `x @ W` in a forward is the value flowing through the
        // layer, while `rhs` is a model parameter that names no arena. Assigned
        // by each arm from what `alloc_inheriting` resolved, so a full span
        // falls back to the pool rather than failing.
        let inherit = self.backing;
        let out_backing;
        let slice = match (&self.slice, &rhs.slice) {
            (CudaStorageSlice::BF16(lhs), CudaStorageSlice::BF16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let cfg = gemm_config(bf16::ONE, bf16::ZERO, (b, m, n, k), lhs_l, rhs_l)?;
                let (mut out, resolved) =
                    unsafe { alloc_inheriting::<bf16>(dev, elem_count, inherit)? };
                out_backing = resolved;
                // Check 16-byte alignment: bf16 is 2 bytes, so offset must be multiple of 8 elements
                // CUDA malloc guarantees 256-byte aligned base, output is fresh allocation
                let known_aligned = (lhs_l.start_offset() * 2).is_multiple_of(16)
                    && (rhs_l.start_offset() * 2).is_multiple_of(16);
                unsafe {
                    gemm_strided_batched_bf16(
                        &*self.device.cublas()?,
                        cfg,
                        rhs,
                        lhs,
                        &mut out,
                        known_aligned,
                    )
                }
                .w()?;
                CudaStorageSlice::BF16(out)
            }
            (CudaStorageSlice::F16(lhs), CudaStorageSlice::F16(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let cfg = gemm_config(f16::ONE, f16::ZERO, (b, m, n, k), lhs_l, rhs_l)?;
                let (mut out, resolved) =
                    unsafe { alloc_inheriting::<f16>(dev, elem_count, inherit)? };
                out_backing = resolved;
                // Check 16-byte alignment: f16 is 2 bytes, so offset must be multiple of 8 elements
                let known_aligned = (lhs_l.start_offset() * 2).is_multiple_of(16)
                    && (rhs_l.start_offset() * 2).is_multiple_of(16);
                unsafe {
                    gemm_strided_batched_f16(
                        &*self.device.cublas()?,
                        cfg,
                        rhs,
                        lhs,
                        &mut out,
                        known_aligned,
                    )
                }
                .w()?;
                CudaStorageSlice::F16(out)
            }
            (CudaStorageSlice::F32(lhs), CudaStorageSlice::F32(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let cfg = gemm_config(1., 0., (b, m, n, k), lhs_l, rhs_l)?;
                let (mut out, resolved) =
                    unsafe { alloc_inheriting::<f32>(dev, elem_count, inherit)? };
                out_backing = resolved;
                // Check 16-byte alignment: f32 is 4 bytes, so offset must be multiple of 4 elements
                let known_aligned = (lhs_l.start_offset() * 4).is_multiple_of(16)
                    && (rhs_l.start_offset() * 4).is_multiple_of(16);
                unsafe {
                    gemm_strided_batched_f32(
                        &*self.device.cublas()?,
                        cfg,
                        rhs,
                        lhs,
                        &mut out,
                        known_aligned,
                    )
                }
                .w()?;
                CudaStorageSlice::F32(out)
            }
            (CudaStorageSlice::F64(lhs), CudaStorageSlice::F64(rhs)) => {
                let lhs = &lhs.slice(lhs_l.start_offset()..);
                let rhs = &rhs.slice(rhs_l.start_offset()..);
                let cfg = gemm_config(1., 0., (b, m, n, k), lhs_l, rhs_l)?;
                let (mut out, resolved) =
                    unsafe { alloc_inheriting::<f64>(dev, elem_count, inherit)? };
                out_backing = resolved;
                unsafe {
                    self.device
                        .cublas()?
                        .gemm_strided_batched(cfg, rhs, lhs, &mut out)
                }
                .w()?;
                CudaStorageSlice::F64(out)
            }
            _ => {
                return Err(
                    CudaError::InternalError("dtype mismatch in matmul op".to_string()).into(),
                )
            }
        };
        let device = dev.clone();
        Ok(Self {
            slice,
            device,
            backing: out_backing,
            anchor: None,
        })
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s: usize,
        dst_s: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        use kernels::simple::fill::FillDType;

        let dev = &self.device;
        let d1 = d1 as u32;
        let d2 = d2 as u32;
        // Nothing to copy so we exit early to avoid launching a kernel and some potential invalid
        // argument with a null pointer.
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let dst_s = dst_s as u32;
        let src_s = src_s as u32;
        let stream = dev.cuda_stream();

        match (&self.slice, &mut dst.slice) {
            (S::U8(s), S::U8(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::U8 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
                Ok(())
            }
            (S::U32(s), S::U32(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::U32 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
                Ok(())
            }
            (S::I64(s), S::I64(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::I64 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
                Ok(())
            }
            (S::BF16(s), S::BF16(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::BF16 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
                Ok(())
            }
            (S::F16(s), S::F16(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::F16 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
                Ok(())
            }
            (S::F32(s), S::F32(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::F32 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
                Ok(())
            }
            (S::F64(s), S::F64(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::F64 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
                Ok(())
            }
            (S::F8E4M3(s), S::F8E4M3(d)) => {
                let src_slice = s.slice(src_o..);
                let dst_slice = d.slice(dst_o..);
                let (src_ptr, _src_guard) = src_slice.device_ptr(&stream);
                let (dst_ptr, _dst_guard) = dst_slice.device_ptr(&stream);
                unsafe {
                    kernels::simple::fill::run_copy2d_op(
                        FillDType::F8E4M3 as i32,
                        src_ptr as *const std::ffi::c_void,
                        dst_ptr as *mut std::ffi::c_void,
                        d1,
                        d2,
                        src_s,
                        dst_s,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
                Ok(())
            }
            _ => Err(CudaError::InternalError(
                "dtype mismatch in copy2d".to_string(),
            ))?,
        }
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_shape = src_l.shape();
        let dims = src_shape.dims();
        let el_count = src_shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }
        let dev = &self.device;

        // Helper macro for FFI-based strided copy
        macro_rules! copy_strided_ffi {
            ($src:expr, $dst:expr, $dtype:expr) => {{
                let (src, mut dst) = slice_src_and_dst($src, src_l, $dst, dst_offset);
                if src_l.is_contiguous() {
                    dev.memcpy_dtod(&src, &mut dst)?
                } else {
                    use kernels::simple::unary::UnaryOp;
                    let stream = dev.cuda_stream();

                    // Prepare dims/strides info for non-contiguous tensors
                    let info = dev.info_table(&[dims, src_l.stride()].concat())?;
                    let (info_ptr, _info_guard) = info.device_ptr(&stream);

                    let (src_ptr, _src_guard) = src.device_ptr(&stream);
                    let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

                    unsafe {
                        kernels::simple::unary::run_unary_op(
                            UnaryOp::Copy as i32,
                            $dtype,
                            el_count,
                            dims.len(),
                            info_ptr as *const usize,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                }
            }};
        }

        use kernels::simple::unary::UnaryDType;
        match (&self.slice, &mut dst.slice) {
            (CudaStorageSlice::BF16(src), CudaStorageSlice::BF16(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::BF16 as i32);
            }
            (CudaStorageSlice::F16(src), CudaStorageSlice::F16(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::F16 as i32);
            }
            (CudaStorageSlice::F32(src), CudaStorageSlice::F32(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::F32 as i32);
            }
            (CudaStorageSlice::F8E4M3(src), CudaStorageSlice::F8E4M3(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::F8E4M3 as i32);
            }
            (CudaStorageSlice::U8(src), CudaStorageSlice::U8(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::U8 as i32);
            }
            (CudaStorageSlice::U32(src), CudaStorageSlice::U32(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::U32 as i32);
            }
            (CudaStorageSlice::I64(src), CudaStorageSlice::I64(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::I64 as i32);
            }
            (CudaStorageSlice::F64(src), CudaStorageSlice::F64(dst)) => {
                copy_strided_ffi!(src, dst, UnaryDType::F64 as i32);
            }
            _ => Err(CudaError::InternalError(
                "dtype mismatch in copy_strided op".to_string(),
            ))?,
        }
        Ok(())
    }
}
