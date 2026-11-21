//! Runs the two-operand kernels (where, comparison and binary operations)
//! through the FFI dispatcher.
//!
//! Each map first tries the dispatcher and writes an output that the kernel
//! fully initialises. A launch with no elements is not made, as in the
//! single-operand maps, because a zero grid is an invalid configuration.

use super::*;

pub(super) struct WhereCond<'a>(pub(super) &'a CudaStorage, pub(super) &'a Layout);

impl Map2 for WhereCond<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        t: &CudaSlice<T>,
        layout_t: &Layout,
        f: &CudaSlice<T>,
        layout_f: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_where");
        let ids_l = &self.1;
        let ids = &self.0;

        // Get condition dtype for FFI
        let cond_dtype = match storage_to_where_cond_dtype(&ids.slice) {
            Some(d) => d,
            None => Err(CudaError::UnexpectedDType {
                msg: "where conditions should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })
            .w()?,
        };

        let shape = ids_l.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let ds =
            dev.info_table(&[dims, ids_l.stride(), layout_t.stride(), layout_f.stride()].concat())?;
        let t = &t.slice(layout_t.start_offset()..);
        let f = &f.slice(layout_f.start_offset()..);

        // Get data dtype for FFI
        let data_dtype = dtype_to_where_data_dtype(T::DTYPE);

        let stream = dev.cuda_stream();

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, el, origin)? };
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (t_ptr, _t_guard) = t.device_ptr(&stream);
            let (f_ptr, _f_guard) = f.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            // Get ids pointer and call FFI inside match arms to keep temporaries alive
            match &ids.slice {
                CudaStorageSlice::U8(slice) => {
                    let s = slice.slice(ids_l.start_offset()..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::ternary::run_where(
                            cond_dtype,
                            data_dtype,
                            el,
                            dims.len(),
                            ds_ptr as *const usize,
                            ids_ptr as *const std::ffi::c_void,
                            t_ptr as *const std::ffi::c_void,
                            f_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                }
                CudaStorageSlice::U32(slice) => {
                    let s = slice.slice(ids_l.start_offset()..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::ternary::run_where(
                            cond_dtype,
                            data_dtype,
                            el,
                            dims.len(),
                            ds_ptr as *const usize,
                            ids_ptr as *const std::ffi::c_void,
                            t_ptr as *const std::ffi::c_void,
                            f_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                }
                CudaStorageSlice::I64(slice) => {
                    let s = slice.slice(ids_l.start_offset()..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::ternary::run_where(
                            cond_dtype,
                            data_dtype,
                            el,
                            dims.len(),
                            ds_ptr as *const usize,
                            ids_ptr as *const std::ffi::c_void,
                            t_ptr as *const std::ffi::c_void,
                            f_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                }
                _ => unreachable!(), // Already checked above
            };
        }
        Ok((out, out_backing))
    }
}

impl<U: crate::op::BinaryOpT> Map2 for U {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        lhs: &CudaSlice<T>,
        lhs_l: &Layout,
        rhs: &CudaSlice<T>,
        rhs_l: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!(U::KERNEL);
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let lhs_start = lhs_l.start_offset();
        let rhs_start = rhs_l.start_offset();
        let stream = dev.cuda_stream();

        // Try to use FFI dispatcher first
        if let Some(op) = kernel_name_to_binary_arith_op(U::KERNEL) {
            let dtype = T::DTYPE;
            let dtype_i32 = dtype_to_binary_dtype(dtype);

            // Prepare dims and strides info for non-contiguous tensors
            let info: Option<Arc<Uploaded<usize>>> =
                if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
                    None
                } else {
                    Some(dev.info_table(&[dims, lhs_l.stride(), rhs_l.stride()].concat())?)
                };

            let lhs_slice = &lhs.slice(lhs_start..);
            let rhs_slice = &rhs.slice(rhs_start..);
            // SAFETY: Allocated memory will be initialized by the kernel
            let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, elem_count, origin)? };
            {
                let info_ptr = match &info {
                    Some(s) => {
                        let (ptr, _guard) = s.device_ptr(&stream);
                        ptr as *const usize
                    }
                    None => std::ptr::null(),
                };
                let (lhs_ptr, _lhs_guard) = lhs_slice.device_ptr(&stream);
                let (rhs_ptr, _rhs_guard) = rhs_slice.device_ptr(&stream);
                let (out_ptr, _out_guard) = out.device_ptr(&stream);

                // Keep info alive for the kernel call
                let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));

                // No elements, no launch — see the unary map.
                if elem_count > 0 {
                    unsafe {
                        kernels::simple::binary::run_binary_arith_op(
                            op,
                            dtype_i32,
                            elem_count,
                            dims.len(),
                            info_ptr,
                            lhs_ptr as *const std::ffi::c_void,
                            rhs_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                }
            }
            return Ok((out, out_backing));
        }

        // All binary operations should be handled by FFI dispatcher above
        Err(CudaError::InternalError(format!(
            "Unrecognized binary kernel '{}' - all operations should use FFI dispatcher",
            U::KERNEL
        )))
        .w()
    }
}

pub(super) struct Cmp(pub(super) CmpOp);

impl Map2Any for Cmp {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        lhs: &CudaSlice<T>,
        lhs_l: &Layout,
        rhs: &CudaSlice<T>,
        rhs_l: &Layout,
        dev: &CudaDevice,
        origin: Backing,
    ) -> Result<OutS> {
        cuda_breadcrumb!("run_binary_cmp_op");
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();
        let lhs_start = lhs_l.start_offset();
        let rhs_start = rhs_l.start_offset();
        let stream = dev.cuda_stream();

        // Use FFI dispatcher
        let op = cmp_op_to_binary_cmp(self.0);
        let dtype = T::DTYPE;
        let dtype_i32 = dtype_to_binary_dtype(dtype);

        // Prepare dims and strides info for non-contiguous tensors
        let info: Option<Arc<Uploaded<usize>>> = if lhs_l.is_contiguous() && rhs_l.is_contiguous() {
            None
        } else {
            Some(dev.info_table(&[dims, lhs_l.stride(), rhs_l.stride()].concat())?)
        };

        let lhs_slice = &lhs.slice(lhs_start..);
        let rhs_slice = &rhs.slice(rhs_start..);
        // SAFETY: Allocated memory will be initialized by the kernel
        let (out, out_backing) = unsafe { alloc_inheriting::<u8>(dev, elem_count, origin)? };
        {
            let info_ptr = match &info {
                Some(s) => {
                    let (ptr, _guard) = s.device_ptr(&stream);
                    ptr as *const usize
                }
                None => std::ptr::null(),
            };
            let (lhs_ptr, _lhs_guard) = lhs_slice.device_ptr(&stream);
            let (rhs_ptr, _rhs_guard) = rhs_slice.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            // Keep info alive for the kernel call
            let _info_guard = info.as_ref().map(|s| s.device_ptr(&stream));

            unsafe {
                kernels::simple::binary::run_binary_cmp_op(
                    op,
                    dtype_i32,
                    elem_count,
                    dims.len(),
                    info_ptr,
                    lhs_ptr as *const std::ffi::c_void,
                    rhs_ptr as *const std::ffi::c_void,
                    out_ptr as *mut u8,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        Ok((S::U8(out), out_backing))
    }
}
