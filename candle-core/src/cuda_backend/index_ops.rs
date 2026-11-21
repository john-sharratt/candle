//! Runs the gather, scatter and index-add kernels, and refuses a scatter whose
//! kernel does not exist.
//!
//! Each operation is a unit struct implementing a map trait, and resolves its
//! index and data dtypes for the dispatcher. An embedding gather reads a
//! resident table, so its output goes to the forward span beside the ids rather
//! than to the table's own arena. `expect_dispatched` turns a missing kernel
//! into an error, because the dispatcher then launches nothing and the
//! destination is left unwritten.

use super::*;

pub(super) struct IndexSelect<'a>(
    pub(super) &'a CudaStorage,
    pub(super) &'a Layout,
    pub(super) usize,
);

impl Map1 for IndexSelect<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        src_l: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_index_select");
        let ids_l = &self.1;
        let ids = &self.0;

        // Get index dtype for FFI
        let idx_dtype = match storage_to_index_dtype(&ids.slice) {
            Some(d) => d,
            None => Err(CudaError::UnexpectedDType {
                msg: "index_select ids should be u8, u32, or i64",
                expected: DType::U32,
                got: ids.dtype(),
            })
            .w()?,
        };

        let ids_shape = ids_l.shape();
        let ids_dims = ids_shape.dims();
        let ds = dev.info_table(&[ids_dims, ids_l.stride()].concat())?;
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-select" }.bt())?,
        };
        let left_size: usize = src_l.dims()[..self.2].iter().product();
        let right_size: usize = src_l.dims()[self.2 + 1..].iter().product();
        let src_dim_size = src_l.dims()[self.2];
        let ids_dim_size = ids_shape.elem_count();
        let dst_el = ids_shape.elem_count() * left_size * right_size;

        // Get data dtype for FFI
        let data_dtype = dtype_to_indexing_data_dtype(T::DTYPE);

        let stream = dev.cuda_stream();

        // The rows belong where the indices are when the table has no arena of
        // its own: an embedding gather reads a resident table — no wave
        // provenance — through ids laid on the forward span, and its output is
        // that forward's residual.
        let origin = if origin.inherit_ticket().is_some() {
            origin
        } else {
            ids.backing
        };
        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, dst_el, origin)? };
        {
            let (ds_ptr, _ds_guard) = ds.device_ptr(&stream);
            let (src_ptr, _src_guard) = src.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            // Get ids pointer based on dtype - need to keep temp slices alive
            let ids_ptr: u64 = match &ids.slice {
                CudaStorageSlice::U32(slice) => {
                    let s = slice.slice(ids_l.start_offset()..);
                    let (ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_index_select(
                            idx_dtype,
                            data_dtype,
                            dst_el,
                            ids_dims.len(),
                            ds_ptr as *const usize,
                            ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            left_size,
                            src_dim_size,
                            ids_dim_size,
                            right_size,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                    ptr
                }
                CudaStorageSlice::U8(slice) => {
                    let s = slice.slice(ids_l.start_offset()..);
                    let (ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_index_select(
                            idx_dtype,
                            data_dtype,
                            dst_el,
                            ids_dims.len(),
                            ds_ptr as *const usize,
                            ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            left_size,
                            src_dim_size,
                            ids_dim_size,
                            right_size,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                    ptr
                }
                CudaStorageSlice::I64(slice) => {
                    let s = slice.slice(ids_l.start_offset()..);
                    let (ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_index_select(
                            idx_dtype,
                            data_dtype,
                            dst_el,
                            ids_dims.len(),
                            ds_ptr as *const usize,
                            ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            left_size,
                            src_dim_size,
                            ids_dim_size,
                            right_size,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                    ptr
                }
                _ => unreachable!(), // Already checked above
            };
            let _ = ids_ptr; // Suppress unused warning
        }
        Ok((out, out_backing))
    }
}

pub(super) struct Gather<'a>(
    pub(super) &'a CudaStorage,
    pub(super) &'a Layout,
    pub(super) usize,
);

impl Map1 for Gather<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        src: &CudaSlice<T>,
        dev: &CudaDevice,
        src_l: &Layout,
        origin: Backing,
    ) -> Result<Out<T>> {
        cuda_breadcrumb!("run_gather");
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "gather" }.bt())?,
        };

        // Get index dtype for FFI
        let idx_dtype = match storage_to_index_dtype(&ids.slice) {
            Some(d) => d,
            None => Err(CudaError::UnexpectedDType {
                msg: "gather ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };

        let el = ids_l.shape().elem_count();
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let ids_dim_sz = ids_l.dims()[dim];

        // Get data dtype for FFI
        let data_dtype = dtype_to_indexing_data_dtype(T::DTYPE);

        let stream = dev.cuda_stream();

        // SAFETY: Set later by running the kernel.
        let (out, out_backing) = unsafe { alloc_inheriting::<T>(dev, el, origin)? };
        {
            let (src_ptr, _src_guard) = src.device_ptr(&stream);
            let (out_ptr, _out_guard) = out.device_ptr(&stream);

            // Get ids pointer - call FFI inside each match arm to keep temporaries alive
            match &ids.slice {
                CudaStorageSlice::U32(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_gather(
                            idx_dtype,
                            data_dtype,
                            el,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            ids_dim_sz,
                            right_sz,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                }
                CudaStorageSlice::U8(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_gather(
                            idx_dtype,
                            data_dtype,
                            el,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            ids_dim_sz,
                            right_sz,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        );
                    }
                }
                CudaStorageSlice::I64(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_gather(
                            idx_dtype,
                            data_dtype,
                            el,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            out_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            ids_dim_sz,
                            right_sz,
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

pub(super) struct IndexAdd<'a>(
    pub(super) &'a CudaStorage,
    pub(super) &'a Layout,
    pub(super) usize,
);

impl Map2InPlace for IndexAdd<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut CudaSlice<T>,
        dst_l: &Layout,
        src: &CudaSlice<T>,
        src_l: &Layout,
        dev: &CudaDevice,
    ) -> Result<()> {
        cuda_breadcrumb!("run_index_add");
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };

        // Get index dtype for FFI
        let idx_dtype = match storage_to_index_dtype(&ids.slice) {
            Some(d) => d,
            None => Err(CudaError::UnexpectedDType {
                msg: "index-add ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };

        let dst = match dst_l.contiguous_offsets() {
            Some((o1, o2)) => dst.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "index-add" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_l.dims()[dim];
        let ids_dim_sz = ids_l.dims()[0];

        // Get data dtype for FFI
        let data_dtype = dtype_to_indexing_data_dtype(T::DTYPE);

        let stream = dev.cuda_stream();

        {
            let (src_ptr, _src_guard) = src.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

            // Get ids pointer and call FFI inside match arms to keep temporaries alive
            let status = match &ids.slice {
                CudaStorageSlice::U32(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_index_add(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            ids_dim_sz,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        )
                    }
                }
                CudaStorageSlice::U8(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_index_add(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            ids_dim_sz,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        )
                    }
                }
                CudaStorageSlice::I64(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_index_add(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            ids_dim_sz,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        )
                    }
                }
                _ => unreachable!(), // Already checked above
            };
            expect_dispatched(status, "index-add", ids.dtype(), T::DTYPE)?;
        }
        Ok(())
    }
}

/// Turn a scatter-family dispatcher's status into an error when it found no
/// kernel for the pair — it launched nothing, so the destination is unwritten.
pub(super) fn expect_dispatched(
    status: i32,
    op: &'static str,
    ids: DType,
    data: DType,
) -> Result<()> {
    if status == kernels::simple::indexing::DISPATCHED {
        return Ok(());
    }
    Err(crate::Error::Msg(format!(
        "{op}: no CUDA kernel for {ids:?} indices over {data:?} data (dispatcher status {status})"
    ))
    .bt())
}

pub(super) struct Scatter<'a>(
    pub(super) &'a CudaStorage,
    pub(super) &'a Layout,
    pub(super) usize,
);

impl Map2InPlace for Scatter<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut CudaSlice<T>,
        dst_l: &Layout,
        src: &CudaSlice<T>,
        src_l: &Layout,
        dev: &CudaDevice,
    ) -> Result<()> {
        cuda_breadcrumb!("run_scatter");
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "scatter" }.bt())?,
        };

        // Get index dtype for FFI
        let idx_dtype = match storage_to_index_dtype(&ids.slice) {
            Some(d) => d,
            None => Err(CudaError::UnexpectedDType {
                msg: "scatter ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };

        let dst = match dst_l.contiguous_offsets() {
            Some((o1, o2)) => dst.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter" }.bt())?,
        };
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_l.dims()[dim];

        // Get data dtype for FFI
        let data_dtype = dtype_to_indexing_data_dtype(T::DTYPE);

        let stream = dev.cuda_stream();

        {
            let (src_ptr, _src_guard) = src.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

            // Get ids pointer and call FFI inside match arms to keep temporaries alive
            let status = match &ids.slice {
                CudaStorageSlice::U32(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_scatter(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        )
                    }
                }
                CudaStorageSlice::U8(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_scatter(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        )
                    }
                }
                CudaStorageSlice::I64(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_scatter(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        )
                    }
                }
                _ => unreachable!(), // Already checked above
            };
            expect_dispatched(status, "scatter", ids.dtype(), T::DTYPE)?;
        }
        Ok(())
    }
}

pub(super) struct ScatterAdd<'a>(
    pub(super) &'a CudaStorage,
    pub(super) &'a Layout,
    pub(super) usize,
);

impl Map2InPlace for ScatterAdd<'_> {
    fn f<T: DeviceRepr + WithDType + ValidAsZeroBits>(
        &self,
        dst: &mut CudaSlice<T>,
        dst_l: &Layout,
        src: &CudaSlice<T>,
        src_l: &Layout,
        dev: &CudaDevice,
    ) -> Result<()> {
        cuda_breadcrumb!("run_scatter_add");
        let ids = &self.0;
        let ids_l = &self.1;
        let dim = self.2;
        let (ids_o1, _) = match ids_l.contiguous_offsets() {
            Some(o12) => o12,
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };

        // Get index dtype for FFI
        let idx_dtype = match storage_to_index_dtype(&ids.slice) {
            Some(d) => d,
            None => Err(CudaError::UnexpectedDType {
                msg: "scatter-add ids should be u8/u32/i64",
                expected: DType::U32,
                got: ids.dtype(),
            })?,
        };

        let dst = match dst_l.contiguous_offsets() {
            Some((o1, o2)) => dst.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };
        let src = match src_l.contiguous_offsets() {
            Some((o1, o2)) => src.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "scatter-add" }.bt())?,
        };
        let left_sz: usize = src_l.dims()[..dim].iter().product();
        let right_sz: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_sz = src_l.dims()[dim];
        let dst_dim_sz = dst_l.dims()[dim];

        // Get data dtype for FFI
        let data_dtype = dtype_to_indexing_data_dtype(T::DTYPE);

        let stream = dev.cuda_stream();

        {
            let (src_ptr, _src_guard) = src.device_ptr(&stream);
            let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);

            // Get ids pointer and call FFI inside match arms to keep temporaries alive
            let status = match &ids.slice {
                CudaStorageSlice::U32(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_scatter_add(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        )
                    }
                }
                CudaStorageSlice::U8(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_scatter_add(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        )
                    }
                }
                CudaStorageSlice::I64(slice) => {
                    let s = slice.slice(ids_o1..);
                    let (ids_ptr, _guard) = s.device_ptr(&stream);
                    unsafe {
                        kernels::simple::indexing::run_scatter_add(
                            idx_dtype,
                            data_dtype,
                            ids_ptr as *const std::ffi::c_void,
                            src_ptr as *const std::ffi::c_void,
                            dst_ptr as *mut std::ffi::c_void,
                            left_sz,
                            src_dim_sz,
                            dst_dim_sz,
                            right_sz,
                            stream.cu_stream() as *mut std::ffi::c_void,
                        )
                    }
                }
                _ => unreachable!(), // Already checked above
            };
            expect_dispatched(status, "scatter-add", ids.dtype(), T::DTYPE)?;
        }
        Ok(())
    }
}
