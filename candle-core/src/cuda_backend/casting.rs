//! Converts a storage's values to another dtype on the device, reusing the
//! buffer in place where it is large enough.
//!
//! An in-place cast writes the converted values into the existing allocation,
//! which is then reinterpreted as the new type. That reinterpretation copies
//! into a fresh buffer, so the old range is retired behind a fence rather than
//! dropped while the cast and copy that read it are still queued. The new
//! storage keeps the backing of the one it came from. A contiguous view with a
//! non-zero start offset is refused, since the in-place cast reads from the
//! base pointer; the allocating conversion covers that case.

use super::*;

impl CudaStorage {
    /// In-place type conversion using CUDA cast_mut kernels.
    ///
    /// This function converts the tensor data to a new dtype in-place when possible.
    /// The buffer must be large enough to hold the destination data type.
    ///
    /// # Returns
    /// - `Ok(true)` if the conversion was performed in-place
    /// - `Ok(false)` if the buffer was too small (caller should fall back to regular to_dtype)
    ///
    /// # Safety
    /// The buffer must have at least `elem_count * max(src_size, dst_size)` bytes allocated.
    pub fn to_dtype_mut(&mut self, layout: &Layout, dtype: DType) -> Result<bool> {
        use kernels::simple::cast::CastDType;

        let src_dtype = self.dtype();

        // No-op if same dtype
        if src_dtype == dtype {
            return Ok(true);
        }

        // Contiguous tensors are required for in-place conversion.
        // The caller (Tensor::to_dtype_mut) ensures this, but we check as a safety measure.
        if !layout.is_contiguous() {
            return Ok(false);
        }

        // The in-place cast reads/writes from the buffer's base pointer (no offset) and
        // the caller resets the layout to `contiguous(offset 0)` afterwards. A contiguous
        // *view* with a non-zero start offset (e.g. the second half of a last-dim
        // `narrow`) would therefore cast the wrong elements and report the wrong region.
        // Fall back to the allocating `to_dtype`, which honours the offset.
        if layout.start_offset() != 0 {
            return Ok(false);
        }

        let elem_count = layout.shape().elem_count();
        let src_size = src_dtype.size_in_bytes();
        let dst_size = dtype.size_in_bytes();

        // Check if buffer is large enough for the destination type
        let buffer_bytes = self.buffer_byte_len();
        let required_bytes = elem_count * dst_size.max(src_size);

        if buffer_bytes < required_bytes {
            // Buffer too small, caller should use regular to_dtype
            return Ok(false);
        }

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

        let src_dtype_i32 = dtype_to_cast(src_dtype);
        let dst_dtype_i32 = dtype_to_cast(dtype);
        let stream = self.device.cuda_stream();

        // Get mutable pointer and perform in-place cast
        let buf_ptr = self.slice.device_ptr_mut(&stream)?;

        unsafe {
            kernels::simple::cast::run_cast_mut(
                src_dtype_i32,
                dst_dtype_i32,
                elem_count,
                buf_ptr,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }

        // Now we need to "reinterpret" the slice as the new dtype.
        // This is safe because:
        // 1. We verified buffer size is sufficient
        // 2. The cast_mut kernel has converted the data in-place
        // 3. The buffer memory layout is compatible
        // `reinterpret_slice_as` allocates a *fresh buffer* and copies into it,
        // so this is a change of ownership, not a retype in place. Two things
        // follow, and both were previously missed:
        //
        //  1. The outgoing slice has to be disposed of according to the backing
        //     it had. A plain assignment would drop it — and dropping a lease
        //     calls `cuMemFreeAsync` on memory owned by an arena.
        //  2. `backing` has to follow the bytes to wherever the *new* buffer
        //     came from. The destination inherits this storage's arena, so a
        //     wave-backed cast stays on the wave; hard-coding either answer gets
        //     one of the two cases wrong (a lease marked `Owned` is a double
        //     free, a pool buffer marked `Lease` is a permanent leak).
        let (fresh, fresh_backing) = self.reinterpret_slice_as(dtype, elem_count)?;
        let previous = std::mem::replace(&mut self.slice, fresh);
        match self.backing {
            // The old range is no storage's any more: an arena that frees takes
            // it back, behind a fence recorded after the cast that reads it.
            Backing::Lease(origin) => {
                let ptr = previous.leak_view();
                if let LeaseOrigin::Wave(ticket) = origin {
                    wave_provenance::wave_release(ticket, ptr);
                }
            }
            // Retired, not dropped: the cast and the copy just issued read it,
            // and while this thread records a wave they have not run yet — a
            // free now would queue ahead of them.
            Backing::Owned => self.device.retire(previous),
        }
        self.backing = fresh_backing;
        // The fresh buffer is not the anchored memory, so this storage no longer has
        // anything of its owner's to keep alive.
        self.anchor = None;

        Ok(true)
    }

    /// Reinterpret the underlying buffer as a different dtype by copying to a new typed buffer.
    ///
    /// After an in-place cast, the buffer contains valid data for the target dtype,
    /// but the CudaSlice still has the original type. This function creates a new
    /// properly-typed CudaSlice by copying the bytes.
    ///
    /// # Safety
    /// The caller must ensure:
    /// - The buffer contains valid data for the target dtype
    /// - The buffer is large enough to hold `elem_count` elements of the target dtype
    pub(super) fn reinterpret_slice_as(
        &self,
        dtype: DType,
        elem_count: usize,
    ) -> Result<(CudaStorageSlice, Backing)> {
        let dev = &self.device;
        let byte_count = elem_count * dtype.size_in_bytes();
        let stream = dev.cuda_stream();

        // Get source pointer
        let src_ptr: cudarc::driver::sys::CUdeviceptr = match &self.slice {
            CudaStorageSlice::U8(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::U32(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::I64(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::BF16(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::F16(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::F32(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::F64(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::F8E4M3(s) => {
                let (ptr, _) = s.device_ptr(&stream);
                ptr as cudarc::driver::sys::CUdeviceptr
            }
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        };

        // Allocate destination buffer of the correct type and copy raw bytes.
        // The retyped buffer holds the same values this storage already holds,
        // so it belongs wherever this storage does.
        let inherit = self.backing;
        // The copy is queued on the launch stream, behind the in-place cast
        // that produced its bytes. A synchronous copy runs on the legacy stream
        // at once — ahead of that cast while this thread records a wave, so it
        // read the bytes before they were cast. The stream is taken after the
        // allocation, which may end the recording segment.
        macro_rules! alloc_and_copy {
            ($ty:ty, $wrapper:path) => {{
                use cudarc::driver::DevicePtrMut;
                let (mut dst, dst_backing) =
                    unsafe { alloc_inheriting::<$ty>(dev, elem_count, inherit)? };
                let launch = dev.cuda_stream();
                let (dst_ptr, _) = dst.device_ptr_mut(&launch);
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        dst_ptr,
                        src_ptr,
                        byte_count,
                        launch.cu_stream(),
                    )
                    .w()?;
                }
                Ok(($wrapper(dst), dst_backing))
            }};
        }

        match dtype {
            DType::U8 => alloc_and_copy!(u8, CudaStorageSlice::U8),
            DType::U32 => alloc_and_copy!(u32, CudaStorageSlice::U32),
            DType::I64 => alloc_and_copy!(i64, CudaStorageSlice::I64),
            DType::BF16 => alloc_and_copy!(bf16, CudaStorageSlice::BF16),
            DType::F16 => alloc_and_copy!(f16, CudaStorageSlice::F16),
            DType::F32 => alloc_and_copy!(f32, CudaStorageSlice::F32),
            DType::F64 => alloc_and_copy!(f64, CudaStorageSlice::F64),
            DType::F8E4M3 => alloc_and_copy!(F8E4M3, CudaStorageSlice::F8E4M3),
        }
    }
}
