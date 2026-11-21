//! Copies device data to host memory, including the asynchronous readbacks the
//! routing path uses.
//!
//! A U32 range can be copied on a given stream into a host buffer. When that
//! buffer is pinned the copy returns without waiting, so the caller synchronises
//! the stream once for however many copies it queued. Scalar and range readbacks
//! must see everything recorded before them. Each copy goes through the checked
//! wrapper, so a sticky or out-of-memory error stays visible to the caller, and
//! a non-U32 storage is refused.

use super::*;

impl CudaStorage {
    /// Copy a range of U32 data from GPU to a host buffer on a specific stream.
    ///
    /// When `dst` is backed by pinned memory (`cuMemAllocHost`), the copy is
    /// truly asynchronous — the CPU returns immediately and the DMA engine
    /// handles the transfer.  This is the async DtoH path used for routing
    /// index transfer without draining the compute pipeline.
    ///
    /// `offset` and `elem_count` identify the sub-range of the device slice
    /// to copy (from `contiguous_offsets` on the tensor layout).  `dst` must
    /// have at least `elem_count` elements.
    ///
    /// Returns `Err` if the storage is not U32.
    pub fn copy_u32_to_host_on_stream(
        &self,
        dst: &mut [u32],
        stream: &Arc<CudaStream>,
        offset: usize,
        elem_count: usize,
    ) -> Result<()> {
        match &self.slice {
            CudaStorageSlice::U32(slice) => {
                if offset + elem_count > slice.len() {
                    crate::bail!(
                        "copy_u32_to_host_on_stream: range {}..{} exceeds slice len {}",
                        offset,
                        offset + elem_count,
                        slice.len(),
                    );
                }
                if dst.len() < elem_count {
                    crate::bail!(
                        "copy_u32_to_host_on_stream: dst too small ({} < {})",
                        dst.len(),
                        elem_count,
                    );
                }
                let view = slice.slice(offset..offset + elem_count);
                // `.w()`, not `Error::wrap` — see `CudaDevice::synchronize`'s
                // comment: this is a real device round-trip and needs the same
                // sticky/OOM-streak visibility as every other CUDA call.
                stream.memcpy_dtoh(&view, &mut dst[..elem_count]).w()?;
                Ok(())
            }
            _ => crate::bail!(
                "copy_u32_to_host_on_stream: expected U32 storage, got {:?}",
                self.dtype(),
            ),
        }
    }

    /// Enqueue a copy of `elem_count` F32 elements starting at `offset` into
    /// `dst` as their bytes, on `stream`, without waiting for it.
    ///
    /// The bytes are the device's — little-endian IEEE-754, which is every host
    /// CUDA runs on — so `dst` receives exactly what a `to_le_bytes` walk of the
    /// values would produce. Into pinned `dst` the copy is asynchronous and the
    /// caller synchronises `stream` once for however many it enqueued: that is
    /// the point, against one synchronising readback per tensor.
    pub fn copy_f32_bytes_to_host_on_stream(
        &self,
        dst: &mut [u8],
        stream: &Arc<CudaStream>,
        offset: usize,
        elem_count: usize,
    ) -> Result<()> {
        match &self.slice {
            CudaStorageSlice::F32(slice) => {
                if offset + elem_count > slice.len() {
                    crate::bail!(
                        "copy_f32_bytes_to_host_on_stream: range {}..{} exceeds slice len {}",
                        offset,
                        offset + elem_count,
                        slice.len(),
                    );
                }
                let bytes = elem_count * std::mem::size_of::<f32>();
                if dst.len() < bytes {
                    crate::bail!(
                        "copy_f32_bytes_to_host_on_stream: dst too small ({} < {})",
                        dst.len(),
                        bytes,
                    );
                }
                let view = slice.slice(offset..offset + elem_count);
                let (src, _guard) = view.device_ptr(stream);
                // SAFETY: `src` is a live device range of exactly `bytes` bytes,
                // held by `_guard` until the copy is enqueued, and `dst[..bytes]` is
                // a host range the caller keeps alive until it synchronises `stream`.
                unsafe { memcpy_dtoh_async(&mut dst[..bytes], src, stream.cu_stream()) }.w()?;
                Ok(())
            }
            _ => crate::bail!(
                "copy_f32_bytes_to_host_on_stream: expected F32 storage, got {:?}",
                self.dtype(),
            ),
        }
    }

    /// Fast scalar transfer - optimized for single-element copies.
    /// Uses slice-based transfer to copy only a single element instead of the whole tensor.
    /// This is much faster than `to_cpu_storage()` for extracting single values.
    pub fn to_cpu_scalar<T>(&self, offset: usize) -> Result<T>
    where
        T: DeviceRepr + WithDType + Copy,
    {
        use cudarc::driver::DeviceSlice;

        // A readback must see everything recorded before it.
        let _eager = self.device.pause_capture()?;
        match &self.slice {
            CudaStorageSlice::U8(slice) if T::DTYPE == DType::U8 => {
                // Create a slice of just one element
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                // SAFETY: We've checked the dtype matches and vec has exactly 1 element
                Ok(unsafe { std::mem::transmute_copy(&vec[0]) })
            }
            CudaStorageSlice::U32(slice) if T::DTYPE == DType::U32 => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(unsafe { std::mem::transmute_copy(&vec[0]) })
            }
            CudaStorageSlice::I64(slice) if T::DTYPE == DType::I64 => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(unsafe { std::mem::transmute_copy(&vec[0]) })
            }
            CudaStorageSlice::F32(slice) if T::DTYPE == DType::F32 => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(unsafe { std::mem::transmute_copy(&vec[0]) })
            }
            CudaStorageSlice::F64(slice) if T::DTYPE == DType::F64 => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(unsafe { std::mem::transmute_copy(&vec[0]) })
            }
            CudaStorageSlice::F16(slice) if T::DTYPE == DType::F16 => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(unsafe { std::mem::transmute_copy(&vec[0]) })
            }
            CudaStorageSlice::BF16(slice) if T::DTYPE == DType::BF16 => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(unsafe { std::mem::transmute_copy(&vec[0]) })
            }
            _ => {
                // Fallback to full transfer for unsupported types
                let cpu_storage = self.to_cpu_storage()?;
                let data = T::cpu_storage_as_slice(&cpu_storage)?;
                Ok(data[offset])
            }
        }
    }

    /// Elements `start..end` copied to the host, and nothing else of the
    /// storage — what a readback of a contiguous view needs, where
    /// [`BackendStorage::to_cpu_storage`] copies everything the view was cut
    /// from.
    pub fn to_cpu_storage_range(&self, start: usize, end: usize) -> Result<CpuStorage> {
        // A readback must see everything recorded before it.
        let _eager = self.device.pause_capture()?;
        macro_rules! range {
            ($slice:expr, $variant:ident) => {{
                let view = $slice.slice(start..end);
                CpuStorage::$variant($slice.stream().memcpy_dtov(&view).w()?)
            }};
        }
        Ok(match &self.slice {
            CudaStorageSlice::U8(s) => range!(s, U8),
            CudaStorageSlice::U32(s) => range!(s, U32),
            CudaStorageSlice::I64(s) => range!(s, I64),
            CudaStorageSlice::BF16(s) => range!(s, BF16),
            CudaStorageSlice::F16(s) => range!(s, F16),
            CudaStorageSlice::F32(s) => range!(s, F32),
            CudaStorageSlice::F64(s) => range!(s, F64),
            CudaStorageSlice::F8E4M3(s) => range!(s, F8E4M3),
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        })
    }

    /// Transfer a single scalar element from GPU to CPU (type-erased version).
    /// This is used as a fast fallback when the compile-time type is unknown.
    /// Returns the value as a CpuStorage containing exactly one element.
    pub fn to_cpu_storage_scalar(&self, offset: usize) -> Result<CpuStorage> {
        use cudarc::driver::DeviceSlice;

        match &self.slice {
            CudaStorageSlice::U8(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::U8(vec))
            }
            CudaStorageSlice::U32(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::U32(vec))
            }
            CudaStorageSlice::I64(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::I64(vec))
            }
            CudaStorageSlice::BF16(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::BF16(vec))
            }
            CudaStorageSlice::F16(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::F16(vec))
            }
            CudaStorageSlice::F32(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::F32(vec))
            }
            CudaStorageSlice::F64(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::F64(vec))
            }
            CudaStorageSlice::F8E4M3(slice) => {
                let single_slice = slice.slice(offset..offset + 1);
                let vec = single_slice.stream().memcpy_dtov(&single_slice).w()?;
                Ok(CpuStorage::F8E4M3(vec))
            }
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        }
    }
}
