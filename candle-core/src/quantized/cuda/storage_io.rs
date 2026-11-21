//! Creates, copies and reads quantized storage, in the kinds that other code
//! builds on.
//!
//! Storage is zeroed, left uninitialised for a kernel that will fill all of it, or
//! allocated from an arena so its bytes come from a budget already paid for.
//! Host-mapped storage lets kernels read quantized data across PCIe without using
//! VRAM, at PCIe bandwidth. Copies into an existing device slot go on a named
//! stream, so a slot can be overwritten while compute runs, and a range read
//! transfers only the bytes asked for. Uninitialised storage is wrong for a
//! q-matmul kernel, which reads the padding past the payload as zeros only when the
//! storage was zeroed.

use super::*;

impl QCudaStorage {
    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes)?;
        Ok(QCudaStorage {
            data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            }),
            device: device.clone(),
            dtype,
            backing: Backing::Owned,
        })
    }

    /// Same as [`QCudaStorage::zeros`], but the bytes are left as they are.
    ///
    /// For storage a kernel fills immediately, where the zero-fill is a full
    /// memset of the buffer discarded microseconds later by the write that
    /// follows it.
    ///
    /// # Safety
    ///
    /// The caller must write the whole `len`-byte data region before anything
    /// reads it. The `MATRIX_ROW_PADDING` tail past `len` is uninitialised too,
    /// so this is wrong for any storage handed to a q-matmul kernel — those
    /// over-read into the padding by design, and [`QCudaStorage::zeros`] is
    /// what makes that read a defined zero.
    pub unsafe fn uninit(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        unsafe { Self::uninit_from(device, el_count, dtype, Backing::Owned) }
    }

    /// [`Self::uninit`] with the storage taken from `origin`'s arena.
    ///
    /// For quantized staging that is written and consumed inside one scope — the
    /// embedding gather, whose bytes are read once by the dequantize that
    /// immediately follows. Carving that from the wave costs no VRAM at all: the
    /// span is already reserved, so the bytes come out of a budget that has
    /// been paid for whether or not anything uses it.
    ///
    /// # Safety
    /// As [`Self::uninit`]: the storage is uninitialised, so every byte a reader
    /// touches must be written first.
    pub unsafe fn uninit_from(
        device: &CudaDevice,
        el_count: usize,
        dtype: GgmlDType,
        origin: Backing,
    ) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let (inner, backing) =
            unsafe { alloc_inheriting::<u8>(device, padded_size_in_bytes, origin)? };
        Ok(QCudaStorage {
            data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            }),
            device: device.clone(),
            dtype,
            backing,
        })
    }

    /// Create a `QCudaStorage` backed by host-mapped (pinned) memory.
    ///
    /// The quantized data is copied into a `cudaHostAlloc`-allocated buffer
    /// with `CU_MEMHOSTALLOC_DEVICEMAP`. CUDA kernels access this memory
    /// transparently over PCIe — **no VRAM is consumed**.
    ///
    /// Returns `(storage, guard)`. The caller must keep `guard` alive for the
    /// lifetime of the storage; dropping it frees the pinned host buffer.
    ///
    /// This is the correct overflow path when VRAM budget is exceeded: tensors
    /// still work with all CUDA matmul kernels, just at PCIe bandwidth.
    pub fn from_host_mapped(
        data: &[u8],
        elem_count: usize,
        dtype: GgmlDType,
        device: &CudaDevice,
    ) -> Result<(Self, HostMappedAlloc)> {
        let size_in_bytes = data.len();
        let padded_size_in_bytes =
            ceil_div(elem_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let (host_ptr, _dev_ptr, guard) = alloc_host_mapped(padded_size_in_bytes)?;

        // Copy quantized data into the pinned host buffer
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), host_ptr, size_in_bytes);
            // Zero the padding region
            if padded_size_in_bytes > size_in_bytes {
                std::ptr::write_bytes(
                    host_ptr.add(size_in_bytes),
                    0u8,
                    padded_size_in_bytes - size_in_bytes,
                );
            }
        }

        // Wrap the pinned host buffer as a CudaSlice via a device H2D copy: the
        // QCudaStorage API needs an owned CudaSlice, and cudarc has no way to wrap an
        // external device pointer (from cuMemHostGetDevicePointer) into a CudaSlice
        // without ownership, so the mapped-host zero-copy path is not expressible here.
        // The HostMappedAlloc guard keeps the pinned buffer alive alongside the VRAM copy.
        let pinned_slice = unsafe { std::slice::from_raw_parts(host_ptr, size_in_bytes) };
        let mut inner = unsafe { device.alloc::<u8>(padded_size_in_bytes)? };
        device.memcpy_htod(pinned_slice, &mut inner.slice_mut(..size_in_bytes))?;

        let storage = QCudaStorage {
            data: std::mem::ManuallyDrop::new(PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            }),
            device: device.clone(),
            dtype,
            backing: Backing::Owned,
        };
        Ok((storage, guard))
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn byte_len(&self) -> usize {
        self.data.len
    }

    pub fn bytes(&self) -> CudaView<'_, u8> {
        self.data.inner.slice(..self.data.len)
    }

    pub fn bytes_mut(&mut self) -> cudarc::driver::CudaViewMut<'_, u8> {
        // `len` is read out first: `data` is `ManuallyDrop`, so both halves of
        // this expression go through one `DerefMut` borrow rather than being
        // disjoint field accesses.
        let len = self.data.len;
        self.data.inner.slice_mut(..len)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len
    }

    /// Get the device pointer for the raw quantized data.
    /// This is used by paged attention kernels that need direct GPU access.
    ///
    /// # Safety
    /// The returned pointer is only valid while the QCudaStorage is alive.
    /// The caller must ensure the stream is synchronized before using the pointer.
    pub fn data_ptr(&self) -> u64 {
        let stream = self.device.cuda_stream();
        let (ptr, _guard) = self.data.inner.device_ptr(&stream);
        ptr
    }

    /// Get a mutable device pointer for the raw quantized data.
    /// This is used by batched quantization kernels that write directly to GPU memory.
    ///
    /// # Safety
    /// The returned pointer is only valid while the QCudaStorage is alive.
    /// The caller must ensure proper synchronization and exclusive access.
    pub fn data_ptr_mut(&mut self) -> u64 {
        let stream = self.device.cuda_stream();
        let (ptr, _guard) = self.data.inner.device_ptr_mut(&stream);
        ptr
    }

    /// Copy the raw quantized data from GPU to CPU as a Vec<u8>
    pub fn data(&self) -> Result<Vec<u8>> {
        let stream = self.device.cuda_stream();
        // Only copy the actual data, not the padding
        let slice = self.data.inner.slice(0..self.data.len);
        let cpu_data = stream.memcpy_dtov(&slice).map_err(crate::Error::wrap)?;
        Ok(cpu_data)
    }

    /// Copy a byte range of the quantized data from GPU to CPU as a `Vec<u8>`.
    ///
    /// Issues a single `cuMemcpyDtoH` for exactly `range.len()` bytes
    /// starting at `range.start` — no kernel, no full-buffer copy.  Use
    /// this whenever the caller only needs a slice; calling
    /// [`Self::data`] and slicing the result on the CPU side would
    /// transfer the entire arena over PCIe just to throw most of it away.
    pub fn data_range(&self, range: std::ops::Range<usize>) -> Result<Vec<u8>> {
        if range.end > self.data.len {
            crate::bail!(
                "data_range: range {:?} exceeds storage byte_len {}",
                range,
                self.data.len,
            );
        }
        let stream = self.device.cuda_stream();
        let slice = self.data.inner.slice(range);
        let cpu_data = stream.memcpy_dtov(&slice).map_err(crate::Error::wrap)?;
        Ok(cpu_data)
    }

    /// Overwrite the contents of this VRAM buffer from host bytes.
    ///
    /// Used by the expert LRU cache to swap expert weights into pre-allocated
    /// VRAM slots without reallocating. The source slice must be exactly
    /// `self.storage_size_in_bytes()` bytes.
    pub fn copy_from_host(&mut self, src: &[u8]) -> Result<()> {
        if src.len() != self.data.len {
            crate::bail!(
                "copy_from_host: expected {} bytes, got {}",
                self.data.len,
                src.len()
            );
        }
        self.device
            .memcpy_htod(src, &mut self.data.inner.slice_mut(..src.len()))?;
        Ok(())
    }

    /// Like [`copy_from_host`](Self::copy_from_host), but issues the H2D
    /// copy on a secondary [`CudaStream`] for DMA overlap.
    ///
    /// The caller must synchronise (event + wait) before using the buffer
    /// on the device's main stream.
    pub fn copy_from_host_on_stream(
        &mut self,
        src: &[u8],
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        if src.len() != self.data.len {
            crate::bail!(
                "copy_from_host_on_stream: expected {} bytes, got {}",
                self.data.len,
                src.len()
            );
        }
        // Goes to the STREAM's `memcpy_htod`, not the device's, so the guard on
        // `CudaDevice::memcpy_htod` never sees it. It writes a whole quantized
        // buffer — an expert slot, say — at a caller-computed destination, so an
        // off-by-one slot index here
        // replaces a contiguous run of resident weights with another expert's
        // bytes: finite, plausibly-shaped, and wrong.
        #[cfg(feature = "tensor-assert")]
        {
            use cudarc::driver::DevicePtrMut;
            let mut dst = self.data.inner.slice_mut(..src.len());
            let (base, _g) = dst.device_ptr_mut(stream);
            crate::readonly_regions::forbid_write(
                "QCudaStorage::copy_from_host_on_stream",
                base,
                src.len(),
            );
        }
        stream
            .memcpy_htod(src, &mut self.data.inner.slice_mut(..src.len()))
            .map_err(crate::Error::wrap)?;
        Ok(())
    }

    /// Copy raw quantized VRAM bytes to a pre-existing host buffer on a stream.
    ///
    /// When `dst` is backed by pinned memory (`cuMemAllocHost`), the copy is
    /// truly asynchronous — the CPU returns immediately and the DMA engine
    /// handles the transfer.  The layer stream's pack build reads each
    /// repacked layer back to the host this way.
    ///
    /// `dst` must be at least `self.storage_size_in_bytes()` bytes.
    pub fn copy_to_host_on_stream(
        &self,
        dst: &mut [u8],
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        let len = self.data.len;
        if dst.len() < len {
            crate::bail!(
                "copy_to_host_on_stream: dst too small ({} < {})",
                dst.len(),
                len,
            );
        }
        let src_slice = self.data.inner.slice(0..len);
        stream
            .memcpy_dtoh(&src_slice, &mut dst[..len])
            .map_err(crate::Error::wrap)?;
        Ok(())
    }
}
