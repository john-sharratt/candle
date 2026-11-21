//! The device-or-host storage behind a quantized tensor, with the byte-level
//! reads and writes that the KV arena code uses on it.
//!
//! Ranged reads copy only the span they were asked for. Copying the whole arena
//! across PCIe to keep a slice of it was the cost these methods exist to remove.
//! Byte writes and quantize-into calls take a byte offset into the storage, and
//! each checks that the destination has room before it writes. Several of the
//! quantize and dequantize paths exist only on CUDA, and they are refused on CPU
//! with an error that names the device.

use super::*;

impl Device {
    pub(super) fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QStorage> {
        match self {
            Device::Cpu => {
                let storage = dtype.cpu_zeros(elem_count);
                Ok(QStorage::Cpu(storage))
            }
            Device::Metal(metal) => {
                let storage = metal::QMetalStorage::zeros(metal, elem_count, dtype)?;
                Ok(QStorage::Metal(storage))
            }
            Device::Cuda(cuda) => {
                let storage = cuda::QCudaStorage::zeros(cuda, elem_count, dtype)?;
                Ok(QStorage::Cuda(storage))
            }
        }
    }
}

pub enum QStorage {
    Cpu(Box<dyn QuantizedType>),
    Metal(metal::QMetalStorage),
    Cuda(cuda::QCudaStorage),
}

impl Clone for QStorage {
    fn clone(&self) -> Self {
        match self {
            QStorage::Cpu(storage) => QStorage::Cpu(storage.clone_box()),
            QStorage::Metal(storage) => QStorage::Metal(storage.clone()),
            QStorage::Cuda(storage) => QStorage::Cuda(storage.clone()),
        }
    }
}

impl QStorage {
    pub(super) fn block_size(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.block_size(),
            QStorage::Metal(storage) => storage.dtype().block_size(),
            QStorage::Cuda(storage) => storage.dtype().block_size(),
        }
    }

    pub(super) fn dtype(&self) -> GgmlDType {
        match self {
            QStorage::Cpu(storage) => storage.dtype(),
            QStorage::Metal(storage) => storage.dtype(),
            QStorage::Cuda(storage) => storage.dtype(),
        }
    }

    pub(super) fn device(&self) -> Device {
        match self {
            QStorage::Cpu(_storage) => Device::Cpu,
            QStorage::Metal(storage) => Device::Metal(storage.device().clone()),
            QStorage::Cuda(storage) => Device::Cuda(storage.device().clone()),
        }
    }

    pub(super) fn size_in_bytes(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.storage_size_in_bytes(),
            QStorage::Metal(storage) => storage.storage_size_in_bytes(),
            QStorage::Cuda(storage) => storage.storage_size_in_bytes(),
        }
    }

    pub(super) fn quantize(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?);
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => storage.quantize(src)?,
            (QStorage::Cuda(storage), Storage::Cuda(src)) => storage.quantize(src)?,
            _ => crate::bail!("Invalid dequantize storage locations do not match"),
        }
        Ok(())
    }

    pub(super) fn dequantize(&self, elem_count: usize) -> Result<Storage> {
        match self {
            QStorage::Cpu(storage) => Ok(Storage::Cpu(storage.dequantize(elem_count)?)),
            QStorage::Metal(storage) => Ok(Storage::Metal(storage.dequantize(elem_count)?)),
            QStorage::Cuda(storage) => Ok(Storage::Cuda(storage.dequantize(elem_count)?)),
        }
    }

    pub(super) fn data(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            QStorage::Cpu(storage) => {
                let data_ptr = storage.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::from(data))
            }
            QStorage::Cuda(storage) => {
                let vec = storage.data()?;
                Ok(Cow::from(vec))
            }
            QStorage::Metal(_) => {
                crate::bail!("data() not implemented for Metal storage");
            }
        }
    }

    /// Read just `range` bytes of the storage's raw quantized data.
    ///
    /// Backend-specific behaviour:
    /// - **CPU**: zero-copy `Cow::Borrowed` of the underlying slice.
    /// - **CUDA**: single ranged `cuMemcpyDtoH` copying only
    ///   `range.len()` bytes — does **not** copy the whole arena and
    ///   then slice on the CPU side.
    /// - **Metal**: ranged blit copying only the requested span.
    ///
    /// Use this everywhere a caller previously did
    /// `&storage.data()?[range]` — the slice form copies the entire
    /// arena over PCIe just to discard most of it.
    pub(super) fn data_range(&self, range: std::ops::Range<usize>) -> Result<Cow<'_, [u8]>> {
        match self {
            QStorage::Cpu(storage) => {
                let data_ptr = storage.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                if range.end > size_in_bytes {
                    crate::bail!(
                        "data_range: range {:?} exceeds storage byte_len {}",
                        range,
                        size_in_bytes,
                    );
                }
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::Borrowed(&data[range]))
            }
            QStorage::Cuda(storage) => {
                let vec = storage.data_range(range)?;
                Ok(Cow::Owned(vec))
            }
            QStorage::Metal(storage) => Ok(Cow::Owned(storage.data_range(range)?)),
        }
    }

    /// Get the CUDA device pointer for the raw quantized data.
    /// Returns None for non-CUDA storage.
    #[cfg(feature = "cuda")]
    pub(super) fn cuda_data_ptr(&self) -> Option<u64> {
        match self {
            QStorage::Cuda(storage) => Some(storage.data_ptr()),
            _ => None,
        }
    }

    /// Get a reference to the inner [`QCudaStorage`], if CUDA-backed.
    #[cfg(feature = "cuda")]
    pub fn as_cuda(&self) -> Option<&cuda::QCudaStorage> {
        match self {
            QStorage::Cuda(s) => Some(s),
            _ => None,
        }
    }

    /// Get a mutable CUDA device pointer for the raw quantized data.
    /// Returns None for non-CUDA storage.
    #[cfg(feature = "cuda")]
    pub(super) fn cuda_data_ptr_mut(&mut self) -> Option<u64> {
        match self {
            QStorage::Cuda(storage) => Some(storage.data_ptr_mut()),
            _ => None,
        }
    }

    /// Copy bytes from `src` into `self` at the given byte offset.
    ///
    /// # Arguments
    /// * `src` - Source storage to copy from
    /// * `byte_offset` - Byte offset in destination where copy begins
    ///
    /// # Errors
    /// Returns an error if:
    /// - Storages are on different devices
    /// - Byte offset + src size exceeds destination size
    pub(super) fn slice_scatter(&mut self, src: &QStorage, byte_offset: usize) -> Result<()> {
        let src_size = src.size_in_bytes();
        let dst_size = self.size_in_bytes();
        if byte_offset + src_size > dst_size {
            crate::bail!(
                "slice_scatter: source ({} bytes) at offset {} exceeds destination ({} bytes)",
                src_size,
                byte_offset,
                dst_size
            )
        }

        match (self, src) {
            (QStorage::Cpu(dst), QStorage::Cpu(src)) => {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_ptr() as *mut u8;
                unsafe {
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr.add(byte_offset), src_size);
                }
                Ok(())
            }
            #[cfg(feature = "cuda")]
            (QStorage::Cuda(dst), QStorage::Cuda(src)) => {
                let device = dst.device().clone();
                let src_view = src.bytes();
                let mut dst_view = dst
                    .bytes_mut()
                    .slice_mut(byte_offset..byte_offset + src_size);
                device.memcpy_dtod(&src_view, &mut dst_view)?;
                Ok(())
            }
            #[cfg(feature = "metal")]
            (QStorage::Metal(_dst), QStorage::Metal(_src)) => {
                crate::bail!("slice_scatter not yet implemented for Metal")
            }
            _ => crate::bail!("slice_scatter: device mismatch between source and destination"),
        }
    }

    /// Copy a byte range from one QStorage to another without dequantization.
    ///
    /// Both source and destination must be on the same device. Copies `byte_len`
    /// bytes from `src` starting at `src_byte_offset` into `self` at `dst_byte_offset`.
    pub(super) fn slice_range_copy(
        &mut self,
        src: &QStorage,
        src_byte_offset: usize,
        dst_byte_offset: usize,
        byte_len: usize,
    ) -> Result<()> {
        let src_size = src.size_in_bytes();
        let dst_size = self.size_in_bytes();
        if src_byte_offset + byte_len > src_size {
            crate::bail!(
                "slice_range_copy: src range {}..{} exceeds src size {}",
                src_byte_offset,
                src_byte_offset + byte_len,
                src_size
            )
        }
        if dst_byte_offset + byte_len > dst_size {
            crate::bail!(
                "slice_range_copy: dst range {}..{} exceeds dst size {}",
                dst_byte_offset,
                dst_byte_offset + byte_len,
                dst_size
            )
        }

        match (self, src) {
            (QStorage::Cpu(dst), QStorage::Cpu(src)) => {
                let src_ptr = src.as_ptr();
                let dst_ptr = dst.as_ptr() as *mut u8;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_ptr.add(src_byte_offset),
                        dst_ptr.add(dst_byte_offset),
                        byte_len,
                    );
                }
                Ok(())
            }
            #[cfg(feature = "cuda")]
            (QStorage::Cuda(dst), QStorage::Cuda(src)) => {
                let device = dst.device().clone();
                let src_view = src
                    .bytes()
                    .slice(src_byte_offset..src_byte_offset + byte_len);
                let mut dst_view = dst
                    .bytes_mut()
                    .slice_mut(dst_byte_offset..dst_byte_offset + byte_len);
                device.memcpy_dtod(&src_view, &mut dst_view)?;
                Ok(())
            }
            // GPU→CPU: DtoH byte copy (warm-tier migration).
            #[cfg(feature = "cuda")]
            (QStorage::Cpu(dst), QStorage::Cuda(src)) => {
                let src_view = src
                    .bytes()
                    .slice(src_byte_offset..src_byte_offset + byte_len);
                let cpu_bytes: Vec<u8> = src.device().memcpy_dtov(&src_view)?;
                let dst_ptr = dst.as_ptr() as *mut u8;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        cpu_bytes.as_ptr(),
                        dst_ptr.add(dst_byte_offset),
                        byte_len,
                    );
                }
                Ok(())
            }
            // CPU→GPU: HtoD byte copy (hot-tier re-injection).
            #[cfg(feature = "cuda")]
            (QStorage::Cuda(dst), QStorage::Cpu(src)) => {
                let device = dst.device().clone();
                let src_ptr = src.as_ptr();
                let src_bytes =
                    unsafe { std::slice::from_raw_parts(src_ptr.add(src_byte_offset), byte_len) };
                let mut dst_view = dst
                    .bytes_mut()
                    .slice_mut(dst_byte_offset..dst_byte_offset + byte_len);
                device.memcpy_htod(src_bytes, &mut dst_view)?;
                Ok(())
            }
            #[cfg(feature = "metal")]
            (QStorage::Metal(_dst), QStorage::Metal(_src)) => {
                crate::bail!("slice_range_copy not yet implemented for Metal")
            }
            _ => crate::bail!("slice_range_copy: device mismatch between source and destination"),
        }
    }

    /// Quantize f32 data directly into this storage at the given byte offset.
    ///
    /// This is optimized for GPU by using CUDA quantization kernels that write
    /// directly to the destination buffer, avoiding intermediate allocations.
    ///
    /// # Arguments
    /// * `src` - Source float storage
    /// * `elem_count` - Number of f32 elements to quantize
    /// * `byte_offset` - Byte offset in destination where quantized data starts
    ///
    /// # Errors
    /// Returns an error if storages are on different devices or CPU is used.
    #[cfg(feature = "cuda")]
    pub(super) fn quantize_into(
        &mut self,
        src: &Storage,
        elem_count: usize,
        byte_offset: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cuda(dst), Storage::Cuda(src)) => {
                dst.quantize_into(src, elem_count, byte_offset)
            }
            (QStorage::Cpu(dst), Storage::Cpu(src)) => {
                let mut tmp = dst.dtype().cpu_zeros(elem_count);
                tmp.from_float(src.as_slice::<f32>()?);
                let src_size = tmp.storage_size_in_bytes();
                let dst_size = dst.storage_size_in_bytes();
                if byte_offset + src_size > dst_size {
                    crate::bail!(
                        "quantize_into: source ({} bytes) at offset {} exceeds destination ({} bytes)",
                        src_size,
                        byte_offset,
                        dst_size
                    )
                }
                let src_ptr = tmp.as_ptr();
                let dst_ptr = dst.as_ptr() as *mut u8;
                unsafe {
                    std::ptr::copy_nonoverlapping(src_ptr, dst_ptr.add(byte_offset), src_size);
                }
                Ok(())
            }
            #[cfg(feature = "metal")]
            (QStorage::Metal(_), _) => {
                crate::bail!("quantize_into: Metal quantization not yet implemented")
            }
            _ => crate::bail!("quantize_into: device mismatch between source and destination"),
        }
    }

    /// Quantize f32 data with fused transpose from [H, T, D] to [H, D, T] layout.
    ///
    /// This fuses the memory layout transformation with quantization to avoid
    /// intermediate allocations. Used for KV cache quantization.
    ///
    /// # Arguments
    /// * `src` - Source float storage with shape [n_head, chunk_size, head_dim]
    /// * `n_head` - Number of heads
    /// * `chunk_size` - Number of tokens (must be 32)
    /// * `head_dim` - Dimension per head
    /// * `byte_offset` - Byte offset in destination
    ///
    /// # Errors
    /// Returns an error if dtype is not Q4_0/Q8_0 or devices mismatch.
    #[cfg(feature = "cuda")]
    pub(super) fn quantize_transposed_into(
        &mut self,
        src: &Storage,
        n_head: usize,
        chunk_size: usize,
        head_dim: usize,
        byte_offset: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cuda(dst), Storage::Cuda(src)) => {
                dst.quantize_transposed_into(src, n_head, chunk_size, head_dim, byte_offset)
            }
            (QStorage::Cpu(_), _) => {
                crate::bail!(
                    "quantize_transposed_into: CPU not supported, use separate transpose + quantize"
                )
            }
            #[cfg(feature = "metal")]
            (QStorage::Metal(_), _) => {
                crate::bail!("quantize_transposed_into: Metal not yet implemented")
            }
            _ => crate::bail!("quantize_transposed_into: device mismatch"),
        }
    }

    /// Dequantize a range of elements directly into a destination storage buffer.
    ///
    /// # Errors
    /// Returns an error if storages are on different devices or CPU is used.
    #[cfg(feature = "cuda")]
    pub(super) fn dequantize_into(
        &self,
        dst: &mut Storage,
        elem_count: usize,
        src_byte_offset: usize,
        dst_elem_offset: usize,
    ) -> Result<()> {
        match (self, dst) {
            (QStorage::Cuda(src), Storage::Cuda(dst)) => {
                src.dequantize_into(dst, elem_count, src_byte_offset, dst_elem_offset)
            }
            (QStorage::Cpu(_), _) => {
                crate::bail!("dequantize_into: CPU not supported, use dequantize()")
            }
            #[cfg(feature = "metal")]
            (QStorage::Metal(_), _) => {
                crate::bail!("dequantize_into: Metal not yet implemented")
            }
            _ => crate::bail!("dequantize_into: device mismatch between source and destination"),
        }
    }
}
