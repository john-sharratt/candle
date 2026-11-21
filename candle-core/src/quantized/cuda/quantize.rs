//! Quantizes float data into block formats on the GPU, including fused transposes
//! that write KV chunks in token-oriented layout.
//!
//! Each entry writes into a destination the caller sized. The transposed paths
//! turn channel-oriented float blocks into token-oriented quantized blocks in one
//! launch, so KV migration needs no intermediate buffer. Only the standard
//! 32-element formats take the fused paths, and the chunk size has to match the
//! block size. A format the block kernel does not name is refused rather than left
//! unwritten. KO weights go through a separate kernel that writes the lane-major
//! layout the int8 matmul reads.

use super::*;

pub(super) fn quantize_q8_1(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    elem_count: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let stream = dev.cuda_stream();
    let (src_ptr, _src_guard) = src.device_ptr(&stream);
    let (dst_ptr, _dst_guard) = dst.device_ptr_mut(&stream);
    unsafe {
        run_quantize_q8_1(
            src_ptr as *const f32,
            dst_ptr as *mut std::ffi::c_void,
            elem_count as i32,
            ky as i32,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }
    Ok(())
}

/// Quantize f32 data to any supported quantized format on GPU.
///
/// # Arguments
/// * `src` - Source f32 data view
/// * `dst` - Destination buffer (must be pre-allocated with correct size)
/// * `elem_count` - Total number of f32 elements to quantize
/// * `dtype` - Target quantized dtype (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K, QAWQ, QAWQG64)
/// * `dev` - CUDA device
///
/// # Returns
/// Result indicating success or error
pub fn quantize_to_dtype(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    elem_count: usize,
    dtype: GgmlDType,
    dev: &CudaDevice,
) -> Result<()> {
    require_block_quantize_kernel(dtype)?;
    let qtype = dtype_to_qtype(dtype)?;
    let stream = dev.cuda_stream();
    let (src_ptr, _src_guard) = src.device_ptr(&stream);
    let (dst_ptr, _dst_guard) = dst.device_ptr_mut(&stream);
    unsafe {
        run_quantize_block(
            src_ptr as *const f32,
            dst_ptr as *mut std::ffi::c_void,
            elem_count as i32,
            qtype as i32,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }
    Ok(())
}

/// Quantize f32 data with fused transpose from [H, T, D] to [H, D, T] layout on GPU.
///
/// This fuses the memory layout transformation with quantization to avoid
/// intermediate allocations. Used for KV cache quantization where:
/// - Input layout: [n_head, chunk_size, head_dim] - channel-oriented float
/// - Output layout: [n_head, head_dim, chunk_size] - token-oriented quant
///
/// # Arguments
/// * `src` - Source f32 tensor with shape [n_head, chunk_size, head_dim]
/// * `dst` - Destination buffer (must be pre-allocated for n_head * head_dim quantized blocks)
/// * `n_head` - Number of KV heads
/// * `chunk_size` - Number of tokens (must be 32 for standard quants)
/// * `head_dim` - Dimension per head
/// * `dtype` - Target quantized dtype (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
/// * `dev` - CUDA device
///
/// # Returns
/// Result indicating success or error
pub fn quantize_transposed_to_dtype(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    n_head: usize,
    chunk_size: usize,
    head_dim: usize,
    dtype: GgmlDType,
    dev: &CudaDevice,
) -> Result<()> {
    // Standard 32-element formats support fused transpose+quantize
    let qtype = match dtype {
        GgmlDType::Q4_0 => QType::Q4_0,
        GgmlDType::Q4_1 => QType::Q4_1,
        GgmlDType::Q5_0 => QType::Q5_0,
        GgmlDType::Q5_1 => QType::Q5_1,
        GgmlDType::Q8_0 => QType::Q8_0,
        GgmlDType::Q8_1 => QType::Q8_1,
        GgmlDType::Q4_KS => QType::Q4_KS,
        GgmlDType::Q8_KS => QType::Q8_KS,
        GgmlDType::Q2_0 => QType::Q2_0,
        GgmlDType::Q3_0 => QType::Q3_0,
        GgmlDType::R16 => QType::R16,
        GgmlDType::Q0 => QType::Q0,
        GgmlDType::Q1_S => QType::Q1_S,
        GgmlDType::Q2_S => QType::Q2_S,
        GgmlDType::Q2_A => QType::Q2_A,
        GgmlDType::Q2_1 => QType::Q2_1,
        GgmlDType::Q3_1 => QType::Q3_1,
        _ => crate::bail!("quantize_transposed_to_dtype: only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q4_KS/Q8_KS supported, got {:?}", dtype),
    };

    // Validate chunk_size matches block size (all standard quants use 32)
    if chunk_size != 32 {
        crate::bail!(
            "quantize_transposed_to_dtype: chunk_size must be 32 for standard quants, got {}",
            chunk_size
        );
    }

    let stream = dev.cuda_stream();
    let (src_ptr, _src_guard) = src.device_ptr(&stream);
    let (dst_ptr, _dst_guard) = dst.device_ptr_mut(&stream);
    unsafe {
        // Use batched version with num_chunks=1
        run_quantize_transposed_batched(
            src_ptr as *const f32,
            dst_ptr as *mut std::ffi::c_void,
            std::ptr::null(), // No src offsets (contiguous)
            std::ptr::null(), // No dst offsets (contiguous)
            1,                // Single chunk
            n_head as i32,
            chunk_size as i32,
            head_dim as i32,
            qtype as i32,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }
    Ok(())
}

/// Batched quantize f32 data with fused transpose for multiple chunks.
///
/// Processes multiple chunks in a single kernel launch for efficient KV cache migration.
/// Fuses the memory layout transformation with quantization:
/// - Input layout: [num_chunks, n_head, chunk_size, head_dim] - channel-oriented float
/// - Output layout: [num_chunks, n_head, head_dim] Q blocks - token-oriented quant
///
/// # Arguments
/// * `src` - Source f32 data pointer (contiguous for all chunks or use offsets)
/// * `dst` - Destination quantized data pointer
/// * `src_offsets` - Per-chunk element offsets into src (None for contiguous)
/// * `dst_offsets` - Per-chunk byte offsets into dst (None for contiguous)
/// * `num_chunks` - Number of chunks to process
/// * `n_head` - Number of KV heads per chunk
/// * `chunk_size` - Tokens per chunk (must be 32 for standard quants)
/// * `head_dim` - Dimension per head
/// * `dtype` - Target dtype (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
/// * `dev` - CUDA device
///
/// # Safety
/// The caller must ensure that `src` and `dst` are valid GPU pointers with
/// sufficient allocated memory for the operation.
#[allow(clippy::too_many_arguments)]
pub unsafe fn quantize_transposed_batched_to_dtype(
    src: *const f32,
    dst: *mut u8,
    src_offsets: Option<&CudaSlice<i32>>,
    dst_offsets: Option<&CudaSlice<i32>>,
    num_chunks: usize,
    n_head: usize,
    chunk_size: usize,
    head_dim: usize,
    dtype: GgmlDType,
    dev: &CudaDevice,
) -> Result<()> {
    // Only standard 32-element quants are supported
    let qtype = match dtype {
        GgmlDType::Q4_0 => QType::Q4_0,
        GgmlDType::Q4_1 => QType::Q4_1,
        GgmlDType::Q5_0 => QType::Q5_0,
        GgmlDType::Q5_1 => QType::Q5_1,
        GgmlDType::Q8_0 => QType::Q8_0,
        GgmlDType::Q8_1 => QType::Q8_1,
        GgmlDType::Q4_KS => QType::Q4_KS,
        GgmlDType::Q8_KS => QType::Q8_KS,
        GgmlDType::Q2_0 => QType::Q2_0,
        GgmlDType::Q3_0 => QType::Q3_0,
        GgmlDType::R16 => QType::R16,
        GgmlDType::Q0 => QType::Q0,
        GgmlDType::Q1_S => QType::Q1_S,
        GgmlDType::Q2_S => QType::Q2_S,
        GgmlDType::Q2_A => QType::Q2_A,
        GgmlDType::Q2_1 => QType::Q2_1,
        GgmlDType::Q3_1 => QType::Q3_1,
        GgmlDType::Q0_V => QType::Q0_V,
        GgmlDType::Q1_A => QType::Q1_A,
        GgmlDType::Q0_X => QType::Q0_X,
        GgmlDType::Q0_M2 => QType::Q0_M2,
        GgmlDType::Q0_M4 => QType::Q0_M4,
        _ => crate::bail!(
            "quantize_transposed_batched: only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q4_KS/Q8_KS supported, got {:?}",
            dtype
        ),
    };

    // Validate chunk_size matches block size
    if chunk_size != 32 {
        crate::bail!(
            "quantize_transposed_batched: chunk_size must be 32, got {}",
            chunk_size
        );
    }

    let stream = dev.cuda_stream();

    // Get src_offsets pointer (or null for contiguous)
    let src_off_ptr = if let Some(offsets) = src_offsets {
        let (ptr, _guard) = offsets.device_ptr(&stream);
        ptr as *const i32
    } else {
        std::ptr::null()
    };

    // Get dst_offsets pointer (or null for contiguous)
    let dst_off_ptr = if let Some(offsets) = dst_offsets {
        let (ptr, _guard) = offsets.device_ptr(&stream);
        ptr as *const i32
    } else {
        std::ptr::null()
    };

    run_quantize_transposed_batched(
        src,
        dst as *mut std::ffi::c_void,
        src_off_ptr,
        dst_off_ptr,
        num_chunks as i32,
        n_head as i32,
        chunk_size as i32,
        head_dim as i32,
        qtype as i32,
        stream.cu_stream() as *mut std::ffi::c_void,
    );
    Ok(())
}

/// Convert candle DType to kernel SrcDType code.
/// Returns: 0=F32, 1=F16, 2=BF16, 3=F8E4M3
/// Map a candle `DType` to the integer src-dtype code expected by the
/// `transpose_quant_batch_typed` CUDA dispatcher. The codes match
/// `GgmlDType` / `QType` / `ArenaFormat` values so the Rust side never
/// has to translate between two numbering schemes — the CUDA enum
/// `SrcDType` in `transpose_batch.cuh` uses the same values.
pub fn dtype_to_src_dtype_code(dtype: crate::DType) -> Result<i32> {
    match dtype {
        crate::DType::F32 => Ok(GgmlDType::F32 as i32),   // 0
        crate::DType::F16 => Ok(GgmlDType::F16 as i32),   // 1
        crate::DType::BF16 => Ok(GgmlDType::BF16 as i32), // 2
        crate::DType::F8E4M3 => Ok(GgmlDType::F8E4M3 as i32), // 34
        _ => crate::bail!(
            "quantize_transposed: source dtype must be F32/F16/BF16/F8E4M3, got {:?}",
            dtype
        ),
    }
}

/// Convert candle DType to GgmlDType for float source types.
pub fn dtype_to_ggml_float(dtype: crate::DType) -> Result<GgmlDType> {
    match dtype {
        crate::DType::F32 => Ok(GgmlDType::F32),
        crate::DType::F16 => Ok(GgmlDType::F16),
        crate::DType::BF16 => Ok(GgmlDType::BF16),
        _ => crate::bail!("dtype_to_ggml_float: expected float dtype, got {:?}", dtype),
    }
}

/// Batched quantize typed data with fused transpose for multiple chunks.
///
/// Like `quantize_transposed_batched_to_dtype` but accepts any supported source dtype
/// (F32, F16, BF16, F8E4M3) and converts inline in the kernel.
///
/// # Arguments
/// * `src` - Source data pointer (F32/F16/BF16/F8E4M3)
/// * `src_dtype` - Source data type
/// * `dst` - Destination quantized data pointer
/// * `src_offsets` - Per-chunk element offsets into src (None for contiguous)
/// * `dst_offsets` - Per-chunk byte offsets into dst (None for contiguous)
/// * `num_chunks` - Number of chunks to process
/// * `n_head` - Number of KV heads per chunk
/// * `chunk_size` - Tokens per chunk (must be 32 for standard quants)
/// * `head_dim` - Dimension per head
/// * `dtype` - Target quantized dtype (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
/// * `dev` - CUDA device
///
/// # Safety
/// The caller must ensure that `src` and `dst` are valid GPU pointers with
/// sufficient allocated memory for the operation.
#[allow(clippy::too_many_arguments)]
pub unsafe fn quantize_transposed_batched_typed(
    src: *const std::ffi::c_void,
    src_dtype: crate::DType,
    dst: *mut u8,
    src_offsets: Option<&CudaSlice<i32>>,
    dst_offsets: Option<&CudaSlice<i32>>,
    num_chunks: usize,
    n_head: usize,
    chunk_size: usize,
    head_dim: usize,
    dtype: GgmlDType,
    dev: &CudaDevice,
) -> Result<()> {
    // Only standard 32-element quants are supported
    let qtype = match dtype {
        GgmlDType::Q4_0 => QType::Q4_0,
        GgmlDType::Q4_1 => QType::Q4_1,
        GgmlDType::Q5_0 => QType::Q5_0,
        GgmlDType::Q5_1 => QType::Q5_1,
        GgmlDType::Q8_0 => QType::Q8_0,
        GgmlDType::Q8_1 => QType::Q8_1,
        GgmlDType::Q4_KS => QType::Q4_KS,
        GgmlDType::Q8_KS => QType::Q8_KS,
        GgmlDType::Q2_0 => QType::Q2_0,
        GgmlDType::Q3_0 => QType::Q3_0,
        GgmlDType::R16 => QType::R16,
        GgmlDType::Q0 => QType::Q0,
        GgmlDType::Q1_S => QType::Q1_S,
        GgmlDType::Q2_S => QType::Q2_S,
        GgmlDType::Q2_A => QType::Q2_A,
        GgmlDType::Q2_1 => QType::Q2_1,
        GgmlDType::Q3_1 => QType::Q3_1,
        GgmlDType::Q0_V => QType::Q0_V,
        GgmlDType::Q1_A => QType::Q1_A,
        GgmlDType::Q0_X => QType::Q0_X,
        GgmlDType::Q0_M2 => QType::Q0_M2,
        GgmlDType::Q0_M4 => QType::Q0_M4,
        _ => crate::bail!(
            "quantize_transposed_batched: only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q4_KS/Q8_KS supported, got {:?}",
            dtype
        ),
    };

    let src_dtype_code = dtype_to_src_dtype_code(src_dtype)?;

    // Validate chunk_size matches block size
    if chunk_size != 32 {
        crate::bail!(
            "quantize_transposed_batched: chunk_size must be 32, got {}",
            chunk_size
        );
    }

    let stream = dev.cuda_stream();

    // Get src_offsets pointer (or null for contiguous)
    let src_off_ptr = if let Some(offsets) = src_offsets {
        let (ptr, _guard) = offsets.device_ptr(&stream);
        ptr as *const i32
    } else {
        std::ptr::null()
    };

    // Get dst_offsets pointer (or null for contiguous)
    let dst_off_ptr = if let Some(offsets) = dst_offsets {
        let (ptr, _guard) = offsets.device_ptr(&stream);
        ptr as *const i32
    } else {
        std::ptr::null()
    };

    run_quantize_transposed_batched_typed(
        src,
        dst as *mut std::ffi::c_void,
        src_off_ptr,
        dst_off_ptr,
        num_chunks as i32,
        n_head as i32,
        chunk_size as i32,
        head_dim as i32,
        qtype as i32,
        src_dtype_code,
        stream.cu_stream() as *mut std::ffi::c_void,
    );
    Ok(())
}

/// Calculate the required buffer size in bytes for quantizing `elem_count` f32 elements.
///
/// # Arguments
/// * `elem_count` - Number of f32 elements to quantize
/// * `dtype` - Target quantized dtype
///
/// # Returns
/// Required buffer size in bytes
pub fn quantized_size(elem_count: usize, dtype: GgmlDType) -> usize {
    let block_size = dtype.block_size();
    let num_blocks = elem_count.div_ceil(block_size);
    num_blocks * dtype.type_size()
}

/// Whether `run_quantize_block` has an arm for `dtype` — the formats its first
/// switch names (`quantized_dispatcher.cu`). Its `default:` returns without
/// launching, so any other format leaves the destination as it was and
/// reports nothing. The KV formats below 2 bits and R16 are encoded by the
/// palette convert kernel or on the host, never here.
pub(super) fn has_block_quantize_kernel(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q8_1
            | GgmlDType::Q4_KS
            | GgmlDType::Q8_KS
            | GgmlDType::Q2_0
            | GgmlDType::Q3_0
            | GgmlDType::Q2_K
            | GgmlDType::Q3_K
            | GgmlDType::Q4_K
            | GgmlDType::Q5_K
            | GgmlDType::Q6_K
            | GgmlDType::Q8_K
            | GgmlDType::QAWQ
            | GgmlDType::QAWQ_G64
    )
}

/// Refuse a block format `run_quantize_block` would silently skip.
pub(super) fn require_block_quantize_kernel(dtype: GgmlDType) -> Result<()> {
    if !has_block_quantize_kernel(dtype) {
        crate::bail!(
            "{dtype:?} has no device block-quantize kernel; run_quantize_block would \
             leave the destination unwritten"
        )
    }
    Ok(())
}

impl QCudaStorage {
    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.memcpy_dtov(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(data.as_ref(), &mut inner.slice_mut(..data.len()))?;
        zero_padding_tail(&self.device.cuda_stream(), &mut inner, data.len())?;
        // Dispose of the buffer being replaced before overwriting it: `data` is
        // `ManuallyDrop`, so an assignment alone would leak it. A lease must be
        // `leak`ed rather than dropped — its memory belongs to an arena — while
        // an owned buffer is freed.
        // SAFETY: the old value is moved out exactly once and not read again.
        let old = unsafe { std::mem::ManuallyDrop::take(&mut self.data) };
        if matches!(self.backing, Backing::Lease(_)) {
            old.inner.leak();
        } else {
            // Retired, not dropped: a launch recorded before this call may
            // still read the old buffer.
            self.device.retire(old);
        }
        self.data = std::mem::ManuallyDrop::new(PaddedCudaSlice {
            inner,
            len: data.len(),
        });
        // The buffer identity just changed, so the old `backing` no longer
        // describes it. On a storage built by `from_leased_device_ptr` this is
        // load-bearing twice over: the quantized bytes have landed in the fresh
        // allocation rather than the arena slot the lease pointed at, and
        // leaving `Lease` set makes `Drop` `leak()` that fresh allocation —
        // a permanent VRAM leak of `padded_len` per call. `Clone` was already
        // taught to force `Owned` for exactly this reason; this is the other
        // place the buffer is replaced.
        self.backing = Backing::Owned;
        Ok(())
    }

    /// Quantize f32 data directly into this storage at the given byte offset.
    ///
    /// This avoids intermediate allocations by using the GPU quantization kernel
    /// to write directly into the target buffer.
    ///
    /// # Arguments
    /// * `src` - Source f32 CUDA storage
    /// * `elem_count` - Number of f32 elements to quantize
    /// * `byte_offset` - Byte offset in destination where quantized data should be written
    ///
    /// # Safety
    /// The caller must ensure:
    /// - byte_offset is block-aligned (multiple of dtype.type_size())
    /// - byte_offset + quantized_size(elem_count, dtype) <= self.storage_size_in_bytes()
    pub fn quantize_into(
        &mut self,
        src: &CudaStorage,
        elem_count: usize,
        byte_offset: usize,
    ) -> Result<()> {
        // Validate input dtype
        let src_f32 = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => data,
            _ => crate::bail!("quantize_into: only f32 can be quantized"),
        };

        // Validate alignment
        let block_size = self.dtype.block_size();
        let type_size = self.dtype.type_size();
        if !byte_offset.is_multiple_of(type_size) {
            crate::bail!(
                "quantize_into: byte_offset {} not aligned to block type_size {}",
                byte_offset,
                type_size
            );
        }

        // Validate bounds
        let quantized_bytes = quantized_size(elem_count, self.dtype);
        if byte_offset + quantized_bytes > self.data.len {
            crate::bail!(
                "quantize_into: offset {} + size {} exceeds storage size {}",
                byte_offset,
                quantized_bytes,
                self.data.len
            );
        }

        // Get pointers and call kernel
        require_block_quantize_kernel(self.dtype)?;
        let qtype = dtype_to_qtype(self.dtype)?;
        let stream = self.device.cuda_stream();
        let (src_ptr, _src_guard) = src_f32.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = self.data.inner.device_ptr_mut(&stream);

        unsafe {
            run_quantize_block(
                src_ptr as *const f32,
                (dst_ptr as *mut u8).add(byte_offset) as *mut std::ffi::c_void,
                elem_count as i32,
                qtype as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }

        // Note: we silence the "unused" warning on block_size since it's used for documentation
        let _ = block_size;

        Ok(())
    }

    /// Quantize f32 data with fused transpose from [H, T, D] to [H, D, T] layout.
    ///
    /// This fuses the memory layout transformation with quantization to avoid
    /// intermediate allocations. Used for KV cache quantization where:
    /// - Input layout: [n_head, chunk_size, head_dim] - channel-oriented float
    /// - Output layout: [n_head, head_dim, chunk_size] - token-oriented quant
    ///
    /// # Arguments
    /// * `src` - Source f32 CUDA storage with shape [n_head, chunk_size, head_dim]
    /// * `n_head` - Number of KV heads
    /// * `chunk_size` - Number of tokens (must be 32 for standard quants)
    /// * `head_dim` - Dimension per head
    /// * `byte_offset` - Byte offset in destination where quantized data should be written
    ///
    /// # Supported Types
    /// Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 (all 32-element formats).
    ///
    /// # Safety
    /// The caller must ensure:
    /// - byte_offset is block-aligned
    /// - byte_offset + n_head * head_dim * type_size <= storage_size_in_bytes()
    /// - src contains n_head * chunk_size * head_dim elements (F32/F16/BF16/F8E4M3)
    pub fn quantize_transposed_into(
        &mut self,
        src: &CudaStorage,
        n_head: usize,
        chunk_size: usize,
        head_dim: usize,
        byte_offset: usize,
    ) -> Result<()> {
        // Standard 32-element formats support fused transpose+quantize
        let qtype = match self.dtype {
            GgmlDType::Q4_0 => QType::Q4_0,
            GgmlDType::Q4_1 => QType::Q4_1,
            GgmlDType::Q5_0 => QType::Q5_0,
            GgmlDType::Q5_1 => QType::Q5_1,
            GgmlDType::Q8_0 => QType::Q8_0,
            GgmlDType::Q8_1 => QType::Q8_1,
            GgmlDType::Q4_KS => QType::Q4_KS,
            GgmlDType::Q8_KS => QType::Q8_KS,
            GgmlDType::Q2_0 => QType::Q2_0,
            GgmlDType::Q3_0 => QType::Q3_0,
            GgmlDType::R16 => QType::R16,
            GgmlDType::Q0 => QType::Q0,
            GgmlDType::Q0_X => QType::Q0_X,
            GgmlDType::Q1_A => QType::Q1_A,
            GgmlDType::Q0_V => QType::Q0_V,
            GgmlDType::Q0_M2 => QType::Q0_M2,
            GgmlDType::Q0_M4 => QType::Q0_M4,
            GgmlDType::Q1_S => QType::Q1_S,
            GgmlDType::Q2_S => QType::Q2_S,
            GgmlDType::Q2_A => QType::Q2_A,
            GgmlDType::Q2_1 => QType::Q2_1,
            GgmlDType::Q3_1 => QType::Q3_1,
            _ => crate::bail!(
                "quantize_transposed_into: only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q4_KS/Q8_KS supported, got {:?}",
                self.dtype
            ),
        };

        // Validate chunk_size (all standard quants use 32)
        if chunk_size != 32 {
            crate::bail!(
                "quantize_transposed_into: chunk_size must be 32 for standard quants, got {}",
                chunk_size
            );
        }

        // Get source pointer and dtype - supports F32/F16/BF16/F8E4M3
        let (src_ptr, src_dtype) = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => {
                let stream = self.device.cuda_stream();
                let (ptr, _guard) = data.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, crate::DType::F32)
            }
            crate::cuda_backend::CudaStorageSlice::F16(data) => {
                let stream = self.device.cuda_stream();
                let (ptr, _guard) = data.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, crate::DType::F16)
            }
            crate::cuda_backend::CudaStorageSlice::BF16(data) => {
                let stream = self.device.cuda_stream();
                let (ptr, _guard) = data.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, crate::DType::BF16)
            }
            crate::cuda_backend::CudaStorageSlice::F8E4M3(data) => {
                let stream = self.device.cuda_stream();
                let (ptr, _guard) = data.device_ptr(&stream);
                (ptr as *const std::ffi::c_void, crate::DType::F8E4M3)
            }
            _ => crate::bail!(
                "quantize_transposed_into: source dtype must be F32/F16/BF16/F8E4M3, got {:?}",
                src.dtype()
            ),
        };

        let src_dtype_code = dtype_to_src_dtype_code(src_dtype)?;

        // Validate alignment
        let type_size = self.dtype.type_size();
        if !byte_offset.is_multiple_of(type_size) {
            crate::bail!(
                "quantize_transposed_into: byte_offset {} not aligned to type_size {}",
                byte_offset,
                type_size
            );
        }

        // Validate bounds: n_head * head_dim Q8_0/Q4_0 blocks
        let num_blocks = n_head * head_dim;
        let required_bytes = num_blocks * type_size;
        if byte_offset + required_bytes > self.data.len {
            crate::bail!(
                "quantize_transposed_into: offset {} + size {} exceeds storage size {}",
                byte_offset,
                required_bytes,
                self.data.len
            );
        }

        // Get destination pointer and call typed kernel
        let stream = self.device.cuda_stream();
        let (dst_ptr, _dst_guard) = self.data.inner.device_ptr_mut(&stream);

        unsafe {
            // Use typed batched version with num_chunks=1
            run_quantize_transposed_batched_typed(
                src_ptr,
                (dst_ptr as *mut u8).add(byte_offset) as *mut std::ffi::c_void,
                std::ptr::null(), // No src offsets (contiguous)
                std::ptr::null(), // No dst offsets (contiguous)
                1,                // Single chunk
                n_head as i32,
                chunk_size as i32,
                head_dim as i32,
                qtype as i32,
                src_dtype_code,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }

        Ok(())
    }
}

/// Quantize F32 weights `[nrows × ncols]` (row-major) → a GPU buffer in the lane-major KO
/// layout the int8 KO matmul reads (`Q2_KO`/`Q3_KO`/`Q4_KO`/`Q5_KO`/`Q6_KO`/`Q8_KO`). The
/// de-interleave runs
/// on the GPU (`run_quantize_ko`), byte-identical to the CPU `ko_quant::quantize_ko` — the
/// symmetric weight counterpart to `quantize_acts_q8a128`. `nrows` is N (output rows), `ncols`
/// is K; `nrows` must be a multiple of 8, `ncols` a multiple of 128.
pub fn quantize_ko_weights(
    data: &[f32],
    nrows: usize,
    ncols: usize,
    dtype: GgmlDType,
    device: &CudaDevice,
) -> Result<CudaSlice<u8>> {
    if !nrows.is_multiple_of(8) || !ncols.is_multiple_of(128) || data.len() != nrows * ncols {
        crate::bail!(
            "quantize_ko_weights: bad shape nrows={nrows} ncols={ncols} len={}",
            data.len()
        );
    }
    let qtype = dtype_to_qtype(dtype)? as i32;
    let bytes = (nrows / 8) * (ncols / 128) * crate::quantized::ko_quant::ko_chunk_bytes(dtype);
    let w_dev = device.memcpy_stod(data)?;
    let mut out = unsafe { device.alloc::<u8>(bytes)? };
    {
        let stream = device.cuda_stream();
        let (wp, _g0) = w_dev.device_ptr(&stream);
        let (op, _g1) = out.device_ptr_mut(&stream);
        unsafe {
            run_quantize_ko(
                wp as *const f32,
                op as *mut std::ffi::c_void,
                nrows as i32,
                ncols as i32,
                qtype,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }
    Ok(out)
}
