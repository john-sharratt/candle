//! Expands quantized weights and KV blocks to float, refusing any format the flat
//! dequantize kernel does not implement.
//!
//! The kernels write into a destination the caller owns, which may be host-mapped
//! memory, and they run in bands of whole blocks so that no single launch lasts
//! long enough to trip the GPU watchdog. A format the dispatch switch does not
//! name launches nothing and would return the allocator's leftover bytes as a
//! success, so each entry checks the format first and either decodes on the host
//! or refuses it by name. KO twins have no flat per-block kernel and are refused
//! the same way, and a float source is widened by a cast or a copy rather than
//! dequantized.

use super::*;

/// Dequantize into a destination the caller already owns.
///
/// `dst_ptr` must name `elem_count` writable f32 the kernel may stamp. **It does not have to
/// be VRAM**: a `cuMemHostAlloc(DEVICEMAP)` pointer from [`alloc_host_mapped`] is a device
/// address like any other, and the kernel writes across PCIe into host RAM. That is what lets
/// [`QCudaStorage::repack_ko_into`] hold its whole-tensor f32 intermediate off the card — see
/// its header for why a buffer that exists for one tensor at load was costing VRAM for the
/// life of the process.
/// Whether `run_dequantize_block` has an arm for `dtype` — exactly the formats
/// its switch names (`quantized_dispatcher.cu`).
///
/// **That switch has no `default:`.** A format it does not name launches
/// nothing, and the caller's freshly allocated destination comes back as `Ok`
/// holding whatever the allocator left there. The KV formats below 2 bits
/// (Q1_S, Q2_S, Q2_A, Q0 and its siblings, Q2_1, Q3_1) were routed to it
/// anyway and read back as magnitudes near 1e34, so every device dequant asks
/// this first: a format it rejects either decodes on the host
/// ([`QCudaStorage::dequantize`]) or is refused.
pub(super) fn has_block_dequant_kernel(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q8_1
            | GgmlDType::Q2_K
            | GgmlDType::Q3_K
            | GgmlDType::Q4_K
            | GgmlDType::Q5_K
            | GgmlDType::Q6_K
            | GgmlDType::Q8_K
            | GgmlDType::QAWQ
            | GgmlDType::QAWQ_G64
            | GgmlDType::Q4_KS
            | GgmlDType::Q8_KS
            | GgmlDType::Q2_0
            | GgmlDType::Q3_0
            | GgmlDType::R16
    )
}

/// Refuse a block format `run_dequantize_block` would silently skip.
pub(super) fn require_block_dequant_kernel(dtype: GgmlDType) -> Result<()> {
    if !has_block_dequant_kernel(dtype) {
        crate::bail!(
            "{dtype:?} has no device dequant kernel; run_dequantize_block would return \
             uninitialised memory for it"
        )
    }
    Ok(())
}

/// Every launch is issued on `stream`.
pub(super) fn dequantize_f32_into(
    data_ptr: u64,
    dtype: GgmlDType,
    elem_count: usize,
    dst_ptr: u64,
    stream: &CudaStream,
) -> Result<()> {
    let launch_stream = stream.cu_stream() as *mut std::ffi::c_void;
    /// Which routine turns this source into f32.
    enum Widen {
        /// A block format with a `QType` slot — the ordinary dequant kernel.
        Block(i32),
        /// MXFP4: no `QType` slot (kept off the locked QTYPE tables), standalone kernel.
        Mxfp4,
        /// **A source that is already float.** Not a dequant at all — a widening cast, or
        /// for F32 a straight copy. It is handled here rather than at the caller so that a
        /// float tensor takes the same banded route as every quantized one: without it
        /// `QMatMul::build` had to dequantize the whole tensor, copy it again through
        /// `force_contiguous`, and quantize it to an intermediate `Q8_0` — four whole-tensor
        /// buffers to produce a twin that this path produces from a 48 MiB band.
        Float(GgmlDType),
    }

    let widen = match dtype {
        GgmlDType::MXFP4 => Widen::Mxfp4,
        GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => Widen::Float(dtype),
        _ => {
            require_block_dequant_kernel(dtype)?;
            Widen::Block(dtype_to_qtype(dtype)? as i32)
        }
    };

    // **Banded, so one launch never runs long enough to trip the GPU watchdog.**
    // `dst_ptr` may be host memory (that is the point — see `repack_ko_into`), and writing f32
    // across PCIe is slow enough that a whole large tensor in a single launch exceeds Windows'
    // default 2 s TDR: the device resets and the next synchronise reports
    // `CUDA_ERROR_UNKNOWN`. Measured here: 64 MiB completes, 256 MiB does not.
    //
    // Bands are whole blocks, so a band's source bytes and destination elements are both
    // contiguous and the kernel needs no offset argument — the pointers carry it. Banding a
    // device destination costs a handful of extra launches and nothing else, so it is not
    // conditional; a path that only bands "when it matters" is one whose common case is
    // untested.
    const BAND_TARGET_BYTES: usize = 48 * 1024 * 1024;
    let bs = dtype.block_size();
    let ts = dtype.type_size();
    let band = (BAND_TARGET_BYTES / std::mem::size_of::<f32>() / bs).max(1) * bs;
    let mut off = 0usize;
    while off < elem_count {
        let n = band.min(elem_count - off);
        let src = data_ptr + (off / bs * ts) as u64;
        let dst = dst_ptr + (off * std::mem::size_of::<f32>()) as u64;
        match widen {
            Widen::Mxfp4 => unsafe {
                candle_kernels::simple::quantized::run_dequantize_mxfp4(
                    src as *const std::ffi::c_void,
                    dst as *mut std::ffi::c_void,
                    n as i32,
                    0,
                    launch_stream,
                );
            },
            Widen::Block(q) => unsafe {
                run_dequantize_block(
                    src as *const std::ffi::c_void,
                    dst as *mut std::ffi::c_void,
                    n as i32,
                    q,
                    DequantOutDType::F32 as i32,
                    launch_stream,
                );
            },
            // `num_dims = 0` / `info = null`: the band is contiguous by construction, so the
            // cast needs no layout.
            Widen::Float(GgmlDType::F16) => unsafe {
                candle_kernels::simple::cast::run_cast_f16_f32(
                    src as *const std::ffi::c_void,
                    dst as *mut f32,
                    n,
                    0,
                    std::ptr::null(),
                    launch_stream,
                );
            },
            Widen::Float(GgmlDType::BF16) => unsafe {
                candle_kernels::simple::cast::run_cast_bf16_f32(
                    src as *const std::ffi::c_void,
                    dst as *mut f32,
                    n,
                    0,
                    std::ptr::null(),
                    launch_stream,
                );
            },
            // Already f32: the "widen" is a copy, and the driver does it without a kernel.
            // Not folded into the cast dispatcher, which has no identity arm. Async on
            // `stream`, ordered with the bands around it like every other arm.
            Widen::Float(_) => {
                let bytes = n * std::mem::size_of::<f32>();
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(dst, src, bytes, stream.cu_stream())
                        .map_err(crate::Error::wrap)?;
                }
            }
        }
        off += n;
    }
    Ok(())
}

pub(super) fn dequantize_f32(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let dst = unsafe { dev.alloc::<f32>(elem_count)? };
    {
        let stream = dev.cuda_stream();
        let (src_ptr, _src_guard) = data.inner.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
        dequantize_f32_into(src_ptr, dtype, elem_count, dst_ptr, &stream)?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

pub(super) fn dequantize_f16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if dtype == GgmlDType::MXFP4 {
        let dst = unsafe { dev.alloc::<f16>(elem_count)? };
        {
            let stream = dev.cuda_stream();
            let (data_ptr, _dg) = data.inner.device_ptr(&stream);
            let (dst_ptr, _og) = dst.device_ptr(&stream);
            unsafe {
                candle_kernels::simple::quantized::run_dequantize_mxfp4(
                    data_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    elem_count as i32,
                    1,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        return Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()));
    }
    require_block_dequant_kernel(dtype)?;
    let qtype = dtype_to_qtype(dtype)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = data.inner.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
        unsafe {
            run_dequantize_block(
                data_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                elem_count as i32,
                qtype as i32,
                DequantOutDType::F16 as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

pub(super) fn dequantize_bf16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    use half::bf16;
    if dtype == GgmlDType::MXFP4 {
        let dst = unsafe { dev.alloc::<bf16>(elem_count)? };
        {
            let stream = dev.cuda_stream();
            let (data_ptr, _dg) = data.inner.device_ptr(&stream);
            let (dst_ptr, _og) = dst.device_ptr(&stream);
            unsafe {
                candle_kernels::simple::quantized::run_dequantize_mxfp4(
                    data_ptr as *const std::ffi::c_void,
                    dst_ptr as *mut std::ffi::c_void,
                    elem_count as i32,
                    2,
                    stream.cu_stream() as *mut std::ffi::c_void,
                );
            }
        }
        return Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()));
    }
    require_block_dequant_kernel(dtype)?;
    let qtype = dtype_to_qtype(dtype)?;
    let dst = unsafe { dev.alloc::<bf16>(elem_count)? };
    {
        let stream = dev.cuda_stream();
        let (data_ptr, _data_guard) = data.inner.device_ptr(&stream);
        let (dst_ptr, _dst_guard) = dst.device_ptr(&stream);
        unsafe {
            run_dequantize_block(
                data_ptr as *const std::ffi::c_void,
                dst_ptr as *mut std::ffi::c_void,
                elem_count as i32,
                qtype as i32,
                DequantOutDType::BF16 as i32,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

impl QCudaStorage {
    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
        fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) {
            let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
            let vec = slice.to_vec();
            T::to_float(&vec, dst)
        }

        let fast_kernel = self.dtype == GgmlDType::MXFP4 || has_block_dequant_kernel(self.dtype);
        if fast_kernel {
            return dequantize_f32(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.

        let buffer = self
            .device
            .memcpy_dtov(&self.data.inner.slice(..self.data.len))?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F64 => deq::<f64>(&buffer, block_len, &mut out),
            GgmlDType::F32 => deq::<f32>(&buffer, block_len, &mut out),
            GgmlDType::U8 => deq::<u8>(&buffer, block_len, &mut out),
            GgmlDType::I8 => deq::<i8>(&buffer, block_len, &mut out),
            GgmlDType::U16 => deq::<u16>(&buffer, block_len, &mut out),
            GgmlDType::I16 => deq::<i16>(&buffer, block_len, &mut out),
            GgmlDType::U32 => deq::<u32>(&buffer, block_len, &mut out),
            GgmlDType::I32 => deq::<i32>(&buffer, block_len, &mut out),
            GgmlDType::U64 => deq::<u64>(&buffer, block_len, &mut out),
            GgmlDType::I64 => deq::<i64>(&buffer, block_len, &mut out),
            GgmlDType::F16 => deq::<half::f16>(&buffer, block_len, &mut out),
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, block_len, &mut out),
            GgmlDType::F8E4M3 => panic!("not implemented"),
            GgmlDType::F8E5M2 => panic!("not implemented"),
            GgmlDType::MXFP4 => {
                deq::<crate::quantized::k_quants::BlockMXFP4>(&buffer, block_len, &mut out)
            }
            GgmlDType::Q4_0 => deq::<crate::quantized::BlockQ4_0>(&buffer, block_len, &mut out),
            GgmlDType::Q4_1 => deq::<crate::quantized::BlockQ4_1>(&buffer, block_len, &mut out),
            GgmlDType::Q5_0 => deq::<crate::quantized::BlockQ5_0>(&buffer, block_len, &mut out),
            GgmlDType::Q5_1 => deq::<crate::quantized::BlockQ5_1>(&buffer, block_len, &mut out),
            GgmlDType::Q8_0 => deq::<crate::quantized::BlockQ8_0>(&buffer, block_len, &mut out),
            GgmlDType::Q8_1 => deq::<crate::quantized::BlockQ8_1>(&buffer, block_len, &mut out),
            GgmlDType::Q2_K => deq::<crate::quantized::BlockQ2_K>(&buffer, block_len, &mut out),
            GgmlDType::Q3_K => deq::<crate::quantized::BlockQ3_K>(&buffer, block_len, &mut out),
            GgmlDType::Q4_K => deq::<crate::quantized::BlockQ4_K>(&buffer, block_len, &mut out),
            GgmlDType::Q5_K => deq::<crate::quantized::BlockQ5_K>(&buffer, block_len, &mut out),
            GgmlDType::Q6_K => deq::<crate::quantized::BlockQ6_K>(&buffer, block_len, &mut out),
            GgmlDType::Q8_K => deq::<crate::quantized::BlockQ8_K>(&buffer, block_len, &mut out),
            GgmlDType::QAWQ => deq::<crate::quantized::BlockQAWQ>(&buffer, block_len, &mut out),
            GgmlDType::QAWQ_G64 => {
                deq::<crate::quantized::BlockQAWQ_G64>(&buffer, block_len, &mut out)
            }
            GgmlDType::Q4_KS => deq::<crate::quantized::BlockQ4_KS>(&buffer, block_len, &mut out),
            GgmlDType::Q8_KS => deq::<crate::quantized::BlockQ8_KS>(&buffer, block_len, &mut out),
            GgmlDType::Q2_0 => deq::<crate::quantized::BlockQ2_0>(&buffer, block_len, &mut out),
            GgmlDType::Q3_0 => deq::<crate::quantized::BlockQ3_0>(&buffer, block_len, &mut out),
            GgmlDType::R16 => deq::<crate::quantized::BlockR16>(&buffer, block_len, &mut out),
            GgmlDType::Q0 => deq::<crate::quantized::BlockQ0>(&buffer, block_len, &mut out),
            GgmlDType::Q1_S => deq::<crate::quantized::BlockQ1S>(&buffer, block_len, &mut out),
            GgmlDType::Q2_S => deq::<crate::quantized::BlockQ2S>(&buffer, block_len, &mut out),
            GgmlDType::Q2_A => deq::<crate::quantized::BlockQ2A>(&buffer, block_len, &mut out),
            GgmlDType::Q2_1 => deq::<crate::quantized::BlockQ2_1>(&buffer, block_len, &mut out),
            GgmlDType::Q3_1 => deq::<crate::quantized::BlockQ3_1>(&buffer, block_len, &mut out),
            GgmlDType::P2 => deq::<crate::quantized::BlockP2>(&buffer, block_len, &mut out),
            GgmlDType::Q0_V => deq::<crate::quantized::BlockQ0V>(&buffer, block_len, &mut out),
            GgmlDType::Q1_A => deq::<crate::quantized::BlockQ1A>(&buffer, block_len, &mut out),
            GgmlDType::Q0_X => deq::<crate::quantized::BlockQ0X>(&buffer, block_len, &mut out),
            GgmlDType::Q0_M2 => deq::<crate::quantized::BlockQ0M2>(&buffer, block_len, &mut out),
            GgmlDType::Q0_M4 => deq::<crate::quantized::BlockQ0M4>(&buffer, block_len, &mut out),
            GgmlDType::Q4_KO => deq::<crate::quantized::BlockQ4_KO>(&buffer, block_len, &mut out),
            GgmlDType::Q5_KO => deq::<crate::quantized::BlockQ5_KO>(&buffer, block_len, &mut out),
            GgmlDType::Q6_KO => deq::<crate::quantized::BlockQ6_KO>(&buffer, block_len, &mut out),
            GgmlDType::Q8_KO => deq::<crate::quantized::BlockQ8_KO>(&buffer, block_len, &mut out),
            // MXFP4_KO is a GPU-only lane-major chunk with no per-128 host block codec; it is
            // never CPU-dequantized (the per-sub fold lives in the int8 kernel).
            GgmlDType::MXFP4_KO => {
                crate::bail!("MXFP4_KO has no CPU dequant path; it is a GPU-only int8 weight")
            }
            // Q2_KO is a GPU-only lane-major crumb chunk; the shape-aware `ko_quant::dequant_q2_ko`
            // (test/prepare only) reconstructs it — this flat per-block path does not apply.
            GgmlDType::Q2_KO => {
                crate::bail!("Q2_KO has no CPU dequant path; it is a GPU-only int8 weight")
            }
            // Q3_KO likewise: a crumb+hi lane-major chunk. `ko_quant::dequant_ko` reconstructs
            // it shape-aware (test/prepare only); this flat per-block path does not apply.
            GgmlDType::Q3_KO => {
                crate::bail!("Q3_KO has no CPU dequant path; it is a GPU-only int8 weight")
            }
        }

        self.device
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    /// **A KO twin has no flat per-block dequant kernel, and silence here is
    /// uninitialised memory.**
    ///
    /// `run_dequantize_block`'s switch carries no KO arm *and no `default:`*, so
    /// a KO dtype reaches it, matches nothing, launches nothing, and the
    /// freshly-`alloc`'d destination is returned as `Ok` — garbage weights with
    /// no error anywhere. The F32 sibling (`dequantize`) has always refused
    /// these by name; these two were never given the same guard, and `Q3_KO`
    /// joining `dtype_to_qtype` made the gap reachable from a plain GGUF load.
    ///
    /// Refused rather than fixed in the kernel because a flat per-block dequant
    /// is not what a KO chunk *is*: it is lane-major over a 128-K tile, and
    /// `ko_quant::dequant_ko` reconstructs it shape-aware for tests and
    /// `prepare`. There is no correct flat kernel to dispatch to.
    pub(super) fn refuse_ko(&self, to: &str) -> Result<()> {
        if self.dtype.is_ko() {
            crate::bail!(
                "{:?} has no flat per-block dequant path, so it cannot be \
                 dequantized to {to}; it is a GPU-only int8 weight read by the \
                 KO matmul kernels",
                self.dtype
            )
        }
        Ok(())
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<CudaStorage> {
        self.refuse_ko("f16")?;
        dequantize_f16(&self.data, self.dtype, elem_count, self.device())
    }

    pub fn dequantize_bf16(&self, elem_count: usize) -> Result<CudaStorage> {
        self.refuse_ko("bf16")?;
        dequantize_bf16(&self.data, self.dtype, elem_count, self.device())
    }

    /// Dequantize data from this storage at the given byte offset into a destination buffer.
    ///
    /// This avoids intermediate allocations by using the GPU dequantization kernel
    /// to read from a specific offset and write directly into the target buffer.
    ///
    /// # Arguments
    /// * `dst` - Destination CUDA storage (f16/bf16/f32)
    /// * `elem_count` - Number of elements to dequantize
    /// * `src_byte_offset` - Byte offset in source (this storage) to read from
    /// * `dst_elem_offset` - Element offset in destination where dequantized data should be written
    ///
    /// # Safety
    /// The caller must ensure:
    /// - src_byte_offset is block-aligned (multiple of dtype.type_size())
    /// - src_byte_offset + quantized_size(elem_count, dtype) <= self.storage_size_in_bytes()
    /// - dst_elem_offset + elem_count <= dst element count
    pub fn dequantize_into(
        &self,
        dst: &mut CudaStorage,
        elem_count: usize,
        src_byte_offset: usize,
        dst_elem_offset: usize,
    ) -> Result<()> {
        use crate::cuda_backend::CudaStorageSlice;
        use candle_kernels::simple::quantized::run_dequantize_block;

        // Validate alignment
        let type_size = self.dtype.type_size();
        if !src_byte_offset.is_multiple_of(type_size) {
            crate::bail!(
                "dequantize_into: src_byte_offset {} not aligned to block type_size {}",
                src_byte_offset,
                type_size
            );
        }

        // Validate source bounds
        let quantized_bytes = quantized_size(elem_count, self.dtype);
        if src_byte_offset + quantized_bytes > self.data.len {
            crate::bail!(
                "dequantize_into: src offset {} + size {} exceeds storage size {}",
                src_byte_offset,
                quantized_bytes,
                self.data.len
            );
        }

        require_block_dequant_kernel(self.dtype)?;
        let qtype = dtype_to_qtype(self.dtype)?;
        let stream = self.device.cuda_stream();
        let (src_ptr, _src_guard) = self.data.inner.device_ptr(&stream);

        match &mut dst.slice {
            CudaStorageSlice::F16(data) => {
                let (dst_ptr, _dst_guard) = data.device_ptr_mut(&stream);
                unsafe {
                    run_dequantize_block(
                        (src_ptr as *const u8).add(src_byte_offset) as *const std::ffi::c_void,
                        (dst_ptr as *mut f16).add(dst_elem_offset) as *mut std::ffi::c_void,
                        elem_count as i32,
                        qtype as i32,
                        DequantOutDType::F16 as i32,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }
            CudaStorageSlice::BF16(data) => {
                let (dst_ptr, _dst_guard) = data.device_ptr_mut(&stream);
                unsafe {
                    run_dequantize_block(
                        (src_ptr as *const u8).add(src_byte_offset) as *const std::ffi::c_void,
                        (dst_ptr as *mut half::bf16).add(dst_elem_offset) as *mut std::ffi::c_void,
                        elem_count as i32,
                        qtype as i32,
                        DequantOutDType::BF16 as i32,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }
            CudaStorageSlice::F32(data) => {
                let (dst_ptr, _dst_guard) = data.device_ptr_mut(&stream);
                unsafe {
                    run_dequantize_block(
                        (src_ptr as *const u8).add(src_byte_offset) as *const std::ffi::c_void,
                        (dst_ptr as *mut f32).add(dst_elem_offset) as *mut std::ffi::c_void,
                        elem_count as i32,
                        qtype as i32,
                        DequantOutDType::F32 as i32,
                        stream.cu_stream() as *mut std::ffi::c_void,
                    );
                }
            }
            _ => crate::bail!("dequantize_into: destination must be f16, bf16, or f32"),
        }

        Ok(())
    }
}

/// Dequantize a lane-major KO chunk buffer → f32 `[nrows × ncols]` (row-major) on the GPU
/// (`run_dequantize_ko`). Inverse of [`quantize_ko_weights`]; matches `ko_quant::dequant_ko`.
pub fn dequant_ko_weights(
    chunk: &CudaSlice<u8>,
    nrows: usize,
    ncols: usize,
    dtype: GgmlDType,
    device: &CudaDevice,
) -> Result<CudaSlice<f32>> {
    let qtype = dtype_to_qtype(dtype)? as i32;
    let mut out = unsafe { device.alloc::<f32>(nrows * ncols)? };
    {
        let stream = device.cuda_stream();
        let (cp, _g0) = chunk.device_ptr(&stream);
        let (op, _g1) = out.device_ptr_mut(&stream);
        unsafe {
            run_dequantize_ko(
                cp as *const std::ffi::c_void,
                op as *mut f32,
                nrows as i32,
                ncols as i32,
                qtype,
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }
    Ok(out)
}

/// Dequantize q8a1024 flat-grouped blocks → f32 `[rows, cols]`.
pub fn dequantize_q8a128(
    blocks: &CudaSlice<u8>,
    rows: usize,
    cols: usize,
    device: &CudaDevice,
) -> Result<CudaSlice<f32>> {
    let mut out = unsafe { device.alloc::<f32>(rows * cols)? };
    {
        let stream = device.cuda_stream();
        let (bp, _g1) = blocks.device_ptr(&stream);
        let (op, _g2) = out.device_ptr_mut(&stream);
        unsafe {
            run_dequantize_q8a128(
                bp as *const std::ffi::c_void,
                op as *mut std::ffi::c_void,
                rows as i32,
                cols as i32,
                2, // F32 output
                stream.cu_stream() as *mut std::ffi::c_void,
            );
        }
    }
    Ok(out)
}
