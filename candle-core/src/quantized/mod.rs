//! Code for GGML and GGUF files
use crate::{Context, CpuStorage, DType, Device, LiveTensor, Result, Shape, Storage, Tensor};
use k_quants::*;
use std::borrow::Cow;
use std::marker::PhantomData;

#[cfg(target_feature = "avx2")]
pub mod avx;
mod dummy_cuda;
mod dummy_metal;
pub mod ggml_file;
pub mod gguf_file;
pub mod int8_matmul_mode;
pub mod k_quants;
pub mod ko_quant;
pub mod prepare;
// Note: the previous `q0_v_test` module has been removed — it tested the OLD
// (sign + shape + curve_pos) Q0_V format that no longer exists. The new
// (curve + scale + centroid) format will get a fresh test suite.
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(not(feature = "metal"))]
mod metal {
    pub use super::dummy_metal::*;
}
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(not(feature = "cuda"))]
mod dummy_pinned_staging;
#[cfg(feature = "cuda")]
pub mod pinned_staging;
#[cfg(feature = "cuda")]
pub mod table_ring;
#[cfg(not(feature = "cuda"))]
pub mod pinned_staging {
    pub use super::dummy_pinned_staging::*;
}
#[cfg(not(feature = "cuda"))]
mod cuda {
    pub use super::dummy_cuda::*;
}

#[cfg(feature = "cuda")]
pub use cuda::cuda_flush_l2;
pub use cuda::get_dispatch_info;
#[cfg(feature = "cuda")]
pub use cuda::grouped_matmul_gemx;
pub use cuda::set_force_dmmv;
#[cfg(feature = "cuda")]
pub use cuda::{
    alloc_host_mapped, get_total_vram_device0, get_vram_info, register_mmap_cuda, HostMappedAlloc,
    MmapRegistration,
};
#[cfg(feature = "cuda")]
pub use cuda::{
    load_repacked, load_repacked_into, load_repacked_on_stream, repack_gemx_to_host,
    repack_to_host, repacked_size_bytes, view_repacked,
};

#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(target_feature = "simd128")]
pub mod simd128;
pub mod utils;
use half::{bf16, f16};

pub use k_quants::GgmlType;

/// A quantized tensor whose device memory stays valid for `'w`.
///
/// `'w` describes the *memory*, not a borrow of any Rust value: an allocation
/// this tensor owns is live for `'static`, while a view onto memory owned
/// elsewhere — a KV arena slot, via [`Self::from_leased_cuda_ptr`] — is live
/// only as long as that owner. [`QTensor`] is the `'static` case and is what
/// nearly all code means.
///
/// The parameter is what stops an arena view outliving its arena. It also
/// keeps a view out of [`QMatMul`]: `CustomOp1` is implemented for `QTensor`
/// alone, so a lease — which carries no matrix-row padding — cannot reach the
/// matmul path at all.
///
/// # Variance
///
/// Covariant in `'w`, so an owned tensor is accepted wherever a shorter-lived
/// one is expected:
///
/// ```
/// use candle_core::quantized::{LiveQTensor, QTensor};
/// fn shorten<'a>(q: QTensor) -> LiveQTensor<'a> { q }
/// ```
///
/// and never the reverse — which is the property that makes the whole
/// parameter worth having:
///
/// ```compile_fail
/// use candle_core::quantized::{LiveQTensor, QTensor};
/// fn lengthen<'a>(q: LiveQTensor<'a>) -> QTensor { q }
/// ```
///
/// A lease therefore cannot become a matmul weight, because [`QMatMul`] holds
/// `Arc<QTensor>`:
///
/// ```compile_fail
/// use candle_core::quantized::{LiveQTensor, QMatMul};
/// use std::sync::Arc;
/// fn weight<'a>(q: LiveQTensor<'a>) -> candle_core::Result<QMatMul> {
///     QMatMul::from_arc(Arc::new(q))
/// }
/// ```
///
/// while an owned one still can — the control that keeps the `compile_fail`
/// above from passing for some unrelated reason:
///
/// ```
/// use candle_core::quantized::{QMatMul, QTensor};
/// use std::sync::Arc;
/// fn weight(q: QTensor) -> candle_core::Result<QMatMul> {
///     QMatMul::from_arc(Arc::new(q))
/// }
/// ```
#[derive(Clone)]
pub struct LiveQTensor<'w> {
    storage: QStorage,
    shape: Shape,
    /// Covariant in `'w`, so a `'static` tensor is usable wherever a shorter
    /// one is expected but never the reverse. `&'w [u8]` rather than
    /// `&'w mut [u8]`: writes go through a raw device pointer, so nothing here
    /// needs invariance, and invariance would reject the coercion above.
    lease: PhantomData<&'w [u8]>,
}

/// The everyday quantized tensor: it owns its memory, so that memory is live
/// for as long as the process cares to keep it.
pub type QTensor = LiveQTensor<'static>;

impl Device {
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QStorage> {
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
    fn block_size(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.block_size(),
            QStorage::Metal(storage) => storage.dtype().block_size(),
            QStorage::Cuda(storage) => storage.dtype().block_size(),
        }
    }

    fn dtype(&self) -> GgmlDType {
        match self {
            QStorage::Cpu(storage) => storage.dtype(),
            QStorage::Metal(storage) => storage.dtype(),
            QStorage::Cuda(storage) => storage.dtype(),
        }
    }

    fn device(&self) -> Device {
        match self {
            QStorage::Cpu(_storage) => Device::Cpu,
            QStorage::Metal(storage) => Device::Metal(storage.device().clone()),
            QStorage::Cuda(storage) => Device::Cuda(storage.device().clone()),
        }
    }

    fn size_in_bytes(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.storage_size_in_bytes(),
            QStorage::Metal(storage) => storage.storage_size_in_bytes(),
            QStorage::Cuda(storage) => storage.storage_size_in_bytes(),
        }
    }

    fn quantize(&mut self, src: &Storage) -> Result<()> {
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

    fn dequantize(&self, elem_count: usize) -> Result<Storage> {
        match self {
            QStorage::Cpu(storage) => Ok(Storage::Cpu(storage.dequantize(elem_count)?)),
            QStorage::Metal(storage) => Ok(Storage::Metal(storage.dequantize(elem_count)?)),
            QStorage::Cuda(storage) => Ok(Storage::Cuda(storage.dequantize(elem_count)?)),
        }
    }

    fn data(&self) -> Result<Cow<'_, [u8]>> {
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
    fn data_range(&self, range: std::ops::Range<usize>) -> Result<Cow<'_, [u8]>> {
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
    fn cuda_data_ptr(&self) -> Option<u64> {
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
    fn cuda_data_ptr_mut(&mut self) -> Option<u64> {
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
    fn slice_scatter(&mut self, src: &QStorage, byte_offset: usize) -> Result<()> {
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
    fn slice_range_copy(
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
    fn quantize_into(
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
    fn quantize_transposed_into(
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
    fn dequantize_into(
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

/// Inference numeric mode for the q8a128 int8 tensor-core matmul path.
///
/// A single knob selecting both halves of the int8 pairing (weight repack via
/// [`QMatMul::repack_for_optimization`] and activation conversion via `cuda::to_dynamic`):
///
/// - [`Int8Mode::Off`] — FP16 GEMM. Weights repack to the gemx float layout, activations stay
///   float. The numeric reference path; no int8 anywhere.
/// - [`Int8Mode::Performance`] — int8 with the *same-width* KO weight twin (`Q4_K`→`Q4_KO`,
///   `Q5_K`→`Q5_KO`, …). Fastest and smallest; takes the per-128 granularity hit on the weight.
/// - [`Int8Mode::Precision`] — int8 with the *stepped-up* KO weight twin (`Q4_K`→`Q5_KO`,
///   `Q5_K`→`Q6_KO`, …). The extra weight bit absorbs the per-128 re-quant error, leaving int8
///   near-lossless versus the source quant. Activations (q8a128) are identical to Performance —
///   only the weight twin differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Int8Mode {
    /// FP16 GEMM — no int8. The numeric reference.
    #[default]
    Off,
    /// int8 with the same-width KO weight twin. Fastest, lossier.
    Performance,
    /// int8 with the stepped-up KO weight twin. Near-lossless versus the source quant.
    Precision,
}

impl Int8Mode {
    /// True when the int8 tensor-core path is active (any mode other than [`Int8Mode::Off`]).
    pub fn is_int8(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// Auto-select the numeric mode for `device`: [`Int8Mode::Precision`] when the device can run
    /// the int8 `m16n8k32` tensor-core MMA (CUDA, compute capability >= 8.0 / Ampere+), otherwise
    /// [`Int8Mode::Off`] (the FP16 reference). Precision is chosen over Performance because its
    /// stepped-up KO twin is near-lossless versus the source quant at no measurable decode cost.
    pub fn auto(device: &crate::Device) -> Self {
        match device {
            #[cfg(feature = "cuda")]
            crate::Device::Cuda(d) if d.supports_int8_mma() => Self::Precision,
            _ => Self::Off,
        }
    }

    /// VRAM-aware [`Int8Mode::auto`]: on an int8-MMA-capable CUDA device, picks
    /// [`Int8Mode::Precision`] (near-lossless, but the *larger* stepped-up weight
    /// twin) only when the weights leave comfortable headroom — the model is at
    /// most ~70% of free VRAM, so the KV cache, activations, and (MoE) hot
    /// experts still fit — and otherwise drops to [`Int8Mode::Performance`] (the
    /// smaller same-width twin) so a tight model still fits. `Off` (FP16) on CPU
    /// / non-int8 devices, or [`Int8Mode::Performance`] if VRAM can't be queried.
    ///
    /// `model_bytes` is the on-disk quantized weight size (e.g. the GGUF length).
    // `model_bytes` is only weighed against free VRAM, which is a CUDA query;
    // every other device answers `Off` without consulting it.
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    pub fn auto_sized(device: &crate::Device, model_bytes: usize) -> Self {
        match device {
            #[cfg(feature = "cuda")]
            crate::Device::Cuda(d) if d.supports_int8_mma() => match device.mem_get_info() {
                // model / free <= 7/10  ⇒  fits with headroom  ⇒  Precision.
                Ok((free, _)) if model_bytes.saturating_mul(10) <= free.saturating_mul(7) => {
                    Self::Precision
                }
                _ => Self::Performance,
            },
            _ => Self::Off,
        }
    }
}

#[repr(u32)]
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32 = 0,
    F16 = 1,
    BF16 = 2,
    /// R16: Raw F16 + reserved Q-capture space (128 bytes per 32 elements)
    R16 = 3,
    /// P2: 2-bit palette index (1 byte per 4 head_dim positions, pure arena routing)
    P2 = 4,
    /// AWQ 4-bit with group size 128
    QAWQ = 5,
    QAWQ_G64 = 6,
    /// Quant blocks
    Q8_0 = 7,
    Q8_1 = 8,
    Q8_K = 9,
    Q8_KS = 10,

    Q6_K = 11,

    Q5_0 = 12,
    Q5_1 = 13,
    Q5_K = 14,

    Q4_0 = 15,
    Q4_1 = 16,
    Q4_K = 17,
    Q4_KS = 18,

    Q3_0 = 19,
    Q3_1 = 20,
    Q3_K = 21,

    Q2_0 = 22,
    Q2_1 = 23,
    Q2_K = 24,
    Q2_S = 25,
    Q2_A = 26,

    Q1_S = 27,

    Q0_V = 28,
    Q1_A = 29,
    Q0_X = 30,
    Q0_M2 = 31,
    Q0_M4 = 32,
    Q0 = 33,

    F8E4M3 = 34,
    F8E5M2 = 35,

    U8 = 36,
    I8 = 37,
    U16 = 38,
    I16 = 39,
    U32 = 40,
    I32 = 41,
    U64 = 42,
    I64 = 43,
    F64 = 44,

    /// Lane-major per-128 affine weight format for the q8a128 int8 tensor-core matmul.
    /// NOT a byte-permute of a stored K-quant: it carries its own per-128 `(scale, min)`
    /// (one pair per 128 K per row) and is re-quantized straight from F32 (see
    /// `ko_quant::quantize_ko`), so it is lossy vs the source — not bit-identical. Q4_KO is
    /// 4-bit; the de-interleaved lane-major layout lets each int8-matmul lane pull its 4
    /// sub-uint32s in one wide LDS. Value 45 mirrors `QTYPE_Q4_KO` / `QType::Q4_KO`; the
    /// Q5/Q6/Q8_KO twins follow at 46-48.
    Q4_KO = 45,
    /// Lane-major per-128 affine KO twins (5/6/8-bit) — the same re-quantized-from-F32 format
    /// as Q4_KO, NOT a permutation of Q5_K/Q6_K/Q8_K. GPU-only weight formats; mirror
    /// `QTYPE_Q*_KO` / `QType::Q*_KO`.
    Q5_KO = 46,
    Q6_KO = 47,
    Q8_KO = 48,

    /// OCP MXFP4 (4-bit micro-scaling FP4): 32 elems/block, one E8M0 scale byte + 16
    /// nibbles indexing the E2M1 table. GGUF file code 39. The native trained format
    /// for the DeepSeek-V4 routed experts. See [`k_quants::BlockMXFP4`].
    MXFP4 = 49,

    /// Lane-major per-sub MXFP4 for the q8a128 int8 tensor-core matmul — the KO twin the
    /// MXFP4 routed experts repack to (an exact byte permutation of the native blocks; no
    /// requant). The E2M1 nibbles are kept **4-bit**; the kernel runs one int32 MMA per
    /// 32-K sub and folds each with its own E8M0 power-of-two scale in FP — the per-32
    /// scales apply exactly. Stays 4-bit → fits in RAM where Q6_KO/Q8_KO don't. GPU-only;
    /// value 50 mirrors `QTYPE_MXFP4_KO` / `QType::MXFP4_KO`. See
    /// `ko_quant::mxfp4_native_to_ko_gpu_chunk` + `loader/mxfp4.cuh`.
    MXFP4_KO = 50,

    /// Lane-major per-128 affine KO twin at **2-bit** — the smallest KO weight. Same
    /// re-quantized-from-F32 per-128 `(scale, min)` affine format as `Q4_KO`, but each value is
    /// a 2-bit crumb (0..3): the 128-K tile's quants pack into 32 B (`block_c_q2_KO_k128`,
    /// `int qs[8]`) vs Q4_KO's 64 B, so a chunk is 288 B / 1024 elems (~2.25 bpw). Its 2-bit
    /// crumb layout mirrors the high-2-bit crumb region Q6_KO already carries (`cr0`/`cr1` at
    /// `lane*8 + sub*2`), used here as the whole value. GPU-only (built by requantizing from F32
    /// on-device, like the other KO twins); read by the `q2_ko_int8_f32_grouped` int8 kernel.
    /// Value 51 mirrors `QTYPE_Q2_KO` / `QType::Q2_KO`. See `ko_quant::quantize_q2_ko`.
    Q2_KO = 51,
}

impl GgmlDType {
    /// True for the lane-major per-128 KO weight formats (`Q4_KO`/`Q5_KO`/`Q6_KO`/`Q8_KO`).
    /// These are the only weight layouts the q8a128 int8 tensor-core matmul can read, so the
    /// `DynamicTensor::Int8` matmul path guards on this.
    pub fn is_ko(self) -> bool {
        matches!(
            self,
            Self::Q2_KO | Self::Q4_KO | Self::Q5_KO | Self::Q6_KO | Self::Q8_KO | Self::MXFP4_KO
        )
    }

    /// The KO weight twin used for int8 optimization, selected by [`Int8Mode`].
    ///
    /// [`Int8Mode::Performance`] picks the **same-width** twin (4-bit→`Q4_KO`, 5-bit→`Q5_KO`,
    /// 6-bit→`Q6_KO`, 8-bit→`Q8_KO`; ≤3-bit→`Q4_KO`, the smallest KO form). It takes the
    /// per-32→per-128 granularity hit on the weight but is the fastest and smallest.
    ///
    /// [`Int8Mode::Precision`] steps the source one notch up the ladder
    /// (`Q4_KO` < `Q5_KO` < `Q6_KO` < `Q8_KO`): 4-bit→`Q5_KO`, 5-bit→`Q6_KO`, ≤3-bit→`Q4_KO`. The
    /// extra bit absorbs the granularity loss so the re-quant of the already-quantized weight is
    /// near-lossless. At the top of the ladder there is no finer twin, so 6-bit→`Q6_KO` and
    /// 8-bit→`Q8_KO` are same-width in both modes.
    ///
    /// Errors for [`Int8Mode::Off`] (no KO twin) and for dtypes with no KO form.
    pub fn to_ko(self, mode: Int8Mode) -> Result<Self> {
        // Already a KO format (e.g. a pre-repacked / prepared weight): the twin is itself, so
        // the optimize/geometry step is a no-op regardless of mode — nothing to repack.
        if self.is_ko() {
            return Ok(self);
        }
        match mode {
            Int8Mode::Off => crate::bail!("to_ko: Int8Mode::Off has no KO weight twin"),
            Int8Mode::Performance => Ok(match self {
                Self::Q2_K | Self::Q3_K => Self::Q4_KO,
                Self::Q4_0 | Self::Q4_1 | Self::Q4_K => Self::Q4_KO,
                Self::Q5_0 | Self::Q5_1 | Self::Q5_K => Self::Q5_KO,
                Self::Q6_K => Self::Q6_KO,
                Self::Q8_0 | Self::Q8_1 | Self::Q8_K => Self::Q8_KO,
                // MXFP4 keeps its native 4-bit E2M1 nibbles (exact byte permutation, no
                // requant to a wider grid; the kernel folds each per-32 E8M0 sub exactly),
                // so it fits in RAM where Q6_KO/Q8_KO don't, with no weight-side loss.
                Self::MXFP4 => Self::MXFP4_KO,
                other => crate::bail!("no KO weight form for {other:?}"),
            }),
            Int8Mode::Precision => Ok(match self {
                Self::Q2_K | Self::Q3_K => Self::Q4_KO,
                Self::Q4_0 | Self::Q4_1 | Self::Q4_K => Self::Q5_KO,
                Self::Q5_0 | Self::Q5_1 | Self::Q5_K => Self::Q6_KO,
                Self::Q6_K => Self::Q6_KO,
                Self::Q8_0 | Self::Q8_1 | Self::Q8_K => Self::Q8_KO,
                // Same 4-bit per-sub twin as Performance — the per-sub fold is already
                // weight-exact, so there is no wider grid to step up to.
                Self::MXFP4 => Self::MXFP4_KO,
                other => crate::bail!("no KO weight form for {other:?}"),
            }),
        }
    }

    /// Map an in-workspace integer (`GgmlDType as u32` discriminant) back to
    /// the corresponding `GgmlDType` variant.  This is the identity mapping —
    /// `from_u32(Self::Q4_K as u32)` returns `Self::Q4_K`.
    ///
    /// ⚠ This is NOT the GGUF/GGML on-disk file-format code.  Use
    /// `from_gguf_file_code` when reading raw integers from GGUF/GGML files.
    ///
    /// Every other integer used for a quant format inside the workspace —
    /// the C++ `QType` in `block_compact.cuh`, `ArenaFormat::*` in
    /// `arena_table.cuh`, `SELECT_FMT_*` in `select_kv_format.cuh`, the
    /// KV-side Rust `QType` in `candle_kernels/src/simple/quantized.rs`,
    /// the matmul-side Rust `QType` in `candle_kernels/src/quantized/api.rs`,
    /// and every `qtype: i32` FFI argument — all use `GgmlDType as u32`.
    pub fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::BF16,
            3 => Self::R16,
            4 => Self::P2,
            5 => Self::QAWQ,
            6 => Self::QAWQ_G64,
            7 => Self::Q8_0,
            8 => Self::Q8_1,
            9 => Self::Q8_K,
            10 => Self::Q8_KS,
            11 => Self::Q6_K,
            12 => Self::Q5_0,
            13 => Self::Q5_1,
            14 => Self::Q5_K,
            15 => Self::Q4_0,
            16 => Self::Q4_1,
            17 => Self::Q4_K,
            18 => Self::Q4_KS,
            19 => Self::Q3_0,
            20 => Self::Q3_1,
            21 => Self::Q3_K,
            22 => Self::Q2_0,
            23 => Self::Q2_1,
            24 => Self::Q2_K,
            25 => Self::Q2_S,
            26 => Self::Q2_A,
            27 => Self::Q1_S,
            28 => Self::Q0_V,
            29 => Self::Q1_A,
            30 => Self::Q0_X,
            31 => Self::Q0_M2,
            32 => Self::Q0_M4,
            33 => Self::Q0,
            34 => Self::F8E4M3,
            35 => Self::F8E5M2,
            36 => Self::U8,
            37 => Self::I8,
            38 => Self::U16,
            39 => Self::I16,
            40 => Self::U32,
            41 => Self::I32,
            42 => Self::U64,
            43 => Self::I64,
            44 => Self::F64,
            45 => Self::Q4_KO,
            46 => Self::Q5_KO,
            47 => Self::Q6_KO,
            48 => Self::Q8_KO,
            49 => Self::MXFP4,
            50 => Self::MXFP4_KO,
            51 => Self::Q2_KO,
            _ => crate::bail!("unknown dtype discriminant {u}"),
        };
        Ok(dtype)
    }

    /// Return the in-workspace integer representation (`GgmlDType as u32`).
    ///
    /// ⚠ This is NOT the GGUF/GGML on-disk file-format code.  Use
    /// `to_gguf_file_code` when writing integers to GGUF/GGML files.
    pub fn to_u32(self) -> u32 {
        self as u32
    }

    /// Decode a GGUF / GGML **on-disk file-format** dtype code into our
    /// in-memory `GgmlDType`.
    ///
    /// ⚠ THIS IS THE SINGLE TRANSLATION BOUNDARY between the on-disk GGUF
    /// integer and everything inside the workspace.  The on-disk numbering
    /// (upstream llama.cpp) differs from the workspace discriminants:
    ///   * On-disk: Q4_K=12, Q5_K=13, Q6_K=14, Q8_K=15, BF16=30, …
    ///   * In-workspace: Q4_K=17, Q5_K=14, Q6_K=11, Q8_K=9, BF16=2, …
    ///
    /// `from_gguf_file_code` and `to_gguf_file_code` are each other's inverse.
    pub(crate) fn from_gguf_file_code(u: u32) -> Result<Self> {
        let dtype = match u {
            // Standard GGML file codes (match llama.cpp ggml.h).
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            // Standard ggml integer / f64 tensor codes (llama.cpp writes these for
            // non-quant tensors such as the hash-routing `tid2eid` table). candle's own
            // writer uses the 231+ codes, but external files use these — read both.
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => Self::BF16,
            // MXFP4 (OCP micro-scaling FP4) — DeepSeek-V4 routed experts.
            39 => Self::MXFP4,
            // AWQ types use IDs 100+ to avoid conflicts with GGML types
            100 => Self::QAWQ,
            101 => Self::QAWQ_G64,
            // Candle-specific extensions use IDs 200+ to stay clear of future GGML / AWQ additions.
            200 => Self::Q4_KS,
            201 => Self::Q8_KS,
            202 => Self::Q2_0,
            203 => Self::Q3_0,
            204 => Self::Q2_S,
            205 => Self::Q2_A,
            206 => Self::Q1_S,
            207 => Self::R16,
            208 => Self::Q0,
            209 => Self::Q3_1,
            210 => Self::Q2_1,
            211 => Self::P2,
            212 => Self::Q0_V,
            213 => Self::Q1_A,
            214 => Self::Q0_X,
            215 => Self::Q0_M2,
            216 => Self::Q0_M4,
            217 => Self::F8E4M3,
            218 => Self::F8E5M2,
            // KO byte-permuted twins — never actually on disk, but kept round-trippable
            // with `to_gguf_file_code` for consistency.
            219 => Self::Q4_KO,
            220 => Self::Q5_KO,
            221 => Self::Q6_KO,
            222 => Self::Q8_KO,
            223 => Self::MXFP4_KO,
            224 => Self::Q2_KO,
            230 => Self::F64,
            231 => Self::U8,
            232 => Self::I8,
            233 => Self::U16,
            234 => Self::I16,
            235 => Self::U32,
            236 => Self::I32,
            237 => Self::U64,
            238 => Self::I64,
            _ => crate::bail!("unknown gguf file dtype code {u}"),
        };
        Ok(dtype)
    }

    /// Encode a `GgmlDType` back into its GGUF / GGML on-disk file-format code.
    ///
    /// ⚠ MUST be the exact inverse of `from_gguf_file_code` above — the output
    /// is the integer written to disk in GGUF tensor metadata, NOT the
    /// `#[repr(u32)]` discriminant of this enum.
    pub(crate) fn to_gguf_file_code(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2_K => 10,
            Self::Q3_K => 11,
            Self::Q4_K => 12,
            Self::Q5_K => 13,
            Self::Q6_K => 14,
            Self::Q8_K => 15,
            Self::BF16 => 30,
            Self::QAWQ => 100,
            Self::QAWQ_G64 => 101,
            Self::Q4_KS => 200,
            Self::Q8_KS => 201,
            Self::Q2_0 => 202,
            Self::Q3_0 => 203,
            Self::Q2_S => 204,
            Self::Q2_A => 205,
            Self::Q1_S => 206,
            Self::R16 => 207,
            Self::Q0 => 208,
            Self::Q3_1 => 209,
            Self::Q2_1 => 210,
            Self::P2 => 211,
            Self::Q0_V => 212,
            Self::Q1_A => 213,
            Self::Q0_X => 214,
            Self::Q0_M2 => 215,
            Self::Q0_M4 => 216,
            Self::F8E4M3 => 217,
            Self::F8E5M2 => 218,
            Self::MXFP4 => 39,
            Self::F64 => 230,
            Self::U8 => 231,
            Self::I8 => 232,
            Self::U16 => 233,
            Self::I16 => 234,
            Self::U32 => 235,
            Self::I32 => 236,
            Self::U64 => 237,
            Self::I64 => 238,
            // KO twins are GPU-only and never actually written to disk; codes kept for
            // exhaustiveness / round-trip with `from_gguf_file_code`.
            Self::Q4_KO => 219,
            Self::Q5_KO => 220,
            Self::Q6_KO => 221,
            Self::Q8_KO => 222,
            Self::MXFP4_KO => 223,
            Self::Q2_KO => 224,
        }
    }

    /// The block dtype
    pub fn cpu_zeros(&self, elem_count: usize) -> Box<dyn QuantizedType> {
        match self {
            Self::F64 => Box::new(vec![f64::zeros(); elem_count]),
            Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
            Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
            Self::F8E4M3 => Box::new(vec![BlockQ8_0::zeros(); elem_count]),
            Self::F8E5M2 => Box::new(vec![BlockQ8_0::zeros(); elem_count]),
            Self::U8 => Box::new(vec![u8::zeros(); elem_count]),
            Self::I8 => Box::new(vec![i8::zeros(); elem_count]),
            Self::U16 => Box::new(vec![u16::zeros(); elem_count]),
            Self::I16 => Box::new(vec![i16::zeros(); elem_count]),
            Self::U32 => Box::new(vec![u32::zeros(); elem_count]),
            Self::I32 => Box::new(vec![i32::zeros(); elem_count]),
            Self::U64 => Box::new(vec![u64::zeros(); elem_count]),
            Self::I64 => Box::new(vec![i64::zeros(); elem_count]),
            Self::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
            Self::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
            Self::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
            Self::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
            Self::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
            Self::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
            Self::Q2_K => Box::new(vec![BlockQ2_K::zeros(); elem_count / BlockQ2_K::BLCK_SIZE]),
            Self::Q3_K => Box::new(vec![BlockQ3_K::zeros(); elem_count / BlockQ3_K::BLCK_SIZE]),
            Self::Q4_K => Box::new(vec![BlockQ4_K::zeros(); elem_count / BlockQ4_K::BLCK_SIZE]),
            Self::Q5_K => Box::new(vec![BlockQ5_K::zeros(); elem_count / BlockQ5_K::BLCK_SIZE]),
            Self::Q6_K => Box::new(vec![BlockQ6_K::zeros(); elem_count / BlockQ6_K::BLCK_SIZE]),
            Self::Q8_K => Box::new(vec![BlockQ8_K::zeros(); elem_count / BlockQ8_K::BLCK_SIZE]),
            Self::BF16 => Box::new(vec![bf16::zeros(); elem_count]),
            Self::QAWQ => Box::new(vec![BlockQAWQ::zeros(); elem_count / BlockQAWQ::BLCK_SIZE]),
            Self::QAWQ_G64 => Box::new(vec![
                BlockQAWQ_G64::zeros();
                elem_count / BlockQAWQ_G64::BLCK_SIZE
            ]),
            Self::Q4_KS => Box::new(vec![
                BlockQ4_KS::zeros();
                elem_count / BlockQ4_KS::BLCK_SIZE
            ]),
            Self::Q8_KS => Box::new(vec![
                BlockQ8_KS::zeros();
                elem_count / BlockQ8_KS::BLCK_SIZE
            ]),
            Self::Q2_0 => Box::new(vec![BlockQ2_0::zeros(); elem_count / BlockQ2_0::BLCK_SIZE]),
            Self::Q3_0 => Box::new(vec![BlockQ3_0::zeros(); elem_count / BlockQ3_0::BLCK_SIZE]),
            Self::R16 => Box::new(vec![BlockR16::zeros(); elem_count / BlockR16::BLCK_SIZE]),
            Self::Q0 => Box::new(vec![BlockQ0::zeros(); elem_count / BlockQ0::BLCK_SIZE]),
            Self::Q1_S => Box::new(vec![BlockQ1S::zeros(); elem_count / BlockQ1S::BLCK_SIZE]),
            Self::Q2_S => Box::new(vec![BlockQ2S::zeros(); elem_count / BlockQ2S::BLCK_SIZE]),
            Self::Q2_A => Box::new(vec![BlockQ2A::zeros(); elem_count / BlockQ2A::BLCK_SIZE]),
            Self::Q2_1 => Box::new(vec![BlockQ2_1::zeros(); elem_count / BlockQ2_1::BLCK_SIZE]),
            Self::Q3_1 => Box::new(vec![BlockQ3_1::zeros(); elem_count / BlockQ3_1::BLCK_SIZE]),
            Self::P2 => Box::new(vec![BlockP2::zeros(); elem_count / BlockP2::BLCK_SIZE]),
            Self::Q0_V => Box::new(vec![BlockQ0V::zeros(); elem_count / BlockQ0V::BLCK_SIZE]),
            Self::Q1_A => Box::new(vec![BlockQ1A::zeros(); elem_count / BlockQ1A::BLCK_SIZE]),
            Self::Q0_X => Box::new(vec![BlockQ0X::zeros(); elem_count / BlockQ0X::BLCK_SIZE]),
            Self::Q0_M2 => Box::new(vec![BlockQ0M2::zeros(); elem_count / BlockQ0M2::BLCK_SIZE]),
            Self::Q0_M4 => Box::new(vec![BlockQ0M4::zeros(); elem_count / BlockQ0M4::BLCK_SIZE]),
            Self::Q4_KO => Box::new(vec![
                BlockQ4_KO::zeros();
                elem_count / BlockQ4_KO::BLCK_SIZE
            ]),
            Self::Q5_KO => Box::new(vec![
                BlockQ5_KO::zeros();
                elem_count / BlockQ5_KO::BLCK_SIZE
            ]),
            Self::Q6_KO => Box::new(vec![
                BlockQ6_KO::zeros();
                elem_count / BlockQ6_KO::BLCK_SIZE
            ]),
            Self::Q8_KO => Box::new(vec![
                BlockQ8_KO::zeros();
                elem_count / BlockQ8_KO::BLCK_SIZE
            ]),
            Self::MXFP4 => Box::new(vec![
                BlockMXFP4::zeros();
                elem_count / BlockMXFP4::BLCK_SIZE
            ]),
            // MXFP4_KO is a GPU-only lane-major chunk (built by repacking MXFP4 on-device,
            // never as CPU blocks) — there is no host block struct to zero-fill.
            Self::MXFP4_KO => {
                panic!("MXFP4_KO has no CPU block form; build it via repack from MXFP4 on CUDA")
            }
            // Q2_KO is a GPU-only lane-major chunk (built by requantizing F32 on-device, or by
            // `ko_quant::quantize_q2_ko` for the host prepare path) — no host block struct.
            Self::Q2_KO => {
                panic!("Q2_KO has no CPU block form; build it via quantize_q2_ko / repack on CUDA")
            }
        }
    }
    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F64 => 8,
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::F8E4M3 => 1,
            Self::F8E5M2 => 1,
            Self::U8 => 1,
            Self::I8 => 1,
            Self::U16 => 2,
            Self::I16 => 2,
            Self::U32 => 4,
            Self::I32 => 4,
            Self::U64 => 8,
            Self::I64 => 8,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2_K => std::mem::size_of::<BlockQ2_K>(),
            Self::Q3_K => std::mem::size_of::<BlockQ3_K>(),
            Self::Q4_K => std::mem::size_of::<BlockQ4_K>(),
            Self::Q5_K => std::mem::size_of::<BlockQ5_K>(),
            Self::Q6_K => std::mem::size_of::<BlockQ6_K>(),
            Self::Q8_K => std::mem::size_of::<BlockQ8_K>(),
            Self::QAWQ => std::mem::size_of::<BlockQAWQ>(),
            Self::QAWQ_G64 => std::mem::size_of::<BlockQAWQ_G64>(),
            Self::Q4_KS => std::mem::size_of::<BlockQ4_KS>(),
            Self::Q8_KS => std::mem::size_of::<BlockQ8_KS>(),
            Self::Q2_0 => std::mem::size_of::<BlockQ2_0>(),
            Self::Q3_0 => std::mem::size_of::<BlockQ3_0>(),
            Self::R16 => std::mem::size_of::<BlockR16>(),
            Self::Q0 => std::mem::size_of::<BlockQ0>(),
            Self::Q1_S => std::mem::size_of::<BlockQ1S>(),
            Self::Q2_S => std::mem::size_of::<BlockQ2S>(),
            Self::Q2_A => std::mem::size_of::<BlockQ2A>(),
            Self::Q2_1 => std::mem::size_of::<BlockQ2_1>(),
            Self::Q3_1 => std::mem::size_of::<BlockQ3_1>(),
            Self::P2 => std::mem::size_of::<BlockP2>(),
            Self::Q0_V => std::mem::size_of::<BlockQ0V>(),
            Self::Q1_A => std::mem::size_of::<BlockQ1A>(),
            Self::Q0_X => std::mem::size_of::<BlockQ0X>(),
            Self::Q0_M2 => std::mem::size_of::<BlockQ0M2>(),
            Self::Q0_M4 => std::mem::size_of::<BlockQ0M4>(),
            // KO compact blocks: same byte size as their K counterpart, just reordered.
            // KO twins are GPU-only lane-major chunks sized by `ko_quant::ko_chunk_bytes`
            // (128 elems/block, ko_chunk_bytes/8 bytes/block) — NOT the CPU `BlockQ*_KO`
            // struct layout. `type_size` is what the GGUF header uses to lay out and slice
            // tensor data (offset accounting on write, length on read), so it MUST equal the
            // bytes `quantize_ko` actually emits or offsets drift (past-EOF reads). Mirrors
            // MXFP4_KO returning `72` (= 576/8) directly rather than a Block struct size.
            Self::Q4_KO => 68,  // ko_chunk_bytes 544 / 8
            Self::Q5_KO => 84,  // ko_chunk_bytes 672 / 8
            Self::Q6_KO => 100, // ko_chunk_bytes 800 / 8
            Self::Q8_KO => 132, // ko_chunk_bytes 1056 / 8
            Self::MXFP4 => std::mem::size_of::<BlockMXFP4>(),
            // 576-byte GPU chunk per 1024 elements (512 nibbles + 32 E8M0 + 32 dm) → 72 B
            // per 128, keeping bytes == n_blocks × type_size for the [N,K] storage.
            Self::MXFP4_KO => 72,
            // 288-byte GPU chunk per 1024 elements (256 B of 2-bit crumbs + 32 dm) → 36 B per
            // 128. `ko_chunk_bytes(Q2_KO) = 288`.
            Self::Q2_KO => 36,
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F64 => 1,
            Self::F16 | Self::BF16 => 1,
            Self::F8E4M3 => 1,
            Self::F8E5M2 => 1,
            Self::U8 => 1,
            Self::I8 => 1,
            Self::U16 => 1,
            Self::I16 => 1,
            Self::U32 => 1,
            Self::I32 => 1,
            Self::U64 => 1,
            Self::I64 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => {
                k_quants::QK_K
            }
            Self::QAWQ | Self::QAWQ_G64 => k_quants::QK_AWQ,
            Self::Q4_KS => k_quants::QK_Q4_KS,
            Self::Q8_KS => k_quants::QK_Q8_KS,
            Self::Q2_0 => k_quants::QK2_0,
            Self::Q3_0 => k_quants::QK3_0,
            Self::R16 => k_quants::QK_R16,
            Self::Q0 => k_quants::QK_Q0,
            Self::Q1_S => k_quants::QK1_S,
            Self::Q2_S => k_quants::QK2_S,
            Self::Q2_A => k_quants::QK2_A,
            Self::Q2_1 => k_quants::QK2_1,
            Self::Q3_1 => k_quants::QK3_1,
            Self::P2 => k_quants::QK_P2,
            Self::Q0_V => k_quants::QK_Q0_V,
            Self::Q1_A => k_quants::QK1_A,
            Self::Q0_X => k_quants::QK_Q0_X,
            Self::Q0_M2 => k_quants::QK_Q0_M2,
            Self::Q0_M4 => k_quants::QK_Q0_M4,
            // KO compact blocks hold 128 elements (K/128 granularity).
            Self::Q4_KO => BlockQ4_KO::BLCK_SIZE,
            Self::Q5_KO => BlockQ5_KO::BLCK_SIZE,
            Self::Q6_KO => BlockQ6_KO::BLCK_SIZE,
            Self::Q8_KO => BlockQ8_KO::BLCK_SIZE,
            Self::MXFP4 => k_quants::QK_MXFP4,
            // K/128 granularity (the collapse is per 128-K tile), like the other KO twins.
            Self::MXFP4_KO => 128,
            // K/128 granularity like the other KO twins (per-128 affine).
            Self::Q2_KO => 128,
        }
    }
}

// A version of GgmlType without `vec_dot` so that it can be dyn boxed.
pub trait QuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage>;
    fn storage_size_in_bytes(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn block_size(&self) -> usize;
    #[allow(clippy::wrong_self_convention)]
    fn from_float(&mut self, xs: &[f32]);
    fn size(&self) -> usize;
    fn clone_box(&self) -> Box<dyn QuantizedType>;
}

impl<T: k_quants::GgmlType + Send + Sync + Clone + 'static> QuantizedType for Vec<T> {
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }

    fn size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }

    fn from_float(&mut self, xs: &[f32]) {
        T::from_float(xs, self)
    }

    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.as_slice(), &mut ys);
        Ok(CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }

    fn clone_box(&self) -> Box<dyn QuantizedType> {
        Box::new(self.clone())
    }
}

impl std::fmt::Debug for LiveQTensor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QTensor[{:?}; {:?}]", self.shape, self.dtype())
    }
}

fn check_shape(shape: &Shape, block_size: usize) -> Result<()> {
    let dims = shape.dims();
    if dims.is_empty() {
        crate::bail!("scalar tensor cannot be quantized {shape:?}")
    }
    if !dims[dims.len() - 1].is_multiple_of(block_size) {
        crate::bail!(
            "quantized tensor must have their last dim divisible by block size {shape:?} {}",
            block_size
        )
    }
    Ok(())
}

/// The constructors, which all allocate.
///
/// They are on `QTensor` rather than on `LiveQTensor<'w>` so that `'w` cannot
/// be *inferred* at a call site: fresh memory is owned, therefore `'static`,
/// and saying so here is what keeps the parameter meaningful everywhere else.
/// The accessors and views live in the `impl<'w>` block below.
impl QTensor {
    pub fn new<S: Into<Shape>>(storage: QStorage, shape: S) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, storage.block_size())?;
        Ok(Self {
            storage,
            shape,
            lease: PhantomData,
        })
    }

    /// Create a zero-initialized quantized tensor.
    ///
    /// Used for arena allocation in quantized KV cache.
    ///
    /// # Arguments
    /// * `shape` - Shape of the tensor (must be divisible by block size)
    /// * `dtype` - Quantization type (Q4_0, Q8_0, etc.)
    /// * `device` - Device to allocate on (CPU, CUDA, Metal)
    ///
    /// # Example
    /// ```ignore
    /// let arena = QTensor::zeros((1024, 128), GgmlDType::Q8_0, &device)?;
    /// ```
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: GgmlDType, device: &Device) -> Result<Self> {
        let shape = shape.into();
        let block_size = dtype.block_size();
        check_shape(&shape, block_size)?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        let storage = device.qzeros(elem_count, dtype)?;
        Self::new(storage, shape)
    }

    pub fn quantize(src: &Tensor, dtype: GgmlDType) -> Result<Self> {
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        // force_contiguous ensures the storage exactly matches the shape,
        // even for narrowed/sliced tensors where is_contiguous() may be true
        // but the underlying storage has more elements.
        let src = src
            .to_dtype(crate::DType::F32)?
            .flatten_all()?
            .force_contiguous()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
            lease: PhantomData,
        })
    }

    /// Create a `QTensor` from raw GGML quantized bytes, backed by host-mapped
    /// (pinned) memory instead of VRAM.
    ///
    /// This is the VRAM-overflow equivalent of [`ggml_file::qtensor_from_ggml`].
    /// The tensor data lives in pinned host RAM and is GPU-accessible over PCIe.
    /// CUDA kernels work transparently — just at PCIe bandwidth instead of VRAM.
    ///
    /// Returns `(qtensor, guard)`. The caller must keep `guard` alive for the
    /// lifetime of the tensor.
    ///
    /// On non-CUDA devices, falls back to `qtensor_from_ggml` (no guard needed).
    #[cfg(feature = "cuda")]
    pub fn from_host_mapped_ggml(
        ggml_dtype: GgmlDType,
        raw_data: &[u8],
        dims: Vec<usize>,
        device: &Device,
    ) -> Result<(Self, cuda::HostMappedAlloc)> {
        match device {
            Device::Cuda(cuda_dev) => {
                let elem_count: usize = dims.iter().product();
                let block_size = ggml_dtype.block_size();
                if !elem_count.is_multiple_of(block_size) {
                    crate::bail!(
                        "element count {elem_count} not divisible by block size {block_size}"
                    );
                }
                let (storage, guard) = cuda::QCudaStorage::from_host_mapped(
                    raw_data, elem_count, ggml_dtype, cuda_dev,
                )?;
                let qt = Self::new(QStorage::Cuda(storage), dims)?;
                Ok((qt, guard))
            }
            _ => {
                // Non-CUDA: no host-mapped concept, use a dummy guard.
                // This path should not normally be hit — callers gate on Device::Cuda.
                crate::bail!("from_host_mapped_ggml requires a CUDA device");
            }
        }
    }
}

/// Everything that reads, writes, or views an existing quantized tensor, and so
/// works the same whether it owns its memory or leases it.
///
/// Methods here that produce a *new* allocation say `QTensor` in their return
/// type rather than `Self`: the result is owned, and inheriting the receiver's
/// `'w` would needlessly shorten it.
impl<'w> LiveQTensor<'w> {
    pub fn dtype(&self) -> GgmlDType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    /// Get a mutable reference to the underlying storage.
    ///
    /// Used by the expert LRU cache to overwrite pre-allocated VRAM slot
    /// contents via `QCudaStorage::copy_from_host`.
    pub fn storage_mut(&mut self) -> &mut QStorage {
        &mut self.storage
    }

    /// Symmetric read counterpart of [`Self::write_bytes_at`]. Copy
    /// `dst.len()` bytes from this QTensor's CPU storage at
    /// `byte_offset` into `dst`. The warm→hot batched-async path uses
    /// this to gather warm-resident arena bytes into pinned host
    /// scratch ahead of the HtoD scatter.
    pub fn read_bytes_at(&self, byte_offset: usize, dst: &mut [u8]) -> Result<()> {
        let src_size = self.storage.size_in_bytes();
        if byte_offset + dst.len() > src_size {
            crate::bail!(
                "read_bytes_at: range {}..{} exceeds storage size {src_size}",
                byte_offset,
                byte_offset + dst.len()
            )
        }
        match &self.storage {
            QStorage::Cpu(storage) => {
                let src_ptr = storage.as_ptr();
                // SAFETY: `src_ptr` is valid for `src_size` bytes;
                // `byte_offset + dst.len()` is bounds-checked above;
                // source and dest don't overlap (dst is caller-owned).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src_ptr.add(byte_offset),
                        dst.as_mut_ptr(),
                        dst.len(),
                    );
                }
                Ok(())
            }
            _ => crate::bail!(
                "read_bytes_at: only CPU storage supported (got {:?})",
                self.storage.device()
            ),
        }
    }

    /// Async HtoD byte-write counterpart of [`Self::write_bytes_at`]:
    /// memcpy `bytes` into this QTensor's CUDA storage at
    /// `byte_offset`, enqueued on `stream`. **CUDA storage only.**
    ///
    /// Does **not** synchronise — the caller is responsible for
    /// calling `stream.synchronize()` (or fencing through subsequent
    /// stream work) before assuming the bytes have landed. For the
    /// transfer to be truly async on the GPU (rather than the driver
    /// inserting a hidden bounce buffer), `bytes` should live in
    /// `cuMemHostAlloc`'d pinned host memory.
    ///
    /// Mirrors the byte-write half of `QStorage::slice_scatter` for
    /// the Cuda case, but takes a raw `&[u8]` source — useful when
    /// the host bytes come from outside the QTensor type system (e.g.
    /// a pinned scratch buffer fed by NVMe / GPU-gather).
    #[cfg(feature = "cuda")]
    pub fn write_bytes_at_async(
        &mut self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        byte_offset: usize,
        bytes: &[u8],
    ) -> Result<()> {
        use crate::cuda_backend::WrapErr;
        let dst_size = self.storage.size_in_bytes();
        if byte_offset + bytes.len() > dst_size {
            crate::bail!(
                "write_bytes_at_async: range {}..{} exceeds storage size {dst_size}",
                byte_offset,
                byte_offset + bytes.len()
            )
        }
        match &mut self.storage {
            QStorage::Cuda(storage) => {
                let mut dst_view = storage
                    .bytes_mut()
                    .slice_mut(byte_offset..byte_offset + bytes.len());
                stream.memcpy_htod(bytes, &mut dst_view).w()?;
                Ok(())
            }
            QStorage::Cpu(_) => {
                crate::bail!("write_bytes_at_async: CPU storage — use the sync `write_bytes_at`")
            }
            _ => crate::bail!(
                "write_bytes_at_async: only CUDA storage supported (got {:?})",
                self.storage.device()
            ),
        }
    }

    /// Async DtoH byte-read counterpart of [`Self::read_bytes_at`]:
    /// memcpy `dst.len()` bytes from this QTensor's CUDA storage at
    /// `byte_offset` into `dst`, enqueued on `stream`. **CUDA storage
    /// only.**
    ///
    /// Does **not** synchronise — caller must `stream.synchronize()`
    /// (or otherwise fence) before reading `dst`. Pinned `dst` (via
    /// `cuMemHostAlloc`) is required for a true async DMA; pageable
    /// `dst` triggers a hidden driver-side bounce buffer.
    #[cfg(feature = "cuda")]
    pub fn read_bytes_at_async(
        &self,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        byte_offset: usize,
        dst: &mut [u8],
    ) -> Result<()> {
        use crate::cuda_backend::WrapErr;
        let src_size = self.storage.size_in_bytes();
        if byte_offset + dst.len() > src_size {
            crate::bail!(
                "read_bytes_at_async: range {}..{} exceeds storage size {src_size}",
                byte_offset,
                byte_offset + dst.len()
            )
        }
        match &self.storage {
            QStorage::Cuda(storage) => {
                let src_view = storage.bytes().slice(byte_offset..byte_offset + dst.len());
                stream.memcpy_dtoh(&src_view, dst).w()?;
                Ok(())
            }
            QStorage::Cpu(_) => {
                crate::bail!("read_bytes_at_async: CPU storage — use the sync `read_bytes_at`")
            }
            _ => crate::bail!(
                "read_bytes_at_async: only CUDA storage supported (got {:?})",
                self.storage.device()
            ),
        }
    }

    /// Memcpy `bytes` into this QTensor's underlying byte slab at
    /// `byte_offset`. **CPU storage only.** The arena KV-cache code in
    /// `candle-nn` treats `QTensor` as a flat slab of quantized bytes
    /// — `slice_scatter` / `slice_range_copy` ultimately do exactly
    /// this same `ptr::copy_nonoverlapping`, just with a `QStorage`
    /// source instead of a raw byte slice. This is the bytes-from-host
    /// entry point the hot→warm batched-async path needs after
    /// pulling a chunk's bytes off the GPU into pinned host scratch.
    ///
    /// No format awareness: the caller is responsible for ensuring
    /// `bytes` is a valid quantized blob (right block count × right
    /// `type_size`).
    pub fn write_bytes_at(&mut self, byte_offset: usize, bytes: &[u8]) -> Result<()> {
        let dst_size = self.storage.size_in_bytes();
        if byte_offset + bytes.len() > dst_size {
            crate::bail!(
                "write_bytes_at: range {}..{} exceeds storage size {dst_size}",
                byte_offset,
                byte_offset + bytes.len()
            )
        }
        match &mut self.storage {
            QStorage::Cpu(storage) => {
                let dst_ptr = storage.as_ptr() as *mut u8;
                // SAFETY: `dst_ptr` is the CPU storage's owned allocation,
                // valid for `dst_size` bytes; `byte_offset + bytes.len()`
                // is bounds-checked above; `bytes` and the destination
                // don't overlap (caller owns `bytes`).
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        bytes.as_ptr(),
                        dst_ptr.add(byte_offset),
                        bytes.len(),
                    );
                }
                Ok(())
            }
            _ => crate::bail!(
                "write_bytes_at: only CPU storage supported (got {:?})",
                self.storage.device()
            ),
        }
    }

    /// Get a shared reference to the underlying storage.
    pub fn storage(&self) -> &QStorage {
        &self.storage
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let storage = self.storage.dequantize(self.shape.elem_count())?;
        let none = crate::op::BackpropOp::none();
        crate::tensor::from_storage(storage, self.shape.clone(), none, false).to_device(device)
    }

    pub fn dequantize_f16(&self, device: &Device) -> Result<Tensor> {
        // In the CUDA case, we have a specialized kernel as this can be useful for volta
        // architectures. https://github.com/huggingface/candle/issues/2136
        match &self.storage {
            QStorage::Cuda(s) => {
                let s = s.dequantize_f16(self.shape.elem_count())?;
                let none = crate::op::BackpropOp::none();
                crate::tensor::from_storage(Storage::Cuda(s), self.shape.clone(), none, false)
                    .to_device(device)
            }
            _ => {
                let s = self.dequantize(device)?.to_dtype(crate::DType::F16)?;
                Ok(s)
            }
        }
    }

    pub fn dequantize_bf16(&self, device: &Device) -> Result<Tensor> {
        // In the CUDA case, we have a specialized kernel path (currently via F16)
        match &self.storage {
            QStorage::Cuda(s) => {
                let s = s.dequantize_bf16(self.shape.elem_count())?;
                let none = crate::op::BackpropOp::none();
                crate::tensor::from_storage(Storage::Cuda(s), self.shape.clone(), none, false)
                    .to_device(device)
            }
            _ => {
                let s = self.dequantize(device)?.to_dtype(crate::DType::BF16)?;
                Ok(s)
            }
        }
    }

    /// Repack quantized weights to K/128 format with embedded scales.
    ///
    /// The K/128 format stores 128 elements per block with scales embedded inline,
    /// enabling efficient 16-thread cooperative loads in CUDA kernels.
    ///
    /// # Returns
    /// A new QTensor with repacked storage
    #[cfg(feature = "cuda")]
    pub fn repack_gemx(&self) -> Result<QTensor> {
        match &self.storage {
            QStorage::Cuda(s) => {
                let new_storage = s.repack_gemx(&self.shape)?;
                Ok(QTensor {
                    storage: QStorage::Cuda(new_storage),
                    shape: self.shape.clone(),
                    lease: PhantomData,
                })
            }
            _ => crate::bail!("repack_gemx is only supported on CUDA"),
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn repack_gemx(&self) -> Result<QTensor> {
        crate::bail!("repack_gemx requires the cuda feature")
    }

    /// Get the size in bytes after GEMX repacking, without actually repacking.
    #[cfg(feature = "cuda")]
    pub fn repacked_size(&self) -> Result<usize> {
        match &self.storage {
            QStorage::Cuda(s) => s.repacked_size(&self.shape),
            _ => crate::bail!("repacked_size is only supported on CUDA"),
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn repacked_size(&self) -> Result<usize> {
        crate::bail!("repacked_size requires the cuda feature")
    }

    /// Check if this tensor's dtype supports GEMX repacking.
    #[cfg(feature = "cuda")]
    pub fn supports_gemx_repacking(&self) -> bool {
        match &self.storage {
            QStorage::Cuda(s) => s.supports_gemx_repacking(),
            _ => false,
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn supports_gemx_repacking(&self) -> bool {
        false
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.size_in_bytes()
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        self.storage.data()
    }

    /// Read `range` bytes of the underlying raw quantized data.
    ///
    /// Replaces the `&qtensor.data()?[range]` pattern: that form pulls
    /// the **entire** arena across PCIe (for CUDA) or the full buffer
    /// (for Metal) and then throws away everything outside `range`.
    /// This method does a single ranged DMA for exactly `range.len()`
    /// bytes — no kernel, no full-buffer copy.
    ///
    /// On CPU storage it returns a borrowed slice (zero copy).  On
    /// CUDA / Metal it returns an owned `Vec<u8>` of just the requested
    /// span.
    pub fn data_range(&self, range: std::ops::Range<usize>) -> Result<Cow<'_, [u8]>> {
        self.storage.data_range(range)
    }

    /// Copy the raw quantized data to a host buffer on the given CUDA stream.
    ///
    /// When `dst` is backed by pinned memory (`cuMemAllocHost`), the copy
    /// is truly asynchronous — the CPU returns immediately.  This is the
    /// D2H eviction path for the two-tier expert cache.
    ///
    /// `dst` must be at least [`storage_size_in_bytes()`](Self::storage_size_in_bytes) bytes.
    #[cfg(feature = "cuda")]
    pub fn copy_data_to_host_on_stream(
        &self,
        dst: &mut [u8],
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    ) -> Result<()> {
        match &self.storage {
            QStorage::Cuda(s) => s.copy_to_host_on_stream(dst, stream),
            _ => crate::bail!("copy_data_to_host_on_stream requires CUDA storage"),
        }
    }

    /// A `QTensor` over quantized device memory it does **not** own.
    ///
    /// The quantized counterpart of [`crate::Tensor::from_leased_cuda_ptr`],
    /// and the way a KV arena slot is handed to the block quantize /
    /// dequantize paths now that an arena is a run of untyped byte slots
    /// (`docs/archived/arena_unification.md` principle 8). Writes through the returned
    /// tensor land in the caller's memory — which is the point: cloning a
    /// `QTensor` is a device-to-device **copy**, so "clone the arena and write
    /// to it" silently writes to a throwaway.
    ///
    /// The view has no matrix-row padding, so it cannot be used as a matmul
    /// operand — `CustomOp1` is implemented for `QTensor` alone, so the result
    /// of this call is rejected at compile time. The block quantize /
    /// dequantize kernels do not read past the data and are the only intended
    /// consumers.
    ///
    /// # Safety
    /// `ptr` must point to at least `ceil(elem_count / block_size) *
    /// type_size` bytes of device memory that is un-aliased for writes.
    ///
    /// The caller also chooses `'w`, and choosing it too long is how this call
    /// goes wrong: it must not outlive the memory `ptr` addresses. Nothing in
    /// the arguments pins it, so an unannotated binding can infer `'static`.
    /// Prefer a wrapper that ties `'w` to a borrow of the owner — the KV
    /// arena's `qslot_view` returns `LiveQTensor<'a>` from `&'a self`, which
    /// discharges this obligation by construction.
    #[cfg(feature = "cuda")]
    pub unsafe fn from_leased_cuda_ptr(
        ptr: u64,
        dtype: GgmlDType,
        elem_count: usize,
        device: &crate::CudaDevice,
        origin: crate::cuda_backend::wave_provenance::LeaseOrigin,
    ) -> Result<Self> {
        let storage =
            cuda::QCudaStorage::from_leased_device_ptr(ptr, elem_count, dtype, device, origin)?;
        Ok(Self {
            storage: QStorage::Cuda(storage),
            shape: Shape::from(elem_count),
            lease: PhantomData,
        })
    }

    /// A `'static` copy of this tensor's contents.
    ///
    /// The sanctioned way to let a lease's data outlive its owner.
    /// [`QCudaStorage::clone`](cuda::QCudaStorage) is a device-to-device copy
    /// that always yields owned storage, so this really does allocate — it is
    /// `Clone` with the lifetime told truthfully, rather than
    /// `Clone::clone`, which must return `Self` and so needlessly inherits
    /// `'w`.
    pub fn to_owned_qtensor(&self) -> QTensor {
        QTensor {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            lease: PhantomData,
        }
    }

    /// Get the CUDA device pointer for the raw quantized data.
    /// Returns None for non-CUDA storage.
    ///
    /// This is used by paged attention kernels that need direct GPU access
    /// to quantized KV cache arenas.
    #[cfg(feature = "cuda")]
    pub fn cuda_data_ptr(&self) -> Option<u64> {
        self.storage.cuda_data_ptr()
    }

    /// Get a mutable CUDA device pointer for the raw quantized data.
    /// Returns None for non-CUDA storage.
    ///
    /// This is used by batched quantization kernels that write directly
    /// to quantized KV cache arenas.
    #[cfg(feature = "cuda")]
    pub fn cuda_data_ptr_mut(&mut self) -> Option<u64> {
        self.storage.cuda_data_ptr_mut()
    }

    /// Copy quantized blocks from `src` into `self` at the given element offset.
    ///
    /// This is the primary mechanism for writing into quantized KV cache arenas.
    /// Both tensors must have the same dtype and be on the same device.
    ///
    /// # Arguments
    /// * `src` - Source QTensor to copy from (must be 1D or have same shape except for scatter dim)
    /// * `elem_offset` - Element offset (must be block-aligned, i.e., multiple of block_size)
    ///
    /// # Requirements
    /// - Both tensors must have the same dtype
    /// - Both tensors must be on the same device
    /// - `elem_offset` must be a multiple of block_size (32)
    /// - `elem_offset + src.elem_count()` must not exceed `self.elem_count()`
    ///
    /// # Example
    /// ```ignore
    /// // Create arena of 256 elements
    /// let arena = QTensor::zeros((256,), GgmlDType::Q8_0, &device)?;
    ///
    /// // Create new KV data (64 elements = 2 blocks)
    /// let new_kv = QTensor::quantize(&new_data, GgmlDType::Q8_0)?;
    ///
    /// // Scatter at offset 64 (block-aligned)
    /// arena.slice_scatter(&new_kv, 64)?;
    /// ```
    pub fn slice_scatter(&mut self, src: &QTensor, elem_offset: usize) -> Result<()> {
        // Validate dtype match
        if self.dtype() != src.dtype() {
            crate::bail!(
                "slice_scatter: dtype mismatch ({:?} vs {:?})",
                self.dtype(),
                src.dtype()
            )
        }

        // Validate device match
        if self.device().location() != src.device().location() {
            crate::bail!("slice_scatter: device mismatch")
        }

        // Validate block alignment
        let block_size = self.storage.block_size();
        if !elem_offset.is_multiple_of(block_size) {
            crate::bail!(
                "slice_scatter: element offset {} is not aligned to block size {}",
                elem_offset,
                block_size
            )
        }

        // Validate bounds
        let src_elems = src.shape.elem_count();
        let dst_elems = self.shape.elem_count();
        if elem_offset + src_elems > dst_elems {
            crate::bail!(
                "slice_scatter: source ({} elements) at offset {} exceeds destination ({} elements)",
                src_elems,
                elem_offset,
                dst_elems
            )
        }

        // Calculate byte offset based on blocks
        let block_offset = elem_offset / block_size;
        let bytes_per_block = self.dtype().type_size();
        let byte_offset = block_offset * bytes_per_block;

        self.storage.slice_scatter(&src.storage, byte_offset)
    }

    /// Copy a range of elements from another QTensor into this one, without
    /// dequantization. Both must have the same dtype and be on the same device.
    /// Offsets must be block-aligned.
    pub fn slice_range_copy(
        &mut self,
        src: &QTensor,
        src_elem_offset: usize,
        dst_elem_offset: usize,
        elem_count: usize,
    ) -> Result<()> {
        if self.dtype() != src.dtype() {
            crate::bail!(
                "slice_range_copy: dtype mismatch ({:?} vs {:?})",
                self.dtype(),
                src.dtype()
            )
        }

        let block_size = self.storage.block_size();
        if !src_elem_offset.is_multiple_of(block_size)
            || !dst_elem_offset.is_multiple_of(block_size)
            || !elem_count.is_multiple_of(block_size)
        {
            crate::bail!(
                "slice_range_copy: offsets/count must be block-aligned (block_size={})",
                block_size
            )
        }

        let bytes_per_block = self.dtype().type_size();
        let src_byte_offset = (src_elem_offset / block_size) * bytes_per_block;
        let dst_byte_offset = (dst_elem_offset / block_size) * bytes_per_block;
        let byte_len = (elem_count / block_size) * bytes_per_block;

        self.storage
            .slice_range_copy(&src.storage, src_byte_offset, dst_byte_offset, byte_len)
    }

    /// Quantize a float tensor directly into this QTensor at the given element offset.
    ///
    /// This is optimized for GPU operations by using CUDA quantization kernels that
    /// write directly to the destination buffer, avoiding intermediate allocations.
    /// This is the preferred method for quantizing data into KV cache arenas.
    ///
    /// # Arguments
    /// * `src` - Source float tensor (will be converted to f32 and made contiguous)
    /// * `elem_offset` - Element offset (must be block-aligned, i.e., multiple of block_size)
    ///
    /// # Requirements
    /// - Both tensors must be on the same CUDA device
    /// - `elem_offset` must be a multiple of block_size (32 for standard, 256 for K-quants)
    /// - `elem_offset + src.elem_count()` must not exceed `self.elem_count()`
    ///
    /// # Example
    /// ```ignore
    /// // Create arena of 256 elements
    /// let arena = QTensor::zeros((256,), GgmlDType::Q8_0, &device)?;
    ///
    /// // Float data to quantize (64 elements = 2 blocks)
    /// let float_data = Tensor::randn(0.0, 1.0, (64,), &device)?;
    ///
    /// // Quantize directly at offset 64 (block-aligned)
    /// arena.quantize_into(&float_data, 64)?;
    /// ```
    #[cfg(feature = "cuda")]
    /// `src` may live on an inference wave: this reads its bytes and quantizes
    /// them into the slot, retaining nothing, so the slot does not outlive the
    /// generation `src` came from.
    pub fn quantize_into(&mut self, src: &LiveTensor<'_>, elem_offset: usize) -> Result<()> {
        // Validate device match
        if self.device().location() != src.device().location() {
            crate::bail!("quantize_into: device mismatch")
        }

        // Validate block alignment
        let block_size = self.storage.block_size();
        if !elem_offset.is_multiple_of(block_size) {
            crate::bail!(
                "quantize_into: element offset {} is not aligned to block size {}",
                elem_offset,
                block_size
            )
        }

        // Validate bounds
        let src_elems = src.elem_count();
        let dst_elems = self.shape.elem_count();
        if elem_offset + src_elems > dst_elems {
            crate::bail!(
                "quantize_into: source ({} elements) at offset {} exceeds destination ({} elements)",
                src_elems,
                elem_offset,
                dst_elems
            )
        }

        // Convert source to f32 and make contiguous
        let src = src
            .to_dtype(crate::DType::F32)?
            .flatten_all()?
            .force_contiguous()?;

        // Calculate byte offset
        let block_offset = elem_offset / block_size;
        let bytes_per_block = self.dtype().type_size();
        let byte_offset = block_offset * bytes_per_block;

        // Get underlying storage and call quantize_into
        let (src_storage, _src_layout) = src.storage_and_layout();
        self.storage
            .quantize_into(&src_storage, src_elems, byte_offset)
    }

    /// Quantize f32 data with fused transpose from [H, T, D] to [H, D, T] layout.
    ///
    /// This fuses the memory layout transformation with quantization to avoid
    /// intermediate allocations. Used for KV cache quantization where:
    /// - Input layout: [n_head, chunk_size, head_dim] - channel-oriented float
    /// - Output layout: [n_head, head_dim, chunk_size] - token-oriented quant
    ///
    /// # Arguments
    /// * `src` - Source tensor with shape [n_head, chunk_size, head_dim]
    /// * `elem_offset` - Element offset in destination (block-aligned for this dtype)
    ///
    /// # Supported Types
    /// Only Q4_0 and Q8_0 are supported for fused transpose+quantize.
    ///
    /// # Example
    /// ```ignore
    /// // Quantize a [8, 32, 128] float chunk into quantized storage at offset 0
    /// quantized_arena.quantize_transposed_into(&float_chunk, 0)?;
    /// // The result is stored as [8, 128] Q8_0/Q4_0 blocks (token-oriented layout)
    /// ```
    #[cfg(feature = "cuda")]
    pub fn quantize_transposed_into(&mut self, src: &Tensor, elem_offset: usize) -> Result<()> {
        // Validate device match
        if self.device().location() != src.device().location() {
            crate::bail!("quantize_transposed_into: device mismatch")
        }

        // Validate supported quantized dtypes (all standard 32-element formats)
        match self.dtype() {
            GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q8_1
            | GgmlDType::Q4_KS
            | GgmlDType::Q8_KS => {}
            other => crate::bail!(
                "quantize_transposed_into: only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q4_KS/Q8_KS supported, got {:?}",
                other
            ),
        }

        // Validate supported source dtypes - kernel handles conversion inline
        match src.dtype() {
            crate::DType::F32 | crate::DType::F16 | crate::DType::BF16 | crate::DType::F8E4M3 => {}
            other => crate::bail!(
                "quantize_transposed_into: source dtype must be F32/F16/BF16/F8E4M3, got {:?}",
                other
            ),
        }

        // Validate input shape
        let dims = src.dims();
        if dims.len() != 3 {
            crate::bail!(
                "quantize_transposed_into: expected 3D tensor [H, T, D], got {}D",
                dims.len()
            )
        }
        let n_head = dims[0];
        let chunk_size = dims[1];
        let head_dim = dims[2];

        // Validate chunk_size is 32 (required for Q4_0/Q8_0 GGML blocks)
        if chunk_size != 32 {
            crate::bail!(
                "quantize_transposed_into: chunk_size must be 32 for Q4_0/Q8_0, got {}",
                chunk_size
            )
        }

        // Validate block alignment
        let block_size = self.storage.block_size();
        if !elem_offset.is_multiple_of(block_size) {
            crate::bail!(
                "quantize_transposed_into: element offset {} not aligned to block size {}",
                elem_offset,
                block_size
            )
        }

        // Validate bounds: output has n_head * head_dim blocks of chunk_size elements each
        let src_elems = src.elem_count();
        let dst_elems = self.shape.elem_count();
        if elem_offset + src_elems > dst_elems {
            crate::bail!(
                "quantize_transposed_into: source ({} elements) at offset {} exceeds destination ({} elements)",
                src_elems,
                elem_offset,
                dst_elems
            )
        }

        // Ensure contiguous layout for correct memory access pattern
        // Kernel assumes [H, T, D] row-major layout with strides [T*D, D, 1]
        let src = src.force_contiguous()?;

        // Calculate byte offset
        let block_offset = elem_offset / block_size;
        let bytes_per_block = self.dtype().type_size();
        let byte_offset = block_offset * bytes_per_block;

        // Get underlying storage and call quantize_transposed_into
        let (src_storage, _src_layout) = src.storage_and_layout();
        self.storage.quantize_transposed_into(
            &src_storage,
            n_head,
            chunk_size,
            head_dim,
            byte_offset,
        )
    }

    /// Dequantize a range of elements from this quantized tensor into a destination tensor.
    ///
    /// This efficiently dequantizes a specific chunk without processing the entire tensor.
    ///
    /// # Arguments
    /// * `dst` - Destination tensor (must be f16, bf16, or f32, contiguous, on same device)
    /// * `src_elem_offset` - Element offset in source (this tensor) to read from
    /// * `dst_elem_offset` - Element offset in destination to write to
    /// * `elem_count` - Number of elements to dequantize
    ///
    /// # Example
    /// ```ignore
    /// // Dequantize 128 elements from offset 256 in quantized tensor to offset 0 in float tensor
    /// quantized_arena.dequantize_into(&mut float_chunk, 256, 0, 128)?;
    /// ```
    #[cfg(feature = "cuda")]
    pub fn dequantize_into(
        &self,
        dst: &mut Tensor,
        src_elem_offset: usize,
        dst_elem_offset: usize,
        elem_count: usize,
    ) -> Result<()> {
        // Validate device match
        if self.device().location() != dst.device().location() {
            crate::bail!("dequantize_into: device mismatch")
        }

        // Validate block alignment
        let block_size = self.storage.block_size();
        if !src_elem_offset.is_multiple_of(block_size) {
            crate::bail!(
                "dequantize_into: source element offset {} is not aligned to block size {}",
                src_elem_offset,
                block_size
            )
        }

        // Validate source bounds
        let src_elems = self.shape.elem_count();
        if src_elem_offset + elem_count > src_elems {
            crate::bail!(
                "dequantize_into: reading {} elements at offset {} exceeds source ({} elements)",
                elem_count,
                src_elem_offset,
                src_elems
            )
        }

        // Validate destination bounds
        let dst_elems = dst.elem_count();
        if dst_elem_offset + elem_count > dst_elems {
            crate::bail!(
                "dequantize_into: writing {} elements at offset {} exceeds destination ({} elements)",
                elem_count,
                dst_elem_offset,
                dst_elems
            )
        }

        // Calculate byte offset in source
        let block_offset = src_elem_offset / block_size;
        let bytes_per_block = self.dtype().type_size();
        let byte_offset = block_offset * bytes_per_block;

        // Get underlying storage and call dequantize_into
        let (mut dst_storage, _dst_layout) = dst.storage_mut_and_layout();
        self.storage
            .dequantize_into(&mut dst_storage, elem_count, byte_offset, dst_elem_offset)
    }

    /// Concatenate multiple quantized tensors along the first dimension (row concat).
    ///
    /// This is primarily intended for CUDA inference optimizations (e.g. fusing Q/K/V
    /// projections into a single matmul) without dequantizing/requantizing.
    ///
    /// Requirements:
    /// - All tensors are rank-2 with shapes (n_i, k)
    /// - All tensors share the same dtype, device, and k
    #[cfg(feature = "cuda")]
    pub fn concat_rows_cuda(qtensors: &[&QTensor]) -> Result<QTensor> {
        if qtensors.is_empty() {
            crate::bail!("concat_rows_cuda requires at least one tensor")
        }

        let dtype = qtensors[0].dtype();
        let device = qtensors[0].device();
        let (mut total_n, k) = qtensors[0].shape.dims2()?;

        for (i, t) in qtensors.iter().enumerate() {
            if t.rank() != 2 {
                crate::bail!(
                    "concat_rows_cuda expects rank-2 tensors, got rank {}",
                    t.rank()
                )
            }
            if t.dtype() != dtype {
                crate::bail!(
                    "concat_rows_cuda dtype mismatch at {i}: {:?} vs {:?}",
                    t.dtype(),
                    dtype
                )
            }
            if t.device().location() != device.location() {
                crate::bail!("concat_rows_cuda device mismatch at {i}")
            }
            let (n_i, k_i) = t.shape.dims2()?;
            if k_i != k {
                crate::bail!("concat_rows_cuda column mismatch at {i}: {} vs {}", k_i, k)
            }
            if i != 0 {
                total_n += n_i;
            }
        }

        match (&qtensors[0].storage, device) {
            (QStorage::Cuda(_), Device::Cuda(cuda_dev)) => {
                let elem_count = total_n * k;
                let mut out_storage = cuda::QCudaStorage::zeros(&cuda_dev, elem_count, dtype)?;
                let mut byte_off = 0usize;

                for (i, t) in qtensors.iter().enumerate() {
                    let (n_i, k_i) = t.shape.dims2()?;
                    let _ = (n_i, k_i);

                    let src = match &t.storage {
                        QStorage::Cuda(s) => s,
                        _ => crate::bail!("concat_rows_cuda expects CUDA storage at {i}"),
                    };
                    let len = src.byte_len();

                    let src_view = src.bytes();
                    let mut dst_view = out_storage.bytes_mut().slice_mut(byte_off..byte_off + len);

                    cuda_dev.memcpy_dtod(&src_view, &mut dst_view)?;
                    byte_off += len;
                }

                let out = QTensor::new(QStorage::Cuda(out_storage), (total_n, k))?;
                Ok(out)
            }
            _ => crate::bail!("concat_rows_cuda is only supported for CUDA QTensor storage"),
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn concat_rows_cuda(_qtensors: &[&QTensor]) -> Result<QTensor> {
        crate::bail!("concat_rows_cuda requires the cuda feature")
    }
}

#[derive(Clone, Debug)]
pub enum QMatMul {
    QTensor(std::sync::Arc<QTensor>),
    Tensor(Tensor),
    TensorF16(Tensor),
}

impl QMatMul {
    /// Get the inner `QTensor` if this is the quantized variant.
    pub fn qtensor(&self) -> Option<&QTensor> {
        match self {
            QMatMul::QTensor(qt) => Some(qt),
            _ => None,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL_F16: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL_F16") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL_BF16: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL_BF16") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl QMatMul {
    pub fn from_arc(qtensor: std::sync::Arc<QTensor>) -> Result<Self> {
        let dequantize = match qtensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => true,
            _ => DEQUANTIZE_ALL.with(|b| *b),
        };
        let t = if dequantize {
            let tensor = qtensor.dequantize(&qtensor.device())?;
            Self::Tensor(tensor)
        } else if DEQUANTIZE_ALL_F16.with(|b| *b) {
            let tensor = qtensor.dequantize_f16(&qtensor.device())?;
            Self::TensorF16(tensor)
        } else if DEQUANTIZE_ALL_BF16.with(|b| *b) {
            let tensor = qtensor.dequantize_bf16(&qtensor.device())?;
            Self::TensorF16(tensor)
        } else {
            Self::QTensor(qtensor)
        };
        Ok(t)
    }

    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Self::from_arc(std::sync::Arc::new(qtensor))
    }

    /// Repack the weight for the inference numeric mode selected by `mode`: any int8 mode → the
    /// lane-major KO format read by the q8a128 int8 tensor-core matmul (`repack_ko`, twin chosen
    /// by [`GgmlDType::to_ko`]); [`Int8Mode::Off`] → the FP GEMX repack read by the dequant-weight
    /// float matmul (`repack_gemx`). The weight-side half of the single `int8mode` knob (paired
    /// with `cuda::to_dynamic` on the activation side); the matmul's KO⇔int8 pairing guard keeps
    /// the two consistent. Requires the quantized (`QTensor`) variant on CUDA.
    /// Non-CUDA build: the KO/GEMX repacked layouts are CUDA-kernel formats, so there is
    /// nothing to repack to — this always errors (the `dummy_cuda` convention).
    #[cfg(not(feature = "cuda"))]
    pub fn repack_for_optimization(&self, _mode: Int8Mode) -> Result<QMatMul> {
        crate::bail!("repack_for_optimization requires the cuda feature")
    }

    #[cfg(feature = "cuda")]
    pub fn repack_for_optimization(&self, mode: Int8Mode) -> Result<QMatMul> {
        let qt = self
            .qtensor()
            .ok_or_else(|| crate::Error::Msg("repack_for_optimization: not a QTensor".into()))?;
        // Expects a COMPACT source weight: re-running on an already-KO-optimized weight would
        // double-process it (and `to_ko`/`repack_gemx` have no KO source). Guard explicitly.
        if qt.dtype().is_ko() {
            crate::bail!(
                "repack_for_optimization: weight is already KO-optimized ({:?}); call on the \
                 compact source weight once",
                qt.dtype()
            );
        }
        let shape = qt.shape().clone();
        let new_storage = match &qt.storage {
            QStorage::Cuda(cs) => {
                if mode.is_int8() {
                    QStorage::Cuda(cs.repack_ko(&shape, qt.dtype().to_ko(mode)?)?)
                } else {
                    QStorage::Cuda(cs.repack_gemx(&shape)?)
                }
            }
            _ => crate::bail!("repack_for_optimization requires CUDA storage"),
        };
        QMatMul::from_qtensor(QTensor {
            storage: new_storage,
            shape,
            lease: PhantomData,
        })
    }

    #[allow(unused_variables)]
    pub fn forward_via_gemx<'w>(&self, xs: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
        match self {
            Self::QTensor(t) => {
                // For CUDA, we need to call the storage directly with compute_type
                match &t.storage {
                    #[cfg(feature = "cuda")]
                    QStorage::Cuda(cuda_storage) => {
                        let storage = xs.storage_and_layout().0;
                        let layout = xs.layout();
                        let cuda_xs = match &*storage {
                            Storage::Cuda(s) => s,
                            _ => crate::bail!("expected CUDA storage for quantized matmul"),
                        };
                        // K/128 blocks have embedded scales, no external scales needed
                        let (out_storage, out_shape) =
                            cuda_storage.fwd_via_gemx(&t.shape, cuda_xs, layout)?;
                        let none = crate::op::BackpropOp::none();
                        Ok(crate::tensor::from_storage(
                            Storage::Cuda(out_storage),
                            out_shape,
                            none,
                            false,
                        ))
                    }
                    #[cfg(not(feature = "cuda"))]
                    QStorage::Cuda(_) => {
                        crate::bail!("CUDA support not compiled")
                    }
                    // For non-CUDA, fall back to standard forward
                    _ => xs.apply_op1_no_bwd(t.as_ref()),
                }
            }
            // For dequantized tensors, use standard matmul. `matmul` records a
            // graph edge and so returns its operand's lifetime; the copy is what
            // makes the result owned. Only the `CANDLE_DEQUANTIZE_ALL` debug
            // path reaches here, so it is not paid in production.
            Self::Tensor(w) => {
                let xs = &xs.to_owned_tensor()?;
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
            Self::TensorF16(w) => {
                let xs = &xs.to_owned_tensor()?;
                let in_dtype = xs.dtype();
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
            }
        }
    }

    /// Matmul over an already-built [`cuda::DynamicTensor`] activation — the keystone the fused
    /// producers feed: an `Int8` operand carries **pre-quantized** q8a128 (emitted directly by a
    /// fused RMSNorm/SwiGLU/attention epilogue, so no standalone quantize launch), a `Float`
    /// operand is a plain FP tensor. Dispatches to [`cuda::dense_qmatmul`], which forks the int8
    /// KO tensor-core path vs the FP path and enforces the KO⇔int8 pairing against this weight.
    /// Returns a result of shape `[lead.., N]` stored at `out_dtype`: the int8 kernel converts its
    /// F32 accumulator on the store, so there is no cast — and no second buffer — afterwards.
    #[cfg(feature = "cuda")]
    pub fn forward_dynamic<'w>(
        &self,
        input: cuda::DynamicTensor<'_, 'w>,
        out_dtype: crate::DType,
    ) -> Result<LiveTensor<'w>> {
        let t = match self {
            Self::QTensor(t) => t,
            _ => crate::bail!("forward_dynamic requires a QTensor weight"),
        };
        let cs = match &t.storage {
            QStorage::Cuda(cs) => cs,
            _ => crate::bail!("forward_dynamic requires CUDA storage"),
        };
        let device = cs.device().clone();
        // Weight is row-major [N, K]; nrows = N is the matmul's output width.
        let nrows = t.shape().dims()[0];
        let wptr = cs.data_ptr();
        let wlen = cs.storage_size_in_bytes();
        let wdtype = t.dtype();
        cuda::dense_qmatmul(input, wptr, wdtype, nrows, wlen, out_dtype, &device)
    }

    /// Fused q/k/v projection in ONE launch: the shared q8a128 activation `op` × the separate KO
    /// weights `q/k/v` (any KO formats — no weight concatenation needed), returning the concatenated
    /// `[lead.., Nq+Nk+Nv]` output cast to `out_dtype`. Float-identical to three separate
    /// [`forward_dynamic`] calls but with full GPU occupancy. Each weight must be a KO QTensor on
    /// CUDA (the int8 twin). See [`cuda::qkv_segmented_matmul`].
    #[cfg(feature = "cuda")]
    pub fn qkv_segmented<'w>(
        op: &cuda::Q8a128Operand<'w>,
        weights: &[&QMatMul],
        out_dtype: crate::DType,
    ) -> Result<LiveTensor<'w>> {
        let mut segs = Vec::with_capacity(weights.len());
        let mut device = None;
        for w in weights {
            let t = match w {
                Self::QTensor(t) => t,
                _ => crate::bail!("qkv_segmented requires KO QTensor weights"),
            };
            let cs = match &t.storage {
                QStorage::Cuda(cs) => cs,
                _ => crate::bail!("qkv_segmented requires CUDA storage"),
            };
            device = Some(cs.device().clone());
            segs.push((cs.data_ptr(), t.dtype(), t.shape().dims()[0]));
        }
        let device = device.ok_or_else(|| crate::Error::Msg("qkv_segmented: no weights".into()))?;
        cuda::qkv_segmented_matmul(op, &segs, out_dtype, &device)
    }

    /// Int8 tensor-core matmul: q8a128 activations × KO weights. The weight must already be the
    /// KO twin QTensor on CUDA (produced by [`QMatMul::repack_for_optimization`] with an int8
    /// `mode`) — the twin choice was baked in at repack time, so here `mode` only selects the
    /// activation form (q8a128 for any non-`Off` mode). Quantizes the activation here (the
    /// **unfused** path, one standalone launch) then runs [`QMatMul::forward_dynamic`]; the fused
    /// producers bypass this by emitting q8a128 themselves and calling `forward_dynamic` directly.
    /// The result comes back in `xs`'s dtype — quantizing the activation is not meant to change the
    /// width the caller sees.
    ///
    /// Non-CUDA build: the q8a128 × KO int8 matmul is a CUDA tensor-core path — always
    /// errors (the `dummy_cuda` convention).
    #[cfg(not(feature = "cuda"))]
    pub fn forward_via_int8<'w>(
        &self,
        _xs: &LiveTensor<'w>,
        _mode: Int8Mode,
    ) -> Result<LiveTensor<'w>> {
        crate::bail!("forward_via_int8 requires the cuda feature")
    }

    #[cfg(feature = "cuda")]
    pub fn forward_via_int8<'w>(
        &self,
        xs: &LiveTensor<'w>,
        mode: Int8Mode,
    ) -> Result<LiveTensor<'w>> {
        let device = match self {
            Self::QTensor(t) => match &t.storage {
                QStorage::Cuda(cs) => cs.device().clone(),
                _ => crate::bail!("forward_via_int8 requires CUDA storage"),
            },
            _ => crate::bail!("forward_via_int8 requires a KO QTensor weight"),
        };
        let out_dtype = xs.dtype();
        let acts = cuda::to_dynamic(xs, mode, &device)?;
        self.forward_dynamic(acts.as_dynamic(), out_dtype)
    }
}

impl crate::CustomOp1 for QTensor {
    fn name(&self) -> &'static str {
        "qmatmul"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        let (n, k) = self.shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().context("empty dst_shape")?;
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self.shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        #[allow(clippy::infallible_destructuring_match)]
        let self_storage = match &self.storage {
            QStorage::Cpu(storage) => storage,
            QStorage::Metal(_) | QStorage::Cuda(_) => crate::bail!("Invalid storage"),
        };
        let slice = storage.as_slice::<f32>()?;
        let slice = &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
        let mut dst_storage = vec![0f32; dst_shape.elem_count()];
        self_storage.matmul_t((dst_shape.elem_count() / n, k, n), slice, &mut dst_storage)?;
        Ok((crate::CpuStorage::F32(dst_storage), dst_shape))
    }

    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Metal(metal) => metal,
            _ => unreachable!("Cannot call metal matmul on non metal QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }

    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Cuda(cuda) => cuda,
            _ => unreachable!("Cannot call cuda matmul on non cuda QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }
}

impl QMatMul {
    /// Project an activation that may live on an inference wave.
    ///
    /// [`crate::Module`] takes `&Tensor`, so it cannot accept a wave-scoped
    /// activation — and that restriction is deliberate everywhere else, because
    /// a module may retain what it is given. This one does not retain, but it
    /// does *inherit*: the output is allocated from whichever arena `xs` came
    /// from, so the result is bounded by `xs`'s generation rather than being
    /// owned. The `Module` impl below is this method at `'static`, where the
    /// distinction collapses and the result really is owned.
    pub fn forward_live<'w>(&self, xs: &LiveTensor<'w>) -> Result<LiveTensor<'w>> {
        match self {
            // MXFP4 has no native matmul kernel (and no native FP4 tensor-core MMA on
            // sm_120), so dequantize the weight (fast GPU kernel / CPU codec) and run a
            // standard matmul. Used by the DeepSeek-V4 routed experts.
            Self::QTensor(t) if t.dtype() == GgmlDType::MXFP4 => {
                let w = t.dequantize(xs.device())?.to_dtype(xs.dtype())?; // [out, in]
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
            Self::QTensor(t) => xs.apply_op1_no_bwd(t.as_ref()),
            Self::Tensor(w) => {
                // The dequantized-weight fallback (`CANDLE_DEQUANTIZE_ALL`),
                // not the production path. `matmul` records a graph edge and so
                // returns the operand's lifetime; copying off the wave first is
                // what makes the result owned, and this path can afford it.
                let xs = &xs.to_owned_tensor()?;
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
            Self::TensorF16(w) => {
                // As the `Tensor` arm: debug-only, so copy off the wave.
                let xs = &xs.to_owned_tensor()?;
                let in_dtype = xs.dtype();
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
            }
        }
    }
}

impl crate::Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_live(xs)
    }
}

#[cfg(test)]
mod ggml_dtype_lock_tests {
    //! These tests pin the exact integer value for every `GgmlDType` variant.
    //! They must stay in lockstep with:
    //!   - The C++ `QType` enum in candle-kernels/src/quantized/block_compact.cuh
    //!     (locked in via `static_assert` in that file)
    //!   - The Rust `QType` enum in candle-kernels/src/quantized/api.rs (which
    //!     has its own matching `qtype_values_are_stable` test)
    //!   - `ArenaFormat::*` in candle-kernels/src/arena_table.cuh
    //!   - `SELECT_FMT_*` in candle-kernels/src/quantize/select_kv_format.cuh
    //!
    //! F32/F16/BF16 (0/1/2) are specific to GgmlDType — they do not appear in
    //! QType, which starts at R16=3.
    use super::GgmlDType;

    #[test]
    fn ggml_dtype_values_are_stable() {
        assert_eq!(GgmlDType::F32 as u32, 0);
        assert_eq!(GgmlDType::F16 as u32, 1);
        assert_eq!(GgmlDType::BF16 as u32, 2);
        assert_eq!(GgmlDType::R16 as u32, 3);
        assert_eq!(GgmlDType::P2 as u32, 4);
        assert_eq!(GgmlDType::QAWQ as u32, 5);
        assert_eq!(GgmlDType::QAWQ_G64 as u32, 6);
        assert_eq!(GgmlDType::Q8_0 as u32, 7);
        assert_eq!(GgmlDType::Q8_1 as u32, 8);
        assert_eq!(GgmlDType::Q8_K as u32, 9);
        assert_eq!(GgmlDType::Q8_KS as u32, 10);
        assert_eq!(GgmlDType::Q6_K as u32, 11);
        assert_eq!(GgmlDType::Q5_0 as u32, 12);
        assert_eq!(GgmlDType::Q5_1 as u32, 13);
        assert_eq!(GgmlDType::Q5_K as u32, 14);
        assert_eq!(GgmlDType::Q4_0 as u32, 15);
        assert_eq!(GgmlDType::Q4_1 as u32, 16);
        assert_eq!(GgmlDType::Q4_K as u32, 17);
        assert_eq!(GgmlDType::Q4_KS as u32, 18);
        assert_eq!(GgmlDType::Q3_0 as u32, 19);
        assert_eq!(GgmlDType::Q3_1 as u32, 20);
        assert_eq!(GgmlDType::Q3_K as u32, 21);
        assert_eq!(GgmlDType::Q2_0 as u32, 22);
        assert_eq!(GgmlDType::Q2_1 as u32, 23);
        assert_eq!(GgmlDType::Q2_K as u32, 24);
        assert_eq!(GgmlDType::Q2_S as u32, 25);
        assert_eq!(GgmlDType::Q2_A as u32, 26);
        assert_eq!(GgmlDType::Q1_S as u32, 27);
        assert_eq!(GgmlDType::Q0_V as u32, 28);
        assert_eq!(GgmlDType::Q1_A as u32, 29);
        assert_eq!(GgmlDType::Q0_X as u32, 30);
        assert_eq!(GgmlDType::Q0_M2 as u32, 31);
        assert_eq!(GgmlDType::Q0_M4 as u32, 32);
        assert_eq!(GgmlDType::Q0 as u32, 33);
        assert_eq!(GgmlDType::F8E4M3 as u32, 34);
        assert_eq!(GgmlDType::F8E5M2 as u32, 35);
        // KO byte-permuted twins — must match QTYPE_Q*_KO / QType::Q*_KO (45-48).
        assert_eq!(GgmlDType::Q4_KO as u32, 45);
        assert_eq!(GgmlDType::Q5_KO as u32, 46);
        assert_eq!(GgmlDType::Q6_KO as u32, 47);
        assert_eq!(GgmlDType::Q8_KO as u32, 48);
    }
}
