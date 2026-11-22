//! A quantized tensor whose device memory stays valid for a stated lifetime, so
//! that an arena view cannot outlive the arena it points into.
//!
//! The lifetime describes the memory rather than a borrow. An owned tensor has
//! the static case, while a view onto a KV arena slot carries the arena's
//! lifetime. The matmul operator accepts only owned tensors, so a view cannot
//! become a weight. Reads, writes and scatters are the same whichever kind the
//! caller holds. Operations that allocate return an owned tensor, since inheriting
//! the receiver's lifetime would shorten a result that owns its memory.

use super::*;

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
    pub(super) storage: QStorage,
    pub(super) shape: Shape,
    /// Covariant in `'w`, so a `'static` tensor is usable wherever a shorter
    /// one is expected but never the reverse. `&'w [u8]` rather than
    /// `&'w mut [u8]`: writes go through a raw device pointer, so nothing here
    /// needs invariance, and invariance would reject the coercion above.
    pub(super) lease: PhantomData<&'w [u8]>,
}

/// The everyday quantized tensor: it owns its memory, so that memory is live
/// for as long as the process cares to keep it.
pub type QTensor = LiveQTensor<'static>;

impl std::fmt::Debug for LiveQTensor<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QTensor[{:?}; {:?}]", self.shape, self.dtype())
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

    /// Fold this weight's DEQUANTIZED NaN count, Inf count and finite min/max
    /// into `name`'s assert slot, returning the tensor so the call chains.
    ///
    /// The values a quantized weight contributes to a matmul are its
    /// dequantized ones, so that is what this measures — a NaN scale and a
    /// finite-but-enormous weight are different faults, and only the
    /// dequantized view tells them apart.
    ///
    /// Unlike [`LiveTensor::assert`] this stages through the dequant kernels on
    /// the default stream, so it belongs at load time or an epoch boundary
    /// rather than inside a wave. Prefer [`Self::assert_once`] for weights.
    #[cfg(feature = "tensor-assert")]
    pub fn assert(&self, name: &str) -> &Self {
        crate::tensor_assert::assert_qtensor(self, name);
        self
    }

    #[cfg(not(feature = "tensor-assert"))]
    #[inline(always)]
    pub fn assert(&self, _name: &str) -> &Self {
        self
    }

    /// [`Self::assert`], but only the first time this name is seen in the
    /// current epoch — the form weights should use, since a weight does not
    /// change between forwards.
    #[cfg(feature = "tensor-assert")]
    pub fn assert_once(&self, name: &str) -> &Self {
        if crate::tensor_assert::should_run_once(name) {
            crate::tensor_assert::assert_qtensor(self, name);
        }
        self
    }

    #[cfg(not(feature = "tensor-assert"))]
    #[inline(always)]
    pub fn assert_once(&self, _name: &str) -> &Self {
        self
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

    /// Repack the weight into the GEMX layout read by the dequant-weight float
    /// matmul, as a standalone [`QTensor`].
    ///
    /// [`QMatMul::repack_for_optimization`] is the inference entry point and
    /// hands back a `QMatMul`; this one keeps the repacked tensor addressable
    /// so its bytes can be inspected, which is what the per-format layout
    /// tests and the matmul benchmark need.
    #[cfg(feature = "cuda")]
    pub fn repack_gemx(&self) -> Result<QTensor> {
        match &self.storage {
            QStorage::Cuda(s) => Ok(QTensor {
                storage: QStorage::Cuda(s.repack_gemx(&self.shape)?),
                shape: self.shape.clone(),
                lease: PhantomData,
            }),
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

    /// Whether `repack_ko_into` can build this tensor's KO twin directly — see
    /// [`crate::quantized::cuda::repackable_to_ko`]. Wider than
    /// [`Self::supports_gemx_repacking`]: a float source has no GEMX kernel but is widened
    /// band-by-band by the same banded route.
    #[cfg(feature = "cuda")]
    pub fn repackable_to_ko(&self) -> bool {
        crate::quantized::cuda::repackable_to_ko(self.dtype())
    }

    #[cfg(not(feature = "cuda"))]
    pub fn repackable_to_ko(&self) -> bool {
        false
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
    /// is truly asynchronous — the CPU returns immediately.  The layer
    /// stream's pack build reads each repacked layer back to the host this way.
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
    /// dequantize paths now that an arena is a run of untyped byte slots.
    /// Writes through the returned
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

        // **The one failure this function cannot survive quietly.** Everything
        // below is a byte append, which is the stacked weight exactly when a
        // row is a contiguous run of blocks — true of every GGUF block format,
        // where `k` is a multiple of the block size so no block spans two rows.
        //
        // It is false of the KO twins. Those are lane-major GPU chunks whose
        // bytes are already permuted into the layout the int8 tensor-core
        // matmul reads (`QCudaStorage::repack_ko`), so rows are interleaved
        // across lanes and appending two of them yields a weight of exactly the
        // right byte count holding nonsense — which surfaces as bad numbers many
        // layers downstream, never as a fault here. Refused by name rather than
        // guessed at, because the caller's fix is to move the stack earlier
        // (onto the GGUF layout, before `QMatMul::from_qtensor_with_mode`
        // repacks) rather than to give up on it.
        if dtype.is_ko() {
            crate::bail!(
                "concat_rows_cuda: {dtype:?} is a KO twin — its bytes are lane-major, already \
                 repacked for the int8 matmul, so its rows are interleaved and a byte append \
                 would build a correctly-sized weight holding nonsense. Stack on the GGUF block \
                 layout instead, BEFORE the repack"
            )
        }

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
