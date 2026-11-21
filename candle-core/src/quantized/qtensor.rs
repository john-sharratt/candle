//! The constructors for a quantized tensor, which allocate fresh memory, and the
//! path that quantizes a float tensor into one.
//!
//! The constructors live on the owned type rather than the lifetime-generic one,
//! so that new memory always carries the static lifetime and the lifetime cannot
//! be inferred as something shorter. Quantizing forces a contiguous layout first,
//! because a narrowed view can report itself contiguous while its storage holds
//! more elements than its shape. A host-mapped tensor lets a weight that does not
//! fit in VRAM be read over PCIe by the same kernels, and the guard that the
//! caller keeps must outlive the tensor.

use super::*;

pub(super) fn check_shape(shape: &Shape, block_size: usize) -> Result<()> {
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
