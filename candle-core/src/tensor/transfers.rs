//! Moves a tensor between devices or dtypes, and takes copies that own their
//! memory.
//!
//! `to_owned_tensor` copies the view's own contents onto a fresh allocation, so
//! the result borrows nothing and wave-scoped data can outlive its owner. It
//! copies the view and not the parent storage. `copy` keeps the source layout,
//! and `detach` drops the graph edge while sharing storage. `to_dtype` casts,
//! and `to_dtype_mut` casts in place on CUDA when the buffer is large enough and
//! not shared. `ensure_dtype` converts only when the current dtype is not in the
//! list it is given.

use super::*;

/// The tensor API, generic over how long the storage is valid.
///
/// Unlike [`crate::quantized::LiveQTensor`], this block is *not* split into
/// `'static` constructors and `'w` accessors, because here the `-> Result<Self>`
/// methods are overwhelmingly **views** — `reshape`, `narrow`, `t`,
/// `broadcast_as` — and a view of a lease has to stay a lease. Returning `Self`
/// is what makes that true by construction. The handful of methods that
/// allocate leave `'w` free, which is sound because owned memory outlives every
/// choice of `'w`; the discipline that matters is applied at the *lease*
/// constructors instead, where `'w` is tied to a borrow of whatever owns the
/// bytes.
impl<'w> LiveTensor<'w> {
    /// A `'static` deep copy of this tensor's contents, compacted.
    ///
    /// The sanctioned way to let wave- or arena-scoped data outlive its owner:
    /// the view is copied onto a fresh owned allocation, so the result borrows
    /// nothing. It really does allocate — that is the point, and it is why this
    /// is not something [`Clone`] could do (it must return `Self`, and so would
    /// inherit `'w` while quietly paying for a copy).
    ///
    /// **This tensor's contents, not its storage's.** The result is a contiguous
    /// buffer of exactly `shape`, so owning one row of a wide block costs one
    /// row. The obvious spelling — clone the storage and keep the layout, which
    /// is what [`Storage::try_clone`] does and what this used to do — copies the
    /// WHOLE parent for every view taken of it, and the call sites here are
    /// overwhelmingly `x.narrow(..)?.to_owned_tensor()?`: one scored row off a
    /// batched head, one sequence's span off a packed wave, the last KV row of a
    /// verify block. Owning `n` rows of an `n`-row block that way is `n²` bytes
    /// and `n` allocations of the full block. Measured on the 128-slot decode
    /// ladder, the per-row copy of the `[128, vocab]` logits block was **881 ms
    /// of a 1,152 ms forward** — 76% of the step, and ~10 GB of VRAM held for
    /// 78 MB of logits.
    ///
    /// Use [`Self::copy`] where the source layout itself must survive the copy;
    /// this one is for getting the values out.
    #[track_caller]
    pub fn to_owned_tensor(&self) -> Result<Tensor> {
        let shape = self.shape();
        // An owned allocation, deliberately NOT the source's arena — inheriting
        // a wave ticket here would put the copy back on the span it is being
        // taken off. It stands on the held ground (`Device::alloc_owned`).
        let mut storage = unsafe { self.device().alloc_owned(shape, self.dtype())? };
        self.storage()
            .copy_strided_src(&mut storage, 0, self.layout())?;
        Ok(from_storage(
            storage,
            shape.clone(),
            BackpropOp::none(),
            false,
        ))
    }

    pub fn copy(&self) -> Result<Self> {
        let op = BackpropOp::new1(self, Op::Copy);
        let tensor_ = TensorInner {
            lease: PhantomData,
            id: TensorId::new(),
            storage: Arc::new(RwLock::new(self.storage().try_clone(self.layout())?)),
            layout: self.layout.clone(),
            op,
            is_variable: false,
            dtype: self.dtype,
            device: self.device.clone(),
        };
        Ok(LiveTensor(Arc::new(tensor_)))
    }

    /// Returns a new tensor detached from the current graph, gradient are not propagated through
    /// this new node. The storage of this tensor is shared with the initial tensor.
    ///
    /// If the tensor is already detached from the computation graph, the same tensor is returned.
    pub fn detach(&self) -> Self {
        if self.op.is_none() && !self.is_variable {
            self.clone()
        } else {
            let tensor_ = TensorInner {
                lease: PhantomData,
                id: TensorId::new(),
                storage: self.storage.clone(),
                layout: self.layout.clone(),
                op: BackpropOp::none(),
                is_variable: false,
                dtype: self.dtype,
                device: self.device.clone(),
            };
            LiveTensor(Arc::new(tensor_))
        }
    }

    /// If the target device is the same as the tensor device, only a shallow copy is performed.
    pub fn to_device(&self, device: &Device) -> Result<Self> {
        if self.device().same_device(device) {
            Ok(self.clone())
        } else {
            let storage = match (&*self.storage(), device) {
                (Storage::Cpu(storage), Device::Cuda(cuda)) => {
                    Storage::Cuda(cuda.storage_from_cpu_storage(storage)?)
                }
                (Storage::Cpu(storage), Device::Metal(metal)) => {
                    Storage::Metal(metal.storage_from_cpu_storage(storage)?)
                }
                (Storage::Cuda(storage), Device::Cpu) => Storage::Cpu(storage.to_cpu_storage()?),
                (Storage::Metal(storage), Device::Cpu) => Storage::Cpu(storage.to_cpu_storage()?),
                (Storage::Cuda(storage), Device::Cuda(cuda)) => {
                    // CUDA→CUDA copy currently round-trips through CPU storage even when the
                    // source and destination GPU ids match.
                    let cpu_storage = storage.to_cpu_storage()?;
                    Storage::Cuda(cuda.storage_from_cpu_storage(&cpu_storage)?)
                }
                (Storage::Cpu(storage), Device::Cpu) => Storage::Cpu(storage.clone()),
                _ => {
                    bail!(
                        "not implemented yet, self.device: {:?}, device: {:?}",
                        self.device(),
                        device
                    )
                }
            };
            let op = BackpropOp::new1(self, Op::ToDevice);
            let tensor_ = TensorInner {
                lease: PhantomData,
                id: TensorId::new(),
                storage: Arc::new(RwLock::new(storage)),
                layout: self.layout.clone(),
                op,
                is_variable: false,
                dtype: self.dtype,
                device: device.clone(),
            };
            Ok(LiveTensor(Arc::new(tensor_)))
        }
    }

    /// Casts the input tensor to the target `dtype`.
    ///
    /// ```rust
    /// use candle_core::{Tensor, Device};
    /// let tensor = Tensor::new(3.14159265358979f64, &Device::Cpu)?;
    /// assert_eq!(tensor.to_scalar::<f64>()?, 3.14159265358979);
    /// let tensor = tensor.to_dtype(candle_core::DType::F32)?;
    /// assert_eq!(tensor.to_scalar::<f32>()?, 3.1415927);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        if self.dtype() == dtype {
            Ok(self.clone())
        } else {
            let shape = self.shape();
            let storage = self.storage().to_dtype(self.layout(), dtype)?;
            let op = BackpropOp::new1(self, Op::ToDType);
            Ok(from_storage(storage, shape.clone(), op, false))
        }
    }

    /// Casts the tensor to the target `dtype` in-place when possible.
    ///
    /// On CUDA, this attempts to perform the conversion in-place if the underlying buffer
    /// is large enough to hold the destination type. If the buffer is too small, or on
    /// non-CUDA backends, it falls back to a regular `to_dtype` and replaces the tensor.
    ///
    /// This is useful for scenarios where you want to minimize memory allocations and
    /// the tensor's buffer was allocated with extra capacity.
    ///
    /// # Example
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let mut tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &Device::Cpu)?;
    /// tensor.to_dtype_mut(DType::F64)?;
    /// assert_eq!(tensor.dtype(), DType::F64);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn to_dtype_mut(&mut self, dtype: DType) -> Result<()> {
        if self.dtype() == dtype {
            return Ok(());
        }

        // Make contiguous first if needed - in-place cast requires contiguous layout
        if !self.is_contiguous() {
            *self = self.contiguous()?;
        }

        // Try in-place conversion first (only works on CUDA with sufficient buffer).
        // Skip it when the storage is shared (e.g. this tensor is a `narrow` view that
        // aliases a sibling view of the same buffer, as the fused gate/up MLP produces):
        // an in-place cast mutates the shared buffer and corrupts the other views.
        let converted_in_place = if Arc::strong_count(&self.storage) > 1 {
            false
        } else {
            let mut storage = self.storage_mut();
            storage.to_dtype_mut(self.layout(), dtype)?
        };

        if !converted_in_place {
            // Fall back to regular to_dtype and replace self (e.g., CPU/Metal or buffer too small)
            *self = self.to_dtype(dtype)?;
        } else {
            // In-place conversion succeeded, but we need to update the tensor's dtype field.
            // Since Tensor is Arc<Tensor_> and dtype is immutable, we need to create a new tensor
            // that shares the same (now converted) storage.
            let shape = self.shape().clone();
            let storage = self.storage.clone(); // Clone the Arc<RwLock<Storage>>
            let op = BackpropOp::new1(self, Op::ToDType);
            let tensor_ = TensorInner {
                lease: PhantomData,
                id: TensorId::new(),
                storage,
                layout: Layout::contiguous(&shape),
                op,
                is_variable: false,
                dtype,
                device: self.device.clone(),
            };
            *self = LiveTensor(Arc::new(tensor_));
        }

        Ok(())
    }

    /// Ensures the tensor is one of the specified `dtype`s. If the tensor's current dtype is
    /// already in the provided list, returns a clone without conversion. Otherwise, converts
    /// to the first dtype in the list.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let tensor = Tensor::new(3.14f32, &Device::Cpu)?;
    /// // Already F32, no conversion needed
    /// let result = tensor.ensure_dtype(&[DType::F32, DType::F64])?;
    /// assert_eq!(result.dtype(), DType::F32);
    ///
    /// // F32 not in list, converts to first dtype (F64)
    /// let result = tensor.ensure_dtype(&[DType::F64, DType::BF16])?;
    /// assert_eq!(result.dtype(), DType::F64);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn ensure_dtype(&self, dtypes: &[DType]) -> Result<Self> {
        if dtypes.is_empty() {
            crate::bail!("ensure_dtype requires at least one dtype")
        }
        if dtypes.contains(&self.dtype()) {
            Ok(self.clone())
        } else {
            self.to_dtype(dtypes[0])
        }
    }

    /// Create a variable based on the values currently stored in a tensor. The storage is always
    /// copied.
    pub(crate) fn make_var(&self) -> Result<Self> {
        let shape = self.shape().clone();
        let mut storage = unsafe {
            self.device()
                .alloc_uninit_from(&shape, self.dtype(), self.wave_ticket())?
        };
        self.storage()
            .copy_strided_src(&mut storage, 0, self.layout())?;
        Ok(from_storage(storage, shape, BackpropOp::none(), true))
    }
}
