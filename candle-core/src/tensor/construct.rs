//! Creates tensors from nothing, from device memory the process owns elsewhere,
//! or from a kernel's output.
//!
//! `zeros`, `ones`, `full`, `arange`, `new` and `from_vec` allocate fresh
//! storage, and `empty` skips the initialising fill, so every element must be
//! written before it is read. `from_leased_cuda_ptr` and
//! `from_anchored_cuda_ptr` view memory an arena owns, and `from_cuda_storage`
//! wraps a kernel's output while keeping the wave's lifetime. Device
//! allocations that carry no wave provenance are reported by `note_ticketless`,
//! and `empty_beside` and `from_vec_beside` place an output on the same wave
//! span as an operand.

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
    /// Build a tensor that **views** device memory this process owns elsewhere,
    /// rather than memory allocated from the pool.
    ///
    /// The resulting storage is [`crate::cuda_backend::Backing::Lease`]: dropping
    /// it releases the view and never frees the memory. This is how tensor-shaped
    /// operations reach into a KV arena slot — the arena owns the bytes for the
    /// process lifetime, so a pool free on them would be a correctness error, not
    /// a leak. See `docs/archived/arena_unification.md` §3.7.
    ///
    /// Views and reshapes share the underlying `Arc<Storage>`, so the lease
    /// travels with them.
    ///
    /// # Safety
    /// `ptr` must point to at least `shape.elem_count()` elements of `dtype`,
    /// be correctly aligned, and remain live — and not written through another
    /// alias — for as long as this tensor or any view of it exists.
    #[cfg(feature = "cuda")]
    pub unsafe fn from_leased_cuda_ptr<S: Into<Shape>>(
        ptr: u64,
        dtype: DType,
        shape: S,
        device: &Device,
        origin: crate::cuda_backend::wave_provenance::LeaseOrigin,
    ) -> Result<Self> {
        let Device::Cuda(cuda) = device else {
            crate::bail!("from_leased_cuda_ptr: expected a CUDA device, got {device:?}");
        };
        let shape = shape.into();
        let storage = crate::CudaStorage::from_leased_device_ptr(
            ptr,
            shape.elem_count(),
            dtype,
            cuda,
            origin,
        )?;
        Ok(from_storage(
            Storage::Cuda(storage),
            shape,
            BackpropOp::none(),
            false,
        ))
    }

    /// Re-describe this tensor's bytes as a `'static` tensor that **borrows**
    /// them, copying nothing.
    ///
    /// The alternative at a channel boundary is [`Self::to_owned_tensor`], which
    /// really copies. `'static` here says the result owns no memory, not that
    /// the memory is immortal — so the caller carries the obligation below.
    ///
    /// Requires a contiguous layout: a lease is an address plus a shape, and a
    /// strided view is neither losslessly describable that way nor useful to the
    /// kernels that consume one.
    ///
    /// # Safety
    /// `self` must outlive every use of the returned tensor.
    #[cfg(feature = "cuda")]
    pub unsafe fn as_foreign_lease(&self) -> Result<Tensor> {
        let (storage, layout) = self.storage_and_layout();
        if !layout.is_contiguous() {
            crate::bail!("as_foreign_lease: requires a contiguous layout");
        }
        let Storage::Cuda(cuda) = &*storage else {
            crate::bail!("as_foreign_lease: expected CUDA storage");
        };
        let dtype = self.dtype();
        let base = cuda.slice.device_ptr(&cuda.device().cuda_stream());
        let ptr = base + (layout.start_offset() * dtype.size_in_bytes()) as u64;
        // An anchored lease stays anchored: the re-lease is one more view of the
        // same memory, and dropping the anchor here would let its owner hand the
        // memory on while this view still reads it.
        match &cuda.anchor {
            Some(anchor) => Tensor::from_anchored_cuda_ptr(
                ptr,
                dtype,
                self.shape().clone(),
                self.device(),
                anchor.clone(),
            ),
            None => Tensor::from_leased_cuda_ptr(
                ptr,
                dtype,
                self.shape().clone(),
                self.device(),
                crate::cuda_backend::wave_provenance::LeaseOrigin::Foreign,
            ),
        }
    }

    /// [`Self::from_leased_cuda_ptr`] for memory whose owner can let it go: the
    /// tensor's storage holds `anchor`, and every view, clone and re-lease of it
    /// shares that storage — so the owner cannot hand the memory to anyone else while
    /// any of them exists. See [`LeaseAnchor`].
    ///
    /// # Safety
    /// `ptr` must point to at least `shape.elem_count()` elements of `dtype`, be
    /// correctly aligned, and stay live while `anchor` is held; nothing may write it
    /// through another alias for as long as this tensor or any view of it exists.
    #[cfg(feature = "cuda")]
    pub unsafe fn from_anchored_cuda_ptr<S: Into<Shape>>(
        ptr: u64,
        dtype: DType,
        shape: S,
        device: &Device,
        anchor: LeaseAnchor,
    ) -> Result<Self> {
        let Device::Cuda(cuda) = device else {
            bail!("from_anchored_cuda_ptr: expected a CUDA device, got {device:?}");
        };
        let shape = shape.into();
        let storage =
            CudaStorage::from_anchored_device_ptr(ptr, shape.elem_count(), dtype, cuda, anchor)?;
        Ok(from_storage(
            Storage::Cuda(storage),
            shape,
            BackpropOp::none(),
            false,
        ))
    }

    /// The wave generation this tensor's storage was carved from, if any — what
    /// an op passes along so its output is allocated beside its input.
    ///
    /// `None` off CUDA and for anything pool-backed, which is the right answer
    /// rather than a fallback: there is no arena to inherit.
    pub fn wave_ticket(&self) -> Option<crate::wave_provenance::WaveTicket> {
        #[cfg(feature = "cuda")]
        {
            self.cuda_backing().inherit_ticket()
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    }

    /// The arena this tensor's storage came from, for an op that wants to
    /// allocate its output alongside it.
    ///
    /// [`crate::cuda_backend::Backing::Owned`] for anything not on CUDA, which
    /// is the right answer rather than a fallback: there is no wave to inherit.
    #[cfg(feature = "cuda")]
    pub fn cuda_backing(&self) -> crate::cuda_backend::Backing {
        let (storage, _) = self.storage_and_layout();
        match &*storage {
            Storage::Cuda(c) => c.backing,
            _ => crate::cuda_backend::Backing::Owned,
        }
    }

    /// Wrap device storage a kernel has just written into a tensor.
    ///
    /// The door out of a hand-rolled CUDA kernel that does not go through
    /// [`crate::CustomOp1`]. `CustomOp1::cuda_fwd` returns `(CudaStorage,
    /// Shape)` with no lifetime parameter, so a kernel whose output lives on an
    /// inference wave cannot say so through that trait — the lifetime is erased
    /// at the boundary and the result comes back as an owned `Tensor`. Calling
    /// the kernel directly and wrapping its storage here keeps `'w`.
    ///
    /// No backprop op is recorded: these are inference kernels, and a graph
    /// edge would want a gradient definition none of them have.
    ///
    /// # Safety
    /// `storage` must hold at least `shape.elem_count()` initialised elements
    /// of its dtype, and `'w` must not outlive the memory behind it. When the
    /// storage is [`crate::cuda_backend::Backing::Lease`] that memory belongs to
    /// someone else — a wave's transient half, a KV arena — and picking `'w`
    /// too long is exactly the mistake the parameter exists to prevent. Prefer
    /// a caller whose own signature pins `'w` to a borrow of the owner.
    #[cfg(feature = "cuda")]
    pub unsafe fn from_cuda_storage<S: Into<Shape>>(
        storage: crate::CudaStorage,
        shape: S,
    ) -> LiveTensor<'w> {
        from_storage(Storage::Cuda(storage), shape, BackpropOp::none(), false)
    }

    pub(crate) fn ones_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let none = BackpropOp::none();
        let shape = shape.into();
        let mut storage = unsafe { device.alloc_uninit(&shape, dtype)? };
        let layout = Layout::contiguous(shape.clone());
        storage.const_set(crate::scalar::Scalar::one(dtype), &layout)?;
        Ok(from_storage(storage, shape, none, is_variable))
    }

    /// Report a device allocation that **cannot** carry wave provenance.
    ///
    /// `zeros`, `ones`, `full`, `arange`, `new` and `from_vec` build a tensor
    /// from nothing — there is no operand to inherit a ticket from, so the
    /// result is always `Backing::Owned`. Outside a wave that is exactly right
    /// (weights, masks, tables). Inside one it is a root provenance break: the
    /// tensor comes from the driver rather than the reservation, and every
    /// tensor derived from it afterwards inherits `Owned` and does the same.
    ///
    /// `#[track_caller]` means the location reported is the *caller's* — the
    /// model line that built the tensor — with no stack walk and no symbol
    /// resolution, and correct under inlining. The detector is armed only around
    /// a wave, so this is one relaxed load everywhere else.
    /// **Device allocations only**, which is the module's stated scope: "Host
    /// allocations are out of scope." Without the check a CPU tensor is
    /// reported exactly like a pool allocation, and the report is read as a
    /// work list — the warm tier's 16 MiB host slab
    /// (`kv_cache::chunked::alloc::claim_slab`, the `ArenaLocation::Cpu` arm)
    /// showed up as the single largest "forbidden allocation" in a live
    /// inventory while touching no VRAM at all.
    #[track_caller]
    pub(super) fn note_ticketless(what: &'static str, bytes: usize, device: &Device) {
        if !device.is_cuda() {
            return;
        }
        crate::forbidden_alloc::record_at(std::panic::Location::caller(), what, bytes);
    }

    /// Creates a new tensor filled with ones.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::ones((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0], (2, 3), &Device::Cpu)?;
    /// // a == b
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    #[track_caller]
    pub fn ones<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        let shape = shape.into();
        Self::note_ticketless(
            "Tensor::ones",
            shape.elem_count() * dtype.size_in_bytes(),
            device,
        );
        Self::ones_impl(shape, dtype, device, false)
    }

    pub fn const_set(&self, value: crate::scalar::Scalar) -> Result<()> {
        self.storage_mut().const_set(value, self.layout())
    }

    pub fn zero_set(&self) -> Result<()> {
        self.const_set(crate::scalar::Scalar::zero(self.dtype()))
    }

    pub fn one_set(&self) -> Result<()> {
        self.const_set(crate::scalar::Scalar::one(self.dtype()))
    }

    /// Creates a new tensor filled with ones with same shape, dtype, and device as the other tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = a.ones_like()?;
    /// // b == a + 1
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    #[track_caller]
    pub fn ones_like(&self) -> Result<Self> {
        Tensor::ones(self.shape(), self.dtype(), self.device())
    }

    // Do not expose outside of the crate, the `is_variable=true` case should only be accessed from
    // the variable module.
    pub(crate) fn zeros_impl<S: Into<Shape>>(
        shape: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let none = BackpropOp::none();
        let shape = shape.into();
        let storage = device.zeros(&shape, dtype)?;
        Ok(from_storage(storage, shape, none, is_variable))
    }

    /// Creates a new tensor filled with zeros.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = Tensor::from_slice(&[0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0], (2, 3), &Device::Cpu)?;
    /// // a == b
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    #[track_caller]
    pub fn zeros<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        let shape = shape.into();
        Self::note_ticketless(
            "Tensor::zeros",
            shape.elem_count() * dtype.size_in_bytes(),
            device,
        );
        Self::zeros_impl(shape, dtype, device, false)
    }

    /// Creates a new tensor filled with zeros with same shape, dtype, and device as the other
    /// tensor.
    ///
    /// ```rust
    /// use candle_core::{Tensor, DType, Device};
    /// let a = Tensor::zeros((2, 3), DType::F32, &Device::Cpu)?;
    /// let b = a.zeros_like()?;
    /// // b is on CPU f32.
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    #[track_caller]
    pub fn zeros_like(&self) -> Result<Self> {
        Tensor::zeros(self.shape(), self.dtype(), self.device())
    }

    /// Allocates a tensor WITHOUT initialising its storage — the bytes are
    /// whatever the allocator returned. This skips the full-width `memset` that
    /// [`Tensor::zeros`] pays, so it is the correct constructor for a buffer that
    /// a kernel (or other op) fully overwrites before anything reads it.
    ///
    /// SAFETY / CONTRACT: every element MUST be written before it is read.
    /// Reading an unwritten element yields an arbitrary value (and, for the
    /// float dtypes, possibly a NaN/Inf bit pattern). Use [`Tensor::zeros`] for
    /// any buffer whose zero value is read before being written — accumulators
    /// (`atomicAdd`/`atomicMax` targets), scatter bases, and ragged padding.
    pub fn empty<S: Into<Shape>>(shape: S, dtype: DType, device: &Device) -> Result<Self> {
        let shape = shape.into();
        // Uninitialised storage: the caller's write is the initialisation.
        let storage = unsafe { device.alloc_uninit(&shape, dtype)? };
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    /// [`Self::empty`], in whichever arena `self` came from.
    ///
    /// The operand-provenance rule spelled out for a **kernel wrapper**. The
    /// generic ops get this for free — they allocate through
    /// `alloc_uninit_from` and inherit their input's arena without knowing what
    /// an arena is — but a hand-written launcher names its own output buffer,
    /// and reaching for [`Self::empty`] there silently drops the chain onto the
    /// pool. Allocating beside an operand keeps the chain intact, and `'w` comes
    /// from that operand, so the result cannot outlive the span it may point
    /// into.
    ///
    /// Carries the same write-before-read contract as [`Self::empty`].
    #[track_caller]
    pub fn empty_beside<S: Into<Shape>>(&self, shape: S, dtype: DType) -> Result<Self> {
        let shape = shape.into();
        let ticket = self.wave_ticket();
        // **"Beside" is a request, not a guarantee.** The output lands on the
        // wave span only if the operand is itself leased from it; an operand
        // that is `Owned` silently yields an owned output, and every tensor
        // derived from it afterwards is owned too. That is how one broken
        // provenance root becomes dozens of "sites" in a report.
        //
        // Named here, with the caller's own file and line, because that is the
        // only place the answer is cheap and exact.
        if ticket.is_none() {
            crate::forbidden_alloc::record_at(
                std::panic::Location::caller(),
                "empty_beside on an operand with no wave ticket",
                shape.elem_count() * dtype.size_in_bytes(),
            );
        }
        // SAFETY: as `empty` — the caller's kernel writes every element before
        // anything reads it.
        let storage = unsafe { self.device().alloc_uninit_from(&shape, dtype, ticket)? };
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    pub(crate) fn rand_impl<S: Into<Shape>, T: crate::FloatDType>(
        lo: T,
        up: T,
        s: S,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let s = s.into();
        let storage = device.rand_uniform(lo, up, &s)?;
        let none = BackpropOp::none();
        Ok(from_storage(storage, s, none, is_variable))
    }

    pub(crate) fn rand_f64_impl<S: Into<Shape>>(
        lo: f64,
        up: f64,
        s: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let s = s.into();
        let storage = device.rand_uniform_f64(lo, up, &s, dtype)?;
        let none = BackpropOp::none();
        Ok(from_storage(storage, s, none, is_variable))
    }

    /// Creates a new tensor initialized with values sampled uniformly between `lo` and `up`.
    pub fn rand<S: Into<Shape>, T: crate::FloatDType>(
        lo: T,
        up: T,
        s: S,
        device: &Device,
    ) -> Result<Self> {
        Self::rand_impl(lo, up, s, device, false)
    }

    pub fn rand_like(&self, lo: f64, up: f64) -> Result<Self> {
        Tensor::rand_f64_impl(lo, up, self.shape(), self.dtype(), self.device(), false)
    }

    pub(crate) fn randn_impl<S: Into<Shape>, T: crate::FloatDType>(
        mean: T,
        std: T,
        s: S,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let s = s.into();
        let storage = device.rand_normal(mean, std, &s)?;
        let none = BackpropOp::none();
        Ok(from_storage(storage, s, none, is_variable))
    }

    pub(crate) fn randn_f64_impl<S: Into<Shape>>(
        mean: f64,
        std: f64,
        s: S,
        dtype: DType,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let s = s.into();
        let storage = device.rand_normal_f64(mean, std, &s, dtype)?;
        let none = BackpropOp::none();
        Ok(from_storage(storage, s, none, is_variable))
    }

    pub fn randn_like(&self, mean: f64, stdev: f64) -> Result<Self> {
        Tensor::randn_f64_impl(
            mean,
            stdev,
            self.shape(),
            self.dtype(),
            self.device(),
            false,
        )
    }

    /// Creates a new tensor initialized with values sampled from a normal distribution with the
    /// specified `mean` and standard deviation `std`.
    pub fn randn<S: Into<Shape>, T: crate::FloatDType>(
        mean: T,
        std: T,
        s: S,
        device: &Device,
    ) -> Result<Self> {
        Self::randn_impl(mean, std, s, device, false)
    }

    /// `#[track_caller]` so the site reported from inside is the model's, not
    /// `Tensor::new`'s — the attribute only propagates through functions that
    /// carry it.
    #[track_caller]
    pub(crate) fn new_impl<A: crate::device::NdArray>(
        array: A,
        shape: Shape,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let n: usize = shape.elem_count();
        let buffer_size: usize = array.shape()?.elem_count();
        if buffer_size != n {
            return Err(Error::ShapeMismatch { buffer_size, shape }.bt());
        }
        let storage = device.storage(array)?;
        // Reported here rather than in `new`: `NdArray` carries no dtype, so the
        // element size is not knowable until the storage exists. Reporting the
        // element *count* as bytes instead was 4x under for F32 — in the very
        // report the rest of this work is justified by.
        Self::note_ticketless("Tensor::new", n * storage.dtype().size_in_bytes(), device);
        let none = BackpropOp::none();
        Ok(from_storage(storage, shape, none, is_variable))
    }

    /// Creates a new tensor on the specified device using the content and shape of the input.
    #[track_caller]
    pub fn new<A: crate::device::NdArray>(array: A, device: &Device) -> Result<Self> {
        let shape = array.shape()?;
        Self::new_impl(array, shape, device, false)
    }

    /// Returns a new tensor with all the elements having the same specified value.
    ///```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::full(3.5, (2, 4), &Device::Cpu)?;
    ///
    /// assert_eq!(a.to_vec2::<f64>()?, &[
    ///     [3.5, 3.5, 3.5, 3.5],
    ///     [3.5, 3.5, 3.5, 3.5],
    /// ]);
    /// # Ok::<(), candle_core::Error>(())
    #[track_caller]
    pub fn full<D: crate::WithDType, S: Into<Shape>>(
        value: D,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let none = BackpropOp::none();
        let shape = shape.into();
        Self::note_ticketless(
            "Tensor::full",
            shape.elem_count() * D::DTYPE.size_in_bytes(),
            device,
        );
        let mut storage = unsafe { device.alloc_uninit(&shape, D::DTYPE)? };
        let layout = Layout::contiguous(shape.clone());
        storage.const_set(value.to_scalar(), &layout)?;
        Ok(from_storage(storage, shape, none, false))
    }

    /// Creates a new 1D tensor from an iterator.
    ///```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::from_iter( [1.0, 2.0, 3.0, 4.0].into_iter(), &Device::Cpu)?;
    ///
    /// assert_eq!(a.to_vec1::<f64>()?, &[1.0, 2.0, 3.0, 4.0]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn from_iter<D: crate::WithDType>(
        iter: impl IntoIterator<Item = D>,
        device: &Device,
    ) -> Result<Self> {
        let data = iter.into_iter().collect::<Vec<_>>();
        let len = data.len();
        Self::from_vec_impl(data, len, device, false)
    }

    /// Creates a new 1D tensor with values from the interval `[start, end)` taken with a common
    /// difference `1` from `start`.
    ///```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::arange(2., 5., &Device::Cpu)?;
    ///
    /// assert_eq!(a.to_vec1::<f64>()?, &[2., 3., 4.]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    #[track_caller]
    pub fn arange<D: crate::WithDType>(start: D, end: D, device: &Device) -> Result<Self> {
        Self::arange_step(start, end, D::one(), device)
    }

    /// Creates a new 1D tensor with values from the interval `[start, end)` taken with a common
    /// difference `step` from `start`.
    ///```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::arange_step(2.0, 4.0, 0.5, &Device::Cpu)?;
    ///
    /// assert_eq!(a.to_vec1::<f64>()?, &[2.0, 2.5, 3.0, 3.5]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    #[track_caller]
    pub fn arange_step<D: crate::WithDType>(
        start: D,
        end: D,
        step: D,
        device: &Device,
    ) -> Result<Self> {
        if D::is_zero(&step) {
            bail!("step cannot be zero")
        }
        // Reported below, where the length is known: an arange's element count
        // falls out of building it, and both of its paths (the native integer
        // generator and the host loop) end at a `len` this can be taken from.
        // Reporting a literal `0` here instead made the tally's arange rows
        // unreadable.
        // Integer aranges generate ON DEVICE where the backend supports it: these are
        // hot-path gather indices, and building them on the host costs a tiny H2D
        // upload per call (a measured WDDM submission storm). `start + i*step` is
        // exact integer arithmetic — identical values to the accumulation loop below.
        // Float aranges always take the host path: its repeated-addition rounding is
        // the documented semantics, which the closed form would not reproduce.
        let int_bits = |v: D| -> Option<u64> {
            match v.to_scalar() {
                crate::scalar::Scalar::U8(x) => Some(x as u64),
                crate::scalar::Scalar::U32(x) => Some(x as u64),
                crate::scalar::Scalar::I64(x) => Some(x as u64),
                _ => None,
            }
        };
        if let (Some(start_bits), Some(step_bits)) = (int_bits(start), int_bits(step)) {
            // Count without materializing — the same loop semantics as the host build.
            let mut len = 0usize;
            let mut current = start;
            if step >= D::zero() {
                while current < end {
                    len += 1;
                    current += step;
                }
            } else {
                while current > end {
                    len += 1;
                    current += step;
                }
            }
            if let Some(storage) = device.arange_int_native(D::DTYPE, start_bits, step_bits, len)? {
                Self::note_ticketless("Tensor::arange", len * D::DTYPE.size_in_bytes(), device);
                return Ok(from_storage(storage, len, BackpropOp::none(), false));
            }
        }
        let mut data = vec![];
        let mut current = start;
        if step >= D::zero() {
            while current < end {
                data.push(current);
                current += step;
            }
        } else {
            while current > end {
                data.push(current);
                current += step;
            }
        }
        let len = data.len();
        Self::note_ticketless("Tensor::arange", len * D::DTYPE.size_in_bytes(), device);
        Self::from_vec_impl(data, len, device, false)
    }

    pub(crate) fn from_vec_impl<S: ShapeWithOneHole, D: crate::WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
        is_variable: bool,
    ) -> Result<Self> {
        let shape = shape.into_shape(data.len())?;
        let storage = device.storage_owned(data)?;
        let none = BackpropOp::none();
        Ok(from_storage(storage, shape, none, is_variable))
    }

    /// Creates a new tensor initialized with values from the input vector. The number of elements
    /// in this vector must be the same as the number of elements defined by the shape.
    /// If the device is cpu, no data copy is made.
    ///```rust
    /// use candle_core::{Tensor, Device};
    /// let a = Tensor::from_vec(vec!{1., 2., 3., 4., 5., 6.}, (2, 3), &Device::Cpu)?;
    ///
    /// assert_eq!(a.to_vec2::<f64>()?, &[
    ///     [1., 2., 3.],
    ///     [4., 5., 6.]
    /// ]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    #[track_caller]
    pub fn from_vec<S: ShapeWithOneHole, D: crate::WithDType>(
        data: Vec<D>,
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        Self::note_ticketless(
            "Tensor::from_vec",
            data.len() * D::DTYPE.size_in_bytes(),
            device,
        );
        Self::from_vec_impl(data, shape, device, false)
    }

    /// [`Self::from_vec`], placed on the same wave span as `self`.
    ///
    /// The host-upload counterpart to [`Self::empty_beside`]. A table built on
    /// the host — a pointer array, a row map, a rotary layout — has no device
    /// operand to inherit provenance from, so `from_vec` can only ever produce
    /// an `Owned` tensor: a driver allocation, inside the wave, from the memory
    /// the reservation deliberately does not cover. Naming an operand that is
    /// already on the span is what puts the upload there instead.
    ///
    /// `self` need only identify the span; its shape and dtype are irrelevant.
    /// If `self` is itself owned this is exactly [`Self::from_vec`] — the report
    /// then names *this* call site, which is the honest answer, because the
    /// provenance was already lost upstream.
    #[cfg(feature = "cuda")]
    #[track_caller]
    pub fn from_vec_beside<S: ShapeWithOneHole, D: crate::WithDType>(
        &self,
        data: Vec<D>,
        shape: S,
    ) -> Result<Self> {
        let ticket = self.wave_ticket();
        if ticket.is_none() {
            Self::note_ticketless(
                "from_vec_beside on an operand with no wave ticket",
                data.len() * D::DTYPE.size_in_bytes(),
                self.device(),
            );
        }
        let shape = shape.into_shape(data.len())?;
        let storage = self.device().storage_owned_on(data, ticket)?;
        Ok(from_storage(storage, shape, BackpropOp::none(), false))
    }

    /// Creates a new tensor initialized with values from the input slice. The number of elements
    /// in this vector must be the same as the number of elements defined by the shape.
    ///```rust
    /// use candle_core::{Tensor, Device};
    /// let values = vec![1., 2., 3., 4., 5., 6., 7., 8.];
    /// let a = Tensor::from_slice(&values[1..7], (2, 3), &Device::Cpu)?;
    ///
    /// assert_eq!(a.to_vec2::<f64>()?, &[
    ///     [2., 3., 4.],
    ///     [5., 6., 7.]
    /// ]);
    /// # Ok::<(), candle_core::Error>(())
    /// ```
    pub fn from_slice<S: ShapeWithOneHole, D: crate::WithDType>(
        array: &[D],
        shape: S,
        device: &Device,
    ) -> Result<Self> {
        let shape = shape.into_shape(array.len())?;
        let storage = device.storage_from_slice(array)?;
        let none = BackpropOp::none();
        Ok(from_storage(storage, shape, none, false))
    }
}
