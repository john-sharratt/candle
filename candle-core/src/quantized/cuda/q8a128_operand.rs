//! Defines the q8a128 activation operand: its backing, its shape, its sum
//! convention, and how its bytes are borrowed or copied.
//!
//! An operand's bytes are either an owned buffer or a tensor's storage, and it
//! carries its leading dimensions and per-128 sum convention because only the
//! bytes say how to read them. Its lifetime is bounded by the generation its bytes
//! were carved from, so it cannot outlive a wave that is reset. Borrowing the
//! bytes costs nothing and owns nothing, which lets an operand cross a channel.
//! Copying off the arena is the only way out of a wave for an operand the borrow
//! checker cannot follow, and `dequantize` returns the activation as the matmul
//! sees it, with the int8 rounding included.

use super::*;

/// The minimum token count M at which the q8a1024 int8 matmul flips from mode-1 to
/// mode-2. Measured crossover: below this, the per-token weight re-read of mode-2's
/// Bm=32 tile costs more than it saves; at/above it, weight reuse wins (the prefill
/// regime). M is the absolute number of token rows fed to the quantizer.
/// q8a128-quantized int8 operand for the tensor-core qmatmul: the packed blocks (stored in
/// the q8a1024 flat-grouped super-block layout — see blocks.cuh), the logical shape
/// `[rows × cols]` = `[M × K]`, and `lead` — the original activation's leading dims
/// (`prod(lead) == rows`) so the matmul can rebuild the output rank (`[lead.., N]`) to match the
/// float path on 3D+ activations. The mode-1/mode-2 flip is NOT stored here: the q8a1024 layout
/// is mode-independent, so the dispatcher chooses the kernel tiling from the token count. Named
/// "operand" (not "acts") because either matmul side could be supplied this way.
/// Backing bytes for a [`Q8a128Operand`]: either an owned `CudaSlice` (the quantizer / fused
/// RMSNorm / SwiGLU producers `alloc` their own buffer) or a `U8` [`Tensor`] of q8a1024 bytes.
/// The latter lets a producer that returns through `apply_op1` (e.g. the B2 decode op, whose
/// output is a tensor) feed the int8 matmul WITHOUT a device copy — the operand just borrows the
/// tensor's bytes. Both expose a device pointer via [`Q8a128Operand::with_device_ptr`].
pub enum Q8a128Data<'w> {
    Owned(CudaSlice<u8>),
    Tensor(crate::LiveTensor<'w>),
}

/// Where a kernel's output goes, and how the storage wrapping it must be marked.
///
/// Returned as a triple — `(ptr, owned, backing)` — rather than as separate
/// decisions, because the pointer and the backing have to agree. A wave range
/// marked `Owned` is a double free; a pool buffer marked `Lease` is a permanent
/// leak. Resolving both in one place is what stops them drifting.
///
/// # Why the wave arm is a bare pointer
///
/// The pool arm owns a `CudaSlice`; the wave arm deliberately does not
/// materialise one. A `CudaSlice` frees on drop, and between resolving an output
/// and wrapping it the kernel launch can fail — every `?` in that window would
/// otherwise put a `cuMemFreeAsync` on arena memory. Staying a raw `u64` until
/// the caller wraps it removes the window entirely.
pub(super) type ResolvedOut<T> = (u64, Option<CudaSlice<T>>, Backing);

/// Resolve a `u8` output from `origin`'s arena, or from the pool.
///
/// `origin` is the operand's backing: a wave-backed operand yields a wave-backed
/// output (the inheritance rule), anything else allocates normally. A phase's
/// *first* buffer has no arena to inherit — its operand is the residual stream,
/// which must stay pooled because it crosses layers — so those call sites pass a
/// synthesised `Backing::Lease(Wave(ticket))` to seed the chain instead.
pub(super) fn resolve_u8_out(
    origin: Backing,
    dev: &CudaDevice,
    bytes: usize,
) -> Result<ResolvedOut<u8>> {
    let ticket = origin.inherit_ticket();
    // The third of the three sites making this decision, attributed like the
    // other two. A `NoTicket` decline here is always a genuine provenance break:
    // the phase-first buffers described above do not reach it, because they seed
    // a synthesised ticket rather than arriving without one.
    if let Some((ptr, ticket)) = wave_alloc_attributed(ticket, bytes, INHERIT_ALIGN)? {
        return Ok((ptr, None, Backing::Lease(LeaseOrigin::Wave(ticket))));
    }
    // SAFETY: the kernel the caller launches next fills this before anything
    // reads it, exactly as when it allocated inline.
    let slice = unsafe { dev.alloc::<u8>(bytes)? };
    let ptr = {
        let stream = dev.cuda_stream();
        let (ptr, _guard) = slice.device_ptr(&stream);
        ptr
    };
    Ok((ptr, Some(slice), Backing::Owned))
}

/// [`resolve_u8_out`] for a typed output.
///
/// Separate because the byte count is `elems * size_of::<T>()`, and getting that
/// wrong is the one mistake this whole mechanism exists to make impossible.
pub(super) fn resolve_typed_out<T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits>(
    origin: Backing,
    dev: &CudaDevice,
    elems: usize,
) -> Result<ResolvedOut<T>> {
    let ticket = origin.inherit_ticket();
    let bytes = elems * std::mem::size_of::<T>();
    // Attributed for the same reason as `alloc_inheriting` — this is the other
    // half of the same decision, and a report that cannot tell the two decline
    // reasons apart sends the reader to the wrong fix.
    if let Some((ptr, ticket)) = wave_alloc_attributed(ticket, bytes, INHERIT_ALIGN)? {
        return Ok((ptr, None, Backing::Lease(LeaseOrigin::Wave(ticket))));
    }
    // SAFETY: filled by the kernel the caller launches next.
    let slice = unsafe { dev.alloc::<T>(elems)? };
    let ptr = {
        let stream = dev.cuda_stream();
        let (ptr, _guard) = slice.device_ptr(&stream);
        ptr
    };
    Ok((ptr, Some(slice), Backing::Owned))
}

/// A resolved matmul destination whose element type was picked at runtime.
///
/// The pool arm owns a typed `CudaSlice`, so the type has to be carried from the
/// resolve to the wrap; the wave arm holds `None` and rides on the raw pointer.
/// One enum for all three matmul entries — they make the same decision and would
/// otherwise each grow their own copy of it.
pub(super) enum OwnedOut {
    F16(Option<CudaSlice<f16>>),
    BF16(Option<CudaSlice<bf16>>),
    F32(Option<CudaSlice<f32>>),
}

/// [`resolve_typed_out`] with the element type chosen from a runtime [`DType`].
pub(super) fn resolve_out(
    dtype: crate::DType,
    origin: Backing,
    dev: &CudaDevice,
    elems: usize,
) -> Result<(u64, OwnedOut, Backing)> {
    match dtype {
        crate::DType::F16 => {
            let (p, o, b) = resolve_typed_out::<f16>(origin, dev, elems)?;
            Ok((p, OwnedOut::F16(o), b))
        }
        crate::DType::BF16 => {
            let (p, o, b) = resolve_typed_out::<bf16>(origin, dev, elems)?;
            Ok((p, OwnedOut::BF16(o), b))
        }
        crate::DType::F32 => {
            let (p, o, b) = resolve_typed_out::<f32>(origin, dev, elems)?;
            Ok((p, OwnedOut::F32(o), b))
        }
        other => crate::bail!("quantized matmul: unsupported output dtype {other:?}"),
    }
}

/// [`tensor_from_out`] for a destination resolved by [`resolve_out`].
pub(super) fn tensor_from_owned_out<'w>(
    owned: OwnedOut,
    ptr: u64,
    backing: Backing,
    shape: crate::Shape,
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    match owned {
        OwnedOut::F16(o) => tensor_from_out::<f16>(o, ptr, backing, shape, device),
        OwnedOut::BF16(o) => tensor_from_out::<bf16>(o, ptr, backing, shape, device),
        OwnedOut::F32(o) => tensor_from_out::<f32>(o, ptr, backing, shape, device),
    }
}

/// The dispatcher's store-width code for an output dtype.
pub(super) fn out_dtype_code(dtype: crate::DType) -> Result<i32> {
    let code = match dtype {
        crate::DType::F16 => OutDType::F16,
        crate::DType::BF16 => OutDType::BF16,
        crate::DType::F32 => OutDType::F32,
        other => crate::bail!("quantized matmul: unsupported output dtype {other:?}"),
    };
    Ok(code as i32)
}

/// Turn a launcher status code into an error.
///
/// The launchers pick a kernel from a static table; a miss leaves the destination
/// holding whatever the arena last put there. Checking the code is what makes that
/// a failure instead of wrong numbers.
///
/// A failed launch says which CUDA call failed, with what error, and the
/// thread's recent launches: a launch inside a recorded wave can fail on its own
/// refusal or report an earlier one, and the bare status told neither apart.
pub(super) fn check_matmul_status(code: i32, op: &str) -> Result<()> {
    let status = MatmulStatus::from_code(code);
    match status.failure() {
        None => Ok(()),
        Some(reason) if status == MatmulStatus::LaunchFailed => {
            // SAFETY: both return static strings the launcher's own thread wrote.
            let (error, site) = unsafe {
                (
                    CStr::from_ptr(qmm_last_launch_error()).to_string_lossy(),
                    CStr::from_ptr(qmm_last_launch_site()).to_string_lossy(),
                )
            };
            crate::bail!(
                "{op}: kernel not launched ({reason}, status {code}): {site} returned {error}; \
                 {}",
                last_cuda_kernel_launch()
            )
        }
        Some(reason) => crate::bail!("{op}: kernel not launched ({reason}, status {code})"),
    }
}

/// Build the q8a128 operand for whichever destination the kernel just wrote.
///
/// Kept next to [`resolve_u8_out`] because the two are halves of one decision:
/// resolve picks where the bytes go, this wraps them without the caller having
/// to re-derive which arm was taken. `sum_scale` is the convention the kernel
/// just wrote the blocks under — a kernel that copies blocks (the MoE gather)
/// passes its source's.
#[allow(clippy::too_many_arguments)]
pub(super) fn q8a128_from_out<'w>(
    owned: Option<CudaSlice<u8>>,
    ptr: u64,
    backing: Backing,
    bytes: usize,
    rows: usize,
    cols: usize,
    sum_scale: SumScale,
    device: &CudaDevice,
) -> Result<Q8a128Operand<'w>> {
    match owned {
        Some(slice) => Ok(Q8a128Operand::new(slice, rows, cols, sum_scale)),
        None => {
            let dev = crate::Device::Cuda(device.clone());
            let origin = match backing {
                Backing::Lease(o) => o,
                Backing::Owned => LeaseOrigin::Foreign,
            };
            // SAFETY: the kernel above filled `bytes` at `ptr`, and `'w` is the
            // generation that range was carved from, so the operand cannot be
            // named after the span is reset.
            let backing_tensor = unsafe {
                crate::LiveTensor::from_leased_cuda_ptr(ptr, crate::DType::U8, bytes, &dev, origin)?
            };
            Ok(Q8a128Operand::from_tensor(
                backing_tensor,
                rows,
                cols,
                sum_scale,
            ))
        }
    }
}

/// Wrap whichever destination a kernel wrote into as a tensor bounded by `'w`.
///
/// The leased arm is why these kernels carry a lifetime: an inherited output
/// lives in a span the next layer resets, so returning `Tensor` (`'static`) over
/// it would be a lie the borrow checker could not catch.
pub(super) fn tensor_from_out<
    'w,
    T: crate::WithDType + crate::cuda_backend::CudaDType + cudarc::driver::DeviceRepr,
>(
    owned: Option<CudaSlice<T>>,
    ptr: u64,
    backing: Backing,
    shape: crate::Shape,
    device: &CudaDevice,
) -> Result<crate::LiveTensor<'w>> {
    match owned {
        Some(slice) => {
            let storage = CudaStorage::wrap_cuda_slice(slice, device.clone());
            // SAFETY: the kernel wrote `shape.elem_count()` elements of `T`.
            Ok(unsafe { crate::LiveTensor::from_cuda_storage(storage, shape) })
        }
        None => {
            let dev = crate::Device::Cuda(device.clone());
            let origin = match backing {
                Backing::Lease(o) => o,
                Backing::Owned => LeaseOrigin::Foreign,
            };
            // SAFETY: as above, and `'w` bounds the carved range.
            unsafe { crate::LiveTensor::from_leased_cuda_ptr(ptr, T::DTYPE, shape, &dev, origin) }
        }
    }
}

/// `'w` is the backing tensor's: the B2 decode context is written into an
/// inference wave's transient half, so an operand wrapping it must not outlive
/// that wave. The owned variant allocates and is free to pick any `'w`.
pub struct Q8a128Operand<'w> {
    pub data: Q8a128Data<'w>,
    pub rows: usize,      // M (flattened token count)
    pub cols: usize,      // K
    pub lead: Vec<usize>, // output leading dims; prod == rows (defaults to [rows])
    /// How this operand's blocks store their per-128 `Σx`.
    ///
    /// **Unlike the matmul mode, this IS an attribute of the bytes.** The mode
    /// only changes how a kernel tiles the same bytes, so the dispatcher derives
    /// it; the sum convention changes what the bytes *mean*, so it travels with
    /// them and the matmul never has to guess. Whoever quantized the activation
    /// chose it — see [`SumScale`]. Every constructor takes it and none
    /// defaults it: a kernel that copies blocks and labels its result with the
    /// wrong one produces wrong numbers, not an error.
    pub sum_scale: SumScale,
}

impl<'w> Q8a128Operand<'w> {
    /// Copy off whatever arena this operand lives in, yielding one that owns its
    /// bytes and so may be `'static`.
    ///
    /// The sanctioned escape from a wave, and the only one: it really copies
    /// rather than laundering the lifetime. Needed where an operand crosses a
    /// boundary the borrow checker cannot follow — the expert-pipeline thread
    /// receives its work over a channel, so a `MoeWorkRequest` cannot carry a
    /// lifetime and must own what it holds.
    pub fn to_owned(&self) -> Result<Q8a128Operand<'static>> {
        let data = match &self.data {
            Q8a128Data::Owned(slice) => Q8a128Data::Owned(slice.try_clone().w()?),
            Q8a128Data::Tensor(t) => Q8a128Data::Tensor(t.to_owned_tensor()?),
        };
        Ok(Q8a128Operand {
            data,
            rows: self.rows,
            cols: self.cols,
            lead: self.lead.clone(),
            sum_scale: self.sum_scale,
        })
    }

    /// The arena this operand's bytes came from, for a kernel that wants to
    /// allocate its output alongside them.
    ///
    /// The `Owned` arm holds a pool `CudaSlice` and so reports
    /// [`Backing::Owned`]; the `Tensor` arm forwards whatever its storage
    /// carries, which is how a wave-backed activation propagates through a
    /// whole expert chain without any call site naming a generation.
    pub fn backing(&self) -> Backing {
        match &self.data {
            Q8a128Data::Owned(_) => Backing::Owned,
            Q8a128Data::Tensor(t) => t.cuda_backing(),
        }
    }

    /// Wrap owned q8a128 blocks of flattened logical shape `[rows × cols]`. `lead` defaults to
    /// `[rows]` (a 2D `[M, N]` output); use [`Self::with_lead`] to preserve higher activation
    /// ranks. The matmul mode (mode-1 vs mode-2 weight-reuse) is NOT stored here — it is a
    /// kernel/tiling property the dispatcher derives from the token count, not an attribute of
    /// the (mode-independent) activation bytes. `sum_scale` is how the blocks
    /// store their per-128 `Σx` — the producer's to name, since a mismatch is read
    /// by the matmul as a wrong number rather than refused.
    pub fn new(data: CudaSlice<u8>, rows: usize, cols: usize, sum_scale: SumScale) -> Self {
        Self {
            data: Q8a128Data::Owned(data),
            rows,
            cols,
            lead: vec![rows],
            sum_scale,
        }
    }

    /// Wrap a contiguous `U8` [`Tensor`] of q8a1024 bytes as an operand, no copy. For producers
    /// that emit through `apply_op1` (the B2 decode context), whose result is a tensor, with
    /// the `Σx` convention that producer wrote.
    pub fn from_tensor(
        data: crate::LiveTensor<'w>,
        rows: usize,
        cols: usize,
        sum_scale: SumScale,
    ) -> Self {
        Self {
            data: Q8a128Data::Tensor(data),
            rows,
            cols,
            lead: vec![rows],
            sum_scale,
        }
    }

    /// Set the output leading dims (the activation's dims minus K); `prod(lead)` must equal
    /// `rows`. Lets the int8 matmul reproduce the activation's rank, e.g. `[B, M, K]→[B, M, N]`.
    pub fn with_lead(mut self, lead: Vec<usize>) -> Self {
        debug_assert_eq!(
            lead.iter().product::<usize>(),
            self.rows,
            "lead must multiply to rows"
        );
        self.lead = lead;
        self
    }

    /// Borrow the owned backing slice; bails on the tensor-backed variant. For callers/tests that
    /// need the raw `CudaSlice` directly (e.g. `dequantize_q8a128`).
    pub fn data_slice(&self) -> Result<&CudaSlice<u8>> {
        match &self.data {
            Q8a128Data::Owned(s) => Ok(s),
            Q8a128Data::Tensor(_) => {
                crate::bail!("Q8a128Operand::data_slice: operand is tensor-backed, not owned")
            }
        }
    }

    /// Consume into the owned backing slice; bails on the tensor-backed variant.
    pub fn into_owned_data(self) -> Result<CudaSlice<u8>> {
        match self.data {
            Q8a128Data::Owned(s) => Ok(s),
            Q8a128Data::Tensor(_) => {
                crate::bail!("Q8a128Operand::into_owned_data: operand is tensor-backed, not owned")
            }
        }
    }

    /// Re-describe these bytes as a `'static` operand that **borrows** them.
    ///
    /// The counterpart to [`Self::to_owned`], and the one to prefer: `to_owned`
    /// copies the whole activation off the arena, which on the MoE path is a
    /// per-layer device copy of the entire ln2 output. This copies nothing — it
    /// wraps the same address as a [`LeaseOrigin::Foreign`] lease, so dropping
    /// the result frees nothing and the producer stays the owner.
    ///
    /// `'static` here is a statement about *ownership*, not lifetime: the result
    /// owns no memory, so there is nothing for a lifetime to bound. That is what
    /// lets it cross a boundary a borrow cannot — a channel, a stored request —
    /// and it is also exactly why the caller carries the obligation below.
    ///
    /// # Safety
    /// `self` must outlive every use of the returned operand: the caller keeps
    /// the frame holding the activation live until every launch that reads the
    /// operand has been issued, on the stream the activation was produced on.
    pub unsafe fn as_foreign_lease(&self, device: &CudaDevice) -> Result<Q8a128Operand<'static>> {
        let bytes = self.byte_len();
        let dev = crate::Device::Cuda(device.clone());
        let leased = self.with_device_ptr(device, |ptr| {
            LiveTensor::from_leased_cuda_ptr(
                ptr,
                crate::DType::U8,
                bytes,
                &dev,
                LeaseOrigin::Foreign,
            )
        })?;
        Ok(Q8a128Operand {
            data: Q8a128Data::Tensor(leased),
            rows: self.rows,
            cols: self.cols,
            lead: self.lead.clone(),
            sum_scale: self.sum_scale,
        })
    }

    /// Reconstruct the float activation these blocks encode, at `dtype`.
    ///
    /// **The int8 round trip is the point, not a defect.** What comes back is
    /// the activation as the *matmul* sees it — the int8 codes times their
    /// per-128 scale — not the float that was quantized. A consumer that needs
    /// to compute against the same operand the base projection multiplies wants
    /// exactly that; one that needs the original float cannot get it here, and
    /// must be handed the tensor before it was quantized.
    ///
    /// One kernel and one buffer at the operand's own backing, so the result
    /// lands in the arena its source came from. This is a full-tensor pass
    /// (hot-path invariant 1), so it belongs on a path that runs once per
    /// projection group rather than once per matmul.
    pub fn dequantize(
        &self,
        dtype: crate::DType,
        device: &CudaDevice,
    ) -> Result<crate::LiveTensor<'w>> {
        let dtype_code = match dtype {
            crate::DType::F16 => 0,
            crate::DType::BF16 => 1,
            crate::DType::F32 => 2,
            d => crate::bail!("Q8a128Operand::dequantize: unsupported dtype {d:?}"),
        };
        let elems = self.rows * self.cols;
        let (out_ptr, owned, out_backing) = resolve_out(dtype, self.backing(), device, elems)?;
        self.with_device_ptr(device, |in_ptr| {
            // SAFETY: `in_ptr` names this operand's q8a1024 blocks for
            // `rows × cols` elements, and `out_ptr` names exactly that many
            // elements of `dtype`.
            unsafe {
                run_dequantize_q8a128(
                    in_ptr as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    self.rows as i32,
                    self.cols as i32,
                    dtype_code,
                    device.cuda_stream().cu_stream() as *mut std::ffi::c_void,
                );
            }
            Ok(())
        })?;
        // The activation's own rank, as `lead` recorded it — so an adapter reads
        // the shape the base projection's operand had, not a flattened one.
        let mut dims = self.lead.clone();
        dims.push(self.cols);
        tensor_from_owned_out(owned, out_ptr, out_backing, dims.into(), device)
    }

    /// Packed size of the q8a1024 blocks backing this operand.
    ///
    /// Delegates to [`q8a1024_byte_len`] rather than recomputing: this is the
    /// size `as_foreign_lease` stamps on the lease it hands out, and a figure that disagrees with what the quantizer actually allocated is
    /// a lease that lies about its own extent.
    pub fn byte_len(&self) -> usize {
        q8a1024_byte_len(self.rows, self.cols)
    }

    /// Run `f` with the raw device pointer to the q8a128 bytes, keeping the backing's
    /// storage/stream guards alive for the duration so the pointer stays valid. Unifies the
    /// owned-slice and tensor-backed variants for the kernel dispatch.
    pub fn with_device_ptr<R>(
        &self,
        device: &CudaDevice,
        f: impl FnOnce(u64) -> Result<R>,
    ) -> Result<R> {
        let stream = device.cuda_stream();
        match &self.data {
            Q8a128Data::Owned(s) => {
                let (ptr, _g) = s.device_ptr(&stream);
                f(ptr)
            }
            Q8a128Data::Tensor(t) => {
                let (storage, layout) = t.storage_and_layout();
                let cuda = match &*storage {
                    crate::Storage::Cuda(c) => c,
                    _ => crate::bail!("Q8a128Operand: tensor backing must be a CUDA tensor"),
                };
                let slice = cuda.as_cuda_slice::<u8>()?.slice(layout.start_offset()..);
                let (ptr, _g) = slice.device_ptr(&stream);
                f(ptr)
            }
        }
    }
}

/// A qmatmul activation/LHS that is either a plain float tensor (F16/BF16/F32 — the
/// dequant-weight float path) or pre-quantized q8a1024 int8 blocks (the int8 tensor-core
/// path). Threading one type through the matmul hides whether it runs in float or int8
/// mode: the int8 arm runs the q8a128 tensor-core path (its tiling chosen by the grid-fill
/// formula `q8a128_dense_tile` at dispatch), the float arm derives it from the tensor's dtype.
/// Two lifetimes, deliberately distinct: `'a` is how long this borrow of the
/// operand lasts, `'w` is the generation the operand's memory belongs to.
///
/// Collapsing them (`Float(&'a LiveTensor<'a>)`) forces the result of a matmul
/// to be bounded by the borrow of a *local variable* rather than by the arena
/// the bytes came from — so a pool-backed activation held in a local could not
/// produce a result that outlives the local, even though the result is owned.
/// Keeping them apart lets `'w` stay `'static` for pool operands and contract
/// to the guard only for wave-backed ones, which is the whole point.
pub enum DynamicTensor<'a, 'w> {
    Float(&'a crate::LiveTensor<'w>),
    Int8(&'a Q8a128Operand<'w>),
}

impl DynamicTensor<'_, '_> {
    /// The arena this activation came from, so a matmul over it can allocate its
    /// output alongside. Forwarding this at every call site is the whole of the
    /// inheritance rule on the consumer side.
    pub fn backing(&self) -> Backing {
        match self {
            Self::Float(t) => t.cuda_backing(),
            Self::Int8(op) => op.backing(),
        }
    }
}

/// Owned activation operand produced by [`to_dynamic`]: a float tensor ([`Int8Mode::Off`]) or
/// pre-quantized q8a128 blocks (any int8 mode). `DynamicTensor` borrows, so this owns the
/// chosen representation and hands out a borrow via [`DynamicActs::as_dynamic`] for the matmul.
pub enum DynamicActs<'w> {
    Float(crate::LiveTensor<'w>),
    Int8(Q8a128Operand<'w>),
}

impl<'w> DynamicActs<'w> {
    /// Borrow as the matmul-facing [`DynamicTensor`].
    pub fn as_dynamic(&self) -> DynamicTensor<'_, 'w> {
        match self {
            DynamicActs::Float(t) => DynamicTensor::Float(t),
            DynamicActs::Int8(op) => DynamicTensor::Int8(op),
        }
    }
}
