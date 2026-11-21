//! Defines the CUDA storage type and the backing that says who owns its device
//! memory.
//!
//! `CudaStorage` pairs a typed device slice with a `Backing`. A pool allocation
//! is freed on drop, a lease releases only a view of memory owned elsewhere, and
//! an arena range is returned to its arena. Allocations that may inherit an
//! operand's arena go through `alloc_inheriting`, which fails when that open
//! phase is full instead of falling back silently, and the storage is stamped
//! with the backing it actually received. A leased storage is disposed of with
//! `leak_view`, because dropping it would ask the driver to free memory it does
//! not own. The arena and the expert weight zone share one reservation, so an
//! arena's frontier must stay below the weight floor.

use super::*;

impl CudaStorageSlice {
    /// Diverge on the tombstone variant.
    ///
    /// [`CudaStorageSlice::Moved`] exists only between the `mem::replace` and
    /// the end of a leased [`CudaStorage`]'s `drop`, and nothing else can
    /// observe one — so every other match over the slice ends here. Returning
    /// `!` lets one arm serve matches of every result type.
    #[cold]
    #[inline(never)]
    pub fn unreachable_moved() -> ! {
        unreachable!("CudaStorageSlice::Moved escaped a leased storage's drop")
    }

    /// Size of the backing allocation in bytes.
    ///
    /// The element count times the element width, so it is comparable across
    /// dtypes — which is what makes it the right unit for allocation
    /// accounting ([`crate::forbidden_alloc`]) and for budget arithmetic.
    pub fn byte_len(&self) -> usize {
        match self {
            CudaStorageSlice::U8(s) => s.len() * std::mem::size_of::<u8>(),
            CudaStorageSlice::U32(s) => s.len() * std::mem::size_of::<u32>(),
            CudaStorageSlice::I64(s) => s.len() * std::mem::size_of::<i64>(),
            CudaStorageSlice::BF16(s) => s.len() * std::mem::size_of::<bf16>(),
            CudaStorageSlice::F16(s) => s.len() * std::mem::size_of::<f16>(),
            CudaStorageSlice::F32(s) => s.len() * std::mem::size_of::<f32>(),
            CudaStorageSlice::F64(s) => s.len() * std::mem::size_of::<f64>(),
            CudaStorageSlice::F8E4M3(s) => s.len() * std::mem::size_of::<F8E4M3>(),
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        }
    }

    /// Device address of the first element, whatever the element type.
    ///
    /// The read-only twin of [`Self::device_ptr_mut`], and it carries the same
    /// caveat: no guard is returned, so the caller keeps the storage alive for
    /// as long as the address is used.
    pub fn device_ptr(&self, stream: &Arc<CudaStream>) -> u64 {
        use cudarc::driver::DevicePtr;
        macro_rules! addr {
            ($s:expr) => {{
                let (ptr, _guard) = $s.device_ptr(stream);
                ptr
            }};
        }
        match self {
            CudaStorageSlice::U8(s) => addr!(s),
            CudaStorageSlice::U32(s) => addr!(s),
            CudaStorageSlice::I64(s) => addr!(s),
            CudaStorageSlice::BF16(s) => addr!(s),
            CudaStorageSlice::F16(s) => addr!(s),
            CudaStorageSlice::F32(s) => addr!(s),
            CudaStorageSlice::F64(s) => addr!(s),
            CudaStorageSlice::F8E4M3(s) => addr!(s),
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        }
    }

    /// Get a mutable device pointer for in-place operations.
    /// Returns the raw pointer that can be passed to FFI functions.
    ///
    /// NOTE: This does NOT return a guard - the caller must ensure proper synchronization.
    pub fn device_ptr_mut(&mut self, stream: &Arc<CudaStream>) -> Result<*mut std::ffi::c_void> {
        use cudarc::driver::DevicePtrMut;
        match self {
            CudaStorageSlice::U8(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::U32(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::I64(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::BF16(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::F16(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::F32(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::F64(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::F8E4M3(s) => {
                let (ptr, _guard) = s.device_ptr_mut(stream);
                Ok(ptr as *mut std::ffi::c_void)
            }
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        }
    }
}

impl CudaStorageSlice {
    /// Release the handle without freeing the memory behind it.
    ///
    /// Calling `leak` rather than merely suppressing the drop is load-bearing:
    /// it waits on the slice's read/write events, destroys them, and decrements
    /// the stream's `Arc`. Bare suppression (`mem::forget`) would strand two
    /// `CudaEvent`s and a stream refcount **per lease** — thousands per second
    /// on the decode path.
    ///
    /// This is the only correct way to dispose of a slice whose memory is owned
    /// elsewhere (an arena slot, a wave range). Dropping one normally calls
    /// `cuMemFreeAsync` on an interior pointer, which the driver rejects into a
    /// silently-recorded error rather than a panic — so the mistake does not
    /// announce itself.
    ///
    /// Answers the view's device address, which the lease's arena may want back
    /// ([`wave_provenance::wave_release`]).
    pub(super) fn leak_view(self) -> u64 {
        match self {
            Self::U8(s) => s.leak(),
            Self::U32(s) => s.leak(),
            Self::I64(s) => s.leak(),
            Self::BF16(s) => s.leak(),
            Self::F16(s) => s.leak(),
            Self::F32(s) => s.leak(),
            Self::F64(s) => s.leak(),
            Self::F8E4M3(s) => s.leak(),
            Self::Moved => Self::unreachable_moved(),
        }
    }
}

impl CudaStorage {
    /// Wrap `len` elements of device memory at `ptr` as a leased storage.
    ///
    /// The memory must outlive every tensor derived from this storage, and must
    /// already be valid for `T` — arena slots satisfy both: they live in a
    /// reservation held for the process lifetime, and zero-on-recycle
    /// (invariant 4) means a slot's bytes are always a legal bit pattern.
    ///
    /// # Safety
    /// `ptr` must point to at least `len` elements of `dtype`, be aligned for
    /// it, and stay live and un-aliased-for-writes for the storage's lifetime.
    pub unsafe fn from_leased_device_ptr(
        ptr: u64,
        len: usize,
        dtype: DType,
        device: &CudaDevice,
        origin: LeaseOrigin,
    ) -> Result<Self> {
        // Wrapping an address issues nothing on the stream: no lease.
        let stream = device.stream_unleased();
        let slice = match dtype {
            DType::U8 => CudaStorageSlice::U8(stream.upgrade_device_ptr::<u8>(ptr, len)),
            DType::U32 => CudaStorageSlice::U32(stream.upgrade_device_ptr::<u32>(ptr, len)),
            DType::I64 => CudaStorageSlice::I64(stream.upgrade_device_ptr::<i64>(ptr, len)),
            DType::BF16 => CudaStorageSlice::BF16(stream.upgrade_device_ptr::<bf16>(ptr, len)),
            DType::F16 => CudaStorageSlice::F16(stream.upgrade_device_ptr::<f16>(ptr, len)),
            DType::F32 => CudaStorageSlice::F32(stream.upgrade_device_ptr::<f32>(ptr, len)),
            DType::F64 => CudaStorageSlice::F64(stream.upgrade_device_ptr::<f64>(ptr, len)),
            DType::F8E4M3 => {
                CudaStorageSlice::F8E4M3(stream.upgrade_device_ptr::<F8E4M3>(ptr, len))
            }
        };
        Ok(Self {
            slice,
            device: device.clone(),
            backing: Backing::Lease(origin),
            anchor: None,
        })
    }

    /// [`Self::from_leased_device_ptr`] for memory whose owner can let it go: the
    /// storage — and every view, clone and re-lease of it — holds `anchor`, so the
    /// owner cannot hand the memory to anyone else while any of them exists.
    ///
    /// # Safety
    /// As [`Self::from_leased_device_ptr`], except that the memory's lifetime is the
    /// anchor's to guarantee: `anchor` must keep the range live while it is held.
    pub unsafe fn from_anchored_device_ptr(
        ptr: u64,
        len: usize,
        dtype: DType,
        device: &CudaDevice,
        anchor: LeaseAnchor,
    ) -> Result<Self> {
        let mut storage =
            Self::from_leased_device_ptr(ptr, len, dtype, device, LeaseOrigin::Foreign)?;
        storage.anchor = Some(anchor);
        Ok(storage)
    }
}

/// Alignment for an inherited output. Matches what `cudaMalloc` guarantees, so a
/// wave-backed output is as aligned as the pool allocation it replaces for every
/// vectorised access a kernel makes.
pub const INHERIT_ALIGN: usize = 256;

/// Allocate an op's output from the arena its **operand** came from — or the
/// pool when the operand has no open phase to inherit — and fail when that
/// phase is open and full ([`wave_provenance::exhausted`]).
///
/// This is the whole operand-provenance rule in one function. The returned
/// [`Backing`] is what the caller must stamp on the output storage: pairing the
/// slice with its backing here — rather than letting the call site name a
/// `Backing` literal next to an allocation it made separately — is what stops
/// the two drifting apart, which is the mistake that turns a wave range into a
/// double free.
///
/// # Safety
///
/// The returned slice is uninitialised. The caller must have the kernel write it
/// before anything reads it, exactly as for a bare `dev.alloc`.
pub unsafe fn alloc_inheriting<T: DeviceRepr>(
    dev: &CudaDevice,
    elem_count: usize,
    from: Backing,
) -> Result<(CudaSlice<T>, Backing)> {
    let ticket = from.inherit_ticket();
    let bytes = elem_count * std::mem::size_of::<T>();
    // Attributed, so the fall-through below is not silent: a site that lost its
    // provenance and a site whose generation closed both end up on `dev.alloc`
    // and are otherwise indistinguishable in any report. An open phase that
    // cannot hold the request is an error here, not a pool allocation.
    if let Some((ptr, ticket)) =
        wave_provenance::wave_alloc_attributed(ticket, bytes, INHERIT_ALIGN)?
    {
        // **The arena and the weight zone share one reservation.**
        //
        // Wave arenas are carved upward from `span_base`; the expert weight zone
        // is carved downward from `span_end`; `weight_floor` is the boundary
        // between them. Nothing in this path knows where that boundary is — it
        // is handed a `ptr` and a length by the arena allocator and wraps them.
        // So an arena whose frontier crosses the floor hands out a range that
        // IS expert weights, and the activation written through it silently
        // replaces them: contiguous, however many bytes the caller asked for,
        // and finite-looking, which is why it surfaces as a NaN several layers
        // downstream rather than as a fault.
        //
        // This is the one allocation path the pool never sees, so the check on
        // `CudaDevice::alloc` cannot cover it.
        crate::readonly_regions::forbid_write("wave arena lease (alloc_inheriting)", ptr, bytes);
        // Dropping this slice bare would ask the driver to free an address
        // inside the span's allocation — one it never handed out — so the
        // driver rejects it and nothing is freed. That is what makes the window
        // between here and the caller stamping `Backing::Lease` harmless.
        let slice = dev
            .stream_unleased()
            .upgrade_device_ptr::<T>(ptr, elem_count);
        return Ok((slice, Backing::Lease(LeaseOrigin::Wave(ticket))));
    }
    Ok((dev.alloc::<T>(elem_count)?, Backing::Owned))
}

impl CudaStorage {
    pub fn wrap_cuda_slice<T: CudaDType>(slice: CudaSlice<T>, device: CudaDevice) -> CudaStorage {
        T::wrap_cuda_slice(slice, device)
    }

    /// Wrap a slice produced by [`alloc_inheriting`], carrying the backing it
    /// resolved to.
    ///
    /// [`Self::wrap_cuda_slice`] stamps `Backing::Owned`, which is a claim of
    /// ownership: dropping such a storage frees the memory. Applying it to a
    /// slice that turned out to be a view over a wave range would hand the
    /// arena's memory to the pool's free path, so an allocation site that can
    /// inherit must carry the resolved backing through to the wrap.
    pub fn wrap_cuda_slice_backed<T: CudaDType>(
        slice: CudaSlice<T>,
        device: CudaDevice,
        backing: Backing,
    ) -> CudaStorage {
        let mut storage = T::wrap_cuda_slice(slice, device);
        storage.backing = backing;
        storage
    }

    /// Wrap a borrowed device range as storage of element type `T`.
    ///
    /// Kernel wrappers use this for outputs written into a transient span:
    /// the span owns the memory and its generation frees it, so the storage
    /// must not.
    ///
    /// # Safety
    /// As [`CudaDType::wrap_leased_ptr`].
    pub unsafe fn wrap_leased_ptr<T: CudaDType>(
        ptr: u64,
        len: usize,
        device: CudaDevice,
        origin: LeaseOrigin,
    ) -> CudaStorage {
        T::wrap_leased_ptr(ptr, len, device, origin)
    }

    pub fn as_cuda_slice<T: CudaDType>(&self) -> Result<&CudaSlice<T>> {
        T::as_cuda_slice(self)
    }

    pub fn as_cuda_slice_mut<T: CudaDType>(&mut self) -> Result<&mut CudaSlice<T>> {
        T::as_cuda_slice_mut(self)
    }

    /// Returns the total byte length of the underlying buffer.
    pub(super) fn buffer_byte_len(&self) -> usize {
        match &self.slice {
            CudaStorageSlice::U8(s) => s.len() * std::mem::size_of::<u8>(),
            CudaStorageSlice::U32(s) => s.len() * std::mem::size_of::<u32>(),
            CudaStorageSlice::I64(s) => s.len() * std::mem::size_of::<i64>(),
            CudaStorageSlice::BF16(s) => s.len() * std::mem::size_of::<bf16>(),
            CudaStorageSlice::F16(s) => s.len() * std::mem::size_of::<f16>(),
            CudaStorageSlice::F32(s) => s.len() * std::mem::size_of::<f32>(),
            CudaStorageSlice::F64(s) => s.len() * std::mem::size_of::<f64>(),
            CudaStorageSlice::F8E4M3(s) => s.len() * std::mem::size_of::<F8E4M3>(),
            CudaStorageSlice::Moved => CudaStorageSlice::unreachable_moved(),
        }
    }
}
