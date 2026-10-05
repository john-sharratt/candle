//! Wave-scoped buffers for the inference loop.
//!
//! An attention output is the archetypal wave intermediate: written by one
//! kernel, consumed by `o_proj`, dead immediately after. Inside the inference
//! loop it comes from the wave's transient half — one cursor bump, no allocator
//! traffic, and the whole half reclaimed when the guard drops
//! (`docs/archived/arena_unification.md` §3.6). On the decode path that replaces an
//! alloc/free pair per layer per forward.
//!
//! The guard bounding these is **layer-scoped**, spanning attention ->
//! `o_proj`. That is deliberate and was learned the hard way: a guard held for
//! the whole forward keeps every layer's output live at once, so consumption
//! grows with depth instead of staying at one layer's working set. Halves
//! alternate per layer, which puts a full layer of same-stream work between one
//! layer's reads and the next reuse of that half.
//!
//! # The guard is the lifetime
//!
//! Every buffer here is allocated *through* a [`WaveGeneration`], so the
//! resulting tensor is a `LiveTensor<'w>` borrowing that guard rather than a
//! `Tensor` claiming `'static`. A wave buffer therefore cannot be named after
//! the guard that frees it has dropped: the compiler rejects the program
//! instead of the kernel reading recycled bytes. Ordering the drops by hand,
//! which is what this replaced, was correct only for as long as everyone
//! remembered to.
//!
//! Outside the inference loop — kernel tests, replay harnesses, the `decode_ab`
//! and `prefill_ab` fixtures — there is no wave, and the caller passes `None`.
//! The absence is real state, not a mode: with no guard there is nothing to
//! bound a lease, so the buffer is allocated and owned in the ordinary way, and
//! `'w` is free because owned memory outlives every choice of it.
//!
//! # Scope
//!
//! *Our* kernels take preallocated leased buffers, because each has an
//! allocation site to redirect. **Interior op outputs — the temporaries candle's
//! own ops allocate — land here too**, but by a different route: they inherit
//! their arena from their operand
//! (`candle::cuda_backend::wave_provenance`), so a chain of forty ops needs no
//! call-site changes at all. What it needs is a *seed*, because the head of a
//! chain reads the residual stream, which crosses layers and lives on the pool
//! with no arena to inherit. [`wave_root`] is that seed, and the norm at the top
//! of each layer half is where it is applied.
//!
//! Two consequences worth stating, both learned by measuring rather than
//! reading. A chain is only on the span from its seed **down to the first op
//! that does not inherit** — one non-inheriting allocation site silently drops
//! everything downstream of it back onto the pool. And a phase whose generation
//! opens but whose chain was never seeded reports a peak of zero while running
//! entirely off the pool, which is indistinguishable from a phase that did
//! nothing; the `wave arenas:` line in the gate is what tells the two apart.
//!
//! The inter-layer hidden state is the deliberate exception: it is the result of
//! a residual add and outlives every layer generation, so it stays owned.
//!
//! The MoE combine target is here too, via [`wave_empty`]. It is *returned*
//! from the expert forward, so nothing inside the MoE code bounds it — the
//! bound comes from one level up, where the layer opens a generation around
//! `ffn_residual`, whose residual update consumes the result. That is the
//! same layer scoping the attention path uses, applied to the layer's other
//! half.

use std::marker::PhantomData;

use candle::cuda_backend::cudarc::driver::{
    CudaSlice, CudaStream, DevicePtr, DeviceRepr, SyncOnDrop,
};
use candle::cuda_backend::wave_provenance::{
    exhausted, wave_alloc, LeaseOrigin, WaveCarve, WaveTicket,
};
use candle::cuda_backend::CudaDType;
use candle::{CudaDevice, CudaStorage, DType, Device, LiveTensor, Result, Shape, Tensor};
use candle_nn::kv_cache::WaveGeneration;

/// Alignment for every wave buffer.
///
/// Matches what `cudaMalloc` guarantees, so a leased buffer is as aligned as
/// the owned one it replaces for every vectorised access the kernels make.
const WAVE_ALIGN: usize = 256;

/// Where a kernel writes its output.
///
/// `'w` is the wave guard the buffer was taken from, and it is what makes the
/// [`Self::into_tensor`] result honest: a leased output borrows the guard, an
/// owned one is free to outlive everything.
pub(crate) enum KernelOutput<'w, T> {
    /// A range of the in-flight wave's half. The wave owns it; this does not.
    ///
    /// `elems`, not bytes — `BumpRange::len` is bytes, and the two are the same
    /// number only for `u8` outputs.
    Leased {
        ptr: u64,
        elems: usize,
        /// The arena this came from, so an op reading the resulting tensor
        /// allocates its own output from the same generation.
        ticket: WaveTicket,
        wave: PhantomData<&'w ()>,
    },
    /// This op's own allocation, freed when the storage drops.
    Owned(CudaSlice<T>),
}

impl<'w, T: CudaDType + DeviceRepr> KernelOutput<'w, T> {
    /// Reserve room for `elem_count` elements of `T`.
    ///
    /// `wave` decides where the memory comes from, and it is the caller's
    /// declared intent rather than an ambient lookup: with a guard the range is
    /// the wave's and borrows it, without one the buffer is owned. There is no
    /// third case where a lease is produced that nothing bounds.
    pub(crate) fn new(
        dev: &CudaDevice,
        elem_count: usize,
        wave: Option<&'w WaveGeneration>,
    ) -> Result<Self> {
        let Some(wave) = wave else {
            // SAFETY: the storage is written by the kernel launched at the call
            // site before anything reads it, exactly as when it was allocated
            // inline there.
            return Ok(Self::Owned(unsafe { dev.alloc::<T>(elem_count)? }));
        };
        let bytes = elem_count * std::mem::size_of::<T>();
        let ticket = wave.ticket();
        let range = wave.alloc(bytes, WAVE_ALIGN)?;
        Ok(Self::Leased {
            ptr: range.ptr,
            elems: elem_count,
            ticket,
            wave: PhantomData,
        })
    }

    /// The destination address, plus the stream guard the owned arm needs.
    ///
    /// A leased range needs none: the half is not handed out again until a
    /// whole layer's work has been issued on the same stream, which
    /// subsumes the per-slice dependency `device_ptr` records. Callers hold the
    /// returned guard across the launch, as they did the one
    /// `CudaSlice::device_ptr` gave them.
    pub(crate) fn device_ptr<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (u64, Option<SyncOnDrop<'a>>) {
        match self {
            Self::Leased { ptr, .. } => (*ptr, None),
            Self::Owned(slice) => {
                let (ptr, guard) = slice.device_ptr(stream);
                (ptr, Some(guard))
            }
        }
    }

    /// Hand the output to candle as storage.
    fn into_storage(self, dev: CudaDevice) -> CudaStorage {
        match self {
            // SAFETY: `ptr` is `elems` elements of `T` in the half pinned by
            // the guard this borrows, so the range outlives the returned
            // storage by construction.
            Self::Leased {
                ptr, elems, ticket, ..
            } => unsafe {
                CudaStorage::wrap_leased_ptr::<T>(ptr, elems, dev, LeaseOrigin::Wave(ticket))
            },
            Self::Owned(slice) => CudaStorage::wrap_cuda_slice(slice, dev),
        }
    }

    /// Hand the output to candle as a tensor bounded by the wave it came from.
    ///
    /// The one place the kernel wrappers turn storage into a tensor. Going
    /// through here rather than `CustomOp1` is what preserves `'w`: that trait
    /// returns `(CudaStorage, Shape)`, which has nowhere to carry it.
    pub(crate) fn into_tensor<S: Into<Shape>>(self, dev: CudaDevice, shape: S) -> LiveTensor<'w> {
        let storage = self.into_storage(dev);
        // SAFETY: the kernel at the call site wrote `shape.elem_count()`
        // elements into this storage before we got here, and `'w` is the
        // guard's own lifetime — carried on `Self` since `new`, so it cannot be
        // widened here.
        unsafe { LiveTensor::from_cuda_storage(storage, shape) }
    }
}

/// The backing that **seeds** a phase's inheritance chain.
///
/// Every other buffer in a phase inherits its arena from an operand, but the
/// first one cannot: its operand is the residual stream, which stays on the pool
/// because it crosses layers. So the head of the chain names the generation
/// directly, and everything downstream follows from it without a single further
/// mention of the wave.
///
/// `Backing::Owned` without a guard, which is the correct answer rather than a
/// fallback: outside a wave there is no arena to seed from.
pub(crate) fn wave_root(wave: Option<&WaveGeneration>) -> candle::cuda_backend::Backing {
    match wave {
        Some(g) => candle::cuda_backend::Backing::Lease(LeaseOrigin::Wave(g.ticket())),
        None => candle::cuda_backend::Backing::Owned,
    }
}

/// An **uninitialised** buffer on the wave's half, or an ordinary one when there
/// is no wave.
///
/// A wave range handed over with no `memset`, for a buffer the caller fully
/// overwrites — hot-path invariant 6. The distinction matters here rather than
/// being a micro-optimisation: this exists to give a *root* operand wave
/// provenance, and a root is by definition something whose every byte is about
/// to be written from somewhere else.
///
/// There is deliberately **no zeroing counterpart**. The one caller that had one
/// was the MoE combine target, on the belief that the deterministic scatter
/// accumulated into it; the scatter defines every element it touches, so the
/// memset was writing the exact bytes the kernel was about to stamp.
///
/// **This is the constructor for a provenance root that has no device operand to
/// inherit from.** `Tensor::empty` can only produce an `Owned` tensor, and
/// `empty_beside` only relays provenance an operand already has — so a chain
/// starting from a buffer the sequence owns across waves (a rewind stash, a
/// carried conv tail) lands wholly on the pool unless it is staged through this
/// first. See [`Tensor::empty_beside`]'s note that "one broken provenance root
/// becomes dozens of sites in a report".
///
/// SAFETY / CONTRACT: as [`candle::Tensor::empty`] — every element must be
/// written before it is read.
pub(crate) fn wave_empty<'w, S: Into<Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
    wave: Option<&'w WaveGeneration>,
) -> Result<LiveTensor<'w>> {
    let shape = shape.into();
    let (Device::Cuda(_), Some(wave)) = (device, wave) else {
        return Tensor::empty(shape, dtype, device);
    };
    let bytes = shape.elem_count() * dtype.size_in_bytes();
    let ticket = wave.ticket();
    let range = wave.alloc(bytes, WAVE_ALIGN)?;
    // SAFETY: `range` is `bytes` of the half pinned by `wave`, nothing else
    // addresses it within this generation, and the returned tensor borrows
    // `wave` so it cannot be named after the guard that reclaims the range.
    unsafe {
        LiveTensor::from_leased_cuda_ptr(range.ptr, dtype, shape, device, LeaseOrigin::Wave(ticket))
    }
}

/// A host-built table uploaded onto the arena a [`WaveTicket`] names, returning
/// a plain [`Tensor`].
///
/// The upload counterpart of [`wave_empty`]: a pointer array, a row map, a
/// rotary layout are built on the host, so there is no device operand whose
/// provenance they could inherit — `Tensor::from_vec` can only ever produce an
/// `Owned` tensor, i.e. a driver allocation inside the wave, from the memory the
/// reservation deliberately does not cover.
///
/// This is the form the per-wave **metadata** uploads take — ragged prefill
/// offsets, page and candidate tables, gathered position ids — which is exactly
/// what [`candle_nn::kv_cache::WAVE_FORWARD_BYTES`] describes its span as
/// holding. They are built deep inside the sweep, by functions that have no
/// business borrowing a generation, and they are `Tensor`-typed because every
/// consumer downstream of them is; a ticket is a `Copy` coordinate, so it
/// reaches them without changing a single signature's lifetime.
///
/// Sound for the same reason as [`wave_empty_ticketed`]: the lease frees nothing
/// on drop and the range's only reclaim is the generation's reset, which cannot
/// happen while the forward that opened it is still running. A ticket whose
/// generation has closed is an ordinary upload, which is a correct answer
/// rather than a silent failure; an open generation with no room is an error.
///
/// The copy is issued **on the device's stream**, and is not waited for.
/// Stream-ordered because the destination is recycled wave memory: the legacy
/// NULL stream does not order against a `NonBlocking` stream (which is what
/// cudarc creates), so an unordered copy can land on addresses the previous
/// generation's kernels are still reading. **No host wait**: for a transfer
/// *from pageable host memory* the driver stages through its own pinned buffer
/// and `cuMemcpyHtoDAsync` returns only once `data` has been copied into it, so
/// the `Vec` may drop when this returns. The DMA to the device may still be
/// outstanding, which is what the stream ordering covers.
pub(crate) fn wave_from_vec_ticketed<D: CudaDType + candle::WithDType, S: Into<Shape>>(
    data: Vec<D>,
    shape: S,
    device: &Device,
    ticket: Option<WaveTicket>,
) -> Result<Tensor> {
    let shape = shape.into();
    if shape.elem_count() != data.len() {
        candle::bail!(
            "wave_from_vec_ticketed: {} elements for a shape of {}",
            data.len(),
            shape.elem_count()
        );
    }
    let bytes = std::mem::size_of_val(data.as_slice());
    let (Device::Cuda(cuda), Some(ticket)) = (device, ticket) else {
        return Tensor::from_vec(data, shape, device);
    };
    let ptr = match wave_alloc(ticket, bytes, WAVE_ALIGN) {
        WaveCarve::Carved(ptr) => ptr,
        WaveCarve::Closed => return Tensor::from_vec(data, shape, device),
        WaveCarve::Exhausted => return Err(exhausted(ticket, bytes)),
    };
    // `ptr` addresses `bytes` the resolver just carved from the ticket's arena
    // and nothing else holds that range in this generation. The upload is
    // ordered behind everything issued before it — recorded into a wave
    // capture's segment, or queued on the compute stream — and returns once
    // `data` has been staged, so the `Vec` may drop when this returns.
    //
    // SAFETY: `data` is `bytes` of plain device-representable values.
    let raw = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };
    cuda.upload_raw(ptr, raw)?;
    // SAFETY: as above. The lease frees nothing on drop, so the range's only
    // reclaim is the generation's reset.
    unsafe { Tensor::from_leased_cuda_ptr(ptr, D::DTYPE, shape, device, LeaseOrigin::Wave(ticket)) }
}

/// Write `data` from the host into `dst`, a contiguous buffer that already
/// exists — the upload that allocates nothing.
///
/// For state a caller holds across forwards and refills each time: the copy
/// lands in place, so the only device memory involved is the buffer's own.
/// Issued on the device's stream and not waited for, for the reasons
/// [`wave_from_vec_ticketed`] gives — the call returns once `data` is staged.
pub(crate) fn upload_into<D: CudaDType + candle::WithDType>(
    dst: &Tensor,
    data: &[D],
) -> Result<()> {
    if dst.dtype() != D::DTYPE || !dst.is_contiguous() || dst.elem_count() != data.len() {
        candle::bail!(
            "upload_into: {} {:?} elements into a {:?} {:?} buffer (contiguous: {})",
            data.len(),
            D::DTYPE,
            dst.dtype(),
            dst.shape(),
            dst.is_contiguous()
        );
    }
    let Device::Cuda(cuda) = dst.device() else {
        candle::bail!("upload_into: the buffer must be on CUDA");
    };
    let stream = cuda.compute_stream();
    let (storage, layout) = dst.storage_and_layout();
    let candle::Storage::Cuda(c) = &*storage else {
        candle::bail!("upload_into: the buffer must be on CUDA");
    };
    let slice = c.as_cuda_slice::<D>()?;
    let (base, _guard) = slice.device_ptr(&stream);
    let at = base + (layout.start_offset() * std::mem::size_of::<D>()) as u64;
    // `at` addresses `data.len()` elements of `dst`'s own storage (the length
    // and contiguity are checked above); the upload is ordered behind every
    // reader of `dst` issued before it — see `wave_from_vec_ticketed`.
    //
    // SAFETY: `data` is plain device-representable values.
    let raw = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    cuda.upload_raw(at, raw)?;
    Ok(())
}

/// [`wave_empty`] for a holder of a [`WaveTicket`] rather than of the guard.
///
/// The ticket is a `Copy` coordinate of an open generation, so a caller that
/// cannot borrow the generation itself can still carve from its arena. The
/// result is a `Tensor`, i.e. `'static`; that is sound because it **owns
/// nothing** — it is a lease, so its drop frees nothing — and the wave's own
/// reset reclaims the range. A ticket whose generation has already closed
/// allocates from the pool; an open generation with no room is an error. It
/// issues no driver call.
///
/// SAFETY / CONTRACT: as [`candle::Tensor::empty`] — every element must be
/// written before it is read.
pub(crate) fn wave_empty_ticketed<S: Into<Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
    ticket: Option<WaveTicket>,
) -> Result<Tensor> {
    let shape = shape.into();
    let bytes = shape.elem_count() * dtype.size_in_bytes();
    let (Device::Cuda(_), Some(ticket)) = (device, ticket) else {
        return Tensor::empty(shape, dtype, device);
    };
    let ptr = match wave_alloc(ticket, bytes, WAVE_ALIGN) {
        WaveCarve::Carved(ptr) => ptr,
        WaveCarve::Closed => return Tensor::empty(shape, dtype, device),
        WaveCarve::Exhausted => return Err(exhausted(ticket, bytes)),
    };
    // SAFETY: `ptr` addresses `bytes` the resolver just carved from the ticket's
    // arena, and no other claimant holds that range within this generation. The
    // lease frees nothing on drop, so the range's only reclaim is the
    // generation's reset.
    unsafe { Tensor::from_leased_cuda_ptr(ptr, dtype, shape, device, LeaseOrigin::Wave(ticket)) }
}
