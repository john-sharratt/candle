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
//! The MoE combine target is here too, via [`wave_zeros`]. It is *returned*
//! from the expert forward, so nothing inside the MoE code bounds it — the
//! bound comes from one level up, where the layer opens a generation around
//! `ffn_forward` and the residual add that consumes the result. That is the
//! same layer scoping the attention path uses, applied to the layer's other
//! half.

use std::marker::PhantomData;

use candle::cuda_backend::cudarc::driver::result::memset_d8_async;
use candle::cuda_backend::cudarc::driver::{
    CudaSlice, CudaStream, DevicePtr, DeviceRepr, SyncOnDrop,
};
use candle::cuda_backend::wave_provenance::{wave_alloc, LeaseOrigin, WaveTicket};
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

/// A zeroed tensor on the wave's half, or an ordinary one when there is no wave.
///
/// For accumulators, where the caller needs the buffer to *start* at zero — the
/// MoE combine target is scattered into, not overwritten, so it cannot take a
/// wave range as-is. The fill is `memset` on the device's stream, which is the
/// same work `Tensor::zeros` does; what it replaces is the allocate/free pair
/// around it, one per MoE layer per forward.
pub(crate) fn wave_zeros<'w, S: Into<Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
    wave: Option<&'w WaveGeneration>,
) -> Result<LiveTensor<'w>> {
    let shape = shape.into();
    let (Device::Cuda(cuda), Some(wave)) = (device, wave) else {
        return Tensor::zeros(shape, dtype, device);
    };
    let stream = cuda.cuda_stream();
    let bytes = shape.elem_count() * dtype.size_in_bytes();
    let ticket = wave.ticket();
    let range = wave.alloc(bytes, WAVE_ALIGN)?;
    // SAFETY: `range` is `bytes` of the half pinned by `wave`, and nothing else
    // addresses it within this generation.
    unsafe { memset_d8_async(range.ptr, 0, bytes, stream.cu_stream()) }
        .map_err(|e| candle::Error::Msg(format!("zeroing a wave buffer: {e}")))?;
    // SAFETY: as above, and the returned tensor borrows `wave`, so it cannot be
    // named after the guard that reclaims the range has dropped.
    unsafe {
        LiveTensor::from_leased_cuda_ptr(range.ptr, dtype, shape, device, LeaseOrigin::Wave(ticket))
    }
}

/// [`wave_zeros`] for a holder of a [`WaveTicket`] rather than of the guard.
///
/// The expert-pipeline thread is the caller that needs this. It cannot borrow
/// the generation — a `&WaveGeneration` does not cross a channel — but the
/// ticket is a `Copy` coordinate and does, and the submitting thread blocks on
/// the response for the whole request, so the generation is open throughout.
///
/// The result is a `Tensor`, i.e. `'static`, because it is handed back over the
/// same channel. That is sound for the same reason the ticket is: the tensor
/// **owns nothing** — it is a lease, so its drop frees nothing — and the wave's
/// own reset reclaims the range. A ticket whose generation has already closed
/// resolves to `None` and this allocates from the pool, which is a correct
/// answer rather than a fallback.
pub(crate) fn wave_zeros_ticketed<S: Into<Shape>>(
    shape: S,
    dtype: DType,
    device: &Device,
    ticket: Option<WaveTicket>,
) -> Result<Tensor> {
    let shape = shape.into();
    let bytes = shape.elem_count() * dtype.size_in_bytes();
    let (Device::Cuda(cuda), Some(ticket)) = (device, ticket) else {
        return Tensor::zeros(shape, dtype, device);
    };
    let Some(ptr) = wave_alloc(ticket, bytes, WAVE_ALIGN) else {
        return Tensor::zeros(shape, dtype, device);
    };
    let stream = cuda.cuda_stream();
    // `memset_d8_async` is a raw driver call, and the driver takes its context
    // from the calling thread. [`wave_zeros`] gets away without this because it
    // runs on the forward thread, where candle has already bound one; this runs
    // on the expert-pipeline thread, which has not, and the call fails with
    // `CUDA_ERROR_INVALID_CONTEXT`. Binding is idempotent.
    stream
        .context()
        .bind_to_thread()
        .map_err(|e| candle::Error::Msg(format!("binding the device context: {e}")))?;
    // SAFETY: `ptr` addresses `bytes` the resolver just carved from the ticket's
    // arena, and no other claimant holds that range within this generation.
    unsafe { memset_d8_async(ptr, 0, bytes, stream.cu_stream()) }
        .map_err(|e| candle::Error::Msg(format!("zeroing a wave buffer: {e}")))?;
    // SAFETY: as above. The lease frees nothing on drop, so the range's only
    // reclaim is the generation's reset.
    unsafe { Tensor::from_leased_cuda_ptr(ptr, dtype, shape, device, LeaseOrigin::Wave(ticket)) }
}
