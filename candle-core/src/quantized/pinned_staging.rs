//! Pinned-memory staging for zero-copy GPU reads over PCIe.
//!
//! The GPU kernel reads compact-move descriptors directly from host pinned
//! memory — no `memcpy`, no driver calls on the submit hot-path.
//!
//! # Memory layout
//!
//! A **bump arena** (default 128 MB) is allocated at construction with
//! `CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED`.  The CUDA
//! driver maps this into the GPU's address space so reads go straight over
//! PCIe.  Allocations ≤ 16 MB are bump-allocated (pointer increment,
//! nanoseconds, zero OS/driver calls).  Larger allocations get a dedicated
//! `cuMemHostAlloc` each and are freed after stream sync.
//!
//! # Generations
//!
//! Because the GPU reads *live* from the arena (no snapshot copy), the arena
//! must not be reset while any kernel might still reference its data.  A
//! **generation** guard ([`Generation`], obtained via
//! [`PinnedStager::begin_generation`]) prevents resets for its lifetime.
//!
//! If the arena fills while a generation is held, an **overflow arena** of
//! the same size and flags is allocated and becomes the new bump target.
//! This preserves zero-copy performance — no fallback to owned buffers or
//! slow memcpy paths.
//!
//! When the last generation drops, a CUDA **event** is recorded on the stream
//! and the reset is DEFERRED — the arenas stay dirty until that fence fires.
//! The next [`PinnedStager::begin_generation`] queries it (non-blocking) and,
//! if it has fired, resets all arenas and frees the overflow slabs (only the
//! original is retained). Dropping a generation therefore never blocks: the
//! previous form synchronised the stream inline, draining the whole GPU
//! pipeline from a destructor. The blocking reset still exists on the one path
//! that genuinely needs the space back — an allocation that finds the arena
//! full with no live generation.
//!
//! # Usage
//!
//! ```ignore
//! let gen = stager.begin_generation();
//! for batch in batches {
//!     let mut buf = stager.alloc(len)?;
//!     write_moves(&mut buf);
//!     let gpu = stager.submit(buf)?;
//!     launch_kernel(gpu.dev_ptr());
//! }
//! drop(gen); // records a fence; the arena resets on the next generation
//! ```

use crate::cuda_backend::WrapErr;
use crate::{CudaDevice, Result};
use cudarc::driver::sys;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr};
use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

/// RAII generation guard.
///
/// While at least one `Generation` is alive, the arena will never be reset —
/// all submitted bump pointers remain valid. When the last generation drops
/// over a dirty arena, a stream event is recorded and the reset is deferred to
/// the next generation, which resets only once that fence has fired.
///
/// Create via [`PinnedStager::begin_generation`].
pub struct Generation {
    inner: Arc<Mutex<PinnedStagerInner>>,
    /// The stager epoch captured when this generation began. Device pointers it
    /// hands out are valid only while the stager's epoch still equals this.
    epoch: u64,
}

impl Drop for Generation {
    fn drop(&mut self) {
        let should_flush = {
            let mut inner = self.inner.lock().unwrap();
            inner.live_generations -= 1;
            inner.live_generations == 0 && inner.arena_dirty
        };
        if should_flush {
            // Record a fence and leave. The arena stays dirty and un-reset until
            // the event fires; `PinnedStagerInner::try_reclaim` does the actual
            // reset on the next generation. Draining the stream here — from a
            // destructor, on whatever thread dropped the last guard — stalled the
            // entire GPU pipeline once per generation.
            // The fence is recorded while HOLDING the lock, deliberately.
            //
            // Recording it outside and re-locking opens a window in which a whole
            // generation can begin and end: that generation records a LATER fence
            // and stores it, then this thread re-locks, still sees
            // `live_generations == 0`, and overwrites it with its own STALE one.
            // The stale event fires earlier, so `try_reclaim` would reset — and
            // `truncate(1)` free — pinned overflow slabs while the newer
            // generation's kernels were still reading them directly over PCIe.
            //
            // The old code was immune to this only because a `synchronize()`
            // cannot be stale; a recorded event can. `record_event` is a cheap
            // driver call and generation drops are no longer hot, so serialising
            // them behind this mutex costs nothing worth the hazard.
            let mut inner = self.inner.lock().unwrap();
            // Re-check under the same lock we will publish under: another
            // generation may have started, in which case the arena is live again
            // and no fence belongs here — that generation's own drop records one.
            if inner.live_generations == 0 {
                let stream = match (inner.explicit_stream.clone(), inner.dev.clone()) {
                    (Some(stream), _) => Some(stream),
                    (None, Some(dev)) => Some(dev.cuda_stream().clone()),
                    (None, None) => None,
                };
                match stream {
                    // No stream to fence against (CPU-only test stager): the
                    // arena cannot be under GPU read, so reset immediately.
                    None => inner.reset_arenas_now(),
                    Some(stream) => match stream.record_event(None) {
                        Ok(e) => inner.pending_reset = Some(e),
                        // A FAILED record is not the same as "no stream", and must
                        // never fall through to an unsynchronised reset: rewinding
                        // the bump pointers and freeing overflow slabs while
                        // kernels may still be reading them over PCIe is a
                        // use-after-free. With no fence to defer behind, fall back
                        // to the blocking behaviour this path replaced.
                        Err(e) => {
                            tracing::warn!(
                                "pinned staging: fence record failed ({e}); \
                                 synchronising before reset"
                            );
                            let _ = stream.synchronize();
                            inner.reset_arenas_now();
                        }
                    },
                }
            }
        }
    }
}

impl Generation {
    /// Allocate a pinned staging buffer of `len` bytes.
    ///
    /// Delegates to the underlying [`PinnedStager::alloc`].  Because a
    /// `Generation` is alive, the arena will never be reset mid-flight —
    /// if the arena fills, an overflow arena is added automatically.
    pub fn alloc(&self, len: usize) -> crate::Result<PinnedBuf> {
        // Re-use the stager's alloc logic via the shared inner.
        // We construct a temporary PinnedStager handle to call its alloc.
        let stager = PinnedStager {
            inner: Arc::clone(&self.inner),
        };
        stager.alloc(len)
    }

    /// Return a device-visible handle to the staging buffer.
    ///
    /// Delegates to the underlying [`PinnedStager::submit`].
    pub fn submit(&self, buf: PinnedBuf) -> crate::Result<GpuBuf> {
        let stager = PinnedStager {
            inner: Arc::clone(&self.inner),
        };
        stager.submit(buf)
    }

    /// The stager epoch this generation was opened at. A device pointer this
    /// generation returned stays valid only while the generation is alive; once
    /// it (and every sibling) drops, the arena resets and the next generation
    /// carries a higher epoch. Cache consumers compare against this to know
    /// whether a previously-handed-out pointer still refers to live memory.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

/// A GPU buffer returned by [`PinnedStager::submit`].
///
/// Holds either a pointer into the device-side arena (zero-alloc) or an
/// individually allocated `CudaSlice` for large transfers.
#[derive(Clone)]
pub struct GpuBuf {
    dev_ptr: u64,
    len: usize,
    /// Holds the `CudaSlice` alive for owned (non-arena) buffers.
    /// `None` for arena buffers — the arena owns the device memory.
    _owned: Option<CudaSlice<u8>>,
}

impl GpuBuf {
    /// Raw device pointer for kernel launches.
    pub fn dev_ptr(&self) -> u64 {
        self.dev_ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// A zero-length staging buffer — nothing was reserved.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Wrap a pre-existing device pointer owned by something else (e.g. a `Tensor`).
    /// The caller must ensure the owning allocation outlives this `GpuBuf`.
    pub fn from_borrowed(dev_ptr: u64, len: usize) -> Self {
        Self {
            dev_ptr,
            len,
            _owned: None,
        }
    }

    /// Take ownership of a `CudaSlice<u8>` and expose it as a `GpuBuf`.
    /// A `CudaDevice` reference is required to resolve the stable device pointer
    /// (a cudarc API requirement; the pointer is valid for the lifetime of the slice).
    pub fn from_raw_owned(slice: CudaSlice<u8>, dev: &CudaDevice) -> Self {
        let stream = dev.cuda_stream();
        let ptr = {
            let (ptr, _guard) = slice.device_ptr(&stream);
            ptr
        }; // guard dropped; pointer remains valid while slice lives
        let len = slice.len();
        Self {
            dev_ptr: ptr,
            len,
            _owned: Some(slice),
        }
    }
}

/// Default arena slab size (128 MB).
const DEFAULT_ARENA_SIZE: usize = 128 * 1024 * 1024;

/// Allocations at or below this threshold use the bump arena.
const BUMP_THRESHOLD: usize = 16 * 1024 * 1024;

/// Alignment for bump allocations (16 bytes — sufficient for uint4 / SSE).
const BUMP_ALIGN: usize = 16;

/// CUDA pinned host memory buffer.
///
/// Transparently either a bump-allocated slice from the arena (zero-cost alloc)
/// or an individually-allocated owned buffer for large transfers.
pub enum PinnedBuf {
    /// Slice of the bump arena. Does NOT free memory on drop — the arena owns it.
    Bump { ptr: *mut u8, len: usize },
    /// Individually allocated via `cuMemHostAlloc`. Freed on drop.
    Owned { ptr: *mut u8, len: usize },
    /// Plain host memory for no-op / non-CUDA staging.
    Host { data: Vec<u8> },
}

// SAFETY: The pinned memory is host-side and not tied to a specific thread.
unsafe impl Send for PinnedBuf {}
// SAFETY: Shared references only give immutable slice access; mutation
// requires &mut PinnedBuf which is exclusive.
unsafe impl Sync for PinnedBuf {}

impl PinnedBuf {
    /// Allocate `len` bytes of write-combined pinned host memory (owned mode).
    pub fn alloc_owned(len: usize) -> Result<Self> {
        // CU_MEMHOSTALLOC_WRITECOMBINED = 0x04 — fast for CPU→GPU
        // burst writes, **slow** for CPU reads (uncached) and slow as
        // a destination of Windows kernel I/O (`ReadFile` and friends
        // bounce through a degraded path when the destination is WC).
        // The HtoD-only staging paths (bg-quantizer DtoH-then-HtoD,
        // PinnedStager arenas) want WC; the cold-load read scratch
        // does not — see [`Self::alloc_owned_default`].
        Self::alloc_owned_with_flags(len, 0x04)
    }

    /// Allocate `len` bytes of pinned host memory with no write-combining
    /// attribute (owned mode).
    ///
    /// Use this when the buffer is a **destination of kernel I/O** —
    /// most notably the cold-load `NVMe → pinned` direct read on
    /// Windows, where WC pages take a degraded `ReadFile` path that
    /// caps at ~7 MB/s. Plain pinned (page-locked, write-back-cached)
    /// memory accepts the NVMe DMA at full sequential bandwidth, and
    /// the subsequent HtoD upload only loses the small WC bonus on
    /// the GPU side (driver handles cache snooping).
    pub fn alloc_owned_default(len: usize) -> Result<Self> {
        Self::alloc_owned_with_flags(len, 0)
    }

    /// Staging allocation that never aborts the process under memory pressure:
    /// try pinned host memory first, and on failure fall back to a **fallible**
    /// plain host `Vec` via `try_reserve_exact` (NOT `vec![]`, which calls
    /// `handle_alloc_error` and aborts). Returns `Err` when the host heap can't
    /// supply `len` contiguous bytes — so a caller under pressure can shrink its
    /// batch and retry, or defer the work, instead of crashing.
    pub fn alloc_default_or_host_fallible(len: usize) -> Result<Self> {
        if let Ok(b) = Self::alloc_owned_default(len) {
            return Ok(b);
        }
        let mut data: Vec<u8> = Vec::new();
        data.try_reserve_exact(len).map_err(|e| {
            crate::Error::Msg(format!(
                "host staging alloc of {len} bytes failed (pinned host + heap both exhausted): {e}"
            ))
        })?;
        data.resize(len, 0);
        Ok(Self::Host { data })
    }

    fn alloc_owned_with_flags(len: usize, flags: u32) -> Result<Self> {
        if len == 0 {
            return Ok(Self::Bump {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
            });
        }
        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            sys::cuMemHostAlloc(&mut host_ptr, len, flags)
                .result()
                .map_err(|e| crate::Error::Msg(format!("cuMemHostAlloc failed: {:?}", e)))?;
        }
        crate::vram::note_host_pinned_alloc(len as u64);
        Ok(Self::Owned {
            ptr: host_ptr as *mut u8,
            len,
        })
    }

    /// A zero-length buffer — nothing was reserved.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Bump { len, .. } | Self::Owned { len, .. } => *len,
            Self::Host { data } => data.len(),
        }
    }

    fn ptr(&self) -> *mut u8 {
        match self {
            Self::Bump { ptr, .. } | Self::Owned { ptr, .. } => *ptr,
            Self::Host { data } => data.as_ptr() as *mut u8,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr(), self.len()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr(), self.len()) }
    }

    /// Returns true if this is a bump-allocated buffer.
    pub fn is_bump(&self) -> bool {
        matches!(self, Self::Bump { .. })
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if let Self::Owned { ptr, len } = self {
            if *len > 0 {
                unsafe {
                    let _ = sys::cuMemFreeHost(*ptr as *mut std::ffi::c_void).result();
                }
                crate::vram::note_host_pinned_free(*len as u64);
            }
        }
        // Bump variants do nothing — the arena owns the memory.
    }
}

impl Deref for PinnedBuf {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl DerefMut for PinnedBuf {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.as_mut_slice()
    }
}

/// A contiguous pinned slab allocated once at construction.
///
/// Allocated with `CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED`
/// so the GPU can read directly from host memory over PCIe — zero memcpy.
struct PinnedArena {
    ptr: *mut u8,
    /// Device-visible pointer into the same pinned memory (mapped).
    dev_ptr: u64,
    capacity: usize,
    /// Current bump offset. Reset to 0 on sync.
    offset: usize,
}

// SAFETY: host-side pinned memory, not thread-specific.
unsafe impl Send for PinnedArena {}

impl PinnedArena {
    fn new(capacity: usize) -> Result<Self> {
        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        unsafe {
            // CU_MEMHOSTALLOC_DEVICEMAP = 0x02
            // CU_MEMHOSTALLOC_WRITECOMBINED = 0x04
            sys::cuMemHostAlloc(&mut host_ptr, capacity, 0x02 | 0x04)
                .result()
                .map_err(|e| {
                    crate::Error::Msg(format!(
                        "cuMemHostAlloc for arena ({} MB) failed: {:?}",
                        capacity / (1024 * 1024),
                        e
                    ))
                })?;
        }
        // Get the device-visible pointer for this mapped allocation.
        let mut dev_ptr: sys::CUdeviceptr = 0;
        unsafe {
            sys::cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0)
                .result()
                .map_err(|e| {
                    crate::Error::Msg(format!(
                        "cuMemHostGetDevicePointer for arena failed: {:?}",
                        e
                    ))
                })?;
        }
        crate::vram::note_host_pinned_alloc(capacity as u64);
        Ok(Self {
            ptr: host_ptr as *mut u8,
            dev_ptr,
            capacity,
            offset: 0,
        })
    }

    /// Try to bump-allocate `len` bytes. Returns `None` if not enough space.
    fn try_alloc(&mut self, len: usize) -> Option<PinnedBuf> {
        // Align up
        let aligned = (self.offset + BUMP_ALIGN - 1) & !(BUMP_ALIGN - 1);
        let end = aligned + len;
        if end > self.capacity {
            return None;
        }
        let ptr = unsafe { self.ptr.add(aligned) };
        self.offset = end;
        Some(PinnedBuf::Bump { ptr, len })
    }

    /// Reset the bump offset. Only safe after a stream sync.
    fn reset(&mut self) {
        self.offset = 0;
    }

    fn used(&self) -> usize {
        self.offset
    }
}

impl Drop for PinnedArena {
    fn drop(&mut self) {
        if self.capacity > 0 {
            unsafe {
                let _ = sys::cuMemFreeHost(self.ptr as *mut std::ffi::c_void).result();
                crate::vram::note_host_pinned_free(self.capacity as u64);
            }
        }
    }
}

/// Staging allocator backed by CUDA pinned memory.
///
/// Small allocations (≤ 16 MB) are bump-allocated from a device-mapped
/// pinned arena for near-zero overhead. Large allocations get their own
/// pinned buffer. Overflow arenas are added during active generations.
///
/// Cloneable via `Arc<Mutex<..>>` — all clones share the same arena and queue.
#[derive(Clone)]
pub struct PinnedStager {
    inner: Arc<Mutex<PinnedStagerInner>>,
}

impl std::fmt::Debug for PinnedStager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedStager").finish_non_exhaustive()
    }
}

struct PinnedStagerInner {
    dev: Option<CudaDevice>,
    /// When set, operations that need a stream (owned-buf H2D copy, sync on flush)
    /// use this explicit stream instead of `dev.cuda_stream()`.
    /// Used by the background quantizer thread which runs on a dedicated CUDA stream.
    explicit_stream: Option<Arc<CudaStream>>,
    /// Pinned arenas for small buffers. The first is allocated at construction;
    /// overflow arenas are added when the current one fills during an active
    /// generation. On flush, all but the first are freed.
    /// Device-mapped: GPU reads directly over PCIe, zero memcpy.
    arenas: Vec<PinnedArena>,
    /// Size in bytes for each arena slab (used when allocating overflow arenas).
    arena_size: usize,
    /// Spent owned (large) pinned buffers awaiting cleanup.
    pending_owned: Vec<PinnedBuf>,
    /// Cumulative byte size of pending owned buffers.
    pending_owned_bytes: usize,
    /// Whether any bump allocations have been submitted since last reset.
    arena_dirty: bool,
    /// Number of bump-allocated buffers that have been handed out but not yet
    /// submitted. The arena must NOT be reset while this is non-zero, because
    /// those buffers point into arena memory that would be reused.
    bump_outstanding: usize,
    /// Number of live [`Generation`] guards. While > 0, the arenas must not
    /// be reset — submitted bump pointers are still potentially in use.
    live_generations: usize,
    /// Event recorded on the stream when the last generation dropped over a
    /// dirty arena. The arena is safe to reset once this has FIRED — until then
    /// kernels may still be reading the bump region directly over PCIe.
    ///
    /// This is what makes the reset deferred instead of synchronous. Dropping
    /// the last generation used to call `stream.synchronize()` inline, draining
    /// the whole GPU pipeline from a destructor on the caller's thread — 8.8% of
    /// sampled CPU in a full-workspace ingest, reached via
    /// `drive_wave` → drop `CpuStorage` → here. Recording an event costs a
    /// queue entry; the reset then happens on the next generation, by which
    /// point the event has almost always fired and the check is a query.
    pending_reset: Option<cudarc::driver::CudaEvent>,
    /// Monotonic counter bumped every time a fresh [`Generation`] begins. A
    /// generation's arena is reset (all bump pointers invalidated) once the last
    /// guard drops, so the epoch captured at `begin` uniquely identifies the
    /// arena's current fill. Consumers that cache a device pointer handed out by
    /// one generation compare epochs to detect a reset before reusing it.
    epoch: u64,
}

impl PinnedStagerInner {
    /// Reset the arenas and drop spent owned buffers. The caller must have
    /// established that no kernel can still be reading them.
    fn reset_arenas_now(&mut self) {
        for a in self.arenas.iter_mut() {
            a.reset();
        }
        self.arenas.truncate(1);
        self.arena_dirty = false;
        self.pending_owned.clear();
        self.pending_owned_bytes = 0;
        self.pending_reset = None;
    }

    /// Reset the arenas if a deferred reset is pending AND its fence has fired.
    ///
    /// Non-blocking: an event that has not fired leaves the arena dirty and the
    /// reset pending, so the caller keeps bump-allocating (into an overflow
    /// arena if needed) exactly as it would while a generation were live. Safe
    /// to call whenever `live_generations == 0`.
    ///
    /// The blocking counterpart is [`PinnedStager::sync_and_reset_arena`], taken
    /// when the arena is full and the space is genuinely needed: its stream sync
    /// subsumes this fence, so no separate blocking wait exists here.
    fn try_reclaim(&mut self) {
        if self.live_generations != 0 {
            return;
        }
        let Some(event) = self.pending_reset.as_ref() else {
            return;
        };
        // `cuEventQuery` is the non-blocking form: SUCCESS means every preceding
        // stream operation has completed, NOT_READY means work is outstanding.
        // Anything else is a real driver error — treat it as not-ready and let
        // the blocking path below deal with it, rather than resetting memory the
        // GPU may still be reading.
        let ready = unsafe { sys::cuEventQuery(event.cu_event()) } == sys::CUresult::CUDA_SUCCESS;
        if ready {
            self.reset_arenas_now();
        }
    }
}

impl PinnedStager {
    /// Returns the arena size to use: `CANDLE_ARENA_MB` env-var override, else 128 MB.
    fn effective_arena_size() -> usize {
        std::env::var("CANDLE_ARENA_MB")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .map(|mb| mb * 1024 * 1024)
            .unwrap_or(DEFAULT_ARENA_SIZE)
    }

    /// Create a new stager with the default 128 MB arena (or `CANDLE_ARENA_MB` override).
    pub fn new(dev: &CudaDevice) -> Self {
        Self::with_arena_size(dev, Self::effective_arena_size())
    }

    /// Create a new stager from a generic [`Device`].
    ///
    /// On CUDA devices this allocates a real pinned arena.
    /// On non-CUDA devices this returns a no-op host-backed stager so callers
    /// can thread the type through uniformly on all builds and devices.
    pub fn new_from_device(device: &crate::Device) -> Self {
        match device {
            crate::Device::Cuda(cuda_dev) => Self::new(cuda_dev),
            _ => Self::noop(),
        }
    }

    /// Create a no-op host-backed stager used on non-CUDA devices.
    pub fn noop() -> Self {
        Self {
            inner: Arc::new(Mutex::new(PinnedStagerInner {
                dev: None,
                explicit_stream: None,
                arenas: Vec::new(),
                arena_size: 0,
                pending_owned: Vec::new(),
                pending_owned_bytes: 0,
                arena_dirty: false,
                bump_outstanding: 0,
                live_generations: 0,
                pending_reset: None,
                epoch: 0,
            })),
        }
    }

    /// Create a new stager with a custom arena size in bytes.
    ///
    /// The arena is allocated upfront as a single pinned slab.
    /// Allocations ≤ 16 MB are bump-allocated from this arena;
    /// larger ones get their own pinned buffer.
    pub fn with_arena_size(dev: &CudaDevice, arena_bytes: usize) -> Self {
        let arena = PinnedArena::new(arena_bytes)
            .expect("failed to allocate pinned arena for PinnedStager");
        Self {
            inner: Arc::new(Mutex::new(PinnedStagerInner {
                dev: Some(dev.clone()),
                explicit_stream: None,
                arenas: vec![arena],
                arena_size: arena_bytes,
                pending_owned: Vec::new(),
                pending_owned_bytes: 0,
                arena_dirty: false,
                bump_outstanding: 0,
                live_generations: 0,
                pending_reset: None,
                epoch: 0,
            })),
        }
    }

    /// Create a stager that uses `stream` for H2D copies and synchronization,
    /// instead of the device's default stream.  Used by the background quantizer
    /// thread which runs all CUDA work on a dedicated second stream.
    pub fn with_stream(dev: &CudaDevice, stream: Arc<CudaStream>) -> Self {
        let arena = PinnedArena::new(DEFAULT_ARENA_SIZE)
            .expect("failed to allocate pinned arena for PinnedStager::with_stream");
        Self {
            inner: Arc::new(Mutex::new(PinnedStagerInner {
                dev: Some(dev.clone()),
                explicit_stream: Some(stream),
                arenas: vec![arena],
                arena_size: DEFAULT_ARENA_SIZE,
                pending_owned: Vec::new(),
                pending_owned_bytes: 0,
                arena_dirty: false,
                bump_outstanding: 0,
                live_generations: 0,
                pending_reset: None,
                epoch: 0,
            })),
        }
    }

    /// Begin a new generation scope.
    ///
    /// While the returned [`Generation`] is alive, the arena will never be
    /// reset — all bump pointers returned by `submit()` remain valid.
    /// When the last generation drops, the stream is synchronised and the
    /// arena is reset automatically.
    ///
    /// If the arena fills during an active generation, a new overflow
    /// arena slab is allocated automatically (same size, same device-mapped
    /// properties). On flush, overflow arenas are freed.
    pub fn begin_generation(&self) -> Generation {
        let mut inner = self.inner.lock().unwrap();
        // Collect the previous generation's deferred reset, if its fence has
        // fired. This is where the reset actually happens in the steady state —
        // by the time the next generation starts, the prior one's work is
        // normally long done, so the query succeeds and costs nothing.
        inner.try_reclaim();
        inner.live_generations += 1;
        inner.epoch += 1;
        let epoch = inner.epoch;
        Generation {
            inner: Arc::clone(&self.inner),
            epoch,
        }
    }

    /// Allocate a pinned staging buffer of `len` bytes.
    ///
    /// Small allocations (≤ 16 MB) are bump-allocated from the arena.
    /// If the arena is full, the stream is synchronised and the arena reset.
    /// Large allocations get a dedicated pinned buffer.
    pub fn alloc(&self, len: usize) -> Result<PinnedBuf> {
        if len == 0 {
            return Ok(PinnedBuf::Bump {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
            });
        }

        {
            let inner = self.inner.lock().unwrap();
            if inner.dev.is_none() {
                return Ok(PinnedBuf::Host {
                    data: vec![0u8; len],
                });
            }
        }

        if len > BUMP_THRESHOLD {
            return PinnedBuf::alloc_owned(len);
        }

        // Fast path: try bump alloc from the current (last) arena.
        {
            let mut inner = self.inner.lock().unwrap();
            let arena = inner.arenas.last_mut().unwrap();
            if let Some(buf) = arena.try_alloc(len) {
                inner.bump_outstanding += 1;
                return Ok(buf);
            }
            // Cannot reset if there are outstanding bump buffers that
            // haven't been submitted yet — they point into arena memory.
            if inner.bump_outstanding > 0 {
                return Err(crate::Error::Msg(format!(
                    "PinnedStager: arena full with {} outstanding bump buffers; \
                     submit them before allocating more",
                    inner.bump_outstanding,
                )));
            }

            // If generations are live, add an overflow arena instead of
            // resetting. This keeps all previously-submitted bump pointers
            // valid while maintaining zero-copy bump allocation performance.
            if inner.live_generations > 0 {
                let overflow = PinnedArena::new(inner.arena_size)?;
                inner.arenas.push(overflow);
                inner.bump_outstanding += 1;
                let idx = inner.arenas.len() - 1;
                return inner.arenas[idx].try_alloc(len).ok_or_else(|| {
                    inner.bump_outstanding -= 1;
                    crate::Error::Msg(format!(
                        "PinnedStager: allocation of {} bytes exceeds arena capacity ({} bytes)",
                        len, inner.arena_size,
                    ))
                });
            }
        }

        // Arena full, no outstanding bumps, no live generations — sync stream
        // to ensure all in-flight reads from the arenas have completed, then
        // reset.
        self.sync_and_reset_arena()?;

        // Re-acquire lock and retry. Hold the lock through the retry so no
        // other thread can race in and fill the freshly-reset arena.
        let mut inner = self.inner.lock().unwrap();
        inner.bump_outstanding += 1;
        let idx = inner.arenas.len() - 1;
        let cap = inner.arenas[0].capacity;
        inner.arenas[idx].try_alloc(len).ok_or_else(|| {
            inner.bump_outstanding -= 1;
            crate::Error::Msg(format!(
                "PinnedStager: allocation of {} bytes exceeds arena capacity ({} bytes)",
                len, cap,
            ))
        })
    }

    /// Return a device-visible handle to the staging buffer.
    ///
    /// For bump-allocated buffers the arena is device-mapped, so the GPU
    /// reads directly from host pinned memory over PCIe — **zero driver
    /// calls**, zero memcpy. For owned (large) buffers it falls back to
    /// `stream.alloc` + `memcpy_htod`.
    pub fn submit(&self, buf: PinnedBuf) -> Result<GpuBuf> {
        let mut inner = self.inner.lock().unwrap();

        if inner.dev.is_none() {
            return Ok(GpuBuf {
                dev_ptr: 0,
                len: buf.len(),
                _owned: None,
            });
        }

        match &buf {
            PinnedBuf::Bump { ptr, len } => {
                // Find which arena this bump buffer belongs to and compute
                // the device-mapped pointer at the same offset.
                let buf_addr = *ptr as usize;
                let dev_ptr = inner
                    .arenas
                    .iter()
                    .find_map(|a| {
                        let base = a.ptr as usize;
                        if buf_addr >= base && buf_addr < base + a.capacity {
                            let offset = buf_addr - base;
                            debug_assert!(offset + *len <= a.capacity);
                            Some(a.dev_ptr + offset as u64)
                        } else {
                            None
                        }
                    })
                    .expect("PinnedBuf::Bump pointer not found in any arena");

                inner.arena_dirty = true;
                debug_assert!(inner.bump_outstanding > 0);
                inner.bump_outstanding -= 1;

                Ok(GpuBuf {
                    dev_ptr,
                    len: *len,
                    _owned: None,
                })
            }
            PinnedBuf::Owned { .. } => {
                let stream = inner.explicit_stream.clone().unwrap_or_else(|| {
                    inner
                        .dev
                        .as_ref()
                        .expect("cuda stager missing device")
                        .cuda_stream()
                });
                let len = buf.len();
                let mut gpu = unsafe { stream.alloc::<u8>(len).w()? };
                stream.memcpy_htod(buf.as_slice(), &mut gpu).w()?;
                let dev_ptr = {
                    let (ptr, _guard) = gpu.device_ptr(&stream);
                    ptr
                };
                inner.pending_owned_bytes += len;
                inner.pending_owned.push(buf);
                Ok(GpuBuf {
                    dev_ptr,
                    len,
                    _owned: Some(gpu),
                })
            }
            PinnedBuf::Host { .. } => Ok(GpuBuf {
                dev_ptr: 0,
                len: buf.len(),
                _owned: None,
            }),
        }
    }

    /// Synchronise the stream and free all pending resources.
    ///
    /// Resets the bump arena and frees any owned buffers.
    pub fn flush(&self) -> Result<()> {
        let needs_sync = {
            let inner = self.inner.lock().unwrap();
            inner.arena_dirty || inner.arenas.len() > 1 || !inner.pending_owned.is_empty()
        };
        if needs_sync {
            self.sync_and_reset_all()?;
        }
        Ok(())
    }

    /// Sync stream, reset arena, free owned buffers.
    fn sync_and_reset_all(&self) -> Result<()> {
        let (dev, explicit_stream) = {
            let inner = self.inner.lock().unwrap();
            (inner.dev.clone(), inner.explicit_stream.clone())
        };
        if let Some(stream) = explicit_stream {
            stream.synchronize().w()?;
        } else if let Some(dev) = dev {
            dev.cuda_stream().synchronize().w()?;
        }
        let mut inner = self.inner.lock().unwrap();
        // Resets all arenas, drops the overflow slabs (keeping only the first),
        // and CLEARS any deferred fence: the stream sync above subsumes it, so
        // leaving `pending_reset` set would strand a fired event for a later
        // `try_reclaim` to act on against an already-reset arena.
        inner.reset_arenas_now();
        Ok(())
    }

    /// Sync stream and reset arenas (for when arena is full).
    fn sync_and_reset_arena(&self) -> Result<()> {
        let (dev, explicit_stream) = {
            let inner = self.inner.lock().unwrap();
            (inner.dev.clone(), inner.explicit_stream.clone())
        };
        if let Some(stream) = explicit_stream {
            stream.synchronize().w()?;
        } else if let Some(dev) = dev {
            dev.cuda_stream().synchronize().w()?;
        }
        let mut inner = self.inner.lock().unwrap();
        // The stream sync above subsumes any deferred fence — an event recorded
        // on this stream has necessarily fired — so this clears `pending_reset`
        // along with the arenas rather than leaving a stale event behind. Also
        // frees any pending owned buffers, since we synced anyway.
        inner.reset_arenas_now();
        Ok(())
    }

    /// Current bytes in the owned cleanup queue.
    pub fn pending_bytes(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.pending_owned_bytes
    }

    /// Current bytes used across all bump arenas.
    pub fn arena_used(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.arenas.iter().map(|a| a.used()).sum()
    }

    /// Number of arena slabs currently allocated (1 = no overflow).
    pub fn arena_count(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.arenas.len()
    }
}

impl Drop for PinnedStagerInner {
    fn drop(&mut self) {
        if self.arena_dirty || !self.pending_owned.is_empty() {
            if let Some(stream) = &self.explicit_stream {
                let _ = stream.synchronize();
            } else if let Some(dev) = &self.dev {
                let _ = dev.cuda_stream().synchronize();
            }
            self.pending_owned.clear();
        }
        // arenas are dropped automatically (each frees its slab)
    }
}

#[cfg(test)]
mod deferred_reset_tests {
    use super::*;

    /// Dropping the last generation must NOT leave the arena permanently dirty.
    ///
    /// The reset moved from the drop (which synchronised the stream inline) to
    /// the next `begin_generation`, gated on a recorded event. The hazard in
    /// that move is a reset that never happens — the arena would grow overflow
    /// slabs forever. On the device-free stager there is no stream to fence
    /// against, so the drop must reset immediately rather than park a fence
    /// nothing will ever fire.
    #[test]
    fn deviceless_drop_resets_immediately_rather_than_parking_a_fence() {
        let stager = PinnedStager::noop();
        {
            let _g = stager.begin_generation();
            let mut inner = stager.inner.lock().unwrap();
            // Stand in for submitted bump allocations.
            inner.arena_dirty = true;
        }
        let inner = stager.inner.lock().unwrap();
        assert_eq!(inner.live_generations, 0, "guard released");
        assert!(
            !inner.arena_dirty,
            "deviceless drop must reset, not defer behind a fence that cannot fire",
        );
        assert!(
            inner.pending_reset.is_none(),
            "no fence should be parked when there is no stream",
        );
    }

    /// A generation still live must block the reset — the whole invariant the
    /// guard exists for. Nesting is what makes the deferred path safe: the
    /// inner drop sees a non-zero count and records nothing.
    #[test]
    fn nested_generation_keeps_arena_live() {
        let stager = PinnedStager::noop();
        let outer = stager.begin_generation();
        {
            let _inner_gen = stager.begin_generation();
            let mut inner = stager.inner.lock().unwrap();
            inner.arena_dirty = true;
        }
        {
            let inner = stager.inner.lock().unwrap();
            assert_eq!(inner.live_generations, 1, "outer guard still held");
            assert!(
                inner.arena_dirty,
                "arena reset while a generation was still live — submitted bump \
                 pointers would dangle",
            );
        }
        drop(outer);
        let inner = stager.inner.lock().unwrap();
        assert!(!inner.arena_dirty, "reset once the last guard released");
    }

    /// `begin_generation` bumps the epoch, so a consumer holding a device
    /// pointer from an earlier generation can detect that the arena underneath
    /// it was reset. Deferring the reset must not stall the epoch.
    #[test]
    fn epoch_advances_per_generation() {
        let stager = PinnedStager::noop();
        let a = stager.begin_generation().epoch();
        let b = stager.begin_generation().epoch();
        assert!(b > a, "epoch must advance so stale pointers are detectable");
    }
}

#[cfg(test)]
mod fallible_alloc_tests {
    use super::*;

    /// The staging fallback must return `Err` — never abort the process — when
    /// neither pinned host memory nor the plain host heap can supply the size.
    /// This is the whole point of `try_reserve_exact` over `vec![0u8; n]`, whose
    /// `handle_alloc_error` aborted a full overnight load.
    #[test]
    fn errs_on_impossible_size_instead_of_aborting() {
        let r = PinnedBuf::alloc_default_or_host_fallible(usize::MAX);
        assert!(r.is_err(), "impossible alloc must return Err, not abort");
    }

    /// A modest size still succeeds (pinned when a CUDA context is present, else
    /// the fallible host heap).
    #[test]
    fn small_alloc_succeeds() {
        let b = PinnedBuf::alloc_default_or_host_fallible(4096).expect("4 KiB staging");
        assert!(b.len() >= 4096);
    }
}
