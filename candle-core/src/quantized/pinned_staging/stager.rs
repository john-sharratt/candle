//! Allocates staging buffers and submits them to the GPU, choosing a bump slot,
//! an owned buffer or a resident copy by size and by use.
//!
//! Small staging buffers are bump-allocated from the mapped arena at near-zero
//! cost, and large ones get their own pinned buffer. A buffer that every block of
//! a kernel reads on its critical path goes to resident device memory instead,
//! because a zero-copy read pays a PCIe round trip on every access. The
//! zero-copy submit path makes no driver call. An arena is reset for reuse only
//! when no generation is live and its fence has fired, or after the stream has
//! been synchronised where the space is genuinely needed.

use super::*;

/// Staging allocator backed by CUDA pinned memory.
///
/// Small allocations (≤ 16 MB) are bump-allocated from a device-mapped
/// pinned arena for near-zero overhead. Large allocations get their own
/// pinned buffer. Overflow arenas are added during active generations.
///
/// Cloneable via `Arc<Mutex<..>>` — all clones share the same arena and queue.
#[derive(Clone)]
pub struct PinnedStager {
    pub(super) inner: Arc<Mutex<PinnedStagerInner>>,
}

impl std::fmt::Debug for PinnedStager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedStager").finish_non_exhaustive()
    }
}

pub(super) struct PinnedStagerInner {
    pub(super) dev: Option<CudaDevice>,
    /// When set, operations that need a stream (owned-buf H2D copy, sync on flush)
    /// use this explicit stream instead of `dev.cuda_stream()`.
    /// Used by the background quantizer thread which runs on a dedicated CUDA stream.
    pub(super) explicit_stream: Option<Arc<CudaStream>>,
    /// Pinned arenas for small buffers. The first is allocated at construction;
    /// overflow arenas are added when the current one fills during an active
    /// generation. On flush, all but the first are freed.
    /// Device-mapped: GPU reads directly over PCIe, zero memcpy.
    pub(super) arenas: Vec<PinnedArena>,
    /// Size in bytes for each arena slab (used when allocating overflow arenas).
    pub(super) arena_size: usize,
    /// Spent owned (large) pinned buffers awaiting cleanup.
    pub(super) pending_owned: Vec<PinnedBuf>,
    /// Cumulative byte size of pending owned buffers.
    pub(super) pending_owned_bytes: usize,
    /// Whether any bump allocations have been submitted since last reset.
    pub(super) arena_dirty: bool,
    /// Number of bump-allocated buffers that have been handed out but not yet
    /// submitted. The arena must NOT be reset while this is non-zero, because
    /// those buffers point into arena memory that would be reused.
    pub(super) bump_outstanding: usize,
    /// Number of live [`Generation`] guards. While > 0, the arenas must not
    /// be reset — submitted bump pointers are still potentially in use.
    pub(super) live_generations: usize,
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
    pub(super) pending_reset: Option<cudarc::driver::CudaEvent>,
    /// Monotonic counter bumped every time a fresh [`Generation`] begins. A
    /// generation's arena is reset (all bump pointers invalidated) once the last
    /// guard drops, so the epoch captured at `begin` uniquely identifies the
    /// arena's current fill. Consumers that cache a device pointer handed out by
    /// one generation compare epochs to detect a reset before reusing it.
    pub(super) epoch: u64,
    /// Where [`PinnedStager::submit_resident`] copies land: carved from
    /// `resident_slabs` and rewound with the arenas, since a resident copy is
    /// read by the same kernels as the pinned bytes it came from.
    pub(super) resident: RingPlan,
    pub(super) resident_slabs: Vec<Held<u8>>,
}

impl PinnedStagerInner {
    /// Reset the arenas and drop spent owned buffers. The caller must have
    /// established that no kernel can still be reading them.
    pub(super) fn reset_arenas_now(&mut self) {
        for a in self.arenas.iter_mut() {
            a.reset();
        }
        self.arenas.truncate(1);
        self.arena_dirty = false;
        self.pending_owned.clear();
        self.pending_owned_bytes = 0;
        self.pending_reset = None;
        if self.resident.reset() {
            // Retired rather than dropped: a reset can run on a thread that is
            // recording a wave, where a free is refused until its segment has
            // launched.
            for slab in self.resident_slabs.drain(..) {
                match &self.dev {
                    Some(dev) => dev.retire(slab),
                    None => drop(slab),
                }
            }
        }
    }

    /// Carve `buf.len()` device bytes from the resident slabs and enqueue the
    /// pinned → device copy into them. The copy is asynchronous (pinned
    /// source), stream-ordered ahead of every launch that follows; the bytes
    /// stay valid until the arenas next reset.
    pub(super) fn copy_to_device(&mut self, buf: &PinnedBuf) -> Result<u64> {
        let dev = self.dev.clone().expect("cuda stager missing device");
        // Leased for the whole copy, which may allocate a slab and queues on
        // the stream.
        let lease = dev.cuda_stream();
        let stream = self
            .explicit_stream
            .clone()
            .unwrap_or_else(|| (*lease).clone());
        let src = buf.as_slice();
        let mut _eager = None;
        let at = match self.resident.carve(buf.len()) {
            Carve::At(offset) => {
                let slab = self
                    .resident_slabs
                    .last()
                    .expect("a carve At names a held slab");
                let at = slab.device_ptr(&stream).0 + offset as u64;
                // **On the device's stream, a carve into a held slab cuts
                // nothing.** Recording, the bytes are staged through the wave's
                // ring and their copy recorded ahead of the launches that read
                // them; in a segment that runs eagerly, the copy is queued
                // behind the launches already issued. Pausing in either cut the
                // wave where the same wave without this copy is not cut, and
                // every segment after it held other layers.
                if self.explicit_stream.is_none() {
                    if dev.record_upload(at, src) {
                        return Ok(at);
                    }
                    if !dev.eager_segment_here() {
                        _eager = dev.pause_capture()?;
                    }
                } else {
                    _eager = dev.pause_capture()?;
                }
                at
            }
            Carve::NewSlab(width) => {
                // An allocation: eager, behind whatever this thread has
                // recorded.
                _eager = dev.pause_capture()?;
                // SAFETY: every byte a launch reads is written by the copy
                // below, into the range carved for it.
                let slab = unsafe { Held::<u8>::alloc_on(&stream, width)? };
                let base = slab.device_ptr(&stream).0;
                self.resident_slabs.push(slab);
                base
            }
        };
        if !src.is_empty() {
            // SAFETY: `at` names `src.len()` bytes of a held slab, carved for
            // this copy alone until the arenas reset; `src` is pinned and stays
            // untouched until the same reset.
            unsafe {
                sys::cuMemcpyHtoDAsync_v2(
                    at,
                    src.as_ptr() as *const std::ffi::c_void,
                    src.len(),
                    stream.cu_stream(),
                )
            }
            .result()
            .w()?;
        }
        drop(lease);
        Ok(at)
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
    pub(super) fn try_reclaim(&mut self) {
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
    pub(super) fn effective_arena_size() -> usize {
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
                resident: RingPlan::default(),
                resident_slabs: Vec::new(),
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
                resident: RingPlan::default(),
                resident_slabs: Vec::new(),
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
                resident: RingPlan::default(),
                resident_slabs: Vec::new(),
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
        //
        // Not on a thread recording a wave: querying an event there is host
        // protocol, and it invalidates the whole capture. The arena then keeps
        // bump-allocating — into an overflow slab if it fills — exactly as it
        // does while a generation is live, and the next generation begun
        // outside the recording collects the reset.
        let recording = inner
            .dev
            .as_ref()
            .is_some_and(|d| d.recording_segment().is_some());
        if !recording {
            inner.try_reclaim();
        }
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
                // Wider than the bump arena takes, so not device-mapped: the
                // kernel reads a device copy, owned by the handle because a
                // transfer this wide is rare and holding a slab its width for
                // every later generation would cost more than the allocation.
                let dev = inner.dev.clone().expect("checked above");
                let _eager = dev.pause_capture()?;
                // Leased for the allocation and the copy queued below.
                let lease = dev.cuda_stream();
                let stream = inner
                    .explicit_stream
                    .clone()
                    .unwrap_or_else(|| (*lease).clone());
                let mut gpu = unsafe { stream_alloc::<u8>(&stream, buf.len())? };
                stream.memcpy_htod(buf.as_slice(), &mut gpu).w()?;
                let dev_ptr = gpu.device_ptr(&stream).0;
                let len = buf.len();
                inner.pending_owned_bytes += len;
                inner.pending_owned.push(buf);
                Ok(GpuBuf {
                    dev_ptr,
                    len,
                    _owned: Some(OwnedDeviceBytes::new(gpu, dev.clone())),
                })
            }
            PinnedBuf::Host { .. } => Ok(GpuBuf {
                dev_ptr: 0,
                len: buf.len(),
                _owned: None,
            }),
        }
    }

    /// Return a handle to a **device-resident** copy of the staging buffer.
    ///
    /// One stream-ordered pinned → device copy into the stager's resident
    /// slabs ([`super::resident_ring`]). This is the right submit for data
    /// every block of a kernel reads on its critical path — a slot header, a
    /// per-row table — where the zero-copy mapping's PCIe round trip per read
    /// is the cost, not the bytes. The bookkeeping is the same as
    /// [`Self::submit`], and so is the lifetime: the copy is valid while the
    /// generation that made it is, and its bytes are reused once the
    /// generation's fence has fired — exactly when the pinned source's are.
    pub fn submit_resident(&self, buf: PinnedBuf) -> Result<GpuBuf> {
        let mut inner = self.inner.lock().unwrap();

        if inner.dev.is_none() || matches!(buf, PinnedBuf::Host { .. }) {
            return Ok(GpuBuf {
                dev_ptr: 0,
                len: buf.len(),
                _owned: None,
            });
        }

        let dev_ptr = inner.copy_to_device(&buf)?;
        let len = buf.len();
        // The resident copy rewinds with the arenas, so a generation that made
        // one must fence and reset like one that bumped.
        inner.arena_dirty = true;
        match &buf {
            PinnedBuf::Bump { .. } => {
                debug_assert!(inner.bump_outstanding > 0);
                inner.bump_outstanding -= 1;
            }
            PinnedBuf::Owned { .. } => {
                inner.pending_owned_bytes += len;
                inner.pending_owned.push(buf);
            }
            PinnedBuf::Host { .. } => unreachable!("handled above"),
        }
        Ok(GpuBuf {
            dev_ptr,
            len,
            _owned: None,
        })
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
    pub(super) fn sync_and_reset_all(&self) -> Result<()> {
        let (dev, explicit_stream) = {
            let inner = self.inner.lock().unwrap();
            (inner.dev.clone(), inner.explicit_stream.clone())
        };
        let _eager = match &dev {
            Some(dev) => dev.pause_capture()?,
            None => None,
        };
        if let Some(stream) = explicit_stream {
            stream.synchronize().w()?;
        } else if let Some(dev) = &dev {
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
    pub(super) fn sync_and_reset_arena(&self) -> Result<()> {
        let (dev, explicit_stream) = {
            let inner = self.inner.lock().unwrap();
            (inner.dev.clone(), inner.explicit_stream.clone())
        };
        let _eager = match &dev {
            Some(dev) => dev.pause_capture()?,
            None => None,
        };
        if let Some(stream) = explicit_stream {
            stream.synchronize().w()?;
        } else if let Some(dev) = &dev {
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
                let _eager = dev.pause_capture();
                let _ = dev.cuda_stream().synchronize();
            }
            self.pending_owned.clear();
        }
        // arenas are dropped automatically (each frees its slab)
    }
}
