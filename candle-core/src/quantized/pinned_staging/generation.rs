//! Keeps a generation's pinned arena alive while kernels may read it, and resets
//! the arena only once the GPU has finished with it.
//!
//! While any generation guard lives, the arena is not reset, so the device
//! pointers handed out stay valid. When the last guard drops, an event is
//! recorded instead of synchronising the stream from a destructor, and the reset
//! waits for that event at the next generation. The event is recorded under the
//! lock that publishes it, so a stale event cannot overwrite a newer one and
//! release memory that a later generation's kernels still read.

use super::*;

/// RAII generation guard.
///
/// While at least one `Generation` is alive, the arena will never be reset —
/// all submitted bump pointers remain valid. When the last generation drops
/// over a dirty arena, a stream event is recorded and the reset is deferred to
/// the next generation, which resets only once that fence has fired.
///
/// Create via [`PinnedStager::begin_generation`].
pub struct Generation {
    pub(super) inner: Arc<Mutex<PinnedStagerInner>>,
    /// The stager epoch captured when this generation began. Device pointers it
    /// hands out are valid only while the stager's epoch still equals this.
    pub(super) epoch: u64,
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
                // The fence must follow every kernel that read the arena, and
                // inside a wave capture those are recorded but not yet issued:
                // issue them first.
                let dev = inner.dev.clone();
                let _eager = dev.as_ref().and_then(|d| d.pause_capture().ok().flatten());
                // Leased for as long as the fence below is being recorded.
                let lease = dev.as_ref().map(|d| d.cuda_stream());
                let stream = match (inner.explicit_stream.clone(), &lease) {
                    (Some(stream), _) => Some(stream),
                    (None, Some(lease)) => Some((**lease).clone()),
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

    /// Return a handle to a device-resident copy of the staging buffer.
    ///
    /// Delegates to the underlying [`PinnedStager::submit_resident`].
    pub fn submit_resident(&self, buf: PinnedBuf) -> crate::Result<GpuBuf> {
        let stager = PinnedStager {
            inner: Arc::clone(&self.inner),
        };
        stager.submit_resident(buf)
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
///
/// **A clone is another handle on the same device bytes**, whichever kind it
/// is. An owned buffer's slice is shared, not copied: cloning one used to clone
/// the `CudaSlice` — a device allocation and a copy per clone, issued on the
/// hot path by every drafter that hands its step a header buffer, and refused
/// outright inside a wave capture — while keeping the original's `dev_ptr`, so
/// the copy was never even read and the original it pointed at was not held.
#[derive(Clone)]
pub struct GpuBuf {
    pub(super) dev_ptr: u64,
    pub(super) len: usize,
    /// Holds the `CudaSlice` alive for owned (non-arena) buffers until the last
    /// handle drops. `None` for arena buffers — the arena owns the device
    /// memory.
    pub(super) _owned: Option<Arc<OwnedDeviceBytes>>,
}

/// An owned buffer's device bytes, retired to their device when the last
/// [`GpuBuf`] naming them drops.
pub(super) struct OwnedDeviceBytes {
    pub(super) slice: Option<CudaSlice<u8>>,
    pub(super) dev: CudaDevice,
}

impl Drop for OwnedDeviceBytes {
    fn drop(&mut self) {
        // A launch this thread recorded may still read the buffer.
        if let Some(slice) = self.slice.take() {
            self.dev.retire(slice);
        }
    }
}

impl OwnedDeviceBytes {
    pub(super) fn new(slice: CudaSlice<u8>, dev: CudaDevice) -> Arc<Self> {
        Arc::new(Self {
            slice: Some(slice),
            dev,
        })
    }
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
        // Reading the address issues nothing on the stream: no lease.
        let stream = dev.stream_unleased();
        let ptr = {
            let (ptr, _guard) = slice.device_ptr(stream);
            ptr
        }; // guard dropped; pointer remains valid while slice lives
        let len = slice.len();
        Self {
            dev_ptr: ptr,
            len,
            _owned: Some(OwnedDeviceBytes::new(slice, dev.clone())),
        }
    }
}
