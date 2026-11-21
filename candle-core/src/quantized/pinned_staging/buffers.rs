//! Provides the staging buffers the engine copies to and from the GPU, pinned or
//! plain as each buffer's reader requires.
//!
//! A buffer is either a slice of the bump arena, which frees nothing on drop, or
//! an owned pinned allocation. The memory kind is chosen by the reader:
//! write-combined memory is fast for CPU writes and slow for CPU reads and for
//! kernel I/O, so a destination of kernel I/O gets plain pinned memory. Staging
//! allocation that cannot be satisfied returns an error rather than aborting the
//! process. Write-combined buffers are recycled through a capped pool, because
//! pinning pages is expensive and the pool is host-wide, and a buffer returns to
//! it only after the copies reading it have finished.

use super::*;

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

    /// Plain pinned memory, as [`Self::alloc_owned_default`], that every CUDA
    /// context treats as pinned — `CU_MEMHOSTALLOC_PORTABLE` (0x01). For a
    /// buffer held process-wide and copied into from whichever device a caller
    /// is on: without the flag, a copy from a second device's context reads it
    /// as pageable memory and the transfer is staged through the driver.
    pub fn alloc_owned_portable(len: usize) -> Result<Self> {
        Self::alloc_owned_with_flags(len, 0x01)
    }

    /// Write-combined pinned memory a kernel reads at its host address —
    /// `CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED` (0x06). For a
    /// buffer the CPU only writes and a kernel only reads, pulled over PCIe by
    /// the kernel itself rather than copied by the copy engine.
    pub fn alloc_owned_mapped(len: usize) -> Result<Self> {
        Self::alloc_owned_with_flags(len, 0x06)
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

    pub(super) fn alloc_owned_with_flags(len: usize, flags: u32) -> Result<Self> {
        if len == 0 {
            return Ok(Self::Bump {
                ptr: std::ptr::NonNull::dangling().as_ptr(),
                len: 0,
            });
        }
        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // Pinning is refused on a recording thread, and the refusal ends its
        // capture: a staging buffer taken mid-wave pauses the wave around it.
        outside_capture(|| unsafe { sys::cuMemHostAlloc(&mut host_ptr, len, flags) })?
            .result()
            .map_err(|e| crate::Error::Msg(format!("cuMemHostAlloc failed: {:?}", e)))?;
        crate::vram::note_host_pinned_alloc(crate::vram::PinnedUse::Staging, len as u64);
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

    pub(super) fn ptr(&self) -> *mut u8 {
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

/// Most bytes the recycler keeps pinned while idle.
///
/// Pinned pages are a scarce host-wide resource, so an unbounded pool would
/// trade a latency problem for a footprint one. At this cap the pool holds the
/// working set of a wide wave's slot-state staging (a few KiB per buffer, two per
/// resident `(layer, slot)`) and hands the surplus back to the driver.
pub(super) const RECYCLER_MAX_BYTES: usize = 64 * 1024 * 1024;

/// Idle write-combined buffers, keyed by exact byte length.
///
/// Exact length rather than a range because every caller sizes through
/// `class_bytes_for`, so the keys are already a short doubling ladder and an
/// exact match is both a hit and a perfect fit.
pub(super) struct RecyclerState {
    pub(super) by_len: std::collections::HashMap<usize, Vec<PinnedBuf>>,
    pub(super) bytes: usize,
}

impl RecyclerState {
    /// An idle buffer of exactly `len` bytes, if one is held.
    pub(super) fn take(&mut self, len: usize) -> Option<PinnedBuf> {
        let bufs = self.by_len.get_mut(&len)?;
        let buf = bufs.pop()?;
        self.bytes -= len;
        Some(buf)
    }

    /// Keep `buf` for reuse, or return it (to be dropped) when that would put
    /// the pool over [`RECYCLER_MAX_BYTES`].
    pub(super) fn give(&mut self, buf: PinnedBuf) -> Option<PinnedBuf> {
        let len = buf.len();
        if len == 0 || self.bytes + len > RECYCLER_MAX_BYTES {
            return Some(buf);
        }
        self.bytes += len;
        self.by_len.entry(len).or_default().push(buf);
        None
    }

    /// Every idle buffer, emptying the pool.
    pub(super) fn drain(&mut self) -> Vec<PinnedBuf> {
        self.bytes = 0;
        self.by_len.drain().flat_map(|(_, bufs)| bufs).collect()
    }
}

pub(super) fn recycler() -> &'static std::sync::Mutex<RecyclerState> {
    static POOL: std::sync::OnceLock<std::sync::Mutex<RecyclerState>> = std::sync::OnceLock::new();
    POOL.get_or_init(|| {
        std::sync::Mutex::new(RecyclerState {
            by_len: std::collections::HashMap::new(),
            bytes: 0,
        })
    })
}

/// A write-combined pinned buffer of `len` bytes, recycled when one is idle.
///
/// **`cuMemHostAlloc` is not cheap and it is not amortised by holding the buffer.**
/// It pins pages, and measured on the Flash-Next prefill gate it costs ~0.98 ms —
/// which was **97.6%** of a KV slot-state rebuild and ~9% of prefill wall at 16
/// slots. It looked amortised because each buffer is allocated once, but there are
/// two per `(layer, slot)` and a prefill only rebuilds each a couple of times, so
/// in practice almost every rebuild paid a first touch: 416 allocations at 16
/// slots, 417 ms. Recycling turns that into two allocations and 414 pool hits.
///
/// Pairs with [`give_recycled_wc`], which is the only sanctioned way back. The
/// pool is **write-combined only** — `alloc_owned`'s flags — because WC memory is
/// fast for CPU writes and slow for CPU reads, so handing one to a reader would be
/// a silent performance fault rather than an error.
pub fn take_recycled_wc(len: usize) -> Result<PinnedBuf> {
    if len > 0 {
        if let Ok(mut pool) = recycler().lock() {
            if let Some(buf) = pool.take(len) {
                return Ok(buf);
            }
        }
    }
    PinnedBuf::alloc_owned(len)
}

/// Hand a buffer from [`take_recycled_wc`] back for reuse.
///
/// **The caller guarantees no copy is still reading it.** The pool hands the same
/// pages to the next taker, so returning a buffer with a `memcpy_htod_async` in
/// flight out of it would let that copy read another owner's bytes — silently,
/// since both are valid pinned memory. Return it only after the fence that
/// retires its copies.
///
/// Only [`PinnedBuf::Owned`] is kept: a `Bump` belongs to a stager arena, and a
/// `Host` vector is cheap to reallocate and is not pinned.
pub fn give_recycled_wc(buf: PinnedBuf) {
    if !matches!(buf, PinnedBuf::Owned { .. }) {
        return;
    }
    if let Ok(mut pool) = recycler().lock() {
        // A refused buffer comes back and drops here, freeing its pages.
        let _ = pool.give(buf);
    }
}

/// Free every idle buffer the recycler holds, unpinning its pages.
///
/// The pool is process-wide, so it outlives the engine whose slot states filled
/// it: without this, a shut-down engine leaves up to [`RECYCLER_MAX_BYTES`] of
/// host-wide pinned pages behind for a taker that may never come. Called once
/// the engine's sequences — every `(layer, slot)` staging pair — have handed
/// their buffers back. A buffer taken out at the time is untouched and returns
/// to the pool as usual.
pub fn release_recycled_wc() {
    // Drop outside the lock: each drop is a `cuMemFreeHost`.
    let idle = match recycler().lock() {
        Ok(mut pool) => pool.drain(),
        Err(_) => return,
    };
    drop(idle);
}

/// Idle buffers held, and their total bytes.
pub fn recycled_stats() -> (usize, usize) {
    match recycler().lock() {
        Ok(pool) => (pool.by_len.values().map(|v| v.len()).sum(), pool.bytes),
        Err(_) => (0, 0),
    }
}

impl Drop for PinnedBuf {
    fn drop(&mut self) {
        if let Self::Owned { ptr, len } = self {
            if *len > 0 {
                free_host(*ptr);
                crate::vram::note_host_pinned_free(crate::vram::PinnedUse::Staging, *len as u64);
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
