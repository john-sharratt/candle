//! Owns one pinned host slab that the GPU reads directly over PCIe, and hands
//! out bump-allocated pieces of it.
//!
//! The slab is mapped so kernels read it at its host address with no copy, and
//! an overflow slab is added during a wave when the first one fills. Freeing
//! pinned memory is refused on a thread that is recording a wave, so the wave is
//! paused around the free. The bump offset is reset only after a stream sync,
//! since a kernel may still be reading the bytes it covers.

use super::*;

/// A contiguous pinned slab allocated once at construction.
///
/// Allocated with `CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED`
/// so the GPU can read directly from host memory over PCIe — zero memcpy.
pub(super) struct PinnedArena {
    pub(super) ptr: *mut u8,
    /// Device-visible pointer into the same pinned memory (mapped).
    pub(super) dev_ptr: u64,
    pub(super) capacity: usize,
    /// Current bump offset. Reset to 0 on sync.
    pub(super) offset: usize,
}

// SAFETY: host-side pinned memory, not thread-specific.
unsafe impl Send for PinnedArena {}

impl PinnedArena {
    pub(super) fn new(capacity: usize) -> Result<Self> {
        let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        // CU_MEMHOSTALLOC_DEVICEMAP = 0x02
        // CU_MEMHOSTALLOC_WRITECOMBINED = 0x04
        // An overflow arena is added mid-generation, so possibly mid-wave: the
        // pinning runs with this thread's wave paused around it.
        outside_capture(|| unsafe { sys::cuMemHostAlloc(&mut host_ptr, capacity, 0x02 | 0x04) })?
            .result()
            .map_err(|e| {
                crate::Error::Msg(format!(
                    "cuMemHostAlloc for arena ({} MB) failed: {:?}",
                    capacity / (1024 * 1024),
                    e
                ))
            })?;
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
        crate::vram::note_host_pinned_alloc(crate::vram::PinnedUse::Staging, capacity as u64);
        Ok(Self {
            ptr: host_ptr as *mut u8,
            dev_ptr,
            capacity,
            offset: 0,
        })
    }

    /// Try to bump-allocate `len` bytes. Returns `None` if not enough space.
    pub(super) fn try_alloc(&mut self, len: usize) -> Option<PinnedBuf> {
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
    pub(super) fn reset(&mut self) {
        self.offset = 0;
    }

    pub(super) fn used(&self) -> usize {
        self.offset
    }
}

impl Drop for PinnedArena {
    fn drop(&mut self) {
        if self.capacity > 0 {
            free_host(self.ptr);
            crate::vram::note_host_pinned_free(
                crate::vram::PinnedUse::Staging,
                self.capacity as u64,
            );
        }
    }
}

/// Unpin and free `ptr`, from `cuMemHostAlloc`, with this thread's wave paused
/// around it: the driver refuses the free on a recording thread, and the
/// refusal ends the capture. A wave that cannot be paused is already broken,
/// and the free goes ahead rather than leak the pages.
pub(super) fn free_host(ptr: *mut u8) {
    let free = || unsafe { sys::cuMemFreeHost(ptr as *mut std::ffi::c_void) };
    if outside_capture(free).is_err() {
        let _ = free();
    }
}
