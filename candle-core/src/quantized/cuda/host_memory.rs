//! Allocates and registers host memory a kernel can address over PCIe, and the
//! guards that free it.
//!
//! Pinned, device-mapped allocations are the only host memory a kernel may
//! dereference. Registering an existing mmap can succeed while its device pointer
//! is refused, and a kernel that then reads it faults on the first forward and
//! poisons the context, so every such buffer is allocated here. Each allocation
//! counts toward the process-wide pinned total, which the host-RAM budget reads as
//! non-pageable, and a failed allocation reports the size asked for and that
//! total. Guards free their memory on drop; on a recording thread the free waits
//! for the wave to pause. A failed registration returns `None`, and the caller
//! must degrade visibly rather than take the slower path silently.

use super::*;

// Host-Mapped / Pinned Memory Primitives

/// Guard for CUDA-registered mmap memory. Automatically unregisters on drop.
///
/// Created by [`register_mmap_cuda`]. Ensures `cuMemHostUnregister` is called
/// when the guard is dropped, even if the mmap outlives it.
pub struct MmapRegistration {
    pub(super) ptr: *mut std::ffi::c_void,
}

// Safety: the pointer is only used in Drop to unregister; the registration
// itself is thread-safe in the CUDA driver.
unsafe impl Send for MmapRegistration {}

unsafe impl Sync for MmapRegistration {}

impl Drop for MmapRegistration {
    fn drop(&mut self) {
        use cudarc::driver::sys;
        unsafe {
            let _ = sys::cuMemHostUnregister(self.ptr).result();
        }
    }
}

/// Guard for `cudaHostAlloc`-allocated memory. Calls `cudaFreeHost` on drop.
///
/// Created by [`alloc_host_mapped`]. The memory is GPU-accessible via PCIe
/// (allocated with `CU_MEMHOSTALLOC_DEVICEMAP`).
pub struct HostMappedAlloc {
    pub(super) host_ptr: *mut std::ffi::c_void,
    pub(super) size: usize,
}

// Safety: the pinned memory is accessible from any thread; the host pointer
// is stable for the lifetime of the allocation.
unsafe impl Send for HostMappedAlloc {}

unsafe impl Sync for HostMappedAlloc {}

impl Drop for HostMappedAlloc {
    fn drop(&mut self) {
        // Refused on a recording thread, ending its capture: freed with this
        // thread's wave paused, or directly when a wave cannot be paused.
        let free = || unsafe { cudarc::driver::sys::cuMemFreeHost(self.host_ptr) };
        if outside_capture(free).is_err() {
            let _ = free();
        }
        crate::vram::note_host_pinned_free(crate::vram::PinnedUse::HostMapped, self.size as u64);
    }
}

/// Register an mmap with CUDA for DMA-accelerated host-to-device transfers.
///
/// On success, returns a [`MmapRegistration`] guard that unregisters the memory
/// on drop. Returns `None` if registration fails (e.g. alignment issues, driver
/// limitations). Callers should fall back to regular `memcpy` in that case.
///
/// The mmap is registered as read-only and device-mapped, so the GPU can read
/// from it directly over PCIe.
pub fn register_mmap_cuda(mmap: &memmap2::Mmap) -> Option<MmapRegistration> {
    use cudarc::driver::sys;
    let ptr = mmap.as_ptr() as *mut std::ffi::c_void;
    let len = mmap.len();
    let register_result = unsafe {
        sys::cuMemHostRegister_v2(
            ptr,
            len,
            sys::CU_MEMHOSTREGISTER_DEVICEMAP | sys::CU_MEMHOSTREGISTER_READ_ONLY,
        )
    };
    match register_result.result() {
        Ok(_) => Some(MmapRegistration { ptr }),
        Err(e) => {
            // Do not swallow this. Callers degrade to a slower path when
            // registration fails, so a silent `None` turns a driver-level
            // refusal into an unexplained performance difference that looks
            // like a tuning problem.
            tracing::warn!(
                target: "candle_core::quantized::cuda",
                len,
                "cuMemHostRegister(DEVICEMAP|READ_ONLY) failed ({e:?}); \
                 falling back to staged host-to-device copies"
            );
            None
        }
    }
}

/// Allocate host memory that is GPU-accessible via PCIe (`cudaHostAllocMapped`).
///
/// Returns `(host_ptr, device_ptr, guard)`. The `guard` frees the memory on drop.
/// GPU kernels use `device_ptr` transparently — hardware handles PCIe transfers.
///
/// This is the building block for VRAM-overflow weight storage: tensors that
/// don't fit in VRAM can live in pinned host memory and still be used by CUDA
/// kernels (at PCIe bandwidth instead of VRAM bandwidth).
///
/// Allocating here is not interchangeable with registering memory you already
/// have. On WDDM, [`register_mmap_cuda`] succeeds on a multi-gigabyte model
/// mmap and `cuMemHostGetDevicePointer` then *refuses* that range, so a
/// registered mmap is not kernel-addressable however large it is. Forcing the
/// host pointer through anyway — on the theory that unified addressing makes it
/// equivalent — faults with `CUDA_ERROR_ILLEGAL_ADDRESS` on the first forward
/// and poisons the context. Anything a kernel must dereference from host memory
/// has to be copied into an allocation from this function.
pub fn alloc_host_mapped(size: usize) -> Result<(*mut u8, u64, HostMappedAlloc)> {
    use cudarc::driver::sys;
    let mut host_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
    unsafe {
        // CU_MEMHOSTALLOC_DEVICEMAP = 0x02
        // The size and the standing pinned total are in the message because an
        // out-of-memory without them is unactionable: pinned allocations fail
        // against what every *other* pinned claim in the process has already
        // taken, and the amount asked for is the only clue to which caller this
        // was. Pinning is refused on a recording thread, ending its capture,
        // so it runs with this thread's wave paused.
        outside_capture(|| sys::cuMemHostAlloc(&mut host_ptr, size, 0x02))?
            .result()
            .map_err(|e| {
                crate::Error::Msg(format!(
                    "cuMemHostAlloc failed for {:.1} MiB (pinned already held: {:.1} GiB): {:?}",
                    size as f64 / (1024.0 * 1024.0),
                    crate::vram::host_pinned_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                    e,
                ))
            })?;
        let mut dev_ptr: sys::CUdeviceptr = 0;
        let res = sys::cuMemHostGetDevicePointer_v2(&mut dev_ptr, host_ptr, 0);
        if let Err(e) = res.result() {
            let _ = outside_capture(|| sys::cuMemFreeHost(host_ptr));
            crate::bail!("cuMemHostGetDevicePointer failed: {:?}", e);
        }
        // Non-pageable by construction, so the host-RAM budget must count it as
        // structural — tracked here rather than at each call site, so no caller
        // can allocate gigabytes of pinned RAM the budget reads as pageable.
        crate::vram::note_host_pinned_alloc(crate::vram::PinnedUse::HostMapped, size as u64);
        let guard = HostMappedAlloc { host_ptr, size };
        Ok((host_ptr as *mut u8, dev_ptr, guard))
    }
}
