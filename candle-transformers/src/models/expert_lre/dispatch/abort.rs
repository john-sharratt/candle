//! The mapped word that tells every waiting worker to give up on a cold expert that
//! will never arrive.
//!
//! It is raised when a cold expert can never be published, such as a failed pack
//! read or a dead stager or pipeline thread. Waiting workers then give their expert
//! up and claim the fault word, and the host fails the forward once it has
//! synchronised, so no token is returned from a layer that could not be computed.
//! The word lives in mapped pinned memory, so raising it is a plain host store and
//! needs no driver call.

use super::*;

/// The word every waiting worker polls, and the means to raise it.
///
/// Raised when a cold expert can never be published — a failed pack read, a
/// stager or pipeline thread that died. Every waiting worker then gives its
/// expert up and claims the fault word (`fault`), the host fails the forward
/// once it has synchronised, and no token computed from the layer is returned.
///
/// The word lives in mapped pinned memory and is raised with a plain host
/// store: raising it must not need the driver.
pub(crate) struct AbortWord {
    pub(super) host: *mut u32,
    pub(super) dev: u64,
}

// SAFETY: a mapped word written only by `raise` (a volatile store of 1) and
// read by the device and by `is_raised`.
unsafe impl Send for AbortWord {}

unsafe impl Sync for AbortWord {}

impl AbortWord {
    pub(crate) fn new() -> Result<Self> {
        let mut raw: *mut c_void = null_mut();
        // SAFETY: a page-locked, device-mapped allocation, freed in `drop`.
        let r = unsafe { sys::cuMemHostAlloc(&mut raw, 4, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            candle::bail!("expert cache: mapped abort word allocation failed: {r:?}");
        }
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe {
                sys::cuMemFreeHost(raw);
            }
            candle::bail!("expert cache: abort word has no device address: {r:?}");
        }
        // SAFETY: four bytes just allocated.
        unsafe { write_volatile(raw as *mut u32, 0) };
        Ok(Self {
            host: raw as *mut u32,
            dev,
        })
    }

    /// The word's device address, for `MoeLive::abort`.
    pub(crate) fn ptr(&self) -> u64 {
        self.dev
    }

    /// End every wait, now and later. Idempotent.
    pub(crate) fn raise(&self) {
        // SAFETY: the mapped word this struct owns.
        unsafe { write_volatile(self.host, 1) };
        fence(Ordering::SeqCst);
    }

    pub(crate) fn is_raised(&self) -> bool {
        // SAFETY: the mapped word this struct owns.
        unsafe { read_volatile(self.host) != 0 }
    }
}

impl Drop for AbortWord {
    fn drop(&mut self) {
        // SAFETY: allocated by `cuMemHostAlloc` in `new`; every reader is
        // joined or synchronized before the cache drops.
        unsafe {
            sys::cuMemFreeHost(self.host as *mut c_void);
        }
    }
}
