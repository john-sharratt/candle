//! Host uploads recorded into a wave capture.
//!
//! A host-to-device copy cannot sit in a recorded segment as a copy: the node
//! would read the caller's buffer when the graph runs, long after it is gone.
//! Ending the segment to run the copy eagerly is correct but costs a graph
//! launch per upload, and a forward uploads small per-wave tables at nearly
//! every layer. So while a wave records, an upload copies its bytes into a
//! pinned, device-mapped ring and records a kernel that copies them from there
//! to the destination — the bytes stay put until that kernel has run.
//!
//! Two rings alternate by wave. Each is fenced by an event recorded when its
//! wave finishes, and is not reused until that event has fired, so a wave's
//! staged bytes outlive every replay that reads them.

use crate::cuda_backend::WrapErr;
use crate::Result;
use candle_kernels::simple::fill::run_copy2d_op;
use cudarc::driver::{sys, CudaEvent, CudaStream};
use std::ffi::c_void;
use std::sync::Arc;

/// Bytes in one ring. A forward's per-wave tables are a few hundred KiB at
/// the widest waves the engine runs; an upload that does not fit is run
/// eagerly instead, so the size bounds only how often that happens.
pub(super) const RING_BYTES: usize = 8 << 20;

/// Every staged upload starts on this boundary, so a copy kernel never reads
/// an unaligned table and two uploads never share a cache line.
const ALIGN: usize = 256;

/// `copy2d`'s dtype code for bytes.
const COPY_U8: i32 = 5;

/// One pinned, device-mapped ring of staged upload bytes.
pub(super) struct StagingRing {
    host: *mut u8,
    device: u64,
    used: usize,
    /// Recorded on the compute stream when the wave that last used this ring
    /// finished; the ring is not written again until it has fired.
    fence: Option<CudaEvent>,
}

// SAFETY: the ring's host bytes are written only by the thread recording the
// wave that holds it, under the hub's lock; the pointers are plain addresses.
unsafe impl Send for StagingRing {}

impl StagingRing {
    pub(super) fn new(stream: &Arc<CudaStream>) -> Result<Self> {
        stream.context().bind_to_thread().w()?;
        let mut host: *mut c_void = std::ptr::null_mut();
        // SAFETY: a fresh pinned allocation, mapped into the device's address
        // space; freed in `Drop`.
        unsafe {
            sys::cuMemHostAlloc(
                &mut host,
                RING_BYTES,
                sys::CU_MEMHOSTALLOC_DEVICEMAP | sys::CU_MEMHOSTALLOC_PORTABLE,
            )
            .result()
            .w()?;
        }
        let mut device: sys::CUdeviceptr = 0;
        // SAFETY: `host` was mapped above.
        let mapped = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut device, host, 0).result() };
        if let Err(e) = mapped {
            // SAFETY: allocated above and not yet handed out.
            unsafe { sys::cuMemFreeHost(host) };
            return Err(e).w();
        }
        Ok(Self {
            host: host as *mut u8,
            device,
            used: 0,
            fence: None,
        })
    }

    /// Make the ring empty for a new wave, waiting out the wave that used it
    /// last. That wave finished long ago in every steady state — its logits
    /// were read back before this one was assembled — so the wait is a check.
    pub(super) fn reopen(&mut self) -> Result<()> {
        if let Some(fence) = self.fence.take() {
            fence.synchronize().w()?;
        }
        self.used = 0;
        Ok(())
    }

    /// Fence the ring behind everything the wave issued on `compute`.
    pub(super) fn close(&mut self, compute: &Arc<CudaStream>) -> Result<()> {
        self.fence = Some(compute.record_event(None).w()?);
        Ok(())
    }

    /// Stage `src` and record its copy to `dst` on `capture`. Returns `false`,
    /// having done nothing, when the ring has no room for it.
    pub(super) fn record(&mut self, capture: &Arc<CudaStream>, dst: u64, src: &[u8]) -> bool {
        let n = src.len();
        let at = self.used.next_multiple_of(ALIGN);
        if n == 0 || n > u32::MAX as usize || at + n > RING_BYTES {
            return n == 0;
        }
        // SAFETY: `[at, at + n)` lies inside the ring, and no recorded launch
        // reads it yet: the ring was reopened empty for this wave and `used`
        // only grows within it.
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.host.add(at), n) };
        // SAFETY: the source is the ring's device mapping of the bytes just
        // written and `dst` holds `n` bytes the caller owns; one row of `n`
        // bytes, launched on the capture stream so it is recorded.
        unsafe {
            run_copy2d_op(
                COPY_U8,
                (self.device + at as u64) as *const c_void,
                dst as *mut c_void,
                1,
                n as u32,
                n as u32,
                n as u32,
                capture.cu_stream() as *mut c_void,
            );
        }
        self.used = at + n;
        true
    }
}

impl Drop for StagingRing {
    fn drop(&mut self) {
        if let Some(fence) = self.fence.take() {
            let _ = fence.synchronize();
        }
        // SAFETY: allocated by `new`, freed once.
        unsafe { sys::cuMemFreeHost(self.host as *mut c_void) };
    }
}
