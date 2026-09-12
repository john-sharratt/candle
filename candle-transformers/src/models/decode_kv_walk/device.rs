//! Reading raw device bytes for a capture.

use candle::cuda_backend::CudaDevice;
use candle::Result;
use cudarc::driver::CudaSlice;

/// Copy `len` bytes from device address `ptr` to the host.
///
/// For memory reachable only as an address — a header table entry, a slice
/// array, a record, a KV band — which is everything a paged kernel reads. The
/// view is forgotten rather than dropped: it borrows memory the capture does
/// not own, and dropping it would free someone else's allocation.
///
/// # Safety
///
/// `ptr` must name at least `len` readable bytes on `dev`.
pub(crate) unsafe fn read_device(dev: &CudaDevice, ptr: u64, len: usize) -> Result<Vec<u8>> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let stream = dev.cuda_stream();
    // SAFETY: the caller's contract — `len` readable bytes at `ptr`.
    let view: CudaSlice<u8> = unsafe { stream.upgrade_device_ptr::<u8>(ptr, len) };
    let host = dev.memcpy_dtov(&view);
    std::mem::forget(view);
    host.map_err(|e| candle::Error::Msg(format!("device read of {len} B at {ptr:#x}: {e}")))
}
