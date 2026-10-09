//! The host→device link's rate, measured once at startup: what read-ahead's
//! window is sized from (`read_ahead`).
//!
//! A copy from pinned host memory into VRAM on the copy stream, timed by the
//! host around a stream synchronize — the best of a few, after one to warm the
//! path. It measures the link the workers read remote experts over, on this
//! machine as it is: a PCIe 3.0 host halves it (`CLAUDE.md`, the 3090 box), and
//! a figure carried over from another machine would size the window twice too
//! wide there.

use candle::cuda_backend::CudaDevice;
use candle::Result;
use cudarc::driver::{sys, CudaStream, DevicePtr};
use std::time::Instant;

/// Bytes one timed copy moves: large enough that the per-copy overhead is
/// noise, small enough to pin on a machine whose page-lock budget is tight.
const PROBE_BYTES: usize = 16 << 20;

/// Timed copies; the fastest is the rate.
const PROBES: usize = 3;

/// Bytes per second from pinned host memory into VRAM on `stream`.
pub(crate) fn measure_link_rate(device: &CudaDevice, stream: &CudaStream) -> Result<f64> {
    copy_rate(device, stream, PROBE_BYTES, PROBES)
}

/// The fastest of `probes` copies of `bytes`, after one untimed.
fn copy_rate(device: &CudaDevice, stream: &CudaStream, bytes: usize, probes: usize) -> Result<f64> {
    let mut host: *mut std::ffi::c_void = std::ptr::null_mut();
    // SAFETY: a page-locked allocation of `bytes`, freed below on every path.
    let r = unsafe { sys::cuMemAllocHost_v2(&mut host, bytes) };
    if r != sys::CUresult::CUDA_SUCCESS {
        candle::bail!("link rate: {bytes} pinned bytes refused: {r:?}");
    }
    let measured = (|| -> Result<f64> {
        // SAFETY: written by every copy before it is read.
        let dst = unsafe { device.alloc::<u8>(bytes)? };
        let (dp, _g) = dst.device_ptr(stream);
        let copy = || -> Result<f64> {
            let t = Instant::now();
            // SAFETY: `host` holds `bytes` pinned bytes and `dp` `bytes` of VRAM;
            // the synchronize below ends the copy before either is reused.
            unsafe {
                let src = std::slice::from_raw_parts(host as *const u8, bytes);
                cudarc::driver::result::memcpy_htod_async(dp, src, stream.cu_stream())
                    .map_err(candle::Error::wrap)?;
            }
            stream.synchronize().map_err(candle::Error::wrap)?;
            Ok(t.elapsed().as_secs_f64())
        };
        copy()?;
        let mut best = f64::INFINITY;
        for _ in 0..probes {
            best = best.min(copy()?);
        }
        Ok(bytes as f64 / best)
    })();
    // SAFETY: allocated above; every copy from it has completed.
    unsafe {
        sys::cuMemFreeHost(host);
    }
    measured
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// The startup measurement returns a rate, and one no PCIe link could beat
    /// by an order of magnitude.
    #[test]
    fn the_link_rate_is_a_plausible_rate() {
        let Ok(Device::Cuda(device)) = Device::new_cuda(0) else {
            return;
        };
        let stream = device.cuda_context().new_stream().unwrap();
        let rate = measure_link_rate(&device, &stream).unwrap();
        assert!(rate > 1e8 && rate < 1e12, "{rate} B/s");
    }

    /// **The link, by copy size**, on the machine the test runs on — what the
    /// read-ahead window is sized from. Prints GB/s; asserts only that each
    /// copy completed.
    ///
    /// `cargo test --release --features cuda -p candle-transformers --lib expert_lre::link_rate::tests::link_rate_sweep -- --ignored --nocapture`
    #[test]
    #[ignore = "a measurement, ~1 s of copies; read its output"]
    fn link_rate_sweep() {
        let Ok(Device::Cuda(device)) = Device::new_cuda(0) else {
            panic!("the sweep needs a CUDA device");
        };
        let stream = device.cuda_context().new_stream().unwrap();
        for mib in [1usize, 4, 16, 64] {
            let rate = copy_rate(&device, &stream, mib << 20, 8).unwrap();
            eprintln!("  {mib:>3} MiB pinned → VRAM: {:.2} GB/s", rate / 1e9);
        }
    }
}
