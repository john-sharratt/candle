//! DRAM shared between the pool and the link: `B_c` in the split
//! (`docs/cpu_expert_compute.md` §3.2, §5.3).
//!
//! A warm expert the GPU copies is a DMA read of the same DRAM the pool reads,
//! so the pool's rate while the copy engine streams (`B_c`) is not its rate
//! alone (`B_d`). How close the two together come to `B_d` is measured, not
//! assumed: the split's model is `B_c ≈ B_d − B_l`, the optimistic case.

use std::ffi::c_void;
use std::ptr::null_mut;
use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use candle::cuda_backend::CudaDevice;
use candle::{bail, Error, Result};
use cudarc::driver::result::memcpy_htod_async;
use cudarc::driver::{sys, CudaStream, DevicePtr};

/// Bytes one copy moves: an expert section's worth of slices, well past the
/// per-copy overhead.
const COPY_BYTES: usize = 64 << 20;

/// Copies pinned host memory into VRAM on `stream`, back to back, until `stop`
/// is set; returns the bytes moved and the seconds it took. Binds the device's
/// context to the calling thread first, so it may run on a thread of its own.
pub fn stream_copies(
    device: &CudaDevice,
    stream: &CudaStream,
    stop: &AtomicBool,
) -> Result<(u64, f64)> {
    device
        .cuda_context()
        .bind_to_thread()
        .map_err(Error::wrap)?;
    let mut host: *mut c_void = null_mut();
    // SAFETY: a page-locked allocation of COPY_BYTES, freed below on every path.
    let r = unsafe { sys::cuMemAllocHost_v2(&mut host, COPY_BYTES) };
    if r != sys::CUresult::CUDA_SUCCESS {
        bail!("shared dram: {COPY_BYTES} pinned bytes refused: {r:?}");
    }
    let moved = (|| -> Result<(u64, f64)> {
        // SAFETY: written by every copy before it is read.
        let dst = unsafe { device.alloc::<u8>(COPY_BYTES)? };
        let (dp, _g) = dst.device_ptr(stream);
        // SAFETY: `host` holds COPY_BYTES pinned bytes and `dp` COPY_BYTES of
        // VRAM; each copy is synchronised before the next is issued.
        let src = unsafe { slice::from_raw_parts(host as *const u8, COPY_BYTES) };
        let t0 = Instant::now();
        let mut bytes = 0u64;
        while !stop.load(Ordering::Relaxed) {
            unsafe {
                memcpy_htod_async(dp, src, stream.cu_stream()).map_err(Error::wrap)?;
            }
            stream.synchronize().map_err(Error::wrap)?;
            bytes += COPY_BYTES as u64;
        }
        Ok((bytes, t0.elapsed().as_secs_f64()))
    })();
    // SAFETY: allocated above; every copy from it has completed.
    unsafe {
        sys::cuMemFreeHost(host);
    }
    moved
}

/// The copy engine's rate alone over `window`.
pub fn copy_rate_alone(device: &CudaDevice, stream: &CudaStream, window: Duration) -> Result<f64> {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let timer = thread::spawn(move || {
        thread::sleep(window);
        flag.store(true, Ordering::Relaxed);
    });
    let (bytes, secs) = stream_copies(device, stream, &stop)?;
    timer.join().expect("timer thread panicked");
    Ok(bytes as f64 / secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::expert_lre::cpu_pool::bandwidth::{read_rate, touched_buffer};
    use crate::models::expert_lre::cpu_pool::topology::{performance_cores, physical_cores};
    use candle::Device;
    use std::process::Command;

    /// Measurements 1–2 together (`docs/cpu_expert_compute.md` §8): the pool's
    /// read rate alone (`B_d`), the copy engine's alone (`B_l`), and both at
    /// once (`B_c` and the link under contention), for the pool layouts that
    /// matter. Reads the live PCIe generation and width while the copies run.
    ///
    /// `cargo test --release -p candle-transformers --features cuda --lib shared_dram_sweep -- --ignored --nocapture`
    #[test]
    #[ignore = "a measurement, ~20 s; read its output"]
    fn shared_dram_sweep() {
        let Ok(Device::Cuda(device)) = Device::new_cuda(0) else {
            panic!("the sweep needs a CUDA device");
        };
        let stream = device.cuda_context().new_stream().unwrap();
        let window = Duration::from_millis(500);
        let cores = physical_cores().unwrap();
        let perf = performance_cores(&cores);
        let buf = touched_buffer(2 << 30);

        let link = copy_rate_alone(&device, &stream, window).unwrap();
        println!("link alone: {:.2} GB/s", link / 1e9);

        let mut layouts: Vec<(String, Vec<u32>)> = [4usize, perf.len()]
            .into_iter()
            .filter(|&n| n <= perf.len())
            .map(|n| {
                (
                    format!("{n} P"),
                    perf[..n].iter().map(|c| c.primary()).collect(),
                )
            })
            .collect();
        layouts.push((
            "all physical".into(),
            cores.iter().map(|c| c.primary()).collect(),
        ));
        layouts.dedup_by(|a, b| a.1 == b.1);

        for (name, sets) in layouts {
            let alone = read_rate(buf.clone(), &sets, window, 3).unwrap().gb_per_s();
            let stop = Arc::new(AtomicBool::new(false));
            let (dev, st, flag) = (device.clone(), stream.clone(), stop.clone());
            let copier = thread::spawn(move || stream_copies(&dev, &st, &flag));
            thread::sleep(Duration::from_millis(50));
            let smi = Command::new("nvidia-smi")
                .args([
                    "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
                    "--format=csv,noheader",
                ])
                .output()
                .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                .unwrap_or_else(|e| format!("nvidia-smi: {e}"));
            let shared = read_rate(buf.clone(), &sets, window, 3).unwrap().gb_per_s();
            stop.store(true, Ordering::Relaxed);
            let (bytes, secs) = copier.join().expect("copier panicked").unwrap();
            let link_shared = bytes as f64 / secs / 1e9;
            println!(
                "{name:>13}: B_d {alone:6.1}  B_c {shared:6.1}  link under load {link_shared:5.2}  \
                 sum {:6.1} GB/s  (link gen,width: {smi})",
                shared + link_shared
            );
        }
    }
}
