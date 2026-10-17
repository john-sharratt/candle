//! The device↔pool handoff's round trip — `t_h` in the split
//! (`docs/cpu_expert_compute.md` §5.9, §8 measurement 0).
//!
//! Mapped words on separate cache lines: the device rings `doorbell` and waits
//! for `done`; a pinned host thread waits for the doorbell and answers on
//! `done`. Iteration *i + 1*'s doorbell can only land after the host answered
//! *i*, so the time between the host seeing two consecutive doorbells is one
//! full round trip — a host store reaching the device, the device's wait
//! noticing it, the device's store reaching the host, and the host noticing
//! that.
//!
//! The device waits one of two ways ([`GpuWait`]): a chain of
//! `cuStreamWriteValue32` / `cuStreamWaitValue32` pairs, which waits in the
//! GPU's front end without holding an SM; or one thread of a kernel polling the
//! mapped word, the shape the dispatch's worker blocks already use.
//!
//! The host waits either by spinning on `pause`, or — where the CPU has WAITPKG —
//! by `umonitor`/`umwait` on the doorbell's line, which idles the core until the
//! line is written. Whether a write arriving over PCIe wakes `umwait` on a given
//! part is exactly what this measures: if it does not, the wait ends only at
//! its TSC deadline, and the round trip is the deadline, not microseconds.

#[cfg(target_arch = "x86_64")]
use std::arch::asm;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__cpuid_count, _rdtsc};
use std::ffi::c_void;
use std::hint::spin_loop;
use std::ptr::{self, null_mut};
use std::sync::atomic::{fence, Ordering};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

use candle::cuda_backend::CudaDevice;
use candle::{bail, Error, Result};
use candle_kernels::simple::handoff_ping::run_handoff_ping;
use cudarc::driver::sys;

/// Bytes between the words: one cache line each, so the host's store to
/// `done` never disturbs the line it monitors.
const LINE: usize = 64;

/// The mapped words, in line order.
const DOORBELL: usize = 0;
const DONE: usize = 1;
const GAVE_UP: usize = 2;
const WORDS: usize = 3;

/// How long the kernel's poll waits for one answer before it gives up.
const KERNEL_LIMIT_NS: u64 = 1_000_000_000;

/// How the device waits for the host's answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuWait {
    /// `cuStreamWaitValue32`, in the front end.
    StreamMemOp,
    /// One kernel thread polling the mapped word.
    KernelPoll,
}

/// How the host thread waits for the doorbell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HostWait {
    /// `pause` in a loop.
    Spin,
    /// `umonitor` + `umwait` in C0.1, re-armed every `deadline_cycles` TSC
    /// cycles (the OS caps a single wait regardless).
    Umwait { deadline_cycles: u64 },
}

/// Whether this CPU has WAITPKG (`umonitor`/`umwait`/`tpause`). Asked once:
/// every wait loop asks it each pass, and `cpuid` serializes the core — and
/// under a hypervisor, which Windows' virtualization-based security puts
/// beneath every process, traps to it for a microsecond or more.
pub fn has_waitpkg() -> bool {
    static WAITPKG: OnceLock<bool> = OnceLock::new();
    *WAITPKG.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            // CPUID leaf 7 subleaf 0 exists on every x86-64 part that runs
            // this code; WAITPKG is ECX bit 5.
            let leaf7 = __cpuid_count(7, 0);
            leaf7.ecx & (1 << 5) != 0
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// The mapped words, one per cache line.
struct Words {
    host: *mut u8,
    dev: sys::CUdeviceptr,
}

impl Words {
    fn new() -> Result<Self> {
        let mut raw: *mut c_void = null_mut();
        // SAFETY: a page-locked, device-mapped allocation, freed in `drop`.
        let r =
            unsafe { sys::cuMemHostAlloc(&mut raw, WORDS * LINE, sys::CU_MEMHOSTALLOC_DEVICEMAP) };
        if r != sys::CUresult::CUDA_SUCCESS {
            bail!("handoff: mapped words refused: {r:?}");
        }
        let mut dev: sys::CUdeviceptr = 0;
        // SAFETY: `raw` was allocated with DEVICEMAP just above.
        let r = unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dev, raw, 0) };
        if r != sys::CUresult::CUDA_SUCCESS {
            // SAFETY: allocated just above and never handed out.
            unsafe { sys::cuMemFreeHost(raw) };
            bail!("handoff: mapped words have no device address: {r:?}");
        }
        // SAFETY: WORDS * LINE bytes just allocated.
        unsafe { ptr::write_bytes(raw as *mut u8, 0, WORDS * LINE) };
        Ok(Self {
            host: raw as *mut u8,
            dev,
        })
    }

    /// Word `w`'s host address.
    fn host(&self, w: usize) -> *mut u32 {
        // SAFETY: inside the WORDS * LINE allocation.
        unsafe { self.host.add(w * LINE) as *mut u32 }
    }

    /// Word `w`'s device address.
    fn dev(&self, w: usize) -> sys::CUdeviceptr {
        self.dev + (w * LINE) as u64
    }

    fn doorbell(&self) -> *mut u32 {
        self.host(DOORBELL)
    }

    fn done(&self) -> *mut u32 {
        self.host(DONE)
    }
}

impl Drop for Words {
    fn drop(&mut self) {
        // SAFETY: allocated in `new`; the caller synchronised the stream first.
        unsafe { sys::cuMemFreeHost(self.host as *mut c_void) };
    }
}

/// Round trips, in nanoseconds, sorted ascending.
#[derive(Debug, Clone)]
pub struct RoundTrips(pub Vec<f64>);

impl RoundTrips {
    pub fn quantile(&self, q: f64) -> f64 {
        let i = ((self.0.len() - 1) as f64 * q).round() as usize;
        self.0[i]
    }
}

/// Write/wait pairs enqueued between flushes of the stream.
const FLUSH_EVERY: u32 = 32;

/// Runs `iterations` handoffs on a stream of `device`'s, with the device
/// waiting by `gpu` and the calling thread as the host side waiting by `host`
/// (pin it first). Returns the round trips.
pub fn round_trips(
    device: &CudaDevice,
    iterations: u32,
    gpu: GpuWait,
    host: HostWait,
) -> Result<RoundTrips> {
    device
        .cuda_context()
        .bind_to_thread()
        .map_err(Error::wrap)?;
    let stream = device.cuda_context().new_stream().map_err(Error::wrap)?;
    let words = Words::new()?;
    let (doorbell, done, gave_up) = (words.dev(DOORBELL), words.dev(DONE), words.dev(GAVE_UP));

    // A second thread issues the device's side: the driver's queue is finite,
    // and an enqueue that blocked on a full queue while the device waited for
    // an answer the same thread would give would be a deadlock.
    let (feeder_device, feeder_stream) = (device.clone(), stream.clone());
    let feeder = thread::spawn(move || -> Result<()> {
        feeder_device
            .cuda_context()
            .bind_to_thread()
            .map_err(Error::wrap)?;
        let cu = feeder_stream.cu_stream();
        match gpu {
            GpuWait::StreamMemOp => {
                for i in 1..=iterations {
                    // SAFETY: both addresses lie in the mapped allocation,
                    // which outlives the stream's work: the caller
                    // synchronises before `words` drops.
                    unsafe {
                        let r = sys::cuStreamWriteValue32_v2(cu, doorbell, i, 0);
                        if r != sys::CUresult::CUDA_SUCCESS {
                            bail!("handoff: cuStreamWriteValue32 refused: {r:?}");
                        }
                        let r = sys::cuStreamWaitValue32_v2(cu, done, i, 0);
                        if r != sys::CUresult::CUDA_SUCCESS {
                            bail!("handoff: cuStreamWaitValue32 refused: {r:?}");
                        }
                        // Under WDDM a submission can sit in the driver's
                        // queue until something flushes it; a query does.
                        if i % FLUSH_EVERY == 0 || i == iterations {
                            sys::cuStreamQuery(cu);
                        }
                    }
                }
            }
            GpuWait::KernelPoll => {
                // SAFETY: the three device addresses lie in the mapped
                // allocation, which outlives the launch (synchronised below).
                unsafe {
                    run_handoff_ping(
                        doorbell as *mut u32,
                        done as *const u32,
                        iterations,
                        KERNEL_LIMIT_NS,
                        gave_up as *mut u32,
                        cu as *mut c_void,
                    );
                    sys::cuStreamQuery(cu);
                }
            }
        }
        Ok(())
    });

    let answered = respond(&words, iterations, host);
    if answered.is_err() {
        // Release every wait still queued, so the stream drains and neither
        // the feeder nor the synchronise below can block.
        // SAFETY: the mapped done word.
        unsafe { ptr::write_volatile(words.done(), u32::MAX) };
    }
    feeder.join().expect("handoff feeder panicked")?;
    stream.synchronize().map_err(Error::wrap)?;
    // SAFETY: the mapped give-up word; the stream has drained.
    let gave_up = unsafe { ptr::read_volatile(words.host(GAVE_UP)) };
    if gave_up != 0 {
        bail!("handoff: the kernel gave up waiting for answer {gave_up}");
    }
    let seen = answered?;
    let mut trips: Vec<f64> = seen
        .windows(2)
        .map(|w| (w[1] - w[0]).as_nanos() as f64)
        .collect();
    trips.sort_by(f64::total_cmp);
    Ok(RoundTrips(trips))
}

/// The host side: wait for each doorbell, answer on `done`, and note when each
/// doorbell was seen.
fn respond(words: &Words, iterations: u32, wait: HostWait) -> Result<Vec<Instant>> {
    let deadline = Duration::from_secs(5);
    let mut seen = Vec::with_capacity(iterations as usize);
    for i in 1..=iterations {
        let t0 = Instant::now();
        loop {
            // SAFETY: the mapped doorbell word; the device writes it.
            if unsafe { ptr::read_volatile(words.doorbell()) } >= i {
                break;
            }
            if t0.elapsed() > deadline {
                bail!("handoff: doorbell {i} never rang ({wait:?})");
            }
            match wait {
                HostWait::Spin => spin_loop(),
                HostWait::Umwait { deadline_cycles } => {
                    // SAFETY: WAITPKG checked by the caller; the address is the
                    // mapped doorbell, re-read after arming so a write between
                    // the read above and the arm is not missed.
                    unsafe {
                        umonitor(words.doorbell() as *const u8);
                        if ptr::read_volatile(words.doorbell()) >= i {
                            break;
                        }
                        umwait(deadline_cycles);
                    }
                }
            }
        }
        seen.push(Instant::now());
        fence(Ordering::SeqCst);
        // SAFETY: the mapped done word; the device waits on it.
        unsafe { ptr::write_volatile(words.done(), i) };
    }
    Ok(seen)
}

/// Arms the monitor on `addr`'s cache line.
#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn umonitor(addr: *const u8) {
    asm!("umonitor {0}", in(reg) addr, options(nostack, preserves_flags));
}

/// `umwait` in C0.1 (the lighter state, faster to wake) until `cycles` TSC
/// ticks from now or the monitored line is written.
#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn umwait(cycles: u64) {
    let deadline = _rdtsc().wrapping_add(cycles);
    asm!(
        "umwait {ctrl:e}",
        ctrl = in(reg) 1u32,
        in("edx") (deadline >> 32) as u32,
        in("eax") deadline as u32,
        options(nostack),
    );
}

/// `tpause` in C0.1 for `cycles` TSC ticks: a timed wait with nothing to
/// monitor, at a fraction of a `pause` loop's power and none of its pressure
/// on the SMT sibling.
#[cfg(target_arch = "x86_64")]
pub(crate) unsafe fn tpause(cycles: u64) {
    let deadline = _rdtsc().wrapping_add(cycles);
    asm!(
        "tpause {ctrl:e}",
        ctrl = in(reg) 1u32,
        in("edx") (deadline >> 32) as u32,
        in("eax") deadline as u32,
        options(nostack),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::expert_lre::cpu_pool::pin::pin_current_thread;
    use crate::models::expert_lre::cpu_pool::topology::{performance_cores, physical_cores};
    use candle::Device;

    /// Measurement 0 (`docs/cpu_expert_compute.md` §8): the round trip for each
    /// device wait (stream memop, kernel poll) × each host wait (spin, and
    /// `umwait` at two re-arm deadlines). A `umwait` whose round trip tracks its
    /// deadline was not woken by the device's write.
    ///
    /// `cargo test --release -p candle-transformers --features cuda --lib handoff_round_trip -- --ignored --nocapture`
    #[test]
    #[ignore = "a measurement, ~5 s; read its output"]
    fn handoff_round_trip() {
        let Ok(Device::Cuda(device)) = Device::new_cuda(0) else {
            panic!("the bench needs a CUDA device");
        };
        let perf = performance_cores(&physical_cores().unwrap());
        pin_current_thread(perf[0].primary()).unwrap();
        let mut modes = vec![HostWait::Spin];
        if has_waitpkg() {
            // ~10 µs and ~1 ms of TSC at a few GHz.
            modes.push(HostWait::Umwait {
                deadline_cycles: 30_000,
            });
            modes.push(HostWait::Umwait {
                deadline_cycles: 3_000_000,
            });
        } else {
            println!("no WAITPKG on this CPU: umwait not measured");
        }
        for gpu in [GpuWait::StreamMemOp, GpuWait::KernelPoll] {
            for &host in &modes {
                let t = round_trips(&device, 5_000, gpu, host).unwrap();
                println!(
                    "{gpu:?} / {host:?}: round trip p50 {:.2} µs  p90 {:.2}  p99 {:.2}  \
                     min {:.2}  max {:.2}",
                    t.quantile(0.5) / 1e3,
                    t.quantile(0.9) / 1e3,
                    t.quantile(0.99) / 1e3,
                    t.quantile(0.0) / 1e3,
                    t.quantile(1.0) / 1e3,
                );
            }
        }
    }
}
