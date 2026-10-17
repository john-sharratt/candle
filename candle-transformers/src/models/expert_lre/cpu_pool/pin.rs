//! Keeping a pool worker on one logical processor.
//!
//! Windows CPU sets are *soft* affinity: the scheduler prefers the set but can
//! still run other threads there, which is why the pool's tasks are small and
//! an idle worker takes a late one's rows (`docs/cpu_expert_compute.md` §5.6).

use std::io;
#[cfg(target_os = "linux")]
use std::mem::{size_of, zeroed};

#[cfg(target_os = "linux")]
use libc::{cpu_set_t, sched_setaffinity, setpriority, CPU_SET, PRIO_PROCESS};
#[cfg(windows)]
use windows_sys::Win32::System::Threading::{
    GetCurrentThread, SetThreadPriority, SetThreadSelectedCpuSets, THREAD_PRIORITY_HIGHEST,
};

/// Restricts the calling thread to one CPU set (a [`topology`](super::topology)
/// id).
#[cfg(windows)]
pub fn pin_current_thread(cpu_set: u32) -> io::Result<()> {
    let ids = [cpu_set];
    let ok = unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), ids.as_ptr(), 1) };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Linux: the CPU set is the logical processor's index.
#[cfg(target_os = "linux")]
pub fn pin_current_thread(cpu_set: u32) -> io::Result<()> {
    // SAFETY: a zeroed cpu_set_t with one bit set, for the calling thread.
    unsafe {
        let mut set: cpu_set_t = zeroed();
        CPU_SET(cpu_set as usize, &mut set);
        if sched_setaffinity(0, size_of::<cpu_set_t>(), &set) != 0 {
            return Err(io::Error::last_os_error());
        }
    }
    Ok(())
}

/// Elsewhere there is no thread affinity to set; the pool runs unpinned.
#[cfg(not(any(windows, target_os = "linux")))]
pub fn pin_current_thread(_cpu_set: u32) -> io::Result<()> {
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

/// Raises the calling thread above the process's ordinary threads.
///
/// A pool worker that is preempted holding a claimed task holds up its whole
/// layer until the scheduler gives it back its core — a quantum, milliseconds,
/// against a layer of tens of microseconds — and a soft CPU set does not stop
/// the scheduler putting another thread there. At a higher priority the
/// scheduler places the process's other threads on other cores instead.
///
/// It is not also taken out of the OS's power throttling (EcoQoS): measured on
/// the laptop's wait harness (`wait_bench`), opting out raised the join
/// latency's p90 from ~1–5 µs to 110–215 µs.
#[cfg(windows)]
pub fn raise_priority() -> io::Result<()> {
    // SAFETY: the calling thread's pseudo-handle.
    let ok = unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST) };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Linux: a higher priority needs `CAP_SYS_NICE`; without it the pool runs at
/// the process's own.
#[cfg(target_os = "linux")]
pub fn raise_priority() -> io::Result<()> {
    // SAFETY: the calling thread (`who = 0`), one nice step up.
    let r = unsafe { setpriority(PRIO_PROCESS, 0, -5) };
    if r != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

/// Elsewhere the pool runs at the process's own priority.
#[cfg(not(any(windows, target_os = "linux")))]
pub fn raise_priority() -> io::Result<()> {
    Err(io::Error::from(io::ErrorKind::Unsupported))
}
