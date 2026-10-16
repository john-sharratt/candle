//! Keeping a pool worker on one logical processor.
//!
//! Windows CPU sets are *soft* affinity: the scheduler prefers the set but can
//! still run other threads there, which is why the pool's tasks are small and
//! an idle worker takes a late one's rows (`docs/cpu_expert_compute.md` §5.6).

use std::io;

/// Restricts the calling thread to one CPU set (a [`topology`](super::topology)
/// id).
#[cfg(windows)]
pub fn pin_current_thread(cpu_set: u32) -> io::Result<()> {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadSelectedCpuSets};

    let ids = [cpu_set];
    let ok = unsafe { SetThreadSelectedCpuSets(GetCurrentThread(), ids.as_ptr(), 1) };
    if ok == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}
