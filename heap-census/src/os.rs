//! What the OS says this process holds: its commit charge, its heap's own
//! summary, and the machine's commit against its limit.
//!
//! The process heap's summary is the figure the census cannot give from
//! inside: `committed − allocated` is memory the heap holds committed while
//! nothing uses it — retention, measured by the heap itself. Reading it takes
//! the heap's lock while it runs, so it is read off every hot path and its own
//! duration is reported with it.

use std::time::Instant;

/// The process heap's own figures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OsHeap {
    /// Bytes in blocks handed out and not freed.
    pub allocated_bytes: u64,
    /// Bytes the heap has committed, in use or not.
    pub committed_bytes: u64,
    /// How long the reading held the heap, in microseconds.
    pub read_us: u64,
}

/// The process, as the OS counts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessMemory {
    /// The commit charge: private memory the process owns, resident or not.
    pub private_bytes: u64,
    /// The process heap's summary, where the platform gives one.
    pub heap: Option<OsHeap>,
}

/// The machine's commit, where the platform states a limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MachineCommit {
    pub committed_bytes: u64,
    pub limit_bytes: u64,
}

/// Read this process: its commit charge and its heap's summary.
pub fn process_memory() -> Option<ProcessMemory> {
    Some(ProcessMemory {
        private_bytes: platform::private_bytes()?,
        heap: platform::heap(),
    })
}

/// The commit charge alone. Locks nothing, so it may be read as often as
/// wanted.
pub fn private_bytes() -> Option<u64> {
    platform::private_bytes()
}

/// The process heap's summary alone. Holds the heap's lock while it runs —
/// every allocation in the process waits for it — so it is read on a timer,
/// not per request.
pub fn heap_summary() -> Option<OsHeap> {
    platform::heap()
}

/// Read the machine's commit.
pub fn machine_commit() -> Option<MachineCommit> {
    platform::machine_commit()
}

/// `read` timed: its answer and the microseconds it took.
fn timed<T>(read: impl FnOnce() -> Option<T>) -> Option<(T, u64)> {
    let t = Instant::now();
    let v = read()?;
    Some((v, t.elapsed().as_micros() as u64))
}

#[cfg(windows)]
mod platform {
    use std::mem::{size_of, zeroed};

    use windows_sys::Win32::System::Memory::{GetProcessHeap, HeapSummary, HEAP_SUMMARY};
    use windows_sys::Win32::System::ProcessStatus::{
        GetPerformanceInfo, GetProcessMemoryInfo, PERFORMANCE_INFORMATION, PROCESS_MEMORY_COUNTERS,
        PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    use super::{timed, MachineCommit, OsHeap};

    pub fn private_bytes() -> Option<u64> {
        // SAFETY: a zeroed counters struct with its size set is what the call
        // fills; the pseudo-handle needs no closing.
        unsafe {
            let mut c: PROCESS_MEMORY_COUNTERS_EX = zeroed();
            c.cb = size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
            let ok = GetProcessMemoryInfo(
                GetCurrentProcess(),
                &mut c as *mut PROCESS_MEMORY_COUNTERS_EX as *mut PROCESS_MEMORY_COUNTERS,
                c.cb,
            );
            (ok != 0).then_some(c.PrivateUsage as u64)
        }
    }

    pub fn heap() -> Option<OsHeap> {
        let (s, read_us) = timed(|| {
            // SAFETY: the summary struct carries its own size; the process heap
            // handle is the process's for its whole life.
            unsafe {
                let mut s: HEAP_SUMMARY = zeroed();
                s.cb = size_of::<HEAP_SUMMARY>() as u32;
                let ok = HeapSummary(GetProcessHeap(), 0, &mut s);
                (ok != 0).then_some(s)
            }
        })?;
        Some(OsHeap {
            allocated_bytes: s.cbAllocated as u64,
            committed_bytes: s.cbCommitted as u64,
            read_us,
        })
    }

    pub fn machine_commit() -> Option<MachineCommit> {
        // SAFETY: as for the process counters.
        unsafe {
            let mut p: PERFORMANCE_INFORMATION = zeroed();
            p.cb = size_of::<PERFORMANCE_INFORMATION>() as u32;
            if GetPerformanceInfo(&mut p, p.cb) == 0 {
                return None;
            }
            let page = p.PageSize as u64;
            Some(MachineCommit {
                committed_bytes: p.CommitTotal as u64 * page,
                limit_bytes: p.CommitLimit as u64 * page,
            })
        }
    }
}

#[cfg(target_os = "linux")]
mod platform {
    use std::fs;

    use super::{timed, MachineCommit, OsHeap};

    /// A `Key:   123 kB` line's value, in bytes.
    fn kib_field(text: &str, key: &str) -> Option<u64> {
        let line = text.lines().find(|l| l.starts_with(key))?;
        let kib: u64 = line[key.len()..].split_whitespace().next()?.parse().ok()?;
        Some(kib * 1024)
    }

    /// Anonymous resident memory plus what of it is swapped: the private
    /// memory the process owns.
    pub fn private_bytes() -> Option<u64> {
        let s = fs::read_to_string("/proc/self/status").ok()?;
        Some(kib_field(&s, "RssAnon:")? + kib_field(&s, "VmSwap:").unwrap_or(0))
    }

    pub fn heap() -> Option<OsHeap> {
        let (m, read_us) = timed(|| {
            // SAFETY: `mallinfo2` reads the allocator's own statistics.
            Some(unsafe { libc::mallinfo2() })
        })?;
        // A block above glibc's mmap threshold is its own mapping, counted in
        // `hblkhd` and not in the arena's `uordblks`: it is allocated and
        // committed both.
        Some(OsHeap {
            allocated_bytes: m.uordblks as u64 + m.hblkhd as u64,
            committed_bytes: (m.uordblks + m.fordblks) as u64 + m.hblkhd as u64,
            read_us,
        })
    }

    pub fn machine_commit() -> Option<MachineCommit> {
        let s = fs::read_to_string("/proc/meminfo").ok()?;
        Some(MachineCommit {
            committed_bytes: kib_field(&s, "Committed_AS:")?,
            limit_bytes: kib_field(&s, "CommitLimit:")?,
        })
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod platform {
    use super::{MachineCommit, OsHeap};

    pub fn private_bytes() -> Option<u64> {
        None
    }

    pub fn heap() -> Option<OsHeap> {
        None
    }

    pub fn machine_commit() -> Option<MachineCommit> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The readings are present and ordered**: a heap holds no more than
    /// it has committed, and commits no more than the process's charge.
    #[test]
    #[cfg(any(windows, target_os = "linux"))]
    fn the_readings_are_present_and_ordered() {
        let p = process_memory().expect("this platform reads its process");
        assert!(p.private_bytes > 0);
        let h = p.heap.expect("this platform summarises its heap");
        assert!(h.allocated_bytes <= h.committed_bytes, "{h:?}");
        assert!(h.committed_bytes <= p.private_bytes, "{h:?} against {p:?}");
        let m = machine_commit().expect("this platform states a commit limit");
        assert!(
            m.committed_bytes > 0 && m.committed_bytes <= m.limit_bytes,
            "{m:?}"
        );
    }
}
