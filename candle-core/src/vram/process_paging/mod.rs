//! One process's paging: how many of its pages have been read back from disk,
//! and how much of it is resident (`docs/elastic_warm_tier.md` §3).
//!
//! A hard fault is a page read from disk into a process — memory it had that
//! the OS had paged out. The count is per process, so a reader can tell its own
//! paging apart from every other process's: a minimized app trimmed to the
//! pagefile raises the machine's pages-in rate (`host_perf`) and nothing here.
//!
//! Windows keeps the count in `SystemProcessInformation` (`HardFaultCount`),
//! Linux in `/proc/<pid>/stat` (`majflt`).

#[cfg(windows)]
mod windows;

#[cfg(target_os = "linux")]
mod linux;

#[cfg(test)]
mod tests;

/// One process's paging counters, as one reading.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessPaging {
    pub pid: u32,
    /// The image or command name.
    pub name: String,
    /// Hard faults since the process started: reads from disk into it.
    pub hard_faults: u64,
    /// Bytes of it resident, shared pages included.
    pub working_set: u64,
    /// Bytes of it resident that are its own.
    pub private_working_set: u64,
    /// Bytes it has committed of its own — resident or in the pagefile.
    pub private_bytes: u64,
}

/// This process's paging, or `None` where the platform has no probe here.
pub fn own_paging() -> Option<ProcessPaging> {
    #[cfg(windows)]
    {
        windows::own()
    }
    #[cfg(target_os = "linux")]
    {
        linux::own()
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        None
    }
}

/// Every running process whose image name is `name`, ignoring case and a
/// trailing `.exe` — for watching another process, as the balloon harness
/// watches the engine. Empty where the platform has no probe here.
pub fn paging_of_named(name: &str) -> Vec<ProcessPaging> {
    let want = bare_name(name);
    #[cfg(windows)]
    let all = windows::all();
    #[cfg(target_os = "linux")]
    let all = linux::all();
    #[cfg(not(any(windows, target_os = "linux")))]
    let all: Vec<ProcessPaging> = Vec::new();
    all.into_iter()
        .filter(|p| bare_name(&p.name) == want)
        .collect()
}

/// `name` lower-cased, without a trailing `.exe`.
pub(crate) fn bare_name(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    lower.strip_suffix(".exe").unwrap_or(&lower).to_string()
}
