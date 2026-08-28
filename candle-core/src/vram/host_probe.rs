//! Host-memory *dynamics* probe: hard-fault paging rate and commit charge.
//!
//! The availability numbers everywhere else in the memory report are **levels**,
//! and a level cannot distinguish "structurally tight but healthy" (an 11 GB
//! pinned expert pool holds available RAM under the floor forever) from
//! "actively thrashing" (working sets being read back from disk). Thrash is a
//! **rate**: pages read from disk per second. This probe measures it directly,
//! plus the commit charge and limit — the ceiling that `cuMemHostAlloc` and
//! staging allocations actually fail against (RAM + pagefile), which is a
//! different exhaustion mode than physical RAM.
//!
//! One `NtQuerySystemInformation(SystemPerformanceInformation)` call yields
//! both. The counters used sit in the stable documented prefix of
//! `SYSTEM_PERFORMANCE_INFORMATION` (unchanged since NT4; the same offsets
//! perfmon's `Memory\Pages Input/sec` and `Committed Bytes` are built on):
//!
//! ```text
//!   offset  0  IdleProcessTime          i64
//!   offset  8  IoReadTransferCount      i64
//!   offset 16  IoWriteTransferCount     i64
//!   offset 24  IoOtherTransferCount     i64
//!   offset 32  IoReadOperationCount     u32
//!   offset 36  IoWriteOperationCount    u32
//!   offset 40  IoOtherOperationCount    u32
//!   offset 44  AvailablePages           u32
//!   offset 48  CommittedPages           u32
//!   offset 52  CommitLimit              u32
//!   offset 56  PeakCommitment           u32
//!   offset 60  PageFaultCount           u32   (soft + hard — not used)
//!   offset 64  CopyOnWriteCount         u32
//!   offset 68  TransitionCount          u32
//!   offset 72  CacheTransitionCount     u32
//!   offset 76  DemandZeroCount          u32
//!   offset 80  PageReadCount            u32   ← pages read FROM DISK (hard)
//!   offset 84  PageReadIoCount          u32   ← read operations (fewer; batched)
//! ```
//!
//! `PageReadCount` is cumulative since boot; [`host_perf`] returns the raw
//! counter and [`pages_in_per_sec`] derives the rate across successive calls.
//! Non-Windows builds return `None` — the callers treat an absent probe as "no
//! veto", never as zero pressure.

/// One reading of the host paging / commit counters.
#[derive(Debug, Clone, Copy)]
pub struct HostPerf {
    /// Cumulative pages read from disk since boot (hard faults + read-ahead).
    pub page_read_count: u64,
    /// Cumulative page-read I/O operations since boot.
    pub page_read_io_count: u64,
    /// Commit charge in bytes (RAM + pagefile currently promised).
    pub commit_total_bytes: u64,
    /// Commit ceiling in bytes — allocations fail against THIS, not free RAM.
    pub commit_limit_bytes: u64,
}

#[cfg(windows)]
pub fn host_perf() -> Option<HostPerf> {
    // SystemPerformanceInformation = 2. The struct is large (~350 bytes and
    // grows across Windows versions); the kernel fills as much as the buffer
    // holds and we only read the stable prefix, so a generous fixed buffer is
    // both forward- and backward-compatible.
    const SYSTEM_PERFORMANCE_INFORMATION: i32 = 2;
    #[link(name = "ntdll")]
    extern "system" {
        fn NtQuerySystemInformation(
            class: i32,
            info: *mut std::ffi::c_void,
            len: u32,
            ret_len: *mut u32,
        ) -> i32;
    }
    let mut buf = [0u8; 1024];
    let mut ret = 0u32;
    let status = unsafe {
        NtQuerySystemInformation(
            SYSTEM_PERFORMANCE_INFORMATION,
            buf.as_mut_ptr() as *mut _,
            buf.len() as u32,
            &mut ret,
        )
    };
    // NT_SUCCESS: negative values are errors; require the prefix we read.
    if status < 0 || (ret as usize) < 88 {
        return None;
    }
    let u32_at = |off: usize| u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as u64;
    // Page counts are in 4 KiB pages on every supported architecture.
    const PAGE: u64 = 4096;
    Some(HostPerf {
        page_read_count: u32_at(80),
        page_read_io_count: u32_at(84),
        commit_total_bytes: u32_at(48) * PAGE,
        commit_limit_bytes: u32_at(52) * PAGE,
    })
}

#[cfg(not(windows))]
pub fn host_perf() -> Option<HostPerf> {
    None
}

/// `MEMORYSTATUSEX` — the layout `GlobalMemoryStatusEx` fills. Fixed since
/// Windows 2000; `length` must be set to the struct's own size before the call.
#[cfg(windows)]
#[repr(C)]
struct MemoryStatusEx {
    length: u32,
    memory_load: u32,
    total_phys: u64,
    avail_phys: u64,
    total_page_file: u64,
    avail_page_file: u64,
    total_virtual: u64,
    avail_virtual: u64,
    avail_extended_virtual: u64,
}

#[cfg(windows)]
fn memory_status() -> Option<MemoryStatusEx> {
    #[link(name = "kernel32")]
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }
    let mut s = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    let ok = unsafe { GlobalMemoryStatusEx(&mut s) };
    (ok != 0).then_some(s)
}

/// Total physical RAM in bytes, or `None` where the platform has no probe here.
///
/// The *level* [`host_ram_budget`] needs as its denominator. Read from the
/// machine rather than passed in because the expert cache decides its warm tier
/// before any caller with a `sysinfo` handle is in scope.
#[cfg(windows)]
pub fn total_physical_ram() -> Option<u64> {
    memory_status().map(|s| s.total_phys).filter(|&b| b > 0)
}

/// Physical RAM the OS says is available **right now**, or `None` where the
/// platform has no probe here.
///
/// Distinct from [`total_physical_ram`] in the way that matters to anyone about
/// to take a large non-pageable allocation: total is what the machine has, this
/// is what it will actually give. A dev box with an editor, a browser and a
/// language server open can be 12 GB down before the process starts, and a
/// pinned pool sized against the total then succeeds by consuming every free
/// page — leaving the *next* pinned allocation in the same process to fail.
///
/// Windows counts free + standby (reclaimable cache) pages, Linux's
/// `MemAvailable` is the kernel's own estimate of the same thing.
#[cfg(windows)]
pub fn available_physical_ram() -> Option<u64> {
    memory_status().map(|s| s.avail_phys).filter(|&b| b > 0)
}

#[cfg(target_os = "linux")]
pub fn available_physical_ram() -> Option<u64> {
    meminfo_field("MemAvailable:")
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn available_physical_ram() -> Option<u64> {
    None
}

/// Physical RAM that was available when this process started, latched once.
///
/// # Why the live reading is the wrong input for a long-lived allocation
///
/// [`available_physical_ram`] answers "what is free at this instant", and the
/// instant the expert cache asks is the worst one in the process: the loader has
/// the checkpoint mapped and has been reading it, so the reading is depressed by
/// the engine's own transient footprint. Measured on the 16 GB dev box across
/// one gate run — 15.65 GiB free before the process, **12.18 GiB at the moment
/// the warm tier was sized**, over 20 GiB free again once it exited. The tier is
/// the largest and longest-lived allocation the engine makes, and it was being
/// sized from the bottom of a trough it had dug itself, then never revisited.
///
/// This is the same measurement taken before any of that: what the machine was
/// willing to give, rather than what it had left mid-load. The pages the live
/// reading was missing are mostly the mapped checkpoint's, which are file-backed
/// and droppable — so they were never the warm tier's competitors in the first
/// place.
///
/// Latched with `get_or_init`, so the first caller decides. [`snapshot_launch`]
/// makes that caller be process start rather than whoever happens to ask first.
pub fn launch_available_ram() -> Option<u64> {
    static LAUNCH_AVAIL: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *LAUNCH_AVAIL.get_or_init(available_physical_ram)
}

/// The least host RAM seen free since the process started, or `u64::MAX` if
/// nothing has sampled yet.
///
/// The number that says how close the warm tier's headroom came to the edge.
/// Sizing rules can only be tuned against it: the tier is allocated once and
/// the pressure it creates shows up later, in whatever allocates next, so a
/// headroom that "worked" is indistinguishable from one that missed by 40 MB
/// unless the trough is recorded as it happens.
use std::sync::atomic::{AtomicU64, Ordering};

static AVAIL_LOW_WATER: AtomicU64 = AtomicU64::new(u64::MAX);

/// Sample free host RAM now and keep it if it is the lowest yet.
///
/// Cheap (one `GlobalMemoryStatusEx` / one `/proc/meminfo` read), but not free —
/// call it at phase boundaries, not per token.
pub fn sample_available_low_water() {
    let Some(now) = available_physical_ram() else {
        return;
    };
    let mut cur = AVAIL_LOW_WATER.load(Ordering::Relaxed);
    while now < cur {
        match AVAIL_LOW_WATER.compare_exchange_weak(cur, now, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(observed) => cur = observed,
        }
    }
}

/// The lowest free-RAM reading any [`sample_available_low_water`] call has seen,
/// or `None` if none has run.
pub fn available_low_water() -> Option<u64> {
    match AVAIL_LOW_WATER.load(Ordering::Relaxed) {
        u64::MAX => None,
        v => Some(v),
    }
}

/// Take the launch reading now, before the process has allocated anything worth
/// counting.
///
/// Called from CUDA device creation, which every path that can reach a warm tier
/// passes through first — a device has to exist before weights can be loaded
/// onto it. Idempotent, and cheap enough to call from anywhere earlier.
pub fn snapshot_launch() {
    let _ = launch_available_ram();
}

/// One field of `/proc/meminfo`, in bytes.
#[cfg(target_os = "linux")]
fn meminfo_field(key: &str) -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        // "MemTotal:       32873416 kB"
        let Some(rest) = line.strip_prefix(key) else {
            continue;
        };
        let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
        return Some(kb * 1024);
    }
    None
}

#[cfg(target_os = "linux")]
pub fn total_physical_ram() -> Option<u64> {
    meminfo_field("MemTotal:")
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn total_physical_ram() -> Option<u64> {
    None
}

/// Pages read from disk per second since the previous call, derived from the
/// cumulative counter. First call (and any call after a probe failure) returns
/// `None` — a rate needs two samples. Callers poll at their own cadence; the
/// module keeps exactly one previous sample, so interleaved callers would
/// shorten each other's windows — there is one caller (the scheduler's report).
pub fn pages_in_per_sec() -> Option<f64> {
    use std::sync::Mutex;
    use std::time::Instant;
    static PREV: Mutex<Option<(Instant, u64)>> = Mutex::new(None);
    let now_count = host_perf()?.page_read_count;
    let now = Instant::now();
    let mut prev = PREV.lock().unwrap();
    let rate = prev.and_then(|(t, c)| {
        let dt = now.duration_since(t).as_secs_f64();
        // Sub-100ms windows amplify counter noise into nonsense rates; also
        // guards the counter's u32 wrap (rare: ~2^32 pages = 16 TiB read).
        (dt > 0.1 && now_count >= c).then(|| (now_count - c) as f64 / dt)
    });
    *prev = Some((now, now_count));
    rate
}

// ── Host-RAM KV budget ───────────────────────────────────────────────────────

/// The host-RAM budget: what the warm KV tier may occupy.
///
/// A *budget*, not a pressure signal. The old throttle compared OS "available"
/// against an absolute floor — a number our own mmap'd weights push down as the
/// page cache fills, so the throttle punished the system for its weights being
/// resident. This reserves the weights explicitly and budgets the remainder:
///
/// ```text
///   buffer           = max(pct% × total, 4 GiB)      — caps the WEIGHTS only
///   weights_reserved = min(weights_mmap, total − buffer)
///   kv_warm_budget   = total − pinned − weights_reserved − os_keep
/// ```
///
/// The buffer is **inclusive of warm KV**: it exists solely so the weights
/// cannot claim the whole machine — warm KV, the OS, and everything else share
/// what the weights leave behind. It is NOT subtracted again when computing the
/// warm budget; only the small fixed `os_keep` floor is, so warm growth cannot
/// starve the OS outright.
///
/// The weights are reserved IN FULL even though they are file-backed: evicting
/// weight pages to hold warm KV trades cheap-tier capacity for hard faults on
/// the inference path. Only when the model is bigger than `total − buffer` is
/// it capped — a machine that cannot hold its model resident must swap by
/// definition, and the budget states that instead of hiding it.
///
/// Dev box, expert tier resident: `31.5 − 14.4 pinned − 0.7 weights − 1 os
/// ≈ 15.4 GiB` of warm budget. The weights term is small because the MoE loader
/// declares only the GGUF's *live* pages — its expert regions move to the pack
/// file at startup and are never read again. 186 GB box:
/// `186 − 12 − 18 − 1 ≈ 155 GiB`.
#[derive(Debug, Clone, Copy)]
pub struct HostRamBudget {
    pub total_bytes: u64,
    pub buffer_bytes: u64,
    pub pinned_bytes: u64,
    pub weights_reserved_bytes: u64,
    /// Whether the weights had to be capped below their full size (the machine
    /// cannot hold the model resident; weight pages will swap).
    pub weights_capped: bool,
    pub kv_warm_budget_bytes: u64,
}

/// Weights-cap buffer percentage: `CANDLE_HOST_RAM_BUFFER_PCT`, default 30 (of
/// total RAM), floored at 4 GiB either way. Cached on first read.
fn buffer_pct() -> u64 {
    use std::sync::OnceLock;
    static V: OnceLock<u64> = OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("CANDLE_HOST_RAM_BUFFER_PCT")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&p| p > 0 && p <= 90)
            .unwrap_or(30)
    })
}

/// Fixed OS floor inside the non-weights region: keeps warm-KV growth from
/// starving the OS itself. Everything else in the buffer region is warm KV's to
/// use.
///
/// It bounds the warm **KV** tier and nothing else — the expert cache's warm tier
/// is bounded by available RAM, which on every machine measured is far tighter,
/// so this never reaches it.
///
/// 1 GiB rather than 2: warm-KV arenas are pageable, so this floor is a
/// throttle on growth rather than a guard against failure — the failure mode it
/// prevents is paging, which `pages_in_per_sec` measures directly and after the
/// fact. A GiB is enough for the OS on any machine that can host this engine at
/// all, and the GiB it releases goes to the tier that has a use for it.
const OS_KEEP_BYTES: u64 = 1024 * 1024 * 1024;

/// Pure budget arithmetic — see [`HostRamBudget`]. Exposed separately from
/// [`host_ram_budget`] so both machines' numbers pin down in unit tests without
/// touching the process-global gauges.
pub fn host_ram_budget_from(
    total: u64,
    pinned: u64,
    weights_mmap: u64,
    pct: u64,
    os_keep: u64,
) -> HostRamBudget {
    const FOUR_GIB: u64 = 4 * 1024 * 1024 * 1024;
    let buffer = (total / 100 * pct).max(FOUR_GIB);
    let weights_cap = total.saturating_sub(buffer);
    let weights_reserved = weights_mmap.min(weights_cap);
    // The buffer is NOT subtracted here — it capped the weights, and warm KV
    // lives inside the region it protected. Only the OS floor comes out.
    let kv = total
        .saturating_sub(pinned)
        .saturating_sub(weights_reserved)
        .saturating_sub(os_keep);
    HostRamBudget {
        total_bytes: total,
        buffer_bytes: buffer,
        pinned_bytes: pinned,
        weights_reserved_bytes: weights_reserved,
        weights_capped: weights_reserved < weights_mmap,
        kv_warm_budget_bytes: kv,
    }
}

/// The live budget for `total_ram`, reading the pinned and weights gauges.
pub fn host_ram_budget(total_ram: u64) -> HostRamBudget {
    host_ram_budget_from(
        total_ram,
        super::host_pinned_bytes(),
        super::weights_mmap_bytes(),
        buffer_pct(),
        OS_KEEP_BYTES,
    )
}

#[cfg(test)]
mod tests {
    // `assert!(cfg!(not(windows)), …)` is the point: the probe is required to
    // succeed on Windows and may legitimately return `None` elsewhere, so the
    // platform constant is exactly what is being asserted.
    #![allow(clippy::assertions_on_constants)]

    use super::*;

    /// On Windows the probe must return live, sane counters; elsewhere `None`.
    #[test]
    fn probe_returns_sane_counters() {
        match host_perf() {
            Some(p) => {
                assert!(p.commit_limit_bytes > 0, "commit limit can never be 0");
                assert!(
                    p.commit_total_bytes <= p.commit_limit_bytes,
                    "commit {} exceeds limit {}",
                    p.commit_total_bytes,
                    p.commit_limit_bytes
                );
                // A booted system has read SOMETHING from disk.
                assert!(p.page_read_count > 0);
            }
            None => assert!(cfg!(not(windows)), "probe must succeed on Windows"),
        }
    }

    const GIB: u64 = 1024 * 1024 * 1024;

    /// The dev box: 31.5 GB RAM, 11.05 GB pinned, 17.3 GB weights. The buffer
    /// caps the weights only (they fit: 17.3 < 31.5 − 9.45); warm KV then gets
    /// what pinned + weights + the OS floor leave — small but REAL, not zero.
    #[test]
    fn dev_box_gets_a_real_warm_budget() {
        // The dev box with the expert cache resident: 14.4 GiB pinned for the
        // warm expert tier, and only the GGUF's *live* pages declared as
        // weights — its expert regions move to the pack file at startup and are
        // never read again, so declaring the whole 17.3 GiB mapping (which this
        // test used to) reserved RAM for bytes nothing faults back in.
        let total = (31.5 * GIB as f64) as u64;
        let pinned = (14.4 * GIB as f64) as u64;
        let weights = (0.7 * GIB as f64) as u64;
        let b = host_ram_budget_from(total, pinned, weights, 30, OS_KEEP_BYTES);
        assert_eq!(
            b.buffer_bytes,
            total / 100 * 30,
            "30% of 31.5 > 4 GiB floor"
        );
        assert!(!b.weights_capped);
        assert_eq!(
            b.kv_warm_budget_bytes,
            total - pinned - weights - OS_KEEP_BYTES
        );
        assert!(
            b.kv_warm_budget_bytes > 15 * GIB,
            "the warm KV tier should have real room once the pack owns the \
             expert bytes, got {}",
            b.kv_warm_budget_bytes
        );
    }

    /// The OS floor is a constant, not a knob. It bounds warm **KV** growth
    /// only — the expert cache's warm tier is bounded by available RAM, which is
    /// far tighter — so a change here is a change to how much the KV tier may
    /// take before it starts paging.
    #[test]
    fn the_os_floor_is_one_gib() {
        assert_eq!(OS_KEEP_BYTES, GIB);
    }

    /// The 186 GB box: weights fully reserved and a huge warm budget remains —
    /// the buffer does NOT come out of it (inclusive semantics).
    #[test]
    fn big_box_budget_is_not_reduced_by_the_buffer() {
        let b = host_ram_budget_from(186 * GIB, 12 * GIB, 18 * GIB, 30, 2 * GIB);
        assert!(!b.weights_capped);
        let expect = 186 * GIB - 12 * GIB - 18 * GIB - 2 * GIB;
        assert_eq!(
            b.kv_warm_budget_bytes, expect,
            "buffer must not be subtracted"
        );
        assert!(b.kv_warm_budget_bytes > 150 * GIB);
    }

    /// A model bigger than `total − buffer` is capped — the machine must swap
    /// weights, and the budget says so explicitly. The freed region then counts
    /// toward warm KV (it is genuinely available RAM).
    #[test]
    fn oversized_weights_are_capped_and_flagged() {
        let b = host_ram_budget_from(16 * GIB, 0, 30 * GIB, 30, 2 * GIB);
        assert!(b.weights_capped);
        // buffer = max(30% x 16, 4) = 4.8 GiB; cap = 11.2 GiB.
        assert_eq!(b.weights_reserved_bytes, 16 * GIB - (16 * GIB / 100 * 30));
        assert_eq!(
            b.kv_warm_budget_bytes,
            16 * GIB - b.weights_reserved_bytes - 2 * GIB
        );
    }

    /// The rate derivation needs two samples and never goes negative.
    #[test]
    fn rate_needs_two_samples_and_is_nonnegative() {
        // First call primes; may be None. Second call (after a real interval)
        // must be Some(non-negative) on Windows.
        let _ = pages_in_per_sec();
        std::thread::sleep(std::time::Duration::from_millis(150));
        match pages_in_per_sec() {
            Some(r) => assert!(r >= 0.0),
            None => assert!(cfg!(not(windows))),
        }
    }
}
