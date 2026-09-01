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

/// How the host's RAM is partitioned between everything that wants it.
///
/// A *budget*, not a pressure signal. The old throttle compared OS "available"
/// against an absolute floor — a number our own mmap'd weights push down as the
/// page cache fills, so the throttle punished the system for its weights being
/// resident. This reserves the weights explicitly and partitions the remainder:
///
/// ```text
///   buffer           = max(pct% × total, 4 GiB)      — caps the WEIGHTS only
///   weights_reserved = min(weights_mmap, total − buffer)
///   tier_pool        = total − weights_reserved − PAGEABLE_RESERVE − os_keep
///   kv_warm_budget   = tier_pool × KV_WARM_SHARE_PCT%
///   expert_pinned    = tier_pool − kv_warm_budget
/// ```
///
/// # One pool, split once — because there are TWO warm tiers
///
/// The expert cache pins a host pool and the KV cache holds warm arenas, and
/// they are the two largest host consumers by an order of magnitude. They used
/// to be sized independently, each against a snapshot that did not contain the
/// other, which over-committed the machine by roughly the size of the pageable
/// region:
///
/// - the expert tier sizes FIRST, while `pinned` is only the embedding, so the
///   KV budget it was handed as a ceiling computed to nearly the whole machine
///   and never bound — leaving it bounded only by launch-time free RAM minus a
///   constant that had been measured on a gate barely exercising warm KV;
/// - the KV budget was then recomputed with the expert pool pinned, but did not
///   subtract the pageable buffer, so it believed it could grow into the region
///   holding the page cache, the mapped checkpoint, and every other process.
///
/// Measured on the 16 GB box (31.5 GiB host) during a tool calibration: expert
/// tier took 13.5 GiB at launch, warm KV grew to 7.1 GiB, and the machine
/// reached 0.9 GiB free while paging at 1,302 pages/sec — with the KV tier still
/// reporting `over_budget = false` at half its budget. The sum of what the two
/// tiers each believed they could have was 28 GiB on a machine whose measured
/// need for everything else is 9.5 GiB.
///
/// So the partition is computed ONCE, from the machine's fixed quantities, and
/// both tiers take their slice from it.
///
/// # `pinned_bytes` is reported, never subtracted
///
/// It is observability only. Subtracting the live pinned gauge is exactly what
/// made the old budget depend on *when* it was called: before the expert pool
/// existed it read one number and after it read another, and the expert tier
/// consumed the first while the KV tier was handed the second. A partition that
/// depends on call order is not a partition.
///
/// # The weights are reserved in full
///
/// Even though they are file-backed: evicting weight pages to hold warm bytes
/// trades cheap-tier capacity for hard faults on the inference path. Only when
/// the model is bigger than `total − buffer` is it capped — a machine that
/// cannot hold its model resident must swap by definition, and the budget states
/// that instead of hiding it.
///
/// Dev box (31.5 GiB, 2.4 GiB live weight pages): `31.5 − 2.4 − 10 − 1 = 18.1`
/// GiB of tier pool → 7.2 KV / 10.9 expert, leaving 10 GiB genuinely pageable.
/// The weights term is small because the MoE loader declares only the GGUF's
/// *live* pages — its expert regions move to the pack file at startup and are
/// never read again.
#[derive(Debug, Clone, Copy)]
pub struct HostRamBudget {
    pub total_bytes: u64,
    pub buffer_bytes: u64,
    /// Live pinned gauge — **reported, not used**. See the type docs.
    pub pinned_bytes: u64,
    pub weights_reserved_bytes: u64,
    /// Whether the weights had to be capped below their full size (the machine
    /// cannot hold the model resident; weight pages will swap).
    pub weights_capped: bool,
    /// Host RAM held back from BOTH tiers for the page cache, the mapped
    /// checkpoint, other processes, and the engine's own non-pinned
    /// allocations. See [`PAGEABLE_RESERVE`].
    pub pageable_reserve_bytes: u64,
    /// What the two warm tiers share. The sum of the two budgets below.
    pub tier_pool_bytes: u64,
    pub kv_warm_budget_bytes: u64,
    /// What the expert cache's warm tier may page-lock.
    pub expert_pinned_budget_bytes: u64,
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

/// Host RAM that must stay pageable, whatever else happens.
///
/// The page cache, the mapped checkpoint, every other process's working set,
/// and the engine's own non-pinned allocations all live here. Pinning into it
/// does not fail — measured directly: a probe took the entire pinnable half of
/// a 31.5 GiB box without the driver once refusing, and free RAM ended at
/// 0.02 GiB. **There is no natural stopping point**, which is why this bound is
/// stated rather than discovered, and why an allocate-until-refusal probe cannot
/// find it: refusal never comes, the machine just starts thrashing.
/// `candle-core/tests/pinned_ceiling_probe.rs` re-measures it on any machine
/// this needs revisiting on.
///
/// # An absolute floor, not a fraction
///
/// This was `total / 2`, from a measured failure at 76 % on a 194 GB machine
/// (148 GB locked, 66 GB of other commit pushed to pagefile). A fraction reads
/// the right way on that machine and the wrong way on a small one: on a 31.5 GiB
/// box half is 15.76 GiB reserved against an OS and application set measured at
/// 9.5 GiB, so the rule bound the warm tier for no reason anyone could point
/// at — the tier stopped growing while 6 GiB sat unused and unusable.
///
/// What the OS and the surrounding applications need does not scale with how
/// much RAM is installed, so the reserve is an absolute quantity. 10 GiB covers
/// the 9.5 GiB measured here with room, and on a large machine it lets pinning
/// go far past half — which is correct, and is what the 194 GB case was really
/// telling us: 46 GB pageable was too little there too.
///
/// It lives here, beside the partition that applies it, rather than beside the
/// expert tier that used to own it: it bounds BOTH warm tiers, and a reserve
/// only one of them subtracts is not a reserve.
pub const PAGEABLE_RESERVE: u64 = 10 * 1024 * 1024 * 1024;

/// The warm KV tier's share of the tier pool, as a percentage.
///
/// **Both tiers are caches with cold tiers**, so this split is a performance
/// trade rather than a correctness one — what matters for correctness is that
/// the two shares SUM to the pool. An expert miss is a synchronous read from the
/// pack file on the pipeline thread; a warm-KV miss is a read from the redo log.
///
/// The expert side takes the clear majority because it gates decode throughput
/// directly, and because it must not shrink: at 30 % the 31.5 GiB dev box yields
/// 12.7 GiB of expert budget, slightly **more** than the 13.5 GiB the tier used
/// to take before the partition existed once its own page-lock ceiling is
/// applied — the honest accounting costs no residency.
///
/// The KV side needs far less than its peak occupancy suggested. That peak —
/// 7.1 GiB across a full tool calibration — was almost entirely K/V already
/// marked for distillation and waiting on a compactor
/// (`Substrate::release_distilled_kv`); released at the mark, the tier holds
/// only live conversation history.
const KV_WARM_SHARE_PCT: u64 = 30;

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
    // What the two warm tiers share, after the machine's fixed obligations.
    //
    // `pinned` is deliberately absent: a partition that changes with the live
    // pinned gauge changes with WHEN it is called, which is the whole defect
    // this replaced — the expert tier read it before pinning and the KV tier
    // after, so each was sized against a machine that did not contain the other.
    // Non-tier pins (the host-mapped embedding, the routing buffer) are small,
    // fixed, and covered by the margin `PAGEABLE_RESERVE` carries over the 9.5
    // GiB it was measured against; the expert tier subtracts them again from its
    // own page-lock ceiling, which is where that belongs.
    let tier_pool = total
        .saturating_sub(weights_reserved)
        .saturating_sub(PAGEABLE_RESERVE)
        .saturating_sub(os_keep);
    let kv = tier_pool / 100 * KV_WARM_SHARE_PCT;
    HostRamBudget {
        total_bytes: total,
        buffer_bytes: buffer,
        pinned_bytes: pinned,
        weights_reserved_bytes: weights_reserved,
        weights_capped: weights_reserved < weights_mmap,
        pageable_reserve_bytes: PAGEABLE_RESERVE,
        tier_pool_bytes: tier_pool,
        kv_warm_budget_bytes: kv,
        // The remainder, not a second percentage: the two must SUM to the pool
        // or the partition leaks, and a rounding gap between two independently
        // computed percentages is exactly how that happens quietly.
        expert_pinned_budget_bytes: tier_pool - kv,
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

    /// **The partition must fit the machine.** Everything the budget hands out,
    /// plus everything it reserves, is exactly the machine — no more.
    ///
    /// This is the assertion the old budget could not make. It computed
    /// `kv_warm = total − pinned − weights − os_keep` and left the pageable
    /// buffer inside that figure, so on the dev box it handed the KV tier 15.4
    /// GiB *on top of* an expert pool that had already page-locked 14.4 — 30 GiB
    /// of promises against 31.5 GiB of RAM, with nothing left for the page
    /// cache, the mapped checkpoint, or any other process. The old test asserted
    /// `kv_warm_budget > 15 GiB` and so pinned the over-commit in place.
    #[test]
    fn the_partition_never_promises_more_than_the_machine_has() {
        for total_gib in [16.0f64, 31.5, 64.0, 186.0] {
            let total = (total_gib * GIB as f64) as u64;
            let weights = (2.4 * GIB as f64) as u64;
            let b = host_ram_budget_from(total, 0, weights, 30, OS_KEEP_BYTES);
            let promised = b.expert_pinned_budget_bytes
                + b.kv_warm_budget_bytes
                + b.weights_reserved_bytes
                + b.pageable_reserve_bytes
                + OS_KEEP_BYTES;
            assert!(
                promised <= total,
                "on a {total_gib} GiB box the budget promises {promised} of {total}",
            );
            // And the two tier shares are exactly the pool — a rounding gap
            // between them is capacity that silently belongs to nobody.
            assert_eq!(
                b.expert_pinned_budget_bytes + b.kv_warm_budget_bytes,
                b.tier_pool_bytes,
                "the two tier budgets must sum to the pool on a {total_gib} GiB box",
            );
        }
    }

    /// The dev box, stated in full so the numbers that broke the machine are
    /// written down: 31.5 GiB with 2.4 GiB of live weight pages.
    ///
    /// Only the GGUF's *live* pages count as weights — its expert regions move
    /// to the pack file at startup and are never read again, so declaring the
    /// whole 17.3 GiB mapping would reserve RAM for bytes nothing faults back in.
    #[test]
    fn dev_box_splits_its_tier_pool() {
        let total = (31.5 * GIB as f64) as u64;
        let weights = (2.4 * GIB as f64) as u64;
        let b = host_ram_budget_from(total, 0, weights, 30, OS_KEEP_BYTES);

        assert_eq!(
            b.buffer_bytes,
            total / 100 * 30,
            "30% of 31.5 > 4 GiB floor"
        );
        assert!(!b.weights_capped);
        assert_eq!(
            b.tier_pool_bytes,
            total - weights - PAGEABLE_RESERVE - OS_KEEP_BYTES,
        );
        // ~18.1 GiB of pool → ~5.4 KV / ~12.7 expert.
        let kv = b.kv_warm_budget_bytes as f64 / GIB as f64;
        let ex = b.expert_pinned_budget_bytes as f64 / GIB as f64;
        assert!((5.2..5.7).contains(&kv), "KV share reads {kv:.2} GiB");
        assert!((12.4..13.0).contains(&ex), "expert share reads {ex:.2} GiB");
        // **The expert tier must not shrink to pay for the honest accounting.**
        // Before the partition it took 13.5 GiB on this machine by sizing from
        // launch-time free RAM, and the whole point of releasing distilled K/V
        // at the mark rather than at compaction is that the KV side no longer
        // needs the 7.1 GiB its peak occupancy once suggested. If a future split
        // pushes the expert budget back under this, decode throughput pays for
        // it — measured 14 % fewer resident experts at a 40 % KV share.
        assert!(
            b.expert_pinned_budget_bytes >= (12.4 * GIB as f64) as u64,
            "the expert tier held ~13.5 GiB before the partition; {ex:.2} GiB \
             would cost resident experts",
        );
    }

    /// **`pinned` may not move the partition.** The defect this replaced was an
    /// ordering one: the expert tier read the budget before page-locking its
    /// pool and the KV tier read it after, so each was sized against a machine
    /// that did not contain the other. A partition that changes with the live
    /// pinned gauge is not a partition.
    #[test]
    fn the_partition_is_the_same_before_and_after_the_pool_is_pinned() {
        let total = (31.5 * GIB as f64) as u64;
        let weights = (2.4 * GIB as f64) as u64;
        let before = host_ram_budget_from(total, 0, weights, 30, OS_KEEP_BYTES);
        let after = host_ram_budget_from(
            total,
            (13.5 * GIB as f64) as u64,
            weights,
            30,
            OS_KEEP_BYTES,
        );
        assert_eq!(
            before.kv_warm_budget_bytes, after.kv_warm_budget_bytes,
            "the KV share moved once the expert pool was pinned",
        );
        assert_eq!(
            before.expert_pinned_budget_bytes, after.expert_pinned_budget_bytes,
            "the expert share moved once the expert pool was pinned",
        );
        assert_eq!(before.tier_pool_bytes, after.tier_pool_bytes);
        // Reported, though — it is still what the machine actually holds.
        assert_eq!(after.pinned_bytes, (13.5 * GIB as f64) as u64);
    }

    /// A machine too small to hold its model still yields a coherent partition
    /// rather than underflowing into a huge one.
    #[test]
    fn a_machine_smaller_than_its_reserves_yields_an_empty_pool() {
        let total = 8 * GIB;
        let weights = 6 * GIB;
        let b = host_ram_budget_from(total, 0, weights, 30, OS_KEEP_BYTES);
        assert!(b.weights_capped, "the model cannot fit beside the buffer");
        assert_eq!(
            b.tier_pool_bytes, 0,
            "no pool is left, and it must read as zero rather than wrap",
        );
        assert_eq!(b.kv_warm_budget_bytes, 0);
        assert_eq!(b.expert_pinned_budget_bytes, 0);
    }

    /// The OS floor is a constant, not a knob. It comes out of the tier pool
    /// before either share is taken, so a change here changes how much BOTH
    /// warm tiers may hold.
    #[test]
    fn the_os_floor_is_one_gib() {
        assert_eq!(OS_KEEP_BYTES, GIB);
    }

    /// The 186 GB box: the reserve stays an absolute 10 GiB, so a large machine
    /// puts nearly all of itself into the tier pool.
    ///
    /// The KV share is far smaller than the 154 GiB the old budget handed out —
    /// deliberately, because that figure was never real: the expert tier was
    /// taking its own pool out of the same RAM at the same time. 47 GiB of warm
    /// KV beside 109 GiB of pinned experts is the honest version of the same
    /// machine.
    #[test]
    fn big_box_splits_a_large_pool() {
        let b = host_ram_budget_from(186 * GIB, 12 * GIB, 18 * GIB, 30, 2 * GIB);
        assert!(!b.weights_capped);
        assert_eq!(
            b.tier_pool_bytes,
            186 * GIB - 18 * GIB - PAGEABLE_RESERVE - 2 * GIB,
        );
        assert!(b.kv_warm_budget_bytes > 45 * GIB);
        assert!(b.expert_pinned_budget_bytes > 105 * GIB);
        assert_eq!(
            b.kv_warm_budget_bytes + b.expert_pinned_budget_bytes,
            b.tier_pool_bytes,
        );
    }

    /// A model bigger than `total − buffer` is capped — the machine must swap
    /// weights, and the budget says so explicitly.
    ///
    /// **And then there is nothing left to give.** The old budget handed the KV
    /// tier the 2.8 GiB between the capped weights and the OS floor, ignoring
    /// that the page cache and the mapped checkpoint have to live somewhere. A
    /// 16 GiB machine asked to hold a 30 GiB model has no warm tier at all, and
    /// saying so is more useful than a budget it can only meet by thrashing.
    #[test]
    fn oversized_weights_leave_no_tier_pool() {
        let b = host_ram_budget_from(16 * GIB, 0, 30 * GIB, 30, 2 * GIB);
        assert!(b.weights_capped);
        // buffer = max(30% x 16, 4) = 4.8 GiB; cap = 11.2 GiB.
        assert_eq!(b.weights_reserved_bytes, 16 * GIB - (16 * GIB / 100 * 30));
        assert_eq!(b.tier_pool_bytes, 0);
        assert_eq!(b.kv_warm_budget_bytes, 0);
        assert_eq!(b.expert_pinned_budget_bytes, 0);
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
