//! Live detection of device allocations that bypass the bump allocator.
//!
//! Every transient buffer on the wave path is supposed to come from a
//! [`crate::cuda_backend`] lease over a bump range — allocated once, budgeted
//! against the domain's span, and reclaimed by a cursor reset rather than a
//! free. Anything that instead reaches the CUDA driver for fresh memory while a
//! wave is running is a **forbidden allocation**: it costs an allocator round
//! trip in the hot loop, and it is memory the span was never sized for.
//!
//! Finding those call sites is the hard part. They are not syntactically
//! distinctive — a forbidden allocation looks exactly like a legitimate one, and
//! the difference is *when* it runs. So this module answers the question at
//! runtime: arm the detector around a wave, and every driver allocation that
//! happens inside that window reports itself with the stack that caused it.
//!
//! # Using it
//!
//! ```no_run
//! # use candle_core::forbidden_alloc;
//! let detector = forbidden_alloc::armed();
//! // ... run the wave ...
//! drop(detector);
//! eprintln!("{}", forbidden_alloc::take_report());
//! ```
//!
//! [`armed`] is the form to prefer: the flag is global, so an early `?` between
//! a bare [`enable`] and its [`disable`] would leave the detector armed for the
//! rest of the process and attribute unrelated allocations to the wave.
//!
//! # One report per distinct stack, not per allocation
//!
//! A wave issues tens of thousands of allocations across its layers, and a hot
//! loop repeats the same handful of call sites every iteration. Printing each
//! one would bury the answer in its own duplicates and slow the run enough to
//! change what it measures. So a stack is printed in full the **first** time it
//! is seen and counted silently thereafter; [`take_report`] then gives the
//! tally. No information is lost — the set of distinct stacks *is* the work
//! list, and the count tells you which entry to fix first.
//!
//! # What is not covered
//!
//! Only device allocations are instrumented, at the point where this crate
//! calls the driver. Host allocations are out of scope. The flag is global
//! rather than per-thread, so a background thread allocating during an armed
//! window is reported too — that is deliberate (the persistence thread shares
//! the device) and the stack says which thread's work it was.
//!
//! Compiled only under the `forbidden_allocations` feature. Without it every
//! entry point below is an empty inlined function and [`record`] costs nothing,
//! so the instrumentation can stay on the hot path permanently.

/// Arm the detector until the returned guard drops.
///
/// Prefer this to [`enable`]/[`disable`]: the flag is process-global, and a
/// `?` that skips the `disable` leaves it armed for every later allocation.
#[must_use = "the detector disarms when this guard drops; `let _ = armed()` disarms immediately"]
pub fn armed() -> Armed {
    enable();
    Armed(())
}

/// Disarms the detector on drop. See [`armed`].
///
/// The flag underneath is a plain on/off, not a depth count, so these guards do
/// not nest: an inner one's drop disarms the outer window too. Arm once, around
/// the region under measurement.
pub struct Armed(());

impl Drop for Armed {
    fn drop(&mut self) {
        disable();
    }
}

#[cfg(feature = "forbidden_allocations")]
mod imp {
    use std::collections::hash_map::Entry;
    use std::collections::HashMap;
    use std::fmt;
    use std::hash::{DefaultHasher, Hash, Hasher};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Mutex, MutexGuard, OnceLock};

    /// Read on every device allocation, so the disarmed path is one relaxed
    /// load and a not-taken branch.
    static ARMED: AtomicBool = AtomicBool::new(false);

    /// What one call site has done since the last [`take_report`].
    #[derive(Debug, Clone)]
    pub struct SiteReport {
        /// The instrumented driver entry point, e.g. `CudaDevice::alloc_zeros`.
        pub site: &'static str,
        /// Allocations attributed to this stack while armed.
        pub calls: u64,
        /// Bytes those allocations requested in total.
        pub bytes: u64,
        /// The stack that reached the driver, captured on first sighting.
        pub backtrace: String,
    }

    /// Every distinct stack seen while armed, worst first.
    #[derive(Debug, Clone, Default)]
    pub struct Report {
        /// Sorted by call count descending, then bytes descending: the order to
        /// work through them, since a hot-loop site repeats every iteration.
        pub sites: Vec<SiteReport>,
        pub total_calls: u64,
        pub total_bytes: u64,
    }

    impl Report {
        /// Whether the armed window was clean.
        pub fn is_clean(&self) -> bool {
            self.sites.is_empty()
        }
    }

    impl fmt::Display for Report {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            if self.is_clean() {
                return write!(f, "forbidden allocations: none");
            }
            writeln!(
                f,
                "forbidden allocations: {} distinct sites, {} allocations, {} bytes",
                self.sites.len(),
                self.total_calls,
                self.total_bytes
            )?;
            for (i, s) in self.sites.iter().enumerate() {
                writeln!(
                    f,
                    "  {:>3}. {:>9} calls {:>13} B  {}  [{}]",
                    i + 1,
                    s.calls,
                    s.bytes,
                    s.site,
                    first_user_frame(&s.backtrace)
                )?;
            }
            Ok(())
        }
    }

    struct Registry {
        /// Keyed by stack hash so a site repeated in a hot loop aggregates
        /// instead of reprinting.
        sites: HashMap<u64, SiteReport>,
    }

    fn registry() -> MutexGuard<'static, Registry> {
        static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
        REGISTRY
            .get_or_init(|| {
                Mutex::new(Registry {
                    sites: HashMap::new(),
                })
            })
            .lock()
            // A panic inside `record` would poison this for the rest of the
            // run, turning a diagnostic into a cascade of failures in code
            // that is only here to observe. The counts are plain integers;
            // there is no invariant a panic could have left broken.
            .unwrap_or_else(|e| e.into_inner())
    }

    pub fn enable() {
        ARMED.store(true, Ordering::Relaxed);
    }

    pub fn disable() {
        ARMED.store(false, Ordering::Relaxed);
    }

    pub fn is_enabled() -> bool {
        ARMED.load(Ordering::Relaxed)
    }

    /// How many frames identify a call site.
    ///
    /// Deep enough to climb out of this crate's dispatch layers and name the
    /// model code that asked for the memory, shallow enough that the same call
    /// site reached from different configs or batch positions still aggregates
    /// into one entry. Raise it if two genuinely different sites are colliding;
    /// lower it if one site is fragmenting across many entries.
    const STACK_DEPTH: usize = 32;

    /// Note that `site` allocated `bytes` of device memory.
    ///
    /// Called from the driver entry points in `cuda_backend`, so it runs inside
    /// an allocation. It never allocates device memory itself and so cannot
    /// recurse.
    ///
    /// Disarmed, this is one relaxed load. Armed, it walks the stack but
    /// **does not symbolise it** — resolving names costs milliseconds per
    /// capture, which across a hot loop would dominate the very thing being
    /// measured. Frame addresses are enough to tell call sites apart; the
    /// symbols are resolved once, when a site is first seen.
    pub fn record(site: &'static str, bytes: usize) {
        if !is_enabled() {
            return;
        }
        let mut hasher = DefaultHasher::new();
        site.hash(&mut hasher);
        let mut depth = 0usize;
        backtrace::trace(|frame| {
            (frame.ip() as usize).hash(&mut hasher);
            depth += 1;
            depth < STACK_DEPTH
        });
        let key = hasher.finish();

        let mut reg = registry();
        match reg.sites.entry(key) {
            Entry::Occupied(mut o) => {
                let s = o.get_mut();
                s.calls += 1;
                s.bytes += bytes as u64;
            }
            Entry::Vacant(v) => {
                // Only now is symbolisation worth its cost. Printing under the
                // lock keeps two threads discovering different sites from
                // interleaving their stacks into one unreadable block.
                let backtrace = format!("{:?}", backtrace::Backtrace::new());
                eprintln!(
                    "forbidden allocation detected: {site} requested {bytes} B of device memory \
                     outside the bump allocator\n{backtrace}"
                );
                v.insert(SiteReport {
                    site,
                    calls: 1,
                    bytes: bytes as u64,
                    backtrace,
                });
            }
        }
    }

    /// Note an allocation whose call site is known **exactly**, from a
    /// `#[track_caller]` caller.
    ///
    /// The cheap twin of [`record`], and the accurate one. A release backtrace
    /// names the nearest surviving symbol, which after inlining is often a
    /// *neighbour* of the real allocator — measured here: an entry attributed to
    /// `lower_tri_mask` was 344,064 B per call, which is no square mask but a
    /// `[84, 2048]` BF16 activation. Fixing what such a stack names is fixing the
    /// wrong function.
    ///
    /// `Location` is a compile-time constant threaded through by the attribute,
    /// so this walks nothing and symbolises nothing, and it survives inlining
    /// because it never depended on the frame surviving.
    pub fn record_at(
        loc: &'static std::panic::Location<'static>,
        what: &'static str,
        bytes: usize,
    ) {
        if !is_enabled() {
            return;
        }
        let mut hasher = DefaultHasher::new();
        loc.file().hash(&mut hasher);
        loc.line().hash(&mut hasher);
        what.hash(&mut hasher);
        let key = hasher.finish();

        let mut reg = registry();
        match reg.sites.entry(key) {
            Entry::Occupied(mut o) => {
                let s = o.get_mut();
                s.calls += 1;
                s.bytes += bytes as u64;
            }
            Entry::Vacant(v) => {
                let at = format!("{}:{}", loc.file(), loc.line());
                eprintln!("forbidden allocation: {what} {bytes} B at {at}");
                v.insert(SiteReport {
                    site: what,
                    calls: 1,
                    bytes: bytes as u64,
                    backtrace: at,
                });
            }
        }
    }

    /// Drain and return everything recorded since the last call.
    pub fn take_report() -> Report {
        let mut sites: Vec<SiteReport> = registry().sites.drain().map(|(_, v)| v).collect();
        sites.sort_by(|a, b| b.calls.cmp(&a.calls).then(b.bytes.cmp(&a.bytes)));
        let total_calls = sites.iter().map(|s| s.calls).sum();
        let total_bytes = sites.iter().map(|s| s.bytes).sum();
        Report {
            sites,
            total_calls,
            total_bytes,
        }
    }

    /// Forget everything recorded so far, without producing a report.
    pub fn reset() {
        registry().sites.clear();
    }

    /// How many user frames make up a site's label.
    ///
    /// **One is not enough, and the reason cost real time.** Symbolisation maps
    /// an instruction pointer to the nearest preceding symbol, so an inlined or
    /// merged callee is reported under whichever neighbour owns that address —
    /// `qkv_segmented_matmul` surfaced as `grouped_matmul_gemx`, its neighbour in
    /// the same object, and a whole diagnosis was built on the wrong function.
    /// A chain is self-correcting: even when the innermost symbol is
    /// misattributed, its callers place it unambiguously.
    ///
    /// **Six rather than three**, because three stopped short of the answer on
    /// the sites that mattered. The `delta_net` cluster labels its top five
    /// entries `empty_beside <- {mix,cuda}::{...}` — five distinct call sites
    /// that agree on their innermost three frames and diverge only above them,
    /// so a three-frame label collapsed them into one row and hid which root was
    /// feeding the cascade. The cost is table width, paid once per report, on a
    /// diagnostic that is off by default.
    const LABEL_FRAMES: usize = 6;

    /// The innermost few frames that belong to this codebase rather than to the
    /// allocator plumbing, innermost first, joined by ` <- `.
    fn first_user_frame(backtrace: &str) -> String {
        /// Matched as **prefixes**, not substrings: `"core::"` as a substring
        /// also matches `candle_core::`, which would hide exactly the frames
        /// this function exists to find.
        const SKIP: [&str; 6] = [
            "backtrace::",
            "candle_core::cuda_backend::",
            "candle_core::forbidden_alloc",
            "core::",
            "cudarc::",
            "std::",
        ];
        let frames: Vec<&str> = backtrace
            .lines()
            .map(str::trim)
            .filter_map(|l| l.split_once(": ").map(|(_, sym)| sym))
            .filter(|sym| !SKIP.iter().any(|s| sym.starts_with(s)))
            .take(LABEL_FRAMES)
            .collect();
        if frames.is_empty() {
            "<unknown>".to_string()
        } else {
            frames.join(" <- ")
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        /// The registry and the flag are process-global, so these tests are not
        /// independent of each other however separate their values look — the
        /// same mistake that made this crate's GPU tests flaky. One lock.
        fn serial() -> MutexGuard<'static, ()> {
            static SERIAL: Mutex<()> = Mutex::new(());
            SERIAL.lock().unwrap_or_else(|e| e.into_inner())
        }

        /// Disarm and drain, so a failing test cannot arm the detector for the
        /// ones that follow it.
        fn fresh() -> MutexGuard<'static, ()> {
            let g = serial();
            disable();
            reset();
            g
        }

        #[test]
        fn disarmed_records_nothing() {
            let _s = fresh();
            record("test::disarmed", 4096);
            assert!(take_report().is_clean());
        }

        #[test]
        fn armed_records_and_disarms_on_drop() {
            let _s = fresh();
            {
                let _armed = crate::forbidden_alloc::armed();
                assert!(is_enabled());
                record("test::armed", 4096);
            }
            assert!(!is_enabled(), "guard must disarm on drop");
            record("test::armed", 4096);

            let report = take_report();
            assert_eq!(report.total_calls, 1, "only the armed call is recorded");
            assert_eq!(report.total_bytes, 4096);
        }

        #[test]
        fn repeats_of_one_stack_aggregate() {
            let _s = fresh();
            let _armed = crate::forbidden_alloc::armed();
            // Same call site, same stack: one entry, three calls. This is the
            // property that keeps a hot loop's output readable.
            for _ in 0..3 {
                record("test::repeat", 100);
            }
            let report = take_report();
            assert_eq!(report.sites.len(), 1, "one distinct stack");
            assert_eq!(report.sites[0].calls, 3);
            assert_eq!(report.sites[0].bytes, 300);
        }

        #[test]
        fn distinct_sites_stay_distinct() {
            let _s = fresh();
            let _armed = crate::forbidden_alloc::armed();
            record("test::site_a", 1);
            record("test::site_b", 2);
            let report = take_report();
            assert_eq!(report.sites.len(), 2);
            assert_eq!(report.total_calls, 2);
            assert_eq!(report.total_bytes, 3);
        }

        #[test]
        fn report_ranks_hottest_first() {
            let _s = fresh();
            let _armed = crate::forbidden_alloc::armed();
            record("test::cold", 1_000_000);
            for _ in 0..5 {
                record("test::hot", 8);
            }
            let report = take_report();
            assert_eq!(
                report.sites[0].site, "test::hot",
                "call count outranks byte count: a hot-loop site repeats every iteration"
            );
        }

        /// The summary line has to name the *caller*, and `candle_core::` frames
        /// are the interesting ones. A substring match on `"core::"` would
        /// swallow them and attribute every site to whatever came next.
        #[test]
        fn summary_frame_skips_the_allocator_not_the_caller() {
            let bt = "\
   0: candle_core::forbidden_alloc::imp::record
   1: candle_core::cuda_backend::device::alloc
   2: core::ptr::drop_in_place
   3: std::sys::pal::unix::thread
   4: candle_core::quantized::cuda::dequantize_f32
   5: candle_transformers::models::batched_layer::forward";
            assert_eq!(
                first_user_frame(bt),
                concat!(
                    "candle_core::quantized::cuda::dequantize_f32",
                    " <- candle_transformers::models::batched_layer::forward"
                )
            );
        }

        /// **The label must survive a misattributed innermost frame.**
        ///
        /// Symbolisation resolves an address to the nearest preceding symbol, so
        /// an inlined or merged callee is reported under whichever neighbour owns
        /// that address. That really happened: `qkv_segmented_matmul` surfaced as
        /// `grouped_matmul_gemx`, its neighbour in the same object, and a whole
        /// diagnosis was built on the wrong function before the callers gave it
        /// away. Carrying the callers is what makes one wrong symbol recoverable
        /// rather than misleading.
        #[test]
        fn the_callers_disambiguate_a_misattributed_symbol() {
            let bt = "\
   0: candle_core::forbidden_alloc::imp::record
   1: candle_core::cuda_backend::device::alloc
   2: candle_core::quantized::cuda::grouped_matmul_gemx
   3: candle_core::quantized::QMatMul::qkv_segmented
   4: candle_transformers::models::quantized_qwen3_moe::project_qkv";
            let label = first_user_frame(bt);
            assert!(
                label.contains("qkv_segmented"),
                "the caller chain must place the site even when frame 0 is wrong: {label}"
            );
            assert_eq!(label.matches(" <- ").count(), LABEL_FRAMES - 1);
        }

        /// A build without debug info symbolises to bare addresses. Passing one
        /// through is the useful answer — it names the stripped build and is
        /// still resolvable against a map file, where `<unknown>` would discard
        /// the only information there was.
        #[test]
        fn summary_frame_passes_through_an_unsymbolised_address() {
            assert_eq!(first_user_frame("   0: 0x7ff6a1b2c3d4"), "0x7ff6a1b2c3d4");
        }

        #[test]
        fn summary_frame_falls_back_when_every_frame_is_skipped() {
            assert_eq!(
                first_user_frame("   0: candle_core::forbidden_alloc::imp::record"),
                "<unknown>"
            );
        }

        #[test]
        fn take_report_drains() {
            let _s = fresh();
            let _armed = crate::forbidden_alloc::armed();
            record("test::drain", 64);
            assert_eq!(take_report().total_calls, 1);
            assert!(
                take_report().is_clean(),
                "a drained report must not repeat its findings"
            );
        }
    }
}

#[cfg(not(feature = "forbidden_allocations"))]
mod imp {
    use std::fmt;
    use std::sync::atomic::{AtomicBool, Ordering};

    #[derive(Debug, Clone)]
    pub struct SiteReport {
        pub site: &'static str,
        pub calls: u64,
        pub bytes: u64,
        pub backtrace: String,
    }

    #[derive(Debug, Clone, Default)]
    pub struct Report {
        pub sites: Vec<SiteReport>,
        pub total_calls: u64,
        pub total_bytes: u64,
    }

    impl Report {
        pub fn is_clean(&self) -> bool {
            true
        }
    }

    impl fmt::Display for Report {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(
                f,
                "forbidden allocations: not measured (build with --features forbidden_allocations)"
            )
        }
    }

    /// Arming without the feature compiled in would report a clean run for a
    /// build that never looked, which reads as "no forbidden allocations" and
    /// is the one wrong answer this module must not give. Say so, once.
    pub fn enable() {
        static WARNED: AtomicBool = AtomicBool::new(false);
        if !WARNED.swap(true, Ordering::Relaxed) {
            eprintln!(
                "forbidden-allocation detector requested but not compiled in; \
                 rebuild with --features forbidden_allocations"
            );
        }
    }

    pub fn disable() {}

    pub fn is_enabled() -> bool {
        false
    }

    #[inline(always)]
    pub fn record(_site: &'static str, _bytes: usize) {}

    #[inline(always)]
    pub fn record_at(
        _loc: &'static std::panic::Location<'static>,
        _what: &'static str,
        _bytes: usize,
    ) {
    }

    pub fn take_report() -> Report {
        Report::default()
    }

    pub fn reset() {}
}

pub use imp::{
    disable, enable, is_enabled, record, record_at, reset, take_report, Report, SiteReport,
};
