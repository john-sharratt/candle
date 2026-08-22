//! Zero-cost profiling infrastructure for hot-path instrumentation.
//!
//! All timing code compiles away completely when the `profile` feature is
//! disabled.  When enabled, accumulates wall-clock durations for named spans
//! and provides a formatted report.
//!
//! # Design
//!
//! - [`ProfileMark`] is `Instant` when profiling, `()` otherwise.
//! - [`profile_now()`] captures a timestamp (or is a no-op).
//! - [`ProfileAccumulator`] is zero-sized when profiling is off; all its
//!   methods are `#[inline(always)]` no-ops that LLVM eliminates entirely.
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::models::profile::{ProfileAccumulator, profile_now};
//!
//! let mut prof = ProfileAccumulator::new();
//! let t = profile_now();
//! // ... work ...
//! prof.record("my_span", t);
//! println!("{}", prof.report("My Profile"));
//! ```

#[cfg(feature = "profile")]
use std::fmt;

#[cfg(feature = "nvtx")]
use candle_kernels::simple::nvtx::{candle_nvtx_range_end, candle_nvtx_range_start};
#[cfg(feature = "nvtx")]
use std::ffi::CString;

// ── Timestamp mark ────────────────────────────────────────────────────

/// Opaque timestamp.  `Instant` when `profile` is enabled, `()` otherwise.
#[cfg(feature = "profile")]
pub type ProfileMark = std::time::Instant;

#[cfg(not(feature = "profile"))]
pub type ProfileMark = ();

/// Capture the current timestamp.  Compiles to nothing when profiling is
/// disabled.
#[cfg(feature = "profile")]
#[inline(always)]
pub fn profile_now() -> ProfileMark {
    std::time::Instant::now()
}

/// No-op when profiling is disabled.
#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn profile_now() -> ProfileMark {}

// ── Accumulator ───────────────────────────────────────────────────────

/// Accumulates wall-clock timing data for named spans.
///
/// When the `profile` feature is disabled this is a zero-sized type and
/// every method is an `#[inline(always)]` no-op — the compiler eliminates
/// them entirely, including the `&mut ProfileAccumulator` parameter at
/// call sites.
pub struct ProfileAccumulator {
    #[cfg(feature = "profile")]
    entries: Vec<ProfileEntry>,
}

#[cfg(feature = "profile")]
struct ProfileEntry {
    name: &'static str,
    total: std::time::Duration,
    count: u64,
}

impl ProfileAccumulator {
    /// Create a new empty accumulator.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "profile")]
            entries: Vec::new(),
        }
    }

    /// Record elapsed time since `start` for the named span.
    #[cfg(feature = "profile")]
    #[inline(always)]
    pub fn record(&mut self, name: &'static str, start: ProfileMark) {
        self.record_duration(name, start.elapsed(), 1);
    }

    /// Record an already measured duration for the named span, with an
    /// explicit event count. This is useful when lower layers return
    /// aggregated timing for multiple fast-reuse or rebuild events.
    #[cfg(feature = "profile")]
    #[inline(always)]
    pub fn record_duration(
        &mut self,
        name: &'static str,
        elapsed: std::time::Duration,
        count: u64,
    ) {
        if count == 0 {
            return;
        }
        if let Some(entry) = self.entries.iter_mut().find(|e| e.name == name) {
            entry.total += elapsed;
            entry.count += count;
        } else {
            self.entries.push(ProfileEntry {
                name,
                total: elapsed,
                count,
            });
        }
    }

    /// No-op when profiling is disabled.
    #[cfg(not(feature = "profile"))]
    #[inline(always)]
    pub fn record(&mut self, _name: &'static str, _start: ProfileMark) {}

    /// No-op when profiling is disabled.
    #[cfg(not(feature = "profile"))]
    #[inline(always)]
    pub fn record_duration(
        &mut self,
        _name: &'static str,
        _elapsed: std::time::Duration,
        _count: u64,
    ) {
    }

    /// Merge another accumulator's data into this one.
    #[cfg(feature = "profile")]
    pub fn merge(&mut self, other: &ProfileAccumulator) {
        for entry in &other.entries {
            if let Some(e) = self.entries.iter_mut().find(|e| e.name == entry.name) {
                e.total += entry.total;
                e.count += entry.count;
            } else {
                self.entries.push(ProfileEntry {
                    name: entry.name,
                    total: entry.total,
                    count: entry.count,
                });
            }
        }
    }

    /// No-op when profiling is disabled.
    #[cfg(not(feature = "profile"))]
    #[inline(always)]
    pub fn merge(&mut self, _other: &ProfileAccumulator) {}

    /// Returns `true` if any spans have been recorded.
    #[cfg(feature = "profile")]
    pub fn has_data(&self) -> bool {
        !self.entries.is_empty()
    }

    /// Always returns `false` when profiling is disabled.
    #[cfg(not(feature = "profile"))]
    #[inline(always)]
    pub fn has_data(&self) -> bool {
        false
    }

    /// Format a human-readable timing report.
    ///
    /// Spans are listed in insertion order (typically execution order).
    #[cfg(feature = "profile")]
    pub fn report(&self, header: &str) -> String {
        use fmt::Write;
        let mut out = String::new();
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "╔══════════════════════════════════════════════════════════════╗"
        );
        let _ = writeln!(out, "║  {:<60}║", header);
        let _ = writeln!(
            out,
            "╠══════════════════════════════════════════════════════════════╣"
        );
        let _ = writeln!(
            out,
            "║ {:<28} {:>11} {:>8} {:>10} ║",
            "Span", "Total", "Count", "Avg"
        );
        let _ = writeln!(
            out,
            "╟──────────────────────────────────────────────────────────────╢"
        );
        for entry in &self.entries {
            let avg = if entry.count > 0 {
                entry.total / entry.count as u32
            } else {
                std::time::Duration::ZERO
            };
            let total_ms = entry.total.as_secs_f64() * 1000.0;
            let avg_ms = avg.as_secs_f64() * 1000.0;
            let _ = writeln!(
                out,
                "║ {:<28} {:>9.2}ms {:>8} {:>8.2}ms ║",
                entry.name, total_ms, entry.count, avg_ms,
            );
        }
        let _ = writeln!(
            out,
            "╚══════════════════════════════════════════════════════════════╝"
        );
        out
    }

    /// Returns a placeholder message when profiling is disabled.
    #[cfg(not(feature = "profile"))]
    pub fn report(&self, _header: &str) -> String {
        String::from("(profiling disabled — rebuild with --features profile)")
    }
}

impl Default for ProfileAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

// ── Snapshot ──────────────────────────────────────────────────────────

/// Serialisable snapshot of profiling data.
///
/// Extracted from a [`ProfileAccumulator`] for cross-thread transfer and
/// per-config collection.  All timing values are in milliseconds.
#[derive(Debug, Clone, Default)]
pub struct ProfileSnapshot {
    /// `(span_name, total_ms, call_count)` in insertion (execution) order.
    pub entries: Vec<(String, f64, u64)>,
}

impl ProfileSnapshot {
    /// Merge another snapshot's entries into this one.
    pub fn merge(&mut self, other: &ProfileSnapshot) {
        for (name, total_ms, count) in &other.entries {
            if let Some(e) = self.entries.iter_mut().find(|(n, _, _)| n == name) {
                e.1 += total_ms;
                e.2 += count;
            } else {
                self.entries.push((name.clone(), *total_ms, *count));
            }
        }
    }
}

impl ProfileAccumulator {
    /// Snapshot current data as a transferable [`ProfileSnapshot`].
    #[cfg(feature = "profile")]
    pub fn snapshot(&self) -> ProfileSnapshot {
        ProfileSnapshot {
            entries: self
                .entries
                .iter()
                .map(|e| (e.name.to_string(), e.total.as_secs_f64() * 1000.0, e.count))
                .collect(),
        }
    }

    /// Returns an empty snapshot when profiling is disabled.
    #[cfg(not(feature = "profile"))]
    #[inline(always)]
    pub fn snapshot(&self) -> ProfileSnapshot {
        ProfileSnapshot::default()
    }

    /// Reset all accumulated spans to zero.
    #[cfg(feature = "profile")]
    pub fn reset(&mut self) {
        self.entries.clear();
    }

    /// No-op when profiling is disabled.
    #[cfg(not(feature = "profile"))]
    #[inline(always)]
    pub fn reset(&mut self) {}
}

// ── Thread-local pipeline profiler ────────────────────────────────────
//
// A separate thread-local accumulator used for pipeline-stage timings that are
// reported alongside the benchmark tables.

#[cfg(feature = "profile")]
use std::cell::RefCell;

#[cfg(feature = "profile")]
thread_local! {
    static PIPELINE_PROF: RefCell<ProfileAccumulator> = RefCell::new(ProfileAccumulator::new());
}

/// Record elapsed time for a pipeline span.
#[cfg(feature = "profile")]
#[inline(always)]
pub fn pipeline_record(name: &'static str, start: ProfileMark) {
    PIPELINE_PROF.with(|prof| prof.borrow_mut().record(name, start));
}

/// Record an already measured duration for a pipeline span.
#[cfg(feature = "profile")]
#[inline(always)]
pub fn pipeline_record_duration(name: &'static str, elapsed: std::time::Duration, count: u64) {
    PIPELINE_PROF.with(|prof| prof.borrow_mut().record_duration(name, elapsed, count));
}

/// No-op when profiling is disabled.
#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn pipeline_record(_name: &'static str, _start: ProfileMark) {}

/// No-op when profiling is disabled.
#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn pipeline_record_duration(_name: &'static str, _elapsed: std::time::Duration, _count: u64) {}

// ── Pipeline span guard ───────────────────────────────────────────────

/// A named pipeline span: wall-clock timing under `profile`, an NVTX range
/// under `nvtx`, and nothing at all under neither.
///
/// The two features are independent on purpose. `profile` costs an `Instant`
/// plus a `RefCell` borrow per span and produces the summary table; `nvtx`
/// costs a driver call and annotates the trace so
/// `nsys stats --report nvtx_kern_sum` attributes every kernel to the span that
/// launched it. A capture run wants the second without paying for the first.
///
/// # Why this exists rather than [`pipeline_record`] alone
///
/// [`pipeline_record`] receives the span name at the END of the span, which is
/// enough to accumulate a duration but not to open an NVTX range — that needs
/// the name at the start. `Span` captures both ends.
///
/// # Ordering
///
/// The span closes when the guard drops, so nested spans come out correctly
/// without any bookkeeping: Rust drops in reverse declaration order, so an
/// enclosing span declared first closes last. Spans that do NOT nest are also
/// fine — the underlying `nvtxRangeStartEx`/`nvtxRangeEnd` pair is the
/// overlapping-capable NVTX API, not the strictly-stacked push/pop one, so
/// ending out of order is legal.
///
/// # Usage
///
/// ```rust,ignore
/// let s = span("decode:select");
/// // ... work ...
/// s.end();          // records here, exactly like `pipeline_record` did
/// ```
///
/// Use [`Span::end`] rather than letting the guard fall out of scope wherever
/// the span ends before its enclosing block does — most of the hot path records
/// mid-block with more work following.
pub struct Span {
    name: &'static str,
    mark: ProfileMark,
    #[cfg(feature = "nvtx")]
    range: u64,
}

/// The NUL-terminated form of a span name, interned per name.
///
/// Span names come from a closed set of `&'static str` literals — a few dozen
/// across the wave — but a range is opened per layer per wave, so building the
/// `CString` on every open would be an allocation on the hot path for no reason.
/// Interning turns it into a pointer lookup. The cache is thread-local, matching
/// the timing accumulator, so there is no lock.
#[cfg(feature = "nvtx")]
fn nvtx_name(name: &'static str) -> &'static CString {
    use std::cell::RefCell;
    use std::collections::HashMap;

    thread_local! {
        static NAMES: RefCell<HashMap<&'static str, &'static CString>> =
            RefCell::new(HashMap::new());
    }
    NAMES.with(|n| {
        *n.borrow_mut().entry(name).or_insert_with(|| {
            // Leaked deliberately: one allocation per distinct span name per
            // thread, for the life of the process, so the pointer handed to NVTX
            // stays valid without a borrow outliving the call.
            let c = CString::new(name).expect("span names are literals without NUL bytes");
            Box::leak(Box::new(c))
        })
    })
}

/// Open a named pipeline span. See [`Span`].
#[inline(always)]
pub fn span(name: &'static str) -> Span {
    Span::new(name)
}

/// Open a span only when `cond` holds, for a phase that is skipped entirely on
/// some waves (decode-only waves have no prefill rows, and vice versa).
///
/// Without this the guard would record a span for the phase that did not run,
/// polluting both the timing table and the NVTX trace with zero-work entries.
/// `drop` the returned value where the phase ends.
#[inline(always)]
pub fn span_if(cond: bool, name: &'static str) -> Option<Span> {
    if cond {
        Some(Span::new(name))
    } else {
        None
    }
}

impl Span {
    /// Open the span: capture the start mark and push the NVTX range.
    #[inline(always)]
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            mark: profile_now(),
            #[cfg(feature = "nvtx")]
            range: {
                // Span names are `&'static str` from a closed set, so the
                // NUL-terminated copy is built once per name and reused rather
                // than allocated on every open.
                let c = nvtx_name(name);
                unsafe { candle_nvtx_range_start(c.as_ptr()) }
            },
        }
    }

    /// Close the span at this exact point, rather than at end of scope.
    #[inline(always)]
    pub fn end(self) {
        drop(self);
    }
}

impl Drop for Span {
    // `ProfileMark` is `()` when `profile` is off — that IS the zero-cost
    // design, so passing it on is a unit arg by construction, not an oversight.
    #[allow(clippy::unit_arg)]
    #[inline(always)]
    fn drop(&mut self) {
        // Close the NVTX range before the host-side bookkeeping, so the
        // accumulator's `RefCell` borrow is not attributed to the span.
        #[cfg(feature = "nvtx")]
        unsafe {
            candle_nvtx_range_end(self.range)
        };
        pipeline_record(self.name, self.mark);
    }
}

/// Snapshot and reset the pipeline profiler, returning a report string.
#[cfg(feature = "profile")]
pub fn pipeline_snapshot_and_reset() -> ProfileSnapshot {
    PIPELINE_PROF.with(|prof| {
        let mut p = prof.borrow_mut();
        let snap = p.snapshot();
        p.reset();
        snap
    })
}

/// Returns empty snapshot when profiling is disabled.
#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn pipeline_snapshot_and_reset() -> ProfileSnapshot {
    ProfileSnapshot::default()
}

/// Synchronize a device before taking a timestamp (for accurate GPU timing).
/// Compiles to nothing when profiling is disabled.
#[cfg(feature = "profile")]
#[inline(always)]
pub fn profile_sync(device: &candle::Device) {
    let _ = device.synchronize();
}

/// No-op when profiling is disabled.
#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn profile_sync(_device: &candle::Device) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// The span must be usable — and cost nothing — regardless of which
    /// features are on. This is the only assertion available when `profile` is
    /// off, and it is the one that catches a broken cfg combination.
    #[test]
    fn a_span_opens_and_closes_under_any_feature_set() {
        let s = span("probe:explicit_end");
        s.end();
        {
            let _s = span("probe:scope_end");
        }
    }

    /// `end()` must record at the call site, exactly as the
    /// `profile_now`/`pipeline_record` pair it replaces did.
    #[cfg(feature = "profile")]
    #[test]
    fn end_records_the_span_under_the_profile_feature() {
        let _ = pipeline_snapshot_and_reset(); // isolate from other spans on this thread
        let s = span("probe:recorded");
        s.end();
        let snap = pipeline_snapshot_and_reset();
        let hit = snap.entries.iter().find(|(n, _, _)| n == "probe:recorded");
        assert!(hit.is_some(), "span did not record: {:?}", snap.entries);
        assert_eq!(hit.unwrap().2, 1, "span recorded the wrong count");
    }

    /// Nested spans both record, and the inner one is contained by the outer.
    /// Drop order (reverse declaration) is what makes NVTX nesting come out
    /// right, so it is worth pinning.
    #[cfg(feature = "profile")]
    #[test]
    fn nested_spans_both_record_with_the_inner_contained() {
        let _ = pipeline_snapshot_and_reset();
        {
            let _outer = span("probe:outer");
            {
                let _inner = span("probe:inner");
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let snap = pipeline_snapshot_and_reset();
        let ms = |name: &str| {
            snap.entries
                .iter()
                .find(|(n, _, _)| n == name)
                .unwrap_or_else(|| panic!("missing {name}: {:?}", snap.entries))
                .1
        };
        assert!(
            ms("probe:outer") >= ms("probe:inner"),
            "outer {} should contain inner {}",
            ms("probe:outer"),
            ms("probe:inner")
        );
    }
}
