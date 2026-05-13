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
    ) {}

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
                .map(|e| {
                    (
                        e.name.to_string(),
                        e.total.as_secs_f64() * 1000.0,
                        e.count,
                    )
                })
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
pub fn pipeline_record_duration(
    name: &'static str,
    elapsed: std::time::Duration,
    count: u64,
) {
    PIPELINE_PROF.with(|prof| prof.borrow_mut().record_duration(name, elapsed, count));
}

/// No-op when profiling is disabled.
#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn pipeline_record(_name: &'static str, _start: ProfileMark) {}

/// No-op when profiling is disabled.
#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn pipeline_record_duration(
    _name: &'static str,
    _elapsed: std::time::Duration,
    _count: u64,
) {}

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
