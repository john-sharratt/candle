//! The thread-local accumulator the hot path records into.
//!
//! Separate from a caller-owned [`ProfileAccumulator`] because the spans are
//! scattered through code that has no profiler to thread through it.
//! Thread-local rather than global so there is no lock on the hot path; the
//! harness snapshots it at phase boundaries.

#[cfg(feature = "profile")]
use super::accumulator::ProfileAccumulator;
use super::accumulator::ProfileSnapshot;
use super::mark::ProfileMark;

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
