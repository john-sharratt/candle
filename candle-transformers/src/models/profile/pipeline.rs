//! The accumulator the hot path records into.
//!
//! Separate from a caller-owned [`ProfileAccumulator`] because the spans are
//! scattered through code that has no profiler to thread through it.
//!
//! **Per-thread storage, process-wide readout.** The record path touches only
//! this thread's accumulator, so the hot path never contends: a wave engine
//! records from the scheduler thread while the persistence thread records its
//! own drain, and neither waits on the other. The readout
//! ([`pipeline_snapshot_and_reset`]) aggregates every thread's table, which is
//! what lets a daemon expose the totals over an API from a request handler that
//! is not the thread that produced them — the first thing a thread-local-only
//! readout gets wrong, because it answers "empty" rather than "not mine".
//!
//! A thread that exits merges its totals into a retired table rather than
//! dropping them, so work done on a short-lived pool thread is still counted.

#[cfg(feature = "profile")]
use super::accumulator::ProfileAccumulator;
use super::accumulator::ProfileSnapshot;
use super::mark::ProfileMark;

#[cfg(feature = "profile")]
mod store {
    use super::ProfileAccumulator;
    use std::sync::{Arc, Mutex, OnceLock, Weak};

    type Shared = Arc<Mutex<ProfileAccumulator>>;

    /// Every live thread's accumulator, weakly held so a thread that exits is
    /// pruned rather than leaked.
    fn live() -> &'static Mutex<Vec<Weak<Mutex<ProfileAccumulator>>>> {
        static LIVE: OnceLock<Mutex<Vec<Weak<Mutex<ProfileAccumulator>>>>> = OnceLock::new();
        LIVE.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// Totals inherited from threads that have exited.
    fn retired() -> &'static Mutex<ProfileAccumulator> {
        static RETIRED: OnceLock<Mutex<ProfileAccumulator>> = OnceLock::new();
        RETIRED.get_or_init(|| Mutex::new(ProfileAccumulator::new()))
    }

    /// Holds one thread's accumulator and, on thread exit, folds whatever it
    /// still holds into [`retired`] so the work is not silently uncounted.
    struct Handle(Shared);

    impl Drop for Handle {
        fn drop(&mut self) {
            let Ok(mine) = self.0.lock() else { return };
            if let Ok(mut r) = retired().lock() {
                r.merge(&mine);
            }
        }
    }

    thread_local! {
        static LOCAL: Handle = {
            let acc: Shared = Arc::new(Mutex::new(ProfileAccumulator::new()));
            if let Ok(mut l) = live().lock() {
                l.push(Arc::downgrade(&acc));
            }
            Handle(acc)
        };
    }

    /// Run `f` against this thread's accumulator. Uncontended by construction —
    /// only this thread writes here, and the readout's brief lock is the sole
    /// other visitor.
    #[inline]
    pub(super) fn with_local<R>(f: impl FnOnce(&mut ProfileAccumulator) -> R) -> Option<R> {
        LOCAL
            .try_with(|h| h.0.lock().ok().map(|mut a| f(&mut a)))
            .ok()
            .flatten()
    }

    /// Snapshot and clear every thread's table plus the retired one, merged.
    pub(super) fn snapshot_and_reset_all() -> super::ProfileSnapshot {
        let mut out = super::ProfileSnapshot::default();
        if let Ok(mut r) = retired().lock() {
            out.merge(&r.snapshot());
            r.reset();
        }
        let Ok(mut l) = live().lock() else { return out };
        // Prune threads that have gone while collecting from those that remain.
        l.retain(|w| match w.upgrade() {
            Some(acc) => {
                if let Ok(mut a) = acc.lock() {
                    out.merge(&a.snapshot());
                    a.reset();
                }
                true
            }
            None => false,
        });
        out
    }
}

/// Record elapsed time for a pipeline span.
#[cfg(feature = "profile")]
#[inline(always)]
pub fn pipeline_record(name: &'static str, start: ProfileMark) {
    store::with_local(|a| a.record(name, start));
}

/// Record an already measured duration for a pipeline span.
#[cfg(feature = "profile")]
#[inline(always)]
pub fn pipeline_record_duration(name: &'static str, elapsed: std::time::Duration, count: u64) {
    store::with_local(|a| a.record_duration(name, elapsed, count));
}

/// No-op when profiling is disabled.
#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn pipeline_record(_name: &'static str, _start: ProfileMark) {}

/// No-op when profiling is disabled.
#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn pipeline_record_duration(_name: &'static str, _elapsed: std::time::Duration, _count: u64) {}

/// Snapshot and reset the pipeline profiler across **every** thread.
///
/// Snapshot-and-reset in one call is deliberate: it makes two consecutive
/// readings a well-defined interval, so a caller can bracket exactly the window
/// it cares about — the seconds a rate collapsed, rather than an average since
/// process start that buries it.
#[cfg(feature = "profile")]
pub fn pipeline_snapshot_and_reset() -> ProfileSnapshot {
    store::snapshot_and_reset_all()
}

/// Returns empty snapshot when profiling is disabled.
#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn pipeline_snapshot_and_reset() -> ProfileSnapshot {
    ProfileSnapshot::default()
}
