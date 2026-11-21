//! A canary on tokio's blocking pool: every [`EVERY`] it runs a task that does
//! nothing there and times how long the pool took to start it.
//!
//! **What it is for.** A pool with every thread stuck runs nothing queued
//! behind them — the synchronous tools, keep's park, the work tenants hand to
//! it — while the async routes, `/v1/status` among them, answer as ever. On
//! WhiteWorld (2026-11-20) `/v1/agent/hold`, whose last step parks the build
//! queue on keep through the pool, and `/v1/logs` hung for an hour beside a
//! `/v1/status` answering in 3 ms, and the process was killed before anything
//! named the wait. The canary names it: `blocking_pool` in `/v1/status`, which
//! a wedge does not reach, and a warning on the log every [`WARN_EVERY`] the
//! pool goes without running it.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use serde::Serialize;

/// How often the canary runs.
pub const EVERY: Duration = Duration::from_secs(5);

/// A start this slow is logged.
pub const SLOW: Duration = Duration::from_secs(2);

/// While the canary waits, a warning this often.
pub const WARN_EVERY: Duration = Duration::from_secs(10);

/// `0` for "none".
struct Canary {
    /// Milliseconds after [`origin`] the waiting canary was queued, plus one.
    waiting_since: AtomicU64,
    /// The last start's latency in microseconds, plus one.
    last_us: AtomicU64,
}

static CANARY: Canary = Canary {
    waiting_since: AtomicU64::new(0),
    last_us: AtomicU64::new(0),
};

fn origin() -> Instant {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    *ORIGIN.get_or_init(Instant::now)
}

fn now_ms() -> u64 {
    origin().elapsed().as_millis() as u64
}

/// The pool as `/v1/status` shows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct PoolView {
    /// How long the last canary waited to start, in milliseconds.
    pub last_start_ms: Option<u64>,
    /// How long the canary now queued has waited, in milliseconds: a pool
    /// running nothing.
    pub waiting_ms: Option<u64>,
}

/// The view from the canary's stored figures, at `now` milliseconds.
fn view_of(waiting_since: u64, last_us: u64, now: u64) -> PoolView {
    PoolView {
        last_start_ms: (last_us > 0).then(|| (last_us - 1) / 1000),
        waiting_ms: (waiting_since > 0).then(|| now.saturating_sub(waiting_since - 1)),
    }
}

/// The pool now.
pub fn view() -> PoolView {
    view_of(
        CANARY.waiting_since.load(Ordering::Acquire),
        CANARY.last_us.load(Ordering::Acquire),
        now_ms(),
    )
}

/// Start the canary on the current runtime.
pub fn start() {
    origin();
    tokio::spawn(async {
        let mut every = tokio::time::interval(EVERY);
        loop {
            every.tick().await;
            run_once().await;
        }
    });
}

/// One canary: queued, waited for with a warning every [`WARN_EVERY`], its
/// latency stored.
async fn run_once() {
    let queued = Instant::now();
    CANARY.waiting_since.store(now_ms() + 1, Ordering::Release);
    let started = tokio::task::spawn_blocking(Instant::now);
    tokio::pin!(started);
    let mut warn = tokio::time::interval_at(tokio::time::Instant::now() + WARN_EVERY, WARN_EVERY);
    let started = loop {
        tokio::select! {
            at = &mut started => break at.ok(),
            _ = warn.tick() => tracing::warn!(
                waited_s = queued.elapsed().as_secs(),
                "the blocking pool has started nothing for {} s: every task queued on it — the \
                 synchronous tools, keep's park, the tenants' blocking work — waits",
                queued.elapsed().as_secs()
            ),
        }
    };
    CANARY.waiting_since.store(0, Ordering::Release);
    let Some(at) = started else {
        return;
    };
    let latency = at.saturating_duration_since(queued);
    CANARY
        .last_us
        .store(latency.as_micros() as u64 + 1, Ordering::Release);
    if latency >= SLOW {
        tracing::warn!(
            waited_ms = latency.as_millis() as u64,
            "the blocking pool took {} ms to start a task",
            latency.as_millis()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The view says how long the last start took and how long the one now
    /// queued has waited**, and nothing it has not seen.
    #[test]
    fn the_view_reads_the_stored_figures() {
        assert_eq!(
            view_of(0, 0, 9_000),
            PoolView {
                last_start_ms: None,
                waiting_ms: None
            }
        );
        assert_eq!(
            view_of(0, 2_500 + 1, 9_000),
            PoolView {
                last_start_ms: Some(2),
                waiting_ms: None
            }
        );
        assert_eq!(
            view_of(4_000 + 1, 1, 9_000),
            PoolView {
                last_start_ms: Some(0),
                waiting_ms: Some(5_000)
            }
        );
    }

    /// **A free pool starts the canary at once; one whose only thread is held
    /// shows the canary waiting** until the thread is let go, then the wait as
    /// the start's latency. The two in one test: they share the process's
    /// canary.
    #[test]
    fn a_held_pool_shows_the_canary_waiting() {
        let rt = tokio::runtime::Builder::new_current_thread()
            .max_blocking_threads(1)
            .enable_time()
            .build()
            .unwrap();
        rt.block_on(async {
            run_once().await;
            assert_eq!(view().waiting_ms, None);
            assert!(view().last_start_ms.is_some_and(|ms| ms < 1_000));

            let (release, held) = std::sync::mpsc::channel::<()>();
            let holder = tokio::task::spawn_blocking(move || held.recv());
            let canary = tokio::spawn(run_once());
            tokio::time::sleep(Duration::from_millis(300)).await;
            assert!(
                view().waiting_ms.is_some_and(|ms| ms >= 200),
                "the canary waits behind the held thread: {:?}",
                view()
            );
            release.send(()).unwrap();
            holder.await.unwrap().unwrap();
            canary.await.unwrap();
            assert_eq!(view().waiting_ms, None);
            assert!(view().last_start_ms.is_some_and(|ms| ms >= 200));
        });
    }
}
