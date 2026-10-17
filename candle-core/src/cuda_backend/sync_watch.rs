//! A blocking stream sync that names what it is stuck behind.
//!
//! A kernel that never finishes leaves every thread that synchronises parked in
//! the driver, and nothing in the log says which kernel it was: measured on
//! `.5`, 2026-10-16, the scheduler sat four minutes in `cuStreamSynchronize`
//! behind a GPU at 100% and 114 W with no line written. So each sync marks its
//! thread as waiting, and one watchdog thread logs a wait that passes
//! [`OVERDUE`] — the thread, how long, and the recent kernel launches of every
//! thread, newest first — once per wait. The sync itself stays the driver's
//! blocking call: polling would add latency to every decode step.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use cudarc::driver::{CudaStream, DriverError};

use super::every_thread_kernel_launches;

/// How long a sync may block before the watchdog reports it.
const OVERDUE: Duration = Duration::from_secs(10);

/// One thread's wait: milliseconds since [`epoch`] it began, 0 when not waiting.
struct Wait {
    thread: String,
    since_ms: AtomicU64,
    reported: AtomicBool,
}

static WAITS: Mutex<Vec<Arc<Wait>>> = Mutex::new(Vec::new());

thread_local! {
    static WAIT: Arc<Wait> = {
        let wait = Arc::new(Wait {
            thread: std::thread::current().name().unwrap_or("unnamed").to_string(),
            since_ms: AtomicU64::new(0),
            reported: AtomicBool::new(false),
        });
        WAITS.lock().unwrap_or_else(|p| p.into_inner()).push(Arc::clone(&wait));
        wait
    };
}

fn epoch() -> Instant {
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    *EPOCH.get_or_init(Instant::now)
}

/// Milliseconds since [`epoch`], never 0 (0 means "not waiting").
fn now_ms() -> u64 {
    epoch().elapsed().as_millis() as u64 + 1
}

/// Whether a wait that began at `since_ms` is overdue at `now_ms`.
fn overdue(since_ms: u64, now_ms: u64) -> bool {
    since_ms != 0 && now_ms.saturating_sub(since_ms) >= OVERDUE.as_millis() as u64
}

fn start_watchdog() {
    static STARTED: OnceLock<()> = OnceLock::new();
    STARTED.get_or_init(|| {
        let spawned = std::thread::Builder::new()
            .name("cuda-sync-watchdog".into())
            .spawn(|| loop {
                std::thread::sleep(Duration::from_secs(1));
                let now = now_ms();
                // Dead threads' waits are dropped here, so the list stays the
                // live threads that have synchronised.
                let waits: Vec<Arc<Wait>> = {
                    let mut w = WAITS.lock().unwrap_or_else(|p| p.into_inner());
                    w.retain(|x| Arc::strong_count(x) > 1);
                    w.clone()
                };
                for w in waits {
                    let since = w.since_ms.load(Ordering::Relaxed);
                    if overdue(since, now) && !w.reported.swap(true, Ordering::Relaxed) {
                        tracing::warn!(
                            thread = %w.thread,
                            waited_s = (now - since) / 1000,
                            "a CUDA stream sync has been blocked past {}s — a kernel on the \
                             stream is not finishing\n{}",
                            OVERDUE.as_secs(),
                            every_thread_kernel_launches()
                        );
                    }
                }
            });
        if let Err(e) = spawned {
            tracing::error!(error = %e, "could not start the CUDA sync watchdog");
        }
    });
}

/// `stream.synchronize()`, watched: a wait past [`OVERDUE`] is reported while
/// it blocks, and again with its total once it returns.
pub(crate) fn synchronize(stream: &CudaStream) -> std::result::Result<(), DriverError> {
    start_watchdog();
    let begun = now_ms();
    WAIT.with(|w| {
        w.reported.store(false, Ordering::Relaxed);
        w.since_ms.store(begun, Ordering::Relaxed);
    });
    let out = stream.synchronize();
    let reported = WAIT.with(|w| {
        w.since_ms.store(0, Ordering::Relaxed);
        w.reported.load(Ordering::Relaxed)
    });
    if reported {
        tracing::warn!(
            waited_ms = now_ms() - begun,
            "the overdue CUDA stream sync returned"
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{overdue, OVERDUE};

    /// A wait is overdue from `OVERDUE` on, and a thread not waiting never is.
    #[test]
    fn a_wait_is_overdue_past_the_threshold() {
        let t = OVERDUE.as_millis() as u64;
        assert!(!overdue(0, 10 * t), "not waiting");
        assert!(!overdue(5, 5 + t - 1));
        assert!(overdue(5, 5 + t));
        assert!(!overdue(50, 10), "a clock read before the wait began");
    }
}
