//! The daemon's single background ingest worker — the ONLY thing that runs
//! an ingest pass after startup.
//!
//! Before this, a filesystem-event burst (`crate::watcher`) and the startup
//! background reconcile could both call `InferenceState::refresh_ingest_layers`
//! at once, running two overlapping `run_dir_pool`/`run_file_pool` calls.
//! `repo_scan`'s process-global pricing statics (`SCAN_KV_BASELINE` /
//! `SCAN_LIVE_CONVS`, re-anchored once per pool call on the assumption that
//! the call is one fresh, non-overlapping pass) and `ingest_report`'s
//! last-write-wins publish are both sound only when passes never overlap.
//! Serialising every pass behind one worker task is what makes that true.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;
use tokio::task::JoinHandle;

/// A running ingest worker. Dropping this without calling [`stop`](Self::stop)
/// leaves the task running detached — always call `stop` during shutdown.
pub struct IngestWorker {
    wake: Arc<Notify>,
    stop: Arc<AtomicBool>,
    task: JoinHandle<()>,
}

/// Spawn the worker on the ambient Tokio runtime. It sleeps until woken, then
/// runs one `pass()`. The first pass to complete is followed by
/// `after_first_pass()` (the ingest normalization warm-up) run inline, before
/// the worker loops back to waiting — which is what guarantees the warm-up
/// can never overlap a pass without any extra locking.
///
/// `wake` is supplied by the caller (rather than returned only on
/// [`IngestWorker`]) so the watcher's debounced callback can hold its own
/// clone and wake the worker without reaching back through it.
/// `tokio::sync::Notify::notify_one` stores one permit when nothing is
/// waiting, so a wake that lands before the worker's first `.await`, or while
/// a pass is already running, is coalesced rather than lost.
///
/// `pass` and `after_first_pass` are blocking (worker-pool spin-up, GPU
/// prefill/decode) exactly like the watcher's own callback, so each runs on
/// `spawn_blocking` rather than the async task itself.
pub fn spawn(
    wake: Arc<Notify>,
    pass: Arc<dyn Fn() + Send + Sync + 'static>,
    after_first_pass: Arc<dyn Fn() + Send + Sync + 'static>,
) -> IngestWorker {
    let stop = Arc::new(AtomicBool::new(false));
    let wake_loop = Arc::clone(&wake);
    let stop_loop = Arc::clone(&stop);
    let task = tokio::spawn(async move {
        let mut first = true;
        loop {
            wake_loop.notified().await;
            if stop_loop.load(Ordering::Relaxed) || candle_conversation::ingest_cancelled() {
                break;
            }
            let started = std::time::Instant::now();
            let p = Arc::clone(&pass);
            if tokio::task::spawn_blocking(move || p()).await.is_err() {
                tracing::error!("ingest worker: pass panicked");
            }
            tracing::info!(
                elapsed_ms = started.elapsed().as_millis() as u64,
                "ingest worker: pass complete",
            );
            if first {
                first = false;
                // Ordering invariant: the warm-up's heavy self-match scan must
                // run AFTER ingest, never during (it freezes the scheduler).
                // Running it here, between this pass and the next
                // `notified().await`, guarantees that without a lock — a wake
                // that arrives during the warm-up is stored as a `Notify`
                // permit and serviced once this returns. Skipped on a
                // shutdown cancel so Ctrl-C isn't held up by a 1-2 minute scan.
                if !stop_loop.load(Ordering::Relaxed) && !candle_conversation::ingest_cancelled() {
                    let w = Arc::clone(&after_first_pass);
                    let _ = tokio::task::spawn_blocking(move || w()).await;
                }
            }
        }
    });
    IngestWorker { wake, stop, task }
}

impl IngestWorker {
    /// Ask for a pass. Cheap and synchronous — callable from anywhere,
    /// including the watcher's blocking-pool callback.
    pub fn wake(&self) {
        self.wake.notify_one();
    }

    /// Stop after the in-flight pass reaches its next unit boundary, and join.
    ///
    /// Never `abort()`: Tokio cannot cancel the `spawn_blocking` closure a
    /// pass runs on, so aborting would return here while a pool is still
    /// minting conversations and writing turns — breaking the daemon
    /// shutdown's "the drain runs on the loader thread OR here, never both"
    /// invariant. The pools already stop cooperatively at a unit boundary on
    /// `candle_conversation::ingest_cancelled()` (the caller is expected to
    /// have raised that before calling `stop`), which is what bounds this
    /// await.
    pub async fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        self.wake.notify_one();
        if self.task.await.is_err() {
            tracing::warn!("ingest worker: task panicked during shutdown");
        }
    }
}
