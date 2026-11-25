//! The daemon's single background ingest worker — the ONLY thing that runs
//! an ingest pass after startup.
//!
//! Several things ask for a pass — startup, a record branch moving in the log
//! (`crate::substrate_git`), a tool round that published a branch — and any
//! of them can ask while another pass runs. `repo_scan`'s process-global
//! pricing statics (`SCAN_KV_BASELINE` / `SCAN_LIVE_CONVS`, re-anchored once
//! per pool call on the assumption that the call is one fresh,
//! non-overlapping pass) and `ingest_report`'s last-write-wins publish are
//! both sound only when passes never overlap. Serialising every pass behind
//! one worker task is what makes that true.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;
use tokio::task::JoinHandle;

/// Raised by every request for a pass, lowered as a pass begins: a request
/// that arrived while one runs. The layers built on others read it between
/// units ([`preempted`]) and stop, so the pass ends and the next one starts
/// from the bottom — a moved branch is read into the layers that read the
/// branches before any layer above them reads another unit.
static PREEMPT: AtomicBool = AtomicBool::new(false);

/// Raised once the process's first pass and its warm-up are done: until then
/// the backlog and the warm-up's counts are empty only because nothing has
/// been counted yet, and the memory is not settled (`crate::settled`).
static FIRST_PASS_WARMED: AtomicBool = AtomicBool::new(false);

/// Whether the process's first ingest pass and its warm-up have finished.
pub fn first_pass_warmed() -> bool {
    FIRST_PASS_WARMED.load(Ordering::SeqCst)
}

/// Ask for a pass through `wake`: what every caller holding the worker's
/// `Notify` calls, so a pass in flight learns it is wanted again.
pub fn request(wake: &Notify) {
    PREEMPT.store(true, Ordering::SeqCst);
    wake.notify_one();
}

/// Whether a pass was asked for since this one began.
pub fn preempted() -> bool {
    PREEMPT.load(Ordering::SeqCst)
}

/// A running ingest worker. Dropping this without calling [`stop`](Self::stop)
/// leaves the task running detached — always call `stop` during shutdown.
pub struct IngestWorker {
    wake: Arc<Notify>,
    stop: Arc<AtomicBool>,
    task: JoinHandle<()>,
}

/// Spawn the worker on the ambient Tokio runtime. It sleeps until woken, then
/// runs one `pass()`, which answers whether it changed a layer. The first pass
/// to complete, and every later pass that changed a layer, is followed by
/// `warm()` (the ingest normalization warm-up, which warms only the files not
/// warmed before) run inline, before the worker loops back to waiting — which
/// is what guarantees the warm-up can never overlap a pass without any extra
/// locking. A file a later pass read is a new timeline with cold backgrounds:
/// warmed only after the first pass, it stayed read against the pool until
/// live traffic happened to teach it.
///
/// `wake` is supplied by the caller (rather than returned only on
/// [`IngestWorker`]) so the origin watchers and the session can each hold a
/// clone and wake the worker without reaching back through it.
/// `tokio::sync::Notify::notify_one` stores one permit when nothing is
/// waiting, so a wake that lands before the worker's first `.await`, or while
/// a pass is already running, is coalesced rather than lost.
///
/// `pass` and `warm` are blocking (git reads, worker-pool spin-up, GPU
/// prefill/decode), so each runs on `spawn_blocking` rather than the async
/// task itself.
pub fn spawn(
    wake: Arc<Notify>,
    pass: Arc<dyn Fn() -> bool + Send + Sync + 'static>,
    warm: Arc<dyn Fn() + Send + Sync + 'static>,
) -> IngestWorker {
    let stop = Arc::new(AtomicBool::new(false));
    let wake_loop = Arc::clone(&wake);
    let stop_loop = Arc::clone(&stop);
    let task = tokio::spawn(async move {
        let mut first = true;
        loop {
            wake_loop.notified().await;
            if stop_loop.load(Ordering::Relaxed) || crate::ingest_cancel::cancelled() {
                break;
            }
            PREEMPT.store(false, Ordering::SeqCst);
            let started = std::time::Instant::now();
            let p = Arc::clone(&pass);
            let changed = match tokio::task::spawn_blocking(move || p()).await {
                Ok(changed) => changed,
                Err(_) => {
                    tracing::error!("ingest worker: pass panicked");
                    false
                }
            };
            tracing::info!(
                elapsed_ms = started.elapsed().as_millis() as u64,
                changed,
                "ingest worker: pass complete",
            );
            if warms_after(first, changed) {
                let was_first = first;
                first = false;
                // Ordering invariant: the warm-up's heavy self-match scan must
                // run AFTER ingest, never during (it freezes the scheduler).
                // Running it here, between this pass and the next
                // `notified().await`, guarantees that without a lock — a wake
                // that arrives during the warm-up is stored as a `Notify`
                // permit and serviced once this returns. Skipped on a
                // shutdown cancel so Ctrl-C isn't held up by the scan.
                if !stop_loop.load(Ordering::Relaxed) && !crate::ingest_cancel::cancelled() {
                    let w = Arc::clone(&warm);
                    let _ = tokio::task::spawn_blocking(move || w()).await;
                    if was_first {
                        FIRST_PASS_WARMED.store(true, Ordering::SeqCst);
                    }
                }
            }
        }
    });
    IngestWorker { wake, stop, task }
}

/// Whether the normalization warm-up follows a pass: the process's first,
/// whatever it changed, and any later one that changed a layer.
fn warms_after(first: bool, changed: bool) -> bool {
    first || changed
}

impl IngestWorker {
    /// Ask for a pass. Cheap and synchronous — callable from anywhere,
    /// including a blocking-pool callback.
    pub fn wake(&self) {
        request(&self.wake);
    }

    /// Stop after the in-flight pass reaches its next unit boundary, and join.
    ///
    /// Never `abort()`: Tokio cannot cancel the `spawn_blocking` closure a
    /// pass runs on, so aborting would return here while a pool is still
    /// minting conversations and writing turns — breaking the daemon
    /// shutdown's "the drain runs on the loader thread OR here, never both"
    /// invariant. The pools already stop cooperatively at a unit boundary on
    /// `crate::ingest_cancel::cancelled()` (the caller is expected to
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

#[cfg(test)]
mod tests {
    use super::warms_after;

    /// **The warm-up follows the first pass and every pass that changed a
    /// layer**, and no other: a later pass that read nothing new has nothing
    /// to warm.
    #[test]
    fn the_warm_up_follows_the_first_pass_and_every_change() {
        assert!(warms_after(true, false));
        assert!(warms_after(true, true));
        assert!(warms_after(false, true));
        assert!(!warms_after(false, false));
    }
}
