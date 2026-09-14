//! Cooperative ingest cancellation — an engine-layer concern.
//!
//! The startup ingest, the background reconcile, and uploads all drive the same
//! engine (scheduler + persistence). A graceful shutdown mid-ingest must stop
//! that ingest at a safe boundary so the persistence thread can drain the tier
//! tail hot→warm→cold instead of losing it. This flag is the signal.
//!
//! It lives HERE (not in the app/daemon) for two reasons: the ingest and the
//! summary-decode wait it needs to interrupt are engine internals
//! ([`crate::Sequence::ingest_scope_roundtrip_indices`]), and the flag must be
//! reachable while the engine is still being constructed on the loader thread —
//! before any [`ConversationEngine`](crate::ConversationEngine) handle is
//! published — which a free function (no engine reference) provides.
//!
//! **Scope & lifecycle.** The flag is process-scoped (one latch for the whole
//! process, not per-engine), so a caller that runs more than one engine — or the
//! test binary, which runs many — must treat it as shared. It is **resettable**:
//! [`reset_ingest_cancel`] clears it at the start of a load so a fresh load (or
//! the next test) never inherits a prior run's cancel, and an embedder can
//! re-arm ingest after a cancelled one.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::task::{Context, Poll, Waker};

static INGEST_CANCEL: AtomicBool = AtomicBool::new(false);

/// Every async waiter parked on the latch, keyed by its future's identity so a
/// re-poll replaces the stored waker instead of accumulating clones, and a
/// dropped future removes its own entry instead of leaking its task reference
/// for the life of the process.
static CANCEL_WAKERS: Mutex<Vec<(u64, Waker)>> = Mutex::new(Vec::new());
static NEXT_WAITER: AtomicU64 = AtomicU64::new(0);

/// Ask every in-flight ingest to stop at its next file / cluster / scope-chunk
/// boundary (and the in-flight summary decode to abandon its wait). Idempotent.
pub fn request_ingest_cancel() {
    INGEST_CANCEL.store(true, Ordering::SeqCst);
    // After the store, so a waiter woken here re-reads the flag as latched —
    // and woken OUTSIDE the registry lock, because a wake may synchronously
    // drop or re-poll another waiter, and both of those take the same lock.
    let woken: Vec<(u64, Waker)> = CANCEL_WAKERS.lock().unwrap().drain(..).collect();
    for (_, w) in woken {
        w.wake();
    }
}

/// Resolves when [`request_ingest_cancel`] latches — the awaitable face of the
/// same latch [`ingest_cancelled`] polls, for the async ingest waits that have
/// no timer to poll it on ([`crate::TurnHandle::wait_cancellable_async`]).
/// Already-latched resolves immediately.
pub fn ingest_cancel_wait() -> IngestCancelWait {
    IngestCancelWait {
        id: NEXT_WAITER.fetch_add(1, Ordering::Relaxed),
    }
}

/// See [`ingest_cancel_wait`].
pub struct IngestCancelWait {
    id: u64,
}

impl Future for IngestCancelWait {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if ingest_cancelled() {
            return Poll::Ready(());
        }
        let mut wakers = CANCEL_WAKERS.lock().unwrap();
        // Re-check under the lock: a cancel between the read above and this
        // registration has already drained the list, so registering after it
        // would park this waiter forever.
        if ingest_cancelled() {
            return Poll::Ready(());
        }
        match wakers.iter_mut().find(|(id, _)| *id == self.id) {
            Some((_, w)) => w.clone_from(cx.waker()),
            None => wakers.push((self.id, cx.waker().clone())),
        }
        Poll::Pending
    }
}

impl Drop for IngestCancelWait {
    fn drop(&mut self) {
        CANCEL_WAKERS
            .lock()
            .unwrap()
            .retain(|(id, _)| *id != self.id);
    }
}

/// Clear the cancel latch so a new load starts un-cancelled. Called at the start
/// of a load (and usable by tests / embedders to re-arm ingest after a prior
/// cancel), since the flag is process-scoped and would otherwise persist across
/// loads within one process.
pub fn reset_ingest_cancel() {
    INGEST_CANCEL.store(false, Ordering::SeqCst);
}

/// Whether a shutdown has asked ingest to stop. Polled in the ingest item loops
/// and the interruptible decode-wait. A `Relaxed` load is sufficient — this is a
/// one-way latch (until [`reset_ingest_cancel`]), not a synchronisation point.
pub fn ingest_cancelled() -> bool {
    INGEST_CANCEL.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The flag is process-scoped, so these tests share it with each other and
    /// with any ingest test in the binary — each one leaves it cleared.

    #[test]
    fn an_already_latched_cancel_resolves_immediately() {
        request_ingest_cancel();
        futures::executor::block_on(ingest_cancel_wait());
        reset_ingest_cancel();
    }

    #[test]
    fn a_parked_waiter_is_woken_by_the_cancel() {
        reset_ingest_cancel();
        let waited = std::thread::spawn(|| {
            futures::executor::block_on(ingest_cancel_wait());
        });
        // Let the waiter park before latching — registration is what is under
        // test. The flag is process-scoped, so a parallel test latching it can
        // resolve the waiter before it ever registers; that finished thread is
        // the other exit from this spin.
        while CANCEL_WAKERS.lock().unwrap().is_empty() && !waited.is_finished() {
            std::thread::yield_now();
        }
        request_ingest_cancel();
        waited.join().expect("the waiter returned");
        reset_ingest_cancel();
    }

    #[test]
    fn a_dropped_waiter_leaves_no_waker_behind() {
        reset_ingest_cancel();
        let wait = ingest_cancel_wait();
        let id = wait.id;
        {
            // Park it once so it registers.
            let mut wait = Box::pin(wait);
            let waker = futures::task::noop_waker();
            let mut cx = Context::from_waker(&waker);
            assert!(wait.as_mut().poll(&mut cx).is_pending());
            assert!(CANCEL_WAKERS.lock().unwrap().iter().any(|(i, _)| *i == id));
        }
        assert!(
            !CANCEL_WAKERS.lock().unwrap().iter().any(|(i, _)| *i == id),
            "the drop did not unregister the waker"
        );
    }
}
