//! Cooperative ingest cancellation, one latch per substrate.
//!
//! The startup ingest, the background reconcile, and uploads all drive the same
//! engine (scheduler + persistence). A graceful shutdown mid-ingest must stop
//! that ingest at a safe boundary so the persistence thread can drain the tier
//! tail hot→warm→cold instead of losing it. A [`CancelToken`] is the signal.
//!
//! **Scope.** One token per substrate: each [`crate::projection::Conversation`]
//! carries the token its engine attachment was built with, so stopping one
//! tenant's ingest leaves another tenant on the same engine running. A daemon
//! creates its token before the engine exists and hands it to the builder
//! ([`crate::models::ModelBuilder::cancel_token`]), so a shutdown that arrives
//! while the engine is still loading reaches the ingest the load is about to
//! start. The token is **resettable** ([`CancelToken::reset`]) so a fresh load
//! never inherits a prior run's cancel.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};

/// A cancellation latch shared by clones.
#[derive(Clone, Default)]
pub struct CancelToken {
    inner: Arc<Latch>,
}

#[derive(Default)]
struct Latch {
    cancelled: AtomicBool,
    /// Every async waiter parked on the latch, keyed by its future's identity
    /// so a re-poll replaces the stored waker instead of accumulating clones,
    /// and a dropped future removes its own entry instead of leaking its task
    /// reference for the life of the token.
    wakers: Mutex<Vec<(u64, Waker)>>,
    next_waiter: AtomicU64,
}

impl std::fmt::Debug for CancelToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancelToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Ask every in-flight ingest on this token to stop at its next file /
    /// cluster / scope-chunk boundary (and the in-flight summary decode to
    /// abandon its wait). Idempotent.
    pub fn request(&self) {
        self.inner.cancelled.store(true, Ordering::SeqCst);
        // After the store, so a waiter woken here re-reads the flag as latched
        // — and woken OUTSIDE the registry lock, because a wake may
        // synchronously drop or re-poll another waiter, and both of those take
        // the same lock.
        let woken: Vec<(u64, Waker)> = self
            .inner
            .wakers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
            .collect();
        for (_, w) in woken {
            w.wake();
        }
    }

    /// Clear the latch so a new load starts un-cancelled.
    pub fn reset(&self) {
        self.inner.cancelled.store(false, Ordering::SeqCst);
    }

    /// Whether a shutdown has asked ingest on this token to stop. A `Relaxed`
    /// load is sufficient — this is a one-way latch (until [`Self::reset`]),
    /// not a synchronisation point.
    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Relaxed)
    }

    /// Resolves when [`Self::request`] latches — the awaitable face of the
    /// latch, for the async ingest waits that have no timer to poll it on
    /// ([`crate::TurnHandle::wait_cancellable_async`]). Already latched
    /// resolves immediately.
    pub fn wait(&self) -> CancelWait {
        CancelWait {
            token: self.clone(),
            id: self.inner.next_waiter.fetch_add(1, Ordering::Relaxed),
        }
    }
}

/// See [`CancelToken::wait`].
pub struct CancelWait {
    token: CancelToken,
    id: u64,
}

impl Future for CancelWait {
    type Output = ();
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<()> {
        if self.token.is_cancelled() {
            return Poll::Ready(());
        }
        let mut wakers = self
            .token
            .inner
            .wakers
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // Re-check under the lock: a cancel between the read above and this
        // registration has already drained the list, so registering after it
        // would park this waiter forever.
        if self.token.is_cancelled() {
            return Poll::Ready(());
        }
        match wakers.iter_mut().find(|(id, _)| *id == self.id) {
            Some((_, w)) => w.clone_from(cx.waker()),
            None => wakers.push((self.id, cx.waker().clone())),
        }
        Poll::Pending
    }
}

impl Drop for CancelWait {
    fn drop(&mut self) {
        self.token
            .inner
            .wakers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|(id, _)| *id != self.id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_already_latched_cancel_resolves_immediately() {
        let token = CancelToken::new();
        token.request();
        futures::executor::block_on(token.wait());
    }

    #[test]
    fn a_parked_waiter_is_woken_by_the_cancel() {
        let token = CancelToken::new();
        let waiting = token.clone();
        let waited = std::thread::spawn(move || {
            futures::executor::block_on(waiting.wait());
        });
        // Let the waiter park before latching — registration is what is under
        // test, and this token's registry is nobody else's.
        while token.inner.wakers.lock().unwrap().is_empty() && !waited.is_finished() {
            std::thread::yield_now();
        }
        token.request();
        waited.join().expect("the waiter returned");
    }

    #[test]
    fn a_dropped_waiter_leaves_no_waker_behind() {
        let token = CancelToken::new();
        let wait = token.wait();
        {
            // Park it once so it registers.
            let mut wait = Box::pin(wait);
            let waker = futures::task::noop_waker();
            let mut cx = Context::from_waker(&waker);
            assert!(wait.as_mut().poll(&mut cx).is_pending());
            assert_eq!(token.inner.wakers.lock().unwrap().len(), 1);
        }
        assert!(
            token.inner.wakers.lock().unwrap().is_empty(),
            "the drop did not unregister the waker"
        );
    }

    /// Two substrates' tokens are independent: stopping one tenant's ingest
    /// does not stop the other's.
    #[test]
    fn one_tokens_cancel_does_not_reach_another() {
        let zend = CancelToken::new();
        let npcd = CancelToken::new();
        zend.request();
        assert!(zend.is_cancelled());
        assert!(!npcd.is_cancelled());
        zend.reset();
        assert!(!zend.is_cancelled());
    }
}
