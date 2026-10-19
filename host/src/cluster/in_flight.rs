//! The requests a gateway sent over this member's sockets that are still
//! running, so a member leaving its clusters closes them only once those have
//! finished (`docs/replicated_substrate.md` §12.4).
//!
//! Draining stops a gateway routing anything new here, but the sockets carry
//! the requests already running — a turn streaming its reply among them — and
//! closing a socket cuts every one of them mid-stream. Only a request that can
//! append is counted ([`writes_records`](super::flush::writes_records)): a log
//! tail or a follow of a live reply is a `GET` that never ends on its own, and
//! waiting on one would hold every shutdown to the limit.
//!
//! A request is in flight until its response body ends or is dropped — the
//! end of a stream of server-sent events, not the return of its head
//! ([`super::held_body::HeldBody`] holds its [`Running`] that long).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::time::{sleep, Instant};

/// How often a wait for the requests in flight looks again.
const LOOK: Duration = Duration::from_millis(100);

/// The longest a member leaving waits for its requests in flight: a turn still
/// running past it is cut when the sockets close.
pub const IN_FLIGHT_WAIT: Duration = Duration::from_secs(300);

/// Requests served off this member's sockets that are still running.
#[derive(Clone, Default)]
pub struct InFlight(Arc<AtomicUsize>);

/// One request in flight, until dropped.
pub struct Running(Arc<AtomicUsize>);

impl Drop for Running {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl InFlight {
    /// A request starts.
    pub fn enter(&self) -> Running {
        self.0.fetch_add(1, Ordering::AcqRel);
        Running(Arc::clone(&self.0))
    }

    pub fn count(&self) -> usize {
        self.0.load(Ordering::Acquire)
    }

    /// Wait until none is running, or at most `limit`, telling `waiting` how
    /// many still are each time it looks. Returns how many were still running
    /// when it stopped.
    pub async fn settle(&self, limit: Duration, mut waiting: impl FnMut(usize)) -> usize {
        let deadline = Instant::now() + limit;
        loop {
            let left = self.count();
            if left == 0 || Instant::now() >= deadline {
                return left;
            }
            waiting(left);
            sleep(LOOK).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A request counts from its start until its guard goes.
    #[test]
    fn a_request_counts_until_it_ends() {
        let f = InFlight::default();
        let a = f.enter();
        let b = f.enter();
        assert_eq!(f.count(), 2);
        drop(a);
        assert_eq!(f.count(), 1);
        drop(b);
        assert_eq!(f.count(), 0);
    }

    /// The wait returns at once with nothing running, and at the limit with
    /// what is still running, reporting it while it waits.
    #[tokio::test(start_paused = true)]
    async fn the_wait_ends_when_nothing_runs_or_at_its_limit() {
        let f = InFlight::default();
        assert_eq!(f.settle(Duration::from_secs(5), |_| {}).await, 0);
        let running = f.enter();
        let mut seen = Vec::new();
        let left = f.settle(Duration::from_millis(250), |n| seen.push(n)).await;
        assert_eq!(left, 1);
        assert_eq!(seen, vec![1, 1, 1]);
        drop(running);
        assert_eq!(f.settle(Duration::from_secs(5), |_| {}).await, 0);
    }
}
