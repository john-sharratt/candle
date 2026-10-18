//! How many bytes a pipe's sender has handed the transport that the peer has
//! not taken, and a wake for whoever waits on that count to fall.
//!
//! The session writer sends catch-up frames only while the socket holds less
//! than its low-water mark unsent (§4.2), so it waits on [`Drain::below`]
//! rather than polling. A poll on a timer paced catch-up by the timer: tokio's
//! timer on Windows ticks at ~15.6 ms, so a 1 ms poll let about one low-water
//! mark out per tick — ~64 MB/s on a link that carries twice that, and a
//! member joining `.5` caught up 35 GB at 30–43 MB/s (2026-10-18).

use std::pin::pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;

/// A sender's unsent byte count; clones share it.
#[derive(Clone, Default)]
pub struct Drain {
    unsent: Arc<AtomicUsize>,
    fell: Arc<Notify>,
}

impl Drain {
    pub fn new() -> Drain {
        Drain::default()
    }

    /// `n` more bytes handed to the transport.
    pub fn add(&self, n: usize) {
        self.unsent.fetch_add(n, Ordering::SeqCst);
    }

    /// `n` bytes the peer has taken.
    pub fn sub(&self, n: usize) {
        self.unsent.fetch_sub(n, Ordering::SeqCst);
        self.fell.notify_waiters();
    }

    pub fn unsent(&self) -> usize {
        self.unsent.load(Ordering::SeqCst)
    }

    /// Resolves once fewer than `n` bytes are unsent: at once if they already
    /// are, else at the first fall that takes the count under `n`.
    pub async fn below(&self, n: usize) {
        loop {
            let mut fell = pin!(self.fell.notified());
            // Registered before the count is read, so a fall between the read
            // and the wait is not missed.
            fell.as_mut().enable();
            if self.unsent() < n {
                return;
            }
            fell.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    use tokio::time::timeout;

    /// `below` resolves at once under the mark, waits above it, and wakes on
    /// the fall that crosses it — not on one that leaves the count above.
    #[tokio::test]
    async fn below_wakes_on_the_fall_that_crosses_the_mark() {
        let d = Drain::new();
        d.below(10).await;
        d.add(30);
        let waiter = tokio::spawn({
            let d = d.clone();
            async move { d.below(10).await }
        });
        tokio::task::yield_now().await;
        d.sub(5);
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished(), "25 unsent is not under 10");
        d.sub(20);
        timeout(Duration::from_secs(1), waiter)
            .await
            .expect("woken by the fall to 5")
            .unwrap();
        assert_eq!(d.unsent(), 5);
    }
}
