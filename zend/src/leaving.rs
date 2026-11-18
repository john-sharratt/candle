//! A member leaving its clusters: the signal every live turn hears, and the
//! count of turns the leave waits on (`docs/turn_handover.md` §4).
//!
//! A turn holds a [`Running`] for as long as it may still write its
//! conversation — its decode, and for a worker the record it writes after. A
//! leave raises the signal and waits for the count to reach zero, so what the
//! turns write on their way out is written before the member's claims go.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::watch;
use tokio::time::{sleep, Instant};

/// How long a leave waits for its turns to stop.
pub const LEAVE_TURNS_WAIT: Duration = Duration::from_secs(30);

/// How often a leave looks at the count while it waits.
const LOOK: Duration = Duration::from_millis(50);

/// The leaving signal and the turns still running.
pub struct Leaving {
    tx: watch::Sender<bool>,
    running: Arc<AtomicUsize>,
}

/// One turn still running, until dropped.
pub struct Running(Arc<AtomicUsize>);

impl Drop for Running {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

impl Default for Leaving {
    fn default() -> Self {
        Self {
            tx: watch::channel(false).0,
            running: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Leaving {
    /// What a turn watches: it changes to `true` once, when the member leaves.
    pub fn watch(&self) -> watch::Receiver<bool> {
        self.tx.subscribe()
    }

    /// Whether the member is leaving.
    pub fn is_leaving(&self) -> bool {
        *self.tx.borrow()
    }

    /// A turn starts running.
    pub fn running(&self) -> Running {
        self.running.fetch_add(1, Ordering::AcqRel);
        Running(Arc::clone(&self.running))
    }

    /// The turns still running.
    pub fn count(&self) -> usize {
        self.running.load(Ordering::Acquire)
    }

    /// Raise the signal and wait, at most `wait`, for every running turn to
    /// end. How many still run when it stops waiting.
    pub async fn leave(&self, wait: Duration) -> usize {
        self.tx.send_replace(true);
        let deadline = Instant::now() + wait;
        loop {
            let left = self.count();
            if left == 0 || Instant::now() >= deadline {
                return left;
            }
            sleep(LOOK).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_leave_with_nothing_running_ends_at_once() {
        let l = Leaving::default();
        assert!(!l.is_leaving());
        assert_eq!(l.leave(Duration::from_secs(5)).await, 0);
        assert!(l.is_leaving());
    }

    /// A turn hears the signal, and the leave waits until it has stopped.
    #[tokio::test]
    async fn a_leave_waits_for_its_turns_to_stop() {
        let l = Arc::new(Leaving::default());
        let running = l.running();
        let mut heard = l.watch();
        let turn = tokio::spawn(async move {
            heard.wait_for(|leaving| *leaving).await.unwrap();
            drop(running);
        });
        assert_eq!(l.leave(Duration::from_secs(5)).await, 0);
        turn.await.unwrap();
        assert_eq!(l.count(), 0);
    }

    /// A turn that does not stop in time is left counted.
    #[tokio::test]
    async fn a_turn_that_does_not_stop_is_counted() {
        let l = Leaving::default();
        let _running = l.running();
        assert_eq!(l.leave(Duration::from_millis(120)).await, 1);
    }
}
