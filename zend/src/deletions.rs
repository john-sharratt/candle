//! Conversations deleted while a reply runs: the signal a running turn waits on
//! beside its stream.
//!
//! A turn outlives its client on purpose — a reload follows it, and it seals
//! as it would have — so nothing about a closed socket ends it. A delete must:
//! nobody can come back to a tombstoned conversation, and every token of a
//! round still running is prefill and decode the engine spends on it while the
//! turns queued behind it wait. Checked only between rounds, a round of tool
//! output already submitted ran to its end: measured on WhiteWorld, 24 readers
//! deleted mid-round held the engine for twelve minutes, every new turn queued
//! behind their prefills.
//!
//! A delete moves the epoch; a running turn wakes, asks the substrate whether
//! its own timeline is the one tombstoned, and if so drops its handle at once,
//! which withdraws it from the engine.

use tokio::sync::watch;

/// The deletion epoch, moved by every conversation delete.
pub struct Deletions {
    epoch: watch::Sender<u64>,
}

impl Default for Deletions {
    fn default() -> Self {
        Self {
            epoch: watch::channel(0).0,
        }
    }
}

impl Deletions {
    /// A conversation was tombstoned: every running turn checks whether it was
    /// its own.
    pub fn deleted(&self) {
        self.epoch.send_modify(|e| *e += 1);
    }

    /// A receiver that wakes at the next delete.
    pub fn watch(&self) -> watch::Receiver<u64> {
        self.epoch.subscribe()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// **A watcher wakes at a delete, and only then.**
    #[tokio::test]
    async fn a_watcher_wakes_at_a_delete() {
        let deletions = Deletions::default();
        let mut rx = deletions.watch();
        let quiet = tokio::time::timeout(Duration::from_millis(50), rx.changed()).await;
        assert!(quiet.is_err(), "nothing deleted, nothing to wake for");
        deletions.deleted();
        tokio::time::timeout(Duration::from_secs(5), rx.changed())
            .await
            .expect("woken by the delete")
            .expect("the sender lives");
    }

    /// A delete made while a turn is between waits is not lost: the next wait
    /// returns at once.
    #[tokio::test]
    async fn a_delete_between_waits_is_seen_at_the_next() {
        let deletions = Deletions::default();
        let mut rx = deletions.watch();
        deletions.deleted();
        tokio::time::timeout(Duration::from_secs(5), rx.changed())
            .await
            .expect("seen without waiting")
            .expect("the sender lives");
    }
}
