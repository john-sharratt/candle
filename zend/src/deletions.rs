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
//!
//! Work a conversation leaves running after its turn — its journal draft, whose
//! questions run on forks of its slot — ends the same way ([`Deletions::unless_deleted`]):
//! dropped where it stands, its forks freed with it. Left to run, a draft for a
//! deleted conversation held a fork's recurrent store for as long as the engine
//! took to reach it — fifteen minutes and more under load, on BlackWorld
//! (2026-11-23), two dozen at once.

use std::future::Future;

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

    /// `work`'s output, or `None` when `gone` — whether the conversation `work`
    /// is for is tombstoned — holds before it finishes: asked before it starts
    /// and again at every delete. `work` is dropped where it stands.
    pub async fn unless_deleted<F: Future>(
        &self,
        work: F,
        gone: impl Fn() -> bool,
    ) -> Option<F::Output> {
        let mut rx = self.watch();
        if gone() {
            return None;
        }
        let mut work = std::pin::pin!(work);
        loop {
            tokio::select! {
                out = &mut work => return Some(out),
                changed = rx.changed() => {
                    if changed.is_err() || gone() {
                        return None;
                    }
                }
            }
        }
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

    /// **Work for a conversation deleted under it is dropped**: a delete of
    /// another conversation leaves it running, its own ends it, and work whose
    /// conversation is already gone never starts.
    #[tokio::test]
    async fn work_for_a_deleted_conversation_is_dropped() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let deletions = Arc::new(Deletions::default());
        let gone = Arc::new(AtomicBool::new(false));
        let started = Arc::new(AtomicBool::new(false));
        let task = {
            let (deletions, gone, started) = (
                Arc::clone(&deletions),
                Arc::clone(&gone),
                Arc::clone(&started),
            );
            tokio::spawn(async move {
                let work = async {
                    started.store(true, Ordering::SeqCst);
                    std::future::pending::<u32>().await
                };
                deletions
                    .unless_deleted(work, || gone.load(Ordering::SeqCst))
                    .await
            })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        deletions.deleted();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!task.is_finished(), "another conversation's delete");
        gone.store(true, Ordering::SeqCst);
        deletions.deleted();
        let out = tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("ended by its own delete")
            .unwrap();
        assert_eq!(out, None);
        assert!(started.load(Ordering::SeqCst));

        let done = deletions.unless_deleted(async { 7 }, || false).await;
        assert_eq!(done, Some(7), "work that finishes is kept");
        let never = deletions
            .unless_deleted(async { panic!("started for a gone conversation") }, || true)
            .await;
        assert_eq!(never, None::<()>);
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
