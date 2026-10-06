//! The persistence mutex, with the foreground ahead of maintenance.
//!
//! A segment-maintenance op relocates a segment in batches and releases the lock
//! between them so a cold load or a seal write waits for one batch, not the whole
//! op. A plain mutex does not deliver that: the maintenance thread re-takes the
//! lock the instant it lets go, before a waiter has woken, so the waiter sat out
//! every batch. Measured on a live daemon: a 4.4 GB segment's 949 batches held
//! the lock for 22,073 ms of a 22,077 ms relocation, and the elevate a decode was
//! waiting on stalled for all 21.6 s of it.
//!
//! So every ordinary acquisition registers itself as waiting while it blocks, and
//! maintenance takes the lock through [`PersistenceLock::lock_behind_waiters`],
//! which does not start its next batch while anyone is registered.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LockResult, Mutex, MutexGuard, TryLockResult};
use std::thread;
use std::time::Duration;

use super::SubstratePersistence;

/// How long maintenance sleeps between looks at the waiter count.
const YIELD_POLL: Duration = Duration::from_millis(1);

/// The substrate's persistence handle behind a mutex that lets foreground
/// callers in ahead of background maintenance.
pub struct PersistenceLock<T = SubstratePersistence> {
    inner: Mutex<T>,
    /// Foreground callers blocked on — or about to block on — `inner`.
    waiting: AtomicUsize,
}

impl<T> PersistenceLock<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: Mutex::new(value),
            waiting: AtomicUsize::new(0),
        }
    }

    /// Take the lock as a foreground caller: maintenance holds off its next
    /// batch until this one has it.
    pub fn lock(&self) -> LockResult<MutexGuard<'_, T>> {
        self.waiting.fetch_add(1, Ordering::AcqRel);
        let guard = self.inner.lock();
        self.waiting.fetch_sub(1, Ordering::AcqRel);
        guard
    }

    pub fn try_lock(&self) -> TryLockResult<MutexGuard<'_, T>> {
        self.inner.try_lock()
    }

    /// Take the lock as background work: wait until no foreground caller is
    /// waiting, then take it. A caller that arrives after the check waits for
    /// one hold, no longer.
    pub fn lock_behind_waiters(&self) -> LockResult<MutexGuard<'_, T>> {
        while self.waiting.load(Ordering::Acquire) > 0 {
            thread::sleep(YIELD_POLL);
        }
        self.inner.lock()
    }

    /// Foreground callers currently waiting.
    pub fn waiting(&self) -> usize {
        self.waiting.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::time::Instant;

    /// A foreground caller waits for the hold in progress and nothing more: the
    /// background loop that let go does not get the lock back first.
    #[test]
    fn a_foreground_caller_goes_ahead_of_the_next_background_hold() {
        let lock = Arc::new(PersistenceLock::new(Vec::<&str>::new()));
        let (started_tx, started_rx) = mpsc::channel();
        let background = {
            let lock = lock.clone();
            thread::spawn(move || {
                for i in 0..200 {
                    let mut g = lock.lock_behind_waiters().unwrap();
                    if i == 0 {
                        started_tx.send(()).unwrap();
                    }
                    g.push("batch");
                    thread::sleep(Duration::from_millis(2));
                }
            })
        };
        started_rx.recv().unwrap();
        let t = Instant::now();
        lock.lock().unwrap().push("foreground");
        let waited = t.elapsed();
        background.join().unwrap();

        let log = lock.lock().unwrap();
        let at = log.iter().position(|e| *e == "foreground").unwrap();
        assert!(
            at < 10,
            "the foreground got in after {at} batches, not after the loop's 200"
        );
        assert!(waited < Duration::from_millis(200), "waited {waited:?}");
        assert_eq!(lock.waiting(), 0);
    }

    /// With nobody waiting, background work takes the lock at once.
    #[test]
    fn an_idle_lock_is_taken_at_once_by_background_work() {
        let lock = PersistenceLock::new(0u32);
        *lock.lock_behind_waiters().unwrap() += 1;
        *lock.lock().unwrap() += 1;
        assert_eq!(*lock.lock().unwrap(), 2);
        assert_eq!(lock.waiting(), 0);
    }
}
