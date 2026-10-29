//! The substrate's lock, timed.
//!
//! Every reader and writer of a substrate — the scheduler's projections, the
//! persistence thread, the redo-log writer, the ingest workers, the API's
//! views — shares one `RwLock`. A view that answers in milliseconds waited
//! 13–18 s for it on BlackWorld (2026-10-29), and nothing said who held it.
//! [`SubstrateLock`] is that lock with the caller of every acquisition recorded
//! (`#[track_caller]`): a wait for it, or a hold of it, longer than [`SLOW`] is
//! logged with the site, so the holder names itself when it lets go and every
//! caller queued behind it says how long it stood there.
//!
//! The shape is `RwLock`'s — `read`, `write`, `try_read`, `try_write`, each a
//! `LockResult` whose poison carries the guard — so a caller written against
//! the std lock reads the same against this one.

use std::ops::{Deref, DerefMut};
use std::panic::Location;
use std::sync::{
    LockResult, PoisonError, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError, TryLockResult,
};
use std::time::{Duration, Instant};

use super::Substrate;

/// A wait for the lock, or a hold of it, at least this long is logged.
pub const SLOW: Duration = Duration::from_millis(250);

/// See the module documentation.
pub struct SubstrateLock(RwLock<Substrate>);

/// Which side of the lock an acquisition took.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Read,
    Write,
}

impl Side {
    fn as_str(self) -> &'static str {
        match self {
            Side::Read => "read",
            Side::Write => "write",
        }
    }
}

/// A shared hold of the substrate, timed from acquisition to drop.
pub struct SubstrateReadGuard<'a> {
    guard: RwLockReadGuard<'a, Substrate>,
    held: Held,
}

/// An exclusive hold of the substrate, timed from acquisition to drop.
pub struct SubstrateWriteGuard<'a> {
    guard: RwLockWriteGuard<'a, Substrate>,
    held: Held,
}

/// Where and since when a guard holds the lock.
struct Held {
    side: Side,
    site: &'static Location<'static>,
    since: Instant,
}

impl Drop for Held {
    fn drop(&mut self) {
        let held = self.since.elapsed();
        if held >= SLOW {
            tracing::warn!(
                target: "candle_conversation::substrate::lock",
                side = self.side.as_str(),
                site = %self.site,
                held_ms = held.as_millis() as u64,
                "the substrate lock was held long: every reader and writer behind it waited"
            );
        }
    }
}

/// Log a wait of `waited` for `side` at `site` when it was slow.
fn note_wait(side: Side, site: &'static Location<'static>, waited: Duration) {
    if waited >= SLOW {
        tracing::warn!(
            target: "candle_conversation::substrate::lock",
            side = side.as_str(),
            site = %site,
            waited_ms = waited.as_millis() as u64,
            "waited long for the substrate lock"
        );
    }
}

/// `result` with its guard, poisoned or not, wrapped by `wrap`.
fn wrap_lock<G, T>(result: LockResult<G>, wrap: impl FnOnce(G) -> T) -> LockResult<T> {
    match result {
        Ok(guard) => Ok(wrap(guard)),
        Err(poisoned) => Err(PoisonError::new(wrap(poisoned.into_inner()))),
    }
}

/// [`wrap_lock`] for a try, which may also have found the lock taken.
fn wrap_try<G, T>(result: TryLockResult<G>, wrap: impl FnOnce(G) -> T) -> TryLockResult<T> {
    match result {
        Ok(guard) => Ok(wrap(guard)),
        Err(TryLockError::WouldBlock) => Err(TryLockError::WouldBlock),
        Err(TryLockError::Poisoned(poisoned)) => Err(TryLockError::Poisoned(PoisonError::new(
            wrap(poisoned.into_inner()),
        ))),
    }
}

impl SubstrateLock {
    pub fn new(substrate: Substrate) -> Self {
        SubstrateLock(RwLock::new(substrate))
    }

    /// A shared hold, waiting for it; the wait and the hold are timed against
    /// the caller's site.
    #[track_caller]
    pub fn read(&self) -> LockResult<SubstrateReadGuard<'_>> {
        let site = Location::caller();
        let asked = Instant::now();
        let result = self.0.read();
        note_wait(Side::Read, site, asked.elapsed());
        wrap_lock(result, |guard| SubstrateReadGuard {
            guard,
            held: Held {
                side: Side::Read,
                site,
                since: Instant::now(),
            },
        })
    }

    /// An exclusive hold, waiting for it; the wait and the hold are timed
    /// against the caller's site.
    #[track_caller]
    pub fn write(&self) -> LockResult<SubstrateWriteGuard<'_>> {
        let site = Location::caller();
        let asked = Instant::now();
        let result = self.0.write();
        note_wait(Side::Write, site, asked.elapsed());
        wrap_lock(result, |guard| SubstrateWriteGuard {
            guard,
            held: Held {
                side: Side::Write,
                site,
                since: Instant::now(),
            },
        })
    }

    /// A shared hold if one is free now; the hold is timed.
    #[track_caller]
    pub fn try_read(&self) -> TryLockResult<SubstrateReadGuard<'_>> {
        let site = Location::caller();
        wrap_try(self.0.try_read(), |guard| SubstrateReadGuard {
            guard,
            held: Held {
                side: Side::Read,
                site,
                since: Instant::now(),
            },
        })
    }

    /// An exclusive hold if one is free now; the hold is timed.
    #[track_caller]
    pub fn try_write(&self) -> TryLockResult<SubstrateWriteGuard<'_>> {
        let site = Location::caller();
        wrap_try(self.0.try_write(), |guard| SubstrateWriteGuard {
            guard,
            held: Held {
                side: Side::Write,
                site,
                since: Instant::now(),
            },
        })
    }
}

impl SubstrateReadGuard<'_> {
    /// The site that took this hold.
    pub fn site(&self) -> &'static Location<'static> {
        self.held.site
    }
}

impl SubstrateWriteGuard<'_> {
    /// The site that took this hold.
    pub fn site(&self) -> &'static Location<'static> {
        self.held.site
    }
}

impl Deref for SubstrateReadGuard<'_> {
    type Target = Substrate;
    fn deref(&self) -> &Substrate {
        &self.guard
    }
}

impl Deref for SubstrateWriteGuard<'_> {
    type Target = Substrate;
    fn deref(&self) -> &Substrate {
        &self.guard
    }
}

impl DerefMut for SubstrateWriteGuard<'_> {
    fn deref_mut(&mut self) -> &mut Substrate {
        &mut self.guard
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;

    /// **A hold names the line that took it**, whichever side: the site is the
    /// caller's, not this module's.
    #[test]
    fn a_hold_names_the_line_that_took_it() {
        let lock = SubstrateLock::new(Substrate::new());
        let line = line!() + 1;
        let read = lock.read().unwrap();
        assert_eq!(read.site().file(), file!());
        assert_eq!(read.site().line(), line);
        drop(read);
        let line = line!() + 1;
        let write = lock.try_write().unwrap();
        assert_eq!(write.site().line(), line);
    }

    /// **A writer waits out a reader, and a try finds the lock taken**: the
    /// timing wraps the std lock without changing what it excludes.
    #[test]
    fn the_lock_excludes_as_the_std_lock_does() {
        let lock = Arc::new(SubstrateLock::new(Substrate::new()));
        let read = lock.read().unwrap();
        assert!(matches!(lock.try_write(), Err(TryLockError::WouldBlock)));
        assert!(lock.try_read().is_ok(), "readers share");
        let (got, wrote) = flume::bounded(1);
        let writer = {
            let lock = Arc::clone(&lock);
            thread::spawn(move || {
                drop(lock.write().unwrap());
                got.send(()).unwrap();
            })
        };
        thread::sleep(Duration::from_millis(50));
        assert!(wrote.try_recv().is_err(), "the writer got past a reader");
        drop(read);
        wrote.recv().unwrap();
        writer.join().unwrap();
    }

    /// **A poisoned lock hands back its guard**, as the std lock does, so a
    /// caller that recovers with `into_inner` still holds a timed guard.
    #[test]
    fn a_poisoned_lock_hands_back_its_guard() {
        let lock = Arc::new(SubstrateLock::new(Substrate::new()));
        let poisoner = Arc::clone(&lock);
        let _ = thread::spawn(move || {
            let _held = poisoner.write().unwrap();
            panic!("poison the lock");
        })
        .join();
        let recovered = lock.read().unwrap_or_else(|e| e.into_inner());
        assert_eq!(recovered.site().file(), file!());
        drop(recovered);
        assert!(matches!(lock.try_write(), Err(TryLockError::Poisoned(_))));
    }
}
