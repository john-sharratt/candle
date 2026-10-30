//! The substrate's versioned root (`docs/substrate_lock_free.md` §3).
//!
//! Every reader and writer of a substrate — the scheduler's projections, the
//! persistence thread, the redo-log writer, the ingest workers, the API's
//! views — shares one [`SubstrateLock`]. It holds no lock. It holds the latest
//! **version**: an `Arc<Substrate>` built of persistent collections, so a clone
//! is O(1) and an edit copies only the paths it changes.
//!
//! - **A read never waits.** [`SubstrateLock::read`] loads the latest version
//!   and hands it out; the reader holds that version for as long as it reads,
//!   whatever commits after, and never sees a half-made edit.
//! - **A write never waits either.** [`SubstrateLock::update`] edits a clone of
//!   the latest version and publishes it by compare-and-swap. When another
//!   writer published first, the edit is run again on the newer version — so an
//!   edit is a pure function of the version it is handed, and does nothing
//!   outside it but allocate residences (whose attempt is tracked, and
//!   discarded when lost — [`residence::begin_attempt`]).
//! - **Residences are outside the version.** KV tier transitions are atomic
//!   steps on the shared [`ResidenceSlab`] and commit nothing.
//!
//! The substrate is an order-independent fold of stamped records, so the edits
//! that race here commute: a retry re-applies the same records to a version
//! that holds more of them, and arrives where either order would have.

use std::ops::Deref;
use std::panic::Location;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;

use super::residence::{self, ResidenceSlab};
use super::Substrate;

/// An edit that took at least this long, or was retried at least
/// [`RETRIES_NOTED`] times, is logged with the site that made it.
pub const SLOW: Duration = Duration::from_millis(250);

/// See [`SLOW`].
pub const RETRIES_NOTED: u32 = 8;

/// See the module documentation.
pub struct SubstrateLock {
    root: ArcSwap<Substrate>,
}

/// One version of the substrate, held for as long as the reader reads.
pub struct SubstrateReadGuard {
    version: Arc<Substrate>,
}

/// Ends a version edit's residence attempt, lost unless marked committed —
/// so an edit that panics discards what it allocated.
struct Attempt<'a> {
    slab: &'a ResidenceSlab,
    committed: bool,
}

impl Drop for Attempt<'_> {
    fn drop(&mut self) {
        residence::end_attempt(self.slab, self.committed);
    }
}

impl SubstrateLock {
    pub fn new(substrate: Substrate) -> Self {
        SubstrateLock {
            root: ArcSwap::from_pointee(substrate),
        }
    }

    /// The latest version. Never waits.
    pub fn read(&self) -> SubstrateReadGuard {
        SubstrateReadGuard {
            version: self.root.load_full(),
        }
    }

    /// The residences every version shares — for tier transitions, which
    /// commit no version.
    pub fn residences(&self) -> Arc<ResidenceSlab> {
        Arc::clone(self.root.load().residences())
    }

    /// Apply `edit` to the latest version and publish the result, returning
    /// what the publishing run of `edit` returned.
    ///
    /// `edit` may run more than once: each time another writer publishes
    /// between this one's load and its swap, it is run again on the newer
    /// version. It must therefore read only the version it is handed and
    /// change nothing outside it — no I/O, no channel sends, no shared
    /// counters. Work with an effect outside the substrate is done by the
    /// caller, after `update` returns.
    #[track_caller]
    pub fn update<R>(&self, mut edit: impl FnMut(&mut Substrate) -> R) -> R {
        let site = Location::caller();
        let started = Instant::now();
        let mut current = self.root.load_full();
        let mut retries = 0u32;
        loop {
            let mut next = (*current).clone();
            let mut attempt = Attempt {
                slab: current.residences(),
                committed: false,
            };
            residence::begin_attempt();
            let out = edit(&mut next);
            let prev = self.root.compare_and_swap(&current, Arc::new(next));
            if Arc::ptr_eq(&prev, &current) {
                attempt.committed = true;
                drop(attempt);
                note_edit(site, started.elapsed(), retries);
                return out;
            }
            drop(attempt);
            retries += 1;
            current = arc_swap::Guard::into_inner(prev);
        }
    }
}

/// Log an edit at `site` that was slow or retried often.
fn note_edit(site: &'static Location<'static>, took: Duration, retries: u32) {
    if took >= SLOW || retries >= RETRIES_NOTED {
        tracing::warn!(
            target: "candle_conversation::substrate::lock",
            site = %site,
            took_ms = took.as_millis() as u64,
            retries,
            "a substrate edit was slow or contended: split it, or move work out of it"
        );
    }
}

impl SubstrateReadGuard {
    /// The version, to keep beyond this guard.
    pub fn version(&self) -> Arc<Substrate> {
        Arc::clone(&self.version)
    }
}

impl Deref for SubstrateReadGuard {
    type Target = Substrate;
    fn deref(&self) -> &Substrate {
        &self.version
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::*;
    use crate::persistence::streams::StreamId;

    /// **A reader keeps its version**: an edit published after the read is
    /// not in it, and the next read has it.
    #[test]
    fn a_reader_keeps_the_version_it_read() {
        let lock = SubstrateLock::new(Substrate::new());
        let before = lock.read();
        lock.update(|s| s.toggle_layer("code"));
        assert!(!before.is_layer_disabled("code"));
        assert!(lock.read().is_layer_disabled("code"));
    }

    /// **No edit is lost under contention**: every one of many concurrent
    /// edits lands, each exactly once in the published result.
    #[test]
    fn concurrent_edits_all_land() {
        let lock = Arc::new(SubstrateLock::new(Substrate::new()));
        let writers: Vec<_> = (0..8)
            .map(|w| {
                let lock = Arc::clone(&lock);
                thread::spawn(move || {
                    for i in 0..200 {
                        let name = format!("w{w}-{i}");
                        lock.update(|s| {
                            s.toggle_layer(&name);
                        });
                    }
                })
            })
            .collect();
        for w in writers {
            w.join().unwrap();
        }
        let s = lock.read();
        for w in 0..8 {
            for i in 0..200 {
                assert!(s.is_layer_disabled(&format!("w{w}-{i}")));
            }
        }
    }

    /// **A lost attempt's residences are discarded**: an edit that allocates
    /// and installs, then loses its swap, leaves the losing residence dead and
    /// empty, and the winner's alive.
    #[test]
    fn a_lost_attempt_discards_its_residences() {
        let lock = SubstrateLock::new(Substrate::new());
        let mut allocated = Vec::new();
        let mut raced = false;
        lock.update(|s| {
            let idx = s.alloc_residence(StreamId(7), None);
            allocated.push(idx);
            if !raced {
                raced = true;
                // Another writer publishes between this attempt's load and swap.
                thread::scope(|t| {
                    t.spawn(|| lock.update(|o| o.toggle_layer("other")));
                });
            }
        });
        assert_eq!(allocated.len(), 2, "the edit ran again");
        let slab = lock.residences();
        assert!(
            slab.get(allocated[0]).is_dead(),
            "the lost attempt's residence"
        );
        assert!(!slab.get(allocated[1]).is_dead(), "the published one's");
    }

    /// **An edit that panics publishes nothing** and leaves the next edit free
    /// to run.
    #[test]
    fn a_panicking_edit_publishes_nothing() {
        let lock = Arc::new(SubstrateLock::new(Substrate::new()));
        let l = Arc::clone(&lock);
        let _ = thread::spawn(move || {
            l.update(|s| {
                s.toggle_layer("half");
                panic!("mid-edit");
            })
        })
        .join();
        assert!(!lock.read().is_layer_disabled("half"));
        lock.update(|s| s.toggle_layer("after"));
        assert!(lock.read().is_layer_disabled("after"));
    }
}
