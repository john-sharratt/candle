//! The one place a substrate's record identities are minted.
//!
//! The persistence layer mints when it appends a record it originates, and the
//! runtime write paths mint when they change the substrate in memory, before
//! the record that carries the change reaches the writer thread. Both draw on
//! one [`Origination`], so every identity this node issues is unique, its
//! `oseq`s never repeat, and each in-memory value holds, from the moment it is
//! set, the stamp its record will carry on disk.
//!
//! Minting takes a lock of its own and nothing else, so it is safe to call
//! with the substrate lock or the persistence lock held.

use std::sync::{Arc, Mutex, MutexGuard};

use super::origination::{now_ms, Origination};
use super::record_identity::RecordIdentity;

/// A shared handle on a substrate's [`Origination`].
#[derive(Clone, Debug)]
pub struct Minter(Arc<Mutex<Origination>>);

impl Minter {
    pub fn new(origination: Origination) -> Minter {
        Minter(Arc::new(Mutex::new(origination)))
    }

    fn lock(&self) -> MutexGuard<'_, Origination> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The identity of the next record this node originates, under claim or
    /// lease `epoch` (zero for none), stamped with the wall clock now.
    pub fn mint(&self, epoch: u64) -> RecordIdentity {
        self.lock().mint(now_ms(), epoch)
    }

    /// Take a record this substrate holds into account (see
    /// [`Origination::observe`]).
    pub fn observe(&self, id: &RecordIdentity) {
        self.lock().observe(id);
    }

    /// The writer's identity state as it stands.
    pub fn origination(&self) -> Origination {
        self.lock().clone()
    }

    /// This node.
    pub fn node(&self) -> u64 {
        self.lock().node()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two handles on one minter draw on one sequence and one clock.
    #[test]
    fn clones_share_one_sequence() {
        let a = Minter::new(Origination::new(9));
        let b = a.clone();
        let first = a.mint(0);
        let second = b.mint(0);
        assert_eq!((first.oseq, second.oseq), (1, 2));
        assert!(second.stamp() > first.stamp());
        assert_eq!(a.origination().next_oseq(), 3);
    }

    /// A record observed through one handle bounds what the other mints.
    #[test]
    fn an_observation_bounds_every_handle() {
        let a = Minter::new(Origination::new(9));
        let b = a.clone();
        let far = RecordIdentity {
            origin: 4,
            oseq: 1,
            gen: 2,
            epoch: 0,
            hlc: u64::MAX >> 1,
        };
        a.observe(&far);
        let next = b.mint(0);
        assert_eq!(next.gen, 2);
        assert!(next.hlc > far.hlc);
    }
}
