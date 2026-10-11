//! The app's leases (§9.2): a world's authority holds `world/<wid>` by Paxos,
//! renewing it while it runs the world, and hands it back when rendezvous names
//! another proposer, when it drains, or when its disk fills. The app follows
//! what becomes of each lease through [`LeaseEvent`]s.
//!
//! A member proposes for a lease only while it is joined, ready to take work
//! and not in fault mode: one that cannot write the world's state must not
//! hold it (§7.7).
//!
//! **A lease is asked for again when the group grows.** One taken under a
//! smaller quorum was accepted by a majority of the smaller group, and the
//! members that have joined since never saw it: a majority of the larger group
//! made of them could grant it a second time. So once the quorum this member
//! counts rises above the one a held lease was last granted under, it renews
//! that lease at once, and a majority of the group as it stands knows it.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::broadcast;

use super::replicator::{Phase, Replicator};

/// A world lease's term (§9.2).
pub const LEASE_TTL: Duration = Duration::from_secs(30);

/// How often its holder renews it.
pub const RENEW_EVERY: Duration = Duration::from_secs(10);

/// How many lease events a slow follower may fall behind by.
pub(super) const EVENTS_KEPT: usize = 1024;

/// What a lease was last asked for with: its term, and the quorum counted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Term {
    pub ttl: Duration,
    pub quorum: usize,
}

/// What became of a lease this member asked for or held.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaseEvent {
    /// This member holds it now — newly, or renewed.
    Held { key: Vec<u8> },
    /// Another member holds it.
    Busy { key: Vec<u8>, holder: u64 },
    /// This member no longer holds it: it ran out unrenewed, or was handed
    /// back.
    Lost { key: Vec<u8> },
    /// Member `by` handed it back, its own writes reaching `through`: the next
    /// holder takes it once it has applied them (§9.6).
    Released { key: Vec<u8>, by: u64, through: u64 },
}

impl LeaseEvent {
    /// The lease this event is about.
    pub fn key(&self) -> &[u8] {
        match self {
            LeaseEvent::Held { key }
            | LeaseEvent::Busy { key, .. }
            | LeaseEvent::Lost { key }
            | LeaseEvent::Released { key, .. } => key,
        }
    }
}

impl Replicator {
    /// Every lease event from now on.
    pub fn lease_events(&self) -> broadcast::Receiver<LeaseEvent> {
        self.leases.subscribe()
    }

    /// Whether this member may propose for a lease now.
    fn may_hold(&self) -> bool {
        let s = self.state();
        s.phase == Phase::Joined && s.fault.is_none() && !s.drained
    }

    /// Ask for `key`'s lease for `ttl`, or renew it. Refused — `false` — while
    /// this member may not hold one.
    pub fn acquire(self: &Arc<Self>, key: &[u8], ttl: Duration) -> bool {
        if !self.may_hold() {
            return false;
        }
        let now = self.now();
        let quorum = self.quorum();
        self.terms
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key.to_vec(), Term { ttl, quorum });
        let outs = {
            let mut p = self.paxos.lock().unwrap_or_else(|e| e.into_inner());
            p.set_quorum(quorum);
            p.acquire(key, ttl, now)
        };
        match outs {
            Ok(outs) => {
                self.dispatch(outs);
                true
            }
            Err(e) => {
                tracing::error!(node = self.node, "the vote journal failed: {e}");
                false
            }
        }
    }

    /// Renew at once every lease held under a smaller quorum than this
    /// member counts now. Run on every consensus tick.
    pub(super) fn reaffirm_leases(self: &Arc<Self>) {
        let quorum = self.quorum();
        let due: Vec<(Vec<u8>, Duration)> = self
            .terms
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|(_, t)| t.quorum < quorum)
            .map(|(k, t)| (k.clone(), t.ttl))
            .collect();
        for (key, ttl) in due {
            if self.holds(&key) {
                self.acquire(&key, ttl);
            }
        }
    }

    /// Whether this member holds `key`'s lease now.
    pub fn holds(&self, key: &[u8]) -> bool {
        let now = self.now();
        self.paxos
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .holds(key, now)
    }

    /// Through which `oseq` this member has applied `origin`'s records.
    pub fn applied_through(&self, origin: u64) -> u64 {
        self.store.applied().through(origin)
    }

    /// This member's own records are written through this `oseq`.
    pub fn own_through(&self) -> u64 {
        self.store.own_through()
    }

    /// Hand `key`'s lease back; `through` is this member's own `oseq`, which
    /// the next holder waits for (§9.6).
    pub fn release_lease(self: &Arc<Self>, key: &[u8], through: u64) {
        let now = self.now();
        self.terms
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(key);
        let outs = {
            let mut p = self.paxos.lock().unwrap_or_else(|e| e.into_inner());
            p.release(key, through, now)
        };
        match outs {
            Ok(outs) => {
                let released = !outs.is_empty();
                self.dispatch(outs);
                if released {
                    let _ = self.leases.send(LeaseEvent::Lost { key: key.to_vec() });
                }
            }
            Err(e) => tracing::error!(node = self.node, "the vote journal failed: {e}"),
        }
    }
}
