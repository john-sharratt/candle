//! Stopping and starting taking work (§12.4): `drain` hands back every claim
//! and lease, as fault mode does, and the gateways route nothing new to the
//! member — it stays connected, keeps sending its own events and answering
//! fetches. `undrain` takes work again once a catch-up has run. Beside them,
//! what the maintenance commands run on replication: a catch-up or a full
//! resync now, a lease hand-back alone, and a lowered `max_members`.

use std::sync::Arc;

use super::catchup::Purpose;
use super::lease::LeaseEvent;
use super::replicator::Replicator;
use super::upgrade::is_upgrade_lease;

/// What a drain handed back.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HandedBack {
    pub claims: usize,
    pub leases: usize,
}

impl Replicator {
    /// Hand back every claim and lease; the number of claims.
    pub(super) async fn hand_back(self: &Arc<Self>) -> usize {
        let store = self.store.clone();
        let through = tokio::task::spawn_blocking(move || store.own_through())
            .await
            .unwrap_or(0);
        self.release_leases(through);
        self.member().map_or(0, |m| m.release_all(through))
    }

    /// Hand back every lease this member holds but an upgrade's; how many.
    /// `through` is its own `oseq`, which the next holder waits for (§9.6).
    /// An `upgrade/<app>` lease stays: the member holding it is draining to
    /// install, and the next member waits for it to be back (§3.9).
    pub fn release_leases(self: &Arc<Self>, through: u64) -> usize {
        let now = self.now();
        let (outs, keys) = {
            let mut p = self.paxos.lock().unwrap_or_else(|e| e.into_inner());
            let held: Vec<Vec<u8>> = p
                .held(now)
                .into_iter()
                .filter(|k| !is_upgrade_lease(k))
                .collect();
            let mut outs = Vec::new();
            for key in &held {
                match p.release(key, through, now) {
                    Ok(o) => outs.extend(o),
                    Err(e) => tracing::error!(node = self.node, "the vote journal failed: {e}"),
                }
            }
            (outs, held)
        };
        self.dispatch(outs);
        for key in &keys {
            let _ = self.leases.send(LeaseEvent::Lost { key: key.clone() });
        }
        keys.len()
    }

    /// Stop taking work, staying connected.
    pub async fn drain(self: &Arc<Self>) -> HandedBack {
        self.state().drained = true;
        let leases = {
            let now = self.now();
            self.paxos
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .held(now)
                .iter()
                .filter(|k| !is_upgrade_lease(k))
                .count()
        };
        let claims = self.hand_back().await;
        if let Some(m) = self.member() {
            m.capacity_changed();
        }
        self.coord_now();
        HandedBack { claims, leases }
    }

    /// Take work again: each socket catches up first, and reports ready once
    /// it has. Refused while a catch-up is running or in fault mode.
    pub fn undrain(self: &Arc<Self>) -> Result<(), String> {
        let sockets = {
            let mut s = self.state();
            if s.fault.is_some() {
                return Err("in fault mode".into());
            }
            if !s.catchups.is_empty() {
                return Err("a catch-up is running".into());
            }
            s.drained = false;
            let sockets: Vec<usize> = s.admitted().into_iter().map(|(c, _)| c).collect();
            for c in &sockets {
                s.ready[*c] = false;
            }
            sockets
        };
        for c in sockets {
            self.start_catchup(c, Purpose::Admission, None);
        }
        if let Some(m) = self.member() {
            m.capacity_changed();
        }
        self.coord_now();
        Ok(())
    }

    pub fn drained(&self) -> bool {
        self.state().drained
    }

    /// Own events neither unsent nor in flight: what a shutdown waits for.
    pub fn outbox_empty(&self) -> bool {
        self.state().outbox.counts() == (0, 0)
    }

    /// Catch up now (§7.2), from `from` or from whichever member each gateway
    /// picks; how many catch-ups started.
    pub fn catch_up_now(self: &Arc<Self>, from: Option<u64>) -> usize {
        let sockets: Vec<usize> = self
            .state()
            .admitted()
            .into_iter()
            .map(|(c, _)| c)
            .collect();
        for c in &sockets {
            self.start_catchup(*c, Purpose::Periodic, from);
        }
        sockets.len()
    }

    /// The full resync now (§7.5): catch up, then offer what it holds.
    pub fn resync(self: &Arc<Self>) -> usize {
        let sockets: Vec<usize> = {
            let mut s = self.state();
            s.offered = false;
            s.admitted().into_iter().map(|(c, _)| c).take(1).collect()
        };
        for c in &sockets {
            self.state().ready[*c] = false;
            self.start_catchup(*c, Purpose::Admission, None);
        }
        sockets.len()
    }

    /// Write `max_members` lower, to `n` (§5.9).
    pub async fn set_quorum(self: &Arc<Self>, n: u32) {
        let store = self.store.clone();
        let _ = tokio::task::spawn_blocking(move || store.set_max_members(n)).await;
        self.refresh_group().await;
    }
}
