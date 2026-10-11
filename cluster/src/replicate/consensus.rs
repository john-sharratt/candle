//! Consensus on a member (§5.9) and the generation a rejoin takes by it
//! (§6.7).
//!
//! The member is an acceptor and a proposer. Paxos frames from the gateways
//! go to its engine; what the engine sends goes out on every admitted socket,
//! the gateways delivering replies to `reply_to` alone. A decision is written
//! as a replicated record, so every member reads it from its substrate. The
//! quorum is a majority of `max_members`, which the member raises when it sees
//! more ready members at once than the substrate holds.
//!
//! **Rejoining.** A member that wrote while cut off catches up first, then
//! proposes `gen/<k>` with itself as the value, `k` one above the greatest
//! decided generation it knows. A decision for another member moves it to
//! `k + 1`. Its own decision fixes what it wrote cut off, which the store then
//! reloads under; without a quorum it stays cut off and tries again.

use std::collections::BTreeSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::replicator::{Phase, Rejoin, Replicator};
use crate::consensus::Out;
use crate::frame::{Frame, MemberState};

/// How long a rejoin waits for its generation's decision.
pub const REJOIN_WAIT: Duration = Duration::from_secs(10);

/// How long a member left cut off for want of a quorum waits to try again.
pub const REJOIN_RETRY: Duration = Duration::from_secs(5);

fn generation_key(k: u64) -> Vec<u8> {
    format!("gen/{k}").into_bytes()
}

impl Replicator {
    /// A Paxos frame from another member.
    pub(super) fn paxos_frame(self: &Arc<Self>, frame: &Frame) {
        let now = self.now();
        let outs = {
            let mut p = self.paxos.lock().unwrap_or_else(|e| e.into_inner());
            p.handle(frame, now)
        };
        match outs {
            Ok(outs) => self.dispatch(outs),
            Err(e) => tracing::error!(node = self.node, "the vote journal failed: {e}"),
        }
    }

    /// Act on what the engine asked for.
    pub(super) fn dispatch(self: &Arc<Self>, outs: Vec<Out>) {
        for out in outs {
            match out {
                Out::Broadcast(frame) | Out::Reply { frame, .. } => {
                    for (_, handle) in self.state().admitted() {
                        let _ = handle.control(frame.clone());
                    }
                }
                Out::Decided { key, value } => self.decided(key, value),
                // Leases are the worlds' (§9.2); no lease is asked for here.
                Out::LeaseHeld { .. } | Out::LeaseBusy { .. } | Out::LeaseLost { .. } => {}
            }
        }
    }

    /// `key` is decided: written as a replicated record — off the runtime, it
    /// syncs — then learnt, and a rejoin waiting on it goes on.
    fn decided(self: &Arc<Self>, key: Vec<u8>, value: Vec<u8>) {
        let me = self.clone();
        tokio::spawn(async move {
            let store = me.store.clone();
            let (k, v) = (key.clone(), value.clone());
            let _ = tokio::task::spawn_blocking(move || store.record_decision(&k, &v)).await;
            {
                let mut p = me.paxos.lock().unwrap_or_else(|e| e.into_inner());
                if let Err(e) = p.learn(&key, &value) {
                    tracing::error!(node = me.node, "the vote journal failed: {e}");
                }
            }
            let rejoin = me.state().rejoin;
            let Some(Rejoin { k, .. }) = rejoin else {
                return;
            };
            if key != generation_key(k) {
                return;
            }
            if value == me.node.to_le_bytes() {
                me.state().rejoin = None;
                me.rejoined(k).await;
            } else {
                me.propose_generation(k + 1).await;
            }
        });
    }

    /// Start taking a generation, once the admission catch-up is done.
    pub(super) fn begin_rejoin(self: &Arc<Self>) {
        let me = self.clone();
        tokio::spawn(async move {
            let store = me.store.clone();
            let Ok((detached, known)) =
                tokio::task::spawn_blocking(move || (store.detached(), store.generation())).await
            else {
                return;
            };
            if !detached {
                // Nothing written cut off: nothing to place, no generation.
                me.rejoined(known).await;
                return;
            }
            me.propose_generation(known + 1).await;
        });
    }

    /// Propose `gen/<k>`, or the first `k` above it no decision names yet.
    async fn propose_generation(self: &Arc<Self>, k: u64) {
        let store = self.store.clone();
        let Ok(k) = tokio::task::spawn_blocking(move || {
            let mut k = k;
            while store.decision(&generation_key(k)).is_some() {
                k += 1;
            }
            k
        })
        .await
        else {
            return;
        };
        self.state().rejoin = Some(Rejoin {
            k,
            deadline: Instant::now() + REJOIN_WAIT,
        });
        let now = self.now();
        let quorum = self.quorum();
        let outs = {
            let mut p = self.paxos.lock().unwrap_or_else(|e| e.into_inner());
            // The quorum as the substrate holds it now: a proposal never runs
            // on a stale one.
            p.set_quorum(quorum);
            p.decide(&generation_key(k), self.node.to_le_bytes().to_vec(), now)
        };
        match outs {
            Ok(outs) => self.dispatch(outs),
            Err(e) => tracing::error!(node = self.node, "the vote journal failed: {e}"),
        }
    }

    /// Generation `k` is this member's: fix what it wrote cut off, rebuild,
    /// and send it (§6.7).
    async fn rejoined(self: &Arc<Self>, k: u64) {
        let store = self.store.clone();
        let _ = tokio::task::spawn_blocking(move || {
            let fixed = store.detached();
            store.rejoin(k);
            if fixed {
                store.reload();
            }
        })
        .await;
        {
            let mut s = self.state();
            s.phase = Phase::Joined;
            s.retry_rejoin = None;
        }
        if let Some(m) = self.member() {
            m.capacity_changed();
        }
        self.send_wake.notify_one();
    }

    /// A majority of `max_members` (§5.9), as last read from the substrate.
    pub(super) fn quorum(&self) -> usize {
        self.state().max_members as usize / 2 + 1
    }

    /// The once-a-tick consensus work: the engine's timers, the quorum, and a
    /// rejoin that found no quorum.
    pub(super) fn consensus_tick(self: &Arc<Self>) {
        let now = self.now();
        let quorum = self.quorum();
        let outs = {
            let mut p = self.paxos.lock().unwrap_or_else(|e| e.into_inner());
            p.set_quorum(quorum);
            p.tick(now)
        };
        match outs {
            Ok(outs) => self.dispatch(outs),
            Err(e) => tracing::error!(node = self.node, "the vote journal failed: {e}"),
        }

        let expired = {
            let mut s = self.state();
            match s.rejoin {
                Some(r) if Instant::now() >= r.deadline => {
                    s.rejoin = None;
                    s.phase = Phase::CutOff;
                    s.retry_rejoin = Some(Instant::now() + REJOIN_RETRY);
                    Some(r.k)
                }
                _ => None,
            }
        };
        if let Some(k) = expired {
            self.paxos
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .abandon(&generation_key(k));
            tracing::warn!(
                node = self.node,
                "no quorum for generation {k}: still cut off, serving locally"
            );
            if let Some(m) = self.member() {
                m.capacity_changed();
            }
        }
        let retry = {
            let mut s = self.state();
            let due = s.retry_rejoin.is_some_and(|at| Instant::now() >= at)
                && s.phase == Phase::CutOff
                && s.ready.iter().any(|r| *r);
            if due {
                s.retry_rejoin = None;
            }
            due
        };
        if retry {
            self.begin_rejoin();
        }
    }

    /// The once-a-second read of what the substrate holds of the group —
    /// `max_members` and the greatest decided generation — and a raise of
    /// `max_members` when more ready members are seen at once.
    pub(super) async fn refresh_group(self: &Arc<Self>) {
        let seen = self.ready_members();
        let store = self.store.clone();
        let Ok((max_members, generation)) = tokio::task::spawn_blocking(move || {
            let held = store.max_members();
            if seen > held {
                store.raise_max_members(seen);
            }
            (store.max_members(), store.generation())
        })
        .await
        else {
            return;
        };
        let mut s = self.state();
        s.max_members = max_members;
        s.generation = generation;
    }

    /// The distinct ready members visible at once across every cluster, this
    /// one included when it is ready.
    fn ready_members(&self) -> u32 {
        let s = self.state();
        let mut ready: BTreeSet<u64> = s
            .members
            .iter()
            .flatten()
            .flat_map(|m| m.members.iter())
            .filter(|m| m.state == MemberState::Ready)
            .map(|m| m.node)
            .collect();
        if s.phase == Phase::Joined && s.ready.iter().any(|r| *r) {
            ready.insert(self.node);
        }
        ready.len() as u32
    }
}
