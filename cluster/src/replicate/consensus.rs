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

use super::lease::LeaseEvent;
use super::quorum_governor::{Look, Step};
use super::replicator::{Phase, Rejoin, Replicator, State};
use crate::consensus::Out;
use crate::frame::{Frame, MemberState};

/// How long a rejoin waits for its generation's decision.
pub const REJOIN_WAIT: Duration = Duration::from_secs(10);

/// How long a member left cut off for want of a quorum waits to try again.
pub const REJOIN_RETRY: Duration = Duration::from_secs(5);

/// How recently a tie-break reply must have arrived on this member's socket
/// for its gateway to count as the voter every round passes through. Leases
/// renew every 10 s, so a live group hears from the voter far more often.
const VOTER_FRESH: Duration = Duration::from_secs(10 * 60);

fn generation_key(k: u64) -> Vec<u8> {
    format!("gen/{k}").into_bytes()
}

impl Replicator {
    /// A Paxos frame from another member.
    pub(super) fn paxos_frame(self: &Arc<Self>, frame: &Frame) {
        // A reply flagged as the tie-break says this member's gateway is the
        // voter, which every round passes through (`super::quorum_governor`).
        let from_voter = match frame {
            Frame::Promise(p) => p.tiebreak,
            Frame::Nack(n) => n.tiebreak,
            Frame::Accepted(a) => a.tiebreak,
            _ => false,
        };
        if from_voter {
            self.state().voter_seen = Some(Instant::now());
        }
        // Another member handing a lease back says how far its writes reach,
        // for whoever takes the lease next (§9.6).
        if let Frame::Release(r) = frame {
            if r.ballot.node != self.node {
                let _ = self.leases.send(LeaseEvent::Released {
                    key: r.key.clone(),
                    by: r.ballot.node,
                    through: r.through,
                });
            }
        }
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
                // Leases are the app's (§9.2): it follows them as events.
                Out::LeaseHeld { key, .. } => {
                    let _ = self.leases.send(LeaseEvent::Held { key });
                }
                Out::LeaseBusy { key, holder } => {
                    let _ = self.leases.send(LeaseEvent::Busy { key, holder });
                }
                Out::LeaseLost { key } => {
                    let _ = self.leases.send(LeaseEvent::Lost { key });
                }
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

    /// Propose `gen/<k>`, or the first `k` above it no decision names yet. A
    /// member in fault mode proposes nothing (§7.7): it stays cut off, and
    /// tries again once it has left and caught up.
    async fn propose_generation(self: &Arc<Self>, k: u64) {
        if self.faulted() {
            let mut s = self.state();
            s.phase = Phase::CutOff;
            s.retry_rejoin = Some(Instant::now() + REJOIN_RETRY);
            return;
        }
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
        let tiebreak = self.tiebreak_allowed();
        let outs = {
            let mut p = self.paxos.lock().unwrap_or_else(|e| e.into_inner());
            // The quorum as the substrate holds it now: a proposal never runs
            // on a stale one.
            p.set_quorum(quorum);
            p.allow_tiebreak(tiebreak);
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
        let fixed = tokio::task::spawn_blocking(move || {
            let fixed = store.detached();
            store.rejoin(k);
            if fixed {
                store.reload();
            }
            fixed
        })
        .await
        .unwrap_or(false);
        {
            let mut s = self.state();
            s.phase = Phase::Joined;
            s.retry_rejoin = None;
            if fixed {
                s.last_rejoin = Some(k);
            }
        }
        if let Some(m) = self.member() {
            m.capacity_changed();
        }
        self.coord_now();
        self.send_wake.notify_one();
    }

    /// `max_members`, or the ready members seen now when more are seen than
    /// that (§5.9). A member arriving in a group still counted as smaller
    /// would otherwise make a majority of one with itself in the second before
    /// `max_members` rises, and take a lease another member holds.
    fn group_size(&self) -> u32 {
        let held = self.state().max_members;
        held.max(self.ready_members())
    }

    /// A majority of [`Self::group_size`].
    pub(super) fn quorum(&self) -> usize {
        self.group_size() as usize / 2 + 1
    }

    /// A gateway's vote completes a round only in a group of two (§5.9).
    pub(super) fn tiebreak_allowed(&self) -> bool {
        self.group_size() == 2
    }

    /// The once-a-tick consensus work: the engine's timers, the quorum, and a
    /// rejoin that found no quorum.
    pub(super) fn consensus_tick(self: &Arc<Self>) {
        let now = self.now();
        let quorum = self.quorum();
        let tiebreak = self.tiebreak_allowed();
        let outs = {
            let mut p = self.paxos.lock().unwrap_or_else(|e| e.into_inner());
            p.set_quorum(quorum);
            p.allow_tiebreak(tiebreak);
            p.tick(now)
        };
        match outs {
            Ok(outs) => self.dispatch(outs),
            Err(e) => tracing::error!(node = self.node, "the vote journal failed: {e}"),
        }
        self.reaffirm_leases();

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
        // A member in fault mode proposes nothing, and a raise is a proposal.
        let may_raise = !self.faulted();
        let store = self.store.clone();
        let Ok((max_members, generation)) = tokio::task::spawn_blocking(move || {
            let held = store.max_members();
            if may_raise && seen > held {
                store.raise_max_members(seen);
            }
            (store.max_members(), store.generation())
        })
        .await
        else {
            return;
        };
        {
            let mut s = self.state();
            s.max_members = max_members;
            s.generation = generation;
        }
        self.govern_quorum().await;
    }

    /// Whether this member sees the whole group: joined, in no fault, not
    /// drained, and admitted and ready on every cluster it is configured for.
    /// Only such a member lowers `max_members` (§5.9).
    fn sees_whole_group(s: &State) -> bool {
        s.phase == Phase::Joined
            && s.fault.is_none()
            && !s.drained
            && s.sockets.iter().all(Option::is_some)
            && s.ready.iter().all(|r| *r)
    }

    /// Look at the group once for the governor (`super::quorum_governor`), and
    /// write the lower `max_members` it asks for.
    async fn govern_quorum(self: &Arc<Self>) {
        let ready_nodes = self.ready_nodes();
        let ready = ready_nodes.len() as u32;
        // The leases this member's acceptor still counts for members that are
        // not ready: what a lowering to the floor waits for.
        let absent_lease_left = {
            let now = self.now();
            self.paxos
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .running_leases(now)
                .into_iter()
                .filter(|(holder, _)| *holder != self.node && !ready_nodes.contains(holder))
                .map(|(_, left)| left)
                .max()
        };
        let step = {
            let mut s = self.state();
            let look = Look {
                now: Instant::now(),
                max: s.max_members,
                ready,
                eligible: Self::sees_whole_group(&s),
                one_gateway: s.sockets.len() == 1
                    && s.voter_seen.is_some_and(|at| at.elapsed() <= VOTER_FRESH),
                absent_lease_left,
            };
            s.governor.observe(look)
        };
        match step {
            Step::Hold => {}
            Step::Propose {
                from,
                to,
                executes_in,
            } => tracing::warn!(
                node = self.node,
                from,
                to,
                ready,
                executes_in_s = executes_in.as_secs(),
                "the group has stayed short of max_members: lowering it unless the members return"
            ),
            Step::Withdraw { to } => tracing::info!(
                node = self.node,
                to,
                ready,
                "the group is whole again, or this member no longer sees all of it: \
                 max_members stays"
            ),
            Step::Execute { from, to } => {
                tracing::warn!(
                    node = self.node,
                    from,
                    to,
                    ready,
                    "the group stayed short and the absent members' leases have lapsed: \
                     max_members lowered"
                );
                let store = self.store.clone();
                let _ = tokio::task::spawn_blocking(move || store.set_max_members(to)).await;
                self.state().max_members = to;
            }
        }
    }

    /// How many distinct ready members are visible at once across every
    /// cluster, this one included when it is ready.
    pub(super) fn ready_members(&self) -> u32 {
        self.ready_nodes().len() as u32
    }

    /// The distinct ready members visible at once across every cluster, this
    /// one included when it is ready.
    fn ready_nodes(&self) -> BTreeSet<u64> {
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
        ready
    }
}
