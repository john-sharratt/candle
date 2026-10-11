//! The `COORD` report a member sends its gateways (§12.1): built once a second
//! from replication's state, the store and the member, and sent on every
//! socket when it has changed or [`COORD_EVERY`] has passed.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::fault::floor;
use super::replicator::{CatchupView, Replicator};
use crate::coord::{
    AppliedLine, ClaimLine, ClusterLine, CommandLine, CoordState, DeferredView, DiskView,
    FaultView, GenView, KeyLine, OutboxView, PaxosView, TombstoneView,
};
use crate::frame::{Ballot, Coord, Frame};

/// How often `COORD` is sent when nothing has changed. The ticker looks for a
/// change once a second.
pub const COORD_EVERY: Duration = Duration::from_secs(2);

/// The maintenance commands `COORD` reports.
pub const COMMANDS_KEPT: usize = 8;

fn ballot(b: Ballot) -> String {
    format!("{}.{:016x}", b.round, b.node)
}

impl Replicator {
    /// What this member reports.
    pub async fn coord(self: &Arc<Self>) -> Option<CoordState> {
        let store = self.store.clone();
        let (applied, deferred, tombstones, disk, detached) =
            tokio::task::spawn_blocking(move || {
                (
                    store.applied(),
                    store.deferred(),
                    store.tombstones(),
                    store.disk(),
                    store.detached(),
                )
            })
            .await
            .ok()?;
        let member = self.member();
        let now = self.now();
        let keys: Vec<KeyLine> = {
            let p = self.paxos.lock().unwrap_or_else(|e| e.into_inner());
            p.keys()
                .into_iter()
                .map(|k| {
                    let status = p.status(&k, now);
                    KeyLine {
                        key: String::from_utf8_lossy(&k).into_owned(),
                        round: status
                            .round
                            .map(|(b, phase)| format!("{} {phase}", ballot(b))),
                        voters: status.voters.iter().map(|n| format!("{n:016x}")).collect(),
                        beaten_by: status.beaten_by.map(ballot),
                        held_ms: status.held_until.map(|until| until.0.saturating_sub(now.0)),
                        promised: status.promised.map(ballot),
                        accepted: status.accepted.map(ballot),
                    }
                })
                .collect()
        };
        // The member's own locks are taken before replication's, never under it.
        let clusters: Vec<ClusterLine> = member
            .as_ref()
            .map(|m| {
                m.clusters()
                    .into_iter()
                    .map(|v| ClusterLine {
                        url: v.url,
                        state: v.state.into(),
                        session: v.session,
                        last_close: v.last_close,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let claims: Vec<ClaimLine> = member
            .as_ref()
            .map(|m| {
                m.own_claims()
                    .into_iter()
                    .map(|(key, epoch)| ClaimLine {
                        key: String::from_utf8_lossy(&key).into_owned(),
                        epoch_ms: epoch.ms,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let s = self.state();
        let mut high: BTreeMap<u64, u64> = BTreeMap::new();
        for m in s.members.iter().flatten().flat_map(|m| m.members.iter()) {
            let h = high.entry(m.node).or_default();
            *h = (*h).max(m.high_water);
        }
        let (unsent, in_flight) = s.outbox.counts();
        Some(CoordState {
            mode: CoordState::mode_of(s.phase, s.drained, s.fault.is_some()).into(),
            detached,
            fault: s.fault.as_ref().map(|f| FaultView {
                reason: f.reason.clone(),
                since_ms: f.since_ms,
                claims_released: f.claims_released,
                voting: true,
                freed: f.freed,
            }),
            gen: GenView {
                current: s.generation,
                last_rejoin: s.last_rejoin,
                decided: s.generation,
            },
            deferred: deferred
                .into_iter()
                .map(|(record_type, origin, count)| DeferredView {
                    record_type,
                    origin: format!("{origin:016x}"),
                    count,
                })
                .collect(),
            clusters,
            paxos: PaxosView {
                max_members: s.max_members,
                quorum: s.max_members as usize / 2 + 1,
                keys,
            },
            claims,
            applied: applied
                .origins()
                .map(|(origin, a)| AppliedLine {
                    origin: format!("{origin:016x}"),
                    through: a.through(),
                    missing: a.gaps().iter().map(|(s, e)| e - s + 1).sum(),
                    high_water: high.get(&origin).copied().unwrap_or(a.top()),
                })
                .collect(),
            outbox: OutboxView { unsent, in_flight },
            catchups: s
                .catchups
                .values()
                .map(|r| CatchupView {
                    cluster: r.socket,
                    purpose: r.purpose,
                    progress: r.progress().clone(),
                })
                .collect(),
            tombstones: TombstoneView {
                held: tombstones.held,
                waiting_age: tombstones.swept,
                deleted: tombstones.deleted,
            },
            disk: DiskView {
                free: disk.free,
                total: disk.total,
                floor: floor(disk.total),
            },
            stats: s.stats.clone(),
            commands: self
                .commands
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .cloned()
                .collect(),
        })
    }

    /// A maintenance command started or ended on this member: reported in
    /// `COORD` at once, the last [`COMMANDS_KEPT`] of them.
    pub fn command(self: &Arc<Self>, line: CommandLine) {
        {
            let mut commands = self.commands.lock().unwrap_or_else(|e| e.into_inner());
            match commands.iter_mut().find(|c| c.job == line.job) {
                Some(c) => *c = line,
                None => {
                    if commands.len() == COMMANDS_KEPT {
                        commands.pop_front();
                    }
                    commands.push_back(line);
                }
            }
        }
        self.coord_now();
    }

    /// Send `COORD` now: its mode changed, which the gateways act on.
    pub(super) fn coord_now(self: &Arc<Self>) {
        let me = self.clone();
        tokio::spawn(async move { me.send_coord().await });
    }

    /// Send `COORD` on every socket if it changed or has not gone for
    /// [`COORD_EVERY`].
    pub(super) async fn send_coord(self: &Arc<Self>) {
        let Some(state) = self.coord().await else {
            return;
        };
        let bytes = state.encode();
        let handles = {
            let mut s = self.state();
            let due = match &s.coord_sent {
                Some((last, at)) => *last != bytes || at.elapsed() >= COORD_EVERY,
                None => true,
            };
            if !due {
                return;
            }
            s.coord_sent = Some((bytes.clone(), Instant::now()));
            s.admitted()
        };
        for (_, h) in handles {
            let _ = h.control(Frame::Coord(Coord {
                state: bytes.clone(),
            }));
        }
    }
}
