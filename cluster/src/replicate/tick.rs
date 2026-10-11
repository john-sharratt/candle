//! The replicator's clock (§7.1, §6.7, §7.7, §12.1): consensus timers every
//! tick, and once a second the disk check, `COORD`, the `ACK`, the gap watch,
//! the periodic catch-up while a socket is down, catch-ups whose answerer went
//! quiet, and cutting off when no socket is left.

use std::mem::replace;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::time::{interval, MissedTickBehavior};

use super::catchup::Purpose;
use super::replicator::{Phase, Replicator};
use super::watch::periodic_due;
use crate::frame::{Ack, Frame};
use crate::member::Link;

/// How often the consensus engine's timers run.
const TICK: Duration = Duration::from_millis(100);

/// How often the slower work runs.
const SECOND: Duration = Duration::from_secs(1);

/// A catch-up whose answerer has said nothing for this long is asked again.
pub const QUIET: Duration = Duration::from_secs(30);

impl Replicator {
    pub(super) async fn ticker(self: Arc<Self>) {
        let mut tick = interval(TICK);
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut last = Instant::now();
        loop {
            tick.tick().await;
            self.consensus_tick();
            if last.elapsed() < SECOND {
                continue;
            }
            last = Instant::now();
            self.second().await;
        }
    }

    async fn second(self: &Arc<Self>) {
        self.check_disk().await;
        self.send_coord().await;
        self.cut_off_if_alone().await;
        self.refresh_group().await;
        // In fault mode a member applies nothing, so it acknowledges nothing
        // and asks for nothing; leaving starts its catch-ups (§7.7).
        if self.faulted() {
            return;
        }
        let store = self.store.clone();
        let Ok(applied) = tokio::task::spawn_blocking(move || store.applied()).await else {
            return;
        };
        let now = Instant::now();
        let (ack, gaps, periodic, quiet) = {
            let mut s = self.state();
            let ack = replace(&mut s.ack_dirty, false).then(|| Ack {
                applied: applied.origins().map(|(o, a)| (o, a.through())).collect(),
            });

            // An origin announcing more than this member holds, for 5 s.
            let highs: Vec<(u64, u64)> = s
                .members
                .iter()
                .flatten()
                .flat_map(|m| m.members.iter())
                .filter(|m| m.node != self.node)
                .map(|m| (m.node, m.high_water))
                .collect();
            let due = s.gaps.observe(highs, &applied, now);
            let gaps: Vec<(usize, u64)> = due
                .into_iter()
                .filter(|origin| {
                    !s.catchups
                        .values()
                        .any(|r| r.purpose == Purpose::Gap(*origin))
                })
                .filter_map(|origin| {
                    let c = s.members.iter().position(|m| {
                        m.as_ref()
                            .is_some_and(|m| m.members.iter().any(|x| x.node == origin))
                    })?;
                    s.sockets[c].is_some().then_some((c, origin))
                })
                .collect();

            // Every 30 s while a socket is down, on the sockets still up.
            let down = s.sockets.iter().any(Option::is_none);
            let periodic: Vec<usize> = if periodic_due(down, s.last_periodic, now) {
                s.last_periodic = now;
                (0..s.sockets.len())
                    .filter(|c| {
                        s.sockets[*c].is_some() && !s.catchups.values().any(|r| r.socket == *c)
                    })
                    .collect()
            } else {
                Vec::new()
            };

            // A catch-up whose answerer went quiet.
            let quiet: Vec<(u64, usize, Purpose)> = s
                .heard
                .iter()
                .filter(|(_, at)| now.duration_since(**at) >= QUIET)
                .filter_map(|(id, _)| s.catchups.get(id).map(|r| (*id, r.socket, r.purpose)))
                .collect();
            for (id, _, _) in &quiet {
                s.drop_catchup(*id);
            }
            (ack, gaps, periodic, quiet)
        };
        if let Some(ack) = ack {
            for (_, handle) in self.state().admitted() {
                let _ = handle.control(Frame::Ack(ack.clone()));
            }
        }
        for (c, origin) in gaps {
            self.start_catchup(c, Purpose::Gap(origin), Some(origin));
        }
        for c in periodic {
            self.start_catchup(c, Purpose::Periodic, None);
        }
        for (_, c, purpose) in quiet {
            tracing::warn!(
                node = self.node,
                cluster = c,
                "a catch-up went quiet: asking again"
            );
            if purpose == Purpose::Admission {
                self.start_catchup(c, Purpose::Admission, None);
            }
        }
    }

    /// With no socket admitted and the member's link cut off, what it writes
    /// from here on is detached (§6.7).
    async fn cut_off_if_alone(self: &Arc<Self>) {
        let alone = self.state().sockets.iter().all(Option::is_none);
        if !alone {
            return;
        }
        let cut_off = self
            .member()
            .is_some_and(|m| m.socket_link() == Link::CutOff);
        if !cut_off || self.state().phase == Phase::CutOff {
            return;
        }
        let store = self.store.clone();
        let _ = tokio::task::spawn_blocking(move || store.detach()).await;
        self.state().phase = Phase::CutOff;
        tracing::warn!(
            node = self.node,
            "cut off from every cluster: writing detached"
        );
    }
}
