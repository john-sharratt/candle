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

/// Unanswered admission catch-ups after which a socket is taken as caught up.
///
/// **One stuck member must not keep another from serving.** The answerer is
/// another member, and a member whose own sends stall answers nothing: on
/// 2026-10-17 WhiteWorld's npcd, ready at the gateway but with its sockets
/// dropping every 45 s, was asked `.5`'s admission catch-up again and again,
/// and `.5`'s npcd never finished loading — so `keep` rolled the build back
/// after its ten minutes. Past this many quiet asks the member serves on what
/// it holds: anything it lacks is a gap the gap watch fetches once an origin
/// announces it, from whichever member can answer.
pub const ADMISSION_QUIET_LIMIT: u32 = 2;

/// Whether a catch-up last heard of at `word` has gone quiet at `now`: no word
/// of it for [`QUIET`], and no answer arriving on its socket either
/// (`arriving`: the last catch-up frame, or byte of a message not yet whole).
/// Over a slow link a reply is still arriving while its frame is not yet whole,
/// and a fetched record of many fragments lands as a whole only at its last.
/// An idle socket's `PING`s are whole messages and no catch-up's, so they never
/// hold one: a request that will get no answer still goes quiet. The gateway
/// says `LOST` when it knows one will not come — and once told `LOST`, the
/// catch-up's own silence is all that counts.
pub fn is_quiet(now: Instant, word: Instant, arriving: Option<Instant>, lost: bool) -> bool {
    let silent = |at: Instant| now.duration_since(at) >= QUIET;
    silent(word) && (lost || arriving.is_none_or(silent))
}

/// What an admission catch-up that went quiet does next, given how many times
/// it already had: ask again, or stop asking and serve.
pub fn after_quiet_admission(quiet_before: u32) -> QuietAdmission {
    if quiet_before + 1 >= ADMISSION_QUIET_LIMIT {
        QuietAdmission::Serve
    } else {
        QuietAdmission::AskAgain
    }
}

/// See [`after_quiet_admission`].
#[derive(Debug, PartialEq, Eq)]
pub enum QuietAdmission {
    AskAgain,
    Serve,
}

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
            let socket_heard = |c: usize| {
                s.sockets
                    .get(c)
                    .and_then(Option::as_ref)
                    .and_then(|h| h.answer_heard())
                    .map(|at| at.into_std())
            };
            let quiet: Vec<(u64, usize, Purpose)> = s
                .heard
                .iter()
                .filter_map(|(id, at)| {
                    let r = s.catchups.get(id)?;
                    is_quiet(now, *at, socket_heard(r.socket), s.lost.contains(id))
                        .then_some((*id, r.socket, r.purpose))
                })
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
            if purpose != Purpose::Admission {
                tracing::warn!(
                    node = self.node,
                    cluster = c,
                    "a catch-up went quiet; dropped, the gap watch asks again"
                );
                continue;
            }
            let quiet_before = {
                let mut s = self.state();
                let n = s.admission_quiet[c];
                s.admission_quiet[c] = n + 1;
                n
            };
            match after_quiet_admission(quiet_before) {
                QuietAdmission::AskAgain => {
                    tracing::warn!(
                        node = self.node,
                        cluster = c,
                        "an admission catch-up went quiet: asking again"
                    );
                    self.start_catchup(c, Purpose::Admission, None);
                }
                QuietAdmission::Serve => {
                    tracing::warn!(
                        node = self.node,
                        cluster = c,
                        asks = quiet_before + 1,
                        "an admission catch-up went unanswered {} times: serving on what this \
                         member holds; what it lacks arrives as gaps",
                        quiet_before + 1
                    );
                    self.finished(Purpose::Admission, c).await;
                }
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

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::{after_quiet_admission, is_quiet, QuietAdmission, ADMISSION_QUIET_LIMIT, QUIET};

    /// **A reply still arriving is not silence.** AlienWorld's gap catch-ups
    /// were dropped every 32 s while BlackWorld's ~3 KB/s kept arriving on the
    /// socket — a listing of any size takes longer than 30 s to arrive whole —
    /// and asked again from scratch, for ever. A message part-way across holds
    /// a catch-up; an idle socket (its pings whole, nothing arriving) does not,
    /// or one that will get no answer would never be asked again — which held
    /// AlienWorld's for half an hour when an idle socket's pings counted. Once
    /// the gateway says `LOST`, its own silence alone drops it.
    #[test]
    fn a_catch_up_is_quiet_only_when_no_message_is_arriving_unless_lost() {
        let t0 = Instant::now();
        let later = t0 + QUIET;
        let just = Duration::from_secs(1);
        // No word, and no message arriving: quiet.
        assert!(is_quiet(later, t0, Some(t0), false));
        assert!(is_quiet(later, t0, None, false));
        // No word, but a message arriving: a reply on its way.
        assert!(!is_quiet(later, t0, Some(later - just), false));
        // Told LOST: its silence is all that counts.
        assert!(is_quiet(later, t0, Some(later - just), true));
        // A word within QUIET: never quiet.
        assert!(!is_quiet(later, later - just, None, true));
        assert!(!is_quiet(later - just, t0, None, false));
    }

    /// The first quiet admission is asked again; the one that reaches the
    /// limit serves.
    #[test]
    fn a_quiet_admission_is_asked_again_then_served() {
        assert_eq!(ADMISSION_QUIET_LIMIT, 2);
        assert_eq!(after_quiet_admission(0), QuietAdmission::AskAgain);
        assert_eq!(after_quiet_admission(1), QuietAdmission::Serve);
        assert_eq!(after_quiet_admission(7), QuietAdmission::Serve);
    }
}
