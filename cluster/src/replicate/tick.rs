//! The replicator's clock (§7.1, §6.7, §7.7, §12.1): consensus timers every
//! tick, and once a second the disk check, `COORD`, the `ACK`, the gap watch,
//! the periodic catch-up while a socket is down, catch-ups whose answerer went
//! quiet, and cutting off when no socket is left.

use std::future::Future;
use std::mem::replace;
use std::pin::pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use applied_set::AppliedSet;
use tokio::time::{interval, timeout, MissedTickBehavior};

use super::catchup::Purpose;
use super::receipt_watch::Seen;
use super::replicator::{Phase, Replicator};
use super::watch::periodic_due;
use crate::frame::{Ack, Frame};
use crate::member::Link;

/// How often the consensus engine's timers run.
const TICK: Duration = Duration::from_millis(100);

/// How often the slower work runs.
const SECOND: Duration = Duration::from_secs(1);

/// A step of the slower work taking this long is logged, and again each time
/// this passes while it still waits.
const STEP_WATCH: Duration = Duration::from_secs(30);

/// A catch-up whose answerer has said nothing for this long is asked again.
pub const QUIET: Duration = Duration::from_secs(30);

/// Run `step`, saying so while it has not finished. Everything after a step
/// waits on it: a tick that stopped left a gap catch-up silent for an hour,
/// never dropped, with the log saying nothing about where it stood.
async fn watched<F: Future>(step: &'static str, fut: F) -> F::Output {
    let started = Instant::now();
    let mut fut = pin!(fut);
    loop {
        match timeout(STEP_WATCH, &mut fut).await {
            Ok(out) => return out,
            Err(_) => tracing::warn!(
                step,
                waited_s = started.elapsed().as_secs(),
                "the replicator's tick is waiting on a step: nothing after it runs"
            ),
        }
    }
}

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

/// How often a joined member records that it is (§7.8). The absence that
/// decides what an open drops is counted in days, so a minute is exact enough
/// and costs one small file write a minute.
pub const JOINED_MARK: Duration = Duration::from_secs(60);

/// Whether the store is to be marked joined at `now`: joined, with a socket
/// whose admission catch-up has finished, and not marked within
/// [`JOINED_MARK`].
pub fn joined_mark_due(
    phase: Phase,
    caught_up: bool,
    marked: Option<Instant>,
    now: Instant,
) -> bool {
    phase == Phase::Joined
        && caught_up
        && marked.is_none_or(|at| now.duration_since(at) >= JOINED_MARK)
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
        watched("check_disk", self.check_disk()).await;
        watched("send_coord", self.send_coord()).await;
        watched("cut_off_if_alone", self.cut_off_if_alone()).await;
        watched("refresh_group", self.refresh_group()).await;
        watched("mark_joined", self.mark_joined()).await;
        self.note_load();
        self.watch_receipts();
        // In fault mode a member applies nothing, so it acknowledges nothing
        // and asks for nothing; leaving starts its catch-ups (§7.7).
        if self.faulted() {
            return;
        }
        let store = self.store.clone();
        let read = tokio::task::spawn_blocking(move || (store.applied(), store.own_through()));
        let Ok((applied, own_through)) = watched("applied", read).await else {
            return;
        };
        let now = Instant::now();
        let (ack, gaps, periodic, quiet) = {
            let mut s = self.state();
            let ack = replace(&mut s.ack_dirty, false).then(|| Ack {
                applied: ack_lines(&applied, self.node, own_through),
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
            for blind in s.gaps.blind(highs.iter().copied(), &applied, now) {
                tracing::warn!(
                    node = self.node,
                    origin = format_args!("{:016x}", blind.origin),
                    announced = blind.high,
                    top = blind.top,
                    missing = blind.missing,
                    "this member holds an origin past what it announces, so no catch-up will \
                     ask for what it lacks: the origin's position is stuck"
                );
            }
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

    /// Log a stall of this member's receipts, and its end (`receipt_watch`).
    fn watch_receipts(&self) {
        let (seen, unsent, oldest) = {
            let mut s = self.state();
            let (unsent, in_flight) = s.outbox.counts();
            let oldest = s.outbox.oldest_in_flight();
            (
                s.receipts.observe(in_flight, Instant::now()),
                unsent,
                oldest,
            )
        };
        match seen {
            Seen::Quiet => {}
            Seen::Stalled { in_flight, secs } => tracing::warn!(
                node = format_args!("{:016x}", self.node),
                in_flight,
                unsent,
                stalled_s = secs,
                oldest_oseq = oldest.map(|(o, _)| o),
                cluster = oldest.map(|(_, c)| c),
                "own events in flight have had no receipt: the gateway is not taking them in"
            ),
            Seen::Resumed { in_flight, secs } => tracing::info!(
                node = format_args!("{:016x}", self.node),
                in_flight,
                unsent,
                stalled_s = secs,
                "own events are being receipted again"
            ),
        }
    }

    /// Record that the member is joined and caught up, when due
    /// ([`joined_mark_due`]).
    async fn mark_joined(self: &Arc<Self>) {
        let now = Instant::now();
        {
            let mut s = self.state();
            let caught_up = s
                .ready
                .iter()
                .zip(&s.sockets)
                .any(|(ready, socket)| *ready && socket.is_some());
            if !joined_mark_due(s.phase, caught_up, s.joined_marked, now) {
                return;
            }
            s.joined_marked = Some(now);
        }
        let store = self.store.clone();
        let _ = tokio::task::spawn_blocking(move || store.mark_joined()).await;
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

/// An `ACK`'s lines: every origin's contiguous position, and for this member's
/// own, all it may say it wrote. That line is what peers learn its high water
/// from, so it cannot stop at a hole: one `oseq` a purge left in BlackWorld's
/// zend held its announcement 60 million below its top, and every peer read the
/// origin as complete and fetched nothing it had lost (2026-10-31).
fn ack_lines(applied: &AppliedSet, node: u64, own_through: u64) -> Vec<(u64, u64)> {
    applied
        .origins()
        .map(|(origin, held)| {
            let line = if origin == node {
                own_through.max(held.through())
            } else {
                held.through()
            };
            (origin, line)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use applied_set::AppliedSet;

    use super::Phase;
    use super::{
        ack_lines, after_quiet_admission, is_quiet, joined_mark_due, QuietAdmission,
        ADMISSION_QUIET_LIMIT, JOINED_MARK, QUIET,
    };

    /// **A member's own line passes a hole in what it holds of itself**;
    /// another origin's stays contiguous.
    #[test]
    fn an_ack_says_all_this_member_wrote_past_a_hole() {
        let mut applied = AppliedSet::default();
        for oseq in [1, 2, 4, 5] {
            applied.insert(7, oseq);
            applied.insert(9, oseq);
        }
        assert_eq!(ack_lines(&applied, 7, 5), [(7, 5), (9, 2)]);
        assert_eq!(
            ack_lines(&applied, 7, 0),
            [(7, 2), (9, 2)],
            "never below what it holds contiguously"
        );
    }

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

    /// Marked joined only while joined and caught up on some socket, and at
    /// most once a [`JOINED_MARK`].
    #[test]
    fn a_joined_caught_up_member_is_marked_once_a_minute() {
        let t0 = Instant::now();
        let just = Duration::from_secs(1);
        assert!(joined_mark_due(Phase::Joined, true, None, t0));
        assert!(!joined_mark_due(Phase::Joined, false, None, t0));
        assert!(!joined_mark_due(Phase::CutOff, true, None, t0));
        assert!(!joined_mark_due(
            Phase::Joined,
            true,
            Some(t0),
            t0 + JOINED_MARK - just
        ));
        assert!(joined_mark_due(
            Phase::Joined,
            true,
            Some(t0),
            t0 + JOINED_MARK
        ));
    }
}
