//! Telling a waiting turn what it waits behind ([`TurnEvent::PrefillQueued`]).
//!
//! A turn's prefill can stand still for tens of seconds — queued behind a creep
//! cohort, an ingest sweep, other turns' prompts — and from the caller's side
//! that silence looked like a hang: a zen-code tool round on `.5` (2026-10-18)
//! sat 12–22 s between its `PrefillProgress { 0, N }` and the next. So while a
//! prompt makes no progress the scheduler tells it, every
//! [`WAIT_EVENT_EVERY`], how long it has waited and how much work stands ahead
//! of it.
//!
//! "Ahead" is what the scheduler will run before this turn's next row: the
//! creep cohort in flight, and every other waiting prompt or ingest of at least
//! this turn's band that came before it — a lower band rides beside it, never
//! in front.
//!
//! **A turn behind work that moves is not stalled.** Each tell says whether a
//! prefill window ran since the last one (`ahead_moved`), and a caller's stall
//! clock counts that as the turn moving (`TurnHandle::moves`). Beside a person's
//! decode a `Low` cohort rides one layer a window, about once in 65 steps'
//! worth of decode (`throttle_ratio`), so a sweep of a deep cohort outlasts the
//! 60 s a background wait allows. Told only that they waited, every file read
//! queued behind it was given up mid-read on BlackWorld (2026-11-19): nine files
//! at once, 66 s into a 58 s reply, each read thrown away and begun again.
//!
//! [`TurnEvent::PrefillQueued`]: crate::handle::TurnEvent::PrefillQueued

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::handle::TurnEvent;

/// How often a waiting turn is told it is still waiting.
pub(super) const WAIT_EVENT_EVERY: Duration = Duration::from_secs(1);
/// How often the scheduler looks for waiting turns at all.
const SWEEP_EVERY: Duration = Duration::from_millis(250);

/// One piece of prompt work, in the order the scheduler will reach it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Work {
    /// The turn's sequence, or `None` for work no caller waits on (an ingest).
    pub(super) turn: Option<usize>,
    /// Its band's rank (`band::rank`).
    pub(super) band: usize,
    /// Prompt tokens it still has to run.
    pub(super) remaining: usize,
    /// Its prompt has advanced this far.
    pub(super) offset: usize,
}

/// What stands ahead of one waiting turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(super) struct Ahead {
    pub(super) tokens: usize,
    pub(super) turns: usize,
}

/// The work ahead of `queue[at]`: `running` — the cohort in flight, which the
/// turn is not in — and every piece before it in `queue` of at least its band.
pub(super) fn ahead_of(queue: &[Work], at: usize, running: Ahead) -> Ahead {
    let band = queue[at].band;
    queue[..at]
        .iter()
        .filter(|w| w.band >= band && w.remaining > 0)
        .fold(running, |a, w| Ahead {
            tokens: a.tokens + w.remaining,
            turns: a.turns + 1,
        })
}

/// One turn told it waits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Told {
    pub(super) turn: usize,
    /// Since its offset last moved.
    pub(super) waited: Duration,
    /// Whether a prefill window ran since it was last told, or since it was
    /// first seen waiting.
    pub(super) ahead_moved: bool,
}

/// One turn's wait, as last seen.
#[derive(Debug, Clone, Copy)]
struct Seen {
    offset: usize,
    moved_at: Instant,
    told: Option<Instant>,
    /// [`PrefillWaits::windows`] when it was last told, or first seen.
    windows: u64,
}

/// When each turn last moved, and when it was last told it waits.
#[derive(Debug, Default)]
pub(super) struct PrefillWaits {
    last_sweep: Option<Instant>,
    seen: HashMap<usize, Seen>,
    /// Prefill windows the scheduler has run: every forward that advanced the
    /// creep cohort or a chunk, whatever its members.
    windows: u64,
}

impl PrefillWaits {
    /// Whether a sweep is due: at most every [`SWEEP_EVERY`].
    pub(super) fn due(&mut self, now: Instant) -> bool {
        if self
            .last_sweep
            .is_some_and(|t| now.duration_since(t) < SWEEP_EVERY)
        {
            return false;
        }
        self.last_sweep = Some(now);
        true
    }

    /// A prefill window ran: the work ahead of every waiting turn moved.
    pub(super) fn window_ran(&mut self) {
        self.windows += 1;
    }

    /// The turns in `queue` to tell now: a turn whose offset has not moved for
    /// [`WAIT_EVENT_EVERY`] and was not told within it, with how long it has
    /// stood still and whether a window ran since it was last told. A turn
    /// first seen starts its clock now; one no longer in the queue is
    /// forgotten.
    pub(super) fn stalled(&mut self, now: Instant, queue: &[Work]) -> Vec<Told> {
        let mut next = HashMap::with_capacity(queue.len());
        let mut tell = Vec::new();
        for w in queue {
            let Some(turn) = w.turn else { continue };
            let mut seen = match self.seen.get(&turn) {
                Some(s) if s.offset == w.offset => *s,
                _ => Seen {
                    offset: w.offset,
                    moved_at: now,
                    told: None,
                    windows: self.windows,
                },
            };
            let waited = now.duration_since(seen.moved_at);
            let due = waited >= WAIT_EVENT_EVERY
                && seen
                    .told
                    .is_none_or(|t| now.duration_since(t) >= WAIT_EVENT_EVERY);
            if due {
                tell.push(Told {
                    turn,
                    waited,
                    ahead_moved: self.windows != seen.windows,
                });
                seen.told = Some(now);
                seen.windows = self.windows;
            }
            next.insert(turn, seen);
        }
        self.seen = next;
        tell
    }
}

impl super::Scheduler {
    /// Tell every turn whose prefill has stood still for a second what it is
    /// waiting behind ([`TurnEvent::PrefillQueued`]). Cheap when nothing
    /// waits, and looked at no more than every [`SWEEP_EVERY`].
    ///
    /// [`TurnEvent::PrefillQueued`]: crate::handle::TurnEvent::PrefillQueued
    pub(super) fn report_prefill_waits(&mut self) {
        let now = Instant::now();
        if !self.prefill_waits.due(now) {
            return;
        }
        // The cohort in flight is running, not waiting: its members are not
        // told, and it stands ahead of everyone else.
        let in_cohort: Vec<usize> = self
            .wave_prefill_members
            .iter()
            .map(|m| m.seq_id())
            .collect();
        let running = Ahead {
            tokens: self
                .wave_prefill_members
                .iter()
                .map(|m| match m {
                    super::WaveMember::Prefill { advance, .. }
                    | super::WaveMember::Section { advance, .. } => *advance,
                })
                .sum(),
            turns: in_cohort.len(),
        };
        let band = |s: &Self, seq| super::band::rank(s.decode_priority_or_high(seq));
        let mut queue: Vec<Work> = Vec::new();
        for p in &self.active_prefills {
            let seq = p.work.sequence_id;
            if p.error.is_some() || p.finished.is_some() || in_cohort.contains(&seq.0) {
                continue;
            }
            let remaining = p.work.tokens.len().saturating_sub(p.offset);
            if remaining > 0 {
                queue.push(Work {
                    turn: Some(seq.0),
                    band: band(self, seq),
                    remaining,
                    offset: p.offset,
                });
            }
        }
        for w in &self.prefill_queue {
            queue.push(Work {
                turn: Some(w.sequence_id.0),
                band: band(self, w.sequence_id),
                remaining: w.tokens.len(),
                offset: 0,
            });
        }
        for s in &self.active_section_ingests {
            if s.error.is_some() || in_cohort.contains(&s.sequence_id.0) {
                continue;
            }
            queue.push(Work {
                turn: None,
                band: band(self, s.sequence_id),
                remaining: s.tokens.len().saturating_sub(s.offset),
                offset: s.offset,
            });
        }
        for s in &self.section_queue {
            queue.push(Work {
                turn: None,
                band: band(self, s.sequence_id),
                remaining: s.tokens.len(),
                offset: 0,
            });
        }
        let tell = self.prefill_waits.stalled(now, &queue);
        for Told {
            turn,
            waited,
            ahead_moved,
        } in tell
        {
            let Some(at) = queue.iter().position(|w| w.turn == Some(turn)) else {
                continue;
            };
            let ahead = ahead_of(&queue, at, running);
            let event = TurnEvent::PrefillQueued {
                ahead_tokens: ahead.tokens,
                ahead_turns: ahead.turns,
                waiting_ms: waited.as_millis() as u64,
                ahead_moved,
            };
            let tx = self
                .active_prefills
                .iter()
                .find(|p| p.work.sequence_id.0 == turn)
                .map(|p| &p.work.event_tx)
                .or_else(|| {
                    self.prefill_queue
                        .iter()
                        .find(|w| w.sequence_id.0 == turn)
                        .map(|w| &w.event_tx)
                });
            if let Some(tx) = tx {
                let _ = tx.send(event);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn(seq: usize, band: usize, remaining: usize) -> Work {
        Work {
            turn: Some(seq),
            band,
            remaining,
            offset: 0,
        }
    }

    fn ingest(band: usize, remaining: usize) -> Work {
        Work {
            turn: None,
            band,
            remaining,
            offset: 0,
        }
    }

    /// Ahead of a turn: the running cohort, and what came before it of at
    /// least its band — a lower band rides beside it, never in front.
    #[test]
    fn ahead_is_the_cohort_and_earlier_work_of_its_band_or_above() {
        let queue = [
            ingest(1, 8_000),
            turn(1, 2, 300),
            ingest(1, 4_000),
            turn(2, 2, 50),
        ];
        let running = Ahead {
            tokens: 8_190,
            turns: 1,
        };
        assert_eq!(
            ahead_of(&queue, 3, running),
            Ahead {
                tokens: 8_490,
                turns: 2
            },
            "the cohort and turn 1; the Normal ingests ride beside a High turn"
        );
        assert_eq!(
            ahead_of(&queue, 2, Ahead::default()),
            Ahead {
                tokens: 8_300,
                turns: 2
            }
        );
        assert_eq!(ahead_of(&queue, 0, Ahead::default()), Ahead::default());
    }

    /// A turn is told once it has stood still a second, then once a second
    /// while it waits; moving resets its clock, and a turn that left is
    /// forgotten.
    #[test]
    fn a_still_turn_is_told_each_second_and_a_moving_one_never() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut w = PrefillWaits::default();
        let q = [turn(7, 2, 100)];
        assert!(w.stalled(at(0), &q).is_empty(), "just seen");
        assert!(w.stalled(at(900), &q).is_empty(), "still under a second");
        assert_eq!(w.stalled(at(1_000), &q), vec![told(7, 1_000, false)]);
        assert!(
            w.stalled(at(1_500), &q).is_empty(),
            "told within the second"
        );
        assert_eq!(w.stalled(at(2_000), &q), vec![told(7, 2_000, false)]);
        let moved = [Work { offset: 64, ..q[0] }];
        assert!(
            w.stalled(at(2_500), &moved).is_empty(),
            "it moved: the clock restarts"
        );
        assert!(w.stalled(at(3_400), &moved).is_empty());
        assert_eq!(w.stalled(at(3_500), &moved), vec![told(7, 1_000, false)]);
        assert!(w.stalled(at(3_600), &[]).is_empty());
        assert!(w.seen.is_empty(), "a turn no longer queued is forgotten");
    }

    fn told(turn: usize, waited_ms: u64, ahead_moved: bool) -> Told {
        Told {
            turn,
            waited: Duration::from_millis(waited_ms),
            ahead_moved,
        }
    }

    /// **Behind work that moves, a still turn is told so.** A window run since
    /// the turn was last told — or since it was first seen — reads as the work
    /// ahead having moved; a tell with no window since reads as not. The turn's
    /// own clock still runs from its own offset: what is ahead moving is not the
    /// turn moving, and it says how long it has itself stood still.
    #[test]
    fn a_tell_says_whether_a_window_ran_since_the_last() {
        let t0 = Instant::now();
        let at = |ms: u64| t0 + Duration::from_millis(ms);
        let mut w = PrefillWaits::default();
        let q = [turn(7, 0, 100)];
        assert!(w.stalled(at(0), &q).is_empty(), "just seen");
        w.window_ran();
        assert_eq!(
            w.stalled(at(1_000), &q),
            vec![told(7, 1_000, true)],
            "a window ran after it was first seen"
        );
        assert_eq!(
            w.stalled(at(2_000), &q),
            vec![told(7, 2_000, false)],
            "none since the last tell"
        );
        w.window_ran();
        w.window_ran();
        assert_eq!(w.stalled(at(3_000), &q), vec![told(7, 3_000, true)]);
        assert_eq!(w.stalled(at(4_000), &q), vec![told(7, 4_000, false)]);
    }

    /// A window that ran before a turn queued is not work it waited behind.
    #[test]
    fn a_window_before_the_turn_queued_is_not_counted() {
        let t0 = Instant::now();
        let mut w = PrefillWaits::default();
        w.window_ran();
        let q = [turn(3, 0, 10)];
        assert!(w.stalled(t0, &q).is_empty());
        assert_eq!(
            w.stalled(t0 + Duration::from_secs(1), &q),
            vec![told(3, 1_000, false)]
        );
    }

    /// Ingest work carries no caller, so it is never told — only counted.
    #[test]
    fn ingests_are_counted_not_told() {
        let t0 = Instant::now();
        let mut w = PrefillWaits::default();
        let q = [ingest(1, 8_000)];
        w.stalled(t0, &q);
        assert!(w.stalled(t0 + Duration::from_secs(5), &q).is_empty());
    }
}
