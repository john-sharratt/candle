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
//! [`TurnEvent::PrefillQueued`]: crate::handle::TurnEvent::PrefillQueued

use std::collections::HashMap;
use std::time::{Duration, Instant};

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

/// When each turn last moved, and when it was last told it waits.
#[derive(Debug, Default)]
pub(super) struct PrefillWaits {
    last_sweep: Option<Instant>,
    /// Per turn: the offset last seen, when it last changed, when last told.
    seen: HashMap<usize, (usize, Instant, Option<Instant>)>,
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

    /// The turns in `queue` to tell now, with how long each has stood still:
    /// a turn whose offset has not moved for [`WAIT_EVENT_EVERY`] and was not
    /// told within it. A turn first seen starts its clock now; one no longer in
    /// the queue is forgotten.
    pub(super) fn stalled(&mut self, now: Instant, queue: &[Work]) -> Vec<(usize, Duration)> {
        let mut next = HashMap::with_capacity(queue.len());
        let mut tell = Vec::new();
        for w in queue {
            let Some(turn) = w.turn else { continue };
            let (moved_at, told) = match self.seen.get(&turn) {
                Some(&(offset, moved_at, told)) if offset == w.offset => (moved_at, told),
                _ => (now, None),
            };
            let waited = now.duration_since(moved_at);
            let due = waited >= WAIT_EVENT_EVERY
                && told.is_none_or(|t| now.duration_since(t) >= WAIT_EVENT_EVERY);
            let told = if due {
                tell.push((turn, waited));
                Some(now)
            } else {
                told
            };
            next.insert(turn, (w.offset, moved_at, told));
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
        for (turn, waited) in tell {
            let Some(at) = queue.iter().position(|w| w.turn == Some(turn)) else {
                continue;
            };
            let ahead = ahead_of(&queue, at, running);
            let event = crate::handle::TurnEvent::PrefillQueued {
                ahead_tokens: ahead.tokens,
                ahead_turns: ahead.turns,
                waiting_ms: waited.as_millis() as u64,
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
        assert_eq!(w.stalled(at(1_000), &q), vec![(7, Duration::from_secs(1))]);
        assert!(
            w.stalled(at(1_500), &q).is_empty(),
            "told within the second"
        );
        assert_eq!(w.stalled(at(2_000), &q), vec![(7, Duration::from_secs(2))]);
        let moved = [Work { offset: 64, ..q[0] }];
        assert!(
            w.stalled(at(2_500), &moved).is_empty(),
            "it moved: the clock restarts"
        );
        assert!(w.stalled(at(3_400), &moved).is_empty());
        assert_eq!(
            w.stalled(at(3_500), &moved),
            vec![(7, Duration::from_secs(1))]
        );
        assert!(w.stalled(at(3_600), &[]).is_empty());
        assert!(w.seen.is_empty(), "a turn no longer queued is forgotten");
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
