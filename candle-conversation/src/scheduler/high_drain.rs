//! How fast this card drains its `High` prefill backlog, per wall second.
//!
//! What a member advertises as its prefill rate is what the gateway divides
//! that backlog, and a placed turn's prompt, by. The rate model's figure is a
//! forward's throughput on a card doing nothing else, at the width admission
//! would offer — and beside running decodes no forward runs that way: the trade
//! narrows the cohort, the decodes take their steps, and the prompt drains at a
//! fraction of it. On BlackWorld (a965e3a07, 2026-10-19) heavy-read's backlog
//! of 30–54K `High` tokens drained at a few hundred a second while the model,
//! learning nothing from 128-row cohorts it judged too narrow to fit, held at
//! 1,670 — every finish priced minutes early.
//!
//! So the meter counts the `High` prefill tokens the sweeps actually advanced
//! while a `High` backlog stood, over the wall time it stood, both decayed with
//! one time constant: the rate a turn joining that backlog gets now, measured
//! rather than modelled. Time with no backlog standing teaches nothing about a
//! queue's drain, and is not counted.

use std::time::Instant;

/// The meter's memory: evidence older than this weighs `1/e` of fresh. A
/// minute is the gateway's own horizon — its rows are fifteen seconds apart,
/// and a turn placed now finishes inside a few of them.
const TAU_SECS: f64 = 60.0;

/// The standing time the meter needs before its rate is the advertised one:
/// a sixth of its memory, so a backlog that has just formed is not judged by
/// the sweep or two it has had.
const MIN_STANDING_SECS: f64 = TAU_SECS / 6.0;

#[derive(Debug, Default)]
pub(super) struct HighDrain {
    /// Decayed `High` tokens advanced while a backlog stood.
    tokens: f64,
    /// Decayed seconds a backlog stood.
    standing_secs: f64,
    /// When the meter last advanced, and whether a backlog stood then.
    at: Option<(Instant, bool)>,
}

impl HighDrain {
    /// Advance the meter to `now`, with `standing` whether a `High` backlog
    /// stands. The interval since the last advance counts as standing when a
    /// backlog stood at its start — the state the sweeps in it ran under.
    pub(super) fn advance(&mut self, now: Instant, standing: bool) {
        if let Some((then, stood)) = self.at {
            let dt = now.saturating_duration_since(then).as_secs_f64();
            let keep = (-dt / TAU_SECS).exp();
            self.tokens *= keep;
            self.standing_secs *= keep;
            if stood {
                self.standing_secs += dt;
            }
        }
        self.at = Some((now, standing));
    }

    /// Count `tokens` of `High` prefill a sweep advanced — only while a
    /// backlog stands, so a turn that arrived and drained between two advances
    /// is not read as a queue's drain.
    pub(super) fn drained(&mut self, tokens: usize) {
        if self.at.is_some_and(|(_, stood)| stood) {
            self.tokens += tokens as f64;
        }
    }

    /// Tokens a second the backlog drained at, once it has stood long enough
    /// to say; `None` before.
    pub(super) fn rate(&self) -> Option<f64> {
        (self.standing_secs >= MIN_STANDING_SECS).then(|| self.tokens / self.standing_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// The heavy-read shape: a backlog standing for a minute, drained at 300
    /// tokens a second in sweeps, reads as 300 — not as anything the sweeps'
    /// own forward throughput says.
    #[test]
    fn a_standing_backlog_reads_its_drain_per_wall_second() {
        let mut m = HighDrain::default();
        let t0 = Instant::now();
        m.advance(t0, true);
        for i in 1..=60 {
            m.drained(300);
            m.advance(t0 + Duration::from_secs(i), true);
        }
        let r = m.rate().expect("a minute of standing backlog");
        assert!((r - 300.0).abs() < 300.0 * 0.05, "read {r}");
    }

    /// No backlog, no evidence: an idle card's drain is not a queue's.
    #[test]
    fn time_with_no_backlog_teaches_nothing() {
        let mut m = HighDrain::default();
        let t0 = Instant::now();
        m.advance(t0, false);
        for i in 1..=120 {
            m.drained(2_000);
            m.advance(t0 + Duration::from_secs(i), false);
        }
        assert_eq!(m.rate(), None);
    }

    /// A backlog that has only just formed is not judged yet.
    #[test]
    fn a_fresh_backlog_is_not_judged_by_its_first_sweeps() {
        let mut m = HighDrain::default();
        let t0 = Instant::now();
        m.advance(t0, true);
        m.drained(4_000);
        m.advance(t0 + Duration::from_secs(2), true);
        assert_eq!(m.rate(), None);
    }

    /// The drain follows the card: a slower minute after a fast one moves the
    /// figure most of the way down within that minute.
    #[test]
    fn a_slower_drain_is_followed() {
        let mut m = HighDrain::default();
        let t0 = Instant::now();
        m.advance(t0, true);
        let mut t = 0;
        for _ in 0..120 {
            m.drained(1_600);
            t += 1;
            m.advance(t0 + Duration::from_secs(t), true);
        }
        for _ in 0..60 {
            m.drained(200);
            t += 1;
            m.advance(t0 + Duration::from_secs(t), true);
        }
        let r = m.rate().expect("standing");
        assert!(r < 800.0, "read {r} a minute into a 200 tok/s drain");
    }
}
