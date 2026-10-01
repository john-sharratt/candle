//! Lower-priority work waits while higher-priority work is running.
//!
//! A conversation's priority is its target layer's `decode_priority`: the
//! dialogue is `High`, the ingest layers `Low`. While any `High` sequence is
//! queued, prefilling or decoding, `Low` (and `Normal`) sequences are paused —
//! not admitted, and held out of every forward — and they stay paused for
//! [`COOLDOWN`] after the last higher-priority activity, so a conversation
//! running tool rounds keeps the device between its rounds rather than handing
//! it back to ingest for the few seconds each tool takes.
//!
//! Pausing is scheduling only: a paused sequence keeps its K/V, its recurrent
//! state and its place in line, and resumes where it stopped.
//!
//! # Why it exists
//!
//! Ingest shares every forward and every residency structure with the
//! dialogue. With sixteen file workers sealing turns, the dialogue's
//! reprojection paid for their churn and its decode rows rode forwards widened
//! by theirs — measured on a live turn, ingest was the difference between a
//! scan served from cache and one rebuilt from scratch every reprojection. The
//! previous rule only capped ingest decode rows beside a dialogue decode; it
//! kept ingest prefills and scans running, and it lifted the instant the
//! dialogue's decode ended, between every tool round.

use std::time::{Duration, Instant};

use super::{Scheduler, SequenceId};
use crate::projection::DecodePriority;

/// How long lower-priority work stays paused after the last higher-priority
/// activity. Long enough to span a tool round — the dialogue is idle while its
/// tools run — so a multi-round turn keeps the device throughout.
pub const COOLDOWN: Duration = Duration::from_secs(15);

/// How long the loop waits for a request when every sequence with work is
/// paused, before looking again. Short against [`COOLDOWN`], so paused work
/// resumes within this of its release; a request ends the wait at once.
pub const PAUSED_POLL: Duration = Duration::from_millis(250);

/// A band's rank: higher outranks lower.
fn rank(p: DecodePriority) -> usize {
    match p {
        DecodePriority::Low => 0,
        DecodePriority::Normal => 1,
        DecodePriority::High => 2,
    }
}

/// When each priority band was last seen with work, by [`rank`].
#[derive(Debug, Default, Clone)]
pub struct PriorityPause {
    last_active: [Option<Instant>; 3],
}

impl PriorityPause {
    /// Record that a sequence of priority `p` has work at `now` — queued,
    /// prefilling or decoding.
    pub fn observe(&mut self, p: DecodePriority, now: Instant) {
        let slot = &mut self.last_active[rank(p)];
        if slot.is_none_or(|t| t < now) {
            *slot = Some(now);
        }
    }

    /// Whether work of priority `p` is paused at `now`: some strictly higher
    /// band had work within [`COOLDOWN`].
    pub fn paused(&self, p: DecodePriority, now: Instant) -> bool {
        self.last_active[rank(p) + 1..]
            .iter()
            .flatten()
            .any(|&t| now.saturating_duration_since(t) < COOLDOWN)
    }
}

impl Scheduler {
    /// Record the priority of every sequence with work right now — queued,
    /// prefilling, or decoding. Called where each gate decides, so a gate
    /// always judges against the present. A queued dialogue turn counts, so
    /// ingest stops the moment a question arrives rather than once it is
    /// admitted.
    ///
    /// Only a priority that resolves is recorded. A slot with no resolvable
    /// target (a summary probe, a slot reloaded without its projection) is not
    /// paused — it counts as `High` for its own gating — but it is not evidence
    /// that a conversation is running either, so it must not hold ingest and
    /// the normalization warm-up back for a cooldown after it.
    pub(super) fn observe_priorities(&mut self) {
        let now = Instant::now();
        let ids: Vec<SequenceId> = self
            .active_decodes
            .iter()
            .filter(|(_, s)| !s.finished)
            .map(|(&id, _)| id)
            .chain(
                self.active_prefills
                    .iter()
                    .filter(|p| p.error.is_none())
                    .map(|p| p.work.sequence_id),
            )
            .chain(self.prefill_queue.iter().map(|w| w.sequence_id))
            .collect();
        for id in ids {
            if let Some(p) = self.decode_layer_priority(id) {
                self.priority_pause.observe(p, now);
            }
        }
    }

    /// In-flight prefills that can advance: not errored and not paused.
    ///
    /// What admission's progress gates count. A paused prefill holds a slot and
    /// its KV but moves no token, so counting it as "in flight" let a queue of
    /// paused ingest prefills keep the width full and the keep-one-alive rule
    /// silent — while the dialogue turn queued behind them, the very thing
    /// pausing them, was never admitted.
    pub(super) fn running_prefills(&self) -> usize {
        self.active_prefills
            .iter()
            .filter(|p| p.error.is_none() && !self.priority_paused(p.work.sequence_id))
            .count()
    }

    /// Whether sequence `id` is paused behind higher-priority work.
    pub(super) fn priority_paused(&self, id: SequenceId) -> bool {
        self.priority_pause
            .paused(self.decode_priority_or_high(id), Instant::now())
    }

    /// Whether there is sequence work and all of it is paused — the loop then
    /// waits for a request instead of spinning through quanta that would run
    /// nothing.
    ///
    /// Every entry counts, finished or not: a finished prefill still needs
    /// promoting and a finished decode still needs sealing, and if it belongs to
    /// the band that is running it is not paused, so the loop proceeds. A `High`
    /// sequence is never paused (no band outranks it), so this can hold only
    /// while nothing but lower-priority work remains in its cooldown. Section
    /// ingests and deferred glue are not paused by priority, so either one
    /// keeps the loop running.
    pub(super) fn all_work_paused(&mut self) -> bool {
        if !self.active_section_ingests.is_empty() || !self.deferred_glue_fires.is_empty() {
            return false;
        }
        self.observe_priorities();
        let ids: Vec<SequenceId> = self
            .active_decodes
            .keys()
            .copied()
            .chain(self.active_prefills.iter().map(|p| p.work.sequence_id))
            .chain(self.prefill_queue.iter().map(|w| w.sequence_id))
            .collect();
        !ids.is_empty() && ids.iter().all(|&id| self.priority_paused(id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use DecodePriority::{High, Low, Normal};

    #[test]
    fn nothing_is_paused_when_nothing_higher_has_run() {
        let now = Instant::now();
        let mut p = PriorityPause::default();
        p.observe(Low, now);
        assert!(!p.paused(Low, now));
        assert!(!p.paused(Normal, now));
        assert!(!p.paused(High, now));
    }

    /// A running dialogue pauses ingest, and never itself.
    #[test]
    fn high_work_pauses_everything_below_it() {
        let now = Instant::now();
        let mut p = PriorityPause::default();
        p.observe(High, now);
        p.observe(Low, now);
        assert!(p.paused(Low, now));
        assert!(p.paused(Normal, now));
        assert!(!p.paused(High, now));
    }

    /// The cooldown: still paused just inside it, released exactly at it.
    #[test]
    fn the_pause_outlives_the_high_work_by_the_cooldown() {
        let t0 = Instant::now();
        let mut p = PriorityPause::default();
        p.observe(High, t0);
        assert!(p.paused(Low, t0 + COOLDOWN - Duration::from_millis(1)));
        assert!(!p.paused(Low, t0 + COOLDOWN));
    }

    /// Each round of a multi-round turn renews the pause, so ingest never gets
    /// the gap between rounds.
    #[test]
    fn repeated_high_activity_keeps_renewing_the_pause() {
        let t0 = Instant::now();
        let mut p = PriorityPause::default();
        for round in 0..5u64 {
            p.observe(High, t0 + Duration::from_secs(10 * round));
        }
        assert!(p.paused(Low, t0 + Duration::from_secs(40 + 14)));
        assert!(!p.paused(Low, t0 + Duration::from_secs(40 + 15)));
    }

    /// Normal pauses Low but not High; an older observation never rewinds a
    /// newer one.
    #[test]
    fn normal_outranks_low_only_and_time_never_goes_back() {
        let t0 = Instant::now();
        let mut p = PriorityPause::default();
        p.observe(Normal, t0 + Duration::from_secs(5));
        p.observe(Normal, t0);
        assert!(p.paused(Low, t0 + Duration::from_secs(19)));
        assert!(!p.paused(High, t0 + Duration::from_secs(5)));
        assert!(!p.paused(Normal, t0 + Duration::from_secs(5)));
    }
}
