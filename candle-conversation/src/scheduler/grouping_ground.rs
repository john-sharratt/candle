//! The engine's side of [`super::grouping`]: what a reading counts, and what a
//! phase holds.
//!
//! Everything is counted below `Urgent`. A person is waiting on `Urgent` work
//! — a live dialogue, the check a draw waits on — so it is admitted in every
//! phase, and it is not the work a phase groups.

use std::time::{Duration, Instant};

use super::band::rank;
use super::grouping::{Phase, Reading};
use super::{Scheduler, SequenceId};
use crate::projection::DecodePriority;

impl Scheduler {
    /// Whether `prio` is below `Urgent`: the work grouping counts and holds.
    fn grouped(prio: DecodePriority) -> bool {
        prio != DecodePriority::Urgent
    }

    /// The engine as [`super::grouping::Grouping::step`] reads it, at `now`.
    fn grouping_reading(&self, now: Instant) -> Reading {
        let resident = self.model.expert_stats().and_then(|s| {
            let total = (s.total_experts as u64).saturating_mul(s.expert_slot_bytes as u64);
            (total > 0).then(|| s.resident_vram_bytes as f64 / total as f64)
        });
        let band_of = |id: SequenceId| self.decode_priority_or_high(id);
        let queued = self
            .prefill_queue
            .iter()
            .filter(|w| Self::grouped(band_of(w.sequence_id)))
            .map(|w| (w.tokens.len(), w.submitted_at));
        let pending = self
            .pending_submits
            .iter()
            .filter(|p| Self::grouped(band_of(p.parent())))
            .map(|p| (p.work.tokens.len(), p.work.submitted_at));
        let sections = self
            .section_queue
            .iter()
            .filter(|s| Self::grouped(s.band))
            .map(|s| (s.tokens.len(), s.queued_at));
        let (mut waiting_rows, mut oldest) = (0usize, None::<Instant>);
        for (rows, at) in queued.chain(pending).chain(sections) {
            waiting_rows += rows;
            oldest = Some(oldest.map_or(at, |o| o.min(at)));
        }
        let prefilling: usize = self
            .active_prefills
            .iter()
            .filter(|p| p.error.is_none() && p.finished.is_none())
            .filter(|p| Self::grouped(band_of(p.work.sequence_id)))
            .map(|p| p.work.tokens.len().saturating_sub(p.offset))
            .sum();
        let ingesting: usize = self
            .active_section_ingests
            .iter()
            .filter(|s| s.error.is_none() && Self::grouped(s.band))
            .map(|s| s.tokens.len().saturating_sub(s.offset))
            .sum();
        Reading {
            resident,
            waiting_rows,
            oldest_wait: oldest.map_or(Duration::ZERO, |o| now.saturating_duration_since(o)),
            sweeping_rows: prefilling + ingesting,
            decodes: self
                .foreground_decodes()
                .filter(|&id| Self::grouped(band_of(id)))
                .count(),
            sweep: self
                .sweep_forecast
                .sweep_secs()
                .filter(|s| s.is_finite() && *s >= 0.0)
                .map(Duration::from_secs_f64),
        }
    }

    /// Read the engine and move the grouping to the phase it calls for. A
    /// change out of a decode phase reopens admission at once, so the prompts
    /// it held are offered on this pass rather than at the next completion.
    pub(super) fn advance_grouping(&mut self) {
        let now = Instant::now();
        let reading = self.grouping_reading(now);
        let Some(t) = self.grouping.step(now, &reading) else {
            return;
        };
        if t.from == Phase::Decode {
            self.settled_since_admit = true;
        }
        tracing::info!(
            target: "candle_conversation::scheduler::grouping",
            from = t.from.name(),
            to = t.to.name(),
            why = t.why,
            lasted_ms = t.lasted.as_millis() as u64,
            resident_pct = t.resident.map(|r| (r * 100.0).round() as u64),
            waiting_rows = reading.waiting_rows,
            oldest_wait_ms = reading.oldest_wait.as_millis() as u64,
            sweeping_rows = reading.sweeping_rows,
            decodes = reading.decodes,
            "grouping phase",
        );
    }

    /// Whether the phase holds a prompt or section of band `prio`: a decode
    /// phase holds what is below `Urgent` while a decode it does not outrank
    /// is running. A prompt above every grouped decode is what they would be
    /// set aside for, as `Ground::outranks_active` has it, so it is not held.
    pub(super) fn grouping_holds(&self, prio: DecodePriority) -> bool {
        if self.grouping.phase() != Phase::Decode || !Self::grouped(prio) {
            return false;
        }
        self.foreground_decodes().any(|id| {
            let band = self.decode_priority_or_high(id);
            Self::grouped(band) && rank(band) >= rank(prio)
        })
    }

    /// Whether the phase holds the work the keep-one-alive rule would force
    /// in: the highest band queued. While a decode phase holds it the engine
    /// is not stalled — its decodes step, and the phase ends on the prompts'
    /// age ([`super::grouping::MAX_WAIT`]).
    pub(super) fn grouping_holds_head(&self) -> bool {
        let queued = self
            .prefill_queue
            .iter()
            .map(|w| self.decode_priority_or_high(w.sequence_id))
            .chain(
                self.pending_submits
                    .iter()
                    .map(|p| self.decode_priority_or_high(p.parent())),
            )
            .chain(self.section_queue.iter().map(|s| s.band))
            .max_by_key(|&b| rank(b));
        queued.is_some_and(|b| self.grouping_holds(b))
    }

    /// Whether a creep of band `work` runs beside its own band's decodes at
    /// full share: in a prefill phase the group it carries is the phase's
    /// work, and held to the share a live band keeps beside its own decodes
    /// it would sweep a third of the device while the slice runs out.
    /// Decodes that outrank it still protect themselves by their ratios.
    pub(super) fn grouping_widens_creep(&self, work: DecodePriority) -> bool {
        self.grouping.phase() == Phase::Prefill && Self::grouped(work)
    }
}

#[cfg(test)]
mod tests {
    use super::super::grouping::Grouping;
    use super::super::tests::{boundary_state, make_test_scheduler};
    use super::*;
    use DecodePriority::{High, Low, Normal, Urgent};

    /// A scheduler decoding one sequence of each band in `bands`, standing in
    /// `phase`.
    fn decoding(phase: Phase, bands: &[DecodePriority]) -> Scheduler {
        let (mut sched, _tx) = make_test_scheduler();
        for &band in bands {
            let id = SequenceId(sched.session.create_sequence().expect("create"));
            let (state, _rx) = boundary_state();
            sched.active_decodes.insert(id, state);
            sched.slot_bands.insert(id, band);
        }
        sched.grouping = Grouping::standing(phase, 0.23);
        sched
    }

    /// A decode phase holds every band below `Urgent` that a running grouped
    /// decode does not outrank, and nothing `Urgent`.
    #[test]
    fn a_decode_phase_holds_what_its_decodes_do_not_outrank() {
        let sched = decoding(Phase::Decode, &[High]);
        assert!(sched.grouping_holds(High));
        assert!(sched.grouping_holds(Normal));
        assert!(sched.grouping_holds(Low));
        assert!(!sched.grouping_holds(Urgent), "a person is waiting on it");

        let below = decoding(Phase::Decode, &[Normal]);
        assert!(
            !below.grouping_holds(High),
            "it outranks every grouped decode"
        );
        assert!(below.grouping_holds(Normal));
        assert!(below.grouping_holds(Low));
    }

    /// An `Urgent` decode is not the phase's work: it holds nothing.
    #[test]
    fn an_urgent_decode_holds_nothing() {
        let sched = decoding(Phase::Decode, &[Urgent]);
        for band in [High, Normal, Low] {
            assert!(!sched.grouping_holds(band), "{band:?}");
        }
    }

    /// Outside a decode phase nothing is held.
    #[test]
    fn only_a_decode_phase_holds() {
        for phase in [Phase::Off, Phase::Mixed, Phase::Prefill] {
            let sched = decoding(phase, &[High, Normal, Low]);
            for band in [Urgent, High, Normal, Low] {
                assert!(!sched.grouping_holds(band), "{phase:?} {band:?}");
            }
        }
    }

    /// In a prefill phase a creep stops yielding to its own band's decodes —
    /// `High` beside `High` takes the whole share — while decodes that
    /// outrank it keep their ratio, and `Urgent` keeps its own.
    #[test]
    fn a_prefill_phase_creeps_beside_its_own_band_at_full_share() {
        let sched = decoding(Phase::Prefill, &[High]);
        assert_eq!(sched.airtime_ratio_over(High), 1);
        assert_eq!(sched.airtime_ratio_over(Low), 64, "the decode outranks it");
        let decode = decoding(Phase::Decode, &[High]);
        assert_eq!(
            decode.airtime_ratio_over(High),
            2,
            "the share a live band keeps"
        );
        let urgent = decoding(Phase::Prefill, &[Urgent]);
        assert_eq!(urgent.airtime_ratio_over(Urgent), 2, "not grouped");
        assert_eq!(urgent.airtime_ratio_over(High), 64);
    }

    /// A card reporting no expert cache — the test model — stays off, however
    /// much it decodes.
    #[test]
    fn a_card_without_an_expert_cache_never_groups() {
        let (mut sched, _tx) = make_test_scheduler();
        for _ in 0..3 {
            let id = SequenceId(sched.session.create_sequence().expect("create"));
            let (state, _rx) = boundary_state();
            sched.active_decodes.insert(id, state);
            sched.slot_bands.insert(id, High);
        }
        sched.advance_grouping();
        sched.advance_grouping();
        assert_eq!(sched.grouping.phase(), Phase::Off);
        assert!(!sched.grouping_holds(High));
    }
}
