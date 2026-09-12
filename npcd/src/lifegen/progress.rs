//! What the generating overlay reads while a life is being written.
//!
//! # The same shape as the loading screen, and a different instance
//!
//! [`crate::engine::loading`] is the right *shape* — named phases, a counter
//! within one, a detail line — and the ladder happens to fit it exactly, four
//! rungs where startup has five. It is the wrong *instance*: startup progress
//! is global and singular, while a generation belongs to one character and
//! several can be in flight. So this is its own type with its own snapshot,
//! reached through a job id.
//!
//! # Priming and fan-out are reported separately
//!
//! A forked wave finishes in clumps. A bar that sits at 0/12 and then jumps to
//! 12/12 has told the operator nothing, and worse, has told them nothing during
//! precisely the seconds they are wondering whether it is stuck.
//!
//! So the two stages are distinct. [`Stage::Priming`] is the shared prefix being
//! prefilled — one long step with no counter, honestly reported as such —
//! and [`Stage::Generating`] carries both a completed count and how many forks
//! are in flight, which moves as each fork hits its own end.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use serde::Serialize;

use super::plan::Phase;

/// Which half of a phase is running.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Stage {
    /// Prefilling the shared prefix. One decode's worth of work with nothing to
    /// count, so the bar reports the stage rather than inventing a fraction.
    Priming,
    /// Forks in flight against a primed prefix.
    Generating,
    /// Writing documents to disk.
    Writing,
}

impl Stage {
    pub fn label(self) -> &'static str {
        match self {
            Stage::Priming => "Priming the shared context",
            Stage::Generating => "Generating",
            Stage::Writing => "Writing documents",
        }
    }
}

/// How a run ended.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Outcome {
    Done,
    /// Stopped by the operator. Whatever completed before the stop is kept —
    /// a cancelled run is a shorter run, not a discarded one.
    Cancelled,
    Failed {
        error: String,
    },
}

/// A generation's state at a moment.
#[derive(Clone, Debug, Serialize)]
pub struct GenSnapshot {
    pub phase: Phase,
    pub phase_label: &'static str,
    pub stage: Stage,
    pub stage_label: &'static str,
    /// Phases already finished, derived from position so a phase with nothing
    /// to do still counts as behind you.
    pub completed: Vec<Phase>,
    /// 0.0–1.0 within the phase.
    pub progress: f32,
    pub done: u64,
    pub total: u64,
    /// Forks currently decoding. Zero while priming.
    pub in_flight: u64,
    pub unit: &'static str,
    pub detail: String,
    pub elapsed_ms: u64,
    /// `None` while running.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<Outcome>,
}

/// Shared, mutable generation progress.
#[derive(Debug)]
pub struct GenProgress {
    inner: Mutex<Inner>,
    /// Read by the generator between forks. Separate from the mutex so a
    /// cancel never waits on a snapshot, and a snapshot never waits on a
    /// cancel.
    cancel: AtomicBool,
}

#[derive(Debug)]
struct Inner {
    phase: Phase,
    stage: Stage,
    done: u64,
    total: u64,
    in_flight: u64,
    detail: String,
    started: Instant,
    outcome: Option<Outcome>,
}

impl Default for GenProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl GenProgress {
    pub fn new() -> Self {
        GenProgress {
            inner: Mutex::new(Inner {
                phase: Phase::Story,
                stage: Stage::Priming,
                done: 0,
                total: 0,
                in_flight: 0,
                detail: String::new(),
                started: Instant::now(),
                outcome: None,
            }),
            cancel: AtomicBool::new(false),
        }
    }

    /// Enter a phase, priming. Clears the counter and the detail: a count
    /// carried over from the previous phase reads as progress through this one.
    pub fn enter(&self, phase: Phase) {
        let mut g = self.inner.lock().unwrap();
        g.phase = phase;
        g.stage = Stage::Priming;
        g.done = 0;
        g.total = 0;
        g.in_flight = 0;
        g.detail = String::new();
    }

    /// The prefix is primed; `total` forks are about to run.
    pub fn fanning_out(&self, total: u64) {
        let mut g = self.inner.lock().unwrap();
        g.stage = Stage::Generating;
        g.total = total;
        g.done = 0;
    }

    pub fn stage(&self, stage: Stage) {
        self.inner.lock().unwrap().stage = stage;
    }

    /// How many forks are decoding right now.
    pub fn in_flight(&self, n: u64) {
        self.inner.lock().unwrap().in_flight = n;
    }

    /// One fork landed.
    pub fn completed_one(&self, detail: impl Into<String>) {
        let mut g = self.inner.lock().unwrap();
        g.done += 1;
        g.detail = detail.into();
    }

    pub fn detail(&self, detail: impl Into<String>) {
        self.inner.lock().unwrap().detail = detail.into();
    }

    /// Ask the run to stop at the next fork boundary.
    ///
    /// Not a kill. A wave in flight is left to finish, because the alternative
    /// is a half-written document and a slot the engine still thinks is busy.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    pub fn finish(&self, outcome: Outcome) {
        let mut g = self.inner.lock().unwrap();
        g.in_flight = 0;
        g.outcome = Some(outcome);
    }

    pub fn is_finished(&self) -> bool {
        self.inner.lock().unwrap().outcome.is_some()
    }

    pub fn snapshot(&self) -> GenSnapshot {
        let g = self.inner.lock().unwrap();
        GenSnapshot {
            phase: g.phase,
            phase_label: g.phase.label(),
            stage: g.stage,
            stage_label: g.stage.label(),
            completed: Phase::ALL
                .iter()
                .take_while(|p| **p != g.phase)
                .copied()
                .collect(),
            progress: if g.total == 0 {
                0.0
            } else {
                (g.done as f32 / g.total as f32).clamp(0.0, 1.0)
            },
            done: g.done,
            total: g.total,
            in_flight: g.in_flight,
            unit: g.phase.unit(),
            detail: g.detail.clone(),
            elapsed_ms: g.started.elapsed().as_millis() as u64,
            outcome: g.outcome.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_run_is_priming_the_first_phase() {
        let p = GenProgress::new();
        let s = p.snapshot();
        assert_eq!(s.phase, Phase::Story);
        assert_eq!(s.stage, Stage::Priming);
        assert!(s.completed.is_empty());
        assert!(s.outcome.is_none());
        assert!(!p.is_finished());
    }

    /// **Priming has nothing to count, and says so rather than inventing a
    /// fraction.** A bar guessing at 40% through a prefill is a lie the
    /// operator will calibrate against.
    #[test]
    fn priming_reports_no_total_and_a_zero_bar() {
        let p = GenProgress::new();
        p.enter(Phase::Months);
        let s = p.snapshot();
        assert_eq!(s.stage, Stage::Priming);
        assert_eq!((s.done, s.total, s.in_flight), (0, 0, 0));
        assert_eq!(s.progress, 0.0);
        assert_eq!(s.stage_label, "Priming the shared context");
    }

    /// The two stages are distinct because a forked wave finishes in clumps,
    /// and a bar that jumps 0/12 → 12/12 says nothing during the seconds the
    /// operator is wondering whether it is stuck.
    #[test]
    fn fanning_out_switches_stage_and_takes_a_total() {
        let p = GenProgress::new();
        p.enter(Phase::Months);
        p.fanning_out(12);
        p.in_flight(12);
        let s = p.snapshot();
        assert_eq!(s.stage, Stage::Generating);
        assert_eq!((s.done, s.total, s.in_flight), (0, 12, 12));

        p.completed_one("1998-09");
        p.in_flight(11);
        let s = p.snapshot();
        assert_eq!((s.done, s.in_flight), (1, 11));
        assert_eq!(s.detail, "1998-09");
        assert!((s.progress - 1.0 / 12.0).abs() < 1e-6);
    }

    #[test]
    fn completed_phases_are_derived_from_position() {
        let p = GenProgress::new();
        p.enter(Phase::Days);
        assert_eq!(
            p.snapshot().completed,
            vec![Phase::Story, Phase::Years, Phase::Months]
        );
    }

    /// A count carried into the next phase reads as progress through it — the
    /// one piece of state that must not survive a transition.
    #[test]
    fn entering_a_phase_clears_the_previous_counter_and_detail() {
        let p = GenProgress::new();
        p.enter(Phase::Months);
        p.fanning_out(12);
        p.completed_one("1998-09");
        p.enter(Phase::Days);
        let s = p.snapshot();
        assert_eq!((s.done, s.total, s.in_flight), (0, 0, 0));
        assert_eq!(s.detail, "");
        assert_eq!(s.unit, "days", "the unit follows the phase entered");
    }

    /// **Cancel is a request, not a kill.** A wave in flight finishes, because
    /// the alternative is a half-written document and a slot the engine still
    /// thinks is busy.
    #[test]
    fn cancelling_is_visible_before_the_run_ends() {
        let p = GenProgress::new();
        assert!(!p.is_cancelled());
        p.cancel();
        assert!(p.is_cancelled());
        assert!(!p.is_finished(), "a cancel request is not an ending");
        p.finish(Outcome::Cancelled);
        assert!(p.is_finished());
        assert_eq!(p.snapshot().outcome, Some(Outcome::Cancelled));
    }

    #[test]
    fn a_failure_carries_its_reason() {
        let p = GenProgress::new();
        p.finish(Outcome::Failed {
            error: "no engine".into(),
        });
        assert_eq!(
            p.snapshot().outcome,
            Some(Outcome::Failed {
                error: "no engine".into()
            })
        );
    }

    #[test]
    fn progress_is_clamped_and_never_divides_by_zero() {
        let p = GenProgress::new();
        p.fanning_out(0);
        assert_eq!(p.snapshot().progress, 0.0);
        p.fanning_out(2);
        for _ in 0..5 {
            p.completed_one("x");
        }
        assert_eq!(p.snapshot().progress, 1.0);
    }

    #[test]
    fn elapsed_runs_across_the_whole_generation_not_the_phase() {
        let p = GenProgress::new();
        std::thread::sleep(std::time::Duration::from_millis(12));
        p.enter(Phase::Days);
        assert!(p.snapshot().elapsed_ms >= 10);
    }

    #[test]
    fn every_stage_names_itself() {
        for s in [Stage::Priming, Stage::Generating, Stage::Writing] {
            assert!(!s.label().is_empty());
        }
    }
}
