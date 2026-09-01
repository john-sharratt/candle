//! Daemon load-state machine — what the loading screen reads.
//!
//! The counterpart of `zend/src/loading.rs`, with npcd's own phases. A restart
//! is not instant here: the weights are gigabytes, the redo log has to be
//! replayed, and the mind's layers have to be diffed against what the substrate
//! already holds. That is tens of seconds during which the console must show
//! something truthful rather than an empty shell or a spinner that means
//! nothing.
//!
//! # The phases, and why each is separately visible
//!
//! A single "starting…" tells an operator nothing about *where* a slow start is
//! slow, and the four phases fail for completely different reasons — a missing
//! GGUF, a corrupt segment, an unreadable mind directory. Naming each one turns
//! "npcd is hanging" into "npcd is on Replaying substrate, 40 000 of 250 000
//! turns", which is a fact somebody can act on.
//!
//! [`LoadProgress`] is the single source of truth. The loader advances it with
//! [`LoadProgress::set_step`], reports within a step via
//! [`LoadProgress::set_progress`], and finishes with
//! [`LoadProgress::mark_ready`]. `GET /v1/status` reads a snapshot.

use std::sync::Mutex;
use std::time::Instant;

use serde::Serialize;

/// Phases of startup, in execution order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadStep {
    /// Fetch (if absent) and load the GGUF onto the card.
    Model,
    /// Replay the redo log at `.substrate/` into the in-RAM substrate.
    Substrate,
    /// Prefill the tool catalog's calibration examples so tool selection is
    /// calibrated rather than cold. See `engine::tools`.
    ///
    /// **Before [`LoadStep::Layers`], and the order is load-bearing.** Every
    /// layer frames on the shared system prompt, and the tool catalog is part of
    /// that prompt. A turn prefilled while the catalog is still absent captures
    /// its KV — and the wide-Q signature the gather matches against — under a
    /// prompt that is not the one any character will ever think under. The
    /// documents would be *in* the substrate and subtly mismatched to every
    /// query made of them, which is the worst kind of wrong: nothing fails, and
    /// retrieval is quietly worse than it should be.
    Calibrating,
    /// Diff the mind directory against what the substrate holds and ingest the
    /// difference — the step that puts a world's documents where a character can
    /// reach them.
    Layers,
    /// Bring the cast up: one inbox and one scheduled heartbeat per character.
    Waking,
}

impl LoadStep {
    /// Canonical order. Each phase depends on the one before it: the substrate
    /// needs the model's engine, calibration needs the substrate to prefill
    /// into, the layers need the prompt calibration completes, and the cast
    /// needs the world it is about to think about.
    pub const ALL: &'static [LoadStep] = &[
        LoadStep::Model,
        LoadStep::Substrate,
        LoadStep::Calibrating,
        LoadStep::Layers,
        LoadStep::Waking,
    ];

    /// The line the overlay renders.
    pub fn label(self) -> &'static str {
        match self {
            LoadStep::Model => "Loading model",
            LoadStep::Substrate => "Replaying substrate",
            LoadStep::Layers => "Ingesting mind layers",
            LoadStep::Calibrating => "Calibrating tools",
            LoadStep::Waking => "Waking the cast",
        }
    }

    /// The noun this step's counter counts, rendered as "N / M unit" beside the
    /// bar. Empty where the counter is a scaled fraction rather than a count of
    /// anything a person would name, in which case only the bar shows.
    pub fn unit(self) -> &'static str {
        match self {
            LoadStep::Model => "layers",
            LoadStep::Substrate => "turns",
            LoadStep::Layers => "files",
            LoadStep::Calibrating => "tools",
            LoadStep::Waking => "characters",
        }
    }
}

/// The loading state at a moment.
#[derive(Clone, Debug, Serialize)]
pub struct LoadingSnapshot {
    pub current: LoadStep,
    pub label: &'static str,
    /// 0.0–1.0 within the current step.
    pub progress: f32,
    pub completed: Vec<LoadStep>,
    /// Absolute position inside the step. `total == 0` means nothing countable
    /// has been reported yet, and the bar reads zero rather than guessing.
    pub progressed: u64,
    pub total: u64,
    pub unit: String,
    /// A specific thing being worked on right now — the file being ingested, the
    /// character being woken. Sub-step detail, so one phase can name an
    /// arbitrary number of items without needing a phase each.
    pub detail: String,
    pub elapsed_ms: u64,
}

/// Shared, mutable load progress. A `None` snapshot means ready — the console
/// drops the overlay and shows the app.
pub struct LoadProgress {
    inner: Mutex<Inner>,
}

enum Inner {
    Loading {
        current: LoadStep,
        progressed: u64,
        total: u64,
        unit: String,
        detail: String,
        started: Instant,
    },
    Ready,
}

impl Default for LoadProgress {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadProgress {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner::Loading {
                current: LoadStep::Model,
                progressed: 0,
                total: 0,
                unit: LoadStep::Model.unit().to_string(),
                detail: String::new(),
                started: Instant::now(),
            }),
        }
    }

    /// Enter a phase. Resets the counter and the detail line, because a count
    /// carried over from the previous phase is worse than no count — it reads as
    /// progress through the new one.
    pub fn set_step(&self, step: LoadStep) {
        let mut g = self.inner.lock().unwrap();
        let started = match &*g {
            Inner::Loading { started, .. } => *started,
            Inner::Ready => Instant::now(),
        };
        *g = Inner::Loading {
            current: step,
            progressed: 0,
            total: 0,
            unit: step.unit().to_string(),
            detail: String::new(),
            started,
        };
    }

    /// Report position within the current phase.
    pub fn set_progress(&self, progressed: u64, total: u64) {
        if let Inner::Loading {
            progressed: p,
            total: t,
            ..
        } = &mut *self.inner.lock().unwrap()
        {
            *p = progressed;
            *t = total;
        }
    }

    /// Name the specific item being worked on.
    pub fn set_detail(&self, detail: impl Into<String>) {
        if let Inner::Loading { detail: d, .. } = &mut *self.inner.lock().unwrap() {
            *d = detail.into();
        }
    }

    /// Override the counted noun — the `Layers` phase counts whatever the
    /// layer being ingested counts, which the phase itself cannot know.
    pub fn set_unit(&self, unit: impl Into<String>) {
        if let Inner::Loading { unit: u, .. } = &mut *self.inner.lock().unwrap() {
            *u = unit.into();
        }
    }

    /// Startup is over. Every later snapshot is `None`.
    pub fn mark_ready(&self) {
        *self.inner.lock().unwrap() = Inner::Ready;
    }

    pub fn is_ready(&self) -> bool {
        matches!(&*self.inner.lock().unwrap(), Inner::Ready)
    }

    /// `None` once ready.
    pub fn snapshot(&self) -> Option<LoadingSnapshot> {
        let g = self.inner.lock().unwrap();
        let Inner::Loading {
            current,
            progressed,
            total,
            unit,
            detail,
            started,
        } = &*g
        else {
            return None;
        };
        // Everything strictly before the current step, by the canonical order.
        // Derived rather than accumulated so a phase that is skipped entirely
        // still counts as done — which is the normal case for `Layers` on a
        // daemon with no mind directory.
        let completed = LoadStep::ALL
            .iter()
            .take_while(|s| *s != current)
            .copied()
            .collect();
        Some(LoadingSnapshot {
            current: *current,
            label: current.label(),
            progress: if *total == 0 {
                0.0
            } else {
                (*progressed as f32 / *total as f32).clamp(0.0, 1.0)
            },
            completed,
            progressed: *progressed,
            total: *total,
            unit: unit.clone(),
            detail: detail.clone(),
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_progress_is_on_the_first_step_and_not_ready() {
        let p = LoadProgress::new();
        let s = p.snapshot().expect("loading");
        assert_eq!(s.current, LoadStep::Model);
        assert!(s.completed.is_empty());
        assert_eq!(s.progress, 0.0);
        assert!(!p.is_ready());
    }

    /// The completed list is derived from position, not accumulated as steps
    /// run. A phase that does no work — `Layers` with no mind directory — is
    /// still behind you once you are past it, and an accumulated list would
    /// leave a gap in the overlay's tick marks.
    #[test]
    fn completed_is_everything_before_the_current_step() {
        let p = LoadProgress::new();
        p.set_step(LoadStep::Layers);
        let s = p.snapshot().expect("loading");
        assert_eq!(
            s.completed,
            vec![LoadStep::Model, LoadStep::Substrate, LoadStep::Calibrating]
        );
    }

    /// **Calibration precedes the layers, and the order is load-bearing.**
    ///
    /// Every layer frames on the shared system prompt, and the tool catalog is
    /// part of it. A document prefilled before the catalog exists captures its
    /// KV under a prompt no character will ever think under — the turns land,
    /// nothing errors, and retrieval is quietly worse than it should be for the
    /// life of that substrate.
    #[test]
    fn the_prompt_is_complete_before_anything_is_prefilled_against_it() {
        let pos = |want: LoadStep| LoadStep::ALL.iter().position(|s| *s == want).unwrap();
        assert!(
            pos(LoadStep::Calibrating) < pos(LoadStep::Layers),
            "layers are ingested before the prompt they frame on is finished"
        );
        // And both sit after the substrate they write into, and before the cast
        // that reads them.
        assert!(pos(LoadStep::Substrate) < pos(LoadStep::Calibrating));
        assert!(pos(LoadStep::Layers) < pos(LoadStep::Waking));
    }

    #[test]
    fn ready_has_no_snapshot() {
        let p = LoadProgress::new();
        p.mark_ready();
        assert!(p.snapshot().is_none());
        assert!(p.is_ready());
    }

    /// A count carried into the next phase reads as progress through it. This is
    /// the one piece of state that must not survive a transition.
    #[test]
    fn entering_a_step_clears_the_previous_steps_counter_and_detail() {
        let p = LoadProgress::new();
        p.set_progress(400, 500);
        p.set_detail("qwen3.5-9b.gguf");
        p.set_step(LoadStep::Substrate);
        let s = p.snapshot().expect("loading");
        assert_eq!((s.progressed, s.total), (0, 0));
        assert_eq!(s.progress, 0.0);
        assert_eq!(s.detail, "");
        assert_eq!(s.unit, "turns", "the unit follows the step it entered");
    }

    /// A total of zero is "nothing counted yet", not "zero of zero done" — the
    /// difference between a bar at 0% and a division by zero.
    #[test]
    fn an_uncounted_step_reads_zero_rather_than_dividing_by_zero() {
        let p = LoadProgress::new();
        p.set_progress(7, 0);
        assert_eq!(p.snapshot().unwrap().progress, 0.0);
    }

    #[test]
    fn progress_is_clamped_to_the_step() {
        let p = LoadProgress::new();
        p.set_progress(900, 500);
        assert_eq!(p.snapshot().unwrap().progress, 1.0);
    }

    /// Every step must name itself and its unit, or the overlay renders a blank
    /// line where a phase should be.
    #[test]
    fn every_step_has_a_label() {
        for s in LoadStep::ALL {
            assert!(!s.label().is_empty(), "{s:?} has no label");
        }
        assert_eq!(
            LoadStep::ALL.len(),
            5,
            "a step was added without a decision"
        );
    }

    /// Elapsed time runs across the whole load, not per phase — an operator
    /// watching a slow start wants "this has been going 90 seconds", and a timer
    /// that reset each phase would never show more than the current one.
    #[test]
    fn elapsed_survives_a_step_change() {
        let p = LoadProgress::new();
        std::thread::sleep(std::time::Duration::from_millis(12));
        p.set_step(LoadStep::Waking);
        assert!(p.snapshot().unwrap().elapsed_ms >= 10);
    }
}
