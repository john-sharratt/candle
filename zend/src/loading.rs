//! Daemon load-state machine — drives the frontend's loading screen.
//!
//! The daemon's startup walks through a fixed sequence of phases:
//!
//! 1. **Model** — fetch and load the GGUF weights.
//! 2. **Substrate** — replay the redo log into the in-RAM substrate.
//! 3. **Sections** — prefill the projection schema's pinned sections.
//! 4. **RepoScan** — *(future)* index the workspace's code into the substrate.
//! 5. **CodeRead** — *(future)* run the first per-file prefill pass.
//!
//! Steps 4 and 5 currently have no work to do — they transition through
//! instantly. Wiring them up is left for the integrations that own them.
//!
//! `LoadProgress` is the single source of truth; the daemon advances it
//! via [`Self::set_step`], reports intra-step progress via
//! [`Self::set_progress`], and finalises with [`Self::mark_ready`]. The
//! `GET /v1/status` endpoint reads a snapshot.

use std::sync::Mutex;
use std::time::Instant;

/// Phases of daemon startup, in the order they execute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadStep {
    Model,
    Substrate,
    Compacting,
    Sections,
    RepoScan,
    CodeRead,
}

impl LoadStep {
    /// Canonical ordering of all steps — used to derive the "completed"
    /// list from the current step (everything before it is done).
    pub const ALL: &'static [LoadStep] = &[
        LoadStep::Model,
        LoadStep::Substrate,
        LoadStep::Compacting,
        LoadStep::Sections,
        LoadStep::RepoScan,
        LoadStep::CodeRead,
    ];

    /// Human-readable label rendered in the loading overlay.
    pub fn label(self) -> &'static str {
        match self {
            LoadStep::Model => "Loading model",
            LoadStep::Substrate => "Loading substrate",
            LoadStep::Compacting => "Compacting substrate",
            LoadStep::Sections => "Prefilling sections",
            LoadStep::RepoScan => "Scanning repository",
            LoadStep::CodeRead => "Reading code",
        }
    }
}

/// Snapshot of the loading state at a moment in time.
#[derive(Clone, Debug)]
pub struct LoadingSnapshot {
    pub current: LoadStep,
    pub progress: f32,
    pub completed: Vec<LoadStep>,
}

/// Shared, mutable load-progress state. `None`-snapshot means the daemon
/// is fully ready and the frontend should show the chat view.
pub struct LoadProgress {
    inner: Mutex<Inner>,
}

enum Inner {
    Loading {
        current: LoadStep,
        /// Real `(current, total)` counter — layers for model load,
        /// section-content chars for prefill, etc. `0/0` means "no
        /// progress reported yet for this step" → bar reads 0%.
        progressed: u64,
        total: u64,
        /// Wall-clock instant the current step was entered — used for
        /// per-step elapsed-time logging at transitions.
        started: Instant,
    },
    Ready,
}

impl LoadProgress {
    /// Begin at the first step (`Model`).
    pub fn new() -> Self {
        tracing::info!(step = LoadStep::Model.label(), "load step started");
        Self {
            inner: Mutex::new(Inner::Loading {
                current: LoadStep::Model,
                progressed: 0,
                total: 0,
                started: Instant::now(),
            }),
        }
    }

    /// Take a snapshot. `None` once [`Self::mark_ready`] has been called.
    pub fn snapshot(&self) -> Option<LoadingSnapshot> {
        match &*self.inner.lock().unwrap() {
            Inner::Loading {
                current,
                progressed,
                total,
                ..
            } => {
                let progress = if *total > 0 {
                    (*progressed as f32 / *total as f32).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let completed: Vec<LoadStep> = LoadStep::ALL
                    .iter()
                    .copied()
                    .take_while(|s| s != current)
                    .collect();
                Some(LoadingSnapshot {
                    current: *current,
                    progress,
                    completed,
                })
            }
            Inner::Ready => None,
        }
    }

    /// Whether the daemon has finished loading.
    ///
    /// Kept as a typed boolean API for callers that don't want to
    /// parse `snapshot() -> Option<LoadingSnapshot>` themselves; only
    /// the unit tests exercise it today, hence `#[allow(dead_code)]`.
    #[allow(dead_code)]
    pub fn is_ready(&self) -> bool {
        matches!(*self.inner.lock().unwrap(), Inner::Ready)
    }

    /// Advance to `step`. Progress resets to `0/0`. Callers don't need
    /// to "complete" the previous step explicitly — moving forward
    /// implicitly marks everything before `step` as done.
    ///
    /// Logs the previous step's elapsed wall-clock time at INFO. A
    /// transition to the same step (e.g. a redundant call right after
    /// `new()`) is a no-op so timings stay clean.
    pub fn set_step(&self, step: LoadStep) {
        let mut guard = self.inner.lock().unwrap();
        if let Inner::Loading {
            current, started, ..
        } = &*guard
        {
            if *current == step {
                return;
            }
            tracing::info!(
                step = current.label(),
                elapsed_ms = started.elapsed().as_millis() as u64,
                "load step complete",
            );
        }
        tracing::info!(step = step.label(), "load step started");
        *guard = Inner::Loading {
            current: step,
            progressed: 0,
            total: 0,
            started: Instant::now(),
        };
    }

    /// Report real progress within the current step. `current` and
    /// `total` are domain-specific integers — layers loaded out of
    /// total layers, section content chars done out of total, etc.
    /// `total == 0` is allowed and means "indeterminate" (snapshot
    /// returns 0% in that case).
    ///
    /// Cheap to call on a tight loop (one mutex acquisition).
    pub fn set_step_progress(&self, current: u64, total: u64) {
        let mut guard = self.inner.lock().unwrap();
        if let Inner::Loading {
            progressed, total: t, ..
        } = &mut *guard
        {
            *progressed = current.min(total);
            *t = total;
        }
    }

    /// Finalise: the daemon is ready, all phases complete. Subsequent
    /// `snapshot()` calls return `None`. Logs the last step's elapsed
    /// time before flipping over.
    pub fn mark_ready(&self) {
        let mut guard = self.inner.lock().unwrap();
        if let Inner::Loading {
            current, started, ..
        } = &*guard
        {
            tracing::info!(
                step = current.label(),
                elapsed_ms = started.elapsed().as_millis() as u64,
                "load step complete",
            );
        }
        tracing::info!("daemon ready");
        *guard = Inner::Ready;
    }
}

impl Default for LoadProgress {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_at_model_step_with_zero_progress() {
        let p = LoadProgress::new();
        let snap = p.snapshot().unwrap();
        assert_eq!(snap.current, LoadStep::Model);
        assert_eq!(snap.progress, 0.0);
        assert!(snap.completed.is_empty());
    }

    #[test]
    fn set_step_derives_completed_list_from_canonical_order() {
        let p = LoadProgress::new();
        p.set_step(LoadStep::Sections);
        let snap = p.snapshot().unwrap();
        assert_eq!(snap.current, LoadStep::Sections);
        assert_eq!(snap.completed, vec![LoadStep::Model, LoadStep::Substrate]);
    }

    #[test]
    fn mark_ready_clears_the_snapshot() {
        let p = LoadProgress::new();
        p.mark_ready();
        assert!(p.snapshot().is_none());
        assert!(p.is_ready());
    }

    #[test]
    fn set_step_progress_reports_a_real_fraction() {
        let p = LoadProgress::new();
        p.set_step_progress(3, 12);
        assert!((p.snapshot().unwrap().progress - 0.25).abs() < 1e-6);
    }

    #[test]
    fn set_step_progress_with_zero_total_reads_zero() {
        let p = LoadProgress::new();
        p.set_step_progress(5, 0);
        assert_eq!(p.snapshot().unwrap().progress, 0.0);
    }

    #[test]
    fn set_step_resets_progress_to_zero() {
        let p = LoadProgress::new();
        p.set_step_progress(8, 10);
        p.set_step(LoadStep::Sections);
        assert_eq!(p.snapshot().unwrap().progress, 0.0);
    }
}
