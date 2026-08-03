//! Daemon load-state machine — drives the frontend's loading screen.
//!
//! The daemon's startup walks through a fixed sequence of phases:
//!
//! 1. **Model** — fetch and load the GGUF weights.
//! 2. **Substrate** — replay the redo log into the in-RAM substrate.
//! 3. **Sections** — prefill the projection schema's pinned sections.
//! 4. **Ingesting** — run the schema-declared ingest passes (one per projection
//!    layer that carries an `ingest:` descriptor — folder scans, per-file reads).
//!    Each layer's human label rides the `detail` sub-status; a schema with no
//!    ingest layers transitions through this step instantly.
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
    /// Forced whole-store redo-log compaction (`--compact-substrate`). Skipped
    /// unless the flag is set — reclaim is normally incremental/background.
    Compacting,
    Sections,
    CalibratingSections,
    /// The schema-driven ingest phase: every projection layer that declares an
    /// `ingest:` descriptor is populated here, in schema order. The specific
    /// layer's display label ("Scanning repository", "Reading code", …) is
    /// surfaced through the `detail` sub-status, so this one step covers an
    /// arbitrary number of ingest layers.
    Ingesting,
}

impl LoadStep {
    /// Canonical ordering of all steps — used to derive the "completed"
    /// list from the current step (everything before it is done).
    pub const ALL: &'static [LoadStep] = &[
        LoadStep::Model,
        LoadStep::Substrate,
        LoadStep::Compacting,
        LoadStep::Sections,
        LoadStep::CalibratingSections,
        LoadStep::Ingesting,
    ];

    /// Human-readable label rendered in the loading overlay.
    pub fn label(self) -> &'static str {
        match self {
            LoadStep::Model => "Loading model",
            LoadStep::Substrate => "Loading substrate",
            LoadStep::Compacting => "Compacting substrate",
            LoadStep::Sections => "Prefilling tool sections",
            LoadStep::CalibratingSections => "Calibrating sections",
            LoadStep::Ingesting => "Ingesting workspace",
        }
    }

    /// The noun the step's `(progressed, total)` counter counts — rendered next to
    /// the bar as an absolute "N / M unit" readout. Empty when the counter is a
    /// scaled fraction (e.g. section prefill reports bytes scaled to 10 000) or
    /// otherwise not a meaningful discrete count, in which case only the bar
    /// shows. The `Ingesting` step's unit varies per ingest layer (folders vs
    /// sections vs files) and is set explicitly via [`LoadProgress::set_step_unit`].
    pub fn unit(self) -> &'static str {
        match self {
            LoadStep::Model => "layers",
            LoadStep::Substrate => "turns",
            LoadStep::Compacting => "",
            LoadStep::Sections => "",
            LoadStep::CalibratingSections => "",
            LoadStep::Ingesting => "",
        }
    }
}

/// Snapshot of the loading state at a moment in time.
#[derive(Clone, Debug)]
pub struct LoadingSnapshot {
    pub current: LoadStep,
    pub progress: f32,
    pub completed: Vec<LoadStep>,
    /// Absolute progress within the current step: `progressed` of `total`
    /// `unit`s done (e.g. 137 of 1000 files). `total == 0` means the step has
    /// no measurable count yet. `unit` is empty when the counter is not a
    /// meaningful discrete count (scaled fraction) — the frontend then shows
    /// only the bar, no "N / M" readout. Owned because ingest units are
    /// projection-YAML-defined (per layer), not compile-time constants.
    pub progressed: u64,
    pub total: u64,
    pub unit: String,
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
        /// The noun `progressed`/`total` count (e.g. "files"). Defaults to the
        /// step's [`LoadStep::unit`]; the ingest phase overrides it per layer via
        /// [`LoadProgress::set_step_unit`] with the layer's YAML-defined unit.
        /// Empty ⇒ no absolute readout shown.
        unit: String,
        /// Cumulative tokens prefilled during this step (ingest stat).
        prefill_tokens: u64,
        /// Wall-clock instant the current step was entered — used for
        /// per-step elapsed-time logging at transitions.
        started: Instant,
    },
    Ready,
}

/// Live token counters for an in-flight ingest step, polled to drive the
/// upload modal's per-stage stats (ingested tokens & t/s). The whole-file
/// summary is no longer decoded inline — it's the async summary tree's root,
/// tracked separately by the upload's analysis phase — so there are no
/// summary token counters here.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IngestStats {
    pub prefill_tokens: u64,
}

impl LoadProgress {
    /// Begin at the first step (`Model`) and announce it — the daemon's real
    /// model-load lifecycle. Use [`Self::silent`] for throwaway progress sinks
    /// that report sub-progress but are not that lifecycle.
    pub fn new() -> Self {
        tracing::info!(step = LoadStep::Model.label(), "load step started");
        Self::silent()
    }

    /// Like [`Self::new`] but **does not** emit the "load step started" log.
    /// For throwaway progress handles — the workspace watcher's repo-map /
    /// code-reading refreshes and upload ingests create one per call just to
    /// satisfy the progress parameter, and are not the model-load lifecycle.
    /// Without this, every filesystem-event burst would spuriously log
    /// "load step started Loading model".
    pub fn silent() -> Self {
        Self {
            inner: Mutex::new(Inner::Loading {
                current: LoadStep::Model,
                progressed: 0,
                total: 0,
                unit: LoadStep::Model.unit().to_string(),
                prefill_tokens: 0,
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
                unit,
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
                    progressed: *progressed,
                    total: *total,
                    unit: unit.clone(),
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
            unit: step.unit().to_string(),
            prefill_tokens: 0,
            started: Instant::now(),
        };
    }

    /// Override the unit noun for the current step's absolute readout. Used by
    /// the ingest phase, whose single [`LoadStep::Ingesting`] step covers layers
    /// that count different things (folders, sections, files) — each layer sets
    /// its own YAML-defined unit as it begins. No-op once ready.
    pub fn set_step_unit(&self, unit: &str) {
        if let Inner::Loading { unit: u, .. } = &mut *self.inner.lock().unwrap() {
            *u = unit.to_string();
        }
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
            progressed,
            total: t,
            ..
        } = &mut *guard
        {
            *progressed = current.min(total);
            *t = total;
        }
    }

    /// Read the raw `(current, total)` step counters — the inverse of
    /// [`Self::set_step_progress`]. `total == 0` means indeterminate.
    /// Cheap; used to poll an in-flight step's progress (e.g. streaming the
    /// code-read ingest bar during an upload).
    pub fn step_progress(&self) -> (u64, u64) {
        let guard = self.inner.lock().unwrap();
        if let Inner::Loading {
            progressed, total, ..
        } = &*guard
        {
            (*progressed, *total)
        } else {
            (0, 0)
        }
    }

    /// Add `n` to the running prefilled-token count for the current step.
    /// Cheap; called once per carved scope as the ingest reads a file.
    pub fn add_prefill_tokens(&self, n: u64) {
        if let Inner::Loading { prefill_tokens, .. } = &mut *self.inner.lock().unwrap() {
            *prefill_tokens += n;
        }
    }

    /// Snapshot the current step's ingest token counters — polled by the upload
    /// SSE loop to stream the per-stage "ingested tokens & t/s" stat to the GUI.
    pub fn ingest_stats(&self) -> IngestStats {
        if let Inner::Loading { prefill_tokens, .. } = &*self.inner.lock().unwrap() {
            IngestStats {
                prefill_tokens: *prefill_tokens,
            }
        } else {
            IngestStats::default()
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

    /// `silent()` has the same initial state as `new()` (it only differs by not
    /// emitting the "load step started" log — the property that keeps a
    /// watcher-refresh burst from spuriously logging "Loading model").
    #[test]
    fn silent_starts_at_model_step_like_new() {
        let p = LoadProgress::silent();
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
        assert_eq!(
            snap.completed,
            vec![LoadStep::Model, LoadStep::Substrate, LoadStep::Compacting]
        );
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

    #[test]
    fn snapshot_exposes_absolute_counts_and_step_default_unit() {
        let p = LoadProgress::new(); // Model step: unit "layers"
        p.set_step_progress(3, 12);
        let snap = p.snapshot().unwrap();
        assert_eq!(snap.progressed, 3);
        assert_eq!(snap.total, 12);
        assert_eq!(snap.unit, "layers");
    }

    #[test]
    fn set_step_unit_overrides_and_step_change_resets_to_default() {
        let p = LoadProgress::new();
        // The ingest step has no fixed unit (its layers count different things).
        p.set_step(LoadStep::Ingesting);
        assert_eq!(p.snapshot().unwrap().unit, "");
        p.set_step_unit("files");
        p.set_step_progress(10, 1000);
        let snap = p.snapshot().unwrap();
        assert_eq!(snap.unit, "files");
        assert_eq!(snap.progressed, 10);
        assert_eq!(snap.total, 1000);
        // Advancing to another step resets the unit to that step's default.
        p.set_step(LoadStep::Substrate);
        assert_eq!(p.snapshot().unwrap().unit, "turns");
    }
}
