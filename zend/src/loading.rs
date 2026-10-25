//! Daemon load-state machine — drives the frontend's loading screen.
//!
//! The daemon's startup walks through a fixed sequence of phases:
//!
//! 1. **Pack** — find the model pack, downloading or preparing it if absent,
//!    then wait for the process's other tenants to arrive at the shared engine.
//! 2. **Opening** — open the substrate; a store written before identities is
//!    stamped here, once, and that is minutes on a large one.
//! 3. **Model** — load the weights onto the card.
//! 4. **Substrate** — replay the redo log into the in-RAM substrate.
//! 3. **Connecting** — with `--cluster`, join the gateways.
//! 4. **Catch-up** — with a cluster reached, fetch what the group wrote while
//!    this member was away.
//! 5. **Sections** — prefill the projection schema's pinned sections, then
//!    calibrate them.
//! 6. **Self-check** — with `--self-check`, ask every stored conversation
//!    whether it is intact (`crate::self_check`).
//! 7. **Priming** — read the priming chain the base conversation descends
//!    from (`crate::branch_ingest::prime`).
//! 8. **Normalizing** — relearn the tool catalog's score-normalization levels.
//! 9. **Preloading** — decode and place on the GPU what a projection's belief
//!    scan reads, so the first question asked does not pay for it.
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
    /// The model pack found — downloaded or prepared if absent — and the other
    /// tenants of the process arrived at the shared engine.
    Pack,
    /// The substrate opened, and stamped with identities if it was written
    /// before them. Separate from the model so a minutes-long stamping reads
    /// as itself, with its records counted, and not as a model stuck at 0%.
    Opening,
    Model,
    Substrate,
    /// Forced whole-store redo-log compaction (`--compact-substrate`). Skipped
    /// unless the flag is set — reclaim is normally incremental/background.
    Compacting,
    /// Joining the cluster (`--cluster`, `docs/replicated_substrate.md` §7.6):
    /// dialling every gateway until one admits this member, every one has
    /// failed an attempt, or ten seconds have passed. Skipped standalone.
    Connecting,
    /// Catching up on what the group wrote while this member was away
    /// (§7.2, §7.6), and taking a generation for what it wrote cut off.
    /// Skipped standalone and when no cluster was reached.
    CatchUp,
    /// A repository the log names no branch of yet pulled from `origin`
    /// through an agent, by the member holding its lease — this one, or another
    /// it waits for (`docs/substrate_git.md` §5). A fresh substrate pulls every
    /// repository in full: candle alone is twenty packs and over a minute.
    /// Skipped when the log already names every repository.
    PullingRepositories,
    /// Every repository the workspace names brought up to the log: its packs
    /// taken into its store. Skipped when the workspace names none.
    Repositories,
    Sections,
    CalibratingSections,
    /// Every stored conversation asked whether it is intact (`--self-check`).
    /// Skipped unless the flag is set. Before the priming chain and every
    /// ingest, so a conversation it tombstones is read again by them.
    SelfCheck,
    /// The priming chain (`BranchIngest::prime`): the workspace listing, then
    /// each repository's root listing and anchor documents, read in order.
    /// Before `ready`, because the base conversation descends from it — the
    /// first question asked must already have it. Links read on an earlier
    /// boot are not read again, so a restart pays for what changed.
    Priming,
    /// The tool catalog's score-normalization hit levels, relearned from its
    /// corpus. They are runtime-only, so every start pays this — last, once the
    /// corpus is complete, and before `ready`, because a query scored against
    /// cold levels ranks the tools differently rather than merely lower.
    Normalizing,
    /// Everything a projection's belief scan reads — the ingested files' and
    /// folders' signatures and the tool catalog's gallery — decoded and resident
    /// on the GPU (`ConversationEngine::preload_galleries`). Before `ready`,
    /// because otherwise the first question after a start pays for it ahead of
    /// its first token: about a second at a thousand ingested files.
    Preloading,
}

impl LoadStep {
    /// Canonical ordering of all steps — used to derive the "completed"
    /// list from the current step (everything before it is done).
    pub const ALL: &'static [LoadStep] = &[
        LoadStep::Pack,
        LoadStep::Opening,
        LoadStep::Model,
        LoadStep::Substrate,
        LoadStep::Compacting,
        LoadStep::Connecting,
        LoadStep::CatchUp,
        LoadStep::PullingRepositories,
        LoadStep::Repositories,
        LoadStep::Sections,
        LoadStep::CalibratingSections,
        LoadStep::SelfCheck,
        LoadStep::Priming,
        LoadStep::Normalizing,
        LoadStep::Preloading,
    ];

    /// Whether reaching this step proves the build: every step before it tests
    /// the binary — the model loaded, the substrate opened and caught up, the
    /// tool sections prefilled and calibrated, so real forwards have run — and
    /// every step from it on is reading whose length is the data's. Priming
    /// after an ingest key change reads every repository's anchors again: an
    /// hour, and nothing about the build that the forwards before it did not
    /// already show.
    pub fn proves_the_build(self) -> bool {
        match self {
            LoadStep::Pack
            | LoadStep::Opening
            | LoadStep::Model
            | LoadStep::Substrate
            | LoadStep::Compacting
            | LoadStep::Connecting
            | LoadStep::CatchUp
            | LoadStep::PullingRepositories
            | LoadStep::Repositories
            | LoadStep::Sections
            | LoadStep::CalibratingSections => false,
            LoadStep::SelfCheck
            | LoadStep::Priming
            | LoadStep::Normalizing
            | LoadStep::Preloading => true,
        }
    }

    /// The step's id on the wire — what a page keys on.
    pub fn id(self) -> &'static str {
        match self {
            LoadStep::Pack => "pack",
            LoadStep::Opening => "opening",
            LoadStep::Model => "model",
            LoadStep::Substrate => "substrate",
            LoadStep::Compacting => "compacting",
            LoadStep::Connecting => "connecting",
            LoadStep::CatchUp => "catchup",
            LoadStep::PullingRepositories => "pulling_repositories",
            LoadStep::Repositories => "repositories",
            LoadStep::Sections => "sections",
            LoadStep::CalibratingSections => "calibrating_sections",
            LoadStep::SelfCheck => "self_check",
            LoadStep::Priming => "priming",
            LoadStep::Normalizing => "normalizing",
            LoadStep::Preloading => "preloading",
        }
    }

    /// Human-readable label rendered in the loading overlay.
    pub fn label(self) -> &'static str {
        match self {
            LoadStep::Pack => "Finding the model pack",
            LoadStep::Opening => "Opening the substrate",
            LoadStep::Model => "Loading model",
            LoadStep::Substrate => "Loading substrate",
            LoadStep::Compacting => "Compacting substrate",
            LoadStep::Connecting => "Connecting to the cluster",
            LoadStep::CatchUp => "Catching up",
            LoadStep::PullingRepositories => "Pulling repositories",
            LoadStep::Repositories => "Loading repositories",
            LoadStep::Sections => "Prefilling tool sections",
            LoadStep::CalibratingSections => "Calibrating sections",
            LoadStep::SelfCheck => "Checking conversations",
            LoadStep::Priming => "Reading the repositories",
            LoadStep::Normalizing => "Normalizing scores",
            LoadStep::Preloading => "Preloading projections",
        }
    }

    /// The noun the step's `(progressed, total)` counter counts — rendered next to
    /// the bar as an absolute "N / M unit" readout. Empty when the counter is a
    /// scaled fraction (e.g. section prefill reports bytes scaled to 10 000) or
    /// otherwise not a meaningful discrete count, in which case only the bar
    /// shows.
    pub fn unit(self) -> &'static str {
        match self {
            // No absolute readout: what the model step counts depends on the
            // architecture. A MoE checkpoint reports experts repacked into the
            // expert pack (25,088 of them on Flash-Next, and the bulk of a cold
            // load's wall time), a dense one reports transformer blocks mounted.
            // One static noun cannot name both, so only the bar shows.
            LoadStep::Model => "",
            LoadStep::Pack => "",
            // Records rewritten while a store is stamped; nothing counts while
            // it is only read, and the bar then shows nothing.
            LoadStep::Opening => "records",
            LoadStep::Substrate => "turns",
            LoadStep::Compacting => "",
            LoadStep::Connecting => "clusters",
            LoadStep::CatchUp => "records",
            LoadStep::PullingRepositories | LoadStep::Repositories => "packs",
            LoadStep::Sections => "",
            LoadStep::CalibratingSections => "",
            LoadStep::SelfCheck => "conversations",
            LoadStep::Priming => "documents",
            LoadStep::Normalizing => "",
            LoadStep::Preloading => "",
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
    /// `unit`s done (e.g. 137 of 1000 turns). `total == 0` means the step has
    /// no measurable count yet. `unit` is empty when the counter is not a
    /// meaningful discrete count (scaled fraction) — the frontend then shows
    /// only the bar, no "N / M" readout.
    pub progressed: u64,
    pub total: u64,
    pub unit: String,
    /// Milliseconds since the load began — across every step, not the
    /// current one.
    pub elapsed_ms: u64,
    /// Tokens prefilled during the current step.
    pub prefill_tokens: u64,
    /// The other tenants of the process this load is waiting for before the
    /// shared engine can be built — the last line of the overlay while any
    /// remain.
    pub waiting_for: Vec<&'static str>,
}

/// Shared, mutable load-progress state. `None`-snapshot means the daemon
/// is fully ready and the frontend should show the chat view.
pub struct LoadProgress {
    inner: Mutex<Inner>,
    /// When the load began.
    began: Instant,
    /// See [`LoadingSnapshot::waiting_for`]. Apart from the step state: the
    /// wait sits inside the model step and does not reset with a step.
    waiting_for: Mutex<Vec<&'static str>>,
}

enum Inner {
    Loading {
        current: LoadStep,
        /// Real `(current, total)` counter — layers for model load,
        /// section-content chars for prefill, etc. `0/0` means "no
        /// progress reported yet for this step" → bar reads 0%.
        progressed: u64,
        total: u64,
        /// The noun `progressed`/`total` count (e.g. "turns") — the step's
        /// [`LoadStep::unit`]. Empty ⇒ no absolute readout shown.
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
        tracing::info!(step = LoadStep::Pack.label(), "load step started");
        Self::silent()
    }

    /// Like [`Self::new`] but **does not** emit the "load step started" log.
    /// For throwaway progress handles — ingest refreshes and upload ingests
    /// create one per call just to satisfy the progress parameter, and are not
    /// the model-load lifecycle, so they must not log "load step started
    /// Loading model".
    pub fn silent() -> Self {
        Self {
            inner: Mutex::new(Inner::Loading {
                current: LoadStep::Pack,
                progressed: 0,
                total: 0,
                unit: LoadStep::Pack.unit().to_string(),
                prefill_tokens: 0,
                started: Instant::now(),
            }),
            began: Instant::now(),
            waiting_for: Mutex::new(Vec::new()),
        }
    }

    /// The tenants this load is waiting for now; empty once it is not.
    pub fn set_waiting_for(&self, tenants: &[&'static str]) {
        if !tenants.is_empty() {
            tracing::info!(tenants = ?tenants, "waiting for the other tenants to arrive");
        }
        *self.waiting_for.lock().unwrap() = tenants.to_vec();
    }

    /// Take a snapshot. `None` once [`Self::mark_ready`] has been called.
    pub fn snapshot(&self) -> Option<LoadingSnapshot> {
        match &*self.inner.lock().unwrap() {
            Inner::Loading {
                current,
                progressed,
                total,
                unit,
                prefill_tokens,
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
                    elapsed_ms: self.began.elapsed().as_millis() as u64,
                    prefill_tokens: *prefill_tokens,
                    waiting_for: self.waiting_for.lock().unwrap().clone(),
                })
            }
            Inner::Ready => None,
        }
    }

    /// The step in progress, or `None` once ready.
    pub fn current_step(&self) -> Option<LoadStep> {
        match &*self.inner.lock().unwrap() {
            Inner::Loading { current, .. } => Some(*current),
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

    /// Name what the current step's counters count, where it changes within
    /// the step — the substrate open counts MiB while it reads and records
    /// while it rewrites. [`Self::set_step`] resets it to the step's own unit.
    pub fn set_step_unit(&self, unit: &str) {
        let mut guard = self.inner.lock().unwrap();
        if let Inner::Loading { unit: u, .. } = &mut *guard {
            if u != unit {
                *u = unit.to_string();
            }
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

    /// **The build is proven once section calibration is behind it** — the
    /// steps that prove it are exactly those after it, in the load's own order,
    /// so a step added later lands on the side of the line its place says.
    #[test]
    fn the_build_is_proven_from_the_step_after_calibration() {
        let at = LoadStep::ALL
            .iter()
            .position(|s| *s == LoadStep::CalibratingSections)
            .unwrap();
        for (i, step) in LoadStep::ALL.iter().enumerate() {
            assert_eq!(step.proves_the_build(), i > at, "{step:?}");
        }
        assert!(
            LoadStep::Priming.proves_the_build(),
            "priming is reading, not the build"
        );
        assert!(
            !LoadStep::Sections.proves_the_build(),
            "the first forwards are the build"
        );
    }

    #[test]
    fn new_starts_at_the_pack_step_with_zero_progress() {
        let p = LoadProgress::new();
        let snap = p.snapshot().unwrap();
        assert_eq!(snap.current, LoadStep::Pack);
        assert_eq!(snap.progress, 0.0);
        assert!(snap.completed.is_empty());
    }

    /// `silent()` has the same initial state as `new()` (it only differs by not
    /// emitting the "load step started" log — the property that keeps an
    /// ingest refresh from spuriously logging a load step).
    #[test]
    fn silent_starts_at_the_pack_step_like_new() {
        let p = LoadProgress::silent();
        let snap = p.snapshot().unwrap();
        assert_eq!(snap.current, LoadStep::Pack);
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
            vec![
                LoadStep::Pack,
                LoadStep::Opening,
                LoadStep::Model,
                LoadStep::Substrate,
                LoadStep::Compacting,
                LoadStep::Connecting,
                LoadStep::CatchUp,
                LoadStep::PullingRepositories,
                LoadStep::Repositories
            ]
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

    /// The model step counts different things per architecture (experts on a
    /// MoE repack, blocks on a dense mount), so it carries no unit noun — the
    /// absolute counters are still exposed, only the "N / M <unit>" readout is
    /// suppressed.
    #[test]
    fn snapshot_exposes_absolute_counts_and_the_model_step_has_no_unit() {
        let p = LoadProgress::new();
        p.set_step(LoadStep::Model);
        p.set_step_progress(3, 12);
        let snap = p.snapshot().unwrap();
        assert_eq!(snap.progressed, 3);
        assert_eq!(snap.total, 12);
        assert_eq!(snap.unit, "");
    }

    /// A step that does name a discrete unit still reports it — the counterpart
    /// to the model step's empty noun above.
    #[test]
    fn a_step_with_a_discrete_unit_reports_it() {
        let p = LoadProgress::new();
        p.set_step(LoadStep::Substrate);
        p.set_step_progress(40, 100);
        assert_eq!(p.snapshot().unwrap().unit, "turns");
    }

    /// Prefilled tokens count within a step and reset with it.
    #[test]
    fn prefill_tokens_count_within_a_step() {
        let p = LoadProgress::new();
        p.set_step(LoadStep::Priming);
        p.add_prefill_tokens(300);
        p.add_prefill_tokens(200);
        assert_eq!(p.snapshot().unwrap().prefill_tokens, 500);
        p.set_step(LoadStep::Normalizing);
        assert_eq!(p.snapshot().unwrap().prefill_tokens, 0);
    }

    /// Whom the load waits for is reported until it is told nobody, and a
    /// step change does not clear it — the wait is inside the model step.
    #[test]
    fn the_wait_for_another_tenant_is_reported_until_it_ends() {
        let p = LoadProgress::silent();
        assert!(p.snapshot().unwrap().waiting_for.is_empty());
        p.set_waiting_for(&["npcd"]);
        assert_eq!(p.snapshot().unwrap().waiting_for, vec!["npcd"]);
        p.set_step(LoadStep::Substrate);
        assert_eq!(p.snapshot().unwrap().waiting_for, vec!["npcd"]);
        p.set_waiting_for(&[]);
        assert!(p.snapshot().unwrap().waiting_for.is_empty());
    }

    /// **The substrate's stamping is its own step, with its records counted**,
    /// between the pack and the weights — the order they run in. A store
    /// stamped on open reads as that, not as a model sitting at 0%.
    #[test]
    fn opening_the_substrate_is_a_counted_step_before_the_weights() {
        let pos = |s: LoadStep| LoadStep::ALL.iter().position(|x| *x == s).unwrap();
        assert!(pos(LoadStep::Pack) < pos(LoadStep::Opening));
        assert!(pos(LoadStep::Opening) < pos(LoadStep::Model));
        let p = LoadProgress::silent();
        p.set_step(LoadStep::Opening);
        p.set_step_progress(250, 1_000);
        let snap = p.snapshot().unwrap();
        assert_eq!((snap.progressed, snap.total), (250, 1_000));
        assert_eq!(snap.unit, "records");
        assert_eq!(snap.completed, vec![LoadStep::Pack]);
    }

    /// A step's unit can change within it, and a new step brings its own.
    #[test]
    fn a_step_unit_changes_within_the_step_and_resets_with_the_next() {
        let p = LoadProgress::silent();
        p.set_step(LoadStep::Opening);
        p.set_step_unit("MiB");
        p.set_step_progress(40, 900);
        assert_eq!(p.snapshot().unwrap().unit, "MiB");
        p.set_step(LoadStep::Substrate);
        assert_eq!(p.snapshot().unwrap().unit, "turns");
    }

    /// Every step has its own id, in the snake case a page keys on.
    #[test]
    fn every_step_has_a_distinct_id() {
        let ids: Vec<&str> = LoadStep::ALL.iter().map(|s| s.id()).collect();
        assert_eq!(
            ids,
            [
                "pack",
                "opening",
                "model",
                "substrate",
                "compacting",
                "connecting",
                "catchup",
                "pulling_repositories",
                "repositories",
                "sections",
                "calibrating_sections",
                "self_check",
                "priming",
                "normalizing",
                "preloading",
            ]
        );
    }
}
