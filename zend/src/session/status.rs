//! Reports the daemon's state for `GET /v1/status` and the substrate's
//! maintenance state, and words a waiting prefill.
//!
//! A snapshot carries `loading` while any startup step is in flight.
//! `started_at_ms` changes only across restarts, so a frontend can detect one.
//! The sidebar list is gated on the redo log still existing, so a wiped
//! substrate shows no conversations even while its in-RAM copy remains. A
//! waiting prefill is worded as what is ahead of it and how long it has waited,
//! in whole seconds, so the text changes at most once a second.

use super::*;
use crate::calibrate::owed;
use crate::settled::{settled, Settled};

/// The last maintenance operation the redo log ran: `(op_name, unix_millis)`.
/// `None` when nothing has run since the log was opened.
pub type LastMaintenanceOp = Option<(String, u64)>;

/// Redo-log maintenance state for `/v1/status`:
/// `(segment_count, last_op, compaction_in_progress)`.
pub type MaintenanceStatus = (usize, LastMaintenanceOp, bool);

/// Snapshot returned by `GET /v1/status`. `loading` is `None` once the
/// daemon is fully ready (chat unlocked); `Some` while any startup step
/// is still in flight. `started_at_ms` changes only across daemon
/// restarts — the frontend uses it to detect a restart and re-fetch.
pub struct StatusSnapshot {
    pub loading: Option<LoadingSnapshot>,
    pub detail: String,
    pub started_at_ms: u64,
    /// Pending background ingest work, merged across every ingest layer (and
    /// uploads, which share the same per-file pool). `None` when the queue is
    /// empty — the GUI hides its bar on this.
    pub ingest_backlog: Option<crate::ingest_backlog::BacklogSnapshot>,
    /// `(scored, total)` of the dialogue normalization replay while it drains
    /// after a load or a re-arm; `None` once the score backgrounds are warm.
    pub normalizing: Option<(usize, usize)>,
    /// Units the calibration passes found uncalibrated and no pass has
    /// committed since ([`crate::calibrate::owed`]).
    pub calibration_owed: usize,
    /// `(warmed, total)` of the ingest files' normalization warm-up while it
    /// runs; `None` when none is.
    pub warming: Option<(usize, usize)>,
}

impl StatusSnapshot {
    /// Whether the memory is settled, and what it waits on ([`crate::settled`]).
    pub fn settled(&self) -> Settled {
        settled(
            self.ingest_backlog.as_ref(),
            self.calibration_owed,
            self.warming,
            self.normalizing,
        )
    }
}

impl ZendSession {
    /// Snapshot for `GET /v1/status`. `loading=None` once every startup
    /// step is complete (chat unlocked); `Some` while any step is still
    /// in flight. `detail` is the active step's free-form sub-status
    /// (e.g. download progress). `started_at_ms` is fixed for the
    /// lifetime of this `ZendSession` and changes only across daemon
    /// restarts.
    pub fn status_snapshot(&self) -> StatusSnapshot {
        StatusSnapshot {
            loading: self.load_progress.snapshot(),
            detail: self.status_rx.borrow().clone(),
            started_at_ms: self.started_at_ms,
            ingest_backlog: crate::ingest_backlog::snapshot(),
            normalizing: self
                .inference
                .load_full()
                .and_then(|state| state.engine.warm_replay_progress()),
            calibration_owed: owed::global().total(),
            warming: self
                .inference
                .load_full()
                .and_then(|state| state.engine.ingest_warm_progress()),
        }
    }

    /// Sidebar entries for `GET /v1/conversations`. Built from the
    /// in-RAM substrate (the `RecordType::Label` records carry every
    /// conversation's `(conv_id, label)` pair), but **gated on the
    /// on-disk redo log still existing**.
    ///
    /// The gate exists because the dev workflow of wiping the
    /// `substrate/` dir behind a running daemon used to leave ghost
    /// conversations in the sidebar: the daemon's in-RAM substrate
    /// kept the old `known_conversations` snapshot even though disk
    /// was empty. With the gate, deleting the log forces the next
    /// refresh to return an empty list — matching what the user
    /// observes on disk.
    ///
    /// Sorted newest-first by `conv_id` string descending (the frontend
    /// supplies `Date.now()` ids, so lexicographic-descending == newest).
    /// The titler's internal timeline is excluded. Archived
    /// conversations are filtered out unless `include_archived` is
    /// set — that's the "show archived" checkbox at the bottom of
    /// the sidebar.
    /// The segmented redo log's maintenance state — `(segment_count, last_op)`
    /// — for the `/v1/status` compaction indicator. `None` until the engine is
    /// loaded.
    pub fn substrate_maintenance(&self) -> Option<MaintenanceStatus> {
        let state = self.inference.load_full()?;
        Some(state.substrate.maintenance_status())
    }

    /// How this boot loaded its prompt sections — restored from the redo log or
    /// prefilled — counted since the engine was built. `None` until the model is
    /// loaded.
    pub fn section_loads(&self) -> Option<SectionLoads> {
        let state = self.inference.load_full()?;
        let conv = { state.engine.conversation() };
        Some(conv.section_loads())
    }

    /// `self-check`, run now (`docs/replicated_substrate.md` §12.4): every
    /// stored conversation asked whether it is intact, as `--self-check
    /// --dry-run` asks at load — reporting, tombstoning nothing. Blocks until
    /// done; `None` until the model is loaded.
    pub fn self_check_now(
        &self,
        progress: &dyn Fn(u64, u64),
    ) -> Option<anyhow::Result<SelfCheckReport>> {
        let state = self.inference.load_full()?;
        let mut layers: HashSet<LayerId> = state
            .ingest_layers
            .iter()
            .filter(|il| state.ingest_bases.contains_key(&il.name))
            .filter_map(|il| state.refresh_builder.id_for_layer(&il.name))
            .collect();
        if let Some(dialogue) = state.refresh_builder.id_for_layer("dialogue") {
            layers.insert(dialogue);
        }
        let base = state.base_conv.unsealed_asker();
        Some(self_check::run(
            &state.engine,
            &state.refresh_config,
            &layers,
            base,
            SelfCheck::DryRun,
            progress,
        ))
    }
}

/// The status a waiting prefill shows (`TurnEvent::PrefillQueued`): what it
/// waits behind and for how long, in whole seconds, so the text changes once a
/// second at most — "Waiting for the GPU — 2 turns (18K tokens) ahead, 4 s".
pub(super) fn queued_status(ahead_tokens: usize, ahead_turns: usize, waiting_ms: u64) -> String {
    let secs = waiting_ms / 1000;
    if ahead_turns == 0 {
        return format!("Waiting for the GPU — {secs} s");
    }
    let tokens = if ahead_tokens >= 1000 {
        format!("{}K", ahead_tokens / 1000)
    } else {
        ahead_tokens.to_string()
    };
    let turns = if ahead_turns == 1 { "turn" } else { "turns" };
    format!("Waiting for the GPU — {ahead_turns} {turns} ({tokens} tokens) ahead, {secs} s")
}
