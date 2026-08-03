//! `GET /v1/status` — daemon loading state.
//!
//! The frontend polls this every second. While the daemon is loading,
//! the response carries the structured state machine
//! ([`crate::loading::LoadStep`]) so the loading overlay can render a
//! progress bar plus a "what's done / what's now" checklist. Once
//! every step completes, `state` flips to `"ready"` and the chat view
//! unlocks.

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};
use serde::Serialize;

use crate::session::ZendSession;

pub async fn status(State(session): State<Arc<ZendSession>>) -> Json<StatusBody> {
    let snap = session.status_snapshot();
    let loading = snap.loading.map(|s| LoadingBody {
        current: s.current.label(),
        progress: s.progress,
        completed: s.completed.iter().map(|step| step.label()).collect(),
        progressed: s.progressed,
        total: s.total,
        unit: s.unit.clone(),
    });
    let maintenance = session
        .substrate_maintenance()
        .map(|(segments, last, running)| MaintenanceBody {
            segments,
            last_op: last.as_ref().map(|(label, _)| label.clone()),
            last_op_unix: last.map(|(_, unix)| unix),
            running,
        });
    Json(StatusBody {
        state: if loading.is_some() {
            "loading"
        } else {
            "ready"
        },
        started_at_ms: snap.started_at_ms,
        detail: snap.detail,
        loading,
        maintenance,
        build: super::build_id(),
    })
}

/// Response body for `GET /v1/status`.
///
/// - `state` is `"ready"` once every startup step is complete (chat
///   unlocked) and `"loading"` while any step is still in flight.
/// - `started_at_ms` is the daemon's boot timestamp. Stays fixed for
///   the lifetime of the daemon process and changes only across
///   restarts — the frontend uses it to detect daemon restarts and
///   re-fetch the conversations sidebar.
/// - `detail` is the free-form sub-status string for the current step
///   (e.g. `"Downloading shard 3/8"`). Empty when nothing to report.
/// - `loading` carries the structured state-machine view: which step
///   is active, how far through, and what's already done. `null` once
///   `state == "ready"`.
#[derive(Serialize)]
pub struct StatusBody {
    pub state: &'static str,
    pub started_at_ms: u64,
    pub detail: String,
    pub loading: Option<LoadingBody>,
    /// Segmented redo-log maintenance state — segment count and the last
    /// drop/compact/combine op. `null` until the engine is loaded.
    pub maintenance: Option<MaintenanceBody>,
    /// Hash of the embedded web build. The frontend captures this on load and
    /// force-reloads when it changes (daemon rebuilt with new UI assets).
    pub build: &'static str,
}

/// Segmented redo-log maintenance view sent inside `StatusBody.maintenance` —
/// drives the GUI's compaction indicator (§11).
#[derive(Serialize)]
pub struct MaintenanceBody {
    /// Total segment files in `.substrate/` (sealed + the one active).
    pub segments: usize,
    /// Human label of the last maintenance op (e.g. `"dropped segment 3"`),
    /// or `null` if none has run this session.
    pub last_op: Option<String>,
    /// Unix-seconds timestamp of the last op, or `null`.
    pub last_op_unix: Option<u64>,
    /// `true` while a maintenance op's I/O is in flight — drives the GUI's live
    /// spinner (vs. the settled ✓ shown for the last completed op).
    pub running: bool,
}

/// `POST /v1/debug/maintenance` — force one background-maintenance op now
/// (seal the active + compact a dead-carrying sealed segment, waiving the
/// age/ratio gates), for exercising the in-flight compaction path while chat is
/// live. Runs under the same phased locking as background maintenance, so it
/// doesn't stall inference. `503` until the model is loaded.
pub async fn force_maintenance(
    State(session): State<Arc<ZendSession>>,
) -> Result<Json<ForceMaintenanceBody>, StatusCode> {
    match session.force_maintenance() {
        Some((ran, segments, last_op)) => Ok(Json(ForceMaintenanceBody {
            ran,
            segments,
            last_op: last_op.map(|(label, _)| label),
        })),
        None => Err(StatusCode::SERVICE_UNAVAILABLE),
    }
}

/// Response body for `POST /v1/debug/maintenance`.
#[derive(Serialize)]
pub struct ForceMaintenanceBody {
    /// Whether a drop/compact/combine op actually ran (`false` = nothing was
    /// eligible even under the forced gates — e.g. the sealed segment is 100% live).
    pub ran: bool,
    /// Total segment files after the pass.
    pub segments: usize,
    /// Label of the op that ran (e.g. `"compacted segment 12"`), or `null`.
    pub last_op: Option<String>,
}

/// Structured loading-state view sent inside `StatusBody.loading`.
#[derive(Serialize)]
pub struct LoadingBody {
    /// Human label of the currently active step (e.g. `"Loading model"`).
    pub current: &'static str,
    /// Progress within the current step in `0.0..=1.0`. May be `0.0`
    /// when the step has no measurable sub-progress (most steps do not).
    pub progress: f32,
    /// Labels of steps that have already completed, in execution order.
    pub completed: Vec<&'static str>,
    /// Absolute progress within the current step: `progressed` of `total`
    /// `unit`s (e.g. 137 of 1000 files). The frontend renders "N / M unit"
    /// beside the bar. `total == 0` ⇒ no count yet; `unit == ""` ⇒ the counter
    /// is a scaled fraction, not a discrete count, so only the bar shows.
    pub progressed: u64,
    pub total: u64,
    pub unit: String,
}
