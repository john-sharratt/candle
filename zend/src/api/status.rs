//! `GET /v1/status` — daemon loading state.
//!
//! The frontend polls this every second. While the daemon is loading,
//! the response carries the structured state machine
//! ([`crate::loading::LoadStep`]) so the loading overlay can render a
//! progress bar plus a "what's done / what's now" checklist. Once
//! every step completes, `state` flips to `"ready"` and the chat view
//! unlocks.

use std::sync::Arc;

use axum::{extract::State, Json};
use serde::Serialize;

use crate::session::ZendSession;

pub async fn status(State(session): State<Arc<ZendSession>>) -> Json<StatusBody> {
    let snap = session.status_snapshot();
    let loading = snap.loading.map(|s| LoadingBody {
        current: s.current.label(),
        progress: s.progress,
        completed: s.completed.iter().map(|step| step.label()).collect(),
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
}
