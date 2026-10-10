//! `POST /v1/substrate/zend/maintenance` — force one background-maintenance op
//! now (seal the active + compact a dead-carrying sealed segment, waiving the
//! age/ratio gates), for exercising the in-flight compaction path while chat is
//! live. Runs under the same phased locking as background maintenance, so it
//! doesn't stall inference. `503` until the model is loaded.
//!
//! zend's load state, maintenance and ingest backlog reach `/v1/status` through
//! its line there (`crate::tenant`), not through a route of its own.

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};
use serde::Serialize;

use crate::session::ZendSession;

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

/// Response body for `POST /v1/substrate/zend/maintenance`.
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
