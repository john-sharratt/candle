//! `POST /v1/substrate/zend/sealed-fresh/anchors` — the anchor sweep: a
//! sealed-fresh question run with each placed section's ends kept fresh over a
//! sealed K/V transplant, across a grid of head and tail sizes
//! (`session::sealed_fresh_sweep`).

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};

use crate::session::sealed_fresh_sweep::{AnchorSweep, AnchorSweepReq};
use crate::session::ZendSession;

pub async fn run(
    State(session): State<Arc<ZendSession>>,
    Json(req): Json<AnchorSweepReq>,
) -> Result<Json<AnchorSweep>, (StatusCode, String)> {
    tokio::task::spawn_blocking(move || session.sealed_fresh_anchor_sweep(&req))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))
}
