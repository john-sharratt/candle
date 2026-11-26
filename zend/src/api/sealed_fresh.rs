//! `POST /v1/substrate/zend/sealed-fresh` — one question decoded sealed and
//! fresh over exactly the same tokens, and the first decoded token where the
//! two part (`session::sealed_fresh`).

use std::sync::Arc;

use axum::{extract::State, http::StatusCode, Json};

use crate::session::sealed_fresh::{SealedFresh, SealedFreshReq};
use crate::session::ZendSession;

pub async fn run(
    State(session): State<Arc<ZendSession>>,
    Json(req): Json<SealedFreshReq>,
) -> Result<Json<SealedFresh>, (StatusCode, String)> {
    tokio::task::spawn_blocking(move || session.sealed_fresh(&req))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("{e:#}")))
}
