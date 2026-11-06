//! `POST /v1/conversations/{id}/restore` — an archived conversation's turns
//! decoded again onto another conversation (`session::restore`), answered when
//! the last of them has sealed.

use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    Json,
};
use serde::Deserialize;

use super::chat::{apply_tools_dial, dial_selection};
use crate::session::restore::{RestoreSpec, Restored};
use crate::session::ZendSession;
use crate::types::ToolMode;

#[derive(Deserialize)]
pub struct RestoreBody {
    /// The conversation it is restored as.
    pub into: String,
    #[serde(default)]
    pub effort: Option<u8>,
    #[serde(default)]
    pub verbosity: Option<u8>,
    #[serde(default)]
    pub think: Option<bool>,
    #[serde(default)]
    pub identity: Option<String>,
}

pub async fn restore(
    State(session): State<Arc<ZendSession>>,
    Path(id): Path<String>,
    Json(body): Json<RestoreBody>,
) -> Result<Json<Restored>, (StatusCode, String)> {
    let tools_mode = ToolMode::Comprehensive;
    let mut selection = dial_selection(body.effort, body.verbosity, body.think);
    apply_tools_dial(&mut selection, tools_mode);
    let spec = RestoreSpec {
        from: id,
        into: body.into,
        selection,
        tools_mode,
        identity: body.identity,
    };
    tokio::task::spawn_blocking(move || session.restore_conversation(&spec))
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("restore task: {e}"),
            )
        })?
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("{e:#}")))
}
