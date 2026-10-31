//! `GET /v1/npc/:nid/discord/trace` — what a character's bot has done on
//! Discord, for an admin (`docs/npcd_discord.md` §5.5). The `Admin` bar is
//! the route's; any living character answers, whoever owns it.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;

use super::report::report;
use crate::api::{err, Authored};
use crate::npcs;

/// `GET /v1/npc/:nid/discord/trace`
pub async fn get_trace(State(s): State<Arc<Authored>>, Path(nid): Path<String>) -> Response {
    let Some(npc) = npcs::npc_id_of_wire(&nid).filter(|npc| s.npcs.view().is_living(*npc)) else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    Json(report(&s, npc).await).into_response()
}
