//! `POST /v1/npc/:nid/trial`: one turn put to a suspended character and
//! answered as one call, nothing of it kept (`docs/npc_trials.md`).
//!
//! The caller's ownership is checked first, then the body, then that the
//! character is suspended, then that there is an engine to decode on: each
//! refusal says the first thing that stands in the way.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::rejection::JsonRejection;
use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;

use crate::api::{err, Authored};
use crate::engine::no_engine;
use crate::engine::pulse::owned_or_admin;
use crate::engine::trial::{TrialBody, TrialRefused};
use crate::suspend;

/// `POST /v1/npc/:nid/trial` — decode one trial turn on a suspended character.
pub async fn trial(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<String>,
    headers: HeaderMap,
    body: Result<Json<TrialBody>, JsonRejection>,
) -> Response {
    let received = Instant::now();
    // The owner's, or any living character for an admin (`docs/npc_trials.md`
    // §1): an admin measures the cast, whoever owns it.
    let npc_id = match owned_or_admin(&s, &headers, &nid).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let trial = match body
        .map_err(|e| e.body_text())
        .and_then(|Json(b)| b.check())
    {
        Ok(t) => t,
        Err(why) => return err(StatusCode::BAD_REQUEST, "bad_trial", &why),
    };
    if let Some(r) = suspend::not_suspended(&s, npc_id) {
        return r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("a trial turn");
    };
    match rt.trial(npc_id, &trial, received).await {
        Ok(answer) => Json(answer).into_response(),
        Err(TrialRefused::NotLoaded) => no_engine("a trial turn"),
        Err(TrialRefused::BadTools(why)) => err(
            StatusCode::BAD_REQUEST,
            "bad_trial",
            &format!("its tools make no grammar: {why}"),
        ),
        Err(TrialRefused::Failed(e)) => {
            err(StatusCode::CONFLICT, "not_answered", &format!("{e:#}"))
        }
    }
}
