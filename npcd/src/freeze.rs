//! Freezing the cast, over HTTP: one character, or every character the caller
//! may manage, and the refusal every route that would run a frozen character's
//! model answers with.
//!
//! Frozen is `state: suspended` ([`SUSPENDED`]), whoever set it — these routes,
//! or an owner's Suspend on the character's own page. A suspended character
//! takes no turn of any kind: its scheduler never makes it due, the guardian
//! does not scan it, the round does not judge its mission stale, its lease
//! does not move it, and the routes that decode for it directly (`ask`,
//! `reflect`, `dream`, `journal/draft`) refuse it with [`frozen`]. Everything
//! delivered to it meanwhile — a message, a thought, a line in an interaction,
//! what its world does around it — waits in its inbox, and the turn after the
//! thaw reads it bounded per source (`Scheduler::freeze`).
//!
//! The state is the character's record ([`crate::npcs::Npcs::freeze`]), so it
//! survives a restart and replicates; the member that writes it brings its own
//! scheduler along at once, and every other member as the record reaches it
//! (`Runtime::follow_cast`).

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};
use web::auth::Role;

use crate::api::{err, name_personality, now_ms, npc_err, owner_of, Authored};
use crate::npcs::{npc_id_of_wire, Manager, SUSPENDED};

/// The `409` a route that would decode for a frozen character answers with,
/// or `None` for a character that is not frozen.
pub fn frozen(s: &Authored, npc_id: u64) -> Option<Response> {
    let held = s
        .npcs
        .view()
        .payload(npc_id)
        .is_some_and(|n| n.state == SUSPENDED);
    held.then(|| {
        err(
            StatusCode::CONFLICT,
            "npc_frozen",
            "that character is frozen and takes no turn until it is unfrozen",
        )
    })
}

/// `{"frozen": bool}` out of a request body, or the `400` for a body without
/// one.
fn wanted(body: &Value) -> Result<bool, Box<Response>> {
    body.get("frozen").and_then(Value::as_bool).ok_or_else(|| {
        Box::new(err(
            StatusCode::BAD_REQUEST,
            "frozen_missing",
            "say which: {\"frozen\": true} or {\"frozen\": false}",
        ))
    })
}

/// The caller, and whether they manage every character or only their own.
async fn caller(s: &Arc<Authored>, headers: &HeaderMap) -> Result<(String, bool), Box<Response>> {
    let (id, me) = owner_of(s, headers).await?;
    let admin = s.roles.of(Some(&id)).at_least(Role::Admin);
    Ok((me, admin))
}

/// The manager a caller is: an admin manages every character, anybody else
/// their own.
fn manager(me: &str, admin: bool) -> Manager<'_> {
    if admin {
        Manager::Admin(me)
    } else {
        Manager::Owner(me)
    }
}

/// Bring this member's scheduler to the cast just written.
fn follow(s: &Authored) {
    if let Some(rt) = s.runtime.as_ref() {
        rt.follow_frozen();
    }
}

/// `POST /v1/npc/:nid/freeze` — suspend one character, or resume it.
///
/// Body `{"frozen": true|false}`. Its owner's, or any admin's — the same act as
/// the owner's Suspend / Resume: resuming puts back the state a freeze of the
/// whole cast found it in, or `idle`. Answers with the character as it now
/// stands, its `state` and the `frozen` derived from it among its fields.
pub async fn post_freeze(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let frozen = match wanted(&body) {
        Ok(f) => f,
        Err(r) => return *r,
    };
    let (me, admin) = match caller(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let Some(npc_id) = npc_id_of_wire(&nid) else {
        return err(StatusCode::NOT_FOUND, "npc_not_found", "no such character");
    };
    match s.npcs.freeze(npc_id, manager(&me, admin), frozen, now_ms()) {
        Ok(mut v) => {
            follow(&s);
            tracing::info!(npc = %nid, frozen, "character freeze set");
            name_personality(&mut v, &*s.personalities.read().await);
            Json(v).into_response()
        }
        Err(e) => npc_err(e),
    }
}

/// `POST /v1/npcs/freeze` — suspend every character the caller may manage
/// (their own, or for an admin the whole cast), or resume every one such a
/// freeze suspended.
///
/// Body `{"frozen": true|false}`. Freezing keeps each character's state to put
/// back; thawing puts back only what a freeze kept, so a character suspended
/// any other way stays suspended. Answers `{"frozen": bool, "changed": n}`.
pub async fn post_freeze_all(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let frozen = match wanted(&body) {
        Ok(f) => f,
        Err(r) => return *r,
    };
    let (me, admin) = match caller(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    match s.npcs.freeze_all(manager(&me, admin), frozen) {
        Ok(changed) => {
            follow(&s);
            tracing::info!(frozen, changed, admin, "cast freeze set");
            Json(json!({ "frozen": frozen, "changed": changed })).into_response()
        }
        Err(e) => npc_err(e),
    }
}
