//! Suspending the cast, over HTTP: one character, or every character the
//! caller may manage, and the refusal every route that would run a suspended
//! character's model answers with.
//!
//! Suspended is `state: suspended` ([`SUSPENDED`]), whoever set it — these
//! routes, or an edit of the character's state. A suspended character takes no
//! turn of any kind: its scheduler never makes it due, the guardian does not
//! scan it, the round does not count the time against its mission
//! (`RoundState::hold`), and the routes that decode for it directly (`ask`,
//! `reflect`, `dream`, `journal/draft`) refuse it with [`refused`]. Everything
//! delivered to it meanwhile — a message, a thought, a line in an interaction,
//! what its world does around it — waits in its inbox, and the turn after it
//! resumes reads it bounded per source (`Scheduler::suspend`).
//!
//! Its conversation is let go as it is suspended (`Runtime::follow_suspended`):
//! its engine slot and resident K/V are freed for other work, and the turn it
//! takes once resumed rejoins the conversation from the log.
//!
//! A lease may still move it, as it moves any character: a hand-back takes no
//! turn, and the member it lands on holds it suspended from its first moment,
//! since every member follows the record. Pinning it instead would stall a
//! member's drain — a deploy's restart among them — on a character that is
//! doing nothing.
//!
//! The state is the character's record ([`crate::npcs::Npcs::suspend`]), so it
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

/// The `409` a route that would decode for a suspended character answers
/// with, or `None` for a character that is not suspended.
pub fn refused(s: &Authored, npc_id: u64) -> Option<Response> {
    let held = s
        .npcs
        .view()
        .payload(npc_id)
        .is_some_and(|n| n.state == SUSPENDED);
    held.then(|| {
        err(
            StatusCode::CONFLICT,
            "npc_suspended",
            "that character is suspended and takes no turn until it is resumed",
        )
    })
}

/// `{"suspended": bool}` out of a request body, or the `400` for a body
/// without one.
fn wanted(body: &Value) -> Result<bool, Box<Response>> {
    body.get("suspended")
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            Box::new(err(
                StatusCode::BAD_REQUEST,
                "suspended_missing",
                "say which: {\"suspended\": true} or {\"suspended\": false}",
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
pub fn follow(s: &Authored) {
    if let Some(rt) = s.runtime.as_ref() {
        rt.follow_suspended();
    }
}

/// `POST /v1/npc/:nid/suspend` — suspend one character, or resume it.
///
/// Body `{"suspended": true|false}`. Its owner's, or any admin's: the
/// character page's Suspend / Resume. Resuming puts back the state a
/// suspension of the whole cast found it in, or `idle`. Answers with the
/// character as it now stands.
pub async fn post_suspend(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path(nid): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let suspended = match wanted(&body) {
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
    match s
        .npcs
        .suspend(npc_id, manager(&me, admin), suspended, now_ms())
    {
        Ok(mut v) => {
            follow(&s);
            tracing::info!(npc = %nid, suspended, "character suspension set");
            name_personality(&mut v, &*s.personalities.read().await);
            Json(v).into_response()
        }
        Err(e) => npc_err(e),
    }
}

/// `POST /v1/npcs/suspend` — suspend every character the caller may manage
/// (their own, or for an admin the whole cast), or resume every one such a
/// suspension suspended.
///
/// Body `{"suspended": true|false}`. Suspending keeps each character's state
/// to put back; resuming puts back only what was kept, so a character
/// suspended any other way stays suspended. Answers
/// `{"suspended": bool, "changed": n}`.
pub async fn post_suspend_all(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let suspended = match wanted(&body) {
        Ok(f) => f,
        Err(r) => return *r,
    };
    let (me, admin) = match caller(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    match s.npcs.suspend_all(manager(&me, admin), suspended) {
        Ok(changed) => {
            follow(&s);
            tracing::info!(suspended, changed, admin, "cast suspension set");
            Json(json!({ "suspended": suspended, "changed": changed })).into_response()
        }
        Err(e) => npc_err(e),
    }
}
