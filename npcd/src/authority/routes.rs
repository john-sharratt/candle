//! The authority's routes (`docs/replicated_substrate.md` §9.3, §9.6):
//!
//! | Route | |
//! |---|---|
//! | `POST /v1/world/:wid/op` | a write another member makes to a world this member runs — answered only over a cluster socket, from a member, and only by the world's authority |
//! | `POST /v1/pulse/npcleases/:nid/release` | an admin frees a character's lease, for a holder that will not return |

use std::sync::Arc;

use axum::body::to_bytes;
use axum::extract::{Path, Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use cluster::dispatch::Dispatch;
use host::cluster::up::from_member;
use host::guard::{post, Api};
use serde_json::json;
use web::auth::{Role, Roles};

use super::ops::WorldOp;
use super::role::Role as WorldRole;
use crate::engine::runtime::Runtime;

/// The largest op a member sends: a call line and a decision.
const OP_LIMIT: usize = 1 << 20;

pub fn api(roles: Roles) -> Api<Arc<Runtime>> {
    Api::new(roles)
        .route(
            "/v1/world/:wid/op",
            Role::Unauthenticated,
            Dispatch::owner_path("wid"),
            post(op),
        )
        .route(
            "/v1/pulse/npcleases/:nid/release",
            Role::Admin,
            Dispatch::Any,
            post(release),
        )
}

fn refuse(status: StatusCode, code: &str, detail: String) -> Response {
    (status, Json(json!({ "error": code, "detail": detail }))).into_response()
}

/// `POST /v1/world/:wid/op`.
async fn op(State(rt): State<Arc<Runtime>>, Path(wid): Path<String>, req: Request) -> Response {
    let Some(from) = from_member(&req) else {
        return refuse(
            StatusCode::FORBIDDEN,
            "members_only",
            "a world's writes are taken only from its cluster's members".into(),
        );
    };
    if !matches!(rt.standing.role(&wid), WorldRole::Authority { .. }) {
        return refuse(
            StatusCode::CONFLICT,
            "not_the_authority",
            format!("this member does not run world `{wid}` now"),
        );
    }
    let body = match to_bytes(req.into_body(), OP_LIMIT).await {
        Ok(b) => b,
        Err(e) => return refuse(StatusCode::BAD_REQUEST, "unreadable", e.to_string()),
    };
    let op: WorldOp = match serde_json::from_slice(&body) {
        Ok(op) => op,
        Err(e) => return refuse(StatusCode::BAD_REQUEST, "bad_op", e.to_string()),
    };
    match rt.make(&wid, op, Some(from)).await {
        Ok(reply) => Json(reply).into_response(),
        Err(e) => refuse(
            StatusCode::UNPROCESSABLE_ENTITY,
            "refused",
            format!("{e:#}"),
        ),
    }
}

/// `POST /v1/pulse/npcleases/:nid/release`.
async fn release(State(rt): State<Arc<Runtime>>, Path(nid): Path<u64>) -> Response {
    let Some(at) = rt.bodies.bound(nid) else {
        return refuse(
            StatusCode::NOT_FOUND,
            "no_such_character",
            format!("character {nid} has no body in a hosted world"),
        );
    };
    match rt
        .world_write(&at.world, WorldOp::Release { npc: nid })
        .await
    {
        Ok(reply) => Json(reply).into_response(),
        Err(e) => refuse(StatusCode::BAD_GATEWAY, "unreached", format!("{e:#}")),
    }
}
