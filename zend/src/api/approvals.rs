//! The approvals a person's dialogue with Zen asked for, settled from their
//! own signed-in browser (`docs/zen_tool_reach.md` §5).
//!
//! - `GET /v1/approvals/:id` — the approval: the call held, why, and where it
//!   stands.
//! - `POST /v1/approvals/:id/grant` — run it once, exactly as asked, as the
//!   caller; answers what the call answered.
//! - `POST /v1/approvals/:id/refuse` — settle it unrun.
//!
//! An approval is settled only by the person it waits on: the caller's key
//! names the record, so another person's id names nothing. Nothing a turn
//! writes reaches these routes — a grant is a request the person's browser
//! makes.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::Json;
use candle_conversation::persistence::SharedSubstrate;
use serde_json::{json, Value};

use crate::mcp::approvals::{self, Approval};
use crate::mcp::caller::Caller;
use crate::mcp::ops_tools::{run_granted, OpsCall};
use crate::mcp::workers::now_ms;
use crate::session::ZendSession;

/// The longest a granted call may wait on what it started.
const GRANTED_WAIT: Duration = Duration::from_secs(120);

/// What a route answers when it cannot.
fn refused(status: StatusCode, why: impl Into<String>) -> (StatusCode, Json<Value>) {
    (status, Json(json!({"error": why.into()})))
}

/// The signed-in Admin calling.
fn admin(headers: &HeaderMap, session: &ZendSession) -> Result<Caller, (StatusCode, Json<Value>)> {
    match Caller::of(headers, session.roles()) {
        Some(c) if c.is_admin() => Ok(c),
        Some(_) => Err(refused(StatusCode::FORBIDDEN, "approvals are an Admin's")),
        None => Err(refused(
            StatusCode::UNAUTHORIZED,
            "sign in to settle an approval",
        )),
    }
}

/// The substrate the approvals are kept in, or why it is not open yet.
fn records_of(session: &ZendSession) -> Result<SharedSubstrate, (StatusCode, Json<Value>)> {
    session.shared_substrate().ok_or_else(|| {
        refused(
            StatusCode::SERVICE_UNAVAILABLE,
            "the model is still loading",
        )
    })
}

/// An approval as the GUI draws it.
fn view(a: &Approval) -> Value {
    json!({
        "id": a.id,
        "tool": a.tool,
        "args": a.args,
        "why": a.why,
        "state": a.state,
        "expired": a.expired(now_ms()),
        "conversation": a.conversation,
    })
}

pub async fn read(
    State(session): State<Arc<ZendSession>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let caller = admin(&headers, &session)?;
    let records = records_of(&session)?;
    approvals::get(&records, &caller.key(), &id)
        .map(|a| Json(view(&a)))
        .ok_or_else(|| {
            refused(
                StatusCode::NOT_FOUND,
                format!("there is no approval `{id}` of yours"),
            )
        })
}

/// Settle `id` for the caller, granted or not.
async fn settle(
    session: Arc<ZendSession>,
    headers: HeaderMap,
    id: String,
    grant: bool,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let caller = admin(&headers, &session)?;
    let records = records_of(&session)?;
    let approval = approvals::settle(&records, &caller.key(), &id, grant, now_ms())
        .map_err(|e| refused(StatusCode::CONFLICT, e))?;
    if !grant {
        return Ok(Json(view(&approval)));
    }
    let call = OpsCall {
        session: Arc::clone(&session),
        lineage: approval.conversation.clone(),
        caller: Some(caller),
        progress: None,
        ceiling: GRANTED_WAIT.as_secs(),
        worker: false,
    };
    let answer = run_granted(&approval, &call).await;
    let mut out = view(&approval);
    out["answer"] = json!(answer.text);
    out["is_error"] = json!(answer.is_error);
    Ok(Json(out))
}

pub async fn grant(
    State(session): State<Arc<ZendSession>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    settle(session, headers, id, true).await
}

pub async fn refuse(
    State(session): State<Arc<ZendSession>>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    settle(session, headers, id, false).await
}
