//! `POST /v1/mcp/worker/continue {owner, thread, name, call_turn, through}`:
//! take a worker a slower member handed here at a round boundary
//! (`crate::worker_hop`, `crate::mcp::workers::continue_handed`).
//!
//! Only a member asks, over a cluster socket, naming this member with
//! `x-tokera-node`. This member answers once it holds the conversation
//! through the handing member's commit `through`, and runs the round's tools
//! itself: `202` when it has taken the worker, an error — and the handing
//! member goes on with it — otherwise.

use std::sync::Arc;

use axum::body::to_bytes;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use host::cluster::up::from_member;
use serde::Deserialize;
use serde_json::json;

use crate::mcp::threads::conv_id;
use crate::mcp::workers::{continue_handed, CONTINUE_HOLD_WAIT};
use crate::session::ZendSession;

/// The most a request to take a worker may be.
const BODY_LIMIT: usize = 16 << 10;

/// What the handing member names.
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Handed {
    pub owner: String,
    pub thread: String,
    pub name: String,
    pub call_turn: u32,
    /// Its newest commit of the worker's conversation.
    pub through: u64,
}

fn answer(status: StatusCode, error: &str, detail: String) -> Response {
    (status, Json(json!({ "error": error, "detail": detail }))).into_response()
}

/// `POST /v1/mcp/worker/continue`.
pub async fn post(State(session): State<Arc<ZendSession>>, req: Request) -> Response {
    let Some(by) = from_member(&req) else {
        return answer(
            StatusCode::FORBIDDEN,
            "members_only",
            "a worker is handed on by the member running it, over a cluster socket".into(),
        );
    };
    let bytes = match to_bytes(req.into_body(), BODY_LIMIT).await {
        Ok(b) => b,
        Err(e) => return answer(StatusCode::BAD_REQUEST, "unreadable", e.to_string()),
    };
    let handed: Handed = match serde_json::from_slice(&bytes) {
        Ok(h) => h,
        Err(e) => return answer(StatusCode::BAD_REQUEST, "bad_body", e.to_string()),
    };
    let conv_id = conv_id(&handed.owner, &handed.thread, &handed.name);
    if !session
        .holds_through(&conv_id, by, handed.through, CONTINUE_HOLD_WAIT)
        .await
    {
        return answer(
            StatusCode::CONFLICT,
            "not_held",
            format!(
                "{conv_id} is not held here through {by:016x}'s commit {} yet",
                handed.through
            ),
        );
    }
    match continue_handed(
        &session,
        &handed.owner,
        &handed.thread,
        &handed.name,
        handed.call_turn,
    ) {
        Ok(_) => (
            StatusCode::ACCEPTED,
            Json(json!({ "taken": conv_id, "call_turn": handed.call_turn })),
        )
            .into_response(),
        Err(e) => answer(StatusCode::CONFLICT, "not_taken", e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A handing names exactly what the member taking it needs**, and
    /// nothing else is accepted.
    #[test]
    fn a_handing_is_read_exactly() {
        let h: Handed = serde_json::from_str(
            r#"{"owner":"o1","thread":"car-notes","name":"clean","call_turn":7,"through":42}"#,
        )
        .unwrap();
        assert_eq!(
            h,
            Handed {
                owner: "o1".into(),
                thread: "car-notes".into(),
                name: "clean".into(),
                call_turn: 7,
                through: 42,
            }
        );
        assert!(serde_json::from_str::<Handed>(
            r#"{"owner":"o1","thread":"t","name":"n","call_turn":7,"through":42,"extra":1}"#
        )
        .is_err());
    }
}
