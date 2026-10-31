//! `POST /v1/mcp/worker/continue` with a [`Handing`]: take a worker a slower
//! member handed here at a round boundary (`crate::worker_hop`,
//! `crate::mcp::workers::continue_handed`).
//!
//! Only a member asks, over a cluster socket, naming this member with
//! `x-tokera-node`, and signs what it asks under the members' handing key: the
//! gateway's stamp names the member whose stream carried it, but that stream
//! also carries what clients sent the member's own port, so the stamp alone
//! proves nothing. This member answers once it holds the conversation through
//! the handing member's commit, and runs the round's tools itself: `202` when
//! it has taken the worker, an error — and the handing member goes on with
//! it — otherwise.

use std::sync::Arc;

use axum::body::to_bytes;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use host::cluster::up::from_member;
use serde_json::json;

use crate::mcp::threads::conv_id;
use crate::mcp::workers::{continue_handed, now_ms, Handing, CONTINUE_HOLD_WAIT};
use crate::session::ZendSession;
use crate::worker_hop::{fresh, verify, PROOF_HEADER};

/// The most a request to take a worker may be.
const BODY_LIMIT: usize = 16 << 10;

/// Why a handing is refused: its status, its error's code and detail.
type Refusal = (StatusCode, &'static str, String);

fn answer((status, error, detail): Refusal) -> Response {
    (status, Json(json!({ "error": error, "detail": detail }))).into_response()
}

/// What the request says about who sent it, before its body is read: the
/// member the gateway names, and the signature it carries.
fn sender(req: &Request) -> Result<(u64, String), Refusal> {
    let by = from_member(req).ok_or_else(|| {
        (
            StatusCode::FORBIDDEN,
            "members_only",
            "a worker is handed on by the member running it, over a cluster socket".to_string(),
        )
    })?;
    let proof = req
        .headers()
        .get(PROOF_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
        .ok_or_else(|| {
            (
                StatusCode::FORBIDDEN,
                "unsigned",
                format!("a handing is signed in `{PROOF_HEADER}`"),
            )
        })?;
    Ok((by, proof))
}

/// `body`, read as a handing signed by `proof` at a time fresh at `now_ms`.
pub fn read_signed(
    body: &[u8],
    proof: &str,
    verified: impl Fn(&[u8], &str) -> bool,
    now_ms: u64,
) -> Result<Handing, Refusal> {
    if !verified(body, proof) {
        return Err((
            StatusCode::FORBIDDEN,
            "bad_signature",
            "the handing is not signed with the members' key".into(),
        ));
    }
    let handing: Handing = serde_json::from_slice(body)
        .map_err(|e| (StatusCode::BAD_REQUEST, "bad_body", e.to_string()))?;
    if !fresh(handing.signed_ms, now_ms) {
        return Err((
            StatusCode::FORBIDDEN,
            "stale",
            "the handing was signed too long ago".into(),
        ));
    }
    Ok(handing)
}

/// `POST /v1/mcp/worker/continue`.
pub async fn post(State(session): State<Arc<ZendSession>>, req: Request) -> Response {
    let (by, proof) = match sender(&req) {
        Ok(s) => s,
        Err(refused) => return answer(refused),
    };
    let bytes = match to_bytes(req.into_body(), BODY_LIMIT).await {
        Ok(b) => b,
        Err(e) => return answer((StatusCode::BAD_REQUEST, "unreadable", e.to_string())),
    };
    let handing = match read_signed(&bytes, &proof, verify, now_ms()) {
        Ok(h) => h,
        Err(refused) => return answer(refused),
    };
    let conv_id = conv_id(&handing.owner, &handing.thread, &handing.name);
    if !session
        .holds_through(&conv_id, by, handing.through, CONTINUE_HOLD_WAIT)
        .await
    {
        return answer((
            StatusCode::CONFLICT,
            "not_held",
            format!(
                "{conv_id} is not held here through {by:016x}'s commit {} yet",
                handing.through
            ),
        ));
    }
    match continue_handed(&session, &handing, by) {
        Ok(_) => (
            StatusCode::ACCEPTED,
            Json(json!({ "taken": conv_id, "call_turn": handing.call_turn })),
        )
            .into_response(),
        Err(e) => answer((StatusCode::CONFLICT, "not_taken", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker_hop::{sign_with, verify_with, FRESH_MS};

    const KEY: [u8; 32] = [3; 32];

    fn handing(signed_ms: u64) -> Handing {
        Handing {
            from: 0xa,
            to: 0xb,
            owner: "o1".into(),
            thread: "car-notes".into(),
            name: "clean".into(),
            call_turn: 7,
            iteration: 12,
            through: 42,
            signed_ms,
        }
    }

    /// **A handing is taken only as a member signed it, freshly**: exactly
    /// what it names, refused when unsigned by the key, stale, or carrying
    /// anything more.
    #[test]
    fn a_handing_is_read_only_as_a_member_signed_it() {
        let verified = |b: &[u8], p: &str| verify_with(&KEY, b, p);
        let now = 1_800_000_000_000;
        let body = serde_json::to_vec(&handing(now)).unwrap();
        let proof = sign_with(&KEY, &body);
        assert_eq!(
            read_signed(&body, &proof, verified, now).unwrap(),
            handing(now)
        );
        assert_eq!(
            read_signed(&body, &sign_with(&[4; 32], &body), verified, now)
                .unwrap_err()
                .1,
            "bad_signature"
        );
        assert_eq!(
            read_signed(&body, &proof, verified, now + FRESH_MS + 1)
                .unwrap_err()
                .1,
            "stale"
        );
        let more = br#"{"from":1,"to":2,"owner":"o","thread":"t","name":"n","call_turn":7,"iteration":1,"through":4,"signed_ms":1,"extra":1}"#;
        assert_eq!(
            read_signed(more, &sign_with(&KEY, more), verified, 1)
                .unwrap_err()
                .1,
            "bad_body"
        );
    }
}
