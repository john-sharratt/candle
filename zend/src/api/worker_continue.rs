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
//!
//! **A member that lacks the handing member's records says which.** When it
//! cannot take the worker within [`CONTINUE_HOLD_WAIT`] and lacks any of the
//! handing member's records through the handing's `written`, it answers
//! [`MISSING`] with the ranges it lacks; the handing member may ask again
//! carrying them under the same signature (`crate::hand_records`), and they
//! are applied here as the stream's arrivals are before the worker is taken.

use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::to_bytes;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use host::cluster::up::from_member;
use serde_json::{json, Value};

use crate::hand_records::{self, count, PUSH_BUDGET};
use crate::mcp::threads::conv_id;
use crate::mcp::workers::{continue_handed, now_ms, Handing, CONTINUE_HOLD_WAIT, MISSING};
use crate::session::ZendSession;
use crate::worker_hop::{self, fresh, verify, PROOF_HEADER};

/// The most a request to take a worker may be: the handing, and the records
/// it carries with each one's head.
pub const BODY_LIMIT: usize = PUSH_BUDGET as usize + (1 << 20);

/// How often a member waiting to take a worker looks again.
const LOOK_AGAIN: Duration = Duration::from_millis(200);

/// Why a handing is refused: its status, its error's code and detail.
type Refusal = (StatusCode, &'static str, String);

fn answer((status, error, detail): Refusal) -> Response {
    (status, Json(json!({ "error": error, "detail": detail }))).into_response()
}

/// The answer of a member that lacks `missing` of the handing member's
/// records, and could not take the worker because of `detail`.
pub fn missing_body(detail: &str, missing: &[(u64, u64)]) -> Value {
    json!({ "error": MISSING, "detail": detail, "missing": missing })
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

/// What a signed handing's body holds.
type Signed<'a> = (Handing, Vec<(u64, &'a [u8])>);

/// `body`, read as a handing signed by `proof` at a time fresh at `now_ms`,
/// and the records it carries.
pub fn read_signed<'a>(
    body: &'a [u8],
    proof: &str,
    verified: impl Fn(&[u8], &str) -> bool,
    now_ms: u64,
) -> Result<Signed<'a>, Refusal> {
    if !verified(body, proof) {
        return Err((
            StatusCode::FORBIDDEN,
            "bad_signature",
            "the handing is not signed with the members' key".into(),
        ));
    }
    let (json, records) =
        hand_records::decode(body).map_err(|e| (StatusCode::BAD_REQUEST, "bad_body", e))?;
    let handing: Handing = serde_json::from_slice(json)
        .map_err(|e| (StatusCode::BAD_REQUEST, "bad_body", e.to_string()))?;
    if !fresh(handing.signed_ms, now_ms) {
        return Err((
            StatusCode::FORBIDDEN,
            "stale",
            "the handing was signed too long ago".into(),
        ));
    }
    Ok((handing, records))
}

/// Take `handing`'s worker once this member holds its conversation through
/// `by`'s commit and the worker is `by`'s to hand on, looking again until
/// [`CONTINUE_HOLD_WAIT`] passes; the last refusal when it never is.
async fn take(
    session: &Arc<ZendSession>,
    handing: &Handing,
    by: u64,
    conv_id: &str,
) -> Result<(), Refusal> {
    let until = Instant::now() + CONTINUE_HOLD_WAIT;
    loop {
        let held = session
            .holds_through(conv_id, by, handing.through, Duration::ZERO)
            .await;
        let refused = if held {
            match continue_handed(session, handing, by) {
                Ok(_) => return Ok(()),
                Err(e) => (StatusCode::CONFLICT, "not_taken", e),
            }
        } else {
            (
                StatusCode::CONFLICT,
                "not_held",
                format!(
                    "{conv_id} is not held here through {by:016x}'s commit {} yet",
                    handing.through
                ),
            )
        };
        if Instant::now() >= until {
            return Err(refused);
        }
        tokio::time::sleep(LOOK_AGAIN).await;
    }
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
    let (handing, records) = match read_signed(&bytes, &proof, verify, now_ms()) {
        Ok(h) => h,
        Err(refused) => return answer(refused),
    };
    let conv_id = conv_id(&handing.owner, &handing.thread, &handing.name);
    if !records.is_empty() {
        let carried = records.len();
        let owned: Vec<(u64, Vec<u8>)> =
            records.into_iter().map(|(o, b)| (o, b.to_vec())).collect();
        let taken = match worker_hop::member().and_then(|m| m.replicator().cloned()) {
            Some(replicator) => replicator.apply_carried(by, owned).await,
            None => 0,
        };
        tracing::info!(
            conv_id = %conv_id,
            from = %format!("{by:016x}"),
            carried,
            taken,
            "mcp: a handing carried records of its member's this member lacked"
        );
    }
    match take(&session, &handing, by, &conv_id).await {
        Ok(()) => (
            StatusCode::ACCEPTED,
            Json(json!({ "taken": conv_id, "call_turn": handing.call_turn })),
        )
            .into_response(),
        Err(refused) => {
            let missing = session.lacks_of(by, handing.written).await;
            if missing.is_empty() {
                return answer(refused);
            }
            let detail = format!(
                "{}; this member lacks {} of {by:016x}'s records through {}",
                refused.2,
                count(&missing),
                handing.written
            );
            (StatusCode::CONFLICT, Json(missing_body(&detail, &missing))).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mcp::workers::lacked;
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
            written: 45,
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
            (handing(now), vec![])
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
        let more = br#"{"from":1,"to":2,"owner":"o","thread":"t","name":"n","call_turn":7,"iteration":1,"through":4,"written":4,"signed_ms":1,"extra":1}"#;
        assert_eq!(
            read_signed(more, &sign_with(&KEY, more), verified, 1)
                .unwrap_err()
                .1,
            "bad_body"
        );
    }

    /// **The records a handing carries are under its signature**: read back
    /// as sent, and one changed in flight refuses the whole handing.
    #[test]
    fn a_handings_records_are_signed_with_it() {
        let verified = |b: &[u8], p: &str| verify_with(&KEY, b, p);
        let now = 1_800_000_000_000;
        let json = serde_json::to_vec(&handing(now)).unwrap();
        let body = hand_records::encode(&json, &[(43, vec![1, 2, 3]), (45, vec![9])]);
        let proof = sign_with(&KEY, &body);
        assert_eq!(
            read_signed(&body, &proof, verified, now).unwrap(),
            (handing(now), vec![(43, &[1, 2, 3][..]), (45, &[9][..])])
        );
        let mut changed = body.clone();
        *changed.last_mut().unwrap() = 8;
        assert_eq!(
            read_signed(&changed, &proof, verified, now).unwrap_err().1,
            "bad_signature"
        );
    }

    /// **A member that lacks records says which, and the handing member reads
    /// them back**; any other refusal names none.
    #[test]
    fn a_refusal_names_the_records_its_member_lacks() {
        let missing = vec![(11, 14), (21, 24)];
        let body = serde_json::to_vec(&missing_body("not held", &missing)).unwrap();
        assert_eq!(lacked(409, &body), Some(missing));
        let other = br#"{"error":"not_taken","detail":"no round"}"#;
        assert_eq!(lacked(409, other), None);
        assert_eq!(lacked(403, &body), None);
        assert_eq!(lacked(409, b"not json"), None);
    }
}
