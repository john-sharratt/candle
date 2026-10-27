//! `POST /v1/prompt/authorize {purpose, prompt}` (`docs/guest_draw_orchestration.md`
//! §1): this member's check, answered as a signed authorization.
//!
//! Called by a member — another member drawing, over a cluster socket — and by
//! an Admin, whose `"mint": true` signs without the check: the Admin vouches
//! for content cleared another way. Users do not call it: it would be a free
//! oracle for probing the filter.

use std::sync::Arc;

use axum::body::to_bytes;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use host::cluster::up::from_member;
use serde::Deserialize;
use serde_json::{json, Value};
use web::auth::Role;

use super::check::{local, Compliance};
use crate::api::{now_ms, Authored};
use crate::compliance::Denial;
use crate::identity::require;

/// The most a prompt to authorize may be.
const BODY_LIMIT: usize = 64 << 10;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AuthorizeBody {
    purpose: String,
    prompt: String,
    /// An Admin's: sign without the check.
    #[serde(default)]
    mint: bool,
}

/// `POST /v1/prompt/authorize`.
pub async fn post_authorize(State(s): State<Arc<Authored>>, req: Request) -> Response {
    let member = from_member(&req).is_some();
    let admin = require(req.headers(), &s.roles, Role::Admin).is_ok();
    if !member && !admin {
        return answer(
            StatusCode::FORBIDDEN,
            json!({ "error": "members_only",
                    "detail": "prompts are authorized for this cluster's members and its admins" }),
        );
    }
    let body = match to_bytes(req.into_body(), BODY_LIMIT).await {
        Ok(b) => b,
        Err(e) => {
            return answer(
                StatusCode::BAD_REQUEST,
                json!({ "error": "unreadable", "detail": e.to_string() }),
            )
        }
    };
    let body: AuthorizeBody = match serde_json::from_slice(&body) {
        Ok(b) => b,
        Err(e) => {
            return answer(
                StatusCode::BAD_REQUEST,
                json!({ "error": "bad_body", "detail": e.to_string() }),
            )
        }
    };
    if body.mint {
        if !admin {
            return answer(
                StatusCode::FORBIDDEN,
                json!({ "error": "admins_only", "detail": "only an admin mints an authorization" }),
            );
        }
        let a = s.authorizations.sign(&body.purpose, &body.prompt, now_ms());
        return answer(StatusCode::OK, json!({ "authorization": a }));
    }
    match local(
        &Compliance(Arc::clone(&s)),
        &s.authorizations,
        &s.authorized,
        &body.purpose,
        &body.prompt,
    )
    .await
    {
        Ok(a) => answer(StatusCode::OK, json!({ "authorization": a })),
        Err(d @ Denial::Refused) => answer(
            StatusCode::FORBIDDEN,
            json!({ "error": "prompt_declined", "detail": d.to_string(), "retry": false }),
        ),
        Err(d @ Denial::Unavailable(_)) => answer(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({ "error": "check_unavailable", "detail": d.to_string(), "retry": true }),
        ),
    }
}

fn answer(status: StatusCode, body: Value) -> Response {
    (status, Json(body)).into_response()
}
