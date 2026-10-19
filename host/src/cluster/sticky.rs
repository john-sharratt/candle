//! Claiming a sticky key before serving it (`docs/replicated_substrate.md`
//! §5.4). A request this member serves, from a gateway or its own listener, on
//! a route the app's table declares `Sticky`, is checked against the claims
//! every gateway relays: a key another live member holds is answered `409`
//! naming the holder, and any other is claimed here — announced on every
//! socket — before the request runs.
//!
//! When the member's claims follow use (`docs/cooperative_admission.md` §2.2),
//! the request uses its claim until its response has ended, and the claim goes
//! with its last use: the key's next request is placed afresh.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::extract::Request;
use axum::http::header::CONTENT_LENGTH;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use cluster::dispatch::{extract, Dispatch, Extracted, RouteTable, BODY_LIMIT};
use cluster::member::{AppMember, Check, ClaimUse};
use serde_json::json;
use web::cluster::route::HOLDER_HEADER;

use super::held_body::HeldBody;

/// What the check decided.
pub enum Sticky {
    /// Serve it — the request, its body buffered if its key needed it — and,
    /// when the member's claims follow use, the claim's use for as long as the
    /// response lasts.
    Serve(Request, Option<ClaimUse>),
    /// Answer it with this: another member holds its key, or its body could
    /// not be read for the key.
    Answer(Response),
}

/// The `409` for a key `holder` holds.
pub fn held(key: &str, holder: u64) -> Response {
    let holder = format!("{holder:016x}");
    let mut res = (
        StatusCode::CONFLICT,
        Json(json!({
            "error": "held_elsewhere",
            "detail": format!("`{key}` is being served by member {holder}"),
            "holder": holder,
        })),
    )
        .into_response();
    if let Ok(v) = HeaderValue::from_str(&holder) {
        res.headers_mut().insert(HOLDER_HEADER, v);
    }
    res
}

/// Check `req` against `table` and `member`'s claims.
pub async fn check(member: &Arc<AppMember>, table: &RouteTable, req: Request) -> Sticky {
    let method = req.method().as_str().to_string();
    let path = req.uri().path().to_string();
    let Some(m) = table.lookup(&method, &path) else {
        return Sticky::Serve(req, None);
    };
    let Dispatch::Sticky(sources) = &m.route.dispatch else {
        return Sticky::Serve(req, None);
    };
    let declared = req
        .headers()
        .get(CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    let (key, req) = match extract(sources, &m.params, req.headers(), None) {
        Extracted::Key(k) => (Some(k), req),
        Extracted::None => (None, req),
        // Larger than a gateway reads to route it: no gateway routed it by its
        // body, so it is served unclaimed, as it came.
        Extracted::NeedsBody if declared.is_some_and(|n| n > BODY_LIMIT) => (None, req),
        Extracted::NeedsBody => {
            let (parts, body) = req.into_parts();
            let Ok(bytes) = to_bytes(body, BODY_LIMIT).await else {
                return Sticky::Answer(
                    (
                        StatusCode::PAYLOAD_TOO_LARGE,
                        Json(json!({
                            "error": "too_large",
                            "detail": format!(
                                "a request named by its body may be at most {} MiB",
                                BODY_LIMIT >> 20
                            ),
                        })),
                    )
                        .into_response(),
                );
            };
            let key = match extract(sources, &m.params, &parts.headers, Some(&bytes)) {
                Extracted::Key(k) => Some(k),
                _ => None,
            };
            (key, Request::from_parts(parts, Body::from(bytes)))
        }
    };
    match key {
        Some(k) => match member.claim(k.as_bytes()) {
            Check::Held(holder) => Sticky::Answer(held(&k, holder)),
            Check::Mine | Check::Took(_) => {
                let used = member.follows_use().then(|| member.use_claim(k.as_bytes()));
                Sticky::Serve(req, used)
            }
        },
        None => Sticky::Serve(req, None),
    }
}

/// `res`, holding `used` until its body has ended.
pub fn hold(res: Response, used: Option<ClaimUse>) -> Response {
    match used {
        Some(u) => res.map(|body| Body::new(HeldBody::new(body, u))),
        None => res,
    }
}
