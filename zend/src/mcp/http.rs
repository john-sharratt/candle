//! The routes of the remote MCP endpoint, `/mcp`
//! (`docs/claude_zend_gateway.md` §3, §12).
//!
//! Sign-in is the gateway's: it runs the authorization server, and on `/mcp`
//! it turns the client's bearer token into an identity stated on `X-Tokera-*`,
//! as it does a session cookie (`web::auth::mcp`). The member believes those
//! headers only from the gateways it trusts, so the caller here is whoever
//! they name. A request naming nobody is refused with the `WWW-Authenticate`
//! challenge that starts a client's sign-in at the gateway — which is why the
//! route is registered `Role::Unauthenticated`: the route guard's own `401`
//! carries no challenge.
//!
//! A `tools/call` from a client that accepts an event stream is answered on
//! one: progress every [`PROGRESS_EVERY`] while the tool runs — which keeps
//! Claude Code's idle timer from firing — then the result.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json, Response};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use web::auth::mcp::metadata::RESOURCE_DOC_PATH;

use super::caller::Caller;
use super::jsonrpc::{self, Message};
use super::origin;
use super::protocol::{self, MODERN, UNSUPPORTED_VERSION};
use super::state::{self, Client};
use super::tools::{self, Ctx};
use super::workers::now_ms;
use crate::session::ZendSession;

/// How often a running `tools/call` reports progress on its stream.
pub const PROGRESS_EVERY: Duration = Duration::from_secs(15);

const SESSION_HEADER: &str = "mcp-session-id";
const VERSION_HEADER: &str = "mcp-protocol-version";

fn loading() -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, "30")],
        Json(json!({"error": "loading", "message": "zend is still loading its model"})),
    )
        .into_response()
}

fn json_rpc(status: StatusCode, body: Value) -> Response {
    (status, Json(body)).into_response()
}

/// The `WWW-Authenticate` value a refused request carries, pointing the client
/// at the gateway's protected-resource document on `origin`; `invalid_token`
/// when it sent a token the gateway did not take.
pub fn challenge(origin: &str, sent_a_token: bool) -> String {
    let doc = format!("{origin}{RESOURCE_DOC_PATH}");
    if sent_a_token {
        format!(r#"Bearer error="invalid_token", resource_metadata="{doc}""#)
    } else {
        format!(r#"Bearer resource_metadata="{doc}""#)
    }
}

/// The refusal that starts a client's sign-in.
fn unauthorized(headers: &HeaderMap) -> Response {
    let sent_a_token = headers.contains_key(header::AUTHORIZATION);
    let mut res = (
        StatusCode::UNAUTHORIZED,
        Json(json!({"error": "unauthorized", "message": "sign in to use zend"})),
    )
        .into_response();
    if let Ok(v) = HeaderValue::from_str(&challenge(&origin::of(headers), sent_a_token)) {
        res.headers_mut().insert(header::WWW_AUTHENTICATE, v);
    }
    res
}

fn read_header<'a>(headers: &'a HeaderMap, name: &str) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::trim)
        .filter(|v| !v.is_empty())
}

/// A browser's `Origin`, when it names somewhere other than this origin, is
/// refused (DNS rebinding); the Claude clients send none.
fn foreign_origin(headers: &HeaderMap) -> bool {
    read_header(headers, header::ORIGIN.as_str()).is_some_and(|o| o != origin::of(headers))
}

fn wants_stream(headers: &HeaderMap) -> bool {
    read_header(headers, header::ACCEPT.as_str()).is_some_and(|a| a.contains("text/event-stream"))
}

/// `POST /mcp`.
pub async fn post(
    State(session): State<Arc<ZendSession>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if foreign_origin(&headers) {
        return json_rpc(
            StatusCode::FORBIDDEN,
            jsonrpc::error(
                Value::Null,
                jsonrpc::INVALID_REQUEST,
                "this origin may not call /mcp",
            ),
        );
    }
    let Some(caller) = Caller::of(&headers, session.roles()) else {
        return unauthorized(&headers);
    };
    let Some(records) = session.shared_substrate() else {
        return loading();
    };
    let message = match jsonrpc::parse(&body) {
        Ok(m) => m,
        Err(e) => return json_rpc(StatusCode::BAD_REQUEST, e),
    };
    let (id, method, params) = match message {
        Message::Notification { .. } | Message::Response => {
            return StatusCode::ACCEPTED.into_response();
        }
        Message::Request { id, method, params } => (id, method, params),
    };
    let version = read_header(&headers, VERSION_HEADER);
    if let Some(v) = version {
        if !protocol::supported().contains(&v) {
            return json_rpc(
                StatusCode::BAD_REQUEST,
                json!({"jsonrpc": "2.0", "id": id, "error": {
                    "code": UNSUPPORTED_VERSION,
                    "message": "Unsupported protocol version",
                    "data": {"supported": protocol::supported(), "requested": v}
                }}),
            );
        }
    }
    let modern = version == Some(MODERN);
    let owner = caller.key();
    let now = now_ms();
    let failed = |id: Value, e: anyhow::Error| {
        json_rpc(
            StatusCode::INTERNAL_SERVER_ERROR,
            jsonrpc::error(id, jsonrpc::INTERNAL_ERROR, &e.to_string()),
        )
    };

    // Which session the call is on: a 2026-07-28 client's is keyed by its
    // person and client; a handshake client's is the one its `initialize`
    // opened. Both are records, so any member answers.
    let (mcp_id, mcp_session) = if modern {
        let client = Client::of(protocol::modern_client(&params).unwrap_or(""));
        let key = state::stateless_key(&owner, client);
        match state::ensure(&records, &key, &owner, client, MODERN, now) {
            Ok(s) => (key, s),
            Err(e) => return failed(id, e),
        }
    } else if method == "initialize" {
        let asked = params.get("protocolVersion").and_then(Value::as_str);
        let version = protocol::negotiate(asked);
        let client_name = params
            .get("clientInfo")
            .and_then(|c| c.get("name"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let id_opened = match state::open(&records, &owner, Client::of(client_name), version, now) {
            Ok(opened) => opened,
            Err(e) => return failed(id, e),
        };
        let mut res = json_rpc(
            StatusCode::OK,
            jsonrpc::result(id, protocol::initialize_result(version)),
        );
        if let Ok(v) = HeaderValue::from_str(&id_opened) {
            res.headers_mut().insert(SESSION_HEADER, v);
        }
        return res;
    } else {
        let Some(sid) = read_header(&headers, SESSION_HEADER) else {
            return json_rpc(
                StatusCode::BAD_REQUEST,
                jsonrpc::error(
                    id,
                    jsonrpc::INVALID_REQUEST,
                    "no session: send `initialize` first",
                ),
            );
        };
        match state::get(&records, sid, now) {
            Some(s) if s.owner == owner => (sid.to_string(), s),
            _ => {
                return json_rpc(
                    StatusCode::NOT_FOUND,
                    jsonrpc::error(
                        id,
                        jsonrpc::INVALID_REQUEST,
                        "unknown session: initialize again",
                    ),
                )
            }
        }
    };

    match method.as_str() {
        "server/discover" => json_rpc(
            StatusCode::OK,
            jsonrpc::result(id, protocol::discover_result()),
        ),
        "ping" if !modern => json_rpc(StatusCode::OK, jsonrpc::result(id, json!({}))),
        "tools/list" => json_rpc(
            StatusCode::OK,
            jsonrpc::result(
                id,
                protocol::tools_list_result(tools::definitions(), modern),
            ),
        ),
        "tools/call" => {
            let Some(name) = params
                .get("name")
                .and_then(Value::as_str)
                .map(str::to_string)
            else {
                return json_rpc(
                    StatusCode::OK,
                    jsonrpc::error(id, jsonrpc::INVALID_PARAMS, "`name` names the tool to call"),
                );
            };
            let args = params.get("arguments").cloned().unwrap_or(json!({}));
            let ctx = Ctx {
                session: Arc::clone(&session),
                caller,
                mcp: mcp_session,
                mcp_id,
            };
            let progress = protocol::progress_token(&params);
            if wants_stream(&headers) {
                stream_call(ctx, name, args, id, progress, modern)
            } else {
                let answer = tools::call(&ctx, &name, &args).await;
                json_rpc(
                    StatusCode::OK,
                    jsonrpc::result(id, protocol::call_result(answer.result(), modern)),
                )
            }
        }
        other => json_rpc(
            if modern {
                StatusCode::NOT_FOUND
            } else {
                StatusCode::OK
            },
            jsonrpc::error(
                id,
                jsonrpc::METHOD_NOT_FOUND,
                &format!("`{other}` is not a method /mcp answers"),
            ),
        ),
    }
}

/// Run a `tools/call` on an event stream: progress while it runs, then the
/// result, which ends the stream. The tool runs to its end whether or not the
/// client stays — the work is zend's either way.
fn stream_call(
    ctx: Ctx,
    name: String,
    args: Value,
    id: Value,
    progress: Option<Value>,
    modern: bool,
) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(8);
    tokio::spawn(async move {
        let call = tools::call(&ctx, &name, &args);
        tokio::pin!(call);
        let mut tick = tokio::time::interval(PROGRESS_EVERY);
        tick.tick().await;
        let mut beats: u64 = 0;
        let answer = loop {
            tokio::select! {
                answer = &mut call => break answer,
                _ = tick.tick() => {
                    beats += 1;
                    let event = match &progress {
                        Some(token) => Event::default().data(
                            jsonrpc::notification("notifications/progress", json!({
                                "progressToken": token,
                                "progress": beats,
                                "message": format!("{name} working, {} s", beats * PROGRESS_EVERY.as_secs()),
                            }))
                            .to_string(),
                        ),
                        None => Event::default().comment("working"),
                    };
                    // A client gone is no reason to stop: the result is kept by the tool.
                    let _ = tx.send(Ok(event)).await;
                }
            }
        };
        let done = jsonrpc::result(id, protocol::call_result(answer.result(), modern));
        let _ = tx.send(Ok(Event::default().data(done.to_string()))).await;
    });
    let mut res = Sse::new(ReceiverStream::new(rx)).into_response();
    res.headers_mut()
        .insert("x-accel-buffering", HeaderValue::from_static("no"));
    res.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-cache, no-transform"),
    );
    res
}

/// `GET /mcp`: no server-initiated stream is offered.
pub async fn get() -> Response {
    StatusCode::METHOD_NOT_ALLOWED.into_response()
}

/// `DELETE /mcp`: end a handshake session.
pub async fn delete(State(session): State<Arc<ZendSession>>, headers: HeaderMap) -> Response {
    let Some(caller) = Caller::of(&headers, session.roles()) else {
        return unauthorized(&headers);
    };
    let Some(sid) = read_header(&headers, SESSION_HEADER) else {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    };
    let Some(records) = session.shared_substrate() else {
        return loading();
    };
    match state::get(&records, sid, now_ms()) {
        Some(s) if s.owner == caller.key() => match state::close(&records, sid) {
            Ok(()) => StatusCode::NO_CONTENT.into_response(),
            Err(_) => StatusCode::INTERNAL_SERVER_ERROR.into_response(),
        },
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&'static str, &'static str)]) -> HeaderMap {
        let mut h = HeaderMap::new();
        for (k, v) in pairs {
            h.insert(*k, HeaderValue::from_static(v));
        }
        h
    }

    #[test]
    fn a_browser_from_elsewhere_is_refused_and_a_client_with_no_origin_is_not() {
        let base = [("host", "code.tokera.com"), ("x-forwarded-proto", "https")];
        assert!(!foreign_origin(&headers(&base)));
        let mut same = base.to_vec();
        same.push(("origin", "https://code.tokera.com"));
        assert!(!foreign_origin(&headers(&same)));
        let mut other = base.to_vec();
        other.push(("origin", "https://evil.example"));
        assert!(foreign_origin(&headers(&other)));
    }

    #[test]
    fn the_refusal_points_at_the_gateways_resource_document() {
        let res = unauthorized(&headers(&[
            ("host", "192.168.0.237:18444"),
            ("x-forwarded-host", "code.tokera.com"),
            ("x-forwarded-proto", "https"),
        ]));
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            res.headers()[header::WWW_AUTHENTICATE],
            r#"Bearer resource_metadata="https://code.tokera.com/.well-known/oauth-protected-resource/mcp""#
        );
        let sent = unauthorized(&headers(&[
            ("host", "code.tokera.com"),
            ("authorization", "Bearer x"),
        ]));
        assert!(sent.headers()[header::WWW_AUTHENTICATE]
            .to_str()
            .unwrap()
            .starts_with(r#"Bearer error="invalid_token""#));
    }
}
