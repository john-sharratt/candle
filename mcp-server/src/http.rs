//! The routes of an MCP endpoint, for any [`Server`]: `POST` a message, `GET`
//! (no server-initiated stream is offered), `DELETE` a handshake session.
//!
//! Sign-in is the gateway's: it runs the authorization server, and turns a
//! client's bearer token into an identity stated on `X-Tokera-*`
//! (`web::auth::mcp`). A request naming nobody is refused with the
//! `WWW-Authenticate` challenge that starts a client's sign-in at the gateway.
//!
//! A `tools/call` from a client that accepts an event stream is answered on
//! one: a progress notification for each step the tool says of the work it
//! waits on, and at least every [`PROGRESS_EVERY`], which keeps Claude Code's
//! idle timer from firing; then the result. Notifications are shown, not read
//! into Claude's context.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::{ConnectInfo, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json, Response};
use serde_json::{json, Value};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use web::auth::mcp::metadata::RESOURCE_DOC_PATH;

use crate::client::Client;
use crate::jsonrpc::{self, Message};
use crate::origin;
use crate::protocol::{self, MODERN, UNSUPPORTED_VERSION};
use crate::server::{Call, NoCaller, Server};
use crate::session::stateless_key;

/// How often a running `tools/call` reports progress on its stream.
pub const PROGRESS_EVERY: Duration = Duration::from_secs(15);

/// The header a handshake session is named on.
pub const SESSION_HEADER: &str = "mcp-session-id";
/// The header a request names its protocol version on.
pub const VERSION_HEADER: &str = "mcp-protocol-version";

pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

fn json_rpc(status: StatusCode, body: Value) -> Response {
    (status, Json(body)).into_response()
}

fn unavailable(message: &str) -> Response {
    (
        StatusCode::SERVICE_UNAVAILABLE,
        [(header::RETRY_AFTER, "30")],
        Json(json!({"error": "unavailable", "message": message})),
    )
        .into_response()
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

/// The refusal that starts a client's sign-in to `title`.
fn unauthorized(headers: &HeaderMap, title: &str) -> Response {
    let sent_a_token = headers.contains_key(header::AUTHORIZATION);
    let mut res = (
        StatusCode::UNAUTHORIZED,
        Json(json!({"error": "unauthorized", "message": format!("sign in to use {title}")})),
    )
        .into_response();
    if let Ok(v) = HeaderValue::from_str(&challenge(&origin::of(headers), sent_a_token)) {
        res.headers_mut().insert(header::WWW_AUTHENTICATE, v);
    }
    res
}

/// The refusal of a request that names no caller: the sign-in challenge for
/// nobody, a plain `403` for somebody who may not use `title`.
fn refused(headers: &HeaderMap, title: &str, why: NoCaller) -> Response {
    match why {
        NoCaller::SignedOut => unauthorized(headers, title),
        NoCaller::NotAllowed(message) => (
            StatusCode::FORBIDDEN,
            Json(json!({"error": "forbidden", "message": message})),
        )
            .into_response(),
    }
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

/// `POST`: one JSON-RPC message.
pub async fn post<S: Server>(
    State(server): State<Arc<S>>,
    peer: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    if foreign_origin(&headers) {
        return json_rpc(
            StatusCode::FORBIDDEN,
            jsonrpc::error(
                Value::Null,
                jsonrpc::INVALID_REQUEST,
                "this origin may not call this server",
            ),
        );
    }
    let identity = server.identity();
    let caller = match server.caller(&headers, peer.map(|ConnectInfo(a)| a.ip())) {
        Ok(c) => c,
        Err(why) => return refused(&headers, &identity.title, why),
    };
    let store = match server.store() {
        Ok(s) => s,
        Err(why) => return unavailable(&why),
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
    let owner = server.owner(&caller);
    let now = now_ms();
    let failed = |id: Value, e: anyhow::Error| {
        json_rpc(
            StatusCode::INTERNAL_SERVER_ERROR,
            jsonrpc::error(id, jsonrpc::INTERNAL_ERROR, &e.to_string()),
        )
    };

    // Which session the call is on: a 2026-07-28 client's is keyed by its
    // person and client; a handshake client's is the one its `initialize`
    // opened.
    let (session_id, session) = if modern {
        let client = Client::of(protocol::modern_client(&params).unwrap_or(""));
        let key = stateless_key(&owner, client, protocol::on_behalf(&params));
        match store.ensure(&key, &owner, client, MODERN, now) {
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
        // What the client calls itself and the agent it sends: how a client
        // `Client::of` does not know yet is recognised in the log.
        tracing::info!(
            client = client_name,
            agent = read_header(&headers, header::USER_AGENT.as_str()).unwrap_or(""),
            owner = %owner,
            protocol = version,
            "mcp: initialize"
        );
        let opened = match store.open(&owner, Client::of(client_name), version, now) {
            Ok(opened) => opened,
            Err(e) => return failed(id, e),
        };
        let mut res = json_rpc(
            StatusCode::OK,
            jsonrpc::result(
                id,
                protocol::initialize_result(identity, version, &origin::of(&headers)),
            ),
        );
        if let Ok(v) = HeaderValue::from_str(&opened) {
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
        match store.get(sid, now) {
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

    let mut call = Call::<S> {
        caller,
        headers,
        session_id,
        session,
        store,
        progress: None,
    };
    match method.as_str() {
        "server/discover" => json_rpc(
            StatusCode::OK,
            jsonrpc::result(
                id,
                protocol::discover_result(identity, &origin::of(&call.headers)),
            ),
        ),
        "ping" if !modern => json_rpc(StatusCode::OK, jsonrpc::result(id, json!({}))),
        "tools/list" => match S::tools(&server, &call).await {
            Ok(tools) => json_rpc(
                StatusCode::OK,
                jsonrpc::result(id, protocol::tools_list_result(tools, modern)),
            ),
            // Said as an error, so the client asks again rather than caching a
            // list missing tools.
            Err(why) => json_rpc(
                StatusCode::SERVICE_UNAVAILABLE,
                jsonrpc::error(
                    id,
                    jsonrpc::INTERNAL_ERROR,
                    &format!("the tools could not be listed; try again shortly ({why})"),
                ),
            ),
        },
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
            let progress = protocol::progress_token(&params);
            if wants_stream(&call.headers) {
                let (steps_tx, steps) = mpsc::unbounded_channel();
                call.progress = Some(steps_tx);
                stream_call(server, call, name, args, id, progress, steps, modern)
            } else {
                let answer = S::call(&server, &call, &name, &args).await;
                let result = server.result(&answer);
                // Answered in the response itself, which nothing here can see
                // fail; settled behind it, so the answer does not wait on it.
                tokio::spawn(async move { S::delivered(&server, &call, &answer).await });
                json_rpc(
                    StatusCode::OK,
                    jsonrpc::result(id, protocol::call_result(result, modern)),
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
                &format!("`{other}` is not a method this server answers"),
            ),
        ),
    }
}

/// Run a `tools/call` on an event stream: progress while it runs, then the
/// result, which ends the stream. Each step the tool says on `steps` is a
/// progress notification of its own, and a quiet [`PROGRESS_EVERY`] repeats
/// the last one. The tool runs to its end whether or not the client stays —
/// the work is the server's either way — but once the client has gone, the
/// tool is told ([`Call::abandoned`]) and what it answers is not delivered: a
/// server calling on to another drops that call, so the server behind it
/// does not take an answer nobody read as delivered either.
#[allow(clippy::too_many_arguments)]
fn stream_call<S: Server>(
    server: Arc<S>,
    call: Call<S>,
    name: String,
    args: Value,
    id: Value,
    progress: Option<Value>,
    mut steps: mpsc::UnboundedReceiver<String>,
    modern: bool,
) -> Response {
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(8);
    tokio::spawn(async move {
        let running = S::call(&server, &call, &name, &args);
        tokio::pin!(running);
        let mut tick = tokio::time::interval(PROGRESS_EVERY);
        tick.tick().await;
        let mut sent: u64 = 0;
        let mut last = format!("{name} working");
        let event = |sent: u64, message: &str| match &progress {
            Some(token) => Event::default().data(
                jsonrpc::notification(
                    "notifications/progress",
                    json!({
                        "progressToken": token,
                        "progress": sent,
                        "message": message,
                    }),
                )
                .to_string(),
            ),
            None => Event::default().comment("working"),
        };
        let mut gone = false;
        let answer = loop {
            let message = tokio::select! {
                answer = &mut running => break answer,
                // The client went: the tool is told by its steps closing.
                _ = tx.closed(), if !gone => {
                    gone = true;
                    steps.close();
                    continue;
                }
                Some(step) = steps.recv(), if !gone => {
                    tick.reset();
                    step
                }
                _ = tick.tick(), if !gone => last.clone(),
            };
            sent += 1;
            let _ = tx.send(Ok(event(sent, &message))).await;
            last = message;
        };
        if gone {
            // What delivering the answer would change stays unchanged, for
            // the next call.
            return;
        }
        // Steps said as the tool finished are still told, before its answer.
        while let Ok(step) = steps.try_recv() {
            sent += 1;
            let _ = tx.send(Ok(event(sent, &step))).await;
        }
        let done = jsonrpc::result(id, protocol::call_result(server.result(&answer), modern));
        if tx
            .send(Ok(Event::default().data(done.to_string())))
            .await
            .is_ok()
        {
            S::delivered(&server, &call, &answer).await;
        }
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

/// `GET`: no server-initiated stream is offered.
pub async fn get() -> Response {
    StatusCode::METHOD_NOT_ALLOWED.into_response()
}

/// `DELETE`: end a handshake session.
pub async fn delete<S: Server>(
    State(server): State<Arc<S>>,
    peer: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
) -> Response {
    let caller = match server.caller(&headers, peer.map(|ConnectInfo(a)| a.ip())) {
        Ok(c) => c,
        Err(why) => return refused(&headers, &server.identity().title, why),
    };
    let Some(sid) = read_header(&headers, SESSION_HEADER) else {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    };
    let store = match server.store() {
        Ok(s) => s,
        Err(why) => return unavailable(&why),
    };
    match store.get(sid, now_ms()) {
        Some(s) if s.owner == server.owner(&caller) => match store.close(sid) {
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

    /// **Somebody who may not use the server is refused without a
    /// challenge**: signing in again changes nothing, and a challenge would
    /// send their client round the sign-in to be refused at its end.
    #[test]
    fn a_caller_not_allowed_is_refused_without_a_challenge() {
        let h = headers(&[("host", "code.tokera.com"), ("authorization", "Bearer x")]);
        let res = refused(&h, "Zen", NoCaller::NotAllowed("not on the list".into()));
        assert_eq!(res.status(), StatusCode::FORBIDDEN);
        assert!(res.headers().get(header::WWW_AUTHENTICATE).is_none());
        let res = refused(&h, "Zen", NoCaller::SignedOut);
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        assert!(res.headers().get(header::WWW_AUTHENTICATE).is_some());
    }

    #[test]
    fn the_refusal_points_at_the_gateways_resource_document() {
        let res = unauthorized(
            &headers(&[
                ("host", "192.168.0.237:8095"),
                ("x-forwarded-host", "code.tokera.com"),
                ("x-forwarded-proto", "https"),
            ]),
            "Zen",
        );
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            res.headers()[header::WWW_AUTHENTICATE],
            r#"Bearer resource_metadata="https://code.tokera.com/.well-known/oauth-protected-resource/mcp""#
        );
        let sent = unauthorized(
            &headers(&[("host", "code.tokera.com"), ("authorization", "Bearer x")]),
            "Zen",
        );
        assert!(sent.headers()[header::WWW_AUTHENTICATE]
            .to_str()
            .unwrap()
            .starts_with(r#"Bearer error="invalid_token""#));
    }
}
