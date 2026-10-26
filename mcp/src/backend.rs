//! One backend MCP server, called through the gateway as the caller
//! (`docs/mcp_app.md` §1, §3).
//!
//! Every request goes to the gateway's entrance at `/v1/mcp`, names the
//! backend's site in `Host`, and presents the caller's own access token, which
//! the gateway honours from this app's address and routes as it would the
//! person's own request. It speaks the 2026-07-28 protocol, which keeps no
//! session to open: each request names the caller's client, and the session
//! here it is made on behalf of, so the backend keeps what it holds per
//! session of ours.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use axum::http::header;
use futures::StreamExt;
use mcp_server::http::{PROGRESS_EVERY, VERSION_HEADER};
use mcp_server::protocol::{text_result, CLIENT_META, MODERN, SESSION_META};
use mcp_server::Session;
use reqwest::StatusCode;
use serde_json::{json, Value};
use tokio::time::timeout;

use crate::caller::Caller;
use crate::sse::Reader;

/// How long listing a backend's tools may take.
const LIST_TIMEOUT: Duration = Duration::from_secs(60);

/// How long a backend may say nothing before its call is given up: a server
/// on `mcp_server` says something on a call's stream at least every
/// [`PROGRESS_EVERY`], so four of those missed is a backend gone, not a slow
/// one.
const IDLE: Duration = Duration::from_secs(4 * PROGRESS_EVERY.as_secs());

/// One backend: what it is called, and the site it is reached at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Backend {
    pub name: String,
    pub host: String,
}

/// The session a request is made on behalf of.
pub struct OnBehalf<'a> {
    pub id: &'a str,
    pub session: &'a Session,
}

/// What one message read off a backend's answer is.
#[derive(Debug, Clone, PartialEq)]
pub enum Read {
    /// A step of the work: a progress notification's message.
    Progress(String),
    /// The answer.
    Result(Value),
    /// The answer, refused.
    Error(String),
    /// Anything else, passed over.
    Other,
}

/// What `message` is.
pub fn read(message: &Value) -> Read {
    if message.get("method").and_then(Value::as_str) == Some("notifications/progress") {
        return match message["params"].get("message").and_then(Value::as_str) {
            Some(m) => Read::Progress(m.to_string()),
            None => Read::Other,
        };
    }
    if let Some(result) = message.get("result") {
        return Read::Result(result.clone());
    }
    match message.get("error") {
        // A JSON-RPC error: its message.
        Some(Value::Object(error)) => Read::Error(
            error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("the call failed")
                .to_string(),
        ),
        // A route's own refusal, from before the MCP server — the gateway or
        // the daemon's sign-in check: `{"error": "<kind>", "message": "…"}`.
        Some(Value::String(kind)) => {
            Read::Error(match message.get("message").and_then(Value::as_str) {
                Some(said) => format!("{kind}: {said}"),
                None => kind.clone(),
            })
        }
        _ => Read::Other,
    }
}

/// The gateway, as the backends are reached through it.
pub struct Link {
    client: reqwest::Client,
    base: String,
    next_id: AtomicU64,
}

impl Link {
    pub fn new(base: &str) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .build()?;
        Ok(Self {
            client,
            base: base.trim_end_matches('/').to_string(),
            next_id: AtomicU64::new(1),
        })
    }

    /// The request carrying `method` to `backend` as `caller`, on behalf of
    /// `on`; asking for progress when `progress`.
    pub fn request(
        &self,
        backend: &Backend,
        caller: &Caller,
        on: &OnBehalf,
        method: &str,
        mut params: Value,
        progress: bool,
    ) -> reqwest::RequestBuilder {
        let mut meta = json!({
            CLIENT_META: {"name": on.session.client.name()},
            SESSION_META: on.id,
        });
        if progress {
            meta["progressToken"] = json!(1);
        }
        params["_meta"] = meta;
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let req = self
            .client
            .post(format!("{}/v1/mcp", self.base))
            .header(header::HOST, &backend.host)
            .header(VERSION_HEADER, MODERN)
            .header(header::ACCEPT, "application/json, text/event-stream")
            .json(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}));
        match caller.authorization() {
            Some((k, v)) => req.header(k, v),
            None => req,
        }
    }

    /// `backend`'s tools, as `caller` may call them.
    pub async fn tools(
        &self,
        backend: &Backend,
        caller: &Caller,
        on: &OnBehalf<'_>,
    ) -> Result<Vec<Value>, String> {
        let req = self
            .request(backend, caller, on, "tools/list", json!({}), false)
            .timeout(LIST_TIMEOUT);
        let result = answer(backend, req, &|_| {}).await?;
        result
            .get("tools")
            .and_then(Value::as_array)
            .cloned()
            .ok_or_else(|| format!("{} listed no tools", backend.name))
    }

    /// Run tool `name` on `backend` as `caller`, telling each step of its work
    /// to `step`; its `tools/call` result. A failure is an error result.
    pub async fn call(
        &self,
        backend: &Backend,
        caller: &Caller,
        on: &OnBehalf<'_>,
        name: &str,
        args: &Value,
        step: &(dyn Fn(String) + Sync),
    ) -> Value {
        let params = json!({"name": name, "arguments": args});
        let req = self.request(backend, caller, on, "tools/call", params, true);
        match answer(backend, req, step).await {
            Ok(result) => result,
            Err(why) => text_result(&why, true),
        }
    }
}

/// What a backend answering a server error says: it is down or restarting —
/// not a refusal of what was asked — so the same call is worth repeating.
fn unavailable(name: &str, status: StatusCode, said: &str) -> String {
    format!(
        "{name} is unavailable ({status}: {said}) — it is restarting or down, not refusing \
         this call; try the same call again in a minute or two"
    )
}

/// What a backend answering `status` with `said` means for the call, in
/// words that say whether to repeat it: a server error, a sign-in it did not
/// take, and a route it did not have are the backend's state, not a refusal of
/// what was asked — each passes while a restarting backend comes back.
fn refusal(name: &str, status: StatusCode, said: &str) -> String {
    let said = if said.is_empty() {
        String::new()
    } else {
        format!(": {said}")
    };
    match status.as_u16() {
        _ if status.is_server_error() => unavailable(name, status, said.trim_start_matches(": ")),
        401 => format!(
            "{name} did not accept your sign-in ({status}{said}) — while it restarts this passes \
             in a minute or two; if it persists, sign in to the connector again"
        ),
        404 => format!(
            "{name} has no MCP route right now ({status}{said}) — it is restarting; try the same \
             call again in a minute or two"
        ),
        _ => format!("{name} refused ({status}){said}"),
    }
}

/// Send `req` and read its answer — one JSON-RPC response, or an event stream
/// of progress ending in one — telling each progress message to `step`.
async fn answer(
    backend: &Backend,
    req: reqwest::RequestBuilder,
    step: &(dyn Fn(String) + Sync),
) -> Result<Value, String> {
    let name = &backend.name;
    let silent = || format!("{name} went silent for {}s", IDLE.as_secs());
    let res = timeout(IDLE, req.send())
        .await
        .map_err(|_| silent())?
        .map_err(|e| format!("{name} could not be reached: {e}"))?;
    let status = res.status();
    let streamed = res
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|t| t.starts_with("text/event-stream"));
    if !streamed {
        let bytes = timeout(IDLE, res.bytes())
            .await
            .map_err(|_| silent())?
            .map_err(|e| format!("{name}'s answer was cut off: {e}"))?;
        return match serde_json::from_slice::<Value>(&bytes).map(|m| read(&m)) {
            Ok(Read::Result(r)) => Ok(r),
            Ok(Read::Error(e)) if status.is_success() => Err(format!("{name}: {e}")),
            Ok(Read::Error(e)) => Err(refusal(name, status, &e)),
            _ => {
                let body: String = String::from_utf8_lossy(&bytes)
                    .trim()
                    .chars()
                    .take(300)
                    .collect();
                Err(refusal(name, status, &body))
            }
        };
    }
    let mut stream = res.bytes_stream();
    let mut reader = Reader::default();
    while let Some(chunk) = timeout(IDLE, stream.next()).await.map_err(|_| silent())? {
        let chunk = chunk.map_err(|e| format!("{name}'s answer was cut off: {e}"))?;
        for event in reader.feed(&chunk) {
            let Ok(message) = serde_json::from_str::<Value>(&event.data) else {
                continue;
            };
            match read(&message) {
                Read::Progress(line) => step(line),
                Read::Result(r) => return Ok(r),
                Read::Error(e) => return Err(format!("{name}: {e}")),
                Read::Other => {}
            }
        }
    }
    Err(format!("{name} ended the call without an answer"))
}

#[cfg(test)]
mod tests {
    use mcp_server::Client;

    use super::*;

    fn caller() -> Caller {
        Caller {
            provider: "google".into(),
            sub: "boss".into(),
            token: "t0k".into(),
        }
    }

    #[test]
    fn a_message_is_progress_an_answer_a_refusal_or_nothing() {
        assert_eq!(
            read(
                &json!({"jsonrpc": "2.0", "method": "notifications/progress",
                         "params": {"progressToken": 1, "progress": 2, "message": "round 3"}})
            ),
            Read::Progress("round 3".into())
        );
        assert_eq!(
            read(&json!({"jsonrpc": "2.0", "id": 4, "result": {"isError": false}})),
            Read::Result(json!({"isError": false}))
        );
        assert_eq!(
            read(&json!({"jsonrpc": "2.0", "id": 4,
                         "error": {"code": -32601, "message": "no such method"}})),
            Read::Error("no such method".into())
        );
        assert_eq!(
            read(&json!({"jsonrpc": "2.0", "method": "notifications/other"})),
            Read::Other
        );
    }

    /// **A route's own refusal keeps its words**: the body a sign-in check
    /// answers before any MCP server is reached is said, not reduced to "the
    /// call failed".
    #[test]
    fn a_routes_refusal_is_said_in_its_own_words() {
        assert_eq!(
            read(&json!({"error": "unauthorized", "message": "sign in to use Bots"})),
            Read::Error("unauthorized: sign in to use Bots".into())
        );
        assert_eq!(
            read(&json!({"error": "upstream_down"})),
            Read::Error("upstream_down".into())
        );
    }

    /// **A sign-in not taken and a missing route are said as what they are**
    /// — the backend's state while it restarts — and only another status is
    /// a refusal.
    #[test]
    fn each_refusal_says_whether_to_repeat_the_call() {
        assert_eq!(
            refusal(
                "bot",
                StatusCode::UNAUTHORIZED,
                "unauthorized: sign in to use Bots"
            ),
            "bot did not accept your sign-in (401 Unauthorized: unauthorized: sign in to use \
             Bots) — while it restarts this passes in a minute or two; if it persists, sign in \
             to the connector again"
        );
        assert_eq!(
            refusal("bot", StatusCode::NOT_FOUND, ""),
            "bot has no MCP route right now (404 Not Found) — it is restarting; try the same \
             call again in a minute or two"
        );
        assert_eq!(
            refusal("bot", StatusCode::FORBIDDEN, "forbidden: Admins only"),
            "bot refused (403 Forbidden): forbidden: Admins only"
        );
        assert_eq!(
            refusal("bot", StatusCode::BAD_GATEWAY, "upstream_unavailable"),
            unavailable("bot", StatusCode::BAD_GATEWAY, "upstream_unavailable")
        );
    }

    /// **A server error says the backend is down, not that it refused**, and
    /// that the same call is worth repeating.
    #[test]
    fn a_server_error_is_said_as_unavailable() {
        assert_eq!(
            unavailable(
                "bot",
                StatusCode::SERVICE_UNAVAILABLE,
                "upstream_unavailable"
            ),
            "bot is unavailable (503 Service Unavailable: upstream_unavailable) — it is \
             restarting or down, not refusing this call; try the same call again in a minute or two"
        );
    }

    /// **A request names its backend's site, presents the caller's token, and
    /// says whom it is on behalf of** — the client and the session here — so
    /// the backend keeps what it holds per session of ours.
    #[test]
    fn a_request_names_its_site_and_states_its_caller_and_session() {
        let link = Link::new("http://192.168.0.5/").unwrap();
        let backend = Backend {
            name: "npcd".into(),
            host: "bot.tokera.com".into(),
        };
        let session = Session::new("o", Client::ClaudeCode, MODERN, 0);
        let on = OnBehalf {
            id: "s1",
            session: &session,
        };
        let req = link
            .request(
                &backend,
                &caller(),
                &on,
                "tools/call",
                json!({"name": "bot_list"}),
                true,
            )
            .build()
            .unwrap();
        assert_eq!(req.url().as_str(), "http://192.168.0.5/v1/mcp");
        assert_eq!(req.headers()[header::HOST], "bot.tokera.com");
        assert_eq!(req.headers()[header::AUTHORIZATION], "Bearer t0k");
        assert_eq!(req.headers()[VERSION_HEADER], MODERN);
        assert!(
            req.headers().get("x-tokera-user").is_none(),
            "the token names the caller, never headers of the app's own"
        );
        let body: Value = serde_json::from_slice(req.body().unwrap().as_bytes().unwrap()).unwrap();
        assert_eq!(body["method"], "tools/call");
        assert_eq!(
            body["params"],
            json!({
                "name": "bot_list",
                "_meta": {
                    "io.modelcontextprotocol/clientInfo": {"name": "claude-code"},
                    "com.tokera/session": "s1",
                    "progressToken": 1
                }
            })
        );
    }
}
