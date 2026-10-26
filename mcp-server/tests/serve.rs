//! The routes served for a stand-in app, driven as a client drives them: the
//! handshake and its session, the stateless era and the session a caller in
//! front names, a call answered on an event stream with its progress, and the
//! refusals.

use std::net::IpAddr;
use std::sync::{Arc, Mutex};

use axum::body::{to_bytes, Body};
use axum::http::{header, HeaderMap, Request, StatusCode};
use axum::response::Response;
use axum::routing::post;
use axum::Router;
use mcp_server::http::{self, SESSION_HEADER, VERSION_HEADER};
use mcp_server::protocol::{text_result, MODERN};
use mcp_server::{Call, Identity, LocalStore, Server, Store};
use serde_json::{json, Value};
use tokio::sync::Notify;
use tokio::time::{sleep, timeout, Duration};
use tower::ServiceExt;

/// An app with one tool, `remember {text}`, which holds its text on the
/// session and answers what was held there before; and `slow`, which says two
/// steps of its work.
struct Stand {
    identity: Identity,
    store: Arc<LocalStore>,
    delivered: Mutex<Vec<String>>,
    /// Set by `linger` once its client has gone.
    left: Notify,
}

impl Server for Stand {
    type Caller = String;
    type Answer = String;

    fn identity(&self) -> &Identity {
        &self.identity
    }

    fn caller(&self, headers: &HeaderMap, _peer: Option<IpAddr>) -> Option<String> {
        headers
            .get("x-who")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    }

    fn owner(&self, caller: &String) -> String {
        caller.clone()
    }

    fn store(&self) -> Result<Arc<dyn Store>, String> {
        Ok(self.store.clone())
    }

    async fn tools(_this: &Arc<Self>, _call: &Call<Self>) -> Result<Value, String> {
        Ok(json!([{"name": "remember", "inputSchema": {"type": "object"}}]))
    }

    async fn call(this: &Arc<Self>, call: &Call<Self>, name: &str, args: &Value) -> String {
        match name {
            "linger" => {
                call.abandoned().await;
                this.left.notify_one();
                "lingered".into()
            }
            "remember" => {
                let before = call.session.held("text").unwrap_or("nothing").to_string();
                let text = args["text"].as_str().unwrap_or_default();
                call.store
                    .hold(&call.session_id, "text", text, http::now_ms())
                    .unwrap();
                format!("{} held {before}", call.caller)
            }
            "slow" => {
                call.step("first");
                call.step("second");
                "done".into()
            }
            other => format!("no {other}"),
        }
    }

    fn result(&self, answer: &String) -> Value {
        text_result(answer, false)
    }

    async fn delivered(this: &Arc<Self>, _call: &Call<Self>, answer: &String) {
        this.delivered.lock().unwrap().push(answer.clone());
    }
}

fn stand() -> (Router, Arc<Stand>) {
    let s = Arc::new(Stand {
        identity: Identity {
            name: "stand".into(),
            title: "Stand".into(),
            description: "A stand-in.".into(),
            instructions: "Use it.".into(),
            icon: None,
            version: "0".into(),
        },
        store: Arc::new(LocalStore::in_memory()),
        delivered: Mutex::new(Vec::new()),
        left: Notify::new(),
    });
    let router = Router::new()
        .route(
            "/mcp",
            post(http::post::<Stand>)
                .get(http::get)
                .delete(http::delete::<Stand>),
        )
        .with_state(s.clone());
    (router, s)
}

fn rpc(who: Option<&str>, headers: &[(&str, &str)], body: Value) -> Request<Body> {
    let mut req = Request::post("/mcp")
        .header(header::HOST, "code.tokera.com")
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(who) = who {
        req = req.header("x-who", who);
    }
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    req.body(Body::from(body.to_string())).unwrap()
}

async fn send(router: &Router, req: Request<Body>) -> Response {
    router.clone().oneshot(req).await.unwrap()
}

async fn body(res: Response) -> String {
    String::from_utf8(to_bytes(res.into_body(), 1 << 20).await.unwrap().to_vec()).unwrap()
}

fn call(id: u64, name: &str, args: Value, meta: Value) -> Value {
    json!({"jsonrpc": "2.0", "id": id, "method": "tools/call",
           "params": {"name": name, "arguments": args, "_meta": meta}})
}

#[tokio::test]
async fn a_handshake_opens_a_session_its_owner_alone_calls_on() {
    let (router, s) = stand();
    let init = send(
        &router,
        rpc(
            Some("ann"),
            &[],
            json!({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                   "params": {"protocolVersion": "2025-11-25", "clientInfo": {"name": "claude-ai"}}}),
        ),
    )
    .await;
    assert_eq!(init.status(), StatusCode::OK);
    let sid = init.headers()[SESSION_HEADER].to_str().unwrap().to_string();
    let answered: Value = serde_json::from_str(&body(init).await).unwrap();
    assert_eq!(answered["result"]["serverInfo"]["name"], "stand");
    assert_eq!(answered["result"]["instructions"], "Use it.");

    let first = send(
        &router,
        rpc(
            Some("ann"),
            &[(SESSION_HEADER, &sid)],
            call(2, "remember", json!({"text": "blue"}), json!({})),
        ),
    )
    .await;
    assert_eq!(
        body(first).await,
        r#"{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"ann held nothing"}],"isError":false}}"#
    );
    let second = send(
        &router,
        rpc(
            Some("ann"),
            &[(SESSION_HEADER, &sid)],
            call(3, "remember", json!({"text": "red"}), json!({})),
        ),
    )
    .await;
    assert!(body(second).await.contains("ann held blue"));

    // Someone else on Ann's session is told it is unknown.
    let other = send(
        &router,
        rpc(
            Some("bob"),
            &[(SESSION_HEADER, &sid)],
            call(4, "remember", json!({"text": "x"}), json!({})),
        ),
    )
    .await;
    assert_eq!(other.status(), StatusCode::NOT_FOUND);

    // Delivered behind the answer, once each.
    for _ in 0..100 {
        if s.delivered.lock().unwrap().len() == 2 {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(
        *s.delivered.lock().unwrap(),
        ["ann held nothing", "ann held blue"]
    );
}

/// **The stateless era keeps one session per person and client — and per
/// session of the server calling on their behalf.**
#[tokio::test]
async fn a_stateless_caller_is_kept_apart_by_the_session_it_calls_on_behalf_of() {
    let (router, _) = stand();
    let meta = |session: &str| {
        json!({"io.modelcontextprotocol/clientInfo": {"name": "claude-code"},
               "com.tokera/session": session})
    };
    let modern = [(VERSION_HEADER, MODERN)];
    for (id, session, text) in [(1, "s1", "blue"), (2, "s2", "red")] {
        let res = send(
            &router,
            rpc(
                Some("ann"),
                &modern,
                call(id, "remember", json!({"text": text}), meta(session)),
            ),
        )
        .await;
        let text = body(res).await;
        assert!(text.contains("ann held nothing"), "{text}");
        assert!(text.contains(r#""resultType":"complete""#), "{text}");
    }
    let again = send(
        &router,
        rpc(
            Some("ann"),
            &modern,
            call(3, "remember", json!({"text": "green"}), meta("s1")),
        ),
    )
    .await;
    assert!(body(again).await.contains("ann held blue"));
}

#[tokio::test]
async fn a_call_on_an_event_stream_says_each_step_then_answers() {
    let (router, _) = stand();
    let res = send(
        &router,
        rpc(
            Some("ann"),
            &[
                (VERSION_HEADER, MODERN),
                (
                    header::ACCEPT.as_str(),
                    "application/json, text/event-stream",
                ),
            ],
            call(9, "slow", json!({}), json!({"progressToken": "p"})),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    let text = body(res).await;
    let data: Vec<Value> = text
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    assert_eq!(data.len(), 3, "{text}");
    assert_eq!(data[0]["params"]["message"], "first");
    assert_eq!(data[0]["params"]["progressToken"], "p");
    assert_eq!(data[1]["params"]["message"], "second");
    assert_eq!(data[1]["params"]["progress"], 2);
    assert_eq!(data[2]["id"], 9);
    assert_eq!(data[2]["result"]["content"][0]["text"], "done");
}

/// **A client gone from its stream is noticed, and nothing is delivered**:
/// the tool learns it was abandoned, and its answer is not taken as read.
#[tokio::test]
async fn a_client_gone_from_its_stream_is_noticed_and_nothing_is_delivered() {
    let (router, s) = stand();
    let res = send(
        &router,
        rpc(
            Some("ann"),
            &[
                (VERSION_HEADER, MODERN),
                (header::ACCEPT.as_str(), "text/event-stream"),
            ],
            call(1, "linger", json!({}), json!({"progressToken": 1})),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::OK);
    drop(res);
    timeout(Duration::from_secs(5), s.left.notified())
        .await
        .expect("the tool was told its client left");
    sleep(Duration::from_millis(50)).await;
    assert!(s.delivered.lock().unwrap().is_empty());
}

#[tokio::test]
async fn nobody_is_sent_to_sign_in_and_a_list_is_its_tools() {
    let (router, _) = stand();
    let res = send(
        &router,
        rpc(
            None,
            &[],
            json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        ),
    )
    .await;
    assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    assert!(res.headers()[header::WWW_AUTHENTICATE]
        .to_str()
        .unwrap()
        .starts_with("Bearer resource_metadata="));
    let listed = send(
        &router,
        rpc(
            Some("ann"),
            &[(VERSION_HEADER, MODERN)],
            json!({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}),
        ),
    )
    .await;
    let v: Value = serde_json::from_str(&body(listed).await).unwrap();
    assert_eq!(v["result"]["tools"][0]["name"], "remember");
    assert_eq!(v["result"]["cacheScope"], "private");
}
