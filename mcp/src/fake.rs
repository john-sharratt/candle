//! A fake gateway with one backend behind it, on a local port, for the
//! aggregator's tests: it lists `get_user_context`, a tool that only reads,
//! one that writes and a diary's; issues conversation refs under the key the aggregators
//! open them with, as zend does; keeps refs' uses as zend's record does; and
//! notes every call it is sent, with the ref its `_meta` named.

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::routing::post;
use axum::{Json, Router};
use mcp_server::http::now_ms;
use mcp_server::protocol::{MODERN, QUIET_META, REF_META};
use mcp_server::refs::{named_in, Refs, DIARY_HEADER, OPENING};
use mcp_server::{Call, Client, Identity, SealedStore, Session};
use serde_json::{json, Value};
use tempfile::TempDir;
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tokio::time::{sleep, timeout};

use crate::app::App;
use crate::backend::{Backend, Link};
use crate::caller::Caller;
use crate::context;
use crate::guides;
use crate::last_listed::LastListed;
use crate::listing::Listing;
use crate::liveness::Seen;
use crate::routes::Routes;
use crate::served::Served;

/// The key refs are issued and opened with in these tests.
pub const KEY: [u8; 32] = [9; 32];

/// The backend's tool that only reads, the one that writes, and a diary's.
pub const LOOK: &str = "journal_read";
pub const MAKE: &str = "image_generate";
pub const DIARY: &str = "diary_read";

/// One call the backend was sent: its tool, its arguments, the ref its
/// `_meta` named, whether it asked not to be logged, and the diary key its
/// header named.
#[derive(Debug, Clone, PartialEq)]
pub struct Sent {
    pub tool: String,
    pub args: Value,
    pub conversation_ref: Option<String>,
    pub quiet: bool,
    pub diary: Option<String>,
}

/// The fake backend: what it was sent, the uses it was told, and the gate
/// its listing waits at.
pub struct Fake {
    pub lists: AtomicUsize,
    pub sent: Mutex<Vec<Sent>>,
    /// Each conversation's last use, as zend's record keeps it.
    pub used: Mutex<HashMap<String, u64>>,
    refs: Refs,
    open: AtomicBool,
    gate: Notify,
}

impl Fake {
    /// A backend whose listing answers at once.
    pub fn open() -> Arc<Fake> {
        let f = Self::gated();
        f.release();
        f
    }

    /// A backend whose listing waits until [`Fake::release`].
    pub fn gated() -> Arc<Fake> {
        Arc::new(Fake {
            lists: AtomicUsize::new(0),
            sent: Mutex::new(Vec::new()),
            used: Mutex::new(HashMap::new()),
            refs: Refs::new(KEY),
            open: AtomicBool::new(false),
            gate: Notify::new(),
        })
    }

    pub fn release(&self) {
        self.open.store(true, Ordering::SeqCst);
        self.gate.notify_waiters();
    }

    async fn wait_open(&self) {
        loop {
            let opened = self.gate.notified();
            if self.open.load(Ordering::SeqCst) {
                return;
            }
            opened.await;
        }
    }

    /// The calls of `tool` it was sent.
    pub fn sent(&self, tool: &str) -> Vec<Sent> {
        self.sent
            .lock()
            .unwrap()
            .iter()
            .filter(|s| s.tool == tool)
            .cloned()
            .collect()
    }

    /// What it answers `tool` with `args`, called in the conversation
    /// `named` by `owner`.
    fn answer(&self, owner: &str, tool: &str, args: &Value, named: Option<&str>) -> Value {
        let text = |t: String| json!({"content": [{"type": "text", "text": t}], "isError": false});
        match tool {
            // As zend begins one: the ref the aggregator vouched for in
            // `_meta` answered as it is; one it did not, renewed keeping its
            // flag; none, a new conversation.
            context::GET => {
                let private = args.get("private_mode").and_then(Value::as_bool) == Some(true);
                let r = match args.get("ref").and_then(Value::as_str) {
                    Some(given) if named == Some(given) => given.to_string(),
                    Some(given) => {
                        let was = self.refs.open(owner, given).is_ok_and(|r| r.private);
                        self.refs.issue(owner, was || private, now_ms()).0
                    }
                    None => self.refs.issue(owner, private, now_ms()).0,
                };
                text(format!("{OPENING}{r}`. Pass it.\n\nRemembered: tea."))
            }
            context::REF_SEEN => {
                let Some(r) = named.and_then(|n| self.refs.open(owner, n).ok()) else {
                    return json!({"content": [{"type": "text", "text": "no conversation"}], "isError": true});
                };
                let mut used = self.used.lock().unwrap();
                if let Some(at) = args.get("used_ms").and_then(Value::as_u64) {
                    let held = used.entry(r.conversation()).or_insert(at);
                    *held = (*held).max(at);
                }
                text(json!({"used_ms": used.get(&r.conversation())}).to_string())
            }
            _ => text("done".into()),
        }
    }
}

async fn serve(
    State(fake): State<Arc<Fake>>,
    headers: HeaderMap,
    Json(req): Json<Value>,
) -> Json<Value> {
    let id = req["id"].clone();
    let result = match req["method"].as_str() {
        Some("tools/list") => {
            fake.lists.fetch_add(1, Ordering::SeqCst);
            fake.wait_open().await;
            json!({"tools": [
                {"name": context::GET, "annotations": {"readOnlyHint": true},
                 "inputSchema": {"type": "object", "properties": {"ref": {"type": "string"}}}},
                {"name": LOOK, "annotations": {"readOnlyHint": true},
                 "inputSchema": {"type": "object", "properties": {"what": {"type": "string"}}}},
                {"name": MAKE, "inputSchema": {"type": "object"}},
                {"name": DIARY, "inputSchema": {"type": "object"}},
            ]})
        }
        _ => {
            let params = &req["params"];
            let tool = params["name"].as_str().unwrap_or("").to_string();
            let named = params["_meta"][REF_META].as_str().map(str::to_string);
            let args = params["arguments"].clone();
            fake.sent.lock().unwrap().push(Sent {
                tool: tool.clone(),
                args: args.clone(),
                conversation_ref: named.clone(),
                quiet: params["_meta"][QUIET_META] == true,
                diary: headers
                    .get(DIARY_HEADER)
                    .and_then(|v| v.to_str().ok())
                    .map(str::to_string),
            });
            fake.answer(&boss().key(), &tool, &args, named.as_deref())
        }
    };
    Json(json!({"jsonrpc": "2.0", "id": id, "result": result}))
}

/// The person every test call is made as.
pub fn boss() -> Caller {
    caller("boss")
}

pub fn caller(sub: &str) -> Caller {
    Caller {
        provider: "google".into(),
        sub: sub.into(),
        token: "t0k".into(),
    }
}

/// The fake gateway, serving on a local port: its address.
pub async fn gateway(fake: &Arc<Fake>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = Router::new()
        .route("/v1/mcp", post(serve))
        .with_state(Arc::clone(fake));
    tokio::spawn(async move { axum::serve(listener, router).await });
    format!("http://{addr}")
}

/// An aggregator over the backend at `gateway`, whose routes are known from
/// an earlier run and which has listed nothing yet.
pub fn app(gateway: &str, state: &TempDir) -> Arc<App> {
    let routes = state.path().join("routes.json");
    fs::write(
        &routes,
        json!({LOOK: "zen", MAKE: "zen", DIARY: "zen", "get_user_context": "zen",
               "conversation_ref_seen": "zen"})
        .to_string(),
    )
    .unwrap();
    Arc::new(App {
        identity: Identity {
            name: "mcp".into(),
            title: "mcp".into(),
            description: String::new(),
            instructions: String::new(),
            icon: None,
            version: "0".into(),
            list_changed: true,
        },
        store: Arc::new(SealedStore),
        backends: vec![Backend {
            name: "zen".into(),
            host: "code.tokera.com".into(),
            path: "/v1/mcp".into(),
            memory: None,
            diaries: None,
        }],
        link: Link::new(gateway).unwrap(),
        routes: Routes::kept_in(&routes, vec!["zen".into()]),
        last_listed: LastListed::default(),
        served: Served::default(),
        library: guides::embedded().unwrap(),
        read_only: RwLock::new(BTreeSet::new()),
        refs: Refs::new(KEY),
        seen: Seen::default(),
        counters: Arc::default(),
        listing: Listing::default(),
    })
}

/// A call by `caller` through `client`, on the MCP session `session`.
pub fn call_as(caller: Caller, client: Client, session: &str) -> Call<App> {
    Call {
        caller,
        headers: Default::default(),
        session_id: session.to_string(),
        session: Session::new("o", client, MODERN, 0),
        conversation_ref: None,
        quiet: false,
        store: Arc::new(SealedStore),
        progress: None,
    }
}

/// A call by the boss through Claude Code on `session`.
pub fn call(session: &str) -> Call<App> {
    call_as(boss(), Client::ClaudeCode, session)
}

/// The ref a `get_user_context` answer opens with — on its first line, as
/// every instruction says it is.
pub fn ref_of(answer: &Value) -> String {
    let text = context::text_of(answer).unwrap();
    named_in(text)
        .unwrap_or_else(|| panic!("the answer's first line names no ref: {text}"))
        .to_string()
}

/// Wait until `done` holds, for at most five seconds.
pub async fn until(what: &str, done: impl Fn() -> bool) {
    timeout(Duration::from_secs(5), async {
        while !done() {
            sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("{what}"));
}
