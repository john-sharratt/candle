//! A call a server in front asks not be logged leaves no line
//! (`docs/private_mode.md` §4.4). Its own test binary, since it reads what
//! the one subscriber of the process wrote: a call any test beside it made
//! would be in the same log.

use std::io;
use std::net::IpAddr;
use std::sync::{Arc, Mutex};

use axum::body::{to_bytes, Body};
use axum::http::{header, HeaderMap, Request};
use axum::routing::post;
use axum::Router;
use mcp_server::http::{self, VERSION_HEADER};
use mcp_server::protocol::{text_result, MODERN};
use mcp_server::{Call, Identity, LocalStore, NoCaller, Server, Store};
use serde_json::{json, Value};
use tower::ServiceExt;
use tracing::subscriber::set_global_default;
use tracing_subscriber::fmt::MakeWriter;

/// An app with one tool, `echo`, answering what it was given.
struct Echo {
    identity: Identity,
    store: Arc<LocalStore>,
}

impl Server for Echo {
    type Caller = String;
    type Answer = String;

    fn identity(&self) -> &Identity {
        &self.identity
    }

    fn caller(&self, headers: &HeaderMap, _peer: Option<IpAddr>) -> Result<String, NoCaller> {
        headers
            .get("x-who")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
            .ok_or(NoCaller::SignedOut)
    }

    fn owner(&self, caller: &String) -> String {
        caller.clone()
    }

    fn store(&self) -> Result<Arc<dyn Store>, String> {
        Ok(self.store.clone())
    }

    async fn tools(_this: &Arc<Self>, _call: &Call<Self>) -> Result<Value, String> {
        Ok(json!([{"name": "echo", "inputSchema": {"type": "object"}}]))
    }

    async fn call(_this: &Arc<Self>, _call: &Call<Self>, _name: &str, args: &Value) -> String {
        args["text"].as_str().unwrap_or_default().to_string()
    }

    fn tools_version(&self) -> Option<&str> {
        None
    }

    fn list_changed(&self, _call: &Call<Self>) -> bool {
        false
    }

    fn result(&self, answer: &String) -> Value {
        text_result(answer, false)
    }

    async fn delivered(_this: &Arc<Self>, _call: &Call<Self>, _answer: &String) {}
}

/// What the subscriber writes, kept to be read back.
#[derive(Clone, Default)]
struct Kept(Arc<Mutex<Vec<u8>>>);

impl Kept {
    fn text(&self) -> String {
        String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
    }
}

impl io::Write for Kept {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for Kept {
    type Writer = Kept;

    fn make_writer(&'a self) -> Kept {
        self.clone()
    }
}

async fn echo(router: &Router, id: u64, text: &str, meta: Value) -> String {
    let body = json!({"jsonrpc": "2.0", "id": id, "method": "tools/call",
                      "params": {"name": "echo", "arguments": {"text": text}, "_meta": meta}});
    let req = Request::post("/mcp")
        .header(header::HOST, "code.tokera.com")
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-who", "ann")
        .header(VERSION_HEADER, MODERN)
        .body(Body::from(body.to_string()))
        .unwrap();
    let res = router.clone().oneshot(req).await.unwrap();
    String::from_utf8(to_bytes(res.into_body(), 1 << 20).await.unwrap().to_vec()).unwrap()
}

/// **A quiet call is handled and leaves no line; any other is logged.**
#[tokio::test]
async fn a_quiet_call_leaves_no_line_in_the_log() {
    let kept = Kept::default();
    set_global_default(
        tracing_subscriber::fmt()
            .with_writer(kept.clone())
            .with_ansi(false)
            .finish(),
    )
    .unwrap();
    let app = Arc::new(Echo {
        identity: Identity {
            name: "echo".into(),
            title: "Echo".into(),
            description: "A stand-in.".into(),
            instructions: String::new(),
            icon: None,
            version: "0".into(),
            list_changed: false,
        },
        store: Arc::new(LocalStore::in_memory()),
    });
    let router = Router::new()
        .route("/mcp", post(http::post::<Echo>))
        .with_state(app);
    let client = json!({"name": "claude-code"});
    let quiet = json!({"io.modelcontextprotocol/clientInfo": client, "com.tokera/quiet": true});
    assert!(echo(&router, 1, "SYNTHETIC-QUIET", quiet).await.contains("SYNTHETIC-QUIET"));
    assert!(!kept.text().contains("mcp: handled"), "{}", kept.text());
    let plain = json!({"io.modelcontextprotocol/clientInfo": client});
    assert!(echo(&router, 2, "spoken", plain).await.contains("spoken"));
    let logged = kept.text();
    assert_eq!(logged.matches("mcp: handled").count(), 1, "{logged}");
    assert!(logged.contains("echo"), "{logged}");
    assert!(!logged.contains("SYNTHETIC-QUIET"), "{logged}");
}
