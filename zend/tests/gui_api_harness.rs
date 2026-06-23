//! GUI/API integration harness.
//!
//! Boots the **real** zend axum router (`zend::api::router`) on an ephemeral
//! port with a model-less [`ZendSession`] and drives it over real HTTP/WS. The
//! model only loads on `start_loading`, so everything here runs with no GPU and
//! no GGUF on disk — this is the live-daemon contract gate for the GUI
//! (docs/zend_ui_redesign.md §7) and runs in CPU CI.
//!
//! What it proves end-to-end against a running server:
//!   * the daemon serves the GUI shell + the `zend-api.*` seam scripts,
//!   * `/v1/status` and `/v1/conversations` answer with the shapes the live
//!     adapter consumes,
//!   * unknown paths 404 and model-gated writes return 503,
//!   * `/ws/logs` frames each line as structured JSON `{ts,level,target,msg}`
//!     (decision 6) over a real WebSocket.

use std::net::SocketAddr;
use std::sync::Arc;

use zend::api;
use zend::config::DaemonConfig;
use zend::log_broadcast::{BusWriter, LogBus};
use zend::session::ZendSession;

struct Harness {
    addr: SocketAddr,
    log: Arc<LogBus>,
    _tmp: tempfile::TempDir,
}

/// Boot the router on `127.0.0.1:0` and return the bound address. The temp
/// workspace + log bus are kept alive by the returned guard.
async fn boot() -> Harness {
    let tmp = tempfile::tempdir().expect("tempdir");
    let config = DaemonConfig {
        workspace: tmp.path().to_path_buf(),
        ..Default::default()
    };
    let log = LogBus::new();
    let session = Arc::new(ZendSession::new(config, log.clone()));
    let app = api::router(session);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    tokio::spawn(async move {
        let _ = axum::serve(listener, app.into_make_service()).await;
    });

    Harness {
        addr,
        log,
        _tmp: tmp,
    }
}

#[tokio::test]
async fn serves_gui_shell_and_seam_scripts() {
    let hz = boot().await;
    let base = format!("http://{}", hz.addr);
    let client = reqwest::Client::new();

    // GUI shell at root
    let r = client.get(format!("{base}/")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let ct = r
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    assert!(ct.contains("text/html"), "content-type was {ct}");
    let body = r.text().await.unwrap();
    assert!(body.contains("id=\"app\""), "shell missing #app root");
    assert!(
        body.contains("zend-api.js"),
        "shell missing seam script tag"
    );

    // the seam scripts the GUI loads must all be served
    for path in [
        "zend-api.js",
        "zend-api.mock.js",
        "zend-api.live.js",
        "favicon.svg",
    ] {
        let r = client.get(format!("{base}/{path}")).send().await.unwrap();
        assert_eq!(r.status(), 200, "asset {path} not served");
    }
}

#[tokio::test]
async fn model_independent_api_contract() {
    let hz = boot().await;
    let base = format!("http://{}", hz.addr);
    let client = reqwest::Client::new();

    // status answers with a `state` field even before any model load
    let r = client
        .get(format!("{base}/v1/status"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let v: serde_json::Value = r.json().await.unwrap();
    assert!(v.get("state").is_some(), "status missing state: {v}");
    // build id for the hot-reload check (frontend force-reloads when it changes)
    assert!(
        v.get("build")
            .and_then(|b| b.as_str())
            .is_some_and(|s| !s.is_empty()),
        "status missing build id: {v}"
    );

    // conversations: model-less + empty tempdir -> an empty list, well-shaped
    let r = client
        .get(format!("{base}/v1/conversations"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let v: serde_json::Value = r.json().await.unwrap();
    assert!(
        v["conversations"].is_array(),
        "conversations not an array: {v}"
    );

    // unknown path -> 404 (fallback)
    let r = client
        .get(format!("{base}/no-such-asset"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);

    // archive is a model-gated write -> 503 when no engine is loaded
    let r = client
        .post(format!("{base}/v1/conversations/whatever/archive"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);

    // conversation history (and its windowed-substrate panel data, now folded
    // into this endpoint) is model-gated -> 503 without an engine
    let r = client
        .get(format!("{base}/v1/conversations/whatever"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);
}

#[tokio::test]
async fn conversation_files_upload_list_get_delete() {
    let hz = boot().await;
    let base = format!("http://{}", hz.addr);
    let client = reqwest::Client::new();

    // empty to start
    let r = client
        .get(format!("{base}/v1/conversations/c1/files"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let v: serde_json::Value = r.json().await.unwrap();
    assert_eq!(v["files"].as_array().unwrap().len(), 0);

    // upload a file (multipart) -> SSE progress; the stream carries file_done
    let form = reqwest::multipart::Form::new().part(
        "file",
        reqwest::multipart::Part::text("fn main() {}\n").file_name("main.rs"),
    );
    let r = client
        .post(format!("{base}/v1/conversations/c1/files"))
        .multipart(form)
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let sse = r.text().await.unwrap();
    assert!(sse.contains("file_start"), "no file_start in {sse}");
    assert!(sse.contains("\"partIndex\""), "no part events in {sse}");
    assert!(sse.contains("file_done"), "no file_done in {sse}");
    assert!(sse.contains("[DONE]"));

    // it now lists, with id + metadata
    let r = client
        .get(format!("{base}/v1/conversations/c1/files"))
        .send()
        .await
        .unwrap();
    let v: serde_json::Value = r.json().await.unwrap();
    let files = v["files"].as_array().unwrap();
    assert_eq!(files.len(), 1);
    assert_eq!(files[0]["name"], "main.rs");
    assert_eq!(files[0]["ext"], "RS");
    assert_eq!(files[0]["kind"], "code");
    let id = files[0]["id"].as_u64().unwrap();

    // content reconstructs byte-exact
    let r = client
        .get(format!("{base}/v1/conversations/c1/files/{id}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(r.text().await.unwrap(), "fn main() {}\n");

    // delete -> 204, then gone
    let r = client
        .delete(format!("{base}/v1/conversations/c1/files/{id}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 204);
    let r = client
        .get(format!("{base}/v1/conversations/c1/files/{id}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
}

#[tokio::test]
async fn ws_logs_streams_structured_json() {
    use futures_util::StreamExt;

    let hz = boot().await;

    // Push one line into the session's log bus via a scoped fmt subscriber.
    {
        let sub = tracing_subscriber::fmt()
            .with_ansi(false)
            .with_writer(BusWriter(hz.log.clone()))
            .finish();
        tracing::subscriber::with_default(sub, || {
            tracing::info!(target: "zend::harness", "structured frame check");
        });
    }
    assert!(
        !hz.log.recent().is_empty(),
        "log bus did not capture the line"
    );

    let url = format!("ws://{}/ws/logs", hz.addr);
    let (mut ws, _resp) = tokio_tungstenite::connect_async(url.as_str())
        .await
        .expect("ws connect");

    // First replayed frame is our backlog line, framed as structured JSON.
    let msg = ws.next().await.expect("a frame").expect("frame ok");
    let text = msg.into_text().expect("text frame");
    let v: serde_json::Value = serde_json::from_str(&text).expect("frame is JSON");
    for field in ["ts", "level", "target", "msg"] {
        assert!(v.get(field).is_some(), "frame missing {field}: {v}");
    }
    assert_eq!(v["level"], "INFO");
    assert_eq!(v["target"], "zend::harness");
    assert_eq!(v["msg"], "structured frame check");
}
