//! GUI/API integration harness.
//!
//! Boots zend as the host serves it — the host's shared endpoints and zend's
//! own routes in one router, behind the host's `web` server with the UIs picked
//! by host name and read from disk — on an ephemeral port, with a model-less
//! [`ZendSession`], and drives it over real HTTP/WS. The model only loads on
//! `start_loading`, so everything here runs with no GPU and no GGUF on disk —
//! this is the live-daemon contract gate for the GUI (docs/zend_ui_redesign.md
//! §7) and runs in CPU CI.
//!
//! What it proves end-to-end against a running server:
//!   * the server serves zend's GUI shell, its `zend-api.*` seam scripts, and
//!     the shared front-end at `/common/`, from `zend/web/` on disk,
//!   * `/v1/status`, `/v1/me` and `/v1/conversations` answer with the shapes
//!     the live adapter consumes — zend's own fields under `tenants.zend`,
//!   * unknown paths 404 and model-gated writes return 503,
//!   * `/ws/logs` frames each line as structured JSON
//!     `{ts,level,target,msg,tenant,seq}` over a real WebSocket.

use std::io::Write;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use host::access::LocalSignin;
use host::endpoints::{routes, Shared};
use host::logs::LogBus;
use host::model::ModelView;
use host::run::merge_routes;
use host::server::{builder, Listen};
use host::stopping::Stopping;
use host::telemetry::Telemetry;
use host::ui::{build_id, COMMON};
use host::Reports;
use web::auth::Roles;
use web::log_file::RotatingFileLog;
use web::Roots;
use zend::api;
use zend::config::DaemonConfig;
use zend::session::ZendSession;
use zend::tenant::{zend_ui, ZendReports, WEB};

struct Harness {
    addr: SocketAddr,
    /// The log file feeding the host's bus, as the process's subscriber writes
    /// it.
    log: RotatingFileLog,
    _tmp: tempfile::TempDir,
}

/// Boot the assembled server on `127.0.0.1:0` and return the bound address.
/// The temp workspace + log bus are kept alive by the returned guard.
async fn boot() -> Harness {
    let tmp = tempfile::tempdir().expect("tempdir");
    let config =
        DaemonConfig::new(zend::workspace::single_repo(tmp.path(), "project").expect("workspace"));
    let session = Arc::new(ZendSession::new(config));
    let log = LogBus::new(&["zend"]);
    let log_file = RotatingFileLog::new(tmp.path(), "harness.log").expect("log file");
    log.attach(&log_file);
    // `boss` is the admin, so a test can say which caller it means by which
    // subject it sends.
    let roles: Roles = serde_yaml::from_str("admins:\n  - sub: boss\n").unwrap();
    let shared = Arc::new(Shared {
        started_at_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
        build: build_id(&[&WEB, &COMMON], "harness"),
        model: ModelView::of_prescribed(),
        tenants: vec![(
            "zend",
            Arc::new(ZendReports::new(Arc::clone(&session))) as Arc<dyn Reports>,
        )],
        log: Arc::clone(&log),
        telemetry: Telemetry::new(ModelView::of_prescribed()),
        roles: roles.clone(),
        cluster: None,
        stopping: Arc::new(Stopping::new()),
        host: "harness".into(),
    });
    let all = merge_routes(
        routes(shared, Roots::embedded(&[&COMMON])),
        vec![("zend", api::routes(session, roles))],
    )
    .expect("zend's routes do not overlap the host's");
    let listen = Listen {
        bind: "127.0.0.1".parse().unwrap(),
        port: 0,
        tls_port: None,
        gateways: Vec::new(),
    };
    let app = builder(
        &listen,
        &[("zend", zend_ui())],
        all.router,
        LocalSignin::default(),
        None,
    )
    .expect("the site table builds")
    .router();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local_addr");
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await;
    });

    Harness {
        addr,
        log: log_file,
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

    // the seam scripts the GUI loads must all be served, and the shared
    // front-end at `/common/`
    for path in [
        "zend-api.js",
        "zend-api.mock.js",
        "zend-api.live.js",
        "favicon.png",
        "common/base.css",
        "common/lib/dom.js",
        "common/lib/estate.css",
    ] {
        let r = client.get(format!("{base}/{path}")).send().await.unwrap();
        assert_eq!(r.status(), 200, "asset {path} not served");
    }
    // a pretty URL resolves to its page
    let r = client.get(format!("{base}/perf")).send().await.unwrap();
    assert_eq!(r.status(), 200);
}

#[tokio::test]
async fn model_independent_api_contract() {
    let hz = boot().await;
    let base = format!("http://{}", hz.addr);
    let client = reqwest::Client::new();

    // status answers with a `state` field even before any model load, and
    // zend's own line under its name
    let r = client
        .get(format!("{base}/v1/status"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let v: serde_json::Value = r.json().await.unwrap();
    assert_eq!(v["state"], "loading", "status: {v}");
    assert_eq!(v["loading"]["current"], "pack", "status: {v}");
    assert_eq!(v["tenants"]["zend"]["state"], "loading", "status: {v}");
    assert!(
        v["tenants"]["zend"].get("ingest_backlog").is_some(),
        "zend's line carries its backlog: {v}"
    );
    // build id for the hot-reload check (frontend force-reloads when it changes)
    assert!(
        v.get("build")
            .and_then(|b| b.as_str())
            .is_some_and(|s| s.len() == 16),
        "status missing build id: {v}"
    );

    // signed out is an answer: the role, and zend's modes for it
    let r = client.get(format!("{base}/v1/me")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let v: serde_json::Value = r.json().await.unwrap();
    assert_eq!(
        v,
        serde_json::json!({
            "role": "unauthenticated",
            "tenants": {"zend": {"tool_modes": ["none", "restricted"], "default_tools": "restricted"}},
        })
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

    // an unknown asset -> 404; an unknown API path -> 404
    for path in ["no-such-asset.js", "v1/no-such-route"] {
        let r = client.get(format!("{base}/{path}")).send().await.unwrap();
        assert_eq!(r.status(), 404, "{path}");
    }

    // the substrate views are an admin's: signed out, refused
    let r = client
        .get(format!("{base}/v1/substrate/zend/system-prompt"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);

    // **A turn must name its conversation.** There is no default conversation
    // for unaddressed turns to accumulate on, so a chat request without a
    // `conv_id` is refused outright rather than filed somewhere. Checked here
    // rather than in a unit test because the refusal is the handler's, and it
    // has to happen before the session is touched — which is why it answers
    // 400 with no engine loaded, where every model-gated route below answers
    // 503.
    for body in [
        serde_json::json!({ "model": "zen-code", "messages": [] }),
        serde_json::json!({ "model": "zen-code", "conv_id": "   ", "messages": [] }),
    ] {
        let r = client
            .post(format!("{base}/v1/chat/completions"))
            .json(&body)
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 400, "expected a refusal for {body}");
        let v: serde_json::Value = r.json().await.unwrap();
        assert_eq!(v["error"]["code"], "missing_conv_id", "body was {v}");
        assert_eq!(v["error"]["param"], "conv_id", "body was {v}");
    }

    // `passthrough` carries the client's own context and never reaches the
    // substrate, so it names no conversation and must not be refused for it.
    // Model-gated like every other engine route, hence 503 and not 400.
    let r = client
        .post(format!("{base}/v1/chat/completions"))
        .json(&serde_json::json!({ "model": "passthrough", "messages": [] }))
        .send()
        .await
        .unwrap();
    assert_ne!(r.status(), 400, "passthrough must not require a conv_id");

    // archive is a model-gated write -> 503 when no engine is loaded
    let r = client
        .post(format!("{base}/v1/conversations/whatever/archive"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);

    // conversation history is model-gated -> 503 without an engine
    let r = client
        .get(format!("{base}/v1/conversations/whatever"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);

    // so is the projection panel's data: one point in full, and the context
    // for a point the client already holds
    let r = client
        .get(format!("{base}/v1/conversations/whatever/projections/0/0"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);
    let r = client
        .post(format!(
            "{base}/v1/conversations/whatever/projection-context"
        ))
        .json(&serde_json::json!({ "turns": [] }))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);
}

/// A conversation's files are its substrate's (`docs/replicated_substrate.md`
/// §10.2): until the model and its substrate load, every files route
/// answers `503`.
#[tokio::test]
async fn conversation_files_wait_for_the_substrate() {
    let hz = boot().await;
    let base = format!("http://{}", hz.addr);
    let client = reqwest::Client::new();
    let r = client
        .get(format!("{base}/v1/conversations/c1/files"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);
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
    assert_eq!(r.status(), 503);
    let r = client
        .get(format!("{base}/v1/conversations/c1/files/7"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);
    let r = client
        .delete(format!("{base}/v1/conversations/c1/files/7"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 503);
}

/// The log pane is any signed-in caller's, never a stranger's: it names paths
/// on this host. Signed out, the upgrade is refused with a status the browser
/// can read.
#[tokio::test]
async fn ws_logs_is_refused_to_a_signed_out_caller() {
    let hz = boot().await;
    let r = reqwest::Client::new()
        .get(format!("http://{}/ws/logs", hz.addr))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);
}

#[tokio::test]
async fn ws_logs_streams_structured_json() {
    use futures_util::StreamExt;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;

    let mut hz = boot().await;

    // One formatted line into the log file, as the subscriber writes it.
    writeln!(
        hz.log,
        "2026-01-01T00:00:01.0Z  INFO zend::harness: structured frame check"
    )
    .unwrap();

    // Signed in, on the gateway's headers — a loopback peer is trusted.
    let mut req = format!("ws://{}/ws/logs", hz.addr)
        .into_client_request()
        .unwrap();
    req.headers_mut()
        .insert("x-tokera-user", "boss".parse().unwrap());
    req.headers_mut()
        .insert("x-tokera-provider", "google".parse().unwrap());
    let (mut ws, _resp) = tokio_tungstenite::connect_async(req)
        .await
        .expect("ws connect");

    // The first replayed frame is our line, read back from the file, framed as
    // structured JSON. A replayed line carries no bus position.
    let msg = ws.next().await.expect("a frame").expect("frame ok");
    let text = msg.into_text().expect("text frame");
    let v: serde_json::Value = serde_json::from_str(&text).expect("frame is JSON");
    for field in ["ts", "level", "target", "msg", "tenant", "seq"] {
        assert!(v.get(field).is_some(), "frame missing {field}: {v}");
    }
    assert_eq!(v["level"], "INFO");
    assert_eq!(v["target"], "zend::harness");
    assert_eq!(v["tenant"], "zend");
    assert_eq!(v["msg"], "structured frame check");
    assert_eq!(v["seq"], 0);
}
