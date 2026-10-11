//! The gateway with members, over real sockets (`docs/replicated_substrate.md`
//! §5, §14.1): members dial `/v1/cluster/zend` over a real WebSocket and serve
//! an axum router on the streams the gateway opens; a plain daemon stands in as
//! the route's configured upstream. The modes, dispatch, UI-pinning, claims,
//! aggregate and socket-drop tables run here.

use std::convert::Infallible;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::ws::{Message as WsMessage, WebSocketUpgrade};
use axum::extract::Path as UrlPath;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::response::IntoResponse;
use axum::routing::get;
use axum::Router;
use cluster::dial::Connect;
use cluster::dispatch::{Dispatch, KeySource, Merge, Route, RouteTable};
use cluster::frame::{BuildId, Capacity, MemberState, ModelId};
use cluster::gateway::Hub;
use cluster::handshake::GroupKey;
use cluster::member::{AppMember, Check, Identity, Link};
use cluster::pipe::socket::{split, SocketRx, SocketTx};
use cluster::pipe::tungstenite::Tungstenite;
use cluster::session::VirtualConn;
use futures_util::{SinkExt, StreamExt};
use http_body_util::BodyExt;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::time::sleep;
use tokio_tungstenite::tungstenite::Message as TMessage;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use tower::Service;
use web::cluster::{ClusterPeer, Gateway};
use web::{Builder, Config};

fn key() -> GroupKey {
    GroupKey::from_bytes([5; 32])
}

async fn spawn(router: Router) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    });
    addr
}

fn header(h: &HeaderMap, name: &str) -> String {
    h.get(name)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string()
}

/// The configured upstream: says it answered, and what hop mark it saw.
fn dumb_daemon() -> Router {
    Router::new().fallback(|headers: HeaderMap| async move {
        axum::Json(json!({"by": "dumb", "hop": header(&headers, "x-tokera-hop")}))
    })
}

/// What every member declares.
fn table() -> RouteTable {
    RouteTable::new(vec![
        Route::new(&[], "/v1/where", Dispatch::Best),
        Route::new(&[], "/v1/conversations/:id", Dispatch::sticky_path("id")),
        Route::new(
            &["POST"],
            "/v1/chat/completions",
            Dispatch::Sticky(vec![
                KeySource::Header("x-tokera-conversation".into()),
                KeySource::Body("conv_id".into()),
            ]),
        ),
        Route::new(&[], "/v1/status", Dispatch::Aggregate(Merge::Envelope)),
        Route::new(&[], "/v1/hang", Dispatch::Best),
        Route::new(&[], "/v1/held", Dispatch::Best),
        Route::new(&[], "/v1/stream", Dispatch::Best),
        Route::new(&[], "/ws/echo", Dispatch::Best),
        Route::new(&[], "/ws/logs", Dispatch::Aggregate(Merge::Lines)),
    ])
    .unwrap()
}

/// A member's own router: what a daemon would serve, reduced to what the
/// tables need.
fn member_api(node: u64, app: Arc<AppMember>) -> Router {
    let me = format!("member-{node}");
    let who = me.clone();
    let page = me.clone();
    let status = me.clone();
    let line = me.clone();
    Router::new()
        .route(
            "/v1/where",
            get(move |headers: HeaderMap| async move {
                axum::Json(json!({
                    "by": who,
                    "hop": header(&headers, "x-tokera-hop"),
                    "cookie": header(&headers, "cookie"),
                    "email": header(&headers, "x-tokera-email"),
                    "from_member": header(&headers, "x-tokera-from-member"),
                }))
            }),
        )
        .route(
            "/v1/conversations/:id",
            get(move |UrlPath(id): UrlPath<String>| async move {
                match app.claim(id.as_bytes()) {
                    Check::Held(holder) => (
                        StatusCode::CONFLICT,
                        [("x-tokera-holder", format!("{holder:016x}"))],
                        "held elsewhere",
                    )
                        .into_response(),
                    _ => format!("{me} serves {id}").into_response(),
                }
            }),
        )
        .route(
            "/v1/status",
            get(move || async move {
                if node == 9 {
                    sleep(Duration::from_secs(30)).await;
                }
                axum::Json(json!({"state": "ready", "by": status}))
            }),
        )
        .route(
            "/v1/hang",
            get(|| async {
                sleep(Duration::from_secs(600)).await;
                "never"
            }),
        )
        // A key whose holder this member knows of and the gateway does not.
        .route(
            "/v1/held",
            get(|| async {
                (
                    StatusCode::CONFLICT,
                    [("x-tokera-holder", "00000000000000aa")],
                    "held elsewhere",
                )
            }),
        )
        .route(
            "/v1/stream",
            get(|| async {
                let first =
                    futures_util::stream::once(async { Ok::<_, Infallible>("first chunk") });
                let rest = futures_util::stream::once(async {
                    sleep(Duration::from_secs(600)).await;
                    Ok::<_, Infallible>("never")
                });
                Body::from_stream(first.chain(rest))
            }),
        )
        .route(
            "/ws/echo",
            get(|ws: WebSocketUpgrade| async move {
                ws.on_upgrade(|mut socket| async move {
                    while let Some(Ok(m)) = socket.recv().await {
                        if socket.send(m).await.is_err() {
                            return;
                        }
                    }
                })
            }),
        )
        .route(
            "/ws/logs",
            get(move |ws: WebSocketUpgrade| async move {
                ws.on_upgrade(move |mut socket| async move {
                    let text = json!({"msg": "hello", "from": line}).to_string();
                    let _ = socket.send(WsMessage::Text(text)).await;
                    while socket.recv().await.is_some() {}
                })
            }),
        )
        .fallback(move || async move { format!("<p>page from {page}</p>") })
}

/// Serve `router` on a stream the gateway opened, as a member's host does.
fn serve_stream(conn: VirtualConn, router: Router) {
    tokio::spawn(async move {
        let svc = service_fn(move |req: Request<Incoming>| {
            let mut router = router.clone();
            async move {
                let mut req = req.map(Body::new);
                req.extensions_mut().insert(ClusterPeer { cluster: 0 });
                router.call(req).await
            }
        });
        let _ = hyper::server::conn::http1::Builder::new()
            .serve_connection(TokioIo::new(conn), svc)
            .with_upgrades()
            .await;
    });
}

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

struct Dial(String);

impl Connect for Dial {
    type Tx = SocketTx;
    type Rx = SocketRx<Ws, Tungstenite>;
    async fn connect(&mut self) -> Result<(SocketTx, SocketRx<Ws, Tungstenite>), String> {
        let (ws, _) = connect_async(self.0.as_str())
            .await
            .map_err(|e| e.to_string())?;
        Ok(split::<Tungstenite, _, _>(ws))
    }
}

struct Member {
    app: Arc<AppMember>,
    task: tokio::task::JoinHandle<()>,
}

impl Member {
    fn leave(&self) {
        self.app.close_all("test");
        self.task.abort();
    }
}

fn member(gateway: SocketAddr, node: u64, power: u32) -> Member {
    let ready = Arc::new(AtomicBool::new(true));
    let app = AppMember::new(
        Identity {
            app: "zend".into(),
            node,
            host: format!("m{node}"),
            build: BuildId {
                count: 7,
                hash: "h7".into(),
                local: String::new(),
            },
            model: Some(ModelId {
                arch: "Qwen4Exp".into(),
                weights: [1; 32],
                tokenizer: [2; 32],
                rope: "yarn".into(),
                describe: "Flash-Next".into(),
            }),
            repos: vec![],
        },
        key(),
        vec![format!("ws://{gateway}")],
        table(),
        Box::new(move || Capacity {
            state: if ready.load(Ordering::SeqCst) {
                MemberState::Ready
            } else {
                MemberState::Loading
            },
            power,
            idle: true,
            slots_free: 2,
            decodes: 0,
            prefill_backlog: 0,
            vram_free_mib: 1000,
            cores: 8,
            ..Capacity::default()
        }),
        None,
        Instant::now(),
    );
    let router = member_api(node, app.clone());
    let url = format!("ws://{gateway}/v1/cluster/zend");
    let task = tokio::spawn({
        let app = app.clone();
        async move {
            app.run(0, Dial(url), move |conn| serve_stream(conn, router.clone()))
                .await;
        }
    });
    Member { app, task }
}

fn content_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

async fn gateway(dumb: SocketAddr) -> (SocketAddr, Arc<Hub>) {
    let yaml = format!(
        r#"
cluster: {{group_key_file: "secrets/group.key"}}
sites:
  - name: zend
    hosts: ["code.test"]
    default: true
    api:
      - {{prefix: /, upstream: "http://{dumb}", cluster: zend}}
      - {{prefix: /plain, upstream: "http://{dumb}"}}
"#
    );
    let cfg = Config::from_yaml(&yaml, content_dir()).unwrap();
    let hub = Hub::new(key(), "gw-test");
    let router = Builder::new(cfg)
        .cluster(Gateway::with_hub(hub.clone()))
        .router();
    (spawn(router).await, hub)
}

struct Got {
    status: u16,
    headers: HeaderMap,
    body: String,
}

impl Got {
    fn json(&self) -> Value {
        serde_json::from_str(&self.body).unwrap_or(Value::Null)
    }
}

async fn ask(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
    body: &str,
) -> Got {
    let stream = TcpStream::connect(addr).await.unwrap();
    let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .unwrap();
    tokio::spawn(conn);
    let mut req = Request::builder()
        .method(method)
        .uri(path)
        .header("host", "code.test");
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let res = sender
        .send_request(req.body(Body::from(body.to_string())).unwrap())
        .await
        .unwrap();
    let status = res.status().as_u16();
    let headers = res.headers().clone();
    let body =
        String::from_utf8_lossy(&res.into_body().collect().await.unwrap().to_bytes()).into_owned();
    Got {
        status,
        headers,
        body,
    }
}

async fn fetch(addr: SocketAddr, path: &str) -> Got {
    ask(addr, "GET", path, &[], "").await
}

fn both_ready(hub: &Hub) -> bool {
    let status = hub.status(0);
    status.members.iter().filter(|m| m.state == "ready").count() == 2
}

async fn until(what: &str, mut ok: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ok() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        sleep(Duration::from_millis(10)).await;
    }
}

/// **Modes** (§5.7): a route with `cluster:` and no ready member goes to its
/// configured URL, marked dumb; the first ready member switches it; the last
/// leaving reverts it; a route without `cluster:` never switches.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_mode_follows_the_ready_members() {
    let dumb = spawn(dumb_daemon()).await;
    let (gw, hub) = gateway(dumb).await;

    let r = fetch(gw, "/v1/where").await.json();
    assert_eq!(r, json!({"by": "dumb", "hop": "dumb"}));

    let m = member(gw, 1, 100);
    until("clustered", || hub.clustered("zend")).await;
    let r = fetch(gw, "/v1/where").await.json();
    assert_eq!(r["by"], "member-1");
    assert_eq!(r["hop"], "", "a member is never sent the dumb mark");
    assert_eq!(fetch(gw, "/plain/x").await.json()["by"], "dumb");

    m.leave();
    until("dumb again", || !hub.clustered("zend")).await;
    assert_eq!(fetch(gw, "/v1/where").await.json()["by"], "dumb");
}

/// **Dispatch, claims and pinning** (§5.2–§5.4): `Best` takes the most
/// powerful idle member; a sticky key stays with the member that claimed it;
/// a body key is read from the buffered body; a page's UI requests stay on one
/// member through its cookie, which API requests ignore; the cookie and the
/// session token never reach a member; `x-tokera-node` is an admin's.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn requests_go_where_their_dispatch_says() {
    let dumb = spawn(dumb_daemon()).await;
    let (gw, hub) = gateway(dumb).await;
    let a = member(gw, 1, 100);
    let b = member(gw, 2, 300);
    until("both ready", || both_ready(&hub)).await;

    assert_eq!(fetch(gw, "/v1/where").await.json()["by"], "member-2");

    // The first request for a conversation lands Best and is claimed there;
    // every later one follows the claim.
    let first = fetch(gw, "/v1/conversations/c-1").await;
    assert_eq!(first.body, "member-2 serves c-1");
    until("the claim reached the gateway", || {
        hub.status(0).claims.iter().any(|c| c.key == "c-1")
    })
    .await;
    for _ in 0..5 {
        assert_eq!(
            fetch(gw, "/v1/conversations/c-1").await.body,
            "member-2 serves c-1"
        );
    }
    // The other member, told of the claim, refuses it.
    assert_eq!(a.app.claim(b"c-1"), Check::Held(2));

    // UI requests: the first is pinned and the cookie set; the rest follow it.
    let page = fetch(gw, "/index.html").await;
    let cookie = header(&page.headers, "set-cookie");
    assert!(cookie.starts_with("tokera_ui_zend="), "{cookie}");
    let pair = cookie.split(';').next().unwrap().to_string();
    for path in ["/app.js", "/style.css", "/", "/index.html"] {
        let again = ask(gw, "GET", path, &[("cookie", &pair)], "").await;
        assert_eq!(again.body, page.body, "{path}");
        assert!(header(&again.headers, "set-cookie").is_empty());
    }
    // An API request ignores the pin, and no member sees a cookie.
    let api = ask(gw, "GET", "/v1/where", &[("cookie", &pair)], "")
        .await
        .json();
    assert_eq!(api["by"], "member-2");
    assert_eq!(api["cookie"], "");

    // Addressing one member is an admin's.
    let refused = ask(
        gw,
        "GET",
        "/v1/where",
        &[("x-tokera-node", "0000000000000001")],
        "",
    )
    .await;
    assert_eq!(refused.status, 403);
    a.leave();
    b.leave();
}

/// **A body key** is read from the buffered body — at most 8 MiB — and a header
/// key ahead of it wins without the body being read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_body_key_routes_by_the_buffered_body() {
    let dumb = spawn(dumb_daemon()).await;
    let (gw, hub) = gateway(dumb).await;
    let a = member(gw, 1, 100);
    let b = member(gw, 2, 300);
    until("both ready", || both_ready(&hub)).await;
    // The weaker member holds the conversation, so `Best` would go elsewhere.
    assert!(matches!(a.app.claim(b"c-body"), Check::Took(_)));
    until("the claim reached the gateway", || {
        hub.status(0).claims.iter().any(|c| c.key == "c-body")
    })
    .await;
    // The members' fallback answers any method with the page, naming itself.
    let by_body = ask(
        gw,
        "POST",
        "/v1/chat/completions",
        &[],
        r#"{"conv_id":"c-body"}"#,
    )
    .await;
    assert_eq!(by_body.body, "<p>page from member-1</p>");
    let fresh = ask(
        gw,
        "POST",
        "/v1/chat/completions",
        &[],
        r#"{"messages":[]}"#,
    )
    .await;
    assert_eq!(fresh.body, "<p>page from member-2</p>");
    let by_header = ask(
        gw,
        "POST",
        "/v1/chat/completions",
        &[("x-tokera-conversation", "c-body")],
        r#"{"conv_id":"another"}"#,
    )
    .await;
    assert_eq!(by_header.body, "<p>page from member-1</p>");
    let too_big = format!(r#"{{"conv_id":"c","pad":"{}"}}"#, "x".repeat(9 << 20));
    let r = ask(gw, "POST", "/v1/chat/completions", &[], &too_big).await;
    assert_eq!(r.status, 413);
    a.leave();
    b.leave();
}

/// **Aggregates** (§5.5): `/v1/status` answers an envelope of every ready
/// member, with the group block, marked `x-tokera-aggregate`; a member past
/// the 3 s deadline is listed as such and does not hold the answer up.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_aggregate_lists_every_member_and_marks_the_late() {
    let dumb = spawn(dumb_daemon()).await;
    let (gw, hub) = gateway(dumb).await;
    let _a = member(gw, 1, 100);
    let _slow = member(gw, 9, 100);
    until("both ready", || both_ready(&hub)).await;
    let t = Instant::now();
    let r = fetch(gw, "/v1/status").await;
    let took = t.elapsed();
    assert!(took < Duration::from_millis(3_800), "{took:?}");
    assert_eq!(header(&r.headers, "x-tokera-aggregate"), "1");
    let v = r.json();
    let members = v["members"].as_array().unwrap();
    assert_eq!(members.len(), 2);
    let fast = members
        .iter()
        .find(|m| m["node"] == "0000000000000001")
        .unwrap();
    assert_eq!(fast["status"], 200);
    assert_eq!(fast["body"]["by"], "member-1");
    assert_eq!(fast["host"], "m1");
    let slow = members
        .iter()
        .find(|m| m["node"] == "0000000000000009")
        .unwrap();
    assert_eq!(
        (slow["status"].clone(), slow["error"].clone()),
        (json!(504), json!("deadline"))
    );
    assert_eq!(v["group"]["members"], 2);
    assert_eq!(v["group"]["model_lock"], "Flash-Next");
    assert_eq!(v["group"]["floor"]["count"], 7);
}

/// **A merged socket** (§5.2 `/ws/logs`): one browser WebSocket carries every
/// member's lines, each tagged with its node.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_merged_socket_carries_every_members_lines() {
    let dumb = spawn(dumb_daemon()).await;
    let (gw, hub) = gateway(dumb).await;
    let _a = member(gw, 1, 100);
    let _b = member(gw, 2, 100);
    until("both ready", || both_ready(&hub)).await;
    let (mut ws, _) = connect_async(format!("ws://{gw}/ws/logs")).await.unwrap();
    let mut from = Vec::new();
    while from.len() < 2 {
        let TMessage::Text(t) = ws.next().await.unwrap().unwrap() else {
            continue;
        };
        let v: Value = serde_json::from_str(&t).unwrap();
        assert_eq!(v["msg"], "hello");
        // Each line is tagged with the member that sent it.
        let node = v["node"].as_str().unwrap().to_string();
        let n = u64::from_str_radix(&node, 16).unwrap();
        assert_eq!(v["from"], format!("member-{n}"));
        assert_eq!(v["host"], format!("m{n}"));
        from.push(node);
    }
    from.sort();
    assert_eq!(from, ["0000000000000001", "0000000000000002"]);
    let _ = ws.close(None).await;
}

/// **A dropped socket** (§5.6): an exchange with no response head yet gets a
/// `503`; a body in progress is aborted rather than ended; a tunnelled
/// WebSocket closes `1011`; nothing is replayed on another member.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_dropped_socket_fails_what_rides_it() {
    let dumb = spawn(dumb_daemon()).await;
    let (gw, hub) = gateway(dumb).await;
    let m = member(gw, 1, 100);
    until("ready", || hub.clustered("zend")).await;

    // A response not yet begun.
    let hanging = tokio::spawn(fetch(gw, "/v1/hang"));
    // A body under way.
    let stream = TcpStream::connect(gw).await.unwrap();
    let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .unwrap();
    tokio::spawn(conn);
    let res = sender
        .send_request(
            Request::get("/v1/stream")
                .header("host", "code.test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), 200);
    let mut body = res.into_body();
    let first = body.frame().await.unwrap().unwrap().into_data().unwrap();
    assert_eq!(&first[..], b"first chunk");
    // A tunnelled socket.
    let (mut ws, _) = connect_async(format!("ws://{gw}/ws/echo")).await.unwrap();
    ws.send(TMessage::Text("ping".into())).await.unwrap();
    assert_eq!(
        ws.next().await.unwrap().unwrap(),
        TMessage::Text("ping".into())
    );

    sleep(Duration::from_millis(100)).await;
    m.leave();

    let hung = hanging.await.unwrap();
    assert_eq!(hung.status, 503, "{}", hung.body);
    let rest = body.frame().await;
    assert!(
        matches!(rest, Some(Err(_))),
        "the body was ended, not aborted: {rest:?}"
    );
    match ws.next().await {
        Some(Ok(TMessage::Close(Some(frame)))) => assert_eq!(u16::from(frame.code), 1011),
        other => panic!("expected a 1011 close, got {other:?}"),
    }
    let failures = hub.status(0).traffic;
    assert!(failures.failed_before_head >= 1);
    assert!(failures.aborted_mid_body >= 1);
    assert!(failures.tunnels_closed >= 1);
}

/// **A key held through another cluster** (§5.4): the member reached answers
/// `409` naming the holder, and the gateway answers `503` with `Retry-After`,
/// since the next attempt may land on a gateway the holder is joined to.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_holder_elsewhere_is_answered_retry() {
    let dumb = spawn(dumb_daemon()).await;
    let (gw, hub) = gateway(dumb).await;
    let _m = member(gw, 1, 100);
    until("ready", || hub.clustered("zend")).await;
    let r = fetch(gw, "/v1/held").await;
    assert_eq!(r.status, 503, "{}", r.body);
    assert_eq!(header(&r.headers, "retry-after"), "1");
    assert_eq!(hub.status(0).traffic.holder_elsewhere, 1);
}

/// **The status page** is an admin's, and with no sign-in configured nobody is.
#[tokio::test]
async fn the_status_page_is_an_admins() {
    let dumb = spawn(dumb_daemon()).await;
    let (gw, hub) = gateway(dumb).await;
    assert_eq!(fetch(gw, "/gateway").await.status, 401);
    assert_eq!(fetch(gw, "/gateway/status.json").await.status, 401);
    for command in ["drain-gateway", "undrain-gateway"] {
        let path = format!("/gateway/command/{command}");
        assert_eq!(ask(gw, "POST", &path, &[], "").await.status, 401);
    }
    let member_command = "/gateway/member/zend/0000000000000001/v1/cluster/command/drain";
    assert_eq!(ask(gw, "POST", member_command, &[], "").await.status, 401);
    assert!(!hub.draining(), "a refused command does nothing");
}

/// **The cluster socket** refuses an app that is none of the three, and a
/// request that is not an HTTP/1.1 upgrade.
#[tokio::test]
async fn the_endpoint_refuses_what_cannot_be_a_member() {
    let dumb = spawn(dumb_daemon()).await;
    let (gw, _hub) = gateway(dumb).await;
    assert_eq!(fetch(gw, "/v1/cluster/other").await.status, 404);
    // A plain GET with no upgrade is refused by the WebSocket handshake.
    assert_eq!(fetch(gw, "/v1/cluster/zend").await.status, 400);
}

/// **Which member a request came from** (§9.3): a request a member's own
/// stream carried reaches the member it is routed to naming the sender; a
/// client claiming to be a member is not believed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn only_a_members_own_stream_names_it() {
    let dumb = spawn(dumb_daemon()).await;
    let (gw, hub) = gateway(dumb).await;
    let m = member(gw, 3, 100);
    until("joined", || hub.status(0).members.len() == 1).await;
    until("clustered", || hub.clustered("zend")).await;

    let forged = ask(
        gw,
        "GET",
        "/v1/where",
        &[("x-tokera-from-member", "00000000000000aa")],
        "",
    )
    .await;
    assert_eq!(forged.json()["from_member"], "", "{}", forged.body);

    let conn = m.app.open_up().expect("a socket to open a stream on");
    let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(conn))
        .await
        .unwrap();
    tokio::spawn(connection);
    let res = sender
        .send_request(
            Request::get("/v1/where")
                .header("host", "code.test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body = res.into_body().collect().await.unwrap().to_bytes();
    let got: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(got["from_member"], "0000000000000003");
    m.leave();
}

/// **A member dialling the real endpoint** over a real WebSocket is joined.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_member_dialling_in_is_joined() {
    let dumb = spawn(dumb_daemon()).await;
    let (gw, _hub) = gateway(dumb).await;
    let m = member(gw, 1, 100);
    until("joined", || m.app.link() == Link::Joined).await;
    m.leave();
}
