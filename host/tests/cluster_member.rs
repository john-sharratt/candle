//! A member host and a gateway, over real sockets (`docs/replicated_substrate.md`
//! §5.8, §14.1 "own listener"): the member's own listener forwards a request
//! up while joined — carrying the identity it resolved — and the cluster routes
//! it back down its socket; `/v1/status` and a request a gateway marked dumb
//! are answered where they arrive; a member that reaches no cluster serves
//! locally; a sticky key is claimed by the member that serves it.

use std::net::{IpAddr, SocketAddr};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::{Path as UrlPath, Request};
use axum::Router;
use cluster::consensus::votes::MemoryVotes;
use cluster::dispatch::{Dispatch, Merge};
use cluster::frame::{BuildId, ModelId};
use cluster::gateway::{state_name, Hub};
use cluster::handshake::GroupKey;
use cluster::member::{Check, Link};
use cluster::replicate::{MemoryStore, Replication, Store};
use cluster::upgrade::artifacts::RunProbe;
use cluster::upgrade::runtime::{InstallJob, Installer, Setup, Upgrader};
use cluster::version::info::{platform, BuildInfo, Opens, ZEND};
use cluster_dial::Target;
use host::access::LocalSignin;
use host::cluster::leave::drain_and_send;
use host::cluster::{Cluster, ClusterSettings, Join};
use host::commands::jobs::Job;
use host::commands::parse::{Body as CommandBody, Command};
use host::commands::Commands;
use host::guard::{dispatch_table, get, post, Api};
use host::model::ModelView;
use host::server::{builder, Listen};
use host::stopping::Stopping;
use host::telemetry::Telemetry;
use host::TenantUi;
use http_body_util::BodyExt;
use hyper_util::rt::TokioIo;
use include_dir::{include_dir, Dir};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::runtime::Handle;
use tokio::task::spawn_blocking;
use tokio::time::sleep;
use web::auth::Role;
use web::cluster::{ClusterPeer, Gateway};
use web::{Builder, Config};

fn key() -> GroupKey {
    GroupKey::from_bytes([9; 32])
}

/// An installer for members that are never asked to install.
struct NoInstall;

impl Installer for NoInstall {
    fn install(&self, _: InstallJob) -> Result<(), String> {
        Err("this test installs nothing".into())
    }
}

/// A member's upgrades at build 7, keeping its state under `dir`.
fn upgrader(dir: &Path) -> Arc<Upgrader> {
    let up = Upgrader::new(
        Setup {
            info: BuildInfo {
                kind: ZEND.into(),
                count: 7,
                hash: "h7".into(),
                local: String::new(),
                platform: platform(),
                opens: Opens::default(),
            },
            exe: std::env::current_exe().unwrap(),
            upgrade_dir: dir.join("upgrade"),
            keep_builds: 5,
            failed_from: None,
        },
        Arc::new(NoInstall),
        Arc::new(RunProbe),
    );
    up.start();
    up
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

/// A port nothing listens on.
async fn closed_port() -> SocketAddr {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    l.local_addr().unwrap()
}

/// The member's API: each route says whether it was reached over a socket,
/// and who it was told the caller is; one writes a key into the member's
/// substrate, another says whether the member holds it.
fn api(store: Arc<MemoryStore>) -> Api<()> {
    let writer = store.clone();
    let write = move |UrlPath(k): UrlPath<String>| {
        let s = writer.clone();
        async move {
            s.write(k.as_bytes(), b"written");
            format!("{:x}", s.node())
        }
    };
    let read = move |UrlPath(k): UrlPath<String>| {
        let s = store.clone();
        async move { axum::Json(json!({"held": s.value(k.as_bytes()).is_some(), "node": s.node()})) }
    };
    let seen = |req: Request| async move {
        let email = req
            .headers()
            .get("x-tokera-email")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        axum::Json(json!({
            "socket": req.extensions().get::<ClusterPeer>().is_some(),
            "email": email,
        }))
    };
    Api::new(Default::default())
        .route(
            "/v1/where",
            Role::Unauthenticated,
            Dispatch::Best,
            get(seen),
        )
        .route(
            "/v1/status",
            Role::Unauthenticated,
            Dispatch::Aggregate(Merge::Envelope),
            get(seen),
        )
        .route(
            "/v1/conversations/:id",
            Role::Unauthenticated,
            Dispatch::sticky_path("id"),
            get(|UrlPath(id): UrlPath<String>| async move { format!("served {id}") }),
        )
        // A write is a POST: a member flushes what a request appended before
        // its response ends only for a method that can append.
        .route(
            "/v1/write/:k",
            Role::Unauthenticated,
            Dispatch::Best,
            post(write),
        )
        .route(
            "/v1/read/:k",
            Role::Unauthenticated,
            Dispatch::Any,
            get(read),
        )
}

struct Member {
    addr: SocketAddr,
    cluster: Arc<Cluster>,
    join: Join,
    store: Arc<MemoryStore>,
}

/// The UI every member here serves.
static UI: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/tests/ui/zend");

/// A member whose upgrade state lives under `dir`, serving [`UI`].
async fn member(gateway: SocketAddr, node: u64, dir: &Path) -> Member {
    let settings = ClusterSettings {
        targets: vec![Target::parse(&format!("http://{gateway}"), None).unwrap()],
        key: key(),
        affinity_idle: Duration::from_secs(600),
        quorum_max: None,
        host: format!("m{node}"),
        build: BuildId {
            count: 7,
            hash: "h7".into(),
            local: String::new(),
        },
        agent: None,
    };
    let cluster = Cluster::new(
        settings,
        &["zend"],
        Telemetry::new(ModelView::of_prescribed()),
        Handle::current(),
        upgrader(dir),
    )
    .unwrap();
    let store = Arc::new(MemoryStore::new(node));
    let routes = api(store.clone());
    let table = dispatch_table(routes.declared()).unwrap();
    cluster.link("zend").unwrap().set_routes(table);
    let uis = vec![(
        "zend",
        TenantUi {
            hosts: vec![],
            content: &UI,
            default: true,
        },
    )];
    let listen = Listen {
        bind: IpAddr::from([127, 0, 0, 1]),
        port: 0,
        tls_port: None,
        gateways: vec![],
    };
    let router = builder(
        &listen,
        &uis,
        routes.into_routes(()).router,
        LocalSignin::default(),
        Some(cluster.clone()),
    )
    .unwrap()
    .router();
    let addr = spawn(router).await;
    let join = Join::new(cluster.clone(), "zend").unwrap();
    Member {
        addr,
        cluster,
        join,
        store,
    }
}

impl Member {
    /// Connect as `node`, as the tenant's loader does, catch up, and say it is
    /// ready.
    async fn connect(&self, node: u64) -> Link {
        let join = self.join.clone();
        let replication =
            Replication::open(self.store.clone(), Box::new(MemoryVotes::new())).unwrap();
        let link = spawn_blocking(move || {
            let link = join.connect(
                node,
                Some(ModelId {
                    arch: "Qwen4Exp".into(),
                    weights: [1; 32],
                    tokenizer: [2; 32],
                    rope: "yarn".into(),
                    describe: "Flash-Next".into(),
                }),
                replication,
                || 0,
                || false,
                |_| {},
            );
            join.catch_up(|| false, |_| {});
            link
        })
        .await
        .unwrap();
        self.cluster.link("zend").unwrap().set_ready(true);
        link
    }
}

async fn gateway(upstream: &str) -> (SocketAddr, Arc<Hub>) {
    let yaml = format!(
        r#"
cluster: {{group_key_file: "unused.key"}}
sites:
  - name: zend
    default: true
    api:
      - {{prefix: /, upstream: "{upstream}", cluster: zend}}
"#
    );
    let cfg = Config::from_yaml(&yaml, Path::new(env!("CARGO_MANIFEST_DIR"))).unwrap();
    let hub = Hub::new(key(), "gw-member-test");
    let router = Builder::new(cfg)
        .cluster(Gateway::with_hub(hub.clone()))
        .router();
    (spawn(router).await, hub)
}

async fn ask(addr: SocketAddr, path: &str, headers: &[(&str, &str)]) -> (u16, String) {
    send(addr, "GET", path, headers).await
}

async fn send(
    addr: SocketAddr,
    method: &str,
    path: &str,
    headers: &[(&str, &str)],
) -> (u16, String) {
    let stream = TcpStream::connect(addr).await.unwrap();
    let (mut sender, conn) = hyper::client::conn::http1::handshake(TokioIo::new(stream))
        .await
        .unwrap();
    tokio::spawn(conn);
    let mut req = hyper::Request::builder()
        .method(method)
        .uri(path)
        .header("host", "code.test");
    for (k, v) in headers {
        req = req.header(*k, *v);
    }
    let res = sender
        .send_request(req.body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = res.status().as_u16();
    let body = res.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8_lossy(&body).into_owned())
}

async fn ask_json(addr: SocketAddr, path: &str, headers: &[(&str, &str)]) -> Value {
    let (status, body) = ask(addr, path, headers).await;
    assert_eq!(status, 200, "{path}: {body}");
    serde_json::from_str(&body).unwrap()
}

/// Wait for `read` to give `want`; on a timeout, fail with what it last gave.
async fn until_eq<T: PartialEq + std::fmt::Debug>(what: &str, read: impl Fn() -> T, want: T) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let now = read();
        if now == want {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for {what} to be {want:?}: it is {now:?}"
        );
        sleep(Duration::from_millis(10)).await;
    }
}

async fn until(what: &str, mut ok: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !ok() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        sleep(Duration::from_millis(10)).await;
    }
}

/// **Joined**: a request on the member's own listener goes up to the gateway
/// with the identity the listener believed, and the cluster routes it back
/// down the member's socket; `/v1/status` and a request an allowlisted gateway
/// marked dumb are answered on the listener; a sticky key is claimed by the
/// member that serves it, and the gateway learns the claim.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_joined_member_forwards_its_listeners_requests_up() {
    let dir = tempfile::TempDir::new().unwrap();
    let dumb = closed_port().await;
    let (gw, hub) = gateway(&format!("http://{dumb}")).await;
    let m = member(gw, 1, dir.path()).await;
    assert_eq!(m.connect(1).await, Link::Joined);
    until("clustered", || hub.clustered("zend")).await;

    // Loopback is believed, so the identity set here is the one carried up.
    let signed_in = [
        ("x-tokera-user", "g-1"),
        ("x-tokera-provider", "google"),
        ("x-tokera-email", "boss@example.com"),
    ];
    let up = ask_json(m.addr, "/v1/where", &signed_in).await;
    assert_eq!(up, json!({"socket": true, "email": "boss@example.com"}));
    // Through the gateway directly, the same member answers off its socket.
    assert_eq!(ask_json(gw, "/v1/where", &[]).await["socket"], true);
    // The status route never forwards.
    assert_eq!(ask_json(m.addr, "/v1/status", &[]).await["socket"], false);
    // A request a gateway sent here in dumb mode is never sent back up.
    let dumb_hop = ask_json(m.addr, "/v1/where", &[("x-tokera-hop", "dumb")]).await;
    assert_eq!(dumb_hop["socket"], false);

    // A conversation reached on the listener is claimed where it is served.
    let (status, body) = ask(m.addr, "/v1/conversations/c-9", &[]).await;
    assert_eq!((status, body.as_str()), (200, "served c-9"));
    until("the claim reached the gateway", || {
        hub.status(0)
            .claims
            .iter()
            .any(|c| c.key == "c-9" && c.node == "0000000000000001")
    })
    .await;
    m.cluster.close_all("test over");
}

/// **A write is seen by the next request, wherever it lands** (§6.4): a
/// request through the gateway writes on the member it reaches, whose response
/// ends only after the write's event is on its socket; the next request, routed
/// to either member, is served only after the records ahead of it are applied,
/// and finds the write.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_write_is_seen_by_the_next_request_on_any_member() {
    let dir = tempfile::TempDir::new().unwrap();
    let dumb = closed_port().await;
    let (gw, hub) = gateway(&format!("http://{dumb}")).await;
    let a = member(gw, 1, dir.path()).await;
    let b = member(gw, 2, dir.path()).await;
    assert_eq!(a.connect(1).await, Link::Joined);
    assert_eq!(b.connect(2).await, Link::Joined);
    until("both routable", || {
        [1, 2]
            .iter()
            .all(|n| hub.member("zend", *n).is_some_and(|m| m.routable()))
    })
    .await;

    let mut readers = std::collections::BTreeSet::new();
    for k in 0..20 {
        let (status, _) = send(gw, "POST", &format!("/v1/write/k{k}"), &[]).await;
        assert_eq!(status, 200);
        for _ in 0..2 {
            let read = ask_json(gw, &format!("/v1/read/k{k}"), &[]).await;
            assert_eq!(read["held"], true, "k{k} read on member {}", read["node"]);
            readers.insert(read["node"].as_u64().unwrap());
        }
    }
    assert_eq!(readers.len(), 2, "reads reached both members");
    a.cluster.close_all("test over");
    b.cluster.close_all("test over");
}

/// Wait for the gateway's log to say `text` of `node`; on a timeout, fail with
/// what it did say.
async fn until_logged(hub: &Hub, node: &str, text: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let entries = hub.log.read(None, Some(node), 1000);
        if entries.iter().any(|e| e.text.contains(text)) {
            return;
        }
        if Instant::now() >= deadline {
            let said: Vec<&str> = entries.iter().map(|e| e.text.as_str()).collect();
            panic!("the log never said {text:?} of {node}; it said {said:#?}");
        }
        sleep(Duration::from_millis(10)).await;
    }
}

/// Wait for `job` to end; its last line.
async fn ended(job: &Job) -> Value {
    until("the job ended", || job.ended()).await;
    job.snapshot().pop().unwrap()
}

/// **Commands on a member** (§12.4): `drain` hands back its claims and the
/// gateway routes it nothing new, `undrain` makes it routable again, and each
/// command shows in the gateway's coordination log through `COORD` — who ran
/// it, and how it ended. `set-quorum` writes `max_members` lower.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn commands_run_on_a_member_and_reach_the_log() {
    let dir = tempfile::TempDir::new().unwrap();
    let dumb = closed_port().await;
    let (gw, hub) = gateway(&format!("http://{dumb}")).await;
    let a = member(gw, 1, dir.path()).await;
    let b = member(gw, 2, dir.path()).await;
    assert_eq!(a.connect(1).await, Link::Joined);
    assert_eq!(b.connect(2).await, Link::Joined);
    let routable = |n| hub.member("zend", n).is_some_and(|m| m.routable());
    until("both routable", || routable(1) && routable(2)).await;
    // Seen together, the two raise the group's size to two.
    until("max_members two", || b.store.max_members() == 2).await;
    let b_member = b.cluster.link("zend").unwrap().member().unwrap().clone();
    assert!(matches!(b_member.claim(b"c-2"), Check::Took(_)));

    let commands = Arc::new(Commands::new(
        Some(b.cluster.clone()),
        vec![],
        Arc::new(Stopping::new()),
        "m2".into(),
    ));
    let drain = commands
        .start(
            Command::Drain,
            &CommandBody::default(),
            "ops@example.com".into(),
        )
        .unwrap();
    assert_eq!(
        ended(&drain).await["result"],
        json!({"zend": {"claims": 1, "leases": 0}})
    );
    until("b draining", || {
        hub.member("zend", 2)
            .is_some_and(|m| !m.routable() && state_name(m.state()) == "draining")
    })
    .await;
    let b_hex = "0000000000000002";
    until_logged(
        &hub,
        b_hex,
        r#"command drain (job 1) by ops@example.com, with {"app":null}"#,
    )
    .await;
    until_logged(&hub, b_hex, "command drain (job 1) ended: ok").await;

    let undrain = commands
        .start(
            Command::Undrain,
            &CommandBody::default(),
            "ops@example.com".into(),
        )
        .unwrap();
    assert_eq!(ended(&undrain).await["event"], "done");
    until("b routable again", || routable(2)).await;

    // A leaves for good; B, alone, is told the group is one.
    a.cluster.close_all("retired");
    until("a gone", || hub.member("zend", 1).is_none()).await;
    sleep(Duration::from_secs(1)).await;
    let quorum = commands
        .start(
            Command::SetQuorum { max_members: 1 },
            &CommandBody::default(),
            "ops@example.com".into(),
        )
        .unwrap();
    assert_eq!(
        ended(&quorum).await["result"],
        json!({"zend": {"max_members": 1, "quorum": 1}})
    );
    let b_link = b.cluster.link("zend").unwrap().member().unwrap().clone();
    until_eq(
        "b's max_members, and the group b sees",
        || {
            let group: Vec<(u64, String)> = b_link
                .group()
                .iter()
                .map(|m| (m.node, state_name(m.state).to_string()))
                .collect();
            (b.store.max_members(), group)
        },
        (1, vec![(2, "ready".to_string())]),
    )
    .await;
    until_logged(&hub, b_hex, "max_members 2 → 1, quorum 1").await;
    b.cluster.close_all("test over");
}

/// **Leaving first** (§12.4): before the process stops, a member drains and
/// sends what it wrote; the stop's steps are what its loading page shows.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stopping_member_drains_and_sends_first() {
    let dir = tempfile::TempDir::new().unwrap();
    let dumb = closed_port().await;
    let (gw, hub) = gateway(&format!("http://{dumb}")).await;
    let a = member(gw, 1, dir.path()).await;
    let b = member(gw, 2, dir.path()).await;
    assert_eq!(a.connect(1).await, Link::Joined);
    assert_eq!(b.connect(2).await, Link::Joined);
    for k in 0..50 {
        b.store.write(format!("b/{k}").as_bytes(), b"x");
    }
    let stopping = Stopping::new();
    assert_eq!(drain_and_send(&b.cluster, &stopping).await, 0);
    until("b draining", || {
        hub.member("zend", 2)
            .is_some_and(|m| state_name(m.state()) == "draining")
    })
    .await;
    until("a holds everything b wrote", || {
        (0..50).all(|k| a.store.value(format!("b/{k}").as_bytes()).is_some())
    })
    .await;
    assert!(matches!(
        stopping.step().map(|s| s.label),
        Some("Flushing" | "Sending")
    ));
    a.cluster.close_all("test over");
    b.cluster.close_all("test over");
}

/// **Cut off**: a member whose every cluster has failed one attempt serves its
/// listener's requests itself, at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_member_that_reaches_no_cluster_serves_locally() {
    let dir = tempfile::TempDir::new().unwrap();
    let nowhere = closed_port().await;
    let m = member(nowhere, 2, dir.path()).await;
    let started = Instant::now();
    assert_eq!(m.connect(2).await, Link::CutOff);
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "{:?}",
        started.elapsed()
    );
    let local = ask_json(m.addr, "/v1/where", &[]).await;
    assert_eq!(local["socket"], false);
}
