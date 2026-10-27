//! H6, the aggregate half (§14.2): an 8-member `Aggregate(Envelope)` with one
//! member that never answers in time. The gateway answers at its deadline plus
//! a little, with the slow member marked and the other seven present. Members
//! run in process over memory pipes on a paused clock, so the 3 s deadline
//! costs no wall time. Budget 1 s.

use std::net::IpAddr;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant as WallInstant};

use axum::body::{to_bytes, Body};
use axum::http::Request;
use axum::routing::get;
use axum::Router;
use cluster::dial::Connect;
use cluster::dispatch::{Dispatch, Merge, Route, RouteTable};
use cluster::frame::{BuildId, Capacity, MemberState, ModelId};
use cluster::gateway::Hub;
use cluster::handshake::GroupKey;
use cluster::member::{AppMember, Identity};
use cluster::pipe::memory::{pair, MemoryRx, MemoryTx};
use cluster::session::VirtualConn;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper_util::rt::TokioIo;
use serde_json::{json, Value};
use tokio::time::{sleep, Instant};
use tower::{Service, ServiceExt};
use web::cluster::Gateway;
use web::{Builder, Config};

const MEMBERS: u64 = 8;
const SLOW: u64 = 8;
const BUDGET: Duration = Duration::from_secs(1);

fn key() -> GroupKey {
    GroupKey::from_bytes([6; 32])
}

struct ToHub(Arc<Hub>);

impl Connect for ToHub {
    type Tx = MemoryTx;
    type Rx = MemoryRx;
    async fn connect(&mut self) -> Result<(MemoryTx, MemoryRx), String> {
        let (m, g) = pair(1 << 20);
        let hub = self.0.clone();
        tokio::spawn(async move {
            hub.serve("zend", IpAddr::from([127, 0, 0, 1]), g.tx, g.rx, |_| {})
                .await;
        });
        Ok((m.tx, m.rx))
    }
}

fn serve_stream(conn: VirtualConn, router: Router) {
    tokio::spawn(async move {
        let svc = service_fn(move |req: Request<Incoming>| {
            let mut router = router.clone();
            async move { router.call(req.map(Body::new)).await }
        });
        let _ = hyper::server::conn::http1::Builder::new()
            .serve_connection(TokioIo::new(conn), svc)
            .await;
    });
}

fn member(hub: &Arc<Hub>, node: u64) -> Arc<AppMember> {
    let app = AppMember::new(
        Identity {
            app: "zend".into(),
            node,
            host: format!("m{node}"),
            build: BuildId {
                count: 1,
                hash: "h".into(),
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
            leaseholder: true,
        },
        key(),
        vec!["mem".into()],
        RouteTable::new(vec![Route::new(
            &[],
            "/v1/status",
            Dispatch::Aggregate(Merge::Envelope),
        )])
        .unwrap(),
        Box::new(|| Capacity {
            state: MemberState::Ready,
            power: 100,
            idle: true,
            slots_free: 1,
            decodes: 0,
            prefill_backlog: 0,
            vram_free_mib: 0,
            cores: 1,
            ..Capacity::default()
        }),
        None,
        WallInstant::now(),
    );
    let router = Router::new().route(
        "/v1/status",
        get(move || async move {
            if node == SLOW {
                sleep(Duration::from_secs(60)).await;
            }
            axum::Json(json!({"node": node}))
        }),
    );
    tokio::spawn(app.clone().run(0, ToHub(hub.clone()), move |conn| {
        serve_stream(conn, router.clone())
    }));
    app
}

#[tokio::test(start_paused = true)]
async fn h6_an_eight_member_aggregate_answers_at_its_deadline() {
    let wall = WallInstant::now();
    let hub = Hub::new(key(), "gw-h6");
    let cfg = Config::from_yaml(
        r#"
cluster: {group_key_file: "unused.key"}
sites:
  - name: zend
    default: true
    api:
      - {prefix: /, upstream: "http://127.0.0.1:9", cluster: zend}
"#,
        Path::new(env!("CARGO_MANIFEST_DIR")),
    )
    .unwrap();
    let router = Builder::new(cfg)
        .cluster(Gateway::with_hub(hub.clone()))
        .router();
    let _members: Vec<_> = (1..=MEMBERS).map(|n| member(&hub, n)).collect();
    let ready = |hub: &Hub| {
        let status = hub.status(0);
        status.members.iter().filter(|m| m.state == "ready").count()
    };
    while ready(&hub) < MEMBERS as usize {
        sleep(Duration::from_millis(10)).await;
    }

    let asked = Instant::now();
    let res = router
        .oneshot(
            Request::get("/v1/status")
                .header("host", "code.test")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let body: Value =
        serde_json::from_slice(&to_bytes(res.into_body(), 1 << 20).await.unwrap()).unwrap();
    let took = asked.elapsed();
    assert!(
        took >= Duration::from_secs(3) && took <= Duration::from_millis(3_100),
        "answered after {took:?} of virtual time"
    );
    let members = body["members"].as_array().unwrap();
    assert_eq!(members.len(), MEMBERS as usize);
    for m in members {
        let node = u64::from_str_radix(m["node"].as_str().unwrap(), 16).unwrap();
        if node == SLOW {
            assert_eq!(
                (m["status"].clone(), m["error"].clone()),
                (json!(504), json!("deadline"))
            );
        } else {
            assert_eq!(m["status"], 200);
            assert_eq!(m["body"]["node"], node);
        }
    }
    let elapsed = wall.elapsed();
    println!("H6 (aggregate): {MEMBERS} members, one late, answered in {took:?} virtual, {elapsed:?} wall");
    assert!(elapsed <= BUDGET, "took {elapsed:?}");
}
