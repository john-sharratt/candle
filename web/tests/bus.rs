//! The gateway's bus over real sockets (`docs/mcp_file_tools.md` §5): a
//! service subscribes over `/gateway/events` with its key, is refused what its
//! key does not name, and answers a gate request on its reply topic.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use cluster::gateway::Hub;
use cluster::handshake::GroupKey;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio::time::timeout;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{connect_async, MaybeTlsStream, WebSocketStream};
use web::bus;
use web::cluster::bearer_key::BearerKey;
use web::cluster::service_key::ServiceKey;
use web::cluster::Gateway;
use web::config::{ServiceKeyConfig, ServiceTopics};

type Ws = WebSocketStream<MaybeTlsStream<TcpStream>>;

const KEY: &str = "5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e";

fn gateway() -> Arc<Gateway> {
    let yenbot = ServiceKey::new(
        BearerKey::from_hex(KEY).unwrap(),
        &ServiceKeyConfig {
            name: "yenbot".into(),
            key_file: "unused".into(),
            routes: vec![],
            topics: ServiceTopics {
                subscribe: vec!["repo/yenbot".into(), "gate/yenbot".into()],
                publish: vec!["gate/yenbot/result/*".into()],
            },
        },
    );
    Gateway::with_hub_and_services(
        Hub::new(GroupKey::from_bytes([5; 32]), "bus-test"),
        None,
        None,
        vec![yenbot],
    )
}

async fn spawn(gw: Arc<Gateway>) -> SocketAddr {
    let router: Router = bus::http::router(gw, vec!["127.0.0.1".parse().unwrap()]);
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

async fn connect(addr: SocketAddr, key: Option<&str>) -> Result<Ws, String> {
    let mut req = format!("ws://{addr}/gateway/events")
        .into_client_request()
        .unwrap();
    if let Some(k) = key {
        req.headers_mut()
            .insert("authorization", format!("Bearer {k}").parse().unwrap());
    }
    connect_async(req)
        .await
        .map(|(ws, _)| ws)
        .map_err(|e| e.to_string())
}

async fn send(ws: &mut Ws, frame: Value) {
    ws.send(Message::Text(frame.to_string())).await.unwrap();
}

async fn next(ws: &mut Ws) -> Value {
    loop {
        let frame = timeout(Duration::from_secs(5), ws.next())
            .await
            .expect("a frame within 5 s")
            .expect("the socket is open")
            .unwrap();
        if let Message::Text(t) = frame {
            return serde_json::from_str(&t).unwrap();
        }
    }
}

/// Wait until the bus reaches `n` subscribers on `topic` — the socket's
/// subscribe frame is applied after the upgrade, asynchronously.
async fn until_subscribed(gw: &Gateway, topic: &str) {
    for _ in 0..500 {
        if gw.bus.publish(topic, json!({"probe": true})) > 0 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("nothing subscribed to {topic}");
}

/// **A socket without a service key is refused at the upgrade.**
#[tokio::test]
async fn the_socket_takes_a_service_key() {
    let addr = spawn(gateway()).await;
    assert!(connect(addr, None).await.is_err());
    assert!(connect(addr, Some(&"ab".repeat(32))).await.is_err());
    assert!(connect(addr, Some(KEY)).await.is_ok());
}

/// **A service hears its own topics, and is refused the rest by name.**
#[tokio::test]
async fn a_service_hears_its_topics_and_is_refused_the_rest() {
    let gw = gateway();
    let addr = spawn(gw.clone()).await;
    let mut ws = connect(addr, Some(KEY)).await.unwrap();
    send(
        &mut ws,
        json!({"op": "subscribe", "topics": ["repo/yenbot", "repo/candle"]}),
    )
    .await;
    assert_eq!(
        next(&mut ws).await,
        json!({"op": "refused", "topic": "repo/candle", "why": "not subscribable by service:yenbot"})
    );
    until_subscribed(&gw, "repo/yenbot").await;
    let probe = next(&mut ws).await;
    assert_eq!(probe["topic"], "repo/yenbot");
    gw.bus.publish(
        "repo/yenbot",
        json!({"commit": "4f1c", "paths": ["apps/a/app.yaml"]}),
    );
    gw.bus.publish("repo/candle", json!({"commit": "nope"}));
    let mut heard = next(&mut ws).await;
    while heard["payload"]["probe"] == json!(true) {
        heard = next(&mut ws).await;
    }
    assert_eq!(heard["op"], "message");
    assert_eq!(heard["topic"], "repo/yenbot");
    assert_eq!(
        heard["payload"],
        json!({"commit": "4f1c", "paths": ["apps/a/app.yaml"]})
    );
    send(
        &mut ws,
        json!({"op": "publish", "topic": "repo/yenbot", "payload": {}}),
    )
    .await;
    assert_eq!(
        next(&mut ws).await,
        json!({"op": "refused", "topic": "repo/yenbot", "why": "not publishable by service:yenbot"})
    );
}

/// **The gate's round trip**: a request published on `gate/yenbot` reaches the
/// service, which answers on the reply topic, and the request has its answer.
#[tokio::test]
async fn a_service_answers_a_gate_request() {
    let gw = gateway();
    let addr = spawn(gw.clone()).await;
    let mut ws = connect(addr, Some(KEY)).await.unwrap();
    send(
        &mut ws,
        json!({"op": "subscribe", "topics": ["gate/yenbot"]}),
    )
    .await;
    until_subscribed(&gw, "gate/yenbot").await;
    let asking = {
        let gw = gw.clone();
        tokio::spawn(async move {
            gw.bus
                .request(
                    "gate/yenbot",
                    json!({"id": "01JD", "commit": "4f1c"}),
                    "gate/yenbot/result/01JD",
                    Duration::from_secs(5),
                )
                .await
        })
    };
    let mut ask = next(&mut ws).await;
    while ask["payload"]["probe"] == json!(true) {
        ask = next(&mut ws).await;
    }
    assert_eq!(ask["payload"]["id"], "01JD");
    send(
        &mut ws,
        json!({"op": "publish", "topic": "gate/yenbot/result/01JD",
               "payload": {"ok": false, "reason": "apps/a/pages/home.js:14 uses a literal colour"}}),
    )
    .await;
    assert_eq!(
        asking.await.unwrap(),
        Ok(json!({"ok": false, "reason": "apps/a/pages/home.js:14 uses a literal colour"}))
    );
}

/// **A member on its bus stream needs no key and has every topic**; the same
/// request on the gateway's own router, which is where a member's listener
/// forwards its clients, is refused.
#[tokio::test]
async fn a_member_publishes_on_its_bus_stream_alone() {
    let gw = gateway();
    let members = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let on_stream = members.local_addr().unwrap();
    let router = bus::http::member_router(gw.clone());
    tokio::spawn(async move {
        axum::serve(
            members,
            router.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap();
    });
    let forwarded = spawn(gw).await;
    let client = reqwest::Client::new();
    let post = |addr: SocketAddr| {
        client
            .post(format!("http://{addr}/gateway/events/publish"))
            .json(&json!({"topic": "gate/yenbot/result/7", "payload": {"ok": true}}))
            .send()
    };
    assert_eq!(post(on_stream).await.unwrap().status(), 200);
    assert_eq!(post(forwarded).await.unwrap().status(), 401);
}

/// **Publishing over HTTP takes a key, and only the key's topics.**
#[tokio::test]
async fn publishing_over_http_is_the_keys() {
    let addr = spawn(gateway()).await;
    let client = reqwest::Client::new();
    let url = format!("http://{addr}/gateway/events/publish");
    let post = |topic: &'static str, key: Option<&'static str>| {
        let mut req = client
            .post(&url)
            .json(&json!({"topic": topic, "payload": {"ok": true}}));
        if let Some(k) = key {
            req = req.bearer_auth(k);
        }
        req.send()
    };
    let ok = post("gate/yenbot/result/7", Some(KEY)).await.unwrap();
    assert_eq!(ok.status(), 200);
    assert_eq!(ok.json::<Value>().await.unwrap(), json!({"reached": 0}));
    assert_eq!(post("repo/yenbot", Some(KEY)).await.unwrap().status(), 403);
    assert_eq!(
        post("gate/yenbot/result/7", None).await.unwrap().status(),
        401
    );
}
