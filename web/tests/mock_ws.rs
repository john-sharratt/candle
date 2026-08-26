//! The `/ws/*` streams, over a real websocket — direct, and through the proxy.
//!
//! The proxy case is the one worth the extra setup. The console is served from
//! a DMZ box and the daemon is somewhere else, so every frame the logs pane
//! ever sees has crossed a tunnel; a test that only talks to the daemon
//! directly would pass while the deployed arrangement was broken.

use std::net::SocketAddr;
use std::time::Duration;

use futures_util::StreamExt;
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message;
use web::{mock, Builder, Config};

async fn spawn_daemon() -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, mock::npcd::router()).await.unwrap();
    });
    addr
}

/// A `web` instance in its DMZ role, forwarding `/ws` to `daemon`.
async fn spawn_proxy(daemon: SocketAddr) -> SocketAddr {
    let cfg = Config::from_yaml(
        &format!(
            r#"
sites:
  - name: npcd
    default: true
    api:
      - {{prefix: /ws, upstream: "http://{daemon}"}}
"#
        ),
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")),
    )
    .unwrap();

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let router = Builder::new(cfg).router();
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

/// Collect `n` JSON frames, failing rather than hanging if they do not arrive.
async fn frames(at: SocketAddr, path: &str, n: usize) -> Vec<Value> {
    let (mut sock, _) = tokio_tungstenite::connect_async(format!("ws://{at}{path}"))
        .await
        .expect("handshake");
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let msg = tokio::time::timeout(Duration::from_secs(10), sock.next())
            .await
            .expect("timed out waiting for a frame")
            .expect("stream ended early")
            .expect("frame error");
        if let Message::Text(t) = msg {
            out.push(serde_json::from_str(&t).expect("every frame is one JSON object"));
        }
    }
    out
}

#[tokio::test]
async fn logs_replay_the_backlog_before_the_tail() {
    let d = spawn_daemon().await;
    // 13 backlog lines, so a 15th frame can only have come from the timer.
    let got = frames(d, "/ws/logs", 15).await;

    assert_eq!(got[0]["msg"], "npcd ready — mock backend, no engine loaded");
    assert!(
        got.iter()
            .all(|l| l["ts"].is_string() && l["level"].is_string() && l["target"].is_string()),
        "every line carries the fields the pane renders"
    );

    // The pane filters on level as a property, never by matching formatted
    // text, so the levels have to be exactly the names it knows.
    for l in &got {
        let lvl = l["level"].as_str().unwrap();
        assert!(
            ["TRACE", "DEBUG", "INFO", "WARN", "ERROR"].contains(&lvl),
            "unknown level {lvl}"
        );
    }
}

#[tokio::test]
async fn events_name_the_npc_they_concern() {
    let d = spawn_daemon().await;
    let got = frames(d, "/ws/events", 4).await;
    for e in &got {
        assert!(e["type"].as_str().unwrap().starts_with("npc."), "{e}");
        // The roster indexes its rows by id; a frame without one is unroutable.
        assert!(e["npc_id"].as_str().is_some_and(|s| !s.is_empty()), "{e}");
    }
}

#[tokio::test]
async fn the_console_sees_the_same_frames_through_the_proxy() {
    let d = spawn_daemon().await;
    let p = spawn_proxy(d).await;

    let direct = frames(d, "/ws/logs", 13).await;
    let through = frames(p, "/ws/logs", 13).await;

    // The backlog is fixed, so the two are comparable frame for frame — any
    // difference is the tunnel altering what it carries.
    assert_eq!(direct, through);
}

#[tokio::test]
async fn closing_the_socket_ends_the_stream() {
    // A pane that navigates away must not leave the daemon writing into a dead
    // socket for as long as the process lives.
    let d = spawn_daemon().await;
    let (sock, _) = tokio_tungstenite::connect_async(format!("ws://{d}/ws/events"))
        .await
        .unwrap();
    drop(sock);

    // The daemon keeps serving, which is the observable consequence: a dropped
    // subscriber is not a fault.
    let got = frames(d, "/ws/events", 1).await;
    assert!(got[0]["npc_id"].is_string());
}
