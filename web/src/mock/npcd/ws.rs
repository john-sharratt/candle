//! The two websocket streams: `/ws/logs` and `/ws/events`.
//!
//! Both are push. A console pane that polls has to choose between a stale view
//! and a request every few seconds against a daemon that is trying to decode;
//! a socket that says nothing until something happens costs neither. The
//! polling versions of these panes are gone, not retained alongside.
//!
//! Every frame is one JSON object on its own line, so a reader is
//! `JSON.parse(ev.data)` with no framing of its own to get wrong. The shapes
//! follow `docs/npc_api_gui_design.md` §§ on the logs pane and the roster.
//!
//! The bodies here are mock: the lines and events are synthesised on a timer
//! rather than tapped off a running engine. The wire contract is the real one
//! and does not change when the engine lands behind it.

use std::time::Duration;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use serde_json::{json, Value};

use super::schema;

/// Cadence of synthesised traffic. Slow enough to read, fast enough that a pane
/// left open visibly moves.
const LOG_PERIOD: Duration = Duration::from_millis(1_400);
const EVENT_PERIOD: Duration = Duration::from_millis(2_600);

pub async fn logs(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(|socket| stream(socket, LOG_PERIOD, seed_logs(), log_line))
}

pub async fn events(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(|socket| stream(socket, EVENT_PERIOD, Vec::new(), event))
}

/// Replay the backlog, then emit one frame per tick until the peer goes away.
///
/// The backlog matters: a pane opened at minute ten would otherwise start
/// empty and stay nearly empty, which reads as "nothing is happening" rather
/// than "you just got here". Sending it over the same socket also removes the
/// seed-then-subscribe race, where a line arriving between the two is either
/// lost or shown twice.
async fn stream(
    mut socket: WebSocket,
    period: Duration,
    backlog: Vec<Value>,
    next: fn(u64) -> Value,
) {
    for item in backlog {
        if socket.send(Message::Text(item.to_string())).await.is_err() {
            return;
        }
    }

    let mut tick = tokio::time::interval(period);
    let mut n: u64 = 0;
    loop {
        tokio::select! {
            _ = tick.tick() => {
                n += 1;
                if socket.send(Message::Text(next(n).to_string())).await.is_err() {
                    return;
                }
            }
            // Reading is what notices a close, and a client that sends nothing
            // still needs its pings answered — which axum does inside `recv`.
            incoming = socket.recv() => match incoming {
                Some(Ok(_)) => {}
                _ => return,
            },
        }
    }
}

fn seed_logs() -> Vec<Value> {
    schema::log_lines()["lines"]
        .as_array()
        .cloned()
        .unwrap_or_default()
}

/// One synthesised log line. The rotation covers every level the pane filters
/// on, so the level selector has something to do without waiting for a real
/// error to happen.
fn log_line(n: u64) -> Value {
    const LINES: [(&str, &str, &str); 8] = [
        ("DEBUG", "tick", "npc …4281 gate 0.42 → tick scheduled"),
        (
            "TRACE",
            "projection",
            "gather: 31 turns, 14880/16000 tok, 4 dropped (budget)",
        ),
        ("INFO", "narrator", "rendered a_88211 in 197ms"),
        (
            "DEBUG",
            "persistence",
            "checkpoint written · 38 records · 764 KiB",
        ),
        ("WARN", "monitor", "npc …4283 overlap 0.39 → band=fixated"),
        ("INFO", "scheduler", "batch composition: 3 npcs / decode"),
        ("DEBUG", "tick", "npc …4285 idle → heartbeat deferred 90s"),
        (
            "ERROR",
            "image",
            "job_img_2 abandoned: relief exhausted before slot claim",
        ),
    ];
    let (level, target, msg) = LINES[(n as usize) % LINES.len()];
    json!({ "ts": clock(n), "level": level, "target": target, "msg": msg })
}

/// One daemon-wide event. The roster subscribes to these instead of re-listing:
/// each frame names the NPC it concerns and carries only what changed.
fn event(n: u64) -> Value {
    const NPCS: [&str; 4] = [
        "10237749914772934281",
        "10237749914772934283",
        "10237749914772934284",
        "10237749914772934286",
    ];
    let id = NPCS[(n as usize) % NPCS.len()];
    match n % 4 {
        0 => {
            json!({ "type": "npc.tick", "npc_id": id, "pending_events": n % 7, "state": "ticking" })
        }
        1 => json!({ "type": "npc.state", "npc_id": id, "state": "active" }),
        2 => json!({ "type": "npc.monitor", "npc_id": id, "overlap": 0.11 + (n % 5) as f64 * 0.04,
                     "band": if n % 5 == 4 { "fixated" } else { "healthy" } }),
        _ => json!({ "type": "npc.state", "npc_id": id, "state": "idle" }),
    }
}

/// A wall clock for the frame's `ts`. Mock lines carry a timestamp because the
/// pane renders one; it advances with the stream so the column is monotonic.
fn clock(n: u64) -> String {
    let base = 6 * 3600 + 14 * 60 + 2;
    let t = base + n * 3;
    format!("{:02}:{:02}:{:02}", t / 3600 % 24, t / 60 % 60, t % 60)
}
