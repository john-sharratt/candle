//! `GET /gateway/events`: the bus as a WebSocket (`docs/mcp_file_tools.md`
//! §5.2), for a subscriber. Frames are JSON:
//!
//! ```json
//! {"op": "subscribe", "topics": ["repo/yenbot"]}
//! {"op": "publish",   "topic": "gate/yenbot/result/01JD", "payload": {…}}
//! {"op": "message",   "topic": "repo/yenbot", "payload": {…}, "at_ms": 1793931066831}
//! {"op": "refused",   "topic": "repo/candle", "why": "not subscribable by service:yenbot"}
//! ```
//!
//! The first two go up, the last two come down. A topic the caller may not
//! use is refused by name and the rest of the frame stands.

use std::net::SocketAddr;
use std::ptr;

use axum::extract::ws::{Message as Frame, WebSocket, WebSocketUpgrade};
use axum::extract::{ConnectInfo, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::{json, Value};

use super::http::{caller, BusState, Caller};
use super::topic;

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
enum Up {
    Subscribe {
        topics: Vec<String>,
    },
    Publish {
        topic: String,
        #[serde(default)]
        payload: Value,
    },
}

/// Whose socket it is, kept once the upgrade is done.
#[derive(Clone)]
enum Who {
    Member,
    Service(usize),
}

pub async fn upgrade(
    State(s): State<BusState>,
    peer: Option<ConnectInfo<SocketAddr>>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    let who = match caller(&s, &headers, peer.map(|p| p.0)) {
        Some(Caller::Member) => Who::Member,
        Some(Caller::Service(key)) => {
            match s.gateway.service_keys.iter().position(|k| ptr::eq(k, key)) {
                Some(i) => Who::Service(i),
                None => return StatusCode::UNAUTHORIZED.into_response(),
            }
        }
        None => return StatusCode::UNAUTHORIZED.into_response(),
    };
    ws.on_upgrade(move |socket| serve(s, who, socket))
}

fn as_caller<'a>(s: &'a BusState, who: &Who) -> Caller<'a> {
    match who {
        Who::Member => Caller::Member,
        Who::Service(i) => Caller::Service(&s.gateway.service_keys[*i]),
    }
}

fn name(s: &BusState, who: &Who) -> String {
    match who {
        Who::Member => "a member".into(),
        Who::Service(i) => s.gateway.service_keys[*i].identity().sub.clone(),
    }
}

fn refusal(topic: &str, why: String) -> Frame {
    Frame::Text(json!({"op": "refused", "topic": topic, "why": why}).to_string())
}

async fn serve(s: BusState, who: Who, mut socket: WebSocket) {
    let bus = &s.gateway.bus;
    let mut sub = bus.subscribe(Vec::new());
    loop {
        tokio::select! {
            message = sub.messages.recv() => {
                // `None`: the bus dropped this subscriber for falling behind.
                let Some(m) = message else { break };
                let frame = json!({"op": "message", "topic": m.topic, "payload": m.payload, "at_ms": m.at_ms});
                if socket.send(Frame::Text(frame.to_string())).await.is_err() {
                    break;
                }
            }
            frame = socket.recv() => {
                let text = match frame {
                    Some(Ok(Frame::Text(t))) => t,
                    Some(Ok(Frame::Ping(_) | Frame::Pong(_) | Frame::Binary(_))) => continue,
                    Some(Ok(Frame::Close(_)) | Err(_)) | None => break,
                };
                for reply in handle(&s, &who, sub.id, &text) {
                    if socket.send(reply).await.is_err() {
                        bus.unsubscribe(sub.id);
                        return;
                    }
                }
            }
        }
    }
    bus.unsubscribe(sub.id);
}

/// What one frame up does, and the frames it answers with.
fn handle(s: &BusState, who: &Who, id: u64, text: &str) -> Vec<Frame> {
    let caller = as_caller(s, who);
    let up: Up = match serde_json::from_str(text) {
        Ok(up) => up,
        Err(e) => return vec![refusal("", format!("not a frame: {e}"))],
    };
    match up {
        Up::Subscribe { topics } => {
            let (ok, no): (Vec<String>, Vec<String>) =
                topics.into_iter().partition(|t| caller.may_subscribe(t));
            s.gateway.bus.extend(id, &ok);
            no.iter()
                .map(|t| refusal(t, format!("not subscribable by {}", name(s, who))))
                .collect()
        }
        Up::Publish { topic, payload } => {
            if !topic::valid(&topic) {
                return vec![refusal(&topic, "not a topic".into())];
            }
            if !caller.may_publish(&topic) {
                return vec![refusal(
                    &topic,
                    format!("not publishable by {}", name(s, who)),
                )];
            }
            s.gateway.bus.publish(&topic, payload);
            Vec::new()
        }
    }
}
