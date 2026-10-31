//! A fake of Discord's gateway, on a port of the test's own.
//!
//! Each connection it accepts plays the next [`Script`]: `HELLO`, then to an
//! `IDENTIFY` either `READY` or a close, and to a `RESUME` `RESUMED`. It
//! answers heartbeats, keeps every payload a shard sent it, and delivers
//! what [`FakeGateway::dispatch`] is given to the connection that is open.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode;
use tokio_tungstenite::tungstenite::protocol::CloseFrame;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{accept_async, WebSocketStream};

/// The bot's user, as `READY` names it.
pub const BOT_USER: u64 = 222;

/// The session `READY` opens.
pub const SESSION: &str = "fake-session";

/// What one connection does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    /// `READY` to an `IDENTIFY`, `RESUMED` to a `RESUME`, and held open.
    Serve,
    /// `READY`, then the connection dropped with no close.
    DropAfterReady,
    /// Closed with this code in answer to an `IDENTIFY`.
    Close(u16),
}

/// A running fake gateway.
pub struct FakeGateway {
    /// Its base, as a shard's [`super::shard::Config::gateway`].
    pub url: String,
    received: Arc<Mutex<Vec<Value>>>,
    open: Arc<Mutex<Option<mpsc::UnboundedSender<Value>>>>,
    accepted: Arc<AtomicUsize>,
    seq: Arc<AtomicU64>,
}

impl FakeGateway {
    /// Every payload a shard sent, over every connection, in order.
    pub fn received(&self) -> Vec<Value> {
        self.received.lock().unwrap().clone()
    }

    /// How many connections it has accepted.
    pub fn accepted(&self) -> usize {
        self.accepted.load(Ordering::SeqCst)
    }

    /// Send `message` to the open connection as a `MESSAGE_CREATE`.
    pub fn dispatch(&self, message: Value) {
        let s = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
        let frame = json!({ "op": 0, "t": "MESSAGE_CREATE", "s": s, "d": message });
        self.open
            .lock()
            .unwrap()
            .as_ref()
            .expect("a connection is open")
            .send(frame)
            .unwrap();
    }
}

/// A fake playing `scripts`, one per connection, the last for every
/// connection after them; with `heartbeat_ms` the interval its `HELLO` asks
/// for.
pub async fn start(scripts: Vec<Script>, heartbeat_ms: u64) -> FakeGateway {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("ws://{}", listener.local_addr().unwrap());
    let fake = FakeGateway {
        url: url.clone(),
        received: Arc::new(Mutex::new(Vec::new())),
        open: Arc::new(Mutex::new(None)),
        accepted: Arc::new(AtomicUsize::new(0)),
        seq: Arc::new(AtomicU64::new(0)),
    };
    let held = Held {
        url,
        heartbeat_ms,
        received: Arc::clone(&fake.received),
        open: Arc::clone(&fake.open),
        seq: Arc::clone(&fake.seq),
    };
    let accepted = Arc::clone(&fake.accepted);
    tokio::spawn(async move {
        loop {
            let Ok((tcp, _)) = listener.accept().await else {
                return;
            };
            let n = accepted.fetch_add(1, Ordering::SeqCst);
            let script = scripts
                .get(n)
                .or(scripts.last())
                .copied()
                .unwrap_or(Script::Serve);
            let held = held.clone();
            tokio::spawn(async move { held.play(tcp, script).await });
        }
    });
    fake
}

#[derive(Clone)]
struct Held {
    url: String,
    heartbeat_ms: u64,
    received: Arc<Mutex<Vec<Value>>>,
    open: Arc<Mutex<Option<mpsc::UnboundedSender<Value>>>>,
    seq: Arc<AtomicU64>,
}

impl Held {
    async fn play(self, tcp: TcpStream, script: Script) {
        let Ok(mut ws) = accept_async(tcp).await else {
            return;
        };
        let (tx, mut dispatches) = mpsc::unbounded_channel();
        *self.open.lock().unwrap() = Some(tx);
        let hello = json!({ "op": 10, "d": { "heartbeat_interval": self.heartbeat_ms } });
        if send(&mut ws, &hello).await.is_err() {
            return;
        }
        loop {
            tokio::select! {
                frame = dispatches.recv() => match frame {
                    Some(frame) => {
                        if send(&mut ws, &frame).await.is_err() {
                            return;
                        }
                    }
                    None => return,
                },
                msg = ws.next() => {
                    let Some(Ok(Message::Text(text))) = msg else {
                        return;
                    };
                    let Ok(payload) = serde_json::from_str::<Value>(&text) else {
                        continue;
                    };
                    self.received.lock().unwrap().push(payload.clone());
                    match payload["op"].as_u64() {
                        Some(1) => {
                            if send(&mut ws, &json!({ "op": 11 })).await.is_err() {
                                return;
                            }
                        }
                        Some(2) => match script {
                            Script::Close(code) => {
                                let _ = ws
                                    .close(Some(CloseFrame {
                                        code: CloseCode::from(code),
                                        reason: "scripted".into(),
                                    }))
                                    .await;
                                return;
                            }
                            Script::Serve | Script::DropAfterReady => {
                                if send(&mut ws, &self.ready()).await.is_err() {
                                    return;
                                }
                                if script == Script::DropAfterReady {
                                    return;
                                }
                            }
                        },
                        Some(6) => {
                            let s = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
                            let resumed = json!({ "op": 0, "t": "RESUMED", "s": s, "d": {} });
                            if send(&mut ws, &resumed).await.is_err() {
                                return;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    fn ready(&self) -> Value {
        let s = self.seq.fetch_add(1, Ordering::SeqCst) + 1;
        json!({
            "op": 0,
            "t": "READY",
            "s": s,
            "d": {
                "v": 10,
                "user": { "id": BOT_USER.to_string(), "username": "Zen", "bot": true },
                "session_id": SESSION,
                "resume_gateway_url": self.url,
                "guilds": [],
            },
        })
    }
}

async fn send(ws: &mut WebSocketStream<TcpStream>, payload: &Value) -> Result<(), ()> {
    ws.send(Message::Text(payload.to_string()))
        .await
        .map_err(|_| ())
}
