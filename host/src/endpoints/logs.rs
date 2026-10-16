//! `WS /ws/logs` — the process's log lines, every tenant's, as the console's
//! log pane reads them.
//!
//! One JSON object per frame — `{ts, level, target, msg, tenant, seq}` — so a
//! reader is `JSON.parse(ev.data)` with no framing of its own to get wrong.
//!
//! # Neither lost nor repeated
//!
//! The replay is the log file's last [`REPLAY_BYTES`], read back from disk, and
//! the subscription is taken under the file's lock in the same moment
//! ([`LogBus::replay`]): a line written before it is in the replay, one written
//! after is in the stream, and none is in both.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use tokio::sync::broadcast::error::RecvError;

use super::Shared;
use crate::logs::{LogBus, LogLine};

/// How much of the file a reader is shown before the live tail: enough to see
/// what happened just before you looked.
pub const REPLAY_BYTES: u64 = 256 * 1024;

pub async fn logs(ws: WebSocketUpgrade, State(s): State<Arc<Shared>>) -> Response {
    let bus = Arc::clone(&s.log);
    ws.on_upgrade(move |socket| stream(socket, bus))
}

async fn stream(mut socket: WebSocket, bus: Arc<LogBus>) {
    let (mut rx, replay) = bus.replay(REPLAY_BYTES);
    for line in &replay {
        if send(&mut socket, line).await.is_err() {
            return;
        }
    }
    loop {
        tokio::select! {
            got = rx.recv() => match got {
                Ok(line) => {
                    if send(&mut socket, &line).await.is_err() {
                        return;
                    }
                }
                // The console fell behind — a paused pane, a slow tab, a burst.
                // Say so in the stream rather than silently closing the gap: a
                // reader needs to know the sequence in front of them is not
                // complete.
                Err(RecvError::Lagged(n)) => {
                    bus.note_dropped(n);
                    let notice = LogLine {
                        ts: String::new(),
                        level: "WARN".to_owned(),
                        target: "host::logs".to_owned(),
                        msg: format!("{n} lines dropped — the console fell behind"),
                        tenant: String::new(),
                        seq: 0,
                    };
                    if send(&mut socket, &notice).await.is_err() {
                        return;
                    }
                }
                Err(RecvError::Closed) => return,
            },
            // Reading is what notices a close, and a client that sends nothing
            // still needs its pings answered — which axum does inside `recv`.
            incoming = socket.recv() => match incoming {
                Some(Ok(_)) => {}
                _ => return,
            },
        }
    }
}

async fn send(socket: &mut WebSocket, line: &LogLine) -> Result<(), ()> {
    let text = serde_json::to_string(line).map_err(|_| ())?;
    socket.send(Message::Text(text)).await.map_err(|_| ())
}
