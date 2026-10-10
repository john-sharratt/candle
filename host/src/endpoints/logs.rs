//! `WS /ws/logs` — the process's log lines, every tenant's, as the console's
//! log pane reads them.
//!
//! One JSON object per frame — `{ts, level, target, msg, tenant, seq}` — so a
//! reader is `JSON.parse(ev.data)` with no framing of its own to get wrong.
//!
//! # Neither lost nor repeated
//!
//! The subscription is taken *before* the ring is read, so a line written
//! between the two is delivered rather than lost; and every line carries the
//! bus's `seq`, so whatever the subscription then delivers that the replay
//! already covered is skipped rather than shown twice.

use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::State;
use axum::response::Response;
use tokio::sync::broadcast::error::RecvError;

use super::Shared;
use crate::logs::{LogBus, LogLine};

pub async fn logs(ws: WebSocketUpgrade, State(s): State<Arc<Shared>>) -> Response {
    let bus = Arc::clone(&s.log);
    ws.on_upgrade(move |socket| stream(socket, bus))
}

/// The lines a reader that subscribed and then took `replay` should show, in
/// order: the replay, then whatever comes after its last `seq`.
pub struct Cursor {
    last: u64,
}

impl Cursor {
    /// Start after `replay`.
    pub fn after(replay: &[LogLine]) -> Self {
        Self {
            last: replay.last().map_or(0, |l| l.seq),
        }
    }

    /// Whether `line` from the subscription is new to this reader.
    pub fn admit(&mut self, line: &LogLine) -> bool {
        if line.seq <= self.last {
            return false;
        }
        self.last = line.seq;
        true
    }
}

async fn stream(mut socket: WebSocket, bus: Arc<LogBus>) {
    let mut rx = bus.subscribe();
    let replay = bus.recent();
    let mut cursor = Cursor::after(&replay);
    for line in &replay {
        if send(&mut socket, line).await.is_err() {
            return;
        }
    }
    loop {
        tokio::select! {
            got = rx.recv() => match got {
                Ok(line) => {
                    if cursor.admit(&line) && send(&mut socket, &line).await.is_err() {
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

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;
    use crate::logs::BusWriter;

    /// **A line written between subscribing and reading the ring is shown
    /// once.** It is in both — the subscription saw it and so did the replay —
    /// and the cursor skips the subscription's copy.
    #[tokio::test]
    async fn a_line_in_both_the_replay_and_the_stream_is_shown_once() {
        let bus = LogBus::new(&["zend"]);
        let mut w = BusWriter::new(bus.clone());
        writeln!(w, "2026-01-01T00:00:01.0Z  INFO zend: before").unwrap();

        let mut rx = bus.subscribe();
        writeln!(w, "2026-01-01T00:00:02.0Z  INFO zend: between").unwrap();
        let replay = bus.recent();
        writeln!(w, "2026-01-01T00:00:03.0Z  INFO zend: after").unwrap();

        let mut cursor = Cursor::after(&replay);
        let mut shown: Vec<String> = replay.iter().map(|l| l.msg.clone()).collect();
        for _ in 0..2 {
            let line = rx.recv().await.unwrap();
            if cursor.admit(&line) {
                shown.push(line.msg);
            }
        }
        assert_eq!(shown, ["before", "between", "after"]);
    }

    /// An empty ring admits everything.
    #[test]
    fn an_empty_replay_admits_the_first_line() {
        let mut c = Cursor::after(&[]);
        let line = |seq| LogLine {
            ts: String::new(),
            level: "INFO".into(),
            target: String::new(),
            msg: String::new(),
            tenant: String::new(),
            seq,
        };
        assert!(c.admit(&line(1)));
        assert!(!c.admit(&line(1)));
        assert!(c.admit(&line(2)));
    }
}
