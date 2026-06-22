use std::sync::Arc;

use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    response::Response,
};
use tokio::sync::broadcast::error::RecvError;

use crate::session::ZendSession;

pub async fn handler(ws: WebSocketUpgrade, State(session): State<Arc<ZendSession>>) -> Response {
    ws.on_upgrade(move |socket| serve(socket, session))
}

async fn serve(mut socket: WebSocket, session: Arc<ZendSession>) {
    // Replay recent history so the pane isn't blank on first connect.
    for line in session.log.recent() {
        if socket.send(Message::Text(frame(&line))).await.is_err() {
            return;
        }
    }

    let mut rx = session.log.subscribe();
    loop {
        match rx.recv().await {
            Ok(line) => {
                if socket.send(Message::Text(frame(&line))).await.is_err() {
                    break;
                }
            }
            Err(RecvError::Lagged(n)) => {
                let notice = format!(
                    "0000-00-00T00:00:00Z  WARN zend::log_broadcast: dropped {} log lines",
                    n
                );
                if socket.send(Message::Text(frame(&notice))).await.is_err() {
                    break;
                }
            }
            Err(RecvError::Closed) => break,
        }
    }
}

/// Frame a formatted log line as the structured JSON the UI consumes
/// (`{ ts, level, target, msg }` — docs/zend_ui_redesign.md §2.6). Falls back to
/// the raw line if serialization somehow fails.
fn frame(line: &str) -> String {
    serde_json::to_string(&crate::log_line::parse(line)).unwrap_or_else(|_| line.to_string())
}
