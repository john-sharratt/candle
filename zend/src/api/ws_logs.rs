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
        if socket.send(Message::Text(line)).await.is_err() {
            return;
        }
    }

    let mut rx = session.log.subscribe();
    loop {
        match rx.recv().await {
            Ok(line) => {
                if socket.send(Message::Text(line)).await.is_err() {
                    break;
                }
            }
            Err(RecvError::Lagged(n)) => {
                let notice = format!(
                    "0000-00-00T00:00:00Z  WARN zend::log_broadcast: [dropped {} log lines]",
                    n
                );
                if socket.send(Message::Text(notice)).await.is_err() {
                    break;
                }
            }
            Err(RecvError::Closed) => break,
        }
    }
}
