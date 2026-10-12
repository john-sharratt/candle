//! One job's output, followed live for a browser (`GET
//! /v1/agents/{node}/jobs/{id}/stream`).
//!
//! The log is read from the agent a chunk at a time — straight on while a
//! read comes back full, every [`FOLLOW_EVERY`] once caught up — and sent on as
//! text, until the job has ended and its log has been read to the end. It is
//! read only while the browser holds the stream: the page opens it when an
//! agent's row is expanded and closes it when the row is collapsed, so an agent
//! nobody is watching costs nothing.
//!
//! Events, each `data` a JSON object:
//!
//! | event | data | |
//! |---|---|---|
//! | `skipped` | `{bytes}` | the head of a log longer than [`BACKLOG`], not sent |
//! | `chunk` | `{text}` | output, in order |
//! | `end` | `{size}` | the job has ended and all of its log was sent |
//! | `error` | `{message}` | the agent could not be read; the stream ends |

use std::convert::Infallible;
use std::time::Duration;

use agent::client::{Agents, At};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::json;
use tokio::sync::mpsc;
use tokio::time::sleep;
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use super::utf8::Utf8Stream;

/// How often a log that has been read to its end is read again.
pub const FOLLOW_EVERY: Duration = Duration::from_millis(300);

/// The most of a log's head a new reader is sent: a longer log is joined at
/// its last [`BACKLOG`] bytes.
pub const BACKLOG: u64 = 512 * 1024;

/// Where a reader joining a log `size` bytes long begins.
pub fn join_at(size: u64) -> u64 {
    size.saturating_sub(BACKLOG)
}

fn event(name: &str, data: serde_json::Value) -> Event {
    Event::default().event(name).data(data.to_string())
}

/// Follow job `id` on agent `at` into `tx` until it ends, the agent cannot be
/// read, or the browser lets go.
async fn follow(agents: Agents, at: At, id: String, tx: mpsc::Sender<Event>) {
    let mut text = Utf8Stream::default();
    let mut from = 0;
    let mut joined = false;
    loop {
        let chunk = match agents.tail(at, &id, from).await {
            Ok(c) => c,
            Err(e) => {
                let _ = tx
                    .send(event("error", json!({ "message": e.to_string() })))
                    .await;
                return;
            }
        };
        if !joined {
            joined = true;
            let start = join_at(chunk.size);
            if start > 0 {
                if tx
                    .send(event("skipped", json!({ "bytes": start })))
                    .await
                    .is_err()
                {
                    return;
                }
                from = start;
                continue;
            }
        }
        let bytes = match chunk.bytes() {
            Ok(b) => b,
            Err(e) => {
                let _ = tx
                    .send(event("error", json!({ "message": e.to_string() })))
                    .await;
                return;
            }
        };
        from += bytes.len() as u64;
        let out = text.push(&bytes);
        if !out.is_empty()
            && tx
                .send(event("chunk", json!({ "text": out })))
                .await
                .is_err()
        {
            return;
        }
        let caught_up = from >= chunk.size;
        if chunk.ended && caught_up {
            let rest = text.finish();
            if !rest.is_empty() {
                let _ = tx.send(event("chunk", json!({ "text": rest }))).await;
            }
            let _ = tx.send(event("end", json!({ "size": from }))).await;
            return;
        }
        if caught_up {
            // Nobody is reading: stop asking the agent.
            if tx.is_closed() {
                return;
            }
            sleep(FOLLOW_EVERY).await;
        }
    }
}

/// The SSE response following job `id` on agent `at`.
pub fn stream(agents: Agents, at: At, id: String) -> Response {
    let (tx, rx) = mpsc::channel::<Event>(64);
    tokio::spawn(follow(agents, at, id, tx));
    let body = ReceiverStream::new(rx).map(Ok::<Event, Infallible>);
    Sse::new(body)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A reader joins a short log at its first byte, and a long one at its
    /// last [`BACKLOG`] bytes.
    #[test]
    fn a_reader_joins_a_long_log_near_its_end() {
        assert_eq!(join_at(0), 0);
        assert_eq!(join_at(BACKLOG), 0);
        assert_eq!(join_at(BACKLOG + 10), 10);
        assert_eq!(join_at(3 * BACKLOG), 2 * BACKLOG);
    }
}
