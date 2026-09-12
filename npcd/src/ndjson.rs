//! A response that arrives in pieces, one JSON object per line.
//!
//! # Why NDJSON and not SSE
//!
//! The browser's `EventSource` — the thing that makes Server-Sent Events
//! convenient — can only issue a `GET`, and every generation route here takes a
//! body. Working around that means putting a world id, a personality id and a
//! seed into a query string, which is the shape this daemon deliberately does
//! not use for anything a caller composes.
//!
//! Read through `fetch` instead, both formats are read the same way: take the
//! response's reader, decode, split on newlines. SSE's `data:` prefixes and
//! blank-line framing would be two more things to strip. So the wire format is
//! one compact JSON object per line, and the client is six lines of parsing.
//!
//! # The contract
//!
//! Every stream ends with exactly one terminal line — `done` or `error` — and a
//! consumer that has not seen one has an incomplete answer even if the
//! connection closed cleanly. The terminal `done` carries the **whole** result,
//! not just the last fragment: a consumer may show the fragments as they arrive
//! and then replace what it showed with the final text, which is what makes the
//! preview's occasional resynchronisation invisible.

use std::convert::Infallible;

use axum::body::Body;
use axum::http::{header, HeaderName, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::{json, Value};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_stream::StreamExt;

/// The nginx-family "do not buffer this response" header. Not in `http`'s set
/// of standard names, so it is spelled out once here rather than parsed at each
/// use — the constructor is const, so a typo is a compile error.
const ACCEL_BUFFERING: HeaderName = HeaderName::from_static("x-accel-buffering");

/// `application/x-ndjson`, plus the headers that keep a stream a stream.
///
/// `no-cache` and `no-transform` are not decoration: a proxy that buffers to
/// compress delivers the whole body at once, which is exactly the big-bang
/// response streaming exists to replace. This daemon sits behind the estate
/// gateway, so the hop that would do it is real.
pub fn stream(rx: UnboundedReceiver<Value>) -> Response {
    let lines = UnboundedReceiverStream::new(rx).map(|v| {
        let mut line = v.to_string();
        line.push('\n');
        Ok::<_, Infallible>(line)
    });
    (
        [
            (header::CONTENT_TYPE, "application/x-ndjson"),
            (header::CACHE_CONTROL, "no-cache, no-transform"),
            // Nginx-family proxies buffer a response body by default and would
            // hold every fragment until the generation finished.
            (ACCEL_BUFFERING, "no"),
        ],
        Body::from_stream(lines),
    )
        .into_response()
}

/// A failure that happened before the stream opened.
///
/// Sent as an ordinary status + JSON body, not as a one-line stream: a caller
/// that asked for an unknown world should get a 404 it can branch on, and
/// burying that in a 200 with an `error` line means every client has to parse
/// the body to find out whether the request worked.
pub fn refuse(status: StatusCode, error: &str, detail: &str) -> Response {
    (
        status,
        axum::Json(json!({ "error": error, "detail": detail })),
    )
        .into_response()
}

/// A terminal `error` line, for a failure *after* the stream opened.
///
/// By then the status line is long gone, so the only way to report is in band.
pub fn error_line(error: &str, detail: &str, retry: bool) -> Value {
    json!({ "event": "error", "error": error, "detail": detail, "retry": retry })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use tokio::sync::mpsc::unbounded_channel;

    async fn body_of(r: Response) -> String {
        let bytes = to_bytes(r.into_body(), 1 << 20).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    /// **One object per line, and every line complete.** A consumer splits on
    /// newlines, so an object containing a raw newline would be delivered as two
    /// unparseable halves. `serde_json`'s compact form never emits one, and this
    /// is the test that says so out loud.
    #[tokio::test]
    async fn each_event_is_one_whole_line() {
        let (tx, rx) = unbounded_channel();
        tx.send(json!({ "event": "loading" })).unwrap();
        tx.send(json!({ "event": "token", "text": "a\nb" }))
            .unwrap();
        tx.send(json!({ "event": "done", "tokens": 2 })).unwrap();
        drop(tx);

        let text = body_of(stream(rx)).await;
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        for line in lines {
            let v: Value = serde_json::from_str(line).expect("a line that will not parse");
            assert!(v["event"].is_string());
        }
        assert!(
            text.contains(r"a\nb"),
            "a newline inside a fragment was not escaped and split the line"
        );
    }

    /// The headers are load-bearing: a buffering hop between here and the
    /// browser turns a stream back into the single response it replaced.
    #[tokio::test]
    async fn the_response_is_marked_unbufferable() {
        let (tx, rx) = unbounded_channel::<Value>();
        drop(tx);
        let r = stream(rx);
        let h = r.headers();
        assert_eq!(h[header::CONTENT_TYPE], "application/x-ndjson");
        assert!(h[header::CACHE_CONTROL]
            .to_str()
            .unwrap()
            .contains("no-transform"));
        assert_eq!(h["x-accel-buffering"], "no");
    }

    /// A refusal keeps its status. A caller branching on 404 must not have to
    /// parse a 200 body to discover the world did not exist.
    #[tokio::test]
    async fn a_refusal_is_a_status_not_a_stream() {
        let r = refuse(StatusCode::NOT_FOUND, "world_not_found", "battle-cites");
        assert_eq!(r.status(), StatusCode::NOT_FOUND);
        let v: Value = serde_json::from_str(&body_of(r).await).unwrap();
        assert_eq!(v["error"], "world_not_found");
    }

    #[test]
    fn an_error_line_says_whether_to_retry() {
        let v = error_line("no_room", "the card was full", true);
        assert_eq!(v["event"], "error");
        assert_eq!(v["retry"], true);
    }
}
