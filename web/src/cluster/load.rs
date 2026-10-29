//! A member's answer, carrying the request's [`Load`] until its body is done
//! (§5.3). A chat streams its answer for as long as it decodes, so the request
//! is work for the member until the last byte, not until the headers: the load
//! rides inside the body and is dropped with it — finished, or abandoned by a
//! client that went away. As it drops, the request's `answered` line says when
//! its first byte and its end came (`docs/cooperative_admission.md` §4.6).

use std::time::Instant;

use axum::body::Body;
use axum::http::StatusCode;
use axum::response::Response;
use cluster::gateway::Load;
use futures::StreamExt;

/// The request a load is carried for, as its `answered` line names it.
pub struct Answering {
    pub req: u64,
    pub app: String,
    pub node: u64,
    /// When the gateway began routing it.
    pub started: Instant,
}

/// What the `answered` line reports, kept beside the load in the body.
struct Answered {
    answering: Answering,
    status: u16,
    first_byte: Option<Instant>,
    bytes: u64,
}

impl Drop for Answered {
    fn drop(&mut self) {
        let a = &self.answering;
        let ms = |at: Instant| at.duration_since(a.started).as_millis() as u64;
        tracing::info!(
            target: "cluster::gateway::placement",
            req = a.req,
            app = a.app,
            member = %format!("{:016x}", a.node),
            status = self.status,
            first_byte_ms = self.first_byte.map(ms),
            total_ms = ms(Instant::now()),
            bytes = self.bytes,
            "answered"
        );
    }
}

/// `res` with `load` held until its body is finished or dropped, and its
/// `answered` line logged then. An upgraded connection — a websocket — is a
/// session, not a request, and holds none.
pub fn carry(res: Response, load: Option<Load>, answering: Answering) -> Response {
    let Some(load) = load else {
        return res;
    };
    if res.status() == StatusCode::SWITCHING_PROTOCOLS {
        return res;
    }
    let answered = Answered {
        status: res.status().as_u16(),
        answering,
        first_byte: None,
        bytes: 0,
    };
    let (parts, body) = res.into_parts();
    Response::from_parts(parts, ride(body, load, answered))
}

/// `body` carrying `keep` and `answered` until it is finished or dropped,
/// counting into `answered` as its bytes pass.
fn ride<K: Send + 'static>(body: Body, keep: K, answered: Answered) -> Body {
    // One value, borrowed whole in the closure, so all of it moves into the
    // body: a closure naming only a field of it would copy that field out and
    // leave the rest — the load, the record — to drop as soon as the response
    // is built.
    let mut held = (keep, answered);
    Body::from_stream(body.into_data_stream().map(move |chunk| {
        let (_, answered) = &mut held;
        if let Ok(bytes) = &chunk {
            answered.first_byte.get_or_insert_with(Instant::now);
            answered.bytes += bytes.len() as u64;
        }
        chunk
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use http_body_util::BodyExt;

    fn answering() -> Answering {
        Answering {
            req: 1,
            app: "zend".into(),
            node: 1,
            started: Instant::now(),
        }
    }

    /// No load leaves the response as it was; an upgrade holds none.
    #[tokio::test]
    async fn a_response_without_load_is_untouched() {
        let res = carry(Response::new(Body::from("ok")), None, answering());
        let bytes = res.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&bytes[..], b"ok");
    }

    /// **What a body carries rides in it until the body is done**: built, the
    /// body still holds it; read to the end and dropped, it lets it go.
    #[tokio::test]
    async fn what_a_body_carries_drops_with_the_body() {
        let keep = Arc::new(());
        let answered = Answered {
            answering: answering(),
            status: 200,
            first_byte: None,
            bytes: 0,
        };
        let body = ride(Body::from("abcde"), keep.clone(), answered);
        assert_eq!(Arc::strong_count(&keep), 2, "held by the body");
        let bytes = body.collect().await.unwrap().to_bytes();
        assert_eq!(&bytes[..], b"abcde");
        assert_eq!(Arc::strong_count(&keep), 1, "let go with the body");
    }
}
