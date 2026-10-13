//! A response served off a socket ends only after the events its request
//! caused are on that socket, ahead of its end (`docs/replicated_substrate.md`
//! §6.4). A request the gateway routes afterwards, to any member, then lands
//! behind those events on that member's ordered lane.
//!
//! A `GET` or a `HEAD` writes nothing, so its response is not held for this.
//! The end of a request that can append is what the gateway reads as complete:
//!
//! - **a body of unknown length** (chunked — a stream of server-sent events) ends
//!   at its terminating chunk, written once the body says it is done. Its
//!   frames pass through as they come; only the end waits for the flush.
//! - **a body of known length** ends with its last byte. Each frame is held
//!   until the next is known, so the last one goes out after the flush.
//! - **a response with no body** — a `204`, an upgrade — ends at its head, so
//!   the flush runs before the head is returned ([`flush_first`]).

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{ready, Context, Poll};

use axum::body::Body;
use bytes::Bytes;
use cluster::replicate::Replicator;
use http_body::{Frame, SizeHint};
use hyper::http::Method;

/// Whether a request can append substrate records, so its response waits
/// until those records are on the socket (§6.4).
///
/// A `GET` or a `HEAD` reads. Waiting out the writer's queue for one held
/// the console's status poll behind the cast's seals until the gateway's
/// deadline, and the page read that deadline as the daemon going away.
pub fn writes_records(method: &Method) -> bool {
    !matches!(*method, Method::GET | Method::HEAD)
}

type Flushing = Pin<Box<dyn Future<Output = ()> + Send>>;

/// Whether a response with this body must be flushed before its head goes.
pub fn flush_first(body: &Body) -> bool {
    http_body::Body::size_hint(body).exact() == Some(0)
}

enum Stage {
    /// Reading the inner body.
    Streaming,
    /// The inner body is done; the flush runs.
    Flushing(Flushing),
    /// Flushed: what was held goes, then the end.
    Flushed,
}

/// A body that ends only after its request's events are on socket `cluster`.
pub struct Flushed {
    inner: Body,
    exact: bool,
    /// The last frame read, held until the next is known (known length).
    held: Option<Frame<Bytes>>,
    replicator: Arc<Replicator>,
    cluster: usize,
    stage: Stage,
}

impl Flushed {
    pub fn new(inner: Body, replicator: Arc<Replicator>, cluster: usize) -> Flushed {
        Flushed {
            exact: http_body::Body::size_hint(&inner).exact().is_some(),
            inner,
            held: None,
            replicator,
            cluster,
            stage: Stage::Streaming,
        }
    }
}

impl http_body::Body for Flushed {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        let this = &mut *self;
        loop {
            match &mut this.stage {
                Stage::Streaming => match ready!(Pin::new(&mut this.inner).poll_frame(cx)) {
                    Some(Ok(frame)) if this.exact => {
                        if let Some(previous) = this.held.replace(frame) {
                            return Poll::Ready(Some(Ok(previous)));
                        }
                    }
                    Some(other) => return Poll::Ready(Some(other)),
                    None => {
                        let r = this.replicator.clone();
                        let c = this.cluster;
                        this.stage = Stage::Flushing(Box::pin(async move { r.flush(c).await }));
                    }
                },
                Stage::Flushing(flushing) => {
                    ready!(flushing.as_mut().poll(cx));
                    this.stage = Stage::Flushed;
                }
                Stage::Flushed => return Poll::Ready(this.held.take().map(Ok)),
            }
        }
    }

    fn is_end_stream(&self) -> bool {
        matches!(self.stage, Stage::Flushed) && self.held.is_none()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_read_does_not_wait_out_the_writer() {
        assert!(!writes_records(&Method::GET));
        assert!(!writes_records(&Method::HEAD));
        assert!(writes_records(&Method::POST));
        assert!(writes_records(&Method::PUT));
        assert!(writes_records(&Method::PATCH));
        assert!(writes_records(&Method::DELETE));
    }
}
