//! A response body that keeps a guard until the response has ended — the end
//! of a stream of server-sent events, not the return of its head — or until the
//! body is dropped unread because the client went away.
//!
//! What it holds is whatever must last exactly as long as the answer: a
//! request counted in flight ([`super::in_flight`]), a claim in use
//! ([`super::sticky`]).

use std::pin::Pin;
use std::task::{Context, Poll};

use axum::body::Body;
use bytes::Bytes;
use http_body::{Frame, SizeHint};

/// `inner`, holding `guard` until it ends.
pub struct HeldBody<G> {
    inner: Body,
    guard: Option<G>,
}

impl<G> HeldBody<G> {
    pub fn new(inner: Body, guard: G) -> Self {
        Self {
            inner,
            guard: Some(guard),
        }
    }
}

impl<G: Unpin> http_body::Body for HeldBody<G> {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Self::Error>>> {
        let this = &mut *self;
        let polled = Pin::new(&mut this.inner).poll_frame(cx);
        if let Poll::Ready(None) = polled {
            this.guard = None;
        }
        polled
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> SizeHint {
        self.inner.size_hint()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use axum::body::to_bytes;

    struct Count(Arc<AtomicUsize>);

    impl Drop for Count {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// The guard goes when the body is read to its end, not when the head was
    /// returned.
    #[tokio::test]
    async fn the_guard_goes_at_the_bodys_end() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let body = Body::new(HeldBody::new(
            Body::from("data: [DONE]\n\n"),
            Count(Arc::clone(&dropped)),
        ));
        assert_eq!(dropped.load(Ordering::Acquire), 0);
        let bytes = to_bytes(body, usize::MAX).await.unwrap();
        assert_eq!(&bytes[..], b"data: [DONE]\n\n");
        assert_eq!(dropped.load(Ordering::Acquire), 1);
    }

    /// A body dropped unread — the client went away — lets the guard go too.
    #[test]
    fn a_dropped_body_lets_its_guard_go() {
        let dropped = Arc::new(AtomicUsize::new(0));
        let body = HeldBody::new(Body::from("partial"), Count(Arc::clone(&dropped)));
        drop(body);
        assert_eq!(dropped.load(Ordering::Acquire), 1);
    }
}
