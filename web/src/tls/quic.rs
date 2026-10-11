//! HTTP/3 over QUIC.
//!
//! Every request is handed to the same router the TCP entrance uses, so a site,
//! a proxied daemon and a generated page answer identically whichever protocol
//! asked. The bodies stream both ways: a request body is read frame by frame as
//! the router consumes it, and a response body — an SSE stream included — is
//! sent frame by frame as it is produced, never collected.

use std::net::SocketAddr;

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::header::{self, HeaderName};
use axum::http::{Request, Response};
use axum::Router;
use bytes::{Buf, Bytes};
use futures::Stream;
use h3::error::{Code, StreamError};
use h3::quic::{BidiStream, RecvStream};
use h3::server::{Connection as H3Connection, RequestStream};
use h3_quinn::Connection as QuinnConnection;
use http_body_util::BodyExt;
use quinn::{Endpoint, Incoming};
use tokio::sync::{mpsc, watch};
use tower::Service;

use super::origin::origin_form;
use super::SecureEntrance;

/// Connection-specific headers, which HTTP/3 forbids (RFC 9114 §4.2): the
/// framing they describe belongs to HTTP/1.1, and a peer receiving one must
/// treat the response as malformed.
const CONNECTION_SPECIFIC: [HeaderName; 5] = [
    header::CONNECTION,
    header::TRANSFER_ENCODING,
    header::UPGRADE,
    HeaderName::from_static("keep-alive"),
    HeaderName::from_static("proxy-connection"),
];

/// Accept QUIC connections until `stop` flips, each on its own task, each
/// request on a task of its own beneath it; then stop accepting, tell every
/// open connection to finish (`GOAWAY`), and wait for every request in flight.
pub async fn serve(endpoint: Endpoint, router: Router, mut stop: watch::Receiver<bool>) {
    // Every connection task, and every request task beneath one, holds a clone
    // of `open`. The drain is over when the last one is dropped.
    let (open, mut drained) = mpsc::channel::<()>(1);
    loop {
        let incoming = tokio::select! {
            incoming = endpoint.accept() => match incoming {
                Some(i) => i,
                None => break,
            },
            _ = stop.changed() => break,
        };
        tokio::spawn(connection(
            incoming,
            router.clone(),
            stop.clone(),
            open.clone(),
        ));
    }
    endpoint.set_server_config(None);
    drop(open);
    let _ = drained.recv().await;
    endpoint.close(0u32.into(), b"shutting down");
}

/// One QUIC connection: its requests until the client ends it, or until `stop`
/// flips — then the client is told no more will be taken, and the requests it
/// already sent are answered.
async fn connection(
    incoming: Incoming,
    router: Router,
    mut stop: watch::Receiver<bool>,
    open: mpsc::Sender<()>,
) {
    let conn = match incoming.await {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!(error = %e, "h3: handshake failed");
            return;
        }
    };
    let peer = conn.remote_address();
    let mut h3 = match H3Connection::<_, Bytes>::new(QuinnConnection::new(conn)).await {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!(%peer, error = %e, "h3: connection setup failed");
            return;
        }
    };
    let mut stopping = *stop.borrow();
    if stopping {
        let _ = h3.shutdown(0).await;
    }
    loop {
        let accepted = if stopping {
            h3.accept().await
        } else {
            tokio::select! {
                accepted = h3.accept() => accepted,
                _ = stop.changed() => {
                    stopping = true;
                    let _ = h3.shutdown(0).await;
                    continue;
                }
            }
        };
        match accepted {
            Ok(Some(resolver)) => {
                let router = router.clone();
                let open = open.clone();
                tokio::spawn(async move {
                    let _open = open;
                    match resolver.resolve_request().await {
                        Ok((req, stream)) => {
                            if let Err(e) = answer(router, peer, req, stream).await {
                                tracing::debug!(%peer, error = %e, "h3: request ended");
                            }
                        }
                        Err(e) => tracing::debug!(%peer, error = %e, "h3: bad request"),
                    }
                });
            }
            Ok(None) => break,
            Err(e) => {
                if !e.is_h3_no_error() {
                    tracing::debug!(%peer, error = %e, "h3: connection ended");
                }
                break;
            }
        }
    }
}

/// One request through the router, and its response back down the stream.
async fn answer<S>(
    mut router: Router,
    peer: SocketAddr,
    req: Request<()>,
    stream: RequestStream<S, Bytes>,
) -> Result<(), StreamError>
where
    S: BidiStream<Bytes> + Send + 'static,
    S::RecvStream: Send + 'static,
    S::SendStream: Send,
{
    let (mut send, recv) = stream.split();
    let (parts, ()) = req.into_parts();
    let mut req = Request::from_parts(parts, Body::from_stream(request_body(recv)));
    origin_form(&mut req);
    req.extensions_mut().insert(ConnectInfo(peer));
    req.extensions_mut().insert(SecureEntrance);
    let res = match router.call(req).await {
        Ok(r) => r,
        Err(never) => match never {},
    };

    let (mut parts, mut body) = res.into_parts();
    for h in CONNECTION_SPECIFIC {
        parts.headers.remove(&h);
    }
    send.send_response(Response::from_parts(parts, ())).await?;
    while let Some(frame) = body.frame().await {
        let frame = match frame {
            Ok(f) => f,
            // The response is already under way, so there is no status left to
            // change: the stream is reset, which a client reads as a truncated
            // response rather than a complete one.
            Err(e) => {
                tracing::debug!(%peer, error = %e, "h3: response body failed");
                send.stop_stream(Code::H3_INTERNAL_ERROR);
                return Ok(());
            }
        };
        match frame.into_data() {
            Ok(data) => send.send_data(data).await?,
            Err(frame) => {
                if let Ok(trailers) = frame.into_trailers() {
                    send.send_trailers(trailers).await?;
                }
            }
        }
    }
    send.finish().await
}

/// The request body, as the stream of chunks the router reads.
fn request_body<S>(
    recv: RequestStream<S, Bytes>,
) -> impl Stream<Item = Result<Bytes, StreamError>> + Send + 'static
where
    S: RecvStream + Send + 'static,
{
    futures::stream::unfold(Some(recv), |state| async move {
        let mut recv = state?;
        match recv.recv_data().await {
            Ok(Some(mut buf)) => {
                let chunk = buf.copy_to_bytes(buf.remaining());
                Some((Ok(chunk), Some(recv)))
            }
            Ok(None) => None,
            // One error ends the body; nothing after it is read.
            Err(e) => Some((Err(e), None)),
        }
    })
}
