//! HTTP/2 and HTTP/1.1 over TLS — the half of the entrance every client can
//! reach, and the one that tells a client HTTP/3 is there (`Alt-Svc`).

use std::convert::Infallible;
use std::time::Duration;

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::http::{header, HeaderValue};
use axum::Router;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::Request;
use hyper_util::rt::{TokioExecutor, TokioIo, TokioTimer};
use hyper_util::server::conn::auto::Builder as ConnBuilder;
use hyper_util::server::graceful::GracefulShutdown;
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::time::timeout;
use tokio_rustls::TlsAcceptor;
use tower::Service;

use super::origin::origin_form;
use super::SecureEntrance;

/// How long a client has to finish the TLS handshake. On a public port a
/// connection that opens and sends nothing is routine, and without a bound
/// each one holds a task and a socket for ever.
const HANDSHAKE: Duration = Duration::from_secs(10);

/// How long an HTTP/1.1 client has to send a request's headers — including the
/// next request on a kept-alive connection.
const HEADERS: Duration = Duration::from_secs(30);

/// Accept TLS connections until `stop` flips, each served on its own task in
/// whichever protocol ALPN settled on; then stop accepting and wait for every
/// open connection to finish what it is answering — HTTP/2 is sent `GOAWAY`,
/// an idle HTTP/1.1 connection is closed.
pub async fn serve(
    listener: TcpListener,
    acceptor: TlsAcceptor,
    router: Router,
    alt_svc: HeaderValue,
    mut stop: watch::Receiver<bool>,
) {
    let graceful = GracefulShutdown::new();
    loop {
        let (sock, peer) = tokio::select! {
            accepted = listener.accept() => match accepted {
                Ok(a) => a,
                Err(e) => {
                    // Out of descriptors, mostly. Accepting again at once
                    // would spin on the same error, so the loop rests first —
                    // what `axum::serve` does for the plain entrance.
                    tracing::warn!(error = %e, "tls: accept failed");
                    tokio::time::sleep(Duration::from_secs(1)).await;
                    continue;
                }
            },
            _ = stop.changed() => break,
        };
        // Small response writes must not wait on the client's delayed ACK — the
        // same reason the plain entrance sets it.
        let _ = sock.set_nodelay(true);
        let acceptor = acceptor.clone();
        let router = router.clone();
        let alt_svc = alt_svc.clone();
        let watcher = graceful.watcher();
        tokio::spawn(async move {
            let tls = match timeout(HANDSHAKE, acceptor.accept(sock)).await {
                Ok(Ok(s)) => s,
                // Scanners and abandoned handshakes are routine on a public port.
                Ok(Err(e)) => {
                    tracing::debug!(%peer, error = %e, "tls: handshake failed");
                    return;
                }
                Err(_) => {
                    tracing::debug!(%peer, "tls: handshake timed out");
                    return;
                }
            };
            let svc = service_fn(move |req: Request<Incoming>| {
                let mut router = router.clone();
                let alt_svc = alt_svc.clone();
                async move {
                    let mut req = req.map(Body::new);
                    origin_form(&mut req);
                    req.extensions_mut().insert(ConnectInfo(peer));
                    req.extensions_mut().insert(SecureEntrance);
                    let mut res = match router.call(req).await {
                        Ok(r) => r,
                        Err(never) => match never {},
                    };
                    res.headers_mut().insert(header::ALT_SVC, alt_svc);
                    Ok::<_, Infallible>(res)
                }
            });
            let mut builder = ConnBuilder::new(TokioExecutor::new());
            builder
                .http1()
                .timer(TokioTimer::new())
                .header_read_timeout(HEADERS);
            // With upgrades, so a websocket over HTTP/1.1 reaches the proxy's
            // tunnel exactly as it does on the plain entrance.
            let conn = builder
                .serve_connection_with_upgrades(TokioIo::new(tls), svc)
                .into_owned();
            if let Err(e) = watcher.watch(conn).await {
                tracing::debug!(%peer, error = %e, "tls: connection ended");
            }
        });
    }
    drop(listener);
    graceful.shutdown().await;
}
