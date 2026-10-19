//! A request a gateway sent over this member's socket (§5.1), served exactly
//! as the TLS entrance serves a TCP connection — the whole router, Host-based
//! site dispatch, UI files and upgrades included — with a synthetic peer and
//! the [`ClusterPeer`] mark: its identity headers are believed, and it is served
//! here, never forwarded up again (§5.8).

use std::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::ConnectInfo;
use axum::response::Response;
use axum::Router;
use cluster::replicate::Replicator;
use cluster::session::VirtualConn;
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::Request;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ConnBuilder;
use tower::Service;
use web::cluster::ClusterPeer;

use super::flush::{self, flush_first, Flushed};
use super::held_body::HeldBody;
use super::in_flight::InFlight;

/// The address a request off a socket appears to come from: no address at
/// all, so nothing that trusts a peer by address — `--local-signin`, the
/// gateway allowlist — applies to it.
pub const SOCKET_PEER: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0);

/// Serve `conn`, opened on the socket to cluster `cluster`, with `router`.
/// Each response ends only after the events its request caused are on that
/// socket, through `replicator` ([`flush`](super::flush)), and a request that
/// can append counts in `in_flight` until its response ends.
pub fn serve(
    router: Router,
    conn: VirtualConn,
    cluster: usize,
    replicator: Option<Arc<Replicator>>,
    in_flight: InFlight,
) {
    tokio::spawn(async move {
        let svc = service_fn(move |req: Request<Incoming>| {
            let mut router = router.clone();
            let replicator = replicator.clone();
            let in_flight = in_flight.clone();
            async move {
                // A read appends nothing. Ending it after the writer's queue
                // would hold the console's status poll behind every seal the
                // cast has in flight, and a leaving member does not wait on it.
                if !flush::writes_records(req.method()) {
                    return Ok::<_, Infallible>(route(&mut router, req, cluster).await);
                }
                let running = in_flight.enter();
                let res = route(&mut router, req, cluster).await;
                let Some(r) = replicator else {
                    return Ok(res.map(|body| Body::new(HeldBody::new(body, running))));
                };
                if flush_first(res.body()) {
                    r.flush(cluster).await;
                    return Ok(res.map(|body| Body::new(HeldBody::new(body, running))));
                }
                Ok(res.map(|body| {
                    Body::new(HeldBody::new(
                        Body::new(Flushed::new(body, r, cluster)),
                        running,
                    ))
                }))
            }
        });
        let conn = ConnBuilder::new(TokioExecutor::new())
            .serve_connection_with_upgrades(TokioIo::new(conn), svc)
            .await;
        if let Err(e) = conn {
            tracing::debug!(error = %e, "a gateway's stream ended");
        }
    });
}

/// `req` through `router`, marked as a request off socket `cluster`.
async fn route(router: &mut Router, req: Request<Incoming>, cluster: usize) -> Response {
    let mut req = req.map(Body::new);
    req.extensions_mut().insert(ConnectInfo(SOCKET_PEER));
    req.extensions_mut().insert(ClusterPeer { cluster });
    match router.call(req).await {
        Ok(res) => res,
        Err(never) => match never {},
    }
}
