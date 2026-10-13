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

/// The address a request off a socket appears to come from: no address at
/// all, so nothing that trusts a peer by address — `--local-signin`, the
/// gateway allowlist — applies to it.
pub const SOCKET_PEER: SocketAddr = SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0);

/// Serve `conn`, opened on the socket to cluster `cluster`, with `router`.
/// Each response ends only after the events its request caused are on that
/// socket, through `replicator` ([`flush`](super::flush)).
pub fn serve(
    router: Router,
    conn: VirtualConn,
    cluster: usize,
    replicator: Option<Arc<Replicator>>,
) {
    tokio::spawn(async move {
        let svc = service_fn(move |req: Request<Incoming>| {
            let mut router = router.clone();
            let replicator = replicator.clone();
            async move {
                let writes = flush::writes_records(req.method());
                let mut req = req.map(Body::new);
                req.extensions_mut().insert(ConnectInfo(SOCKET_PEER));
                req.extensions_mut().insert(ClusterPeer { cluster });
                let res = match router.call(req).await {
                    Ok(res) => res,
                    Err(never) => match never {},
                };
                let Some(r) = replicator else {
                    return Ok::<_, Infallible>(res);
                };
                // A read appends nothing. Ending it after the writer's queue
                // would hold the console's status poll behind every seal the
                // cast has in flight.
                if !writes {
                    return Ok(res);
                }
                if flush_first(res.body()) {
                    r.flush(cluster).await;
                    return Ok(res);
                }
                Ok(res.map(|body| Body::new(Flushed::new(body, r, cluster))))
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
