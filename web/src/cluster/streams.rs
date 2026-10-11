//! Streams a member opens towards the gateway (§5.1, §5.8).
//!
//! - **Towards its own app**: a request its own listener took while joined.
//!   The gateway serves each with its own router — the same path as a
//!   browser's request — marked [`MemberOrigin`], so its identity headers are
//!   believed and it is routed for the member's app.
//! - **Towards `group:agent`**: the member's jobs, each sent to an agent
//!   ([`jobs`](super::jobs), §11.5). Any other group is refused.

use std::convert::Infallible;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::ConnectInfo;
use cluster::gateway::MemberStream;
use cluster::member::GROUP;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::Request;
use hyper_util::rt::TokioIo;
use tower::Service;

use super::jobs::{self, AGENT_APP};
use super::{Gateway, MemberOrigin};

/// Serve `stream`, which the member at `ip` opened.
pub fn serve(gateway: &Arc<Gateway>, ip: IpAddr, stream: MemberStream) {
    if let Some(group) = stream.target.strip_prefix(GROUP) {
        if group == AGENT_APP {
            serve_jobs(gateway, ip, stream);
        } else {
            tracing::debug!(group, "a member opened a stream to a group that takes none");
            stream.conn.reset();
        }
        return;
    }
    let Some(router) = gateway.router() else {
        stream.conn.reset();
        return;
    };
    let open = gateway.hub.traffic.stream();
    let origin = MemberOrigin {
        app: stream.app,
        node: stream.node,
    };
    tokio::spawn(async move {
        let _open = open;
        let svc = service_fn(move |req: Request<Incoming>| {
            let mut router = router.clone();
            let origin = origin.clone();
            async move {
                let mut req = req.map(Body::new);
                req.extensions_mut()
                    .insert(ConnectInfo(SocketAddr::new(ip, 0)));
                req.extensions_mut().insert(origin);
                match router.call(req).await {
                    Ok(res) => Ok::<_, Infallible>(res),
                    Err(never) => match never {},
                }
            }
        });
        let conn = http1::Builder::new()
            .serve_connection(TokioIo::new(stream.conn), svc)
            .with_upgrades();
        if let Err(e) = conn.await {
            tracing::debug!(error = %e, "member stream ended");
        }
    });
}

/// Each request on `stream` goes to an agent.
fn serve_jobs(gateway: &Arc<Gateway>, ip: IpAddr, stream: MemberStream) {
    let open = gateway.hub.traffic.stream();
    let gateway = gateway.clone();
    let from = stream.node;
    tokio::spawn(async move {
        let _open = open;
        let svc = service_fn(move |req: Request<Incoming>| {
            let gateway = gateway.clone();
            async move {
                Ok::<_, Infallible>(jobs::answer(&gateway, from, ip, req.map(Body::new)).await)
            }
        });
        if let Err(e) = http1::Builder::new()
            .serve_connection(TokioIo::new(stream.conn), svc)
            .await
        {
            tracing::debug!(error = %e, "member job stream ended");
        }
    });
}
