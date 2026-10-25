//! Streams a member opens towards the gateway (§5.1, §5.8).
//!
//! - **Towards its own app**: a request its own listener took while joined.
//!   The gateway serves each with its own router — the same path as a
//!   browser's request — marked [`MemberOrigin`], so its identity headers are
//!   believed and it is routed for the member's app.
//! - **Towards `group:agent`**: the member's jobs, each sent to an agent
//!   ([`jobs`](super::jobs), §11.5).
//! - **Towards `group:keep`**: an Admin's requests to a machine's keep, each
//!   sent to the keep of the machine it names ([`keeps`](super::keeps)). Any
//!   other group is refused.

use std::convert::Infallible;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::ConnectInfo;
use cluster::gateway::{MemberStream, KEEP};
use cluster::member::GROUP;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::Request;
use hyper_util::rt::TokioIo;
use tower::Service;

use super::jobs::{self, AGENT_APP};
use super::{keeps, Gateway, MemberOrigin};

/// Serve `stream`, which the member at `ip` opened.
pub fn serve(gateway: &Arc<Gateway>, ip: IpAddr, stream: MemberStream) {
    if let Some(group) = stream.target.strip_prefix(GROUP) {
        if group == AGENT_APP {
            serve_group(gateway, ip, stream, Group::Agents);
        } else if group == KEEP {
            serve_group(gateway, ip, stream, Group::Keeps);
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

/// The groups a member's stream may reach.
#[derive(Clone, Copy)]
enum Group {
    Agents,
    Keeps,
}

/// Each request on `stream` goes to a member of `group`.
fn serve_group(gateway: &Arc<Gateway>, ip: IpAddr, stream: MemberStream, group: Group) {
    let open = gateway.hub.traffic.stream();
    let gateway = gateway.clone();
    let from = stream.node;
    tokio::spawn(async move {
        let _open = open;
        let svc = service_fn(move |req: Request<Incoming>| {
            let gateway = gateway.clone();
            async move {
                let req = req.map(Body::new);
                Ok::<_, Infallible>(match group {
                    Group::Agents => jobs::answer(&gateway, from, ip, req).await,
                    Group::Keeps => keeps::answer(&gateway, from, ip, req).await,
                })
            }
        });
        if let Err(e) = http1::Builder::new()
            .serve_connection(TokioIo::new(stream.conn), svc)
            .await
        {
            tracing::debug!(error = %e, "member group stream ended");
        }
    });
}
