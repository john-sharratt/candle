//! Streams a member opens towards the gateway (§5.1, §5.8): a request its own
//! listener took while joined. The gateway serves each with its own router —
//! the same path as a browser's request — marked [`MemberOrigin`], so its
//! identity headers are believed and it is routed for the member's app.

use std::convert::Infallible;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::body::Body;
use axum::extract::ConnectInfo;
use cluster::gateway::MemberStream;
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::Request;
use hyper_util::rt::TokioIo;
use tower::Service;

use super::{Gateway, MemberOrigin};

/// Serve `stream`, which the member at `ip` opened.
pub fn serve(gateway: &Arc<Gateway>, ip: IpAddr, stream: MemberStream) {
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
