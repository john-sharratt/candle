//! This member's own listener (`docs/replicated_substrate.md` §5.8): the
//! `--host`/`--port` server and the TLS entrance. Per app, a request arriving
//! here is forwarded up to a gateway while the app is joined, served here while
//! it is cut off, and answered `503` while it is still connecting or stopped.
//! Some requests are always served here: one that came over a socket, one an
//! allowlisted gateway marked `x-tokera-hop: dumb`, `GET /v1/status`, and every
//! request to a local build.
//!
//! A request served here on a sticky route is claimed first ([`sticky`]).
//!
//! Forwarding up writes the request onto a stream to a gateway with the
//! identity this listener resolved — believed headers from an allowlisted peer,
//! or `--local-signin` — and nothing a client sent itself. The answer streams
//! back down the same stream; an upgrade becomes a tunnel through it.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::extract::{ConnectInfo, Request, State};
use axum::http::{header, HeaderValue, StatusCode, Uri};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use cluster::member::{decide, Arrival, Serve, DUMB_HOP, DUMB_HOP_VALUE};
use cluster::session::VirtualConn;
use hyper::client::conn::http1;
use hyper::upgrade::OnUpgrade;
use hyper_util::rt::TokioIo;
use serde_json::json;
use tokio::io::copy_bidirectional;
use web::auth::Gateways;
use web::cluster::ClusterPeer;
use web::proxy::{HOP_BY_HOP, IDENTITY_HEADERS};
use web::Config;

use super::sticky::{self, Sticky};
use super::{AppLink, Cluster};
use crate::access::LocalSignin;

/// How long a caller is told to wait while the app cannot serve.
const RETRY_AFTER: Duration = Duration::from_secs(5);

pub struct Listener {
    pub cluster: Arc<Cluster>,
    /// The host's site table, which names the app a request is for.
    pub sites: Arc<Config>,
    /// Whose `x-tokera-*` headers — the dumb mark among them — are believed.
    pub gateways: Gateways,
    pub local_signin: LocalSignin,
}

pub async fn layer(State(l): State<Arc<Listener>>, req: Request, next: Next) -> Response {
    let host = req
        .headers()
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);
    let app = l.sites.site_for(host.as_deref()).name.clone();
    let Some(link) = l.cluster.link(&app).cloned() else {
        return next.run(req).await;
    };
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(a)| *a);
    let arrival = Arrival {
        path: req.uri().path(),
        on_socket: req.extensions().get::<ClusterPeer>().is_some(),
        dumb_from_gateway: req
            .headers()
            .get(DUMB_HOP)
            .is_some_and(|v| v.as_bytes() == DUMB_HOP_VALUE.as_bytes())
            && peer.is_some_and(|p| l.gateways.trusts(p.ip())),
    };
    // Before the tenant connects there is no link to forward on: the tenant is
    // still loading, and answers for itself.
    let serve = match link.member() {
        Some(m) => decide(&m.link(), l.cluster.build().is_local(), arrival),
        None => Serve::Local,
    };
    match serve {
        Serve::Local => here(&link, req, next).await,
        Serve::ForwardUp => match link.member().and_then(|m| m.open_up()) {
            Some(conn) => up(&l, conn, peer, req).await,
            // The last socket went between the decision and the stream: cut
            // off, so served here.
            None => here(&link, req, next).await,
        },
        Serve::Unavailable(why) => unavailable(&app, &why),
    }
}

async fn here(link: &AppLink, req: Request, next: Next) -> Response {
    match (link.member(), link.routes.get()) {
        (Some(m), Some(table)) => match sticky::check(m, table, req).await {
            Sticky::Serve(req, used) => sticky::hold(next.run(req).await, used),
            Sticky::Answer(res) => res,
        },
        _ => next.run(req).await,
    }
}

fn unavailable(app: &str, why: &str) -> Response {
    let mut res = (
        StatusCode::SERVICE_UNAVAILABLE,
        Json(json!({
            "error": "cluster_unavailable",
            "detail": format!("{app} cannot serve this yet: {why}"),
        })),
    )
        .into_response();
    res.headers_mut().insert(
        header::RETRY_AFTER,
        HeaderValue::from(RETRY_AFTER.as_secs()),
    );
    res
}

/// Forward `req` up a stream to a gateway.
async fn up(l: &Listener, conn: VirtualConn, peer: Option<SocketAddr>, req: Request) -> Response {
    let (mut parts, body) = req.into_parts();
    let client_upgrade = parts.extensions.remove::<OnUpgrade>();
    let upgrading = parts.headers.contains_key(header::UPGRADE);
    if !peer.is_some_and(|p| l.gateways.trusts(p.ip())) {
        for name in IDENTITY_HEADERS {
            parts.headers.remove(name);
        }
    }
    l.local_signin.apply(&mut parts.headers, peer);
    parts.headers.remove(DUMB_HOP);
    for h in HOP_BY_HOP {
        parts.headers.remove(&h);
    }
    if upgrading {
        parts
            .headers
            .insert(header::CONNECTION, HeaderValue::from_static("upgrade"));
    }
    if let Some(pq) = parts.uri.path_and_query() {
        if let Ok(uri) = pq.as_str().parse::<Uri>() {
            parts.uri = uri;
        }
    }
    let req = Request::from_parts(parts, body);
    let (mut sender, connection) = match http1::handshake(TokioIo::new(conn)).await {
        Ok(h) => h,
        Err(e) => return failed(&e.to_string()),
    };
    tokio::spawn(async move {
        if let Err(e) = connection.with_upgrades().await {
            tracing::debug!(error = %e, "a stream up to the gateway ended");
        }
    });
    let mut res = match sender.send_request(req).await {
        Ok(r) => r,
        Err(e) => return failed(&e.to_string()),
    };
    if res.status() == StatusCode::SWITCHING_PROTOCOLS {
        if let (Some(cu), Some(uu)) = (client_upgrade, res.extensions_mut().remove::<OnUpgrade>()) {
            tokio::spawn(async move {
                if let Ok((c, u)) = tokio::try_join!(cu, uu) {
                    let _ = copy_bidirectional(&mut TokioIo::new(c), &mut TokioIo::new(u)).await;
                }
            });
        }
    }
    let upgraded = res.status() == StatusCode::SWITCHING_PROTOCOLS;
    let (rp, rb) = res.into_parts();
    let mut out = Response::from_parts(rp, Body::new(rb));
    for h in HOP_BY_HOP {
        out.headers_mut().remove(&h);
    }
    if upgraded {
        out.headers_mut()
            .insert(header::CONNECTION, HeaderValue::from_static("upgrade"));
    }
    out
}

/// The stream up failed before an answer: the client retries, and lands here
/// cut off or on a socket that came back.
fn failed(why: &str) -> Response {
    tracing::debug!("forwarding up failed: {why}");
    unavailable("this app", "its gateway did not answer")
}
