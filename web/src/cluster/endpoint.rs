//! `GET /v1/cluster/:app` (§3.1): the WebSocket a member dials, merged ahead of
//! site routing on every host and both entrances — and, as
//! `/v1/cluster/gateway`, the link a peer gateway dials (§13.4). HTTP/1.1 only
//! — HTTP/2 and HTTP/3 cannot carry the upgrade, and are answered `426`.

use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::extract::{ConnectInfo, Path, Request, State};
use axum::http::{header, StatusCode, Version};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use cluster::gateway::APPS;
use cluster::peer::GATEWAY_APP;

use super::upgrade::upgrade;
use super::{streams, Gateway};
use crate::tls::SecureEntrance;

/// The largest message a member sends: an `EVENT` fragment of 1 MiB and its
/// header, with room to spare.
const MAX_MESSAGE: usize = 2 << 20;

pub fn router(gateway: Arc<Gateway>) -> Router {
    Router::new()
        .route("/v1/cluster/:app", get(dial_in))
        .with_state(gateway)
}

async fn dial_in(
    State(gateway): State<Arc<Gateway>>,
    Path(app): Path<String>,
    peer: Option<ConnectInfo<SocketAddr>>,
    req: Request,
) -> Response {
    if req.version() != Version::HTTP_11 {
        return (
            StatusCode::UPGRADE_REQUIRED,
            [(header::UPGRADE, "websocket")],
            "the cluster socket is a WebSocket over HTTP/1.1",
        )
            .into_response();
    }
    // A peer gateway's link (§13.4), secure when it came in on the TLS
    // entrance, whose certificate the peer verified.
    if app == GATEWAY_APP {
        let Some(peering) = gateway.peering.clone() else {
            return (
                StatusCode::NOT_FOUND,
                "this gateway has no `cluster.peers`: it peers with no one",
            )
                .into_response();
        };
        let secure = req.extensions().get::<SecureEntrance>().is_some();
        return upgrade(req, MAX_MESSAGE, move |tx, rx| async move {
            peering.accept(tx, rx, secure).await;
        });
    }
    let Some(app) = APPS.iter().find(|a| **a == app).copied() else {
        return (
            StatusCode::NOT_FOUND,
            format!("no app `{app}`: one of {}", APPS.join(", ")),
        )
            .into_response();
    };
    let ip = peer.map_or(IpAddr::from([0, 0, 0, 0]), |ConnectInfo(a)| a.ip());
    upgrade(req, MAX_MESSAGE, move |tx, rx| async move {
        gateway.tick_upgrades();
        let hub = gateway.hub.clone();
        hub.serve(app, ip, tx, rx, |s| streams::serve(&gateway, ip, s))
            .await;
    })
}
