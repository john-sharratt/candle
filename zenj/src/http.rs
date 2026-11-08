//! What `zenj` serves, on its listener and on the streams its cluster socket
//! carries alike (§3.1, §13):
//!
//! | Route | |
//! |---|---|
//! | `/v1/zenj/mcp` | the tools |
//! | `GET /v1/zenj/status` | the live view, for a page or a watcher |
//! | `GET /v1/zenj/events?since=N` | the event log after a cursor |
//! | `GET /v1/status` | what keep's guard reads: `refuse` while music plays |
//!
//! Identity headers are believed only from the socket and from a listed
//! gateway, as the aggregator believes them; from anyone else they are
//! dropped before anything reads them.

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;

use axum::extract::{ConnectInfo, Query, Request, State};
use axum::middleware::{from_fn_with_state, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use mcp_server::http;
use serde_json::{json, Value};
use web::cluster::ClusterPeer;
use web::proxy::IDENTITY_HEADERS;

use crate::live::Live;
use crate::tools::Dj;

pub const MCP_PATH: &str = "/v1/zenj/mcp";
pub const STATUS_PATH: &str = "/v1/zenj/status";
pub const EVENTS_PATH: &str = "/v1/zenj/events";
/// keep's guard reads every service's status here.
pub const KEEP_STATUS_PATH: &str = "/v1/status";

/// What keep's guard is told: while music plays, every switch, restart and
/// trial on this machine is refused.
pub fn keep_status(l: &Live) -> Value {
    let mut v = json!({"app": "zenj", "live": l.live, "mode": l.mode.name()});
    if l.live {
        v["refuse"] = json!("music is playing at the party (zenj is live)");
    }
    v
}

async fn status(State(dj): State<Arc<Dj>>) -> Json<Value> {
    Json(serde_json::to_value(&**dj.live.load()).unwrap_or(Value::Null))
}

async fn keep(State(dj): State<Arc<Dj>>) -> Json<Value> {
    Json(keep_status(&dj.live.load()))
}

async fn events(State(dj): State<Arc<Dj>>, Query(q): Query<HashMap<String, String>>) -> Response {
    let since = q.get("since").and_then(|s| s.parse().ok()).unwrap_or(0);
    let got = dj.events.since(since, 500);
    Json(json!({"events": got, "cursor": got.last().map_or(dj.events.last().max(since), |e| e.id)}))
        .into_response()
}

/// Whether a request's identity is believed.
pub fn believed(on_socket: bool, peer: Option<IpAddr>, gateways: &[IpAddr]) -> bool {
    on_socket || peer.is_some_and(|p| gateways.contains(&p))
}

async fn believe(State(gateways): State<Arc<[IpAddr]>>, mut req: Request, next: Next) -> Response {
    let on_socket = req.extensions().get::<ClusterPeer>().is_some();
    let peer = req
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ConnectInfo(a)| a.ip());
    if !believed(on_socket, peer, &gateways) {
        for name in IDENTITY_HEADERS {
            req.headers_mut().remove(name);
        }
    }
    next.run(req).await
}

pub fn router(dj: Arc<Dj>, gateways: Arc<[IpAddr]>) -> Router {
    Router::new()
        .route(
            MCP_PATH,
            post(http::post::<Dj>)
                .get(http::get)
                .delete(http::delete::<Dj>),
        )
        .route(STATUS_PATH, get(status))
        .route(EVENTS_PATH, get(events))
        .route(KEEP_STATUS_PATH, get(keep))
        .with_state(dj)
        .layer(from_fn_with_state(gateways, believe))
}

#[cfg(test)]
mod tests {
    use axum::body::{to_bytes, Body};
    use axum::http::Request as HttpRequest;
    use tower::ServiceExt;

    use super::*;
    use crate::tools::fixture::dj;

    /// **While music plays, keep is told to refuse**; silent, nothing is
    /// refused.
    #[test]
    fn keep_is_refused_while_music_plays() {
        let quiet = keep_status(&Live::default());
        assert_eq!(
            quiet,
            json!({"app": "zenj", "live": false, "mode": "human"})
        );
        let live = keep_status(&Live {
            live: true,
            ..Live::default()
        });
        assert_eq!(
            live["refuse"],
            "music is playing at the party (zenj is live)"
        );
    }

    #[test]
    fn identity_is_believed_from_the_socket_or_a_listed_gateway() {
        let gw: IpAddr = "127.0.0.1".parse().unwrap();
        let other: IpAddr = "192.168.0.9".parse().unwrap();
        assert!(believed(true, None, &[gw]));
        assert!(believed(false, Some(gw), &[gw]));
        assert!(!believed(false, Some(other), &[gw]));
        assert!(!believed(false, None, &[gw]));
    }

    /// **keep's status answers on the path keep reads**, from the live view.
    #[tokio::test]
    async fn the_status_keep_reads_is_served() {
        let (dj, _rx) = dj();
        dj.live.store(Arc::new(Live {
            live: true,
            ..Live::default()
        }));
        let r = router(Arc::new(dj), Arc::from(vec![]));
        let res = r
            .oneshot(
                HttpRequest::get(KEEP_STATUS_PATH)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let body: Value =
            serde_json::from_slice(&to_bytes(res.into_body(), 1 << 16).await.unwrap()).unwrap();
        assert_eq!(body["live"], true);
        assert!(body["refuse"].is_string());
    }
}
