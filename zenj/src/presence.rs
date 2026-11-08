//! ZenJ joined to the gateway as the app `zenj` (§3.1): it dials out,
//! declares its routes, and serves the streams the gateway opens with the
//! same router its listener serves. Like keep and the aggregator it is a
//! service: admitted on whatever build it runs, holding no substrate, never
//! rolled out — its machine's is released there, between parties.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::Router;
use cluster::card;
use cluster::dispatch::{Dispatch, Route, RouteTable};
use cluster::frame::{Capacity, MemberState};
use cluster::gateway::ZENJ;
use cluster::handshake::GroupKey;
use cluster::machine::measure;
use cluster::member::{kept_node, AppMember, Identity};
use cluster::session::VirtualConn;
use cluster::version::this_build;
use cluster_dial::{tls_connector, Dial, Target};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::Request;
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as ConnBuilder;
use tower::Service;
use web::cluster::ClusterPeer;

use crate::config::Config;
use crate::http::{EVENTS_PATH, MCP_PATH, STATUS_PATH};

/// The routes the socket declares, each to any member: there is one, on the
/// machine the decks are cabled to.
pub fn table() -> RouteTable {
    RouteTable::new(vec![
        Route::new(&["POST", "GET", "DELETE"], MCP_PATH, Dispatch::Any),
        Route::new(&["GET"], STATUS_PATH, Dispatch::Any),
        Route::new(&["GET"], EVENTS_PATH, Dispatch::Any),
    ])
    .expect("ZenJ's routes are distinct")
}

/// Ready from the moment it joins.
fn capacity() -> Capacity {
    let machine = measure();
    Capacity {
        state: MemberState::Ready,
        idle: true,
        cores: machine.cores,
        load_milli: machine.load_milli,
        os: machine.os,
        mem_free_mib: machine.mem_free_mib,
        ..Capacity::default()
    }
}

fn serve(router: Router, conn: VirtualConn) {
    tokio::spawn(async move {
        let svc = service_fn(move |req: Request<Incoming>| {
            let mut router = router.clone();
            async move {
                let mut req = req.map(Body::new);
                req.extensions_mut().insert(ClusterPeer { cluster: 0 });
                match router.call(req).await {
                    Ok(res) => Ok::<_, Infallible>(res),
                    Err(never) => match never {},
                }
            }
        });
        if let Err(e) = ConnBuilder::new(TokioExecutor::new())
            .serve_connection(TokioIo::new(conn), svc)
            .await
        {
            tracing::debug!(error = %e, "a gateway's stream ended");
        }
    });
}

/// Join `config`'s cluster as `zenj`, serving `router` to the streams its
/// gateway opens.
pub fn join(config: &Config, router: Router) -> anyhow::Result<()> {
    let c = &config.cluster;
    let node = kept_node(&config.state.join("node"))?;
    let key = GroupKey::load(&c.group_key).map_err(|e| anyhow::anyhow!("group key: {e}"))?;
    let target = Target::parse(&c.urls, Some(&c.name))?;
    let member = AppMember::new(
        Identity {
            app: ZENJ.to_string(),
            node,
            host: c.host.clone(),
            card: card::this(),
            build: this_build(),
            model: None,
            repos: Vec::new(),
            leaseholder: false,
        },
        key,
        vec![target.url.clone()],
        table(),
        Box::new(capacity),
        None,
        Instant::now(),
    );
    let tls = tls_connector()?;
    tracing::info!(node = %format!("{node:016x}"), urls = %target.url, "joining the cluster as zenj");
    tokio::spawn(async move {
        let stopped = member
            .run(0, Dial::new(Arc::new(target), ZENJ, tls), move |conn| {
                serve(router.clone(), conn)
            })
            .await;
        tracing::warn!("zenj: left the cluster: {stopped:?}");
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_socket_declares_zenjs_routes_only() {
        let t = table();
        for m in ["POST", "GET", "DELETE"] {
            assert_eq!(t.lookup(m, MCP_PATH).unwrap().route.dispatch, Dispatch::Any);
        }
        assert!(t.lookup("GET", STATUS_PATH).is_some());
        assert!(t.lookup("POST", STATUS_PATH).is_none());
        assert!(
            t.lookup("GET", "/v1/status").is_none(),
            "keep's is the listener's alone"
        );
        assert!(t.lookup("POST", "/v1/mcp").is_none());
    }
}
