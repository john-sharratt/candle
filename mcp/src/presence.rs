//! The aggregator joined to the gateway as the app `mcp`
//! (`docs/mcp_app.md` §6): it dials out, declares its routes, and serves the
//! requests the gateway sends over that socket with the same router its
//! listener serves. The gateway deals `/mcp` to every joined member in turn,
//! so a second machine's aggregator shares the load by joining, with no port
//! of its own open to the gateway.
//!
//! Like keep, it registers no upgrader and holds no substrate: keep releases
//! each machine's on its own, and the gateway admits it on whatever build it
//! runs (`cluster::gateway::SERVICES`).

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::body::Body;
use axum::Router;
use cluster::dispatch::{Dispatch, Route, RouteTable};
use cluster::frame::{Capacity, MemberState};
use cluster::gateway::MCP;
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

/// The one path the aggregator serves.
pub const PATH: &str = "/mcp";

/// The routes the socket declares: `/mcp`, to any member. Every member reads
/// a session another opened, since sessions live in their ids, so nothing
/// sticks a caller to one.
pub fn table() -> RouteTable {
    RouteTable::new(vec![Route::new(
        &["POST", "GET", "DELETE"],
        PATH,
        Dispatch::Any,
    )])
    .expect("the aggregator declares one route")
}

/// A member is ready from the moment it joins: it loads nothing.
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

/// Serve `conn`, a stream the gateway opened, with `router`: each request
/// marked as off the socket, whose identity headers are believed
/// (`crate::believed`).
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

/// Join `config`'s cluster as the app `mcp`, serving `router` to the streams
/// its gateway opens. The socket dials and redials on its own task.
pub fn join(config: &Config, router: Router) -> anyhow::Result<()> {
    let c = &config.cluster;
    let node = kept_node(&config.state.join("node"))?;
    let key = GroupKey::load(&c.group_key).map_err(|e| anyhow::anyhow!("group key: {e}"))?;
    let target = Target::parse(&c.urls, Some(&c.name))?;
    let member = AppMember::new(
        Identity {
            app: MCP.to_string(),
            node,
            host: c.host.clone(),
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
    tracing::info!(node = %format!("{node:016x}"), urls = %target.url, "joining the cluster as mcp");
    tokio::spawn(async move {
        let stopped = member
            .run(0, Dial::new(Arc::new(target), MCP, tls), move |conn| {
                serve(router.clone(), conn)
            })
            .await;
        tracing::warn!("mcp: left the cluster: {stopped:?}");
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `/mcp` is declared for the three methods the server answers, dealt to
    /// any member; nothing else is the aggregator's.
    #[test]
    fn the_socket_declares_mcp_to_any_member() {
        let t = table();
        for method in ["POST", "GET", "DELETE"] {
            let m = t.lookup(method, PATH).expect(method);
            assert_eq!(m.route.dispatch, Dispatch::Any);
        }
        assert!(t.lookup("PUT", PATH).is_none());
        assert!(t.lookup("POST", "/v1/mcp").is_none());
    }

    #[test]
    fn a_member_is_ready_as_it_joins() {
        let c = capacity();
        assert_eq!((c.state, c.idle), (MemberState::Ready, true));
    }
}
