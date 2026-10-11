//! The gateway's cluster (`docs/replicated_substrate.md` §3–§5, §12): members
//! dial `/v1/cluster/:app` and, once ready, serve the routes that name their
//! app in place of the configured upstream.
//!
//! | Module | |
//! |---|---|
//! | [`endpoint`] | the socket a member dials, and its admission |
//! | [`codec`] | axum's WebSocket messages as the cluster pipe's |
//! | [`streams`] | requests members forward up, served by this gateway's router |
//! | [`jobs`] | a member's jobs, each sent to the agent §11.3 chooses |
//! | [`route`] | where a request for a clustered app goes: dispatch, keys, pins |
//! | [`transport`] | one request to one member, over a stream on its socket |
//! | [`aggregate`], [`merge_ws`] | a request every ready member answers |
//! | [`tunnel`] | a WebSocket to a member, closed `1011` when its socket drops |
//! | [`page`] | the status page |
//!
//! The state itself — admission, members, claims, the log — is the `cluster`
//! crate's [`Hub`]; this module is its HTTP.

pub mod aggregate;
pub mod codec;
pub mod endpoint;
pub mod jobs;
pub mod merge_ws;
pub mod page;
pub mod route;
pub mod streams;
pub mod transport;
pub mod tunnel;

use std::net::SocketAddr;
use std::sync::{Arc, OnceLock};

use anyhow::Context;
use axum::Router;
use cluster::gateway::Hub;
use cluster::handshake::GroupKey;

use crate::config::Cluster as ClusterConfig;

/// A request a gateway sent this member over its socket (§5.1): set by the
/// member's host on every request it serves off a cluster stream. Its
/// identity headers are believed, as from a trusted gateway's address, and it
/// is always served where it arrived, never forwarded up again (§5.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClusterPeer {
    /// The socket's index among the member's `--cluster`s.
    pub cluster: usize,
}

/// A request that arrived on a stream a member opened (§5.8): this gateway
/// believes its identity headers, because the socket is authenticated, and
/// routes it for `app` whatever its `Host` says.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemberOrigin {
    pub app: String,
    pub node: u64,
}

pub struct Gateway {
    pub hub: Arc<Hub>,
    /// This gateway's own router, which serves the streams members open.
    /// Set once the router exists, which holds this gateway in its state.
    router: OnceLock<Router>,
}

impl Gateway {
    /// From the `cluster:` block, reading the group key; named in the log by
    /// the block's `name`, or else by the address it binds.
    pub fn new(cfg: &ClusterConfig, bind: SocketAddr) -> anyhow::Result<Arc<Gateway>> {
        let key = GroupKey::load(&cfg.group_key_file)
            .with_context(|| "the cluster's group key".to_string())?;
        let name = cfg.name.clone().unwrap_or_else(|| bind.to_string());
        Ok(Arc::new(Gateway {
            hub: Hub::new(key, name),
            router: OnceLock::new(),
        }))
    }

    /// A gateway around `hub`, for tests that build the hub themselves.
    pub fn with_hub(hub: Arc<Hub>) -> Arc<Gateway> {
        Arc::new(Gateway {
            hub,
            router: OnceLock::new(),
        })
    }

    pub fn set_router(&self, router: Router) {
        // Set once, when the server's router is built.
        let _ = self.router.set(router);
    }

    fn router(&self) -> Option<Router> {
        self.router.get().cloned()
    }
}
