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
//! | [`mirror`] | certification's samples, and a mirrored call compared |
//! | [`aggregate`], [`merge_ws`] | a request every ready member answers |
//! | [`tunnel`] | a WebSocket to a member, closed `1011` when its socket drops |
//! | [`page`] | the status page |
//! | [`peers`] | the peer gateways: node id, identity, dial loops (§13.4) |
//! | [`upgrades`] | the gateway's upgrade commands, the forced rollout's plan |
//!
//! The state itself — admission, members, claims, the log — is the `cluster`
//! crate's [`Hub`]; this module is its HTTP.

pub mod aggregate;
pub mod codec;
pub mod endpoint;
pub mod jobs;
pub mod merge_ws;
pub mod mirror;
pub mod page;
pub mod peers;
pub mod route;
pub mod streams;
pub mod transport;
pub mod tunnel;
pub mod upgrades;

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, Context};
use axum::Router;
use cluster::gateway::upgrades::UpgradeConfig;
use cluster::gateway::Hub;
use cluster::handshake::GroupKey;
use cluster::peer::Peering;
use serde_json::Value;
use tokio::time::{interval, MissedTickBehavior};

use crate::acme::names::groups;
use crate::acme::peers::AcmePeers;
use crate::config::{Cluster as ClusterConfig, Config};

/// How often the upgrades are judged: verdicts due, members left behind.
const UPGRADE_TICK: Duration = Duration::from_secs(1);

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
    /// The peer gateways answering the same names (§13.4), when
    /// `cluster.peers` names any.
    pub peering: Option<Arc<Peering>>,
    /// The ACME shared with the peers, once the TLS entrance's certificates
    /// are loaded.
    acme: OnceLock<Arc<AcmePeers>>,
    /// The sites a certificate is ordered for, for the Peers block.
    sites: Vec<String>,
    /// This gateway's own router, which serves the streams members open.
    /// Set once the router exists, which holds this gateway in its state.
    router: OnceLock<Router>,
    /// Set once the upgrades' ticker runs.
    ticking: OnceLock<()>,
}

impl Gateway {
    /// From `cfg`'s `cluster:` block, reading the group key; named in the log
    /// by the block's `name`, or else by the address it binds. With
    /// `cluster.peers`, its peering too.
    pub fn new(cfg: &Config) -> anyhow::Result<Arc<Gateway>> {
        let cluster: &ClusterConfig = cfg
            .cluster
            .as_ref()
            .ok_or_else(|| anyhow!("no `cluster:` block"))?;
        let key = GroupKey::load(&cluster.group_key_file)
            .with_context(|| "the cluster's group key".to_string())?;
        let name = cluster
            .name
            .clone()
            .unwrap_or_else(|| cfg.server.bind.to_string());
        let hub = Hub::new(key.clone(), name);
        hub.set_upgrade_config(UpgradeConfig {
            grace: Duration::from_secs(cluster.floor_grace_minutes * 60),
            canary_pct: cluster.canary_share.min(100),
        });
        let peering = peers::peering(cfg, &key, &hub)?;
        Ok(Arc::new(Gateway {
            hub,
            peering,
            acme: OnceLock::new(),
            sites: groups(cfg).into_iter().map(|g| g.site).collect(),
            router: OnceLock::new(),
            ticking: OnceLock::new(),
        }))
    }

    /// The ACME shared with the peers: set once, when the TLS entrance's
    /// renewal loop is built.
    pub fn set_acme(&self, acme: Arc<AcmePeers>) {
        let _ = self.acme.set(acme);
    }

    /// The Peers block (§12.2): links, dials and refusals, and the ACME state
    /// shared over them.
    pub fn peers_view(&self) -> Option<Value> {
        let peering = self.peering.as_ref()?;
        let mut view = serde_json::to_value(peering.view()).ok()?;
        if let Some(acme) = self.acme.get() {
            view["acme"] = serde_json::to_value(acme.view(&self.sites)).ok()?;
        }
        Some(view)
    }

    /// Judge upgrades once a second from now on (§3.8): started by the first
    /// member socket, on the runtime that serves it.
    pub fn tick_upgrades(&self) {
        if self.ticking.set(()).is_err() {
            return;
        }
        let hub = self.hub.clone();
        tokio::spawn(async move {
            let mut tick = interval(UPGRADE_TICK);
            tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                hub.upgrade_tick();
            }
        });
    }

    /// A gateway around `hub`, for tests that build the hub themselves.
    pub fn with_hub(hub: Arc<Hub>) -> Arc<Gateway> {
        Arc::new(Gateway {
            hub,
            peering: None,
            acme: OnceLock::new(),
            sites: Vec::new(),
            router: OnceLock::new(),
            ticking: OnceLock::new(),
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
