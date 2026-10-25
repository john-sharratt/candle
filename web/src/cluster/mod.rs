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
//! | [`bearer_key`] | the keys that stand in for a session: status, ops |
//! | [`ops_key`] | the admin an operator on the box acts as, for emergencies |
//! | [`peers`] | the peer gateways: node id, identity, dial loops (§13.4) |
//! | [`upgrade`] | the socket's WebSocket upgrade, over a transport that stamps its reads |
//! | [`upgrades`] | the gateway's upgrade commands, the forced rollout's plan |
//!
//! The state itself — admission, members, claims, the log — is the `cluster`
//! crate's [`Hub`]; this module is its HTTP.

pub mod aggregate;
pub mod bearer_key;
pub mod endpoint;
pub mod jobs;
pub mod keeps;
pub mod load;
pub mod merge_ws;
pub mod mirror;
pub mod ops_key;
pub mod page;
pub mod peers;
pub mod route;
pub mod streams;
pub mod transport;
pub mod tunnel;
pub mod upgrade;
pub mod upgrades;

use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{anyhow, Context};
use axum::http::HeaderValue;
use axum::response::Response;
use axum::Router;
use cluster::consensus::votes::{FileJournal, DEFAULT_SIZE};
use cluster::gateway::upgrades::UpgradeConfig;
use cluster::gateway::Hub;
use cluster::handshake::GroupKey;
use cluster::peer::Peering;
use serde_json::Value;
use tokio::time::{interval, MissedTickBehavior};

use crate::acme::names::groups;
use crate::acme::peers::AcmePeers;
use crate::auth::forwarded::identify;
use crate::auth::{Auth, Identity};
use crate::config::{Cluster as ClusterConfig, Config};
use axum::http::HeaderMap;
use bearer_key::BearerKey;
use ops_key::OpsKey;

/// How often the upgrades are judged: verdicts due, members left behind.
const UPGRADE_TICK: Duration = Duration::from_secs(1);

/// The header naming who answered a request: `<host> <node>`, the machine's
/// name and its 16-hex node, the node as `x-tokera-holder` writes one. A
/// member sets it on everything its router answers; a gateway passes a
/// member's through untouched and names itself only on what it answers alone.
pub const SERVED_BY_HEADER: &str = "x-served-by";

/// The [`SERVED_BY_HEADER`] value for `host`, and `node` once there is one.
pub fn served_by(host: &str, node: Option<u64>) -> Option<HeaderValue> {
    let value = match node {
        Some(node) => format!("{host} {node:016x}"),
        None => host.to_string(),
    };
    HeaderValue::from_str(&value).ok()
}

/// Name `value` as who served `res`, unless whoever answered it already did:
/// a response that came back through a gateway keeps the member's name.
pub fn stamp_served_by(res: &mut Response, value: Option<HeaderValue>) {
    if res.headers().contains_key(SERVED_BY_HEADER) {
        return;
    }
    if let Some(v) = value {
        res.headers_mut().insert(SERVED_BY_HEADER, v);
    }
}

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
    /// This gateway's node: its peering's, or the one kept beside its key.
    pub node: u64,
    /// The peer gateways answering the same names (§13.4), when
    /// `cluster.peers` names any.
    pub peering: Option<Arc<Peering>>,
    /// The key that reads the status without a session, when
    /// `cluster.status_key_file` names one.
    pub status_key: Option<BearerKey>,
    /// The key an operator acts as an admin with, without a session, when
    /// `cluster.ops_key` names one ([`OpsKey`]).
    pub ops_key: Option<OpsKey>,
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
        let (voter, peer_n) = match &peering {
            Some(p) => (p.node(), cluster.peers.len()),
            None => (peers::local_node(&cluster.group_key_file)?, 0),
        };
        let journal = cluster.group_key_file.with_file_name("gateway-votes");
        hub.enable_tiebreak(
            voter,
            peer_n,
            Box::new(
                FileJournal::open(&journal, DEFAULT_SIZE)
                    .with_context(|| format!("opening {}", journal.display()))?,
            ),
        )?;
        let status_key = cluster
            .status_key_file
            .as_deref()
            .map(BearerKey::load)
            .transpose()?;
        let ops_key = cluster
            .ops_key
            .as_ref()
            .map(|o| OpsKey::load(&o.file, &o.admin))
            .transpose()?;
        Ok(Arc::new(Gateway {
            hub,
            node: voter,
            peering,
            status_key,
            ops_key,
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
        Gateway::with_hub_and_keys(hub, None, None)
    }

    /// [`Self::with_hub`], with its status key and its ops key.
    pub fn with_hub_and_keys(
        hub: Arc<Hub>,
        status_key: Option<BearerKey>,
        ops_key: Option<OpsKey>,
    ) -> Arc<Gateway> {
        Arc::new(Gateway {
            hub,
            node: 0,
            peering: None,
            status_key,
            ops_key,
            acme: OnceLock::new(),
            sites: Vec::new(),
            router: OnceLock::new(),
            ticking: OnceLock::new(),
        })
    }

    /// Who a request to one of this gateway's own routes is: its session, the
    /// ops key's admin when it presents that key ([`OpsKey`]), or — on a
    /// request a member sent up its socket, `origin` — the identity the member
    /// states, believed as the site routes believe it, since the member proved
    /// the group key and vouches for its caller by the same rules.
    pub fn caller(
        &self,
        auth: Option<&Auth>,
        headers: &HeaderMap,
        origin: Option<&MemberOrigin>,
    ) -> Option<Identity> {
        auth.and_then(|a| a.identity(headers))
            .or_else(|| self.ops_key.as_ref().and_then(|k| k.caller(headers)))
            .or_else(|| origin.and_then(|_| identify(headers).ok()))
    }

    pub fn set_router(&self, router: Router) {
        // Set once, when the server's router is built.
        let _ = self.router.set(router);
    }

    fn router(&self) -> Option<Router> {
        self.router.get().cloned()
    }

    /// The [`SERVED_BY_HEADER`] this gateway puts on what it answers itself.
    pub fn served_by(&self) -> Option<HeaderValue> {
        served_by(self.hub.name(), Some(self.node))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn served_by_names_the_machine_and_its_node() {
        assert_eq!(
            served_by("whiteworld", Some(0x31f6_62ee_0edb_ae23)).unwrap(),
            "whiteworld 31f662ee0edbae23"
        );
        assert_eq!(served_by("whiteworld", None).unwrap(), "whiteworld");
    }

    /// **A member's stated identity is believed on its own socket only**: the
    /// same headers from anyone else name nobody.
    #[test]
    fn a_member_states_its_callers_identity_over_its_socket_only() {
        let gateway = Gateway::with_hub(Hub::new(GroupKey::from_bytes([7; 32]), "g"));
        let mut headers = HeaderMap::new();
        headers.insert("x-tokera-user", HeaderValue::from_static("u1"));
        headers.insert("x-tokera-provider", HeaderValue::from_static("google"));
        let origin = MemberOrigin {
            app: "zend".into(),
            node: 1,
        };
        assert_eq!(
            gateway.caller(None, &headers, Some(&origin)).map(|i| i.sub),
            Some("u1".to_string())
        );
        assert_eq!(gateway.caller(None, &headers, None), None);
    }

    #[test]
    fn a_members_name_is_kept_over_the_gateways() {
        let mut res = Response::new(axum::body::Body::empty());
        stamp_served_by(&mut res, served_by("blackworld", Some(1)));
        stamp_served_by(&mut res, served_by("dmz", Some(2)));
        assert_eq!(
            res.headers()[SERVED_BY_HEADER],
            "blackworld 0000000000000001"
        );
    }
}
