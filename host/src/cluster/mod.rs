//! A daemon as a cluster member (`docs/replicated_substrate.md` §3.2, §5):
//! with `--cluster`, each tenant's app dials every gateway named, and while
//! joined serves the requests they route here and forwards up the ones its own
//! listener takes.
//!
//! | Module | |
//! |---|---|
//! | [`dial`] | one socket's connection: TLS, the certificate's name, the WebSocket |
//! | [`serve`] | a request a gateway sent over a socket, served by this host's router |
//! | [`flush`] | its response ending only after the events it caused (§6.4) |
//! | [`listener`] | the own listener: forward up when joined, serve when cut off (§5.8) |
//! | [`sticky`] | claiming a sticky key before serving it (§5.4) |
//! | [`capacity`] | what `CAPACITY` reports (§5.3) |
//! | [`store`] | a tenant's substrate as replication drives it (§6, §7) |
//! | [`disk`] | the volume a substrate is on, for fault mode (§7.7) |
//! | [`leave`] | draining and sending before the process stops (§12.4) |
//! | [`up`] | requests between members, and the stamp naming the sender (§9.3) |
//! | [`view`] | the `cluster` block of `/v1/status`, and the `connecting` and `catchup` steps |
//!
//! The protocol — handshake, sessions, claims, reconnects, replication — is
//! the `cluster` crate's [`AppMember`]; this module is its process.

pub mod capacity;
pub mod dial;
pub mod disk;
pub mod flush;
pub mod leave;
pub mod listener;
pub mod serve;
pub mod sticky;
pub mod store;
pub mod up;
pub mod view;

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::{available_parallelism, sleep};
use std::time::{Duration, Instant};

use axum::Router;
use candle_conversation::phase_ring::latest_wave;
use cluster::dispatch::RouteTable;
use cluster::frame::{BuildId, MemberState, ModelId};
use cluster::handshake::GroupKey;
use cluster::member::{AppMember, ClusterView, Identity, Link};
use cluster::replicate::{CatchupView, Phase, Replication, ReplicationView};
use cluster::version::this_build;
use sysinfo::System;
use tokio::runtime::Handle;
use tokio::time::interval;
use tokio_rustls::TlsConnector;

use capacity::{capacity, Machine, Wave};
use dial::{tls_connector, Dial, Target};

use crate::telemetry::Telemetry;
use crate::tenant::Reports;

/// How often a member lets go of the claims it has not served for its idle
/// timeout.
const RELEASE_EVERY: Duration = Duration::from_secs(30);

/// How often the loading step's detail is refreshed while it waits.
const CONNECTING_POLL: Duration = Duration::from_millis(250);

/// `--cluster` and its company, as parsed.
#[derive(Debug, Clone)]
pub struct ClusterSettings {
    pub targets: Vec<Target>,
    pub key: GroupKey,
    /// How long a claim lives unserved (`--cluster-affinity-idle`).
    pub affinity_idle: Duration,
    /// `max_members` to write as a set at startup (`--cluster-quorum-max`).
    pub quorum_max: Option<u32>,
    /// This machine's name in `HELLO` and on the status page.
    pub host: String,
    /// The build this member reports: this binary's ([`this_build`]).
    pub build: BuildId,
}

impl ClusterSettings {
    /// From the flags: every `--cluster`, verified by `--cluster-name` where it
    /// names an address, keyed by `--group-key`.
    pub fn parse(
        urls: &[String],
        name: Option<&str>,
        key: Option<&Path>,
        affinity_idle: Duration,
        quorum_max: Option<u32>,
        host: String,
    ) -> anyhow::Result<Option<ClusterSettings>> {
        if urls.is_empty() {
            if key.is_some() {
                anyhow::bail!("--group-key names a cluster's key, and no --cluster is given");
            }
            if quorum_max.is_some() {
                anyhow::bail!(
                    "--cluster-quorum-max sets a cluster's quorum, and no --cluster is given"
                );
            }
            return Ok(None);
        }
        if quorum_max == Some(0) {
            anyhow::bail!("--cluster-quorum-max is at least 1: a group has at least this member");
        }
        let Some(key) = key else {
            anyhow::bail!("--cluster needs --group-key: the key a member proves it holds");
        };
        let targets = urls
            .iter()
            .map(|u| Target::parse(u, name))
            .collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Some(ClusterSettings {
            targets,
            key: GroupKey::load(key)?,
            affinity_idle,
            quorum_max,
            host,
            build: this_build(),
        }))
    }
}

/// The process's membership: one [`AppLink`] per tenant.
pub struct Cluster {
    settings: ClusterSettings,
    targets: Vec<Arc<Target>>,
    tls: TlsConnector,
    build: BuildId,
    runtime: Handle,
    telemetry: Arc<Telemetry>,
    /// This host's whole router, which serves the requests gateways send.
    router: OnceLock<Router>,
    links: BTreeMap<&'static str, Arc<AppLink>>,
}

/// One tenant's app: unjoined until the tenant opens its substrate and
/// [`connects`](Self::connect) with its node and model.
pub struct AppLink {
    app: &'static str,
    /// The app's dispatch table: the host's routes and this tenant's.
    routes: OnceLock<RouteTable>,
    /// Whether the tenant has finished loading — `CAPACITY`'s state.
    ready: AtomicBool,
    member: OnceLock<Arc<AppMember>>,
    /// The `oseq` of the last record the tenant's substrate wrote, for
    /// `UNCLAIM`'s `through`.
    own_oseq: OnceLock<Box<dyn Fn() -> u64 + Send + Sync>>,
}

impl Cluster {
    /// The membership of the tenants named `apps`, on `runtime`.
    pub fn new(
        settings: ClusterSettings,
        apps: &[&'static str],
        telemetry: Arc<Telemetry>,
        runtime: Handle,
    ) -> anyhow::Result<Arc<Cluster>> {
        let targets = settings.targets.iter().cloned().map(Arc::new).collect();
        Ok(Arc::new(Cluster {
            targets,
            tls: tls_connector()?,
            build: settings.build.clone(),
            runtime,
            telemetry,
            router: OnceLock::new(),
            links: apps
                .iter()
                .map(|app| {
                    (
                        *app,
                        Arc::new(AppLink {
                            app,
                            routes: OnceLock::new(),
                            ready: AtomicBool::new(false),
                            member: OnceLock::new(),
                            own_oseq: OnceLock::new(),
                        }),
                    )
                })
                .collect(),
            settings,
        }))
    }

    pub fn link(&self, app: &str) -> Option<&Arc<AppLink>> {
        self.links.get(app)
    }

    pub fn links(&self) -> impl Iterator<Item = &Arc<AppLink>> {
        self.links.values()
    }

    pub fn targets(&self) -> &[Arc<Target>] {
        &self.targets
    }

    pub fn build(&self) -> &BuildId {
        &self.build
    }

    /// The router gateway requests are served by, once the server built it.
    pub fn set_router(&self, router: Router) {
        let _ = self.router.set(router);
    }

    fn router(&self) -> Option<Router> {
        self.router.get().cloned()
    }

    /// Start `link`'s sockets as `node`, its substrate bound to `model` and
    /// replicated through `replication`.
    fn start(
        self: &Arc<Self>,
        link: &Arc<AppLink>,
        node: u64,
        model: Option<ModelId>,
        replication: Option<Replication>,
    ) {
        let routes = loop {
            if let Some(r) = link.routes.get() {
                break r.clone();
            }
            sleep(CONNECTING_POLL);
        };
        let telemetry = self.telemetry.clone();
        let ready = link.clone();
        let cores = available_parallelism().map_or(1, |n| n.get()) as u16;
        let member = AppMember::new(
            Identity {
                app: link.app.to_string(),
                node,
                host: self.settings.host.clone(),
                build: self.build.clone(),
                model,
                repos: Vec::new(),
            },
            self.settings.key.clone(),
            self.targets.iter().map(|t| t.url.clone()).collect(),
            routes,
            Box::new(move || {
                let state = if ready.ready.load(Ordering::Relaxed) {
                    MemberState::Ready
                } else {
                    MemberState::Loading
                };
                let wave = latest_wave().map(|w| Wave {
                    age_s: w.age_s,
                    backlog: w.sample.backlog,
                    decodes: w.sample.decodes,
                    slots: w.sample.slots,
                    used_mib: w.sample.used_mib,
                    budget_mib: w.sample.budget_mib,
                });
                let load = System::load_average();
                capacity(
                    state,
                    telemetry.capacity(),
                    wave,
                    Machine {
                        cores,
                        load_milli: (load.one * 1000.0) as u32,
                    },
                )
            }),
            replication,
            Instant::now(),
        );
        if link.member.set(member.clone()).is_err() {
            return;
        }
        for (c, target) in self.targets.iter().enumerate() {
            let dial = Dial {
                target: target.clone(),
                app: link.app,
                tls: self.tls.clone(),
            };
            let cluster = self.clone();
            let member = member.clone();
            let replicator = member.replicator().cloned();
            self.runtime.spawn(async move {
                let stopped = member
                    .clone()
                    .run(c, dial, move |conn| {
                        if let Some(router) = cluster.router() {
                            serve::serve(router, conn, c, replicator.clone());
                        } else {
                            conn.reset();
                        }
                    })
                    .await;
                tracing::warn!(
                    app = member.app(),
                    cluster = c,
                    "cluster socket stopped: {stopped:?}"
                );
            });
        }
        let idle = self.settings.affinity_idle;
        let link = link.clone();
        self.runtime.spawn(async move {
            let mut tick = interval(RELEASE_EVERY);
            loop {
                tick.tick().await;
                let through = link.own_oseq.get().map_or(0, |f| f());
                if let Some(m) = link.member.get() {
                    let released = m.release_idle(idle, through);
                    if released > 0 {
                        tracing::info!(app = m.app(), released, "released idle claims");
                    }
                }
            }
        });
    }
}

/// Every second, tell each app's gateways whether its tenant is ready — the
/// state `CAPACITY` carries, and what makes the app clustered.
pub fn watch_readiness(tenants: Vec<(Arc<AppLink>, Arc<dyn Reports>)>) {
    tokio::spawn(async move {
        let mut tick = interval(Duration::from_secs(1));
        loop {
            tick.tick().await;
            for (link, reports) in &tenants {
                link.set_ready(reports.status().await.state == "ready");
            }
        }
    });
}

impl Cluster {
    /// The process is stopping: every app's replication stops and its sockets
    /// close `1001`, before the tenants release their substrates.
    pub fn close_all(&self, reason: &str) {
        for link in self.links.values() {
            if let Some(m) = link.member() {
                m.shutdown(reason);
            }
        }
    }
}

impl AppLink {
    pub fn app(&self) -> &'static str {
        self.app
    }

    /// The app's dispatch table, once the host has merged its routes.
    pub fn set_routes(&self, routes: RouteTable) {
        let _ = self.routes.set(routes);
    }

    /// The tenant finished loading, or went back to it: `CAPACITY` says so at
    /// once.
    pub fn set_ready(&self, ready: bool) {
        let was = self.ready.swap(ready, Ordering::Relaxed);
        if was != ready {
            if let Some(m) = self.member.get() {
                m.capacity_changed();
            }
        }
    }

    pub fn member(&self) -> Option<&Arc<AppMember>> {
        self.member.get()
    }

    /// Where the app stands; `Starting` until the tenant connects.
    pub fn state(&self) -> Link {
        self.member.get().map_or(Link::Starting, |m| m.link())
    }

    /// Where its sockets stand, whatever its replication's phase.
    pub fn socket_state(&self) -> Link {
        self.member
            .get()
            .map_or(Link::Starting, |m| m.socket_link())
    }

    /// Its replication, as `/v1/status` shows it.
    pub fn replication(&self) -> Option<ReplicationView> {
        self.member.get()?.replicator().map(|r| r.view())
    }

    /// Every cluster's socket, for the loading step and `/v1/status`.
    pub fn clusters(&self) -> Vec<ClusterView> {
        self.member.get().map(|m| m.clusters()).unwrap_or_default()
    }
}

/// What a tenant joins its app's clusters through, from the loader thread
/// that opened its substrate.
#[derive(Clone)]
pub struct Join {
    cluster: Arc<Cluster>,
    link: Arc<AppLink>,
}

impl fmt::Debug for Join {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Join").field("app", &self.link.app).finish()
    }
}

impl Join {
    pub fn new(cluster: Arc<Cluster>, app: &str) -> Option<Join> {
        let link = cluster.link(app)?.clone();
        Some(Join { cluster, link })
    }

    pub fn link(&self) -> &Arc<AppLink> {
        &self.link
    }

    /// Dial every cluster as `node`, the substrate bound to `model` and
    /// replicated through `replication`, and wait — the `connecting` loading
    /// step (§7.6) — until the app is joined, cut off, or stopped by a
    /// terminal refusal, or `cancelled` says the process is stopping. Each poll
    /// shows `progress` every cluster's socket.
    ///
    /// A stopped app is the build itself being wrong: the tenant stays on its
    /// loading page with the reason ([`Join::hold`]) rather than serving.
    pub fn connect(
        &self,
        node: u64,
        model: Option<ModelId>,
        replication: Replication,
        own_oseq: impl Fn() -> u64 + Send + Sync + 'static,
        cancelled: impl Fn() -> bool,
        mut progress: impl FnMut(&[ClusterView]),
    ) -> Link {
        let _ = self.link.own_oseq.set(Box::new(own_oseq));
        self.cluster
            .start(&self.link, node, model, Some(replication));
        let replicator = self.link.member().and_then(|m| m.replicator().cloned());
        if let (Some(n), Some(r)) = (self.cluster.settings.quorum_max, replicator) {
            tracing::warn!(
                app = self.link.app,
                max_members = n,
                "--cluster-quorum-max: writing the group's max_members lower"
            );
            self.cluster
                .runtime
                .spawn(async move { r.set_quorum(n).await });
        }
        loop {
            progress(&self.link.clusters());
            let link = self.link.socket_state();
            if let Link::Stopped(close) = &link {
                tracing::error!(
                    app = self.link.app,
                    "the cluster refused this build: {close} — serving nothing until the binary \
                     is replaced"
                );
            }
            if link != Link::Starting || cancelled() {
                return link;
            }
            sleep(CONNECTING_POLL);
        }
    }

    /// The `catchup` loading step (§7.6): wait until a socket's admission
    /// catch-up has finished and any rejoin is done, or the app is cut off —
    /// it then serves locally — or `cancelled` says the process is stopping.
    /// Each poll shows `progress` the catch-ups running.
    pub fn catch_up(&self, cancelled: impl Fn() -> bool, mut progress: impl FnMut(&[CatchupView])) {
        let Some(r) = self.link.member().and_then(|m| m.replicator().cloned()) else {
            return;
        };
        loop {
            progress(&r.catchups());
            let ready = r.caught_up() && r.phase() == Phase::Joined;
            if ready || self.link.state() == Link::CutOff || cancelled() {
                return;
            }
            sleep(CONNECTING_POLL);
        }
    }

    /// After a terminal refusal: keep showing every cluster's socket until
    /// `cancelled` says the process is stopping.
    pub fn hold(&self, cancelled: impl Fn() -> bool, mut progress: impl FnMut(&[ClusterView])) {
        while !cancelled() {
            progress(&self.link.clusters());
            sleep(CONNECTING_POLL);
        }
    }
}
