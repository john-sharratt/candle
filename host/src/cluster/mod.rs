//! A daemon as a cluster member (`docs/replicated_substrate.md` §3.2, §5):
//! with `--cluster`, each tenant's app dials every gateway named, and while
//! joined serves the requests they route here and forwards up the ones its own
//! listener takes.
//!
//! | Module | |
//! |---|---|
//! | [`serve`] | a request a gateway sent over a socket, served by this host's router |
//! | [`flush`] | its response ending only after the events it caused (§6.4) |
//! | [`in_flight`] | the requests still running, which a member leaving lets finish (§12.4) |
//! | [`listener`] | the own listener: forward up when joined, serve when cut off (§5.8) |
//! | [`sticky`] | claiming a sticky key before serving it (§5.4) |
//! | [`capacity`] | what `CAPACITY` reports (§5.3) |
//! | [`store`] | a tenant's substrate as replication drives it (§6, §7) |
//! | [`disk`] | the volume a substrate is on, for fault mode (§7.7) |
//! | [`leave`] | draining and sending before the process stops (§12.4) |
//! | [`up`] | requests between members, and the stamp naming the sender (§9.3) |
//! | [`upgrade`] | what the upgrades start from, and installing a build (§3.9) |
//! | [`view`] | the `cluster` block of `/v1/status`, and the `connecting` and `catchup` steps |
//!
//! The protocol — handshake, sessions, claims, reconnects, replication — is
//! the `cluster` crate's [`AppMember`]; one socket's connection — TLS, the
//! certificate's name, the WebSocket — is `cluster-dial`'s; this module is its
//! process.
//!
//! The sockets run on their own runtime. The HTTP runtime also serves turns,
//! and a worker blocked on the engine mutex — held across a scheduler wait
//! while compaction has the persistence lock — stops being polled. A reader
//! that is not polled for 45 s closes with "no frame from the gateway", both
//! apps together, because they share that runtime. Gateway frames are what
//! keep the socket admitted; they are read here, off that runtime. A request
//! the gateway opens is handed back to the HTTP runtime to be served.

pub mod capacity;
pub mod disk;
pub mod flush;
pub mod held_body;
pub mod in_flight;
pub mod leave;
pub mod listener;
pub mod serve;
pub mod sticky;
pub mod store;
pub mod up;
pub mod upgrade;
pub mod view;

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::sleep;
use std::time::{Duration, Instant};

use agent::checkouts::Checkouts;
use agent::config::AgentConfig;
use agent::jobs::Jobs;
use agent::member::{node_in, Agent, APP as AGENT_APP};
use axum::Router;
use candle_conversation::guest::reach;
use candle_conversation::phase_ring::{latest_wave, learned_rates, rates_read_every};
use cluster::dispatch::RouteTable;
use cluster::frame::{BuildId, ModelId, RepoAd};
use cluster::handshake::GroupKey;
use cluster::machine::measure;
use cluster::member::{AppMember, ClusterView, Identity, Link, CAPACITY_EVERY};
use cluster::replicate::{CatchupView, Phase, Replication, ReplicationView};
use cluster::upgrade::runtime::Upgrader;
use cluster::version::this_build;
use cluster_dial::{tls_connector, Dial, Target};
use tokio::runtime::{Handle, Runtime};
use tokio::time::interval;
use tokio_rustls::TlsConnector;
use zend_vfs::sandbox::programs::policy;
use zend_vfs::Priority;

use candle::gpu_poison::device_failing;
use capacity::{capacity, member_state, Rates, Wave};
use in_flight::InFlight;

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
    /// `--agent`: the in-process agent's file, its checkouts and slots.
    pub agent: Option<AgentConfig>,
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
        agent: Option<&Path>,
        host: String,
    ) -> anyhow::Result<Option<ClusterSettings>> {
        if urls.is_empty() {
            if key.is_some() {
                anyhow::bail!("--group-key names a cluster's key, and no --cluster is given");
            }
            // `--agent` with no cluster is no error: that agent runs in this
            // process alone, what a standalone daemon reads and writes its
            // repositories through (`docs/substrate_git.md` §9,
            // [`standalone_agent`]).
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
            agent: agent.map(AgentConfig::load).transpose()?,
        }))
    }
}

/// The process's membership: one [`AppLink`] per tenant.
pub struct Cluster {
    settings: ClusterSettings,
    targets: Vec<Arc<Target>>,
    tls: TlsConnector,
    build: BuildId,
    /// Serves gateway requests. Not the runtime the sockets live on.
    runtime: Handle,
    /// Reader, writer and session for every app socket. Kept alive for the
    /// process; dropped with a short timeout so a test that drops the cluster
    /// does not wait out a dial loop.
    sockets: Option<Runtime>,
    telemetry: Arc<Telemetry>,
    /// This host's whole router, which serves the requests gateways send.
    router: OnceLock<Router>,
    links: BTreeMap<&'static str, Arc<AppLink>>,
    /// The agent in this process, with `--agent` (§11.5).
    agent: Option<Arc<Jobs>>,
    /// The process's upgrades, across every app it serves (§3.8–§3.11).
    upgrader: Arc<Upgrader>,
    /// The requests gateways sent that are still running, which a member
    /// leaving lets finish before its sockets close.
    in_flight: InFlight,
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
    /// The repositories the tenant holds, for `HELLO` (§10.1).
    repos: OnceLock<Vec<RepoAd>>,
    /// Whether rendezvous may name this member (`Identity::leaseholder`); a
    /// leaseholder unless the tenant says otherwise before it joins.
    leaseholder: OnceLock<bool>,
}

impl Drop for Cluster {
    fn drop(&mut self) {
        self.close_all("stopped");
        if let Some(sockets) = self.sockets.take() {
            sockets.shutdown_timeout(Duration::from_secs(1));
        }
    }
}

impl Cluster {
    /// The membership of the tenants named `apps`, on `runtime`, upgrading
    /// through `upgrader`.
    pub fn new(
        settings: ClusterSettings,
        apps: &[&'static str],
        telemetry: Arc<Telemetry>,
        runtime: Handle,
        upgrader: Arc<Upgrader>,
    ) -> anyhow::Result<Arc<Cluster>> {
        let targets: Vec<Arc<Target>> = settings.targets.iter().cloned().map(Arc::new).collect();
        let tls = tls_connector()?;
        let agent = match &settings.agent {
            Some(cfg) => Some(start_agent(&settings, cfg, &targets, &tls, &runtime)?),
            None => None,
        };
        // Four workers. A session task can sit in one while the readers and
        // writers are still polled on the others.
        let sockets = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .thread_name("cluster-socket")
            .enable_all()
            .build()
            .map_err(|e| anyhow::anyhow!("could not start the cluster socket runtime: {e}"))?;
        // A watchdog on that runtime: a reader it does not poll for 45 s closes
        // both apps' sockets as "no frame from the gateway", and nothing else
        // says the runtime was the one that stalled.
        sockets.spawn(async {
            const EVERY: Duration = Duration::from_millis(500);
            let mut last = std::time::Instant::now();
            loop {
                tokio::time::sleep(EVERY).await;
                let late = last.elapsed().saturating_sub(EVERY);
                if late >= Duration::from_secs(2) {
                    tracing::warn!(
                        late_ms = late.as_millis() as u64,
                        "the cluster socket runtime was not polled on time"
                    );
                }
                last = std::time::Instant::now();
            }
        });
        Ok(Arc::new(Cluster {
            targets,
            agent,
            upgrader,
            tls,
            build: settings.build.clone(),
            runtime,
            in_flight: InFlight::default(),
            sockets: Some(sockets),
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
                            repos: OnceLock::new(),
                            leaseholder: OnceLock::new(),
                        }),
                    )
                })
                .collect(),
            settings,
        }))
    }

    fn sockets(&self) -> &Runtime {
        self.sockets
            .as_ref()
            .expect("the socket runtime is taken only when the cluster is dropped")
    }

    pub fn link(&self, app: &str) -> Option<&Arc<AppLink>> {
        self.links.get(app)
    }

    /// The agent in this process, with `--agent`.
    pub fn agent(&self) -> Option<&Arc<Jobs>> {
        self.agent.as_ref()
    }

    /// The process's upgrades.
    pub fn upgrader(&self) -> &Arc<Upgrader> {
        &self.upgrader
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

    /// This machine's name, as its `HELLO` gives it.
    pub fn host(&self) -> &str {
        &self.settings.host
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
        // The capacity frame below is the published rates' one reader: the
        // scheduler prices them as often as it reads them.
        rates_read_every(CAPACITY_EVERY);
        let member = AppMember::new(
            Identity {
                app: link.app.to_string(),
                node,
                host: self.settings.host.clone(),
                build: self.build.clone(),
                model,
                repos: link.repos.get().cloned().unwrap_or_default(),
                leaseholder: link.leaseholder.get().copied().unwrap_or(true),
            },
            self.settings.key.clone(),
            self.targets.iter().map(|t| t.url.clone()).collect(),
            routes,
            Box::new(move || {
                let state = member_state(ready.ready.load(Ordering::Relaxed), device_failing());
                let wave = latest_wave().map(|w| Wave {
                    age_s: w.age_s,
                    backlog: w.sample.backlog,
                    decodes: w.sample.decodes,
                    slots: w.sample.slots,
                    used_mib: w.sample.used_mib,
                    budget_mib: w.sample.budget_mib,
                });
                let rates = learned_rates().map(|(r, _)| Rates {
                    prefill_tok_s: r.prefill_tok_s,
                    decode_step_next_ms: r.decode_step_ms_plus_one,
                    decode_step_wide_ms: r.decode_step_ms_plus_wide,
                    prefill_tok_s_wide: r.prefill_tok_s_plus_wide,
                    prompt_tokens: r.prompt_tokens,
                    reply_tokens: r.reply_tokens,
                    tokens_per_step: r.tokens_per_step,
                    high_backlog_tokens: r.high_backlog_tokens,
                });
                capacity(
                    state,
                    telemetry.capacity(),
                    wave,
                    rates,
                    measure(),
                    reach::stated(),
                )
            }),
            replication,
            Instant::now(),
        );
        if link.member.set(member.clone()).is_err() {
            return;
        }
        self.upgrader.register(&member);
        for (c, target) in self.targets.iter().enumerate() {
            let dial = Dial::new(target.clone(), link.app, self.tls.clone());
            let cluster = self.clone();
            let member = member.clone();
            let replicator = member.replicator().cloned();
            let http = self.runtime.clone();
            self.sockets().spawn(async move {
                let stopped = member
                    .clone()
                    .run(c, dial, move |conn| {
                        let http = http.clone();
                        let cluster = cluster.clone();
                        let replicator = replicator.clone();
                        // Served on the HTTP runtime. The socket runtime only
                        // reads and writes frames; a handler that blocks must
                        // not stop it doing that.
                        http.spawn(async move {
                            if let Some(router) = cluster.router() {
                                let in_flight = cluster.in_flight.clone();
                                serve::serve(router, conn, c, replicator, in_flight);
                            } else {
                                conn.reset();
                            }
                        });
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

/// The agent of a daemon with no cluster: the agent file's checkouts and slots
/// in this process, joined to nothing. Its jobs — every fetch, push and command
/// — are the tenants' only way to their repositories (`docs/substrate_git.md`
/// §9). Low priority, as [`start_agent`]'s is.
pub fn standalone_agent(cfg: &AgentConfig) -> anyhow::Result<Arc<Jobs>> {
    let checkouts = Checkouts::open(cfg, &policy())?;
    tracing::info!("an agent in process, with no cluster");
    Ok(Jobs::new(
        checkouts,
        cfg.slots(measure().cores),
        Priority::Below,
    ))
}

/// `--agent`: open the agent file's checkouts and join every cluster's agents
/// on a socket of its own, beside the tenants' (§11.5). Its node is kept in
/// its jobs folder, apart from any tenant's substrate.
///
/// It is an agent at low priority: the host runs a model member too, so its
/// commands run below normal priority — a build never takes the processor the
/// decode needs — and the gateways send its jobs to it only when no dedicated
/// agent can take them.
fn start_agent(
    settings: &ClusterSettings,
    cfg: &AgentConfig,
    targets: &[Arc<Target>],
    tls: &TlsConnector,
    runtime: &Handle,
) -> anyhow::Result<Arc<Jobs>> {
    let checkouts = Checkouts::open(cfg, &policy())?;
    let jobs = Jobs::new(checkouts, cfg.slots(measure().cores), Priority::Below);
    let node = node_in(&cfg.jobs_dir)?;
    let _on = runtime.enter();
    let agent = Agent::new(
        node,
        settings.host.clone(),
        settings.key.clone(),
        targets.iter().map(|t| t.url.clone()).collect(),
        Arc::clone(&jobs),
    );
    for (c, target) in targets.iter().enumerate() {
        agent.dial(c, Dial::new(Arc::clone(target), AGENT_APP, tls.clone()));
    }
    tracing::info!(node = %format!("{node:016x}"), "joined the cluster's agents in process");
    Ok(jobs)
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
    /// Wait, at most `limit`, for the requests gateways sent that are still
    /// running — a turn streaming its reply among them — telling `waiting` how
    /// many are left. Closing the sockets under them would cut each one
    /// mid-stream. Returns how many were still running.
    pub async fn settle_requests(&self, limit: Duration, waiting: impl FnMut(usize)) -> usize {
        self.in_flight.settle(limit, waiting).await
    }

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

    /// The repositories the tenant holds, before it joins: its `HELLO`
    /// advertises them, and `Sticky` and `Best` consider only members holding
    /// a conversation's repositories (§10.1).
    pub fn set_repos(&self, repos: Vec<RepoAd>) {
        let _ = self.repos.set(repos);
    }

    /// Whether rendezvous may name this member, before it joins: `false` keeps
    /// it out of every lease while it replicates and serves (§9.5).
    pub fn set_leaseholder(&self, leaseholder: bool) {
        if self.member.get().is_some() || self.leaseholder.set(leaseholder).is_err() {
            tracing::error!(
                app = self.app,
                leaseholder,
                "a member's leaseholding was set after it was admitted, or twice — its admission \
                 stands as it was"
            );
        }
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

    /// A key for `purpose` that every member of the group derives alike
    /// ([`GroupKey::derive`]): what a tenant signs something with that any
    /// member must accept.
    pub fn derive_key(&self, purpose: &str) -> [u8; 32] {
        self.cluster.settings.key.derive(purpose)
    }

    /// The agent in this process, with `--agent`: what the tenant's jobs go
    /// to while it is cut off from every cluster (§11.5).
    pub fn in_process_agent(&self) -> Option<Arc<Jobs>> {
        self.cluster.agent().cloned()
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
            if ready || cancelled() {
                return;
            }
            // Cut off before catching up: the load goes on detached, and the
            // catch-up resumes once a gateway admits it again. Said, because
            // the step reads done either way.
            if self.link.state() == Link::CutOff {
                let catchups = r.catchups();
                tracing::warn!(
                    behind = %view::catchup_detail(&catchups),
                    "cut off before catching up — loading on detached; the catch-up resumes \
                     once rejoined"
                );
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
