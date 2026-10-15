//! A daemon's repositories kept with its log (`docs/substrate_git.md` §5, §7,
//! §8): loaded in full during its loading phase, then followed.
//!
//! [`Repos::open`] opens a store per repository. With `--cluster`, each store
//! fetches through the cluster's agents, and the lease `git/<repo>` is held
//! through its consensus; standalone, each fetches from its checkout's own
//! `origin`, whose branches are adopted once.
//!
//! [`Repos::load`] blocks until every repository is whole in the store: the
//! log names its branches and every pack they need has arrived and been
//! taken. A repository no member has pulled yet is pulled now, through an
//! agent, by the member holding its lease; every other member waits for it to
//! replicate. `max_depth` and every other bound on what is *read into
//! conversations* play no part here: the store holds every object of every
//! branch.
//!
//! [`Repos::follow`] then keeps them on a thread: every tick publishes this
//! member's pushes and syncs; the lease holder fetches every [`FETCH_EVERY`].

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use agent::client::Agents;
use agent::fetcher::AgentFetch;
use candle_conversation::persistence::SharedSubstrate;
use cluster::replicate::Replicator;
use tokio::runtime::Handle;
use zend_vfs::{fetch_here, Oid, RefName, RemoteUrl, Repo};

use crate::import::Importer;
use crate::records::{Alone, Lease, Log};
use crate::store::Store;
use crate::adopt;

/// How often the follow thread ticks: publishes and syncs, all local.
pub const TICK: Duration = Duration::from_secs(2);

/// How often the lease holder fetches origin: a job on an agent, and a request
/// to origin.
pub const FETCH_EVERY: Duration = Duration::from_secs(15);

/// How long the import lease lives unrenewed, and how often it is renewed.
const LEASE_TTL: Duration = Duration::from_secs(120);
const LEASE_RENEW: Duration = Duration::from_secs(30);

/// How long a load waits between rounds while a repository is incomplete.
const LOAD_ROUND: Duration = Duration::from_secs(1);

/// What the cluster gives the importers: the consensus the lease is held
/// through, and the agents fetches are made by, on the runtime their sockets
/// live on.
pub struct Cluster {
    pub replicator: Arc<Replicator>,
    pub agents: Agents,
    pub runtime: Handle,
}

/// The import lease, held through the cluster's consensus.
struct ClusterLease {
    replicator: Arc<Replicator>,
    renewed: Mutex<HashMap<String, Instant>>,
}

impl Lease for ClusterLease {
    fn hold(&self, key: &str) -> bool {
        let mut renewed = self.renewed.lock().unwrap();
        let fresh = renewed
            .get(key)
            .is_some_and(|at| at.elapsed() < LEASE_RENEW);
        if fresh && self.replicator.holds(key.as_bytes()) {
            return true;
        }
        let held = self.replicator.acquire(key.as_bytes(), LEASE_TTL);
        if held {
            renewed.insert(key.to_string(), Instant::now());
        } else {
            renewed.remove(key);
        }
        held
    }
}

/// Where a repository loads from: its name, and for a daemon without a
/// cluster the checkout whose `origin` it fetches and whose branches it adopts.
pub struct Source {
    pub name: String,
    pub checkout: Option<PathBuf>,
}

/// One repository's state during a load.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Loading {
    pub repo: String,
    /// Packs held of those the log names.
    pub held: usize,
    pub listed: usize,
    /// Whether the store holds every branch where the log does.
    pub whole: bool,
    /// Why a pull failed, while it does.
    pub error: Option<String>,
}

type OnChange = Arc<dyn Fn(&str) + Send + Sync>;

struct Inner {
    /// Each store, its importer, and whether it has anywhere to pull from.
    repos: Vec<(Store, Importer, bool)>,
    shared: SharedSubstrate,
    on_change: Mutex<Option<OnChange>>,
}

/// A daemon's repositories in the substrate.
#[derive(Clone)]
pub struct Repos {
    inner: Arc<Inner>,
}

impl Repos {
    /// Open a store under `root` for every source, fetching through `cluster`'s
    /// agents when given, else from each checkout's `origin`. A checkout's
    /// branches are adopted only without a cluster: with one, the agents'
    /// view of origin is the only source, so a stale local clone never leaks
    /// in.
    pub fn open(
        root: &Path,
        sources: &[Source],
        shared: SharedSubstrate,
        cluster: Option<Cluster>,
    ) -> anyhow::Result<Self> {
        let lease: Arc<dyn Lease> = match &cluster {
            Some(c) => Arc::new(ClusterLease {
                replicator: Arc::clone(&c.replicator),
                renewed: Mutex::new(HashMap::new()),
            }),
            None => Arc::new(Alone),
        };
        let mut repos = Vec::new();
        for source in sources {
            let name = &source.name;
            let mut store = Store::open(root, name)?;
            let checkout = source
                .checkout
                .as_deref()
                .filter(|dir| dir.join(".git").exists())
                .and_then(|dir| Repo::open(dir).ok());
            let mut pullable = true;
            store = match (&cluster, &checkout) {
                (Some(c), _) => store.with_origin(
                    &RemoteUrl::parse(&format!("agent:{name}"))?,
                    AgentFetch::shared(c.agents.clone(), c.runtime.clone(), name.clone()),
                )?,
                (None, Some(repo)) => {
                    match repo.remotes()?.into_iter().find(|r| r.name.as_str() == "origin") {
                        Some(origin) => store
                            .with_origin(&RemoteUrl::parse(&origin.fetch_url)?, fetch_here())?,
                        None => {
                            pullable = false;
                            store
                        }
                    }
                }
                (None, None) => {
                    pullable = false;
                    store
                }
            };
            if cluster.is_none() {
                if let Some(repo) = &checkout {
                    match adopt(&shared, name, repo) {
                        Ok(p) if !p.refs.is_empty() => tracing::info!(
                            repo = %name, packs = p.packs, branches = p.refs.len(),
                            "adopted the checkout's branches into the substrate"
                        ),
                        Ok(_) => {}
                        Err(e) => tracing::warn!(repo = %name, "adopting the checkout: {e:#}"),
                    }
                }
            }
            let importer = Importer::new(Arc::clone(&lease)).fetching_every(FETCH_EVERY);
            repos.push((store, importer, pullable));
        }
        Ok(Self {
            inner: Arc::new(Inner {
                repos,
                shared,
                on_change: Mutex::new(None),
            }),
        })
    }

    /// Block until every repository is whole in its store, reporting each
    /// round to `report`. A repository the log names no branch of is pulled
    /// now by whichever member holds its lease. Returns early, incomplete,
    /// when `cancelled` says so.
    pub fn load(
        &self,
        cancelled: &dyn Fn() -> bool,
        report: &dyn Fn(&[Loading]),
    ) -> anyhow::Result<()> {
        let mut state: Vec<Loading> = self
            .inner
            .repos
            .iter()
            .map(|(s, _, _)| Loading {
                repo: s.name().to_string(),
                ..Loading::default()
            })
            .collect();
        loop {
            for ((store, importer, pullable), loading) in self.inner.repos.iter().zip(state.iter_mut()) {
                if loading.whole {
                    continue;
                }
                let named = !branches(&self.inner.shared, store.name())?.is_empty();
                let ticked = if named {
                    importer.tick(store, &self.inner.shared)
                } else {
                    importer.pull(store, &self.inner.shared)
                };
                match ticked {
                    Ok(t) => {
                        loading.held = t.synced.held;
                        loading.listed = t.synced.listed;
                        loading.error = t.fetch_error;
                        // Nowhere to pull from and nothing in the log: an empty
                        // repository, whole as it is.
                        loading.whole = whole(store, &self.inner.shared)?
                            || (!*pullable && !named);
                    }
                    Err(e) => loading.error = Some(format!("{e:#}")),
                }
            }
            report(&state);
            if state.iter().all(|l| l.whole) || cancelled() {
                return Ok(());
            }
            thread::sleep(LOAD_ROUND);
        }
    }

    /// Call `f` with a repository's name whenever one of its record branches
    /// moves.
    pub fn on_change(&self, f: OnChange) {
        *self.inner.on_change.lock().unwrap() = Some(f);
    }

    /// The store of repository `name`.
    pub fn store(&self, name: &str) -> Option<&Store> {
        self.inner
            .repos
            .iter()
            .map(|(s, _, _)| s)
            .find(|s| s.name() == name)
    }

    /// Keep every repository with the log and with origin on a thread of its
    /// own, until `stopped` says so.
    pub fn follow(&self, stopped: fn() -> bool) -> anyhow::Result<()> {
        let inner = Arc::clone(&self.inner);
        thread::Builder::new()
            .name("substrate-git".into())
            .spawn(move || follow(inner, stopped))?;
        Ok(())
    }
}

/// The log's branches of `repo`, the deleted left out.
pub fn branches(shared: &SharedSubstrate, repo: &str) -> anyhow::Result<BTreeMap<RefName, Oid>> {
    Log::new(shared, repo)?
        .refs()?
        .into_iter()
        .filter(|(name, _)| name.starts_with("refs/heads/"))
        .filter_map(|(name, e)| Some((name, e.oid?)))
        .map(|(name, oid)| Ok((RefName::parse(&name)?, Oid::parse(&oid)?)))
        .collect()
}

/// Whether `store` holds every branch the log names, where the log has it.
fn whole(store: &Store, shared: &SharedSubstrate) -> anyhow::Result<bool> {
    let log = branches(shared, store.name())?;
    Ok(!log.is_empty() && store.record()? == log)
}

fn follow(inner: Arc<Inner>, stopped: fn() -> bool) {
    let mut last_error: HashMap<String, String> = HashMap::new();
    loop {
        thread::sleep(TICK);
        if stopped() {
            return;
        }
        for (store, importer, _) in &inner.repos {
            let name = store.name().to_string();
            let error = match importer.tick(store, &inner.shared) {
                Ok(t) => {
                    if !t.synced.moved.is_empty() {
                        if let Some(f) = inner.on_change.lock().unwrap().clone() {
                            f(&name);
                        }
                    }
                    t.fetch_error
                }
                Err(e) => Some(format!("{e:#}")),
            };
            match error {
                Some(e) if last_error.get(&name) != Some(&e) => {
                    tracing::warn!(repo = %name, "keeping the repository with the substrate: {e}");
                    last_error.insert(name, e);
                }
                Some(_) => {}
                None => {
                    if last_error.remove(&name).is_some() {
                        tracing::info!(repo = %name, "repository reachable again");
                    }
                }
            }
        }
    }
}
