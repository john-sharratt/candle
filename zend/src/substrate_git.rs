//! zend's repositories read from the substrate (`docs/substrate_git.md` §7).
//!
//! With `substrate_git: true`, each repository `workspace.yaml` names is read
//! from the object store derived from the log, `<workspace>/substrate/git/<name>`,
//! instead of a folder under the workspace. The repository keeps its name, so
//! every tool, path and ingest key is unchanged; only the folder behind it
//! moves. A member then needs no checkouts: its stores fill from the log.
//!
//! One thread keeps every store with the log and with `origin`
//! ([`substrate_git::Importer`]): it publishes this member's pushes, fetches
//! through the cluster's agents on the member holding the repository's lease,
//! and syncs. A record branch that moves wakes the ingest worker.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use agent::client::Agents;
use agent::fetcher::AgentFetch;
use candle_conversation::persistence::{SharedSubstrate, SUBSTRATE_DIR};
use cluster::replicate::Replicator;
use substrate_git::{adopt, Alone, Importer, Lease, Store};
use tokio::runtime::Handle;
use zend_vfs::{fetch_here, RemoteUrl, Repo, Workspace};

/// The stores' folder inside the substrate.
pub const GIT_DIR: &str = "git";

/// How often the thread ticks every repository.
const TICK: Duration = Duration::from_secs(2);

/// How long the import lease lives unrenewed, and how often it is renewed.
const LEASE_TTL: Duration = Duration::from_secs(120);
const LEASE_RENEW: Duration = Duration::from_secs(30);

/// Where `root`'s stores are.
pub fn stores_root(root: &Path) -> PathBuf {
    root.join(SUBSTRATE_DIR).join(GIT_DIR)
}

/// `root`'s workspace with every manifest repository read from its store,
/// each store made empty when absent, plus the uploads repository.
pub fn open_workspace(root: &Path) -> anyhow::Result<Workspace> {
    let specs = Workspace::manifest(root)?;
    let mut workspace = Workspace::new(root, specs.clone())?;
    for spec in &specs {
        let store = Store::open(&stores_root(root), &spec.name)?;
        workspace = workspace.with_dir(&spec.name, store.dir())?;
    }
    crate::workspace::with_uploads(workspace)
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

/// What the cluster gives the importers: the consensus the lease is held
/// through, and the agents fetches are made by.
pub struct Cluster {
    pub replicator: Arc<Replicator>,
    pub agents: Agents,
    pub runtime: Handle,
}

/// Called when a record branch moves: the ingest worker's wake, once it exists.
static ON_CHANGE: Mutex<Option<Arc<dyn Fn() + Send + Sync>>> = Mutex::new(None);

/// Wake `f` whenever a record branch moves.
pub fn on_change(f: Arc<dyn Fn() + Send + Sync>) {
    *ON_CHANGE.lock().unwrap() = Some(f);
}

fn changed() {
    if let Some(f) = ON_CHANGE.lock().unwrap().clone() {
        f();
    }
}

/// Bring every repository of `names` up to the log once, adopting a checkout
/// under `root` the log has not seen, then keep them there on a thread.
pub fn start(
    root: &Path,
    names: &[String],
    shared: SharedSubstrate,
    cluster: Option<Cluster>,
) -> anyhow::Result<()> {
    let lease: Arc<dyn Lease> = match &cluster {
        Some(c) => Arc::new(ClusterLease {
            replicator: Arc::clone(&c.replicator),
            renewed: Mutex::new(HashMap::new()),
        }),
        None => Arc::new(Alone),
    };
    let mut repos = Vec::new();
    for name in names {
        let local = root.join(name);
        let checkout = local
            .join(".git")
            .exists()
            .then(|| Repo::open(&local).ok())
            .flatten();
        let mut store = Store::open(&stores_root(root), name)?;
        store = match (&cluster, &checkout) {
            (Some(c), _) => store.with_origin(
                &RemoteUrl::parse(&format!("agent:{name}"))?,
                AgentFetch::shared(c.agents.clone(), c.runtime.clone(), name.clone()),
            )?,
            // Standalone: origin as the checkout has it, over this machine's
            // own credentials.
            (None, Some(repo)) => match repo
                .remotes()?
                .into_iter()
                .find(|r| r.name.as_str() == "origin")
            {
                Some(origin) => {
                    store.with_origin(&RemoteUrl::parse(&origin.fetch_url)?, fetch_here())?
                }
                None => store,
            },
            (None, None) => store,
        };
        if let Some(repo) = &checkout {
            match adopt(&shared, name, repo) {
                Ok(p) if p.packs > 0 || !p.refs.is_empty() => {
                    tracing::info!(repo = %name, packs = p.packs, refs = p.refs.len(), "adopted the checkout's branches into the substrate")
                }
                Ok(_) => {}
                Err(e) => tracing::warn!(repo = %name, "adopting the checkout: {e:#}"),
            }
        }
        let importer = Importer::new(Arc::clone(&lease));
        match importer.tick(&store, &shared) {
            Ok(t) => tracing::info!(
                repo = %name,
                fetched = t.fetched,
                taken = t.synced.taken,
                waiting = t.synced.waiting,
                branches = t.synced.moved.len(),
                "repository brought up to the substrate"
            ),
            Err(e) => tracing::warn!(repo = %name, "bringing the repository up: {e:#}"),
        }
        repos.push((store, importer));
    }
    thread::Builder::new()
        .name("substrate-git".into())
        .spawn(move || follow(repos, shared))?;
    Ok(())
}

fn follow(repos: Vec<(Store, Importer)>, shared: SharedSubstrate) {
    let mut last_error: HashMap<String, String> = HashMap::new();
    loop {
        thread::sleep(TICK);
        if crate::ingest_cancel::cancelled() {
            return;
        }
        let mut moved = false;
        for (store, importer) in &repos {
            let error = match importer.tick(store, &shared) {
                Ok(t) => {
                    moved |= !t.synced.moved.is_empty();
                    t.fetch_error
                }
                Err(e) => Some(format!("{e:#}")),
            };
            let name = store.name().to_string();
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
        if moved {
            changed();
        }
    }
}
