//! zend's repositories read from the substrate (`docs/substrate_git.md` §7).
//!
//! Each repository `workspace.yaml` names is read from the object store
//! derived from the log, `<workspace>/substrate/git/<name>`; there is no folder
//! for it under the workspace. The repository keeps its name, so every tool,
//! path and ingest key is unchanged. A member needs no checkouts: its stores
//! fill from the log, and origin is reached only through an agent — the
//! cluster's, or the one in this process (`docs/substrate_git.md` §9).
//!
//! Every repository is loaded whole during the [`LoadStep::Repositories`] step
//! — pulled through an agent first when no member has — and then followed
//! ([`substrate_git::member::Repos`]) until [`Repositories::stop`]. A record
//! branch that moves wakes the ingest worker.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};

use ::substrate_git::member::{Access, Loading, Repos};
use agent::client::Agents;
use agent::jobs::Jobs;
use host::cluster::Join;
use tokio::runtime::Handle;
use ::substrate_git::Store;
use candle_conversation::persistence::{SharedSubstrate, SUBSTRATE_DIR};
use tokio::sync::watch;
use zend_vfs::Workspace;

use crate::loading::{LoadProgress, LoadStep};

/// The stores' folder inside the substrate.
pub const GIT_DIR: &str = "git";

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

/// The agents this daemon's repositories are read and written through: the
/// cluster's, reaching the one in this process while cut off — or, with no
/// cluster, the one in this process alone (`docs/substrate_git.md` §9).
/// `None` when there is neither: a daemon with no `agent:` file and no
/// cluster has no way to a repository.
pub fn agents(cluster: Option<&Join>, agent: Option<&Arc<Jobs>>) -> Option<Agents> {
    match cluster.and_then(|join| Some((join, join.link().member()?))) {
        Some((join, member)) => Some(Agents::new(Arc::clone(member), join.in_process_agent())),
        None => agent.map(|jobs| Agents::in_process(Arc::clone(jobs))),
    }
}

/// How the repositories are reached: [`agents`], on `runtime`, and the
/// cluster's consensus for the import lease when there is a cluster.
pub fn access(
    cluster: Option<&Join>,
    agent: Option<&Arc<Jobs>>,
    runtime: Handle,
) -> anyhow::Result<Access> {
    let agents = agents(cluster, agent).ok_or_else(|| {
        anyhow::anyhow!(
            "the repositories are read and written through an agent, and there is none: give \
             the process an `agent:` file naming a checkout of each"
        )
    })?;
    let replicator = cluster
        .and_then(|join| join.link().member())
        .and_then(|member| member.replicator().cloned());
    Ok(Access {
        agents,
        runtime,
        replicator,
    })
}

/// A load's repositories, followed on a thread of their own until
/// [`Repositories::stop`] — or until dropped, unwaited.
pub struct Repositories {
    repos: Repos,
    /// What the follower asks each round: once set, it ends.
    stop: Arc<AtomicBool>,
    /// Disconnected once the follower has ended: its one sender lives in the
    /// closure the follower owns, dropped as the thread returns or unwinds.
    exited: Mutex<Option<mpsc::Receiver<()>>>,
}

impl Repositories {
    /// Wake `f` whenever a record branch moves — and once now, for a branch
    /// that moved between the load and this call, which no later round of
    /// the follower reports again.
    pub fn on_change(&self, f: Arc<dyn Fn() + Send + Sync>) {
        let wake = Arc::clone(&f);
        self.repos.on_change(Arc::new(move |_: &str| wake()));
        f();
    }

    /// Stop following and wait for the follower to end: at most the round in
    /// flight. Blocking.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
        let exited = self.exited.lock().unwrap().take();
        if let Some(exited) = exited {
            // Nothing is ever sent: this returns once the sender is dropped.
            let _ = exited.recv();
        }
    }
}

impl Drop for Repositories {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// Load every repository of `names` whole, through `access`, then follow them
/// on a thread.
pub fn load(
    root: &Path,
    names: &[String],
    shared: SharedSubstrate,
    access: Access,
    progress: &LoadProgress,
    status: &watch::Sender<String>,
) -> anyhow::Result<Repositories> {
    let repos = Repos::open(&stores_root(root), names, shared, access)?;
    progress.set_step(LoadStep::Repositories);
    repos.load(&crate::ingest_cancel::cancelled, &|state: &[Loading]| {
        let held: usize = state.iter().map(|l| l.held).sum();
        let listed: usize = state.iter().map(|l| l.listed).sum();
        progress.set_step_progress(held as u64, listed as u64);
        status.send(detail(state)).ok();
    })?;
    status.send(String::new()).ok();
    let stop = Arc::new(AtomicBool::new(false));
    let (exited_tx, exited) = mpsc::channel::<()>();
    let stopped: Arc<dyn Fn() -> bool + Send + Sync> = {
        let stop = Arc::clone(&stop);
        let exited_tx = Mutex::new(exited_tx);
        Arc::new(move || {
            // Held, never sent on: dropped with the follower's closure.
            let _ = &exited_tx;
            stop.load(Ordering::Relaxed)
        })
    };
    repos.follow(stopped)?;
    Ok(Repositories {
        repos,
        stop,
        exited: Mutex::new(Some(exited)),
    })
}

/// What the loading page says while repositories load: each one still
/// incomplete, and why a pull is failing.
fn detail(state: &[Loading]) -> String {
    state
        .iter()
        .filter(|l| !l.whole)
        .map(|l| match &l.error {
            Some(e) => format!("{}: {e}", l.repo),
            None if l.listed == 0 => format!("{}: waiting for a pull", l.repo),
            None => format!("{}: {} of {} packs", l.repo, l.held, l.listed),
        })
        .collect::<Vec<_>>()
        .join("; ")
}
