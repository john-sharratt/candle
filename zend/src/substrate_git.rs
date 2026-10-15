//! zend's repositories read from the substrate (`docs/substrate_git.md` §7).
//!
//! With `substrate_git: true`, each repository `workspace.yaml` names is read
//! from the object store derived from the log, `<workspace>/substrate/git/<name>`,
//! instead of a folder under the workspace. The repository keeps its name, so
//! every tool, path and ingest key is unchanged; only the folder behind it
//! moves. A member then needs no checkouts: its stores fill from the log.
//!
//! Every repository is loaded whole during the [`LoadStep::Repositories`] step
//! — pulled through an agent first when no member has — and then followed
//! ([`substrate_git::member::Repos`]). A record branch that moves wakes the
//! ingest worker.

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use ::substrate_git::member::{Cluster, Loading, Repos, Source};
use ::substrate_git::Store;
use candle_conversation::persistence::{SharedSubstrate, SUBSTRATE_DIR};
use tokio::sync::watch;
use zend_vfs::Workspace;

use crate::loading::{LoadProgress, LoadStep};

/// The stores' folder inside the substrate.
pub const GIT_DIR: &str = "git";

/// This daemon's repositories, once loaded.
static REPOS: OnceLock<Repos> = OnceLock::new();

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

/// Load every repository of `names` whole — a checkout under `root` is the
/// source only without a cluster — then follow them on a thread.
pub fn load(
    root: &Path,
    names: &[String],
    shared: SharedSubstrate,
    cluster: Option<Cluster>,
    progress: &LoadProgress,
    status: &watch::Sender<String>,
) -> anyhow::Result<()> {
    let sources: Vec<Source> = names
        .iter()
        .map(|name| Source {
            name: name.clone(),
            checkout: Some(root.join(name)),
        })
        .collect();
    let repos = Repos::open(&stores_root(root), &sources, shared, cluster)?;
    progress.set_step(LoadStep::Repositories);
    repos.load(&crate::ingest_cancel::cancelled, &|state: &[Loading]| {
        let held: usize = state.iter().map(|l| l.held).sum();
        let listed: usize = state.iter().map(|l| l.listed).sum();
        progress.set_step_progress(held as u64, listed as u64);
        status.send(detail(state)).ok();
    })?;
    status.send(String::new()).ok();
    repos.follow(Arc::new(crate::ingest_cancel::cancelled))?;
    let _ = REPOS.set(repos);
    Ok(())
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

/// Wake `f` whenever a record branch moves.
pub fn on_change(f: Arc<dyn Fn() + Send + Sync>) {
    if let Some(repos) = REPOS.get() {
        repos.on_change(Arc::new(move |_| f()));
    }
}
