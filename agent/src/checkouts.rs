//! The repositories an agent holds (§11.1): for each, its checkout's
//! sandbox — which runs a job's command there, one at a time — the source a
//! job's store reads the repository's branches through, and its origin.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cluster::frame::RepoAd;
use cluster::machine::toolchains;
use zend_vfs::{CommandPolicy, GitSource, Repo, Sandbox, SandboxServer, ORIGIN};

use crate::config::AgentConfig;

/// One repository's checkout.
pub struct Checkout {
    pub name: String,
    /// Its folder: where a run's build outputs are, for staging.
    pub dir: PathBuf,
    /// The agent holds the credentials to push to its origin.
    pub push: bool,
    /// Origin's fetch URL; empty when the checkout has no origin.
    pub origin: String,
    pub source: Arc<GitSource>,
    pub server: SandboxServer,
}

impl Checkout {
    /// The checkout's repository: what is fetched, unbundled and pushed.
    pub fn repo(&self) -> &Repo {
        self.server.sandbox().repo()
    }
}

/// Every checkout, by name.
pub struct Checkouts {
    by_name: BTreeMap<String, Checkout>,
    /// The programs the sandboxes start that are on this machine's `PATH`.
    toolchains: Vec<String>,
    /// The folder the job logs are written to, which also keeps the agent's
    /// node and its switch.
    jobs_dir: PathBuf,
    /// Where a run stages what it built, for this machine's keep.
    stage_dir: Option<PathBuf>,
}

impl Checkouts {
    /// Open each checkout `cfg` names, its sandbox starting what `policy`
    /// allows and logging to the configured jobs folder; each recovered first
    /// from any job a crash cut short.
    pub fn open(cfg: &AgentConfig, policy: &CommandPolicy) -> anyhow::Result<Checkouts> {
        let allowed: Vec<&str> = policy.programs().collect();
        let toolchains = toolchains(&allowed);
        let mut by_name = BTreeMap::new();
        for c in &cfg.checkouts {
            let what = |e: &dyn std::fmt::Display| anyhow::anyhow!("checkout `{}`: {e}", c.name);
            let repo = Repo::open(&c.dir).map_err(|e| what(&e))?;
            let origin = repo
                .remotes()
                .map_err(|e| what(&e))?
                .into_iter()
                .find(|r| r.name.as_str() == ORIGIN)
                .map(|r| r.fetch_url)
                .unwrap_or_default();
            let source = GitSource::open(&c.dir).map_err(|e| what(&e))?;
            let sandbox = Sandbox::new(repo, policy.clone());
            if let Err(e) = sandbox.recover() {
                tracing::warn!("checkout `{}`: {e}", c.name);
            }
            let server = SandboxServer::new(sandbox, &cfg.jobs_dir).map_err(|e| what(&e))?;
            by_name.insert(
                c.name.clone(),
                Checkout {
                    name: c.name.clone(),
                    dir: c.dir.clone(),
                    push: c.push,
                    origin,
                    source,
                    server,
                },
            );
        }
        Ok(Checkouts {
            by_name,
            toolchains,
            jobs_dir: cfg.jobs_dir.clone(),
            stage_dir: cfg.stage_dir.clone(),
        })
    }

    /// The folder the job logs are written to.
    pub fn jobs_dir(&self) -> &Path {
        &self.jobs_dir
    }

    /// Where a run stages what it built; `None` when this agent stages nothing.
    pub fn stage_dir(&self) -> Option<&Path> {
        self.stage_dir.as_deref()
    }

    /// What `CAPACITY` advertises as its toolchains (§11.2).
    pub fn toolchains(&self) -> &[String] {
        &self.toolchains
    }

    pub fn get(&self, name: &str) -> Option<&Checkout> {
        self.by_name.get(name)
    }

    pub fn all(&self) -> impl Iterator<Item = &Checkout> {
        self.by_name.values()
    }

    /// The repositories `HELLO` advertises (§11.2).
    pub fn ads(&self) -> Vec<RepoAd> {
        self.all()
            .map(|c| RepoAd {
                name: c.name.clone(),
                origin: c.origin.clone(),
                push: c.push,
            })
            .collect()
    }
}
