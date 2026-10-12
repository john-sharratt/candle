//! The mind as a repository an agent holds (`docs/replicated_substrate.md`
//! §10.3): `--mind-repo <name>`, as zend names a repository in its workspace.
//! With `--cluster` this is how a member has a mind. It keeps a **mirror** of
//! the repository — a git working tree of its own under `--data`, at
//! `mind/<name>` — whose `origin` names the agent, `agent:<name>`, never a URL:
//! no fetch or push of it leaves this machine but through an agent. Every read
//! of the mind folder reads the mirror; its fetches go through an agent holding
//! `<name>` ([`agent::fetcher::AgentFetch`]), its pushes through one with push
//! rights.
//!
//! A mirror is ready once it holds the record's `projection.yaml`. A member
//! whose mirror is not ready starts with no mind, and once it has joined it
//! brings the mirror up to the record through an agent ([`MindRepo::bring_up`])
//! and restarts onto it — the schema, the libraries and the grammar built from
//! them are read once, at start.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{bail, Context};
use zend_vfs::{BranchName, Fetch, GitError, Oid, RemoteName, RemoteUrl, Repo, ORIGIN};

use crate::projection::SCHEMA_FILE;

/// Where under `--data` the mirrors are kept.
pub const MIRRORS: &str = "mind";

/// The mind repository a member reads through its mirror.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MindRepo {
    /// Its name, as the agents hold it.
    pub name: String,
    /// The record's branch.
    pub branch: BranchName,
    /// The mirror: `<data>/mind/<name>`.
    pub dir: PathBuf,
}

impl MindRepo {
    /// `name` on `branch`, mirrored under `data`. A name is one folder: no
    /// separator, nothing that walks out of the mirrors' folder.
    pub fn new(data: &Path, name: &str, branch: &str) -> anyhow::Result<MindRepo> {
        let bad = name.is_empty()
            || name == "."
            || name == ".."
            || name.contains(['/', '\\', ':'])
            || name.starts_with('-');
        if bad {
            bail!("--mind-repo `{name}` is not a repository's name");
        }
        let branch =
            BranchName::parse(branch).with_context(|| format!("--mind-branch `{branch}`"))?;
        Ok(MindRepo {
            name: name.to_string(),
            branch,
            dir: data.join(MIRRORS).join(name),
        })
    }

    /// The `origin` a mirror names: the agents, not a URL.
    pub fn origin_url(&self) -> String {
        format!("agent:{}", self.name)
    }

    /// Whether the mirror holds a mind yet: the record's schema.
    pub fn ready(&self) -> bool {
        self.dir.join(SCHEMA_FILE).is_file()
    }

    /// The mirror, its fetches made by `fetcher`: made empty on the record's
    /// branch, its `origin` the agents', when there is none yet.
    pub fn open(&self, fetcher: Arc<dyn Fetch>) -> Result<Repo, GitError> {
        let repo = if self.dir.join(".git").exists() {
            Repo::open(&self.dir)?
        } else {
            let repo = Repo::init(&self.dir, &self.branch)?;
            repo.add_remote(
                &RemoteName::parse(ORIGIN)?,
                &RemoteUrl::parse(&self.origin_url())?,
            )?;
            repo
        };
        Ok(repo.fetching_through(fetcher))
    }

    /// Bring the mirror up to the record through `fetcher` and check it out.
    /// The record's commit; `None` while origin has no such branch.
    pub fn bring_up(&self, fetcher: Arc<dyn Fetch>) -> Result<Option<Oid>, GitError> {
        let repo = self.open(fetcher)?;
        let pulled = repo.pull_branch(&self.branch)?;
        let Some(tip) = pulled.tip else {
            return Ok(None);
        };
        repo.force_checkout_branch(&self.branch)?;
        Ok(Some(tip))
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::process::Command;

    use zend_vfs::{FetchBundle, FetchSpec, RefUpdate};

    use super::*;

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args([
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "-c",
                "init.defaultBranch=main",
            ])
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    }

    /// An agent's checkout, standing in for the agent: what it bundles is
    /// what a fetch job answers with.
    struct FromCheckout(PathBuf);

    impl Fetch for FromCheckout {
        fn fetch(
            &self,
            repo: &Repo,
            remote: &RemoteName,
            spec: &FetchSpec,
        ) -> Result<Vec<RefUpdate>, GitError> {
            let agent = Repo::open(&self.0)?;
            agent.fetch_here(remote, &FetchSpec::AllBranches)?;
            let fetched: FetchBundle =
                agent.bundle_for_fetch(remote, spec, &repo.tracking_tips(remote)?)?;
            repo.take_fetch(remote, spec, &fetched)
        }
    }

    /// A mind's origin with one commit, and an agent's checkout of it.
    fn origin_and_agent(root: &Path) -> (PathBuf, PathBuf) {
        let origin = root.join("origin.git");
        fs::create_dir(&origin).unwrap();
        git(&origin, &["init", "-q", "--bare", "-b", "main"]);
        let seed = root.join("seed");
        fs::create_dir(&seed).unwrap();
        git(&seed, &["init", "-q", "-b", "main"]);
        fs::write(seed.join(SCHEMA_FILE), b"layers: []\n").unwrap();
        git(&seed, &["add", "."]);
        git(&seed, &["commit", "-q", "-m", "first"]);
        git(&seed, &["push", "-q", origin.to_str().unwrap(), "main"]);
        let agent = root.join("agent");
        git(
            root,
            &[
                "clone",
                "-q",
                origin.to_str().unwrap(),
                agent.to_str().unwrap(),
            ],
        );
        (origin, agent)
    }

    /// A name is one folder; the mirror is under `--data`.
    #[test]
    fn a_mind_repo_is_a_folder_name_under_data() {
        let r = MindRepo::new(Path::new("/srv/npcd"), "mind", "main").unwrap();
        assert_eq!(r.dir, Path::new("/srv/npcd").join("mind").join("mind"));
        assert_eq!(r.origin_url(), "agent:mind");
        for bad in ["", "..", "a/b", "a\\b", "c:x", "-x"] {
            assert!(
                MindRepo::new(Path::new("/d"), bad, "main").is_err(),
                "{bad}"
            );
        }
        assert!(MindRepo::new(Path::new("/d"), "mind", "bad..branch").is_err());
    }

    /// **A member with no way to origin gets its mind through an agent**: the
    /// mirror is made, named `agent:mind`, brought to the record and checked
    /// out — ready — and later follows the record the same way.
    #[test]
    fn the_mirror_is_made_and_brought_up_through_an_agent() {
        let root = tempfile::tempdir().unwrap();
        let (origin, agent) = origin_and_agent(root.path());
        let repo = MindRepo::new(&root.path().join("data"), "mind", "main").unwrap();
        assert!(!repo.ready());
        let fetcher: Arc<dyn Fetch> = Arc::new(FromCheckout(agent.clone()));
        let first = repo.bring_up(fetcher.clone()).unwrap().unwrap();
        assert!(repo.ready());
        assert_eq!(
            git(&repo.dir, &["remote", "get-url", "origin"]).trim(),
            "agent:mind"
        );
        assert_eq!(
            git(&repo.dir, &["rev-parse", "HEAD"]).trim(),
            first.to_string()
        );

        let seed = root.path().join("seed");
        fs::write(seed.join(SCHEMA_FILE), b"layers: [eras]\n").unwrap();
        git(&seed, &["commit", "-q", "-am", "second"]);
        git(&seed, &["push", "-q", origin.to_str().unwrap(), "main"]);
        let second = repo.bring_up(fetcher).unwrap().unwrap();
        assert_ne!(second, first);
        assert_eq!(
            fs::read_to_string(repo.dir.join(SCHEMA_FILE))
                .unwrap()
                .replace("\r\n", "\n"),
            "layers: [eras]\n"
        );
    }

    /// A record without the branch leaves the mirror made and not ready.
    #[test]
    fn a_branch_origin_lacks_leaves_the_mirror_not_ready() {
        let root = tempfile::tempdir().unwrap();
        let (_origin, agent) = origin_and_agent(root.path());
        let repo = MindRepo::new(&root.path().join("data"), "mind", "absent").unwrap();
        let fetcher: Arc<dyn Fetch> = Arc::new(FromCheckout(agent));
        assert_eq!(repo.bring_up(fetcher).unwrap(), None);
        assert!(!repo.ready());
    }
}
