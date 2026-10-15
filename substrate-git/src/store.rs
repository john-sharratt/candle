//! The object store a member derives from its log (`docs/substrate_git.md` §4).
//!
//! One repository per name under a root — `<substrate>/git/<repo>` — with an
//! empty working tree, since nothing reads its folder: a conversation reads a
//! branch's tree, and commands run on agents. Its packs are the log's packs.
//!
//! The log's branches are the store's **record**: `origin`'s tracking refs,
//! `refs/remotes/origin/<branch>`, which is where zend reads a repository's
//! branches when it has an origin (`zend_vfs::Repo::record_branches`). A sync
//! only moves a tracking ref forward, or where the log says `forced`, so a
//! member's own push — which moves its tracking ref as it lands — is never
//! rewound before it is published ([`crate::Importer`]). Local branches are
//! the member's own and never touched.
//!
//! Deleting the store loses nothing a member has published: the next sync
//! rebuilds it from the log.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use anyhow::{bail, Context};
use zend_vfs::{
    BranchName, Fetch, Oid, RefName, RefOp, RefTransaction, RemoteName, RemoteUrl, Repo, Rev,
    UrlKind,
};

use crate::records::{Log, Records};

/// What the store's HEAD names: a branch never made, so nothing is checked out
/// and no branch the log moves is the current one.
const STORE_HEAD: &str = "refs/heads/substrate-store";

/// Where the record's branches are: `origin`'s tracking refs.
pub(crate) const TRACKING: &str = "refs/remotes/origin/";

/// The remote the importer fetches into.
pub(crate) const IMPORT: &str = "import";

/// One repository's derived object store.
pub struct Store {
    name: String,
    dir: PathBuf,
    repo: Repo,
}

/// What a sync did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Synced {
    /// Packs taken into the store.
    pub taken: usize,
    /// Packs the log names whose blob has not arrived.
    pub waiting: usize,
    /// Packs the log names that the store holds, and how many it names.
    pub held: usize,
    pub listed: usize,
    /// Branches the log names that the store does not yet hold where the log
    /// does: their packs have not all arrived.
    pub behind: usize,
    /// Record refs set or deleted, with where they point now.
    pub moved: Vec<(RefName, Option<Oid>)>,
}

impl Store {
    /// The store for `name` under `root`, made empty when absent. Its `origin`
    /// is `agent:<name>` until [`Self::with_origin`] says otherwise.
    pub fn open(root: &Path, name: &str) -> anyhow::Result<Self> {
        let dir = root.join(name);
        if !dir.join(".git").is_dir() {
            fs::create_dir_all(&dir).with_context(|| dir.display().to_string())?;
            Repo::init(&dir, &BranchName::parse("main")?)
                .with_context(|| format!("the object store at {}", dir.display()))?;
        }
        // Nothing is checked out here, so HEAD names a ref of the store's own,
        // never a branch.
        let pointed = Command::new("git")
            .current_dir(&dir)
            .args(["symbolic-ref", "HEAD", STORE_HEAD])
            .output()
            .with_context(|| format!("pointing HEAD in {}", dir.display()))?;
        if !pointed.status.success() {
            bail!(
                "pointing HEAD in {}: {}",
                dir.display(),
                String::from_utf8_lossy(&pointed.stderr).trim()
            );
        }
        let repo =
            Repo::open(&dir).with_context(|| format!("the object store at {}", dir.display()))?;
        let store = Self {
            name: name.to_string(),
            dir,
            repo,
        };
        store.set_origin(&RemoteUrl::parse(&format!("agent:{name}"))?, false)?;
        Ok(store)
    }

    /// This store fetching from `url` as `origin`, through `fetcher`.
    pub fn with_origin(mut self, url: &RemoteUrl, fetcher: Arc<dyn Fetch>) -> anyhow::Result<Self> {
        self.set_origin(url, true)?;
        self.repo = self.repo.fetching_through(fetcher);
        Ok(self)
    }

    /// Point `origin` — the record, and where pushes go — and `import` — what
    /// the importer fetches into — at `url`.
    fn set_origin(&self, url: &RemoteUrl, replace: bool) -> anyhow::Result<()> {
        let remotes = self.repo.remotes()?;
        for name in ["origin", IMPORT] {
            let name = RemoteName::parse(name)?;
            if !remotes.iter().any(|r| r.name == name) {
                self.repo.add_remote(&name, url)?;
            } else if replace {
                self.repo.set_remote_url(&name, url, UrlKind::Fetch)?;
            }
        }
        Ok(())
    }

    /// `origin`'s branches as the importer last fetched them, by their name in
    /// the log. Apart from the record, so a fetch never moves what zend reads
    /// before the log does.
    pub fn imported(&self) -> anyhow::Result<BTreeMap<RefName, Oid>> {
        self.branches_under(&format!("refs/remotes/{IMPORT}/"))
    }

    /// The repository's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Where the store is.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// The store as a repository: read branches, build commits, bundle a push.
    pub fn repo(&self) -> &Repo {
        &self.repo
    }

    /// The record's branches as the store holds them, by their name in the
    /// log (`refs/heads/<branch>`).
    pub fn record(&self) -> anyhow::Result<BTreeMap<RefName, Oid>> {
        self.branches_under(TRACKING)
    }

    fn branches_under(&self, folder: &str) -> anyhow::Result<BTreeMap<RefName, Oid>> {
        let mut out = BTreeMap::new();
        for (name, oid) in self.repo.refs_in(folder)? {
            let branch = &name.as_str()[folder.len()..];
            if branch != "HEAD" {
                out.insert(RefName::parse(&format!("refs/heads/{branch}"))?, oid);
            }
        }
        Ok(out)
    }

    /// Bring the store up to `records`: take every pack it lacks whose blob has
    /// arrived, then move every record branch whose packs are all here.
    pub fn sync(&self, records: &dyn Records) -> anyhow::Result<Synced> {
        let log = Log::new(records, &self.name)?;
        let mut synced = Synced::default();
        let mut held = BTreeSet::new();
        for (sha, entry) in log.packs()? {
            if self.pack_path(&entry.checksum).is_file() {
                held.insert(sha);
                continue;
            }
            match log.pack_bytes(&sha)? {
                Some(bytes) => {
                    self.repo
                        .take_pack(&bytes)
                        .with_context(|| format!("pack {sha} of {}", self.name))?;
                    synced.taken += 1;
                    held.insert(sha);
                }
                None => synced.waiting += 1,
            }
        }

        synced.held = held.len();
        synced.listed = synced.held + synced.waiting;
        let current = self.record()?;
        let mut txn = RefTransaction::new();
        for (name, entry) in log.refs()? {
            let Some(branch) = name.strip_prefix("refs/heads/") else {
                continue;
            };
            if !entry.needs.iter().all(|sha| held.contains(sha)) {
                synced.behind += 1;
                continue;
            }
            let name = RefName::parse(&name)?;
            let tracking = RefName::parse(&format!("{TRACKING}{branch}"))?;
            let new = match &entry.oid {
                Some(oid) => {
                    let oid = Oid::parse(oid)?;
                    if !self.repo.has_object(&oid)? {
                        synced.behind += 1;
                        continue;
                    }
                    Some(oid)
                }
                None => None,
            };
            let op = match (current.get(&name), &new) {
                (Some(old), Some(new)) if old == new => continue,
                (Some(old), Some(new)) => {
                    let forward = self
                        .repo
                        .is_ancestor(&Rev::Oid(old.clone()), &Rev::Oid(new.clone()))?;
                    if !forward && !entry.forced {
                        continue;
                    }
                    RefOp::Update {
                        name: tracking,
                        new: new.clone(),
                        old: old.clone(),
                    }
                }
                (None, Some(new)) => RefOp::Create {
                    name: tracking,
                    new: new.clone(),
                },
                (Some(old), None) => RefOp::Delete {
                    name: tracking,
                    old: old.clone(),
                },
                (None, None) => continue,
            };
            synced.moved.push((name, new));
            txn = txn.push(op);
        }
        if !txn.ops().is_empty() {
            self.repo.update_refs(&txn)?;
        }
        Ok(synced)
    }

    fn pack_path(&self, checksum: &str) -> PathBuf {
        self.dir
            .join(".git")
            .join("objects")
            .join("pack")
            .join(format!("pack-{checksum}.pack"))
    }
}
