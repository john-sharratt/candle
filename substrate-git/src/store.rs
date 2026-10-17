//! The object database a member reads one repository through
//! (`docs/substrate_git.md` §4): no folder, its objects the log's packs, its
//! refs in memory.
//!
//! The log's branches are the database's **record**: `origin`'s tracking refs,
//! `refs/remotes/origin/<branch>`, which is where zend reads a repository's
//! branches when it has an origin (`zend_vfs::Repo::record_branches`). A sync
//! never moves a tracking ref back to an ancestor of what it holds unless the
//! log says `forced`, so a member's own push — which moves its tracking ref as
//! it lands — is never rewound before it is published ([`crate::Importer`]).
//! A tip on another line is taken: the log reached it through a forced write
//! this member missed. Local branches are the member's own and never touched.
//!
//! Nothing a member holds here is lost at a restart: its refs are rebuilt
//! from the log by the next sync.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::Context;
use zend_vfs::{
    fetch_here, Fetch, ObjectDb, Oid, RefName, RefOp, RefTransaction, RemoteName, RemoteUrl, Repo,
    Rev, UrlKind, DEFAULT_PACK_BUDGET,
};

use crate::pack_store::LogPacks;
use crate::records::{Log, Records};

/// Where the record's branches are: `origin`'s tracking refs.
pub(crate) const TRACKING: &str = "refs/remotes/origin/";

/// The remote the importer fetches into.
pub(crate) const IMPORT: &str = "import";

/// The object database of repository `name`, its packs read from `records`.
pub fn open_db(records: Arc<dyn Records>, name: &str) -> anyhow::Result<ObjectDb> {
    Log::new(records.as_ref(), name)?;
    let packs = Arc::new(LogPacks::new(records, name));
    ObjectDb::open(name, packs, DEFAULT_PACK_BUDGET)
        .with_context(|| format!("the object database of {name}"))
}

/// One repository's object database.
pub struct Store {
    name: String,
    db: ObjectDb,
    repo: Repo,
    /// What makes the repository's fetches.
    fetcher: Arc<dyn Fetch>,
}

/// What a sync did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Synced {
    /// Packs taken into the database.
    pub taken: usize,
    /// Packs the log names whose blobs have not all arrived.
    pub waiting: usize,
    /// Packs whose blobs arrived but whose index could not be read, by
    /// SHA-256, with why. Every branch waits while one is, as for a pack not
    /// arrived.
    pub failed: Vec<(String, String)>,
    /// Packs the log names that the database holds, and how many it names.
    pub held: usize,
    pub listed: usize,
    /// Branches the log names that the database does not yet hold where the
    /// log does: their packs have not all arrived.
    pub behind: usize,
    /// Record refs set or deleted, with where they point now.
    pub moved: Vec<(RefName, Option<Oid>)>,
}

impl Store {
    /// Repository `name`, read from `records`. Its `origin` is `agent:<name>`
    /// until [`Self::with_origin`] says otherwise.
    pub fn open(records: Arc<dyn Records>, name: &str) -> anyhow::Result<Self> {
        Self::over(open_db(records, name)?)
    }

    /// The repository `db` holds — a database [`open_db`] opened, shared with
    /// whatever else holds it. Its `origin` is `agent:<name>` until
    /// [`Self::with_origin`] says otherwise.
    pub fn over(db: ObjectDb) -> anyhow::Result<Self> {
        let name = db.name().to_string();
        let repo = Repo::in_substrate(&db)?;
        let store = Self {
            name: name.clone(),
            db,
            repo,
            fetcher: fetch_here(),
        };
        store.set_origin(&RemoteUrl::parse(&format!("agent:{name}"))?)?;
        Ok(store)
    }

    /// This store fetching from `url` as `origin`, through `fetcher`.
    pub fn with_origin(mut self, url: &RemoteUrl, fetcher: Arc<dyn Fetch>) -> anyhow::Result<Self> {
        self.set_origin(url)?;
        self.repo = Repo::in_substrate(&self.db)?.fetching_through(Arc::clone(&fetcher));
        self.fetcher = fetcher;
        Ok(self)
    }

    /// What makes the repository's fetches: for another handle on the same
    /// database to fetch as this one does.
    pub fn fetcher(&self) -> Arc<dyn Fetch> {
        Arc::clone(&self.fetcher)
    }

    /// Point `origin` — the record, and where pushes go — and `import` — what
    /// the importer fetches into — at `url`.
    fn set_origin(&self, url: &RemoteUrl) -> anyhow::Result<()> {
        let remotes = self.repo.remotes()?;
        for name in ["origin", IMPORT] {
            let name = RemoteName::parse(name)?;
            if remotes.iter().any(|r| r.name == name) {
                self.repo.set_remote_url(&name, url, UrlKind::Fetch)?;
            } else {
                self.repo.add_remote(&name, url)?;
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

    /// The store as a repository: read branches, build commits, bundle a push.
    pub fn repo(&self) -> &Repo {
        &self.repo
    }

    /// The database the repository is held in.
    pub fn objects(&self) -> &ObjectDb {
        &self.db
    }

    /// The record's branches as the database holds them, by their name in the
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

    /// Packs the log names that the database holds, and how many it names —
    /// read, with nothing taken in and no ref moved: what a load can say
    /// before its first round has run.
    pub fn pack_counts(&self, records: &dyn Records) -> anyhow::Result<(usize, usize)> {
        let listed = Log::new(records, &self.name)?.packs()?;
        let held: BTreeSet<String> = self.db.held_packs().into_iter().collect();
        let n_held = listed.keys().filter(|sha| held.contains(*sha)).count();
        Ok((n_held, listed.len()))
    }

    /// Bring the database up to `records`: take every pack whose blobs have
    /// arrived, then, once every pack the log lists is here, move every record
    /// branch whose own `needs` are too — never back to an ancestor unless
    /// forced.
    pub fn sync(&self, records: &dyn Records) -> anyhow::Result<Synced> {
        let log = Log::new(records, &self.name)?;
        let refreshed = self.db.refresh()?;
        let held: BTreeSet<String> = self.db.held_packs().into_iter().collect();
        let mut synced = Synced {
            taken: refreshed.added,
            failed: refreshed.failed,
            ..Synced::default()
        };
        for sha in log.packs()?.keys() {
            synced.listed += 1;
            if held.contains(sha) {
                synced.held += 1;
            } else if !synced.failed.iter().any(|(f, _)| f == sha) {
                synced.waiting += 1;
            }
        }

        let current = self.record()?;
        let mut txn = RefTransaction::new();
        for (name, entry) in log.refs()? {
            let Some(branch) = name.strip_prefix("refs/heads/") else {
                continue;
            };
            // Every pack the log lists, not only the ref's own `needs`: a ref
            // names only the packs written with it, and its parents may be in
            // any other. A tracking ref never names a commit whose history the
            // database cannot read whole.
            if synced.listed > synced.held || !entry.needs.iter().all(|sha| held.contains(sha)) {
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
                    // Never back to an ancestor of what the record holds,
                    // unless forced: that is the log behind this member's own
                    // push, or a stale write. A tip on another line is taken —
                    // the log moved there through a forced write this member
                    // missed, and `forced` belongs to that one entry, not to
                    // the commits written on top of it since.
                    let backward = self
                        .repo
                        .is_ancestor(&Rev::Oid(new.clone()), &Rev::Oid(old.clone()))?;
                    if backward && !entry.forced {
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
}
