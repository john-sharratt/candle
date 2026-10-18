//! Keeping the log's branches with `origin` (`docs/substrate_git.md` §5, §6.2).
//!
//! Every member runs an [`Importer`] per repository. Each [`Importer::tick`]:
//!
//! 1. **Publishes this member's pushes.** A push moves the store's record —
//!    `origin`'s tracking ref — as it lands; a record branch ahead of the log
//!    is published, packs first, then the ref.
//! 2. **Imports**, on the one member holding the lease `git/<repo>`: fetches
//!    every branch of `origin` into the store's `import` remote — through an
//!    agent, with `--cluster` — and publishes every branch that moved ahead of
//!    the log. A branch `origin` rewrote or deleted is published only when two
//!    ticks in a row agree on it, so a fetch racing another member's push
//!    cannot rewind a branch (§5.3).
//! 3. **Syncs** the store up to the log.
//!
//! The fetch lands apart from the record, so nothing zend reads moves before
//! the log does.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use zend_vfs::{FetchSpec, Oid, RefName, RemoteName, Rev};

use crate::publish::{publish, Published};
use crate::records::{import_lease, Lease, Log, Records};
use crate::store::{Store, Synced, IMPORT};

/// What one tick did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Imported {
    /// This member's pushes published.
    pub pushed: Published,
    /// Whether this member held the lease and fetched.
    pub fetched: bool,
    /// Why the fetch failed, when this member held the lease and could not.
    pub fetch_error: Option<String>,
    /// `origin`'s fast-forwards published.
    pub moved: Published,
    /// `origin`'s rewrites and deletions published after two agreeing looks.
    pub forced: Published,
    /// What the sync after them did.
    pub synced: Synced,
    /// How long each part of the round took.
    pub took: Took,
}

/// A round's time, by part: what a slow load is spent on.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Took {
    /// Publishing this member's pushes.
    pub publish: Duration,
    /// Fetching `origin` and publishing what moved; zero when this member did
    /// not fetch.
    pub fetch: Duration,
    /// Syncing the store up to the log.
    pub sync: Duration,
}

impl Took {
    /// Both rounds' time, part by part.
    pub fn add(&mut self, other: &Took) {
        self.publish += other.publish;
        self.fetch += other.fetch;
        self.sync += other.sync;
    }
}

/// Whether a round fetches `origin`, when this member holds the lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Fetching {
    /// Once [`Importer::fetching_every`] has passed since the last.
    IfDue,
    /// Whether due or not.
    Now,
    /// Not at all.
    Never,
}

/// The importer of one repository.
pub struct Importer {
    lease: Arc<dyn Lease>,
    /// Rewrites seen once, waiting for a second look.
    suspect: Mutex<HashMap<RefName, Option<Oid>>>,
    /// How often the lease holder fetches; every tick when zero.
    fetch_every: Duration,
    last_fetch: Mutex<Option<Instant>>,
    /// Set while this member, holding the lease, is fetching `origin` — what a
    /// load reports as the repository being pulled while the round still runs.
    fetching: AtomicBool,
}

impl Importer {
    pub fn new(lease: Arc<dyn Lease>) -> Self {
        Self {
            lease,
            suspect: Mutex::new(HashMap::new()),
            fetch_every: Duration::ZERO,
            last_fetch: Mutex::new(None),
            fetching: AtomicBool::new(false),
        }
    }

    /// Whether this member is fetching `origin` now, holding the lease.
    pub fn fetching(&self) -> bool {
        self.fetching.load(Ordering::Relaxed)
    }

    /// This importer fetching at most every `every` — each fetch is a job on
    /// an agent and a request to origin — while its ticks still publish and
    /// sync every time.
    pub fn fetching_every(mut self, every: Duration) -> Self {
        self.fetch_every = every;
        self
    }

    /// The member last found holding `repo`'s import lease, when another
    /// does.
    pub fn holder(&self, repo: &str) -> Option<u64> {
        self.lease.holder(&import_lease(repo))
    }

    /// One round: publish this member's pushes, import if holding the lease
    /// and a fetch is due, sync.
    pub fn tick(&self, store: &Store, records: &dyn Records) -> anyhow::Result<Imported> {
        self.round(store, records, Fetching::IfDue)
    }

    /// [`Self::tick`], fetching now if holding the lease, due or not: a
    /// member's first load of a repository the log names no branch of.
    pub fn pull(&self, store: &Store, records: &dyn Records) -> anyhow::Result<Imported> {
        self.round(store, records, Fetching::Now)
    }

    /// [`Self::tick`] with no fetch: this member's pushes published and the
    /// store synced to the log. A load of a repository the log already names
    /// is whole from the log alone; origin's newer commits are the follower's
    /// to bring in, and a fetch here only made the load wait on origin.
    pub fn settle(&self, store: &Store, records: &dyn Records) -> anyhow::Result<Imported> {
        self.round(store, records, Fetching::Never)
    }

    fn round(
        &self,
        store: &Store,
        records: &dyn Records,
        fetching: Fetching,
    ) -> anyhow::Result<Imported> {
        let mut out = Imported::default();
        let repo = store.repo();
        let due = match fetching {
            Fetching::Now => true,
            Fetching::Never => false,
            Fetching::IfDue => {
                let last = *self.last_fetch.lock().unwrap();
                last.is_none_or(|at| at.elapsed() >= self.fetch_every)
            }
        };

        let started = Instant::now();
        let log = log_branches(records, store.name())?;
        let pushed = ahead_of(store, &log, &store.record()?)?;
        out.pushed = publish(records, store.name(), repo, &pushed, false)?;
        out.took.publish = started.elapsed();

        let started = Instant::now();
        let fetched = if due && self.lease.hold(&import_lease(store.name())) {
            *self.last_fetch.lock().unwrap() = Some(Instant::now());
            self.fetching.store(true, Ordering::Relaxed);
            let result = repo.fetch(&RemoteName::parse(IMPORT)?, &FetchSpec::AllBranches);
            self.fetching.store(false, Ordering::Relaxed);
            match result {
                Ok(_) => Some(true),
                // An agent or origin out of reach: what is held is still
                // published and synced, and the next tick tries again.
                Err(e) => {
                    out.fetch_error = Some(e.to_string());
                    Some(false)
                }
            }
        } else {
            None
        };
        if fetched == Some(true) {
            out.fetched = true;
            let log = log_branches(records, store.name())?;
            let origin = store.imported()?;
            let moved = ahead_of(store, &log, &origin)?;

            let mut suspect = self.suspect.lock().unwrap();
            let mut seen = HashMap::new();
            let mut rewritten = Vec::new();
            let mut look = |name: &RefName, new: Option<Oid>| {
                if suspect.get(name) == Some(&new) {
                    rewritten.push((name.clone(), new));
                } else {
                    seen.insert(name.clone(), new);
                }
            };
            for (name, new) in &origin {
                let held = log.get(name).cloned().flatten();
                let listed = moved.iter().any(|(n, _)| n == name);
                if !listed && held.as_ref() != Some(new) && !behind(store, held.as_ref(), new)? {
                    look(name, Some(new.clone()));
                }
            }
            for (name, held) in &log {
                if held.is_some() && !origin.contains_key(name) {
                    look(name, None);
                }
            }
            *suspect = seen;
            drop(suspect);

            out.moved = publish(records, store.name(), repo, &moved, false)?;
            out.forced = publish(records, store.name(), repo, &rewritten, true)?;
        } else if fetched.is_none() && due {
            self.suspect.lock().unwrap().clear();
        }
        if fetched.is_some() {
            out.took.fetch = started.elapsed();
        }

        let started = Instant::now();
        out.synced = store.sync(records)?;
        out.took.sync = started.elapsed();
        Ok(out)
    }
}

/// The log's branches: `None` for one it deleted.
fn log_branches(
    records: &dyn Records,
    repo: &str,
) -> anyhow::Result<BTreeMap<RefName, Option<Oid>>> {
    Log::new(records, repo)?
        .refs()?
        .into_iter()
        .filter(|(name, _)| name.starts_with("refs/heads/"))
        .map(|(name, e)| {
            let oid = e.oid.as_deref().map(Oid::parse).transpose()?;
            Ok((RefName::parse(&name)?, oid))
        })
        .collect()
}

/// The branches of `here` ahead of the log: new to it, or descending from
/// what it holds. A branch the log deleted is not ahead.
fn ahead_of(
    store: &Store,
    log: &BTreeMap<RefName, Option<Oid>>,
    here: &BTreeMap<RefName, Oid>,
) -> anyhow::Result<Vec<(RefName, Option<Oid>)>> {
    let repo = store.repo();
    let mut out = Vec::new();
    for (name, new) in here {
        let ahead = match log.get(name) {
            None => true,
            Some(None) => false,
            Some(Some(old)) if old == new => false,
            Some(Some(old)) => {
                repo.has_object(old)?
                    && repo.is_ancestor(&Rev::Oid(old.clone()), &Rev::Oid(new.clone()))?
            }
        };
        if ahead {
            out.push((name.clone(), Some(new.clone())));
        }
    }
    Ok(out)
}

/// Whether `here` is merely behind the log's `held`, which is no rewrite.
fn behind(store: &Store, held: Option<&Oid>, here: &Oid) -> anyhow::Result<bool> {
    let Some(held) = held else {
        return Ok(false);
    };
    let repo = store.repo();
    Ok(repo.has_object(held)?
        && repo.is_ancestor(&Rev::Oid(here.clone()), &Rev::Oid(held.clone()))?)
}
