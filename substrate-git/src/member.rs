//! A daemon's repositories kept with its log (`docs/substrate_git.md` §5, §7,
//! §8): loaded in full during its loading phase, then followed.
//!
//! [`Repos::open`] opens a store per repository, every fetch made by an agent
//! holding the repository by name: the cluster's, or with no cluster the one
//! in this process. A daemon has no checkout and no way to `origin` of its
//! own. With a cluster, the lease `git/<repo>` is held through its consensus.
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
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use agent::client::Agents;
use agent::fetcher::AgentFetch;
use candle_conversation::persistence::SharedSubstrate;
use cluster::replicate::Replicator;
use tokio::runtime::Handle;
use zend_vfs::{Oid, RefName, RemoteUrl};

use crate::import::Importer;
use crate::records::{import_lease, Alone, Lease, Log};
use crate::store::{Store, Synced};

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

/// How a daemon reaches its repositories: the agents every fetch is made by —
/// the cluster's, or with no cluster the one in this process — on the runtime
/// their jobs run on, and the cluster's consensus the import lease is held
/// through, when there is a cluster (`docs/substrate_git.md` §9).
pub struct Access {
    pub agents: Agents,
    pub runtime: Handle,
    /// `None` with no cluster: this daemon imports every repository itself.
    pub replicator: Option<Arc<Replicator>>,
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

/// One repository's state during a load.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Loading {
    pub repo: String,
    /// Packs held of those the log names.
    pub held: usize,
    pub listed: usize,
    /// Whether the store holds every branch where the log does.
    pub whole: bool,
    /// Why the repository is not whole yet, when there is more to say than
    /// its packs: a pull failing, or another member's pull awaited.
    pub error: Option<String>,
}

type OnChange = Arc<dyn Fn(&str) + Send + Sync>;

struct Inner {
    /// Each store and its importer.
    repos: Vec<(Store, Importer)>,
    shared: SharedSubstrate,
    on_change: Mutex<Option<OnChange>>,
}

/// A daemon's repositories in the substrate.
#[derive(Clone)]
pub struct Repos {
    inner: Arc<Inner>,
}

impl Repos {
    /// Open a store under `root` for every repository in `names`, every fetch
    /// made as `agent:<name>` through `access`'s agents — the cluster's, or
    /// with no cluster the one in this process. Nothing is adopted from a
    /// local checkout: the agents' view of origin is the only source, so a
    /// stale clone never leaks in.
    pub fn open(
        root: &Path,
        names: &[String],
        shared: SharedSubstrate,
        access: Access,
    ) -> anyhow::Result<Self> {
        let lease: Arc<dyn Lease> = match &access.replicator {
            Some(replicator) => Arc::new(ClusterLease {
                replicator: Arc::clone(replicator),
                renewed: Mutex::new(HashMap::new()),
            }),
            None => Arc::new(Alone),
        };
        let mut repos = Vec::new();
        for name in names {
            // Every fetch through an agent holding the repository by name: the
            // daemon has no checkout and no way to origin of its own.
            let store = Store::open(root, name)?.with_origin(
                &RemoteUrl::parse(&format!("agent:{name}"))?,
                AgentFetch::shared(access.agents.clone(), access.runtime.clone(), name.clone()),
            )?;
            let importer = Importer::new(Arc::clone(&lease)).fetching_every(FETCH_EVERY);
            repos.push((store, importer));
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
    /// now by whichever member holds its lease, and is whole when that pull
    /// finds origin empty; every other member waits for it, saying so in the
    /// report. Returns early, incomplete, when `cancelled` says so.
    pub fn load(
        &self,
        cancelled: &dyn Fn() -> bool,
        report: &dyn Fn(&[Loading]),
    ) -> anyhow::Result<()> {
        let mut state: Vec<Loading> = self
            .inner
            .repos
            .iter()
            .map(|(s, _)| Loading {
                repo: s.name().to_string(),
                ..Loading::default()
            })
            .collect();
        loop {
            for ((store, importer), loading) in self.inner.repos.iter().zip(state.iter_mut()) {
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
                        // A pull that found origin with no branch: an empty
                        // repository, whole as it is.
                        let empty = t.fetched
                            && store.imported()?.is_empty()
                            && branches(&self.inner.shared, store.name())?.is_empty();
                        loading.whole = empty || whole(store, &self.inner.shared)?;
                        loading.error = match failed_packs(&t.synced).or(t.fetch_error) {
                            Some(e) => Some(e),
                            // Not this member's to pull: it waits, as long as
                            // it takes, for the member holding the lease.
                            None if !t.fetched && !named => Some(format!(
                                "waiting for the member holding `{}` to pull it",
                                import_lease(store.name())
                            )),
                            None => None,
                        };
                    }
                    Err(e) => loading.error = Some(format!("{e:#}")),
                }
            }
            report(&state);
            if state.iter().all(|l| l.whole) || wait_or_stop(LOAD_ROUND, cancelled) {
                return Ok(());
            }
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
            .map(|(s, _)| s)
            .find(|s| s.name() == name)
    }

    /// Keep every repository with the log and with origin on a thread of its
    /// own, until `stopped` says so.
    pub fn follow(&self, stopped: Arc<dyn Fn() -> bool + Send + Sync>) -> anyhow::Result<()> {
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

/// How often a wait looks whether it was asked to stop.
const STOP_POLL: Duration = Duration::from_millis(100);

/// Wait `period`, ending early when `stopped` says so; whether it did.
fn wait_or_stop(period: Duration, stopped: &dyn Fn() -> bool) -> bool {
    let until = Instant::now() + period;
    loop {
        if stopped() {
            return true;
        }
        let left = until.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return false;
        }
        thread::sleep(left.min(STOP_POLL));
    }
}

/// Why a sync's packs could not all be taken — every branch waits on them —
/// or `None` when none failed.
fn failed_packs(synced: &Synced) -> Option<String> {
    let (sha, why) = synced.failed.first()?;
    Some(format!(
        "{} pack(s) cannot be taken, and no branch moves until they are; pack {sha}: {why}",
        synced.failed.len()
    ))
}

/// Whether `store` holds every branch the log names, where the log has it.
fn whole(store: &Store, shared: &SharedSubstrate) -> anyhow::Result<bool> {
    let log = branches(shared, store.name())?;
    Ok(!log.is_empty() && store.record()? == log)
}

fn follow(inner: Arc<Inner>, stopped: Arc<dyn Fn() -> bool + Send + Sync>) {
    let mut last_error: HashMap<String, String> = HashMap::new();
    // The record as last reported, per repository. A move is any change to it,
    // whatever made it: a sync taking another member's branch, or this
    // member's own push landing — which moves the record before the sync
    // sees the log agree. Empty at first, so the first round reports what
    // moved between the load and the follower starting.
    let mut reported: HashMap<String, BTreeMap<RefName, Oid>> = HashMap::new();
    loop {
        if wait_or_stop(TICK, &*stopped) {
            return;
        }
        for (store, importer) in &inner.repos {
            // Between repositories too: a stop waits for one repository's
            // round at most, never every repository's.
            if stopped() {
                return;
            }
            let name = store.name().to_string();
            let error = match importer.tick(store, &inner.shared) {
                Ok(t) => {
                    match store.record() {
                        Ok(record) if reported.get(&name) != Some(&record) => {
                            reported.insert(name.clone(), record);
                            if let Some(f) = inner.on_change.lock().unwrap().clone() {
                                f(&name);
                            }
                        }
                        Ok(_) => {}
                        Err(e) => tracing::warn!(repo = %name, "reading the record: {e}"),
                    }
                    failed_packs(&t.synced).or(t.fetch_error)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// **A stop ends a wait at once**, and an unasked wait runs its period.
    #[test]
    fn a_wait_ends_when_stopped_and_runs_its_period_otherwise() {
        let started = Instant::now();
        assert!(wait_or_stop(Duration::from_secs(30), &|| true));
        assert!(started.elapsed() < Duration::from_secs(1));
        let started = Instant::now();
        assert!(!wait_or_stop(Duration::from_millis(150), &|| false));
        assert!(started.elapsed() >= Duration::from_millis(150));
    }

    /// **A failed pack names itself and how many failed**; none, nothing.
    #[test]
    fn failed_packs_are_reported_by_name() {
        assert_eq!(failed_packs(&Synced::default()), None);
        let synced = Synced {
            failed: vec![("ab".into(), "bad".into()), ("cd".into(), "worse".into())],
            ..Synced::default()
        };
        let said = failed_packs(&synced).unwrap();
        assert!(said.starts_with("2 pack(s) cannot be taken"), "{said}");
        assert!(said.ends_with("pack ab: bad"), "{said}");
    }
}
