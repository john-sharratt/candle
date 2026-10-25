//! A daemon's repositories kept with its log (`docs/substrate_git.md` §5, §7,
//! §8): loaded in full during its loading phase, then followed.
//!
//! [`Repos::open`] opens a store per repository, every fetch made by an agent
//! holding the repository by name: the cluster's, or with no cluster the one
//! in this process. A daemon has no checkout and no way to `origin` of its
//! own. With a cluster, the lease `git/<repo>` is held through its consensus.
//!
//! [`Repos::load`] blocks until every repository is whole in its object
//! database: the log names its branches and every pack they need has arrived
//! and been taken in. A repository the log names is loaded from the log alone —
//! no fetch: origin's newer commits are the follower's, and fetching here made
//! a restart wait on origin for over a minute with nothing to bring. A
//! repository no member has pulled yet is pulled now, through an agent, by the
//! member holding its lease; every other member waits for it to replicate. `max_depth` and every other bound on what is *read into
//! conversations* play no part here: the store holds every object of every
//! branch.
//!
//! [`Repos::follow`] then keeps them on a thread: every tick publishes this
//! member's pushes and syncs; the lease holder fetches every [`FETCH_EVERY`].

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use agent::client::Agents;
use agent::fetcher::AgentFetch;
use candle_conversation::persistence::SharedSubstrate;
use cluster::replicate::{LeaseEvent, Replicator};
use tokio::runtime::Handle;
use tokio::sync::broadcast;
use zend_vfs::{ObjectDb, Oid, RefName, RemoteUrl};

use crate::import::{Importer, Took};
use crate::records::{import_lease, Alone, Lease, Log};
use crate::store::{open_db, Store, Synced};

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
///
/// **Held means held.** [`Replicator::acquire`] answers whether the request was
/// sent, not whether the lease was granted — the round completes later, and
/// its outcome arrives as a [`LeaseEvent`]. Taking the send for the grant had a
/// member that had just joined fetching as though it held a lease another
/// member held, beside it.
///
/// **A lease another member holds is not asked for.** A request raises the
/// ballot under the holder's renewal, and one every load round — a second —
/// kept the holder's term from completing, so nobody held it long enough to
/// pull and a joining member waited on the log for good. A refusal naming the
/// holder stands for one lease term; the lease is asked for again only once
/// that member's term could have run out unrenewed.
struct ClusterLease {
    replicator: Arc<Replicator>,
    renewed: Mutex<HashMap<String, Instant>>,
    events: Mutex<broadcast::Receiver<LeaseEvent>>,
    /// The member each lease was found held by, and when.
    busy: Mutex<HashMap<Vec<u8>, (u64, Instant)>>,
}

impl ClusterLease {
    fn new(replicator: Arc<Replicator>) -> Self {
        let events = Mutex::new(replicator.lease_events());
        Self {
            replicator,
            renewed: Mutex::new(HashMap::new()),
            events,
            busy: Mutex::new(HashMap::new()),
        }
    }

    /// Take in what became of the lease requests sent since the last look.
    fn take_events(&self) {
        let mut events = self.events.lock().unwrap();
        let mut busy = self.busy.lock().unwrap();
        loop {
            match events.try_recv() {
                Ok(LeaseEvent::Busy { key, holder }) => {
                    busy.insert(key, (holder, Instant::now()));
                }
                Ok(
                    LeaseEvent::Held { key }
                    | LeaseEvent::Lost { key }
                    | LeaseEvent::Released { key, .. },
                ) => {
                    busy.remove(&key);
                }
                // Fallen behind: what was missed is learned again from the
                // next refusal.
                Err(broadcast::error::TryRecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        }
    }
}

impl Lease for ClusterLease {
    fn hold(&self, key: &str) -> bool {
        self.take_events();
        let mut renewed = self.renewed.lock().unwrap();
        if self.replicator.holds(key.as_bytes()) {
            let fresh = renewed
                .get(key)
                .is_some_and(|at| at.elapsed() < LEASE_RENEW);
            if !fresh && self.replicator.acquire(key.as_bytes(), LEASE_TTL) {
                renewed.insert(key.to_string(), Instant::now());
            }
            return true;
        }
        renewed.remove(key);
        let held_elsewhere = self
            .busy
            .lock()
            .unwrap()
            .get(key.as_bytes())
            .is_some_and(|(_, at)| at.elapsed() < LEASE_TTL);
        if !held_elsewhere {
            // Asked for; granted, if it is, by a later round's `holds`.
            self.replicator.acquire(key.as_bytes(), LEASE_TTL);
        }
        false
    }

    fn holder(&self, key: &str) -> Option<u64> {
        self.take_events();
        self.busy
            .lock()
            .unwrap()
            .get(key.as_bytes())
            .filter(|(_, at)| at.elapsed() < LEASE_TTL)
            .map(|(holder, _)| *holder)
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
    /// Whether a round is running on it now.
    pub busy: bool,
    /// Whether this member, holding its lease, is fetching it from `origin`
    /// now — read while the round runs, so a pull reads as itself and not as
    /// the wait for the lease that came before it.
    pub pulling: bool,
    /// Answers the running round has taken in from origin, each a pack not yet
    /// in [`Self::held`] — read while the round runs, so a long first pull,
    /// many answers inside one round, shows its progress before the round
    /// ends. Zero between rounds, when `held` counts them.
    pub received: usize,
}

/// How often a load reports while its rounds run, so a pull in progress shows.
const REPORT_EVERY: Duration = Duration::from_millis(500);

type OnChange = Arc<dyn Fn(&str) + Send + Sync>;

struct Inner {
    /// Each store and its importer.
    repos: Vec<(Store, Importer)>,
    /// Each store's answers taken in from origin, ever (`AgentFetch`).
    taken: Vec<Arc<AtomicUsize>>,
    shared: SharedSubstrate,
    on_change: Mutex<Option<OnChange>>,
}

/// A daemon's repositories in the substrate.
#[derive(Clone)]
pub struct Repos {
    inner: Arc<Inner>,
}

impl Repos {
    /// Open the object database of every repository in `names`, over
    /// `shared`'s log, every fetch made as `agent:<name>` through `access`'s
    /// agents — the cluster's, or with no cluster the one in this process.
    /// Nothing is adopted from a local checkout: the agents' view of origin is
    /// the only source, so a stale clone never leaks in.
    pub fn open(names: &[String], shared: SharedSubstrate, access: Access) -> anyhow::Result<Self> {
        let mut dbs = Vec::new();
        for name in names {
            dbs.push(open_db(Arc::new(shared.clone()), name)?);
        }
        Self::over(dbs, shared, access)
    }

    /// [`Self::open`] over databases already open — [`open_db`] over
    /// `shared` — which every other holder of them goes on sharing: what the
    /// daemon reads and what is kept with the log are one database.
    pub fn over(
        dbs: Vec<ObjectDb>,
        shared: SharedSubstrate,
        access: Access,
    ) -> anyhow::Result<Self> {
        let lease: Arc<dyn Lease> = match &access.replicator {
            Some(replicator) => Arc::new(ClusterLease::new(Arc::clone(replicator))),
            None => Arc::new(Alone),
        };
        let mut repos = Vec::new();
        let mut taken = Vec::new();
        for db in dbs {
            let name = db.name().to_string();
            let answers = Arc::new(AtomicUsize::new(0));
            // Every fetch through an agent holding the repository by name: the
            // daemon has no checkout and no way to origin of its own.
            let store = Store::over(db)?.with_origin(
                &RemoteUrl::parse(&format!("agent:{name}"))?,
                AgentFetch::shared(
                    access.agents.clone(),
                    access.runtime.clone(),
                    name.clone(),
                    Arc::clone(&answers),
                ),
            )?;
            let importer = Importer::new(Arc::clone(&lease)).fetching_every(FETCH_EVERY);
            repos.push((store, importer));
            taken.push(answers);
        }
        Ok(Self {
            inner: Arc::new(Inner {
                repos,
                taken,
                shared,
                on_change: Mutex::new(None),
            }),
        })
    }

    /// Block until every repository is whole in its store, reporting to
    /// `report` before the first round and as each repository's round ends.
    /// A repository the log names no branch of is pulled now by whichever
    /// member holds its lease, and is whole when that pull finds origin empty;
    /// every other member waits for it, saying so in the report. Returns
    /// early, incomplete, when `cancelled` says so.
    ///
    /// The repositories' rounds run side by side — each its own fetch, its
    /// own agent job and its own database — so a load takes as long as its
    /// slowest repository, not the sum of them.
    pub fn load(
        &self,
        cancelled: &dyn Fn() -> bool,
        report: &dyn Fn(&[Loading]),
    ) -> anyhow::Result<()> {
        let shared = &self.inner.shared;
        let mut state: Vec<Loading> = Vec::with_capacity(self.inner.repos.len());
        for (store, _) in &self.inner.repos {
            let (held, listed) = store.pack_counts(shared)?;
            state.push(Loading {
                repo: store.name().to_string(),
                held,
                listed,
                ..Loading::default()
            });
        }
        let mut took = vec![(0usize, Took::default()); state.len()];
        let taken_now = || -> Vec<usize> {
            self.inner
                .taken
                .iter()
                .map(|t| t.load(Ordering::Relaxed))
                .collect()
        };
        let started = Instant::now();
        report(&state);
        loop {
            for l in state.iter_mut().filter(|l| !l.whole) {
                l.busy = true;
            }
            // Answers a round takes are its packs: counted here while it runs,
            // in `held` once it ends.
            let at_round_start = taken_now();
            let received =
                |i: usize| self.inner.taken[i].load(Ordering::Relaxed) - at_round_start[i];
            report(&state);
            thread::scope(|s| -> anyhow::Result<()> {
                let (tx, rx) = mpsc::channel();
                for (i, ((store, importer), loading)) in
                    self.inner.repos.iter().zip(&state).enumerate()
                {
                    if loading.whole {
                        continue;
                    }
                    let tx = tx.clone();
                    let loading = loading.clone();
                    s.spawn(move || {
                        let _ = tx.send((i, round(store, importer, shared, loading)));
                    });
                }
                drop(tx);
                loop {
                    let (i, ended) = match rx.recv_timeout(REPORT_EVERY) {
                        Ok(got) => got,
                        // Rounds still running: whether each is fetching now,
                        // and how many answers its fetch has taken in.
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            let mut changed = false;
                            for (i, (l, (_, importer))) in
                                state.iter_mut().zip(&self.inner.repos).enumerate()
                            {
                                let pulling = l.busy && importer.fetching();
                                if pulling != l.pulling {
                                    l.pulling = pulling;
                                    if pulling {
                                        // The lease is held now: what the last
                                        // round said about waiting for it is over.
                                        l.error = None;
                                    }
                                    changed = true;
                                }
                                let got = received(i);
                                if got != l.received {
                                    l.received = got;
                                    changed = true;
                                }
                            }
                            if changed {
                                report(&state);
                            }
                            continue;
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    };
                    let (mut loading, t) = ended?;
                    loading.pulling = false;
                    loading.received = 0;
                    took[i].0 += 1;
                    took[i].1.add(&t);
                    if loading.whole {
                        let (rounds, t) = took[i];
                        tracing::info!(
                            repo = %loading.repo,
                            rounds,
                            packs = loading.listed,
                            publish_ms = t.publish.as_millis() as u64,
                            fetch_ms = t.fetch.as_millis() as u64,
                            sync_ms = t.sync.as_millis() as u64,
                            whole_after_ms = started.elapsed().as_millis() as u64,
                            "repository loaded"
                        );
                    }
                    state[i] = loading;
                    report(&state);
                }
                Ok(())
            })?;
            if state.iter().all(|l| l.whole) || wait_or_stop(LOAD_ROUND, cancelled) {
                return Ok(());
            }
        }
    }
}

/// One load round of one repository: settled to the log when the log names its
/// branches — whole from the log alone, with origin's newer commits left to the
/// follower — or pulled when the log names none of them yet. `loading` is its
/// state going in; the state coming out, and what the round's parts took.
fn round(
    store: &Store,
    importer: &Importer,
    shared: &SharedSubstrate,
    mut loading: Loading,
) -> anyhow::Result<(Loading, Took)> {
    loading.busy = false;
    let named = !branches(shared, store.name())?.is_empty();
    let ticked = if named {
        importer.settle(store, shared)
    } else {
        importer.pull(store, shared)
    };
    let t = match ticked {
        Ok(t) => t,
        Err(e) => {
            loading.error = Some(format!("{e:#}"));
            return Ok((loading, Took::default()));
        }
    };
    loading.held = t.synced.held;
    loading.listed = t.synced.listed;
    // A pull that found origin with no branch: an empty repository, whole as
    // it is.
    let empty =
        t.fetched && store.imported()?.is_empty() && branches(shared, store.name())?.is_empty();
    loading.whole = empty || whole(store, shared)?;
    loading.error = match failed_packs(&t.synced).or(t.fetch_error) {
        Some(e) => Some(e),
        // Not this member's to pull: it waits, as long as it takes, for the
        // member holding the lease.
        None if !t.fetched && !named => Some(match importer.holder(store.name()) {
            Some(holder) => format!(
                "waiting for member {holder:016x}, holding `{}`, to pull it",
                import_lease(store.name())
            ),
            None => format!(
                "waiting for the lease `{}` to pull it",
                import_lease(store.name())
            ),
        }),
        None => None,
    };
    Ok((loading, t.took))
}

impl Repos {
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
