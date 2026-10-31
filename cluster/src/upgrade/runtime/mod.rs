//! A process's upgrades (`docs/replicated_substrate.md` §3.5, §3.8–§3.11): one
//! [`Upgrader`] per process, across every app it serves — a `motherd` installs
//! one binary for both. Each app's member hands it the build traffic its
//! sockets carry, and once a second it looks at what it knows and acts.
//!
//! | Concern | Where |
//! |---|---|
//! | what to install, under which lease, and what to fetch ahead | [`decide`] |
//! | the group's verdicts, rollouts and pauses, read from the substrate | [`facts`] |
//! | one app's store reads for them, in flight across ticks | [`facts_read`] |
//! | announcing what it holds, and a candidate's offer | [`announce`] |
//! | fetching an artifact, and serving one | [`fetch`] |
//! | a candidate's verdicts and its `certify` proposal | [`candidate`] |
//! | the upgrade commands: verdicts, pauses and rollouts by hand, rollback | [`commands`] |
//! | taking the lease, the install, `ok`, and handing the lease back | [`install`] |
//! | what it serves | [`served`] |
//! | the builds it never tries again | [`refused`] |
//! | what `COORD` reports | [`view`] |

mod announce;
mod candidate;
pub mod commands;
pub mod decide;
mod facts;
mod facts_read;
mod failed_install;
mod fetch;
mod install;
pub mod refused;
pub mod served;
mod setup;
mod view;

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tokio::time::{interval, MissedTickBehavior};

use super::artifacts::Probe;
use super::decisions::build_name;
use super::parts::Parts;
use super::watchdog::{Failure, FAILED};
use crate::coord::{KeptLine, VerdictLine};
use crate::frame::{BuildArtifacts, Frame};
use crate::member::AppMember;
use facts_read::FactsRead;
use failed_install::failed_install;

pub use decide::{prefetch, target, Facts, Lease, Target};
pub use install::{InstallJob, Installer, HANDBACK_TTL, LEASE_TTL};
pub use served::{HeldAs, Item, OfferFile};
pub use setup::Setup;

/// How often the upgrades look at what they know.
pub const TICK: Duration = Duration::from_secs(1);

/// How often a member announces what it holds, when nothing prompts it.
pub const ANNOUNCE_EVERY: Duration = Duration::from_secs(30);

/// The longest a tick waits for a store to answer what the upgrades ask it —
/// the census and the decisions. Each is an in-RAM read that answers in
/// milliseconds while the store's own thread is free, so a store that takes
/// this long is held by other work (on zend, anything holding the persistence
/// thread). It is the same 10 s after which replication reports records ahead
/// of a request not applied, so a store held this long says so in both places
/// at once; and it is well inside a `COORD` report's period, so the step that
/// names the stall is reported while the stall lasts.
pub const STORE_WAIT: Duration = Duration::from_secs(10);

#[derive(Debug, Default)]
struct State {
    /// Every build heard offered or held, its artifacts merged, by
    /// `(count, hash)`.
    known: BTreeMap<(u32, String), BuildArtifacts>,
    /// What it serves, once scanned.
    served: Vec<Item>,
    scanned: bool,
    fetch: Option<fetch::Fetching>,
    candidates: BTreeMap<String, candidate::Candidacy>,
    facts: Facts,
    /// The latest verdict on each build, per app, for `COORD`.
    verdicts: BTreeMap<String, Vec<VerdictLine>>,
    step: String,
    stranded: Option<String>,
    failure: Option<Failure>,
    announce_due: bool,
    last_announce: Option<Instant>,
    /// An install was handed to the installer: the process is stopping.
    installing: bool,
    /// Per app: its own build is rejected, and it stopped taking work.
    rejected_self: BTreeSet<String>,
    /// Upgrade leases the new process takes and hands back once ready.
    handback: install::Handback,
    /// When it last asked for an upgrade lease another member holds.
    lease_try: Option<Instant>,
    /// Every record type its substrates hold, as last read.
    census: Vec<u8>,
    /// The builds it keeps, as last scanned.
    kept: Vec<KeptLine>,
    /// A build an admin's `rollback` or `upgrade-now` named for this member.
    forced: Option<Target>,
    /// Per app: a facts read that has not answered yet.
    reads: BTreeMap<String, FactsRead>,
    /// The step while a store has not answered: what it is blocked on.
    blocked: Option<String>,
}

/// See the module documentation.
pub struct Upgrader {
    setup: Setup,
    parts: Parts,
    installer: Arc<dyn Installer>,
    probe: Arc<dyn Probe + Send + Sync>,
    apps: Mutex<Vec<Arc<AppMember>>>,
    s: Mutex<State>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl Upgrader {
    /// The upgrades of a process set up as `setup`, installing through
    /// `installer`, asking binaries their build through `probe`.
    pub fn new(
        setup: Setup,
        installer: Arc<dyn Installer>,
        probe: Arc<dyn Probe + Send + Sync>,
    ) -> Arc<Upgrader> {
        let dir = setup.upgrade_dir.clone();
        let mut refused = refused::read(&dir);
        if let Some(build) = &failed_install(&setup, &dir) {
            match refused::refuse(&dir, build) {
                Ok(r) => refused = r,
                Err(e) => tracing::error!("the failed build {build} was not recorded: {e}"),
            }
            tracing::warn!(
                build,
                "restored after a failed install: it is not tried again"
            );
        }
        let own = build_name(&setup.info.build());
        if refused.contains(&own) {
            match refused::forgive(&dir, &own) {
                Ok(r) => refused = r,
                Err(e) => tracing::error!("the running build {own} was not forgiven: {e}"),
            }
        }
        let failure = fs::read(dir.join(FAILED))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok());
        let facts = Facts {
            own: Some(setup.info.clone()),
            refused,
            ..Facts::default()
        };
        Arc::new(Upgrader {
            parts: Parts::new(&dir),
            setup,
            installer,
            probe,
            apps: Mutex::new(Vec::new()),
            s: Mutex::new(State {
                facts,
                failure,
                announce_due: true,
                ..State::default()
            }),
        })
    }

    fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.s)
    }

    pub fn setup(&self) -> &Setup {
        &self.setup
    }

    /// `member` is one of this process's apps: its build traffic comes here.
    pub fn register(self: &Arc<Self>, member: &Arc<AppMember>) {
        member.set_upgrader(Arc::downgrade(self));
        lock(&self.apps).push(member.clone());
        let mut s = self.state();
        s.facts.apps.push(member.app().to_string());
        s.facts
            .nodes
            .insert(member.app().to_string(), format!("{:016x}", member.node()));
    }

    fn apps(&self) -> Vec<Arc<AppMember>> {
        lock(&self.apps).clone()
    }

    fn member(&self, app: &str) -> Option<Arc<AppMember>> {
        self.apps().into_iter().find(|m| m.app() == app)
    }

    /// Start looking, once a second, on the runtime this is called on.
    pub fn start(self: &Arc<Self>) {
        self.start_every(TICK);
    }

    /// Start looking every `every`.
    pub fn start_every(self: &Arc<Self>, every: Duration) {
        let me = self.clone();
        tokio::spawn(async move {
            let mut tick = interval(every);
            tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                me.tick().await;
            }
        });
    }

    /// A socket of `app` was admitted: announce what it holds there.
    pub fn admitted(&self, _app: &str, _c: usize) {
        self.state().announce_due = true;
    }

    /// Build traffic on socket `c` of `app`.
    pub fn frame(self: &Arc<Self>, app: &str, c: usize, frame: Frame) {
        match frame {
            Frame::BuildOffer(b) | Frame::BuildHave(b) => self.heard(b),
            Frame::BuildFetch(f) => self.serve_fetch(app, c, f),
            Frame::BuildChunk(chunk) => self.on_chunk(chunk),
            Frame::Verdict(v) => self.on_verdict(app, c, v),
            _ => {}
        }
    }

    /// A build offered or held somewhere: its artifacts become known.
    fn heard(&self, b: BuildArtifacts) {
        let mut s = self.state();
        let entry = s
            .known
            .entry((b.build.count, b.build.hash.clone()))
            .or_insert_with(|| BuildArtifacts {
                build: b.build.clone(),
                artifacts: Vec::new(),
            });
        for a in b.artifacts {
            if !entry.artifacts.iter().any(|x| x.sha256 == a.sha256) {
                entry.artifacts.push(a);
            }
        }
        s.facts.known = s.known.values().map(|b| b.build.clone()).collect();
    }

    /// Once a second: read the group's decisions, act on its own build's
    /// verdicts, finish what an install left, certify, announce, fetch, and
    /// install.
    async fn tick(self: &Arc<Self>) {
        self.tick_within(STORE_WAIT).await;
    }

    /// [`Self::tick`], waiting up to `store_wait` for each store call. While a
    /// store has not answered, the facts are stale, so nothing decided from
    /// them runs — no certification proposal, no install; what an install
    /// left is still finished, and holdings still announced and fetched.
    async fn tick_within(self: &Arc<Self>, store_wait: Duration) {
        if self.state().installing {
            return;
        }
        if !self.state().scanned {
            self.scan().await;
        }
        let fresh = self.refresh_facts(store_wait).await;
        if fresh {
            self.own_verdicts().await;
        }
        self.finish_install().await;
        if fresh {
            self.candidate_tick(store_wait).await;
        }
        self.announce_tick().await;
        self.fetch_tick().await;
        if fresh {
            self.install_tick().await;
        }
    }
}

#[cfg(test)]
mod tick_tests;
