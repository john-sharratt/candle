//! One app of a member (§3.2, §5.4, §5.8): a socket to each of its clusters,
//! dialled and redialled on its own; what each gateway said — `MEMBERS`, the
//! claims, the floor it was admitted under — and the requests the gateways
//! route here, handed to the host's HTTP server as streams. The host asks it
//! whether the app is joined, for a stream up to a gateway, and whether a
//! sticky key may be served here. Build traffic goes to the process's
//! [`Upgrader`]; a socket that admitted the member below the floor carries
//! nothing else (§3.5).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::future::pending;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, Weak};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::sync::watch;
use tokio::time::{interval, timeout, MissedTickBehavior};

use super::claims::{Check, MemberClaims};
use super::link::{Link, Links};
use crate::clock::now_ms;
use crate::close::{Close, CloseCode};
use crate::coord::{CoordState, UpgradeView};
use crate::dial::{dial, Attempt, Connect, Stopped};
use crate::dispatch::RouteTable;
use crate::frame::{
    Admit, BuildId, Capacity, Claim, Complete, Coord, Epoch, Frame, Hello, MemberInfo, MemberState,
    Members, ModelId, RepoAd, Routes, Unclaim,
};
use crate::handshake::{GroupKey, HandshakeError, PROTO};
use crate::replicate::{Phase, Replication, Replicator};
use crate::session::{Ended, Incoming, Session, SessionHandle, VirtualConn};
use crate::upgrade::decisions::{certify_prefix, latest, Settled, Verdict};
use crate::upgrade::runtime::Upgrader;

/// Work using one of this member's claims ([`AppMember::use_claim`]); dropping
/// it ends the use.
pub struct ClaimUse {
    member: Arc<AppMember>,
    key: Vec<u8>,
}

impl Drop for ClaimUse {
    fn drop(&mut self) {
        self.member.end_use(&self.key);
    }
}

impl fmt::Debug for ClaimUse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClaimUse")
            .field("key", &String::from_utf8_lossy(&self.key))
            .finish()
    }
}

/// How often `CAPACITY` is sent when nothing prompts it sooner (§4.1).
pub const CAPACITY_EVERY: Duration = Duration::from_secs(1);

/// A member with no substrate sends `COORD` every this many `CAPACITY`s.
const COORD_TICKS: u64 = 2;

/// The prefix of a stream's target that names another app's group (§5.1).
pub const GROUP: &str = "group:";

/// The longest a request waits for the records ahead of it to be applied.
pub const APPLY_WAIT: Duration = Duration::from_secs(10);

/// An admission replaces the previous socket. The close that ended that socket
/// described it, and it does not describe the one now admitted.
fn note_admitted(view: &mut ClusterView, session: u64) {
    view.state = "admitted";
    view.session = Some(session);
    view.last_close = None;
}

/// Who this member is to an app's gateways.
#[derive(Debug, Clone)]
pub struct Identity {
    pub app: String,
    pub node: u64,
    pub host: String,
    pub build: BuildId,
    /// The model the substrate is bound to; `None` for an agent.
    pub model: Option<ModelId>,
    pub repos: Vec<RepoAd>,
}

/// What the host measures for `CAPACITY` (§5.3).
pub type CapacitySource = Box<dyn Fn() -> Capacity + Send + Sync>;

/// One cluster's socket, as the loading page and `/v1/status` show it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ClusterView {
    pub url: String,
    /// `dialling`, `admitted`, `retrying` or `stopped`.
    pub state: &'static str,
    pub attempts: u64,
    pub session: Option<u64>,
    /// Seconds until the next attempt, while retrying.
    pub retry_in_s: Option<u64>,
    pub last_close: Option<String>,
}

#[derive(Debug, Clone)]
struct Dialling {
    view: ClusterView,
    retry_at: Option<Instant>,
}

pub struct AppMember {
    id: Identity,
    key: GroupKey,
    routes: RouteTable,
    capacity: CapacitySource,
    /// The app's substrate replication; an agent has no substrate (§11).
    replicator: Option<Arc<Replicator>>,
    links: Mutex<Links>,
    sockets: Mutex<Vec<Option<SessionHandle>>>,
    members: Mutex<Vec<Option<Members>>>,
    claims: Mutex<MemberClaims>,
    dialling: Mutex<Vec<Dialling>>,
    rr: AtomicUsize,
    /// Bumped on every change of a socket's state.
    changed: watch::Sender<u64>,
    /// Set when a 4410 stops the app on every cluster, or the member shuts
    /// down: no socket is dialled again.
    stop: watch::Sender<Option<Close>>,
    /// The floor each socket was admitted under (§3.5).
    floors: Mutex<Vec<Option<BuildId>>>,
    /// The process's upgrades, which take the build traffic (§3.9).
    upgrader: OnceLock<Weak<Upgrader>>,
    /// The group rejected this member's build: its `HELLO` says so (§3.11).
    rejected: AtomicBool,
    /// Told each key another member takes from this one
    /// ([`AppMember::on_release`]).
    released: OnceLock<ReleaseHook>,
    /// Told each key another member lets go, with the point its next holder
    /// must hold it through ([`AppMember::on_unclaim`]).
    unclaimed: OnceLock<UnclaimHook>,
    /// The `oseq` of a key's own last state record in this member's origin
    /// ([`AppMember::key_through`]).
    key_through: OnceLock<KeyThroughHook>,
    /// Keys let go that this member reported holding complete, re-sent whole on
    /// every admission until each is next claimed ([`AppMember::report_complete`]).
    reported: Mutex<BTreeSet<Vec<u8>>>,
}

/// What an app does when another member takes a key from its member.
pub type ReleaseHook = Box<dyn Fn(&[u8]) + Send + Sync>;

/// What an app does when another member lets a key go: the key, its last
/// holder, and the `oseq` in that holder's origin its next holder must hold
/// the key's records through.
pub type UnclaimHook = Box<dyn Fn(&[u8], u64, u64) + Send + Sync>;

/// The `oseq` of a key's own last state record in this member's origin, when
/// the app can name one.
pub type KeyThroughHook = Box<dyn Fn(&[u8]) -> Option<u64> + Send + Sync>;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The close that stopped the app — on another cluster, or by its own
/// shutdown — once one has.
async fn stopped_elsewhere(stop: &mut watch::Receiver<Option<Close>>) -> Close {
    loop {
        if let Some(close) = stop.borrow_and_update().clone() {
            return close;
        }
        if stop.changed().await.is_err() {
            return pending().await;
        }
    }
}

impl AppMember {
    /// The app as `id` on `clusters` (their URLs, for display), declaring
    /// `routes`, measuring itself with `capacity`, replicating through
    /// `replication`; dialling from `started`.
    pub fn new(
        id: Identity,
        key: GroupKey,
        clusters: Vec<String>,
        routes: RouteTable,
        capacity: CapacitySource,
        replication: Option<Replication>,
        started: Instant,
    ) -> Arc<AppMember> {
        let n = clusters.len();
        Arc::new_cyclic(|me| AppMember {
            claims: Mutex::new(MemberClaims::new(id.node)),
            replicator: replication.map(|r| Replicator::new(r, n, me.clone())),
            id,
            key,
            routes,
            capacity,
            links: Mutex::new(Links::new(n, started)),
            sockets: Mutex::new(vec![None; n]),
            members: Mutex::new(vec![None; n]),
            dialling: Mutex::new(
                clusters
                    .into_iter()
                    .map(|url| Dialling {
                        view: ClusterView {
                            url,
                            state: "dialling",
                            ..ClusterView::default()
                        },
                        retry_at: None,
                    })
                    .collect(),
            ),
            rr: AtomicUsize::new(0),
            changed: watch::channel(0).0,
            stop: watch::channel(None).0,
            floors: Mutex::new(vec![None; n]),
            upgrader: OnceLock::new(),
            rejected: AtomicBool::new(false),
            released: OnceLock::new(),
            unclaimed: OnceLock::new(),
            key_through: OnceLock::new(),
            reported: Mutex::new(BTreeSet::new()),
        })
    }

    /// Call `hook` with each key another member lets go, its last holder and
    /// the `UNCLAIM`'s `through`: what an app awaits before it reports the key
    /// complete ([`Self::report_complete`]). The first hook set is kept.
    pub fn on_unclaim(&self, hook: UnclaimHook) {
        let _ = self.unclaimed.set(hook);
    }

    /// Name a key's own last state record when this member lets the key go:
    /// its `UNCLAIM` then carries that `oseq`, not the member's whole origin, so
    /// a member holding that conversation may take its next turn whatever else
    /// of this one's it lacks (`docs/cooperative_admission.md` §2.2). A key the
    /// hook names nothing for goes with the origin's `oseq`. The first hook set
    /// is kept.
    pub fn key_through(&self, hook: KeyThroughHook) {
        let _ = self.key_through.set(hook);
    }

    /// The `oseq` a key let go is announced with: its own, when the app names
    /// one, otherwise `fallback`.
    fn through_for(&self, key: &[u8], fallback: u64) -> u64 {
        self.key_through
            .get()
            .and_then(|hook| hook(key))
            .unwrap_or(fallback)
    }

    /// Report `keys`, each let go by another member, as held complete here:
    /// sent now as `COMPLETE`, those not already reported, and again on every
    /// admission, until each is next claimed by anyone.
    pub fn report_complete(&self, keys: Vec<Vec<u8>>) {
        let fresh: Vec<Vec<u8>> = {
            let mut reported = lock(&self.reported);
            keys.into_iter()
                .filter(|k| reported.insert(k.clone()))
                .collect()
        };
        if !fresh.is_empty() {
            self.announce(Frame::Complete(Complete { keys: fresh }));
        }
    }

    /// The keys reported complete and not claimed since.
    pub fn reported_complete(&self) -> Vec<Vec<u8>> {
        lock(&self.reported).iter().cloned().collect()
    }

    /// Call `hook` with each key another member takes that this member held or
    /// last let go — a later claim heard. From then on that member writes it,
    /// so whatever the app keeps of it in memory is no longer current. A key
    /// this member let go and nobody has taken since is not reported: what the
    /// app keeps of it is still exact, and the common next request — the same
    /// conversation's next turn, placed back here — reuses it. The first hook
    /// set is kept.
    pub fn on_release(&self, hook: ReleaseHook) {
        let _ = self.released.set(hook);
    }

    /// Tell the app's hook `key` is no longer this member's. Never under the
    /// claims lock: the hook takes the app's own.
    fn lost(&self, key: &[u8]) {
        if let Some(hook) = self.released.get() {
            hook(key);
        }
    }

    /// Whether the group rejected this member's build, as its substrate's
    /// certification verdicts last said.
    pub fn set_rejected(&self, rejected: bool) {
        self.rejected.store(rejected, Ordering::Relaxed);
    }

    /// Read the verdicts on this member's build from its substrate, before
    /// its first `HELLO`.
    async fn read_rejected(&self) {
        let Some(r) = &self.replicator else {
            return;
        };
        let prefix = certify_prefix(self.app(), &self.id.build);
        let verdict = latest::<Verdict>(&r.decisions(&prefix).await, &prefix);
        self.set_rejected(verdict.is_some_and(|(_, v)| v.result == Settled::Fail));
    }

    /// Hand the build traffic to `upgrader` (§3.9).
    pub fn set_upgrader(&self, upgrader: Weak<Upgrader>) {
        let _ = self.upgrader.set(upgrader);
    }

    fn upgrader(&self) -> Option<Arc<Upgrader>> {
        self.upgrader.get()?.upgrade()
    }

    /// What `COORD` reports of upgrades, once the process has its upgrades.
    pub fn upgrade_view(&self) -> Option<UpgradeView> {
        Some(self.upgrader()?.view(self.app()))
    }

    /// The build this member runs.
    pub fn build(&self) -> &BuildId {
        &self.id.build
    }

    /// Whether a socket admitted under `floor` admitted this member below it,
    /// to upgrade (§3.5). A local build is never below the floor (§3.10).
    fn below(&self, floor: &BuildId) -> bool {
        !self.id.build.is_local() && floor.count > self.id.build.count
    }

    /// The floor's build, while some socket admitted this member below it.
    pub fn upgrading(&self) -> Option<BuildId> {
        lock(&self.floors)
            .iter()
            .flatten()
            .find(|f| self.below(f))
            .cloned()
    }

    /// The floor socket `c` was admitted under.
    pub fn floor(&self, c: usize) -> Option<BuildId> {
        lock(&self.floors).get(c).cloned().flatten()
    }

    /// The admitted sockets, by cluster.
    pub fn admitted_sockets(&self) -> Vec<usize> {
        lock(&self.sockets)
            .iter()
            .enumerate()
            .filter_map(|(c, h)| h.as_ref().map(|_| c))
            .collect()
    }

    /// Send `frame` on socket `c`'s control lane.
    pub fn send_on(&self, c: usize, frame: Frame) -> bool {
        let handle = lock(&self.sockets).get(c).cloned().flatten();
        handle.is_some_and(|h| h.control(frame).is_ok())
    }

    /// Send `frame` on socket `c`'s catch-up lane: spare capacity only (§7.4).
    pub fn send_catchup_on(&self, c: usize, frame: Frame) -> bool {
        let handle = lock(&self.sockets).get(c).cloned().flatten();
        handle.is_some_and(|h| h.catchup(frame).is_ok())
    }

    pub fn app(&self) -> &str {
        &self.id.app
    }

    pub fn node(&self) -> u64 {
        self.id.node
    }

    /// The machine's name its `HELLO` gives.
    pub fn host(&self) -> &str {
        &self.id.host
    }

    fn bump(&self) {
        self.changed.send_modify(|n| *n += 1);
    }

    /// Dial cluster `c` through `connector` until a terminal close, handing
    /// every stream a gateway opens to `streams` — once every record that
    /// arrived before it is applied (§6.4).
    pub async fn run<C, F>(self: Arc<Self>, c: usize, connector: C, streams: F) -> Stopped
    where
        C: Connect,
        F: Fn(VirtualConn) + Send + Sync + 'static,
    {
        if let Some(r) = &self.replicator {
            r.start();
        }
        self.read_rejected().await;
        let mut stop = self.stop.subscribe();
        let this = &self;
        let streams = &Arc::new(streams);
        let dialled = dial(
            connector,
            self.key.clone(),
            || this.hello(),
            |admit, session| this.session(c, admit, session, streams),
            |attempt, wait| this.attempted(c, attempt, wait),
        );
        let stopped = tokio::select! {
            stopped = dialled => stopped,
            // Another cluster closed the app 4410, or the member is shutting
            // down: this socket stops too, and is not dialled again.
            close = stopped_elsewhere(&mut stop) => {
                if let Some(h) = lock(&self.sockets)[c].take() {
                    h.close(Close::new(
                        CloseCode::GoingAway,
                        format!("the app stopped: {}", close.reason),
                    ));
                }
                Stopped::App(close)
            }
        };
        if let Stopped::App(close) = &stopped {
            lock(&self.links).stop(close.clone());
            self.stop.send_if_modified(|s| {
                let first = s.is_none();
                if first {
                    *s = Some(close.clone());
                }
                first
            });
        }
        {
            let mut d = lock(&self.dialling);
            d[c].view.state = "stopped";
            d[c].retry_at = None;
            d[c].view.last_close = Some(match &stopped {
                Stopped::Socket(close) | Stopped::App(close) => close.to_string(),
            });
        }
        self.bump();
        stopped
    }

    fn hello(&self) -> Hello {
        Hello {
            proto: PROTO,
            app: self.id.app.clone(),
            node: self.id.node,
            host: self.id.host.clone(),
            build: self.id.build.clone(),
            rejected: self.rejected.load(Ordering::Relaxed),
            model: self.id.model.clone(),
            power: (self.capacity)().power,
            repos: self.id.repos.clone(),
            now_ms: now_ms(),
            // `join` draws a fresh nonce.
            nonce: [0; 32],
        }
    }

    fn attempted(&self, c: usize, attempt: &Attempt, wait: Option<Duration>) {
        {
            let mut d = lock(&self.dialling);
            let entry = &mut d[c];
            entry.view.attempts += 1;
            entry.view.session = None;
            entry.view.state = "retrying";
            entry.retry_at = wait.map(|w| Instant::now() + w);
            let close = match attempt {
                Attempt::ConnectFailed(e) => Some(format!("could not connect: {e}")),
                Attempt::Handshake(HandshakeError::Closed(c) | HandshakeError::Refused(c)) => {
                    Some(c.to_string())
                }
                Attempt::Handshake(HandshakeError::Dropped) => {
                    Some("the socket dropped during the handshake".into())
                }
                Attempt::Session(Ended::Closed(c) | Ended::Refused(c)) => Some(c.to_string()),
                Attempt::Session(Ended::Dropped) => Some("the socket dropped".into()),
                Attempt::Session(Ended::Dead) => Some("no frame from the gateway for 45 s".into()),
            };
            // Logged as well as shown: the next admission clears the view, so
            // a socket that ends and is readmitted at once left no trace of why.
            // Offline, every dial is refused by design, every few seconds:
            // that is the state, not news.
            if crate::offline::is_offline() {
                tracing::debug!(
                    app = %self.app(),
                    cluster = c,
                    attempt = entry.view.attempts,
                    "offline: the cluster socket is not dialled"
                );
            } else {
                tracing::warn!(
                    app = %self.app(),
                    cluster = c,
                    attempt = entry.view.attempts,
                    why = close.as_deref().unwrap_or("—"),
                    "a cluster socket ended"
                );
            }
            entry.view.last_close = close;
        }
        lock(&self.links).failed(c);
        self.bump();
    }

    async fn session<F>(&self, c: usize, admit: Admit, session: Session, streams: &Arc<F>) -> Ended
    where
        F: Fn(VirtualConn) + Send + Sync + 'static,
    {
        let handle = session.handle.clone();
        let lost: Vec<Vec<u8>> = {
            let mut claims = lock(&self.claims);
            admit
                .claims
                .iter()
                .filter(|claim| claims.heard(&claim.key, claim.epoch))
                .map(|claim| claim.key.clone())
                .collect()
        };
        for key in &lost {
            self.lost(key);
        }
        if admit.routes_wanted {
            let _ = handle.control(Frame::Routes(Routes {
                table: self.routes.encode(),
            }));
        }
        // Admitted below the floor, the member upgrades: it takes no traffic,
        // applies and sends no events, holds no claim and does not vote
        // (§3.5). Its replication waits for the build that replaces it.
        let floor = BuildId {
            count: admit.floor_count,
            hash: admit.floor_hash.clone(),
            local: String::new(),
        };
        let upgrading = self.below(&floor);
        lock(&self.floors)[c] = Some(floor);
        if !upgrading {
            for (key, epoch) in lock(&self.claims).own() {
                let _ = handle.control(Frame::Claim(Claim { key, epoch }));
            }
            // A gateway that admits this member again has forgotten what it
            // reported: each key still unclaimed goes to it whole.
            let reported = self.reported_complete();
            if !reported.is_empty() {
                let _ = handle.control(Frame::Complete(Complete { keys: reported }));
            }
        }
        lock(&self.sockets)[c] = Some(handle.clone());
        if let Some(r) = self.replicator.as_ref().filter(|_| !upgrading) {
            r.admitted(c, handle.clone());
        }
        let _ = handle.control(Frame::Capacity(self.capacity_for(c)));
        if let Some(u) = self.upgrader() {
            u.admitted(self.app(), c);
        }
        {
            let mut d = lock(&self.dialling);
            note_admitted(&mut d[c].view, admit.session);
            d[c].retry_at = None;
        }
        lock(&self.links).ready(c);
        self.bump();

        let mut incoming = session.incoming;
        let mut ended = session.ended;
        let mut tick = interval(CAPACITY_EVERY);
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut ticks: u64 = 0;
        let how = loop {
            tokio::select! {
                arrived = incoming.recv() => match arrived {
                    Some(Incoming::Frame(frame)) => self.on_frame(c, frame, upgrading),
                    // Nothing is routed to an upgrading member; a stream that
                    // raced the admission is refused.
                    Some(Incoming::Stream(conn)) if upgrading => conn.reset(),
                    Some(Incoming::Stream(conn)) => match &self.replicator {
                        // Served behind the records the gateway relayed
                        // ahead of it, so it sees what they wrote.
                        Some(r) => {
                            let applied = r.applied_so_far();
                            let streams = streams.clone();
                            tokio::spawn(async move {
                                if timeout(APPLY_WAIT, applied).await.is_err() {
                                    tracing::warn!(
                                        "records ahead of a request are not applied after {:?}: \
                                         serving it",
                                        APPLY_WAIT
                                    );
                                }
                                streams(conn);
                            });
                        }
                        None => streams(conn),
                    },
                    Some(Incoming::Event { whole, .. }) => {
                        if let Some(r) = self.replicator.as_ref().filter(|_| !upgrading) {
                            r.event(c, whole);
                        }
                    }
                    None => break (&mut ended).await.unwrap_or(Ended::Dropped),
                },
                _ = tick.tick() => {
                    let _ = handle.control(Frame::Capacity(self.capacity_for(c)));
                    ticks += 1;
                    // A member with no substrate has no replication to send
                    // its COORD: it reports its upgrade itself (§12.1).
                    if self.replicator.is_none() && ticks.is_multiple_of(COORD_TICKS) {
                        if let Some(u) = self.upgrader() {
                            let state = CoordState {
                                mode: "joined".into(),
                                upgrade: Some(u.view(self.app())),
                                ..CoordState::default()
                            };
                            let _ = handle.control(Frame::Coord(Coord {
                                state: state.encode(),
                            }));
                        }
                    }
                }
            }
        };
        lock(&self.sockets)[c] = None;
        lock(&self.members)[c] = None;
        lock(&self.floors)[c] = None;
        if let Some(r) = &self.replicator {
            r.closed(c);
        }
        how
    }

    fn on_frame(&self, c: usize, frame: Frame, upgrading: bool) {
        match frame {
            Frame::Members(m) => {
                if let Some(r) = &self.replicator {
                    r.frame(c, Frame::Members(m.clone()));
                }
                lock(&self.members)[c] = Some(m);
            }
            Frame::Claim(claim) => {
                // Claimed again: whatever this member reported of it is for a
                // release that is over.
                lock(&self.reported).remove(&claim.key);
                let lost = lock(&self.claims).heard(&claim.key, claim.epoch);
                if lost {
                    self.lost(&claim.key);
                }
            }
            Frame::Unclaim(u) => {
                lock(&self.claims).heard_release(&u.key, u.epoch);
                if let Some(hook) = self.unclaimed.get() {
                    hook(&u.key, u.epoch.node, u.through);
                }
            }
            // Build traffic is the process's upgrades' (§3.9).
            f @ (Frame::BuildOffer(_)
            | Frame::BuildHave(_)
            | Frame::BuildFetch(_)
            | Frame::BuildChunk(_)
            | Frame::Verdict(_)) => {
                if let Some(u) = self.upgrader() {
                    u.frame(self.app(), c, f);
                }
            }
            // Substrate and consensus traffic is the replicator's, but for a
            // member upgrading, which takes part in neither.
            other => {
                if let Some(r) = self.replicator.as_ref().filter(|_| !upgrading) {
                    r.frame(c, other);
                }
            }
        }
    }

    /// What `CAPACITY` reports on socket `c`: `upgrading` when that socket
    /// admitted it below the floor (§3.5), `fault` with a full disk or a failed
    /// certification (§7.7, §3.8), `draining` while drained (§12.4), and ready
    /// only once the member is in the group and that socket's admission
    /// catch-up has finished (§7.2).
    fn capacity_for(&self, c: usize) -> Capacity {
        let mut cap = (self.capacity)();
        if self.floor(c).is_some_and(|f| self.below(&f)) {
            cap.state = MemberState::Upgrading;
            return cap;
        }
        if let Some(r) = &self.replicator {
            if r.faulted() {
                cap.state = MemberState::Fault;
            } else if r.drained() {
                cap.state = MemberState::Draining;
            } else if cap.state == MemberState::Ready && !r.socket_ready(c) {
                cap.state = MemberState::Loading;
            }
        }
        cap
    }

    /// The app's link, as its own listener decides by (§5.8): its sockets'
    /// state, and where it stands in the group (§6.6, §6.7).
    pub fn link(&self) -> Link {
        let sockets = self.socket_link();
        let Some(r) = &self.replicator else {
            return sockets;
        };
        match (sockets, r.phase()) {
            (Link::Stopped(close), _) => Link::Stopped(close),
            (_, Phase::Reloading) => Link::Reloading,
            (Link::Joined, Phase::Rejoining) => Link::Rejoining,
            (Link::Joined, Phase::CutOff) => Link::CutOff,
            (sockets, _) => sockets,
        }
    }

    /// The link by its sockets alone.
    pub fn socket_link(&self) -> Link {
        lock(&self.links).state(Instant::now())
    }

    /// The app's replication, when it has a substrate.
    pub fn replicator(&self) -> Option<&Arc<Replicator>> {
        self.replicator.as_ref()
    }

    /// A receiver that wakes on every change of a socket's state.
    pub fn watch(&self) -> watch::Receiver<u64> {
        self.changed.subscribe()
    }

    /// Every cluster's socket as it stands.
    pub fn clusters(&self) -> Vec<ClusterView> {
        let now = Instant::now();
        lock(&self.dialling)
            .iter()
            .map(|d| {
                let mut v = d.view.clone();
                v.retry_in_s = d
                    .retry_at
                    .map(|at| at.saturating_duration_since(now).as_secs());
                v
            })
            .collect()
    }

    /// A stream up to a gateway for a request this member's own listener took
    /// (§5.8), dealt round robin across the admitted sockets.
    pub fn open_up(&self) -> Option<VirtualConn> {
        self.open_to(&self.id.app)
    }

    /// Whether any socket is admitted: a stream up can be opened.
    pub fn admitted(&self) -> bool {
        lock(&self.sockets).iter().any(Option::is_some)
    }

    /// A stream to another app's group through a gateway — `group:agent` for
    /// a job (§5.1, §11.5) — dealt round robin across the admitted sockets.
    pub fn open_group(&self, app: &str) -> Option<VirtualConn> {
        self.open_to(&format!("{GROUP}{app}"))
    }

    fn open_to(&self, target: &str) -> Option<VirtualConn> {
        let sockets: Vec<SessionHandle> = lock(&self.sockets).iter().flatten().cloned().collect();
        if sockets.is_empty() {
            return None;
        }
        let at = self.rr.fetch_add(1, Ordering::Relaxed);
        (0..sockets.len()).find_map(|i| sockets[(at + i) % sockets.len()].open(target).ok())
    }

    /// Whether `node` is a member of any cluster this member is joined to.
    fn live(&self, node: u64) -> bool {
        node == self.id.node
            || lock(&self.members)
                .iter()
                .flatten()
                .any(|m| m.members.iter().any(|x| x.node == node))
    }

    /// Before serving sticky `key` (§5.4): served here, or `Held` by another
    /// live member. A key taken is announced on every socket.
    pub fn claim(&self, key: &[u8]) -> Check {
        let check = {
            let mut claims = lock(&self.claims);
            claims.serve(key, now_ms(), Instant::now(), |n| self.live(n))
        };
        if let Check::Took(epoch) = check {
            self.announce(Frame::Claim(Claim {
                key: key.to_vec(),
                epoch,
            }));
        }
        check
    }

    /// Let go of the claims idle for `idle`, announcing each with this
    /// member's `oseq` at `through`.
    pub fn release_idle(&self, idle: Duration, through: u64) -> usize {
        let gone = lock(&self.claims).idle(Instant::now(), idle);
        for (key, epoch) in &gone {
            self.announce(Frame::Unclaim(Unclaim {
                key: key.clone(),
                epoch: *epoch,
                through: self.through_for(key, through),
            }));
        }
        gone.len()
    }

    /// The live member holding `key`'s claim, if one does — without claiming
    /// it.
    pub fn holder_of(&self, key: &[u8]) -> Option<u64> {
        let holder = lock(&self.claims).holder(key)?;
        self.live(holder).then_some(holder)
    }

    /// The claims this member holds, by key.
    pub fn own_claims(&self) -> Vec<(Vec<u8>, Epoch)> {
        lock(&self.claims).own()
    }

    /// Every member of the app this member sees across its clusters — the
    /// union of their `MEMBERS` lists, one entry per node, in node order. What
    /// rendezvous weighs (§9.5).
    pub fn group(&self) -> Vec<MemberInfo> {
        let mut by_node: BTreeMap<u64, MemberInfo> = BTreeMap::new();
        for m in lock(&self.members).iter().flatten() {
            for info in &m.members {
                by_node.entry(info.node).or_insert_with(|| info.clone());
            }
        }
        by_node.into_values().collect()
    }

    /// Route `key` here, for a lease this member holds (§9.6): its claim is
    /// taken and announced on every socket once, and kept until
    /// [`Self::let_go`].
    pub fn hold(&self, key: &[u8]) {
        let taken = lock(&self.claims).hold(key, now_ms(), Instant::now());
        if let Some(epoch) = taken {
            self.announce(Frame::Claim(Claim {
                key: key.to_vec(),
                epoch,
            }));
        }
    }

    /// Stop routing `key` here, its lease gone; `through` is this member's own
    /// `oseq`, which the next holder waits for.
    pub fn let_go(&self, key: &[u8], through: u64) {
        let Some(epoch) = lock(&self.claims).let_go(key) else {
            return;
        };
        self.announce(Frame::Unclaim(Unclaim {
            key: key.to_vec(),
            epoch,
            through: self.through_for(key, through),
        }));
    }

    /// Hold each claim only while work uses it, and hand it back the moment the
    /// last use ends (`docs/cooperative_admission.md` §2.2): an app whose every
    /// piece of work is placed afresh — a conversation's next turn on whichever
    /// member serves it soonest — rather than kept where it last ran.
    pub fn follow_use(&self) {
        lock(&self.claims).follow_use();
    }

    /// Whether this member's claims last only while used.
    pub fn follows_use(&self) -> bool {
        lock(&self.claims).follows_use()
    }

    /// Work using this member's claim on `key` — a request being answered, a
    /// turn running — until the guard drops. When claims follow use and that
    /// was the last, the claim goes, announced with this member's own `oseq`
    /// as it stands then: what the key's next holder must have applied.
    pub fn use_claim(self: &Arc<Self>, key: &[u8]) -> ClaimUse {
        lock(&self.claims).begin_use(key);
        ClaimUse {
            member: Arc::clone(self),
            key: key.to_vec(),
        }
    }

    /// This member's own records are written through this `oseq`; 0 with no
    /// substrate to replicate.
    fn own_through(&self) -> u64 {
        self.replicator.as_ref().map_or(0, |r| r.own_through())
    }

    fn end_use(&self, key: &[u8]) {
        let Some(epoch) = lock(&self.claims).end_use(key) else {
            return;
        };
        self.announce(Frame::Unclaim(Unclaim {
            key: key.to_vec(),
            epoch,
            through: self.through_for(key, self.own_through()),
        }));
    }

    /// Hand back every claim this member holds — it is about to stop taking
    /// work — announcing each with this member's `oseq` at `through`.
    pub fn release_all(&self, through: u64) -> usize {
        self.release_idle(Duration::ZERO, through)
    }

    /// Send `CAPACITY` now, on a change the host saw (§4.1).
    pub fn capacity_changed(&self) {
        let sockets: Vec<(usize, SessionHandle)> = lock(&self.sockets)
            .iter()
            .enumerate()
            .filter_map(|(c, h)| h.clone().map(|h| (c, h)))
            .collect();
        for (c, handle) in sockets {
            let _ = handle.control(Frame::Capacity(self.capacity_for(c)));
        }
    }

    fn announce(&self, frame: Frame) {
        for handle in lock(&self.sockets).iter().flatten() {
            let _ = handle.control(frame.clone());
        }
    }

    /// Close every socket with `1001`.
    pub fn close_all(&self, reason: &str) {
        for handle in lock(&self.sockets).iter().flatten() {
            handle.close(Close::new(CloseCode::GoingAway, reason));
        }
    }

    /// The member is shutting down: its replication stops, its dialling ends
    /// and its sockets close `1001`, before the tenant releases its substrate.
    /// It does not dial again: a member that rejoined between this and the
    /// process ending would be admitted with its replication stopped, and
    /// counted in the group it has left.
    pub fn shutdown(&self, reason: &str) {
        if let Some(r) = &self.replicator {
            r.shutdown();
        }
        let close = Close::new(CloseCode::GoingAway, reason);
        self.stop.send_if_modified(|s| {
            let first = s.is_none();
            if first {
                *s = Some(close.clone());
            }
            first
        });
        self.close_all(reason);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A member on no cluster: its announcements go nowhere, its claims are its
    /// own.
    fn alone() -> Arc<AppMember> {
        AppMember::new(
            Identity {
                app: "zend".into(),
                node: 1,
                host: "m".into(),
                build: BuildId {
                    count: 1,
                    hash: "h".into(),
                    local: String::new(),
                },
                model: None,
                repos: Vec::new(),
            },
            GroupKey::from_bytes([1; 32]),
            Vec::new(),
            RouteTable::new(Vec::new()).unwrap(),
            Box::new(Capacity::default),
            None,
            Instant::now(),
        )
    }

    /// **A claim following use goes with its last use; the app is told only
    /// when another member takes the key.** A request and a turn overlap: the
    /// request ending first keeps it, the turn ending lets it go. Nobody has
    /// taken it, so what the app keeps is still exact and it is told nothing;
    /// another member's claim heard next tells it once.
    #[test]
    fn a_key_let_go_is_reported_when_another_member_takes_it() {
        let member = alone();
        member.follow_use();
        let told: Arc<Mutex<Vec<Vec<u8>>>> = Arc::default();
        let into = Arc::clone(&told);
        member.on_release(Box::new(move |key| lock(&into).push(key.to_vec())));

        assert!(matches!(member.claim(b"c-1"), Check::Took(_)));
        let request = member.use_claim(b"c-1");
        let turn = member.use_claim(b"c-1");
        drop(request);
        assert_eq!(member.holder_of(b"c-1"), Some(1), "the turn still uses it");
        drop(turn);
        assert_eq!(member.holder_of(b"c-1"), None);
        assert!(lock(&told).is_empty(), "let go, and nobody took it");

        let taken = Claim {
            key: b"c-1".to_vec(),
            epoch: Epoch {
                ms: now_ms() + 1_000,
                node: 2,
            },
        };
        member.on_frame(0, Frame::Claim(taken.clone()), false);
        assert_eq!(*lock(&told), vec![b"c-1".to_vec()]);
        member.on_frame(0, Frame::Claim(taken), false);
        assert_eq!(lock(&told).len(), 1, "told once");
    }

    /// Without following use the claim outlives its uses, and nothing is told.
    #[test]
    fn a_member_not_following_use_keeps_the_claim() {
        let member = alone();
        let told: Arc<Mutex<Vec<Vec<u8>>>> = Arc::default();
        let into = Arc::clone(&told);
        member.on_release(Box::new(move |key| lock(&into).push(key.to_vec())));
        member.claim(b"c-1");
        drop(member.use_claim(b"c-1"));
        assert_eq!(member.holder_of(b"c-1"), Some(1));
        assert!(lock(&told).is_empty());
    }

    /// **Completeness is reported once per release, and ends with the next
    /// claim.** Another member's release reaches the app with its holder and
    /// its point; a key reported complete stays reported — re-sent on every
    /// admission — until anyone claims it again, and reporting it twice sends
    /// nothing new. A key let go here goes with the point the app names for it,
    /// or the origin's when it names none.
    #[test]
    fn a_key_reported_complete_stays_reported_until_it_is_claimed() {
        /// A release heard: its key, its last holder and its point.
        type Heard = Vec<(Vec<u8>, u64, u64)>;
        let member = alone();
        let heard: Arc<Mutex<Heard>> = Arc::default();
        let into = Arc::clone(&heard);
        member.on_unclaim(Box::new(move |key, by, through| {
            lock(&into).push((key.to_vec(), by, through));
        }));
        let gone = Unclaim {
            key: b"c-1".to_vec(),
            epoch: Epoch { ms: 5, node: 2 },
            through: 900,
        };
        member.on_frame(0, Frame::Unclaim(gone), false);
        assert_eq!(*lock(&heard), vec![(b"c-1".to_vec(), 2, 900)]);

        member.report_complete(vec![b"c-1".to_vec(), b"c-2".to_vec()]);
        member.report_complete(vec![b"c-1".to_vec()]);
        assert_eq!(
            member.reported_complete(),
            vec![b"c-1".to_vec(), b"c-2".to_vec()]
        );
        let taken = Claim {
            key: b"c-1".to_vec(),
            epoch: Epoch { ms: 6, node: 3 },
        };
        member.on_frame(0, Frame::Claim(taken), false);
        assert_eq!(member.reported_complete(), vec![b"c-2".to_vec()]);

        assert_eq!(member.through_for(b"c-9", 40), 40, "no hook: the origin's");
        member.key_through(Box::new(|key| (key == b"c-9").then_some(7)));
        assert_eq!(member.through_for(b"c-9", 40), 7);
        assert_eq!(
            member.through_for(b"c-8", 40),
            40,
            "named nothing: the origin's"
        );
    }

    /// A socket that died and was admitted again is admitted. The close that
    /// killed the previous one stays out of the status body, and the attempt
    /// count is the history of dials, not a count of the socket now up.
    #[test]
    fn admission_clears_the_close_that_ended_the_previous_socket() {
        let mut view = ClusterView {
            url: "https://192.168.0.5:9443".into(),
            state: "retrying",
            attempts: 16,
            session: None,
            retry_in_s: Some(4),
            last_close: Some("no frame from the gateway for 45 s".into()),
        };
        note_admitted(&mut view, 50);
        assert_eq!(view.state, "admitted");
        assert_eq!(view.session, Some(50));
        assert_eq!(view.last_close, None);
        assert_eq!(view.attempts, 16);
        assert_eq!(view.retry_in_s, Some(4));
    }
}
