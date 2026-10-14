//! One app of a member (§3.2, §5.4, §5.8): a socket to each of its clusters,
//! dialled and redialled on its own; what each gateway said — `MEMBERS`, the
//! claims, the floor it was admitted under — and the requests the gateways
//! route here, handed to the host's HTTP server as streams. The host asks it
//! whether the app is joined, for a stream up to a gateway, and whether a
//! sticky key may be served here. Build traffic goes to the process's
//! [`Upgrader`]; a socket that admitted the member below the floor carries
//! nothing else (§3.5).

use std::collections::BTreeMap;
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
    Admit, BuildId, Capacity, Claim, Coord, Epoch, Frame, Hello, MemberInfo, MemberState, Members,
    ModelId, RepoAd, Routes, Unclaim,
};
use crate::handshake::{GroupKey, HandshakeError, PROTO};
use crate::replicate::{Phase, Replication, Replicator};
use crate::session::{Ended, Incoming, Session, SessionHandle, VirtualConn};
use crate::upgrade::decisions::{certify_prefix, latest, Settled, Verdict};
use crate::upgrade::runtime::Upgrader;

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
    /// Set when a 4410 stops the app on every cluster.
    stop: watch::Sender<Option<Close>>,
    /// The floor each socket was admitted under (§3.5).
    floors: Mutex<Vec<Option<BuildId>>>,
    /// The process's upgrades, which take the build traffic (§3.9).
    upgrader: OnceLock<Weak<Upgrader>>,
    /// The group rejected this member's build: its `HELLO` says so (§3.11).
    rejected: AtomicBool,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The close that stopped the app on another cluster, once one has.
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
        })
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
            // Another cluster closed the app 4410: this socket stops too.
            close = stopped_elsewhere(&mut stop) => {
                if let Some(h) = lock(&self.sockets)[c].take() {
                    h.close(Close::new(
                        CloseCode::GoingAway,
                        format!("the app stopped on another cluster: {}", close.reason),
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
        {
            let mut claims = lock(&self.claims);
            for claim in &admit.claims {
                claims.heard(&claim.key, claim.epoch);
            }
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
                lock(&self.claims).heard(&claim.key, claim.epoch);
            }
            Frame::Unclaim(u) => lock(&self.claims).heard_release(&u.key, u.epoch),
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
                through,
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
            through,
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

    /// The member is shutting down: its sockets close `1001` and its
    /// replication stops, before the tenant releases its substrate.
    pub fn shutdown(&self, reason: &str) {
        if let Some(r) = &self.replicator {
            r.shutdown();
        }
        self.close_all(reason);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
