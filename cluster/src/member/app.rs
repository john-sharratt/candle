//! One app of a member (§3.2, §5.4, §5.8): a socket to each of its clusters,
//! dialled and redialled on its own; what each gateway said — `MEMBERS`, the
//! claims — and the requests the gateways route here, handed to the host's
//! HTTP server as streams. The host asks it whether the app is joined, for a
//! stream up to a gateway, and whether a sticky key may be served here.

use std::future::pending;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::sync::watch;
use tokio::time::{interval, MissedTickBehavior};

use super::claims::{Check, MemberClaims};
use super::link::{Link, Links};
use crate::clock::now_ms;
use crate::close::{Close, CloseCode};
use crate::dial::{dial, Attempt, Connect, Stopped};
use crate::dispatch::RouteTable;
use crate::frame::{
    Admit, BuildId, Capacity, Claim, Frame, Hello, Members, ModelId, Routes, Unclaim,
};
use crate::handshake::{GroupKey, HandshakeError, PROTO};
use crate::session::{Ended, Incoming, Session, SessionHandle, VirtualConn};

/// How often `CAPACITY` is sent when nothing prompts it sooner (§4.1).
pub const CAPACITY_EVERY: Duration = Duration::from_secs(1);

/// Who this member is to an app's gateways.
#[derive(Debug, Clone)]
pub struct Identity {
    pub app: String,
    pub node: u64,
    pub host: String,
    pub build: BuildId,
    /// The model the substrate is bound to; `None` for an agent.
    pub model: Option<ModelId>,
    pub repos: Vec<String>,
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
    /// `routes`, measuring itself with `capacity`; dialling from `started`.
    pub fn new(
        id: Identity,
        key: GroupKey,
        clusters: Vec<String>,
        routes: RouteTable,
        capacity: CapacitySource,
        started: Instant,
    ) -> Arc<AppMember> {
        let n = clusters.len();
        Arc::new(AppMember {
            claims: Mutex::new(MemberClaims::new(id.node)),
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
        })
    }

    pub fn app(&self) -> &str {
        &self.id.app
    }

    pub fn node(&self) -> u64 {
        self.id.node
    }

    fn bump(&self) {
        self.changed.send_modify(|n| *n += 1);
    }

    /// Dial cluster `c` through `connector` until a terminal close, handing
    /// every stream a gateway opens to `streams`.
    pub async fn run<C, F>(self: Arc<Self>, c: usize, connector: C, streams: F) -> Stopped
    where
        C: Connect,
        F: Fn(VirtualConn) + Send + Sync,
    {
        let mut stop = self.stop.subscribe();
        let this = &self;
        let streams = &streams;
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

    async fn session<F>(&self, c: usize, admit: Admit, session: Session, streams: &F) -> Ended
    where
        F: Fn(VirtualConn) + Send + Sync,
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
        let _ = handle.control(Frame::Capacity((self.capacity)()));
        for (key, epoch) in lock(&self.claims).own() {
            let _ = handle.control(Frame::Claim(Claim { key, epoch }));
        }
        lock(&self.sockets)[c] = Some(handle.clone());
        {
            let mut d = lock(&self.dialling);
            d[c].view.state = "admitted";
            d[c].view.session = Some(admit.session);
            d[c].retry_at = None;
        }
        lock(&self.links).ready(c);
        self.bump();

        let mut incoming = session.incoming;
        let mut ended = session.ended;
        let mut tick = interval(CAPACITY_EVERY);
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let how = loop {
            tokio::select! {
                arrived = incoming.recv() => match arrived {
                    Some(Incoming::Frame(frame)) => self.on_frame(c, frame),
                    Some(Incoming::Stream(conn)) => streams(conn),
                    // No member of this build writes events, so none arrive.
                    Some(Incoming::Event { .. }) => {}
                    None => break (&mut ended).await.unwrap_or(Ended::Dropped),
                },
                _ = tick.tick() => {
                    let _ = handle.control(Frame::Capacity((self.capacity)()));
                }
            }
        };
        lock(&self.sockets)[c] = None;
        lock(&self.members)[c] = None;
        how
    }

    fn on_frame(&self, c: usize, frame: Frame) {
        match frame {
            Frame::Members(m) => lock(&self.members)[c] = Some(m),
            Frame::Claim(claim) => {
                lock(&self.claims).heard(&claim.key, claim.epoch);
            }
            Frame::Unclaim(u) => lock(&self.claims).heard_release(&u.key, u.epoch),
            // Consensus, catch-up and builds are not taken part in by a
            // member of this build.
            _ => {}
        }
    }

    /// The app's link, as its own listener decides by (§5.8).
    pub fn link(&self) -> Link {
        lock(&self.links).state(Instant::now())
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
        let sockets: Vec<SessionHandle> = lock(&self.sockets).iter().flatten().cloned().collect();
        if sockets.is_empty() {
            return None;
        }
        let at = self.rr.fetch_add(1, Ordering::Relaxed);
        (0..sockets.len()).find_map(|i| sockets[(at + i) % sockets.len()].open(&self.id.app).ok())
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

    /// Send `CAPACITY` now, on a change the host saw (§4.1).
    pub fn capacity_changed(&self) {
        self.announce(Frame::Capacity((self.capacity)()));
    }

    fn announce(&self, frame: Frame) {
        for handle in lock(&self.sockets).iter().flatten() {
            let _ = handle.control(frame.clone());
        }
    }

    /// Close every socket with `1001`: the member is shutting down.
    pub fn close_all(&self, reason: &str) {
        for handle in lock(&self.sockets).iter().flatten() {
            handle.close(Close::new(CloseCode::GoingAway, reason));
        }
    }
}
