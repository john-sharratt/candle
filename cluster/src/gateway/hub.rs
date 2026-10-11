//! A gateway's live cluster (§3–§5): every member socket from its `HELLO` to
//! its close, over whatever transport carries it. [`Hub::serve`] runs one
//! socket — the handshake, admission, the session, the frames the gateway
//! keeps or relays — and the rest answers the proxy: which member a request
//! goes to, a stream to it, and the state the status page shows.
//!
//! What it relays (§4.3):
//!
//! - a live `EVENT` to every other member of the app, but one upgrading, in
//!   fault mode or lagging; each sender is receipted for what is relayed;
//! - catch-up traffic — `HEADERS`, an addressed `EVENT`, `DONE` — to the one
//!   member its `reply_to` names; a `CATCHUP` to the member it names, or one
//!   other ready member round robin, or answered `DONE` here when there is
//!   none; a `FETCH` to the member it names; a `HAVE` to every other member;
//! - `CLAIM` and `UNCLAIM`, kept for routing and sent to the app's other
//!   members; `PREPARE`, `PROPOSE` and `RELEASE` to the app's other members;
//!   `PROMISE`, `NACK` and `ACCEPTED` to the member their `reply_to` names.
//!
//! It keeps `CAPACITY`, `ROUTES`, `ACK` and `COORD`, and sends `MEMBERS` on
//! every change and every 5 s.
//!
//! **Queues** (§4.4). A member whose ordered lane holds more than
//! [`LAG_CAP`] of live events unsent becomes **lagging**: no further live
//! event is queued for it and no request is routed to it, while what is
//! queued still drains. It is ready again once the lane is empty and its
//! `ACK`s cover every other member's `high_water`. A catch-up frame that would
//! take a member's catch-up lane past [`CATCHUP_CAP`] is dropped and counted;
//! the requester fetches again.

use std::collections::{HashMap, VecDeque};
use std::net::IpAddr;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use tokio::time::{interval, MissedTickBehavior};

use super::admission::{Admission, Admitted};
use super::coord_log::changes;
use super::log::{CoordLog, Entry};
use super::paxos_log::paxos_step;
use super::registry::{MemberView, Pick, Registry};
use super::status::{
    AppView, ClaimView, FaultRow, HubStatus, MemberRow, Queues, Refusal, SocketReport,
};
use super::traffic::Traffic;
use crate::clock::now_ms;
use crate::close::{Close, CloseCode};
use crate::coord::CoordState;
use crate::delivery::ReceiptPacer;
use crate::dispatch::{Dispatch, RouteTable};
use crate::frame::{name, Admit, Frame, Hello, Members};
use crate::handshake::{admit, GroupKey, HandshakeError, ProofLimiter};
use crate::pipe::{Message, PipeRx, PipeTx};
use crate::session::{start, Ended, Incoming, Role, SessionHandle, VirtualConn};

/// How often every member is sent `MEMBERS` whether or not anything changed.
pub const MEMBERS_EVERY: Duration = Duration::from_secs(5);

/// How many refused handshakes the status page lists.
pub const REFUSALS_KEPT: usize = 100;

/// The apps a member may join (§3.1).
pub const APPS: [&str; 3] = ["zend", "npcd", "agent"];

/// Live events unsent on a member's ordered lane past which it is lagging.
pub const LAG_CAP: usize = 256 << 20;

/// Bytes on a member's catch-up lane past which a catch-up frame is dropped.
pub const CATCHUP_CAP: usize = 16 << 20;

/// How often each socket's `RECEIPT` timer and lagging state are looked at.
const RECEIPT_TICK: Duration = Duration::from_millis(100);

/// A stream a member opened towards the gateway (§5.1): served by the
/// gateway's own router as a request for `app`.
pub struct MemberStream {
    pub app: String,
    pub node: u64,
    pub conn: VirtualConn,
}

pub(super) struct Socket {
    pub session: u64,
    pub handle: SessionHandle,
    pub connected_ms: u64,
    /// Its clock against this gateway's at admission, from its `HELLO`.
    pub clock_offset_ms: i64,
    /// Its last `COORD` (§12.1).
    pub coord: Option<CoordState>,
    pub pacer: ReceiptPacer,
    /// Its last `ACK`: what it has applied of each origin.
    pub acks: Vec<(u64, u64)>,
    /// Live events not queued for it while it lagged, and catch-up frames
    /// dropped at its cap.
    pub skipped: u64,
    pub dropped: u64,
}

#[derive(Debug, Clone, Copy, Default)]
struct Mode {
    clustered: bool,
    since_ms: u64,
}

#[derive(Default)]
pub(super) struct State {
    admission: Admission,
    pub registry: Registry,
    pub sockets: HashMap<(String, u64), Socket>,
    limiter: ProofLimiter,
    refusals: VecDeque<Refusal>,
    modes: HashMap<String, Mode>,
    last_close: HashMap<(String, u64), String>,
    /// Each member's last `COORD`, kept across its sockets: a member that
    /// reconnects is logged for what changed while it was away, not again for
    /// everything it reports.
    reported: HashMap<(String, u64), CoordState>,
    /// Admitting no sockets (§12.4 `drain-gateway`).
    pub draining: bool,
}

pub struct Hub {
    key: GroupKey,
    /// This gateway's name in the coordination log.
    name: String,
    state: Mutex<State>,
    pub log: CoordLog,
    pub traffic: Traffic,
}

pub(super) fn hex(node: u64) -> String {
    format!("{node:016x}")
}

impl Hub {
    pub fn new(key: GroupKey, name: impl Into<String>) -> Arc<Hub> {
        Arc::new(Hub {
            key,
            name: name.into(),
            state: Mutex::new(State::default()),
            log: CoordLog::new(),
            traffic: Traffic::new(),
        })
    }

    pub(super) fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub(super) fn note(&self, app: &str, subject: String, text: String) {
        self.log.push(Entry {
            at_ms: now_ms(),
            gateway: self.name.clone(),
            app: app.to_string(),
            subject,
            text,
        });
    }

    fn refused(&self, app: &str, ip: IpAddr, who: Option<(u64, String)>, close: &Close) {
        let (node, host) = who.map_or((String::new(), String::new()), |(n, h)| (hex(n), h));
        self.note(
            app,
            node.clone(),
            format!("refused {}: {}", close.code, close.reason),
        );
        let mut s = self.state();
        if s.refusals.len() == REFUSALS_KEPT {
            s.refusals.pop_back();
        }
        s.refusals.push_front(Refusal {
            at_ms: now_ms(),
            app: app.to_string(),
            ip: ip.to_string(),
            node,
            host,
            code: close.code,
            reason: close.reason.clone(),
        });
    }

    /// The admission of `hello` on the socket for `app` (§3.3–§3.6).
    fn decide(&self, app: &str, hello: &Hello) -> Result<(Admit, Admitted), Close> {
        if hello.app != app {
            return Err(Close::new(
                CloseCode::Malformed,
                format!("a HELLO for {} on the {app} socket", hello.app),
            ));
        }
        let mut s = self.state();
        let locked_before = s.admission.lock(app).is_some();
        let admitted = s.admission.decide(hello, now_ms())?;
        let lock = (!locked_before)
            .then(|| s.admission.lock(app).map(|m| m.describe.clone()))
            .flatten();
        let (_, members) = s.registry.listing(app);
        let claims = s.registry.claims(app).table();
        // The most any other member has acknowledged of this node's own
        // events: every one at or below it is held somewhere (§7.5).
        let own_applied_through = s
            .sockets
            .iter()
            .filter(|((a, n), _)| a == app && *n != hello.node)
            .flat_map(|(_, x)| x.acks.iter())
            .filter(|(origin, _)| *origin == hello.node)
            .map(|(_, through)| *through)
            .max()
            .unwrap_or(0);
        drop(s);
        if let Some(model) = lock {
            self.note(app, hex(hello.node), format!("model lock set: {model}"));
        }
        Ok((
            Admit {
                session: admitted.session,
                members,
                claims,
                floor_count: admitted.floor.count,
                floor_hash: admitted.floor.hash.clone(),
                // A new session has sent no table on this gateway yet.
                routes_wanted: true,
                own_applied_through,
            },
            admitted,
        ))
    }

    /// Run one member socket for `app`, from `ip`, until it closes. Streams
    /// the member opens are handed to `streams`.
    pub async fn serve<T, R, F>(
        self: &Arc<Self>,
        app: &str,
        ip: IpAddr,
        mut tx: T,
        mut rx: R,
        streams: F,
    ) where
        T: PipeTx,
        R: PipeRx,
        F: Fn(MemberStream) + Send + Sync,
    {
        if !self.state().limiter.allows(ip, Instant::now()) {
            let close = Close::new(
                CloseCode::Proof,
                format!(
                    "{ip} failed more than 10 proofs in a minute, and is held off for a minute"
                ),
            );
            let _ = tx.send(Message::Close(close.clone())).await;
            self.refused(app, ip, None, &close);
            return;
        }
        if self.draining() {
            let close = Close::new(
                CloseCode::GoingAway,
                "this gateway is draining: carry on through the other clusters",
            );
            let _ = tx.send(Message::Close(close)).await;
            return;
        }
        let mut seen = None;
        let mut decided = None;
        let handshake = admit(
            &mut tx,
            &mut rx,
            &self.key,
            |h| seen = Some((h.node, h.host.clone())),
            |h| {
                let (admit, admitted) = self.decide(app, h)?;
                decided = Some(admitted);
                Ok(admit)
            },
        )
        .await;
        let hello = match handshake {
            Ok((hello, _)) => hello,
            Err(e) => {
                // Admitted, and the ADMIT never reached it.
                if let (Some(a), Some((node, _))) = (&decided, &seen) {
                    self.state().admission.left(app, *node, a.session);
                }
                match e {
                    HandshakeError::Refused(close) => {
                        if close.code == CloseCode::Proof.code() {
                            self.state().limiter.failed(ip, Instant::now());
                        }
                        self.refused(app, ip, seen, &close);
                    }
                    HandshakeError::Closed(close) => self.refused(app, ip, seen, &close),
                    HandshakeError::Dropped => {}
                }
                return;
            }
        };
        let admitted = decided.expect("an ADMIT was decided");
        self.run(app, hello, admitted, tx, rx, streams).await;
    }

    async fn run<T, R, F>(
        &self,
        app: &str,
        hello: Hello,
        admitted: Admitted,
        tx: T,
        rx: R,
        streams: F,
    ) where
        T: PipeTx,
        R: PipeRx,
        F: Fn(MemberStream) + Send + Sync,
    {
        let node = hello.node;
        let session = start(Role::Gateway, tx, rx);
        let handle = session.handle.clone();
        {
            let mut s = self.state();
            let connected_ms = now_ms();
            let replaced = s.sockets.insert(
                (app.to_string(), node),
                Socket {
                    session: admitted.session,
                    handle: handle.clone(),
                    connected_ms,
                    clock_offset_ms: hello.now_ms as i64 - connected_ms as i64,
                    coord: None,
                    pacer: ReceiptPacer::new(Instant::now()),
                    acks: Vec::new(),
                    skipped: 0,
                    dropped: 0,
                },
            );
            if let Some(old) = replaced {
                old.handle.close(Close::new(
                    CloseCode::Replaced,
                    format!(
                        "node {} was admitted again, as session {}",
                        hex(node),
                        admitted.session
                    ),
                ));
            }
            s.registry.join(MemberView {
                node,
                host: hello.host.clone(),
                app: app.to_string(),
                build: hello.build.clone(),
                weight: hello.power,
                session: admitted.session,
                capacity: None,
                routes: None,
                high_water: 0,
                lagging: false,
            });
        }
        let local = if hello.build.is_local() {
            format!(", local {}", hello.build.local)
        } else {
            String::new()
        };
        self.note(
            app,
            hex(node),
            format!(
                "admitted {}: build {} ({}){local}, session {}{}",
                hello.host,
                hello.build.count,
                hello.build.hash,
                admitted.session,
                admitted
                    .replaces
                    .map_or(String::new(), |old| format!(", replacing session {old}")),
            ),
        );
        self.members_changed(app);

        let mut incoming = session.incoming;
        let mut ended = session.ended;
        let mut tick = interval(MEMBERS_EVERY);
        tick.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let mut receipts = interval(RECEIPT_TICK);
        receipts.set_missed_tick_behavior(MissedTickBehavior::Delay);
        let how = loop {
            tokio::select! {
                arrived = incoming.recv() => match arrived {
                    Some(Incoming::Frame(frame)) => {
                        if let Err(close) = self.on_frame(app, node, frame) {
                            handle.close(close.clone());
                            break Ended::Refused(close);
                        }
                    }
                    Some(Incoming::Stream(conn)) => streams(MemberStream {
                        app: app.to_string(),
                        node,
                        conn,
                    }),
                    Some(Incoming::Event { whole, frame, .. }) => {
                        self.relay_event(app, node, whole, frame);
                    }
                    None => break (&mut ended).await.unwrap_or(Ended::Dropped),
                },
                _ = tick.tick() => {
                    let (epoch, members) = self.state().registry.listing(app);
                    let _ = handle.control(Frame::Members(Members { epoch, members }));
                }
                _ = receipts.tick() => self.receipt_tick(app, node),
            }
        };

        let left = {
            let mut s = self.state();
            let key = (app.to_string(), node);
            if s.sockets
                .get(&key)
                .is_some_and(|x| x.session == admitted.session)
            {
                s.sockets.remove(&key);
            }
            let locked = s.admission.lock(app).is_some();
            s.admission.left(app, node, admitted.session);
            let cleared = locked && s.admission.lock(app).is_none();
            s.last_close.insert(key, describe(&how));
            (s.registry.leave(app, node, admitted.session), cleared)
        };
        self.note(app, hex(node), format!("socket closed: {}", describe(&how)));
        if left.1 {
            self.note(
                app,
                hex(node),
                "model lock cleared: the group is empty".into(),
            );
        }
        if left.0 {
            self.members_changed(app);
        }
    }

    /// A frame from `node`: kept, relayed, or a protocol violation.
    fn on_frame(&self, app: &str, node: u64, frame: Frame) -> Result<(), Close> {
        if let Some((key, text)) = paxos_step(node, &frame) {
            self.note(app, key, text);
        }
        match frame {
            Frame::Capacity(c) => {
                if self.state().registry.capacity(app, node, c) {
                    self.members_changed(app);
                }
            }
            Frame::Routes(r) => {
                let table = RouteTable::decode(&r.table)
                    .map_err(|e| Close::new(CloseCode::Malformed, format!("ROUTES: {e}")))?;
                self.state().registry.routes(app, node, table);
            }
            Frame::Claim(c) => {
                if c.epoch.node != node {
                    return Err(Close::new(
                        CloseCode::Malformed,
                        format!(
                            "a CLAIM from {} names node {}",
                            hex(node),
                            hex(c.epoch.node)
                        ),
                    ));
                }
                let took = self.state().registry.claims(app).claim(&c.key, c.epoch);
                if took {
                    self.note(
                        app,
                        String::from_utf8_lossy(&c.key).into_owned(),
                        format!("claimed by {} at {}", hex(node), c.epoch.ms),
                    );
                }
                self.to_others(app, node, Frame::Claim(c));
            }
            Frame::Unclaim(u) => {
                let released = self.state().registry.claims(app).unclaim(&u.key, u.epoch);
                if released {
                    self.note(
                        app,
                        String::from_utf8_lossy(&u.key).into_owned(),
                        format!("released by {} through oseq {}", hex(node), u.through),
                    );
                }
                self.to_others(app, node, Frame::Unclaim(u));
            }
            Frame::Coord(c) => {
                let state = CoordState::decode(&c.state).ok_or_else(|| {
                    Close::new(
                        CloseCode::Malformed,
                        "a COORD that is not a coordination state",
                    )
                })?;
                let lines = {
                    let mut s = self.state();
                    let key = (app.to_string(), node);
                    let Some(socket) = s.sockets.get_mut(&key) else {
                        return Ok(());
                    };
                    socket.coord = Some(state.clone());
                    let lines = changes(s.reported.get(&key), &state);
                    s.reported.insert(key, state);
                    lines
                };
                for line in lines {
                    self.note(app, hex(node), line);
                }
            }
            f @ (Frame::Prepare(_) | Frame::Propose(_) | Frame::Release(_)) => {
                self.to_others(app, node, f)
            }
            Frame::Promise(p) => {
                let to = p.reply_to;
                self.to_member(app, to, Frame::Promise(p));
            }
            Frame::Nack(n) => {
                let to = n.reply_to;
                self.to_member(app, to, Frame::Nack(n));
            }
            Frame::Accepted(a) => {
                let to = a.reply_to;
                self.to_member(app, to, Frame::Accepted(a));
            }
            f @ (Frame::Members(_) | Frame::Receipt(_) | Frame::Verdict(_)) => {
                return Err(Close::new(
                    CloseCode::Malformed,
                    format!("{} is the gateway's to send", name(f.kind())),
                ));
            }
            f @ (Frame::Ack(_)
            | Frame::Catchup(_)
            | Frame::Headers(_)
            | Frame::Fetch(_)
            | Frame::Done(_)
            | Frame::Have(_)) => self.relay_substrate(app, node, f),
            // What this gateway neither keeps nor relays: build traffic and
            // world pulses.
            _ => {}
        }
        Ok(())
    }

    pub(super) fn to_others(&self, app: &str, from: u64, frame: Frame) {
        let s = self.state();
        for ((a, n), socket) in &s.sockets {
            if a == app && *n != from {
                let _ = socket.handle.control(frame.clone());
            }
        }
    }

    pub(super) fn to_member(&self, app: &str, node: u64, frame: Frame) {
        if let Some(socket) = self.state().sockets.get(&(app.to_string(), node)) {
            let _ = socket.handle.control(frame);
        }
    }

    /// The app's membership changed: every member is told, and a mode switch
    /// is logged (§5.7).
    fn members_changed(&self, app: &str) {
        let (switched, clustered) = {
            let mut s = self.state();
            let (epoch, members) = s.registry.listing(app);
            for ((a, _), socket) in &s.sockets {
                if a == app {
                    let _ = socket.handle.control(Frame::Members(Members {
                        epoch,
                        members: members.clone(),
                    }));
                }
            }
            let clustered = s.registry.clustered(app);
            let mode = s.modes.entry(app.to_string()).or_default();
            let switched = mode.clustered != clustered;
            if switched {
                *mode = Mode {
                    clustered,
                    since_ms: now_ms(),
                };
            }
            (switched, clustered)
        };
        if switched {
            let text = if clustered {
                "clustered: a member is ready"
            } else {
                "dumb: no member is ready, routes go to their configured upstreams"
            };
            self.note(app, String::new(), text.into());
        }
    }

    /// Close every member socket with `1001`: this gateway is shutting down,
    /// and its members carry on through their other clusters (§3.4).
    pub fn close_all(&self, reason: &str) {
        for socket in self.state().sockets.values() {
            socket
                .handle
                .close(Close::new(CloseCode::GoingAway, reason));
        }
    }

    /// Where a request goes (§5.2).
    pub fn pick(
        &self,
        app: &str,
        dispatch: &Dispatch,
        key: Option<&str>,
        pinned: Option<u64>,
    ) -> Pick {
        self.state().registry.pick(app, dispatch, key, pinned)
    }

    /// Open a stream to `node`'s socket for `app`, towards `target`.
    pub fn open(&self, app: &str, node: u64, target: &str) -> Option<VirtualConn> {
        let handle = self
            .state()
            .sockets
            .get(&(app.to_string(), node))?
            .handle
            .clone();
        handle.open(target).ok()
    }

    /// The dispatch table `app`'s ready members declared.
    pub fn table(&self, app: &str) -> Option<RouteTable> {
        self.state().registry.table(app).cloned()
    }

    pub fn clustered(&self, app: &str) -> bool {
        self.state().registry.clustered(app)
    }

    pub fn member(&self, app: &str, node: u64) -> Option<MemberView> {
        self.state().registry.member(app, node).cloned()
    }

    /// `app`'s floor, model lock and member count, for an aggregated
    /// `/v1/status`'s `group` block (§5.5).
    pub fn group(&self, app: &str) -> AppView {
        app_view(&self.state(), app)
    }

    /// Everything the status page shows (§12.2).
    pub fn status(&self, log_limit: usize) -> HubStatus {
        let now = now_ms();
        let s = self.state();
        let mut members = Vec::new();
        for app in APPS {
            for m in s.registry.members(app) {
                let socket = s.sockets.get(&(app.to_string(), m.node));
                let report = socket.map_or(SocketReport::default(), |x| SocketReport {
                    connected_ms: x.connected_ms,
                    clock_offset_ms: x.clock_offset_ms,
                    coord: x.coord.clone(),
                    queues: Queues {
                        live_bytes: x.handle.gauges().live(),
                        catchup_bytes: x.handle.gauges().catchup(),
                        skipped: x.skipped,
                        dropped: x.dropped,
                    },
                });
                members.push(MemberRow::of(
                    m,
                    report,
                    s.last_close.get(&(app.to_string(), m.node)).cloned(),
                ));
            }
        }
        let faults = members.iter().filter_map(FaultRow::of).collect();
        let claims = APPS
            .iter()
            .flat_map(|app| {
                s.registry
                    .claims_table(app)
                    .into_iter()
                    .map(move |c| ClaimView {
                        app: app.to_string(),
                        key: String::from_utf8_lossy(&c.key).into_owned(),
                        node: hex(c.epoch.node),
                        epoch_ms: c.epoch.ms,
                    })
            })
            .collect();
        HubStatus {
            gateway: self.name.clone(),
            now_ms: now,
            draining: s.draining,
            apps: APPS.iter().map(|a| app_view(&s, a)).collect(),
            faults,
            members,
            claims,
            refusals: s.refusals.iter().cloned().collect(),
            traffic: self.traffic.view(now),
            log: self.log.read(None, None, log_limit),
        }
    }
}

fn app_view(s: &State, app: &str) -> AppView {
    let mode = s.modes.get(app).copied().unwrap_or_default();
    let quorum = s
        .sockets
        .iter()
        .filter(|((a, _), _)| a == app)
        .filter_map(|(_, x)| x.coord.as_ref())
        .map(|c| (c.paxos.max_members, c.paxos.quorum))
        .max();
    AppView {
        max_members: quorum.map(|(m, _)| m),
        quorum: quorum.map(|(_, q)| q),
        app: app.to_string(),
        mode: if mode.clustered { "clustered" } else { "dumb" },
        since_ms: mode.since_ms,
        floor: s.admission.floor(app).cloned(),
        floor_set_by: s
            .admission
            .floor_set_by(app)
            .map(|(n, h)| format!("{} on {h}", hex(n))),
        model_lock: s.admission.lock(app).map(|m| m.describe.clone()),
        members: s.registry.members(app).len(),
        ready: s.registry.members(app).iter().filter(|m| m.ready()).count(),
    }
}

fn describe(how: &Ended) -> String {
    match how {
        Ended::Closed(c) => format!("closed by the member, {c}"),
        Ended::Refused(c) => format!("closed by this gateway, {c}"),
        Ended::Dropped => "the transport dropped".into(),
        Ended::Dead => "no frame for 45 s".into(),
    }
}
