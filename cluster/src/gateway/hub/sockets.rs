//! A member socket's life in the gateway, from its handshake to its close.
//!
//! `decide` admits a `HELLO` only once the node's previous admission is
//! complete, so two dials of one node cannot unseat each other. `serve` runs one
//! socket, and undoes a half-made replacement if its task ends first. `run`
//! owns the session from admission to close. Frames from a session that has
//! been replaced are dropped, since a late claim or capacity report would
//! describe a process that is gone. Coordination decisions go to the cluster
//! log, and `close_all` closes every member socket with `1001` on shutdown so
//! the members carry on through their other clusters.

use super::*;

impl Hub {
    pub fn new(key: GroupKey, name: impl Into<String>) -> Arc<Hub> {
        Arc::new(Hub {
            key,
            name: name.into(),
            state: Mutex::new(State::default()),
            tiebreak: Mutex::new(None),
            log: CoordLog::new(),
            traffic: Traffic::new(),
        })
    }

    /// Vote as the tie-break acceptor (§5.9). `peers` is how many other
    /// gateways are configured; zero means this gateway is the only one.
    /// `store` is its vote journal.
    pub fn enable_tiebreak(
        &self,
        node: u64,
        peers: usize,
        store: Box<dyn crate::consensus::votes::VoteStore>,
    ) -> std::io::Result<()> {
        let opened = TieBreak::open(node, peers, store, WallMillis::now())?;
        *self.tiebreak.lock().unwrap_or_else(|e| e.into_inner()) = Some(opened);
        Ok(())
    }

    /// A peered gateway's node id, from its `PEER_HELLO`. Remembered after the
    /// link drops, so the lowest id stays the only voter.
    pub fn note_peer(&self, node: u64) {
        if let Some(tb) = self
            .tiebreak
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
        {
            tb.note_peer(node);
        }
    }

    pub(in super::super) fn state(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// A coordination decision, kept in the log the cluster page and the audit
    /// read, and written to the gateway's own log beside its placement lines —
    /// a member going lagging, syncing or silent changes where requests go, so
    /// the two are read together.
    pub(in super::super) fn note(&self, app: &str, subject: String, text: String) {
        tracing::info!(
            target: "cluster::gateway::coord",
            app,
            subject = %subject,
            "{text}"
        );
        self.log.push(Entry {
            at_ms: now_ms(),
            gateway: self.name.clone(),
            app: app.to_string(),
            subject,
            text,
        });
    }

    /// A coordination decision made beside the hub — a peer gateway linked or
    /// refused, a certificate order's step (§13.4) — for the log.
    pub fn record(&self, app: &str, subject: String, text: String) {
        self.note(app, subject, text);
    }

    /// This gateway's name in the log.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub(super) fn refused(&self, app: &str, ip: IpAddr, who: Option<(u64, String)>, close: &Close) {
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
    pub(in super::super) fn decide(
        &self,
        app: &str,
        hello: &Hello,
    ) -> Result<(Admit, Admitted), Close> {
        if hello.app != app {
            return Err(Close::new(
                CloseCode::Malformed,
                format!("a HELLO for {} on the {app} socket", hello.app),
            ));
        }
        let mut s = self.state();
        let key = (app.to_string(), hello.node);
        // **An admission of the node already decided completes first.** Its
        // handshake is past its proof and bounded by `ADMIT_SEND`, so it is
        // either taken or undone in moments; a newer `HELLO` meanwhile is
        // refused to dial again rather than taking its place, so two
        // overlapping dials of one node cannot unseat each other in turn and
        // neither ever land. One standing past `PENDING_OVERDUE` breaks the
        // invariant its guard keeps, and is reported, not taken over.
        if let Some(p) = s.pending.get(&key) {
            if p.since.elapsed() >= PENDING_OVERDUE {
                tracing::error!(
                    app,
                    node = %hex(hello.node),
                    session = p.session,
                    "an admission has been pending past any handshake: its task \
                     neither ran nor undid it"
                );
            }
            return Err(Close::new(
                CloseCode::Retry,
                format!(
                    "an admission of node {} is completing: dial again",
                    hex(hello.node)
                ),
            ));
        }
        let locked_before = s.admission.lock(app).is_some();
        let State {
            admission, held, ..
        } = &mut *s;
        let admitted = admission.decide(hello, now_ms(), |b| held.holds(b))?;
        let lock = (!locked_before)
            .then(|| s.admission.lock(app).map(|m| m.describe.clone()))
            .flatten();
        let (_, members) = s.registry.listing(app);
        // **The replacement starts here, under the guard the table is read
        // under** (`docs/replicated_substrate.md` §5.4): the node's standing
        // session, if any, is marked replaced, so nothing it still reports
        // lands after this table was built — it is held back instead, in case
        // the handshake fails (`undo_pending`) — and every claim relay and
        // plan for the node from now on waits for the new socket in
        // `pending`.
        let claims = s.registry.claims(app).table();
        if let Some(old) = s.sockets.get(&key) {
            old.replaced.store(true, Ordering::SeqCst);
        }
        s.pending.insert(
            key,
            Pending {
                session: admitted.session,
                since: Instant::now(),
                queue: Vec::new(),
                held: Held::default(),
                overflowed: false,
            },
        );
        drop(s);
        if let Some(model) = lock {
            self.note(app, hex(hello.node), format!("model lock set: {model}"));
        }
        Ok((
            Admit {
                session: admitted.session,
                members,
                claims,
                // No floor yet — only local builds have joined — is floor 0.
                floor_count: admitted.floor.as_ref().map_or(0, |f| f.count),
                floor_hash: admitted
                    .floor
                    .as_ref()
                    .map_or(String::new(), |f| f.hash.clone()),
                // A new session has sent no table on this gateway yet.
                routes_wanted: true,
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
        // Armed from the moment the admission is decided: should the handshake
        // fail, or this task be dropped or panic before `run` takes the
        // admission's pending, the replacement it began is undone, and the
        // node's session stays as it was.
        let handshake = admit(
            &mut tx,
            &mut rx,
            &self.key,
            |h| seen = Some((h.node, h.host.clone())),
            |h| {
                let (admit, admitted) = self.decide(app, h)?;
                decided = Some(PendingGuard {
                    hub: Arc::clone(self),
                    app: app.to_string(),
                    node: h.node,
                    admitted,
                    armed: true,
                });
                Ok(admit)
            },
        )
        .await;
        let hello = match handshake {
            Ok((hello, _)) => hello,
            Err(e) => {
                drop(decided);
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
        let guard = decided.expect("an ADMIT was decided");
        self.run(app, ip, hello, guard, tx, rx, streams).await;
    }

    /// The admitted session's life, on a socket that came from `ip`. `guard`
    /// holds its admission until its pending is taken.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn run<T, R, F>(
        self: &Arc<Self>,
        app: &str,
        ip: IpAddr,
        hello: Hello,
        mut guard: PendingGuard,
        tx: T,
        rx: R,
        streams: F,
    ) where
        T: PipeTx,
        R: PipeRx,
        F: Fn(MemberStream) + Send + Sync,
    {
        let node = hello.node;
        let admitted = guard.admitted.clone();
        let session = if admitted.behind {
            start_behind(tx, rx)
        } else {
            start(Role::Gateway, tx, rx)
        };
        let handle = session.handle.clone();
        // Its `CAPACITY` does not cross protocols: it stands as upgrading from
        // here to the end of its session.
        let capacity = admitted.behind.then(|| Capacity {
            state: MemberState::Upgrading,
            ..Capacity::default()
        });
        let replaced_flag = Arc::new(AtomicBool::new(false));
        {
            let mut s = self.state();
            let connected_ms = now_ms();
            // What was meant for the node since its ADMIT's table was built —
            // unless too much arrived meanwhile: then this session goes,
            // having changed nothing, and its admission is undone. Disarmed
            // first: taken either way it is no longer the guard's to undo, and
            // a pending not its own fails this session alone, not — through a
            // guard dropping mid-panic — the whole gateway.
            guard.armed = false;
            let taken = take_pending(&mut s, app, node, &admitted);
            let waiting = match taken {
                Taken::Ready(waiting) => waiting,
                Taken::Overflowed(applied) => {
                    drop(s);
                    let why = "more changed during the handshake than it could be brought up \
                               to date with: dial again";
                    // Not 4409: no session of the node displaced this socket
                    // for good, and the member dials again.
                    handle.close(Close::new(CloseCode::Retry, why));
                    self.note(
                        app,
                        hex(node),
                        format!("session {} not taken: {why}", admitted.session),
                    );
                    self.follow(app, node, applied);
                    return;
                }
            };
            let replaced = s.sockets.insert(
                (app.to_string(), node),
                Socket {
                    session: admitted.session,
                    handle: handle.clone(),
                    addr: ip,
                    replaced: Arc::clone(&replaced_flag),
                    connected_ms,
                    clock_offset_ms: hello.now_ms as i64 - connected_ms as i64,
                    coord: None,
                    pacer: ReceiptPacer::new(Instant::now()),
                    capacity_at: Instant::now(),
                    skipped: 0,
                    dropped: 0,
                    pulses_dropped: 0,
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
            // Joining a group that already has a member serving: it serves
            // nothing until it has caught up with it, whatever it reports.
            let syncing = s.registry.has_routable(app, node);
            s.registry.join(MemberView {
                node,
                host: hello.host.clone(),
                card: hello.card,
                app: app.to_string(),
                build: hello.build.clone(),
                weight: hello.power,
                session: admitted.session,
                repos: hello.repos.clone(),
                capacity,
                routes: None,
                high_water: 0,
                applied: Vec::new(),
                lagging: false,
                syncing,
                silent: false,
                in_flight: 0,
                drawing: 0,
                handed: 0,
                closing: false,
            });
            // Sent under the guard that put the socket in place, before any
            // frame relayed after it.
            for frame in waiting {
                let _ = handle.control(frame);
            }
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
        self.on_admitted(app, node, &hello, &admitted, &handle);

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
                        if let Err(close) =
                            self.on_frame(app, node, &replaced_flag, frame, &handle)
                        {
                            handle.close(close.clone());
                            break Ended::Refused(close);
                        }
                    }
                    Some(Incoming::Stream(conn)) => streams(MemberStream {
                        app: app.to_string(),
                        node,
                        target: conn.target().to_string(),
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
                    self.plan_tick();
                }
                _ = receipts.tick() => {
                    self.receipt_tick(app, node);
                    self.silence_tick(app, node);
                }
            }
        };

        let (left, gone, lost) = {
            let mut s = self.state();
            let key = (app.to_string(), node);
            // A replaced session leaving takes nothing of the node's with it.
            let mut gone = Vec::new();
            let mut lost = Vec::new();
            if s.sockets
                .get(&key)
                .is_some_and(|x| x.session == admitted.session)
            {
                s.sockets.remove(&key);
                if !s.sockets.keys().any(|(_, n)| *n == node) {
                    s.held.forget(node);
                }
                gone = s.upgrades.left(app, node);
                lost = s.open.left(app, node);
            }
            let locked = s.admission.lock(app).is_some();
            s.admission.left(app, node, admitted.session);
            let cleared = locked && s.admission.lock(app).is_none();
            s.last_close.insert(key, describe(&how));
            (
                (s.registry.leave(app, node, admitted.session), cleared),
                gone,
                lost,
            )
        };
        self.act(gone);
        // What it was answering will not be answered: its requesters ask again
        // now, rather than waiting out a silence.
        for (requester, request) in lost {
            self.to_member(
                app,
                requester,
                Frame::Lost(Lost {
                    request,
                    reply_to: requester,
                }),
            );
        }
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

    /// A frame from `node` on a session `replaced` says whether a newer one has
    /// replaced: kept, relayed, or a protocol violation. A vote is also
    /// answered on `reply_on`, the socket it arrived on.
    pub(in super::super) fn on_frame(
        self: &Arc<Self>,
        app: &str,
        node: u64,
        replaced: &AtomicBool,
        frame: Frame,
        reply_on: &SessionHandle,
    ) -> Result<(), Close> {
        // **A replaced session reports nothing of its node's state.** Its
        // frames still queued here are from a socket the node has reconnected
        // past: a process that still runs says its state again on its new
        // session, and one that restarted holds nothing it said. Kept, a late
        // `CLAIM` would name the node holder of a key its new process never
        // took, and a late `CAPACITY` would describe a process that is gone.
        // A report is applied under the guard the flag is set under
        // ([`Self::decide`]), so it either lands before the new session's
        // table was built or not at all; while the replacement can still be
        // undone it is held back rather than dropped ([`hold_back`]).
        // Checked and decoded on arrival, held back or not: a report held is
        // one that will apply.
        let frame = match check_report(frame, node)? {
            Arrived::Other(frame) => frame,
            Arrived::Report(report) => {
                let done = {
                    let mut s = self.state();
                    if replaced.load(Ordering::SeqCst) {
                        hold_back(&mut s, app, node, replaced, report);
                        return Ok(());
                    }
                    apply_report(&mut s, app, node, report)
                };
                self.follow(app, node, [done]);
                return Ok(());
            }
        };
        if let Some((key, text)) = paxos_step(node, &frame) {
            self.note(app, key, text);
        }
        match frame {
            Frame::BuildFetch(f) => self.route_fetch(f),
            Frame::BuildChunk(c) => {
                if let Some(h) = self.handle_of(c.reply_to) {
                    let _ = h.catchup(Frame::BuildChunk(c));
                }
            }
            f @ (Frame::Prepare(_) | Frame::Propose(_) | Frame::Release(_)) => {
                self.tiebreak(app, &f, reply_on);
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
            f @ (Frame::Members(_) | Frame::Receipt(_) | Frame::Verdict(_) | Frame::Lost(_)) => {
                return Err(Close::new(
                    CloseCode::Malformed,
                    format!("{} is the gateway's to send", name(f.kind())),
                ));
            }
            f @ (Frame::Catchup(_)
            | Frame::Headers(_)
            | Frame::Fetch(_)
            | Frame::Done(_)
            | Frame::Have(_)) => self.relay_substrate(app, node, f),
            f @ Frame::Pulse(_) => self.relay_pulse(app, node, f),
            _ => {}
        }
        Ok(())
    }

    /// Answer a vote on the socket it arrived on, when this gateway is the
    /// tie-break voter (§5.9). A journal error is logged and the frame is
    /// still relayed: the members' own votes are the round.
    pub(super) fn tiebreak(&self, app: &str, frame: &Frame, reply_on: &SessionHandle) {
        let reply = {
            let mut guard = self.tiebreak.lock().unwrap_or_else(|e| e.into_inner());
            let Some(tb) = guard.as_mut() else {
                return;
            };
            let (now, wall) = (tb.now(), WallMillis::now());
            match tb.consider(app, frame, now, wall) {
                Ok(reply) => reply,
                Err(e) => {
                    tracing::error!(app, "tie-break vote failed: {e}");
                    None
                }
            }
        };
        if let Some(reply) = reply {
            let _ = reply_on.control(reply);
        }
    }

    /// Every other admitted member, of every app: build traffic reaches the
    /// agents too, which take their binary from a model member's build
    /// (§3.9, §4.3).
    pub(super) fn to_every_other(&self, from: u64, frame: Frame) {
        let s = self.state();
        let mut sent = Vec::new();
        for ((_, n), socket) in &s.sockets {
            if *n != from && !sent.contains(n) {
                sent.push(*n);
                let _ = socket.handle.control(frame.clone());
            }
        }
    }

    /// Do what follows from state reports of `node`'s applied under the guard
    /// ([`apply_report`]), in order: their log lines, their relays, a changed
    /// listing, the upgrade actions they set off, and — after a `CAPACITY`,
    /// which says what a card can take — placing the units waiting for room
    /// (§4.3).
    pub(in super::super) fn follow(
        self: &Arc<Self>,
        app: &str,
        node: u64,
        applied: impl IntoIterator<Item = Applied>,
    ) {
        for done in applied {
            for (subject, text) in done.notes {
                self.note(app, subject, text);
            }
            for frame in done.to_others {
                self.to_others(app, node, frame);
            }
            for frame in done.to_every_other {
                self.to_every_other(node, frame);
            }
            if done.members_changed {
                self.members_changed(app);
            }
            self.act(done.actions);
            if done.capacity {
                self.release_waiting();
            }
        }
    }

    /// `frame` to every member of `app` but `from`. A claim relay is also
    /// kept for each session admitted whose socket is not yet in place
    /// ([`Pending`]): it brings that session's `ADMIT` table up to date.
    pub(in super::super) fn to_others(&self, app: &str, from: u64, frame: Frame) {
        let mut s = self.state();
        let State {
            sockets, pending, ..
        } = &mut *s;
        for ((a, n), socket) in sockets.iter() {
            if a == app && *n != from {
                let _ = socket.handle.control(frame.clone());
            }
        }
        if matches!(frame, Frame::Claim(_) | Frame::Unclaim(_)) {
            for ((a, n), p) in pending.iter_mut() {
                if a == app && *n != from {
                    p.relay(frame.clone());
                }
            }
        }
    }

    /// `frame` to `node` of `app`, on whichever socket is its current one
    /// now. A request or reply for a session still in its handshake is not
    /// kept for it: its process never issued what it answers, and a request
    /// to a replaced process fails over by its requester's own retry.
    pub(in super::super) fn to_member(&self, app: &str, node: u64, frame: Frame) {
        if let Some(socket) = self.state().sockets.get(&(app.to_string(), node)) {
            let _ = socket.handle.control(frame);
        }
    }

    /// The app's membership changed: every member is told, and a mode switch
    /// is logged (§5.7).
    pub(super) fn members_changed(&self, app: &str) {
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
}
