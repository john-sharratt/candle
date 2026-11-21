//! Where each request goes, and how a request for a full card waits at the
//! gateway for one with room.
//!
//! A pick counts the request as load on the member it reaches until the
//! returned load drops, in the same hold of the state, so two picks made at once
//! each see the other. While a candidate certifies, its share of live traffic
//! goes to it, and a peer request is placed away from the sender and away from
//! a candidate. A conversation turn or a draw waits at the gateway while every
//! card it may use is full, for as long as its client waits; other units give
//! up after a bounded hold. Waiting units are placed most urgent first, then
//! oldest, when a card's count falls or a capacity report arrives.

use super::*;

/// Hold, for each draw waiting at the gateway, the card it waits for: it takes
/// no new unit but a draw until the draw is placed or gone
/// (`docs/static_placement.md` §4). Derived from the queue as it stands, so a
/// reservation ends with the draw that made it; the cards' load is read once
/// for them all.
pub(super) fn reserve(s: &mut State) {
    if !s.queue.waiting().any(|ask| ask.draw.is_some()) {
        s.registry.set_reserved(BTreeSet::new());
        return;
    }
    let cards = s.registry.cards();
    let reserved: BTreeSet<CardKey> = s
        .queue
        .waiting()
        .filter_map(|ask| s.registry.draw_target(&ask.app, ask.draw?, &cards))
        .collect();
    s.registry.set_reserved(reserved);
}

impl Hub {
    /// A socket of `node`, in whichever app it has one.
    pub(super) fn handle_of(&self, node: u64) -> Option<SessionHandle> {
        self.state()
            .sockets
            .iter()
            .find(|((_, n), _)| *n == node)
            .map(|(_, s)| s.handle.clone())
    }

    /// A `BUILD_FETCH` goes to the member it names; naming none, to the next
    /// holder of the artifact but the requester, round robin — so fetches
    /// spread across every member holding it (§3.9).
    pub(super) fn route_fetch(&self, mut f: BuildFetch) {
        if f.from_node == 0 {
            let mut s = self.state();
            let holders: Vec<u64> = s
                .held
                .holders_any(&f.artifact)
                .into_iter()
                .filter(|n| *n != f.reply_to)
                .collect();
            if holders.is_empty() {
                return;
            }
            f.from_node = holders[s.fetch_rr % holders.len()];
            s.fetch_rr = s.fetch_rr.wrapping_add(1);
        }
        if let Some(h) = self.handle_of(f.from_node) {
            let _ = h.control(Frame::BuildFetch(f));
        }
    }

    /// Where a request goes (§5.2): while a candidate certifies, its share
    /// of the live traffic goes to it (§3.8). A request that is work for the
    /// member it reaches is counted against it until the returned
    /// [`Load`] drops — in the same hold of the state as the pick, so two
    /// picks at once each see the other (§5.3).
    pub fn pick(
        self: &Arc<Self>,
        app: &str,
        dispatch: &Dispatch,
        draw: Option<u64>,
        key: Option<&str>,
        pinned: Option<u64>,
    ) -> Picked {
        self.pick_from(app, dispatch, draw, key, pinned, None)
    }

    /// [`Self::pick`] for a request `sender`, a member of `app`, sent up asking
    /// for a peer ([`super::registry::Registry::pick_from`]): a `Best` request
    /// is placed away from the sender, and away from a canary while one
    /// certifies — certification samples clients' traffic, and an unproven
    /// build is not handed the members' own. Any other is placed as a client's
    /// is.
    pub fn pick_from(
        self: &Arc<Self>,
        app: &str,
        dispatch: &Dispatch,
        draw: Option<u64>,
        key: Option<&str>,
        pinned: Option<u64>,
        sender: Option<u64>,
    ) -> Picked {
        let mut s = self.state();
        self.pick_in(&mut s, app, dispatch, draw, key, pinned, sender, None)
    }

    /// [`Self::pick_from`] under a hold of the state the caller already has.
    /// An `Any` request whose `route` not every ready member declares goes to
    /// one that does, in turn, and waits while none is ready — as does a
    /// sticky one naming a key no member here holds.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pick_in(
        self: &Arc<Self>,
        s: &mut State,
        app: &str,
        dispatch: &Dispatch,
        draw: Option<u64>,
        key: Option<&str>,
        pinned: Option<u64>,
        sender: Option<u64>,
        route: Option<&RouteAt>,
    ) -> Picked {
        let State {
            registry,
            upgrades,
            sockets,
            ..
        } = s;
        // A session that has ended is still listed until its socket's cleanup
        // leaves, and no stream opens to it in between: route it nothing.
        for ((a, node), socket) in sockets.iter() {
            if a == app && socket.handle.is_gone() {
                registry.closing(app, *node, socket.session);
            }
        }
        if let Some(members) = registry.standing_changed(app) {
            tracing::info!(
                target: "cluster::gateway::placement",
                app,
                members = %members,
                "routing standing"
            );
        }
        // A sticky request naming a key nobody here holds, on a route some
        // member's table leaves out, is placed afresh among the members
        // declaring it, as an `Any` one is; once a member holds the key, the
        // key decides. A route every member with a table declares — a
        // conversation's turn — is placed as ever, its handovers with it.
        let narrowed = match (dispatch, route) {
            (Dispatch::Any, Some(at)) => registry.declaring(app, &at.method, &at.path),
            (Dispatch::Sticky(_), Some(at)) if key.is_some_and(|k| !registry.is_held(app, k)) => {
                registry.declaring_by_table(app, &at.method, &at.path)
            }
            _ => None,
        };
        if let (Some(at), Some(nodes)) = (route, narrowed) {
            let pick = registry
                .round_robin_among(app, &nodes)
                .map_or(Pick::Wait, Pick::Member);
            let declaring: Vec<String> = nodes.iter().map(|n| hex(*n)).collect();
            tracing::info!(
                target: "cluster::gateway::placement",
                app,
                route = %format!("{} {}", at.method, at.path),
                declaring = %declaring.join(" "),
                picked = ?pick,
                "routed among the members declaring the route"
            );
            return Picked { pick, load: None };
        }
        let canary = upgrades.canary(app);
        let peer_of = sender.filter(|_| matches!(dispatch, Dispatch::Best));
        let pick = match (canary, peer_of) {
            (canary, Some(sender)) => {
                let ask = PeerAsk {
                    sender: Some(sender),
                    canary,
                };
                registry.pick_from(app, dispatch, draw, key, pinned, ask)
            }
            (Some(candidate), None) => {
                registry.pick_canary(app, dispatch, draw, key, pinned, candidate, &mut || {
                    upgrades.take_share(app)
                })
            }
            (None, None) => registry.pick(app, dispatch, draw, key, pinned),
        };
        let work = matches!(
            dispatch,
            Dispatch::Best | Dispatch::Sticky(_) | Dispatch::Owner(_)
        ) && !is_errand(dispatch, key, draw);
        let load = match pick {
            Pick::Member(node) if work => registry
                .begin(app, node, draw.is_some())
                .map(|session| Load::new(Arc::clone(self), app, node, session, draw.is_some())),
            _ => None,
        };
        // **The first live conversation on a card plans the cast at once**
        // (`docs/static_placement.md` §5.3): its characters leave within a turn,
        // not on the next tick.
        if let (Pick::Member(node), true, true) = (&pick, work, app == LATENCY_APP) {
            let card = registry.member(app, *node).map(CardKey::of);
            let now = Instant::now();
            if card.is_some_and(|c| !s.planner.busy(&c, now)) {
                plan_in(s, now);
            }
        }
        Picked { pick, load }
    }

    /// [`Self::pick_from`], waiting at the gateway while every card the unit
    /// may go to is full (`docs/static_placement.md` §3.4): answered with the
    /// pick once a card has room, most urgent and then oldest first. A client's
    /// conversation turn and a client's draw wait as long as their client does
    /// ([`Placing::waits_for_its_client`]); any other unit is answered `None`
    /// once it has waited [`HOLD`].
    pub async fn place(self: &Arc<Self>, placing: Placing) -> Option<Picked> {
        let urgency = Urgency::of(&placing.app, placing.sender);
        let unbounded = placing.waits_for_its_client();
        let (seq, mut rx) = {
            let mut s = self.state();
            reserve(&mut s);
            let picked = self.pick_in(
                &mut s,
                &placing.app,
                &placing.dispatch,
                placing.draw,
                placing.key.as_deref(),
                placing.pinned,
                placing.sender,
                placing.route.as_ref(),
            );
            if picked.pick != Pick::Wait {
                return Some(picked);
            }
            let (tx, rx) = oneshot::channel();
            let app = placing.app.clone();
            let seq = s.queue.push(placing, RuntimeInstant::now(), tx);
            tracing::info!(
                target: "cluster::gateway::placement",
                app,
                urgency = urgency.name(),
                depth = s.queue.len(),
                "queued"
            );
            (seq, rx)
        };
        // A client's conversation turn, and a client's draw — a card draws one
        // at a time, so a burst waits here for the cards to come free — wait as
        // long as their client. Nothing else is held past the hold.
        if unbounded {
            return rx.await.ok();
        }
        let since = Instant::now();
        match timeout(HOLD, &mut rx).await {
            Ok(Ok(picked)) => Some(picked),
            Ok(Err(_)) => None,
            Err(_) => {
                // Placed in the moment the hold ran out: taken, not refused.
                if !self.state().queue.forget(seq) {
                    if let Ok(picked) = rx.try_recv() {
                        return Some(picked);
                    }
                }
                tracing::info!(
                    target: "cluster::gateway::placement",
                    waited_ms = since.elapsed().as_millis() as u64,
                    "refused"
                );
                None
            }
        }
    }

    /// How many units are waiting for a card with room (§4.3).
    pub fn waiting(&self) -> usize {
        self.state().queue.len()
    }

    /// Place every waiting unit that now fits (§3.4), most urgent and then
    /// oldest first; one that still fits nowhere waits on, and does not hold
    /// back those behind it that fit elsewhere. Each is taken out alone, so the
    /// rest go on holding their cards while it is placed — a draw waiting for
    /// a card keeps it from the conversations released ahead of it. Called
    /// whenever a card's count falls and whenever a `CAPACITY` arrives. A unit
    /// placed for a client gone meanwhile is uncounted as its pick drops —
    /// here, outside the state.
    pub(in super::super) fn release_waiting(self: &Arc<Self>) {
        let mut placed = Vec::new();
        {
            let mut s = self.state();
            if s.queue.len() == 0 {
                // Nothing waits, so nothing holds a card.
                s.registry.set_reserved(BTreeSet::new());
                return;
            }
            // The waiting draws' cards, held for the walk, and held again after
            // each draw it places: the draws still waiting may now want
            // another card.
            reserve(&mut s);
            for seq in s.queue.order() {
                let Some(w) = s.queue.take(seq) else {
                    continue;
                };
                // A draw that has waited its bound takes its card as it stands.
                let overdue = w.ask.draw.is_some() && w.since.elapsed() >= RESERVE_FOR;
                s.registry.set_draw_overdue(overdue);
                let picked = self.pick_in(
                    &mut s,
                    &w.ask.app,
                    &w.ask.dispatch,
                    w.ask.draw,
                    w.ask.key.as_deref(),
                    w.ask.pinned,
                    w.ask.sender,
                    w.ask.route.as_ref(),
                );
                s.registry.set_draw_overdue(false);
                if picked.pick == Pick::Wait {
                    s.queue.keep(w);
                } else {
                    if w.ask.draw.is_some() {
                        reserve(&mut s);
                    }
                    placed.push((w, picked));
                }
            }
        }
        for (w, picked) in placed {
            tracing::info!(
                target: "cluster::gateway::placement",
                req = w.ask.req,
                app = w.ask.app,
                urgency = w.urgency.name(),
                picked = ?picked.pick,
                waited_ms = w.since.elapsed().as_millis() as u64,
                "released"
            );
            let _ = w.tx.send(picked);
        }
    }

    /// Which agent of `app` runs a job (§11.3).
    pub fn pick_agent(&self, app: &str, lineage: Option<&str>, need: &Need) -> Choice {
        self.state().registry.pick_agent(app, lineage, need)
    }

    /// The programs the ready agents of `app` holding `repo` start.
    pub fn agent_programs(&self, app: &str, repo: &str) -> Vec<String> {
        self.state().registry.agent_programs(app, repo)
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
    pub fn table(&self, app: &str) -> Option<Arc<RouteTable>> {
        self.state().registry.table(app).cloned()
    }

    /// The table a request for `method` on `path` is matched in
    /// ([`super::registry::Registry::table_for`]) — shared, so the hub's lock
    /// is held for a scan and a reference count.
    pub fn table_for(&self, app: &str, method: &str, path: &str) -> Option<Arc<RouteTable>> {
        self.state().registry.table_for(app, method, path).cloned()
    }

    /// How a request for `method` on `path` to `app` is routed, decided under
    /// one hold of the hub: no member admitted; none ready, and which one
    /// stands by; or the table it is matched in.
    pub fn routing(&self, app: &str, method: &str, path: &str) -> Routing {
        let s = self.state();
        if let Some(table) = s.registry.table_for(app, method, path) {
            return Routing::Table(Arc::clone(table));
        }
        match standing_of(&s.registry, app) {
            Standing::Unadmitted => Routing::Unadmitted,
            Standing::By(node) => Routing::StandingBy(node),
        }
    }

    /// The members of `app` a request may go to now, in node order, each with
    /// the address its socket came from: what a forwarder of datagrams
    /// spreads its flows over (`docs/battlecities_app.md` §5).
    pub fn serving(&self, app: &str) -> Vec<(u64, IpAddr)> {
        let s = self.state();
        let mut serving: Vec<(u64, IpAddr)> = s
            .registry
            .members(app)
            .iter()
            .filter(|m| m.routable() && !m.silent)
            .filter_map(|m| {
                let socket = s.sockets.get(&(app.to_string(), m.node))?;
                Some((m.node, socket.addr))
            })
            .collect();
        serving.sort_unstable_by_key(|(node, _)| *node);
        serving
    }
}
