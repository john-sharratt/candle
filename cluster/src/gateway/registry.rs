//! A gateway's view of each app's members (§5): who is admitted, their
//! capacity and dispatch table, the claims they announced — and from that,
//! which member a request goes to.
//!
//! An app is **clustered** on this gateway while it has a ready member, and
//! **dumb** otherwise: its routes then go to their configured upstream (§5.7).

use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, HashMap};

use applied_set::AppliedSet;

use super::agents::{choose, Choice, Need};
use super::claims::Claims;
use crate::dispatch::{Dispatch, RouteTable};
use crate::frame::{BuildId, Capacity, ClaimInfo, MemberInfo, MemberState, RepoAd, PLACEMENT_WIDE};

/// One admitted member.
#[derive(Debug, Clone)]
pub struct MemberView {
    pub node: u64,
    pub host: String,
    pub app: String,
    pub build: BuildId,
    /// Its power at admission, fixed for the session (§9.5).
    pub weight: u32,
    pub session: u64,
    /// The repositories its `HELLO` advertised (§10.1, §11.2).
    pub repos: Vec<RepoAd>,
    /// Its last `CAPACITY`; a member is loading until it sends one.
    pub capacity: Option<Capacity>,
    pub routes: Option<RouteTable>,
    /// Its own highest `oseq`, from its `ACK`s (§7.1).
    pub high_water: u64,
    /// Its last `ACK`: through which `oseq` it has applied each origin.
    pub applied: Vec<(u64, u64)>,
    /// Its ordered lane passed its cap: it is sent no live event and routed
    /// no request until it has drained and caught up (§4.4).
    pub lagging: bool,
    /// Admitted while the group already had a member serving: routed no
    /// request until its own `ACK`s cover what every other origin has written
    /// — the gateway's judgement, not the member's. Unlike lagging, it is sent
    /// live events throughout, so it catches up rather than chasing them.
    pub syncing: bool,
    /// Requests this gateway has handed it and it has not finished answering
    /// (§5.3): counted at the pick, so a burst sees its own earlier picks
    /// before the member's next `CAPACITY` reports them.
    pub in_flight: u32,
    /// Jobs handed to it, an agent, since its last `CAPACITY` (§11.3): a job
    /// is started and then read back, so it holds no request open to count,
    /// and the next `CAPACITY` reports it running. Until then the choice
    /// counts it, so a burst of jobs is spread instead of all reading the
    /// same free slot.
    pub handed: u32,
    /// Requests handed to it since its last `CAPACITY`: prompts its reported
    /// backlog does not hold yet. A request handed earlier is either queued,
    /// and in that backlog, or past its prefill and decoding, and in its
    /// decode step — so `finish_us` charges a nominal prompt for these alone.
    pub placed: u32,
    /// Its session has ended and the socket's cleanup has not yet run its
    /// leave: no stream can open to it, so it is routed nothing. A request
    /// picked for it in that window was answered `503` while another member
    /// stood ready.
    pub closing: bool,
}

/// How many of an origin's latest writes a member may still lack and count
/// as in sync: events in flight when its `ACK` was sent.
pub const SYNC_SLACK: u64 = 64;

impl MemberView {
    pub fn state(&self) -> MemberState {
        self.capacity
            .as_ref()
            .map_or(MemberState::Loading, |c| c.state)
    }

    pub fn ready(&self) -> bool {
        self.state() == MemberState::Ready
    }

    /// Whether it holds `origin`'s records through `through`: its own, or as
    /// its last `ACK` reported them applied.
    pub fn holds_through(&self, origin: u64, through: u64) -> bool {
        origin == self.node
            || self
                .applied
                .iter()
                .any(|(o, t)| *o == origin && *t >= through)
    }

    /// Ready, not lagging, not still syncing and not closing: a request may be
    /// routed to it.
    pub fn routable(&self) -> bool {
        self.ready() && !self.lagging && !self.syncing && !self.closing
    }

    pub fn info(&self) -> MemberInfo {
        MemberInfo {
            node: self.node,
            host: self.host.clone(),
            app: self.app.clone(),
            build: self.build.clone(),
            weight: self.weight,
            state: self.state(),
            high_water: self.high_water,
        }
    }
}

/// Where a request goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pick {
    /// The app has no ready member here: the route's configured upstream.
    Configured,
    Member(u64),
    /// A UI file from a newly pinned member: set its cookie.
    Pinned(u64),
    /// Every ready member, for an aggregate.
    All(Vec<u64>),
}

#[derive(Debug, Default)]
struct AppMembers {
    members: BTreeMap<u64, MemberView>,
    claims: Claims,
    rr: usize,
    /// Bumped on every change, for `MEMBERS`.
    epoch: u64,
    /// The highest `oseq` each origin was heard at, kept when its member
    /// leaves: what a catch-up is owed while nobody is here to answer it.
    seen: BTreeMap<u64, u64>,
    /// The last release of each sticky key: who let it go, and through which
    /// of its own records. Dropped once every member a request may go to holds
    /// them.
    handovers: HashMap<Vec<u8>, Handover>,
}

/// A sticky key let go ([`Registry::released`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Handover {
    by: u64,
    through: u64,
}

#[derive(Debug, Default)]
pub struct Registry {
    apps: HashMap<String, AppMembers>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    fn app(&mut self, app: &str) -> &mut AppMembers {
        self.apps.entry(app.to_string()).or_default()
    }

    pub fn join(&mut self, member: MemberView) {
        let a = self.app(&member.app.clone());
        a.members.insert(member.node, member);
        a.epoch += 1;
    }

    /// `node`'s `session` ended; a newer session of the node stays.
    pub fn leave(&mut self, app: &str, node: u64, session: u64) -> bool {
        let a = self.app(app);
        if a.members.get(&node).is_some_and(|m| m.session == session) {
            a.members.remove(&node);
            a.epoch += 1;
            return true;
        }
        false
    }

    /// `node`'s session `session` has ended: route it nothing until its leave.
    /// A newer session of the same node is left alone.
    pub fn closing(&mut self, app: &str, node: u64, session: u64) {
        if let Some(m) = self
            .apps
            .get_mut(app)
            .and_then(|a| a.members.get_mut(&node))
            .filter(|m| m.session == session)
        {
            m.closing = true;
        }
    }

    /// `node`'s `CAPACITY`; whether its state changed, which `MEMBERS` reports.
    pub fn capacity(&mut self, app: &str, node: u64, capacity: Capacity) -> bool {
        let a = self.app(app);
        let Some(m) = a.members.get_mut(&node) else {
            return false;
        };
        let changed = m.capacity.as_ref().map(|c| c.state) != Some(capacity.state);
        if changed {
            a.epoch += 1;
        }
        m.capacity = Some(capacity);
        // What it was handed is in what it reports now.
        m.handed = 0;
        m.placed = 0;
        changed
    }

    pub fn routes(&mut self, app: &str, node: u64, routes: RouteTable) {
        if let Some(m) = self.app(app).members.get_mut(&node) {
            m.routes = Some(routes);
        }
    }

    /// `node`'s `ACK`: what it has applied of each origin, its own highest
    /// `oseq` among them.
    pub fn applied(&mut self, app: &str, node: u64, applied: Vec<(u64, u64)>) {
        let a = self.app(app);
        if let Some(own) = applied.iter().find(|(o, _)| *o == node) {
            let seen = a.seen.entry(node).or_default();
            *seen = (*seen).max(own.1);
            if let Some(m) = a.members.get_mut(&node) {
                m.high_water = m.high_water.max(own.1);
            }
        }
        if let Some(m) = a.members.get_mut(&node) {
            m.applied = applied;
        }
    }

    /// `key`'s claim was let go by `by`, its own records then reaching
    /// `through` (§5.4): the key's next request goes to a member that holds
    /// them (`docs/cooperative_admission.md` §2.2).
    pub fn released(&mut self, app: &str, key: &[u8], by: u64, through: u64) {
        self.app(app)
            .handovers
            .insert(key.to_vec(), Handover { by, through });
    }

    /// Whether a member of `app` other than `except` is serving now: what a
    /// member joining beside it waits for being in sync with.
    pub fn has_routable(&self, app: &str, except: u64) -> bool {
        self.apps
            .get(app)
            .is_some_and(|a| a.members.values().any(|m| m.node != except && m.routable()))
    }

    /// Whether `acks` — a member's applied-through per origin — cover what the
    /// members serving now hold (`serving`, the most any of them has applied of
    /// each origin), within [`SYNC_SLACK`]. Not every write ever seen: one a
    /// member away wrote cut off is held by nobody serving, and a newcomer
    /// waiting on it would wait as long as that member is gone, while exactly
    /// as complete as the members answering.
    pub fn in_sync(node: u64, acks: &[(u64, u64)], serving: &BTreeMap<u64, u64>) -> bool {
        serving.iter().all(|(origin, top)| {
            *origin == node
                || acks
                    .iter()
                    .any(|(o, through)| o == origin && through.saturating_add(SYNC_SLACK) >= *top)
        })
    }

    /// Mark `node` syncing or not; whether that changed.
    pub fn set_syncing(&mut self, app: &str, node: u64, syncing: bool) -> bool {
        let a = self.app(app);
        let Some(m) = a.members.get_mut(&node) else {
            return false;
        };
        let changed = m.syncing != syncing;
        m.syncing = syncing;
        if changed {
            a.epoch += 1;
        }
        changed
    }

    /// Whether `since` lacks writes this gateway has seen an origin other than
    /// `requester` make — records a member now away still holds. With nobody
    /// to answer, such a catch-up is not done: it waits for an answerer.
    pub fn owed(&mut self, app: &str, requester: u64, since: &AppliedSet) -> bool {
        self.app(app)
            .seen
            .iter()
            .any(|(origin, top)| *origin != requester && since.through(*origin) < *top)
    }

    /// Mark `node` lagging or not; whether that changed.
    pub fn set_lagging(&mut self, app: &str, node: u64, lagging: bool) -> bool {
        let a = self.app(app);
        let Some(m) = a.members.get_mut(&node) else {
            return false;
        };
        let changed = m.lagging != lagging;
        m.lagging = lagging;
        changed
    }

    /// A request handed to `node`: one more in flight there. Returns the
    /// member's session, which [`Self::end`] names, or `None` for a node not
    /// admitted.
    pub fn begin(&mut self, app: &str, node: u64) -> Option<u64> {
        let m = self.app(app).members.get_mut(&node)?;
        m.in_flight += 1;
        m.placed += 1;
        Some(m.session)
    }

    /// A request [`Self::begin`] counted against `node`'s `session` is
    /// answered. A session since replaced counted its own requests, so the
    /// newer one's count is left alone.
    pub fn end(&mut self, app: &str, node: u64, session: u64) {
        if let Some(m) = self.app(app).members.get_mut(&node) {
            if m.session == session {
                m.in_flight = m.in_flight.saturating_sub(1);
            }
        }
    }

    pub fn claims(&mut self, app: &str) -> &mut Claims {
        &mut self.app(app).claims
    }

    /// The app's claims, as `ADMIT` and the status page list them.
    pub fn claims_table(&self, app: &str) -> Vec<ClaimInfo> {
        self.apps
            .get(app)
            .map(|a| a.claims.table())
            .unwrap_or_default()
    }

    pub fn member(&self, app: &str, node: u64) -> Option<&MemberView> {
        self.apps.get(app)?.members.get(&node)
    }

    pub fn members(&self, app: &str) -> Vec<&MemberView> {
        self.apps
            .get(app)
            .map(|a| a.members.values().collect())
            .unwrap_or_default()
    }

    /// `MEMBERS`: the app's epoch and every admitted member.
    pub fn listing(&self, app: &str) -> (u64, Vec<MemberInfo>) {
        self.apps.get(app).map_or((0, Vec::new()), |a| {
            (a.epoch, a.members.values().map(MemberView::info).collect())
        })
    }

    /// The member of the app that joined this gateway first — the lowest
    /// admission session — among those admitted and not ready: loading,
    /// catching up, upgrading. While no member is ready, every request for the
    /// app goes to it (§5.7), so its loading page and its own endpoints — the
    /// performance and agents pages read them — stand in for the app.
    pub fn standing_by(&self, app: &str) -> Option<u64> {
        self.apps
            .get(app)?
            .members
            .values()
            .filter(|m| !m.ready())
            .min_by_key(|m| m.session)
            .map(|m| m.node)
    }

    /// Whether the app is clustered here: it has a ready member.
    pub fn clustered(&self, app: &str) -> bool {
        self.apps
            .get(app)
            .is_some_and(|a| a.members.values().any(MemberView::ready))
    }

    /// The dispatch table the app's ready members declared.
    pub fn table(&self, app: &str) -> Option<&RouteTable> {
        self.apps
            .get(app)?
            .members
            .values()
            .filter(|m| m.ready())
            .find_map(|m| m.routes.as_ref())
    }

    /// The members a request may go to: ready, and not lagging.
    fn ready(&self, app: &str) -> Vec<&MemberView> {
        self.ready_but(app, None)
    }

    /// The members a request may go to, but `except`.
    fn ready_but(&self, app: &str, except: Option<u64>) -> Vec<&MemberView> {
        let Some(a) = self.apps.get(app) else {
            return Vec::new();
        };
        let routable: Vec<&MemberView> = a
            .members
            .values()
            .filter(|m| m.routable() && Some(m.node) != except)
            .collect();
        if !routable.is_empty() {
            return routable;
        }
        // Nobody in sync is serving — the one that was has left: the members
        // still syncing are the most anyone has, and serve rather than leave
        // the app answering nothing.
        a.members
            .values()
            .filter(|m| m.ready() && !m.lagging && !m.closing && Some(m.node) != except)
            .collect()
    }

    /// The next routable member other than `except`, round robin: the
    /// answerer of a `CATCHUP` that names none (§7.2).
    pub fn answerer(&mut self, app: &str, except: u64) -> Option<u64> {
        let nodes: Vec<u64> = self
            .ready(app)
            .iter()
            .map(|m| m.node)
            .filter(|n| *n != except)
            .collect();
        if nodes.is_empty() {
            return None;
        }
        let a = self.app(app);
        let pick = nodes[a.rr % nodes.len()];
        a.rr = a.rr.wrapping_add(1);
        Some(pick)
    }

    /// The ready member with the most power to spare per request (§5.3):
    /// `power / (1 + in_flight)`, among those with room — idle, or with more
    /// free slots than it has requests in flight — or among them all when none
    /// has. Of equal shares, an idle member; then the lowest node.
    fn best(&self, app: &str) -> Option<u64> {
        self.best_but(app, None)
    }

    fn best_but(&self, app: &str, except: Option<u64>) -> Option<u64> {
        best_of(self.ready_but(app, except))
    }

    /// Where a request for sticky `key`, which no member here holds, goes
    /// (`docs/cooperative_admission.md` §2.2): `Best` among the members that
    /// hold the records its last holder wrote before letting it go — never one
    /// that would wait for them. When none does — the last holder has left
    /// before anyone applied its last turn — among them all, which is the most
    /// anyone holds (§2.4). A handover every candidate already holds is
    /// forgotten.
    fn best_for(&mut self, app: &str, key: Option<&str>) -> Option<u64> {
        let handover = key.and_then(|k| self.apps.get(app)?.handovers.get(k.as_bytes()).copied());
        let Some(h) = handover else {
            return self.best(app);
        };
        let (pick, everyone_holds) = {
            let ready = self.ready(app);
            let holding: Vec<&MemberView> = ready
                .iter()
                .copied()
                .filter(|m| m.holds_through(h.by, h.through))
                .collect();
            let everyone_holds = holding.len() == ready.len();
            let pick = if holding.is_empty() {
                best_of(ready)
            } else {
                best_of(holding)
            };
            (pick, everyone_holds)
        };
        if everyone_holds {
            if let Some(k) = key {
                self.app(app).handovers.remove(k.as_bytes());
            }
        }
        pick
    }

    fn round_robin(&mut self, app: &str) -> Option<u64> {
        self.round_robin_but(app, None)
    }

    fn round_robin_but(&mut self, app: &str, except: Option<u64>) -> Option<u64> {
        let nodes: Vec<u64> = self.ready_but(app, except).iter().map(|m| m.node).collect();
        if nodes.is_empty() {
            return None;
        }
        let a = self.app(app);
        let pick = nodes[a.rr % nodes.len()];
        a.rr = a.rr.wrapping_add(1);
        Some(pick)
    }

    /// Which agent of `app` runs a job of `lineage` needing `need` (§11.3): the
    /// lineage's holder while it has a free slot, and after `waited` the best
    /// idle agent. The agent chosen is counted as handed the job until its
    /// next `CAPACITY`.
    pub fn pick_agent(
        &mut self,
        app: &str,
        lineage: Option<&str>,
        need: &Need,
        waited: bool,
    ) -> Choice {
        let holder = lineage.and_then(|k| {
            self.apps
                .get(app)
                .and_then(|a| a.claims.holder(k.as_bytes()))
        });
        let choice = choose(&self.ready(app), holder, need, waited);
        if let Choice::Run(node) = choice {
            if let Some(m) = self.app(app).members.get_mut(&node) {
                m.handed += 1;
            }
        }
        choice
    }

    /// Where a request with `dispatch` goes. `key` is its sticky or owner key;
    /// `pinned` the member its page came from (`tokera_ui_<app>`).
    pub fn pick(
        &mut self,
        app: &str,
        dispatch: &Dispatch,
        key: Option<&str>,
        pinned: Option<u64>,
    ) -> Pick {
        if !self.clustered(app) {
            return Pick::Configured;
        }
        let ready_here =
            |r: &Registry, node: u64| r.member(app, node).is_some_and(MemberView::routable);
        let node = match dispatch {
            // Every member loaded, a syncing one included: each answers from
            // what it holds, and one rejoining holds what it wrote cut off
            // that no other member has yet.
            Dispatch::Aggregate(_) => {
                return Pick::All(
                    self.apps
                        .get(app)
                        .map(|a| {
                            a.members
                                .values()
                                .filter(|m| m.ready())
                                .map(|m| m.node)
                                .collect()
                        })
                        .unwrap_or_default(),
                );
            }
            Dispatch::Best => self.best(app),
            Dispatch::Any => self.round_robin(app),
            // A key held by a member here goes to it — syncing, or still
            // loading, included: the conversation is the holder's, possibly
            // with turns it wrote cut off that no other member has, and a
            // holder still loading answers its loading page until it serves.
            // One held elsewhere goes `Best`, and the member reached names the
            // holder (§5.4).
            Dispatch::Sticky(_) | Dispatch::Owner(_) => {
                let holder = key.and_then(|k| {
                    self.apps
                        .get(app)
                        .and_then(|a| a.claims.holder(k.as_bytes()))
                });
                match holder {
                    Some(h) if self.member(app, h).is_some() => Some(h),
                    _ => self.best_for(app, key),
                }
            }
            Dispatch::Ui => {
                if let Some(p) = pinned.filter(|p| ready_here(self, *p)) {
                    return Pick::Member(p);
                }
                return self.round_robin(app).map_or(Pick::Configured, Pick::Pinned);
            }
        };
        node.map_or(Pick::Configured, Pick::Member)
    }

    /// Where a request goes while `canary` certifies (§3.8). A `Best` or
    /// `Any` request, or a sticky one whose key no member here holds, is the
    /// candidate's when `take` says this one is its share, and otherwise goes
    /// to the baseline — the candidate left out, unless it is the only member
    /// ready. Every other request is picked as ever.
    pub fn pick_canary(
        &mut self,
        app: &str,
        dispatch: &Dispatch,
        key: Option<&str>,
        pinned: Option<u64>,
        canary: u64,
        take: &mut dyn FnMut() -> bool,
    ) -> Pick {
        if !self.clustered(app) {
            return Pick::Configured;
        }
        let routable =
            |r: &Registry, node: u64| r.member(app, node).is_some_and(MemberView::routable);
        let shared = match dispatch {
            Dispatch::Best | Dispatch::Any => true,
            Dispatch::Sticky(_) => key
                .and_then(|k| {
                    self.apps
                        .get(app)
                        .and_then(|a| a.claims.holder(k.as_bytes()))
                })
                .is_none_or(|h| self.member(app, h).is_none()),
            _ => false,
        };
        // A key let go is the canary's to take only once it holds what the
        // last holder wrote, as for any member.
        let canary_holds = key
            .and_then(|k| self.apps.get(app)?.handovers.get(k.as_bytes()).copied())
            .is_none_or(|h| {
                self.member(app, canary)
                    .is_some_and(|m| m.holds_through(h.by, h.through))
            });
        if !shared || !canary_holds || !routable(self, canary) {
            return self.pick(app, dispatch, key, pinned);
        }
        if take() {
            return Pick::Member(canary);
        }
        let baseline = match dispatch {
            Dispatch::Any => self.round_robin_but(app, Some(canary)),
            _ => self.best_but(app, Some(canary)),
        };
        Pick::Member(baseline.unwrap_or(canary))
    }
}

/// The prefill a turn brings, in tokens, as the gateway reckons it: a
/// request carries only its new message, and the prompt a member prefills is
/// what its projection makes of the conversation — a figure the gateway does
/// not see. The same for every member, so it weighs each one's backlog and
/// prefill rate alike.
pub const NOMINAL_PROMPT_TOKENS: u64 = 2048;

/// The reply a turn decodes, in tokens, as the gateway reckons it.
pub const NOMINAL_REPLY_TOKENS: u64 = 512;

/// The best of `ready` (§5.3): among those with room, or among them all when
/// none has — the soonest finish when every one of them has said what a turn
/// costs it ([`finish_us`]), otherwise the largest share by [`rank`].
fn best_of(ready: Vec<&MemberView>) -> Option<u64> {
    // Every candidate as the pick saw it, for the placement line below.
    let seen: Vec<String> = ready.iter().map(|m| placement_view(m)).collect();
    let roomy: Vec<&MemberView> = ready.iter().copied().filter(|m| has_room(m)).collect();
    let pool = if roomy.is_empty() { ready } else { roomy };
    let finishes: Option<Vec<(u64, &MemberView)>> =
        pool.iter().map(|m| finish_us(m).map(|f| (f, *m))).collect();
    let (by, pick) = match finishes {
        Some(f) if !f.is_empty() => (
            "finish",
            f.into_iter()
                .min_by(|(fa, a), (fb, b)| fa.cmp(fb).then(rank(b, a)))
                .map(|(_, m)| m.node),
        ),
        _ => (
            "power",
            pool.into_iter().max_by(|a, b| rank(a, b)).map(|m| m.node),
        ),
    };
    tracing::info!(
        target: "cluster::gateway::placement",
        picked = %pick.map_or_else(|| "none".to_string(), |n| format!("{n:016x}")),
        by,
        candidates = %seen.join("; "),
        "placed"
    );
    pick
}

/// One candidate as a placement weighed it: its node, whether it had room,
/// its requests in flight, the figures it advertised, and the finish they give.
fn placement_view(m: &MemberView) -> String {
    let c = capacity_of(m);
    format!(
        "{:016x} room={} in_flight={} placed={} backlog={} prefill_tok_s={} step_next_us={} step_wide_us={} finish_ms={}",
        m.node,
        has_room(m),
        m.in_flight,
        m.placed,
        c.prefill_backlog,
        c.prefill_tok_s,
        c.decode_step_next_us,
        c.decode_step_wide_us,
        finish_us(m).map_or_else(|| "-".to_string(), |f| (f / 1000).to_string()),
    )
}

/// When a turn placed on `m` now would finish, in microseconds from now, by
/// what it says a turn costs it (`docs/cooperative_admission.md` §2.1, §3
/// "earliest advertised finish"): the prefill already queued there, the
/// turns handed to it and not yet in its queue, and this one, at its
/// prefill rate; then the reply at the decode step those turns and this one
/// give it ([`step_with_us`]). `None` until it has learned both rates — a
/// member is never placed by figures it has not measured.
fn finish_us(m: &MemberView) -> Option<u64> {
    let c = capacity_of(m);
    if c.prefill_tok_s == 0 || c.decode_step_next_us == 0 {
        return None;
    }
    // A request handed since the report and already answered brought no
    // prompt that is still waiting: only those still in flight can be.
    let pending = m.placed.min(m.in_flight);
    let tokens = u64::from(c.prefill_backlog) + (u64::from(pending) + 1) * NOMINAL_PROMPT_TOKENS;
    let prefill_us = tokens * 1_000_000 / u64::from(c.prefill_tok_s);
    Some(prefill_us + NOMINAL_REPLY_TOKENS * step_with_us(c, pending + 1))
}

/// The decode step with `extra` rows more than `c` reports decoding, in
/// microseconds: its one-more-row step at 1 and its wide step at
/// [`PLACEMENT_WIDE`], on the line through them, extended past the wide
/// point. A wide step it has not learned, or one no slower, is flat.
fn step_with_us(c: &Capacity, extra: u32) -> u64 {
    let next = u64::from(c.decode_step_next_us);
    let wide = u64::from(c.decode_step_wide_us);
    if wide <= next {
        return next;
    }
    let beyond = u64::from(extra.saturating_sub(1));
    next + (wide - next) * beyond / u64::from(PLACEMENT_WIDE - 1)
}

fn capacity_of(m: &MemberView) -> &Capacity {
    m.capacity.as_ref().expect("a ready member sent capacity")
}

/// Whether `m` can take another request: idle, or with a free slot its
/// requests in flight have not already spoken for.
fn has_room(m: &MemberView) -> bool {
    let c = capacity_of(m);
    c.idle || c.slots_free > m.in_flight
}

/// `a` against `b` for [`Registry::best`]: the larger share of power per
/// request — `power / (1 + in_flight)`, compared cross-multiplied so no share
/// rounds — then idle over busy, then the lower node.
fn rank(a: &MemberView, b: &MemberView) -> Ordering {
    let (ca, cb) = (capacity_of(a), capacity_of(b));
    let share_a = u64::from(ca.power) * (1 + u64::from(b.in_flight));
    let share_b = u64::from(cb.power) * (1 + u64::from(a.in_flight));
    share_a
        .cmp(&share_b)
        .then(ca.idle.cmp(&cb.idle))
        .then(Reverse(a.node).cmp(&Reverse(b.node)))
}

#[cfg(test)]
mod tests;
