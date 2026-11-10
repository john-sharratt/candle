//! A gateway's view of each app's members (§5): who is admitted, their
//! capacity and dispatch table, the claims they announced. Which of them a
//! request goes to is [`pick`]'s, by the policy in [`super::placement`].
//!
//! An app is **clustered** on this gateway while it has a ready member, and
//! **dumb** otherwise: its routes then go to their configured upstream (§5.7).

mod cards;
mod handover;
mod member;
mod pick;
mod standing;

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use applied_set::AppliedSet;

use super::agents::named_standby;
use super::claims::Claims;
use crate::dispatch::RouteTable;
use crate::frame::{Capacity, ClaimInfo, MemberInfo};
use crate::machines::MachineTable;
use handover::Handover;

pub use cards::{CardKey, CardLoad, Cards, HOSTING_APP, LATENCY_APP, MODEL_APPS};
pub use member::{Held, MemberView, PeerAsk, SYNC_SLACK};
pub use pick::is_errand;

#[cfg(test)]
use super::agents::{Choice, Need};
#[cfg(test)]
use crate::dispatch::Dispatch;
#[cfg(test)]
use crate::frame::{BuildId, MemberState, RepoAd};

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
    /// Every card the request may go to is full (`docs/cooperative_admission.md`
    /// §4.2): it waits at the gateway until one has room (§4.3).
    Wait,
    /// No machine the request may go to takes its work at all
    /// (`docs/static_placement.md` §3.2, §4): there is no room to wait for, and
    /// the gateway refuses it.
    Unplaced,
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
    /// The machine this app's round robin last gave a unit to
    /// (`docs/static_placement.md` §3.2), as `(rank, node)` — the app's own, so
    /// another app's picks never move it.
    cursor: Cell<Option<(u32, u64)>>,
    /// The members' standing last logged ([`Registry::standing_changed`]).
    standing: String,
}

#[derive(Debug, Default)]
pub struct Registry {
    apps: HashMap<String, AppMembers>,
    /// What each machine takes (`docs/static_placement.md` §2). A machine it
    /// does not name is given no work.
    machines: MachineTable,
    /// The cards held for a draw waiting at the gateway for their turns to end
    /// (`docs/static_placement.md` §4), as the hub last derived them from its
    /// queue.
    reserved: BTreeSet<CardKey>,
    /// The cursor of an app with no members yet: nothing is placed by it.
    no_cursor: Cell<Option<(u32, u64)>>,
    /// The draw the hub is placing has waited its bound for a card's turns to
    /// end, and takes its card as it stands (`docs/static_placement.md` §4).
    draw_overdue: bool,
    /// The hosts whose agents stand by (§11.3): given a job only when no
    /// other agent can run it.
    standby: BTreeSet<String>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Name the hosts whose agents stand by.
    pub fn set_standby(&mut self, hosts: BTreeSet<String>) {
        self.standby = hosts;
    }

    /// The machine table placement and the plan read from now on.
    pub fn set_machines(&mut self, machines: MachineTable) {
        self.machines = machines;
    }

    pub fn machines(&self) -> &MachineTable {
        &self.machines
    }

    /// Hold `cards` for the draws waiting for them: they take no new unit but a
    /// draw until the hub says otherwise.
    pub fn set_reserved(&mut self, cards: BTreeSet<CardKey>) {
        self.reserved = cards;
    }

    /// Whether the draw placed next has waited its bound, and takes its card as
    /// it stands.
    pub fn set_draw_overdue(&mut self, overdue: bool) {
        self.draw_overdue = overdue;
    }

    /// Whether the agents on `host` stand by.
    pub fn stands_by(&self, host: &str) -> bool {
        named_standby(&self.standby, host)
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
    ///
    /// **Its claims stand** — only a holder releases one (§5.4) — **but each
    /// key it held is placed as released** while it is away: its next request
    /// goes among the members that hold the node's records through the highest
    /// of its own the gateway heard (`docs/cooperative_admission.md` §2.4), not
    /// to whichever member places first, which may not have applied its last
    /// turn. A hint for this gateway's placement only: nothing is relayed and no
    /// claim changes. The node is each key's current holder, so the hint
    /// replaces any earlier release of it — but one that is this same release,
    /// by this node through these records, which keeps the members known to
    /// hold it complete. A real release replaces it in turn.
    pub fn leave(&mut self, app: &str, node: u64, session: u64) -> bool {
        let a = self.app(app);
        if a.members.get(&node).is_some_and(|m| m.session == session) {
            a.members.remove(&node);
            a.epoch += 1;
            let through = a.seen.get(&node).copied().unwrap_or(0);
            for key in a.claims.keys_of(node) {
                let same = a
                    .handovers
                    .get(&key)
                    .is_some_and(|h| h.by == node && h.through == through);
                if !same {
                    a.handovers.insert(
                        key,
                        Handover {
                            by: node,
                            through,
                            complete: BTreeSet::new(),
                        },
                    );
                }
            }
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
        changed
    }

    pub fn routes(&mut self, app: &str, node: u64, routes: RouteTable) {
        if let Some(m) = self.app(app).members.get_mut(&node) {
            m.routes = Some(Arc::new(routes));
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
        // A release every placeable member now holds places like no release
        // at all — like any unit among them — so it is forgotten here rather than
        // waiting for its key's next request, which may never come: a
        // conversation not resumed, a picture never edited. What stands is
        // only what is still replicating.
        let placeable: Vec<&MemberView> = a.members.values().filter(|m| m.placeable()).collect();
        if !placeable.is_empty() {
            a.handovers
                .retain(|_, h| !placeable.iter().all(|m| h.held_by(m)));
        }
    }

    /// `key`'s claim was let go by `by`, its own records then reaching
    /// `through` (§5.4): the key's next request goes to a member that holds
    /// them (`docs/cooperative_admission.md` §2.2).
    pub fn released(&mut self, app: &str, key: &[u8], by: u64, through: u64) {
        self.app(app).handovers.insert(
            key.to_vec(),
            Handover {
                by,
                through,
                complete: BTreeSet::new(),
            },
        );
    }

    /// Whether a release of `key` stands that `node` is known to hold
    /// complete (`COMPLETE`).
    pub fn holds_release(&self, app: &str, key: &[u8], node: u64) -> bool {
        self.apps
            .get(app)
            .and_then(|a| a.handovers.get(key))
            .is_some_and(|h| h.complete.contains(&node))
    }

    /// `node` holds each of `keys` complete (`COMPLETE`): every record of the
    /// conversation it names through its release, and every turn of it whole.
    /// A key with no release standing is ignored — it is held, or every member
    /// already holds it.
    pub fn completed(&mut self, app: &str, node: u64, keys: &[Vec<u8>]) {
        let a = self.app(app);
        for key in keys {
            if let Some(h) = a.handovers.get_mut(key) {
                h.complete.insert(node);
            }
        }
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
    ///
    /// **Nor a record its own origin lacks.** An `ACK` is gap-free, so one hole
    /// pins it below everything held past the hole; when the origin itself
    /// misses that record, nobody can ever fill it, and WhiteWorld's zend sat
    /// syncing for good on 2026-11-01 holding all but one of BlackWorld's 74
    /// million records — the one BlackWorld lacked too. So an origin is also
    /// covered when the member holds up to what is served (`held`, from its
    /// `COORD`) and misses no more of it than the origin does of itself
    /// (`origin_missing`).
    pub fn in_sync(
        node: u64,
        acks: &[(u64, u64)],
        serving: &BTreeMap<u64, u64>,
        held: &BTreeMap<u64, Held>,
        origin_missing: &BTreeMap<u64, u64>,
    ) -> bool {
        serving.iter().all(|(origin, top)| {
            let reaches = |at: u64| at.saturating_add(SYNC_SLACK) >= *top;
            *origin == node
                || acks
                    .iter()
                    .any(|(o, through)| o == origin && reaches(*through))
                || held.get(origin).is_some_and(|h| {
                    reaches(h.top) && h.missing <= origin_missing.get(origin).copied().unwrap_or(0)
                })
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

    /// Mark `node` silent or not; whether that changed.
    pub fn set_silent(&mut self, app: &str, node: u64, silent: bool) -> bool {
        let Some(m) = self.app(app).members.get_mut(&node) else {
            return false;
        };
        let changed = m.silent != silent;
        m.silent = silent;
        changed
    }

    /// A request handed to `node`, a draw when `draw`: one more in flight
    /// there. Returns the member's session, which [`Self::end`] names, or
    /// `None` for a node not admitted.
    pub fn begin(&mut self, app: &str, node: u64, draw: bool) -> Option<u64> {
        let m = self.app(app).members.get_mut(&node)?;
        m.in_flight += 1;
        if draw {
            m.drawing += 1;
        }
        Some(m.session)
    }

    /// A request [`Self::begin`] counted against `node`'s `session` is
    /// answered. A session since replaced counted its own requests, so the
    /// newer one's count is left alone.
    pub fn end(&mut self, app: &str, node: u64, session: u64, draw: bool) {
        if let Some(m) = self.app(app).members.get_mut(&node) {
            if m.session == session {
                m.in_flight = m.in_flight.saturating_sub(1);
                if draw {
                    m.drawing = m.drawing.saturating_sub(1);
                }
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

    /// Whether any member of `app` is admitted, ready or not.
    pub fn admits(&self, app: &str) -> bool {
        self.apps.get(app).is_some_and(|a| !a.members.is_empty())
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
    pub fn table(&self, app: &str) -> Option<&Arc<RouteTable>> {
        self.apps
            .get(app)?
            .members
            .values()
            .filter(|m| m.ready())
            .find_map(|m| m.routes.as_ref())
    }

    /// The table a request for `method` on `path` is matched in: a ready
    /// member's that declares a route answering it — tables differ while a
    /// build that adds a route rolls out, and where a member declares a route
    /// only while it holds what the route needs — else [`Self::table`].
    pub fn table_for(&self, app: &str, method: &str, path: &str) -> Option<&Arc<RouteTable>> {
        self.apps
            .get(app)?
            .members
            .values()
            .filter(|m| m.ready())
            .filter_map(|m| m.routes.as_ref())
            .find(|t| t.answers(method, path))
            .or_else(|| self.table(app))
    }
}

#[cfg(test)]
mod card_tests;
#[cfg(test)]
mod tests;
