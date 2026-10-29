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

use std::collections::{BTreeMap, BTreeSet, HashMap};

use applied_set::AppliedSet;

use super::claims::Claims;
use crate::dispatch::RouteTable;
use crate::frame::{Capacity, ClaimInfo, MemberInfo};
use handover::Handover;

pub use cards::{CardKey, CardLoad, Cards, Fences, HOSTING_APP, LATENCY_APP};
pub use member::{MemberView, PeerAsk, SYNC_SLACK};

#[cfg(test)]
use super::agents::{Choice, Need};
#[cfg(test)]
use super::placement::{
    finish_us, step_with_us, token_ns_with, TurnSize, NOMINAL_PROMPT_TOKENS, NOMINAL_REPLY_TOKENS,
};
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

#[derive(Debug, Default)]
pub struct Registry {
    apps: HashMap<String, AppMembers>,
    /// The plan's fences (`docs/cooperative_admission.md` §4.4).
    fences: Fences,
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
        // A release every placeable member now holds places like no release
        // at all — `Best` among them — so it is forgotten here rather than
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

    /// A request handed to `node`, a draw when `draw`: one more in flight
    /// there. Returns the member's session, which [`Self::end`] names, or
    /// `None` for a node not admitted.
    pub fn begin(&mut self, app: &str, node: u64, draw: bool) -> Option<u64> {
        let m = self.app(app).members.get_mut(&node)?;
        m.in_flight += 1;
        m.placed += 1;
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
}

#[cfg(test)]
mod card_tests;
#[cfg(test)]
mod tests;
