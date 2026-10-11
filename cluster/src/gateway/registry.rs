//! A gateway's view of each app's members (§5): who is admitted, their
//! capacity and dispatch table, the claims they announced — and from that,
//! which member a request goes to.
//!
//! An app is **clustered** on this gateway while it has a ready member, and
//! **dumb** otherwise: its routes then go to their configured upstream (§5.7).

use std::cmp::Reverse;
use std::collections::{BTreeMap, HashMap};

use super::agents::{choose, Choice, Need};
use super::claims::Claims;
use crate::dispatch::{Dispatch, RouteTable};
use crate::frame::{BuildId, Capacity, ClaimInfo, MemberInfo, MemberState, RepoAd};

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
    /// Its ordered lane passed its cap: it is sent no live event and routed
    /// no request until it has drained and caught up (§4.4).
    pub lagging: bool,
}

impl MemberView {
    pub fn state(&self) -> MemberState {
        self.capacity
            .as_ref()
            .map_or(MemberState::Loading, |c| c.state)
    }

    pub fn ready(&self) -> bool {
        self.state() == MemberState::Ready
    }

    /// Ready, and not lagging: a request may be routed to it.
    pub fn routable(&self) -> bool {
        self.ready() && !self.lagging
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
        changed
    }

    pub fn routes(&mut self, app: &str, node: u64, routes: RouteTable) {
        if let Some(m) = self.app(app).members.get_mut(&node) {
            m.routes = Some(routes);
        }
    }

    /// `node`'s own highest `oseq`, from its `ACK`.
    pub fn high_water(&mut self, app: &str, node: u64, high_water: u64) {
        if let Some(m) = self.app(app).members.get_mut(&node) {
            m.high_water = m.high_water.max(high_water);
        }
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
        self.apps
            .get(app)
            .map(|a| a.members.values().filter(|m| m.routable()).collect())
            .unwrap_or_default()
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

    /// The highest-power idle ready member; with none idle, the one with the
    /// most free slots (§5.3).
    fn best(&self, app: &str) -> Option<u64> {
        let ready = self.ready(app);
        let cap = |m: &&MemberView| m.capacity.clone().expect("a ready member sent capacity");
        ready
            .iter()
            .filter(|m| cap(m).idle)
            .max_by_key(|m| (cap(m).power, Reverse(m.node)))
            .or_else(|| {
                ready
                    .iter()
                    .max_by_key(|m| (cap(m).slots_free, Reverse(m.node)))
            })
            .map(|m| m.node)
    }

    fn round_robin(&mut self, app: &str) -> Option<u64> {
        let nodes: Vec<u64> = self.ready(app).iter().map(|m| m.node).collect();
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
    /// idle agent.
    pub fn pick_agent(
        &self,
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
        choose(&self.ready(app), holder, need, waited)
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
            Dispatch::Aggregate(_) => {
                return Pick::All(self.ready(app).iter().map(|m| m.node).collect());
            }
            Dispatch::Best => self.best(app),
            Dispatch::Any => self.round_robin(app),
            // A key held by a member here goes to it. One held elsewhere goes
            // `Best`, and the member reached names the holder (§5.4).
            Dispatch::Sticky(_) | Dispatch::Owner(_) => {
                let holder = key.and_then(|k| {
                    self.apps
                        .get(app)
                        .and_then(|a| a.claims.holder(k.as_bytes()))
                });
                match holder {
                    Some(h) if ready_here(self, h) => Some(h),
                    _ => self.best(app),
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
}

#[cfg(test)]
mod tests;
