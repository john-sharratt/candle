//! Which member a request goes to (§5.2): its dispatch names how — every
//! member, round robin, a key's holder, `Best` — and the eligible pool is
//! handed to [`crate::gateway::placement`] to decide among, with what is
//! placed on each card and what each machine takes. When every eligible card
//! is full the request waits (`docs/static_placement.md` §3.4).

use super::{CardKey, Cards, MemberView, PeerAsk, Pick, Registry};
use crate::dispatch::{reaches, Dispatch};
use crate::gateway::agents::{choose_with_standby, programs, Choice, Need};
use crate::gateway::placement::{
    draw_target, guest_reach, has_room, place, within_reach, Placed, Weigh,
};

impl Registry {
    /// The members a request may go to: ready, and not lagging.
    pub(super) fn ready(&self, app: &str) -> Vec<&MemberView> {
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
            return speaking(routable);
        }
        // Nobody in sync is serving — the one that was has left: the members
        // still syncing are the most anyone has, and serve rather than leave
        // the app answering nothing.
        speaking(
            a.members
                .values()
                .filter(|m| m.ready() && !m.lagging && !m.closing && Some(m.node) != except)
                .collect(),
        )
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

    /// What a placement of `app`'s unit weighs now, over `cards`.
    fn weigh<'a>(&'a self, app: &str, cards: &'a Cards) -> Weigh<'a> {
        Weigh {
            cards,
            machines: &self.machines,
            reserved: &self.reserved,
            cursor: self.apps.get(app).map_or(&self.no_cursor, |a| &a.cursor),
            draw_overdue: self.draw_overdue,
        }
    }

    /// The card a draw of `area` pixels for `app` waits for, when it would
    /// wait for one (`docs/static_placement.md` §4): what the hub holds back
    /// from new units while that draw is queued. `cards` is every card's load,
    /// read once for every draw the hub asks about.
    pub fn draw_target(&self, app: &str, area: u64, cards: &Cards) -> Option<CardKey> {
        draw_target(
            self.placeable_but(app, Some(area), None),
            &self.weigh(app, cards),
        )
    }

    /// The ready member the static placement gives the unit to ([`place`]);
    /// [`Placed::Full`] when no eligible card has room.
    fn best(&self, app: &str, draw: Option<u64>, weigh: &Weigh<'_>) -> Placed {
        self.best_but(app, draw, None, weigh)
    }

    fn best_but(
        &self,
        app: &str,
        draw: Option<u64>,
        except: Option<u64>,
        weigh: &Weigh<'_>,
    ) -> Placed {
        place(app, self.placeable_but(app, draw, except), draw, weigh)
    }

    /// [`Self::best`] for a member asking for a peer ([`Self::pick_from`]):
    /// neither `sender` nor `canary` while another qualifies; then not the
    /// canary; then anyone.
    fn best_for_peer(
        &self,
        app: &str,
        draw: Option<u64>,
        ask: PeerAsk,
        weigh: &Weigh<'_>,
    ) -> Placed {
        let pool = self.placeable_but(app, draw, None);
        let without = |out: &[Option<u64>]| -> Vec<&MemberView> {
            pool.iter()
                .copied()
                .filter(|m| !out.contains(&Some(m.node)))
                .collect()
        };
        let peers = without(&[ask.sender, ask.canary]);
        if !peers.is_empty() {
            return place(app, peers, draw, weigh);
        }
        let proven = without(&[ask.canary]);
        if !proven.is_empty() {
            return place(app, proven, draw, weigh);
        }
        place(app, pool, draw, weigh)
    }

    /// The members a request may be placed on, but `except`: for a draw of
    /// `draw` pixels, those whose card reaches it ([`within_reach`]).
    fn placeable_but(&self, app: &str, draw: Option<u64>, except: Option<u64>) -> Vec<&MemberView> {
        let placeable = self.apps.get(app).map_or_else(Vec::new, |a| {
            a.members
                .values()
                .filter(|m| m.placeable() && Some(m.node) != except)
                .collect()
        });
        within_reach(speaking(placeable), draw)
    }

    /// Where a request for sticky `key`, which no member here holds, goes
    /// (`docs/cooperative_admission.md` §2.2): placed among the members that
    /// hold the records its last holder wrote before letting it go — never one
    /// that would wait for them. When none does — the last holder has left
    /// before anyone applied its last turn — among them all, which is the most
    /// anyone holds (§2.4). A handover every candidate already holds is
    /// forgotten.
    fn best_for(
        &mut self,
        app: &str,
        draw: Option<u64>,
        key: Option<&str>,
        cards: &Cards,
    ) -> Placed {
        let handover = key.and_then(|k| self.apps.get(app)?.handovers.get(k.as_bytes()).cloned());
        let Some(h) = handover else {
            return self.best(app, draw, &self.weigh(app, cards));
        };
        let (pick, everyone_holds) = {
            let weigh = &self.weigh(app, cards);
            let ready = self.placeable_but(app, draw, None);
            let holding: Vec<&MemberView> =
                ready.iter().copied().filter(|m| h.held_by(m)).collect();
            let everyone_holds = holding.len() == ready.len();
            let pick = if holding.is_empty() {
                place(app, ready, draw, weigh)
            } else {
                place(app, holding, draw, weigh)
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
        self.round_robin_among(app, &nodes)
    }

    /// The next of `nodes` in `app`'s turn; `None` when there are none.
    pub fn round_robin_among(&mut self, app: &str, nodes: &[u64]) -> Option<u64> {
        if nodes.is_empty() {
            return None;
        }
        let a = self.app(app);
        let pick = nodes[a.rr % nodes.len()];
        a.rr = a.rr.wrapping_add(1);
        Some(pick)
    }

    /// The ready members of `app` whose own table declares a route answering
    /// `method` on `path`, when not every ready member does: a member that
    /// lacks a route — a build without it yet, or a member that serves it only
    /// while it holds what the route needs — is never sent it. `None` when
    /// every ready member declares it.
    pub fn declaring(&self, app: &str, method: &str, path: &str) -> Option<Vec<u64>> {
        let ready = self.ready_but(app, None);
        let declaring: Vec<u64> = ready
            .iter()
            .filter(|m| {
                m.routes
                    .as_ref()
                    .is_some_and(|t| t.lookup(method, path).is_some())
            })
            .map(|m| m.node)
            .collect();
        (declaring.len() < ready.len()).then_some(declaring)
    }

    /// Which agent of `app` runs a job of `lineage` needing `need` (§11.3): the
    /// lineage's holder while it has a free slot, and after `waited` the best
    /// idle agent; a standby agent only when no other can run it. The agent
    /// chosen is counted as handed the job until its next `CAPACITY`.
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
        let choice = choose_with_standby(&self.ready(app), &self.standby, holder, need, waited);
        if let Choice::Run(node) = choice {
            if let Some(m) = self.app(app).members.get_mut(&node) {
                m.handed += 1;
            }
        }
        choice
    }

    /// The programs the ready agents of `app` holding `repo` start.
    pub fn agent_programs(&self, app: &str, repo: &str) -> Vec<String> {
        programs(&self.ready(app), repo)
    }

    /// Where a request with `dispatch` goes. `draw` is the pixel area it draws,
    /// on a route declared [`crate::dispatch::Reach::Draw`]; `key` its sticky or
    /// owner key; `pinned` the member its page came from (`tokera_ui_<app>`).
    ///
    /// A draw is placed among the members whose card reaches it. A key's
    /// holder still serves it whatever its reach: the key is the holder's, and
    /// its own refusal names what it draws.
    pub fn pick(
        &mut self,
        app: &str,
        dispatch: &Dispatch,
        draw: Option<u64>,
        key: Option<&str>,
        pinned: Option<u64>,
    ) -> Pick {
        self.pick_from(app, dispatch, draw, key, pinned, PeerAsk::default())
    }

    /// [`Self::pick`] for a request a member of `app` sent up asking for a
    /// peer ([`PeerAsk`]): placed `Best`, it goes to a member that is neither
    /// the sender nor the canary when one can take it; else back to the sender
    /// rather than to an unproven canary; and to the canary only when it is
    /// all there is (`docs/guest_draw_orchestration.md` §3). A member asks a
    /// peer for what its own engine is too busy to do — a draw's prompt
    /// authorization while it draws.
    pub fn pick_from(
        &mut self,
        app: &str,
        dispatch: &Dispatch,
        draw: Option<u64>,
        key: Option<&str>,
        pinned: Option<u64>,
        ask: PeerAsk,
    ) -> Pick {
        if !self.clustered(app) {
            return Pick::Configured;
        }
        let cards = self.cards();
        let ready_here = |r: &Registry, node: u64| {
            r.member(app, node)
                .is_some_and(|m| m.routable() && !m.silent)
        };
        let placed = match dispatch {
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
            Dispatch::Best if ask == PeerAsk::default() => {
                self.best(app, draw, &self.weigh(app, &cards))
            }
            Dispatch::Best => self.best_for_peer(app, draw, ask, &self.weigh(app, &cards)),
            Dispatch::Any => self.round_robin(app).map_or(Placed::Nobody, Placed::To),
            // A request routed by its subject that names none and draws
            // nothing — a listing, a read of the caller's own — is no unit of
            // work: it goes in turn, never held back for a card's room.
            Dispatch::Owner(_) if is_errand(dispatch, key, draw) => {
                self.round_robin(app).map_or(Placed::Nobody, Placed::To)
            }
            // A key held by a member here goes to it — syncing, or still
            // loading, included: the conversation is the holder's, possibly
            // with turns it wrote cut off that no other member has, and a
            // holder still loading answers its loading page until it serves.
            // It is never held back for room (rule 1 before rule 3). One held
            // elsewhere goes `Best`, and the member reached names the holder
            // (§5.4).
            Dispatch::Sticky(_) | Dispatch::Owner(_) => {
                let holder = key.and_then(|k| {
                    self.apps
                        .get(app)
                        .and_then(|a| a.claims.holder(k.as_bytes()))
                });
                match holder {
                    Some(h) if self.member(app, h).is_some() => {
                        routed(app, dispatch, h, "holder");
                        Placed::To(h)
                    }
                    _ => self.best_for(app, draw, key, &cards),
                }
            }
            Dispatch::Ui => {
                if let Some(p) = pinned.filter(|p| ready_here(self, *p)) {
                    return Pick::Member(p);
                }
                return self.round_robin(app).map_or(Pick::Configured, Pick::Pinned);
            }
        };
        match placed {
            Placed::To(node) => Pick::Member(node),
            Placed::Full => Pick::Wait,
            Placed::Nobody => Pick::Configured,
        }
    }

    /// Where a request goes while `canary` certifies (§3.8). A `Best` or
    /// `Any` request, or a sticky one whose key no member here holds, is the
    /// candidate's when `take` says this one is its share, and otherwise goes
    /// to the baseline — the candidate left out, unless it is the only member
    /// ready. Every other request is picked as ever. A draw the candidate's
    /// card does not reach is never its share.
    #[allow(clippy::too_many_arguments)]
    pub fn pick_canary(
        &mut self,
        app: &str,
        dispatch: &Dispatch,
        draw: Option<u64>,
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
            .and_then(|k| self.apps.get(app)?.handovers.get(k.as_bytes()).cloned())
            .is_none_or(|h| self.member(app, canary).is_some_and(|m| h.held_by(m)));
        let canary_reaches = draw.is_none_or(|area| {
            self.member(app, canary)
                .is_some_and(|m| reaches(guest_reach(m), area))
        });
        let speaks = self.member(app, canary).is_some_and(|m| !m.silent);
        if !shared || !canary_holds || !canary_reaches || !routable(self, canary) || !speaks {
            return self.pick(app, dispatch, draw, key, pinned);
        }
        let cards = self.cards();
        // The candidate's share is placed like any unit: only while its card
        // has room (rule 3).
        let canary_room = matches!(dispatch, Dispatch::Any)
            || self
                .member(app, canary)
                .is_some_and(|m| has_room(m, &self.weigh(app, &cards), draw.is_some()));
        if canary_room && take() {
            routed(app, dispatch, canary, "canary");
            return Pick::Member(canary);
        }
        let baseline = match dispatch {
            Dispatch::Any => self
                .round_robin_but(app, Some(canary))
                .map_or(Placed::Nobody, Placed::To),
            _ => self.best_but(app, draw, Some(canary), &self.weigh(app, &cards)),
        };
        match baseline {
            Placed::To(node) => Pick::Member(node),
            Placed::Full => Pick::Wait,
            Placed::Nobody if canary_room => Pick::Member(canary),
            Placed::Nobody => Pick::Wait,
        }
    }
}

/// Of `members`, those not silent (§4.4) — or all of them when every one is:
/// a member that stopped reporting is passed over while another can take the
/// request, and never leaves the app answering nothing on its own judgement.
fn speaking(members: Vec<&MemberView>) -> Vec<&MemberView> {
    if members.iter().all(|m| m.silent) {
        return members;
    }
    members.into_iter().filter(|m| !m.silent).collect()
}

/// Whether a request placed by `dispatch` with `key` and `draw` is an errand
/// rather than a unit of work: routed by its subject, it names none and
/// draws nothing, so no member's card is the place for it and it waits for
/// no room ([`Registry::pick_from`]), and it is not counted against the
/// member that answers it.
pub fn is_errand(dispatch: &Dispatch, key: Option<&str>, draw: Option<u64>) -> bool {
    matches!(dispatch, Dispatch::Owner(_)) && key.is_none() && draw.is_none()
}

/// A pick made by something other than the placement policy — a key's
/// holder, a canary's share — named for the placement log
/// (`docs/cooperative_admission.md` §4.6). `Best` and draws log their own.
fn routed(app: &str, dispatch: &Dispatch, node: u64, by: &str) {
    tracing::info!(
        target: "cluster::gateway::placement",
        app,
        dispatch = %dispatch.name(),
        member = %format!("{node:016x}"),
        by,
        "routed"
    );
}
