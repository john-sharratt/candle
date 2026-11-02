//! Cards (`docs/cooperative_admission.md` §4.2): the members of one process,
//! across every app, whose work the gateway counts together.

use std::collections::{BTreeMap, HashMap};

use super::{MemberView, Registry};
use crate::gateway::planner::{CardView, Host};

/// The app whose requests are live conversations for the plan
/// (`docs/static_placement.md` §5.1): a chat, a passthrough or MCP
/// conversation.
pub const LATENCY_APP: &str = "zend";

/// The app whose members host characters, which the plan lends (§5).
pub const HOSTING_APP: &str = "npcd";

/// The apps whose members run the model, whose units the machine table places
/// (`docs/static_placement.md` §3.2).
pub const MODEL_APPS: [&str; 2] = [LATENCY_APP, HOSTING_APP];

/// A card, by what its members' `HELLO`s named.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum CardKey {
    /// The process's card id.
    Card(u64),
    /// A member that names no card is a card of its own.
    Alone { app: String, node: u64 },
}

impl CardKey {
    pub fn of(m: &MemberView) -> CardKey {
        match m.card {
            0 => CardKey::Alone {
                app: m.app.clone(),
                node: m.node,
            },
            card => CardKey::Card(card),
        }
    }
}

/// What the gateway has placed on one card, summed over its members.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CardLoad {
    /// Requests handed to its members and not yet answered.
    pub in_flight: u32,
    /// Draws among `in_flight`.
    pub drawing: u32,
    /// Of `in_flight`, those of [`LATENCY_APP`].
    pub latency: u32,
}

impl CardLoad {
    /// The units of work on the card that a placement counts against its
    /// machine's `max` (`docs/static_placement.md` §3.1): every app's requests
    /// in flight there, but its draws, which hold the card in their own way.
    pub fn work(&self) -> u32 {
        self.in_flight.saturating_sub(self.drawing)
    }
}

/// Every card's load, by key.
pub type Cards = HashMap<CardKey, CardLoad>;

impl Registry {
    /// Every card as the plan reads it (`docs/static_placement.md` §5): its
    /// zend work — requests in flight, and conversations its zend member holds
    /// the claim of, a worker's delegated task among them — and its npcd
    /// members, each with its machine's share of the cast, zero for one that
    /// takes no lease, and eligible to host only while it may be routed to and
    /// nothing in its process still loads (rule 1, [`MemberView::may_host`]).
    ///
    /// **A delegated task makes its card busy as a live conversation does.** A
    /// worker's turns run inside its zend member and never pass the gateway, so
    /// no request of theirs is in flight here; but each holds its
    /// conversation's claim while it runs (`docs/cooperative_admission.md`
    /// §2.2), and the claim is what says the card is at work.
    pub fn card_views(&self) -> Vec<CardView> {
        let loads = self.cards();
        let claimed = self
            .apps
            .get(LATENCY_APP)
            .map(|a| a.claims.held_by())
            .unwrap_or_default();
        let mut views: BTreeMap<CardKey, CardView> = BTreeMap::new();
        for m in self.apps.values().flat_map(|a| a.members.values()) {
            let key = CardKey::of(m);
            let view = views.entry(key.clone()).or_insert_with(|| CardView {
                latency: loads.get(&key).map_or(0, |l| l.latency),
                key,
                hosts: Vec::new(),
            });
            if m.app == LATENCY_APP {
                view.latency += claimed.get(&m.node).copied().unwrap_or(0);
            }
            if m.app == HOSTING_APP {
                view.hosts.push(Host {
                    node: m.node,
                    share: match m.weight {
                        0 => 0,
                        _ => self.machines.characters(&m.host),
                    },
                    eligible: m.may_host(),
                });
            }
        }
        views.into_values().collect()
    }

    /// What the gateway has placed on each card, across every app.
    pub fn cards(&self) -> Cards {
        let mut cards = Cards::new();
        for m in self.apps.values().flat_map(|a| a.members.values()) {
            let load = cards.entry(CardKey::of(m)).or_default();
            load.in_flight += m.in_flight;
            load.drawing += m.drawing;
            if m.app == LATENCY_APP {
                load.latency += m.in_flight;
            }
        }
        cards
    }
}
