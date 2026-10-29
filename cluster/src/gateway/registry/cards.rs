//! Cards (`docs/cooperative_admission.md` §4.2): the members of one process,
//! across every app, whose work the gateway counts together.

use std::collections::{BTreeMap, HashMap};

use super::{MemberView, Registry};
use crate::gateway::planner::{CardView, Host};

/// The app whose requests are latency work for the plan (§4.4): a live chat, a
/// passthrough or MCP conversation.
pub const LATENCY_APP: &str = "zend";

/// The app whose members host characters, which the plan lends (§4.4).
pub const HOSTING_APP: &str = "npcd";

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
    /// Requests handed to its members since each one's last `CAPACITY`.
    pub placed: u32,
    /// Draws among `in_flight`.
    pub drawing: u32,
    /// Of `in_flight`, those of [`LATENCY_APP`].
    pub latency: u32,
}

impl CardLoad {
    /// The units handed to the card that its reported figures cannot hold
    /// yet: placed since its last `CAPACITY`, and still in flight.
    pub fn pending(&self) -> u32 {
        self.placed.min(self.in_flight)
    }
}

/// Every card's load, by key.
pub type Cards = HashMap<CardKey, CardLoad>;

/// The thousandths of each fenced card the NPC floor holds back from latency
/// work (§4.4), as the plan last set them. A card not listed holds none back.
pub type Fences = HashMap<CardKey, u32>;

impl Registry {
    /// The plan's fences, which latency placement reads from now on.
    pub fn set_fences(&mut self, fences: Fences) {
        self.fences = fences;
    }

    pub fn fences(&self) -> &Fences {
        &self.fences
    }

    /// Every card as the plan reads it (§4.4): its latency units in flight,
    /// the bulk work its members report outstanding, and its npcd members —
    /// each eligible to host only while it may be routed to (rule 1).
    pub fn card_views(&self) -> Vec<CardView> {
        let loads = self.cards();
        let mut views: BTreeMap<CardKey, CardView> = BTreeMap::new();
        for m in self.apps.values().flat_map(|a| a.members.values()) {
            let key = CardKey::of(m);
            let view = views.entry(key.clone()).or_insert_with(|| CardView {
                latency: loads.get(&key).map_or(0, |l| l.latency),
                key,
                bulk: 0,
                hosts: Vec::new(),
            });
            let bulk = m.capacity.as_ref().map_or(0, |c| c.bulk_backlog);
            view.bulk = view.bulk.max(bulk);
            if m.app == HOSTING_APP {
                view.hosts.push(Host {
                    node: m.node,
                    weight: m.weight,
                    eligible: m.routable(),
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
            load.placed += m.placed;
            load.drawing += m.drawing;
            if m.app == LATENCY_APP {
                load.latency += m.in_flight;
            }
        }
        cards
    }
}
