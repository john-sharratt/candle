//! The plan for the cast (`docs/static_placement.md` §5): every
//! [`PLAN_EVERY`], and at once when a card becomes busy — and free again
//! [`busy::RETURN_AFTER`] after its last live conversation — the gateway weighs
//! each npcd member by its machine's share of the cast. A card busy with a live
//! conversation lends nothing while a free card can host — unless every
//! hosting card is busy, when each lends by its share and the characters run
//! beneath the conversations. The weights go to npcd's members as `PLAN`.

mod busy;
mod settle;

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use super::registry::CardKey;
use busy::Busy;
use settle::Settled;

/// How often the gateway plans when nothing forces it to sooner.
pub const PLAN_EVERY: Duration = Duration::from_secs(5);

/// One npcd member, as the plan weighs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Host {
    pub node: u64,
    /// Its machine's share of the cast (`host/machines.yaml`): zero for a
    /// machine the table gives none, and for a member that takes no lease.
    pub share: u32,
    /// Ready, in sync, not lagging, not closing, nothing in its process still
    /// loading: it may be lent a character (rule 1). The plan settles it
    /// ([`settle::SETTLE`]) before it moves anything.
    pub eligible: bool,
}

/// One card, as the plan reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardView {
    pub key: CardKey,
    /// Live conversations — zend requests — in flight on it.
    pub latency: u32,
    /// Its npcd members.
    pub hosts: Vec<Host>,
}

/// One card's place in a plan, for the plan log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardPlan {
    pub key: CardKey,
    pub latency: u32,
    pub busy: bool,
}

/// A plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Planned {
    pub epoch: u64,
    /// Whether the weights moved since the plan before.
    pub changed: bool,
    /// `(node, weight)` for every npcd member.
    pub weights: Vec<(u64, u32)>,
    pub cards: Vec<CardPlan>,
}

/// The gateway's planner.
#[derive(Debug, Default)]
pub struct Planner {
    cards: BTreeMap<CardKey, Busy>,
    hosts: BTreeMap<u64, Settled>,
    epoch: u64,
    weights: Vec<(u64, u32)>,
    last: Option<Instant>,
}

impl Planner {
    /// Whether a plan is due at `now`.
    pub fn due(&self, now: Instant) -> bool {
        self.last
            .is_none_or(|last| now.duration_since(last) >= PLAN_EVERY)
    }

    /// Whether the plan last saw `key` busy at `now`: a card it has not seen
    /// busy is one whose first live conversation calls for a plan at once.
    pub fn busy(&self, key: &CardKey, now: Instant) -> bool {
        self.cards.get(key).is_some_and(|b| b.busy(now))
    }

    /// The plan last made: its epoch and weights, for a member just admitted.
    /// `None` before the first.
    pub fn current(&self) -> Option<(u64, Vec<(u64, u32)>)> {
        (self.epoch > 0).then(|| (self.epoch, self.weights.clone()))
    }

    /// Plan the cast over `cards` at `now`.
    pub fn plan(&mut self, now: Instant, cards: &[CardView]) -> Planned {
        self.last = Some(now);
        self.cards
            .retain(|key, _| cards.iter().any(|c| &c.key == key));
        self.hosts
            .retain(|node, _| cards.iter().flat_map(|c| &c.hosts).any(|h| h.node == *node));
        for c in cards {
            self.cards
                .entry(c.key.clone())
                .or_default()
                .observe(now, c.latency > 0);
            for h in &c.hosts {
                self.hosts
                    .entry(h.node)
                    .or_insert_with(|| Settled::new(h.eligible, now))
                    .observe(h.eligible, now);
            }
        }
        let busy: Vec<bool> = cards.iter().map(|c| self.cards[&c.key].busy(now)).collect();
        let lends = |h: &Host| h.share > 0 && self.hosts[&h.node].eligible();
        // A free card that can host takes the cast off the busy ones; with
        // none, every card lends by its share and the cast runs beneath the
        // conversations.
        let free_hosts = cards
            .iter()
            .zip(&busy)
            .any(|(c, busy)| !busy && c.hosts.iter().any(lends));
        let mut weights: Vec<(u64, u32)> = cards
            .iter()
            .zip(&busy)
            .flat_map(|(c, busy)| {
                c.hosts.iter().map(move |h| {
                    let weight = match lends(h) && !(*busy && free_hosts) {
                        true => h.share,
                        false => 0,
                    };
                    (h.node, weight)
                })
            })
            .collect();
        weights.sort_unstable();
        let changed = weights != self.weights;
        if changed {
            self.epoch += 1;
            self.weights = weights.clone();
        }
        Planned {
            epoch: self.epoch,
            changed,
            weights,
            cards: cards
                .iter()
                .zip(busy)
                .map(|(c, busy)| CardPlan {
                    key: c.key.clone(),
                    latency: c.latency,
                    busy,
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests;
