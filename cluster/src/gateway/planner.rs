//! Rules 4 and 5 (`docs/cooperative_admission.md` §4.4–§4.5): every
//! [`PLAN_EVERY`] the gateway plans the movable work. A card busy with latency
//! or bulk work lends no characters — unless it holds part of the NPC floor,
//! or every card is busy — and each card's weight moves a quarter of its
//! power at a time, at most every [`STEP_EVERY`]. The result goes to npcd's
//! members as `PLAN`, and its fences to latency placement.

mod busy;
mod fence;

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use super::registry::{CardKey, Fences};
use busy::Busy;
use fence::{fence, Candidate};

pub use fence::NPC_FLOOR_PERCENT;

/// How often the gateway plans.
pub const PLAN_EVERY: Duration = Duration::from_secs(5);

/// How often a card's weight may move one quarter towards its target.
pub const STEP_EVERY: Duration = Duration::from_secs(30);

/// A weight is this many steps of a card's power, from none to all of it.
const QUARTERS: u32 = 4;

/// One npcd member, as the plan weighs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Host {
    pub node: u64,
    /// The power it was admitted at (`MEMBERS`' weight): what it lends by
    /// while its card is free, and zero for a member that takes no lease.
    pub weight: u32,
    /// Ready, in sync, not lagging, not closing: it may be lent a character
    /// (rule 1).
    pub eligible: bool,
}

/// One card, as the plan reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardView {
    pub key: CardKey,
    /// zend units in flight on it.
    pub latency: u32,
    /// Bulk units its members report outstanding.
    pub bulk: u32,
    /// Its npcd members.
    pub hosts: Vec<Host>,
}

impl CardView {
    /// The power it hosts characters with: its eligible npcd members'.
    fn power(&self) -> u32 {
        self.hosts
            .iter()
            .filter(|h| h.eligible)
            .map(|h| h.weight)
            .sum()
    }
}

/// One card's place in a plan, for the plan log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CardPlan {
    pub key: CardKey,
    pub latency: u32,
    pub bulk: u32,
    pub busy: bool,
    /// Thousandths of its power the floor holds back, zero when unfenced.
    pub fence_milli: u32,
    /// Its weight in quarters of its power, and the quarters it moves towards.
    pub quarters: u32,
    pub target: u32,
}

/// A plan.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Planned {
    pub epoch: u64,
    /// Whether the weights moved since the plan before.
    pub changed: bool,
    /// `(node, weight)` for every npcd member.
    pub weights: Vec<(u64, u32)>,
    pub fences: Fences,
    pub cards: Vec<CardPlan>,
}

#[derive(Debug, Clone, Copy)]
struct CardState {
    busy: Busy,
    quarters: u32,
    moved: Option<Instant>,
}

impl Default for CardState {
    /// A card the plan has just met lends by its whole power, as rendezvous
    /// over the admitted powers already does.
    fn default() -> Self {
        CardState {
            busy: Busy::default(),
            quarters: QUARTERS,
            moved: None,
        }
    }
}

/// The gateway's planner.
#[derive(Debug, Default)]
pub struct Planner {
    cards: BTreeMap<CardKey, CardState>,
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

    /// The plan last made: its epoch and weights, for a member just admitted.
    /// `None` before the first.
    pub fn current(&self) -> Option<(u64, Vec<(u64, u32)>)> {
        (self.epoch > 0).then(|| (self.epoch, self.weights.clone()))
    }

    /// Plan the movable work over `cards` at `now`.
    pub fn plan(&mut self, now: Instant, cards: &[CardView]) -> Planned {
        self.last = Some(now);
        self.cards
            .retain(|key, _| cards.iter().any(|c| &c.key == key));
        for c in cards {
            self.cards.entry(c.key.clone()).or_default().busy.observe(
                now,
                c.latency > 0,
                c.bulk > 0,
            );
        }
        let hosting: Vec<usize> = (0..cards.len()).filter(|&i| cards[i].power() > 0).collect();
        let busy: Vec<bool> = cards
            .iter()
            .map(|c| self.cards[&c.key].busy.busy(now))
            .collect();
        let every_busy = !hosting.is_empty() && hosting.iter().all(|&i| busy[i]);
        let candidates: Vec<Candidate> = hosting
            .iter()
            .map(|&i| Candidate {
                index: i,
                latency: cards[i].latency,
                power: cards[i].power(),
            })
            .collect();
        let fenced: BTreeMap<usize, u32> = fence(&candidates).into_iter().collect();

        let mut plans = Vec::new();
        let mut weights = Vec::new();
        let mut fences = Fences::new();
        for (i, c) in cards.iter().enumerate() {
            let is_busy = busy[i];
            let fence_milli = fenced.get(&i).copied().unwrap_or(0);
            let target = if !is_busy || every_busy {
                QUARTERS
            } else if fence_milli > 0 {
                (fence_milli * QUARTERS).div_ceil(1000).max(1)
            } else {
                0
            };
            let state = self.cards.get_mut(&c.key).expect("observed above");
            let may_move = state
                .moved
                .is_none_or(|at| now.duration_since(at) >= STEP_EVERY);
            if state.quarters != target && may_move {
                state.quarters = if target > state.quarters {
                    state.quarters + 1
                } else {
                    state.quarters - 1
                };
                state.moved = Some(now);
            }
            for h in &c.hosts {
                let weight = match (h.eligible, h.weight) {
                    (false, _) | (_, 0) => 0,
                    (true, w) => (w * state.quarters / QUARTERS).max(u32::from(state.quarters > 0)),
                };
                weights.push((h.node, weight));
            }
            if fence_milli > 0 {
                fences.insert(c.key.clone(), fence_milli);
            }
            plans.push(CardPlan {
                key: c.key.clone(),
                latency: c.latency,
                bulk: c.bulk,
                busy: is_busy,
                fence_milli,
                quarters: state.quarters,
                target,
            });
        }
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
            fences,
            cards: plans,
        }
    }
}

#[cfg(test)]
mod tests;
