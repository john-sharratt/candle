//! The tower: where it stands, how deep it is dug in, and what it is making.
//!
//! # One machine, a handful of enormous verbs
//!
//! Relocating, laying siege and drilling are not three tools. They are one
//! machine doing one of the things it does, so `command_tower { action, … }`
//! carries them all and a new capability is a new action rather than a new verb
//! in the catalogue.
//!
//! # What it can do is what it can afford
//!
//! Relocation is dimensional folding, and the world is explicit that "energy
//! reserves deplete as towers accumulate transit potential". So the action list
//! is not a constant — it is [`Tower::actions`], computed from the stockpile,
//! and a tower that cannot pay for a fold **does not offer one**.
//!
//! That is the whole reason this is worth modelling rather than narrating. The
//! resource economy enforces itself through the grammar, where a character
//! cannot argue with it, instead of through a refusal it reads and tries again.
//!
//! # Keeper acts here and nowhere else
//!
//! A mind with no body has no `move_to`, no `touch` and no `gather`; every act
//! it has runs through the tower's own systems. This module is therefore most of
//! Keeper's world, and the reason `Availability::Embodied` exists.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::sim::field::Resource;

/// What the tower is doing with itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Posture {
    /// Standing on its legs, going nowhere.
    #[default]
    Standing,
    /// Drilled into the ground. Cannot fold until it surfaces.
    DugIn,
    /// Committed against a target.
    Besieging,
}

/// A grid reference. Integers because a coordinate is a number and means
/// nothing as a category — the one place in the catalogue a scalar is plainly
/// right, and therefore the one place a bounds check has to be real.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Coord {
    pub x: i32,
    pub y: i32,
}

impl Coord {
    pub fn new(x: i32, y: i32) -> Coord {
        Coord { x, y }
    }
}

/// How far a coordinate may sit from the origin before it is off the map.
///
/// A coordinate outside this is well-formed JSON and a nonsense place, which is
/// exactly the failure mode a scalar argument has and an enumerated one cannot.
pub const EDGE: i32 = 4_096;

/// Whether a coordinate is on the map at all.
pub fn on_map(c: Coord) -> bool {
    c.x.abs() <= EDGE && c.y.abs() <= EDGE
}

/// What it costs to fold the tower once.
pub const FOLD_COST: u64 = 500;
/// What it costs to open a siege.
pub const SIEGE_COST: u64 = 300;
/// What one metre of drilling costs.
pub const DRILL_COST_PER_METRE: u64 = 2;

/// One thing the fabricators can be told to make.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recipe {
    pub id: String,
    /// What a character calls it: "bolt rounds".
    pub name: String,
    /// What one batch costs, by resource.
    pub cost: Vec<(Resource, u64)>,
    /// How many of the item one batch yields.
    pub yields: u32,
}

impl Recipe {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        cost: &[(Resource, u64)],
        yields: u32,
    ) -> Recipe {
        Recipe {
            id: id.into(),
            name: name.into(),
            cost: cost.to_vec(),
            yields,
        }
    }
}

/// A batch on one of the eight queues.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Batch {
    pub recipe: String,
    pub count: u32,
    /// Which of the eight it is on.
    pub queue: u8,
}

/// How many production queues the tower runs. The world says eight.
pub const QUEUES: u8 = 8;

/// The tower's own state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tower {
    pub name: String,
    pub at: Coord,
    pub posture: Posture,
    /// Metres below the surface. Zero when standing.
    pub depth: u32,
    pub shields: bool,
    /// What it is besieging, when it is.
    pub besieging: Option<String>,
    stock: BTreeMap<String, u64>,
    recipes: BTreeMap<String, Recipe>,
    queued: Vec<Batch>,
}

impl Tower {
    pub fn new(name: impl Into<String>, at: Coord) -> Tower {
        Tower {
            name: name.into(),
            at,
            posture: Posture::Standing,
            depth: 0,
            shields: false,
            besieging: None,
            stock: BTreeMap::new(),
            recipes: BTreeMap::new(),
            queued: Vec::new(),
        }
    }

    pub fn stock_of(&self, r: Resource) -> u64 {
        self.stock.get(r.name()).copied().unwrap_or(0)
    }

    pub fn put(&mut self, r: Resource, amount: u64) {
        let slot = self.stock.entry(r.name().to_string()).or_insert(0);
        *slot = slot.saturating_add(amount);
    }

    /// Spend, all or nothing. A partial draw would leave a caller believing it
    /// had paid for something it has not.
    pub fn draw(&mut self, r: Resource, amount: u64) -> bool {
        let have = self.stock_of(r);
        if have < amount {
            return false;
        }
        self.stock.insert(r.name().to_string(), have - amount);
        true
    }

    pub fn learn(&mut self, recipe: Recipe) {
        self.recipes.insert(recipe.id.clone(), recipe);
    }

    pub fn recipe_by_name(&self, name: &str) -> Option<&Recipe> {
        let want = name.trim().to_lowercase();
        self.recipes
            .values()
            .find(|r| r.name.to_lowercase() == want || r.id == want)
    }

    /// Whether the stockpile covers one batch of this recipe.
    pub fn affords(&self, recipe: &Recipe, count: u32) -> bool {
        recipe
            .cost
            .iter()
            .all(|(r, per)| self.stock_of(*r) >= per.saturating_mul(count as u64))
    }

    /// What `produce`'s `what` offers — **only what the stockpile covers now**,
    /// so the economy is enforced by the grammar rather than by a refusal.
    pub fn makeable(&self) -> Vec<String> {
        self.recipes
            .values()
            .filter(|r| self.affords(r, 1))
            .map(|r| r.name.clone())
            .collect()
    }

    /// Put a batch on a queue, spending what it costs.
    ///
    /// `Err` carries what is short, because "you cannot afford it" without
    /// saying what of is a refusal a character cannot act on.
    pub fn queue(&mut self, recipe_name: &str, count: u32, queue: u8) -> Result<Batch, String> {
        let Some(recipe) = self.recipe_by_name(recipe_name).cloned() else {
            return Err(format!("Nothing here knows how to make {recipe_name}."));
        };
        if count == 0 {
            return Err("A batch of none is not a batch.".into());
        }
        if queue == 0 || queue > QUEUES {
            return Err(format!("There are {QUEUES} queues, numbered 1 to {QUEUES}."));
        }
        if !self.affords(&recipe, count) {
            let short: Vec<String> = recipe
                .cost
                .iter()
                .filter(|(r, per)| self.stock_of(*r) < per.saturating_mul(count as u64))
                .map(|(r, per)| {
                    format!(
                        "{} {} short",
                        per.saturating_mul(count as u64) - self.stock_of(*r),
                        r.name()
                    )
                })
                .collect();
            return Err(format!("The stockpile will not cover it: {}.", short.join(", ")));
        }
        for (r, per) in &recipe.cost {
            self.draw(*r, per.saturating_mul(count as u64));
        }
        let batch = Batch {
            recipe: recipe.id,
            count,
            queue,
        };
        self.queued.push(batch.clone());
        Ok(batch)
    }

    pub fn queued(&self) -> &[Batch] {
        &self.queued
    }

    /// Which of the eight are free, as strings, for the `queue` argument.
    pub fn free_queues(&self) -> Vec<String> {
        (1..=QUEUES)
            .filter(|q| !self.queued.iter().any(|b| b.queue == *q))
            .map(|q| q.to_string())
            .collect()
    }

    /// **What the tower can actually do right now.**
    ///
    /// The live set behind `command_tower`'s `action`, and the reason a tower
    /// too poor to fold is never invited to try. Each rule is a fact about the
    /// world rather than a policy:
    ///
    /// - folding costs energy it may not have, and a tower in the ground has to
    ///   come up first;
    /// - a siege is a commitment, so it is not offered while one stands;
    /// - drilling and surfacing are opposites and exactly one is available;
    /// - shields are a toggle, so only the half that changes anything shows.
    pub fn actions(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.posture != Posture::DugIn && self.stock_of(Resource::Energy) >= FOLD_COST {
            out.push("relocate".to_string());
        }
        if self.besieging.is_none() && self.stock_of(Resource::Energy) >= SIEGE_COST {
            out.push("siege".to_string());
        }
        if self.besieging.is_some() {
            out.push("lift siege".to_string());
        }
        if self.posture == Posture::DugIn {
            out.push("surface".to_string());
        } else {
            out.push("drill down".to_string());
        }
        out.push(if self.shields {
            "drop shields".to_string()
        } else {
            "raise shields".to_string()
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rich() -> Tower {
        let mut t = Tower::new("Redoubt", Coord::new(10, -4));
        t.put(Resource::Energy, 2_000);
        t.put(Resource::Metal, 400);
        t.put(Resource::Ore, 100);
        t.learn(Recipe::new(
            "bolt",
            "bolt rounds",
            &[(Resource::Metal, 20)],
            30,
        ));
        t.learn(Recipe::new(
            "companion",
            "a companion",
            &[(Resource::Nanobots, 50), (Resource::Metal, 200)],
            1,
        ));
        t
    }

    #[test]
    fn a_tower_offers_only_what_it_can_pay_for() {
        let mut t = rich();
        assert!(t.actions().contains(&"relocate".to_string()));

        t.draw(Resource::Energy, 1_900);
        assert!(
            !t.actions().contains(&"relocate".to_string()),
            "a tower with 100 energy was invited to fold at a cost of 500"
        );
        // The cheaper act is still there, which is what makes this a live set
        // rather than an on/off switch.
        assert!(!t.actions().contains(&"siege".to_string()));
    }

    #[test]
    fn a_tower_in_the_ground_cannot_fold_until_it_surfaces() {
        let mut t = rich();
        t.posture = Posture::DugIn;
        t.depth = 40;
        let acts = t.actions();
        assert!(!acts.contains(&"relocate".to_string()));
        assert!(acts.contains(&"surface".to_string()));
        assert!(
            !acts.contains(&"drill down".to_string()),
            "offered to dig while already dug in"
        );
    }

    #[test]
    fn a_siege_is_not_offered_twice_and_lifting_one_is_offered_only_during() {
        let mut t = rich();
        assert!(!t.actions().contains(&"lift siege".to_string()));
        t.besieging = Some("Ash Keep".into());
        let acts = t.actions();
        assert!(!acts.contains(&"siege".to_string()));
        assert!(acts.contains(&"lift siege".to_string()));
    }

    #[test]
    fn shields_offer_only_the_half_that_would_change_anything() {
        let mut t = rich();
        assert!(t.actions().contains(&"raise shields".to_string()));
        t.shields = true;
        assert!(t.actions().contains(&"drop shields".to_string()));
        assert!(!t.actions().contains(&"raise shields".to_string()));
    }

    #[test]
    fn only_what_the_stockpile_covers_is_offered_for_making() {
        let t = rich();
        // Metal covers bolts; there are no nanobots at all, so no companion.
        assert_eq!(t.makeable(), vec!["bolt rounds".to_string()]);
    }

    #[test]
    fn queueing_a_batch_spends_the_stock_and_lands_on_the_queue() {
        let mut t = rich();
        let before = t.stock_of(Resource::Metal);
        let batch = t.queue("bolt rounds", 3, 1).expect("affordable");
        assert_eq!(batch.count, 3);
        assert_eq!(t.stock_of(Resource::Metal), before - 60);
        assert_eq!(t.queued().len(), 1);
    }

    #[test]
    fn a_batch_that_cannot_be_paid_for_says_what_is_short() {
        let mut t = rich();
        let err = t.queue("a companion", 1, 1).unwrap_err();
        assert!(err.contains("nanobots"), "{err}");
        assert!(err.contains("50"), "the shortfall was not named: {err}");
        assert!(t.queued().is_empty(), "a refused batch was queued anyway");
        assert_eq!(t.stock_of(Resource::Metal), 400, "a refused batch spent stock");
    }

    #[test]
    fn a_queue_outside_the_eight_is_refused_and_says_the_range() {
        let mut t = rich();
        assert!(t.queue("bolt rounds", 1, 0).unwrap_err().contains("1 to 8"));
        assert!(t.queue("bolt rounds", 1, 9).unwrap_err().contains("1 to 8"));
    }

    #[test]
    fn a_batch_of_none_is_refused() {
        let mut t = rich();
        assert!(t.queue("bolt rounds", 0, 1).is_err());
    }

    #[test]
    fn a_used_queue_stops_being_offered_as_free() {
        let mut t = rich();
        assert_eq!(t.free_queues().len(), 8);
        t.queue("bolt rounds", 1, 3).unwrap();
        let free = t.free_queues();
        assert_eq!(free.len(), 7);
        assert!(!free.contains(&"3".to_string()));
    }

    #[test]
    fn drawing_more_than_is_stocked_draws_nothing() {
        let mut t = rich();
        assert!(!t.draw(Resource::Ore, 101));
        assert_eq!(t.stock_of(Resource::Ore), 100, "a refused draw still spent");
        assert!(t.draw(Resource::Ore, 100));
        assert_eq!(t.stock_of(Resource::Ore), 0);
    }

    #[test]
    fn a_coordinate_off_the_edge_is_off_the_map() {
        assert!(on_map(Coord::new(0, 0)));
        assert!(on_map(Coord::new(EDGE, -EDGE)));
        assert!(!on_map(Coord::new(EDGE + 1, 0)));
        assert!(!on_map(Coord::new(0, -EDGE - 1)));
    }
}
