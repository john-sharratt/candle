//! The world outside the walls: what can be taken out of the ground, what is
//! trying to kill you, and the standing orders you have given yourself about it.
//!
//! # The simulator fights; a stance says how
//!
//! Nothing here fires a weapon. A decode takes on the order of a hundred
//! milliseconds and a firefight does not wait, so aiming belongs to formula code
//! and the character's part is to set a **stance** — a posture, who to prefer,
//! and a set of constraints to honour throughout.
//!
//! This is not a concession to latency. It is the same split the catalogue
//! already runs on: `say` carries what a character means and the narrator writes
//! the words. Here `engage` carries what a character intends and the simulator
//! resolves the shots. In both cases the model supplies substance and something
//! faster supplies mechanism, and neither can produce what the other did not
//! license.
//!
//! # Every value here is one the simulator branches on
//!
//! A posture, a priority and a filter are all closed sets. There is no free-text
//! argument anywhere in this file, and that is deliberate: formula code cannot
//! read a sentence, so a string handed to it would be written every turn, read
//! by nothing, and look exactly like control while being a no-op.
//!
//! The tactic that will not enumerate is *spoken* — a `tell` to the squad, which
//! the narrator renders and the other characters perceive. That lands where
//! language works.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The six things a tower runs on.
///
/// Closed because the world says it is closed: energy fuels operations, ore
/// becomes metal, gold buys, crystals refine into nanobots, and a tower short of
/// any one of them is dependent on somebody who is not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resource {
    Energy,
    Metal,
    Ore,
    Gold,
    Crystals,
    Nanobots,
}

impl Resource {
    pub const ALL: &'static [Resource] = &[
        Resource::Energy,
        Resource::Metal,
        Resource::Ore,
        Resource::Gold,
        Resource::Crystals,
        Resource::Nanobots,
    ];

    pub fn name(&self) -> &'static str {
        match self {
            Resource::Energy => "energy",
            Resource::Metal => "metal",
            Resource::Ore => "ore",
            Resource::Gold => "gold",
            Resource::Crystals => "crystals",
            Resource::Nanobots => "nanobots",
        }
    }

    pub fn parse(s: &str) -> Option<Resource> {
        let want = s.trim().to_lowercase();
        Resource::ALL.iter().copied().find(|r| r.name() == want)
    }
}

/// Something in the ground, in a wreck, or in a cache, that a body can take.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Deposit {
    pub id: String,
    /// What a character calls it: "the ore seam", "the burnt-out carrier".
    pub name: String,
    /// `area/node` — where it is.
    pub at: String,
    pub resource: Resource,
    /// What is left. A worked-out deposit stays in the world at zero, because
    /// finding out it is empty is worth a walk and being able to see that it was
    /// once worth something is worth more.
    pub remaining: u32,
    /// How much one act of gathering takes.
    pub yield_per_act: u32,
}

impl Deposit {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        at: impl Into<String>,
        resource: Resource,
        remaining: u32,
    ) -> Deposit {
        Deposit {
            id: id.into(),
            name: name.into(),
            at: at.into(),
            resource,
            remaining,
            yield_per_act: 10,
        }
    }

    pub fn worked_out(&self) -> bool {
        self.remaining == 0
    }

    /// Take one act's worth. `0` from a worked-out deposit.
    pub fn work(&mut self) -> u32 {
        let took = self.yield_per_act.min(self.remaining);
        self.remaining -= took;
        took
    }
}

/// What sort of Zenling it is. Decides how the simulator treats it and, for the
/// catalogue, whether an air-only turret can reach it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Breed {
    Chicken,
    Drone,
    Stinger,
    Shooter,
    Mech,
}

impl Breed {
    /// Whether it flies, which is what *air only* and *ground only* select on.
    pub fn airborne(&self) -> bool {
        matches!(self, Breed::Drone | Breed::Stinger)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Breed::Chicken => "chicken",
            Breed::Drone => "drone",
            Breed::Stinger => "stinger",
            Breed::Shooter => "shooter",
            Breed::Mech => "mech",
        }
    }
}

/// Something hostile, standing somewhere.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hostile {
    pub id: String,
    pub name: String,
    pub breed: Breed,
    /// `area/node`.
    pub at: String,
    pub health: u32,
    /// Whether it is shooting at anybody, which is what the *whatever is firing
    /// on us* priority selects.
    pub firing: bool,
}

impl Hostile {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        breed: Breed,
        at: impl Into<String>,
    ) -> Hostile {
        Hostile {
            id: id.into(),
            name: name.into(),
            breed,
            at: at.into(),
            health: 100,
            firing: false,
        }
    }

    pub fn down(&self) -> bool {
        self.health == 0
    }
}

/// What a body is doing about a fight. One per body, replaced rather than
/// stacked — a character has one stance, and issuing a new one is how it
/// changes its mind.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stance {
    pub posture: String,
    /// The one hostile singled out, if any.
    pub target: Option<String>,
    /// What to do about everything that is not the target.
    pub priority: Option<String>,
    /// Constraints honoured throughout, in the order given.
    pub filters: Vec<String>,
}

/// The postures a stance may take. Closed, and the simulator branches on each.
pub const POSTURES: &[&str] = &[
    "press",
    "hold",
    "fall back",
    "break off",
    "flank",
    "suppress",
    "cover",
    "ambush",
    "hold fire",
];

/// How to choose among everything that is not the named target.
pub const PRIORITIES: &[&str] = &[
    "nearest",
    "greatest threat",
    "air",
    "ground",
    "wounded",
    "whatever is firing on us",
];

/// Standing constraints the simulator honours while it resolves an act.
///
/// **The world publishes what it honours**, so a filter nothing reads is
/// unrepresentable rather than silently ignored. That is the whole guard against
/// this list becoming the free-text guidance string wearing a different coat.
pub const FILTERS: &[&str] = &[
    "avoid player",
    "focus on player",
    "stay in cover",
    "keep formation",
    "hold this ground",
    "conserve ammunition",
    "nothing heavy",
    "free fire",
    "spare noncombatants",
];

/// Everything outside the walls, for one world.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Field {
    deposits: BTreeMap<String, Deposit>,
    hostiles: BTreeMap<String, Hostile>,
    stances: BTreeMap<String, Stance>,
}

impl Field {
    pub fn new() -> Field {
        Field::default()
    }

    pub fn is_empty(&self) -> bool {
        self.deposits.is_empty() && self.hostiles.is_empty()
    }

    pub fn seed_deposit(&mut self, d: Deposit) {
        self.deposits.insert(d.id.clone(), d);
    }

    pub fn seed_hostile(&mut self, h: Hostile) {
        self.hostiles.insert(h.id.clone(), h);
    }

    pub fn deposit(&self, id: &str) -> Option<&Deposit> {
        self.deposits.get(id)
    }

    pub fn hostile(&self, id: &str) -> Option<&Hostile> {
        self.hostiles.get(id)
    }

    pub fn hostiles(&self) -> impl Iterator<Item = &Hostile> {
        self.hostiles.values()
    }

    pub fn deposits(&self) -> impl Iterator<Item = &Deposit> {
        self.deposits.values()
    }

    /// What `gather`'s `what` offers where a body stands.
    ///
    /// A worked-out deposit is **not** offered — the act would take nothing, and
    /// an act that cannot do anything is absent rather than available and
    /// futile, which is the rule the whole catalogue keeps.
    pub fn extractable_at(&self, place: &str) -> Vec<String> {
        self.deposits
            .values()
            .filter(|d| d.at == place && !d.worked_out())
            .map(|d| d.name.clone())
            .collect()
    }

    /// What `engage`'s `target` offers. Anything standing here and not down.
    pub fn hostiles_at(&self, place: &str) -> Vec<String> {
        self.hostiles
            .values()
            .filter(|h| h.at == place && !h.down())
            .map(|h| h.name.clone())
            .collect()
    }

    pub fn deposit_by_name_at(&self, place: &str, name: &str) -> Option<&Deposit> {
        let want = name.trim().to_lowercase();
        self.deposits
            .values()
            .find(|d| d.at == place && (d.name.to_lowercase() == want || d.id == want))
    }

    /// Work a deposit, returning what came out of it and what it was.
    pub fn work(&mut self, place: &str, name: &str) -> Option<(Resource, u32)> {
        let want = name.trim().to_lowercase();
        let d = self
            .deposits
            .values_mut()
            .find(|d| d.at == place && (d.name.to_lowercase() == want || d.id == want))?;
        let took = d.work();
        Some((d.resource, took))
    }

    pub fn hostile_by_name_at(&self, place: &str, name: &str) -> Option<&Hostile> {
        let want = name.trim().to_lowercase();
        self.hostiles
            .values()
            .find(|h| h.at == place && !h.down() && (h.name.to_lowercase() == want || h.id == want))
    }

    /// Set what a body is doing about the fight, replacing whatever stood.
    pub fn set_stance(&mut self, body: &str, stance: Stance) {
        self.stances.insert(body.to_string(), stance);
    }

    pub fn stance(&self, body: &str) -> Option<&Stance> {
        self.stances.get(body)
    }

    /// Drop a body's stance — what breaking off, or leaving the world, means.
    pub fn clear_stance(&mut self, body: &str) -> bool {
        self.stances.remove(body).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ridge() -> Field {
        let mut f = Field::new();
        f.seed_deposit(Deposit::new(
            "east_seam",
            "the ore seam",
            "waste/ridge",
            Resource::Ore,
            25,
        ));
        f.seed_hostile(Hostile::new("d1", "a drone", Breed::Drone, "waste/ridge"));
        f.seed_hostile(Hostile::new("m1", "a mech", Breed::Mech, "waste/ridge"));
        f
    }

    #[test]
    fn every_resource_round_trips_through_its_own_name() {
        for r in Resource::ALL {
            assert_eq!(Resource::parse(r.name()), Some(*r));
        }
        assert_eq!(Resource::parse("unobtanium"), None);
    }

    #[test]
    fn working_a_seam_takes_from_it_until_it_is_out() {
        let mut f = ridge();
        assert_eq!(
            f.work("waste/ridge", "the ore seam"),
            Some((Resource::Ore, 10))
        );
        assert_eq!(
            f.work("waste/ridge", "the ore seam"),
            Some((Resource::Ore, 10))
        );
        // Twenty-five, so the third act yields the remainder and not a full ten.
        assert_eq!(
            f.work("waste/ridge", "the ore seam"),
            Some((Resource::Ore, 5))
        );
        assert_eq!(
            f.work("waste/ridge", "the ore seam"),
            Some((Resource::Ore, 0))
        );
    }

    #[test]
    fn a_worked_out_seam_stops_being_offered_but_stays_in_the_world() {
        let mut f = ridge();
        for _ in 0..3 {
            f.work("waste/ridge", "the ore seam");
        }
        assert!(
            f.extractable_at("waste/ridge").is_empty(),
            "an act that can take nothing was still offered"
        );
        assert!(
            f.deposit("east_seam").is_some(),
            "the seam vanished; a character cannot see it was ever worth something"
        );
    }

    #[test]
    fn nothing_is_extractable_or_hostile_in_a_room_that_holds_neither() {
        let f = ridge();
        assert!(f.extractable_at("tower/hall").is_empty());
        assert!(f.hostiles_at("tower/hall").is_empty());
    }

    #[test]
    fn a_downed_hostile_is_no_longer_offered_as_a_target() {
        let mut f = ridge();
        assert_eq!(f.hostiles_at("waste/ridge").len(), 2);
        f.hostiles.get_mut("d1").unwrap().health = 0;
        assert_eq!(f.hostiles_at("waste/ridge"), vec!["a mech".to_string()]);
        assert!(f.hostile_by_name_at("waste/ridge", "a drone").is_none());
    }

    #[test]
    fn what_flies_is_what_air_only_can_reach() {
        assert!(Breed::Drone.airborne() && Breed::Stinger.airborne());
        assert!(!Breed::Mech.airborne() && !Breed::Chicken.airborne());
        assert!(!Breed::Shooter.airborne());
    }

    #[test]
    fn a_stance_replaces_rather_than_stacks() {
        let mut f = ridge();
        f.set_stance(
            "c1",
            Stance {
                posture: "press".into(),
                ..Stance::default()
            },
        );
        f.set_stance(
            "c1",
            Stance {
                posture: "fall back".into(),
                filters: vec!["free fire".into()],
                ..Stance::default()
            },
        );
        let s = f.stance("c1").unwrap();
        assert_eq!(s.posture, "fall back");
        assert_eq!(s.filters, vec!["free fire".to_string()]);
    }

    #[test]
    fn breaking_off_clears_the_stance_and_clearing_twice_is_honest_about_it() {
        let mut f = ridge();
        f.set_stance("c1", Stance::default());
        assert!(f.clear_stance("c1"));
        assert!(!f.clear_stance("c1"));
        assert!(f.stance("c1").is_none());
    }

    #[test]
    fn the_closed_sets_hold_the_values_the_simulator_branches_on() {
        // Named explicitly, because a value silently disappearing from one of
        // these is a tactic a character can no longer express.
        assert!(POSTURES.contains(&"fall back") && POSTURES.contains(&"break off"));
        assert!(PRIORITIES.contains(&"whatever is firing on us"));
        assert!(FILTERS.contains(&"avoid player") && FILTERS.contains(&"focus on player"));
    }

    #[test]
    fn an_empty_field_says_so_which_is_what_keeps_engage_out_of_a_vault() {
        assert!(Field::new().is_empty());
        assert!(!ridge().is_empty());
    }
}
