//! What the world can say about a thing a journal claim names.
//!
//! A [`Snapshot`] is a flat table of facts taken from the sim at one instant, read
//! back by subject and attribute. It is built before a draft starts and owns what
//! it holds, so the draft reads a still picture however long it takes and the
//! world's lock is never held across a decode.
//!
//! A subject or attribute the table does not hold reads as `None`, which the
//! check treats as "no answer" rather than a disagreement — a claim about
//! something the sim does not model stays prose.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::engine::journal::verify::World;
use crate::sim::tower::Tower;

/// Subject the tower's own state is read under.
pub const TOWER: &str = "tower";
/// Subject the tower's stock is read under, one attribute per resource.
pub const STOCKPILE: &str = "stockpile";

/// Facts about the world at one instant.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Snapshot {
    facts: BTreeMap<(String, String), String>,
}

impl Snapshot {
    /// A snapshot that knows nothing, for a world with nothing to read.
    pub fn empty() -> Self {
        Self::default()
    }

    /// Add one fact. Subject and attribute are matched without regard to case.
    pub fn with(mut self, subject: &str, attribute: &str, value: impl ToString) -> Self {
        self.facts
            .insert(key(subject, attribute), value.to_string());
        self
    }

    /// The tower's posture, depth, shields, what it besieges and how long its
    /// energy lasts, and what it holds of every resource.
    ///
    /// A value that is not there — no siege, energy that does not run out — is
    /// left out, so it reads as no answer rather than as the word "null".
    pub fn of_tower(tower: &Tower) -> Self {
        let mut snap = Self::empty()
            .with(TOWER, "posture", scalar(&json!(tower.posture)))
            .with(TOWER, "depth", tower.depth)
            .with(TOWER, "shields", tower.shields)
            .with(TOWER, "energy_per_minute", tower.draw_per_minute());
        if let Some(target) = &tower.besieging {
            snap = snap.with(TOWER, "besieging", target);
        }
        if let Some(minutes) = tower.minutes_of_energy() {
            snap = snap.with(TOWER, "minutes_of_energy", minutes);
        }
        for (name, amount) in tower.stockpile() {
            snap = snap.with(STOCKPILE, name, amount);
        }
        snap
    }
}

impl World for Snapshot {
    fn read(&self, subject: &str, attribute: &str) -> Option<String> {
        self.facts.get(&key(subject, attribute)).cloned()
    }
}

fn key(subject: &str, attribute: &str) -> (String, String) {
    (
        subject.trim().to_lowercase(),
        attribute.trim().to_lowercase(),
    )
}

/// A JSON string without its quotes; any other value as JSON writes it.
fn scalar(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use npc_map::load::MapSet;

    use super::*;
    use crate::sim::field::Resource;
    use crate::sim::seed::battle_cities;
    use crate::sim::tower::Posture;
    use crate::sim::upkeep::{BASE_DRAW, RESERVE};

    fn tower_of(map: &MapSet) -> Tower {
        battle_cities(Some(map))
            .tower
            .expect("battle cities has a tower")
    }

    fn maps() -> MapSet {
        MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
            .expect("the shipped maps must load")
    }

    /// A fact is read back however its subject and attribute are cased or spaced.
    #[test]
    fn a_fact_reads_back_without_regard_to_case_or_padding() {
        let snap = Snapshot::empty().with("Tower", "Shields", true);

        assert_eq!(snap.read("tower", "shields").as_deref(), Some("true"));
        assert_eq!(snap.read("  TOWER ", "Shields ").as_deref(), Some("true"));
    }

    /// What the snapshot does not hold is no answer, never an empty string.
    #[test]
    fn an_unknown_thing_reads_as_no_answer() {
        let snap = Snapshot::empty().with("tower", "depth", 0);

        assert_eq!(snap.read("tower", "colour"), None);
        assert_eq!(snap.read("reactor", "depth"), None);
        assert_eq!(Snapshot::empty().read("tower", "depth"), None);
    }

    /// The tower reads its own state, in the words a claim would use.
    #[test]
    fn the_tower_reads_its_posture_depth_and_shields() {
        let map = maps();
        let mut tower = tower_of(&map);
        tower.shields = true;
        tower.depth = 12;
        tower.posture = Posture::DugIn;

        let snap = Snapshot::of_tower(&tower);

        assert_eq!(snap.read(TOWER, "shields").as_deref(), Some("true"));
        assert_eq!(snap.read(TOWER, "depth").as_deref(), Some("12"));
        assert_eq!(
            snap.read(TOWER, "posture").as_deref(),
            Some(scalar(&json!(Posture::DugIn)).as_str())
        );
    }

    /// Stock is read per resource, zeros included, so "we have no X" is a fact the
    /// world can confirm rather than an absence it cannot.
    #[test]
    fn the_stockpile_reads_every_resource() {
        let map = maps();
        let tower = tower_of(&map);

        let snap = Snapshot::of_tower(&tower);

        for (name, amount) in tower.stockpile() {
            assert_eq!(
                snap.read(STOCKPILE, name),
                Some(amount.to_string()),
                "{name}"
            );
        }
        assert_eq!(
            snap.read(STOCKPILE, Resource::Energy.name()),
            Some(tower.stock_of(Resource::Energy).to_string())
        );
    }

    /// How long the energy lasts is the same figure the station's reading gives.
    #[test]
    fn the_tower_reads_how_long_its_energy_lasts() {
        let map = maps();
        let tower = tower_of(&map);
        let energy = tower.stock_of(Resource::Energy);

        let snap = Snapshot::of_tower(&tower);

        assert_eq!(
            snap.read(TOWER, "minutes_of_energy"),
            Some(((energy - RESERVE) / BASE_DRAW).to_string())
        );
        assert_eq!(
            snap.read(TOWER, "energy_per_minute"),
            Some(BASE_DRAW.to_string())
        );
    }

    /// A tower that is besieging nothing says nothing about a siege, and one that
    /// is besieging names the target.
    #[test]
    fn a_siege_is_read_only_while_there_is_one() {
        let map = maps();
        let mut tower = tower_of(&map);

        assert_eq!(Snapshot::of_tower(&tower).read(TOWER, "besieging"), None);

        tower.besieging = Some("the gate".into());
        assert_eq!(
            Snapshot::of_tower(&tower)
                .read(TOWER, "besieging")
                .as_deref(),
            Some("the gate")
        );
    }

    /// A tower whose energy never runs out has no figure for how long it lasts.
    #[test]
    fn energy_that_never_runs_out_has_no_minutes() {
        let map = maps();
        let mut tower = tower_of(&map);
        tower.posture = Posture::DugIn;

        assert_eq!(
            Snapshot::of_tower(&tower).read(TOWER, "minutes_of_energy"),
            None
        );
    }
}
