//! The clock that lets the tower's state move, and tells the crew when it does.
//!
//! [`crate::sim::upkeep`] says what time does to a tower and
//! [`crate::sim::decisions`] says what that asks of somebody. Neither reads a
//! clock, for the reason [`crate::engine::rooms`] keeps its own: a test that had
//! to wait out a contact's approach in real minutes would not be written. This
//! is the one place that does, and it hands them the time that has passed.
//!
//! # Only while somebody is there
//!
//! A tower nobody is in has no crew to be drained on behalf of, and a world
//! that advanced while empty would greet its first arrival with a contact
//! already on the doorstep and a stockpile spent for reasons nobody witnessed —
//! the same argument [`crate::engine::rooms`] makes for a room. So an empty
//! world does not run its clock, and does not owe it afterwards.
//!
//! # A step is bounded
//!
//! The daemon can stall — a suspended machine, a long pause under a debugger —
//! and the first beat after it would otherwise be asked to account for the whole
//! gap in one step. Time past [`MAX_STEP`] in a single beat is not charged.
//!
//! # Where it is said
//!
//! In every occupied room of the area the tower's consoles stand in. A klaxon
//! that sounded in one room would be a contact only the people standing there
//! knew of, which is the situation a decision with an owner is meant to end.

use std::time::{Duration, Instant};

use npc_map::salience::Weight;
use npc_map::world::{Where, World};

use crate::sim::Sim;

/// The most time one beat may account for.
pub const MAX_STEP: Duration = Duration::from_secs(5);

/// One world's tower clock.
pub struct Watch {
    started: Instant,
    last: Duration,
}

impl Default for Watch {
    fn default() -> Watch {
        Watch::new()
    }
}

impl Watch {
    pub fn new() -> Watch {
        Watch {
            started: Instant::now(),
            last: Duration::ZERO,
        }
    }

    /// Let the tower have the time since the last beat.
    ///
    /// Returns how many lines were said to the crew.
    pub fn watch(&mut self, world: &mut World, sim: &mut Sim) -> usize {
        self.watch_at(world, sim, self.started.elapsed())
    }

    /// The same, at a stated point in the run rather than at the real one.
    pub fn watch_at(&mut self, world: &mut World, sim: &mut Sim, since: Duration) -> usize {
        let passed = since.saturating_sub(self.last).min(MAX_STEP);
        self.last = since;

        let Some(tower) = sim.tower.as_mut() else {
            return 0;
        };
        if world.actors().next().is_none() {
            return 0;
        }

        let happened = tower.advance(passed);
        let alarms = sim.watch_tower();

        let Some(area) = sim.tower_area().map(str::to_string) else {
            return 0;
        };
        let mut rooms: Vec<Where> = world
            .actors()
            .filter(|a| a.at.area == area)
            .map(|a| a.at.clone())
            .collect();
        rooms.sort();
        rooms.dedup();

        let mut said = 0;
        for room in &rooms {
            for (text, weight) in happened
                .iter()
                .map(|t| (t, Weight::Note))
                .chain(alarms.iter().map(|t| (t, Weight::Wake)))
            {
                match world.stir(room, text, weight) {
                    Ok(()) => said += 1,
                    Err(e) => tracing::warn!(room = ?room, "the tower could not be heard: {e:?}"),
                }
            }
        }
        said
    }
}

#[cfg(test)]
mod tests {
    use npc_map::world::Happening;
    use npc_map::MapSet;

    use super::*;
    use crate::sim::field::Resource;
    use crate::sim::seed;
    use crate::sim::upkeep::{BASE_DRAW, CONTACT_EVERY};

    fn shipped() -> MapSet {
        MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
            .expect("the shipped maps must load")
    }

    /// The tower's world, a body in its foundry, and the sim behind it.
    fn tower_world() -> (World, Sim) {
        let map = shipped();
        let sim = seed::battle_cities(Some(&map));
        let mut w = World::new(map);
        w.enter("c1", "Vael", Where::new("tower-redoubt", "foundry"))
            .unwrap();
        (w, sim)
    }

    fn stirred(w: &World) -> Vec<(Where, String, Weight)> {
        w.log()
            .iter()
            .filter_map(|e| match &e.what {
                Happening::Stirred { text, weight } => {
                    Some((e.place.clone(), text.clone(), *weight))
                }
                _ => None,
            })
            .collect()
    }

    fn energy(sim: &Sim) -> u64 {
        sim.tower.as_ref().unwrap().stock_of(Resource::Energy)
    }

    /// Beats of five seconds up to `until`, which is what the daemon's own beats
    /// add up to.
    fn run(watch: &mut Watch, w: &mut World, sim: &mut Sim, until: Duration) {
        let mut at = Duration::ZERO;
        while at < until {
            at += MAX_STEP;
            watch.watch_at(w, sim, at);
        }
    }

    /// **The stockpile falls while the crew is in.**
    #[test]
    fn the_stockpile_drains_while_somebody_is_there() {
        let (mut w, mut sim) = tower_world();
        let before = energy(&sim);
        run(
            &mut Watch::new(),
            &mut w,
            &mut sim,
            Duration::from_secs(600),
        );
        assert_eq!(energy(&sim), before - 10 * BASE_DRAW);
    }

    /// **A world nobody is in does not run its clock, and does not owe it.**
    #[test]
    fn an_empty_tower_does_not_age_and_owes_nothing_afterwards() {
        let map = shipped();
        let mut sim = seed::battle_cities(Some(&map));
        let mut w = World::new(map);
        let before = energy(&sim);
        let mut watch = Watch::new();
        run(&mut watch, &mut w, &mut sim, Duration::from_secs(3_600));
        assert_eq!(energy(&sim), before, "an empty tower was drained");

        w.enter("c1", "Vael", Where::new("tower-redoubt", "foundry"))
            .unwrap();
        watch.watch_at(&mut w, &mut sim, Duration::from_secs(3_605));
        assert!(
            before - energy(&sim) <= BASE_DRAW,
            "the hour spent empty was charged on arrival"
        );
    }

    /// **A stalled daemon is not a skipped hour.**
    #[test]
    fn a_long_gap_is_charged_as_one_bounded_step() {
        let (mut w, mut sim) = tower_world();
        let before = energy(&sim);
        let mut watch = Watch::new();
        watch.watch_at(&mut w, &mut sim, Duration::from_secs(3_600));
        let spent = before - energy(&sim);
        assert!(spent <= BASE_DRAW, "an hour's gap cost {spent}");
    }

    /// **A contact is sighted and said to everyone in the tower**, loud enough
    /// to wake them, and what it asks is on the board for somebody to take.
    #[test]
    fn a_contact_is_sounded_in_every_occupied_room_of_the_tower() {
        let (mut w, mut sim) = tower_world();
        w.enter("c2", "Ulysses", Where::new("tower-redoubt", "bridge"))
            .unwrap();
        let rooms: Vec<Where> = w.actors().map(|a| a.at.clone()).collect();
        run(&mut Watch::new(), &mut w, &mut sim, CONTACT_EVERY);

        let said = stirred(&w);
        let klaxon: Vec<_> = said
            .iter()
            .filter(|(_, text, _)| text.contains("closing on the tower"))
            .collect();
        assert!(!klaxon.is_empty(), "nothing was said: {said:?}");
        assert!(klaxon.iter().all(|(_, _, weight)| *weight == Weight::Wake));
        for room in &rooms {
            assert!(
                klaxon.iter().any(|(place, _, _)| place == room),
                "{room:?} was not told: {klaxon:?}"
            );
        }
        assert!(
            sim.unheld_orders().iter().any(|o| o.contains("contact 1")),
            "the decision was not on the board"
        );
    }

    /// **Said once, not every beat.**
    #[test]
    fn a_contact_is_sounded_once() {
        let (mut w, mut sim) = tower_world();
        let mut watch = Watch::new();
        run(
            &mut watch,
            &mut w,
            &mut sim,
            CONTACT_EVERY + Duration::from_secs(120),
        );
        let n = stirred(&w)
            .iter()
            .filter(|(_, text, _)| text.contains("closing on the tower"))
            .count();
        assert_eq!(n, 1, "the same contact was sounded {n} times");
    }

    /// Somebody outside the tower is not in earshot of its klaxon.
    #[test]
    fn the_klaxon_does_not_sound_outside_the_tower() {
        let (mut w, mut sim) = tower_world();
        let out = Where::new("the-waste", "ruins");
        w.enter("c3", "Wren", out.clone()).unwrap();
        run(&mut Watch::new(), &mut w, &mut sim, CONTACT_EVERY);

        assert!(
            stirred(&w).iter().all(|(place, _, _)| place != &out),
            "the klaxon sounded in the waste"
        );
    }

    /// A world with no tower has nothing to say.
    #[test]
    fn a_world_with_no_tower_says_nothing() {
        let map = shipped();
        let mut sim = seed::vault(Some(&map));
        let mut w = World::new(map);
        w.enter("m1", "Maker-01", Where::new("vault-casting", "green-room"))
            .unwrap();
        let mut watch = Watch::new();
        assert_eq!(watch.watch_at(&mut w, &mut sim, Duration::from_secs(5)), 0);
        assert!(stirred(&w).is_empty());
    }
}
