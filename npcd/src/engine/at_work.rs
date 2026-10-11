//! A Maker at work does not hear the building's background.
//!
//! **What a room is doing is not what a story is about.** The building speaks
//! every fifteen to forty seconds — a roll of tape unsticks itself, the light
//! ring flickers, a smell of hot plastic comes and goes — so that a character
//! with nothing to do has something real to react to (`engine::rooms`). A Maker
//! drafting at a desk is not that character: its mission's steps and their
//! outcomes keep it turning. And the building's lines were the newest and most
//! concrete words in front of it each turn, so a Maker asked to write a scene
//! "moment by moment" wrote them in. A Conan story set in the Contested Cities
//! in 2950 came back with the vault's tape and plastic in it, and a reviewer's
//! verdict ended on the state of the vault's ceiling bracket.
//!
//! So while a body carries document work and stands where documents are worked
//! on, the building's atmosphere — anything it says below the rung that wakes
//! somebody — is not carried to it. A fault loud enough to wake still is, and so
//! is everything people do and say.

use npc_map::salience::Weight;
use npc_map::witness::Witnessed;
use npc_map::world::{Happening, World};

use crate::sim::Sim;

/// The act a desk offers that says documents are worked on here.
const AT_A_DESK: &str = "file_read";

/// Whether `body` is at work: it carries a mission with a document to write or
/// review, and stands at a desk where that work is done.
pub fn at_work(world: &World, sim: &Sim, body: &str) -> bool {
    let carrying = sim.missions.active(body).is_some_and(|m| m.work.is_some());
    if !carrying {
        return false;
    }
    let Some(here) = world.actor(body).map(|a| a.at.clone()) else {
        return false;
    };
    sim.part_offers(&format!("{}/{}", here.area, here.node), AT_A_DESK)
}

/// Whether `w` is the building's atmosphere: a line it said with nobody behind
/// it, too slight to wake anybody.
pub fn atmosphere(w: &Witnessed) -> bool {
    matches!(w.what, Happening::Stirred { weight, .. } if weight < Weight::Wake)
}

/// What of `events` reaches `body`: everything, unless it is at work, when the
/// building's atmosphere is left out.
pub fn heard(world: &World, sim: &Sim, body: &str, events: &[Witnessed]) -> Vec<Witnessed> {
    let working = at_work(world, sim, body);
    events
        .iter()
        .filter(|w| !(working && atmosphere(w)))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use npc_map::witness::since;
    use npc_map::world::Where;
    use npc_map::MapSet;

    use super::*;
    use crate::engine::mission::{Mission, Origin, Todo, Work};
    use crate::sim::seed;

    fn vault() -> (World, Sim) {
        let map = MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
            .expect("the shipped vault must load");
        let sim = seed::vault(Some(&map));
        (World::new(map), sim)
    }

    fn drafting() -> Mission {
        Mission::new(
            "Write Conan's life where the record has nothing.",
            vec![Todo::new(
                "write layers/life/conan/2950 The Door.md and commit it",
            )],
            Origin::Generated {
                generator: "life-event".into(),
                target: "life:conan".into(),
                operation: 1,
                step: "write".into(),
            },
        )
        .with_work(Work {
            writes: "layers/life/conan/2950 The Door.md".into(),
            reads: Vec::new(),
            min_words: 250,
            edit_optional: false,
            anew: false,
            checks: Vec::new(),
            tools: Vec::new(),
        })
    }

    /// The room says one atmospheric line and one loud enough to wake, and
    /// returns what `m1` hears of them.
    fn heard_at(w: &mut World, sim: &Sim, room: &Where) -> Vec<String> {
        w.mark_seen("m1");
        w.stir(room, "A roll of tape unsticks itself.", Weight::Note)
            .unwrap();
        w.stir(room, "A breaker goes somewhere in the vault.", Weight::Wake)
            .unwrap();
        heard(w, sim, "m1", &since(w, "m1"))
            .into_iter()
            .filter_map(|e| match e.what {
                Happening::Stirred { text, .. } => Some(text),
                _ => None,
            })
            .collect()
    }

    /// **A Maker drafting at a desk is not handed the building's background**,
    /// and a fault loud enough to wake it still reaches it.
    #[test]
    fn a_maker_at_work_at_a_desk_does_not_hear_the_atmosphere() {
        let (mut w, mut sim) = vault();
        let desk = Where::new("vault-story", "first-room");
        w.enter("m1", "Paxon Vael", desk.clone()).unwrap();
        sim.missions.assign("m1", drafting());
        assert_eq!(
            heard_at(&mut w, &sim, &desk),
            vec!["A breaker goes somewhere in the vault.".to_string()]
        );
    }

    /// With no document to work on, the building is what there is to notice.
    #[test]
    fn a_maker_with_no_work_hears_the_room() {
        let (mut w, sim) = vault();
        let desk = Where::new("vault-story", "first-room");
        w.enter("m1", "Paxon Vael", desk.clone()).unwrap();
        assert_eq!(heard_at(&mut w, &sim, &desk).len(), 2);
    }

    /// Carrying work but away from any desk — walking to it, or to the table —
    /// is not at work.
    #[test]
    fn a_maker_away_from_a_desk_hears_the_room() {
        let (mut w, mut sim) = vault();
        let lobby = Where::new("vault-command", "command-room");
        w.enter("m1", "Paxon Vael", lobby.clone()).unwrap();
        sim.missions.assign("m1", drafting());
        assert!(!at_work(&w, &sim, "m1"));
        assert_eq!(heard_at(&mut w, &sim, &lobby).len(), 2);
    }
}
