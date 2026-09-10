//! What the room timer has to be.
//!
//! The interesting assertions here are all about *where* and *when*, because
//! those are the two things a single building for the whole world would get
//! wrong: the same rat in two rooms at once, and a slow burn that ran itself
//! out against an empty floor before anybody walked in.

use std::time::Duration;

use npc_map::salience::{weight, Weight};
use npc_map::witness::{narrate, since};
use npc_map::world::{Happening, Where, World};
use npc_map::MapSet;

use crate::engine::event::Salience;
use crate::engine::rooms::{rung, Rooms, WAIT};
use crate::engine::stir::Stirring;

fn vault() -> World {
    let map = MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"))
        .expect("the shipped vault must load");
    World::new(map)
}

fn at(node: &str) -> Where {
    Where::new("vault-casting", node)
}

/// Long enough that every room has certainly been asked at least once.
fn past_the_wait(n: u32) -> Duration {
    WAIT.end * n
}

/// Every stirring in the log, with where it happened.
fn stirrings(w: &World) -> Vec<(Where, String)> {
    w.log()
        .iter()
        .filter_map(|e| match &e.what {
            Happening::Stirred { text, .. } => Some((e.place.clone(), text.clone())),
            _ => None,
        })
        .collect()
}

#[test]
fn a_room_nobody_is_standing_in_is_never_asked() {
    // Not only an economy. A room advanced while empty burns its whole slow
    // burn against a floor nobody was on, so the first Maker to walk into the
    // coolant gallery finds it pooling for reasons nobody witnessed.
    let mut w = vault();
    w.enter("m1", "Maker-01", at("green-room")).unwrap();
    let mut rooms = Rooms::new();

    for i in 1..=20 {
        rooms.stir_at(&mut w, past_the_wait(i));
    }
    assert_eq!(rooms.len(), 1, "a room with nobody in it was fitted");
    for (place, _) in stirrings(&w) {
        assert_eq!(
            place,
            at("green-room"),
            "the building spoke into an empty room"
        );
    }
}

#[test]
fn two_rooms_are_two_buildings() {
    // The reason this is per room rather than per world: a rat is in a room. One
    // building for the whole vault would have the same rat bolt across two
    // floors at once.
    let mut w = vault();
    w.enter("m1", "Maker-01", at("green-room")).unwrap();
    w.enter("m2", "Maker-02", at("band-one")).unwrap();
    let mut rooms = Rooms::new();

    for i in 1..=30 {
        rooms.stir_at(&mut w, past_the_wait(i));
    }
    assert_eq!(rooms.len(), 2);

    let said = stirrings(&w);
    let green: Vec<&String> = said
        .iter()
        .filter(|(p, _)| *p == at("green-room"))
        .map(|(_, t)| t)
        .collect();
    let band: Vec<&String> = said
        .iter()
        .filter(|(p, _)| *p == at("band-one"))
        .map(|(_, t)| t)
        .collect();
    assert!(!green.is_empty() && !band.is_empty(), "{said:?}");
    assert_ne!(
        green, band,
        "the two rooms ran identically, so they are one building in two places"
    );
}

#[test]
fn a_room_waits_between_looks() {
    // The metronome runs at 2 Hz. Without the per-room timer every room would
    // speak twice a second, which is the four-second treadmill this whole
    // design was built to remove, in better prose.
    let mut w = vault();
    w.enter("m1", "Maker-01", at("green-room")).unwrap();
    let mut rooms = Rooms::new();

    let mut spoke = 0;
    // Ten minutes of world, sampled every half second, exactly as the daemon
    // does it.
    let run = Duration::from_secs(600);
    for beat in 0..1200u32 {
        spoke += rooms.stir_at(&mut w, Duration::from_millis(500) * beat);
    }

    // **Derived from `WAIT`, not written down twice.** The rate is a dial, and
    // a test that hard-codes today's setting fails on the next turn of it for
    // the wrong reason — saying "the room is not waiting" when the room is
    // waiting exactly as long as it was told to.
    let most = run.as_secs() / WAIT.start.as_secs();
    let fewest = run.as_secs() / WAIT.end.as_secs();
    assert!(
        (fewest..=most).contains(&(spoke as u64)),
        "{spoke} stirrings in ten minutes, outside the {fewest}–{most} a \
         {WAIT:?} timer admits"
    );
}

#[test]
fn the_first_look_is_a_wait_away_and_not_on_arrival() {
    // A Maker walking into a room should meet the room, not a fan changing note
    // in the same instant it arrives.
    let mut w = vault();
    w.enter("m1", "Maker-01", at("green-room")).unwrap();
    let mut rooms = Rooms::new();
    assert_eq!(rooms.stir_at(&mut w, Duration::ZERO), 0);
    assert_eq!(
        rooms.stir_at(&mut w, WAIT.start - Duration::from_secs(1)),
        0
    );
}

#[test]
fn a_stirring_is_read_only_in_the_room_it_happened_in() {
    // A building noise that carried through a doorway would put the same rat in
    // two rooms, and a character that walked next door to look would find the
    // one place it certainly is not.
    let mut w = vault();
    w.enter("m1", "Maker-01", at("green-room")).unwrap();
    w.enter("m2", "Maker-02", at("band-one")).unwrap();
    w.mark_seen("m1");
    w.mark_seen("m2");

    w.stir(
        &at("green-room"),
        "The lights in the ceiling flicker and steady.",
        Weight::Note,
    )
    .unwrap();

    let here = since(&w, "m1");
    assert_eq!(here.len(), 1, "{here:?}");
    assert!(here[0].here);
    assert!(
        since(&w, "m2").is_empty(),
        "it carried into the next room: {:?}",
        since(&w, "m2")
    );
}

#[test]
fn a_stirring_reads_as_a_sentence_with_nobody_attached() {
    // The rule the whole `stir` module is written to: the line names its own
    // subject, so `narrate` must stand it on its own rather than putting it in
    // somebody's hands.
    let mut w = vault();
    w.enter("m1", "Maker-01", at("green-room")).unwrap();
    w.enter("m2", "Maker-02", at("green-room")).unwrap();
    w.mark_seen("m1");

    w.say("m2", "somebody should look at the redoubt").unwrap();
    w.stir(
        &at("green-room"),
        "The lights in the ceiling flicker and steady.",
        Weight::Note,
    )
    .unwrap();

    let seen = since(&w, "m1");
    let text = narrate(&w, &seen).expect("something happened");
    assert!(
        text.contains("The lights in the ceiling flicker and steady."),
        "{text}"
    );
    assert!(
        !text.contains("Maker-02 The lights"),
        "the building's doing was attributed to somebody: {text}"
    );
}

#[test]
fn what_the_building_does_is_worth_what_it_says_it_is() {
    // Every other happening's weight follows from its kind. A fan changing note
    // and a breaker going are the same kind of thing, so the weight has to ride
    // on the event.
    let mut w = vault();
    w.enter("m1", "Maker-01", at("green-room")).unwrap();
    w.mark_seen("m1");
    w.stir(
        &at("green-room"),
        "A breaker goes somewhere in the vault.",
        Weight::Preempt,
    )
    .unwrap();
    w.stir(
        &at("green-room"),
        "A pen rolls to the edge of the table.",
        Weight::Ambient,
    )
    .unwrap();

    let seen = since(&w, "m1");
    assert_eq!(weight(&seen[0]), Weight::Preempt);
    assert_eq!(weight(&seen[1]), Weight::Ambient);
}

#[test]
fn the_two_scales_meet_at_the_maps_own_bars() {
    // `stir` reasons in floats, the map in named rungs. The mapping is a lookup
    // against the bars the map already enforces, not a judgement — anything
    // that would preempt does, and anything that would end a standing wait
    // wakes.
    let s = |v: Salience| rung(&Stirring::new("t", "A thing happened in the room.", v));
    assert_eq!(s(Salience::URGENT), Weight::Preempt);
    assert_eq!(s(Salience::new(Salience::PREEMPT_AT)), Weight::Preempt);
    assert_eq!(s(Salience::new(Salience::ROUSES_AT)), Weight::Wake);
    assert_eq!(s(Salience::NORMAL), Weight::Note);
    assert_eq!(s(Salience::IDLE), Weight::Ambient);
}

#[test]
fn a_room_hears_the_recordings_its_building_holds() {
    // The tannoy's words come off the map, and the map puts them on the
    // *building* — so a room three levels down still hears them without the
    // level repeating the list.
    let w = vault();
    let said = w.map().announcements_for("vault-casting");
    assert!(
        said.len() >= 200,
        "the casting floor cannot hear the vault's tannoy: {} recordings",
        said.len()
    );
    assert_eq!(
        said,
        w.map().announcements_for("creators-vault"),
        "a level heard something different from its building"
    );
}

#[test]
fn a_world_with_no_recordings_still_runs() {
    // The engine holds no announcements of its own. A map that declares none is
    // a building nobody left a message in, not a bug.
    let w = vault();
    assert!(w.map().announcements_for("nowhere-at-all").is_empty());
}
