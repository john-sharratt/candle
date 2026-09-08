//! What carries, and how far.
//!
//! Two rules under test, and they are the ones the whole social design rests
//! on. **A body witnesses only its own room and what it can see into** — not
//! the next room along, not the level below, not a corridor round the corner.
//! And **the unfiltered record exists, is a different module, and is never
//! what a body reads.**
//!
//! The scope tests are exhaustive rather than illustrative: every node on
//! every level is checked against every other, because a leak here would be
//! silent. The prose would still read correctly and the building would simply
//! stop mattering.

use npc_map::stream::Stream;
use npc_map::witness::{narrate, since, Reach, Scope};
use npc_map::world::{Happening, Where, World};
use npc_map::MapSet;

fn vault() -> World {
    World::new(
        MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps"))
            .expect("the shipped vault must load"),
    )
}

fn casting(node: &str) -> Where {
    Where::new("vault-casting", node)
}

/// Every place in the vault, as `(area, node)`.
fn everywhere(world: &World) -> Vec<Where> {
    world
        .map()
        .children("creators-vault")
        .iter()
        .flat_map(|a| {
            a.nodes
                .iter()
                .map(|n| Where::new(a.id.clone(), n.id.clone()))
                .collect::<Vec<_>>()
        })
        .collect()
}

// =========================================================================
// Scope: exactly here, plus what can be seen
// =========================================================================

#[test]
fn a_body_reaches_its_own_room_and_nothing_it_cannot_see() {
    // Exhaustive over the whole building: for every pair of places, the
    // nearness is Here if they are the same place, InSight if the map says so,
    // and OutOfReach otherwise. There is no fourth case and no exception.
    let w = vault();
    let places = everywhere(&w);
    for from in &places {
        let scope = Scope::at(&w, from);
        let node = w.node(from).expect("every place is on the map");
        for to in &places {
            let expected = if to == from {
                Reach::Here
            } else if to.area == from.area && node.visible.contains(&to.node) {
                Reach::InSight
            } else {
                Reach::OutOfReach
            };
            assert_eq!(
                scope.reach(to),
                expected,
                "{from} -> {to}: the scope disagrees with the map"
            );
        }
    }
}

#[test]
fn every_level_is_out_of_reach_from_every_other() {
    // A claim taken on the casting level is felt on the portrait level,
    // because holding is global. Seeing is not. Nothing on another level ever
    // carries, whatever it is.
    let w = vault();
    let places = everywhere(&w);
    for from in &places {
        let scope = Scope::at(&w, from);
        for to in places.iter().filter(|p| p.area != from.area) {
            assert_eq!(
                scope.reach(to),
                Reach::OutOfReach,
                "{from} can reach {to} on another level"
            );
        }
    }
}

#[test]
fn a_scope_lists_the_same_places_it_reaches() {
    // `places()` is what a caller asks when it wants the set rather than the
    // predicate, and the two must not be able to disagree.
    let w = vault();
    for from in everywhere(&w) {
        let scope = Scope::at(&w, &from);
        let listed = scope.places();
        for place in &listed {
            assert_ne!(
                scope.reach(place),
                Reach::OutOfReach,
                "{from} lists {place} but cannot reach it"
            );
        }
        assert!(listed.contains(&from), "{from} does not list itself");
    }
}

#[test]
fn most_of_the_building_is_out_of_reach_from_anywhere_in_it() {
    // A sanity check on the sanity checks: if sight were accidentally wired
    // to everything, every test above would still pass while meaning nothing.
    let w = vault();
    let places = everywhere(&w);
    let total = places.len();
    for from in &places {
        let reachable = Scope::at(&w, from).places().len();
        assert!(
            reachable * 4 < total,
            "{from} can see {reachable} of {total} places, which is not a building"
        );
    }
}

// =========================================================================
// What that scope means for a body
// =========================================================================

#[test]
fn a_room_two_doors_away_witnesses_nothing() {
    // Band one and band two both open off the ring, one door apart, and see
    // nothing of each other.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("band-two")).unwrap();
    w.mark_seen("m2");
    w.take("m1", Some("cindy")).unwrap();
    w.say("m1", "anything at all").unwrap();

    assert!(since(&w, "m2").is_empty());
}

#[test]
fn the_corridor_between_two_rooms_witnesses_both() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-two")).unwrap();
    w.enter("m2", "Maker-02", casting("band-three")).unwrap();
    w.enter("watcher", "Maker-09", casting("cross-upper"))
        .unwrap();
    w.mark_seen("watcher");

    w.take("m1", Some("cindy")).unwrap();
    w.take("m2", Some("r-okonkwo")).unwrap();

    let seen = since(&w, "watcher");
    assert_eq!(seen.len(), 2);
    // Both lit up; neither subject carried.
    for s in &seen {
        assert_eq!(s.what, Happening::TookStation { subject: None });
        assert!(!s.here);
    }
    let t = narrate(&w, &seen).unwrap();
    assert!(!t.contains("cindy"), "{t}");
    assert!(!t.contains("r-okonkwo"), "{t}");
}

#[test]
fn walking_changes_what_a_body_can_witness() {
    // Scope is a function of where the body is, not of who it is. Stepping
    // out of a room stops the room carrying to it.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("band-one")).unwrap();

    w.mark_seen("m2");
    w.say("m1", "in the room").unwrap();
    assert_eq!(since(&w, "m2").len(), 1);

    // Out to the corridor: it can still see band one, so a body moving there
    // carries, but nothing said in it does.
    w.set_off("m2", casting("ring-north")).unwrap();
    w.settle();
    w.mark_seen("m2");
    w.say("m1", "still in the room").unwrap();
    assert!(since(&w, "m2").is_empty());

    // Round to the far side: now band one carries nothing at all.
    w.set_off("m2", casting("green-room")).unwrap();
    w.settle();
    w.mark_seen("m2");
    w.take("m1", Some("cindy")).unwrap();
    assert!(since(&w, "m2").is_empty());
}

#[test]
fn every_room_can_witness_something_happening_in_it() {
    // The floor under the scope rules: wherever a body stands, what happens
    // in that room reaches it. A room that witnessed nothing would be a room
    // nobody could ever be told anything in.
    let mut w = vault();
    for place in everywhere(&w) {
        w.enter("a", "Maker-A", place.clone()).unwrap();
        w.enter("b", "Maker-B", place.clone()).unwrap();
        w.mark_seen("a");
        w.say("b", "something").unwrap();
        let seen = since(&w, "a");
        assert_eq!(seen.len(), 1, "{place} witnessed nothing said in it");
        assert!(seen[0].here, "{place} did not count itself as here");
    }
}

// =========================================================================
// The unfiltered record is a different thing entirely
// =========================================================================

#[test]
fn the_stream_sees_what_no_body_could() {
    // The dividing line, stated as a test. Six Makers on six levels, and one
    // reader that has all of it.
    let mut w = vault();
    let mut s = Stream::new();
    let places = [
        ("vault-command", "anteroom"),
        ("vault-chronicle", "early-range"),
        ("vault-story", "first-room"),
        ("vault-cartography", "north-survey"),
        ("vault-casting", "band-one"),
        ("vault-portraits", "north-studio"),
    ];
    for (i, (level, node)) in places.iter().enumerate() {
        w.enter(
            format!("m{i}"),
            format!("Maker-{i:02}"),
            Where::new(*level, *node),
        )
        .unwrap();
    }
    s.catch_up(&w, "board");
    for i in 0..places.len() {
        w.say(&format!("m{i}"), "a word").unwrap();
    }

    // Every body witnessed nobody but itself, and its own doings do not count.
    for i in 0..places.len() {
        assert!(
            since(&w, &format!("m{i}")).is_empty(),
            "m{i} witnessed another level"
        );
    }
    // The board has all six.
    assert_eq!(s.drain(&w, "board").len(), 6);
}

#[test]
fn the_stream_is_never_narrowed_the_way_a_witness_is() {
    let mut w = vault();
    let mut s = Stream::new();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("ring-north")).unwrap();
    s.catch_up(&w, "persister");
    w.mark_seen("m2");
    w.take("m1", Some("cindy")).unwrap();

    // The corridor gets the console, not the character.
    assert_eq!(
        since(&w, "m2")[0].what,
        Happening::TookStation { subject: None }
    );
    // The record keeps the character.
    let recorded = s.drain(&w, "persister");
    assert_eq!(recorded.len(), 1);
    assert_eq!(
        recorded[0].what,
        Happening::TookStation {
            subject: Some("cindy".into())
        }
    );
}

#[test]
fn readers_hold_their_own_place_and_do_not_hold_each_other_up() {
    let mut w = vault();
    let mut s = Stream::new();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    s.catch_up(&w, "board");
    s.catch_up(&w, "persister");

    w.say("m1", "one").unwrap();
    w.say("m1", "two").unwrap();

    // The board keeps up; the persister has stalled for both events.
    assert_eq!(s.drain(&w, "board").len(), 2);
    assert_eq!(s.peek(&w, "persister").len(), 2);

    w.say("m1", "three").unwrap();
    assert_eq!(s.drain(&w, "board").len(), 1);
    assert_eq!(s.drain(&w, "persister").len(), 3);
}

#[test]
fn peeking_does_not_spend_what_draining_does() {
    let mut w = vault();
    let mut s = Stream::new();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    s.catch_up(&w, "board");
    w.say("m1", "one").unwrap();

    assert_eq!(s.peek(&w, "board").len(), 1);
    assert_eq!(s.peek(&w, "board").len(), 1);
    assert_eq!(s.drain(&w, "board").len(), 1);
    assert_eq!(s.peek(&w, "board").len(), 0);
}

#[test]
fn a_reader_can_be_wound_back_to_replay() {
    let mut w = vault();
    let mut s = Stream::new();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    let mark = w.now();
    w.say("m1", "one").unwrap();
    w.say("m1", "two").unwrap();

    s.seek("replay", mark);
    assert_eq!(s.drain(&w, "replay").len(), 2);
    assert_eq!(s.drain(&w, "replay").len(), 0);
    s.seek("replay", mark);
    assert_eq!(s.drain(&w, "replay").len(), 2);
}

#[test]
fn a_new_reader_gets_the_whole_record_unless_it_asks_not_to() {
    // A persister created after the fact must not have a hole in it, so the
    // default is the beginning. A board that only wants what happens next
    // says so.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.say("m1", "before either reader existed").unwrap();

    let mut s = Stream::new();
    assert_eq!(s.peek(&w, "persister").len(), 2);
    s.catch_up(&w, "board");
    assert_eq!(s.peek(&w, "board").len(), 0);
}

#[test]
fn a_stalled_reader_is_visible_as_a_cursor_left_behind() {
    let mut w = vault();
    let mut s = Stream::new();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    s.catch_up(&w, "board");
    s.catch_up(&w, "persister");
    w.say("m1", "one").unwrap();
    s.drain(&w, "board");

    let behind: Vec<&str> = s
        .readers()
        .filter(|(_, at)| *at < w.now())
        .map(|(name, _)| name)
        .collect();
    assert_eq!(behind, vec!["persister"]);
}
