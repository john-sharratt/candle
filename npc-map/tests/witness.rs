//! What sixteen Makers saw happen, and what they could not.
//!
//! The percept scenarios live next door and test a point in time. These test
//! the change between two points: that a body sees what others did, never its
//! own doings, only once, and never more of them than the walls allow.

use npc_map::witness::{narrate, since};
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

/// The stream as a body would recall it, flattened.
fn told(world: &World, id: &str) -> String {
    narrate(world, &since(world, id))
        .unwrap_or_default()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// =========================================================================
// A deed reaches the room as a sentence
// =========================================================================

/// **`show`'s `what` is a predicate, and the room reads it as one.**
///
/// Every other happening gets its verb from `verb_phrase` (`said …`, `came
/// in`, `took …`); this one is taken as given, because only the caller knows
/// whether the deed was a gesture, a hand laid on somebody, or a body going
/// still. A caller that passes a bare fragment therefore reaches the other
/// characters with the verb missing.
///
/// That is not hypothetical: nine gestures arrived live as *"Yaelis Vayne
/// towards the table, indicating the standing orders"* — a fragment, next to a
/// pause that read correctly because it passes a predicate.
#[test]
fn a_deed_reaches_the_room_with_its_verb_attached() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("band-one")).unwrap();
    w.mark_seen("m2");
    w.show("m1", None, "gestures towards the table").unwrap();

    let seen = told(&w, "m2");
    assert!(
        seen.contains("Maker-01 gestures towards the table"),
        "the deed lost its verb: {seen}"
    );
}

/// The target is named **once**. The aiming belongs to `to`; a caller that also
/// writes it into `what` gets it twice — which is how one act came to read
/// "to Wren: steadying her, at Wren".
#[test]
fn an_aimed_deed_names_who_it_was_aimed_at_exactly_once() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("band-one")).unwrap();
    w.enter("m3", "Maker-03", casting("band-one")).unwrap();
    w.mark_seen("m3");
    w.show("m1", Some("m2"), "does it: steadying her").unwrap();

    let seen = told(&w, "m3");
    assert_eq!(
        seen.matches("Maker-02").count(),
        1,
        "the target was named more than once: {seen}"
    );
    assert!(seen.contains("does it: steadying her"), "{seen}");
}

// =========================================================================
// The stream is typed first and prose second
// =========================================================================

#[test]
fn what_was_witnessed_is_available_without_rendering_it() {
    // A dispatch board, a replay and an audit all want this and none of them
    // want a paragraph. That is half the reason the stream is its own module.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("band-one")).unwrap();
    w.mark_seen("m2");
    w.take("m1", Some("cindy")).unwrap();

    let seen = since(&w, "m2");
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].name, "Maker-01");
    assert!(seen[0].here);
    assert_eq!(
        seen[0].what,
        Happening::TookStation {
            subject: Some("cindy".into())
        }
    );
}

#[test]
fn a_body_never_witnesses_its_own_doings() {
    // An NPC does not need telling what it just did, and a stream that
    // reports it reads like a machine narrating a machine.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.mark_seen("m1");
    w.take("m1", Some("cindy")).unwrap();
    w.say("m1", "there").unwrap();
    assert!(since(&w, "m1").is_empty());
    assert!(narrate(&w, &since(&w, "m1")).is_none());
}

#[test]
fn a_change_is_told_once_and_then_is_spent() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("band-one")).unwrap();
    w.mark_seen("m2");
    w.take("m1", Some("cindy")).unwrap();

    assert!(told(&w, "m2").contains("Maker-01 took cindy."));
    w.mark_seen("m2");
    assert!(told(&w, "m2").is_empty());
}

#[test]
fn a_body_that_never_looks_accumulates_everything() {
    // The stream is not lost by not being read, which is what lets a body be
    // woken by something that happened while it was busy.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("band-one")).unwrap();
    w.mark_seen("m2");

    w.take("m1", Some("cindy")).unwrap();
    w.say("m1", "one").unwrap();
    w.say("m1", "two").unwrap();
    w.release("m1").unwrap();

    assert_eq!(since(&w, "m2").len(), 4);
}

// =========================================================================
// It reads as a memory of the last minute, not as a log
// =========================================================================

#[test]
fn a_run_by_one_body_becomes_one_sentence() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("band-one")).unwrap();
    w.mark_seen("m2");

    w.say("m1", "the redoubt burned twice").unwrap();
    w.take("m1", Some("cindy")).unwrap();

    let t = told(&w, "m2");
    assert_eq!(t, "Maker-01 said the redoubt burned twice and took cindy.");
}

#[test]
fn two_bodies_get_a_sentence_each() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("band-one")).unwrap();
    w.enter("m3", "Maker-03", casting("band-one")).unwrap();
    w.mark_seen("m3");

    w.take("m1", Some("cindy")).unwrap();
    w.take("m2", Some("r-okonkwo")).unwrap();

    let t = told(&w, "m3");
    assert_eq!(t, "Maker-01 took cindy. Maker-02 took r-okonkwo.");
}

#[test]
fn a_place_named_once_in_a_sentence_is_not_named_again() {
    // Watched from the corridor, a body leaving a station and then the room
    // is two clauses about one place. Naming it twice is what a log does.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("ring-north")).unwrap();
    w.take("m1", Some("cindy")).unwrap();
    w.mark_seen("m2");

    w.set_off("m1", casting("green-room")).unwrap();
    w.tick();

    let t = told(&w, "m2");
    assert!(
        t.contains("left a station dark in band one and left"),
        "{t}"
    );
    assert!(!t.contains("and left band one"), "{t}");
}

#[test]
fn going_somewhere_takes_a_motion_preposition() {
    // You are *in* band one; you go *into* it. Reusing the standing
    // preposition produced "went in band one".
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("ring-north")).unwrap();
    w.enter("m2", "Maker-02", casting("ring-north")).unwrap();
    w.mark_seen("m2");

    w.set_off("m1", casting("band-one")).unwrap();
    w.tick();
    let t = told(&w, "m2");
    assert!(t.contains("went into band one"), "{t}");
    assert!(!t.contains("went in band one"), "{t}");
}

// =========================================================================
// Walls narrow what carries
// =========================================================================

#[test]
fn a_subject_is_stripped_on_the_way_out_of_the_room() {
    // The asymmetry the green room exists for. The world still records what
    // was taken — the narrowing is a perception rule, applied on reading.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("ring-north")).unwrap();
    w.mark_seen("m2");
    w.take("m1", Some("cindy")).unwrap();

    let seen = since(&w, "m2");
    assert_eq!(seen.len(), 1);
    assert!(!seen[0].here);
    assert_eq!(seen[0].what, Happening::TookStation { subject: None });
    assert!(told(&w, "m2").contains("Maker-01 lit a station in band one."));
    assert!(!told(&w, "m2").contains("cindy"));

    // The record is whole; only the reading was narrowed.
    assert!(matches!(
        w.log().last().map(|e| &e.what),
        Some(Happening::TookStation { subject: Some(_) })
    ));
}

#[test]
fn speech_does_not_leave_the_room_it_was_spoken_in() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("ring-north")).unwrap();
    w.enter("m3", "Maker-03", casting("band-one")).unwrap();
    w.mark_seen("m2");
    w.mark_seen("m3");

    w.say("m1", "the redoubt burned twice").unwrap();

    assert!(told(&w, "m3").contains("Maker-01 said the redoubt burned twice."));
    assert!(told(&w, "m2").is_empty());
}

#[test]
fn a_room_out_of_sight_carries_nothing_at_all() {
    // Band one and the green room are on opposite sides of the ring.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("green-room")).unwrap();
    w.mark_seen("m2");
    w.take("m1", Some("cindy")).unwrap();

    assert!(since(&w, "m2").is_empty());
}

#[test]
fn another_level_is_never_witnessed() {
    let mut w = vault();
    w.enter(
        "m1",
        "Maker-01",
        Where::new("vault-portraits", "north-studio"),
    )
    .unwrap();
    w.enter("m2", "Maker-02", casting("band-one")).unwrap();
    w.mark_seen("m2");
    w.take("m1", Some("cindy")).unwrap();

    assert!(since(&w, "m2").is_empty());
}

#[test]
fn the_gallery_witnesses_the_map_room_because_the_map_says_so() {
    // The one sightline in the vault that is not a doorway.
    let mut w = vault();
    w.enter(
        "m1",
        "Maker-01",
        Where::new("vault-cartography", "map-room"),
    )
    .unwrap();
    w.enter("m2", "Maker-02", Where::new("vault-cartography", "gallery"))
        .unwrap();
    w.mark_seen("m2");
    w.take("m1", Some("the-geography")).unwrap();

    let t = told(&w, "m2");
    assert!(t.contains("lit a station in the map room"), "{t}");
    assert!(!t.contains("the-geography"), "{t}");
}

// =========================================================================
// The whole crew
// =========================================================================

#[test]
fn a_busy_room_still_recalls_in_a_paragraph() {
    let mut w = vault();
    for i in 1..=8 {
        w.enter(
            format!("m{i}"),
            format!("Maker-{i:02}"),
            casting("green-room"),
        )
        .unwrap();
    }
    w.mark_seen("m1");
    for i in 2..=8 {
        w.say(&format!("m{i}"), "a word").unwrap();
    }

    let t = told(&w, "m1");
    assert!(
        t.split_whitespace().count() < 80,
        "a recollection of {} words is not a paragraph:\n{t}",
        t.split_whitespace().count()
    );
    assert!(t.contains("Maker-08 said"), "{t}");
}

#[test]
fn nothing_witnessed_is_never_narrated_as_nothing() {
    // No "nothing happened" line: an empty stream produces no text at all,
    // so a quiet turn costs nothing.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("green-room")).unwrap();
    w.mark_seen("m1");
    assert!(narrate(&w, &since(&w, "m1")).is_none());
}
