//! Calling after somebody who walked out while the speaker was deciding.
//!
//! A character reads its situation, spends seconds deciding what to do, and
//! then acts. The room it read can change under the decision: the one it means
//! to answer has stepped out. What the character could address was fixed at the
//! moment it began deciding ([`World::begin_decision`]), and a line aimed at
//! somebody who was in that company is delivered to them wherever they have
//! got to, rather than refused because the room moved.

use npc_map::witness::{narrate, since};
use npc_map::world::{Happening, Refused, Voice, Where, World};
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

/// Another building altogether: nothing in the casting area can see into it.
fn out_of_sight() -> Where {
    Where::new("vault-chronicle", "core")
}

fn told(world: &World, id: &str) -> Option<String> {
    narrate(world, &since(world, id))
}

/// `m1` and `m2` together in band-one, both caught up, `m1` mid-decision.
fn two_in_a_room() -> World {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("band-one")).unwrap();
    w.mark_seen("m1");
    w.mark_seen("m2");
    w.begin_decision("m1");
    w
}

fn walk_out(w: &mut World, id: &str, to: Where) {
    w.set_off(id, to.clone()).unwrap();
    w.settle();
    assert_eq!(w.actor(id).unwrap().at, to, "{id} must have arrived");
    assert!(
        w.actor(id).unwrap().walk.is_none(),
        "the walk must have finished"
    );
}

// =========================================================================
// The line is delivered
// =========================================================================

#[test]
fn a_line_is_delivered_to_somebody_who_walked_out_of_sight_during_the_decision() {
    let mut w = two_in_a_room();
    walk_out(&mut w, "m2", out_of_sight());

    assert_eq!(w.just_left("m1", "Maker-02"), Some("m2".to_string()));
    w.call_after("m1", "m2", "take the north stair", Voice::Said)
        .unwrap();

    let heard = told(&w, "m2").expect("the addressee heard nothing");
    assert!(heard.contains("Maker-01 told you"), "{heard}");
    assert!(heard.contains("take the north stair"), "{heard}");
}

#[test]
fn the_words_are_logged_where_the_speaker_decided_not_where_they_stand_now() {
    let mut w = two_in_a_room();
    walk_out(&mut w, "m1", casting("ring-north"));
    walk_out(&mut w, "m2", out_of_sight());

    w.call_after("m1", "m2", "wait for me", Voice::Said)
        .unwrap();

    let last = w.log().last().unwrap();
    assert_eq!(last.place, casting("band-one"));
    assert_eq!(last.actor, "m1");
    assert!(matches!(
        &last.what,
        Happening::Said { to: Some(t), voice: Voice::Said, .. } if t == "m2"
    ));
    assert!(told(&w, "m2").unwrap().contains("Maker-01 told you"));
}

#[test]
fn it_is_delivered_when_the_speaker_walked_out_as_well() {
    let mut w = two_in_a_room();
    walk_out(&mut w, "m1", casting("ring-north"));
    walk_out(&mut w, "m2", out_of_sight());

    assert_eq!(w.just_left("m1", "Maker-02"), Some("m2".to_string()));
}

#[test]
fn a_whisper_called_after_keeps_its_words_for_the_addressee() {
    let mut w = two_in_a_room();
    walk_out(&mut w, "m2", out_of_sight());
    w.call_after("m1", "m2", "not a word to the others", Voice::Whispered)
        .unwrap();

    let heard = told(&w, "m2").unwrap();
    assert!(heard.contains("not a word to the others"), "{heard}");
}

#[test]
fn nobody_else_out_of_sight_hears_it() {
    let mut w = two_in_a_room();
    w.enter("m3", "Maker-03", out_of_sight()).unwrap();
    w.mark_seen("m3");
    walk_out(&mut w, "m2", out_of_sight());
    w.call_after("m1", "m2", "for you alone", Voice::Said)
        .unwrap();

    assert!(
        !since(&w, "m3")
            .iter()
            .any(|s| matches!(s.what, Happening::Said { .. })),
        "a bystander out of sight overheard a line aimed at somebody else"
    );
}

#[test]
fn a_leaver_one_doorway_away_is_reached_as_before() {
    let mut w = two_in_a_room();
    walk_out(&mut w, "m2", casting("ring-north"));
    w.call_after("m1", "m2", "the stair", Voice::Said).unwrap();
    assert!(told(&w, "m2").unwrap().contains("Maker-01 told you"));
}

// =========================================================================
// Where the decision was made
// =========================================================================

#[test]
fn a_decision_remembers_the_room_it_began_in_until_it_ends() {
    let mut w = two_in_a_room();
    assert_eq!(w.decided_at("m1"), Some(&casting("band-one")));
    assert_eq!(w.decided_at("m2"), None, "m2 is not deciding");

    walk_out(&mut w, "m1", casting("ring-north"));
    assert_eq!(
        w.decided_at("m1"),
        Some(&casting("band-one")),
        "walking on moved the room the decision was made in"
    );

    w.end_decision("m1");
    assert_eq!(w.decided_at("m1"), None);
}

// =========================================================================
// What is refused
// =========================================================================

#[test]
fn somebody_who_was_not_in_the_company_is_refused() {
    let mut w = two_in_a_room();
    w.enter("m3", "Maker-03", casting("ring-north")).unwrap();

    assert_eq!(w.just_left("m1", "Maker-03"), None);
    assert_eq!(
        w.call_after("m1", "m3", "hello", Voice::Said),
        Err(Refused::NotHere { who: "m3".into() })
    );
}

#[test]
fn somebody_who_arrived_after_the_decision_began_is_refused() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.begin_decision("m1");
    w.enter("m2", "Maker-02", casting("band-one")).unwrap();
    walk_out(&mut w, "m2", out_of_sight());

    assert_eq!(w.just_left("m1", "Maker-02"), None);
    assert_eq!(
        w.call_after("m1", "m2", "hello", Voice::Said),
        Err(Refused::NotHere { who: "m2".into() })
    );
}

#[test]
fn a_speaker_who_has_not_begun_deciding_calls_after_nobody() {
    let mut w = two_in_a_room();
    w.end_decision("m1");
    walk_out(&mut w, "m2", out_of_sight());

    assert_eq!(w.just_left("m1", "Maker-02"), None);
    assert_eq!(
        w.call_after("m1", "m2", "hello", Voice::Said),
        Err(Refused::NotHere { who: "m2".into() })
    );
}

#[test]
fn a_new_decision_replaces_the_company() {
    let mut w = two_in_a_room();
    walk_out(&mut w, "m2", out_of_sight());
    w.begin_decision("m1");

    assert_eq!(w.just_left("m1", "Maker-02"), None);
}

#[test]
fn somebody_still_with_the_speaker_is_not_called_after() {
    let w = two_in_a_room();
    assert_eq!(w.just_left("m1", "Maker-02"), None);
}

#[test]
fn a_body_that_left_the_world_cannot_be_called_after() {
    let mut w = two_in_a_room();
    w.leave("m2").unwrap();

    assert_eq!(w.just_left("m1", "Maker-02"), None);
    assert_eq!(
        w.call_after("m1", "m2", "hello", Voice::Said),
        Err(Refused::NotHere { who: "m2".into() })
    );
}

#[test]
fn nobody_calls_after_themselves() {
    let mut w = two_in_a_room();
    assert_eq!(
        w.call_after("m1", "m1", "hello", Voice::Said),
        Err(Refused::SpeakingToYourself)
    );
}

#[test]
fn only_a_whole_name_matches_regardless_of_case() {
    let mut w = two_in_a_room();
    walk_out(&mut w, "m2", out_of_sight());

    assert_eq!(w.just_left("m1", "maker-02"), Some("m2".to_string()));
    assert_eq!(w.just_left("m1", " Maker-02 "), Some("m2".to_string()));
    assert_eq!(w.just_left("m1", "Maker"), None);
    assert_eq!(w.just_left("m1", "Nobody At All"), None);
}

#[test]
fn a_body_that_is_not_in_the_world_has_no_company() {
    let w = vault();
    assert_eq!(w.just_left("ghost", "Maker-02"), None);
}

#[test]
fn deciding_does_not_move_the_clock() {
    let mut w = two_in_a_room();
    let before = w.now();
    w.begin_decision("m1");
    w.end_decision("m1");
    assert_eq!(w.now(), before);
}
