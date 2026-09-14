//! Who hears what, and what it means to each of them.
//!
//! One rule, and everything here is a consequence of it: **delivery is by
//! place, direction is by address.** Speech lands in the room and everybody
//! standing there gets it; who it was aimed at is a fact about the utterance,
//! not a filter on who receives it.
//!
//! That split is what makes a shared room worth standing in. Being told
//! something and watching somebody else be told it are different facts about
//! the same event, and an NPC that could not tell them apart would be an NPC
//! for whom a room full of people is a room full of noise.

use npc_map::witness::{narrate, since};
use npc_map::world::{Happening, Refused, Voice, Where, World};
use npc_map::MapSet;

fn vault() -> World {
    World::new(
        MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps"))
            .expect("the shipped vault must load"),
    )
}

fn green(node: &str) -> Where {
    Where::new("vault-casting", node)
}

/// Three Makers in the green room and one out on the ring, all caught up.
fn room() -> World {
    let mut w = vault();
    for (id, name, at) in [
        ("m1", "Maker-01", "green-room"),
        ("m2", "Maker-02", "green-room"),
        ("m3", "Maker-03", "green-room"),
        ("out", "Maker-09", "ring-south"),
    ] {
        w.enter(id, name, green(at)).unwrap();
    }
    for id in ["m1", "m2", "m3", "out"] {
        w.mark_seen(id);
    }
    w
}

fn said(world: &World, id: &str) -> Option<String> {
    narrate(world, &since(world, id))
}

// =========================================================================
// One utterance, three readings
// =========================================================================

#[test]
fn being_told_and_overhearing_are_different_facts_about_one_event() {
    let mut w = room();
    w.tell("m1", "m2", "get out of here").unwrap();

    // The one it was aimed at. Reported, not quoted: an utterance carries what
    // somebody meant to convey, not the words they used, so quotation marks
    // would attribute a sentence nobody spoke.
    assert_eq!(
        said(&w, "m2").unwrap(),
        "Maker-01 told you get out of here."
    );
    // The one standing next to them.
    assert_eq!(
        said(&w, "m3").unwrap(),
        "Maker-01 told Maker-02 get out of here."
    );
    // Both heard it. Neither heard a different thing.
    assert_eq!(since(&w, "m2").len(), 1);
    assert_eq!(since(&w, "m3").len(), 1);
}

#[test]
fn speech_to_the_room_is_aimed_at_nobody_and_reads_that_way() {
    let mut w = room();
    w.say("m1", "the redoubt burned twice").unwrap();

    for who in ["m2", "m3"] {
        assert_eq!(
            said(&w, who).unwrap(),
            "Maker-01 said the redoubt burned twice.",
            "{who}"
        );
        assert!(!since(&w, who)[0].addressed(), "{who}");
    }
}

#[test]
fn only_the_one_addressed_counts_as_addressed() {
    let mut w = room();
    w.tell("m1", "m2", "get out of here").unwrap();
    assert!(since(&w, "m2")[0].addressed());
    assert!(!since(&w, "m3")[0].addressed());
}

#[test]
fn the_address_is_carried_whole_so_the_stream_can_be_read_by_anybody() {
    // The narrowing is done on the way out, per reader. What the world records
    // is the whole utterance, addressee and all.
    let mut w = room();
    w.tell("m1", "m2", "get out of here").unwrap();

    assert_eq!(
        since(&w, "m3")[0].what,
        Happening::Said {
            to: Some("m2".into()),
            words: "get out of here".into(),
            voice: Voice::Said,
        }
    );
    assert_eq!(w.log().last().unwrap().what, since(&w, "m3")[0].what);
}

// =========================================================================
// Delivery is still by place
// =========================================================================

#[test]
fn addressing_somebody_does_not_carry_the_words_to_them_from_elsewhere() {
    // Direction is not delivery. Aiming your voice at a person in another room
    // is not speaking to them; it is not possible, and it is refused.
    let mut w = room();
    assert_eq!(
        w.tell("m1", "out", "get out of here"),
        Err(Refused::NotHere { who: "out".into() })
    );
}

#[test]
fn nothing_said_in_a_room_reaches_the_corridor_outside_it() {
    // The green room is visible from the south run — you can see who is in it.
    // You cannot hear it, which is why walking in to ask is still necessary.
    let mut w = room();
    w.say("m1", "the redoubt burned twice").unwrap();
    w.tell("m1", "m2", "get out of here").unwrap();
    assert!(since(&w, "out").is_empty());
}

#[test]
fn a_body_that_walks_in_hears_what_is_said_after_it_arrives() {
    let mut w = room();
    w.say("m1", "before").unwrap();
    w.set_off("out", green("green-room")).unwrap();
    w.settle();
    w.mark_seen("out");
    w.say("m1", "after").unwrap();

    let heard = said(&w, "out").unwrap();
    assert!(heard.contains("after"), "{heard}");
    assert!(!heard.contains("before"), "{heard}");
}

#[test]
fn a_body_never_hears_itself() {
    let mut w = room();
    w.say("m1", "to the room").unwrap();
    w.tell("m1", "m2", "and to you").unwrap();
    assert!(since(&w, "m1").is_empty());
}

#[test]
fn a_body_cannot_address_itself() {
    let mut w = room();
    assert_eq!(
        w.tell("m1", "m1", "well then"),
        Err(Refused::SpeakingToYourself)
    );
    // Muttering is speech to the room, which is a thing a Maker may do.
    w.say("m1", "well then").unwrap();
}

#[test]
fn addressing_a_body_that_does_not_exist_is_refused() {
    let mut w = room();
    assert_eq!(
        w.tell("m1", "nobody", "hello"),
        Err(Refused::NotHere {
            who: "nobody".into()
        })
    );
}

// =========================================================================
// A room full of people
// =========================================================================

#[test]
fn a_crossfire_of_talk_reads_as_a_room_not_a_transcript() {
    let mut w = room();
    w.tell("m1", "m2", "you have the redoubt").unwrap();
    w.tell("m2", "m1", "since yesterday").unwrap();
    w.say("m3", "somebody should tell the watch").unwrap();

    // The third Maker heard both halves of a conversation it was not in, plus
    // its own words, which it does not need telling about.
    let heard = said(&w, "m3").unwrap();
    assert!(heard.contains("Maker-01 told Maker-02"), "{heard}");
    assert!(heard.contains("Maker-02 told Maker-01"), "{heard}");
    assert!(!heard.contains("the watch"), "{heard}");

    // The two talking each heard one thing aimed at them and one aimed past.
    let one = said(&w, "m1").unwrap();
    assert!(one.contains("Maker-02 told you"), "{one}");
    assert!(one.contains("Maker-03 said"), "{one}");
}

#[test]
fn every_room_in_the_vault_carries_speech_to_everybody_standing_in_it() {
    // The floor under the whole design: there is no room where talking to
    // somebody fails to reach them.
    let mut w = vault();
    let places: Vec<Where> = w
        .map()
        .children("creators-vault")
        .iter()
        .flat_map(|a| {
            a.nodes
                .iter()
                .map(|n| Where::new(a.id.clone(), n.id.clone()))
                .collect::<Vec<_>>()
        })
        .collect();

    for place in places {
        w.enter("a", "Maker-A", place.clone()).unwrap();
        w.enter("b", "Maker-B", place.clone()).unwrap();
        w.enter("c", "Maker-C", place.clone()).unwrap();
        w.mark_seen("b");
        w.mark_seen("c");
        w.tell("a", "b", "here").unwrap();

        assert!(since(&w, "b")[0].addressed(), "{place}: not delivered");
        assert!(
            !since(&w, "c")[0].addressed(),
            "{place}: overheard as aimed"
        );
    }
}
