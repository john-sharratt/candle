//! Getting about the vault, which is most of what happens in it.
//!
//! Three things a Maker does all day: walk somewhere, find out whether it got
//! there, and — for the one trip everybody makes — skip the walk. The rules
//! under test:
//!
//! - **A journey costs its stops.** One leg per tick — anywhere on your own
//!   level is one, the far corner of the building is three — so the shape of
//!   the place is felt rather than declared. There is no physics and no metres.
//! - **The outcome arrives as an event.** By the time a journey has succeeded
//!   or failed the body has had turns in between, so the answer is news and is
//!   delivered like news.
//! - **A journey that ends early says so.** Diverted, teleported or sat down —
//!   never dropped in silence, because a body waiting on an answer that never
//!   comes waits for ever.
//! - **One primitive moves a body**, whether this world walks it a leg at a
//!   time or a game drives it over a tile grid.
//! - **Teleporting goes to one place.** Everything else is walked, which is
//!   what stops the convenience dissolving the building.

use npc_map::stream::Stream;
use npc_map::witness::{narrate, since};
use npc_map::world::{Happening, Lost, Refused, Where, World};
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

fn command(node: &str) -> Where {
    Where::new("vault-command", node)
}

/// Everything one body has been told since it last looked.
fn told(world: &World, id: &str) -> Vec<Happening> {
    since(world, id).into_iter().map(|w| w.what).collect()
}

// =========================================================================
// A journey costs its distance
// =========================================================================

#[test]
fn setting_off_says_how_far_it_is_and_moves_nobody_yet() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    let stops = w.set_off("m1", casting("green-room")).unwrap();

    assert!(stops > 0);
    assert_eq!(w.actor("m1").unwrap().at, casting("band-one"));
    let a = w.actor("m1").unwrap();
    assert_eq!(a.walk.as_ref().unwrap().to_go(&a.at), stops);
}

#[test]
fn anywhere_on_your_own_level_is_one_stop() {
    // No physics and no metres. Within a level a body goes where it is going,
    // because there is nowhere on the way worth stopping at — and most
    // movement in the vault is exactly this.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    for room in ["relations", "green-room", "watch", "roster-room", "core"] {
        assert_eq!(w.set_off("m1", casting(room)), Ok(1), "{room}");
        assert_eq!(w.tick(), 1);
        assert_eq!(w.actor("m1").unwrap().at, casting(room));
        assert!(w.actor("m1").unwrap().walk.is_none());
    }
    // Nobody is on their way, so nothing moves and no tick is spent.
    assert_eq!(w.tick(), 0);
}

#[test]
fn crossing_the_building_is_out_to_the_lift_up_and_along() {
    // Three stops, and they are the three a person would describe: you go to
    // the lift, you ride it, you walk to the room. Those are the points
    // somebody would stop and reconsider, which is why they are the points a
    // mind gets a turn at.
    let mut w = vault();
    w.enter(
        "m1",
        "Maker-01",
        Where::new("vault-portraits", "north-studio"),
    )
    .unwrap();
    assert_eq!(w.set_off("m1", command("command-room")), Ok(3));

    w.tick();
    assert_eq!(w.actor("m1").unwrap().at.node, "core", "not at the lift");
    assert_eq!(w.actor("m1").unwrap().at.area, "vault-portraits");

    w.tick();
    assert_eq!(
        w.actor("m1").unwrap().at,
        Where::new("vault-command", "core")
    );

    w.tick();
    assert_eq!(w.actor("m1").unwrap().at, command("command-room"));
}

#[test]
fn crossing_the_building_costs_more_than_crossing_a_level() {
    // The whole reason the vault is a building. If these were the same number
    // there would be no reason to be on one level rather than another — and
    // the raw distance is nine doorways against two, so the stop count is a
    // compression of the building, not a flattening of it.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-two")).unwrap();
    w.enter("m2", "Maker-02", casting("band-two")).unwrap();

    assert_eq!(w.set_off("m1", casting("band-three")), Ok(1));
    assert_eq!(w.set_off("m2", command("command-room")), Ok(3));
}

#[test]
fn a_body_between_stops_is_always_somewhere_real() {
    // There is no space between rooms. A body part-way through a journey is
    // standing at a lift, can be seen there, and can be talked to.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.set_off("m1", command("command-room")).unwrap();
    while w.tick() > 0 {
        let at = w.actor("m1").unwrap().at.clone();
        assert!(w.node(&at).is_some(), "{at} is not a place");
        assert!(w.actors_at(&at).iter().any(|a| a.id == "m1"));
    }
}

#[test]
fn everyone_who_sets_off_together_stays_level() {
    // One tick is one moment. Four bodies that leave together arrive together,
    // rather than in whatever order the actor map happens to hold them — a
    // watcher must not see them cross a lift lobby in single file.
    let mut w = vault();
    for i in 1..=4 {
        w.enter(
            format!("m{i}"),
            format!("Maker-{i:02}"),
            casting("band-one"),
        )
        .unwrap();
        w.set_off(&format!("m{i}"), command("command-room"))
            .unwrap();
    }
    assert_eq!(w.tick(), 4);
    let places: Vec<Where> = (1..=4)
        .map(|i| w.actor(&format!("m{i}")).unwrap().at.clone())
        .collect();
    assert!(places.windows(2).all(|p| p[0] == p[1]), "{places:?}");
    // And they share the tick, not merely the place.
    let ticks: Vec<u64> = w
        .log()
        .iter()
        .filter(|e| e.what == Happening::Arrived && e.place == places[0])
        .map(|e| e.at)
        .collect();
    assert_eq!(ticks.len(), 4);
    assert!(ticks.windows(2).all(|t| t[0] == t[1]), "{ticks:?}");
}

#[test]
fn the_whole_lift_ride_is_one_stop_however_many_levels_it_crosses() {
    // You do not get out on every floor. Five portal hops, one stop.
    let mut w = vault();
    w.enter("m1", "Maker-01", Where::new("vault-portraits", "core"))
        .unwrap();
    assert_eq!(w.set_off("m1", Where::new("vault-command", "core")), Ok(1));
    w.tick();
    assert_eq!(
        w.actor("m1").unwrap().at,
        Where::new("vault-command", "core")
    );
}

#[test]
fn there_is_no_setting_off_for_somewhere_that_does_not_exist() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    assert!(matches!(
        w.set_off("m1", casting("the-moon")),
        Err(Refused::NoSuchPlace(_))
    ));
    // Refused before standing up: a body that knows the building knows this
    // without walking to a door to find out.
    assert!(w.actor("m1").unwrap().walk.is_none());
    assert!(w.log().iter().all(|e| e.what != Happening::Left));
}

#[test]
fn setting_off_for_where_you_stand_is_no_journey() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    let before = w.now();
    assert_eq!(w.set_off("m1", casting("band-one")), Ok(0));
    assert!(w.actor("m1").unwrap().walk.is_none());
    assert_eq!(w.now(), before, "a journey of nowhere spent a tick");
}

// =========================================================================
// The outcome comes back as an event
// =========================================================================

#[test]
fn a_body_is_told_it_arrived_and_told_nothing_before_that() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    let moves = w.set_off("m1", casting("relations")).unwrap();
    w.mark_seen("m1");

    for _ in 1..moves {
        w.tick();
        assert!(told(&w, "m1").is_empty(), "told mid-journey");
    }
    w.tick();
    assert_eq!(
        told(&w, "m1"),
        vec![Happening::GotThere {
            toward: casting("relations")
        }]
    );
}

#[test]
fn the_arrival_a_body_is_told_about_is_its_own_and_reads_that_way() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.set_off("m1", casting("green-room")).unwrap();
    w.mark_seen("m1");
    w.settle();

    let seen = since(&w, "m1");
    assert_eq!(seen.len(), 1);
    assert!(seen[0].mine());
    let text = narrate(&w, &seen).unwrap();
    assert!(text.starts_with("You got to the green room"), "{text}");
    assert!(!text.contains("Maker-01"), "{text}");
}

#[test]
fn a_body_is_still_told_nothing_about_what_it_merely_did() {
    // The rule the outcome is an exception to. Setting off, sitting down and
    // speaking are all things the actor knows about already.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.mark_seen("m1");
    w.say("m1", "right").unwrap();
    w.take("m1", Some("cindy")).unwrap();
    w.release("m1").unwrap();
    w.set_off("m1", casting("green-room")).unwrap();
    assert!(told(&w, "m1").is_empty());
}

#[test]
fn nobody_else_learns_where_a_body_was_going_or_that_it_got_there() {
    // Intent is not a visible thing and neither is arriving. What the room sees
    // is a person leaving and a person coming in.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("band-one")).unwrap();
    w.mark_seen("m2");
    w.set_off("m1", casting("green-room")).unwrap();
    w.settle();

    let heard = told(&w, "m2");
    assert!(!heard.is_empty(), "m2 saw nobody leave");
    assert!(
        heard
            .iter()
            .all(|h| matches!(h, Happening::Left | Happening::Arrived)),
        "{heard:?}"
    );
}

#[test]
fn a_corridor_sees_the_traffic_through_it() {
    // What the delay buys. A Maker standing on the ring watches four others
    // cross it on their way somewhere, and that is how it knows the level is
    // busy without anybody telling it.
    let mut w = vault();
    w.enter("watcher", "Maker-09", casting("ring-north"))
        .unwrap();
    for i in 1..=4 {
        w.enter(
            format!("m{i}"),
            format!("Maker-{i:02}"),
            casting("band-one"),
        )
        .unwrap();
        w.set_off(&format!("m{i}"), casting("relations")).unwrap();
    }
    w.mark_seen("watcher");
    w.settle();

    let names: Vec<String> = since(&w, "watcher").into_iter().map(|s| s.name).collect();
    for i in 1..=4 {
        assert!(
            names.contains(&format!("Maker-{i:02}")),
            "Maker-{i:02} crossed unseen"
        );
    }
}

#[test]
fn a_body_crossing_your_view_is_one_thing_that_happened() {
    // The raw stream is a footstep at a time — left, came in, left, went onto
    // the east run — and it gets worse the busier the level is, which is the
    // wrong way round. What a watcher recalls is that somebody crossed.
    let mut w = vault();
    w.enter("watcher", "Maker-09", casting("ring-north"))
        .unwrap();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.mark_seen("watcher");
    w.set_off("m1", casting("relations")).unwrap();
    w.settle();

    let text = narrate(&w, &since(&w, "watcher")).unwrap();
    assert_eq!(text.matches("Maker-01").count(), 1, "{text}");
    assert!(text.matches('.').count() == 1, "{text}");
    // It names where the body was last seen, not every room on the way.
    assert!(text.contains("band one"), "{text}");
}

#[test]
fn two_bodies_crossing_at_once_do_not_cut_each_other_in_half() {
    // The condensing is per body. A second Maker walking through the same
    // corridor interleaves with the first in the log and must not split its
    // journey into two sentences.
    let mut w = vault();
    w.enter("watcher", "Maker-09", casting("ring-north"))
        .unwrap();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("core")).unwrap();
    w.mark_seen("watcher");
    w.set_off("m1", casting("relations")).unwrap();
    w.set_off("m2", casting("band-three")).unwrap();
    w.settle();

    let text = narrate(&w, &since(&w, "watcher")).unwrap();
    assert_eq!(text.matches("Maker-01").count(), 1, "{text}");
    assert_eq!(text.matches("Maker-02").count(), 1, "{text}");
}

// =========================================================================
// A game that owns its own movement drives the same primitive
// =========================================================================

#[test]
fn a_body_put_on_its_own_route_carries_on() {
    // Battle Cities walks a tile grid at its own pace and says where the body
    // ended up. Nothing about how long a journey takes lives in this crate, so
    // a caller may cover a route in whatever increments it likes.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.set_off("m1", command("command-room")).unwrap();
    w.mark_seen("m1");

    // One doorway at a time — finer than this world's own legs.
    w.place("m1", casting("ring-north")).unwrap();
    assert!(
        w.actor("m1").unwrap().walk.is_some(),
        "the journey was lost"
    );
    assert!(told(&w, "m1").is_empty(), "told something mid-journey");

    w.place("m1", casting("core")).unwrap();
    assert!(w.actor("m1").unwrap().walk.is_some());

    w.place("m1", command("command-room")).unwrap();
    assert_eq!(
        told(&w, "m1"),
        vec![Happening::GotThere {
            toward: command("command-room")
        }]
    );
}

#[test]
fn a_body_put_off_its_route_has_lost_the_journey() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.set_off("m1", command("command-room")).unwrap();
    w.mark_seen("m1");

    w.place("m1", casting("green-room")).unwrap();
    assert_eq!(
        told(&w, "m1"),
        vec![Happening::LostTheWay {
            toward: command("command-room"),
            why: Lost::Diverted,
        }]
    );
    assert!(w.actor("m1").unwrap().walk.is_none());
}

#[test]
fn every_way_a_body_moves_looks_the_same_from_outside() {
    // Walked, driven or teleported, a watcher sees a body leave and a body
    // come in — because all three go through one place.
    let mut w = vault();
    w.enter("watcher", "Maker-09", command("command-room"))
        .unwrap();
    for (i, arrive) in [0usize, 1, 2].iter().enumerate() {
        let id = format!("m{i}");
        w.enter(&id, format!("Maker-{i:02}"), casting("band-one"))
            .unwrap();
        w.mark_seen("watcher");
        match arrive {
            0 => {
                w.set_off(&id, command("command-room")).unwrap();
                w.settle();
            }
            1 => w.place(&id, command("command-room")).unwrap(),
            _ => w.teleport(&id).unwrap(),
        }
        let seen: Vec<Happening> = since(&w, "watcher")
            .into_iter()
            .filter(|s| s.actor == id)
            .map(|s| s.what)
            .collect();
        assert_eq!(seen, vec![Happening::Arrived], "arrival {arrive}");
    }
}

// =========================================================================
// A journey that ends early says so
// =========================================================================

#[test]
fn setting_off_somewhere_else_reports_the_journey_it_replaced() {
    // Changed its mind at the lift, which is where a journey can be changed —
    // a same-level trip is one stop and offers no chance to.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.set_off("m1", command("command-room")).unwrap();
    w.mark_seen("m1");
    w.tick();
    assert_eq!(w.actor("m1").unwrap().at, casting("core"));
    w.set_off("m1", casting("green-room")).unwrap();

    assert_eq!(
        told(&w, "m1"),
        vec![Happening::LostTheWay {
            toward: command("command-room"),
            why: Lost::Diverted,
        }]
    );
    assert_eq!(
        w.actor("m1").unwrap().walk.as_ref().unwrap().toward,
        casting("green-room")
    );
}

#[test]
fn stopping_at_a_station_reports_the_journey_it_ended() {
    // Stood up to go to the relations table, thought better of it, and sat
    // back down at a terminal in the room it was already in.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.set_off("m1", casting("relations")).unwrap();
    w.mark_seen("m1");
    w.take("m1", Some("cindy")).unwrap();

    assert_eq!(
        told(&w, "m1"),
        vec![Happening::LostTheWay {
            toward: casting("relations"),
            why: Lost::SatDown,
        }]
    );
    assert!(w.actor("m1").unwrap().walk.is_none());
}

#[test]
fn a_recall_reports_the_journey_it_interrupted() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.set_off("m1", casting("relations")).unwrap();
    w.mark_seen("m1");
    w.teleport("m1").unwrap();

    assert_eq!(
        told(&w, "m1"),
        vec![Happening::LostTheWay {
            toward: casting("relations"),
            why: Lost::Teleported,
        }]
    );
}

#[test]
fn a_lost_journey_reads_as_the_answer_to_the_question_that_was_asked() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.set_off("m1", command("command-room")).unwrap();
    w.mark_seen("m1");
    w.tick();
    w.set_off("m1", casting("green-room")).unwrap();

    let text = narrate(&w, &since(&w, "m1")).unwrap();
    // It names where the body was trying to get to, not where it is standing.
    assert!(text.contains("never got to the command room"), "{text}");
    assert!(text.contains("having set off somewhere else"), "{text}");
}

#[test]
fn no_journey_ever_ends_without_an_answer() {
    // The property that keeps a body from waiting for ever: every walk that
    // begins produces exactly one outcome, whichever way it goes.
    let mut w = vault();
    let mut s = Stream::new();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();

    w.set_off("m1", casting("relations")).unwrap();
    w.tick();
    w.set_off("m1", casting("green-room")).unwrap(); // diverted
    w.tick();
    w.take("m1", None).ok(); // no station on a corridor; changes nothing
    w.settle(); // arrives
    w.set_off("m1", command("command-room")).unwrap();
    w.teleport("m1").unwrap(); // recalled off it
    w.set_off("m1", command("long-bench")).unwrap();
    w.settle(); // arrives

    let (set_out, answered) = s
        .drain(&w, "audit")
        .iter()
        .fold((0, 0), |(o, a), e| match e.what {
            Happening::SetOut { .. } => (o + 1, a),
            _ if e.what.is_outcome() => (o, a + 1),
            _ => (o, a),
        });
    assert_eq!(set_out, 4);
    assert_eq!(answered, set_out, "a journey went unanswered");
}

// =========================================================================
// Recall: one destination, no distance
// =========================================================================

#[test]
fn a_recall_arrives_at_once_and_from_anywhere() {
    let mut w = vault();
    for (i, level) in [
        "vault-command",
        "vault-chronicle",
        "vault-story",
        "vault-cartography",
        "vault-casting",
        "vault-portraits",
    ]
    .iter()
    .enumerate()
    {
        let id = format!("m{i}");
        w.enter(&id, format!("Maker-{i:02}"), Where::new(*level, "core"))
            .unwrap();
        w.teleport(&id).unwrap();
        assert_eq!(w.actor(&id).unwrap().at, command("command-room"));
    }
    // Nobody walked: no step was ever taken.
    assert_eq!(w.tick(), 0);
}

#[test]
fn a_recall_is_a_journey_out_that_still_has_to_be_walked_back() {
    // The asymmetry that keeps the building meaningful. Getting called in is
    // free; going back to work is not.
    let mut w = vault();
    w.enter(
        "m1",
        "Maker-01",
        Where::new("vault-portraits", "north-studio"),
    )
    .unwrap();
    let out = w
        .set_off("m1", command("command-room"))
        .expect("the vault is connected");
    w.teleport("m1").unwrap();

    let back = w
        .set_off("m1", Where::new("vault-portraits", "north-studio"))
        .unwrap();
    assert!(back > 1, "the way back was free");
    assert_eq!(out, back, "the building is not the same shape both ways");
}

#[test]
fn being_recalled_to_where_you_were_walking_is_arriving() {
    // A journey answered by its own destination succeeded, however it got
    // there. Telling a Maker it never reached the command room while it stands
    // in the command room is a plain falsehood, and the kind an NPC would then
    // act on.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.set_off("m1", command("command-room")).unwrap();
    w.mark_seen("m1");
    w.tick();
    w.teleport("m1").unwrap();

    assert_eq!(
        told(&w, "m1"),
        vec![Happening::GotThere {
            toward: command("command-room")
        }]
    );
    let text = narrate(&w, &since(&w, "m1")).unwrap();
    assert_eq!(text, "You got to the command room.");
}

#[test]
fn a_recall_lets_go_of_what_was_held() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.take("m1", Some("cindy")).unwrap();
    w.teleport("m1").unwrap();
    assert!(w.actor("m1").unwrap().hold.is_none());
    assert!(w.holder_of("cindy").is_none());
}

#[test]
fn a_recall_from_the_command_room_changes_nothing() {
    let mut w = vault();
    w.enter("m1", "Maker-01", command("command-room")).unwrap();
    let before = w.now();
    w.teleport("m1").unwrap();
    assert_eq!(w.now(), before);
    assert_eq!(w.actor("m1").unwrap().at, command("command-room"));
}

#[test]
fn a_recall_is_seen_as_an_arrival_by_the_room_it_lands_in() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", command("command-room")).unwrap();
    w.mark_seen("m2");
    w.teleport("m1").unwrap();

    assert_eq!(told(&w, "m2"), vec![Happening::Arrived]);
}

#[test]
fn a_world_with_nowhere_to_be_recalled_to_says_so() {
    // Recall is a property of a map, not of the engine. A building that never
    // named one refuses rather than guessing at a room, because a guess would
    // silently give every world a free journey its author never granted it.
    const INN: &str = "
id: the-anchor
kind: building
name: The Anchor
summary: An inn on the harbour road.
nodes:
  - { id: core, kind: core, name: the door }
  - { id: taproom, kind: social, name: the taproom, off: [core] }
";
    let inn: npc_map::Area = serde_yaml::from_str(INN).expect("the inn must parse");
    let mut w = World::new(MapSet::from_areas([inn]).expect("the inn must load"));
    w.enter("a", "Ana", Where::new("the-anchor", "taproom"))
        .unwrap();

    assert_eq!(w.teleport("a"), Err(Refused::NoTeleport));
    // And walking still works, so what is missing is the shortcut and not the
    // building.
    assert_eq!(w.set_off("a", Where::new("the-anchor", "core")), Ok(1));
}

#[test]
fn a_level_inherits_the_recall_point_of_the_building_it_is_in() {
    // Six level files, one recall point, written once. A map that repeated it
    // per level would be a map where five copies could disagree.
    let w = vault();
    for level in w.map().children("creators-vault") {
        assert_eq!(
            w.map().teleport_to(&level.id),
            Some(command("command-room")),
            "{} does not know where it is recalled to",
            level.id
        );
    }
}
