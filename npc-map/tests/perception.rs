//! Sixteen Makers in one vault, and what each of them can tell.
//!
//! These are scenarios rather than unit tests. The unit tests prove a function
//! does what it says; these prove the *building* behaves — that a claim taken
//! on one level is felt on another, that a full room refuses a seventeenth
//! body, that what one Maker does shows up in what the next one sees, and that
//! none of them can read a colleague's work off the corridor wall.
//!
//! Every room type in the vault appears here at least once, because the shapes
//! differ: a room with sixteen stations, a room with one, a room with a
//! station that claims nothing, a store, three kinds of shared table, a
//! corridor with nothing in it at all.

use npc_map::perceive::{percept, within_reach};
use npc_map::route;
use npc_map::world::{Refused, Where, World};
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

/// One line of the flattened percept, for assertions that do not care where
/// the renderer put its newlines.
fn flat(world: &World, id: &str) -> String {
    percept(world, id)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Walk somebody to a room on their own level and let them get there.
///
/// Setting a scene, not testing movement — how long a journey takes and what a
/// body is told about it is [`movement`](../movement.rs)'s subject. Here it is
/// only how somebody came to be standing where the percept finds them.
fn walk(world: &mut World, id: &str, node: &str) {
    let to = Where::new(world.actor(id).expect("an actor").at.area.clone(), node);
    world.set_off(id, to).expect("a way there");
    world.settle();
}

/// A crew, entered at the lift of the casting level.
fn crew(world: &mut World, n: usize) -> Vec<String> {
    (1..=n)
        .map(|i| {
            let id = format!("m{i:02}");
            world
                .enter(&id, format!("Maker-{i:02}"), casting("core"))
                .expect("the lift exists");
            id
        })
        .collect()
}

// =========================================================================
// One body, one room
// =========================================================================

#[test]
fn a_maker_alone_at_a_station_is_told_where_it_is_and_what_it_holds() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.take("m1", Some("r-okonkwo")).unwrap();

    let p = flat(&w, "m1");
    assert!(
        p.contains("You are working in band one, holding r-okonkwo."),
        "{p}"
    );
    // How many are left, not how many there are: the total is architecture.
    assert!(p.contains("Five character terminals stand free."), "{p}");
}

#[test]
fn a_percept_never_repeats_what_the_memory_already_says() {
    // Exits, capacity and which level a room is on do not change, so paying
    // for them every turn for every Maker is the waste the split exists to
    // avoid. This is the rule the whole module follows from.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.take("m1", Some("r-okonkwo")).unwrap();

    let p = flat(&w, "m1");
    // The way out of band one is the north run, and it always will be.
    assert!(!p.contains("north run"), "{p}");
    // Band one has six terminals, and it always will have.
    assert!(!p.contains("six"), "{p}");
    // Band one is on the casting level, and always will be — and its name is
    // unique in the vault, so it needs no qualifying.
    assert!(!p.contains("casting level"), "{p}");
}

#[test]
fn a_body_alone_in_a_quiet_corridor_gets_one_sentence() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("ring-north")).unwrap();

    let p = percept(&w, "m1");
    assert_eq!(p.trim(), "You are on the north run of the casting level.");
    // Nothing is within reach in a corridor, which is the whole point of
    // hanging tools on parts.
    assert!(within_reach(&w, "m1").is_empty());
}

#[test]
fn a_repeated_name_is_qualified_and_a_unique_one_is_not() {
    // Every level has a north run; only one has a green room. Saying "of the
    // casting level" where it is not needed is the same waste as repeating an
    // exit, and leaving it off where it is needed is an ambiguity.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("ring-north")).unwrap();
    assert!(flat(&w, "m1").contains("of the casting level"));

    walk(&mut w, "m1", "green-room");
    let p = flat(&w, "m1");
    assert!(p.contains("You are in the green room."), "{p}");
    assert!(!p.contains("casting level"), "{p}");
}

#[test]
fn a_room_with_no_stations_says_nothing_about_stations() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("green-room")).unwrap();
    let p = flat(&w, "m1");
    assert!(p.contains("You are in the green room."), "{p}");
    assert!(!p.contains("stand free"), "{p}");
    // But sitting and talking are still within reach.
    let tools = within_reach(&w, "m1");
    assert!(tools.contains(&"room.talk"), "{tools:?}");
}

// =========================================================================
// Every kind of room in the vault produces a percept
// =========================================================================

#[test]
fn every_room_in_the_building_can_be_perceived() {
    let mut w = vault();
    let levels: Vec<String> = w
        .map()
        .children("creators-vault")
        .iter()
        .map(|a| a.id.clone())
        .collect();
    for level in levels {
        let nodes: Vec<String> = w
            .map()
            .get(&level)
            .unwrap()
            .nodes
            .iter()
            .map(|n| n.id.clone())
            .collect();
        for node in nodes {
            let place = Where::new(level.clone(), node.clone());
            w.enter("wanderer", "The Wanderer", place).unwrap();
            let p = flat(&w, "wanderer");
            assert!(!p.is_empty(), "{level}/{node} perceives nothing");
            // A percept always grounds the body it belongs to.
            assert!(
                p.starts_with("You are "),
                "{level}/{node} does not say where it is:\n{p}"
            );
            assert!(p.ends_with('.'), "{level}/{node} ends mid-sentence:\n{p}");
        }
    }
}

#[test]
fn the_watch_is_a_station_that_holds_nobody() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("watch")).unwrap();

    // It binds nothing, so naming a subject is refused rather than ignored.
    assert_eq!(
        w.take("m1", Some("r-okonkwo")).unwrap_err(),
        Refused::SubjectRefused
    );
    w.take("m1", None).unwrap();

    let p = flat(&w, "m1");
    assert!(p.contains("You are working in the watch."), "{p}");
    assert!(!p.contains("holding"), "{p}");
    assert!(within_reach(&w, "m1").contains(&"cast.read_all"));
}

#[test]
fn a_station_that_claims_something_refuses_to_be_taken_blank() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    assert_eq!(
        w.take("m1", None).unwrap_err(),
        Refused::SubjectNeeded {
            binds: "one character".into()
        }
    );
}

#[test]
fn a_store_has_nothing_to_sit_at_but_plenty_to_reach() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("roster-room")).unwrap();
    assert_eq!(
        w.take("m1", Some("anyone")).unwrap_err(),
        Refused::NothingToWorkAt
    );
    let tools = within_reach(&w, "m1");
    assert!(tools.contains(&"roster.read"), "{tools:?}");
    assert!(tools.contains(&"roster.take_unheld"), "{tools:?}");
}

#[test]
fn the_map_is_one_station_and_the_second_comer_waits() {
    let mut w = vault();
    let map_room = Where::new("vault-cartography", "map-room");
    w.enter("m1", "Maker-01", map_room.clone()).unwrap();
    w.enter("m2", "Maker-02", map_room).unwrap();

    w.take("m1", Some("the-geography")).unwrap();
    assert_eq!(
        w.take("m2", Some("the-geography")).unwrap_err(),
        Refused::AlreadyHeld {
            subject: "the-geography".into(),
            by: "Maker-01".into()
        }
    );
    // Even a different subject is refused: there is only one table.
    assert_eq!(
        w.take("m2", Some("something-else")).unwrap_err(),
        Refused::EveryStationTaken { of: 1 }
    );
}

// =========================================================================
// Many bodies
// =========================================================================

#[test]
fn sixteen_makers_fit_on_the_casting_level_and_a_seventeenth_does_not() {
    let mut w = vault();
    let ids = crew(&mut w, 17);

    // Six in band one, six in band two, four in band three: sixteen seats.
    let plan = [("band-one", 6), ("band-two", 6), ("band-three", 4)];
    let mut i = 0;
    for (room, seats) in plan {
        for _ in 0..seats {
            walk(&mut w, &ids[i], room);
            w.take(&ids[i], Some(&format!("character-{i:02}"))).unwrap();
            i += 1;
        }
    }
    assert_eq!(i, 16);

    // The seventeenth finds every band full.
    for (room, seats) in plan {
        walk(&mut w, &ids[16], room);
        assert_eq!(
            w.take(&ids[16], Some("character-16")).unwrap_err(),
            Refused::EveryStationTaken { of: seats }
        );
    }
}

#[test]
fn a_full_room_still_lets_a_body_stand_in_it() {
    // Capacity is on the stations, not on the floor: a Maker with nowhere to
    // sit can still walk in, and that is what makes the green room work.
    let mut w = vault();
    let ids = crew(&mut w, 5);
    for (i, id) in ids.iter().enumerate().take(4) {
        walk(&mut w, id, "band-three");
        w.take(id, Some(&format!("c{i}"))).unwrap();
    }
    walk(&mut w, &ids[4], "band-three");
    let p = flat(&w, &ids[4]);
    assert!(p.contains("Nothing here is free."), "{p}");
    assert!(p.contains("are working"), "{p}");
}

#[test]
fn who_is_working_and_who_is_not_are_told_apart() {
    let mut w = vault();
    let ids = crew(&mut w, 4);
    for id in &ids {
        walk(&mut w, id, "band-one");
    }
    w.take(&ids[0], Some("a")).unwrap();
    w.take(&ids[1], Some("b")).unwrap();

    let p = flat(&w, &ids[3]);
    assert!(p.contains("Maker-01 and Maker-02 are working"), "{p}");
    assert!(p.contains("Maker-03 is not"), "{p}");
}

// =========================================================================
// The claim is global
// =========================================================================

#[test]
fn a_character_held_on_one_level_cannot_be_taken_on_another() {
    // The reason to claim a character is coherence, and coherence does not
    // partition by floor. This is the invariant the whole design rests on.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter(
        "m2",
        "Maker-02",
        Where::new("vault-portraits", "north-studio"),
    )
    .unwrap();

    w.take("m1", Some("r-okonkwo")).unwrap();
    assert_eq!(
        w.take("m2", Some("r-okonkwo")).unwrap_err(),
        Refused::AlreadyHeld {
            subject: "r-okonkwo".into(),
            by: "Maker-01".into()
        }
    );

    // And once the first lets go, the easel two levels up can have them.
    w.release("m1").unwrap();
    w.take("m2", Some("r-okonkwo")).unwrap();
    assert_eq!(w.holder_of("r-okonkwo").unwrap().id, "m2");
}

#[test]
fn one_body_holds_one_thing_at_a_time() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.take("m1", Some("a")).unwrap();
    assert_eq!(
        w.take("m1", Some("b")).unwrap_err(),
        Refused::AlreadyAtAStation
    );
}

#[test]
fn walking_out_frees_the_claim_for_everybody_else() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("band-two")).unwrap();
    w.take("m1", Some("cindy")).unwrap();
    assert!(w.take("m2", Some("cindy")).is_err());

    walk(&mut w, "m1", "green-room");
    w.take("m2", Some("cindy")).unwrap();
    assert_eq!(w.holder_of("cindy").unwrap().id, "m2");
}

#[test]
fn releasing_without_holding_anything_is_refused_rather_than_ignored() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("green-room")).unwrap();
    assert_eq!(w.release("m1").unwrap_err(), Refused::NotAtAStation);
}

// =========================================================================
// What one does, the next sees — as state, not as news
// =========================================================================

#[test]
fn a_station_taken_by_one_maker_is_a_station_the_next_sees_taken() {
    let mut w = vault();
    let ids = crew(&mut w, 2);
    for id in &ids {
        walk(&mut w, id, "band-one");
    }
    w.take(&ids[0], Some("cindy")).unwrap();

    // The change shows in the snapshot without any event being read: the
    // percept is computed from the world, not from a stream.
    let p = flat(&w, &ids[1]);
    assert!(p.contains("Five character terminals stand free."), "{p}");
    assert!(p.contains("Maker-01 is working."), "{p}");
}

#[test]
fn a_percept_is_the_same_answer_however_often_it_is_asked() {
    // The point of splitting the stream out. While the events lived in here,
    // asking twice gave two answers, because the first ask spent the news.
    let mut w = vault();
    let ids = crew(&mut w, 2);
    for id in &ids {
        walk(&mut w, id, "band-one");
    }
    w.take(&ids[0], Some("cindy")).unwrap();

    let once = flat(&w, &ids[1]);
    let twice = flat(&w, &ids[1]);
    w.mark_seen(&ids[1]);
    let after_reading_the_stream = flat(&w, &ids[1]);

    assert_eq!(once, twice);
    assert_eq!(once, after_reading_the_stream);
}

#[test]
fn a_percept_carries_no_news_of_its_own() {
    // Nothing that happened, however recent, appears here. That is the
    // stream's, and it is a different module.
    let mut w = vault();
    let ids = crew(&mut w, 2);
    for id in &ids {
        walk(&mut w, id, "band-one");
    }
    w.say(&ids[0], "the redoubt burned twice").unwrap();
    w.take(&ids[0], Some("cindy")).unwrap();

    let p = flat(&w, &ids[1]);
    assert!(!p.contains("said"), "{p}");
    assert!(!p.contains("took"), "{p}");
    assert!(!p.contains("came in"), "{p}");
    assert!(!p.contains("redoubt"), "{p}");
}

// =========================================================================
// Sight is limited on purpose
// =========================================================================

#[test]
fn what_a_station_holds_cannot_be_read_from_the_corridor() {
    // The asymmetry the green room exists for. Break this and nobody ever
    // needs to ask anybody anything.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("ring-north")).unwrap();
    w.take("m1", Some("cindy")).unwrap();

    let p = flat(&w, "m2");
    // Who is in there is visible.
    assert!(p.contains("You can see Maker-01 in band one."), "{p}");
    // What they are holding is not.
    assert!(!p.contains("cindy"), "{p}");
}

#[test]
fn a_room_out_of_sight_reports_nothing_at_all() {
    // Band one and the green room are on opposite sides of the ring and see
    // nothing of each other.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("green-room")).unwrap();
    w.take("m1", Some("cindy")).unwrap();

    let p = flat(&w, "m2");
    assert!(!p.contains("Maker-01"), "{p}");
    assert!(!p.contains("band one"), "{p}");
}

#[test]
fn the_gallery_sees_the_map_room_because_the_map_says_so() {
    // The one sightline in the vault that is not a doorway. It exists so the
    // geography can be argued about by somebody not holding it.
    let mut w = vault();
    w.enter(
        "m1",
        "Maker-01",
        Where::new("vault-cartography", "map-room"),
    )
    .unwrap();
    w.enter("m2", "Maker-02", Where::new("vault-cartography", "gallery"))
        .unwrap();
    w.take("m1", Some("the-geography")).unwrap();

    let p = flat(&w, "m2");
    assert!(p.contains("You can see Maker-01 in the map room."), "{p}");
    assert!(!p.contains("the-geography"), "{p}");
}

// =========================================================================
// Getting about
// =========================================================================

#[test]
fn a_maker_walks_to_a_room_without_being_told_the_way() {
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("core")).unwrap();
    walk(&mut w, "m1", "relations");
    assert_eq!(w.actor("m1").unwrap().at, casting("relations"));
}

#[test]
fn a_room_named_on_the_wrong_level_is_nowhere() {
    // The early range is a real room, on the chronicle level. Asked for on the
    // casting level it is not a place at all — a body walks to a *place*, and
    // a place is a level and a room together.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("core")).unwrap();
    let err = w.set_off("m1", casting("early-range")).unwrap_err();
    assert!(matches!(err, Refused::NoSuchPlace(_)), "{err:?}");

    // Asked for where it actually is, it is a long walk and not a refusal.
    let moves = w
        .set_off("m1", Where::new("vault-chronicle", "early-range"))
        .expect("the lift joins the levels");
    assert!(moves > 1, "another level was one step away");
}

#[test]
fn every_room_on_every_level_is_walkable_from_the_lift() {
    let w = vault();
    let levels: Vec<String> = w
        .map()
        .children("creators-vault")
        .iter()
        .map(|a| a.id.clone())
        .collect();
    for level in levels {
        let nodes: Vec<String> = w
            .map()
            .get(&level)
            .unwrap()
            .nodes
            .iter()
            .map(|n| n.id.clone())
            .collect();
        for node in nodes {
            assert!(
                route::distance(
                    w.map(),
                    &Where::new(&level, "core"),
                    &Where::new(&level, &node)
                )
                .is_some(),
                "{level}: no way from the lift to {node}"
            );
        }
    }
}

// =========================================================================
// The shared tables, which take two
// =========================================================================

#[test]
fn two_holders_can_meet_at_a_table_neither_of_them_could_use_alone() {
    // The relations table is a fixture, not a station: it claims nothing and
    // seats several, so both holders can be there at once. Whether the tool
    // requires two is the tool's business; the room's business is letting
    // them stand together, and it does.
    let mut w = vault();
    w.enter("m1", "Maker-01", casting("band-one")).unwrap();
    w.enter("m2", "Maker-02", casting("band-two")).unwrap();
    w.take("m1", Some("cindy")).unwrap();
    w.take("m2", Some("r-okonkwo")).unwrap();

    // Walking to the table lets go of both — which is the honest consequence
    // of "holds it until you leave", and something the settling tool will
    // have to reckon with.
    walk(&mut w, "m1", "relations");
    walk(&mut w, "m2", "relations");
    assert!(w.holder_of("cindy").is_none());

    // You stand *at* a table, never in one — which the map has to say,
    // because English gives no rule for telling a table from a room.
    let p = flat(&w, "m1");
    assert!(p.contains("You are at the relations table."), "{p}");
    assert!(p.contains("Maker-02 is here."), "{p}");
    assert!(within_reach(&w, "m1").contains(&"character.settle_relation"));
}

#[test]
fn each_level_has_its_own_shared_table_within_reach() {
    let mut w = vault();
    for (level, node, tool) in [
        (
            "vault-chronicle",
            "concordance",
            "chronicle.settle_boundary",
        ),
        ("vault-story", "long-table", "story.read_aloud"),
        ("vault-cartography", "road-table", "place.settle_route"),
        ("vault-casting", "relations", "character.settle_relation"),
        ("vault-portraits", "likeness", "portrait.settle_likeness"),
    ] {
        w.enter("m1", "Maker-01", Where::new(level, node)).unwrap();
        let tools = within_reach(&w, "m1");
        assert!(tools.contains(&tool), "{level}/{node}: {tools:?}");
    }
}

// =========================================================================
// The whole crew, moving
// =========================================================================

#[test]
fn a_crew_spread_across_the_building_each_see_only_their_own_level() {
    let mut w = vault();
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
    for (i, (_, _)) in places.iter().enumerate() {
        let p = flat(&w, &format!("m{i}"));
        for (j, (_, other)) in places.iter().enumerate() {
            if i != j {
                assert!(!p.contains(other), "m{i} can see m{j}'s room:\n{p}");
            }
        }
    }
}

#[test]
fn the_percept_stays_small_however_busy_the_room_is() {
    // Sixteen bodies in one room is the worst case, and the percept has to
    // stay a tail rather than becoming the context.
    let mut w = vault();
    let ids = crew(&mut w, 16);
    for id in &ids {
        walk(&mut w, id, "green-room");
    }
    let p = percept(&w, &ids[0]);
    assert!(
        p.split_whitespace().count() < 120,
        "a percept of {} words is not a tail:\n{p}",
        p.split_whitespace().count()
    );
}

#[test]
fn no_percept_ever_tells_a_maker_what_it_ought_to_do() {
    // The same rule as the memory, and harder to hold here: a live percept
    // knows something useful, and that is exactly when the temptation to add
    // "you could take it" is strongest.
    let mut w = vault();
    let ids = crew(&mut w, 3);
    for id in &ids {
        walk(&mut w, id, "band-one");
    }
    w.take(&ids[0], Some("cindy")).unwrap();
    w.say(&ids[0], "somebody should look at the redoubt")
        .unwrap();

    let banned = [
        "you should",
        "you could",
        "you must",
        "why not",
        "consider ",
        "try ",
    ];
    for id in &ids {
        // Speech is quoted verbatim and is a character's business, not the
        // renderer's, so the check is on the frame around it.
        let p = flat(&w, id).to_lowercase();
        let frame = p.split("said:").next().unwrap_or_default().to_string();
        for phrase in banned {
            assert!(!frame.contains(phrase), "{id} was advised: {phrase:?}\n{p}");
        }
    }
}
