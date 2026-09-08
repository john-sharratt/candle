//! The vault the Makers work in, loaded from the map files that ship with it.
//!
//! These assert the building rather than the loader — that it holds together,
//! that it has the shape the design calls for, and that the memory generated
//! from it says the things a Maker has to know. A change to a map file that
//! breaks one of these has changed the building, which is a decision, not an
//! accident.

use npc_map::{describe, AreaKind, Known, MapSet, NodeKind, PartKind};

fn vault() -> MapSet {
    MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps"))
        .expect("the shipped vault must load and validate")
}

/// Generated prose is wrapped, so a sentence a test cares about is usually
/// split across two lines. Assertions run against the flattened text: what is
/// being checked is the wording, and where the line breaks fall is the
/// renderer's business and is covered by its own tests.
fn flat(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Every part in the catalogue, by id.
fn parts(set: &MapSet) -> Vec<&npc_map::Part> {
    let mut ids: Vec<&str> = Vec::new();
    for area in set.areas() {
        for node in &area.nodes {
            for placement in &node.parts {
                if !ids.contains(&placement.part()) {
                    ids.push(placement.part());
                }
            }
        }
    }
    ids.into_iter().filter_map(|id| set.part(id)).collect()
}

#[test]
fn a_part_clause_stands_on_its_own_in_a_sentence() {
    // Several parts stand in one room and their clauses run together, so a
    // `short` that opens with a bare pronoun silently attaches itself to
    // whichever part happened to be listed first, and a fragment reads as the
    // tail of the clause before it. Both faults are invisible in the part's
    // own file and only appear once it is standing next to something else.
    let set = vault();
    for part in parts(&set) {
        let Some(short) = &part.short else { continue };
        let flat = flat(short);
        let first = flat.split_whitespace().next().unwrap_or_default();
        assert!(
            !matches!(first, "It" | "Its" | "They" | "Their" | "These" | "Those"),
            "`{}` opens with `{first}`, which will attach to the wrong part: {flat}",
            part.id
        );
        assert!(
            flat.ends_with('.'),
            "`{}` is not a finished sentence: {flat}",
            part.id
        );
        // A clause with no verb is a fragment. Not a grammar checker — just
        // the handful of shapes these actually take, which is enough to catch
        // "Every settlement, ruin and road, and which of them nobody has
        // written." before it reaches a level.
        let verbs = [
            " is ",
            " are ",
            " was ",
            " were ",
            " can ",
            " lie ",
            " lies ",
            " hang ",
            " hangs ",
            " keeps ",
            " lists ",
            " reports ",
            " runs ",
            " shows ",
            " overlooks ",
            " move ",
            " names ",
        ];
        assert!(
            verbs.iter().any(|v| flat.contains(v)),
            "`{}` has no verb and will read as a fragment: {flat}",
            part.id
        );
    }
}

#[test]
fn the_vault_loads_and_every_join_holds() {
    let set = vault();
    let building = set.get("creators-vault").expect("the building");
    assert_eq!(building.kind, AreaKind::Building);
    assert_eq!(set.children("creators-vault").len(), 6);
}

#[test]
fn the_levels_are_numbered_one_to_six_in_the_order_the_building_names_them() {
    let set = vault();
    let ordinals: Vec<u32> = set
        .children("creators-vault")
        .iter()
        .map(|l| l.ordinal.expect("every level is numbered"))
        .collect();
    assert_eq!(ordinals, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn no_working_level_is_a_bottleneck_for_sixteen_makers() {
    // The rule is that no level can quietly become the constraint: if the
    // standing orders send the whole crew to one level, that level holds
    // them. More than sixteen is fine — the cartography level has a
    // seventeenth station for the map itself, which is held whole and by one
    // Maker, and is a different kind of work from the sixteen beside it.
    let set = vault();
    for level in set.children("creators-vault") {
        // The command level is the one nothing is made on. It does hold posts
        // — the intake, the enquiry desk — but each stands alone, so an order
        // can never queue the crew behind a bank of them. A second desk of the
        // same kind here would be a workroom on the level that is supposed to
        // have none.
        if level.id == "vault-command" {
            for node in &level.nodes {
                for (part, n) in set.parts_of(node, PartKind::Station) {
                    assert_eq!(
                        n, 1,
                        "{}: `{}` stands {n} times, and the command level holds \
                         single posts rather than banks of desks",
                        level.id, part.id
                    );
                }
            }
            continue;
        }
        let work: u32 = level
            .nodes
            .iter()
            .flat_map(|n| set.parts_of(n, PartKind::Station))
            .filter(|(part, _)| part.binds.is_some())
            .map(|(_, n)| n)
            .sum();
        assert!(
            work >= 16,
            "{} affords {work} binding stations, so sixteen Makers cannot all work there",
            level.id
        );
    }
}

#[test]
fn the_map_is_the_one_thing_held_whole_and_by_one_maker() {
    let set = vault();
    let level = set.get("vault-cartography").expect("the cartography level");
    let map = level.node("map-room").expect("the map room");
    assert_eq!(
        set.stations_at(map),
        1,
        "moving a river is not a local edit"
    );
    let table = set.part("map-table").expect("the map table");
    assert_eq!(table.binds.as_deref(), Some("the whole geography"));
}

#[test]
fn every_level_is_entered_at_a_core_that_the_lift_reaches() {
    let set = vault();
    let building = set.get("creators-vault").unwrap();
    for level in set.children("creators-vault") {
        assert_eq!(
            level.of_kind(NodeKind::Core).count(),
            1,
            "{} needs exactly one way in",
            level.id
        );
        let reached = building.portals.iter().any(|p| {
            p.between
                .iter()
                .any(|end| end == &format!("{}/core", level.id))
        });
        assert!(reached, "no portal reaches {}", level.id);
    }
}

#[test]
fn a_station_that_binds_something_says_what() {
    let set = vault();
    for (id, part) in [
        ("chronicle-terminal", "one era"),
        ("story-desk", "one gap in the record"),
        ("survey-desk", "one place"),
        ("map-table", "the whole geography"),
        ("character-terminal", "one character"),
        ("easel", "one character"),
    ] {
        let def = set.part(id).unwrap_or_else(|| panic!("no part `{id}`"));
        assert_eq!(def.binds.as_deref(), Some(part), "{id}");
    }
    // The watch is the deliberate exception: a station that holds nothing,
    // because it writes nothing.
    assert!(set.part("watch-desk").unwrap().binds.is_none());
}

#[test]
fn every_station_carries_the_tools_it_makes_reachable() {
    // The whole point of parts: what an NPC can do is a function of what its
    // body is next to. A station with no tools is a seat.
    let set = vault();
    for level in set.children("creators-vault") {
        for node in &level.nodes {
            for (part, _) in set.parts_of(node, PartKind::Station) {
                assert!(
                    !part.tools.is_empty(),
                    "{}/{} places `{}`, which affords nothing",
                    level.id,
                    node.id,
                    part.id
                );
            }
        }
    }
}

#[test]
fn the_tool_surface_is_read_off_the_room_a_body_is_in() {
    let set = vault();
    let casting = set.get("vault-casting").expect("the casting level");

    let band = casting.node("band-one").expect("band one");
    let at_a_station = set.tools_at(band);
    assert!(
        at_a_station.contains(&"character.write_beliefs"),
        "{at_a_station:?}"
    );

    // Step into the corridor and the same tool is gone, because the terminal
    // is not within reach any more.
    let corridor = casting.node("ring-north").expect("the north run");
    assert!(set.tools_at(corridor).is_empty());

    // And a tool that belongs to another level is never reachable here.
    assert!(
        !at_a_station.contains(&"chronicle.rewrite_page"),
        "{at_a_station:?}"
    );
}

#[test]
fn every_level_has_somewhere_to_sit_that_is_not_a_workstation() {
    let set = vault();
    for level in set.children("creators-vault") {
        let seats: u32 = level
            .of_kind(NodeKind::Social)
            .flat_map(|n| set.parts_of(n, PartKind::Seat))
            .map(|(_, n)| n)
            .sum();
        assert!(
            seats >= 4,
            "{} has {seats} seats away from the work",
            level.id
        );
    }
}

#[test]
fn the_building_memory_names_every_level_a_maker_knows() {
    let set = vault();
    let text = flat(&describe::building(&set, "creators-vault", &Known::All));
    for level in set.children("creators-vault") {
        assert!(
            text.contains(&level.name),
            "the building memory never mentions {}:\n{text}",
            level.name
        );
    }
    assert!(text.contains("six levels"), "{text}");
}

#[test]
fn a_maker_that_knows_only_two_levels_is_told_only_those() {
    let set = vault();
    let known = Known::Only(
        ["vault-command".to_string(), "vault-casting".to_string()]
            .into_iter()
            .collect(),
    );
    let text = flat(&describe::building(&set, "creators-vault", &known));
    assert!(text.contains("the casting level"), "{text}");
    assert!(!text.contains("the chronicle"), "{text}");
    assert!(text.contains("two levels"), "{text}");
}

#[test]
fn a_level_memory_gives_the_shape_of_the_route_in_one_sentence() {
    let set = vault();
    let text = flat(&describe::level(&set, "vault-casting"));
    assert!(
        text.contains(
            "Seven rooms open off the ring, which runs right round the level, \
             and the lift and the stair open onto it."
        ),
        "{text}"
    );
}

#[test]
fn no_level_memory_names_a_corridor() {
    // Corridors are plumbing. Naming them made a reader hold six abstract
    // names just to work out where a room was, and nobody ever wants to go to
    // one — which door leads where is perception's job, at the moment
    // somebody is actually walking.
    let set = vault();
    for level in set.children("creators-vault") {
        let text = describe::level(&set, &level.id);
        for passage in level.of_kind(NodeKind::Passage) {
            assert!(
                !text.contains(&passage.name),
                "{} names the corridor {:?}:\n{text}",
                level.id,
                passage.name
            );
        }
    }
}

#[test]
fn a_level_memory_groups_its_rooms_by_what_they_are_for() {
    let set = vault();
    let text = flat(&describe::level(&set, "vault-casting"));
    assert!(text.contains("The work here:"), "{text}");
    assert!(text.contains("To consult:"), "{text}");
    assert!(text.contains("For company:"), "{text}");
}

#[test]
fn identical_work_rooms_are_described_once_between_them() {
    let set = vault();
    let text = flat(&describe::level(&set, "vault-casting"));
    assert!(
        text.contains(
            "band one, band two and band three — sixteen character terminals between them."
        ),
        "{text}"
    );
    assert!(
        text.contains("Each takes one character and holds it until you leave."),
        "{text}"
    );
}

#[test]
fn a_level_memory_says_what_can_be_done_where() {
    let set = vault();
    let text = flat(&describe::level(&set, "vault-casting"));
    // The clause comes off the part, so it says the same thing in every room
    // that has one — and it says it once even where three rooms have them.
    assert!(
        text.contains("Who a character is, what they want, what they hold true"),
        "{text}"
    );
    assert!(
        text.contains("What stands between two characters is settled at it"),
        "{text}"
    );
}

#[test]
fn sight_is_stated_as_a_rule_and_only_the_exceptions_are_listed() {
    let set = vault();
    let casting = flat(&describe::level(&set, "vault-casting"));
    assert!(
        casting.contains("You can see into a room from the corridor it opens off"),
        "{casting}"
    );
    // Every sightline on the casting level is a doorway, so nothing else is
    // worth a reader's attention and nothing else is said.
    assert!(!casting.contains("and between"), "{casting}");

    // The cartography level has the one sightline that is not a door.
    let carto = flat(&describe::level(&set, "vault-cartography"));
    // The gallery's rail overlooks the map room without reaching it, and the
    // Maker holding the map can see who is watching — so it is stated as one
    // fact between the two rather than twice, once each way.
    assert!(
        carto.contains("and between the map room and the survey gallery"),
        "{carto}"
    );
}

#[test]
fn the_building_memory_indexes_what_is_taken_up_where() {
    // This is what lets a Maker route a job to a level without being told
    // which level. It is generated from `binds` alone.
    let set = vault();
    let text = flat(&describe::building(&set, "creators-vault", &Known::All));
    assert!(
        text.contains("What a station takes up, and where:"),
        "{text}"
    );
    assert!(
        text.contains("One character, at sixteen stations on the casting level"),
        "{text}"
    );
    assert!(
        text.contains("The whole geography, at one station"),
        "{text}"
    );
}

#[test]
fn a_level_memory_says_what_the_level_has_not_got() {
    let set = vault();
    let text = flat(&describe::level(&set, "vault-story"));
    assert!(
        text.contains("There is no way off this level except the lift and the stair."),
        "{text}"
    );
}

#[test]
fn the_same_map_describes_the_same_way_every_time() {
    let set = vault();
    let once = describe::level(&set, "vault-chronicle");
    let twice = describe::level(&set, "vault-chronicle");
    assert_eq!(
        once, twice,
        "description must be a pure function of the map"
    );
}

#[test]
fn no_memory_tells_a_maker_what_it_ought_to_do() {
    // A map describes; it never instructs. This is worth a test because the
    // failure is invisible at runtime: one instructional line here is cached,
    // shared by every Maker, and shapes all of them identically for ever
    // without appearing in any log.
    let set = vault();
    let banned = [
        "you should",
        "you must",
        "make sure",
        "remember to",
        "be sure to",
        "try to",
    ];
    for level in set.children("creators-vault") {
        let text = describe::level(&set, &level.id).to_lowercase();
        for phrase in banned {
            assert!(
                !text.contains(phrase),
                "{} instructs rather than describes: {phrase:?}",
                level.id
            );
        }
    }
}
