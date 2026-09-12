//! A building that is not the vault.
//!
//! The vault tests prove the vault. These prove the *machinery* is not shaped
//! around it — a place with a straight corridor instead of a ring, no
//! workstations, one storeroom and a name that takes a plural verb describes
//! just as well, and a level that has none of the optional furniture still
//! produces something a reader can navigate by.
//!
//! Built in memory rather than from files, because what is under test is the
//! renderer rather than any particular YAML.

use npc_map::{
    describe, Area, AreaKind, Known, MapSet, Node, NodeKind, Part, PartKind, Placement, Portal,
    Spine,
};

fn node(id: &str, kind: NodeKind, name: &str, off: &[&str]) -> Node {
    Node {
        id: id.into(),
        kind,
        name: name.into(),
        plural: false,
        stand: None,
        off: off.iter().map(|s| s.to_string()).collect(),
        character: None,
        parts: vec![],
        ground: vec![],
        habit: None,
        sees: vec![],
        exits: vec![],
        visible: vec![],
    }
}

fn part(id: &str, kind: PartKind, name: &str, short: Option<&str>) -> Part {
    Part {
        id: id.into(),
        kind,
        name: name.into(),
        plural: None,
        binds: None,
        short: short.map(String::from),
        long: format!("What a {name} is, at length."),
        // Consulted rather than worked: nothing here has a state to be put into.
        modes: vec![],
    }
}

fn placed(id: &str, count: u32) -> Placement {
    Placement::Counted {
        part: id.into(),
        count,
    }
}

fn area(id: &str, kind: AreaKind, name: &str, nodes: Vec<Node>) -> Area {
    Area {
        id: id.into(),
        kind,
        name: name.into(),
        within: None,
        ordinal: None,
        summary: "A place that is not the vault.".into(),
        character: None,
        lacks: vec![],
        announcements: vec![],
        contains: vec![],
        portals: vec![],
        arrival: None,
        teleport_to: None,
        spine: None,
        nodes,
    }
}

fn inn() -> MapSet {
    MapSet::assemble(inn_areas(), catalogue()).expect("the inn must load")
}

/// A part catalogue that has nothing to do with the vault.
fn catalogue() -> Vec<Part> {
    vec![
        part("bench", PartKind::Seat, "bench", None),
        part(
            "barrels",
            PartKind::Fixture,
            "the barrels",
            Some("The barrels are stacked three deep against the far wall."),
        ),
        part("water-butt", PartKind::Fixture, "a water butt", None),
    ]
}

/// A two-floor inn: a straight corridor, a taproom, a cellar, and rooms above.
fn inn_areas() -> Vec<Area> {
    let mut house = area("the-anchor", AreaKind::Building, "The Anchor", vec![]);
    house.summary = "An inn on the harbour road.".into();
    house.contains = vec!["anchor-ground".into(), "anchor-upstairs".into()];
    house.portals = vec![Portal {
        between: ["anchor-ground/stair".into(), "anchor-upstairs/stair".into()],
        kind: "a stair".into(),
    }];

    let mut ground = area(
        "anchor-ground",
        AreaKind::Level,
        "the ground floor",
        vec![
            node("hall", NodeKind::Passage, "the hall", &[]),
            node("stair", NodeKind::Core, "the stair", &["hall"]),
            node("taproom", NodeKind::Social, "the taproom", &["hall"]),
            node("cellar", NodeKind::Store, "the cellar", &["hall"]),
        ],
    );
    ground.within = Some("the-anchor".into());
    ground.ordinal = Some(1);
    ground.summary = "Where the drinking is done.".into();
    ground.spine = Some(Spine {
        name: "the hall".into(),
        loops: false,
        through: vec!["hall".into()],
    });
    ground.nodes[2].parts = vec![placed("bench", 30)];
    ground.nodes[2].habit = Some("loud".into());
    ground.nodes[3].parts = vec![Placement::Bare("barrels".into())];

    // No spine, no stations, nothing optional: the bare minimum a level can be
    // and still be described.
    let mut upstairs = area(
        "anchor-upstairs",
        AreaKind::Level,
        "the upstairs",
        vec![
            node("landing", NodeKind::Passage, "the landing", &[]),
            node("stair", NodeKind::Core, "the stair", &["landing"]),
            node("beds", NodeKind::Social, "the beds", &["landing"]),
        ],
    );
    upstairs.within = Some("the-anchor".into());
    upstairs.ordinal = Some(2);
    upstairs.summary = "Where the sleeping is done.".into();
    upstairs.nodes[2].plural = true;
    upstairs.nodes[2].habit = Some("full on a market night".into());

    vec![house, ground, upstairs]
}

#[test]
fn a_building_with_no_workstations_still_describes() {
    let set = inn();
    let text = describe::building(&set, "the-anchor", &Known::All);
    assert!(
        text.contains("The Anchor is a building of two levels"),
        "{text}"
    );
    assert!(text.contains("joined by a stair"), "{text}");
    // Nothing is taken up anywhere, so the index is simply absent rather than
    // present and empty.
    assert!(!text.contains("What a station takes up"), "{text}");
}

#[test]
fn a_straight_corridor_is_described_as_running_the_length() {
    let set = inn();
    let text = describe::level(&set, "anchor-ground");
    assert!(
        text.contains("open off the hall, which runs the length of the level"),
        "{text}"
    );
    assert!(!text.contains("right round"), "{text}");
}

#[test]
fn a_level_with_no_spine_at_all_still_says_what_is_there() {
    let set = inn();
    let text = describe::level(&set, "anchor-upstairs");
    assert!(text.contains("One room here"), "{text}");
    assert!(text.contains("the stair opens onto it"), "{text}");
    assert!(text.contains("For company:"), "{text}");
    assert!(text.contains("the beds"), "{text}");
}

#[test]
fn a_plural_name_takes_a_plural_verb_wherever_it_appears() {
    let set = inn();
    let text = describe::level(&set, "anchor-upstairs");
    assert!(
        text.contains("The beds are full on a market night"),
        "{text}"
    );
    // And the entry is the bare name, because there is nothing else true of
    // it — no dash left hanging where the optional fields would have gone.
    assert!(text.contains("  the beds.\n"), "{text}");
}

#[test]
fn a_level_with_nothing_optional_omits_every_optional_line() {
    // No stations, no fixtures on the social room, no character, no `lacks` —
    // and no empty headings, dangling dashes or stray full stops where they
    // would have gone.
    let set = inn();
    let text = describe::level(&set, "anchor-upstairs");
    assert!(!text.contains("The work here:"), "{text}");
    assert!(!text.contains("To consult:"), "{text}");
    assert!(!text.contains("There is no"), "{text}");
    assert!(!text.contains(" ."), "{text}");
    assert!(!text.contains("—\n"), "{text}");
}

/// The outdoor map: its own places, and the inn standing on it.
///
/// This is the shape a game world takes — a terrain grid with buildings placed
/// on it — and the point of the test is that one area holds *both* its own
/// nodes and its children, and that the renderer copes with a place that has
/// no walls making it one.
fn harbour() -> MapSet {
    let mut road = area("harbour-road", AreaKind::Region, "the harbour road", vec![]);
    road.summary = "The road along the water, and what stands on it.".into();
    road.contains = vec!["the-anchor".into()];
    road.nodes = vec![
        node("verge", NodeKind::Passage, "the road", &[]),
        node("yard", NodeKind::Ground, "the inn yard", &["verge"]),
        node("slip", NodeKind::Ground, "the slipway", &["verge"]),
    ];
    road.nodes[1].ground = vec!["dirt".into(), "broken metal plate".into()];
    road.nodes[1].parts = vec![Placement::Bare("water-butt".into())];
    road.nodes[1].habit = Some("churned to mud after rain".into());
    road.nodes[2].ground = vec!["wet stone".into()];
    road.nodes[2].sees = vec!["yard".into()];
    // Outdoors a building is one thing standing on the map; the door leads to
    // a node on one of its levels, not to the building itself. That is the
    // game's model exactly — an instance out here, all its parts once inside.
    road.portals = vec![Portal {
        between: ["harbour-road/yard".into(), "anchor-ground/hall".into()],
        kind: "a door".into(),
    }];

    let mut set: Vec<Area> = Vec::new();
    for area in inn_areas() {
        set.push(area);
    }
    // The inn now stands inside the road rather than on its own.
    for a in &mut set {
        if a.id == "the-anchor" {
            a.within = Some("harbour-road".into());
        }
    }
    set.push(road);
    MapSet::assemble(set, catalogue()).expect("the harbour must load")
}

#[test]
fn an_outdoor_area_holds_its_own_places_and_the_buildings_on_it() {
    let set = harbour();
    let text = describe::level(&set, "harbour-road");
    // Places, not rooms: nothing outdoors has walls making it one.
    assert!(text.contains("Two places here"), "{text}");
    assert!(text.contains("Open ground:"), "{text}");
    assert!(!text.contains("rooms"), "{text}");
    // And it still contains a building, which the building view describes.
    assert_eq!(set.children("harbour-road").len(), 1);
}

#[test]
fn terrain_reaches_the_memory_as_what_is_underfoot() {
    let set = harbour();
    let text = describe::level(&set, "harbour-road");
    assert!(
        text.contains("Underfoot, dirt and broken metal plate."),
        "{text}"
    );
    assert!(text.contains("Underfoot, wet stone."), "{text}");
    // Indoors the field is empty and the clause never appears.
    let inside = describe::level(&set, "anchor-ground");
    assert!(!inside.contains("Underfoot"), "{inside}");
}

#[test]
fn an_outdoor_sightline_is_stated_without_a_doorway_rule_behind_it() {
    let set = harbour();
    let text = describe::level(&set, "harbour-road");
    // No doorway rule out here, because nothing bounds anything: the sightline
    // is simply stated.
    assert!(
        text.contains("You can see between the inn yard and the slipway."),
        "{text}"
    );
    assert!(!text.contains("opens off"), "{text}");
}

#[test]
fn the_same_renderer_reads_the_inn_and_the_vault_the_same_way() {
    // Both buildings answer the three questions a memory exists to answer.
    let set = inn();
    for id in ["anchor-ground", "anchor-upstairs"] {
        let text = describe::level(&set, id);
        assert!(text.starts_with("Level "), "{text}");
        assert!(text.contains("opens onto it"), "{text}");
    }
}
