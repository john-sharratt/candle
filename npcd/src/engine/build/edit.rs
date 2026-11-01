//! Each world-building act as the map edits it makes (`docs/zen_moon_mind.md`
//! §14) — checked against the map as it stands, before anything changes. Pure:
//! the map and the room in, the edits or the refusal out.

use npc_map::part::{Part, PartKind, PartLink, Placement};
use npc_map::schema::{Node, NodeKind, Where};
use npc_map::{MapEdit, MapSet};

use crate::bridge::finding::{Finding, ONBOARDED};

/// A name as an id: lower case, its words joined by hyphens —
/// `the kitchen` is `kitchen`, `Yen's phone` is `yens-phone`.
pub fn slug(name: &str) -> String {
    let words: Vec<String> = name
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '\'')
        .map(|w| w.replace('\'', "").to_ascii_lowercase())
        .filter(|w| !w.is_empty() && w != "the")
        .collect();
    words.join("-")
}

/// Whether `room` is a node of the mirror, where the world may build.
fn mirrored(map: &MapSet, room: &Where) -> Result<(), String> {
    match map.is_mirrored(&room.area)
        && map
            .get(&room.area)
            .and_then(|a| a.node(&room.node))
            .is_some()
    {
        true => Ok(()),
        false => Err(
            "This room is not a mirror of a real place: nothing here is yours to change.".into(),
        ),
    }
}

/// The names of the parts standing in `room`, with their ids.
fn parts_in(map: &MapSet, room: &Where) -> Vec<(String, String)> {
    let Some(node) = map.get(&room.area).and_then(|a| a.node(&room.node)) else {
        return Vec::new();
    };
    map.parts_at(node)
        .map(|(p, _)| (p.name.clone(), p.id.clone()))
        .collect()
}

/// The part anywhere in the catalogue that stands for `finding`'s device.
fn standing_for<'a>(map: &'a MapSet, finding: &Finding) -> Option<&'a Part> {
    map.areas()
        .flat_map(|a| a.nodes.iter())
        .flat_map(|n| map.parts_at(n))
        .map(|(p, _)| p)
        .find(|p| {
            p.link
                .as_ref()
                .is_some_and(|l| l.hardware == finding.hardware)
        })
}

fn link_of(finding: &Finding) -> PartLink {
    PartLink {
        profile: finding.profile.clone(),
        address: finding.address.clone(),
        hardware: finding.hardware.clone(),
        ..Default::default()
    }
}

/// Refused when `room` already holds a part called `name`.
fn named_once(map: &MapSet, room: &Where, name: &str) -> Result<(), String> {
    match parts_in(map, room)
        .iter()
        .any(|(n, _)| n.eq_ignore_ascii_case(name.trim()))
    {
        true => Err(format!(
            "This room already has a {name}: give it a name of its own."
        )),
        false => Ok(()),
    }
}

/// `onboard_part`: the device `finding` found, defined as a part of the room
/// that stands for it, and placed there.
pub fn onboard_part(
    map: &MapSet,
    room: &Where,
    finding: &Finding,
    name: &str,
    description: &str,
) -> Result<Vec<MapEdit>, String> {
    mirrored(map, room)?;
    if let Some(p) = standing_for(map, finding) {
        return Err(format!(
            "{} is already the {} — it is in the world.",
            finding.label(),
            p.name
        ));
    }
    let id = finding.part_id();
    // **Offboarded once, onboarded again**: the device keeps the part it had,
    // placed back and linked at the address it has now.
    if map.part(&id).is_some() {
        return Ok(vec![
            MapEdit::LinkPart {
                part: id.clone(),
                link: link_of(finding),
            },
            MapEdit::PlacePart {
                at: room.clone(),
                placement: Placement::Bare(id),
            },
        ]);
    }
    named_once(map, room, name)?;
    let part = Part {
        id: id.clone(),
        kind: PartKind::Fixture,
        name: name.trim().to_string(),
        plural: None,
        binds: None,
        short: Some(description.trim().to_string()),
        long: finding.line(),
        modes: vec![],
        link: Some(link_of(finding)),
    };
    Ok(vec![
        MapEdit::DefinePart(Box::new(part)),
        MapEdit::PlacePart {
            at: room.clone(),
            placement: Placement::Bare(id),
        },
    ])
}

/// `onboard_thing`: a thing really in the room, on no network, as a part of
/// it.
pub fn onboard_thing(
    map: &MapSet,
    room: &Where,
    name: &str,
    description: &str,
) -> Result<Vec<MapEdit>, String> {
    mirrored(map, room)?;
    named_once(map, room, name)?;
    let id = format!("thing-{}-{}", room.node, slug(name));
    if map.part(&id).is_some() {
        return Err(format!("There is a {name} here already."));
    }
    let part = Part {
        id: id.clone(),
        kind: PartKind::Fixture,
        name: name.trim().to_string(),
        plural: None,
        binds: None,
        short: Some(description.trim().to_string()),
        long: description.trim().to_string(),
        modes: vec![],
        link: None,
    };
    Ok(vec![
        MapEdit::DefinePart(Box::new(part)),
        MapEdit::PlacePart {
            at: room.clone(),
            placement: Placement::Bare(id),
        },
    ])
}

/// Whether part `id` may be linked to a device: one with no link yet, placed
/// once in the whole map — a link is the part's, so a part placed in several
/// rooms would make every one of them the device.
pub fn linkable(map: &MapSet, id: &str) -> bool {
    let placed = map
        .areas()
        .flat_map(|a| a.nodes.iter())
        .flat_map(|n| n.parts.iter())
        .filter(|p| p.part() == id)
        .count();
    placed == 1 && map.part(id).is_some_and(|p| p.link.is_none())
}

/// `link_part`: the part called `name` in the room is the device `finding`
/// found.
pub fn link_part(
    map: &MapSet,
    room: &Where,
    name: &str,
    finding: &Finding,
) -> Result<Vec<MapEdit>, String> {
    mirrored(map, room)?;
    let Some((_, id)) = parts_in(map, room)
        .into_iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name.trim()))
    else {
        return Err(format!("There is no {name} in this room."));
    };
    if map.part(&id).and_then(|p| p.link.as_ref()).is_some() {
        return Err(format!("The {name} already stands for a device."));
    }
    if !linkable(map, &id) {
        return Err(format!(
            "The {name} is a kind of thing that stands in other rooms too: it cannot be one \
             device. Onboard the device as a part of its own."
        ));
    }
    if let Some(p) = standing_for(map, finding) {
        return Err(format!("{} is already the {}.", finding.label(), p.name));
    }
    Ok(vec![MapEdit::LinkPart {
        part: id,
        link: link_of(finding),
    }])
}

/// The links that follow devices `found` at a new address: each placed part
/// standing for a found device's hardware at another address, linked where
/// it is now.
pub fn moved(map: &MapSet, found: &[Finding]) -> Vec<MapEdit> {
    let mut out: Vec<MapEdit> = Vec::new();
    for (part, _) in map
        .areas()
        .flat_map(|a| a.nodes.iter())
        .flat_map(|n| map.parts_at(n))
    {
        let Some(link) = &part.link else {
            continue;
        };
        let Some(f) = found
            .iter()
            .find(|f| f.hardware == link.hardware && f.address != link.address)
        else {
            continue;
        };
        let already = out
            .iter()
            .any(|e| matches!(e, MapEdit::LinkPart { part: p, .. } if *p == part.id));
        if !already {
            out.push(MapEdit::LinkPart {
                part: part.id.clone(),
                link: PartLink {
                    address: f.address.clone(),
                    ..link.clone()
                },
            });
        }
    }
    out
}

/// Whether part `id` is one a survey onboarded — a device the world's
/// builders put in, and so may take out.
pub fn onboarded(id: &str) -> bool {
    id.starts_with(ONBOARDED)
}

/// `offboard_part`: the part called `name`, a device onboarded from a survey,
/// taken out of the room. A part of the room as it was authored is never taken
/// out, linked or not.
pub fn offboard_part(map: &MapSet, room: &Where, name: &str) -> Result<Vec<MapEdit>, String> {
    mirrored(map, room)?;
    let Some((_, id)) = parts_in(map, room)
        .into_iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name.trim()))
    else {
        return Err(format!("There is no {name} in this room."));
    };
    if !onboarded(&id) {
        return Err(format!(
            "The {name} was not onboarded from a survey: it is part of the room as it was \
             authored."
        ));
    }
    Ok(vec![MapEdit::UnplacePart {
        at: room.clone(),
        part: id,
    }])
}

/// `discover_door`: a room of the house opening off this one — in this
/// room's own area, never a new one.
pub fn discover_door(
    map: &MapSet,
    room: &Where,
    name: &str,
    description: &str,
) -> Result<Vec<MapEdit>, String> {
    mirrored(map, room)?;
    let id = slug(name);
    if id.is_empty() {
        return Err("Name the room.".into());
    }
    let area = map.get(&room.area).expect("checked mirrored");
    if area.node(&id).is_some()
        || area
            .nodes
            .iter()
            .any(|n| n.name.eq_ignore_ascii_case(name.trim()))
    {
        return Err(format!("This house already has {name}."));
    }
    let name = name.trim();
    let name = match name.to_ascii_lowercase().starts_with("the ") {
        true => name.to_string(),
        false => format!("the {name}"),
    };
    Ok(vec![MapEdit::AddNode {
        area: room.area.clone(),
        node: Box::new(Node {
            id,
            kind: NodeKind::Social,
            name,
            plural: false,
            stand: None,
            off: vec![room.node.clone()],
            character: Some(description.trim().to_string()),
            parts: vec![],
            ground: vec![],
            habit: None,
            sees: vec![],
            exits: vec![],
            visible: vec![],
        }),
    }])
}

/// The rooms off `room` that may be undiscovered: in its area, hanging off it,
/// and discovered by the world's builders — an edit in `built` added it. A room
/// as it was authored is never taken away, any more than an authored part.
pub fn doors(map: &MapSet, room: &Where, built: &[MapEdit]) -> Vec<(String, Where)> {
    let discovered = |id: &str| {
        built.iter().any(
            |e| matches!(e, MapEdit::AddNode { area, node } if *area == room.area && node.id == id),
        )
    };
    if mirrored(map, room).is_err() {
        return Vec::new();
    }
    let Some(area) = map.get(&room.area) else {
        return Vec::new();
    };
    let Some(here) = area.node(&room.node) else {
        return Vec::new();
    };
    here.exits
        .iter()
        .filter_map(|id| area.node(id))
        .filter(|n| n.off.contains(&room.node) && discovered(&n.id))
        .filter(|n| n.kind != NodeKind::Passage && n.kind != NodeKind::Core)
        .map(|n| (n.name.clone(), Where::new(room.area.clone(), n.id.clone())))
        .collect()
}

/// `undiscover_door`: the room called `name` off this one, which the world's
/// builders discovered (`built`), taken away.
pub fn undiscover_door(
    map: &MapSet,
    room: &Where,
    name: &str,
    built: &[MapEdit],
) -> Result<Vec<MapEdit>, String> {
    let Some((_, at)) = doors(map, room, built)
        .into_iter()
        .find(|(n, _)| n.eq_ignore_ascii_case(name.trim()))
    else {
        return Err(format!("No room called {name} opens off this one."));
    };
    Ok(vec![MapEdit::RemoveNode(at)])
}

/// `describe_room`: what this room is like, rewritten.
pub fn describe_room(
    map: &MapSet,
    room: &Where,
    description: &str,
) -> Result<Vec<MapEdit>, String> {
    mirrored(map, room)?;
    Ok(vec![MapEdit::Describe {
        at: room.clone(),
        character: description.trim().to_string(),
    }])
}

#[cfg(test)]
mod tests {
    use npc_map::schema::{Area, AreaKind};

    use super::*;

    fn node(id: &str, kind: NodeKind, name: &str, off: &[&str], parts: &[&str]) -> Node {
        Node {
            id: id.into(),
            kind,
            name: name.into(),
            plural: false,
            stand: None,
            off: off.iter().map(|s| s.to_string()).collect(),
            character: None,
            parts: parts
                .iter()
                .map(|p| Placement::Bare(p.to_string()))
                .collect(),
            ground: vec![],
            habit: None,
            sees: vec![],
            exits: vec![],
            visible: vec![],
        }
    }

    fn area(
        id: &str,
        kind: AreaKind,
        within: Option<&str>,
        mirrored: bool,
        nodes: Vec<Node>,
    ) -> Area {
        Area {
            id: id.into(),
            kind,
            name: id.into(),
            within: within.map(String::from),
            ordinal: None,
            summary: "a place".into(),
            character: None,
            lacks: vec![],
            announcements: vec![],
            contains: vec![],
            portals: vec![],
            arrival: None,
            teleport_to: None,
            spine: None,
            mirrored,
            nodes,
        }
    }

    fn fixture(id: &str, name: &str) -> Part {
        Part {
            id: id.into(),
            kind: PartKind::Fixture,
            name: name.into(),
            plural: None,
            binds: None,
            short: None,
            long: "x".into(),
            modes: vec![],
            link: None,
        }
    }

    /// A mirrored home with a front room holding a Chromecast, and a moon.
    fn world() -> MapSet {
        let mut home = area("home", AreaKind::Building, None, true, vec![]);
        home.contains = vec!["flat".into()];
        let flat = area(
            "flat",
            AreaKind::Level,
            Some("home"),
            false,
            vec![
                node(
                    "front-room",
                    NodeKind::Social,
                    "the front room",
                    &["hall"],
                    &["cast"],
                ),
                node("hall", NodeKind::Passage, "the hall", &[], &[]),
            ],
        );
        let moon = area(
            "moon",
            AreaKind::Level,
            None,
            false,
            vec![node("watch", NodeKind::Social, "the watch room", &[], &[])],
        );
        MapSet::assemble([home, flat, moon], [fixture("cast", "Chromecast")]).unwrap()
    }

    fn room() -> Where {
        Where::new("flat", "front-room")
    }

    fn sonos() -> Finding {
        Finding {
            address: "192.168.0.89".into(),
            hardware: "f0:f6:c1:a0:79:b4".into(),
            names: vec!["SonosZP".into()],
            maker: "Sonos, Inc.".into(),
            profile: "sonos".into(),
            ..Finding::default()
        }
    }

    fn apply(map: &MapSet, edits: Vec<MapEdit>) -> MapSet {
        edits
            .into_iter()
            .fold(map.clone(), |m, e| m.apply(&e).unwrap())
    }

    #[test]
    fn a_name_is_slugged_without_its_article() {
        assert_eq!(slug("the kitchen"), "kitchen");
        assert_eq!(slug("Yen's phone"), "yens-phone");
        assert_eq!(slug("  The  Main Bedroom "), "main-bedroom");
    }

    /// **A device found is onboarded once, as a part linked to it**; the same
    /// device again, a name the room already has, and a room outside the
    /// mirror are each refused.
    #[test]
    fn a_device_is_onboarded_once_and_only_in_the_mirror() {
        let map = world();
        let edits =
            onboard_part(&map, &room(), &sonos(), "Sonos speaker", "The Sonos plays.").unwrap();
        let built = apply(&map, edits);
        let part = built.part("device-f0f6c1a079b4").unwrap();
        assert_eq!(part.name, "Sonos speaker");
        assert_eq!(part.short.as_deref(), Some("The Sonos plays."));
        assert_eq!(part.link.as_ref().unwrap().address, "192.168.0.89");
        let front = built.get("flat").unwrap().node("front-room").unwrap();
        assert!(front
            .parts
            .iter()
            .any(|p| p.part() == "device-f0f6c1a079b4"));

        assert_eq!(
            onboard_part(&built, &room(), &sonos(), "speaker", "x").unwrap_err(),
            "SonosZP at 192.168.0.89 is already the Sonos speaker — it is in the world."
        );
        let other = Finding {
            hardware: "aa:bb:cc:dd:ee:ff".into(),
            ..sonos()
        };
        assert!(onboard_part(&built, &room(), &other, "chromecast", "x")
            .unwrap_err()
            .starts_with("This room already has a chromecast"));
        assert!(
            onboard_part(&map, &Where::new("moon", "watch"), &sonos(), "x", "y")
                .unwrap_err()
                .starts_with("This room is not a mirror")
        );
    }

    /// **A part in the room is linked to the device it is**, once; and only a
    /// device onboarded from a survey is offboarded — an authored part never,
    /// linked or not.
    #[test]
    fn a_part_is_linked_once_and_only_an_onboarded_device_offboarded() {
        let map = world();
        let cast = Finding {
            address: "192.168.0.214".into(),
            hardware: "14:ae:85:71:eb:44".into(),
            profile: "device".into(),
            ..Finding::default()
        };
        let authored =
            "The Chromecast was not onboarded from a survey: it is part of the room as it was \
             authored.";
        assert_eq!(
            offboard_part(&map, &room(), "Chromecast").unwrap_err(),
            authored
        );
        let linked = apply(&map, link_part(&map, &room(), "chromecast", &cast).unwrap());
        assert_eq!(
            linked.part("cast").unwrap().link.as_ref().unwrap().hardware,
            "14:ae:85:71:eb:44"
        );
        assert!(link_part(&linked, &room(), "Chromecast", &sonos()).is_err());
        assert!(
            onboard_part(&linked, &room(), &cast, "cast", "x").is_err(),
            "already linked"
        );
        assert_eq!(
            offboard_part(&linked, &room(), "Chromecast").unwrap_err(),
            authored,
            "linked, still authored"
        );
        let speaker = apply(
            &linked,
            onboard_part(&linked, &room(), &sonos(), "Sonos speaker", "x").unwrap(),
        );
        let gone = apply(
            &speaker,
            offboard_part(&speaker, &room(), "Sonos speaker").unwrap(),
        );
        let front = gone.get("flat").unwrap().node("front-room").unwrap();
        assert_eq!(front.parts.len(), 1, "the Chromecast stays");
        assert!(link_part(&map, &room(), "sofa", &cast).is_err());
    }

    /// **A part standing in other rooms too cannot be one device**: linking it
    /// would make every one of them the device.
    #[test]
    fn a_part_placed_in_several_rooms_is_not_linked() {
        let (mut areas, mut parts) = world().authored();
        parts.push(fixture("seat", "seat"));
        for a in areas.iter_mut() {
            for n in a
                .nodes
                .iter_mut()
                .filter(|n| n.id == "front-room" || n.id == "watch")
            {
                n.parts.push(Placement::Bare("seat".into()));
            }
        }
        let map = MapSet::assemble(areas, parts).unwrap();
        assert!(!linkable(&map, "seat"));
        assert!(linkable(&map, "cast"));
        assert!(link_part(&map, &room(), "seat", &sonos())
            .unwrap_err()
            .starts_with("The seat is a kind of thing"));
    }

    /// **A device found at a new address is followed there**: its part's link
    /// moves, and a device where it was is left alone.
    #[test]
    fn a_device_that_moved_is_linked_where_it_is() {
        let map = world();
        let built = apply(
            &map,
            onboard_part(&map, &room(), &sonos(), "Sonos speaker", "x").unwrap(),
        );
        assert_eq!(moved(&built, &[sonos()]), [], "where it was");
        let there = Finding {
            address: "192.168.0.95".into(),
            ..sonos()
        };
        let edits = moved(&built, &[there]);
        assert_eq!(edits.len(), 1);
        let after = apply(&built, edits);
        assert_eq!(
            after
                .part("device-f0f6c1a079b4")
                .unwrap()
                .link
                .as_ref()
                .unwrap()
                .address,
            "192.168.0.95"
        );
    }

    /// **A device offboarded is onboarded again** as the part it was, linked
    /// at the address it has now.
    #[test]
    fn an_offboarded_device_is_onboarded_again() {
        let map = world();
        let built = apply(
            &map,
            onboard_part(&map, &room(), &sonos(), "Sonos speaker", "The Sonos plays.").unwrap(),
        );
        let gone = apply(
            &built,
            offboard_part(&built, &room(), "Sonos speaker").unwrap(),
        );
        let moved = Finding {
            address: "192.168.0.90".into(),
            ..sonos()
        };
        let back = apply(
            &gone,
            onboard_part(&gone, &room(), &moved, "speaker", "x").unwrap(),
        );
        let part = back.part("device-f0f6c1a079b4").unwrap();
        assert_eq!(part.name, "Sonos speaker", "the part it was");
        assert_eq!(part.link.as_ref().unwrap().address, "192.168.0.90");
        assert!(onboard_part(&back, &room(), &moved, "speaker", "x").is_err());
    }

    /// **A thing with no network is a part too, and a room is described.**
    #[test]
    fn a_thing_is_onboarded_and_a_room_described() {
        let map = world();
        let built = apply(
            &map,
            onboard_thing(&map, &room(), "sofa", "The sofa faces the TV.").unwrap(),
        );
        assert_eq!(built.part("thing-front-room-sofa").unwrap().name, "sofa");
        assert!(onboard_thing(&built, &room(), "Sofa", "x").is_err());
        let described = apply(
            &built,
            describe_room(&built, &room(), "The main room.").unwrap(),
        );
        assert_eq!(
            described
                .get("flat")
                .unwrap()
                .node("front-room")
                .unwrap()
                .character
                .as_deref(),
            Some("The main room.")
        );
    }

    /// **A room is discovered in this house only, and only a room discovered
    /// is undiscovered** — never a new area, never the passage, never a room
    /// as it was authored.
    #[test]
    fn a_room_is_discovered_in_this_house_and_only_a_discovered_one_undiscovered() {
        let map = world();
        let edits = discover_door(&map, &room(), "kitchen", "Where the family cooks.").unwrap();
        assert!(matches!(&edits[0], MapEdit::AddNode { area, .. } if area == "flat"));
        let made = edits.clone();
        let built = apply(&map, edits);
        let kitchen = built.get("flat").unwrap().node("kitchen").unwrap();
        assert_eq!(kitchen.name, "the kitchen");
        assert!(discover_door(&built, &room(), "the kitchen", "x").is_err());
        assert_eq!(
            doors(&built, &room(), &made).len(),
            1,
            "the kitchen, not the hall"
        );
        assert_eq!(
            doors(&built, &room(), &[]),
            [],
            "not discovered by the builders"
        );
        let gone = apply(
            &built,
            undiscover_door(&built, &room(), "the kitchen", &made).unwrap(),
        );
        assert!(gone.get("flat").unwrap().node("kitchen").is_none());
        assert!(undiscover_door(&map, &room(), "the hall", &made).is_err());
        assert!(discover_door(&map, &Where::new("moon", "watch"), "annex", "x").is_err());
    }
}
