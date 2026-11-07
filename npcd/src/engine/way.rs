//! The way a body that walks takes to a place, as the compass tells it.
//!
//! `move_to` reaches the rooms of the body's own level and the rooms a door
//! leads to from one of them ([`crate::engine::body::destinations`]); anywhere
//! else is the lift. A place off the shaft — a room of the Sharratts' home,
//! through a door from the groups level — is ridden to by the level its door
//! is on, not by its own, which the lift does not serve, and reached through
//! the door's room; and a body standing in one walks back through the door
//! before it can reach the lift at all.

use npc_map::schema::NodeKind;
use npc_map::world::{Where, World};

/// How a walking body reaches a place.
#[derive(Debug, PartialEq, Eq)]
pub enum Way {
    /// `move_to` it from here — by way of `through`, the room a door opens
    /// into in its area, when it lies beyond that room.
    Walk { through: Option<String> },
    /// Ride the lift to the level named `level`, then `move_to` it, by way of
    /// `through` as a walk is — after first walking out to `first`, the room
    /// through a door onto the shaft, when the body stands off it.
    Ride {
        level: String,
        first: Option<String>,
        through: Option<String>,
    },
}

/// The way from `here` to `to`. `None` when neither its level nor any level
/// with a door into its area is served by the lift, or the body stands off the
/// shaft with no door onto it.
pub fn way(w: &World, here: &Where, to: &Where) -> Option<Way> {
    if to.area == here.area {
        return Some(Way::Walk { through: None });
    }
    if let Some(door) = door_into(w, &here.area, &to.area) {
        return Some(Way::Walk {
            through: beyond(w, door, to),
        });
    }
    let (floor, through) = match w.floor_of(to) {
        Some(floor) => (floor, None),
        None => (0..w.shaft().len()).find_map(|i| {
            let door = door_into(w, w.floor_area(i)?, &to.area)?;
            Some((i, beyond(w, door, to)))
        })?,
    };
    let level = w.floor_name(floor)?.to_string();
    let first = match w.floor_of(here) {
        Some(_) => None,
        None => Some(door_onto_shaft(w, &here.area)?),
    };
    Some(Way::Ride {
        level,
        first,
        through,
    })
}

/// The name of the door's room, when `to` lies beyond it rather than being it.
fn beyond(w: &World, door: &Where, to: &Where) -> Option<String> {
    match door == to {
        true => None,
        false => w.node(door).map(|n| n.name.clone()),
    }
}

/// A room in `into` that a door out of one of `from`'s rooms opens onto — a
/// landing excepted, since a portal between landings is the lift, which is
/// ridden.
fn door_into<'w>(w: &'w World, from: &str, into: &str) -> Option<&'w Where> {
    doors_out(w, from)
        .find(|at| at.area == into && w.node(at).is_some_and(|n| n.kind != NodeKind::Core))
}

/// The name of a room a door from `area` opens into on a level the lift serves.
fn door_onto_shaft(w: &World, area: &str) -> Option<String> {
    doors_out(w, area)
        .filter(|at| w.floor_of(at).is_some())
        .find_map(|at| w.node(at).filter(|n| n.kind != NodeKind::Core))
        .map(|n| n.name.clone())
}

/// Everywhere a door out of one of `area`'s rooms leads.
fn doors_out<'w>(w: &'w World, area: &str) -> impl Iterator<Item = &'w Where> + 'w {
    let area = area.to_string();
    w.map()
        .get(&area)
        .into_iter()
        .flat_map(|a| a.nodes.iter())
        .flat_map(move |n| w.map().ways_from(&Where::new(area.clone(), n.id.clone())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use npc_map::schema::{Area, AreaKind, Node, Portal};
    use npc_map::MapSet;

    fn node(id: &str, kind: NodeKind, off: &[&str]) -> Node {
        Node {
            id: id.into(),
            kind,
            name: format!("the {id}"),
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

    fn area(id: &str, kind: AreaKind, ordinal: Option<u32>, nodes: Vec<Node>) -> Area {
        Area {
            id: id.into(),
            kind,
            name: format!("the {id} level"),
            within: None,
            ordinal,
            summary: "a place".into(),
            character: None,
            lacks: vec![],
            announcements: vec![],
            contains: vec![],
            portals: vec![],
            arrival: None,
            teleport_to: None,
            spine: None,
            mirrored: false,
            nodes,
        }
    }

    /// Two levels on the lift, and a home off the shaft — a front room through
    /// a door from the lower level's family room, and a back hall beyond it.
    fn moon() -> World {
        let mut home = area(
            "home",
            AreaKind::Building,
            None,
            vec![
                node("front-room", NodeKind::Social, &["back-hall"]),
                node("back-hall", NodeKind::Social, &[]),
            ],
        );
        home.portals = vec![Portal {
            between: ["groups/family-room".into(), "home/front-room".into()],
            kind: "a door".into(),
        }];
        let set = MapSet::from_areas([
            area(
                "command",
                AreaKind::Level,
                Some(1),
                vec![
                    node("core", NodeKind::Core, &["office"]),
                    node("office", NodeKind::Work, &[]),
                ],
            ),
            area(
                "groups",
                AreaKind::Level,
                Some(2),
                vec![
                    node("core", NodeKind::Core, &["family-room"]),
                    node("family-room", NodeKind::Social, &[]),
                ],
            ),
            home,
        ])
        .expect("the test map assembles");
        World::new(set)
    }

    fn walk(through: Option<&str>) -> Option<Way> {
        Some(Way::Walk {
            through: through.map(str::to_string),
        })
    }

    fn ride(level: &str, first: Option<&str>, through: Option<&str>) -> Option<Way> {
        Some(Way::Ride {
            level: level.into(),
            first: first.map(str::to_string),
            through: through.map(str::to_string),
        })
    }

    #[test]
    fn a_room_on_this_level_is_walked_to() {
        let w = moon();
        assert_eq!(
            way(
                &w,
                &Where::new("command", "core"),
                &Where::new("command", "office")
            ),
            walk(None)
        );
    }

    #[test]
    fn a_room_on_another_level_is_ridden_to_by_its_own() {
        let w = moon();
        assert_eq!(
            way(
                &w,
                &Where::new("command", "office"),
                &Where::new("groups", "family-room")
            ),
            ride("the groups level", None, None)
        );
    }

    /// **A room off the shaft is ridden to by the level its door is on**, and
    /// walked to from that level — the lift has no floor of its own for it.
    #[test]
    fn a_room_through_a_door_is_ridden_to_by_the_door_s_level() {
        let w = moon();
        let front = Where::new("home", "front-room");
        assert_eq!(
            way(&w, &Where::new("command", "office"), &front),
            ride("the groups level", None, None)
        );
        assert_eq!(way(&w, &Where::new("groups", "core"), &front), walk(None));
        assert_eq!(
            way(&w, &front, &Where::new("groups", "family-room")),
            walk(None)
        );
    }

    /// **A room beyond the door's own is reached through it**, from its level
    /// and from a level that rides there.
    #[test]
    fn a_room_beyond_the_door_is_reached_through_the_door_s_room() {
        let w = moon();
        let back = Where::new("home", "back-hall");
        assert_eq!(
            way(&w, &Where::new("groups", "core"), &back),
            walk(Some("the front-room"))
        );
        assert_eq!(
            way(&w, &Where::new("command", "office"), &back),
            ride("the groups level", None, Some("the front-room"))
        );
    }

    /// **Off the shaft, the way out starts back through the door.**
    #[test]
    fn a_body_off_the_shaft_walks_out_through_its_door_first() {
        let w = moon();
        assert_eq!(
            way(
                &w,
                &Where::new("home", "back-hall"),
                &Where::new("command", "office")
            ),
            ride("the command level", Some("the family-room"), None)
        );
        // On the level its door opens onto, the way is a walk through it.
        assert_eq!(
            way(
                &w,
                &Where::new("home", "back-hall"),
                &Where::new("groups", "core")
            ),
            walk(Some("the family-room"))
        );
    }
}
