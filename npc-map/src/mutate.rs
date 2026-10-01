//! Reshaping a map while the world is running.
//!
//! A [`MapEdit`] is one change to the authored half of a [`MapSet`] — a room
//! added, a door cut, a terminal set down. [`MapSet::apply`] makes it the way
//! everything else about a map is made: it takes the authored areas and parts,
//! applies the edit to the copies, and assembles a whole new set from them, so
//! the edited map is woven, joined and validated exactly as a loaded one is.
//! There is no half-edited map: an edit either yields a set that validates or an
//! error saying why not, and the set it was applied to is untouched.
//!
//! # Wire form
//!
//! Adjacently tagged — `{"op": "add_node", "with": { … }}` — so a client writes
//! the operation by name and its arguments beside it.
//!
//! # Instance ids survive an edit
//!
//! The rebuild carries the set's part offsets forward ([`MapSet::offsets`]), so
//! an id already handed out — `terminal~4` — keeps naming the same machine
//! however the map around it changes. A placement is therefore set down once per
//! node and part: [`MapEdit::PlacePart`] refuses a node that already places the
//! part, because growing a kept placement would run its ordinals into the next
//! one's.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::load::MapSet;
use crate::part::Placement;
use crate::schema::{Area, Node, Portal, Where};

/// One change to a map.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", content = "with", rename_all = "snake_case")]
pub enum MapEdit {
    /// A new area. When it says it is `within` a parent that does not yet list
    /// it, the parent's `contains` gains it in the same edit.
    AddArea(Box<Area>),
    /// A new node in an existing area.
    AddNode { area: String, node: Box<Node> },
    /// A new portal, declared by the area that contains both ends.
    AddPortal { area: String, portal: Portal },
    /// A part set down in a node that does not place it yet.
    PlacePart { at: Where, placement: Placement },
    /// A node taken out, with every door, sightline and spine stop that named it.
    RemoveNode(Where),
    /// An empty area taken out.
    RemoveArea(String),
    /// A portal taken out, named by its two ends.
    RemovePortal { area: String, between: [String; 2] },
    /// Every placement of a part taken out of a node.
    UnplacePart { at: Where, part: String },
    /// A node's name changed.
    Retitle { at: Where, name: String },
}

impl MapEdit {
    /// The id of the area this edit touches — the one authored file it changes.
    pub fn area(&self) -> &str {
        match self {
            MapEdit::AddArea(area) => &area.id,
            MapEdit::AddNode { area, .. }
            | MapEdit::AddPortal { area, .. }
            | MapEdit::RemovePortal { area, .. } => area,
            MapEdit::PlacePart { at, .. }
            | MapEdit::UnplacePart { at, .. }
            | MapEdit::Retitle { at, .. }
            | MapEdit::RemoveNode(at) => &at.area,
            MapEdit::RemoveArea(id) => id,
        }
    }
}

impl MapSet {
    /// The map this one becomes under `edit`, or why it cannot.
    pub fn apply(&self, edit: &MapEdit) -> Result<MapSet> {
        let (mut areas, parts) = self.authored();
        match edit {
            MapEdit::AddArea(area) => {
                if areas.iter().any(|a| a.id == area.id) {
                    bail!("there is already an area `{}`", area.id);
                }
                if let Some(parent) = &area.within {
                    let Some(parent) = areas.iter_mut().find(|a| &a.id == parent) else {
                        bail!(
                            "`{}` sits within `{}`, which is not an area",
                            area.id,
                            parent
                        );
                    };
                    if !parent.contains.contains(&area.id) {
                        parent.contains.push(area.id.clone());
                    }
                }
                areas.push((**area).clone());
            }
            MapEdit::AddNode { area, node } => {
                let target = area_mut(&mut areas, area)?;
                if target.node(&node.id).is_some() {
                    bail!("`{area}` already has a node `{}`", node.id);
                }
                target.nodes.push((**node).clone());
            }
            MapEdit::AddPortal { area, portal } => {
                area_mut(&mut areas, area)?.portals.push(portal.clone());
            }
            MapEdit::PlacePart { at, placement } => {
                let node = node_mut(&mut areas, at)?;
                if node.parts.iter().any(|p| p.part() == placement.part()) {
                    bail!("`{at}` already places `{}`", placement.part());
                }
                node.parts.push(placement.clone());
            }
            MapEdit::RemoveNode(at) => {
                let area = area_mut(&mut areas, &at.area)?;
                if area.node(&at.node).is_none() {
                    bail!("`{at}` is not a node here");
                }
                let reference = at.to_string();
                if area.arrival.as_deref() == Some(&reference)
                    || area.teleport_to.as_deref() == Some(&reference)
                {
                    bail!("`{at}` is where `{}` arrives or teleports to", area.id);
                }
                area.nodes.retain(|n| n.id != at.node);
                for node in &mut area.nodes {
                    node.off.retain(|id| id != &at.node);
                    node.sees.retain(|id| id != &at.node);
                }
                if let Some(spine) = &mut area.spine {
                    spine.through.retain(|id| id != &at.node);
                }
                for other in &mut areas {
                    other
                        .portals
                        .retain(|p| !p.between.iter().any(|end| end == &reference));
                }
            }
            MapEdit::RemoveArea(id) => {
                let area = area_mut(&mut areas, id)?;
                if !area.contains.is_empty() {
                    bail!("`{id}` still contains {}", area.contains.join(", "));
                }
                if !area.nodes.is_empty() {
                    bail!("`{id}` still holds nodes");
                }
                areas.retain(|a| &a.id != id);
                for other in &mut areas {
                    other.contains.retain(|child| child != id);
                    other.portals.retain(|p| {
                        !p.between
                            .iter()
                            .any(|end| end.split('/').next() == Some(id))
                    });
                }
            }
            MapEdit::RemovePortal { area, between } => {
                let target = area_mut(&mut areas, area)?;
                let before = target.portals.len();
                target.portals.retain(|p| &p.between != between);
                if target.portals.len() == before {
                    bail!(
                        "`{area}` has no portal between {} and {}",
                        between[0],
                        between[1]
                    );
                }
            }
            MapEdit::UnplacePart { at, part } => {
                let node = node_mut(&mut areas, at)?;
                let before = node.parts.len();
                node.parts.retain(|p| p.part() != part);
                if node.parts.len() == before {
                    bail!("`{at}` does not place `{part}`");
                }
            }
            MapEdit::Retitle { at, name } => {
                if name.trim().is_empty() {
                    bail!("a node cannot be retitled to nothing");
                }
                node_mut(&mut areas, at)?.name = name.clone();
            }
        }
        MapSet::assemble_from(areas, parts, self.offsets())
    }
}

fn area_mut<'a>(areas: &'a mut [Area], id: &str) -> Result<&'a mut Area> {
    match areas.iter_mut().find(|a| a.id == id) {
        Some(area) => Ok(area),
        None => bail!("`{id}` is not an area"),
    }
}

fn node_mut<'a>(areas: &'a mut [Area], at: &Where) -> Result<&'a mut Node> {
    let area = area_mut(areas, &at.area)?;
    match area.nodes.iter_mut().find(|n| n.id == at.node) {
        Some(node) => Ok(node),
        None => bail!("`{at}` is not a node here"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::part::{Part, PartKind};
    use crate::schema::{AreaKind, NodeKind};

    fn part(id: &str) -> Part {
        Part {
            id: id.into(),
            kind: PartKind::Station,
            name: id.into(),
            plural: None,
            binds: None,
            short: None,
            long: "l".into(),
            modes: vec![],
        }
    }

    fn node(id: &str, kind: NodeKind, off: &[&str], parts: Vec<Placement>) -> Node {
        Node {
            id: id.into(),
            kind,
            name: id.into(),
            plural: false,
            stand: None,
            off: off.iter().map(|s| s.to_string()).collect(),
            character: None,
            parts,
            ground: vec![],
            habit: None,
            sees: vec![],
            exits: vec![],
            visible: vec![],
        }
    }

    fn counted(part: &str, count: u32) -> Placement {
        Placement::Counted {
            part: part.into(),
            count,
        }
    }

    fn area(id: &str, nodes: Vec<Node>) -> Area {
        Area {
            id: id.into(),
            kind: AreaKind::Level,
            name: id.into(),
            within: None,
            ordinal: None,
            summary: "s".into(),
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

    fn set() -> MapSet {
        let hall = area(
            "hall",
            vec![
                node("core", NodeKind::Core, &[], vec![counted("desk", 2)]),
                node(
                    "annex",
                    NodeKind::Social,
                    &["core"],
                    vec![counted("desk", 1)],
                ),
            ],
        );
        MapSet::assemble([hall], [part("desk"), part("lamp")]).unwrap()
    }

    fn at(node: &str) -> Where {
        Where::new("hall", node)
    }

    fn ids(map: &MapSet, node: &str) -> Vec<String> {
        map.instances_at(&at(node)).iter().map(|i| i.id()).collect()
    }

    #[test]
    fn an_edit_names_the_one_area_it_touches() {
        let edits = [
            (MapEdit::AddArea(Box::new(area("yard", vec![]))), "yard"),
            (MapEdit::RemoveNode(at("annex")), "hall"),
            (MapEdit::RemoveArea("yard".into()), "yard"),
            (
                MapEdit::RemovePortal {
                    area: "hall".into(),
                    between: ["a".into(), "b".into()],
                },
                "hall",
            ),
            (
                MapEdit::Retitle {
                    at: at("core"),
                    name: "Core".into(),
                },
                "hall",
            ),
        ];
        for (edit, expected) in edits {
            assert_eq!(edit.area(), expected);
        }
    }

    #[test]
    fn a_node_added_is_woven_in() {
        let map = set()
            .apply(&MapEdit::AddNode {
                area: "hall".into(),
                node: Box::new(node("store", NodeKind::Store, &["core"], vec![])),
            })
            .unwrap();
        assert!(map.node_at(&at("store")).is_some());
        assert!(map
            .node_at(&at("core"))
            .unwrap()
            .exits
            .contains(&"store".to_string()));
    }

    #[test]
    fn a_node_removed_takes_the_doors_to_it() {
        let map = set().apply(&MapEdit::RemoveNode(at("annex"))).unwrap();
        assert!(map.node_at(&at("annex")).is_none());
        assert!(!map
            .node_at(&at("core"))
            .unwrap()
            .exits
            .contains(&"annex".to_string()));
    }

    #[test]
    fn a_door_to_nowhere_is_refused_and_the_old_map_stands() {
        let before = set();
        let err = before
            .apply(&MapEdit::AddNode {
                area: "hall".into(),
                node: Box::new(node("store", NodeKind::Store, &["nowhere"], vec![])),
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a node here"), "{err}");
        assert!(before.node_at(&at("store")).is_none());
    }

    #[test]
    fn instance_ids_do_not_move_when_another_room_gains_the_same_part() {
        let before = set();
        assert_eq!(ids(&before, "core"), vec!["desk~0", "desk~1"]);
        assert_eq!(ids(&before, "annex"), vec!["desk~2"]);
        let after = before
            .apply(&MapEdit::AddNode {
                area: "hall".into(),
                node: Box::new(node(
                    "aaa",
                    NodeKind::Store,
                    &["core"],
                    vec![counted("desk", 3)],
                )),
            })
            .unwrap();
        assert_eq!(ids(&after, "core"), vec!["desk~0", "desk~1"]);
        assert_eq!(ids(&after, "annex"), vec!["desk~2"]);
        assert_eq!(ids(&after, "aaa"), vec!["desk~3", "desk~4", "desk~5"]);
    }

    #[test]
    fn a_part_is_placed_once_per_node() {
        let map = set()
            .apply(&MapEdit::PlacePart {
                at: at("annex"),
                placement: Placement::Bare("lamp".into()),
            })
            .unwrap();
        assert_eq!(ids(&map, "annex"), vec!["desk~2", "lamp~0"]);
        let err = map
            .apply(&MapEdit::PlacePart {
                at: at("annex"),
                placement: counted("desk", 2),
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("already places"), "{err}");
    }

    #[test]
    fn a_placement_past_the_limit_is_refused() {
        let err = set()
            .apply(&MapEdit::PlacePart {
                at: at("annex"),
                placement: counted("lamp", 257),
            })
            .unwrap_err()
            .to_string();
        assert!(err.contains("more than the"), "{err}");
    }

    #[test]
    fn a_part_unplaced_leaves_the_node() {
        let map = set()
            .apply(&MapEdit::UnplacePart {
                at: at("core"),
                part: "desk".into(),
            })
            .unwrap();
        assert!(ids(&map, "core").is_empty());
        assert_eq!(ids(&map, "annex"), vec!["desk~2"]);
    }

    #[test]
    fn a_retitle_changes_the_name_and_refuses_a_blank_one() {
        let map = set()
            .apply(&MapEdit::Retitle {
                at: at("annex"),
                name: "the long annex".into(),
            })
            .unwrap();
        assert_eq!(map.node_at(&at("annex")).unwrap().name, "the long annex");
        assert!(set()
            .apply(&MapEdit::Retitle {
                at: at("annex"),
                name: "  ".into()
            })
            .is_err());
    }

    #[test]
    fn an_area_with_children_is_not_removed() {
        let mut hall = area("hall", vec![node("core", NodeKind::Core, &[], vec![])]);
        hall.contains = vec!["wing".into()];
        let mut wing = area("wing", vec![]);
        wing.within = Some("hall".into());
        let map = MapSet::assemble([hall, wing], Vec::<Part>::new()).unwrap();
        let err = map
            .apply(&MapEdit::RemoveArea("hall".into()))
            .unwrap_err()
            .to_string();
        assert!(err.contains("still contains"), "{err}");
        let map = map.apply(&MapEdit::RemoveArea("wing".into())).unwrap();
        assert!(map.get("wing").is_none());
        assert!(map.get("hall").unwrap().contains.is_empty());
    }

    #[test]
    fn the_wire_form_is_adjacently_tagged() {
        let edit: MapEdit = serde_yaml::from_str(
            "op: retitle\nwith:\n  at: { area: hall, node: annex }\n  name: the annex\n",
        )
        .unwrap();
        let map = set().apply(&edit).unwrap();
        assert_eq!(map.node_at(&at("annex")).unwrap().name, "the annex");
    }
}
