//! Changing the shape of a world while it runs.
//!
//! Everywhere else in this crate the map is frozen once [`MapSet::assemble`] has
//! woven and checked it: bodies move, the lift runs, events accrue, but the
//! rooms and the ways between them do not change. This is where they can — where
//! a Maker adds a level, drowns a room, or opens a gate that was not there
//! before (effector design Appendix F).
//!
//! # A mutation is apply-to-a-copy, re-derive, validate, swap
//!
//! There is no partial edit of a live [`MapSet`], and that is the whole safety
//! of it. A [`MapEdit`] is applied to a **clone of the authored data**
//! ([`MapSet::authored`]), and the result is handed straight back through
//! [`MapSet::assemble`] — the same pipeline a load runs: [`weave`](crate::load)
//! doors and sight both ways, index the portals, and [`validate`](crate::
//! validate) the whole set. So every derived thing (`exits`, `visible`, the
//! portal graph) is recomputed from scratch, and every invariant the loader
//! enforces is enforced again:
//!
//! - a door is mutual and derived — a one-way `off` is impossible to write, and
//!   `exits`/`visible` are never hand-set (they are overwritten by the re-weave);
//! - a portal is both-way, or it is dropped and reported;
//! - the whole set validates, or the mutation is refused and **nothing changes**.
//!
//! A mutation that would break the map therefore returns `Err` and the caller
//! keeps the map it had. There is no state in which half an edit has landed,
//! because the edit only ever lands as a whole new set that already passed the
//! same gate a freshly-loaded world does.
//!
//! # This is the map only; the world around it is [`World::reshape`]
//!
//! [`MapSet::apply`] produces a new set and no more. Swapping it into a running
//! [`World`](crate::world::World) — re-deriving the lift shaft, and relocating
//! any body left standing where a node used to be — is [`crate::world::World::
//! reshape`], which calls this and then reconciles the state that hangs off the
//! map.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::load::MapSet;
use crate::part::Placement;
use crate::schema::{Area, Node, Portal, Where};

/// One change to the shape of a world.
///
/// Each variant names a target by id, so an edit that names nothing real is
/// refused with a clear reason *before* the rebuild, rather than surfacing as a
/// weave or validate error against a set the caller cannot see. Everything that
/// survives the target check is then handed to [`MapSet::assemble`], whose own
/// errors carry the structural reason (a door to nowhere, a stranded level).
///
/// [`Area`] and [`Node`] are boxed because they are large and the enum would
/// otherwise be sized to its biggest variant on every value.
///
/// **The wire form is adjacently tagged** — `{ "op": "add_node", "with": {…} }`
/// — so an operator or embedder route ([`crate::world`]'s reshape surface) reads
/// one straight off a request body, and every variant (a bare string, a
/// [`Where`], a struct) tags uniformly. This is the operator/embedder surface's
/// format, not the NPC device's; it carries whole `Area`/`Node` structures, which
/// is the honest shape of "add this room", not a thing a character types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", content = "with", rename_all = "snake_case")]
pub enum MapEdit {
    /// Add a whole area — a level, a region, a building.
    AddArea(Box<Area>),
    /// Remove an area and everything in it.
    RemoveArea(String),
    /// Add a node to an existing area.
    AddNode { area: String, node: Box<Node> },
    /// Remove one node from an area.
    RemoveNode(Where),
    /// Rename a node in place, without moving it or what it opens off.
    Retitle { at: Where, name: String },
    /// Add a portal to an area (which must hold, or reach, both ends).
    AddPortal { area: String, portal: Portal },
    /// Remove a portal from an area, matched on the pair it joins in either
    /// order.
    RemovePortal { area: String, between: [String; 2] },
    /// Place a part in a node.
    PlacePart { at: Where, placement: Placement },
    /// Remove every placement of one part from a node.
    UnplacePart { at: Where, part: String },
}

impl MapSet {
    /// Apply one edit, returning the rebuilt set — or `Err`, leaving `self`
    /// untouched.
    ///
    /// The transaction described in the module docs: clone the authored data,
    /// apply the edit to it, and re-[`assemble`](MapSet::assemble). The parts
    /// catalogue is carried through unchanged — topology mutation reshapes where
    /// things stand, never what kinds of thing exist.
    pub fn apply(&self, edit: &MapEdit) -> Result<MapSet> {
        let (mut areas, parts) = self.authored();
        edit.apply_to(&mut areas)?;
        // Carries this set's own instance offsets forward ([`part_offsets`])
        // so the edit can only ever add new instance ids, never renumber one
        // already handed out.
        MapSet::assemble_from(areas, parts, self.offsets())
    }
}

impl MapEdit {
    /// Apply this edit to the authored areas, in place.
    ///
    /// Only the target-existence checks live here — "no such area", "a node with
    /// that id is already here". Structural validity (a door to nowhere, a level
    /// with no core) is [`assemble`](MapSet::assemble)'s to judge over the whole
    /// rebuilt set, because it is a property of the set and not of the edit.
    fn apply_to(&self, areas: &mut Vec<Area>) -> Result<()> {
        match self {
            MapEdit::AddArea(area) => {
                if areas.iter().any(|a| a.id == area.id) {
                    bail!("an area `{}` is already here", area.id);
                }
                areas.push((**area).clone());
            }
            MapEdit::RemoveArea(id) => {
                let before = areas.len();
                areas.retain(|a| &a.id != id);
                if areas.len() == before {
                    bail!("no area `{id}` to remove");
                }
            }
            MapEdit::AddNode { area, node } => {
                let a = area_mut(areas, area)?;
                if a.nodes.iter().any(|n| n.id == node.id) {
                    bail!("`{area}` already holds a node `{}`", node.id);
                }
                a.nodes.push((**node).clone());
            }
            MapEdit::RemoveNode(at) => {
                let a = area_mut(areas, &at.area)?;
                let before = a.nodes.len();
                a.nodes.retain(|n| n.id != at.node);
                if a.nodes.len() == before {
                    bail!("no node `{at}` to remove");
                }
            }
            MapEdit::Retitle { at, name } => {
                let a = area_mut(areas, &at.area)?;
                let node = a
                    .nodes
                    .iter_mut()
                    .find(|n| n.id == at.node)
                    .ok_or_else(|| anyhow::anyhow!("no node `{at}` to rename"))?;
                node.name = name.clone();
            }
            MapEdit::AddPortal { area, portal } => {
                area_mut(areas, area)?.portals.push(portal.clone());
            }
            MapEdit::RemovePortal { area, between } => {
                let a = area_mut(areas, area)?;
                let before = a.portals.len();
                a.portals.retain(|p| !joins_same(&p.between, between));
                if a.portals.len() == before {
                    bail!(
                        "no portal between `{}` and `{}` on `{area}`",
                        between[0],
                        between[1]
                    );
                }
            }
            MapEdit::PlacePart { at, placement } => {
                node_mut(areas, at)?.parts.push(placement.clone());
            }
            MapEdit::UnplacePart { at, part } => {
                let node = node_mut(areas, at)?;
                let before = node.parts.len();
                node.parts.retain(|p| p.part() != part);
                if node.parts.len() == before {
                    bail!("no `{part}` placed at `{at}` to remove");
                }
            }
        }
        Ok(())
    }
}

impl MapEdit {
    /// The id of the area this edit changes — the file a persistence layer must
    /// rewrite (or, for [`MapEdit::RemoveArea`], remove) after a reshape.
    ///
    /// Every edit changes exactly one area: an area itself, the area a node or
    /// portal or placement lives in. That single id is what lets writeback touch
    /// one authored file rather than rewriting the whole map (and stripping the
    /// comments off every untouched one).
    pub fn area(&self) -> &str {
        match self {
            MapEdit::AddArea(a) => &a.id,
            MapEdit::RemoveArea(id) => id,
            MapEdit::AddNode { area, .. }
            | MapEdit::AddPortal { area, .. }
            | MapEdit::RemovePortal { area, .. } => area,
            MapEdit::RemoveNode(at)
            | MapEdit::Retitle { at, .. }
            | MapEdit::PlacePart { at, .. }
            | MapEdit::UnplacePart { at, .. } => &at.area,
        }
    }

    /// Whether this edit removes its area outright — the one case writeback
    /// deletes a file rather than rewriting it.
    pub fn removes_area(&self) -> bool {
        matches!(self, MapEdit::RemoveArea(_))
    }
}

/// The authored area with this id, mutably, or a clear "no such area".
fn area_mut<'a>(areas: &'a mut [Area], id: &str) -> Result<&'a mut Area> {
    areas
        .iter_mut()
        .find(|a| a.id == id)
        .ok_or_else(|| anyhow::anyhow!("no area `{id}`"))
}

/// The authored node at a place, mutably, or a clear "no such area/node".
fn node_mut<'a>(areas: &'a mut [Area], at: &Where) -> Result<&'a mut Node> {
    let a = area_mut(areas, &at.area)?;
    a.nodes
        .iter_mut()
        .find(|n| n.id == at.node)
        .ok_or_else(|| anyhow::anyhow!("no node `{at}`"))
}

/// Whether two portals join the same pair of ends, in either order — the match
/// [`MapEdit::RemovePortal`] identifies a portal by, since a portal is declared
/// once but reads both ways.
fn joins_same(a: &[String; 2], b: &[String; 2]) -> bool {
    (a[0] == b[0] && a[1] == b[1]) || (a[0] == b[1] && a[1] == b[0])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::part::{Part, PartKind, Placement};
    use crate::schema::{Area, AreaKind, Node, NodeKind, Where};

    /// A catalogue part, the smallest that validates — a named fixture.
    fn part(id: &str) -> Part {
        Part {
            id: id.into(),
            kind: PartKind::Fixture,
            name: id.into(),
            plural: None,
            binds: None,
            short: None,
            long: "what it is".into(),
            modes: vec![],
        }
    }

    fn node(id: &str, kind: NodeKind, off: &[&str]) -> Node {
        Node {
            id: id.into(),
            kind,
            name: id.into(),
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

    fn area(id: &str, nodes: Vec<Node>) -> Area {
        Area {
            id: id.into(),
            kind: AreaKind::Level,
            name: id.into(),
            within: None,
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
            nodes,
        }
    }

    /// A one-level set with a core and a hall, the smallest thing that validates.
    fn base() -> MapSet {
        MapSet::from_areas([area(
            "a",
            vec![
                node("core", NodeKind::Core, &["hall"]),
                node("hall", NodeKind::Passage, &[]),
            ],
        )])
        .expect("the base set validates")
    }

    /// **Adding a node re-weaves the doors both ways.** The new room opens off
    /// the hall; after the rebuild the hall opens back onto it, though only one
    /// end was written — the same guarantee a fresh load gives.
    #[test]
    fn adding_a_node_weaves_the_door_both_ways() {
        let set = base();
        let grown = set
            .apply(&MapEdit::AddNode {
                area: "a".into(),
                node: Box::new(node("green", NodeKind::Social, &["hall"])),
            })
            .expect("a room off the hall validates");
        let hall = grown.get("a").unwrap().node("hall").unwrap();
        assert!(
            hall.exits.contains(&"green".to_string()),
            "the door was not woven back: {:?}",
            hall.exits
        );
        // The original set is untouched — a mutation returns a new set.
        assert!(set.get("a").unwrap().node("green").is_none());
    }

    /// **A node that opens off nothing real is refused, and nothing changes.**
    /// The rebuild's weave catches the door to nowhere, so `apply` returns `Err`
    /// and the caller keeps the set it had.
    #[test]
    fn a_node_that_opens_off_nowhere_is_refused() {
        let set = base();
        let err = set
            .apply(&MapEdit::AddNode {
                area: "a".into(),
                node: Box::new(node("green", NodeKind::Social, &["nowhere"])),
            })
            .expect_err("a door to nowhere cannot validate");
        assert!(err.to_string().contains("not a node here"), "{err}");
    }

    /// **Removing a node is refused when another room still opens off it.** The
    /// hall opens off the core; removing the core leaves a door to nowhere, which
    /// the re-weave will not have.
    #[test]
    fn removing_a_node_others_open_off_is_refused() {
        let set = MapSet::from_areas([area(
            "a",
            vec![
                node("core", NodeKind::Core, &[]),
                node("hall", NodeKind::Passage, &["core"]),
            ],
        )])
        .unwrap();
        let err = set
            .apply(&MapEdit::RemoveNode(Where::new("a", "core")))
            .expect_err("a room still opens off the core");
        assert!(err.to_string().contains("not a node here"), "{err}");
    }

    /// **A placement lands, and reads back at the node.** The part must be one
    /// the catalogue holds — a placement of an unknown part does not validate, so
    /// the set is assembled with a real one first.
    #[test]
    fn a_part_can_be_placed_and_unplaced() {
        let set = MapSet::assemble(
            [area(
                "a",
                vec![
                    node("core", NodeKind::Core, &["hall"]),
                    node("hall", NodeKind::Passage, &[]),
                ],
            )],
            [part("a-board")],
        )
        .expect("the base set validates");
        let placed = set
            .apply(&MapEdit::PlacePart {
                at: Where::new("a", "hall"),
                placement: Placement::Bare("a-board".into()),
            })
            .expect("a placement of a real part does not break the map");
        let hall = placed.get("a").unwrap().node("hall").unwrap();
        assert_eq!(hall.parts.len(), 1);
        assert_eq!(hall.parts[0].part(), "a-board");

        let bare = placed
            .apply(&MapEdit::UnplacePart {
                at: Where::new("a", "hall"),
                part: "a-board".into(),
            })
            .expect("removing it does not break the map");
        assert!(bare
            .get("a")
            .unwrap()
            .node("hall")
            .unwrap()
            .parts
            .is_empty());
    }

    /// **A runtime edit never renumbers an instance id already handed out.**
    /// Area `b` places two `board`s before the edit, at ordinals `0` and `1`.
    /// Adding a new area `a` — which sorts *before* `b` in the `BTreeMap` walk
    /// [`part_offsets`] uses — with a `board` of its own must not shift `b`'s
    /// ordinals down to make room; the new placement is appended after them
    /// instead. This is the failure this crate's own id-stability promise
    /// names (`crate::instance`'s module doc): a URL a character followed
    /// before the edit has to still resolve to the same thing after it.
    #[test]
    fn a_runtime_edit_never_renumbers_an_existing_instance_id() {
        let set = MapSet::assemble(
            [area(
                "b",
                vec![node("core", NodeKind::Core, &["hall"]), {
                    let mut hall = node("hall", NodeKind::Passage, &[]);
                    hall.parts = vec![Placement::Counted {
                        part: "board".into(),
                        count: 2,
                    }];
                    hall
                }],
            )],
            [part("board")],
        )
        .expect("the base set validates");
        let before: Vec<u32> = set
            .instances_at(&Where::new("b", "hall"))
            .iter()
            .map(|i| i.ordinal())
            .collect();
        assert_eq!(before, vec![0, 1], "the base placements start at 0");

        let mut new_area = area("a", vec![node("core", NodeKind::Core, &[])]);
        new_area.nodes[0].parts = vec![Placement::Bare("board".into())];
        let edited = set
            .apply(&MapEdit::AddArea(Box::new(new_area)))
            .expect("adding an earlier-sorting area validates");

        // `b`'s instances kept the exact ordinals they had before the edit.
        let after: Vec<u32> = edited
            .instances_at(&Where::new("b", "hall"))
            .iter()
            .map(|i| i.ordinal())
            .collect();
        assert_eq!(after, before, "an existing instance's id moved");

        // The new placement in `a` was appended after them, not slotted in
        // front by virtue of `a` sorting first.
        let new_ordinal = edited
            .instances_at(&Where::new("a", "core"))
            .first()
            .expect("the new board placed")
            .ordinal();
        assert_eq!(
            new_ordinal, 2,
            "the new placement did not append at the end"
        );
    }

    /// **A placement of a part the catalogue does not hold is refused.** The
    /// validator will not have a node standing something that does not exist, so
    /// the transaction fails and the map is unchanged.
    #[test]
    fn placing_an_unknown_part_is_refused() {
        let set = base();
        let err = set
            .apply(&MapEdit::PlacePart {
                at: Where::new("a", "hall"),
                placement: Placement::Bare("a-ghost".into()),
            })
            .expect_err("an unknown part cannot be placed");
        assert!(err.to_string().contains("not a part"), "{err}");
    }

    /// **Renaming a node keeps its doors.** The title changes; `exits` are
    /// re-derived from the untouched `off`, so the room stays where it was.
    #[test]
    fn retitle_keeps_the_doors() {
        let set = base();
        let renamed = set
            .apply(&MapEdit::Retitle {
                at: Where::new("a", "hall"),
                name: "the long hall".into(),
            })
            .expect("a rename is always valid");
        let hall = renamed.get("a").unwrap().node("hall").unwrap();
        assert_eq!(hall.name, "the long hall");
        assert!(hall.exits.contains(&"core".to_string()));
    }

    /// **The wire form round-trips, and names the area it touches.** An operator
    /// route reads a [`MapEdit`] straight off a request body, so the tagged form
    /// has to parse back to the same edit — and [`MapEdit::area`] must name the
    /// file writeback rewrites. (The exact JSON shape is pinned in the npcd route
    /// test, which has `serde_json`; here it round-trips through `serde_yaml`, the
    /// serializer this crate carries.)
    #[test]
    fn the_wire_form_round_trips_and_names_its_area() {
        let edit = MapEdit::AddNode {
            area: "vault-casting".into(),
            node: Box::new(node("annex", NodeKind::Social, &["green-room"])),
        };
        let wire = serde_yaml::to_string(&edit).expect("serialises");
        assert!(wire.contains("op: add_node"), "{wire}");
        let back: MapEdit = serde_yaml::from_str(&wire).expect("parses back");
        assert_eq!(back.area(), "vault-casting");
        assert!(!back.removes_area());

        // A bare-string variant tags the same way and reports a removal.
        let drown = MapEdit::RemoveArea("vault-annex".into());
        let back: MapEdit = serde_yaml::from_str(&serde_yaml::to_string(&drown).unwrap()).unwrap();
        assert_eq!(back.area(), "vault-annex");
        assert!(back.removes_area());
    }

    /// **An edit that names nothing real is refused with a clear reason**, before
    /// any rebuild.
    #[test]
    fn an_edit_on_a_missing_target_is_refused() {
        let set = base();
        let err = set
            .apply(&MapEdit::RemoveNode(Where::new("a", "ghost")))
            .expect_err("there is no such node");
        assert!(err.to_string().contains("no node"), "{err}");
        let err = set
            .apply(&MapEdit::AddNode {
                area: "nowhere".into(),
                node: Box::new(node("x", NodeKind::Social, &[])),
            })
            .expect_err("there is no such area");
        assert!(err.to_string().contains("no area"), "{err}");
    }
}
