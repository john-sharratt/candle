//! Reading a directory of map files into one joined world.
//!
//! Every `.yaml` in the directory is one [`Area`], and areas find each other by
//! id — `contains` names children, `within` names a parent, a [`Portal`] names
//! `area/node` at each end. Nothing refers to a file name or a path, so a world
//! grows by adding files and a place can be moved between them without editing
//! anything that points at it.
//!
//! The deployed home for these is `<mind>/map/`, alongside the rest of the
//! authored corpus. [`MapSet::load_dir`] takes any directory, so the copy in
//! this repository and the installed copy load the same way.
//!
//! # Weaving
//!
//! An author writes what a room opens *off*; loading turns that into doors at
//! both ends, closes a looping spine, and works out what is visible from
//! where. Everything derived lives in `Node::exits` and `Node::visible`, which
//! are never read from a file. The point is that the two ends of a door cannot
//! disagree, because only one of them was ever written down.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::part::{Part, PartKind};
use crate::schema::{Area, Node, NodeKind, Where};
use crate::validate;

/// Every area that was loaded, and the catalogue of parts they place.
#[derive(Debug, Clone, Default)]
pub struct MapSet {
    areas: BTreeMap<String, Area>,
    parts: BTreeMap<String, Part>,
    /// Every node that leaves its area, and where it comes out. Woven from the
    /// portals of whichever area declares them — both ways, since a lift that
    /// only went up would be a lift nobody could come back down.
    ways: BTreeMap<Where, Vec<Where>>,
}

impl MapSet {
    /// Load every `.yaml` in `dir`, weave it, and check it holds together.
    ///
    /// Validation runs here rather than on demand: a map that does not
    /// validate cannot be walked, and finding that out at the moment an NPC
    /// tries to leave a room is finding out far too late.
    pub fn load_dir(dir: impl AsRef<Path>) -> Result<MapSet> {
        let dir = dir.as_ref();
        let areas: Vec<Area> = read_yaml(dir)?;
        if areas.is_empty() {
            bail!("no map files in {}", dir.display());
        }
        // Parts live one level down, because they are a catalogue rather than
        // a place: the same terminal is placed in several buildings, and a
        // world grows by adding rooms far more often than by adding kinds of
        // thing to stand in one.
        let parts_dir = dir.join("parts");
        let parts: Vec<Part> = if parts_dir.is_dir() {
            read_yaml(&parts_dir)?
        } else {
            Vec::new()
        };
        MapSet::assemble(areas, parts)
    }

    /// Build a set from areas already in memory. The tests use this; so does
    /// anything generating a world rather than reading one.
    pub fn from_areas(areas: impl IntoIterator<Item = Area>) -> Result<MapSet> {
        MapSet::assemble(areas, Vec::new())
    }

    /// Build a set from areas and a part catalogue.
    pub fn assemble(
        areas: impl IntoIterator<Item = Area>,
        parts: impl IntoIterator<Item = Part>,
    ) -> Result<MapSet> {
        let mut set = MapSet::default();
        for part in parts {
            if let Some(clash) = set.parts.get(&part.id) {
                bail!(
                    "two parts share the id `{}`: `{}` and `{}`",
                    part.id,
                    clash.name,
                    part.name
                );
            }
            set.parts.insert(part.id.clone(), part);
        }
        for mut area in areas {
            if let Some(clash) = set.areas.get(&area.id) {
                bail!(
                    "two areas share the id `{}`: `{}` and `{}`",
                    area.id,
                    clash.name,
                    area.name
                );
            }
            weave(&mut area)?;
            set.areas.insert(area.id.clone(), area);
        }
        // Portals join two areas, so they can only be woven once every area is
        // in — which is why they are an index on the set rather than a field on
        // a node, where the weave for one file would have to reach into
        // another.
        set.ways = ways(&set.areas);
        validate::check(&set)?;
        Ok(set)
    }

    /// Where a node leads outside its own area — one step, both ways.
    pub fn ways_from(&self, at: &Where) -> &[Where] {
        self.ways.get(at).map(Vec::as_slice).unwrap_or(&[])
    }

    /// The place bodies in this area can teleport to, if it has one.
    ///
    /// Looked up on the area itself and then on its parents, so a level
    /// inherits its building's destination without every file repeating it.
    pub fn teleport_to(&self, area: &str) -> Option<Where> {
        let mut here = self.get(area)?;
        loop {
            if let Some(to) = &here.teleport_to {
                return Where::parse(to).filter(|w| self.node_at(w).is_some());
            }
            here = self.get(here.within.as_deref()?)?;
        }
    }

    /// The node at a place, wherever in the set it is.
    pub fn node_at(&self, at: &Where) -> Option<&Node> {
        self.get(&at.area)?.node(&at.node)
    }

    /// Where a body arrives when it enters this world at large.
    ///
    /// The whole world's way in. Most bodies do not want this — they belong to
    /// a *part* of the world and should start there — so [`MapSet::arrival_in`]
    /// is the one a caller with somewhere in mind reaches for.
    pub fn arrival(&self) -> Option<Where> {
        let root = self.areas().find(|a| a.within.is_none())?;
        self.arrival_in(&root.id)
    }

    /// Where a body arrives when it enters one part of a world.
    ///
    /// A world is bigger than any one character's part of it, so this is asked
    /// of the part rather than the world: a Maker enters the vault, a soldier
    /// enters a city, and both are the same world.
    ///
    /// Found in the order a person would look:
    ///
    /// 1. **What the area says**, if it names an arrival. Nothing beats being
    ///    told.
    /// 2. **A core, in this area or under it** — the lift and the stair, which
    ///    is what a core *is*: the way in. Validation already walks from a core
    ///    to prove a level is reachable, so a body starting anywhere else could
    ///    begin somewhere the map has never checked leads anywhere.
    /// 3. **Any place at all under it**, for a part too small to have a core.
    ///
    /// `None` for an area that does not exist, or one with nowhere to stand.
    pub fn arrival_in(&self, area: &str) -> Option<Where> {
        // This area and everything under it, parents before children, so a
        // building's own answer beats a level's.
        let mut under = vec![self.get(area)?];
        let mut i = 0;
        while i < under.len() {
            under.extend(self.children(&under[i].id));
            i += 1;
        }
        // Being told beats guessing, at any depth. A world whose only building
        // names its door has named the world's door — descending past that to
        // pick a lift shaft would be ignoring the one part of the map that
        // actually answered.
        let told = under.iter().find_map(|a| {
            a.arrival
                .as_deref()
                .and_then(Where::parse)
                .filter(|at| self.node_at(at).is_some())
        });
        if told.is_some() {
            return told;
        }
        let first = |kind: Option<NodeKind>| {
            under.iter().find_map(|a| {
                a.nodes
                    .iter()
                    .find(|n| kind.is_none_or(|k| n.kind == k))
                    .map(|n| Where::new(a.id.clone(), n.id.clone()))
            })
        };
        first(Some(NodeKind::Core)).or_else(|| first(None))
    }

    pub fn get(&self, id: &str) -> Option<&Area> {
        self.areas.get(id)
    }

    pub fn areas(&self) -> impl Iterator<Item = &Area> {
        self.areas.values()
    }

    /// What the address system plays in an area, taken from the nearest
    /// enclosing area that has anything to play.
    ///
    /// A tannoy is fitted to a *building*, not to a room, so the recordings are
    /// authored on the building and every level and room inside it hears the
    /// same ones. Walking up rather than requiring each area to repeat the list
    /// is what keeps them authored once — and a level that wants its own can
    /// still declare them and be answered first.
    ///
    /// Empty for a building nobody left a message in, which the caller must
    /// treat as silence rather than substituting anything.
    pub fn announcements_for(&self, area: &str) -> &[String] {
        let mut at = self.areas.get(area);
        // Bounded by the depth of the containment chain, and by a hard limit
        // besides: `within` is authored, and a file that names its own parent as
        // itself would otherwise spin here for ever.
        for _ in 0..16 {
            let Some(here) = at else { break };
            if !here.announcements.is_empty() {
                return &here.announcements;
            }
            at = here.within.as_deref().and_then(|up| self.areas.get(up));
        }
        &[]
    }

    pub fn part(&self, id: &str) -> Option<&Part> {
        self.parts.get(id)
    }

    /// What stands in a node, resolved, in the order the node places them.
    pub fn parts_at<'a>(&'a self, node: &'a Node) -> impl Iterator<Item = (&'a Part, u32)> {
        node.parts
            .iter()
            .filter_map(|p| self.part(p.part()).map(|def| (def, p.count())))
    }

    /// The parts of one kind standing in a node.
    pub fn parts_of<'a>(
        &'a self,
        node: &'a Node,
        kind: PartKind,
    ) -> impl Iterator<Item = (&'a Part, u32)> {
        self.parts_at(node).filter(move |(p, _)| p.kind == kind)
    }

    /// How many stations stand in a node — the count that decides how many
    /// bodies can work there at once.
    pub fn stations_at(&self, node: &Node) -> u32 {
        self.parts_of(node, PartKind::Station).map(|(_, n)| n).sum()
    }

    /// Every workstation in an area.
    pub fn stations_in(&self, area: &Area) -> u32 {
        area.nodes.iter().map(|n| self.stations_at(n)).sum()
    }

    /// **The parts within reach at a node**, by id.
    ///
    /// This is the payoff of parts being referenced rather than described in
    /// place: what is standing next to a body is a function of where the body
    /// is, computed from the map.
    ///
    /// It stops here. Which *acts* those parts make available is the engine's
    /// to say — an act names the stations it attaches to — so this crate never
    /// carries a vocabulary it cannot check and cannot use.
    pub fn part_ids_at<'a>(&'a self, node: &'a Node) -> Vec<&'a str> {
        let mut out: Vec<&str> = Vec::new();
        for (part, _) in self.parts_at(node) {
            if !out.contains(&part.id.as_str()) {
                out.push(&part.id);
            }
        }
        out
    }

    /// The children of an area, in the order the parent names them.
    ///
    /// The parent's order is authored — floors run bottom to top or top to
    /// bottom because somebody decided — so it is followed rather than sorted.
    pub fn children(&self, id: &str) -> Vec<&Area> {
        let Some(area) = self.get(id) else {
            return Vec::new();
        };
        area.contains.iter().filter_map(|c| self.get(c)).collect()
    }

    /// Split an `area/node` reference. Used by portals and by anything that
    /// names a place from outside the area holding it.
    pub fn split_ref(reference: &str) -> Option<(&str, &str)> {
        reference.split_once('/')
    }
}

/// Turn what an author wrote into what a walker needs.
///
/// Three passes, each depending on the one before: doors from `off`, doors
/// along a looping spine, then sight — which is *through the doors* plus
/// whatever `sees` adds. Sight comes last because it is defined in terms of
/// the doors, which is also how it is described.
fn weave(area: &mut Area) -> Result<()> {
    let known: BTreeSet<String> = area.nodes.iter().map(|n| n.id.clone()).collect();
    let mut doors: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    for node in &area.nodes {
        for target in &node.off {
            if !known.contains(target) {
                bail!(
                    "`{}`: `{}` opens off `{}`, which is not a node here",
                    area.id,
                    node.id,
                    target
                );
            }
            if target == &node.id {
                bail!("`{}`: `{}` opens off itself", area.id, node.id);
            }
            join(&mut doors, &node.id, target);
        }
    }

    if let Some(spine) = &area.spine {
        for id in &spine.through {
            if !known.contains(id) {
                bail!("`{}`: the spine names `{}`, which is not here", area.id, id);
            }
        }
        // Consecutive passages join, and a loop closes back to the start. The
        // author writes the order; the shape supplies every door along it.
        for pair in spine.through.windows(2) {
            join(&mut doors, &pair[0], &pair[1]);
        }
        if spine.loops && spine.through.len() > 2 {
            let first = spine.through.first().unwrap().clone();
            let last = spine.through.last().unwrap().clone();
            join(&mut doors, &first, &last);
        }
    }

    for node in &mut area.nodes {
        node.exits = doors
            .get(&node.id)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();
    }

    // Sight is through your own doors, plus whatever breaks that rule.
    // Declared one way and made mutual, because a rail you can look down from
    // is a rail that can be looked up at.
    let extra: Vec<(String, String)> = area
        .nodes
        .iter()
        .flat_map(|n| n.sees.iter().map(|s| (n.id.clone(), s.clone())))
        .collect();
    let mut sight: BTreeMap<String, BTreeSet<String>> = area
        .nodes
        .iter()
        .map(|n| (n.id.clone(), n.exits.iter().cloned().collect()))
        .collect();
    for (from, to) in extra {
        if !known.contains(&to) {
            bail!("`{}`: `{}` sees `{}`, which is not here", area.id, from, to);
        }
        join(&mut sight, &from, &to);
    }
    for node in &mut area.nodes {
        node.visible = sight
            .get(&node.id)
            .map(|s| s.iter().cloned().collect())
            .unwrap_or_default();
    }

    Ok(())
}

/// Every portal in the set, indexed by each of its ends.
///
/// Both ends, always. A portal is declared once, by whichever area holds both
/// sides of it, and a door wired one way is the mistake the whole weaving
/// approach exists to make unwritable — a lift that went up and not down would
/// strand a body on the top floor with a route the map says exists.
///
/// Ends that name a node no area has are dropped here and reported by
/// [`validate`], which has the whole set to say so against.
fn ways(areas: &BTreeMap<String, Area>) -> BTreeMap<Where, Vec<Where>> {
    let mut out: BTreeMap<Where, Vec<Where>> = BTreeMap::new();
    let real = |w: &Where| {
        areas
            .get(&w.area)
            .and_then(|a| a.node(&w.node))
            .map(|_| w.clone())
    };
    for area in areas.values() {
        for portal in &area.portals {
            let (Some(a), Some(b)) = (
                Where::parse(&portal.between[0]).as_ref().and_then(real),
                Where::parse(&portal.between[1]).as_ref().and_then(real),
            ) else {
                continue;
            };
            for (from, to) in [(&a, &b), (&b, &a)] {
                let side = out.entry(from.clone()).or_default();
                if !side.contains(to) {
                    side.push(to.clone());
                }
            }
        }
    }
    out
}

/// Every `.yaml` directly in a directory, parsed. Subdirectories are left
/// alone, which is what keeps the part catalogue out of the area listing.
fn read_yaml<T: serde::de::DeserializeOwned>(dir: &Path) -> Result<Vec<T>> {
    let mut out = Vec::new();
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("reading map directory {}", dir.display()))?;
    for entry in entries {
        let path = entry?.path();
        if path.extension().and_then(|e| e.to_str()) != Some("yaml") {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        out.push(
            serde_yaml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?,
        );
    }
    Ok(out)
}

fn join(map: &mut BTreeMap<String, BTreeSet<String>>, a: &str, b: &str) {
    map.entry(a.to_string()).or_default().insert(b.to_string());
    map.entry(b.to_string()).or_default().insert(a.to_string());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{AreaKind, Node, NodeKind, Spine};

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

    #[test]
    fn one_end_of_a_door_is_enough_to_open_both() {
        let set = MapSet::from_areas([area(
            "a",
            vec![
                node("core", NodeKind::Core, &["hall"]),
                node("hall", NodeKind::Passage, &[]),
            ],
        )])
        .unwrap();
        let hall = set.get("a").unwrap().node("hall").unwrap();
        assert_eq!(hall.exits, vec!["core".to_string()]);
    }

    #[test]
    fn a_looping_spine_closes_itself() {
        let mut a = area(
            "a",
            vec![
                node("core", NodeKind::Core, &["n"]),
                node("n", NodeKind::Passage, &[]),
                node("e", NodeKind::Passage, &[]),
                node("s", NodeKind::Passage, &[]),
                node("w", NodeKind::Passage, &[]),
            ],
        );
        a.spine = Some(Spine {
            name: "the ring".into(),
            loops: true,
            through: ["n", "e", "s", "w"].map(String::from).to_vec(),
        });
        let set = MapSet::from_areas([a]).unwrap();
        let n = set.get("a").unwrap().node("n").unwrap();
        // Both ways round the ring, plus the core hanging off it.
        assert_eq!(n.exits, vec!["core", "e", "w"]);
    }

    #[test]
    fn sight_follows_the_doors_without_being_written_down() {
        let set = MapSet::from_areas([area(
            "a",
            vec![
                node("core", NodeKind::Core, &["hall"]),
                node("hall", NodeKind::Passage, &[]),
            ],
        )])
        .unwrap();
        let core = set.get("a").unwrap().node("core").unwrap();
        assert_eq!(core.visible, vec!["hall".to_string()]);
    }

    #[test]
    fn a_sightline_that_is_not_a_door_is_mutual() {
        let mut a = area(
            "a",
            vec![
                node("core", NodeKind::Core, &["hall"]),
                node("hall", NodeKind::Passage, &[]),
                node("gallery", NodeKind::Social, &["hall"]),
                node("pit", NodeKind::Work, &["hall"]),
            ],
        );
        a.nodes[2].sees = vec!["pit".into()];
        let set = MapSet::from_areas([a]).unwrap();
        let level = set.get("a").unwrap();
        assert!(level
            .node("pit")
            .unwrap()
            .visible
            .contains(&"gallery".into()));
        assert!(!level.node("pit").unwrap().exits.contains(&"gallery".into()));
    }

    #[test]
    fn opening_off_a_room_that_is_not_there_is_refused() {
        let err = MapSet::from_areas([area("a", vec![node("core", NodeKind::Core, &["nowhere"])])])
            .unwrap_err()
            .to_string();
        assert!(err.contains("not a node here"), "{err}");
    }

    #[test]
    fn children_come_back_in_the_order_the_parent_names_them() {
        let mut parent = area("parent", vec![node("core", NodeKind::Core, &[])]);
        parent.kind = AreaKind::Building;
        parent.contains = vec!["b".into(), "a".into()];
        let mut a = area("a", vec![node("core", NodeKind::Core, &[])]);
        let mut b = area("b", vec![node("core", NodeKind::Core, &[])]);
        a.within = Some("parent".into());
        b.within = Some("parent".into());
        let set = MapSet::from_areas([parent, a, b]).unwrap();
        let names: Vec<_> = set
            .children("parent")
            .iter()
            .map(|a| a.id.clone())
            .collect();
        assert_eq!(names, vec!["b".to_string(), "a".to_string()]);
    }

    #[test]
    fn a_reference_splits_into_area_and_node() {
        assert_eq!(
            MapSet::split_ref("vault-casting/core"),
            Some(("vault-casting", "core"))
        );
        assert_eq!(MapSet::split_ref("bare"), None);
    }

    #[test]
    fn a_body_arrives_where_the_map_says_it_does() {
        let set = MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps")).unwrap();
        let door = Where::new("vault-command", "command-room");
        // The vault names its door, so entering the vault is entering there.
        assert_eq!(set.arrival_in("creators-vault"), Some(door.clone()));
        // And the world's only building having named one answers for the world.
        assert_eq!(set.arrival(), Some(door));
    }

    #[test]
    fn a_part_of_a_world_is_entered_at_its_own_door_not_the_worlds() {
        // The whole reason arrival is asked of a part rather than a world: a
        // Maker belongs in the vault and a soldier belongs in a city, and both
        // are the same world.
        let set = MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps")).unwrap();
        // A level names no door of its own, so its core is the way in.
        assert_eq!(
            set.arrival_in("vault-casting"),
            Some(Where::new("vault-casting", "core"))
        );
        assert_ne!(
            set.arrival_in("vault-casting"),
            set.arrival_in("creators-vault")
        );
        assert_eq!(set.arrival_in("nowhere"), None);
    }

    #[test]
    fn a_world_that_names_no_way_back_starts_a_body_at_a_core() {
        // The core is the way in — validation already walks from there to prove
        // a level is reachable, so a body starting anywhere else could begin
        // somewhere the map has never checked leads anywhere.
        let mut a = area(
            "level",
            vec![
                node("hall", NodeKind::Social, &["core"]),
                node("core", NodeKind::Core, &[]),
            ],
        );
        a.kind = AreaKind::Level;
        let set = MapSet::from_areas([a]).unwrap();
        assert_eq!(set.arrival(), Some(Where::new("level", "core")));
    }

    #[test]
    fn a_world_with_no_core_at_all_still_has_a_way_in() {
        let mut a = area(
            "field",
            vec![
                node("north", NodeKind::Ground, &["south"]),
                node("south", NodeKind::Ground, &[]),
            ],
        );
        a.kind = AreaKind::Region;
        let set = MapSet::from_areas([a]).unwrap();
        assert_eq!(set.arrival(), Some(Where::new("field", "north")));
    }

    #[test]
    fn wherever_a_body_arrives_is_a_place_it_can_walk_out_of() {
        // The floor under all three answers: an arrival that stranded a body
        // would be a world nobody could ever leave the front door of.
        let set = MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps")).unwrap();
        let at = set.arrival().expect("a way in");
        let node = set.node_at(&at).expect("a real place");
        assert!(!node.exits.is_empty(), "{at} leads nowhere");
    }
}
