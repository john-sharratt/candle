//! One placed thing in one room, and the id that names it.
//!
//! A node places parts by reference — `{ part: terminal, count: 6 }` — so the
//! map says *what kind* of thing stands there and how many. A body works one
//! particular terminal, though, and the world has to tell six of them apart: a
//! claim on the second leaves the others free, and a URL a character followed
//! yesterday has to reach the same machine today. A [`PartInstance`] is that
//! one placement, and [`PartInstance::id`] is its name.
//!
//! # The id is `<part>~<ordinal>`
//!
//! The ordinal is **global to the part** — the world-wide index of this
//! placement among every placement of the same part — not local to the room, so
//! the id carries neither an area nor a node name and stays short. The room's
//! contribution is its offset ([`MapSet`]'s `part_offsets`): the first
//! ordinal its placements of the part start from. A runtime edit carries every
//! offset forward, so an id already handed out keeps naming the same machine
//! when an earlier room gains a placement of the same part.
//!
//! [`PartInstance::ordinal`] is the other number: the placement's position
//! *within its room*, from zero, which is what "the second terminal" counts.

use std::collections::BTreeMap;

use crate::load::MapSet;
use crate::part::Part;
use crate::schema::Where;

/// One placement of a part: a single terminal of the six in a room.
#[derive(Debug, Clone)]
pub struct PartInstance<'a> {
    part: &'a Part,
    at: Where,
    local: u32,
    global: u32,
}

impl<'a> PartInstance<'a> {
    /// The kind of thing this is.
    pub fn part(&self) -> &'a Part {
        self.part
    }

    /// The id of the kind of thing this is — what the instance id is built on.
    pub fn part_id(&self) -> &'a str {
        &self.part.id
    }

    /// The room it stands in.
    pub fn at(&self) -> &Where {
        &self.at
    }

    /// Its position among the placements of this part in its room, from zero.
    pub fn ordinal(&self) -> u32 {
        self.local
    }

    /// Its position among every placement of this part in the world — the
    /// number the id carries.
    pub fn global_ordinal(&self) -> u32 {
        self.global
    }

    /// The id that names exactly this placement: `<part>~<global ordinal>`.
    pub fn id(&self) -> String {
        format!("{}~{}", self.part.id, self.global)
    }
}

impl MapSet {
    /// Every placed part in a node, in the order the node places them, each
    /// placement of a counted part standing alone.
    ///
    /// A placement of a part the catalogue does not hold is skipped, as
    /// validation refuses such a map before it can be walked. Empty for a place
    /// that is not in the map.
    pub fn instances_at(&self, at: &Where) -> Vec<PartInstance<'_>> {
        let Some(node) = self.node_at(at) else {
            return Vec::new();
        };
        let mut next: BTreeMap<&str, u32> = BTreeMap::new();
        let mut out = Vec::new();
        for placement in &node.parts {
            let Some(part) = self.part(placement.part()) else {
                continue;
            };
            let offset = self.part_offset(at, &part.id);
            for _ in 0..placement.count() {
                let local = next.entry(part.id.as_str()).or_default();
                out.push(PartInstance {
                    part,
                    at: at.clone(),
                    local: *local,
                    global: offset + *local,
                });
                *local += 1;
            }
        }
        out
    }

    /// The placement an instance id names, wherever in the world it stands.
    ///
    /// `None` for anything that is not the id of a placement that exists: a
    /// part that is not in the catalogue, an ordinal past the last placement,
    /// and any spelling other than the canonical one (`seat~07` is not `seat~7`,
    /// so one machine never answers to two names).
    pub fn instance(&self, id: &str) -> Option<PartInstance<'_>> {
        let (part_id, ordinal) = id.split_once('~')?;
        let global: u32 = ordinal.parse().ok()?;
        if global.to_string() != ordinal {
            return None;
        }
        self.part(part_id)?;
        self.areas().find_map(|area| {
            area.nodes.iter().find_map(|node| {
                let count: u32 = node
                    .parts
                    .iter()
                    .filter(|p| p.part() == part_id)
                    .map(|p| p.count())
                    .sum();
                let at = Where::new(area.id.clone(), node.id.clone());
                let offset = self.part_offset(&at, part_id);
                (count > 0 && (offset..offset + count).contains(&global))
                    .then(|| {
                        self.instances_at(&at)
                            .into_iter()
                            .find(|i| i.part_id() == part_id && i.global == global)
                    })
                    .flatten()
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::part::{PartKind, Placement};
    use crate::schema::{Area, AreaKind, Node, NodeKind};

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

    fn set() -> MapSet {
        let area = Area {
            id: "hall".into(),
            kind: AreaKind::Level,
            name: "Hall".into(),
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
            nodes: vec![
                node(
                    "annex",
                    NodeKind::Social,
                    &["core"],
                    vec![counted("desk", 1), Placement::Bare("lamp".into())],
                ),
                node(
                    "core",
                    NodeKind::Core,
                    &[],
                    vec![
                        counted("desk", 2),
                        Placement::Bare("lamp".into()),
                        counted("desk", 1),
                    ],
                ),
            ],
        };
        MapSet::assemble([area], [part("desk"), part("lamp")]).unwrap()
    }

    fn at(node: &str) -> Where {
        Where::new("hall", node)
    }

    /// Ordinals run across the world in the order a load walks it — nodes in file
    /// order — and a node's placements of one part share one run.
    #[test]
    fn ids_are_global_to_the_part_and_ordinals_local_to_the_room() {
        let map = set();
        let annex: Vec<(String, u32)> = map
            .instances_at(&at("annex"))
            .iter()
            .map(|i| (i.id(), i.ordinal()))
            .collect();
        assert_eq!(
            annex,
            vec![("desk~0".to_string(), 0), ("lamp~0".to_string(), 0)]
        );

        let core: Vec<(String, u32)> = map
            .instances_at(&at("core"))
            .iter()
            .map(|i| (i.id(), i.ordinal()))
            .collect();
        assert_eq!(
            core,
            vec![
                ("desk~1".to_string(), 0),
                ("desk~2".to_string(), 1),
                ("lamp~1".to_string(), 0),
                ("desk~3".to_string(), 2),
            ]
        );
    }

    #[test]
    fn a_place_that_is_not_in_the_map_holds_nothing() {
        assert!(set().instances_at(&at("nowhere")).is_empty());
    }

    #[test]
    fn an_id_resolves_to_the_room_and_placement_it_names() {
        let map = set();
        let found = map.instance("desk~2").expect("the third desk");
        assert_eq!(found.at(), &at("core"));
        assert_eq!(found.ordinal(), 1);
        assert_eq!(found.part_id(), "desk");

        let first = map.instance("lamp~0").expect("the annex lamp");
        assert_eq!(first.at(), &at("annex"));
    }

    #[test]
    fn an_id_that_names_nothing_resolves_to_nothing() {
        let map = set();
        assert!(map.instance("desk~4").is_none(), "past the last desk");
        assert!(map.instance("ghost~0").is_none(), "not in the catalogue");
        assert!(map.instance("desk").is_none(), "no ordinal");
        assert!(map.instance("desk~x").is_none(), "not a number");
        assert!(map.instance("desk~-1").is_none(), "negative");
    }

    #[test]
    fn one_machine_answers_to_one_name() {
        let map = set();
        assert!(map.instance("desk~2").is_some());
        assert!(map.instance("desk~02").is_none(), "leading zero");
        assert!(map.instance("desk~+2").is_none(), "explicit sign");
    }
}
