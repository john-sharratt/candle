//! What has to be true of a map before anything walks it.
//!
//! These run at load, after weaving. A map that fails one is refused rather
//! than carried, because every fault here shows up later as an NPC standing in
//! a room with no way out, or walking through a door that leads somewhere the
//! description never mentioned — and by then the cause is a long way from the
//! symptom.
//!
//! The list got shorter when doors started being woven rather than authored: a
//! one-way door and a bearing wired up backwards were the two commonest
//! faults, and neither is expressible any more.
//!
//! | Check | The mistake it catches |
//! |---|---|
//! | ids unique | two rooms answering to one name |
//! | the spine is passages | a room named as part of the route |
//! | a room has a way out | a room nobody said what it opens off |
//! | everything reachable | a wing walled off from the rest of the level |
//! | refs resolve | a `contains`, a portal or a `teleport_to` naming nowhere |

use std::collections::{BTreeSet, HashMap, VecDeque};

use anyhow::{bail, Result};

use crate::load::MapSet;
use crate::schema::{Area, NodeKind};

/// Check every area in the set, and the joins between them.
pub fn check(set: &MapSet) -> Result<()> {
    for area in set.areas() {
        check_area(area)?;
        check_parts(set, area)?;
    }
    check_joins(set)?;
    Ok(())
}

/// Every part a room places has to be in the catalogue.
///
/// A misspelt part id would otherwise be a room that quietly holds nothing —
/// no station to work at, no tool within reach — and nothing would say so
/// until an NPC stood in it and found there was nothing to do.
fn check_parts(set: &MapSet, area: &Area) -> Result<()> {
    for node in &area.nodes {
        for placement in &node.parts {
            if set.part(placement.part()).is_none() {
                bail!(
                    "`{}`: `{}` places `{}`, which is not a part",
                    area.id,
                    node.id,
                    placement.part()
                );
            }
        }
    }
    Ok(())
}

fn check_area(area: &Area) -> Result<()> {
    let mut seen = BTreeSet::new();
    for node in &area.nodes {
        if !seen.insert(node.id.as_str()) {
            bail!("`{}`: two nodes share the id `{}`", area.id, node.id);
        }
    }

    // A spine names passages, and only passages. A room in the route means
    // somebody has confused a destination with a way of reaching one, and the
    // description would then walk its reader through a workroom.
    if let Some(spine) = &area.spine {
        for id in &spine.through {
            match area.node(id) {
                None => bail!("`{}`: the spine names `{}`, which is not here", area.id, id),
                Some(n) if n.kind != NodeKind::Passage => bail!(
                    "`{}`: the spine names `{}`, which is a {} rather than a passage",
                    area.id,
                    id,
                    n.kind.slug()
                ),
                Some(_) => {}
            }
        }
    }

    if area.nodes.len() > 1 {
        for node in &area.nodes {
            if node.exits.is_empty() {
                bail!(
                    "`{}`: nothing opens onto `{}` and it opens onto nothing",
                    area.id,
                    node.id
                );
            }
        }
    }

    check_reachable(area)
}

/// Every node reachable from the way in.
///
/// The walk starts at the core, because that is where an NPC arrives. A room
/// reachable only from another room that is itself unreachable is still
/// unreachable, which is why this is a traversal and not a check that every
/// node has a door.
fn check_reachable(area: &Area) -> Result<()> {
    if area.nodes.len() < 2 {
        return Ok(());
    }
    let start = area
        .of_kind(NodeKind::Core)
        .next()
        .or_else(|| area.nodes.first());
    let Some(start) = start else {
        return Ok(());
    };

    let mut seen = BTreeSet::from([start.id.as_str()]);
    let mut queue = VecDeque::from([start.id.as_str()]);
    while let Some(id) = queue.pop_front() {
        let Some(node) = area.node(id) else { continue };
        for exit in &node.exits {
            if seen.insert(exit.as_str()) {
                queue.push_back(exit);
            }
        }
    }

    let stranded: Vec<&str> = area
        .nodes
        .iter()
        .map(|n| n.id.as_str())
        .filter(|id| !seen.contains(id))
        .collect();
    if !stranded.is_empty() {
        bail!(
            "`{}`: no way to reach {} from `{}`",
            area.id,
            stranded.join(", "),
            start.id
        );
    }
    Ok(())
}

/// `contains`, `within` and portals all name areas that have to exist.
fn check_joins(set: &MapSet) -> Result<()> {
    let known: HashMap<&str, &Area> = set.areas().map(|a| (a.id.as_str(), a)).collect();

    for area in set.areas() {
        for child in &area.contains {
            let Some(c) = known.get(child.as_str()) else {
                bail!("`{}` contains `{}`, which was not loaded", area.id, child);
            };
            // A parent claiming a child the child does not claim back is the
            // same one-way mistake as a door, one level up.
            if c.within.as_deref() != Some(area.id.as_str()) {
                bail!(
                    "`{}` contains `{}`, but `{}` says it is within {:?}",
                    area.id,
                    child,
                    child,
                    c.within
                );
            }
        }

        for portal in &area.portals {
            for end in &portal.between {
                let Some((area_id, node_id)) = MapSet::split_ref(end) else {
                    bail!("`{}`: portal end `{}` is not `area/node`", area.id, end);
                };
                let Some(target) = known.get(area_id) else {
                    bail!(
                        "`{}`: portal reaches `{}`, which was not loaded",
                        area.id,
                        area_id
                    );
                };
                if target.node(node_id).is_none() {
                    bail!(
                        "`{}`: portal reaches `{}` in `{}`, which has no such node",
                        area.id,
                        node_id,
                        area_id
                    );
                }
            }
        }

        // A destination that resolves to nothing is worse than none at all: a
        // body would ask to teleport, be told there is nowhere to go, and have
        // no way to find out that the map meant there to be.
        if let Some(to) = &area.teleport_to {
            let Some((area_id, node_id)) = MapSet::split_ref(to) else {
                bail!("`{}`: `teleport_to` `{}` is not `area/node`", area.id, to);
            };
            match known.get(area_id) {
                None => bail!(
                    "`{}`: `teleport_to` reaches `{}`, which was not loaded",
                    area.id,
                    area_id
                ),
                Some(target) if target.node(node_id).is_none() => bail!(
                    "`{}`: `teleport_to` reaches `{}` in `{}`, which has no such node",
                    area.id,
                    node_id,
                    area_id
                ),
                _ => {}
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::{AreaKind, Node, Spine};

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

    fn area(nodes: Vec<Node>) -> Area {
        Area {
            id: "a".into(),
            kind: AreaKind::Level,
            name: "A".into(),
            within: None,
            ordinal: None,
            summary: "s".into(),
            character: None,
            lacks: vec![],
            contains: vec![],
            portals: vec![],
            arrival: None,
            teleport_to: None,
            spine: None,
            nodes,
        }
    }

    #[test]
    fn a_room_nobody_attached_is_refused() {
        let err = MapSet::from_areas([area(vec![
            node("core", NodeKind::Core, &["hall"]),
            node("hall", NodeKind::Passage, &[]),
            node("sealed", NodeKind::Work, &[]),
        ])])
        .unwrap_err()
        .to_string();
        assert!(err.contains("opens onto nothing"), "{err}");
    }

    #[test]
    fn a_wing_reachable_only_from_itself_is_refused() {
        // Both rooms have doors, so the cheap check passes; only the traversal
        // notices the pair is an island.
        let err = MapSet::from_areas([area(vec![
            node("core", NodeKind::Core, &["hall"]),
            node("hall", NodeKind::Passage, &[]),
            node("far", NodeKind::Work, &["further"]),
            node("further", NodeKind::Work, &[]),
        ])])
        .unwrap_err()
        .to_string();
        assert!(err.contains("no way to reach"), "{err}");
    }

    #[test]
    fn a_spine_that_names_a_room_is_refused() {
        let mut a = area(vec![
            node("core", NodeKind::Core, &["hall"]),
            node("hall", NodeKind::Passage, &[]),
            node("shop", NodeKind::Work, &["hall"]),
        ]);
        a.spine = Some(Spine {
            name: "the run".into(),
            loops: false,
            through: vec!["hall".into(), "shop".into()],
        });
        let err = MapSet::from_areas([a]).unwrap_err().to_string();
        assert!(err.contains("rather than a passage"), "{err}");
    }

    #[test]
    fn a_sightline_into_nowhere_is_refused() {
        let mut a = area(vec![
            node("core", NodeKind::Core, &["hall"]),
            node("hall", NodeKind::Passage, &[]),
        ]);
        a.nodes[0].sees = vec!["elsewhere".into()];
        let err = MapSet::from_areas([a]).unwrap_err().to_string();
        assert!(err.contains("sees `elsewhere`"), "{err}");
    }

    #[test]
    fn a_child_that_does_not_claim_its_parent_is_refused() {
        let mut parent = area(vec![node("core", NodeKind::Core, &[])]);
        parent.id = "p".into();
        parent.kind = AreaKind::Building;
        parent.contains = vec!["c".into()];
        let mut child = area(vec![node("core", NodeKind::Core, &[])]);
        child.id = "c".into();
        let err = MapSet::from_areas([parent, child]).unwrap_err().to_string();
        assert!(err.contains("says it is within"), "{err}");
    }

    #[test]
    fn a_portal_to_a_missing_node_is_refused() {
        let mut parent = area(vec![node("core", NodeKind::Core, &[])]);
        parent.id = "p".into();
        parent.kind = AreaKind::Building;
        parent.portals = vec![crate::schema::Portal {
            between: ["p/core".into(), "p/attic".into()],
            kind: "a lift".into(),
        }];
        let err = MapSet::from_areas([parent]).unwrap_err().to_string();
        assert!(err.contains("no such node"), "{err}");
    }
}
