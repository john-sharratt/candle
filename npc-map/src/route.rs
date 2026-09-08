//! How to get from one place to another, and how far it is.
//!
//! A pure question about the map: nobody is standing anywhere in this module,
//! and nothing here changes. [`crate::world`] owns who is walking and how far
//! along they are; this owns the shape of the building they are walking
//! through.
//!
//! # A route crosses levels
//!
//! Doors join rooms within a level and portals join levels to each other, and
//! a body walking from an easel to the command room does not care which kind of
//! link it is crossing. So both are edges of one graph, and a route is a list
//! of places rather than a list of rooms plus a lift ride.
//!
//! # Distance is the whole point
//!
//! Every step of a route is a step a body has to take, one per
//! [`crate::world::World::step`]. That is what makes the vault a building
//! rather than a menu: the green room is near the casting bands and the command
//! room is eleven moves from an easel, so who a Maker runs into depends on where
//! it works, and going upstairs to ask something costs enough to be worth
//! thinking about. Everything social in the design rests on this number being
//! real.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::load::MapSet;
use crate::schema::Where;

/// Everywhere reachable from a place in one move — the doors of its own area
/// and any portal leading out of it.
pub fn steps_from(map: &MapSet, at: &Where) -> Vec<Where> {
    let mut out: Vec<Where> = match map.node_at(at) {
        Some(node) => node
            .exits
            .iter()
            .map(|id| Where::new(at.area.clone(), id.clone()))
            .collect(),
        None => Vec::new(),
    };
    out.extend(map.ways_from(at).iter().cloned());
    out
}

/// The shortest way between two places, both ends included.
///
/// `None` when there is no way at all; a one-element route when you are asking
/// about where you already are. Breadth-first, so the answer is the fewest
/// moves — which is also the fewest ticks, since a move is a tick.
pub fn route(map: &MapSet, from: &Where, to: &Where) -> Option<Vec<Where>> {
    map.node_at(from)?;
    map.node_at(to)?;
    if from == to {
        return Some(vec![from.clone()]);
    }

    let mut came: BTreeMap<Where, Where> = BTreeMap::new();
    let mut seen: BTreeSet<Where> = BTreeSet::from([from.clone()]);
    let mut queue: VecDeque<Where> = VecDeque::from([from.clone()]);

    while let Some(here) = queue.pop_front() {
        for next in steps_from(map, &here) {
            if !seen.insert(next.clone()) {
                continue;
            }
            came.insert(next.clone(), here.clone());
            if &next == to {
                let mut path = vec![to.clone()];
                let mut step = to;
                while let Some(prev) = came.get(step) {
                    path.push(prev.clone());
                    step = prev;
                }
                path.reverse();
                return Some(path);
            }
            queue.push_back(next);
        }
    }
    None
}

/// How many moves a journey takes. Zero when you are already there.
pub fn distance(map: &MapSet, from: &Where, to: &Where) -> Option<usize> {
    route(map, from, to).map(|r| r.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vault() -> MapSet {
        MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/maps")).expect("the vault must load")
    }

    fn at(area: &str, node: &str) -> Where {
        Where::new(area, node)
    }

    #[test]
    fn a_route_to_where_you_stand_is_no_distance_at_all() {
        let m = vault();
        let here = at("vault-casting", "band-one");
        assert_eq!(route(&m, &here, &here).unwrap(), vec![here.clone()]);
        assert_eq!(distance(&m, &here, &here), Some(0));
    }

    #[test]
    fn a_lift_is_a_step_like_any_other() {
        let m = vault();
        let from = at("vault-casting", "core");
        assert!(steps_from(&m, &from).contains(&at("vault-cartography", "core")));
        assert!(steps_from(&m, &from).contains(&at("vault-portraits", "core")));
    }

    #[test]
    fn a_route_climbs_through_the_cores_and_nowhere_else() {
        let m = vault();
        let path = route(
            &m,
            &at("vault-portraits", "north-studio"),
            &at("vault-command", "command-room"),
        )
        .expect("the vault is one connected building");

        // Every change of level happens at a core, because that is the only
        // place the building has a way between them.
        for pair in path.windows(2) {
            if pair[0].area != pair[1].area {
                assert_eq!(
                    pair[0].node, "core",
                    "left {} other than by the lift",
                    pair[0]
                );
                assert_eq!(
                    pair[1].node, "core",
                    "reached {} other than by the lift",
                    pair[1]
                );
            }
        }
        assert_eq!(path.first().unwrap().node, "north-studio");
        assert_eq!(path.last().unwrap().node, "command-room");
    }

    #[test]
    fn every_place_in_the_vault_can_be_walked_to_from_every_other() {
        let m = vault();
        let all: Vec<Where> = m
            .children("creators-vault")
            .iter()
            .flat_map(|a| a.nodes.iter().map(|n| Where::new(&a.id, &n.id)))
            .collect();
        for from in &all {
            for to in &all {
                assert!(
                    distance(&m, from, to).is_some(),
                    "no way from {from} to {to}"
                );
            }
        }
    }

    #[test]
    fn distance_is_what_makes_the_building_worth_crossing() {
        let m = vault();
        // Two rooms off the same cross run are next door but one. The far
        // corner of the top level to the far corner of the bottom is a
        // journey. If those two numbers were alike, being on one level rather
        // than another would mean nothing.
        let near = distance(
            &m,
            &at("vault-casting", "band-two"),
            &at("vault-casting", "band-three"),
        )
        .unwrap();
        let far = distance(
            &m,
            &at("vault-portraits", "north-studio"),
            &at("vault-command", "command-room"),
        )
        .unwrap();
        assert_eq!(near, 2, "two rooms off one corridor");
        assert!(far > near * 3, "far {far}, near {near}");
    }

    #[test]
    fn a_level_is_never_more_than_a_few_rooms_across() {
        // The counterweight to the test above. Distance has to be felt within
        // a level as well as between them, but a level nobody can cross is a
        // level nobody works on — every room is within a handful of moves of
        // every other, and the lift is what costs.
        let m = vault();
        for area in m.children("creators-vault") {
            for from in &area.nodes {
                for to in &area.nodes {
                    let d = distance(
                        &m,
                        &Where::new(&area.id, &from.id),
                        &Where::new(&area.id, &to.id),
                    )
                    .expect("a level is connected");
                    assert!(d <= 5, "{}: {} to {} is {d} moves", area.id, from.id, to.id);
                }
            }
        }
    }

    #[test]
    fn there_is_no_way_to_a_place_that_does_not_exist() {
        let m = vault();
        assert!(route(
            &m,
            &at("vault-casting", "band-one"),
            &at("vault-casting", "the-moon")
        )
        .is_none());
        assert!(route(
            &m,
            &at("vault-casting", "band-one"),
            &at("the-moon", "core")
        )
        .is_none());
    }
}
