//! What a body can do because of where it is standing.
//!
//! The catalog in [`crate::engine::tools`] is what a body can do *anywhere* —
//! speak, move, wait. This is the other half: the things it can do only here,
//! because they work through something standing in the room with it.
//!
//! # Why these are not catalog entries
//!
//! A terminal's tools are **map data**, not code. A part is defined once in the
//! world's files and every room holding one offers the same tools with the same
//! provenance, including rooms written later — so what is reachable is a
//! function of the world, and a compiled-in list would be a second copy of it,
//! free to disagree.
//!
//! That is also why they arrive as identifiers with a sentence rather than as
//! parameter schemas. A part says what it is and what it lets you do; what the
//! call looks like is the business of whatever implements it.
//!
//! # Walking away takes them with you
//!
//! This is the whole payoff of hanging tools on parts rather than on
//! characters. A corridor offers nothing. A Maker at an easel cannot write a
//! chronicle entry — not because a rule forbids it, but because the thing that
//! does that is two floors up. A tool a body is not standing next to is never
//! offered and so cannot be reasoned about wrongly, which is a stronger
//! guarantee than refusing the call would be.

use npc_map::world::World;

use crate::world::Hosted;

/// Something within reach, and what standing next to it lets a body do.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Within {
    /// The part, as the character would name it: "a character terminal".
    pub thing: String,
    /// How many of them are here.
    pub count: u32,
    /// What it is and what it does, authored beside the part so every room
    /// holding one describes it identically.
    pub about: String,
    /// The tools it carries, by identifier.
    pub tools: Vec<String>,
}

/// Everything within reach of a body, in the order the room places it.
///
/// Empty for a corridor, and empty for a body that is not in the world — both
/// of which are the honest answer rather than an error, because "nothing here"
/// is a thing a character routinely stands in the middle of.
pub fn within(hosted: &Hosted, body: &str) -> Vec<Within> {
    hosted.read(|world| in_world(world, body))
}

/// [`within`], against a world the caller is already holding.
///
/// The primitive, because composing a situation happens under the world's lock
/// and taking it again from inside would be a deadlock rather than a slowdown.
pub fn in_world(world: &World, body: &str) -> Vec<Within> {
    let Some(actor) = world.actor(body) else {
        return Vec::new();
    };
    let Some(node) = world.node(&actor.at) else {
        return Vec::new();
    };
    world
        .map()
        .parts_at(node)
        .filter(|(part, _)| !part.tools.is_empty())
        .map(|(part, count)| Within {
            thing: part.count_name(count),
            count,
            about: part.long.clone(),
            tools: part.tools.clone(),
        })
        .collect()
}

/// What is within reach, as a line a character reads.
///
/// Part of the **situation**, not of the system prompt: what a body can do is a
/// function of where it stands, so it changes every time the body moves and a
/// prompt built once per conversation could not carry it. It supersedes with
/// the rest of the situation, which is right — the two always change together.
///
/// Nothing when the room offers nothing, because a corridor saying so every
/// turn is a sentence that adds no fact.
pub fn line(world: &World, body: &str) -> Option<String> {
    let here = in_world(world, body);
    if here.is_empty() {
        return None;
    }
    let things: Vec<String> = here.into_iter().map(|w| w.thing).collect();
    Some(format!("Within reach: {}.", npc_map::text::list(&things)))
}

/// Just the tool identifiers within reach, deduplicated, in a stable order.
///
/// What a caller offering a vocabulary wants; [`within`] is what a caller
/// *describing* the room wants.
pub fn tools(hosted: &Hosted, body: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for part in within(hosted, body) {
        for tool in part.tools {
            if !out.contains(&tool) {
                out.push(tool);
            }
        }
    }
    out
}

/// Whether a tool is reachable from where a body stands.
///
/// The check a caller makes before performing one — and the reason it can be
/// made at all is that reach is computed from the world rather than declared
/// about the character.
pub fn can_reach(hosted: &Hosted, body: &str, tool: &str) -> bool {
    tools(hosted, body).iter().any(|t| t == tool)
}

#[cfg(test)]
mod tests {
    use super::*;
    use npc_map::world::Where;

    fn vault() -> Hosted {
        Hosted::load(
            "creators-vault",
            concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps"),
        )
        .expect("the shipped vault must load")
    }

    fn at(node: &str) -> Where {
        Where::new("vault-casting", node)
    }

    fn standing(node: &str) -> Hosted {
        let h = vault();
        h.with(|w| w.enter("m1", "Maker-01", at(node)).unwrap());
        h
    }

    #[test]
    fn a_working_room_offers_what_stands_in_it() {
        let h = standing("band-one");
        let here = within(&h, "m1");
        assert!(!here.is_empty());

        let terminal = here
            .iter()
            .find(|w| w.thing.contains("terminal"))
            .expect("band one has terminals");
        assert_eq!(terminal.count, 6);
        assert!(terminal.tools.iter().any(|t| t.starts_with("character.")));
        assert!(!terminal.about.is_empty(), "a part with no provenance");
    }

    #[test]
    fn a_corridor_offers_nothing_at_all() {
        // The payoff of hanging tools on parts: walking away takes them with
        // you, and there is nothing to reason wrongly about.
        for run in ["ring-north", "ring-south", "cross-upper"] {
            let h = standing(run);
            assert!(within(&h, run).is_empty(), "{run}");
            assert!(tools(&h, "m1").is_empty(), "{run}");
        }
    }

    #[test]
    fn walking_out_of_a_room_takes_its_tools_with_you() {
        let h = standing("band-one");
        assert!(can_reach(&h, "m1", "character.write_identity"));

        h.with(|w| w.set_off("m1", at("ring-north")).unwrap());
        h.tick();
        assert!(
            !can_reach(&h, "m1", "character.write_identity"),
            "a terminal followed the body out of the room"
        );
    }

    #[test]
    fn a_maker_at_an_easel_cannot_reach_what_is_two_floors_up() {
        // Not because a rule forbids it — because the thing that does it is
        // somewhere else, which is a stronger guarantee than a refusal.
        let h = vault();
        h.with(|w| {
            w.enter(
                "painter",
                "Maker-01",
                Where::new("vault-portraits", "north-studio"),
            )
            .unwrap()
        });
        let reachable = tools(&h, "painter");
        assert!(!reachable.is_empty(), "the studio offers nothing at all");
        assert!(
            !reachable.iter().any(|t| t.starts_with("chronicle.")),
            "{reachable:?}"
        );
    }

    #[test]
    fn every_room_that_offers_a_tool_says_what_the_thing_is() {
        // A tool identifier with no provenance is a verb with no subject. The
        // sentence beside it is what makes it usable, and it is authored once
        // per part rather than per room.
        let h = vault();
        let places: Vec<Where> = h.read(|w| {
            w.map()
                .children("creators-vault")
                .iter()
                .flat_map(|a| {
                    a.nodes
                        .iter()
                        .map(|n| Where::new(a.id.clone(), n.id.clone()))
                        .collect::<Vec<_>>()
                })
                .collect()
        });
        let mut rooms_with_tools = 0;
        for place in places {
            h.with(|w| w.enter("wanderer", "The Wanderer", place.clone()).unwrap());
            for part in within(&h, "wanderer") {
                rooms_with_tools += 1;
                assert!(!part.thing.is_empty(), "{place}");
                assert!(part.about.len() > 20, "{place}: {}", part.about);
                assert!(!part.tools.is_empty(), "{place}");
            }
        }
        assert!(rooms_with_tools > 10, "only {rooms_with_tools} checked");
    }

    #[test]
    fn the_same_part_in_two_rooms_offers_the_same_thing() {
        // Defined once, so every room holding one describes it identically —
        // including rooms written later.
        let h = vault();
        h.with(|w| {
            w.enter("a", "Maker-01", at("band-one")).unwrap();
            w.enter("b", "Maker-02", at("band-two")).unwrap();
        });
        let one = within(&h, "a");
        let two = within(&h, "b");
        let terminal = |v: &[Within]| {
            v.iter()
                .find(|w| w.thing.contains("terminal"))
                .map(|w| (w.about.clone(), w.tools.clone()))
        };
        assert_eq!(terminal(&one), terminal(&two));
    }

    #[test]
    fn tool_identifiers_come_back_once_and_in_a_stable_order() {
        let h = standing("band-one");
        let first = tools(&h, "m1");
        let mut sorted = first.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(first.len(), sorted.len(), "a tool was offered twice");
        assert_eq!(tools(&h, "m1"), first, "the order moved between reads");
    }

    #[test]
    fn a_body_that_is_not_in_the_world_reaches_nothing() {
        let h = vault();
        assert!(within(&h, "nobody").is_empty());
        assert!(tools(&h, "nobody").is_empty());
        assert!(!can_reach(&h, "nobody", "character.read"));
    }
}
