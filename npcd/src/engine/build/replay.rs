//! What the world's builders made, laid over its map
//! (`docs/zen_moon_mind.md` §14): all of it when a world is loaded from its
//! authored map, and what came since when a member takes in the world's state
//! from its authority ([`crate::sim::built::Built::unlaid`]).
//!
//! **In order, each edit against the map as it then stands.** An edit the map
//! already holds — because its owner published it into the authored map — is
//! passed over, so a build laid over a map that already has it changes
//! nothing.

use npc_map::schema::Where;
use npc_map::world::World;
use npc_map::MapEdit;

/// Whether `world`'s map already holds what `edit` makes.
pub fn holds(world: &World, edit: &MapEdit) -> bool {
    let map = world.map();
    let node = |at: &Where| map.get(&at.area).and_then(|a| a.node(&at.node));
    match edit {
        MapEdit::DefinePart(part) => map.part(&part.id).is_some(),
        MapEdit::LinkPart { part, link } => {
            map.part(part).and_then(|p| p.link.as_ref()) == Some(link)
        }
        MapEdit::PlacePart { at, placement } => {
            node(at).is_some_and(|n| n.parts.iter().any(|p| p.part() == placement.part()))
        }
        MapEdit::UnplacePart { at, part } => {
            node(at).is_none_or(|n| !n.parts.iter().any(|p| p.part() == part))
        }
        MapEdit::AddNode { area, node: n } => {
            map.get(area).is_some_and(|a| a.node(&n.id).is_some())
        }
        MapEdit::RemoveNode(at) => node(at).is_none(),
        MapEdit::Describe { at, character } => {
            node(at).is_some_and(|n| n.character.as_ref() == Some(character))
        }
        MapEdit::Retitle { at, name } => node(at).is_some_and(|n| &n.name == name),
        MapEdit::AddArea(a) => map.get(&a.id).is_some(),
        MapEdit::RemoveArea(id) => map.get(id).is_none(),
        MapEdit::AddPortal { .. } | MapEdit::RemovePortal { .. } => false,
    }
}

/// Lay `built` over `world`, in order, passing over what it already holds.
/// Returns how many were laid.
pub fn replay(world: &mut World, built: &[MapEdit]) -> usize {
    let mut laid = 0;
    for edit in built {
        if holds(world, edit) {
            continue;
        }
        match world.reshape(edit) {
            Ok(_) => laid += 1,
            Err(why) => tracing::warn!(?edit, %why, "a built change no longer fits the map"),
        }
    }
    laid
}

#[cfg(test)]
mod tests {
    use npc_map::part::{Part, PartKind, Placement};
    use npc_map::MapSet;

    use super::*;
    use crate::sim::built::Built;

    fn vault() -> World {
        World::new(
            MapSet::load_dir(concat!(env!("CARGO_MANIFEST_DIR"), "/../npc-map/maps")).unwrap(),
        )
    }

    fn speaker() -> Part {
        Part {
            id: "device-f0f6c1a079b4".into(),
            kind: PartKind::Fixture,
            name: "Sonos speaker".into(),
            plural: None,
            binds: None,
            short: None,
            long: "x".into(),
            modes: vec![],
            link: None,
        }
    }

    fn placed(w: &World, room: &Where) -> usize {
        w.node(room)
            .unwrap()
            .parts
            .iter()
            .filter(|p| p.part() == "device-f0f6c1a079b4")
            .count()
    }

    /// **A build laid over a map that already has it changes nothing**: a
    /// part defined is not defined again, and a part placed is not placed
    /// twice.
    #[test]
    fn laying_what_the_map_holds_changes_nothing() {
        let room = Where::new("vault-casting", "green-room");
        let built = vec![
            MapEdit::DefinePart(Box::new(speaker())),
            MapEdit::PlacePart {
                at: room.clone(),
                placement: Placement::Bare("device-f0f6c1a079b4".into()),
            },
            MapEdit::Describe {
                at: room.clone(),
                character: "Quiet.".into(),
            },
        ];
        let mut w = vault();
        assert_eq!(replay(&mut w, &built), 3);
        assert!(built.iter().all(|e| holds(&w, e)));
        assert_eq!(replay(&mut w, &built), 0, "held already");
        assert_eq!(placed(&w, &room), 1);
    }

    /// **A member lays what its authority built since, undoings too**: the
    /// speaker placed and then taken out is out on the member as well, and
    /// a member that laid it all lays nothing again.
    #[test]
    fn a_member_lays_what_was_built_since_and_its_undoing() {
        let room = Where::new("vault-casting", "green-room");
        let mut authority = Built::default();
        let mut member = vault();
        let mut members = Built::default();
        for edit in [
            MapEdit::DefinePart(Box::new(speaker())),
            MapEdit::PlacePart {
                at: room.clone(),
                placement: Placement::Bare("device-f0f6c1a079b4".into()),
            },
        ] {
            authority.laid_now(edit);
        }
        let mut taken = authority.clone();
        taken.carry_laid(&members);
        members = taken;
        replay(&mut member, &members.unlaid());
        assert_eq!(placed(&member, &room), 1);

        authority.laid_now(MapEdit::UnplacePart {
            at: room.clone(),
            part: "device-f0f6c1a079b4".into(),
        });
        let mut taken = authority.clone();
        taken.carry_laid(&members);
        members = taken;
        let fresh = members.unlaid();
        assert_eq!(fresh.len(), 1, "only what came since");
        replay(&mut member, &fresh);
        assert_eq!(placed(&member, &room), 0, "taken out on the member too");
        assert_eq!(members.unlaid(), []);
    }
}
