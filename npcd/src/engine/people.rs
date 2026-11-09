//! Who in a world is a person (`docs/npc_conversations.md` §12): room speech
//! is for people, so the grammar asks whether one is here, or within a
//! shout's reach.
//!
//! A person is a `visitor:{handle}` body in every world — placed by a console
//! session, or by the Battle Cities server for a tower's main avatar.

use npc_map::witness::{carries, Scope};
use npc_map::world::{Voice, Where, World};

use crate::engine::talk::key::VISITOR;

/// Whether a body is a person's.
pub fn is_person(body: &str) -> bool {
    body.strip_prefix(VISITOR).is_some_and(|h| !h.is_empty())
}

/// Whether a shout from `here` reaches any person's body, by perception's own
/// rule — so the grammar never offers a shout that nobody would hear, nor
/// withholds one that somebody would.
pub fn hear_a_shout(world: &World, here: &Where) -> bool {
    world
        .actors()
        .filter(|a| is_person(&a.id))
        .any(|a| carries(Voice::Shouted, Scope::at(world, &a.at).reach(here), false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_person_is_a_visitor_body_with_a_handle() {
        assert!(is_person("visitor:u_8812"));
        assert!(!is_person("visitor:"));
        assert!(!is_person("npc-12"));
        assert!(!is_person("zen"));
    }
}
