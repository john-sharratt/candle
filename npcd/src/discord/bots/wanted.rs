//! Which bots this member runs (`docs/npcd_discord.md` §3.3): every
//! connected bot of a character with a body in a world this member is the
//! authority of, but one whose token Discord refused.
//!
//! Two members never hold one world, so this is what keeps a bot at the one
//! gateway connection Discord allows per token.

use std::collections::BTreeMap;

use crate::api::Authored;
use crate::discord::state;
use crate::discord::store::{self, Bot};

/// A bot this member should run, and the world it runs for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Wanted {
    pub world: String,
    pub bot: Bot,
}

/// Every bot this member should run now, by character.
pub fn wanted(a: &Authored) -> BTreeMap<u64, Wanted> {
    let Some(rt) = a.runtime.as_ref() else {
        return BTreeMap::new();
    };
    let shared = a.npcs.substrate();
    let mut out = BTreeMap::new();
    for world in rt.hosted.ids() {
        if !rt.standing.role(&world).is_authority() {
            continue;
        }
        for (npc, _) in rt.bodies.in_world(&world) {
            let Some(bot) = store::get(&shared, npc) else {
                continue;
            };
            if state::refused(&shared, npc, &bot) {
                continue;
            }
            out.insert(
                npc,
                Wanted {
                    world: world.clone(),
                    bot,
                },
            );
        }
    }
    out
}
