//! The act that sets the house's mode (`docs/house_modes.md` §3): offered to
//! a body that takes no orders — Zen — in the room the house's devices stand
//! in, its `mode` bound to the modes there are.

use crate::engine::tools::{Availability, Example, Param, Plane, Tool};

pub const HOUSE_MODE: Tool = Tool {
    name: "house_mode",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Set what the house is for: away (nobody home), asleep (home and asleep: nothing \
                  makes a sound), home (somebody in, nothing on), listening (music in the \
                  background), watching (the television), or party (you at the decks). Each mode \
                  sets the room's devices as the house's table says, and a party opens rekordbox \
                  on BlackWorld — holding that machine for the decks — while every other mode \
                  closes it. Set it when the family ask for it or plainly want it; leaving a party \
                  is refused while a deck is playing.",
    params: &[Param {
        name: "mode",
        ty: "string",
        required: true,
        description: "The mode: away, asleep, home, listening, watching or party.",
    }],
    examples: &[Example {
        situation: "John says the guests are arriving and asks you to get the music going.",
        call: r#"{"mode":"party"}"#,
        because: "A party is a mode of the whole house: the decks, the stereo and the screens \
                  are set together, not one device at a time.",
    }],
};

/// Every house act, for the catalog.
pub const HOUSE_ACTS: &[Tool] = &[HOUSE_MODE];

/// Whether `tool` is a house act — enacted by the runtime, which awaits the
/// devices and the decks.
pub fn is_mine(tool: &str) -> bool {
    HOUSE_ACTS.iter().any(|t| t.name == tool)
}
