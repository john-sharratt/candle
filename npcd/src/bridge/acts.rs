//! The acts the house's bridges offer (`docs/zen_moon_mind.md` §13, §15),
//! each bound by its one required argument to what stands in the room — a
//! house link, a part linked to a Sonos player — so it is offered only where
//! the thing it works is.

use crate::engine::tools::{Availability, Example, Param, Plane, Tool};

pub const SURVEY_NETWORK: Tool = Tool {
    name: "survey_network",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description:
        "Survey the house network from the house link here: every device on it, what each \
                  says it is, and whether the world can work it. What it finds stays with this \
                  room, to put each device in it as what it is.",
    params: &[Param {
        name: "at",
        ty: "string",
        required: true,
        description: "The house link to survey from.",
    }],
    examples: &[Example {
        situation: "You have been sent to find every device in this room, and there is a house \
                    link here.",
        call: r#"{"at":"house link"}"#,
        because: "A device you have not found cannot be put in the room; the survey is where \
                  finding starts.",
    }],
};

/// A Sonos act that names only its player.
macro_rules! sonos {
    ($name:literal, $desc:literal, $situation:literal, $because:literal) => {
        Tool {
            name: $name,
            at: &[],
            category: "House",
            plane: Plane::World,
            availability: Availability::Embodied,
            description: $desc,
            params: &[Param {
                name: "device",
                ty: "string",
                required: true,
                description: "The Sonos player here, by name.",
            }],
            examples: &[Example {
                situation: $situation,
                call: r#"{"device":"Sonos speaker"}"#,
                because: $because,
            }],
        }
    };
}

pub const SONOS_STATUS: Tool = sonos!(
    "sonos_status",
    "Read what the Sonos player here is doing: playing or paused, what, and how loud.",
    "You want to know whether music is on in the room, and what.",
    "What the room is listening to is in the player, not in anybody's guess."
);

pub const SONOS_PLAY: Tool = sonos!(
    "sonos_play",
    "Start the Sonos player here playing what it has queued, where it last left off.",
    "The family asked for the music back on.",
    "The player plays what it holds; starting it is the whole act."
);

pub const SONOS_PAUSE: Tool = sonos!(
    "sonos_pause",
    "Pause the Sonos player here, so the room is quiet until somebody plays it again.",
    "Somebody is on a call and the music is in the way.",
    "Paused, it picks up where it was."
);

pub const SONOS_VOLUME: Tool = Tool {
    name: "sonos_volume",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description:
        "Set how loud the Sonos player here plays, from 0 (silent) to 100 (as loud as it goes).",
    params: &[
        Param {
            name: "device",
            ty: "string",
            required: true,
            description: "The Sonos player here, by name.",
        },
        Param {
            name: "volume",
            ty: "integer",
            required: true,
            description: "How loud, from 0 to 100; 15 is quiet, 60 fills the room.",
        },
    ],
    examples: &[Example {
        situation: "It is late and the music is loud enough to keep the house up.",
        call: r#"{"device":"Sonos speaker","volume":15}"#,
        because: "Loudness is a number the player holds; set it, do not ask somebody to.",
    }],
};

/// Every bridge act, for the catalog.
pub const BRIDGE_ACTS: &[Tool] = &[
    SURVEY_NETWORK,
    SONOS_STATUS,
    SONOS_PLAY,
    SONOS_PAUSE,
    SONOS_VOLUME,
];

/// Whether `tool` is a bridge act — enacted by the runtime, which awaits the
/// network, and not by the world's synchronous dispatch.
pub fn is_mine(tool: &str) -> bool {
    BRIDGE_ACTS.iter().any(|t| t.name == tool)
}
