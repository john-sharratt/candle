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

/// An AV-receiver act that names only the receiver it works.
macro_rules! receiver {
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
                description: "The AV receiver here, by name, as the room calls it.",
            }],
            examples: &[Example {
                situation: $situation,
                call: r#"{"device":"the stereo"}"#,
                because: $because,
            }],
        }
    };
}

pub const RECEIVER_STATUS: Tool = receiver!(
    "receiver_status",
    "Read what the AV receiver here is doing: on or in standby, how loud, and whether it is muted.",
    "You want to know whether the stereo is on in the room, and how loud.",
    "What the receiver is doing is in the receiver, not in anybody's guess."
);

pub const RECEIVER_ON: Tool = receiver!(
    "receiver_on",
    "Turn the AV receiver here on, out of standby, so the room can play sound through it.",
    "The family want to put music or a film on and the stereo is in standby.",
    "The receiver is off; turning it on is the whole act, and nobody is there to press it."
);

pub const RECEIVER_OFF: Tool = receiver!(
    "receiver_off",
    "Put the AV receiver here into standby, so it draws little and the room is quiet.",
    "The room is empty and the stereo was left on.",
    "A receiver left on all night is a receiver nobody turned off; standby is the act."
);

pub const RECEIVER_MUTE: Tool = receiver!(
    "receiver_mute",
    "Mute the AV receiver here, so its sound stops at once without changing anything else.",
    "The phone is ringing and the room needs to be quiet this moment.",
    "Muting stops the sound and keeps the volume, so unmuting brings it back as it was."
);

pub const RECEIVER_UNMUTE: Tool = receiver!(
    "receiver_unmute",
    "Unmute the AV receiver here, bringing its sound back at the volume it had.",
    "The call is over and the room wants its sound back.",
    "The receiver is muted; unmuting is the act that undoes it."
);

pub const RECEIVER_VOLUME: Tool = Tool {
    name: "receiver_volume",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description:
        "Set how loud the AV receiver here plays, from 0 (silent) to 100 (as loud as it goes).",
    params: &[
        Param {
            name: "device",
            ty: "string",
            required: true,
            description: "The AV receiver here, by name, as the room calls it.",
        },
        Param {
            name: "volume",
            ty: "integer",
            required: true,
            description: "How loud, from 0 to 100; 25 is quiet, 60 fills the room.",
        },
    ],
    examples: &[Example {
        situation: "The stereo is louder than the room needs.",
        call: r#"{"device":"the stereo","volume":30}"#,
        because: "Loudness is a number the receiver holds; set it, do not ask somebody to.",
    }],
};

pub const WAKE_DEVICE: Tool = Tool {
    name: "wake_device",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Wake a device here that is asleep — a PC or a laptop — by its hardware address, so it comes back on the network with nobody at it.",
    params: &[Param {
        name: "device",
        ty: "string",
        required: true,
        description: "The device here to wake, by name, as the room calls it.",
    }],
    examples: &[Example {
        situation: "A workstation here is asleep and something needs it awake.",
        call: r#"{"device":"John's workstation"}"#,
        because: "A sleeping machine wakes on a magic packet, which needs nothing switched on at it.",
    }],
};

/// Every bridge act, for the catalog.
pub const BRIDGE_ACTS: &[Tool] = &[
    SURVEY_NETWORK,
    SONOS_STATUS,
    SONOS_PLAY,
    SONOS_PAUSE,
    SONOS_VOLUME,
    RECEIVER_STATUS,
    RECEIVER_ON,
    RECEIVER_OFF,
    RECEIVER_MUTE,
    RECEIVER_UNMUTE,
    RECEIVER_VOLUME,
    WAKE_DEVICE,
];

/// Whether `tool` is a bridge act — enacted by the runtime, which awaits the
/// network, and not by the world's synchronous dispatch.
pub fn is_mine(tool: &str) -> bool {
    BRIDGE_ACTS.iter().any(|t| t.name == tool)
}
