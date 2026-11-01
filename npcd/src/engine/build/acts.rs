//! The acts by which the world builds itself (`docs/zen_moon_mind.md` §14):
//! every one [`Availability::Granted`] — offered only while a mission granting
//! it is carried, in that mission's room, in the mirror of a real place.

use crate::engine::tools::{Availability, Example, Param, Plane, Tool};

/// A world-building act's parameter.
const fn param(name: &'static str, required: bool, description: &'static str) -> Param {
    Param {
        name,
        ty: "string",
        required,
        description,
    }
}

pub const ONBOARD_PART: Tool = Tool {
    name: "onboard_part",
    at: &[],
    category: "Building",
    plane: Plane::World,
    availability: Availability::Granted,
    description: "Put a device the survey found into this room as what it is: a part of the room \
                  that stands for it, named the way the family would name it.",
    params: &[
        param("from", true, "The device found, as the survey labelled it."),
        param(
            "called",
            true,
            "What it is called in the room, bare and singular: \"Sonos speaker\", \"Yen's phone\".",
        ),
        param(
            "description",
            true,
            "One sentence of where it is and what it is for, naming it: \"The Sonos speaker plays \
             for the whole front room.\"",
        ),
    ],
    examples: &[Example {
        situation: "The survey found a Sonos player in the front room that no part stands for.",
        call: r#"{"from":"SonosZP at 192.168.0.89","called":"Sonos speaker","description":"The Sonos speaker plays for the whole front room."}"#,
        because:
            "A device in the room the world has no part for is a device nobody in the world can \
                  see or work.",
    }],
};

pub const ONBOARD_THING: Tool = Tool {
    name: "onboard_thing",
    at: &[],
    category: "Building",
    plane: Plane::World,
    availability: Availability::Granted,
    description: "Put a thing that is really in this room, and is on no network, into the room as \
                  a part — a sofa, a lamp, a bookshelf — named the way the family would name it.",
    params: &[
        param(
            "called",
            true,
            "What it is called, bare and singular: \"sofa\", \"reading lamp\".",
        ),
        param(
            "description",
            true,
            "One sentence of where it is and what it is for, naming it: \"The sofa faces the \
             television.\"",
        ),
    ],
    examples: &[Example {
        situation:
            "The room's own description says the sofa faces the television, and the room has \
                    no part for it.",
        call: r#"{"called":"sofa","description":"The sofa faces the television across the room."}"#,
        because: "What the room is known to hold belongs in it as a part, where it can be seen.",
    }],
};

pub const LINK_PART: Tool = Tool {
    name: "link_part",
    at: &[],
    category: "Building",
    plane: Plane::World,
    availability: Availability::Granted,
    description: "Say that a part already in this room is a device the survey found — the room's \
                  Chromecast is the Cast device at that address — so the part stands for the real \
                  thing and the world can reach it.",
    params: &[
        param("part", true, "The part in the room, by name."),
        param(
            "from",
            true,
            "The device found that it is, as the survey labelled it.",
        ),
    ],
    examples: &[Example {
        situation:
            "The room has a Chromecast and the survey found a Cast device the room has no other \
                    part for.",
        call: r#"{"part":"Chromecast","from":"Family Room TV at 192.168.0.214"}"#,
        because: "Two parts for one thing would be one too many; the room's part is the device.",
    }],
};

pub const OFFBOARD_PART: Tool = Tool {
    name: "offboard_part",
    at: &[],
    category: "Building",
    plane: Plane::World,
    availability: Availability::Granted,
    description: "Take out of this room a device onboarded from a survey that is no longer there.",
    params: &[
        param("part", true, "The onboarded device in this room, by name."),
        param(
            "why",
            true,
            "Why it is gone: what the survey showed, or what you were told.",
        ),
    ],
    examples: &[Example {
        situation: "The latest survey did not find the old laptop the room still has a part for.",
        call: r#"{"part":"old laptop","why":"It was not on the network in today's survey, and the family said it was given away."}"#,
        because: "A room that keeps what is gone describes a house that is not there.",
    }],
};

pub const DISCOVER_DOOR: Tool = Tool {
    name: "discover_door",
    at: &[],
    category: "Building",
    plane: Plane::World,
    availability: Availability::Granted,
    description:
        "Add a room of the house that opens off this one, and the door to it — a real room \
                  of this house, never anywhere else.",
    params: &[
        param(
            "room",
            true,
            "What the room is called, as it appears mid-sentence: \"the kitchen\".",
        ),
        param(
            "description",
            true,
            "What the room is and what it is like, in a sentence or two.",
        ),
    ],
    examples: &[Example {
        situation: "The front room opens north into the kitchen, and the house has no room for it.",
        call: r#"{"room":"the kitchen","description":"The kitchen, behind the front room, where the family cooks."}"#,
        because:
            "A house the world knows only one room of is a house it cannot follow anybody into.",
    }],
};

pub const UNDISCOVER_DOOR: Tool = Tool {
    name: "undiscover_door",
    at: &[],
    category: "Building",
    plane: Plane::World,
    availability: Availability::Granted,
    description:
        "Take away a room that opens off this one and is not in the house — nobody may be in \
                  it.",
    params: &[
        param("room", true, "The room off this one, by name."),
        param("why", true, "Why it is not there."),
    ],
    examples: &[Example {
        situation: "A room was added for a study the family says the apartment does not have.",
        call: r#"{"room":"the study","why":"The family says there is no study; it was a mistake."}"#,
        because: "A room that is not in the house misleads everybody who walks into it.",
    }],
};

pub const DESCRIBE_ROOM: Tool = Tool {
    name: "describe_room",
    at: &[],
    category: "Building",
    plane: Plane::World,
    availability: Availability::Granted,
    description:
        "Rewrite what this room is like, as you now know it to be: what it is for, how it is \
                  used, what is true of it that its parts do not say.",
    params: &[param(
        "description",
        true,
        "What the room is like, in two or three sentences.",
    )],
    examples: &[Example {
        situation: "Having surveyed the room, you know far more of what it is used for than its \
                    description says.",
        call: r#"{"description":"The family's main room: two workstations and a gaming machine along the east wall, the television and its Chromecast on the west, and music from the Sonos all day."}"#,
        because: "What the room is like is what everybody reads first on coming in.",
    }],
};

/// Every world-building act, for the catalog.
pub const BUILD_ACTS: &[Tool] = &[
    ONBOARD_PART,
    ONBOARD_THING,
    LINK_PART,
    OFFBOARD_PART,
    DISCOVER_DOOR,
    UNDISCOVER_DOOR,
    DESCRIBE_ROOM,
];

/// Whether `tool` is a world-building act — enacted by the runtime, which
/// keeps what it changes.
pub fn is_mine(tool: &str) -> bool {
    BUILD_ACTS.iter().any(|t| t.name == tool)
}
