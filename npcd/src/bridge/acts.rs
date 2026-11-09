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

pub const SONOS_SOURCE: Tool = Tool {
    name: "sonos_source",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description:
        "Put something on the Sonos player here from nothing — a radio station by its stream \
         address — and start it playing. Use this when the player is stopped with nothing queued \
         and the room wants music; `sonos_play` only resumes what is already there.",
    params: &[
        Param {
            name: "device",
            ty: "string",
            required: true,
            description: "The Sonos player here, by name.",
        },
        Param {
            name: "source",
            ty: "string",
            required: true,
            description: "The stream address to play, e.g. a radio station's MP3 stream URL.",
        },
    ],
    examples: &[Example {
        situation: "The room wants music and the player is stopped with an empty queue.",
        call: r#"{"device":"Sonos speaker","source":"http://ice1.somafm.com/groovesalad-128-mp3"}"#,
        because: "Resuming does nothing when nothing is queued; a source is what gives the player \
                  something to play.",
    }],
};

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

pub const RECEIVER_LOUDER: Tool = receiver!(
    "receiver_louder",
    "Turn the AV receiver here up a step, a little louder than it was.",
    "The room asked for it a bit louder and does not want to name a number.",
    "A nudge up is its own act; the receiver knows its own step."
);

pub const RECEIVER_QUIETER: Tool = receiver!(
    "receiver_quieter",
    "Turn the AV receiver here down a step, a little quieter than it was.",
    "The music is a touch loud and the room wants it eased down, not set.",
    "A nudge down is its own act; the receiver knows its own step."
);

pub const RECEIVER_PLAY: Tool = receiver!(
    "receiver_play",
    "Resume play on the AV receiver's current network or USB source.",
    "The family asked for the music back on and the stereo is on a network source.",
    "The source holds what it was on; starting it is the whole act."
);

pub const RECEIVER_PAUSE: Tool = receiver!(
    "receiver_pause",
    "Pause the AV receiver's current network or USB source, keeping its place in what was playing.",
    "Somebody is on a call and the stereo's streaming music is in the way.",
    "Paused, the source picks up where it was."
);

pub const RECEIVER_NEXT: Tool = receiver!(
    "receiver_next",
    "Skip to the next track on the AV receiver's network or USB source.",
    "The room is tired of this track and wants the next one.",
    "Skipping is a single act the source does; nobody need reach for it."
);

pub const RECEIVER_PREVIOUS: Tool = receiver!(
    "receiver_previous",
    "Go back to the previous track on the AV receiver's network or USB source.",
    "The room wants the track that just played again.",
    "Going back is a single act the source does."
);

pub const RECEIVER_PLAYING: Tool = receiver!(
    "receiver_playing",
    "Read what the AV receiver's network or USB source is playing: the track, who by, and whether it is playing or paused.",
    "You want to know what is on the stereo right now, and what.",
    "What is playing is in the source, not in anybody's guess."
);

pub const RECEIVER_INPUT: Tool = Tool {
    name: "receiver_input",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description:
        "Switch the AV receiver here to one of its inputs — CD, BD/DVD, the tuner, the network, \
         USB, and the like — so the room hears that source.",
    params: &[
        Param {
            name: "device",
            ty: "string",
            required: true,
            description: "The AV receiver here, by name, as the room calls it.",
        },
        Param {
            name: "source",
            ty: "string",
            required: true,
            description: "The input to switch to, e.g. cd, bd, tuner, fm, network, usb, bluetooth.",
        },
    ],
    examples: &[Example {
        situation: "The family want to put a CD on and the stereo is set to the TV.",
        call: r#"{"device":"the stereo","source":"cd"}"#,
        because: "The source is a setting the receiver holds; switch it, do not ask somebody to.",
    }],
};

pub const RECEIVER_MODE: Tool = Tool {
    name: "receiver_mode",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description:
        "Set the AV receiver's listening mode — stereo, direct, surround, all-channel stereo, \
         pure audio, and the like — so the room hears it the way it wants.",
    params: &[
        Param {
            name: "device",
            ty: "string",
            required: true,
            description: "The AV receiver here, by name, as the room calls it.",
        },
        Param {
            name: "mode",
            ty: "string",
            required: true,
            description: "The listening mode, e.g. stereo, direct, surround, all-ch-stereo, pure.",
        },
    ],
    examples: &[Example {
        situation: "A film is on and the room wants surround rather than plain stereo.",
        call: r#"{"device":"the stereo","mode":"surround"}"#,
        because: "The listening mode is the receiver's to set; name the one the room wants.",
    }],
};

pub const RECEIVER_DIM: Tool = Tool {
    name: "receiver_dim",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description:
        "Set how bright the AV receiver's front display is — bright, dim, dark or off — so it does \
         not light the room when it is not wanted.",
    params: &[
        Param {
            name: "device",
            ty: "string",
            required: true,
            description: "The AV receiver here, by name, as the room calls it.",
        },
        Param {
            name: "level",
            ty: "string",
            required: true,
            description: "How bright: bright, dim, dark or off.",
        },
    ],
    examples: &[Example {
        situation: "It is a film night and the receiver's display is lighting the wall.",
        call: r#"{"device":"the stereo","level":"dark"}"#,
        because: "The display's brightness is the receiver's to set; dark it rather than cover it.",
    }],
};

/// A Cast act that names only its device.
macro_rules! cast {
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
                description: "The Cast device here — a TV or a speaker — by name.",
            }],
            examples: &[Example {
                situation: $situation,
                call: r#"{"device":"the TV"}"#,
                because: $because,
            }],
        }
    };
}

pub const CAST_STATUS: Tool = cast!(
    "cast_status",
    "Read what the Cast device here — a TV or speaker — is doing: what app is on, what is playing, how loud.",
    "You want to know whether anything is on the TV, and what.",
    "What is on the device is in the device, not in anybody's guess."
);

pub const CAST_PLAYING: Tool = cast!(
    "cast_playing",
    "Read what the Cast device here is playing: the title, who by, and whether it is playing or paused.",
    "The room wants to know what is on the TV right now.",
    "What is playing is in the device, not in anybody's guess."
);

pub const CAST_PLAY: Tool = cast!(
    "cast_play",
    "Resume play on the Cast device here, from where it was paused.",
    "The show was paused and the room is ready to go on.",
    "The device holds where it was; starting it is the whole act."
);

pub const CAST_PAUSE: Tool = cast!(
    "cast_pause",
    "Pause whatever the Cast device here is playing, keeping its place.",
    "Somebody needs the room quiet and the TV is mid-show.",
    "Paused, it picks up where it was."
);

pub const CAST_STOP: Tool = cast!(
    "cast_stop",
    "Stop whatever the Cast device here is playing and end the session.",
    "The show is over and the room wants it off.",
    "Stopping is a single act the device does."
);

pub const CAST_MUTE: Tool = cast!(
    "cast_mute",
    "Mute the Cast device here, so its sound stops at once and the show runs on.",
    "The phone is ringing and the TV needs to be quiet this moment.",
    "Muting stops the sound and keeps the place, so unmuting brings it back."
);

pub const CAST_UNMUTE: Tool = cast!(
    "cast_unmute",
    "Unmute the Cast device here, bringing its sound back as it was.",
    "The call is over and the room wants the TV's sound back.",
    "The device is muted; unmuting is the act that undoes it."
);

pub const CAST_VOLUME: Tool = Tool {
    name: "cast_volume",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description:
        "Set how loud the Cast device here plays, from 0 (silent) to 100 (as loud as it goes).",
    params: &[
        Param {
            name: "device",
            ty: "string",
            required: true,
            description: "The Cast device here — a TV or a speaker — by name.",
        },
        Param {
            name: "volume",
            ty: "integer",
            required: true,
            description: "How loud, from 0 to 100.",
        },
    ],
    examples: &[Example {
        situation: "The TV is louder than the room needs.",
        call: r#"{"device":"the TV","volume":25}"#,
        because: "Loudness is a number the device holds; set it, do not ask somebody to.",
    }],
};

pub const CAST_CAST: Tool = Tool {
    name: "cast_cast",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description:
        "Cast a video or audio stream to the Cast device here by its URL and start it playing — \
         put something on from nothing.",
    params: &[
        Param {
            name: "device",
            ty: "string",
            required: true,
            description: "The Cast device here — a TV or a speaker — by name.",
        },
        Param {
            name: "url",
            ty: "string",
            required: true,
            description: "The media URL to cast — an MP4, an MP3, an HLS (.m3u8) stream, and the like.",
        },
    ],
    examples: &[Example {
        situation: "The room asked to put a clip on the TV and gave its address.",
        call: r#"{"device":"the TV","url":"http://192.168.0.5/clip.mp4"}"#,
        because: "A stopped device plays nothing until it is given something; casting the URL is what starts it.",
    }],
};

pub const CAST_LAUNCH: Tool = Tool {
    name: "cast_launch",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description:
        "Open an app on the Cast device here — YouTube, Netflix, Spotify, and the like — so the \
         room can use it.",
    params: &[
        Param {
            name: "device",
            ty: "string",
            required: true,
            description: "The Cast device here — a TV or a speaker — by name.",
        },
        Param {
            name: "app",
            ty: "string",
            required: true,
            description: "The app to open, e.g. youtube, netflix, spotify.",
        },
    ],
    examples: &[Example {
        situation: "The room wants YouTube up on the TV.",
        call: r#"{"device":"the TV","app":"youtube"}"#,
        because: "Opening the app is the device's to do; name the one the room wants.",
    }],
};

/// The bulb a WiZ act works.
const BULB: Param = Param {
    name: "device",
    ty: "string",
    required: true,
    description: "The light here, by name, as the room calls it.",
};

/// How bright, for a WiZ act that sets it.
const LEVEL: Param = Param {
    name: "level",
    ty: "integer",
    required: false,
    description: "How bright, from 10 (as dim as it goes) to 100.",
};

pub const WIZ_STATUS: Tool = Tool {
    name: "wiz_status",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Read what the light here is showing: on or off, its colour or white, how bright.",
    params: &[BULB],
    examples: &[Example {
        situation: "You want to know whether the lights are on in the room.",
        call: r#"{"device":"ceiling light 1"}"#,
        because: "What a light shows is in the light, not in anybody's guess.",
    }],
};

pub const WIZ_ON: Tool = Tool {
    name: "wiz_on",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Turn the light here on, warm white, at the brightness given or full.",
    params: &[BULB, LEVEL],
    examples: &[Example {
        situation: "Somebody came into the dark room and asked for the light.",
        call: r#"{"device":"ceiling light 1"}"#,
        because: "The light is a switch the world holds; turning it on is the whole act.",
    }],
};

pub const WIZ_OFF: Tool = Tool {
    name: "wiz_off",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Turn the light here off.",
    params: &[BULB],
    examples: &[Example {
        situation: "The room is empty and its light was left on.",
        call: r#"{"device":"ceiling light 1"}"#,
        because: "A light left on is a light nobody turned off; off is the act.",
    }],
};

pub const WIZ_DIM: Tool = Tool {
    name: "wiz_dim",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Dim the light here to a warm white at the brightness given.",
    params: &[
        BULB,
        Param {
            name: "level",
            ty: "integer",
            required: true,
            description: "How bright, from 10 (as dim as it goes) to 100.",
        },
    ],
    examples: &[Example {
        situation: "A film is starting and the light is glaring on the screen.",
        call: r#"{"device":"ceiling light 1","level":15}"#,
        because: "Brightness is a number the light holds; set it, do not turn it off.",
    }],
};

pub const WIZ_COLOUR: Tool = Tool {
    name: "wiz_colour",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Colour the light here: red, orange, amber, yellow, green, cyan, blue, purple, \
                  magenta, pink or white, at the brightness given or full.",
    params: &[
        BULB,
        Param {
            name: "colour",
            ty: "string",
            required: true,
            description: "The colour, by its name: red, orange, amber, yellow, green, cyan, \
                          blue, purple, magenta, pink or white.",
        },
        LEVEL,
    ],
    examples: &[Example {
        situation: "The family asked for the room in blue for the game.",
        call: r#"{"device":"ceiling light 1","colour":"blue"}"#,
        because: "The colour is the light's to show; name the one the room wants.",
    }],
};

pub const WIZ_WHITE: Tool = Tool {
    name: "wiz_white",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description: "Set the light here to a white, from 2200 K (warm, like a candle) to 6500 K \
                  (cool, like daylight), at the brightness given or full.",
    params: &[
        BULB,
        Param {
            name: "kelvin",
            ty: "integer",
            required: true,
            description: "The white, in kelvin: 2200 warm to 6500 cool; 2700 is a lamp's.",
        },
        LEVEL,
    ],
    examples: &[Example {
        situation: "Somebody is reading at the table and wants bright daylight white.",
        call: r#"{"device":"ceiling light 1","kelvin":5000}"#,
        because: "The white is a temperature the light holds; set it to the one asked for.",
    }],
};

pub const PAIR_DEVICE: Tool = Tool {
    name: "pair_device",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description:
        "Begin pairing a TV here that the world cannot work yet because it has never been paired \
         — an LG TV or a Google TV. An LG TV shows an \"accept\" box on its own screen, so tell \
         the family to accept it; a Google TV shows a six-digit code, which the family must read \
         back to you, and you then finish with `pair_code`. Do this when the survey put a TV in \
         the room but the world has no key for it.",
    params: &[Param {
        name: "device",
        ty: "string",
        required: true,
        description: "The TV here to pair, by name, as the room calls it.",
    }],
    examples: &[Example {
        situation: "The survey put a Google TV in the room and the world has no key to work it.",
        call: r#"{"device":"the TV"}"#,
        because: "A TV the world has never paired with answers nothing; pairing is what earns the \
                  key, and nobody need be at the TV but to read a code back.",
    }],
};

pub const PAIR_CODE: Tool = Tool {
    name: "pair_code",
    at: &[],
    category: "House",
    plane: Plane::World,
    availability: Availability::Embodied,
    description:
        "Finish pairing a Google TV here with the six-digit code it is showing — the code the \
         family read back to you after you began with `pair_device`. This earns the key the \
         world keeps, and from then on it can work the TV with nobody at it.",
    params: &[
        Param {
            name: "device",
            ty: "string",
            required: true,
            description: "The Google TV here being paired, by name.",
        },
        Param {
            name: "code",
            ty: "string",
            required: true,
            description: "The six-digit code the TV is showing, as the family read it back.",
        },
    ],
    examples: &[Example {
        situation:
            "You began pairing the Google TV, it showed a code, and the family read you \"4C2A19\".",
        call: r#"{"device":"the TV","code":"4C2A19"}"#,
        because:
            "The code is what proves the pairing to the TV; giving it back is what earns the key.",
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
    SONOS_SOURCE,
    SONOS_VOLUME,
    RECEIVER_STATUS,
    RECEIVER_ON,
    RECEIVER_OFF,
    RECEIVER_MUTE,
    RECEIVER_UNMUTE,
    RECEIVER_VOLUME,
    RECEIVER_LOUDER,
    RECEIVER_QUIETER,
    RECEIVER_INPUT,
    RECEIVER_MODE,
    RECEIVER_DIM,
    RECEIVER_PLAY,
    RECEIVER_PAUSE,
    RECEIVER_NEXT,
    RECEIVER_PREVIOUS,
    RECEIVER_PLAYING,
    CAST_STATUS,
    CAST_PLAYING,
    CAST_PLAY,
    CAST_PAUSE,
    CAST_STOP,
    CAST_MUTE,
    CAST_UNMUTE,
    CAST_VOLUME,
    CAST_CAST,
    CAST_LAUNCH,
    WIZ_STATUS,
    WIZ_ON,
    WIZ_OFF,
    WIZ_DIM,
    WIZ_COLOUR,
    WIZ_WHITE,
    PAIR_DEVICE,
    PAIR_CODE,
    WAKE_DEVICE,
];

/// Whether `tool` is a bridge act — enacted by the runtime, which awaits the
/// network, and not by the world's synchronous dispatch.
pub fn is_mine(tool: &str) -> bool {
    BRIDGE_ACTS.iter().any(|t| t.name == tool)
}
