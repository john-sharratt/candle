//! The acts of a mind with no body (`docs/zen_everywhere.md` §5): find the
//! terminals it can reach, connect to one, and — connected — work what stands
//! behind it.
//!
//! Every one is [`Availability::Granted`]: it is in the grammar only while
//! [`super::grants`] names it, which is the three terminal acts for a mind
//! that takes no orders, and a terminal's own verbs only while that mind is
//! connected to it. An unconnected terminal's verbs are not there to be
//! called.
//!
//! The acts that work a Sonos player, a receiver, a Cast device or a light are
//! the house's own ([`crate::bridge::acts`]): each is bound to the devices of
//! its kind, and a connection is what puts a device in that set.

use crate::engine::tools::{Availability, Example, Param, Plane, Tool};

/// An act of a terminal that names nothing.
macro_rules! bare {
    ($name:literal, $category:literal, $desc:literal, $situation:literal, $because:literal) => {
        Tool {
            name: $name,
            at: &[],
            category: $category,
            plane: Plane::World,
            availability: Availability::Granted,
            description: $desc,
            params: &[],
            examples: &[Example {
                situation: $situation,
                call: "{}",
                because: $because,
            }],
        }
    };
}

/// An act of a terminal that names one thing.
macro_rules! one {
    ($name:literal, $category:literal, $desc:literal, $arg:literal, $ty:literal, $argdesc:literal,
     $situation:literal, $call:literal, $because:literal) => {
        Tool {
            name: $name,
            at: &[],
            category: $category,
            plane: Plane::World,
            availability: Availability::Granted,
            description: $desc,
            params: &[Param {
                name: $arg,
                ty: $ty,
                required: true,
                description: $argdesc,
            }],
            examples: &[Example {
                situation: $situation,
                call: $call,
                because: $because,
            }],
        }
    };
}

pub const TERMINAL_LIST: Tool = bare!(
    "terminal_list",
    "Terminals",
    "List every terminal you can reach now — each a real thing with a way in: the house's \
     lights, a television, the house network, the cluster, a machine — with what it is and \
     whether it answers. Use for: finding out what you can reach, and the exact name to give \
     `terminal_connect`. Triggered by: \"what can you reach\", \"which terminals do you have\", \
     wanting to look at something outside you without knowing its terminal's name.",
    "You want to look in on the house and do not know what is there to look through.",
    "You are nowhere, so nothing is in front of you until you connect to it; the list is where \
     looking starts."
);

pub const TERMINAL_CONNECT: Tool = one!(
    "terminal_connect",
    "Terminals",
    "Connect to a terminal — the only way you reach anything outside you. You hold one \
     connection: connecting to another lets the last go. Connected, what the terminal can do is \
     yours to do, and it tells you what that is. Use for: any question or errand about the \
     cluster, a machine, the lights, a television or the house network — connect first, then \
     act with what the terminal gives you; until you are connected you know nothing of it and \
     can only say so. Triggered by: \"check the cluster\", \"are the machines up\", \"is the \
     television on\", \"turn the lights off\", or having told somebody you would look at a \
     thing you are not connected to.",
    "terminal",
    "string",
    "The terminal to connect to, by its name as `terminal_list` gave it.",
    "It is late, and you mean to see whether the lights were left on.",
    r#"{"terminal":"the lights"}"#,
    "A thing is worked through its terminal; connecting is how you come to be at it."
);

pub const TERMINAL_DISCONNECT: Tool = bare!(
    "terminal_disconnect",
    "Terminals",
    "Let go of the terminal you are connected to: its verbs are yours no longer, and you are \
     connected to nothing until you connect again.",
    "You have done what you connected for.",
    "A connection held is a thing you are still attending to; letting it go is saying you are \
     done there."
);

pub const REST: Tool = bare!(
    "rest",
    "Terminals",
    "Stop, and be woken by whatever reaches you next: a message, a terminal telling you \
     something changed, a Drone's work come back. You name nothing to wait for and no time; the \
     next thing that happens is what wakes you. Use for: the moment after you have done what \
     was asked and told whoever asked what came of it, and any moment with nothing new to act \
     on — a reading that says what the last one said is nothing new. Never to put off \
     something asked of you: do it, say what came of it, then rest. Triggered by: nothing left \
     to do, having just reported, a reading that has not changed.",
    "You read the cluster, told John what it said, and nothing else has reached you.",
    "Reading again what has not changed is not watching: what changes is told to you, and it \
     wakes you."
);

// ── a television ───────────────────────────────────────────────────────────

pub const TV_STATUS: Tool = bare!(
    "tv_status",
    "Television",
    "Read what the television you are connected to is doing: on or off, what is on it, how \
     loud.",
    "Somebody asked whether the television was left on.",
    "What the television is doing is in the television, not in anybody's guess."
);

pub const TV_ON: Tool = bare!(
    "tv_on",
    "Television",
    "Turn the television you are connected to on, waking it over the network.",
    "The family asked for the television on.",
    "It wakes on a packet sent to it; nobody need be at it."
);

pub const TV_OFF: Tool = bare!(
    "tv_off",
    "Television",
    "Turn the television you are connected to off, so its screen is dark and it draws little \
     until it is turned on again.",
    "The room is empty and the television was left on.",
    "A television left on all night is one nobody turned off."
);

pub const TV_VOLUME: Tool = one!(
    "tv_volume",
    "Television",
    "Set how loud the television you are connected to plays, from 0 (silent) to 100.",
    "volume",
    "integer",
    "How loud, from 0 to 100; 15 is quiet, 40 fills the room.",
    "It is late and the television is loud enough to keep the house up.",
    r#"{"volume":12}"#,
    "Loudness is a number the television holds; set it."
);

pub const TV_LOUDER: Tool = bare!(
    "tv_louder",
    "Television",
    "Turn the television you are connected to up a step, a little louder than it was.",
    "Somebody asked for it a little louder.",
    "A nudge up is its own act; the television knows its own step."
);

pub const TV_QUIETER: Tool = bare!(
    "tv_quieter",
    "Television",
    "Turn the television you are connected to down a step, a little quieter than it was.",
    "Somebody asked for it a little quieter.",
    "A nudge down is its own act."
);

pub const TV_MUTE: Tool = bare!(
    "tv_mute",
    "Television",
    "Mute the television you are connected to, so its sound stops at once.",
    "The phone is ringing and the room needs to be quiet this moment.",
    "Muting stops the sound and keeps the volume."
);

pub const TV_UNMUTE: Tool = bare!(
    "tv_unmute",
    "Television",
    "Unmute the television you are connected to, bringing its sound back at the volume it had.",
    "The call is over and the room wants its sound back.",
    "The television is muted; unmuting undoes it."
);

pub const TV_PLAY: Tool = bare!(
    "tv_play",
    "Television",
    "Resume what the television you are connected to was playing.",
    "The show was paused and the room is ready to go on.",
    "It holds where it was; starting it is the whole act."
);

pub const TV_PAUSE: Tool = bare!(
    "tv_pause",
    "Television",
    "Pause what the television you are connected to is playing, keeping its place.",
    "Somebody needs the room quiet and the television is mid-show.",
    "Paused, it picks up where it was."
);

pub const TV_STOP: Tool = bare!(
    "tv_stop",
    "Television",
    "Stop what the television you are connected to is playing, ending it rather than keeping \
     its place.",
    "The show is over and the room wants it off the screen.",
    "Stopping is a single act the television does."
);

pub const TV_LAUNCH: Tool = one!(
    "tv_launch",
    "Television",
    "Open an app on the television you are connected to — YouTube, Netflix, Spotify, Plex and \
     the like.",
    "app",
    "string",
    "The app to open, by its name: youtube, netflix, spotify, plex, disney, prime.",
    "The family asked for Netflix on the television.",
    r#"{"app":"netflix"}"#,
    "Opening the app is the television's to do; name the one wanted."
);

pub const TV_YOUTUBE: Tool = one!(
    "tv_youtube",
    "Television",
    "Play a YouTube video on the television you are connected to, by its link or its id.",
    "video",
    "string",
    "The video: its YouTube link, or the id in it.",
    "Somebody sent a video's link and asked for it on the television.",
    r#"{"video":"https://www.youtube.com/watch?v=dQw4w9WgXcQ"}"#,
    "The link is all it takes; the television finds the video."
);

pub const TV_INPUT: Tool = one!(
    "tv_input",
    "Television",
    "Switch the television you are connected to to one of its inputs.",
    "input",
    "string",
    "The input: hdmi1, hdmi2, hdmi3, hdmi4, av or component.",
    "The console is on the second HDMI socket and the television is on the first.",
    r#"{"input":"hdmi2"}"#,
    "The input is a setting the television holds; switch it."
);

pub const TV_BUTTON: Tool = one!(
    "tv_button",
    "Television",
    "Press one button of the remote on the television you are connected to.",
    "button",
    "string",
    "The button: home, back, up, down, left, right, ok, menu or info.",
    "A dialog is on the screen and the room wants it dismissed.",
    r#"{"button":"back"}"#,
    "What a hand on the remote would press, pressed from here."
);

pub const TV_SHOW: Tool = one!(
    "tv_show",
    "Television",
    "Show a short message on the screen of the television you are connected to, over whatever \
     is on.",
    "message",
    "string",
    "The words to show, a line.",
    "Dinner is ready and the family are watching a film.",
    r#"{"message":"Dinner is ready."}"#,
    "A word on the screen reaches the room without stopping what it is watching."
);

// ── the house network ──────────────────────────────────────────────────────

pub const DEVICE_RENAME: Tool = Tool {
    name: "device_rename",
    at: &[],
    category: "House network",
    plane: Plane::World,
    availability: Availability::Granted,
    description: "Give a device on the house network the name it is to go by — what it is \
                  listed as and worked under from now on. A bulb calls itself by a code; this \
                  is how it becomes the kitchen light.",
    params: &[
        Param {
            name: "device",
            ty: "string",
            required: true,
            description: "The device, by the name it goes by now.",
        },
        Param {
            name: "called",
            ty: "string",
            required: true,
            description: "What it is to be called, one to forty characters.",
        },
    ],
    examples: &[Example {
        situation: "The survey found a bulb that calls itself wiz_d8a011, and the family say it \
                    is the one over the kitchen table.",
        call: r#"{"device":"wiz_d8a011","called":"kitchen light"}"#,
        because: "A thing with a name a person would use is a thing a person can ask for.",
    }],
};

pub const PRESENCE_WATCH: Tool = Tool {
    name: "presence_watch",
    at: &[],
    category: "House network",
    plane: Plane::World,
    availability: Availability::Granted,
    description: "Watch a handset on the house network as one person's: you are told when it \
                  joins the network and when it has left — which is that person coming home \
                  and going out.",
    params: &[
        Param {
            name: "device",
            ty: "string",
            required: true,
            description: "The handset, by the name it goes by.",
        },
        Param {
            name: "who",
            ty: "string",
            required: true,
            description: "Whose it is, by the name you call them.",
        },
    ],
    examples: &[Example {
        situation: "The survey found a phone called Pixel-8, and it is Kate's.",
        call: r#"{"device":"Pixel-8","who":"Kate"}"#,
        because: "A phone goes where its person goes; watching it is knowing who is home \
                  without asking.",
    }],
};

pub const PRESENCE_FORGET: Tool = one!(
    "presence_forget",
    "House network",
    "Stop watching a handset: you are no longer told when it comes and goes.",
    "device",
    "string",
    "The handset, by the name it goes by.",
    "A handset you watch has been given away.",
    r#"{"device":"Pixel-8"}"#,
    "A phone that is nobody's any more says nothing about who is home."
);

// ── the cluster ────────────────────────────────────────────────────────────

pub const CLUSTER_STATUS: Tool = bare!(
    "cluster_status",
    "Cluster",
    "Read how the cluster stands: every member, the build each runs, whether it is ready, and \
     anything in fault. Use for: the first look at the cluster, and any question of whether \
     the machines are up, serving, or on the same build. Triggered by: \"is everything up\", \
     \"are the machines serving\", \"what build is each member on\", \"did the rollout reach \
     every member\".",
    "You are on your watch and have not looked at the cluster since morning.",
    "A member that has fallen out says nothing; it is seen only by looking."
);

pub const CLUSTER_SERVICES: Tool = bare!(
    "cluster_services",
    "Cluster",
    "Read what each machine's keep runs: every service's installed build, whether it runs, and \
     how its last trial ended. Use for: whether a deploy took on each machine, and what is \
     installed as against what the cluster reports. Triggered by: \"did the deploy take\", \
     \"is a trial running\", \"what is installed on that machine\".",
    "A deploy went out this afternoon and you want to know it took everywhere.",
    "What a machine runs is in its keep, not in what was meant to be deployed."
);

pub const CLUSTER_LOGS: Tool = Tool {
    name: "cluster_logs",
    at: &[],
    category: "Cluster",
    plane: Plane::World,
    availability: Availability::Granted,
    description: "Search the cluster's logs — every member's, gathered at the gateway — for \
                  what they say about a symptom, oldest first. Use for: why something happened, \
                  once you have a word for it. Triggered by: \"why did it restart\", \"what do \
                  the logs say about\", \"find the error\"; with no word to search for, \
                  `cluster_log` reads the newest lines.",
    params: &[
        Param {
            name: "grep",
            ty: "string",
            required: true,
            description: "What a line must hold: a word, a phrase or a regular expression.",
        },
        Param {
            name: "level",
            ty: "string",
            required: false,
            description: "The least level kept: INFO, WARN or ERROR. Left out, every line.",
        },
        Param {
            name: "since",
            ty: "string",
            required: false,
            description: "Only lines at or after this moment, UTC, such as 2026-11-15T09:20.",
        },
    ],
    examples: &[Example {
        situation: "A member restarted an hour ago and nobody knows why.",
        call: r#"{"grep":"panicked|restart","level":"WARN"}"#,
        because: "Why a thing fell over is in what it wrote as it fell.",
    }],
};

pub const CLUSTER_MEMBER: Tool = one!(
    "cluster_member",
    "Cluster",
    "Read one member's own status, whole, as it reports itself: its state and build, each app it \
     hosts, whether it is really joined, what it has not yet sent, and its upgrade.",
    "node",
    "string",
    "The member, by its node id: the sixteen hex digits `cluster_status` lists it under.",
    "The cluster's summary says a member is ready, and you doubt it is joined.",
    r#"{"node":"09dd3987369d9cd8"}"#,
    "The summary is one line a member; what a member says of itself is the whole of it."
);

pub const CLUSTER_MEMORY: Tool = one!(
    "cluster_memory",
    "Cluster",
    "Read one member's memory as it reports it: the machine's RAM, what holds its card, and how \
     full each part is.",
    "node",
    "string",
    "The member, by its node id: the sixteen hex digits `cluster_status` lists it under.",
    "A member has slowed, and you want to know whether its card is full.",
    r#"{"node":"09dd3987369d9cd8"}"#,
    "Slow is a symptom; what is holding the card is the cause."
);

pub const CLUSTER_AUDIT: Tool = bare!(
    "cluster_audit",
    "Cluster",
    "Read what was done to the cluster in the last day, newest first: each command a member ran \
     and who sent it, with members joining, leaving and being refused.",
    "A member is on a build you did not expect, and you want to know who put it there.",
    "Before deciding something is wrong, see what was done on purpose."
);

pub const CLUSTER_LOG: Tool = one!(
    "cluster_log",
    "Cluster",
    "Read the newest lines of one log, oldest first — the cluster's own, or a site's — when you \
     want what it is saying now and have no word to search for.",
    "source",
    "string",
    "The log to read: `cluster` for the machines' daemons, or a site's name.",
    "A deploy has just gone out and you want the cluster's last lines as they are.",
    r#"{"source":"cluster"}"#,
    "With nothing to search for, the newest lines are where to start."
);

pub const CLUSTER_JOBS: Tool = bare!(
    "cluster_jobs",
    "Cluster",
    "List the jobs running on the cluster now — builds, test runs, deploys, tasks, maintenance — \
     each with its handle, where it runs and how long it has run.",
    "A machine is busy and you want to know with what.",
    "A machine working hard is usually working on something somebody started."
);

pub const CLUSTER_JOB: Tool = one!(
    "cluster_job",
    "Cluster",
    "Read where one job is, and once it has ended how it went: a build's or test run's results \
     and first failures, a deploy's trial, a task's answer.",
    "job",
    "string",
    "The job, by its handle, as `cluster_jobs` or the call that started it gave it.",
    "You started a test run a while ago and want to know how it ended.",
    r#"{"job":"build:76c3a0f4d39bd9db:Be-t_qMsg8U"}"#,
    "A job you started is yours to see through to how it ended."
);

pub const CLUSTER_JOB_READ: Tool = Tool {
    name: "cluster_job_read",
    at: &[],
    category: "Cluster",
    plane: Plane::World,
    availability: Availability::Granted,
    description: "Read one page of a job's output — two hundred lines: a build's log, a deploy's \
                  build and trial, a task's entries and answer.",
    params: &[
        Param {
            name: "job",
            ty: "string",
            required: true,
            description: "The job, by its handle.",
        },
        Param {
            name: "page",
            ty: "string",
            required: false,
            description: "The page to read, from 1. Left out, the first.",
        },
    ],
    examples: &[Example {
        situation: "A test run failed and its status names page 3 as where the failure is.",
        call: r#"{"job":"build:76c3a0f4d39bd9db:Be-t_qMsg8U","page":"3"}"#,
        because: "What failed is in the output, on the page the status named.",
    }],
};

pub const CLUSTER_JOB_STOP: Tool = one!(
    "cluster_job_stop",
    "Cluster",
    "Stop one job you started: a build or test run is killed where it is, and nothing it built \
     is kept.",
    "job",
    "string",
    "The job, by its handle.",
    "A test run you started is the wrong one, and it is holding a machine.",
    r#"{"job":"build:76c3a0f4d39bd9db:Be-t_qMsg8U"}"#,
    "A job that answers nothing useful is a machine kept from one that would."
);

pub const CLUSTER_JOB_START: Tool = Tool {
    name: "cluster_job_start",
    at: &[],
    category: "Cluster",
    plane: Plane::World,
    availability: Availability::Granted,
    description: "Start a build, a test run or a check of the code on a build machine, as it \
                  stands on the repository's main branch. It answers at once with the job's \
                  handle; `cluster_job` says how it went. One program a job, never a shell line. \
                  Writing or changing code is not this: that is a Drone's `coding-task`.",
    params: &[
        Param {
            name: "repo",
            ty: "string",
            required: true,
            description: "The repository, by its name, such as candle.",
        },
        Param {
            name: "program",
            ty: "string",
            required: true,
            description: "The program alone, such as cargo.",
        },
        Param {
            name: "args",
            ty: "string",
            required: true,
            description: "Its arguments, a space between each, exactly as they are to be \
                          passed.",
        },
    ],
    examples: &[Example {
        situation: "John asked whether the web crate's tests still pass on main.",
        call: r#"{"repo":"candle","program":"cargo","args":"test -p web"}"#,
        because: "Whether the tests pass is found by running them, on a machine kept for it.",
    }],
};

// ── a machine ──────────────────────────────────────────────────────────────

pub const MACHINE_CHECK: Tool = Tool {
    name: "machine_check",
    at: &[],
    category: "Machine",
    plane: Plane::World,
    availability: Availability::Granted,
    description: "Run one of the checks of the machine you are connected to. A check looks and \
                  changes nothing, and its report begins with what is different from the last \
                  time it ran. The machine's own list says what each one reports and what \
                  arguments it takes. Use for: how one machine is doing — its health, its \
                  disks, what listens on it, what runs on it, what its logs say. Triggered by: \
                  \"is that machine healthy\", \"what is using the disk\", \"has anything \
                  changed on it\".",
    params: &[
        Param {
            name: "check",
            ty: "string",
            required: true,
            description: "The check to run, by its name.",
        },
        Param {
            name: "args",
            ty: "string",
            required: false,
            description: "The check's arguments, when it takes any — one \"name: value\" to a \
                          line.",
        },
    ],
    examples: &[
        Example {
            situation: "You are on your watch, connected to a machine, and want to know nothing \
                        is listening on it that was not yesterday.",
            call: r#"{"check":"listeners"}"#,
            because: "A port that was not open is the thing a watch is for, and the report says \
                      it first.",
        },
        Example {
            situation: "The machine's errors named a disk, and you want what its system log \
                        said about it today.",
            call: r#"{"check":"logs","args":"log: System\ngrep: disk\nsince: 24h"}"#,
            because: "A search is asked of a named log for a word; nothing else is run.",
        },
    ],
};

/// The acts a mind with no body always has: finding its terminals,
/// connecting to one, and stopping until something reaches it.
pub const REACH_ACTS: &[&str] = &["terminal_list", "terminal_connect", "rest"];

/// What a paired television offers.
pub const TV_ACTS: &[&str] = &[
    "tv_status",
    "tv_on",
    "tv_off",
    "tv_volume",
    "tv_louder",
    "tv_quieter",
    "tv_mute",
    "tv_unmute",
    "tv_play",
    "tv_pause",
    "tv_stop",
    "tv_launch",
    "tv_youtube",
    "tv_input",
    "tv_button",
    "tv_show",
];

/// What the house network's terminal offers beyond the survey and the wake.
pub const NETWORK_ACTS: &[&str] = &["device_rename", "presence_watch", "presence_forget"];

/// What the cluster's terminal offers.
pub const CLUSTER_ACTS: &[&str] = &[
    "cluster_status",
    "cluster_services",
    "cluster_logs",
    "cluster_member",
    "cluster_memory",
    "cluster_audit",
    "cluster_log",
    "cluster_jobs",
    "cluster_job",
    "cluster_job_read",
    "cluster_job_stop",
    "cluster_job_start",
];

/// What a machine's terminal offers.
pub const MACHINE_ACTS: &[&str] = &["machine_check"];

/// Every terminal act, for the catalog.
pub const TERMINAL_ACTS: &[Tool] = &[
    TERMINAL_LIST,
    TERMINAL_CONNECT,
    TERMINAL_DISCONNECT,
    REST,
    TV_STATUS,
    TV_ON,
    TV_OFF,
    TV_VOLUME,
    TV_LOUDER,
    TV_QUIETER,
    TV_MUTE,
    TV_UNMUTE,
    TV_PLAY,
    TV_PAUSE,
    TV_STOP,
    TV_LAUNCH,
    TV_YOUTUBE,
    TV_INPUT,
    TV_BUTTON,
    TV_SHOW,
    DEVICE_RENAME,
    PRESENCE_WATCH,
    PRESENCE_FORGET,
    CLUSTER_STATUS,
    CLUSTER_SERVICES,
    CLUSTER_LOGS,
    CLUSTER_MEMBER,
    CLUSTER_MEMORY,
    CLUSTER_AUDIT,
    CLUSTER_LOG,
    CLUSTER_JOBS,
    CLUSTER_JOB,
    CLUSTER_JOB_READ,
    CLUSTER_JOB_STOP,
    CLUSTER_JOB_START,
    MACHINE_CHECK,
];

/// Whether `tool` is a terminal act — enacted by the runtime, which awaits the
/// network, and not by the world's synchronous dispatch.
pub fn is_mine(tool: &str) -> bool {
    TERMINAL_ACTS.iter().any(|t| t.name == tool)
}

/// The terminal acts that leave nothing to act on: letting a terminal go, and
/// stopping.
const ENDINGS: &[&str] = &["terminal_disconnect", "rest"];

/// Whether what `tool` came back with is the point of making it — a reading,
/// a list, whether a thing took the call — so the mind that made it is owed a
/// turn to use it.
pub fn answers(tool: &str) -> bool {
    is_mine(tool) && !ENDINGS.contains(&tool)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Every act a terminal is said to offer is in the catalogue**, and
    /// every act in it is offered by something.
    #[test]
    fn every_offered_act_is_in_the_catalogue() {
        let offered: Vec<&str> = REACH_ACTS
            .iter()
            .chain(["terminal_disconnect"].iter())
            .chain(TV_ACTS)
            .chain(NETWORK_ACTS)
            .chain(CLUSTER_ACTS)
            .chain(MACHINE_ACTS)
            .copied()
            .collect();
        for name in &offered {
            assert!(
                is_mine(name),
                "`{name}` is offered and not in the catalogue"
            );
        }
        for tool in TERMINAL_ACTS {
            assert!(
                offered.contains(&tool.name),
                "`{}` is offered by nothing",
                tool.name
            );
            assert_eq!(tool.availability, Availability::Granted, "{}", tool.name);
        }
    }

    /// **A reading is owed a turn to use it; an ending is not.** Every
    /// terminal act answers but the two that leave nothing to act on.
    #[test]
    fn every_terminal_act_answers_but_the_endings() {
        for tool in [
            "terminal_list",
            "terminal_connect",
            "cluster_status",
            "tv_on",
        ] {
            assert!(answers(tool), "{tool}");
        }
        for tool in ["rest", "terminal_disconnect", "tell", "survey_network"] {
            assert!(!answers(tool), "{tool}");
        }
    }
}
