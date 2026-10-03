//! The acts a mission is lived through: take one up at the command desk, and
//! report how it went back at the desk.
//!
//! # One place: the command table
//!
//! A mission is taken up and answered for at the command table (`order-table`),
//! and carried out in the world between. All three acts are `AtPart` on that
//! part, so they are in a character's grammar only while it stands at the table —
//! it is called there when the table opens (see `Runtime::TO_THE_TABLE` and the
//! tannoy the command-table API sends), takes one up, goes and does it, and
//! comes back to report and take the next. Keeping the acts to the table is also
//! what keeps a character *out working* free of them: its grammar while it
//! travels and reads and talks is the ordinary one, unchanged by carrying a
//! mission.
//!
//! The rich shape of a mission (the ask, the steps, the answer, the pass/fail
//! report) lives in [`crate::engine::mission`]; whose it is lives in
//! [`crate::sim::missions`]; these are the verbs that move a mission through its
//! life. The handlers are in [`crate::engine::work`], beside the other acts that
//! run on `Sim` state.

use super::tools::{Availability, Example, Param, Plane, Tool};

/// The command desk, by the part id the map gives it. A mission is taken up and
/// reported here and nowhere else.
pub const COMMAND_DESK: &[&str] = &["order-table"];

/// An act taken at the command desk that names nothing — it acts on the desk and
/// the character standing at it.
macro_rules! desk {
    ($name:literal, $desc:literal, $situation:literal, $because:literal) => {
        Tool {
            name: $name,
            at: COMMAND_DESK,
            category: "Command",
            plane: Plane::World,
            availability: Availability::AtPart,
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

/// An act taken at the command desk that names one thing.
macro_rules! desk_on {
    ($name:literal, $desc:literal, $arg:literal, $argdesc:literal,
     $situation:literal, $call:literal, $because:literal) => {
        Tool {
            name: $name,
            at: COMMAND_DESK,
            category: "Command",
            plane: Plane::World,
            availability: Availability::AtPart,
            description: $desc,
            params: &[Param {
                name: $arg,
                ty: "string",
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

pub const COLLECT_MISSION: Tool = desk!(
    "collect_mission",
    "Take up a mission from the table — the one set for you, or, if none is, the next thing worth \
     doing. It becomes what you are working on until you come back and report it.",
    "You are at the command table with nothing you have been asked to do, and it is where work is \
     handed out.",
    "A mission taken is a mission somebody can hold you to; standing at the table without one is \
     standing idle where the work is."
);

pub const REPORT_DONE: Tool = desk_on!(
    "report_done",
    "Back at the command table, report the mission you were carrying as done, and say what you \
     found or concluded. This closes it and files your answer, which is how anyone else learns \
     what came of it — and it frees you to take up the next.",
    "account",
    "What you found, made, or concluded — the answer the mission was for.",
    "You have carried out your mission and come back to the table to say what came of it.",
    r#"{"account":"I read the coolant valve and the breaker panel: the valve is open and the breaker is tripped"}"#,
    "Work nobody reported is work nobody can build on, and the answer is the point of having gone."
);

pub const REPORT_STUCK: Tool = desk_on!(
    "report_stuck",
    "Back at the command table, report that the mission cannot be finished, and say why. This \
     closes it as not done — an honest account of what stopped you, not a thing to be ashamed of \
     — and frees you to take up another.",
    "why",
    "What stopped you — what you tried, and where it would not go.",
    "You have carried a mission as far as it will go, and come back to the table to say it will \
     not finish.",
    r#"{"why":"I could not get the reading: the panel is locked and nobody here has the key"}"#,
    "A mission that cannot be done is worth knowing about; a character that abandons one silently \
     leaves it believed to be still in hand."
);

/// Every mission act, for the catalog. All at the command table: a mission is
/// taken up and answered for there, and carried out in the world between.
pub const MISSION_ACTS: &[Tool] = &[COLLECT_MISSION, REPORT_DONE, REPORT_STUCK];

/// Whether a tool is one of these, for the dispatcher.
pub fn is_mine(tool: &str) -> bool {
    MISSION_ACTS.iter().any(|t| t.name == tool)
}
