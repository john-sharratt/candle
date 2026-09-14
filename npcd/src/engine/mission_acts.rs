//! The acts a mission is lived through: take one up at the command desk, record
//! progress on it wherever the work happens, and report how it went back at the
//! desk.
//!
//! # Two homes, on purpose
//!
//! Collecting a mission deliberately is an `AtPart` act on the command desk
//! (`order-table`) — where an operator-lodged mission is picked up. Everything
//! else a character does with a mission is [`Availability::OnMission`], carried
//! with the mission rather than with a place: recording progress and reporting
//! how it went both happen wherever the character is, because the moment a step
//! is finished — or the whole thing is — is wherever the work that finished it
//! happened, not back at a desk. Requiring a walk back to report would strand a
//! character that could not find its way, and the point is that the loop closes
//! on its own. (Most missions never need the desk act at all: the engine hands
//! an idle character its next mission itself — see `Runtime::ensure_mission`.)
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

/// An act taken on the mission you are carrying, wherever you are.
macro_rules! on_mission {
    ($name:literal, $desc:literal, $arg:literal, $argdesc:literal,
     $situation:literal, $call:literal, $because:literal) => {
        Tool {
            name: $name,
            at: &[],
            category: "Command",
            plane: Plane::World,
            availability: Availability::OnMission,
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
    "Take up a mission from the desk — the one set for you, or, if none is, the next thing worth \
     doing. It becomes what you are working on until you report it done.",
    "You are at the command desk with nothing you have been asked to do, and it is where work is \
     handed out.",
    "A mission taken is a mission somebody can hold you to; standing at the desk without one is \
     standing idle where the work is."
);

pub const REPORT_DONE: Tool = on_mission!(
    "report_done",
    "Report the mission you were carrying as done, and say what you found or concluded. This \
     closes it and files your answer, which is how anyone else learns what came of it. Do this \
     once every step is finished.",
    "account",
    "What you found, made, or concluded — the answer the mission was for.",
    "Every step of your mission is done and you are ready to say what came of it.",
    r#"{"account":"the record holds, except the eastern date, which cannot be reconciled with the charge"}"#,
    "Work nobody reported is work nobody can build on, and the answer is the point of having gone."
);

pub const REPORT_STUCK: Tool = on_mission!(
    "report_stuck",
    "Report that the mission cannot be finished, and say why. This closes it as not done — an \
     honest account of what stopped you, not a thing to be ashamed of.",
    "why",
    "What stopped you — what you tried, and where it would not go.",
    "You have carried a mission as far as it will go and it will not finish.",
    r#"{"why":"the record it asked me to read is not filed anywhere I could find, and nobody here has seen it"}"#,
    "A mission that cannot be done is worth knowing about; a character that abandons one silently \
     leaves it believed to be still in hand."
);

pub const STEP_DONE: Tool = on_mission!(
    "step_done",
    "Mark one step of your mission finished, as you finish it. This is how the mission's list \
     tracks where you actually are.",
    "step",
    "The step you have just finished, in the words your mission lists it under.",
    "You have just done one of the things your mission set out, and there are more to go.",
    r#"{"step":"read 'the-charge'"}"#,
    "Ticking it off as it happens is what keeps the standing list honest; leaving it drives you to \
     redo work you have already done."
);

pub const ADD_STEP: Tool = on_mission!(
    "add_step",
    "Add a step to your mission that it did not foresee — work you have discovered you need to do \
     to finish it.",
    "step",
    "The step to add, in your own words.",
    "Working your mission has turned up something it did not list that has to be done for it to be \
     finished.",
    r#"{"step":"ask Wren where the second ledger was moved to"}"#,
    "A mission you discover more of is a mission being taken seriously; the list is yours to keep \
     true to the work, not a fixed order to follow blindly."
);

/// Every mission act, for the catalog.
pub const MISSION_ACTS: &[Tool] = &[
    COLLECT_MISSION,
    REPORT_DONE,
    REPORT_STUCK,
    STEP_DONE,
    ADD_STEP,
];

/// Whether a tool is one of these, for the dispatcher.
pub fn is_mine(tool: &str) -> bool {
    MISSION_ACTS.iter().any(|t| t.name == tool)
}
