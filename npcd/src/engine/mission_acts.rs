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
use crate::sim::Sim;

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

/// **What was found is called `found`.** Named `account`, the field read —
/// inside an `invoke` body, where only its name is in front of the writer — as
/// whose account it was, and Makers filed their own names there turn after
/// turn while shouting the reading they had to the room.
pub const REPORT_DONE: Tool = desk_on!(
    "report_done",
    "Back at the command table, report the mission you were carrying as done, and say what you \
     found or concluded. This closes it and files your answer, which is how anyone else learns \
     what came of it — and it frees you to take up the next.",
    "found",
    "What you found, made, or concluded — the answer the mission was for.",
    "You have carried out your mission and come back to the table to say what came of it.",
    r#"{"found":"I read the coolant valve and the breaker panel: the valve is open and the breaker is tripped"}"#,
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

pub const REPORT_REJECTED: Tool = desk_on!(
    "report_rejected",
    "Back at the command table, reject the draft you were sent to review, and say why it cannot \
     stand. This fails the operation it belonged to and takes the draft out of the record. For \
     faults that run through all of it — what can be mended in place, mend, and report it done.",
    "why",
    "What is wrong with the draft that cannot be put right in place.",
    "You have read a draft for review and it cannot stand: it is set in the wrong era, told in the \
     wrong voice from end to end, or summary where a scene should be.",
    r#"{"why":"It is set in 2950 but the whole scene takes place on the surface, which the era says nobody could reach until 3001"}"#,
    "A draft that stands only because nobody would reject it is lore nobody checked; saying it \
     cannot stand is the review doing its job."
);

/// Every mission act, for the catalog. All at the command table: a mission is
/// taken up and answered for there, and carried out in the world between.
pub const MISSION_ACTS: &[Tool] = &[COLLECT_MISSION, REPORT_DONE, REPORT_STUCK, REPORT_REJECTED];

/// What a body at the table is carrying, as far as the table's acts care.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Carrying {
    /// It carries an open mission.
    pub mission: bool,
    /// That mission is a stage of an operation that judges — a review or a
    /// canon check.
    pub review: bool,
    /// That mission is any stage of an operation.
    pub operation: bool,
    /// It holds a standing order.
    pub order: bool,
    /// That mission writes a document — a draft, or a review that must mend
    /// one — which is what `compose` is for.
    pub drafts: bool,
}

/// What `body` is carrying in `s`.
pub fn carrying(s: &Sim, body: &str) -> Carrying {
    let mission = s.missions.active(body);
    Carrying {
        mission: mission.is_some(),
        review: mission.is_some_and(|m| m.may_reject()),
        operation: mission.is_some_and(|m| m.operation().is_some()),
        order: !s.ledger.held_by(body).is_empty(),
        drafts: mission
            .and_then(|m| m.work.as_ref())
            .is_some_and(|w| !w.edit_optional),
    }
}

/// The bench's working-set verbs an operation's Maker is not offered: what
/// it needs is to read, write or edit one document and commit it.
///
/// **Version control is a trap for one document.** Reviewers carrying an
/// operation stashed their own edits, committed nothing, popped the stash,
/// staged, restored and read the log, round and round, each sure the work was
/// done — `bench_commit` answering "you have nothing open to merge" because the
/// stash had taken it.
const NOT_FOR_OPERATIONS: [&str; 9] = [
    "bench_stash",
    "bench_stash_pop",
    "bench_stage",
    "bench_unstage",
    "bench_restore",
    "bench_branch",
    "bench_blame",
    "bench_log",
    "bench_diff",
];

/// Whether the table offers `tool` to a body carrying `c`.
///
/// **What cannot be done is not offered.** The table listed `collect_mission`
/// beside `report_done` to a character back with its work done; it took the
/// first, was refused, read the refusal as being sent to collect first, and
/// stood at the table going round. One carrying a mission has only the reports;
/// one carrying none has only the collect; only a review can be rejected.
/// `orders_report_done` is for a body holding an order — a character with a
/// mission and no order filed its mission against an order it never held. A
/// Maker on an operation is not offered the bench's working-set verbs (see
/// [`NOT_FOR_OPERATIONS`]).
pub fn offered(tool: &str, c: Carrying) -> bool {
    match tool {
        t if t == COLLECT_MISSION.name => !c.mission,
        t if t == REPORT_DONE.name || t == REPORT_STUCK.name => c.mission,
        t if t == REPORT_REJECTED.name => c.mission && c.review,
        "orders_report_done" => c.order,
        // **A mission's piece is written sitting down.** Written whole as one
        // `file_write` among a Maker's other acts, a piece carried the room it
        // was written in; and a composed draft rewritten by hand afterwards
        // lost what the composing gave it — "I am writing this from the
        // chronicle level, where the light ring hums". Its Maker writes it with
        // `compose` and puts single passages right with `file_edit`.
        "compose" => c.drafts,
        "file_write" => !c.drafts,
        t if NOT_FOR_OPERATIONS.contains(&t) => !c.operation,
        _ => true,
    }
}

/// Whether a tool is one of these, for the dispatcher.
pub fn is_mine(tool: &str) -> bool {
    MISSION_ACTS.iter().any(|t| t.name == tool)
}

#[cfg(test)]
mod tests {
    use super::{offered, Carrying};

    fn carrying(mission: bool, review: bool, order: bool) -> Carrying {
        Carrying {
            mission,
            review,
            operation: review,
            order,
            drafts: false,
        }
    }

    #[test]
    fn the_table_offers_only_what_the_body_can_do_there() {
        // Carrying nothing: take one up, nothing to report.
        let idle = carrying(false, false, false);
        assert!(offered("collect_mission", idle));
        assert!(!offered("report_done", idle));
        assert!(!offered("report_stuck", idle));
        assert!(!offered("report_rejected", idle));
        // Carrying one: report it, take nothing new.
        let busy = carrying(true, false, false);
        assert!(!offered("collect_mission", busy));
        assert!(offered("report_done", busy));
        assert!(offered("report_stuck", busy));
        assert!(!offered("report_rejected", busy), "only a review rejects");
        assert!(offered("report_rejected", carrying(true, true, false)));
        // An order is reported only by whoever holds one.
        assert!(!offered("orders_report_done", busy));
        assert!(offered("orders_report_done", carrying(false, false, true)));
        // Everything else is the table's as ever.
        assert!(offered("present", busy));
        // A Maker on an operation reads, writes and commits; the working-set
        // verbs are for others.
        let drafting = Carrying {
            operation: true,
            ..busy
        };
        assert!(!offered("bench_stash", drafting));
        assert!(!offered("bench_diff", drafting));
        assert!(offered("bench_commit", drafting));
        assert!(offered("file_write", drafting));
        assert!(offered("bench_stash", busy), "a routine mission keeps them");
        // Composing is for a Maker with a document to write, and nobody else —
        // and for that Maker it is how the piece is written whole.
        assert!(!offered("compose", drafting));
        let writing = Carrying {
            drafts: true,
            ..drafting
        };
        assert!(offered("compose", writing));
        assert!(!offered("file_write", writing), "written sitting down");
        assert!(offered("file_edit", writing), "a passage is still mended");
    }
}
