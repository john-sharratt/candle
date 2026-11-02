//! The acts by which a character that takes no orders keeps its agenda
//! (`docs/zen_moon_mind.md` §17.2): make a mission of its own, change one, drop
//! one, close one. Offered wherever its attention is — deciding what it is doing
//! needs no table — and only to a character that keeps an agenda, through the
//! sets its parameters are bound to (`Choices::Priorities`, `Choices::Agenda`).

use crate::engine::tools::{Availability, Example, Param, Plane, Tool};

/// An agenda act's parameter.
const fn param(name: &'static str, required: bool, description: &'static str) -> Param {
    Param {
        name,
        ty: "string",
        required,
        description,
    }
}

pub const MISSION_ADD: Tool = Tool {
    name: "mission_add",
    at: &[],
    category: "Agenda",
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Put a mission of your own on your agenda: something you mean to do, with the \
                  steps that see it done. One at `now` is worked at once, ahead of whatever you \
                  were doing, which waits where you left it. Use it when something you heard \
                  or saw needs doing — not to answer a message, which `reply` does.",
    params: &[
        param(
            "title",
            true,
            "A few words it is named by on your agenda: \"the Sharratts' parcels\".",
        ),
        param(
            "brief",
            true,
            "What is to be done, and why, in a sentence or two.",
        ),
        param(
            "steps",
            true,
            "The steps, one to a line, each something you can be seen to do: \"go to the \
             watch room\", \"post work on the order table\".",
        ),
        param("priority", true, "How soon: `now`, `soon` or `later`."),
    ],
    examples: &[Example {
        situation: "Yen has asked you to make sure John is reminded to collect her parcels \
                    when he is next home.",
        call: r#"{"title":"Yen's parcels","brief":"See that John is reminded to collect Yen's parcels from downstairs.","steps":"post work on the order table","priority":"soon"}"#,
        because: "A request that takes doing is a mission; once it is on your agenda it is \
                  worked through, not worried over.",
    }],
};

pub const MISSION_EDIT: Tool = Tool {
    name: "mission_edit",
    at: &[],
    category: "Agenda",
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Change a mission on your agenda: what it is for, its steps, or how soon. \
                  Steps it has already done stay done.",
    params: &[
        param(
            "mission",
            true,
            "Which mission, by its title on your agenda.",
        ),
        param(
            "brief",
            false,
            "What is to be done now, and why, when that has changed.",
        ),
        param("steps", false, "Its steps now, one to a line."),
        param("priority", false, "How soon now: `now`, `soon` or `later`."),
    ],
    examples: &[Example {
        situation: "John has said the parcels can wait until the weekend.",
        call: r#"{"mission":"Yen's parcels","priority":"later"}"#,
        because: "What changed is when, not what; the mission keeps its steps.",
    }],
};

pub const MISSION_CANCEL: Tool = Tool {
    name: "mission_cancel",
    at: &[],
    category: "Agenda",
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Drop a mission from your agenda that is no longer worth doing, saying why. \
                  The next on your agenda is worked.",
    params: &[
        param(
            "mission",
            true,
            "Which mission, by its title on your agenda.",
        ),
        param("why", true, "Why it is no longer worth doing."),
    ],
    examples: &[Example {
        situation: "John has already collected the parcels.",
        call: r#"{"mission":"Yen's parcels","why":"John collected them himself this afternoon"}"#,
        because: "A mission the world has already settled is noise on your agenda.",
    }],
};

pub const MISSION_DONE: Tool = Tool {
    name: "mission_done",
    at: &[],
    category: "Agenda",
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Close a mission on your agenda as done, saying what you found or made — for \
                  one whose last step is yours to judge. A mission whose steps are all seen \
                  done closes by itself.",
    params: &[
        param(
            "mission",
            true,
            "Which mission, by its title on your agenda.",
        ),
        param(
            "found",
            true,
            "What you found, made or concluded, in a sentence.",
        ),
    ],
    examples: &[Example {
        situation: "You set out to decide what the Sharratts might need this week, and have.",
        call: r#"{"mission":"the Sharratts' week","found":"Yen needs the car on Thursday and the school trip form is due Friday"}"#,
        because: "A judgement is done when you have made it; say what it was.",
    }],
};

/// Every agenda act, in the order offered.
pub const AGENDA_ACTS: &[Tool] = &[MISSION_ADD, MISSION_EDIT, MISSION_CANCEL, MISSION_DONE];

/// Whether a tool is one of these, for the dispatcher.
pub fn is_mine(tool: &str) -> bool {
    AGENDA_ACTS.iter().any(|t| t.name == tool)
}
