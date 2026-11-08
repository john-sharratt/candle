//! The acts by which a drone drives a task of Zen's (`docs/mission_templates.md`
//! §7): every one [`Availability::Granted`] — offered only while a mission
//! granting it is carried, wherever its carrier stands.

use crate::engine::tools::{Availability, Example, Param, Plane, Tool};

const fn param(name: &'static str, required: bool, description: &'static str) -> Param {
    Param {
        name,
        ty: "string",
        required,
        description,
    }
}

pub const TASK_START: Tool = Tool {
    name: "task_start",
    at: &[],
    category: "Work",
    plane: Plane::World,
    availability: Availability::Granted,
    description:
        "Start the work your mission is for on one of Zen's workers: a task in the house's \
                  repositories, which runs on its own and answers on your handset.",
    params: &[
        param(
            "name",
            true,
            "What the task does, in a few words joined by dashes: \"fix-door-sensor\".",
        ),
        param(
            "prompt",
            true,
            "The task in full: what to change or find out, where, and what it must check before \
             it answers.",
        ),
        param(
            "role",
            false,
            "`coder` (the default) for work that changes something, `researcher` to answer a \
             question, `reviewer` to read a change and say what is wrong with it.",
        ),
    ],
    examples: &[Example {
        situation: "Your mission is to fix the door sensor's firmware build.",
        call: r#"{"name":"fix-door-sensor-build","prompt":"The door sensor's firmware no longer builds on main. Find why, fix it on a branch, run its tests, and say what changed.","role":"coder"}"#,
        because: "The work is code, and a worker of Zen's does code; you see it through.",
    }],
};

pub const TASK_REPLY: Tool = Tool {
    name: "task_reply",
    at: &[],
    category: "Work",
    plane: Plane::World,
    availability: Availability::Granted,
    description: "Answer the question your mission's task asked you on your handset, so it can go \
                  on.",
    params: &[param(
        "answer",
        true,
        "The decision itself — one of its options, or exactly what to do instead.",
    )],
    examples: &[Example {
        situation: "The task asked whether to keep the old sensor driver or replace it.",
        call: r#"{"answer":"Replace it; nothing else uses the old driver."}"#,
        because: "It waits until it is answered.",
    }],
};

pub const TASK_NUDGE: Tool = Tool {
    name: "task_nudge",
    at: &[],
    category: "Work",
    plane: Plane::World,
    availability: Availability::Granted,
    description: "Steer your mission's task while it runs, when its news shows it going the wrong \
                  way.",
    params: &[param(
        "message",
        true,
        "What to change and why, in a sentence or two.",
    )],
    examples: &[Example {
        situation: "The task is reading the lift's code; the fault is the door sensor's.",
        call: r#"{"message":"The fault is in the door sensor's firmware, not the lift: look there."}"#,
        because: "A task heading the wrong way wastes its run.",
    }],
};

/// The acts a mission lends to drive its task.
pub const TASK_ACTS: &[Tool] = &[TASK_START, TASK_REPLY, TASK_NUDGE];

/// Whether `tool` is one of these acts.
pub fn is_mine(tool: &str) -> bool {
    TASK_ACTS.iter().any(|t| t.name == tool)
}
