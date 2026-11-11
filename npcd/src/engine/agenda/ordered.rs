//! The mission acts of a character that takes orders, as its owner or an
//! authorizer changes its mission through a conversation
//! (`docs/npc_conversations.md` §7.1).
//!
//! **The agenda's names, another body's specs.** An ordered character keeps no
//! agenda: it carries one mission at a time, named by nothing but its brief,
//! and works no priorities. The agenda's own acts would bind `priority` to a
//! set that is empty for it and so leave the grammar altogether, and their
//! `mission` names an agenda title that does not exist. These are the same
//! four names with what fits an ordered body: `mission` binds the brief of the
//! mission it carries ([`crate::engine::tools::Choices::Agenda`], filled from
//! it), and nothing takes a priority or a title.
//!
//! They are never in [`crate::engine::tools::CATALOG`], which is keyed by
//! name: a turn whose `Within` says it [`orders`](crate::engine::tools::Within::orders)
//! takes them in place of the agenda's ([`counterpart`]), and they are
//! installed in the prompt under members of their own ([`member`]).

use serde_json::{Map, Value};

use crate::engine::tools::{Availability, Example, Param, Plane, Tool};

/// The category these are filed under, which is what tells them from the
/// agenda's acts of the same names.
pub const CATEGORY: &str = "Orders";

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
    category: CATEGORY,
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Take on a mission the person you are talking to gives you, in place of the \
                  one you carry, which is called off. Only for a mission they are asking you to \
                  take on now — not to answer them, which `reply` does.",
    params: &[
        param(
            "brief",
            true,
            "What is to be done, and why, in a sentence or two, as they asked it.",
        ),
        param(
            "steps",
            true,
            "The steps, one to a line, each something you can be seen to do: \"go to the \
             watch room\", \"check the lounge doors\".",
        ),
    ],
    examples: &[Example {
        situation: "John, who owns you, asks you to stop tidying and check the garden doors \
                    are locked before dark.",
        call: r#"{"brief":"Check the garden doors are locked before dark, as John asked.","steps":"go to the garden room\ncheck the garden doors"}"#,
        because: "Your owner has changed what you are doing; the old mission is called off.",
    }],
};

pub const MISSION_EDIT: Tool = Tool {
    name: "mission_edit",
    at: &[],
    category: CATEGORY,
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Change the mission you carry, as the person you are talking to asks: what it \
                  is for, or its steps. Steps already done stay done, and a step waiting on an \
                  authorizer's yes stays.",
    params: &[
        param("mission", true, "The mission you carry."),
        param(
            "brief",
            false,
            "What is to be done now, and why, when that has changed.",
        ),
        param("steps", false, "Its steps now, one to a line."),
    ],
    examples: &[Example {
        situation: "John says the lounge only needs the windows done today, not the whole room.",
        call: r#"{"mission":"Keep the lounge tidy.","steps":"clean the lounge windows"}"#,
        because: "What changed is the steps, not what the mission is for.",
    }],
};

pub const MISSION_CANCEL: Tool = Tool {
    name: "mission_cancel",
    at: &[],
    category: CATEGORY,
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Call off the mission you carry, as the person you are talking to asks, saying \
                  why.",
    params: &[
        param("mission", true, "The mission you carry."),
        param("why", true, "Why it is called off, as they said it."),
    ],
    examples: &[Example {
        situation: "Mira says nobody needs the lounge tidied after all.",
        call: r#"{"mission":"Keep the lounge tidy.","why":"Mira says nobody needs it tidied after all"}"#,
        because: "A mission its owner has called off is not yours to keep working.",
    }],
};

pub const MISSION_DONE: Tool = Tool {
    name: "mission_done",
    at: &[],
    category: CATEGORY,
    plane: Plane::Internal,
    availability: Availability::Always,
    description: "Report the mission you carry done, saying what you found or made — when the \
                  person you are talking to agrees it is done. A step waiting on an \
                  authorizer's yes holds it open until they give it.",
    params: &[
        param("mission", true, "The mission you carry."),
        param(
            "found",
            true,
            "What you found, made or concluded, in a sentence.",
        ),
    ],
    examples: &[Example {
        situation: "John agrees the garden doors are checked and locked.",
        call: r#"{"mission":"Check the garden doors are locked before dark, as John asked.","found":"both garden doors are locked"}"#,
        because: "The mission is done and its owner agrees; say what you found.",
    }],
};

/// Every ordered mission act, in the agenda's order.
pub const ORDERED_ACTS: &[Tool] = &[MISSION_ADD, MISSION_EDIT, MISSION_CANCEL, MISSION_DONE];

/// The ordered act standing in for the agenda's act `name`, when it has one.
pub fn counterpart(name: &str) -> Option<&'static Tool> {
    ORDERED_ACTS.iter().find(|t| t.name == name)
}

/// Whether `t` is one of these rather than the agenda's act of its name.
pub fn is_ordered(t: &Tool) -> bool {
    t.category == CATEGORY
}

/// The prompt member an ordered act is installed under, beside the agenda's
/// act of the same name.
pub fn member(name: &str) -> String {
    format!("ordered/{name}")
}

/// What an ordered mission act asks for, read off its arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Order {
    /// Take this on now, calling off what is carried.
    Add { brief: String, steps: Vec<String> },
    /// Change the carried mission.
    Edit {
        mission: String,
        brief: Option<String>,
        steps: Option<Vec<String>>,
    },
    /// Call the carried mission off.
    Cancel { mission: String, why: String },
    /// Report the carried mission done.
    Done { mission: String, found: String },
}

fn text(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// Steps given one to a line, a leading bullet dropped and blank lines left
/// out — as the agenda's acts take them.
fn steps(args: &Map<String, Value>) -> Option<Vec<String>> {
    let lines: Vec<String> = text(args, "steps")?
        .lines()
        .map(|l| l.trim().trim_start_matches(['-', '*']).trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    (!lines.is_empty()).then_some(lines)
}

/// The order `tool` with `args` makes, or what the character is told when it
/// makes none.
pub fn order(tool: &str, args: &Map<String, Value>) -> Result<Order, String> {
    let mission = || text(args, "mission").ok_or("Say which mission: the one you carry.");
    match tool {
        "mission_add" => {
            let brief = text(args, "brief")
                .ok_or("A mission needs a brief of what is to be done.")?;
            let steps = steps(args).ok_or(
                "Give it at least one step, one to a line, that you can be seen to do.",
            )?;
            Ok(Order::Add { brief, steps })
        }
        "mission_edit" => {
            let mission = mission()?;
            let (brief, steps) = (text(args, "brief"), steps(args));
            if brief.is_none() && steps.is_none() {
                return Err(
                    "You meant to change it, but said nothing to change: a brief, or its steps."
                        .into(),
                );
            }
            Ok(Order::Edit {
                mission,
                brief,
                steps,
            })
        }
        "mission_cancel" => Ok(Order::Cancel {
            mission: mission()?,
            why: text(args, "why").ok_or("Say why it is called off.")?,
        }),
        "mission_done" => Ok(Order::Done {
            mission: mission()?,
            found: text(args, "found").ok_or("Say what you found.")?,
        }),
        other => Err(format!("`{other}` is no act of a mission.")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::agenda::AGENDA_ACTS;

    fn args(json: &str) -> Map<String, Value> {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn each_agenda_act_has_its_ordered_counterpart_under_the_same_name() {
        let names: Vec<&str> = ORDERED_ACTS.iter().map(|t| t.name).collect();
        let agenda: Vec<&str> = AGENDA_ACTS.iter().map(|t| t.name).collect();
        assert_eq!(names, agenda);
        for t in ORDERED_ACTS {
            assert!(is_ordered(t));
            assert!(!t.params.iter().any(|p| p.name == "priority" || p.name == "title"));
        }
        assert_eq!(member("mission_add"), "ordered/mission_add");
        assert!(counterpart("reply").is_none());
    }

    #[test]
    fn an_order_is_read_off_its_arguments() {
        assert_eq!(
            order(
                "mission_add",
                &args(r#"{"brief":"Check the doors.","steps":"- go to the garden\n\n* check the doors"}"#)
            ),
            Ok(Order::Add {
                brief: "Check the doors.".into(),
                steps: vec!["go to the garden".into(), "check the doors".into()],
            })
        );
        assert_eq!(
            order("mission_edit", &args(r#"{"mission":"Tidy.","steps":"dust"}"#)),
            Ok(Order::Edit {
                mission: "Tidy.".into(),
                brief: None,
                steps: Some(vec!["dust".into()]),
            })
        );
        assert_eq!(
            order("mission_cancel", &args(r#"{"mission":"Tidy.","why":"not needed"}"#)),
            Ok(Order::Cancel {
                mission: "Tidy.".into(),
                why: "not needed".into(),
            })
        );
        assert_eq!(
            order("mission_done", &args(r#"{"mission":"Tidy.","found":"it is tidy"}"#)),
            Ok(Order::Done {
                mission: "Tidy.".into(),
                found: "it is tidy".into(),
            })
        );
    }

    #[test]
    fn an_order_missing_what_it_needs_is_refused() {
        assert!(order("mission_add", &args(r#"{"brief":"x","steps":"  \n"}"#)).is_err());
        assert!(order("mission_edit", &args(r#"{"mission":"Tidy."}"#)).is_err());
        assert!(order("mission_cancel", &args(r#"{"mission":"Tidy."}"#)).is_err());
        assert!(order("mission_done", &args(r#"{"found":"x"}"#)).is_err());
        assert!(order("reply", &args("{}")).is_err());
    }
}
