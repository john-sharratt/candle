//! What an agenda act does to the agenda (`docs/zen_moon_mind.md` §17.2).

use serde_json::{Map, Value};

use crate::engine::act::Act;
use crate::engine::body::Outcome;
use crate::engine::mission::{Mission, Origin, Todo};
use crate::sim::agenda::Priority;
use crate::sim::Sim;
use crate::world::Hosted;

fn text(args: &Map<String, Value>, key: &str) -> Option<String> {
    let s = args.get(key)?.as_str()?.trim();
    (!s.is_empty()).then(|| s.to_string())
}

/// Steps given one to a line, a leading bullet dropped and blank lines left
/// out — as `order_post` takes them.
fn steps(args: &Map<String, Value>) -> Option<Vec<String>> {
    let lines: Vec<String> = text(args, "steps")?
        .lines()
        .map(|l| l.trim().trim_start_matches(['-', '*']).trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    (!lines.is_empty()).then_some(lines)
}

/// Perform agenda act `act` for `body`.
pub fn perform(hosted: &Hosted, body: &str, act: &Act) -> Outcome {
    let a = &act.args;
    if !hosted.sim(|s| s.missions.is_unordered(body)) {
        return Outcome::Refused(
            "You keep no agenda of your own: your work is what the table gives you.".into(),
        );
    }
    let said = |r: Result<String, String>| match r {
        Ok(did) => Outcome::Did(did),
        Err(why) => Outcome::Refused(why),
    };
    match act.tool {
        "mission_add" => {
            let (Some(title), Some(brief)) = (text(a, "title"), text(a, "brief")) else {
                return Outcome::Refused(
                    "A mission needs a title to be named by and a brief of what is to be done."
                        .into(),
                );
            };
            let Some(steps) = steps(a) else {
                return Outcome::Refused(
                    "Give it at least one step, one to a line, that you can be seen to do — \
                     \"go to the watch room\", \"post work on the order table\"."
                        .into(),
                );
            };
            let Some(priority) = text(a, "priority").and_then(|p| Priority::parse(&p)) else {
                return Outcome::Refused("Say how soon: `now`, `soon` or `later`.".into());
            };
            let mission = Mission::new(
                brief,
                steps.into_iter().map(Todo::new).collect(),
                Origin::Own,
            );
            said(hosted.with_sim(|s| {
                s.missions
                    .agenda_add(body, mission, &title, priority)
                    .map(|()| worked_now(s, body, &title))
            }))
        }
        "mission_edit" => {
            let Some(title) = text(a, "mission") else {
                return Outcome::Refused("Say which mission, by its title.".into());
            };
            let priority = match text(a, "priority") {
                None => None,
                Some(p) => match Priority::parse(&p) {
                    Some(p) => Some(p),
                    None => {
                        return Outcome::Refused("Say how soon: `now`, `soon` or `later`.".into())
                    }
                },
            };
            let (brief, steps) = (text(a, "brief"), steps(a));
            if brief.is_none() && steps.is_none() && priority.is_none() {
                return Outcome::Refused(
                    "You meant to change it, but said nothing to change: a brief, its steps, or \
                     how soon."
                        .into(),
                );
            }
            said(hosted.with_sim(|s| {
                s.missions
                    .agenda_edit(body, &title, brief, steps, priority)
                    .map(|()| worked_now(s, body, &title))
            }))
        }
        "mission_cancel" => {
            let (Some(title), Some(why)) = (text(a, "mission"), text(a, "why")) else {
                return Outcome::Refused("Say which mission, and why it is dropped.".into());
            };
            said(hosted.with_sim(|s| {
                s.missions
                    .agenda_cancel(body, &title, &why)
                    .map(|_| format!("Dropped: {title}. {}", next_line(s, body)))
            }))
        }
        "mission_done" => {
            let (Some(title), Some(found)) = (text(a, "mission"), text(a, "found")) else {
                return Outcome::Refused("Say which mission, and what you found.".into());
            };
            said(hosted.with_sim(|s| {
                s.missions
                    .agenda_done(body, &title, &found)
                    .map(|_| format!("Done: {title}. {}", next_line(s, body)))
            }))
        }
        other => Outcome::Refused(format!("`{other}` is no act of your agenda.")),
    }
}

/// What a change to the agenda leaves being worked, said after the change.
fn worked_now(s: &Sim, body: &str, title: &str) -> String {
    let top = s
        .missions
        .active(body)
        .and_then(|m| m.agendum.as_ref())
        .map(|a| a.title.clone());
    match top.as_deref() {
        Some(t) if t.eq_ignore_ascii_case(title) => {
            format!("On your agenda, and worked now: {title}.")
        }
        Some(t) => format!("On your agenda: {title}. You are working on {t} first."),
        None => format!("On your agenda: {title}."),
    }
}

/// What is worked next, after a mission leaves the agenda.
fn next_line(s: &Sim, body: &str) -> String {
    match s
        .missions
        .active(body)
        .and_then(|m| m.agendum.as_ref())
        .map(|a| a.title.clone())
    {
        Some(t) => format!("Next on your agenda: {t}."),
        None => "Your agenda is empty.".to_string(),
    }
}
