//! The operator's half of the mission system: lodge a mission for a character,
//! and read how one turned out.
//!
//! `POST /v1/npc/:nid/mission` puts a mission on a character — queued for it to
//! collect at the command desk, or, with `start`, carried and read at once.
//! `GET /v1/npc/:nid/mission` reads what the character is carrying or last
//! finished, its outcome and its answer included. `POST .../mission/step` ticks
//! off or adds a step on the open one. The character's own half — collecting and
//! reporting — is the acts in [`crate::engine::mission_acts`], performed at the
//! desk.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::{err, owner_of, Authored};
use crate::engine::event::{EventKind, Salience};
use crate::engine::mission::{Mission, Origin, StepOutcome, Todo};
use crate::engine::{no_engine, owned, owned_by, speaking_as};
use npc_map::route;

#[derive(Debug, Deserialize)]
pub struct LodgeBody {
    /// The brief — what is being asked of the character.
    prompt: String,
    /// The steps, in order. May be empty: a mission can be carried on its ask
    /// alone, and the character may add steps as it discovers them.
    #[serde(default)]
    todo: Vec<String>,
    /// Carry it at once, rousing the character to read it now, rather than
    /// leaving it on the desk for the character to collect on its own. Absent
    /// means queue it for collection.
    #[serde(default)]
    start: bool,
}

/// `POST /v1/npc/:nid/mission` — lodge a mission for a character.
///
/// Ownership is checked: a mission is a thing asked of somebody's character, so
/// it is a write to something they own.
pub async fn lodge(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<String>,
    headers: HeaderMap,
    Json(body): Json<LodgeBody>,
) -> Response {
    let (id, owner, nid) = match owned_by(&s, &headers, &nid).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    let prompt = body.prompt.trim().to_owned();
    if prompt.is_empty() {
        return err(
            StatusCode::BAD_REQUEST,
            "empty_mission",
            "a mission needs a brief — what is being asked",
        );
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("lodging a mission");
    };
    let Some((hosted, character)) = rt.body_of(nid) else {
        return err(
            StatusCode::CONFLICT,
            "not_in_a_world",
            "that character has no body in a world to carry a mission",
        );
    };

    let mut todo: Vec<Todo> = body
        .todo
        .iter()
        .filter(|step| !step.trim().is_empty())
        .map(Todo::new)
        .collect();
    // **A brief with no steps still names where to go and whom to find.** Lodged
    // on its ask alone, a mission gave the engine nothing to sign off and the
    // character nothing to be told next, and was given up as stuck within a
    // minute. The rooms the character can reach and the people in its world
    // that the brief names become its steps, in the order the brief names them.
    if todo.is_empty() {
        let (rooms, places, people) = hosted.read(|w| {
            let Some(here) = w.actor(&character).map(|a| a.at.clone()) else {
                return (Vec::new(), Vec::new(), Vec::new());
            };
            let reach: Vec<_> = std::iter::once(here.clone())
                .chain(route::reachable_from(w.map(), &here))
                .collect();
            let rooms: Vec<String> = reach
                .iter()
                .filter_map(|at| w.node(at).map(|n| n.name.clone()))
                .collect();
            let places: Vec<String> = reach
                .iter()
                .map(|at| format!("{}/{}", at.area, at.node))
                .collect();
            let people: Vec<String> = w
                .actors()
                .filter(|a| a.id != character)
                .map(|a| a.name.clone())
                .collect();
            (rooms, places, people)
        });
        let machines: Vec<String> = hosted.sim(|s| {
            s.devices
                .iter()
                .filter(|d| places.contains(&d.at))
                .map(|d| d.name.clone())
                .collect()
        });
        todo = steps_named_in(&prompt, &rooms, &machines, &people);
    }
    if !todo.is_empty() {
        todo.push(Todo::report("go back to the table and report it"));
    }
    let steps = todo.len();
    let mission = Mission::new(
        &prompt,
        todo,
        Origin::Lodged {
            by: speaking_as(&id, &owner, &s.roles),
        },
    );

    let start = body.start;
    // When carried at once, read the rendered standing task back so the same
    // text can be put on the character's inbox to rouse it now — otherwise it
    // would sit unread until the next idle nudge, up to a minute and a half off.
    let now_text = hosted.with_sim(|sim| {
        if start {
            sim.missions.assign(&character, mission);
            sim.missions.active(&character).map(|m| m.standing_text())
        } else {
            sim.missions.lodge(&character, mission);
            None
        }
    });
    if let Some(text) = now_text {
        let world_ms = s.world_ms(nid).await;
        rt.scheduler
            .deliver(nid, world_ms, Salience::URGENT, EventKind::Nudge { text });
    }

    Json(json!({
        "lodged": !start,
        "carrying": start,
        "prompt": prompt,
        "steps": steps,
    }))
    .into_response()
}

/// `GET /v1/npc/:nid/mission` — read a character's current or last mission.
///
/// The open one if it is on a mission, otherwise the last it finished — with its
/// outcome and answer, which is the point of asking after a character that has
/// reported and moved on.
pub async fn status(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<String>,
    headers: HeaderMap,
) -> Response {
    let nid = match owned(&s, &headers, &nid).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("reading a mission");
    };
    let Some((hosted, character)) = rt.body_of(nid) else {
        // No body is no mission, not an error: a character out of the world
        // simply has none, which is what a caller wants to know.
        return Json(json!({
            "on_mission": false,
            "mission": Value::Null,
            "finished": Value::Null,
            "lodged": 0,
        }))
        .into_response();
    };
    // The last one it finished as well as the one in hand: a character that
    // reports and takes up the next at once would otherwise hide the outcome an
    // operator came to read behind the mission it has just started.
    let (on_mission, mission, finished, lodged) = hosted.sim(|sim| {
        (
            sim.missions.is_on_mission(&character),
            sim.missions.latest(&character).map(mission_view),
            sim.missions.done(&character).map(mission_view),
            sim.missions.lodged_count(&character),
        )
    });
    Json(json!({
        "on_mission": on_mission,
        "mission": mission,
        "finished": finished,
        "lodged": lodged,
    }))
    .into_response()
}

#[derive(Debug, Deserialize)]
pub struct StepBody {
    /// Sign off the first open step with this text.
    #[serde(default)]
    done: Option<String>,
    /// How the step turned out: `achieved` (the default) or `thwarted`.
    #[serde(default)]
    outcome: Option<StepOutcome>,
    /// Add a step with this text.
    #[serde(default)]
    add: Option<String>,
}

/// `POST /v1/npc/:nid/mission/step` — tick off or add a step on the character's
/// open mission. Either changes the steps its prompt section lists, so the
/// section is sealed again before its next turn.
pub async fn step(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<String>,
    headers: HeaderMap,
    Json(body): Json<StepBody>,
) -> Response {
    let nid = match owned(&s, &headers, &nid).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("changing a mission's steps");
    };
    let Some((hosted, character)) = rt.body_of(nid) else {
        return err(
            StatusCode::CONFLICT,
            "not_in_a_world",
            "that character has no body in a world to carry a mission",
        );
    };
    if body.done.is_none() && body.add.is_none() {
        return err(
            StatusCode::BAD_REQUEST,
            "no_step",
            "say a step to tick off with `done`, or a step to add with `add`",
        );
    }
    let (ticked, added) = hosted.with_sim(|sim| {
        let ticked = body.done.as_deref().map(|step| {
            let outcome = body.outcome.unwrap_or(StepOutcome::Achieved);
            sim.missions.check_off(&character, step, outcome)
        });
        let added = body
            .add
            .as_deref()
            .map(|step| sim.missions.add_todo(&character, step));
        (ticked, added)
    });
    Json(json!({ "ticked": ticked, "added": added })).into_response()
}

#[derive(Debug, Deserialize)]
pub struct TableBody {
    /// Open the command table, or shut it.
    open: bool,
}

/// `POST /v1/pulse/command-table` — open or shut the command table.
///
/// Opening it calls every character to the table — over the chat channel and
/// the tannoy — and sets the standing task, so a character with no mission
/// breaks off, makes its way there, and takes one up (with `collect_mission`),
/// carries it out, reports, and comes back for the next until the table shuts.
/// Shutting it stands the cast down: missions in hand are finished, no new ones
/// given. Admin, like `broadcast`: it reaches the whole cast.
pub async fn command_table(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Json(body): Json<TableBody>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("the command table");
    };
    rt.set_table_open(body.open);

    let (line, salience) = if body.open {
        (
            "The command table is open. Any of you without a task in hand, come to the command \
             table and take one up."
                .to_string(),
            Salience::URGENT,
        )
    } else {
        (
            "The command table is closed. Finish the task you are holding; no new ones are being \
             given out."
                .to_string(),
            Salience::NORMAL,
        )
    };
    // The chat channel — a line on every world's standing channel.
    rt.say_to_all_channels("Command", &line);
    // The tannoy — one announcement to the whole cast, loud enough on opening to
    // break a character off what it is doing. Each reads it on its own clock.
    let mut when: HashMap<u64, u64> = HashMap::new();
    for c in rt.scheduler.census() {
        when.insert(c.npc_id, s.world_ms(c.npc_id).await);
    }
    let reached = rt.scheduler.broadcast(
        |id| when.get(&id).copied().unwrap_or(0),
        salience,
        EventKind::Announcement { text: line },
    );

    Json(json!({ "open": body.open, "called": reached })).into_response()
}

/// `POST /v1/pulse/missions/cancel` — call off every character's mission.
///
/// The whole cast at once — for clearing the board before opening the table so
/// everyone is called to take a fresh one. Admin: it reaches characters the
/// caller does not own.
pub async fn cancel_all(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("cancelling missions");
    };
    let mut when: HashMap<u64, u64> = HashMap::new();
    for c in rt.scheduler.census() {
        when.insert(c.npc_id, s.world_ms(c.npc_id).await);
    }
    let cancelled = rt.cancel_all_missions(|id| when.get(&id).copied().unwrap_or(0));
    Json(json!({ "cancelled": cancelled })).into_response()
}

/// `POST /v1/npc/:nid/mission/cancel` — call off one character's mission.
pub async fn cancel(
    State(s): State<Arc<Authored>>,
    Path(nid): Path<String>,
    headers: HeaderMap,
) -> Response {
    let nid = match owned(&s, &headers, &nid).await {
        Ok(id) => id,
        Err(r) => return *r,
    };
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("cancelling a mission");
    };
    let world_ms = s.world_ms(nid).await;
    Json(json!({ "cancelled": rt.cancel_mission(nid, world_ms) })).into_response()
}

/// What a name in a brief is.
#[derive(Clone, Copy)]
enum Named {
    Room,
    Machine,
    Person,
}

/// The steps a brief names: a journey to each of `rooms`, a reading of each of
/// `machines` and a finding of each of `people` that appears in it, in the order
/// it names them. A name given twice, or inside a longer name also given, is
/// one step.
fn steps_named_in(
    brief: &str,
    rooms: &[String],
    machines: &[String],
    people: &[String],
) -> Vec<Todo> {
    let lower = brief.to_lowercase();
    let mut found: Vec<(usize, Todo)> = Vec::new();
    let mut taken: Vec<(usize, usize)> = Vec::new();
    let mut by_length: Vec<(&String, Named)> = rooms
        .iter()
        .map(|r| (r, Named::Room))
        .chain(machines.iter().map(|m| (m, Named::Machine)))
        .chain(people.iter().map(|p| (p, Named::Person)))
        .collect();
    by_length.sort_by_key(|(name, _)| std::cmp::Reverse(name.len()));
    for (name, kind) in by_length {
        let bare = name.trim().trim_start_matches("the ").to_lowercase();
        if bare.len() < 3 {
            continue;
        }
        let Some(at) = lower.find(&bare) else {
            continue;
        };
        let span = (at, at + bare.len());
        if taken.iter().any(|(a, b)| span.0 < *b && *a < span.1) {
            continue;
        }
        taken.push(span);
        let step = match kind {
            Named::Room => format!("go to {name}"),
            Named::Machine => format!("scan {name} and read what state it is in"),
            Named::Person => format!("find {name}"),
        };
        found.push((at, Todo::new(step)));
    }
    found.sort_by_key(|(at, _)| *at);
    found.into_iter().map(|(_, t)| t).collect()
}

/// One mission, as an operator reads it back.
fn mission_view(m: &Mission) -> Value {
    json!({
        "prompt": m.mission_text(),
        "open": m.is_open(),
        "origin": m.origin,
        "todo": m.todo,
        "answer": m.answer,
        "observed": m.observed,
        "report": m.report.as_ref().map(|r| json!({
            "outcome": r.outcome.as_str(),
            "notes": r.notes,
        })),
    })
}

#[cfg(test)]
mod tests {
    use super::{mission_view, steps_named_in};

    /// A brief with no steps is given a journey to each room and a finding of
    /// each person it names, in its own order — the longer room name winning
    /// over one inside it, and nothing it does not name.
    #[test]
    fn a_brief_with_no_steps_is_given_the_rooms_and_people_it_names() {
        let rooms = [
            "the plant room".to_string(),
            "the room".to_string(),
            "the anteroom".to_string(),
        ];
        let people = ["Paxon Vael".to_string(), "Ione Valtiere".to_string()];
        let machines = ["the coolant valve".to_string()];
        let steps: Vec<String> = steps_named_in(
            "Ask Paxon Vael who was in the plant room, then read the coolant valve.",
            &rooms,
            &machines,
            &people,
        )
        .into_iter()
        .map(|t| t.text)
        .collect();
        assert_eq!(
            steps,
            vec![
                "find Paxon Vael",
                "go to the plant room",
                "scan the coolant valve and read what state it is in"
            ]
        );
        assert!(steps_named_in("Think it over.", &rooms, &machines, &people).is_empty());
    }
    use crate::engine::mission::{Mission, Origin, Outcome, StepOutcome, Todo};

    #[test]
    fn the_view_shows_the_ask_the_steps_and_a_filed_report() {
        let mut m = Mission::new(
            "Check the record.",
            vec![Todo::new("read it"), Todo::new("compare it")],
            Origin::Lodged {
                by: "u_op".to_string(),
            },
        );
        // Open: no report, no answer, steps not yet done.
        let open = mission_view(&m);
        assert_eq!(open["prompt"], "Check the record.");
        assert_eq!(open["open"], true);
        assert_eq!(open["report"], serde_json::Value::Null);
        assert_eq!(open["answer"], serde_json::Value::Null);
        assert_eq!(open["todo"].as_array().unwrap().len(), 2);
        assert_eq!(open["todo"][0]["text"], "read it");
        assert_eq!(open["todo"][0]["done"], false);
        assert_eq!(open["todo"][0]["outcome"], serde_json::Value::Null);

        // Closed: the outcome and answer an operator is asking after.
        m.check_off("read it", StepOutcome::Achieved);
        m.check_off("compare it", StepOutcome::Thwarted);
        m.complete(
            Outcome::Pass,
            "it holds",
            Some("the date is wrong".to_string()),
        );
        let closed = mission_view(&m);
        assert_eq!(closed["open"], false);
        assert_eq!(closed["answer"], "the date is wrong");
        assert_eq!(closed["report"]["outcome"], "pass");
        assert_eq!(closed["report"]["notes"], "it holds");
        assert_eq!(closed["todo"][0]["done"], true);
        assert_eq!(closed["todo"][0]["outcome"], "achieved");
        assert_eq!(closed["todo"][1]["done"], true);
        assert_eq!(closed["todo"][1]["outcome"], "thwarted");
    }
}
