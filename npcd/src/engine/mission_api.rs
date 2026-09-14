//! The operator's half of the mission system: lodge a mission for a character,
//! and read how one turned out.
//!
//! `POST /v1/npc/:nid/mission` puts a mission on a character — queued for it to
//! collect at the command desk, or, with `start`, carried and read at once.
//! `GET /v1/npc/:nid/mission` reads what the character is carrying or last
//! finished, its outcome and its answer included. The character's own half —
//! collecting, recording progress, reporting — is the acts in
//! [`crate::engine::mission_acts`], performed at the desk and out in the world.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::{err, owner_of, Authored};
use crate::engine::event::{EventKind, Salience};
use crate::engine::mission::{Mission, Origin, Todo};
use crate::engine::{no_engine, speaking_as};

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
    Path(nid): Path<u64>,
    headers: HeaderMap,
    Json(body): Json<LodgeBody>,
) -> Response {
    let (id, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    if s.npcs.read().await.visible_to(nid, &owner).is_none() {
        return err(StatusCode::NOT_FOUND, "not_found", "no such character");
    }
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

    let todo: Vec<Todo> = body
        .todo
        .iter()
        .filter(|step| !step.trim().is_empty())
        .map(Todo::new)
        .collect();
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
    Path(nid): Path<u64>,
    headers: HeaderMap,
) -> Response {
    let (_, owner) = match owner_of(&s, &headers).await {
        Ok(v) => v,
        Err(r) => return *r,
    };
    if s.npcs.read().await.visible_to(nid, &owner).is_none() {
        return err(StatusCode::NOT_FOUND, "not_found", "no such character");
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("reading a mission");
    };
    let Some((hosted, character)) = rt.body_of(nid) else {
        // No body is no mission, not an error: a character out of the world
        // simply has none, which is what a caller wants to know.
        return Json(json!({ "on_mission": false, "mission": Value::Null })).into_response();
    };
    let (on_mission, mission) = hosted.sim(|sim| {
        (
            sim.missions.is_on_mission(&character),
            sim.missions.latest(&character).map(mission_view),
        )
    });
    Json(json!({ "on_mission": on_mission, "mission": mission })).into_response()
}

/// One mission, as an operator reads it back.
fn mission_view(m: &Mission) -> Value {
    json!({
        "prompt": m.mission_text(),
        "open": m.is_open(),
        "origin": m.origin,
        "todo": m.todo,
        "answer": m.answer,
        "report": m.report.as_ref().map(|r| json!({
            "outcome": r.outcome.as_str(),
            "notes": r.notes,
        })),
    })
}

#[cfg(test)]
mod tests {
    use super::mission_view;
    use crate::engine::mission::{Mission, Origin, Outcome, Todo};

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

        // Closed: the outcome and answer an operator is asking after.
        m.check_off("read it");
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
    }
}
