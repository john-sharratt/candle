//! The operations the command table holds, over HTTP: read them, put a
//! document through review, edit one, call one off.
//!
//! Every route is Admin, like the rest of the table: an operation's missions
//! are taken up by the whole cast.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::{err, owner_of, Authored};
use crate::engine::mission_gen::gates::Form;
use crate::engine::no_engine;
use crate::engine::runtime::Runtime;
use crate::engine::workflow::{Taker, Where, Workflow};
use crate::opstore::owner::Owner;
use crate::sim::operations::Operation;
use crate::sim::shelf::{Published, Shelf};
use crate::world::Hosted;

/// Where an operation stands, as the page names it: `running`, `succeeded`,
/// `failed` or `cancelled`.
fn state(op: &Operation) -> &'static str {
    match op.run.at {
        Where::NextStep(_) => "running",
        Where::Done => "succeeded",
        Where::Failed(_) => "failed",
        Where::Cancelled(_) => "cancelled",
    }
}

/// What an operator needs of one operation, gathered under the world's lock.
struct Seen {
    op: Operation,
    /// The steps of its workflow in order, and which are the table's.
    steps: Vec<(String, bool)>,
    waiting: Option<String>,
    carrying: Option<String>,
}

/// One operation as an operator reads it: where it stands in its workflow,
/// every step taken and by whom, and the brief of its mission still waiting at
/// the table.
fn view(hosted: &Hosted, seen: &Seen) -> Value {
    let op = &seen.op;
    let name_of = |body: &str| hosted.read(|w| w.actor(body).map(|a| a.name.clone()));
    let history: Vec<Value> = op
        .run
        .history
        .iter()
        .map(|t| {
            let (by, by_name) = match &t.by {
                Taker::Table => ("table".to_string(), Some("the table".to_string())),
                Taker::Actor(b) => (b.clone(), name_of(b)),
            };
            json!({
                "step": t.step,
                "by": by,
                "by_name": by_name,
                "outcome": t.outcome,
                "to": t.to,
                "notes": t.notes,
            })
        })
        .collect();
    let steps: Vec<Value> = seen
        .steps
        .iter()
        .map(|(name, table)| json!({ "name": name, "table": table }))
        .collect();
    json!({
        "id": op.id,
        "name": op.name,
        "objective": op.objective,
        "workflow": op.run.workflow,
        "steps": steps,
        "step": op.step(),
        "state": state(op),
        "finished": op.settled(),
        "round": op.run.round,
        "send_backs": op.run.send_backs,
        "generator": op.generator,
        "target": op.target,
        "document": op.document,
        "carrying": seen.carrying,
        "carrying_name": seen.carrying.as_deref().and_then(name_of),
        "history": history,
        "why": op.why(),
        "waiting_brief": seen.waiting,
        "published": op.published,
        "retry": op.retry.as_ref().map(|r| json!({ "at_ms": r.at_ms, "why": r.why })),
    })
}

/// The steps of `wf` in order, each with whether the table takes it.
fn steps_of(wf: Option<&Workflow>) -> Vec<(String, bool)> {
    wf.map(|w| {
        w.steps
            .iter()
            .map(|s| (s.name.clone(), s.call.is_some()))
            .collect()
    })
    .unwrap_or_default()
}

/// Every operation of one world, newest first.
fn world_ops(hosted: &Hosted) -> Vec<Value> {
    let ops: Vec<Seen> = hosted.sim(|s| {
        let ops = s.missions.operations();
        ops.all()
            .map(|o| Seen {
                op: o.clone(),
                steps: steps_of(ops.workflow_of(o)),
                waiting: s.missions.waiting_brief(o.id).map(str::to_string),
                carrying: s.missions.carrying(o.id).map(str::to_string),
            })
            .collect()
    });
    ops.iter().map(|seen| view(hosted, seen)).collect()
}

/// The workflow a document is put through review on, by its form: a life
/// event's, a story's, or — for any other page of the record — a
/// correction's.
fn workflow_for(path: &str) -> &'static str {
    match Form::of(path) {
        Form::LifeEvent => "life-event",
        Form::Story => "story",
        Form::Other => "correction",
    }
}

/// The first step of `wf` the table takes — where a document already written
/// is put through review.
fn first_reading(wf: &Workflow) -> Option<&str> {
    wf.steps
        .iter()
        .find(|s| s.call.is_some())
        .map(|s| s.name.as_str())
}

/// `GET /v1/pulse/operations` — every world's operations, newest first.
pub async fn list(State(s): State<Arc<Authored>>, headers: HeaderMap) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("reading the operations");
    };
    let mut worlds = serde_json::Map::new();
    for id in rt.hosted.ids() {
        if let Some(hosted) = rt.hosted.get(&id) {
            worlds.insert(id, Value::Array(world_ops(&hosted)));
        }
    }
    Json(json!({ "worlds": worlds })).into_response()
}

#[derive(Debug, Deserialize)]
pub struct ReviewBody {
    /// The mind path of the document to put through review.
    path: String,
    /// Which world; absent is every world with documents.
    #[serde(default)]
    world: Option<String>,
    /// The workflow to put it through; absent is the one for its form — a
    /// life event's, a story's, or a correction's for any other page.
    #[serde(default)]
    workflow: Option<String>,
}

/// `POST /v1/pulse/operations` — open an operation that reviews a document
/// already on the record: it starts at its workflow's first reading by the
/// table, and goes on from there as any other. For work done outside the
/// table.
pub async fn review(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Json(body): Json<ReviewBody>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("opening an operation");
    };
    let path = body.path.trim().trim_start_matches('/').to_string();
    if !rt.mind.as_ref().is_some_and(|m| m.join(&path).is_file()) {
        return err(
            StatusCode::NOT_FOUND,
            "no_document",
            "there is no such document in the mind to review",
        );
    }
    let workflow = body
        .workflow
        .clone()
        .unwrap_or_else(|| workflow_for(&path).to_string());
    let mut opened = Vec::new();
    for id in rt.hosted.ids() {
        if body.world.as_deref().is_some_and(|w| w != id) {
            continue;
        }
        // Each member opens it in the worlds it runs.
        if !rt.standing.writes(&id) {
            continue;
        }
        let Some(hosted) = rt.hosted.get(&id) else {
            continue;
        };
        if !hosted.sim(|sim| sim.bench.has_root()) {
            continue;
        }
        let op = hosted.with_sim(|sim| {
            let wf = sim
                .missions
                .operations()
                .workflow(&workflow)
                .ok_or_else(|| format!("there is no workflow called `{workflow}`"))?;
            let at = first_reading(wf)
                .ok_or_else(|| format!("the workflow `{workflow}` has no step the table reads"))?
                .to_string();
            sim.missions.review_document(&workflow, &at, &path)
        });
        match op {
            Ok(op) => {
                // Its store takes the record as it stands as its base (§10.3).
                rt.ops.pin(&Owner::operation(&id, op));
                opened.push(json!({ "world": id, "operation": op }))
            }
            Err(e) => return err(StatusCode::UNPROCESSABLE_ENTITY, "no_workflow", &e),
        }
    }
    Json(json!({ "path": path, "workflow": workflow, "opened": opened })).into_response()
}

#[derive(Debug, Default, Deserialize)]
pub struct EditBody {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    objective: Option<String>,
    /// The brief of its mission still waiting at the table.
    #[serde(default)]
    brief: Option<String>,
}

/// World `wid`, when this member runs it: an operation is the world's, and only
/// its authority changes one (`docs/replicated_substrate.md` §9.2). A member
/// that only follows the world refuses, `409`, and the gateway sends the next
/// one to the authority.
fn run_here(rt: &Runtime, wid: &str) -> Result<Arc<Hosted>, Box<Response>> {
    let Some(hosted) = rt.hosted.get(wid) else {
        return Err(Box::new(err(
            StatusCode::NOT_FOUND,
            "no_world",
            "no such world is running",
        )));
    };
    if !rt.standing.writes(wid) {
        return Err(Box::new(err(
            StatusCode::CONFLICT,
            "run_elsewhere",
            "another member is this world's authority now — ask again",
        )));
    }
    Ok(hosted)
}

/// `PATCH /v1/pulse/operations/:wid/:oid` — rename an operation, restate what
/// it is for, or rewrite the brief of its waiting mission.
pub async fn edit(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((wid, oid)): Path<(String, u64)>,
    Json(body): Json<EditBody>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("editing an operation");
    };
    let hosted = match run_here(rt, &wid) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let edited = hosted.with_sim(|sim| {
        sim.missions
            .edit_operation(
                oid,
                body.name.as_deref(),
                body.objective.as_deref(),
                body.brief.as_deref(),
            )
            .map(|o| o.id)
    });
    match edited {
        Ok(_) => {
            let one = world_ops(&hosted)
                .into_iter()
                .find(|v| v["id"] == json!(oid))
                .unwrap_or(Value::Null);
            Json(one).into_response()
        }
        Err(e) if e == "no such operation" => err(StatusCode::NOT_FOUND, "no_operation", &e),
        Err(e) => err(StatusCode::UNPROCESSABLE_ENTITY, "bad_edit", &e),
    }
}

#[derive(Debug, Deserialize)]
pub struct StepBody {
    /// The step of the operation's workflow to send it to.
    step: String,
}

/// `POST /v1/pulse/operations/:wid/:oid/step` — send an operation to a step
/// of its workflow, in a new round: a running one leaves whatever it waited on
/// (its step not being carried), a finished one is reopened there — read
/// again by the table, say, or a passed story checked again against the
/// storyline. A kept store resumes as it was; a discarded one starts afresh
/// on the record as it stands now (§10.3).
pub async fn step(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((wid, oid)): Path<(String, u64)>,
    Json(body): Json<StepBody>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("moving an operation");
    };
    let hosted = match run_here(rt, &wid) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let step = body.step.trim();
    match hosted.with_sim(|sim| sim.missions.send_to_step(oid, step)) {
        Ok(()) => {
            rt.ops.pin(&Owner::operation(&wid, oid));
            Json(json!({ "operation": oid, "step": step })).into_response()
        }
        Err(e) if e == "no such operation" => err(StatusCode::NOT_FOUND, "no_operation", &e),
        Err(e) => err(StatusCode::CONFLICT, "not_moved", &e),
    }
}

/// `POST /v1/pulse/operations/:wid/:oid/retry` — offer a publish that is
/// waiting on git at once, rather than when its wait is over (§10.3).
pub async fn retry(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((wid, oid)): Path<(String, u64)>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("offering a publish again");
    };
    let hosted = match run_here(rt, &wid) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    match hosted.with_sim(|sim| sim.missions.retry_now(oid)) {
        true => Json(json!({ "operation": oid, "offered": true })).into_response(),
        false => err(
            StatusCode::CONFLICT,
            "not_waiting",
            "that operation has no publish waiting to be tried again",
        ),
    }
}

/// Where an operation's document stands, against the record: `record` when
/// its store holds what the record holds, `operation` when the store holds
/// work the record does not have yet.
fn standing_of(stored: &str, published: Option<&str>) -> &'static str {
    match published == Some(stored) {
        true => "record",
        false => "operation",
    }
}

/// `GET /v1/pulse/operations/:wid/:oid/document` — the text of operation
/// `oid`'s document as its store holds it (§10.3): the record's, under
/// whatever the operation has changed. 404 `no_document` when neither has it
/// — a draft never committed, or a discarded store.
pub async fn document(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((wid, oid)): Path<(String, u64)>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("reading an operation's document");
    };
    let Some(hosted) = rt.hosted.get(&wid) else {
        return err(
            StatusCode::NOT_FOUND,
            "no_world",
            "no such world is running",
        );
    };
    let Some(op) = hosted.sim(|sim| sim.missions.operations().get(oid).cloned()) else {
        return err(StatusCode::NOT_FOUND, "no_operation", "no such operation");
    };
    let Some(root) = rt.mind.as_ref() else {
        return err(
            StatusCode::NOT_FOUND,
            "no_document",
            "there is no mind to read it from",
        );
    };
    let stored = hosted
        .op_shelf(oid)
        .and_then(|shelf| shelf.read(&op.document));
    let published = Published { root: root.clone() }.read(&op.document);
    match stored {
        Some(text) => Json(json!({
            "path": op.document,
            "where": standing_of(&text, published.as_deref()),
            "words": text.split_whitespace().count(),
            "text": text,
        }))
        .into_response(),
        None => err(
            StatusCode::NOT_FOUND,
            "no_document",
            "neither the record nor the operation holds the document — it was never committed, \
             or the operation's work was discarded",
        ),
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct CancelBody {
    #[serde(default)]
    why: Option<String>,
}

/// `POST /v1/pulse/operations/:wid/:oid/cancel` — call an operation off: its
/// waiting mission leaves the table, and whoever carries a stage of it is stood
/// down.
pub async fn cancel(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((wid, oid)): Path<(String, u64)>,
    body: Option<Json<CancelBody>>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("calling off an operation");
    };
    let hosted = match run_here(rt, &wid) {
        Ok(h) => h,
        Err(r) => return *r,
    };
    let why = body
        .and_then(|Json(b)| b.why)
        .filter(|w| !w.trim().is_empty())
        .unwrap_or_else(|| "called off by an operator".into());
    match hosted.with_sim(|sim| sim.missions.cancel_operation(oid, &why)) {
        true => Json(json!({ "cancelled": oid })).into_response(),
        false => err(
            StatusCode::CONFLICT,
            "not_running",
            "there is no such operation running to call off",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::operations::tests::workflows;
    use crate::sim::operations::Operations;

    /// **A document is put through review on the workflow for its form**, at
    /// that workflow's first reading by the table.
    #[test]
    fn a_document_is_reviewed_on_the_workflow_for_its_form() {
        assert_eq!(
            workflow_for("layers/life/keeper/2786 The Charge.md"),
            "life-event"
        );
        assert_eq!(workflow_for("layers/stories/the-ledger.md"), "story");
        assert_eq!(workflow_for("layers/eras/the-fall.md"), "correction");
        let all = workflows();
        for name in ["life-event", "story", "correction"] {
            let wf = all.iter().find(|w| w.name == name).unwrap();
            assert_eq!(first_reading(wf), Some("read"), "{name}");
        }
    }

    /// **An operation is shown by where it stands in its workflow** and every
    /// step taken.
    #[test]
    fn an_operation_is_shown_by_its_workflow() {
        let mut ops = Operations::default();
        ops.set_workflows(workflows());
        let id = ops
            .open(
                "story",
                None,
                "untold",
                "era:x",
                "Tell X",
                "layers/stories/x.md",
            )
            .unwrap();
        let op = ops.get(id).unwrap();
        assert_eq!(state(op), "running");
        let steps = steps_of(ops.workflow_of(op));
        assert_eq!(
            steps,
            [
                ("write".to_string(), false),
                ("fix".to_string(), false),
                ("read".to_string(), true),
                ("review".to_string(), false),
                ("reread".to_string(), true),
                ("canon".to_string(), false),
                ("publish".to_string(), false),
            ]
        );
        ops.cancel(id, "not wanted");
        assert_eq!(state(ops.get(id).unwrap()), "cancelled");
    }

    /// **A document is the record's when its store holds what the record
    /// does**, and the operation's while it holds work the record lacks.
    #[test]
    fn a_document_stands_on_the_record_or_in_its_operation() {
        assert_eq!(standing_of("the charge", Some("the charge")), "record");
        assert_eq!(
            standing_of("the charge, mended", Some("the charge")),
            "operation"
        );
        assert_eq!(standing_of("a new draft", None), "operation");
    }
}
