//! The operations the command table holds, over HTTP: read them, put a
//! document through review, edit one, call one off.
//!
//! Every route is Admin, like the rest of the table: an operation's missions
//! are taken up by the whole cast.

use std::path::Path as FsPath;
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
use crate::engine::workflow::{Taker, Where, Workflow};
use crate::sim::operations::Operation;
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
            Ok(op) => opened.push(json!({ "world": id, "operation": op })),
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
    let Some(hosted) = rt.hosted.get(&wid) else {
        return err(
            StatusCode::NOT_FOUND,
            "no_world",
            "no such world is running",
        );
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
/// storyline.
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
    let Some(hosted) = rt.hosted.get(&wid) else {
        return err(
            StatusCode::NOT_FOUND,
            "no_world",
            "no such world is running",
        );
    };
    let step = body.step.trim();
    match hosted.with_sim(|sim| sim.missions.send_to_step(oid, step)) {
        Ok(()) => Json(json!({ "operation": oid, "step": step })).into_response(),
        Err(e) if e == "no such operation" => err(StatusCode::NOT_FOUND, "no_operation", &e),
        Err(e) => err(StatusCode::CONFLICT, "not_moved", &e),
    }
}

/// Where an operation's document is now, as a mind path, and whether that is
/// the record or among the rejected. `None` when it is in neither.
///
/// A rejected draft is moved to `rejected/<the operation's name>/<its file>`
/// (`Benches::retire`), so it is looked for there first, then in any other
/// folder of the rejected — a draft an operator moved aside by hand.
fn find_document(root: &FsPath, op: &Operation) -> Option<(String, &'static str)> {
    if root.join(&op.document).is_file() {
        return Some((op.document.clone(), "record"));
    }
    let file = op.document.rsplit('/').next()?;
    let slug: String = op
        .name
        .to_lowercase()
        .replace(' ', "-")
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '-')
        .collect();
    let own = format!("rejected/{slug}/{file}");
    if root.join(&own).is_file() {
        return Some((own, "rejected"));
    }
    let mut folders: Vec<String> = std::fs::read_dir(root.join("rejected"))
        .ok()?
        .flatten()
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    folders.sort();
    folders
        .into_iter()
        .map(|f| format!("rejected/{f}/{file}"))
        .find(|p| root.join(p).is_file())
        .map(|p| (p, "rejected"))
}

/// `GET /v1/pulse/operations/:wid/:oid/document` — the text of the document
/// operation `oid` wrote: from the record, or from where its rejection moved
/// it. 404 `no_document` when it is in neither — a draft never committed.
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
    let found = find_document(root, &op)
        .and_then(|(path, at)| Some((std::fs::read_to_string(root.join(&path)).ok()?, path, at)));
    match found {
        Some((text, path, at)) => Json(json!({
            "path": path,
            "where": at,
            "words": text.split_whitespace().count(),
            "text": text,
        }))
        .into_response(),
        None => err(
            StatusCode::NOT_FOUND,
            "no_document",
            "the document is neither on the record nor among the rejected — it was never committed",
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
    let Some(hosted) = rt.hosted.get(&wid) else {
        return err(
            StatusCode::NOT_FOUND,
            "no_world",
            "no such world is running",
        );
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
            ]
        );
        ops.cancel(id, "not wanted");
        assert_eq!(state(ops.get(id).unwrap()), "cancelled");
    }

    /// **An operation's document is found where it is now**: on the record,
    /// in the operation's own rejected folder, or moved aside by hand into any
    /// other — and nowhere when it was never committed.
    #[test]
    fn an_operations_document_is_found_wherever_it_went() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let mut ops = Operations::default();
        ops.set_workflows(workflows());
        let id = ops
            .open(
                "life-event",
                None,
                "life-event",
                "life:keeper",
                "Keeper's charge",
                "layers/life/keeper/2786 The Charge.md",
            )
            .unwrap();
        let op = ops.get(id).unwrap().clone();
        let put = |rel: &str| {
            let p = root.join(rel);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, "the charge").unwrap();
        };
        assert_eq!(find_document(root, &op), None, "never committed");

        put("rejected/cleanup/2786 The Charge.md");
        assert_eq!(
            find_document(root, &op),
            Some(("rejected/cleanup/2786 The Charge.md".into(), "rejected"))
        );
        let own = format!(
            "rejected/{}/2786 The Charge.md",
            op.name.to_lowercase().replace(' ', "-")
        );
        put(&own);
        assert_eq!(
            find_document(root, &op),
            Some((own, "rejected")),
            "its own first"
        );
        put(&op.document);
        assert_eq!(
            find_document(root, &op),
            Some((op.document.clone(), "record")),
            "the record first of all"
        );
    }
}
