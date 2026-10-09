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
use crate::engine::no_engine;
use crate::sim::operations::Operation;
use crate::world::Hosted;

/// One operation as an operator reads it, with the names of who carried it and
/// the brief of its mission still waiting at the table.
fn view(hosted: &Hosted, op: &Operation, waiting: Option<&str>, carrying: Option<&str>) -> Value {
    let name_of =
        |body: Option<&str>| body.and_then(|b| hosted.read(|w| w.actor(b).map(|a| a.name.clone())));
    json!({
        "id": op.id,
        "name": op.name,
        "objective": op.objective,
        "phase": op.phase,
        "finished": op.phase.finished(),
        "generator": op.generator,
        "target": op.target,
        "document": op.document,
        "writer": op.writer,
        "writer_name": name_of(op.writer.as_deref()),
        "reviewer": op.reviewer,
        "reviewer_name": name_of(op.reviewer.as_deref()),
        "checker": op.checker,
        "checker_name": name_of(op.checker.as_deref()),
        "carrying": carrying,
        "carrying_name": name_of(carrying),
        "reading": op.reading,
        "log": op.log,
        "why": op.why,
        "waiting_brief": waiting,
    })
}

/// Every operation of one world, newest first.
fn world_ops(hosted: &Hosted) -> Vec<Value> {
    let ops: Vec<(Operation, Option<String>, Option<String>)> = hosted.sim(|s| {
        s.missions
            .operations()
            .all()
            .map(|o| {
                (
                    o.clone(),
                    s.missions.waiting_brief(o.id).map(str::to_string),
                    s.missions.carrying(o.id).map(str::to_string),
                )
            })
            .collect()
    });
    ops.iter()
        .map(|(o, w, c)| view(hosted, o, w.as_deref(), c.as_deref()))
        .collect()
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
}

/// `POST /v1/pulse/operations` — open an operation that reviews a document
/// already on the record: the table reads it, and a Maker reviews it. For work
/// done outside the table.
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
    let mut opened = Vec::new();
    for id in rt.hosted.ids() {
        if body.world.as_deref().is_some_and(|w| w != id) {
            continue;
        }
        let Some(hosted) = rt.hosted.get(&id) else {
            continue;
        };
        if hosted.sim(|sim| sim.bench.has_root()) {
            let op = hosted.with_sim(|sim| sim.missions.review_document(&path));
            opened.push(json!({ "world": id, "operation": op }));
        }
    }
    Json(json!({ "path": path, "opened": opened })).into_response()
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

/// `POST /v1/pulse/operations/:wid/:oid/read-again` — send an operation's
/// review, still waiting at the table, back to the table's reading, so it is
/// set again from a fresh reading under the current prompt.
pub async fn read_again(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((wid, oid)): Path<(String, u64)>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("reading an operation again");
    };
    let Some(hosted) = rt.hosted.get(&wid) else {
        return err(
            StatusCode::NOT_FOUND,
            "no_world",
            "no such world is running",
        );
    };
    match hosted.with_sim(|sim| sim.missions.read_again(oid)) {
        true => Json(json!({ "reading": oid })).into_response(),
        false => err(
            StatusCode::CONFLICT,
            "not_waiting",
            "the operation has no review waiting at the table",
        ),
    }
}

/// `POST /v1/pulse/operations/:wid/:oid/check` — send a succeeded life event
/// or story to be checked against the main storyline.
pub async fn check(
    State(s): State<Arc<Authored>>,
    headers: HeaderMap,
    Path((wid, oid)): Path<(String, u64)>,
) -> Response {
    if let Err(r) = owner_of(&s, &headers).await {
        return *r;
    }
    let Some(rt) = s.runtime.as_ref() else {
        return no_engine("checking an operation");
    };
    let Some(hosted) = rt.hosted.get(&wid) else {
        return err(
            StatusCode::NOT_FOUND,
            "no_world",
            "no such world is running",
        );
    };
    match hosted.with_sim(|sim| sim.missions.recheck(oid)) {
        true => Json(json!({ "checking": oid })).into_response(),
        false => err(
            StatusCode::CONFLICT,
            "not_lore",
            "only a life event or story that has passed can be sent to be checked",
        ),
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
    use crate::sim::operations::Operations;

    /// **An operation's document is found where it is now**: on the record,
    /// in the operation's own rejected folder, or moved aside by hand into any
    /// other — and nowhere when it was never committed.
    #[test]
    fn an_operations_document_is_found_wherever_it_went() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let mut ops = Operations::default();
        let id = ops.open(
            "life-event",
            "life:keeper",
            "Keeper's charge",
            "layers/life/keeper/2786 The Charge.md",
        );
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
