//! The API the life editor talks to.
//!
//! # `who` is resolved, never trusted
//!
//! Every route here takes a character id from the URL and turns it into a
//! directory and a filename. So it is checked twice, on purpose:
//!
//! - **Resolved against the personality registry** here, so a request can only
//!   name a character that exists. This is the strong check.
//! - **Shape-checked** in [`super::plan::is_safe_id`], where the path is
//!   actually built. This is the one that still holds if some future caller
//!   forgets the first.
//!
//! A path built from a URL is worth two checks in two files. One of them would
//! be a comment saying the caller had better have been careful.
//!
//! # Editing writes through to disk immediately
//!
//! A save writes the plan, regenerates the affected documents and lets the
//! watcher ingest them. There is no separate publish step, because a plan and a
//! directory that disagree is a state somebody would have to reconcile by hand
//! — and the documents are cheap to write.

use std::sync::Arc;

use axum::extract::{Path as UrlPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use super::consequence::{self, Consequence};
use super::document;
use super::job::NotStarted;
use super::plan::{self, NodeId, Phase, Plan};
use super::seed::{check, Seed};
use crate::api::Authored;

/// An error shaped like every other one this daemon returns.
fn err(code: StatusCode, kind: &str, detail: &str) -> Response {
    (code, Json(json!({ "error": kind, "detail": detail }))).into_response()
}

/// A refusal on its way out of a helper.
///
/// Boxed, and the box is not ceremony: an axum `Response` is large, so a
/// `Result<PathBuf, Response>` makes every success carry the width of a failure
/// on the stack. The refusal path is the rare one, so it is the one that pays.
type Refusal = Box<Response>;

fn refuse(code: StatusCode, kind: &str, detail: &str) -> Refusal {
    Box::new(err(code, kind, detail))
}

/// Resolve a character id to a mind root, refusing anything the registry does
/// not know.
pub(super) async fn resolve(s: &Arc<Authored>, who: &str) -> Result<std::path::PathBuf, Refusal> {
    let Some(mind) = s.mind.root() else {
        return Err(refuse(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_mind",
            "this daemon was started without a mind directory, so it has nowhere to write a life",
        ));
    };
    if !plan::is_safe_id(who) {
        return Err(refuse(
            StatusCode::BAD_REQUEST,
            "bad_character_id",
            "a character id is lowercase letters, digits, `-` and `_`",
        ));
    }
    if s.personalities.read().await.get(who).is_none() {
        return Err(refuse(StatusCode::NOT_FOUND, "personality_not_found", who));
    }
    Ok(mind.to_path_buf())
}

/// Load the plan for a character, or report that there is not one.
pub(super) fn open(mind: &std::path::Path, who: &str) -> Result<Plan, Refusal> {
    match plan::load(mind, who) {
        Ok(Some(p)) => Ok(p),
        Ok(None) => Err(refuse(
            StatusCode::NOT_FOUND,
            "no_life_plan",
            "this character has no life plan yet — give it a seed first",
        )),
        Err(e) => Err(refuse(
            StatusCode::INTERNAL_SERVER_ERROR,
            "life_plan_unreadable",
            &format!("{e:#}"),
        )),
    }
}

/// Persist a plan and bring its documents into line with it.
pub(super) fn commit(mind: &std::path::Path, p: &Plan) -> Result<Value, Refusal> {
    let fail = |e: anyhow::Error| {
        refuse(
            StatusCode::INTERNAL_SERVER_ERROR,
            "life_write_failed",
            &format!("{e:#}"),
        )
    };
    if p.story.content.is_generated() {
        document::write_story(mind, &p.seed.who, &p.story.content.text).map_err(fail)?;
    }
    let written = document::sync(mind, p).map_err(fail)?;
    plan::save(mind, p).map_err(fail)?;
    Ok(json!({ "documents": written }))
}

/// `GET /v1/life/:who` — the plan, with its counts.
pub async fn get_life(State(s): State<Arc<Authored>>, UrlPath(who): UrlPath<String>) -> Response {
    let mind = match resolve(&s, &who).await {
        Ok(m) => m,
        Err(r) => return *r,
    };
    match plan::load(&mind, &who) {
        // A character with no plan is ordinary, not an error — the console
        // renders the seed form rather than a failure.
        Ok(None) => Json(json!({ "who": who, "plan": Value::Null })).into_response(),
        Ok(Some(p)) => {
            let counts = p.counts();
            let job = s.lifegen.for_who(&who).map(|j| j.view());
            Json(json!({
                "who": who,
                "plan": p,
                "counts": counts,
                "prefix_hash": p.prefix_hash(),
                "job": job,
            }))
            .into_response()
        }
        Err(e) => err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "life_plan_unreadable",
            &format!("{e:#}"),
        ),
    }
}

/// `PUT /v1/life/:who/seed` — create or re-seed a life.
///
/// Re-seeding an existing plan keeps everything already written: the years and
/// months are re-laid-out against the new dates, and content for a stratum that
/// still exists survives. A seed edit that silently discarded a generated life
/// would make the dates un-correctable in practice.
pub async fn put_seed(
    State(s): State<Arc<Authored>>,
    UrlPath(who): UrlPath<String>,
    Json(mut body): Json<Seed>,
) -> Response {
    let mind = match resolve(&s, &who).await {
        Ok(m) => m,
        Err(r) => return *r,
    };
    // The id comes from the URL and the body's own is discarded rather than
    // trusted: a document that can name its own file can name somebody else's.
    body.who = who.clone();

    let checked = match check(&body) {
        Ok(c) => c,
        Err(bad) => {
            return (
                StatusCode::BAD_REQUEST,
                Json(json!({
                    "error": "bad_seed",
                    "problems": bad.iter().map(|b| json!({
                        "problem": b,
                        "message": b.message(),
                    })).collect::<Vec<_>>(),
                })),
            )
                .into_response()
        }
    };

    let mut fresh = Plan::new(&checked);
    // **An unreadable plan is refused, not overwritten.** The carry-over below reads the
    // existing life so a re-seed keeps everything already written; matching `Ok(Some(old))`
    // let the `Err` arm fall through to a fresh empty plan, so a plan file that failed to
    // parse — the one moment the authored life is least recoverable — was replaced by a blank
    // one instead of reported.
    if let Err(e) = plan::load(&mind, &who) {
        return err(
            StatusCode::INTERNAL_SERVER_ERROR,
            "life_plan_unreadable",
            &format!("the existing plan could not be read, so re-seeding would discard it: {e:#}"),
        );
    }
    if let Ok(Some(old)) = plan::load(&mind, &who) {
        fresh.story = old.story.clone();
        for y in &mut fresh.years {
            let Some(was) = old.year(y.year) else {
                continue;
            };
            y.content = was.content.clone();
            for m in &mut y.months {
                if let Some(wm) = was.months.iter().find(|x| x.month == m.month) {
                    m.content = wm.content.clone();
                    m.days = wm.days.clone();
                }
            }
        }
    }
    match commit(&mind, &fresh) {
        Ok(v) => {
            Json(json!({ "plan": fresh, "counts": fresh.counts(), "written": v })).into_response()
        }
        Err(r) => *r,
    }
}

/// One node's text, as the editor sends it.
#[derive(Debug, Deserialize)]
pub struct NodeEdit {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub text: String,
}

/// `PUT /v1/life/:who/node/:key` — an operator's edit of one stratum.
///
/// `key` is `story`, `1998`, `1998-09` or `1998-09-14`. Marks the node edited
/// (so a regeneration will not overwrite it) and everything below it stale (so
/// the console can offer to bring the subtree back into line).
pub async fn put_node(
    State(s): State<Arc<Authored>>,
    UrlPath((who, key)): UrlPath<(String, String)>,
    Json(body): Json<NodeEdit>,
) -> Response {
    let mind = match resolve(&s, &who).await {
        Ok(m) => m,
        Err(r) => return *r,
    };
    let mut p = match open(&mind, &who) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let Some(id) = node_id(&key) else {
        return err(
            StatusCode::BAD_REQUEST,
            "bad_node",
            "a node is `story`, `YYYY`, `YYYY-MM` or `YYYY-MM-DD`",
        );
    };
    let Some(c) = p.content_mut(id) else {
        return err(StatusCode::NOT_FOUND, "no_such_node", &key);
    };
    c.edit(body.title, body.text);
    let stale = p.mark_stale_below(id);
    match commit(&mind, &p) {
        Ok(v) => Json(json!({ "node": key, "stale_below": stale, "written": v })).into_response(),
        Err(r) => *r,
    }
}

/// `PUT /v1/life/:who/day/:date/consequences` — what a day must produce.
///
/// **The operator authors these; the model never writes one.** They are checked
/// against the authoring catalog here and injected into the document as
/// `<tool_call>` blocks, so the whole class of malformed, invented and drifted
/// calls cannot arise. See [`super::consequence`].
pub async fn put_consequences(
    State(s): State<Arc<Authored>>,
    UrlPath((who, date)): UrlPath<(String, String)>,
    Json(body): Json<Vec<Consequence>>,
) -> Response {
    let mind = match resolve(&s, &who).await {
        Ok(m) => m,
        Err(r) => return *r,
    };
    let mut p = match open(&mind, &who) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let Some(NodeId::Day { year, month, day }) = node_id(&date) else {
        return err(StatusCode::BAD_REQUEST, "bad_day", "a day is `YYYY-MM-DD`");
    };

    let mut problems: Vec<Value> = Vec::new();
    for c in &body {
        for b in consequence::check(c) {
            problems.push(json!({ "problem": b, "message": b.message() }));
        }
    }
    if !problems.is_empty() {
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "bad_consequences", "problems": problems })),
        )
            .into_response();
    }

    if p.day_mut(year, month, day).is_none() {
        return err(
            StatusCode::NOT_FOUND,
            "no_such_day",
            "that day is not in the plan — add it first",
        );
    }
    // Provisionally set them, then check the whole life in order: a revision
    // that precedes the relationship it revises is well-formed on its own and
    // wrong only in company, so it cannot be caught one call at a time.
    let previous = p.day_mut(year, month, day).map(|d| d.consequences.clone());
    if let Some(d) = p.day_mut(year, month, day) {
        d.consequences = body;
    }
    let ordered: Vec<&[Consequence]> = p.ordered_consequences().iter().map(|(_, c)| *c).collect();
    let bad = consequence::check_ordered(ordered, &p.seeded_entities());
    if !bad.is_empty() {
        if let (Some(d), Some(was)) = (p.day_mut(year, month, day), previous) {
            d.consequences = was;
        }
        return (
            StatusCode::BAD_REQUEST,
            Json(json!({
                "error": "bad_consequence_order",
                "problems": bad.iter().map(|b| json!({
                    "problem": b, "message": b.message(),
                })).collect::<Vec<_>>(),
            })),
        )
            .into_response();
    }

    match commit(&mind, &p) {
        Ok(v) => Json(json!({ "day": date, "written": v })).into_response(),
        Err(r) => *r,
    }
}

/// A day the operator adds by hand.
#[derive(Debug, Deserialize)]
pub struct NewDay {
    /// `YYYY-MM-DD`.
    pub date: String,
    #[serde(default)]
    pub title: String,
}

/// `POST /v1/life/:who/day` — mark a day as one worth remembering.
pub async fn post_day(
    State(s): State<Arc<Authored>>,
    UrlPath(who): UrlPath<String>,
    Json(body): Json<NewDay>,
) -> Response {
    let mind = match resolve(&s, &who).await {
        Ok(m) => m,
        Err(r) => return *r,
    };
    let mut p = match open(&mind, &who) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let Some(NodeId::Day { year, month, day }) = node_id(&body.date) else {
        return err(StatusCode::BAD_REQUEST, "bad_day", "a day is `YYYY-MM-DD`");
    };
    if plan::day_date(year, month, day).is_none() {
        return err(
            StatusCode::BAD_REQUEST,
            "no_such_day",
            "that is not a day the calendar has",
        );
    }
    let Some(d) = p.ensure_day(year, month, day) else {
        return err(
            StatusCode::BAD_REQUEST,
            "outside_the_life",
            "that month is not part of this character's life",
        );
    };
    if d.content.title.is_empty() {
        d.content.title = body.title;
    }
    match commit(&mind, &p) {
        Ok(v) => Json(json!({ "day": body.date, "written": v })).into_response(),
        Err(r) => *r,
    }
}

/// `DELETE /v1/life/:who/day/:date` — a day stops being one worth remembering.
///
/// Its document is removed with it, so the character stops remembering an
/// episode that has been written out of their history.
pub async fn delete_day(
    State(s): State<Arc<Authored>>,
    UrlPath((who, date)): UrlPath<(String, String)>,
) -> Response {
    let mind = match resolve(&s, &who).await {
        Ok(m) => m,
        Err(r) => return *r,
    };
    let mut p = match open(&mind, &who) {
        Ok(p) => p,
        Err(r) => return *r,
    };
    let Some(NodeId::Day { year, month, day }) = node_id(&date) else {
        return err(StatusCode::BAD_REQUEST, "bad_day", "a day is `YYYY-MM-DD`");
    };
    let Some(m) = p.month_mut(year, month) else {
        return err(StatusCode::NOT_FOUND, "no_such_day", &date);
    };
    let before = m.days.len();
    m.days.retain(|d| d.day != day);
    if m.days.len() == before {
        return err(StatusCode::NOT_FOUND, "no_such_day", &date);
    }
    match commit(&mind, &p) {
        Ok(v) => Json(json!({ "removed": date, "written": v })).into_response(),
        Err(r) => *r,
    }
}

/// Which rungs to run.
#[derive(Debug, Deserialize)]
pub struct GenerateRequest {
    #[serde(default)]
    pub phases: Vec<Phase>,
    /// Regenerate these nodes even though they are already written. An edited
    /// node is only ever regenerated by being named here — that is what "sticky"
    /// means in practice.
    #[serde(default)]
    pub redo: Vec<String>,
}

/// `POST /v1/life/:who/generate` — run the ladder.
pub async fn post_generate(
    State(s): State<Arc<Authored>>,
    UrlPath(who): UrlPath<String>,
    Json(body): Json<GenerateRequest>,
) -> Response {
    let mind = match resolve(&s, &who).await {
        Ok(m) => m,
        Err(r) => return *r,
    };
    let mut p = match open(&mind, &who) {
        Ok(p) => p,
        Err(r) => return *r,
    };

    // An explicit redo clears the node so the phase picks it up — including one
    // a human edited, which nothing else will touch.
    //
    // Each redo also **carries its own phase in**. A request to regenerate one
    // month is complete on its own terms; requiring the caller to also name the
    // months phase would mean a mismatch between the two silently regenerated
    // nothing, and "nothing happened" is the hardest failure to notice.
    let mut implied: Vec<Phase> = Vec::new();
    for key in &body.redo {
        let Some(id) = node_id(key) else {
            return err(StatusCode::BAD_REQUEST, "bad_node", key);
        };
        let Some(c) = p.content_mut(id) else {
            return err(StatusCode::NOT_FOUND, "no_such_node", key);
        };
        // **Marked stale, not emptied.** `wants_generation()` is
        // `!edited && (!is_generated() || stale)`, so staleness selects the node just as an
        // empty one does — and leaves the text where it is until something better replaces it.
        //
        // Clearing here meant a redo destroyed the node before the run that was meant to
        // improve it. A decode that yields no prose then leaves `apply` with nothing to
        // write, `document::sync` drops the file (the node no longer reads as generated), and
        // `plan::save` persists the emptied node — the operator's good month gone from both
        // disk and plan, with the job still reporting `done`. Cancelling a multi-node redo did
        // the same to every node it had not reached yet, which is not "a shorter run".
        //
        // `edited` is still cleared: naming a node in `redo` is the explicit override that the
        // sticky rule exists to require.
        c.edited = false;
        c.stale = true;
        if !implied.contains(&id.phase()) {
            implied.push(id.phase());
        }
    }

    let Some(rt) = s.runtime.as_ref() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_engine",
            "this daemon has no inference engine, so it cannot write a life",
        );
    };
    let minds = rt.minds.read().unwrap().clone();
    let Some(minds) = minds else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "no_engine",
            "the model is still loading",
        );
    };

    // No phases and no redo means "everything still missing", which is what the
    // console's one big button asks for. A redo on its own runs exactly the
    // rungs its nodes belong to.
    let mut phases = if body.phases.is_empty() && implied.is_empty() {
        Phase::ALL.to_vec()
    } else {
        body.phases
    };
    phases.extend(implied);
    match s
        .lifegen
        .start(minds.engine(), minds.base_config(), &mind, p, phases)
    {
        Ok(job) => Json(job.view()).into_response(),
        Err(e @ NotStarted::AlreadyRunning { .. }) => (
            StatusCode::CONFLICT,
            Json(json!({ "error": "already_running", "detail": e.message(), "refused": e })),
        )
            .into_response(),
        Err(e) => err(StatusCode::BAD_REQUEST, "nothing_to_do", &e.message()),
    }
}

/// `GET /v1/life/:who/job` — what the overlay reads.
pub async fn get_job(State(s): State<Arc<Authored>>, UrlPath(who): UrlPath<String>) -> Response {
    match s.lifegen.for_who(&who) {
        Some(j) => Json(j.view()).into_response(),
        // No run is not a failure — the console shows the editor rather than an
        // overlay.
        None => Json(Value::Null).into_response(),
    }
}

/// `POST /v1/life/:who/cancel` — stop at the next fork boundary.
pub async fn post_cancel(
    State(s): State<Arc<Authored>>,
    UrlPath(who): UrlPath<String>,
) -> Response {
    match s.lifegen.for_who(&who) {
        Some(j) => {
            j.progress.cancel();
            Json(j.view()).into_response()
        }
        None => err(
            StatusCode::NOT_FOUND,
            "no_generation",
            "nothing is generating for this character",
        ),
    }
}

/// `GET /v1/life/catalog` — the authoring tools a day may be given.
///
/// Served from [`crate::engine::authoring::CATALOG`] so the console's form is
/// built from the same table the executor reads. A second copy in JavaScript
/// would be a second place for the vocabulary to be wrong.
pub async fn get_catalog() -> Response {
    Json(json!({
        "tools": consequence::catalog(),
        "cadences": super::seed::Cadence::ALL.iter().map(|c| json!({
            "value": c,
            "label": c.label(),
            "instruction": c.instruction(),
        })).collect::<Vec<_>>(),
        "phases": Phase::ALL.iter().map(|p| json!({
            "value": p,
            "label": p.label(),
            "unit": p.unit(),
        })).collect::<Vec<_>>(),
    }))
    .into_response()
}

/// `story`, `YYYY`, `YYYY-MM` or `YYYY-MM-DD` as a node address.
///
/// Strict about the shapes, because a key that parsed loosely would address a
/// different node than the console meant and the edit would land somewhere
/// plausible.
pub(super) fn node_id(key: &str) -> Option<NodeId> {
    if key.eq_ignore_ascii_case("story") {
        return Some(NodeId::Story);
    }
    let parts: Vec<&str> = key.split('-').collect();
    let num = |i: usize, width: usize| -> Option<u32> {
        let p = parts.get(i)?;
        (p.len() == width && p.bytes().all(|b| b.is_ascii_digit())).then(|| p.parse().ok())?
    };
    let year = num(0, 4)? as i32;
    match parts.len() {
        1 => Some(NodeId::Year { year }),
        2 => Some(NodeId::Month {
            year,
            month: num(1, 2)?,
        }),
        3 => Some(NodeId::Day {
            year,
            month: num(1, 2)?,
            day: num(2, 2)?,
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_node_key_addresses_the_stratum_it_names() {
        assert_eq!(node_id("story"), Some(NodeId::Story));
        assert_eq!(node_id("STORY"), Some(NodeId::Story));
        assert_eq!(node_id("1998"), Some(NodeId::Year { year: 1998 }));
        assert_eq!(
            node_id("1998-09"),
            Some(NodeId::Month {
                year: 1998,
                month: 9
            })
        );
        assert_eq!(
            node_id("1998-09-14"),
            Some(NodeId::Day {
                year: 1998,
                month: 9,
                day: 14
            })
        );
    }

    /// **A key that parsed loosely would address a different node than the
    /// console meant**, and the edit would land somewhere plausible.
    #[test]
    fn a_malformed_node_key_addresses_nothing() {
        for key in [
            "",
            "1998-9",
            "98-09",
            "1998-09-14-1",
            "1998-",
            "-1998",
            "abcd",
            "1998-ab",
            "1998-09-1",
            "19980914",
        ] {
            assert_eq!(node_id(key), None, "{key:?} was accepted");
        }
    }

    /// **A redo carries its own phase in.** A request to regenerate one month
    /// is complete on its own terms; if the caller also had to name the months
    /// phase, a mismatch between the two would regenerate nothing — and
    /// "nothing happened" is the hardest failure for an operator to notice.
    #[test]
    fn a_node_key_implies_the_phase_that_writes_it() {
        for (key, want) in [
            ("story", Phase::Story),
            ("1998", Phase::Years),
            ("1998-09", Phase::Months),
            ("1998-09-14", Phase::Days),
        ] {
            assert_eq!(node_id(key).unwrap().phase(), want, "{key}");
        }
    }

    /// The console's form is built from the daemon's own catalog, so a tool
    /// added to the engine appears in the editor without a second edit.
    #[tokio::test]
    async fn the_catalog_route_serves_the_engines_own_tables() {
        let body = get_catalog().await.into_body();
        let bytes = axum::body::to_bytes(body, 1 << 20).await.unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            v["tools"].as_array().unwrap().len(),
            crate::engine::authoring::CATALOG.len()
        );
        assert_eq!(v["phases"].as_array().unwrap().len(), Phase::ALL.len());
        assert_eq!(v["cadences"][0]["value"], "quiet");
        assert!(!v["tools"][0]["example"].as_str().unwrap().is_empty());
    }
}
