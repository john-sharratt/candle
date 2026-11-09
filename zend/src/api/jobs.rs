//! A job a conversation started, read for its block in the GUI
//! (`docs/zend_ui_redesign.md` §1.4a): where it stands, and its output a range
//! or a tail at a time, as the signed-in person reads it with the job tools.
//!
//! - `GET /v1/jobs/view?handle=…` — its kind, whether it has ended, its line,
//!   and for a task the conversation its worker runs in, which the GUI draws
//!   with the conversation routes it already has.
//! - `GET /v1/jobs/lines?handle=…&from=N` — up to 200 lines from line `from`;
//!   `&tail=N` instead answers its last `N`. The GUI follows a running job by
//!   asking from the line after the last it holds.
//!
//! The handle is a query value rather than a path segment: a task's handle
//! holds a `/`. A task is read only by the person who started it (its records
//! are under their key); any other kind is an Admin's, as `job_status` rules.

use std::sync::Arc;

use ::cluster::log_read::LogAsk;
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::mcp::caller::Caller;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::threads;
use crate::mcp::tools::job::{source, JobHandle, Standing};
use crate::session::ZendSession;

/// The most lines a range answers.
const PAGE: u32 = 200;

/// The most lines a tail answers.
const MAX_TAIL: u32 = 50;

#[derive(Deserialize)]
pub struct Ask {
    pub handle: String,
    #[serde(default)]
    pub from: Option<u64>,
    #[serde(default)]
    pub tail: Option<u32>,
}

type Refused = (StatusCode, Json<Value>);

fn refused(status: StatusCode, why: impl Into<String>) -> Refused {
    (status, Json(json!({"error": why.into()})))
}

/// The caller and the job they name, or why they may not read it.
fn reading(
    session: &Arc<ZendSession>,
    headers: &HeaderMap,
    handle: &str,
) -> Result<(OpsCall, JobHandle), Refused> {
    let caller = Caller::of(headers, session.roles())
        .ok_or_else(|| refused(StatusCode::UNAUTHORIZED, "sign in to read a job"))?;
    let job = JobHandle::parse(handle).map_err(|e| refused(StatusCode::BAD_REQUEST, e))?;
    if !matches!(job, JobHandle::Task { .. }) && !caller.is_admin() {
        return Err(refused(
            StatusCode::FORBIDDEN,
            format!("{job} is a {}: reading one is an Admin's", job.kind()),
        ));
    }
    let call = OpsCall {
        session: Arc::clone(session),
        lineage: caller.key(),
        caller: Some(caller),
        progress: None,
        ceiling: 30,
        worker: false,
    };
    Ok((call, job))
}

/// What to read of a job's output for `ask`.
pub fn log_ask(ask: &Ask) -> LogAsk {
    match ask.tail {
        Some(n) => LogAsk::Tail {
            lines: n.clamp(1, MAX_TAIL),
        },
        None => LogAsk::Lines {
            from: ask.from.unwrap_or(1).max(1),
            count: PAGE,
        },
    }
}

pub async fn view(
    State(session): State<Arc<ZendSession>>,
    headers: HeaderMap,
    Query(ask): Query<Ask>,
) -> Result<Json<Value>, Refused> {
    let (call, job) = reading(&session, &headers, &ask.handle)?;
    let standing = source::standing(&call, &job)
        .await
        .map_err(|e| refused(StatusCode::NOT_FOUND, e))?;
    let conversation = match &job {
        JobHandle::Task { thread, name } => call
            .caller
            .as_ref()
            .map(|c| threads::conv_id(&c.key(), thread, name)),
        _ => None,
    };
    let turns = conversation
        .as_deref()
        .map(|id| session.conversation_turn_count(id));
    Ok(Json(view_body(&job, &standing, conversation, turns)))
}

/// A job's view: its kind, whether it ended, its line, how long its output
/// is in lines and bytes, and for a task the conversation its worker runs in
/// with how many turns it holds — what the page compares to decide what to
/// read again.
pub fn view_body(
    job: &JobHandle,
    standing: &Standing,
    conversation: Option<String>,
    turns: Option<u32>,
) -> Value {
    let mut body = json!({
        "handle": job.to_string(),
        "kind": job.kind(),
        "ended": standing.ended,
        "line": standing.line,
        "total": standing.total,
        "size": standing.size,
        "conversation": conversation,
    });
    if let Some(turns) = turns {
        body["turns"] = json!(turns);
    }
    body
}

pub async fn lines(
    State(session): State<Arc<ZendSession>>,
    headers: HeaderMap,
    Query(ask): Query<Ask>,
) -> Result<Json<Value>, Refused> {
    let (call, job) = reading(&session, &headers, &ask.handle)?;
    let read = source::read(&call, &job, &log_ask(&ask))
        .await
        .map_err(|e| refused(StatusCode::NOT_FOUND, e))?;
    Ok(Json(json!({
        "lines": read.lines,
        "total": read.total,
        "ended": read.ended,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standing(ended: bool, total: u64, size: u64) -> Standing {
        Standing {
            ended,
            line: "it stands".into(),
            detail: String::new(),
            total,
            size,
            marks: Vec::new(),
        }
    }

    /// **A view answers its output's lines and bytes, and a task's turns**,
    /// each exactly.
    #[test]
    fn a_view_answers_what_the_page_compares() {
        let task = JobHandle::parse("task:review/verify").unwrap();
        assert_eq!(
            view_body(
                &task,
                &standing(false, 12, 480),
                Some("mcp:k:review/verify".into()),
                Some(4)
            ),
            json!({
                "handle": "task:review/verify",
                "kind": "task",
                "ended": false,
                "line": "it stands",
                "total": 12,
                "size": 480,
                "conversation": "mcp:k:review/verify",
                "turns": 4,
            })
        );
        let build = JobHandle::parse("build:76c3a0f4d39bd9db:XeV_I_QPzpg").unwrap();
        assert_eq!(
            view_body(&build, &standing(true, 900, 51_200), None, None),
            json!({
                "handle": "build:76c3a0f4d39bd9db:XeV_I_QPzpg",
                "kind": "build",
                "ended": true,
                "line": "it stands",
                "total": 900,
                "size": 51_200,
                "conversation": null,
            })
        );
    }

    /// **A range reads from where the GUI is, a page at a time; a tail is
    /// bounded.**
    #[test]
    fn a_range_reads_on_and_a_tail_is_bounded() {
        let ask = |from, tail| Ask {
            handle: "build:76c3a0f4d39bd9db:Xe".into(),
            from,
            tail,
        };
        assert_eq!(
            log_ask(&ask(None, None)),
            LogAsk::Lines {
                from: 1,
                count: 200
            }
        );
        assert_eq!(
            log_ask(&ask(Some(401), None)),
            LogAsk::Lines {
                from: 401,
                count: 200
            }
        );
        assert_eq!(log_ask(&ask(None, Some(8))), LogAsk::Tail { lines: 8 });
        assert_eq!(log_ask(&ask(None, Some(900))), LogAsk::Tail { lines: 50 });
    }
}
