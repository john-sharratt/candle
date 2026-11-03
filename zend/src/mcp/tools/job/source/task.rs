//! A task — a worker of Zen's — as a job (`docs/jobs.md` §3): its output is
//! its prompt, each journal entry as it is drafted, its question or its answer
//! once it settles, and its turn's report, built from its replicated record so
//! any member answers for it.

use cluster::log_read::{self, LogAsk, LogRead};

use super::super::{pages, JobHandle, Standing};
use super::Row;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::reported::{self, Mark};
use crate::mcp::threads::{self, State, Worker};
use crate::mcp::tools::turn_report;
use crate::mcp::tools::wait::{handle, result_of};
use crate::mcp::tools::{Answer, LOADING};
use crate::mcp::workers::{self, now_ms};

/// Whose tasks a call reads: the person it acts as.
fn owner(call: &OpsCall) -> Result<String, String> {
    call.caller.as_ref().map(|c| c.key()).ok_or_else(|| {
        "a task is read as the person it works for, and this call names none".to_string()
    })
}

/// Task `thread/name`'s record as it stands now.
fn worker(call: &OpsCall, thread: &str, name: &str) -> Result<(String, Worker), String> {
    let owner = owner(call)?;
    let records = call.session.shared_substrate().ok_or(LOADING)?;
    let w = threads::worker(&records, &owner, thread, name)
        .ok_or_else(|| format!("there is no task `task:{thread}/{name}`"))?;
    Ok((owner, w.observed(now_ms())))
}

/// A task's output, as text: what `job_read` pages.
pub fn output(w: &Worker, now: u64) -> String {
    let mut out = format!("{}: {}\n", handle(w), w.prompt.trim());
    for e in &w.entries {
        out.push_str(&format!(
            "── journal entry {}#{} ──\n{}\n",
            w.path(),
            e.id,
            e.text.trim()
        ));
    }
    match w.state {
        State::Question => out.push_str(&format!(
            "── it asks ──\n{}\n",
            w.question.as_deref().unwrap_or("").trim()
        )),
        State::Done | State::Stuck => {
            out.push_str(&format!("── its answer ──\n{}\n", w.final_text.trim()))
        }
        State::Error => out.push_str(&format!(
            "── it failed ──\n{}\n",
            w.error.as_deref().unwrap_or("the task failed")
        )),
        State::Running => {}
    }
    out.push_str(&format!(
        "── its turn ──\n{}\n",
        turn_report::render(w, now)
    ));
    out
}

pub fn standing(
    call: &OpsCall,
    job: &JobHandle,
    thread: &str,
    name: &str,
) -> Result<Standing, String> {
    let (owner, w) = worker(call, thread, name)?;
    let records = call.session.shared_substrate().ok_or(LOADING)?;
    let seen = reported::last(&records, &owner, thread, name);
    let now = now_ms();
    let text = output(&w, now);
    let total = text.lines().count() as u64;
    let marks = w
        .entries
        .iter()
        .map(|e| e.id)
        .max()
        .filter(|newest| seen.is_none_or(|s| *newest > s))
        .map(|entry| Mark {
            thread: w.thread.clone(),
            name: w.name.clone(),
            entry,
        })
        .into_iter()
        .collect();
    let state = match w.state {
        State::Running => format!("running — {}", w.doing(now)),
        other => format!("{other:?}").to_lowercase(),
    };
    Ok(Standing {
        ended: !w.state.working(),
        line: format!("{job}: {state}"),
        detail: format!("\n{}", result_of(&w, seen)),
        total,
        marks,
    })
}

pub fn read(call: &OpsCall, thread: &str, name: &str, ask: &LogAsk) -> Result<LogRead, String> {
    let (_, w) = worker(call, thread, name)?;
    let text = output(&w, now_ms());
    let mut read = log_read::read(text.as_bytes(), ask).map_err(|e| e.to_string())?;
    read.ended = !w.state.working();
    read.size = text.len() as u64;
    Ok(read)
}

pub fn cancel(call: &OpsCall, job: &JobHandle, thread: &str, name: &str) -> Answer {
    let Some(caller) = &call.caller else {
        return Answer::error(
            "a task is stopped as the person it works for, and this call names none",
        );
    };
    match workers::stop(
        &call.session,
        caller,
        thread,
        name,
        "stopped with job_cancel",
    ) {
        Ok(_) => Answer::ok(format!(
            "Stopping {job}: it stops at its next tool round and answers with what it has; it is \
             then stopped but live, and `repo_task_reply` redirects it. `job_status {{jobs: \
             [\"{job}\"], wait_s: 120}}` says when it has stopped."
        )),
        Err(e) => Answer::error(e),
    }
}

/// Every task of the caller's, a line each.
pub fn list(call: &OpsCall, running: bool) -> Result<Vec<Row>, String> {
    let owner = owner(call)?;
    let records = call.session.shared_substrate().ok_or(LOADING)?;
    let now = now_ms();
    let mut rows = Vec::new();
    for t in threads::threads(&records, &owner) {
        for w in threads::workers(&records, &owner, &t.name) {
            let w = w.observed(now);
            if running && !w.state.live() {
                continue;
            }
            let job = JobHandle::Task {
                thread: w.thread.clone(),
                name: w.name.clone(),
            };
            let state = match w.state {
                State::Running => format!("running — {}", w.doing(now)),
                other => format!("{other:?}").to_lowercase(),
            };
            let lines = output(&w, now).lines().count() as u64;
            rows.push(Row {
                line: format!(
                    "{job}: {state}; `{}`, {} pages",
                    w.prompt
                        .lines()
                        .next()
                        .unwrap_or("")
                        .chars()
                        .take(100)
                        .collect::<String>(),
                    pages(lines)
                ),
                job,
                started_ms: w.started_ms,
            });
        }
    }
    Ok(rows)
}
