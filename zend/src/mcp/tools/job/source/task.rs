//! A task — a worker of Zen's — as a job (`docs/jobs.md` §3): its output is
//! its prompt, each journal entry as it is drafted, its question or its answer
//! once it settles, and its turn's report, built from its replicated record so
//! any member answers for it.

use std::fmt::{self, Write};

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
    let mut out = String::new();
    // Writing into a `String` cannot fail.
    let _ = write_output(w, now, &mut out);
    out
}

/// How long a task's output is, counted as it is written rather than built:
/// what a status and a listing need of it.
pub fn measure(w: &Worker, now: u64) -> Count {
    let mut count = Count::default();
    let _ = write_output(w, now, &mut count);
    count
}

/// A task's output, written to `out`. Every segment ends in a newline, so
/// counting newlines counts its lines.
pub fn write_output(w: &Worker, now: u64, out: &mut impl Write) -> fmt::Result {
    writeln!(out, "{}: {}", handle(w), w.prompt.trim())?;
    for e in &w.entries {
        writeln!(
            out,
            "── journal entry {}#{} ──\n{}",
            w.path(),
            e.id,
            e.text.trim()
        )?;
    }
    match w.state {
        State::Question => writeln!(
            out,
            "── it asks ──\n{}",
            w.question.as_deref().unwrap_or("").trim()
        )?,
        State::Done | State::Stuck => writeln!(out, "── its answer ──\n{}", w.final_text.trim())?,
        State::Error => writeln!(
            out,
            "── it failed ──\n{}",
            w.error.as_deref().unwrap_or("the task failed")
        )?,
        State::Running => {}
    }
    writeln!(out, "── its turn ──\n{}", turn_report::render(w, now))
}

/// The lines and bytes of what was written to it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Count {
    pub lines: u64,
    pub bytes: u64,
}

impl Write for Count {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.lines += s.bytes().filter(|b| *b == b'\n').count() as u64;
        self.bytes += s.len() as u64;
        Ok(())
    }
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
    let count = measure(&w, now);
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
        total: count.lines,
        size: count.bytes,
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
            let lines = measure(&w, now).lines;
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

#[cfg(test)]
mod tests {
    use web::auth::Role;

    use super::*;
    use crate::mcp::threads::Entry;

    const NOW: u64 = 1_800_000;

    fn worker() -> Worker {
        Worker {
            thread: "review".into(),
            name: "v".into(),
            conv_id: "c".into(),
            prompt: "p".into(),
            state: State::Done,
            role: Role::User,
            person: None,
            priority: Default::default(),
            effort: Default::default(),
            kind: Default::default(),
            driven: Default::default(),
            started_ms: 1_000_000,
            turn_ms: 1_000_000,
            beat_ms: 0,
            ended_ms: Some(1_500_000),
            entries_after: None,
            entries: vec![
                Entry {
                    id: 1,
                    text: "first\nline".into(),
                },
                Entry {
                    id: 2,
                    text: "second".into(),
                },
            ],
            final_text: "Verdict: ok".into(),
            error: None,
            question: None,
            activity: None,
            no_entry: None,
            stopped: false,
            resumable: false,
            resumes: 0,
            by_task: false,
            audit: None,
            checked: false,
            runner: None,
        }
    }

    /// **The output is counted as it is written**: the prompt's line, each
    /// entry's header and lines (3 and 2), the answer's header and line, and
    /// the turn's header and report — the same lines and bytes the text has.
    #[test]
    fn a_tasks_output_is_counted_without_building_it() {
        let w = worker();
        let text = output(&w, NOW);
        let report = turn_report::render(&w, NOW).lines().count() as u64;
        let count = measure(&w, NOW);
        assert_eq!(count.lines, 1 + 3 + 2 + 2 + 1 + report);
        assert_eq!(count.lines, text.lines().count() as u64);
        assert_eq!(count.bytes, text.len() as u64);
        assert!(text.starts_with(&format!("{}: p\n── journal entry ", handle(&w))));
    }
}
