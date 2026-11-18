//! A build or test run on an agent, as a job: its output is its log, read on
//! the agent.

use agent::client::{AgentError, Agents};
use agent::wire::JobView;
use cluster::log_read::{Line, LogAsk, LogRead, Summary};

use super::super::{pages, JobHandle, Standing, PAGE_LINES};
use super::agent_miss::{miss, restarting, Miss};
use super::Row;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::repo_job::{describe, host_in, host_of, lost, Handle, NO_AGENTS};
use crate::mcp::tools::service::built_service;
use crate::mcp::tools::Answer;
use crate::mcp::workers::now_ms;

/// The most of each kind of line a summary quotes.
const QUOTED: usize = 6;

pub fn agents(call: &OpsCall) -> Result<Agents, String> {
    call.session.agents().ok_or_else(|| NO_AGENTS.to_string())
}

/// The page line `n` falls on.
pub fn page_of(n: u64) -> u64 {
    (n.max(1) - 1) / PAGE_LINES + 1
}

/// What a summary says beneath a job's line: its results, its failures and
/// errors by line, and the page that reads around the first of them.
pub fn summary_text(job: &JobHandle, s: &Summary) -> String {
    let mut out = String::new();
    let quote = |out: &mut String, head: &str, lines: &[Line], count: u64| {
        if lines.is_empty() {
            return;
        }
        out.push_str(&format!("\n{head}:"));
        for l in lines.iter().take(QUOTED) {
            out.push_str(&format!("\n  {}: {}", l.n, l.text));
        }
        let more = count.saturating_sub(lines.len().min(QUOTED) as u64);
        if more > 0 {
            out.push_str(&format!("\n  …and {more} more"));
        }
    };
    quote(&mut out, "test results", &s.results, s.results.len() as u64);
    quote(&mut out, "failures", &s.failures, s.failure_count);
    quote(&mut out, "errors", &s.errors, s.error_count);
    if s.warning_count > 0 {
        out.push_str(&format!("\n{} warnings", s.warning_count));
    }
    let first = s
        .errors
        .first()
        .into_iter()
        .chain(s.failures.first())
        .map(|l| l.n)
        .min();
    match first {
        Some(n) => out.push_str(&format!(
            "\nThe first is on page {}: `job_read {{job: \"{job}\", page: {}}}`.",
            page_of(n),
            page_of(n)
        )),
        None if s.results.is_empty() => {
            if let Some(l) = s.last.last() {
                out.push_str(&format!("\nlast line {}: {}", l.n, l.text));
            }
        }
        None => {}
    }
    out
}

/// What a failed call to `h`'s agent answers: a job whose agent is
/// restarting stands, not ended, and a status waits on; one an agent that
/// serves does not hold is lost (`agent_miss`).
async fn missed(
    agents: &Agents,
    job: &JobHandle,
    h: &Handle,
    e: &AgentError,
) -> Result<Standing, String> {
    match miss(agents, h.node, e).await {
        Miss::Restarting => Ok(Standing {
            ended: false,
            line: restarting(&job.to_string(), &host_of(agents, h.node).await),
            detail: String::new(),
            total: 0,
            size: 0,
            marks: Vec::new(),
        }),
        Miss::NotHeld | Miss::Failed => Err(lost(h, e)),
    }
}

/// What a failed read of `h`'s log answers: the agent is restarting and the
/// read is asked again, or the job is lost.
pub(super) async fn read_missed(agents: &Agents, h: &Handle, e: &AgentError) -> String {
    match miss(agents, h.node, e).await {
        Miss::Restarting => format!(
            "{}: ask again in a moment — the same call",
            restarting(&format!("build:{h}"), &host_of(agents, h.node).await)
        ),
        Miss::NotHeld | Miss::Failed => lost(h, e),
    }
}

pub async fn standing(call: &OpsCall, job: &JobHandle, h: &Handle) -> Result<Standing, String> {
    let agents = agents(call)?;
    let at = h.at(&agents);
    let view = match agents.view(at, &h.id).await {
        Ok(v) => v,
        Err(e) => return missed(&agents, job, h, &e).await,
    };
    let host = host_of(&agents, h.node).await;
    let ended = view.status.is_finished();
    let read = match agents.read_log(at, &h.id, &LogAsk::Summary).await {
        Ok(r) => r,
        Err(e) => return missed(&agents, job, h, &e).await,
    };
    let detail = match (&read.summary, ended) {
        (Some(s), true) => summary_text(job, s),
        (Some(s), false) => s.last.last().map_or(String::new(), |l| {
            format!("\nlast line {}: {}", l.n, l.text)
        }),
        (None, _) => String::new(),
    };
    Ok(Standing {
        ended,
        line: describe(&job.to_string(), &host, &view, now_ms()),
        detail,
        total: read.total,
        size: read.size,
        marks: Vec::new(),
    })
}

pub async fn read(call: &OpsCall, h: &Handle, ask: &LogAsk) -> Result<LogRead, String> {
    let agents = agents(call)?;
    match agents.read_log(h.at(&agents), &h.id, ask).await {
        Ok(read) => Ok(read),
        Err(e) => Err(read_missed(&agents, h, &e).await),
    }
}

pub async fn cancel(call: &OpsCall, job: &JobHandle, h: &Handle) -> Answer {
    let agents = match agents(call) {
        Ok(a) => a,
        Err(e) => return Answer::error(e),
    };
    let at = h.at(&agents);
    let before = match agents.view(at, &h.id).await {
        Ok(v) => v,
        Err(e) => return Answer::error(read_missed(&agents, h, &e).await),
    };
    let host = host_of(&agents, h.node).await;
    if before.status.is_finished() {
        return Answer::ok(format!(
            "Nothing to cancel: it had already ended.\n{}",
            describe(&job.to_string(), &host, &before, now_ms())
        ));
    }
    match agents.cancel(at, &h.id).await {
        Ok(view) => Answer::ok(format!(
            "Cancelled.\n{}",
            describe(&job.to_string(), &host, &view, now_ms())
        )),
        Err(e) => Answer::error(read_missed(&agents, h, &e).await),
    }
}

/// Every agent's jobs, a build or a deploy by what it runs.
pub async fn list(call: &OpsCall, running: bool, limit: usize) -> Result<Vec<Row>, String> {
    let agents = agents(call)?;
    let listed = agents
        .listing()
        .await
        .map_err(|e| format!("the agents could not be listed: {e}"))?;
    let now = now_ms();
    let mut rows = Vec::new();
    for a in &listed {
        let Ok(node) = u64::from_str_radix(&a.node, 16) else {
            continue;
        };
        let Ok(jobs) = agents.jobs_of(agents.at(node), limit).await else {
            continue;
        };
        for j in jobs.jobs {
            let ended = j.status.is_finished();
            if running && ended {
                continue;
            }
            let h = Handle::new(node, &j.id);
            let job = match built_service(&j.command) {
                Some(_) => JobHandle::Deploy(h),
                None => JobHandle::Build(h),
            };
            let view = JobView {
                id: j.id.clone(),
                repo: j.repo.clone(),
                status: j.status.clone(),
                lines: j.lines,
                bytes: j.bytes,
                outcome: None,
                pushed: Vec::new(),
                fetched: None,
                listed: None,
                staged: None,
                tried: None,
                phase: j.phase.clone(),
                command: j.command.clone(),
                started_ms: j.started_ms,
                ended_ms: j.ended_ms,
            };
            rows.push(Row {
                line: format!(
                    "{}, {} pages",
                    describe(&job.to_string(), &host_in(&listed, node), &view, now),
                    pages(j.lines)
                ),
                job,
                started_ms: j.started_ms,
            });
        }
    }
    Ok(rows)
}
