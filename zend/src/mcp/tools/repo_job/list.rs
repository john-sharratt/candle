//! `repo_job_list {node?, running?, limit?}`: the jobs each agent is running
//! and keeps, newest first, each by its handle — what a caller reads when no
//! one holds a job's handle, a worker's `run_command` and a job started before
//! a compaction among them.

use agent::wire::{AgentJobs, JobLine};
use serde_json::{json, Value};

use super::{command_line, host_in, phase, pinned, place, span, state, Handle, NO_AGENTS};
use crate::mcp::ops;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::{Answer, Ctx};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "repo_job_list";

/// The jobs asked of each agent when the call names no `limit`, and the most.
const DEFAULT_LIMIT: u64 = 20;
const MAX_LIMIT: u64 = 100;

/// The longest command line shown: an arguments list can be a page.
const MAX_COMMAND: usize = 120;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "List the jobs on the build agents — every machine's, or one's — newest first: each \
            job's handle (`<node>:<id>`, what `repo_job_status`, `repo_job_log` and `repo_job_cancel` take), \
            where it is (fetching, waiting for the checkout, running, or how it ended), its repository and \
            branch, its command, how long it has run and how many lines it has printed; and per machine, \
            how many of its job slots are busy. Use it to see what is building where before starting \
            another, and to find a job nobody holds the handle of — a worker's `run_command`, a deploy's \
            build, a job started before a compaction. Jobs live in their agent's process, so an agent's \
            restart forgets them. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node": {"type": "string", "minLength": 1, "maxLength": 32, "description": "One agent only: its machine (`BlackWorld`) or its 16-hex-digit node. Every agent when left out."},
                "running": {"type": "boolean", "description": "Only jobs not yet ended."},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "description": "The most jobs listed per agent; 20 by default."}
            },
            "additionalProperties": false
        }
    })
}

/// One job's line.
fn job_line(node: u64, j: &JobLine, now: u64) -> String {
    let ended = j.status.is_finished();
    let ran = j.ended_ms.unwrap_or(now).saturating_sub(j.started_ms);
    format!(
        "{} {} {}{} `{}`, {} {}, {} lines",
        Handle::new(node, &j.id),
        if ended {
            state(&j.status)
        } else {
            phase(&j.phase, node)
        },
        j.repo,
        j.branch
            .as_deref()
            .map_or(String::new(), |b| format!("@{b}")),
        command_line(&j.command, MAX_COMMAND),
        if ended { "took" } else { "for" },
        span(ran),
        j.lines
    )
}

/// One agent's section: its heading, then its jobs.
pub fn section(host: &str, node: u64, jobs: &AgentJobs, running_only: bool, now: u64) -> String {
    let mut out = format!(
        "{host} ({}): {} of {} slots busy{}",
        place(node),
        jobs.running,
        jobs.slots,
        if jobs.enabled { "" } else { ", switched off" }
    );
    let shown: Vec<&JobLine> = jobs
        .jobs
        .iter()
        .filter(|j| !running_only || !j.status.is_finished())
        .collect();
    if shown.is_empty() {
        out.push_str(&format!(
            "\n  no {}jobs",
            if running_only { "running " } else { "" }
        ));
    }
    for j in shown {
        out.push_str("\n  ");
        out.push_str(&job_line(node, j, now));
    }
    out
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let Some(agents) = call.session.agents() else {
        return Answer::error(NO_AGENTS);
    };
    let listed = match agents.listing().await {
        Ok(l) => l,
        Err(e) => return Answer::error(format!("the agents could not be listed: {e}")),
    };
    let nodes: Vec<(u64, String)> = match args.get("node").and_then(Value::as_str) {
        Some(pin) => match pinned(&listed, pin) {
            Ok(node) => vec![(node, host_in(&listed, node))],
            Err(e) => return Answer::error(e),
        },
        None => listed
            .iter()
            .filter_map(|a| {
                u64::from_str_radix(&a.node, 16)
                    .ok()
                    .map(|n| (n, a.host.clone()))
            })
            .collect(),
    };
    if nodes.is_empty() {
        return Answer::error("no agent is connected");
    }
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_LIMIT)
        .clamp(1, MAX_LIMIT) as usize;
    let running_only = args
        .get("running")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let now = now_ms();
    let mut sections = Vec::with_capacity(nodes.len());
    for (node, host) in nodes {
        sections.push(match agents.jobs_of(agents.at(node), limit).await {
            Ok(jobs) => section(&host, node, &jobs, running_only, now),
            Err(e) => format!("{host} ({}): its jobs could not be read: {e}", place(node)),
        });
    }
    Answer::ok(sections.join("\n"))
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    run(&OpsCall::of(ctx), args).await
}

#[cfg(test)]
mod tests {
    use agent::wire::{JobKind, Phase};
    use zend_vfs::JobStatus;

    use super::*;

    fn line(id: &str, status: JobStatus, phase: Phase, ended_ms: Option<u64>) -> JobLine {
        JobLine {
            id: id.into(),
            repo: "candle".into(),
            kind: JobKind::Run,
            branch: Some("main".into()),
            command: vec!["cargo".into(), "test".into(), "-p".into(), "web".into()],
            status,
            lines: 12,
            bytes: 400,
            started_ms: 1_000,
            ended_ms,
            changed: 0,
            phase,
        }
    }

    fn jobs(lines: Vec<JobLine>) -> AgentJobs {
        AgentJobs {
            enabled: true,
            slots: 2,
            running: 1,
            jobs: lines,
        }
    }

    /// **A job reads as its handle, where it is and what it runs.**
    #[test]
    fn a_section_names_each_job_by_its_handle() {
        let text = section(
            "BlackWorld",
            0x76c3a0f4d39bd9db,
            &jobs(vec![line("abc", JobStatus::Running, Phase::Running, None)]),
            false,
            31_000,
        );
        assert_eq!(
            text,
            "BlackWorld (76c3a0f4d39bd9db): 1 of 2 slots busy\n  76c3a0f4d39bd9db:abc running \
             candle@main `cargo test -p web`, for 30 s, 12 lines"
        );
    }

    /// `running` leaves the ended jobs out, and says so when nothing is left.
    #[test]
    fn running_only_leaves_out_what_ended() {
        let ended = jobs(vec![line(
            "old",
            JobStatus::Cancelled,
            Phase::Ended,
            Some(9_000),
        )]);
        assert!(section("w", 1, &ended, false, 0)
            .contains("0000000000000001:old cancelled candle@main `cargo test -p web`, took 8 s"));
        assert_eq!(
            section("w", 1, &ended, true, 0),
            "w (0000000000000001): 1 of 2 slots busy\n  no running jobs"
        );
    }
}
