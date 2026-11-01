//! `repo_job_list {node?, running?, limit?}`: the jobs each agent is running and
//! keeps, newest first (`docs/claude_zend_gateway.md` §8.3) — what `repo_job_log`
//! needs when no one holds the job's id, a worker's `run_command` among them.
//! Admins only.

use agent::client::Agents;
use agent::wire::{AgentJobs, JobLine};
use serde_json::{json, Value};
use zend_vfs::JobStatus;

use super::repo_job::{on_host, place};
use super::{Answer, Ctx};
use crate::mcp::ops;

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
        "description": "List the jobs on the build agents, newest first: each job's id, repository, branch, \
            command, whether it is running or how it ended, and how long it took. Use it to find a run nobody \
            holds the id of — a worker's `run_command`, or a job started before a compaction — then read it \
            with `repo_job_log`, which takes the `node` and `job` this answers. Jobs live in their agent's \
            process, so a restart forgets them. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node": {"type": "string", "description": "One agent only: its 16-hex-digit node id, or its machine's name in any case (`BlackWorld`). Every agent when left out."},
                "running": {"type": "boolean", "description": "Only jobs still running."},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "description": "The most jobs asked of each agent (20 by default)."}
            },
            "additionalProperties": false
        }
    })
}

/// `words` as one line, cut at [`MAX_COMMAND`] characters.
fn command_line(words: &[String]) -> String {
    let line = words.join(" ");
    match line.char_indices().nth(MAX_COMMAND) {
        Some((at, _)) => format!("{}…", &line[..at]),
        None => line,
    }
}

/// Whether `status` is a job still going.
fn is_running(status: &JobStatus) -> bool {
    matches!(status, JobStatus::Running | JobStatus::Queued)
}

/// How `status` reads on a job's line.
fn state(status: &JobStatus) -> String {
    match status {
        JobStatus::Queued => "queued".into(),
        JobStatus::Running => "running".into(),
        JobStatus::Exited { code: Some(code) } => format!("exited {code}"),
        JobStatus::Exited { code: None } => "killed by a signal".into(),
        JobStatus::TimedOut => "timed out".into(),
        JobStatus::Cancelled => "cancelled".into(),
        JobStatus::Refused { why } => format!("refused ({why})"),
        JobStatus::Failed { why } => format!("failed ({why})"),
    }
}

/// One job's line.
fn job_line(j: &JobLine) -> String {
    let took = j.ended_ms.map_or(String::new(), |end| {
        format!(", {} s", end.saturating_sub(j.started_ms) / 1000)
    });
    format!(
        "{} {} {}{}{} `{}`, {} lines",
        j.id,
        state(&j.status),
        j.repo,
        j.branch
            .as_deref()
            .map_or(String::new(), |b| format!("@{b}")),
        took,
        command_line(&j.command),
        j.lines
    )
}

/// One agent's section: its heading, then its jobs.
pub fn section(host: &str, node: &str, jobs: &AgentJobs, running_only: bool) -> String {
    let mut out = format!(
        "{host} ({node}): {} of {} slots running{}",
        jobs.running,
        jobs.slots,
        if jobs.enabled { "" } else { ", switched off" }
    );
    let shown: Vec<&JobLine> = jobs
        .jobs
        .iter()
        .filter(|j| !running_only || is_running(&j.status))
        .collect();
    if shown.is_empty() {
        out.push_str(&format!(
            "\n  no {}jobs",
            if running_only { "running " } else { "" }
        ));
    }
    for j in shown {
        out.push_str("\n  ");
        out.push_str(&job_line(j));
    }
    out
}

/// Every agent `pin` selects, as `(node, host)`.
async fn selected(agents: &Agents, pin: Option<&str>) -> Result<Vec<(u64, String)>, String> {
    let listed = agents
        .listing()
        .await
        .map_err(|e| format!("the agents could not be listed: {e}"))?;
    if let Some(pin) = pin {
        let node = match ops::node(pin) {
            Ok(n) => u64::from_str_radix(&n, 16).expect("sixteen hex digits"),
            Err(_) => on_host(&listed, pin)?,
        };
        let host = listed
            .iter()
            .find(|a| a.node == place(node))
            .map_or_else(|| "an agent".to_string(), |a| a.host.clone());
        return Ok(vec![(node, host)]);
    }
    Ok(listed
        .iter()
        .filter_map(|a| {
            u64::from_str_radix(&a.node, 16)
                .ok()
                .map(|n| (n, a.host.clone()))
        })
        .collect())
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    let Some(agents) = ctx.session.agents() else {
        return Answer::error(
            "this member reaches no agent: it has no cluster and no agent of its own",
        );
    };
    let pin = args.get("node").and_then(Value::as_str);
    let nodes = match selected(&agents, pin).await {
        Ok(n) => n,
        Err(e) => return Answer::error(e),
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
    let mut sections = Vec::with_capacity(nodes.len());
    for (node, host) in nodes {
        sections.push(match agents.jobs_of(agents.at(node), limit).await {
            Ok(jobs) => section(&host, &place(node), &jobs, running_only),
            Err(e) => format!("{host} ({}): its jobs could not be read: {e}", place(node)),
        });
    }
    Answer::ok(sections.join("\n"))
}

#[cfg(test)]
mod tests {
    use agent::wire::JobKind;

    use super::*;

    fn line(id: &str, status: JobStatus, ended_ms: Option<u64>) -> JobLine {
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

    /// A job reads as what `repo_job_log` needs and what it ran.
    #[test]
    fn a_section_names_the_node_and_each_jobs_id_and_command() {
        let text = section(
            "BlackWorld",
            "76c3a0f4d39bd9db",
            &jobs(vec![line("abc", JobStatus::Running, None)]),
            false,
        );
        assert_eq!(
            text,
            "BlackWorld (76c3a0f4d39bd9db): 1 of 2 slots running\n  \
             abc running candle@main `cargo test -p web`, 12 lines"
        );
    }

    /// `running` leaves the ended jobs out, and says so when nothing is left.
    #[test]
    fn running_only_leaves_out_what_ended() {
        let ended = jobs(vec![line("old", JobStatus::Cancelled, Some(9_000))]);
        assert!(section("w", "n", &ended, false).contains("old cancelled candle@main, 8 s"));
        assert_eq!(
            section("w", "n", &ended, true),
            "w (n): 1 of 2 slots running\n  no running jobs"
        );
    }

    /// A command line of a page is cut, on a character.
    #[test]
    fn a_long_command_is_cut() {
        let long = vec!["é".repeat(MAX_COMMAND + 5)];
        let cut = command_line(&long);
        assert_eq!(cut.chars().count(), MAX_COMMAND + 1);
        assert!(cut.ends_with('…'));
    }
}
