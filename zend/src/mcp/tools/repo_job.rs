//! `repo_job_start {node?, repo, branch, program, args, timeout_s?}`: run a program
//! on an agent (`docs/claude_zend_gateway.md` §8.3) — a build, a test run —
//! against a branch as the agent fetches it. It answers once the job has
//! started; `repo_job_log` reads it. Admins only.
//!
//! The job goes over this member's own agent stream, placed as the gateway
//! places any job — sticky on the caller, spilling to an idle agent, low
//! priority last, a standby agent only when no other can — or on the agent
//! `node` names, by its node id or by its machine's name. The program is
//! checked against the agent's sandbox allow-list there, not here.

use agent::client::Agents;
use agent::wire::RunJob;
use cluster::gateway::AgentRow;
use serde_json::{json, Value};
use zend_vfs::Snapshot;

use super::{string, Answer, Ctx};
use crate::mcp::ops;

pub const NAME: &str = "repo_job_start";

/// The longest a job may run when the call names no `timeout_s`, and the most.
const DEFAULT_TIMEOUT_S: u64 = 1_800;
const MAX_TIMEOUT_S: u64 = 4 * 3_600;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Run a build or a test run on a build agent — a cluster machine that checks out a \
            repository and runs programs on it, never one serving traffic — against a branch as it stands on \
            origin. Give a program and its arguments, never a shell line; the agent runs only programs on \
            its allow-list — `cargo` — and a program no agent allows is refused as \"no ready agent holds \
            repository … with <program>\". Answers the job id and the agent it runs on; follow it with `repo_job_log`. A \
            job lives in its agent's process: one lost to a restart is reported lost, and is run again. Name \
            `node` to pin an agent; otherwise the gateway picks one. With `stage`, the file the job builds is \
            staged for `service_trial` once the job succeeds. A standby agent (AlienWorld) is given a job \
            only when no other agent can run it, or when `node` names it. Admins only.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node": {"type": "string", "description": "The agent to run on, to pin one: its 16-hex-digit node id, or its machine's name in any case (`BlackWorld`, `AlienWorld`)."},
                "repo": {"type": "string", "description": "The repository's name — `candle`, `battle-cities` — as the repo_map layer lists it."},
                "branch": {"type": "string", "description": "The branch to run on."},
                "program": {"type": "string", "description": "The program, e.g. `cargo`."},
                "args": {"type": "array", "items": {"type": "string"}, "description": "Its arguments."},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT_S, "description": "The longest it may run (1800 by default)."},
                "stage": {"type": "string", "description": "A file the job builds, relative to the checkout — `target/release/motherd.exe` — staged for `service_trial` on that machine once the job exits 0; `repo_job_log` answers where."}
            },
            "required": ["repo", "branch", "program", "args"],
            "additionalProperties": false
        }
    })
}

/// The job `args` asks for.
pub fn job_of(args: &Value) -> Result<RunJob, String> {
    let args_list = args
        .get("args")
        .and_then(Value::as_array)
        .ok_or("`args` is the program's arguments, a list of strings")?
        .iter()
        .map(|a| a.as_str().map(str::to_string))
        .collect::<Option<Vec<String>>>()
        .ok_or("each of `args` is a string")?;
    Ok(RunJob {
        repo: string(args, "repo")?.to_string(),
        branch: string(args, "branch")?.to_string(),
        overlay: Snapshot::default(),
        program: string(args, "program")?.to_string(),
        args: args_list,
        timeout_secs: args
            .get("timeout_s")
            .and_then(Value::as_u64)
            .unwrap_or(DEFAULT_TIMEOUT_S)
            .clamp(1, MAX_TIMEOUT_S),
        stage: args
            .get("stage")
            .and_then(Value::as_str)
            .map(str::to_string),
    })
}

/// An agent's node as `repo_job_log` takes it.
pub fn place(node: u64) -> String {
    format!("{node:016x}")
}

/// The node of the agent on machine `host` among `agents`: a ready one first.
pub fn on_host(agents: &[AgentRow], host: &str) -> Result<u64, String> {
    let on: Vec<&AgentRow> = agents
        .iter()
        .filter(|a| a.host.eq_ignore_ascii_case(host.trim()))
        .collect();
    let Some(agent) = on.iter().find(|a| a.state == "ready").or(on.first()) else {
        let known: Vec<&str> = agents.iter().map(|a| a.host.as_str()).collect();
        return Err(format!(
            "no agent on {host} has a socket on the gateway; agents are on: {}",
            if known.is_empty() {
                "none".to_string()
            } else {
                known.join(", ")
            }
        ));
    };
    u64::from_str_radix(&agent.node, 16).map_err(|_| format!("{:?} is not a node", agent.node))
}

/// The agent `pin` names: a node id, or the machine it runs on.
async fn pinned(agents: &Agents, pin: &str) -> Result<u64, String> {
    if let Ok(n) = ops::node(pin) {
        return Ok(u64::from_str_radix(&n, 16).expect("sixteen hex digits"));
    }
    let listed = agents
        .listing()
        .await
        .map_err(|e| format!("the agents could not be listed: {e}"))?;
    on_host(&listed, pin)
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    let job = match job_of(args) {
        Ok(j) => j,
        Err(e) => return Answer::error(e),
    };
    let Some(agents) = ctx.session.agents() else {
        return Answer::error(
            "this member reaches no agent: it has no cluster and no agent of its own",
        );
    };
    let to = match args.get("node").and_then(Value::as_str) {
        Some(pin) => match pinned(&agents, pin).await {
            Ok(node) => Some(node),
            Err(e) => return Answer::error(e),
        },
        None => None,
    };
    let command = format!("{} {}", job.program, job.args.join(" "));
    let (at, view) = match agents.submit(&ctx.caller.key(), job, to).await {
        Ok(started) => started,
        Err(e) => return Answer::error(format!("the job could not start: {e}")),
    };
    match agents.node_of(at) {
        Ok(node) => Answer::ok(format!(
            "Started `{command}` as job {} on agent {}: {:?}. Read it with `repo_job_log {{node: \
             \"{}\", job: \"{}\"}}`.",
            view.id,
            place(node),
            view.status,
            place(node),
            view.id
        )),
        Err(e) => Answer::error(format!(
            "`{command}` started as job {}, on an agent whose node could not be read: {e}",
            view.id
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_job_is_a_program_and_its_arguments_on_a_branch() {
        let job = job_of(&json!({
            "repo": "candle", "branch": "fix-x", "program": "cargo",
            "args": ["test", "-p", "zend"], "timeout_s": 99_999,
            "stage": "target/release/motherd.exe"
        }))
        .unwrap();
        assert_eq!(job.stage.as_deref(), Some("target/release/motherd.exe"));
        assert_eq!(
            (job.repo.as_str(), job.branch.as_str(), job.program.as_str()),
            ("candle", "fix-x", "cargo")
        );
        assert_eq!(job.args, ["test", "-p", "zend"]);
        assert_eq!(job.timeout_secs, MAX_TIMEOUT_S);
        assert!(job_of(&json!({"repo": "r", "branch": "b", "program": "p", "args": [1]})).is_err());
        assert!(job_of(&json!({"repo": "r", "branch": "b", "program": "p"})).is_err());
    }

    #[test]
    fn an_agent_is_named_by_its_node() {
        assert_eq!(place(0xab), "00000000000000ab");
    }

    fn row(host: &str, node: u64, state: &str) -> AgentRow {
        serde_json::from_value(json!({
            "node": place(node), "host": host,
            "build": {"count": 1, "hash": "h", "local": ""},
            "state": state, "os": "windows", "cores": 8, "load_milli": 0,
            "mem_free_mib": 0, "slots_free": 1, "jobs": 0, "toolchains": ["cargo"],
            "low_priority": false, "standby": host == "alienworld", "repos": [],
            "connected_ms": 0
        }))
        .unwrap()
    }

    /// **An agent is pinned by its machine's name**, in any case, a ready one
    /// first; a machine with no agent is refused, naming those there are.
    #[test]
    fn an_agent_is_pinned_by_its_machine() {
        let agents = [
            row("BlackWorld", 0x1, "ready"),
            row("alienworld", 0x2, "disabled"),
            row("alienworld", 0x3, "ready"),
        ];
        assert_eq!(on_host(&agents, "blackworld"), Ok(0x1));
        assert_eq!(on_host(&agents, " AlienWorld "), Ok(0x3));
        assert_eq!(on_host(&agents[..2], "alienworld"), Ok(0x2));
        assert_eq!(
            on_host(&agents, "whiteworld").unwrap_err(),
            "no agent on whiteworld has a socket on the gateway; agents are on: BlackWorld, \
             alienworld, alienworld"
        );
    }
}
