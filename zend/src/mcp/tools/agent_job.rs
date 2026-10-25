//! `agent_job {node?, repo, branch, program, args, timeout_s?}`: run a program
//! on an agent (`docs/claude_zend_gateway.md` §8.3) — a build, a test run —
//! against a branch as the agent fetches it. It answers once the job has
//! started; `agent_log` reads it. An Admin's.
//!
//! The job goes over this member's own agent stream, placed as the gateway
//! places any job — sticky on the caller, spilling to an idle agent, low
//! priority last — or on the agent `node` names. The program is checked against
//! the agent's sandbox allow-list there, not here.

use agent::wire::RunJob;
use serde_json::{json, Value};
use zend_vfs::Snapshot;

use super::{string, Answer, Ctx};
use crate::mcp::ops;

pub const NAME: &str = "agent_job";

/// The longest a job may run when the call names no `timeout_s`, and the most.
const DEFAULT_TIMEOUT_S: u64 = 1_800;
const MAX_TIMEOUT_S: u64 = 4 * 3_600;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Run a program — a build, a test run — on an agent, against a repository's branch as \
            it stands on origin: a program and its arguments, never a shell line, within the agent's \
            allow-list. Answers the job and the agent it runs on; read it with `agent_log`. A job lives in \
            its agent's process: one lost to a restart is reported lost, and is run again. Name `node` to \
            pin an agent; otherwise the gateway places it. An Admin's.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node": {"type": "string", "description": "The agent's node, to pin one."},
                "repo": {"type": "string", "description": "The repository, as the workspace names it."},
                "branch": {"type": "string", "description": "The branch to run on."},
                "program": {"type": "string", "description": "The program, e.g. `cargo`."},
                "args": {"type": "array", "items": {"type": "string"}, "description": "Its arguments."},
                "timeout_s": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT_S, "description": "The longest it may run (1800 by default)."},
                "stage": {"type": "string", "description": "A file the job builds, relative to the checkout — `target/release/motherd.exe` — staged for that machine's keep once the job exits 0; `agent_log` answers where."}
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

/// An agent's node as `agent_log` takes it.
pub fn place(node: u64) -> String {
    format!("{node:016x}")
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    let job = match job_of(args) {
        Ok(j) => j,
        Err(e) => return Answer::error(e),
    };
    let to = match args.get("node").and_then(Value::as_str) {
        Some(n) => match ops::node(n) {
            Ok(n) => Some(u64::from_str_radix(&n, 16).expect("sixteen hex digits")),
            Err(e) => return Answer::error(e),
        },
        None => None,
    };
    let Some(agents) = ctx.session.agents() else {
        return Answer::error(
            "this member reaches no agent: it has no cluster and no agent of its own",
        );
    };
    let command = format!("{} {}", job.program, job.args.join(" "));
    let (at, view) = match agents.submit(&ctx.caller.key(), job, to).await {
        Ok(started) => started,
        Err(e) => return Answer::error(format!("the job could not start: {e}")),
    };
    match agents.node_of(at) {
        Ok(node) => Answer::ok(format!(
            "Started `{command}` as job {} on agent {}: {:?}. Read it with `agent_log {{node: \"{}\", \
             job: \"{}\"}}`.",
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
}
