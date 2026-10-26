//! `cluster_job {member, job, wait_s?}`: how a maintenance command's job stands
//! (`docs/claude_zend_gateway.md` §8.2). Its stream — a `started` line, `step`
//! lines, one terminal `done` or `error` — is read on the member that ran it, as
//! far as it has come within `wait_s`, and summarised. An Admin's.

use std::time::Duration;

use axum::http::Method;
use serde_json::{json, Value};

use super::{string, wait_secs, Answer, Ctx};
use crate::mcp::ops::{self, why};

pub const NAME: &str = "cluster_job";

/// The steps a summary quotes, from the end.
const STEPS_SHOWN: usize = 12;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Read a maintenance command's job on the member that ran it: whether it is running, \
            done or failed, its last steps, and its result or error. Waits up to `wait_s` for it to end, \
            answering as it stands then. A job lives in its member's memory: one lost to a restart is \
            reported lost — check `cluster_status` and run the command again. An Admin's.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "member": {"type": "string", "description": "The member's node, as `cluster_command` answered it."},
                "job": {"type": "integer", "minimum": 0, "description": "The job's id."},
                "wait_s": {"type": "integer", "minimum": 0, "description": "How long to wait for it to end (50 by default)."}
            },
            "required": ["member", "job"],
            "additionalProperties": false
        }
    })
}

/// A job's stream as far as it was read, summarised: its state, its last
/// steps, and how it ended.
pub fn summarise(ndjson: &str, complete: bool) -> String {
    let lines: Vec<Value> = ndjson
        .lines()
        .filter_map(|l| serde_json::from_str(l).ok())
        .collect();
    let event = |l: &Value| {
        l.get("event")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string()
    };
    let command = lines
        .iter()
        .find(|l| event(l) == "started")
        .and_then(|l| l.get("command").and_then(Value::as_str))
        .unwrap_or("the command")
        .to_string();
    let steps: Vec<String> = lines
        .iter()
        .filter(|l| event(l) == "step")
        .map(|l| {
            format!(
                "- {}: {}",
                l.get("step").and_then(Value::as_str).unwrap_or(""),
                l.get("detail").and_then(Value::as_str).unwrap_or("")
            )
        })
        .collect();
    let last = lines.last();
    let state = match last.map(event).as_deref() {
        Some("done") => format!(
            "`{command}` is done. Result: {}",
            last.and_then(|l| l.get("result"))
                .cloned()
                .unwrap_or(Value::Null)
        ),
        Some("error") => format!(
            "`{command}` failed: {} — {}",
            last.and_then(|l| l.get("error"))
                .and_then(Value::as_str)
                .unwrap_or(""),
            last.and_then(|l| l.get("detail"))
                .and_then(Value::as_str)
                .unwrap_or("")
        ),
        _ if complete => format!("`{command}`'s stream ended with no result: the job was lost"),
        _ => format!("`{command}` is still running"),
    };
    let mut out = state;
    if !steps.is_empty() {
        let from = steps.len().saturating_sub(STEPS_SHOWN);
        out.push_str(&format!(
            "\nSteps ({} in all, the last {}):",
            steps.len(),
            steps.len() - from
        ));
        for s in &steps[from..] {
            out.push('\n');
            out.push_str(s);
        }
    }
    out
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = ops::admin_only(&ctx.caller, NAME) {
        return Answer::error(e);
    }
    let member = match string(args, "member").and_then(ops::node) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let Some(job) = args.get("job").and_then(Value::as_u64) else {
        return Answer::error("`job` is the job's id, as `cluster_command` answered it");
    };
    let wait = wait_secs(ctx, args);
    let reply = match ops::up(
        &ctx.session,
        &ctx.caller,
        Method::GET,
        &format!("/v1/cluster/command/job/{job}"),
        Some(&member),
        None,
        Duration::from_secs(wait.secs.max(1)),
    )
    .await
    {
        Ok(r) => r,
        Err(e) => return Answer::error(e),
    };
    match reply.status {
        200 => {
            let mut text = summarise(&String::from_utf8_lossy(&reply.body), reply.complete);
            // A job still running after a wait its client cut short says so.
            if let Some(note) = wait.note().filter(|_| !reply.complete) {
                text.push_str("\n\n");
                text.push_str(&note);
            }
            Answer::ok(text)
        }
        404 => Answer::error(format!(
            "job {job} is not on {member}: a job is read on the member that ran it, and one lost \
             to a restart is gone — check `cluster_status`, then run the command again"
        )),
        status => Answer::error(format!("job {job} on {member}: {status} — {}", why(&reply))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const STARTED: &str = r#"{"event":"started","command":"drain","args":{"app":"zend"}}"#;
    const STEP: &str = r#"{"event":"step","step":"zend","detail":"3 of 9 handed off"}"#;

    #[test]
    fn a_job_reads_as_running_done_failed_or_lost() {
        assert_eq!(
            summarise(&format!("{STARTED}\n{STEP}\n"), false),
            "`drain` is still running\nSteps (1 in all, the last 1):\n- zend: 3 of 9 handed off"
        );
        let done = format!("{STARTED}\n{{\"event\":\"done\",\"result\":{{\"zend\":\"ok\"}}}}\n");
        assert_eq!(
            summarise(&done, true),
            "`drain` is done. Result: {\"zend\":\"ok\"}"
        );
        let failed = format!(
            "{STARTED}\n{{\"event\":\"error\",\"error\":\"command_failed\",\"detail\":\"no cluster\"}}\n"
        );
        assert_eq!(
            summarise(&failed, true),
            "`drain` failed: command_failed — no cluster"
        );
        assert_eq!(
            summarise(&format!("{STARTED}\n"), true),
            "`drain`'s stream ended with no result: the job was lost"
        );
    }
}
