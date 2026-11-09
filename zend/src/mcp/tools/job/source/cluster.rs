//! A maintenance job on a member, as a job (`docs/jobs.md` §3): its output is
//! the event stream the member keeps for it — `started`, each `step`, one
//! `done` or `error` — a line per event.

use std::time::Duration;

use axum::http::Method;
use cluster::log_read::{self, LogAsk, LogRead};
use serde_json::Value;

use super::super::{pages, JobHandle, Standing};
use super::Row;
use crate::mcp::ops::{json_of, why};
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::cluster_status::ago;
use crate::mcp::tools::Answer;
use crate::mcp::workers::now_ms;

/// How long one read of a job's stream is held open: long enough for what is
/// written to come back, short enough that a running job answers at once.
const READ_FOR: Duration = Duration::from_secs(2);

/// One event as a line of the job's output.
fn event_line(event: &Value) -> String {
    let field = |k: &str| event.get(k).and_then(Value::as_str).unwrap_or("");
    match field("event") {
        "started" => format!(
            "started {} {}",
            field("command"),
            event.get("args").cloned().unwrap_or(Value::Null)
        ),
        "step" => format!("step {}: {}", field("step"), field("detail")),
        "done" => format!(
            "done: {}",
            event.get("result").cloned().unwrap_or(Value::Null)
        ),
        "error" => format!("error {}: {}", field("error"), field("detail")),
        other => format!("{other} {event}"),
    }
}

/// The job's stream as far as it has come, and whether it has ended.
async fn stream(call: &OpsCall, node: &str, n: u64) -> Result<(String, bool), String> {
    let reply = call
        .up(
            Method::GET,
            &format!("/v1/cluster/command/job/{n}"),
            Some(node),
            READ_FOR,
        )
        .await?;
    match reply.status {
        200 => Ok((
            String::from_utf8_lossy(&reply.body).into_owned(),
            reply.complete,
        )),
        404 => Err(format!(
            "cluster:{node}:{n} is not on its member: a job is kept where it ran, and one lost to \
             a restart is gone"
        )),
        status => Err(format!("cluster:{node}:{n}: {status} — {}", why(&reply))),
    }
}

/// The steps a summary quotes, from the end.
const STEPS_SHOWN: usize = 12;

/// A job's stream as far as it was read, summarised: its state, then its last
/// steps.
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
    let mut out = match last.map(event).as_deref() {
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

/// The job's output, as text.
fn output(ndjson: &str) -> String {
    ndjson
        .lines()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .map(|e| event_line(&e) + "\n")
        .collect()
}

pub async fn standing(
    call: &OpsCall,
    job: &JobHandle,
    node: &str,
    n: u64,
) -> Result<Standing, String> {
    let (ndjson, ended) = stream(call, node, n).await?;
    let summary = summarise(&ndjson, ended);
    let (line, detail) = summary
        .split_once('\n')
        .map_or((summary.clone(), String::new()), |(a, b)| {
            (a.to_string(), format!("\n{b}"))
        });
    let text = output(&ndjson);
    Ok(Standing {
        ended,
        line: format!("{job}: {line}"),
        detail,
        total: text.lines().count() as u64,
        size: text.len() as u64,
        marks: Vec::new(),
    })
}

pub async fn read(call: &OpsCall, node: &str, n: u64, ask: &LogAsk) -> Result<LogRead, String> {
    let (ndjson, ended) = stream(call, node, n).await?;
    let text = output(&ndjson);
    let mut read = log_read::read(text.as_bytes(), ask).map_err(|e| e.to_string())?;
    read.ended = ended;
    read.size = text.len() as u64;
    Ok(read)
}

pub fn cancel(job: &JobHandle) -> Answer {
    Answer::error(format!(
        "{job} is a maintenance command, which runs to its end once started: there is no stopping \
         it partway. `job_status {{jobs: [\"{job}\"], wait_s: 240}}` waits for it."
    ))
}

/// Every member's kept maintenance jobs.
pub async fn list(call: &OpsCall, running: bool) -> Result<Vec<Row>, String> {
    let reply = call
        .up(
            Method::GET,
            "/v1/cluster/command/jobs",
            None,
            Duration::from_secs(15),
        )
        .await?;
    let envelope = json_of(&reply).map_err(|e| format!("/v1/cluster/command/jobs: {e}"))?;
    let now = now_ms();
    let mut rows = Vec::new();
    for m in envelope
        .get("members")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let (Some(node), Some(host)) = (
            m.get("node").and_then(Value::as_str),
            m.get("host").and_then(Value::as_str),
        ) else {
            continue;
        };
        for j in m
            .pointer("/body/jobs")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let ended = j.get("ended").and_then(Value::as_bool).unwrap_or(true);
            if running && ended {
                continue;
            }
            let Some(n) = j.get("id").and_then(Value::as_u64) else {
                continue;
            };
            let started = j.get("started_ms").and_then(Value::as_u64).unwrap_or(0);
            let lines = j.get("lines").and_then(Value::as_u64).unwrap_or(0);
            let job = JobHandle::Cluster {
                node: node.to_string(),
                n,
            };
            rows.push(Row {
                line: format!(
                    "{job} on {host}: `{}` — {}, started {}; last: {}, {} pages",
                    j.get("command").and_then(Value::as_str).unwrap_or("?"),
                    if ended { "ended" } else { "running" },
                    ago(started, now),
                    j.get("last").map_or_else(String::new, event_line),
                    pages(lines)
                ),
                job,
                started_ms: started,
            });
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

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

    #[test]
    fn each_event_is_a_line() {
        let ndjson = format!(
            "{}\n{}\n{}\n",
            json!({"event": "started", "command": "drain", "args": {"app": "zend"}}),
            json!({"event": "step", "step": "zend", "detail": "3 of 9 handed off"}),
            json!({"event": "error", "error": "command_failed", "detail": "no cluster"})
        );
        assert_eq!(
            output(&ndjson),
            "started drain {\"app\":\"zend\"}\nstep zend: 3 of 9 handed off\nerror \
             command_failed: no cluster\n"
        );
    }
}
