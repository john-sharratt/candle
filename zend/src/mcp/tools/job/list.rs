//! `job_list {kind?, running?, limit?}`: every job the caller can see, newest
//! first, a line each (`docs/jobs.md` §5). A deploy is listed once, as a
//! deploy, never again as its build.

use serde_json::{json, Value};

use super::handle::KINDS;
use super::source::list;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::{Answer, Ctx};

pub const NAME: &str = "job_list";

const DEFAULT_LIMIT: u64 = 20;
const MAX_LIMIT: u64 = 100;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "List jobs — tasks, builds and test runs, deploys, maintenance jobs — newest first, a line \
            each: its handle, where it runs, where it is, how long it has run, what it is, and how many pages \
            of output it has. Running ones only, unless `running: false`. Use it to see what is already under \
            way before starting more, and to find a job whose handle you no longer have; `job_status` then \
            waits on it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": KINDS, "description": "Only jobs of one kind."},
                "running": {"type": "boolean", "description": "Only jobs still running (true by default); false adds those that ended lately."},
                "limit": {"type": "integer", "minimum": 1, "maximum": MAX_LIMIT, "description": "The most jobs listed; 20 by default."}
            },
            "additionalProperties": false
        }
    })
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let kind = args.get("kind").and_then(Value::as_str);
    if let Some(k) = kind {
        if !KINDS.contains(&k) {
            return Answer::error(format!("`{k}` is not a kind: {}", KINDS.join(", ")));
        }
    }
    let running = args.get("running").and_then(Value::as_bool).unwrap_or(true);
    let limit = args
        .get("limit")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_LIMIT)
        .clamp(1, MAX_LIMIT) as usize;
    let (rows, unread) = list(call, kind, running, limit).await;
    let mut out = if rows.is_empty() {
        format!(
            "No {}{}jobs.",
            if running { "running " } else { "" },
            kind.map_or(String::new(), |k| format!("{k} "))
        )
    } else {
        rows.iter()
            .map(|r| r.line.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    };
    for e in unread {
        out.push_str(&format!("\n({e})"));
    }
    Answer::ok(out)
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    run(&OpsCall::of(ctx), args).await
}
