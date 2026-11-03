//! `job_grep {job, pattern, context?}`: a job's whole output searched where it
//! is written (`docs/jobs.md` §3), each hit with its line number and the page
//! it is on.

use cluster::log_read::{LogAsk, MAX_CONTEXT};
use serde_json::{json, Value};

use super::source::{build::page_of, read};
use super::status::{allowed, job_schema};
use super::JobHandle;
use crate::mcp::ops_tools::OpsCall;
use crate::mcp::tools::{string, Answer, Ctx};

pub const NAME: &str = "job_grep";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Search a job's whole output for `pattern` — plain text, any case, not a regular \
            expression — searched where the output is written, so only the hits come back: each line that \
            holds it, numbered as in the output with the page it is on, and `context` lines around it (0 to \
            5). Past 200 hits it says how many more there were: search for something narrower. Read around a \
            hit with `job_read` on its page.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "job": job_schema(),
                "pattern": {"type": "string", "minLength": 1, "maxLength": 200, "pattern": "[^\"\\\\]+", "description": "The text to find, ignoring case; plain text."},
                "context": {"type": "integer", "minimum": 0, "maximum": MAX_CONTEXT, "description": "Lines to show either side of each hit, 0 to 5; 0 by default."}
            },
            "required": ["job", "pattern"],
            "additionalProperties": false
        }
    })
}

pub async fn run(call: &OpsCall, args: &Value) -> Answer {
    let job = match string(args, "job").and_then(JobHandle::parse) {
        Ok(j) => j,
        Err(e) => return Answer::error(e),
    };
    if let Err(e) = allowed(call, &job) {
        return Answer::error(e);
    }
    let pattern = match string(args, "pattern") {
        Ok(p) => p.to_string(),
        Err(e) => return Answer::error(e),
    };
    let context = args
        .get("context")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .min(u64::from(MAX_CONTEXT)) as u32;
    let ask = LogAsk::Grep {
        pattern: pattern.clone(),
        context,
        from: 1,
    };
    let r = match read(call, &job, &ask).await {
        Ok(r) => r,
        Err(e) => return Answer::error(e),
    };
    let needle = pattern.to_lowercase();
    let hits = r
        .lines
        .iter()
        .filter(|l| l.text.to_lowercase().contains(&needle))
        .count() as u64;
    let mut out = format!(
        "{job}: {} lines hold `{pattern}`{}",
        hits + r.more,
        if r.more > 0 {
            format!(", the first {hits} shown — search for something narrower to see the rest")
        } else {
            String::new()
        }
    );
    for l in &r.lines {
        out.push_str(&format!("\n{} (page {}): {}", l.n, page_of(l.n), l.text));
    }
    Answer::ok(out)
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    run(&OpsCall::of(ctx), args).await
}
