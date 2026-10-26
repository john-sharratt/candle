//! `task_stop {worker, reason?, wait_s?}`: stop a running worker at its next
//! tool round, so it answers with what it has and can be redirected
//! (`docs/claude_zend_gateway.md` §6.2).

use serde_json::{json, Value};

use super::wait::answer_for;
use super::{string, wait_secs, Answer, Ctx};
use crate::mcp::names::Path;
use crate::mcp::workers;

pub const NAME: &str = "task_stop";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Stop a running worker that is going the wrong way — the same fetch failing, a step \
            repeated, a path you no longer want. It stops at its next tool round, never mid-thought: no \
            further tool call runs, and it answers with what it has found and what remains. It is then \
            stopped but live: redirect it with `task_reply`, or end it with `thread_archive`. Waits for it \
            to stop as `task_wait` does. For guidance that should not end its tools, use `task_nudge`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "worker": {"type": "string", "description": "The worker's path, `T:<thread>/<name>`."},
                "reason": {"type": "string", "description": "Why, in a sentence: the worker reads it."},
                "wait_s": {"type": "integer", "minimum": 0, "description": "How long to wait for it to stop before answering `running` (45 by default)."}
            },
            "required": ["worker"],
            "additionalProperties": false
        }
    })
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    let path = match string(args, "worker") {
        Ok(p) => p,
        Err(e) => return Answer::error(e),
    };
    let reason = args.get("reason").and_then(Value::as_str).unwrap_or("");
    let (thread, name) = match Path::parse(path) {
        Ok(Path::Worker { thread, worker }) => (thread, worker),
        Ok(_) => {
            return Answer::error(format!(
                "`{path}` is not a worker; a worker is `T:<thread>/<name>`"
            ))
        }
        Err(e) => return Answer::error(e),
    };
    if let Err(e) = workers::stop(&ctx.session, &ctx.caller, &thread, &name, reason) {
        return Answer::error(e);
    }
    answer_for(ctx, &[(thread, name)], wait_secs(ctx, args)).await
}
