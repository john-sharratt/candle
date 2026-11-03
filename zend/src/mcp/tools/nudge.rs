//! `repo_task_nudge {task, message}`: guidance for a running task, taken at its
//! next tool round (`docs/claude_zend_gateway.md` §6.2).

use serde_json::{json, Value};

use super::job::status::task_schema;
use super::job::JobHandle;
use super::wait::handle;
use super::{string, Answer, Ctx};
use crate::mcp::workers;

pub const NAME: &str = "repo_task_nudge";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Send unasked guidance to a task that is still running — when its journal shows it heading \
            the wrong way, or repeating a step. The message joins its next tool round's results, so it is read \
            at the next point the task looks up, never mid-thought; a task that ends its turn without another \
            tool round never reads it, and the thread's journal says so. It does not stop the task: for that use \
            `job_cancel`, and for a task that asked or is stuck, `repo_task_reply`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": task_schema(),
                "message": {"type": "string", "minLength": 1, "description": "The guidance, in a sentence or two."}
            },
            "required": ["task", "message"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (task, text) = match (string(args, "task"), string(args, "message")) {
        (Ok(h), Ok(t)) => (h, t),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    let (thread, name) = match JobHandle::task(task) {
        Ok(t) => t,
        Err(e) => return Answer::error(e),
    };
    match workers::nudge(&ctx.session, &ctx.caller, &thread, &name, text) {
        Ok(w) => Answer::ok(format!(
            "Queued for {}: it reads it when its next tool round ends.",
            handle(&w)
        )),
        Err(e) => Answer::error(e),
    }
}
