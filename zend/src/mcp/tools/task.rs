//! `zend_task {thread, name, prompt, priority?, wait_s?}`: start a task on a
//! new worker, and wait for it up to `wait_s` (`docs/claude_zend_gateway.md`
//! §3.2, §3.3, §3.4).

use std::time::Duration;

use serde_json::{json, Value};

use super::wait::answer_for;
use super::{string, wait_secs, Answer, Ctx};
use crate::mcp::threads::Priority;
use crate::mcp::workers;

pub const NAME: &str = "zend_task";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Delegate a task to zend: start it on a new worker under a thread, and wait for it \
            up to `wait_s`. The worker is a native zend conversation with provenance over the code's \
            analysis layers and its own tools; its result is its journal entry. State the task in full in \
            `prompt` — what to do, what to verify before answering, and the shape of the final response \
            (verdict, findings, pointers such as file:line). Name the worker for what it does, as a \
            kebab-case slug (`verify-expiry`), never a number. Answers `done` with the worker's final \
            response and its journal entries, or running with its path and what it has done so far — then \
            call `zend_wait` on it, or start more tasks first and wait on them all at once. Pass the thread \
            you attached to.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "thread": {"type": "string", "description": "The thread the task belongs to; created on first use."},
                "name": {"type": "string", "description": "The worker's name: a kebab-case slug of at most 40 characters saying what it does."},
                "prompt": {"type": "string", "description": "The task, stated in full."},
                "priority": {"type": "string", "enum": ["normal", "high"], "description": "How urgently it decodes: `normal` (the default) runs below a live user; `high` only when someone is waiting on it interactively."},
                "wait_s": {"type": "integer", "minimum": 0, "description": "How long to wait before answering `running` (45 by default)."}
            },
            "required": ["thread", "name", "prompt"],
            "additionalProperties": false
        }
    })
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (thread, name, prompt) = match (
        string(args, "thread"),
        string(args, "name"),
        string(args, "prompt"),
    ) {
        (Ok(t), Ok(n), Ok(p)) => (t, n, p),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return Answer::error(e),
    };
    let priority = match Priority::of(args.get("priority").and_then(Value::as_str)) {
        Ok(p) => p,
        Err(e) => return Answer::error(e),
    };
    let worker = match workers::start(&ctx.session, &ctx.caller, thread, name, prompt, priority) {
        Ok(w) => w,
        Err(e) => return Answer::error(e),
    };
    let wanted = [(worker.thread.clone(), worker.name.clone())];
    answer_for(ctx, &wanted, Duration::from_secs(wait_secs(ctx, args))).await
}
