//! `task_start {thread, name, prompt, priority?, effort?, wait_s?}`: start a
//! task on a new worker, and wait for it up to `wait_s`
//! (`docs/claude_zend_gateway.md` §3.2, §3.3, §3.4).

use serde_json::{json, Value};

use super::wait::answer_for;
use super::{string, wait_secs, Answer, Ctx};
use crate::mcp::threads::{Effort, Priority};
use crate::mcp::workers;

pub const NAME: &str = "task_start";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Delegate a task: start it on a new worker under a thread, and wait for it up to \
            `wait_s`. A worker is a model conversation running on zend, on hardware the user owns, with the \
            repository's analysis to read from and its own tools to read, run and edit with; its result is \
            the journal entry it writes. Pass the thread you attached to. State the task in full in `prompt` — \
            what to do, what to verify before answering, and the shape of the final response (verdict, \
            findings, pointers such as file:line). Name the worker for what it does, as a kebab-case slug \
            (`verify-expiry`), never a number. Answers `done` with the worker's final response and its \
            journal entries, or `running` with its path and what it has done so far — then call `task_wait` \
            on it, or start more tasks first and wait on them all at once.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "thread": {"type": "string", "description": "The thread the task belongs to; created on first use."},
                "name": {"type": "string", "description": "The worker's name: a kebab-case slug of at most 40 characters saying what it does."},
                "prompt": {"type": "string", "description": "The task, stated in full."},
                "priority": {"type": "string", "enum": ["normal", "high"], "description": "How soon its turns decode — scheduling only, not how hard it thinks: `normal` (the default) runs below a live user; `high` only when someone is waiting on it interactively."},
                "effort": {"type": "string", "enum": ["balanced", "deep", "exhaustive"], "description": "How hard it thinks (`deep` by default). `exhaustive` for work that must weigh many sources or results against each other — research that reconciles what it fetched, a review that cross-checks findings — where catching a contradiction takes thought; `balanced` for routine, mechanical tasks."},
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
    let effort = match Effort::of(args.get("effort").and_then(Value::as_str)) {
        Ok(e) => e,
        Err(e) => return Answer::error(e),
    };
    let worker = match workers::start(
        &ctx.session,
        &ctx.caller,
        thread,
        name,
        prompt,
        (priority, effort),
    ) {
        Ok(w) => w,
        Err(e) => return Answer::error(e),
    };
    let wanted = [(worker.thread.clone(), worker.name.clone())];
    answer_for(ctx, &wanted, wait_secs(ctx, args)).await
}
