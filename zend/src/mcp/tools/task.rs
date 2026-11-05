//! `repo_task_start {thread, name, prompt, role?, priority?, effort?, wait_s?}`: start a
//! task on a new worker, and wait for it up to `wait_s`; it answers the task's
//! handle, `task:<thread>/<name>` (`docs/claude_zend_gateway.md` §3.2, §3.4,
//! `docs/jobs.md` §1).

use serde_json::{json, Value};

use super::wait::answer_for;
use super::{string, wait_secs, Answer, Ctx};
use crate::mcp::roles::WorkerKind;
use crate::mcp::threads::{Effort, Priority};
use crate::mcp::workers;

pub const NAME: &str = "repo_task_start";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Delegate a task: start it on a new worker under a thread, and wait for it up to \
            `wait_s`. A worker is a model conversation running on hardware the user owns, with the \
            repository's analysis to read from and its own tools to read, run and edit with; its result is \
            the journal entry it writes. Pass the thread you attached to. State the task in full in `prompt` — \
            what to do, what to verify before answering, and the shape of the final response (verdict, \
            findings, pointers such as file:line). Name the worker for what it does, as a kebab-case slug \
            (`verify-expiry`), never a number; a name already used on the thread gets the next free one \
            (`verify-expiry.2`), so use the handle this answers, `task:<thread>/<name>`, not the name asked \
            for. Answers done, with the task's final response and its journal entries, or running, with its \
            handle and what it has done so far — then wait on it with `job_status`, or start more jobs first \
            and wait on them all in one `job_status` call.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "thread": {"type": "string", "description": "The thread the task belongs to; created on first use."},
                "name": {"type": "string", "description": "The worker's name: a kebab-case slug of at most 40 characters saying what it does."},
                "prompt": {"type": "string", "description": "The task, stated in full."},
                "role": {"type": "string", "enum": ["coder", "researcher", "reviewer"], "description": "The kind of work it does, which puts that kind's working rules in its system prompt: `coder` by default, for changing code; `researcher` to answer a question from evidence; `reviewer` to read a change and report what is wrong with it."},
                "priority": {"type": "string", "enum": ["normal", "high"], "description": "How soon its turns decode — scheduling only, not how hard it thinks: today both decode at the top band beside a live dialogue, since someone is waiting on the task; `normal` is the default."},
                "effort": {"type": "string", "enum": ["balanced", "deep", "exhaustive"], "description": "How hard it thinks (`deep` by default). `exhaustive` for work that must weigh many sources or results against each other — research that reconciles what it fetched, a review that cross-checks findings — where catching a contradiction takes thought; `balanced` for routine, mechanical tasks."},
                "wait_s": {"type": "integer", "minimum": 0, "maximum": 3600, "description": "Seconds to wait for it to settle before answering that it runs; 45 by default."}
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
    let kind = match WorkerKind::of(args.get("role").and_then(Value::as_str)) {
        Ok(k) => k,
        Err(e) => return Answer::error(e),
    };
    let worker = match workers::start(
        &ctx.session,
        &ctx.caller,
        thread,
        name,
        prompt,
        (priority, effort, kind),
        false,
    ) {
        Ok(w) => w,
        Err(e) => return Answer::error(e),
    };
    let wanted = [(worker.thread.clone(), worker.name.clone())];
    answer_for(ctx, &wanted, wait_secs(ctx, args)).await
}
