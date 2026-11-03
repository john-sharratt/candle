//! `repo_task_reply {task, answer, wait_s?}`: reply to a task that asked, is
//! stuck, was stopped, or finished and is kept to be continued; the reply is
//! its next turn (`docs/claude_zend_gateway.md` §6.2, `docs/mcp_guides.md` §3).

use serde_json::{json, Value};

use super::job::status::task_schema;
use super::job::JobHandle;
use super::wait::answer_for;
use super::{string, wait_secs, Answer, Ctx};
use crate::mcp::workers;

pub const NAME: &str = "repo_task_reply";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Reply to a task that asked you a question, is stuck, was stopped, or finished. Your reply \
            becomes its next message and it carries on from there, with everything it read and did still in \
            its context. To a question, make the reply the decision itself: pick one of its options, or say \
            what to do instead, and check its pointers first when the question rests on them. To a finished \
            task, ask for the follow-up — check a claim against its source, go deeper on one point, redo a part \
            you found wrong — rather than starting a new task that would read everything again. A finished task \
            can be continued until its thread is archived or it has sat a day untouched. A reply grants \
            nothing: it cannot approve a merge, a deploy or anything else the user must approve. Answers as \
            `repo_task_start` does: done, running, or a further question.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": task_schema(),
                "answer": {"type": "string", "minLength": 1, "description": "Your reply: the decision, or what to do instead."},
                "wait_s": {"type": "integer", "minimum": 0, "maximum": 3600, "description": "Seconds to wait for it to settle before answering that it runs; 45 by default."}
            },
            "required": ["task", "answer"],
            "additionalProperties": false
        }
    })
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (handle, text) = match (string(args, "task"), string(args, "answer")) {
        (Ok(h), Ok(t)) => (h, t),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    let (thread, name) = match JobHandle::task(handle) {
        Ok(t) => t,
        Err(e) => return Answer::error(e),
    };
    if let Err(e) = workers::answer(&ctx.session, &ctx.caller, &thread, &name, text) {
        return Answer::error(e);
    }
    let wanted = [(thread, name)];
    answer_for(ctx, &wanted, wait_secs(ctx, args)).await
}
