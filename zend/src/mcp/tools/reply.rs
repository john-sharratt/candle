//! `task_reply {worker, answer, wait_s?}`: reply to a worker that asked or is
//! stuck; the reply is its next turn (`docs/claude_zend_gateway.md` §6.2).

use serde_json::{json, Value};

use super::wait::answer_for;
use super::{string, wait_secs, Answer, Ctx};
use crate::mcp::names::Path;
use crate::mcp::workers;

pub const NAME: &str = "task_reply";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Reply to a worker — the conversation a task runs on — that asked you a question, is \
            stuck, or was stopped. Your reply becomes its next message and it carries on from there, so make it \
            the decision itself: pick one of its options, or say what to do instead, and check its pointers \
            first when the question rests on them. A reply grants nothing: it cannot approve a merge, a deploy \
            or anything else the user must approve. Answers as `task_start` does: `done`, `running`, or a \
            further question.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "worker": {"type": "string", "description": "The worker's path, `T:<thread>/<name>`."},
                "answer": {"type": "string", "description": "Your reply: the decision, or what to do instead."},
                "wait_s": {"type": "integer", "minimum": 0, "description": "How long to wait before answering `running` (45 by default)."}
            },
            "required": ["worker", "answer"],
            "additionalProperties": false
        }
    })
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (path, text) = match (string(args, "worker"), string(args, "answer")) {
        (Ok(p), Ok(t)) => (p, t),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    let (thread, name) = match Path::parse(path) {
        Ok(Path::Worker { thread, worker }) => (thread, worker),
        Ok(_) => {
            return Answer::error(format!(
                "`{path}` is not a worker; a worker is `T:<thread>/<name>`"
            ))
        }
        Err(e) => return Answer::error(e),
    };
    if let Err(e) = workers::answer(&ctx.session, &ctx.caller, &thread, &name, text) {
        return Answer::error(e);
    }
    let wanted = [(thread, name)];
    answer_for(ctx, &wanted, wait_secs(ctx, args)).await
}
