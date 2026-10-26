//! `thread_attach {thread}`: bind the MCP session to a thread, creating it on
//! first use (`docs/claude_zend_gateway.md` §4).

use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::names;
use crate::mcp::state::ATTACHED;
use crate::mcp::threads;
use crate::mcp::workers::{self, now_ms};

pub const NAME: &str = "thread_attach";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Attach this session to a thread, creating it on first use. A thread is the place for \
            one feature or area of work: the tasks you delegate for it run under it, and its journal keeps \
            every task started and how it ended, and every decision you note. Attach before working on a \
            feature, and again after your context is compacted. Name a thread for what it is about, as a \
            kebab-case slug (`review-layers`, `bc-wearables`); two unrelated pieces of work never share one. \
            Answers the thread's path and its journal. After attaching, `\"self\"` names this thread in the \
            `journal_*` tools.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "thread": {"type": "string", "description": "The thread's name: a kebab-case slug of at most 40 characters."}
            },
            "required": ["thread"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let thread = match string(args, "thread").and_then(|t| names::check(t).map(|()| t)) {
        Ok(t) => t,
        Err(e) => return Answer::error(e),
    };
    let Some(records) = ctx.session.shared_substrate() else {
        return Answer::error("the model is still loading; try again shortly");
    };
    let owner = ctx.caller.key();
    // Under the lock archiving takes, so a sweep reading the thread sees the
    // attach or archives before it — never between.
    let attached = {
        let _held = workers::holding();
        threads::attach(&records, &owner, thread, now_ms())
    };
    if let Err(e) = attached {
        return Answer::error(e.to_string());
    }
    if let Err(e) = ctx.store.hold(&ctx.mcp_id, ATTACHED, thread, now_ms()) {
        return Answer::error(e.to_string());
    }
    let notes = threads::notes(&records, &owner, thread);
    Answer::ok(format!(
        "Attached to T:{thread}.\n\n{}",
        threads::render_journal(&format!("T:{thread}"), &notes)
    ))
}
