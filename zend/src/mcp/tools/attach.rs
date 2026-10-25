//! `zend_attach {thread}`: bind the MCP session to a thread, creating it on
//! first use (`docs/claude_zend_gateway.md` §4).

use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::names;
use crate::mcp::state;
use crate::mcp::threads;
use crate::mcp::workers::now_ms;

pub const NAME: &str = "zend_attach";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Attach this session to a zend thread — Claude's own conversation in zend for one \
            feature or area — creating it on first use. Attach before working on a feature, and again \
            after your context is compacted. A thread is named for what it is about, as a kebab-case slug \
            (`review-layers`, `bc-wearables`); two unrelated pieces of work never share one. Answers the \
            thread's path and its journal: every task started on it and how each ended. After attaching, \
            `\"self\"` names this thread in `journal_read`.",
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
        return Answer::error("zend is still loading its model; try again shortly");
    };
    let owner = ctx.caller.key();
    if let Err(e) = threads::attach(&records, &owner, thread, now_ms()) {
        return Answer::error(e.to_string());
    }
    if let Err(e) = state::attach(&records, &ctx.mcp_id, thread, now_ms()) {
        return Answer::error(e.to_string());
    }
    let notes = threads::notes(&records, &owner, thread);
    Answer::ok(format!(
        "Attached to T:{thread}.\n\n{}",
        threads::render_journal(thread, &notes)
    ))
}
