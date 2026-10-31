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
            `journal_*` tools — but it is the thread last attached by any of the user's conversations open \
            at once, so name the thread (`T:<thread>`) when another may be open.",
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
        let created = threads::thread(&records, &owner, thread).is_none();
        threads::attach(&records, &owner, thread, now_ms()).map(|_| created)
    };
    let created = match attached {
        Ok(created) => created,
        Err(e) => return Answer::error(e.to_string()),
    };
    if let Err(e) = ctx
        .store
        .hold(&ctx.mcp_id, &ctx.mcp, ATTACHED, thread, now_ms())
    {
        return Answer::error(e.to_string());
    }
    let notes = threads::notes(&records, &owner, thread);
    Answer::ok(answer(
        thread,
        created,
        &threads::render_journal(&format!("T:{thread}"), &notes),
    ))
}

/// What an attach says: whether the thread was just created — so a mistyped
/// name is seen at once rather than taken for an empty thread — and its
/// journal.
fn answer(thread: &str, created: bool, journal: &str) -> String {
    let head = if created {
        format!(
            "Created T:{thread}, a new thread. If you meant one that exists, `thread_list` names \
             them."
        )
    } else {
        format!("Attached to T:{thread}.")
    };
    format!("{head}\n\n{journal}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A new thread says it is new**, so a slug typed wrong is not mistaken
    /// for an existing thread with nothing in it.
    #[test]
    fn an_attach_says_whether_it_created_the_thread() {
        assert_eq!(
            answer(
                "review",
                true,
                "Journal of T:review, written by Claude — no entries yet."
            ),
            "Created T:review, a new thread. If you meant one that exists, `thread_list` names \
             them.\n\nJournal of T:review, written by Claude — no entries yet."
        );
        assert_eq!(answer("review", false, "J"), "Attached to T:review.\n\nJ");
    }
}
