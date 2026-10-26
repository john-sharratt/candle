//! `thread_archive {thread}`: archive a finished feature's thread
//! (`docs/claude_zend_gateway.md` §4).
//!
//! A thread is archived only when nothing on it is still working; its workers
//! waiting on an answer, and its finished workers kept to be continued, are
//! archived with it (`crate::mcp::archiving`). Attaching to the thread again
//! reopens it.

use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::archiving::{self, distil_with, Archived, When};
use crate::mcp::names;
use crate::mcp::workers::now_ms;

pub const NAME: &str = "thread_archive";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Archive a thread whose feature is done, so it no longer lists as open. Refused while \
            any worker on it is still running; workers waiting on a reply are archived with it, their \
            questions kept, and finished workers can no longer be continued. Its journal and every worker's \
            evidence stay readable, and attaching to the thread again reopens it. A thread left a day with \
            nothing new is archived on its own, and so is any worker on it left a day.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "thread": {"type": "string", "description": "The thread's name."}
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
    let distil = distil_with(&ctx.session);
    match archiving::archive_thread(
        &records,
        &distil,
        &owner,
        thread,
        "the feature is done",
        When::Now,
        now_ms(),
    ) {
        Ok(Archived::Done { line, failed }) if failed.is_empty() => {
            Answer::ok(format!("Archived T:{thread}: {line}."))
        }
        Ok(Archived::Done { line, failed }) => Answer::ok(format!(
            "Archived T:{thread}: {line}. These workers' conversations could not be distilled and \
             stay whole — call `thread_archive` again, or the sweep archives them after a day: {}.",
            failed.join("; ")
        )),
        Ok(Archived::Already { failed }) if failed.is_empty() => Answer::ok(format!(
            "T:{thread} is archived; every worker on it is too."
        )),
        Ok(Archived::Already { failed }) => Answer::ok(format!(
            "T:{thread} is archived; these workers' conversations still could not be distilled \
             and stay whole: {}.",
            failed.join("; ")
        )),
        // Only the sweep asks for quiet; archiving now is never refused as
        // active.
        Ok(Archived::Active) => Answer::ok(format!("T:{thread} was left open.")),
        Err(e) => Answer::error(e),
    }
}
