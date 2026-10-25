//! `zend_archive {thread}`: archive a finished feature's thread
//! (`docs/claude_zend_gateway.md` §4).
//!
//! A thread is archived only when nothing on it is still working; its workers
//! waiting on an answer are archived with it, their questions kept in their
//! records and the thread's journal. Attaching to the thread again reopens it.

use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::names;
use crate::mcp::threads::{self, Note, State, Thread};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "zend_archive";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Archive a thread whose feature is done, so it no longer lists as open. Refused while \
            any worker on it is still running; workers waiting on an answer are archived with it, their \
            questions kept. Its journal and every worker's evidence stay readable, and attaching to the \
            thread again reopens it.",
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
        return Answer::error("zend is still loading its model; try again shortly");
    };
    let owner = ctx.caller.key();
    let Some(t) = threads::thread(&records, &owner, thread) else {
        return Answer::error(format!("there is no thread `T:{thread}`"));
    };
    let now = now_ms();
    let workers: Vec<_> = threads::workers(&records, &owner, thread)
        .into_iter()
        .map(|w| w.observed(now))
        .collect();
    let running: Vec<String> = workers
        .iter()
        .filter(|w| w.state.working())
        .map(|w| w.path())
        .collect();
    if !running.is_empty() {
        return Answer::error(format!(
            "workers on T:{thread} are still running: {}. Wait for them, then archive",
            running.join(", ")
        ));
    }
    let mut closed = Vec::new();
    for mut w in workers.into_iter().filter(|w| w.state.live()) {
        if let Some(Err(e)) = ctx.session.archive_worker(&w.conv_id) {
            return Answer::error(format!("{} could not be archived: {e}", w.path()));
        }
        w.error = Some(format!(
            "archived with its thread while {}",
            if w.state == State::Question {
                "waiting on an answer"
            } else {
                "stuck"
            }
        ));
        w.state = State::Error;
        w.ended_ms = Some(now);
        if let Err(e) = threads::put_worker(&records, &owner, &w) {
            return Answer::error(e.to_string());
        }
        closed.push(w.path());
    }
    let archived = Thread {
        archived: true,
        ..t
    };
    if let Err(e) = threads::put_thread(&records, &owner, &archived) {
        return Answer::error(e.to_string());
    }
    let note = Note {
        at_ms: now,
        label: "Thread archived:".into(),
        text: if closed.is_empty() {
            "the feature is done".into()
        } else {
            format!(
                "the feature is done; archived with it, unanswered: {}",
                closed.join(", ")
            )
        },
        worker: None,
    };
    if let Err(e) = threads::add_note(&records, &owner, thread, &note) {
        return Answer::error(e.to_string());
    }
    Answer::ok(format!("Archived T:{thread}. {}", note.text))
}
