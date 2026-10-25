//! `journal_note {journal, text}`: record a note on a thread's journal — a
//! decision and its reason, what the user said — as one entry, written by the
//! server in Claude's words (`docs/claude_zend_gateway.md` §4, §5.3).

use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::names::Path;
use crate::mcp::threads::{self, Note};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "journal_note";

/// The longest note: a decision and its reason, not a document.
const MAX_NOTE: usize = 2_000;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Record a note on a thread's journal: a decision and why, something the user said or \
            asked for, a constraint every later task must keep. Note each decision as you make it — the \
            thread's journal is what you re-read after your context is compacted, and what the workers on the \
            thread build on. One note per decision, in a sentence or two. `journal` is `\"self\"` for the \
            attached thread, or `T:<thread>`; workers' journals are written by the workers themselves.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "description": "`\"self\"` or `T:<thread>`."},
                "text": {"type": "string", "description": "The note, at most 2000 characters."}
            },
            "required": ["journal", "text"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (journal, text) = match (string(args, "journal"), string(args, "text")) {
        (Ok(j), Ok(t)) => (j, t),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    if text.chars().count() > MAX_NOTE {
        return Answer::error(format!(
            "the note is longer than {MAX_NOTE} characters: one note per decision, in a sentence or two"
        ));
    }
    let thread = if journal == "self" {
        match &ctx.mcp.attached {
            Some(t) => t.clone(),
            None => return Answer::error(
                "`self` names the attached thread, and none is attached: call `zend_attach` first",
            ),
        }
    } else {
        match Path::parse(journal) {
            Ok(Path::Thread(t)) => t,
            Ok(Path::Worker { .. }) => {
                return Answer::error(
                    "a worker's journal is written by the worker; note on its thread instead",
                )
            }
            Err(e) => return Answer::error(e),
        }
    };
    let Some(records) = ctx.session.shared_substrate() else {
        return Answer::error("zend is still loading its model; try again shortly");
    };
    let owner = ctx.caller.key();
    if threads::thread(&records, &owner, &thread).is_none() {
        return Answer::error(format!(
            "there is no thread `T:{thread}`: attach to it first"
        ));
    }
    let note = Note {
        at_ms: now_ms(),
        label: "Claude said:".into(),
        text: text.to_string(),
        worker: None,
    };
    match threads::add_note(&records, &owner, &thread, &note) {
        Ok(()) => {
            let n = threads::notes(&records, &owner, &thread).len();
            Answer::ok(format!("Noted as T:{thread}#{n}."))
        }
        Err(e) => Answer::error(e.to_string()),
    }
}
