//! `journal_note {journal, text}`: record a note on a thread's or a project's
//! journal — a decision and its reason, what the user said, a standing rule —
//! as one entry, written by the server in Claude's words
//! (`docs/claude_zend_gateway.md` §4, §5.1, §5.3).

use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::names::Path;
use crate::mcp::threads::{self, project_journal, Note};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "journal_note";

/// The longest note: a decision and its reason, not a document.
const MAX_NOTE: usize = 2_000;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Record a note on a thread's or a project's journal: a decision and why, something the \
            user said or asked for, a constraint every later task must keep. One note per decision, in a \
            sentence or two, as you make it — the thread's journal is what you re-read after your context is \
            compacted, and every worker on the thread starts with it in its prompt. `journal` is `\"self\"` \
            for the attached thread, or `T:<thread>`. A standing rule of a whole repository — a convention \
            from its CLAUDE.md, something the user says always holds — goes on its project journal, \
            `P:<repository>`, which every task in every thread starts with. Workers write their own journals.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "description": "`\"self\"`, `T:<thread>` or `P:<repository>`."},
                "text": {"type": "string", "description": "The note, at most 2000 characters."}
            },
            "required": ["journal", "text"],
            "additionalProperties": false
        }
    })
}

/// The journal a note goes on, under the name its notes are kept by.
enum Target {
    Thread(String),
    Project(String),
}

impl Target {
    fn key(&self) -> String {
        match self {
            Self::Thread(t) => t.clone(),
            Self::Project(repo) => project_journal(repo),
        }
    }

    fn path(&self) -> String {
        match self {
            Self::Thread(t) => format!("T:{t}"),
            Self::Project(repo) => project_journal(repo),
        }
    }
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
    let target = if journal == "self" {
        match ctx.attached() {
            Some(t) => Target::Thread(t),
            None => return Answer::error(
                "`self` names the attached thread, and none is attached: call `thread_attach` first",
            ),
        }
    } else {
        match Path::parse(journal) {
            Ok(Path::Thread(t)) => Target::Thread(t),
            Ok(Path::Project(repo)) => Target::Project(repo),
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
    match &target {
        Target::Thread(thread) if threads::thread(&records, &owner, thread).is_none() => {
            return Answer::error(format!(
                "there is no thread `T:{thread}`: attach to it first"
            ));
        }
        Target::Project(repo) if !ctx.session.repository_names().contains(repo) => {
            return Answer::error(format!(
                "there is no repository `{repo}` in this workspace: a project's journal is \
                 `P:<repository>`, one of {}",
                ctx.session.repository_names().join(", ")
            ));
        }
        _ => {}
    }
    let note = Note {
        at_ms: now_ms(),
        label: "Claude said:".into(),
        text: text.to_string(),
        worker: None,
    };
    let key = target.key();
    match threads::add_note(&records, &owner, &key, &note) {
        Ok(()) => {
            let n = threads::notes(&records, &owner, &key).len();
            Answer::ok(format!("Noted as {}#{n}.", target.path()))
        }
        Err(e) => Answer::error(e.to_string()),
    }
}
