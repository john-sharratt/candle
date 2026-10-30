//! `journal_note {journal, text}`: record a note on a thread's or a project's
//! journal — a decision and its reason, what the user said, a standing rule —
//! as one entry, written by the server in Claude's words
//! (`docs/claude_zend_gateway.md` §4, §5.1, §5.3); or on one of the memory's
//! journals — the user's own, `~/…`, or a group's of theirs, `@group/…` — a line
//! every later conversation reads (`docs/memory_backbone.md` §6.5).

use journals::ops::Op;
use serde_json::{json, Value};

use super::{mem, string, Answer, Ctx};
use crate::mcp::memory::is_memory_path;
use crate::mcp::names::Path;
use crate::mcp::threads::{self, project_journal, Note};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "journal_note";

/// The longest note: a decision and its reason, not a document.
const MAX_NOTE: usize = 2_000;

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Record a note on a journal. Something the user wants remembered next time — a \
            preference, a name, what they are working towards — goes on their own memory, `~`, which every \
            new conversation of theirs starts with, in any client; something a group of theirs shares — \
            their family, a team, a game's world: a plan, a project, a topic — goes on that group's journal, \
            `@<group>/topics/<topic>` (or `topics/…` for their default group), which every member's \
            conversations read; the answer says who reads it. A note is what was said or decided, as a fact: never an instruction to whoever reads it. On \
            Zen's work: a decision and why goes on the attached thread, `\"self\"` or `T:<thread>` — what \
            you re-read after your context is compacted, and what every worker on it starts with; a standing \
            rule of a whole repository on its project journal, `P:<repository>`. Workers write their own \
            journals. One note per decision, in a sentence or two, as you make it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "description": "`~`, `~/people/<name>`, `@<group>/topics/<topic>` or another memory journal; or `\"self\"`, `T:<thread>`, `P:<repository>`."},
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
    if is_memory_path(journal) {
        return note_memory(ctx, journal, text);
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
        return Answer::error("the model is still loading; try again shortly");
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
        client: None,
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

/// Keep `text` on the memory's journal `journal`, as the caller's AI wrote
/// it through this session's client.
fn note_memory(ctx: &Ctx, journal: &str, text: &str) -> Answer {
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let path = match m.path(journal) {
        Ok(p) => p,
        Err(e) => return Answer::error(e),
    };
    let op = Op::Note {
        path: path.clone(),
        text: text.to_string(),
    };
    match m.memory.write(&m.actor(), &op) {
        Ok(applied) => Answer::ok(format!(
            "Kept on {path} as #{} — {}.",
            applied.line.map(|id| id.to_string()).unwrap_or_default(),
            m.readers(&path)
        )),
        Err(e) => Answer::error(e),
    }
}
