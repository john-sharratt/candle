//! `journal_create {under, name, cue}`: a new named journal under a topic, a
//! person or a project (`docs/memory_backbone.md` §2.2, §2.3) — refused past
//! ten live sub-journals until one is folded or merged.

use journals::ops::{OnTaken, Op};
use journals::slug::Slug;
use serde_json::{json, Value};

use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "journal_create";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Open a new memory journal for something worth following on its own — a topic, a \
            person in the user's life, a project or part of one — under `topics`, `projects` (the family's, \
            shared with every member) or `~/people` (the user's own), or under a journal already there. \
            `name` is a short kebab-case slug; `cue` says in a line when to open it (\"open for the Da Nang \
            trip: flights, hotels, the itinerary\"), and is what every index shows. A journal holds at most \
            ten live sub-journals: fold or merge one to make room.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "under": {"type": "string", "description": "Where it goes: `topics`, `projects`, `~/people`, or a journal below one, such as `@family/topics/house-build`."},
                "name": {"type": "string", "description": "A kebab-case slug of at most 40 characters, such as `permits`."},
                "cue": {"type": "string", "description": "One line: when to open it."}
            },
            "required": ["under", "name", "cue"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (under, name, cue) = match (
        string(args, "under"),
        string(args, "name"),
        string(args, "cue"),
    ) {
        (Ok(u), Ok(n), Ok(c)) => (u, n, c),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return Answer::error(e),
    };
    let name = match Slug::parse(name) {
        Ok(n) => n,
        Err(e) => return Answer::error(e),
    };
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let under = match m.path(under) {
        Ok(p) => p,
        Err(e) => return Answer::error(e),
    };
    let op = Op::Create {
        under,
        name,
        cue: cue.to_string(),
        on_taken: OnTaken::Refuse,
    };
    match m.memory.write(&m.actor(), &op) {
        Ok(applied) => {
            let path = applied.created.expect("create makes a journal");
            Answer::ok(format!(
                "Opened {path} — {}. Note on it with journal_note.",
                m.readers(&path)
            ))
        }
        Err(e) => Answer::error(e),
    }
}
