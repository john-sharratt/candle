//! `journal_link {under, name, to, cue}`: a reference from one journal to
//! another in its group, like a symbolic link (`docs/memory_backbone.md`
//! §2.5): a path through `under/name` goes on from `to`.

use journals::ops::Op;
use journals::slug::Slug;
use serde_json::{json, Value};

use super::mem::{APPEND_ONLY, PATHS};
use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "journal_link";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": format!("Add a reference, like a symbolic link: under the journal `under`, the name \
            `name` leads to the journal `to`, so a reader walking down from `under` reaches it without going \
            back up. Link a new topic from the journals a reader would come from (a person working on it, a \
            related topic); link a sub-topic that belongs in two places from the second. `cue` says in a line \
            when to follow it, as an index shows it. A reference stays in its group and leads to a journal, \
            never to another reference; at most ten under one journal. {APPEND_ONLY} {PATHS}"),
        "inputSchema": {
            "type": "object",
            "properties": {
                "under": {"type": "string", "description": "The journal the reference sits under, such as `people/yen`."},
                "name": {"type": "string", "description": "The name it is followed by, a kebab-case slug such as `house`."},
                "to": {"type": "string", "description": "The journal it leads to, such as `topics/house-extension`."},
                "cue": {"type": "string", "description": "One line: when to follow it."}
            },
            "required": ["under", "name", "to", "cue"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (under, name, to, cue) = match (
        string(args, "under"),
        string(args, "name"),
        string(args, "to"),
        string(args, "cue"),
    ) {
        (Ok(u), Ok(n), Ok(t), Ok(c)) => (u, n, t, c),
        (Err(e), ..) | (_, Err(e), ..) | (_, _, Err(e), _) | (.., Err(e)) => {
            return Answer::error(e)
        }
    };
    let name = match Slug::parse(name) {
        Ok(n) => n,
        Err(e) => return Answer::error(e),
    };
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    if let Err(e) = m.need_person() {
        return Answer::error(e);
    }
    let (under, to) = match (m.path(under), m.path(to)) {
        (Ok(u), Ok(t)) => (u, t),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    let op = Op::Link {
        under,
        name,
        to,
        cue: cue.to_string(),
    };
    match m.memory.write(&m.actor(), &op) {
        Ok(applied) => Answer::ok(format!("{}.", applied.subject)),
        Err(e) => Answer::error(e),
    }
}
