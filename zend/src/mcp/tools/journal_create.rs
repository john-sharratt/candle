//! `journal_create {under, name, cue}`: a new named journal under a topic or
//! a person (`docs/memory_backbone.md` §2.2, §2.3) — refused past ten live
//! sub-journals, which is a restructuring and a coding task (§2.6).

use journals::ops::{OnTaken, Op};
use journals::slug::Slug;
use serde_json::{json, Value};

use super::mem::{APPEND_ONLY, PATHS};
use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "journal_create";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": format!("Open a new memory journal for something worth following on its own: a topic \
            or a project (`topics`, or under a topic as a sub-topic), or a person (`people`). Find where it \
            belongs by the topics' cues first; a project is a topic at the top, `topics/candle`. `name` is a \
            short kebab-case slug; `cue` says in a line when to open it (\"open for the Da Nang trip: \
            flights, hotels, the itinerary\"), and is what every index shows. Then link it with \
            journal_link from the journals a reader would come from, set the focus on it with \
            journal_focus, and note on it what was said. A journal holds at most ten live sub-journals: an \
            eleventh means part of the tree needs reorganising. {APPEND_ONLY} {PATHS}"),
        "inputSchema": {
            "type": "object",
            "properties": {
                "under": {"type": "string", "description": "Where it goes: `topics`, `people`, or a journal below one, such as `topics/house-extension`."},
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
    if let Err(e) = m.need_person() {
        return Answer::error(e);
    }
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
                "Opened {path} — {}. Link it with journal_link from where a reader would look, \
                 focus on it with journal_focus, and note on it with journal_note.",
                m.readers(&path)
            ))
        }
        Err(e) => Answer::error(e),
    }
}
