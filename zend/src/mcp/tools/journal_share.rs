//! `journal_share {journal, into?, name?}`: copy one of the user's own
//! journals into their family (`docs/memory_backbone.md` §3.6). Nothing moves
//! the other way.

use journals::ops::Op;
use journals::slug::Slug;
use serde_json::{json, Value};

use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "journal_share";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Share one of the user's own journals — a conversation, a journal under `~/people` — \
            with their family: it is copied, with everything below it, into the family's `topics` (or \
            `into` another place under `topics` or `projects`), every member's conversations read it from \
            then on, and the user's own is folded. Only on their word: a shared journal cannot be unshared, \
            only forgotten line by line, and the family's history keeps it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "description": "The user's own journal, such as `~/clients/grok/2026-10-30-2105`."},
                "into": {"type": "string", "description": "Where it goes in the family: `topics` by default, or `projects`, or a journal below either."},
                "name": {"type": "string", "description": "Its name there, when not its own."}
            },
            "required": ["journal"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let journal = match string(args, "journal") {
        Ok(j) => j,
        Err(e) => return Answer::error(e),
    };
    let into = args.get("into").and_then(Value::as_str).unwrap_or("topics");
    let name = match args.get("name").and_then(Value::as_str) {
        Some(n) => match Slug::parse(n) {
            Ok(s) => Some(s),
            Err(e) => return Answer::error(e),
        },
        None => None,
    };
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let (from, into) = match (m.path(journal), m.path(into)) {
        (Ok(f), Ok(i)) => (f, i),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    match m.memory.write(
        &m.actor(),
        &Op::Share {
            from: from.clone(),
            into,
            name,
        },
    ) {
        Ok(applied) => {
            let to = applied.created.expect("a share makes a journal");
            Answer::ok(format!("Shared {from} as {to} — {}.", m.readers(&to)))
        }
        Err(e) => Answer::error(e),
    }
}
