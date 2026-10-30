//! `journal_merge {journal, into}`: one memory journal merged into another
//! (`docs/memory_backbone.md` §2.3) — its lines join the other's by time, its
//! sub-journals move under it, and it is gone.

use journals::ops::Op;
use serde_json::{json, Value};

use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "journal_merge";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Merge one memory journal into another when they turn out to be about the same thing: \
            its lines join the other's in time order, its sub-journals move under it, and the merged one is \
            gone (history keeps it). Both are in the same place — the user's own, or one family's — under \
            `topics`, `projects` or `~/people`.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "description": "The journal to merge away."},
                "into": {"type": "string", "description": "The journal it joins."}
            },
            "required": ["journal", "into"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (from, into) = match (string(args, "journal"), string(args, "into")) {
        (Ok(f), Ok(i)) => (f, i),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let (from, into) = match (m.path(from), m.path(into)) {
        (Ok(f), Ok(i)) => (f, i),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    match m.memory.write(
        &m.actor(),
        &Op::Merge {
            from: from.clone(),
            into: into.clone(),
        },
    ) {
        Ok(_) => Answer::ok(format!("Merged {from} into {into}.")),
        Err(e) => Answer::error(e),
    }
}
