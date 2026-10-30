//! `journal_edit {journal, text}`: change what one line of a memory journal
//! says (`docs/memory_backbone.md` §3.2). Its id, time and writer stay; the
//! commit keeps what it said before.

use journals::ops::Op;
use serde_json::{json, Value};

use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "journal_edit";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Change what one line of a memory journal says, when it has become wrong or a detail \
            changed — `journal` is the line's address, `<path>#<line id>`, as journal_read and journal_grep \
            print it. The line keeps its place and time; history keeps what it said. In a group's \
            journals a member edits their own lines and an owner any.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "description": "The line's address, `<path>#<line id>`."},
                "text": {"type": "string", "description": "What it says now, at most 2000 characters."}
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
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let address = match m.address(journal) {
        Ok(a) => a,
        Err(e) => return Answer::error(e),
    };
    match m.memory.write(
        &m.actor(),
        &Op::Edit {
            address: address.clone(),
            text: text.to_string(),
        },
    ) {
        Ok(_) => Answer::ok(format!("{address} now says: {text}")),
        Err(e) => Answer::error(e),
    }
}
