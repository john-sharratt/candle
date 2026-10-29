//! `journal_forget {journal: "U:", says}`: end the entry of the user's own
//! journal that says `says`, when they ask for something to be forgotten
//! (`docs/mcp_user_context.md` §2.1, §4). A thread's and a project's journals
//! are their work's record, and stay as written.

use serde_json::{json, Value};

use super::{string, Answer, Ctx};
use crate::mcp::memory;
use crate::mcp::names::{Path, USER};

pub const NAME: &str = "journal_forget";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Forget one entry of the user's own journal, `U:` — when they ask you to forget \
            something, or a kept fact has stopped being true. Name the entry by a few words it says \
            (`says: \"Miso\"`), as the user would: the one entry that holds them, ignoring case, is \
            forgotten; when none does, or several do, nothing is forgotten and the answer says which. Only \
            `U:` forgets: a thread's and a project's journals are the record of their work.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "enum": [USER], "description": "`U:`, the user's own journal."},
                "says": {"type": "string", "description": "A few words of the entry to forget."}
            },
            "required": ["journal", "says"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (journal, says) = match (string(args, "journal"), string(args, "says")) {
        (Ok(j), Ok(s)) => (j, s),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    // The journal as an entry's address names it too: `U:#3` is on `U:`.
    let journal = journal.split_once('#').map_or(journal, |(j, _)| j);
    if Path::parse(journal) != Ok(Path::User) {
        return Answer::error(format!(
            "only the user's own journal, `{USER}`, forgets: `{journal}` is the record of its work"
        ));
    }
    let Some(records) = ctx.session.shared_substrate() else {
        return Answer::error("the model is still loading; try again shortly");
    };
    match memory::forget(&records, &ctx.caller.key(), says) {
        Ok(note) => Answer::ok(format!("Forgot: {} {}", note.label, note.text)),
        Err(e) => Answer::error(e),
    }
}
