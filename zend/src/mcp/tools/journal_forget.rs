//! `journal_forget {journal, says}`: forget one line of a memory journal —
//! when the user asks for something to be forgotten, or a kept fact has
//! stopped being true (`docs/memory_backbone.md` §8). The line leaves the
//! journal's current state; the commit that removed it keeps it in history. A
//! thread's and a project's journals are their work's record, and stay as
//! written.

use journals::ops::forget::find_line;
use journals::ops::{Op, Which};
use serde_json::{json, Value};

use super::{mem, string, Answer, Ctx};
use crate::mcp::memory::is_memory_path;

pub const NAME: &str = "journal_forget";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Forget one line of a memory journal — when the user asks you to forget something, or \
            a kept fact has stopped being true. `journal` is the journal, `~` or another; name the line by a \
            few words it says (`says: \"Miso\"`), as the user would: the one line that holds them, ignoring \
            case, is forgotten; when none does, or several do, nothing is forgotten and the answer names \
            them. Or give `journal` as a line's address, `<path>#<line id>`, and no `says`. In a group's \
            journals, a member forgets their own lines and an owner any. A thread's and a project's \
            journals are the record of their work and are not forgotten.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "description": "A memory journal (`~`, `@<group>/…`), or a line's address `<path>#<line id>`."},
                "says": {"type": "string", "description": "A few words of the line to forget."}
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
    if !is_memory_path(journal) {
        return Answer::error(format!(
            "only the memory's journals forget: `{journal}` is the record of its work"
        ));
    }
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let says = args.get("says").and_then(Value::as_str);
    let (path, which) = match (journal.contains('#'), says) {
        (true, None) => match m.address(journal) {
            Ok(a) => (a.path, Which::Id(a.id)),
            Err(e) => return Answer::error(e),
        },
        (true, Some(_)) => return Answer::error("give a line's address or `says`, not both"),
        (false, Some(s)) => match m.path(journal) {
            Ok(p) => (p, Which::Says(s.to_string())),
            Err(e) => return Answer::error(e),
        },
        (false, None) => {
            return Answer::error("say which line: `says` with a few of its words, or its address")
        }
    };
    // What the line said, as this call's tree has it: told back only when the
    // store forgot that same line, since the tree may have moved on.
    let seen = m
        .tree
        .get(&path)
        .and_then(|j| find_line(j, &which, &path).ok())
        .map(|l| (l.id.clone(), l.text.clone()));
    match m.memory.write(
        &m.actor(),
        &Op::Forget {
            path: path.clone(),
            which,
        },
    ) {
        Ok(applied) => Answer::ok(match (applied.line, seen) {
            (Some(id), Some((was, text))) if id == was => format!("Forgot {path}#{id}: {text}"),
            (Some(id), _) => format!("Forgot {path}#{id}."),
            (None, _) => format!("Forgot a line of {path}."),
        }),
        Err(e) => Answer::error(e),
    }
}
