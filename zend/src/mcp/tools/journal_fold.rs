//! `journal_fold {journal, unfold?}`: fold a journal out of its parent's live
//! index, or back (`docs/memory_backbone.md` §2.3). A folded journal keeps
//! everything; it is still read by path and found by grep.

use journals::ops::Op;
use serde_json::{json, Value};

use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "journal_fold";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Fold a memory journal that is finished or quiet out of its parent's index, to make \
            room or tidy — its summary stays in the parent as one line, and it is still read by path and \
            found by journal_grep. With `unfold: true`, bring a folded one back into the index.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "journal": {"type": "string", "description": "The journal, such as `@sharratt/topics/house-build/permits`."},
                "unfold": {"type": "boolean", "description": "Bring it back instead (false by default)."}
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
    let unfold = args.get("unfold").and_then(Value::as_bool).unwrap_or(false);
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let path = match m.path(journal) {
        Ok(p) => p,
        Err(e) => return Answer::error(e),
    };
    let op = if unfold {
        Op::Unfold { path: path.clone() }
    } else {
        Op::Fold { path: path.clone() }
    };
    match m.memory.write(&m.actor(), &op) {
        Ok(_) if unfold => Answer::ok(format!("{path} is back in its parent's index.")),
        Ok(_) => Answer::ok(format!(
            "Folded {path}: its summary stands in its parent; it is still read by path."
        )),
        Err(e) => Answer::error(e),
    }
}
