//! `diary_delete {topic, ids}`: delete entries of an open diary, every one
//! named or none (`docs/private_mode.md` §5.4). With a `diary_write` at the
//! right past moment, it is how a diary is compacted.

use serde_json::{json, Value};

use super::diary::{conversation, DIARIES};
use super::{string, Answer, Ctx};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "diary_delete";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": format!("{DIARIES} Delete entries of an open diary by their ids, as diary_read gives them: \
            every one named, or none when one is not there. To compact, delete many and diary_write one \
            replacement at the first one's moment."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The diary's topic, as diary_list names it."},
                "ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "The entries' ids."}
            },
            "required": ["topic", "ids"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let deleted = conversation(ctx).and_then(|c| {
        let topic = string(args, "topic")?;
        let ids: Vec<String> = args
            .get("ids")
            .and_then(Value::as_array)
            .ok_or("`ids` lists the entries to delete")?
            .iter()
            .map(|v| v.as_str().map(str::to_string).ok_or("`ids` lists ids, as text"))
            .collect::<Result<_, _>>()?;
        ctx.session
            .mcp()
            .diaries()
            .delete(ctx.session.mcp().memory(), &c, topic, &ids, now_ms())
    });
    match deleted {
        Ok(1) => Answer::ok("Deleted 1 entry."),
        Ok(n) => Answer::ok(format!("Deleted {n} entries.")),
        Err(e) => Answer::error(e),
    }
}
