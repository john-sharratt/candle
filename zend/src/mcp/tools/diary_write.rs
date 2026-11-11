//! `diary_write {topic, text, ts?, author?}`: an entry in an open diary,
//! put at its moment — now, or a past one, where it then reads
//! (`docs/private_mode.md` §5.4).

use journals::diaries::book::Author;
use journals::time::Stamp;
use serde_json::{json, Value};

use super::diary::{conversation, DIARIES};
use super::{string, Answer, Ctx};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "diary_write";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": format!("{DIARIES} Write an entry in an open diary. It goes at its moment: now, or the \
            past moment `ts` names, where it then reads among the others. Answers the entry's id."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The diary's topic, as diary_list names it."},
                "text": {"type": "string", "description": "The entry."},
                "ts": {"type": "string", "description": "The moment it is about, RFC 3339 (`2026-11-10T09:00:00Z`); now when left out."},
                "author": {"type": "string", "enum": ["user", "assistant"], "description": "Who wrote it: `user` for the user's own words, `assistant` (the default) for yours."}
            },
            "required": ["topic", "text"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let written = conversation(ctx).and_then(|c| {
        let topic = string(args, "topic")?;
        let text = args
            .get("text")
            .and_then(Value::as_str)
            .ok_or("`text` is missing: the entry")?;
        let now = now_ms();
        let ts = match args.get("ts").and_then(Value::as_str) {
            Some(ts) => {
                Stamp::parse(ts.trim()).map_err(|e| format!("`ts` is not a moment: {e}"))?
            }
            None => Stamp::from_secs((now / 1000) as i64),
        };
        let author = match args.get("author").and_then(Value::as_str) {
            Some(a) => Author::parse(a).ok_or("`author` is `user` or `assistant`")?,
            None => Author::Assistant,
        };
        let id = ctx.session.mcp().diaries().write(
            ctx.session.mcp().memory(),
            &c,
            topic,
            ts,
            author,
            text,
            now,
        )?;
        Ok((id, ts))
    });
    match written {
        Ok((id, ts)) => Answer::ok(format!("Wrote entry `{id}` at {ts}.")),
        Err(e) => Answer::error(e),
    }
}
