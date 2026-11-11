//! `diary_close {}`: lock this conversation's diaries — what was unlocked is
//! dropped and its keys zeroized (`docs/private_mode.md` §5.4). They lock on
//! their own when the conversation's ref expires.

use serde_json::{json, Value};

use super::diary::{conversation, DIARIES};
use super::{Answer, Ctx};

pub const NAME: &str = "diary_close";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": format!("{DIARIES} Lock this conversation's open diaries. They also lock on their own \
            four hours after the conversation's last call."),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false}
    })
}

pub fn call(ctx: &Ctx, _args: &Value) -> Answer {
    match conversation(ctx) {
        Ok(c) => match ctx.session.mcp().diaries().close(&c) {
            true => Answer::ok("Closed this conversation's diaries."),
            false => Answer::ok("No diary was open."),
        },
        Err(e) => Answer::error(e),
    }
}
