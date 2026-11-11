//! `diary_rotate {topic, passphrase, new_passphrase}`: hold an open diary
//! under a new passphrase — sealed again under a new key of its own, wrapped
//! under the new passphrase — after which neither the old passphrase nor the
//! old key opens any version written from then on (`docs/private_mode.md`
//! §5.3).

use serde_json::{json, Value};

use super::diary::{conversation, passphrase, DIARIES};
use super::{string, Answer, Ctx};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "diary_rotate";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": format!("{DIARIES} Move an open diary to a new passphrase: the old one, given again and \
            matching, no longer opens it, and it is closed here — diary_open with the new one opens it. There \
            is no recovering a lost passphrase."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The diary's topic, as diary_list names it."},
                "passphrase": {"type": "string", "description": "The passphrase diary_open was given, again."},
                "new_passphrase": {"type": "string", "description": "The passphrase that holds it from now on."}
            },
            "required": ["topic", "passphrase", "new_passphrase"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let moved = conversation(ctx).and_then(|c| {
        let topic = string(args, "topic")?;
        let old = passphrase(args, "passphrase")?;
        let new = passphrase(args, "new_passphrase")?;
        ctx.session
            .mcp()
            .diaries()
            .rotate(ctx.session.mcp().memory(), &c, topic, old, new, now_ms())
            .map(|()| topic.to_string())
    });
    match moved {
        Ok(topic) => Answer::ok(format!(
            "The diary `{topic}` is held by the new passphrase now, and closed here: diary_open with \
             it opens it again."
        )),
        Err(e) => Answer::error(e),
    }
}
