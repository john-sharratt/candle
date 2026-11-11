//! `diary_create {topic, passphrase}`: a diary on a topic, under the
//! passphrase this conversation opened — repeated, so a typo makes no diary
//! no one can open (`docs/private_mode.md` §5.4).

use serde_json::{json, Value};

use super::diary::{conversation, passphrase, DIARIES};
use super::{string, Answer, Ctx};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "diary_create";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": format!("{DIARIES} Create a diary on a topic, under the passphrase diary_open opened in \
            this conversation, which must be given again here and match."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "What the diary is about, as the user names it."},
                "passphrase": {"type": "string", "description": "The passphrase diary_open was given, again."}
            },
            "required": ["topic", "passphrase"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let made = conversation(ctx).and_then(|c| {
        let topic = string(args, "topic")?;
        let pass = passphrase(args, "passphrase")?;
        ctx.session
            .mcp()
            .diaries()
            .create(ctx.session.mcp().memory(), &c, topic, pass, now_ms())
            .map(|()| topic.to_string())
    });
    match made {
        Ok(topic) => Answer::ok(format!("Created the diary `{topic}`.")),
        Err(e) => Answer::error(e),
    }
}
