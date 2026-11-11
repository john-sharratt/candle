//! `diary_open {passphrase}`: unlock every diary the passphrase holds, for
//! this conversation (`docs/private_mode.md` §5.4). It answers how many, and
//! nothing else.

use serde_json::{json, Value};

use super::diary::{conversation, passphrase, DIARIES};
use super::{Answer, Ctx};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "diary_open";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": format!("{DIARIES} Unlock every diary this passphrase holds, for this conversation, in \
            place of any passphrase it held. Answers how many opened; none is no error — the passphrase is \
            then open to create diaries with."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "passphrase": {"type": "string", "description": "The diary passphrase, exactly as the user gave it."}
            },
            "required": ["passphrase"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let opened = conversation(ctx).and_then(|c| {
        let pass = passphrase(args, "passphrase")?;
        ctx.session.mcp().diaries().open(
            ctx.session.mcp().memory(),
            &ctx.caller.key(),
            &c,
            pass,
            now_ms(),
        )
    });
    match opened {
        Ok(1) => Answer::ok("Opened 1 diary."),
        Ok(n) => Answer::ok(format!("Opened {n} diaries.")),
        Err(e) => Answer::error(e),
    }
}
