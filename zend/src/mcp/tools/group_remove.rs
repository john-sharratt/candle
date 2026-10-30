//! `group_remove {person | bot, group?}`: an owner takes a person or a bot
//! out of a group, or a person leaves one (`docs/memory_backbone.md` §3.5).
//! What they wrote stays, attributed to them.

use journals::ops::Op;
use serde_json::{json, Value};

use super::group_add::{group_of, whom};
use super::{mem, Answer, Ctx};

pub const NAME: &str = "group_remove";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Take a person or a bot out of one of the user's groups — an owner's to do — or, naming \
            the user themselves, leave it. From then on they no longer read the group's journals; what they \
            wrote stays, attributed to them, and so does their journal under `people`. The group's last \
            owner stays. `group` is the conversation's when absent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "person": {"type": "string", "description": "Their slug, as group_add named them."},
                "bot": {"type": "string", "description": "A bot's id, to remove it instead of a person."},
                "group": {"type": "string", "description": "The group, such as `@battle-cities`; the conversation's when absent."}
            },
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let who = match whom(args) {
        Ok(w) => w,
        Err(e) => return Answer::error(e),
    };
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let group = match group_of(&m, args) {
        Ok(g) => g,
        Err(e) => return Answer::error(e),
    };
    match m.memory.write(&m.actor(), &Op::RemoveMember { group, who }) {
        Ok(applied) => Answer::ok(format!("{}.", applied.subject)),
        Err(e) => Answer::error(e),
    }
}
