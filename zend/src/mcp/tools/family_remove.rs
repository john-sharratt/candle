//! `family_remove {person, family?}`: an owner takes a person out of their
//! family (`docs/memory_backbone.md` §3.5). Their own journals stay theirs;
//! their lines in the family's stay, attributed to them.

use journals::ops::Op;
use journals::slug::Slug;
use serde_json::{json, Value};

use super::family_add::family_of;
use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "family_remove";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Take a person out of the user's family — an owner's to do. From then on their \
            conversations no longer read the family's journals; their own journals stay theirs, and what they \
            wrote in the family's stays, attributed to them. The family's last owner stays.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "person": {"type": "string", "description": "Their slug, as family_add named them."},
                "family": {"type": "string", "description": "The family, when the user owns more than one."}
            },
            "required": ["person"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let person = match string(args, "person").and_then(Slug::parse) {
        Ok(p) => p,
        Err(e) => return Answer::error(e),
    };
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let family = match family_of(&m, args) {
        Ok(f) => f,
        Err(e) => return Answer::error(e),
    };
    match m
        .memory
        .write(&m.actor(), &Op::RemoveMember { family, person })
    {
        Ok(applied) => Answer::ok(format!("{}.", applied.subject)),
        Err(e) => Answer::error(e),
    }
}
