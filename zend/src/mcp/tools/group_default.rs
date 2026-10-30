//! `group_default {group}`: the group a person's conversations open in
//! (`docs/memory_backbone.md` §3.5), when they are in more than one.

use journals::ops::Op;
use journals::slug::Slug;
use serde_json::{json, Value};

use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "group_default";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Choose the group the user's conversations open in, when they are in more than one: \
            its master is what every new conversation starts from, a bare path such as `topics/…` means it, \
            and the conversation's own journal is made there. Their other groups are still read by naming \
            them, `@<group>/…`. Only the user's own choice: never on your own.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "group": {"type": "string", "description": "The group, such as `@zen`."}
            },
            "required": ["group"],
            "additionalProperties": false
        }
    })
}

/// The choice a call makes.
pub fn plan(args: &Value) -> Result<Op, String> {
    let group = string(args, "group").and_then(|g| Slug::parse(g.trim_start_matches('@')))?;
    Ok(Op::ChooseDefault { group })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let op = match plan(args) {
        Ok(op) => op,
        Err(e) => return Answer::error(e),
    };
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    if let Err(e) = m.need_person() {
        return Answer::error(e);
    }
    match m.memory.write(&m.actor(), &op) {
        Ok(applied) => Answer::ok(format!("{}.", applied.subject)),
        Err(e) => Answer::error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_choice_names_one_group() {
        assert_eq!(
            plan(&json!({"group": "@zen"})).unwrap(),
            Op::ChooseDefault {
                group: Slug::parse("zen").unwrap(),
            }
        );
        assert_eq!(
            plan(&json!({})).unwrap_err(),
            "`group` is missing: it is a non-empty string"
        );
    }
}
