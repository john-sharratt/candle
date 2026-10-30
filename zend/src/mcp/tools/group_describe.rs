//! `group_describe {about, group?}`: say what one of the user's groups is
//! for (`docs/memory_backbone.md` §3.5) — the line its README and every
//! frame open with. An owner's to write.

use journals::ops::Op;
use serde_json::{json, Value};

use super::group_add::group_of;
use super::mem::Mem;
use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "group_describe";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Say what one of the user's groups is for, in a sentence or two — what its journals \
            hold and who it is — as an owner of it asks. Every conversation in the group, and its README, \
            starts with this. `group` is the group, the conversation's when absent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "about": {"type": "string", "description": "What the group is for, such as `Zen: the family, the candle engine, Zen Code and the papers.`"},
                "group": {"type": "string", "description": "The group, such as `@battle-cities`; the conversation's when absent."}
            },
            "required": ["about"],
            "additionalProperties": false
        }
    })
}

/// The description a call makes, for `m`'s caller.
pub fn plan(m: &Mem, args: &Value) -> Result<Op, String> {
    Ok(Op::DescribeGroup {
        group: group_of(m, args)?,
        about: string(args, "about")?.to_string(),
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let op = match plan(&m, args) {
        Ok(op) => op,
        Err(e) => return Answer::error(e),
    };
    match m.memory.write(&m.actor(), &op) {
        Ok(applied) => Answer::ok(format!("{}.", applied.subject)),
        Err(e) => Answer::error(e),
    }
}

#[cfg(test)]
mod tests {
    use journals::slug::Slug;

    use super::*;
    use crate::mcp::tools::mem::tests::mem;

    #[test]
    fn a_description_names_its_group_or_the_default() {
        let m = mem("john");
        assert_eq!(
            plan(&m, &json!({"about": "Zen."})).unwrap(),
            Op::DescribeGroup {
                group: Slug::parse("zen").unwrap(),
                about: "Zen.".into(),
            }
        );
        assert_eq!(
            plan(
                &m,
                &json!({"about": "The game.", "group": "@battle-cities"})
            )
            .unwrap(),
            Op::DescribeGroup {
                group: Slug::parse("battle-cities").unwrap(),
                about: "The game.".into(),
            }
        );
        assert!(plan(&m, &json!({})).is_err());
    }
}
