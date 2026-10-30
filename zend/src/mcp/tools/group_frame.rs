//! `group_frame {default?, framed}`: the user's choice of which of their
//! groups open every conversation of theirs (`docs/memory_backbone.md`
//! §3.5, §6.4) — their default, and up to two more.

use journals::ops::Op;
use journals::slug::Slug;
use serde_json::{json, Value};

use super::{mem, Answer, Ctx};

pub const NAME: &str = "group_frame";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Choose which of the user's groups open every conversation of theirs, as they ask: \
            `default`, the group a bare `topics/…` or `projects/…` means and new shared journals go to, \
            and `framed`, up to two more whose summaries every conversation also starts with. Their other \
            groups are still read when opened. Only the user's own choice: never on your own.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "default": {"type": "string", "description": "Their default group, such as `@sharratt`; kept as it is when absent."},
                "framed": {"type": "array", "items": {"type": "string"}, "description": "The other groups that open every conversation, besides the default: at most two."}
            },
            "required": ["framed"],
            "additionalProperties": false
        }
    })
}

fn group(text: &str) -> Result<Slug, String> {
    Slug::parse(text.trim().trim_start_matches('@'))
}

/// The choice a call makes.
pub fn plan(args: &Value) -> Result<Op, String> {
    let default = match args.get("default") {
        None => None,
        Some(v) => Some(group(v.as_str().ok_or("`default` is a group")?)?),
    };
    let framed = args
        .get("framed")
        .and_then(Value::as_array)
        .ok_or("`framed` is a list of groups")?
        .iter()
        .map(|v| {
            v.as_str()
                .ok_or_else(|| "`framed` lists something not a group".to_string())
                .and_then(group)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Op::ChooseGroups { default, framed })
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
    match m.memory.write(&m.actor(), &op) {
        Ok(applied) => Answer::ok(format!("{}.", applied.subject)),
        Err(e) => Answer::error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slug(s: &str) -> Slug {
        Slug::parse(s).unwrap()
    }

    #[test]
    fn a_choice_names_a_default_and_the_others_framed() {
        assert_eq!(
            plan(&json!({"default": "@work", "framed": ["@sharratt", "battle-cities"]})).unwrap(),
            Op::ChooseGroups {
                default: Some(slug("work")),
                framed: vec![slug("sharratt"), slug("battle-cities")],
            }
        );
        assert_eq!(
            plan(&json!({"framed": []})).unwrap(),
            Op::ChooseGroups {
                default: None,
                framed: vec![],
            }
        );
        assert_eq!(
            plan(&json!({})).unwrap_err(),
            "`framed` is a list of groups"
        );
        assert_eq!(
            plan(&json!({"framed": [1]})).unwrap_err(),
            "`framed` lists something not a group"
        );
        assert_eq!(
            plan(&json!({"default": 1, "framed": []})).unwrap_err(),
            "`default` is a group"
        );
    }
}
