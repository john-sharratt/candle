//! `group_create {group, name, about?}`: make a group, the caller its first
//! owner (`docs/memory_backbone.md` §3.5) — another household, a team, a
//! game's world — and say what it is for.

use journals::ops::Op;
use journals::slug::Slug;
use serde_json::{json, Value};

use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "group_create";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Make a new group of the user's: its own master journal, with its own people, topics, \
            conversations, threads and bots, which nothing in another group's journals mixes with — another \
            household, a team, a game's world. The user becomes its first owner, with their journal under its \
            `people`. `group` is a short slug (`battle-cities`), `name` how it is called, and `about` what it \
            is for, which its README and every frame open with. Its journals are read at `@<group>/…`; add \
            people and bots with group_add, and make it where the user's conversations open with \
            group_default.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "group": {"type": "string", "description": "A kebab-case slug, such as `battle-cities`."},
                "name": {"type": "string", "description": "How it is called, such as `Battle Cities`."},
                "about": {"type": "string", "description": "What the group is for, in a sentence or two."}
            },
            "required": ["group", "name"],
            "additionalProperties": false
        }
    })
}

/// The group a call makes: its slug, its name, and what it is for when said.
pub fn plan(args: &Value) -> Result<(Slug, String, Option<String>), String> {
    Ok((
        string(args, "group").and_then(|g| Slug::parse(g.trim_start_matches('@')))?,
        string(args, "name")?.to_string(),
        args.get("about")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|a| !a.is_empty())
            .map(str::to_string),
    ))
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let (group, name, about) = match plan(args) {
        Ok(parts) => parts,
        Err(e) => return Answer::error(e),
    };
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    if let Err(e) = m.need_person() {
        return Answer::error(e);
    }
    let said = if about.is_some() {
        ""
    } else {
        " Say what it is for with group_describe."
    };
    let op = Op::CreateGroup {
        group: group.clone(),
        name,
        about,
    };
    match m.memory.write(&m.actor(), &op) {
        Ok(applied) => Answer::ok(format!(
            "{}. Its journals are @{group}/…; add people and bots with group_add.{said}",
            applied.subject
        )),
        Err(e) => Answer::error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_group_is_made_with_what_it_is_for() {
        assert_eq!(
            plan(
                &json!({"group": "@battle-cities", "name": "Battle Cities", "about": " The game. "})
            )
            .unwrap(),
            (
                Slug::parse("battle-cities").unwrap(),
                "Battle Cities".to_string(),
                Some("The game.".to_string()),
            )
        );
        assert_eq!(
            plan(&json!({"group": "work", "name": "Work", "about": " "})).unwrap(),
            (Slug::parse("work").unwrap(), "Work".to_string(), None)
        );
        assert_eq!(
            plan(&json!({"group": "work"})).unwrap_err(),
            "`name` is missing: it is a non-empty string"
        );
    }
}
