//! `group_add {person | bot, name?, emails?, role?, group?}`: an owner adds
//! a person to a group of theirs, or changes their role or accounts; or adds
//! a bot as a reader of its journals (`docs/memory_backbone.md` §3.5).

use journals::files::{Email, Role, Who};
use journals::ops::Op;
use journals::slug::Slug;
use serde_json::{json, Value};

use super::mem::Mem;
use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "group_add";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Add someone to one of the user's groups — an owner's to do. A person (`person`, a short \
            slug such as `yen-nhi`; `name` how they are called; `emails` the Google accounts they sign in with, \
            each bound at its first sign-in) reads and writes the group's journals and keeps their own; \
            called for someone already in it, it adds the emails given or changes their `role`: `member` by \
            default, `owner`, or `reader`, who reads and writes nothing. Only an estate admin lists emails or \
            renames someone: an owner adds people already known here, who join only through an owner who \
            shares a group with them. A bot (`bot`, its id) only ever reads the \
            group's journals. `group` is the group, the user's default when absent. Whoever runs the estate \
            can read every journal; say so to a person added.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "person": {"type": "string", "description": "A kebab-case slug, such as `yen-nhi`."},
                "bot": {"type": "string", "description": "A bot's id, to add it as a reader instead of a person."},
                "name": {"type": "string", "description": "Their name, for someone new."},
                "emails": {"type": "array", "items": {"type": "string"}, "description": "The accounts they sign in with; an estate admin's to list."},
                "role": {"type": "string", "enum": ["member", "owner", "reader"], "description": "`member` by default."},
                "group": {"type": "string", "description": "The group, such as `@battle-cities`; the user's default when absent."}
            },
            "additionalProperties": false
        }
    })
}

/// The group a call names, or the caller's default.
pub fn group_of(m: &Mem, args: &Value) -> Result<Slug, String> {
    match args.get("group").and_then(Value::as_str) {
        Some(g) => Slug::parse(g.trim().trim_start_matches('@')),
        None => m
            .default_group()
            .ok_or_else(|| "you have no default group: name the `group`".to_string()),
    }
}

/// Exactly one of `person` and `bot`, as given.
pub fn whom(args: &Value) -> Result<Who, String> {
    match (args.get("person"), args.get("bot")) {
        (Some(_), None) => string(args, "person")
            .and_then(Slug::parse)
            .map(Who::Person),
        (None, Some(_)) => string(args, "bot").and_then(Slug::parse).map(Who::Bot),
        _ => Err("name a `person` or a `bot`, one of the two".into()),
    }
}

/// The operation a call asks for, in `group`.
pub fn plan(args: &Value, group: Slug) -> Result<Op, String> {
    match whom(args)? {
        Who::Bot(bot) => {
            if ["name", "emails", "role"]
                .iter()
                .any(|k| args.get(k).is_some())
            {
                return Err("a bot has no name, emails or role here: it is a reader".into());
            }
            Ok(Op::AddBot { group, bot })
        }
        Who::Person(person) => {
            let emails = args
                .get("emails")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .filter_map(Value::as_str)
                        .map(Email::normalise)
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();
            let role = args
                .get("role")
                .and_then(Value::as_str)
                .map(Role::parse)
                .transpose()?
                .unwrap_or(Role::Member);
            Ok(Op::AddMember {
                group,
                person,
                name: args.get("name").and_then(Value::as_str).map(str::to_string),
                emails,
                role,
            })
        }
    }
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    if let Err(e) = whom(args) {
        return Answer::error(e);
    }
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let op = match group_of(&m, args).and_then(|g| plan(args, g)) {
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
    use super::*;
    use crate::mcp::tools::mem::tests::mem;

    fn slug(s: &str) -> Slug {
        Slug::parse(s).unwrap()
    }

    #[test]
    fn a_person_is_added_with_their_accounts_and_role() {
        assert_eq!(
            plan(
                &json!({"person": "yen-nhi", "name": "Yen Nhi", "emails": [" Yen.Nhi@Example.com "], "role": "reader"}),
                slug("sharratt")
            )
            .unwrap(),
            Op::AddMember {
                group: slug("sharratt"),
                person: slug("yen-nhi"),
                name: Some("Yen Nhi".into()),
                emails: vec![Email::parse("yen.nhi@example.com").unwrap()],
                role: Role::Reader,
            }
        );
        assert_eq!(
            plan(&json!({"person": "yen"}), slug("sharratt")).unwrap(),
            Op::AddMember {
                group: slug("sharratt"),
                person: slug("yen"),
                name: None,
                emails: vec![],
                role: Role::Member,
            }
        );
    }

    #[test]
    fn a_bot_is_added_as_a_reader_and_nothing_else() {
        assert_eq!(
            plan(&json!({"bot": "c3f1a07e"}), slug("battle-cities")).unwrap(),
            Op::AddBot {
                group: slug("battle-cities"),
                bot: slug("c3f1a07e"),
            }
        );
        assert_eq!(
            plan(&json!({"bot": "c3f1a07e", "role": "owner"}), slug("g")).unwrap_err(),
            "a bot has no name, emails or role here: it is a reader"
        );
    }

    #[test]
    fn a_call_names_one_person_or_one_bot() {
        for args in [json!({}), json!({"person": "a", "bot": "b"})] {
            assert_eq!(
                whom(&args).unwrap_err(),
                "name a `person` or a `bot`, one of the two"
            );
        }
        assert_eq!(
            plan(&json!({"person": "x", "role": "chief"}), slug("g")).unwrap_err(),
            Role::parse("chief").unwrap_err()
        );
    }

    #[test]
    fn a_group_unnamed_is_the_callers_default() {
        let m = mem("john");
        assert_eq!(group_of(&m, &json!({})).unwrap(), slug("sharratt"));
        assert_eq!(
            group_of(&m, &json!({"group": "@battle-cities"})).unwrap(),
            slug("battle-cities")
        );
        assert_eq!(
            group_of(&mem("solo"), &json!({})).unwrap_err(),
            "you have no default group: name the `group`"
        );
    }
}
