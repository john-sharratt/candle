//! `group_add {person | bot, name?, emails?, role?, reads?, group?}`: an owner
//! adds a person to a group of theirs, or changes their role or accounts; or
//! adds a bot as a reader of the subtrees it is granted
//! (`docs/memory_backbone.md` §3.5).

use journals::files::{Email, Role, Who};
use journals::ops::Op;
use journals::path::Below;
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
            each bound at its first sign-in) reads every journal in the group and writes to them, and gets \
            `people/<person>` there; called for someone already in it, it adds the emails given or changes \
            their `role`: `member` by default, `owner`, or `reader`, who reads and writes nothing. Only an \
            estate admin lists emails or renames someone: an owner adds people already known here. A bot \
            (`bot`, its id) only ever reads, and only the subtrees `reads` grants it, such as `topics/game`. \
            `group` is the group, the conversation's when absent. Whoever runs the estate can read every \
            journal; say so to a person added.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "person": {"type": "string", "description": "A kebab-case slug, such as `yen-nhi`."},
                "bot": {"type": "string", "description": "A bot's id, to add it as a reader instead of a person."},
                "name": {"type": "string", "description": "Their name, for someone new."},
                "emails": {"type": "array", "items": {"type": "string"}, "description": "The accounts they sign in with; an estate admin's to list."},
                "role": {"type": "string", "enum": ["member", "owner", "reader"], "description": "`member` by default."},
                "reads": {"type": "array", "items": {"type": "string"}, "description": "A bot's grants: paths below the group, such as `topics/game`."},
                "group": {"type": "string", "description": "The group, such as `@battle-cities`; the conversation's when absent."}
            },
            "additionalProperties": false
        }
    })
}

/// The group a call names, or the caller's own.
pub fn group_of(m: &Mem, args: &Value) -> Result<Slug, String> {
    match args.get("group").and_then(Value::as_str) {
        Some(g) => Slug::parse(g.trim().trim_start_matches('@')),
        None => m
            .group
            .clone()
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
            let reads = args
                .get("reads")
                .and_then(Value::as_array)
                .ok_or(
                    "a bot reads only what it is granted: give `reads`, such as [\"topics/game\"]",
                )?
                .iter()
                .map(|r| {
                    r.as_str()
                        .ok_or_else(|| "each of `reads` is a path".to_string())
                        .and_then(|r| Below::parse(r).map_err(|e| e.to_string()))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Op::AddBot { group, bot, reads })
        }
        Who::Person(person) => {
            if args.get("reads").is_some() {
                return Err("a person reads the whole group: `reads` is a bot's".into());
            }
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

    fn slug(s: &str) -> Slug {
        Slug::parse(s).unwrap()
    }

    #[test]
    fn a_person_is_added_with_their_accounts_and_role() {
        assert_eq!(
            plan(
                &json!({"person": "yen-nhi", "name": "Yen Nhi", "emails": [" Yen.Nhi@Example.com "], "role": "reader"}),
                slug("zen")
            )
            .unwrap(),
            Op::AddMember {
                group: slug("zen"),
                person: slug("yen-nhi"),
                name: Some("Yen Nhi".into()),
                emails: vec![Email::parse("yen.nhi@example.com").unwrap()],
                role: Role::Reader,
            }
        );
        assert_eq!(
            plan(&json!({"person": "yen"}), slug("zen")).unwrap(),
            Op::AddMember {
                group: slug("zen"),
                person: slug("yen"),
                name: None,
                emails: vec![],
                role: Role::Member,
            }
        );
        assert_eq!(
            plan(&json!({"person": "yen", "reads": ["topics"]}), slug("zen")).unwrap_err(),
            "a person reads the whole group: `reads` is a bot's"
        );
    }

    #[test]
    fn a_bot_is_added_as_a_reader_of_its_grants_and_nothing_else() {
        assert_eq!(
            plan(
                &json!({"bot": "c3f1a07e", "reads": ["topics/game"]}),
                slug("battle-cities")
            )
            .unwrap(),
            Op::AddBot {
                group: slug("battle-cities"),
                bot: slug("c3f1a07e"),
                reads: vec![Below::parse("topics/game").unwrap()],
            }
        );
        assert_eq!(
            plan(&json!({"bot": "c3f1a07e"}), slug("g")).unwrap_err(),
            "a bot reads only what it is granted: give `reads`, such as [\"topics/game\"]"
        );
        assert_eq!(
            plan(
                &json!({"bot": "c3f1a07e", "reads": [], "role": "owner"}),
                slug("g")
            )
            .unwrap_err(),
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
}
