//! `group_create {group, name, kind, about?}`: make a group, the caller its
//! first owner (`docs/memory_backbone.md` §3.5) — a family, a team, or a
//! game's world whose bots read it — and say what it is for.

use journals::files::Kind;
use journals::ops::Op;
use journals::slug::Slug;
use serde_json::{json, Value};

use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "group_create";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Make a new group of the user's — people who share its journals, its projects, topics, \
            threads and bots, and any bots added to read them. The user becomes its first owner. `group` is a \
            short slug (`battle-cities`), `name` how it is called, `kind` what it is: `family` (a household), \
            `team` (people working together) or `world` (a game's world), and `about` what it is for, \
            which its README and every frame open with. Its journals are read at `@<group>/…`; add \
            people and bots with group_add, and choose whether it opens every conversation with \
            group_frame.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "group": {"type": "string", "description": "A kebab-case slug, such as `battle-cities`."},
                "name": {"type": "string", "description": "How it is called, such as `Battle Cities`."},
                "kind": {"type": "string", "enum": ["family", "team", "world"], "description": "What it is."},
                "about": {"type": "string", "description": "What the group is for, in a sentence or two."}
            },
            "required": ["group", "name", "kind"],
            "additionalProperties": false
        }
    })
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let group = match string(args, "group").and_then(|g| Slug::parse(g.trim_start_matches('@'))) {
        Ok(g) => g,
        Err(e) => return Answer::error(e),
    };
    let (name, kind) = match (
        string(args, "name"),
        string(args, "kind").and_then(Kind::parse),
    ) {
        (Ok(n), Ok(k)) => (n.to_string(), k),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let made = match m.memory.write(
        &m.actor(),
        &Op::CreateGroup {
            group: group.clone(),
            name,
            kind,
        },
    ) {
        Ok(applied) => applied.subject,
        Err(e) => return Answer::error(e),
    };
    let said = match args.get("about").and_then(Value::as_str) {
        Some(about) => {
            let describe = Op::DescribeGroup {
                group: group.clone(),
                about: about.to_string(),
            };
            match m.memory.write(&m.actor(), &describe) {
                Ok(_) => String::new(),
                Err(e) => {
                    format!(" What it is for was not kept — {e}; say it with group_describe.")
                }
            }
        }
        None => " Say what it is for with group_describe.".to_string(),
    };
    Answer::ok(format!(
        "{made}. Its journals are @{group}/…; add people and bots with group_add.{said}"
    ))
}
