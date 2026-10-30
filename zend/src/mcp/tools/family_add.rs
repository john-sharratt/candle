//! `family_add {person, name?, emails?, role?, family?}`: an owner adds a
//! person to their family, or changes their role or accounts
//! (`docs/memory_backbone.md` §3.5).

use journals::files::{Email, Role};
use journals::ops::Op;
use journals::slug::Slug;
use serde_json::{json, Value};

use super::mem::Mem;
use super::{mem, string, Answer, Ctx};

pub const NAME: &str = "family_add";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Add a person to the user's family — an owner's to do — so their conversations read the \
            family's journals and keep their own: `person` is a short slug for them (`yen-nhi`), `name` how \
            they are called, `emails` the Google accounts they sign in with (each binds at its first \
            sign-in). Called for someone already in the family, it adds the emails given, or changes their \
            `role` (`member` by default, or `owner`). Whoever runs the estate can read every journal; say so \
            to the person added.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "person": {"type": "string", "description": "A kebab-case slug, such as `yen-nhi`."},
                "name": {"type": "string", "description": "Their name, for someone new."},
                "emails": {"type": "array", "items": {"type": "string"}, "description": "The accounts they sign in with."},
                "role": {"type": "string", "enum": ["member", "owner"], "description": "`member` by default."},
                "family": {"type": "string", "description": "The family, when the user owns more than one."}
            },
            "required": ["person"],
            "additionalProperties": false
        }
    })
}

/// The family a call names, or the caller's only one.
pub fn family_of(m: &Mem, args: &Value) -> Result<Slug, String> {
    match args.get("family").and_then(Value::as_str) {
        Some(f) => Slug::parse(f.trim().trim_start_matches('@')),
        None => match m.families().as_slice() {
            [one] => Ok(one.clone()),
            [] => Err("you belong to no family".into()),
            many => Err(format!(
                "you belong to {}: name the `family`",
                many.iter()
                    .map(|f| format!("@{f}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        },
    }
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let person = match string(args, "person").and_then(Slug::parse) {
        Ok(p) => p,
        Err(e) => return Answer::error(e),
    };
    let emails = match args
        .get("emails")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(Email::normalise)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()
    {
        Ok(e) => e.unwrap_or_default(),
        Err(e) => return Answer::error(e),
    };
    let role = match args
        .get("role")
        .and_then(Value::as_str)
        .map(Role::parse)
        .transpose()
    {
        Ok(r) => r.unwrap_or(Role::Member),
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
    let op = Op::AddMember {
        family,
        person,
        name: args.get("name").and_then(Value::as_str).map(str::to_string),
        emails,
        role,
    };
    match m.memory.write(&m.actor(), &op) {
        Ok(applied) => Answer::ok(format!("{}.", applied.subject)),
        Err(e) => Answer::error(e),
    }
}
