//! `diary_list {}`: the diaries open in this conversation, each with how
//! many entries it holds and the first's and last's moments
//! (`docs/private_mode.md` §5.4).

use serde_json::{json, Value};

use super::diary::{conversation, DIARIES};
use super::{Answer, Ctx};
use crate::mcp::diaries::Listed;
use crate::mcp::workers::now_ms;

pub const NAME: &str = "diary_list";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": format!("{DIARIES} The diaries open in this conversation: each topic, how many entries it \
            holds, and when its first and last entries are."),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false}
    })
}

/// The list as answered.
pub fn render(listed: &[Listed]) -> String {
    if listed.is_empty() {
        return "No diary is open.".into();
    }
    let mut out = format!("Open diaries: {}.", listed.len());
    for l in listed {
        let span = match (&l.first, &l.last) {
            (Some(first), Some(last)) => format!(", {first} to {last}"),
            _ => String::new(),
        };
        let entries = match l.entries {
            1 => "1 entry".to_string(),
            n => format!("{n} entries"),
        };
        out.push_str(&format!("\n- `{}` — {entries}{span}", l.topic));
    }
    out
}

pub fn call(ctx: &Ctx, _args: &Value) -> Answer {
    let listed = conversation(ctx).and_then(|c| {
        ctx.session
            .mcp()
            .diaries()
            .list(ctx.session.mcp().memory(), &c, now_ms())
    });
    match listed {
        Ok(l) => Answer::ok(render(&l)),
        Err(e) => Answer::error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_list_says_each_topic_its_entries_and_its_span() {
        assert_eq!(render(&[]), "No diary is open.");
        let listed = [
            Listed {
                topic: "a".into(),
                entries: 2,
                first: Some("2026-11-10T09:00:00Z".into()),
                last: Some("2026-11-11T09:00:00Z".into()),
            },
            Listed {
                topic: "b".into(),
                entries: 0,
                first: None,
                last: None,
            },
        ];
        assert_eq!(
            render(&listed),
            "Open diaries: 2.\n- `a` — 2 entries, 2026-11-10T09:00:00Z to 2026-11-11T09:00:00Z\n- \
             `b` — 0 entries"
        );
    }
}
