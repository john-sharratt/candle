//! `diary_read {topic}`: an open diary, whole — every entry in time order
//! with its id, never summarised or cut (`docs/private_mode.md` §5.4).

use journals::diaries::book::Diary;
use serde_json::{json, Value};

use super::diary::{conversation, DIARIES};
use super::{string, Answer, Ctx};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "diary_read";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": format!("{DIARIES} Read an open diary whole: every entry, in time order, with its id, \
            its moment and who wrote it. Nothing is summarised or cut."),
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "The diary's topic, as diary_list names it."}
            },
            "required": ["topic"],
            "additionalProperties": false
        }
    })
}

/// The diary as answered.
pub fn render(d: &Diary) -> String {
    let entries = match d.entries.len() {
        1 => "1 entry".to_string(),
        n => format!("{n} entries"),
    };
    let mut out = format!(
        "Diary `{}`, created {} — {entries}.",
        d.topic, d.created
    );
    for e in &d.entries {
        out.push_str(&format!(
            "\n\n[{}] {} ({}):\n{}",
            e.id,
            e.ts,
            e.author.name(),
            e.text
        ));
    }
    out
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let read = conversation(ctx).and_then(|c| {
        let topic = string(args, "topic")?;
        ctx.session
            .mcp()
            .diaries()
            .read(ctx.session.mcp().memory(), &c, topic, now_ms())
    });
    match read {
        Ok(d) => Answer::ok(render(&d)),
        Err(e) => Answer::error(e),
    }
}

#[cfg(test)]
mod tests {
    use journals::diaries::book::Author;
    use journals::time::Stamp;

    use super::*;

    #[test]
    fn a_diary_is_read_as_its_entries_in_order() {
        let at = |s| Stamp::parse(s).unwrap();
        let mut d = Diary::new("t", at("2026-11-11T09:00:00Z"));
        let late = d.write(at("2026-11-12T09:00:00Z"), Author::User, "later");
        let early = d.write(at("2026-11-10T09:00:00Z"), Author::Assistant, "earlier");
        assert_eq!(
            render(&d),
            format!(
                "Diary `t`, created 2026-11-11T09:00:00Z — 2 entries.\n\n[{early}] \
                 2026-11-10T09:00:00Z (assistant):\nearlier\n\n[{late}] 2026-11-12T09:00:00Z \
                 (user):\nlater"
            )
        );
    }
}
