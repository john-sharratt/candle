//! `user_episode_note {tool, summary?}`: record on this conversation's
//! journal that a tool did something for the user, through this session's
//! channel (`docs/mcp_user_context.md` §2.3, `docs/memory_backbone.md` §2). The
//! aggregator calls it by name after each call that changed something; it is
//! in no listing, so no model is offered it — what was done is kept whether or
//! not the model thought to note it.

use journals::actor::Actor;
use journals::ops::guard::refuse_instructions;
use journals::ops::Op;
use serde_json::Value;

use super::{mem, string, Answer, Ctx};
use crate::mcp::memory::conversations;

pub const NAME: &str = "user_episode_note";

/// The line an episode is kept as: the tool, and what it was asked — which
/// the aggregator has already cut short (`mcp/src/context.rs` `summary`).
pub fn episode_text(tool: &str, summary: &str) -> String {
    match summary.trim() {
        "" => tool.to_string(),
        s => format!("{tool} — {s}"),
    }
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let tool = match string(args, "tool") {
        Ok(t) => t,
        Err(e) => return Answer::error(e),
    };
    let summary = args.get("summary").and_then(Value::as_str).unwrap_or("");
    // A caller in no group is an answer, not an error: the aggregator tries
    // a failed record again for minutes, and that will not change in them. A
    // passing failure — the repository still being read — stays an error, and
    // the record is tried again until it lands.
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) if mem::lasting(&e) => return Answer::ok(format!("Not recorded: {e}.")),
        Err(e) => return Answer::error(e),
    };
    let path = match conversations::of(&m.memory, &m.person, m.channel, &ctx.mcp_id) {
        Ok(p) => p,
        Err(e) => return Answer::error(e),
    };
    // What a call was asked is the model's own words: when they read as an
    // instruction, that the tool ran is kept and they are not.
    let text = match refuse_instructions(summary) {
        Ok(()) => episode_text(tool, summary),
        Err(_) => episode_text(
            tool,
            "(what it was asked reads as an instruction, and is not kept)",
        ),
    };
    let server = Actor::Server {
        person: m.person.clone(),
        channel: m.channel,
    };
    match m.memory.write(
        &server,
        &Op::Episode {
            path: path.clone(),
            text: text.clone(),
        },
    ) {
        Ok(_) => Answer::ok(format!("Recorded on {path}: {text}.")),
        Err(e) => Answer::error(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_episode_is_its_tool_and_what_it_was_asked() {
        assert_eq!(
            episode_text("image_generate", " a cat in a hat "),
            "image_generate — a cat in a hat"
        );
        assert_eq!(episode_text("bot_list", "  "), "bot_list");
    }
}
