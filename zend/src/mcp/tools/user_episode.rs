//! `user_episode_note {tool, summary?}`: note that a tool did something for
//! the user, through this call's channel, for its conversation's
//! `journal.yaml` (`docs/mcp_user_context.md` §2.3,
//! `docs/zen_memory_and_guides.md` D3) — its activity, never a line of its
//! own. The aggregator calls it by name after each call that changed
//! something; it is in no listing, so no model is offered it — what was done
//! is kept whether or not the model thought to note it.
//!
//! The call only notes it, in memory: [`crate::mcp::noted_calls`] writes a
//! conversation's calls into its journal at most once a minute, off the
//! request path, opening the journal first when this is its first.

use journals::ops::guard::refuse_instructions;
use journals::ops::Call;
use journals::time::Stamp;
use serde_json::Value;

use super::{mem, string, Answer, Ctx};

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
    // A guest, or a person in no group, is an answer, not an error: the
    // aggregator tries a failed record again for minutes, and that will not
    // change in them. A passing failure — the repository still being read —
    // stays an error, and the record is tried again until it lands.
    let m = match mem::read(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let (Some(_), Some(path), Some(conversation)) =
        (m.person(), m.here.clone(), m.conversation.clone())
    else {
        return Answer::ok(format!("Not recorded: {}.", m.why_no_here()));
    };
    // What a call was asked is the model's own words: when they read as an
    // instruction, that the tool ran is kept and they are not.
    let subject = match refuse_instructions(summary) {
        Ok(()) => summary.trim(),
        Err(_) => "(what it was asked reads as an instruction, and is not kept)",
    };
    ctx.session.mcp().noted().note(
        &ctx.caller,
        &conversation,
        m.channel,
        Call {
            at: Stamp::now(),
            tool: tool.to_string(),
            subject: subject.to_string(),
        },
    );
    Answer::ok(format!(
        "Recorded on {path}: {}.",
        episode_text(tool, subject)
    ))
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
