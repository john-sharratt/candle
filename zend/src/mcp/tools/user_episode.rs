//! `user_episode_note {tool, summary?}`: record on the user's own journal
//! that a tool did something for them, through this session's client
//! (`docs/mcp_user_context.md` §2.3). The aggregator calls it by name after
//! each call that changed something; it is in no listing, so no model is
//! offered it — what was done is kept whether or not the model thought to
//! note it.

use serde_json::Value;

use super::{string, Answer, Ctx};
use crate::mcp::memory;
use crate::mcp::workers::now_ms;

pub const NAME: &str = "user_episode_note";

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let tool = match string(args, "tool") {
        Ok(t) => t,
        Err(e) => return Answer::error(e),
    };
    let summary = args.get("summary").and_then(Value::as_str).unwrap_or("");
    let Some(records) = ctx.session.shared_substrate() else {
        return Answer::error("the model is still loading; try again shortly");
    };
    match memory::record_episode(
        &records,
        &ctx.caller.key(),
        ctx.mcp.client.name(),
        tool,
        summary,
        now_ms(),
    ) {
        Ok(()) => Answer::ok(format!(
            "Recorded: {}.",
            memory::episode_text(tool, summary)
        )),
        Err(e) => Answer::error(e.to_string()),
    }
}
