//! `get_user_context`: who the user is, what they and their groups are working
//! on, and what was done for them lately — this conversation first, then up to
//! the person and each group they frame — within the client's budget
//! (`docs/memory_backbone.md` §6.4, §6.5). The aggregator puts the client's
//! guide in front of the answer, and gives the same answer to the first call of
//! a conversation, whatever tool it is (`docs/mcp_user_context.md` §2.5).

use journals::time::Stamp;
use serde_json::{json, Value};

use super::{mem, Answer, Ctx};
use crate::mcp::memory::{context, conversations};

pub const NAME: &str = "get_user_context";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Who this user is, what they asked you to remember, what they and their family or \
            other groups are working on, and what you did together in recent conversations — pictures drawn, bots given \
            missions, work started — in this app first, then in their others; and how to work with them \
            here. Call this before answering anything from this user, especially anything that assumes \
            shared history: \"again\", \"last time\", \"like before\", \"my\", \"our\". It answers at once.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false}
    })
}

/// The context when nothing is remembered for this caller and nothing will
/// be until an owner adds them: an answer, not an error — the aggregator asks
/// again for every call an error leaves without its context. A passing
/// failure stays an error, so it is asked again.
pub fn nothing_kept(why: &str) -> String {
    format!("Nothing is remembered for this user here: {why}. Answer from this conversation alone.")
}

pub fn call(ctx: &Ctx, _args: &Value) -> Answer {
    let m = match mem::open(ctx) {
        Ok(m) => m,
        Err(e) if mem::lasting(&e) => return Answer::ok(nothing_kept(&e)),
        Err(e) => return Answer::error(e),
    };
    // Only found, never opened: a conversation that only reads leaves no
    // journal behind. Its first episode opens one.
    let here = conversations::existing(&m.memory, &ctx.mcp_id);
    Answer::ok(context::render(
        &m.tree,
        &m.person,
        here.as_ref(),
        m.channel,
        Stamp::now(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_caller_with_no_memory_is_told_so_in_an_answer() {
        assert_eq!(
            nothing_kept("this account is in no group yet"),
            "Nothing is remembered for this user here: this account is in no group yet. Answer from this conversation alone."
        );
    }
}
