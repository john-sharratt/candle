//! `get_user_context`: the frame for this conversation — the group's master,
//! who it could be talking to, this conversation and its focus, the journals
//! the focus names, the last conversation on this channel and the cues below
//! — within the client's budget (`docs/memory_backbone.md` §6.4–§6.6). It
//! writes nothing: the conversation's journal is named, and opened by its
//! first write. A guest is given `@guest`'s.
//! The aggregator puts the client's guide in front of the answer, and gives
//! the same answer to the first call of a conversation, whatever tool it is
//! (`docs/mcp_user_context.md` §2.5).

use journals::actor::GUEST;
use journals::slug::Slug;
use journals::time::Stamp;
use serde_json::{json, Value};

use super::{mem, Answer, Ctx};
use crate::mcp::memory::context;
use crate::mcp::memory::who::Who;

pub const NAME: &str = "get_user_context";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Zen's memory, as this conversation starts from it: the master — what the group is and \
            the few things that last — who you could be talking to, this conversation's own journal and what \
            it has worked out so far (who is speaking, what it is about), the journals that names, where the \
            last conversation in this app left off, and every topic by its cue; and how to work with the \
            journals. Call this before answering anything, especially anything that assumes shared history: \
            \"again\", \"last time\", \"like before\", \"my\", \"our\". It answers at once.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false}
    })
}

/// The context when nothing is remembered for this caller and nothing will
/// be until an owner adds them to a group: an answer, not an error — the
/// aggregator asks again for every call an error leaves without its context.
/// A passing failure stays an error, so it is asked again.
pub fn nothing_kept(why: &str) -> String {
    format!("Nothing is remembered for this user here: {why}. Answer from this conversation alone.")
}

pub fn call(ctx: &Ctx, _args: &Value) -> Answer {
    let m = match mem::read(ctx) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let guest = m.who == Who::Guest;
    let Some(group) = m.group.as_ref() else {
        return Answer::ok(nothing_kept(&m.why_no_here()));
    };
    let guests = Slug::parse(GUEST).expect("a name");
    if guest && m.tree.root(&guests).is_none() {
        return Answer::ok(nothing_kept("this account is mapped to no one"));
    }
    Answer::ok(context::render(
        &m.tree,
        group,
        m.here.as_ref(),
        m.channel,
        guest,
        Stamp::now(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_caller_with_no_memory_is_told_so_in_an_answer() {
        assert_eq!(
            nothing_kept("you are in no group yet"),
            "Nothing is remembered for this user here: you are in no group yet. Answer from this conversation alone."
        );
    }
}
