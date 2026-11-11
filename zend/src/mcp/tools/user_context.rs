//! `get_user_context {ref?}`: the conversation's ref, then the frame for it
//! — the group's master, who it could be talking to, this conversation and
//! its focus, the journals the focus names, the last conversation on this
//! channel and the cues below — within the client's budget
//! (`docs/memory_backbone.md` §6.4–§6.6). Without a ref it begins a
//! conversation; with a live one it answers that one ([`begin`],
//! `docs/private_mode.md` §3). It writes nothing: the conversation's journal
//! is named, and opened by its first write. A guest is given `@guest`'s. The
//! aggregator puts the client's guide in front of the answer.

use journals::actor::GUEST;
use journals::scope::Channel;
use journals::slug::Slug;
use journals::time::Stamp;
use journals::view::frame::Budget;
use mcp_server::refs::Ref;
use serde_json::{json, Value};

use super::{mem, Answer, Ctx, LOADING};
use crate::mcp::begin::{begin, ref_line};
use crate::mcp::memory::context;
use crate::mcp::memory::who::Who;
use crate::mcp::outcomes;
use crate::mcp::records::Records;
use crate::mcp::ref_records;
use crate::mcp::workers::now_ms;

pub const NAME: &str = "get_user_context";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Begin or go on with this conversation, and read Zen's memory as it starts from it. Call \
            this first, before any other Zen tool and before answering anything, especially anything that \
            assumes shared history: \"again\", \"last time\", \"like before\", \"my\", \"our\". Its first line is \
            this conversation's `ref`: pass it as `ref` on every Zen call from then on, this one included when \
            you call it again — every other tool refuses a call without it. Then the memory: the master — what \
            the group is and the few things that last — who you could be talking to, this conversation's own \
            journal and what it has worked out so far (who is speaking, what it is about), the journals that \
            names, where the last conversation in this app left off, and every topic by its cue; and how to \
            work with the journals. It answers at once.\n\n\
            Pass `private_mode: true` when beginning the conversation if: the client says it is incognito, \
            private or temporary (a claude.ai incognito chat); the user asks to keep it off the record, private \
            or out of memory; it is confidential work material that must not land in shared memory; the \
            device or account is shared, and what one person says must not be recorded for others; it is \
            personal — health, finances, anything the user says is theirs alone; it is an automated test or \
            experiment that should not write to memory; or the user asks to open or work in a diary. Private \
            mode makes the conversation read-only for shared memory — nothing is written, messaged, run or \
            drawn — it cannot be turned off for the rest of the conversation, and it is the only mode \
            diaries can be used in.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "This conversation's ref, when it has one: an earlier get_user_context's \
                        first line. Leave it out only at the start of a conversation."
                },
                "private_mode": {
                    "type": "boolean",
                    "description": "Begin this conversation in private mode: read-only for shared memory, to \
                        its end, and the only mode diaries work in. Read when the conversation begins; it \
                        changes nothing for one that has its ref."
                }
            },
            "additionalProperties": false
        }
    })
}

/// The context when nothing is remembered for this caller and nothing will
/// be until an owner adds them to a group: an answer, not an error — the
/// aggregator asks again for every call an error leaves without its context.
/// A passing failure stays an error, so it is asked again.
pub fn nothing_kept(why: &str) -> String {
    format!("Nothing is remembered for this user here: {why}. Answer from this conversation alone.")
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let named = args.get("ref").and_then(Value::as_str);
    // The aggregator says a ref is live by sending it in `_meta` too: then it
    // is answered as it is. Otherwise its record says, and while the record
    // cannot be read nothing is renewed on a guess.
    let vouched = ctx.conversation;
    let records = ctx.session.shared_substrate();
    if named.is_some() && vouched.is_none() && records.is_none() {
        return Answer::error(LOADING);
    }
    let now = now_ms();
    let used = |r: &Ref| match vouched {
        Some(v) if v == *r => Some(now),
        _ => records
            .as_ref()
            .and_then(|s| ref_records::used(s as &dyn Records, &r.conversation())),
    };
    let begun = match begin(
        ctx.session.mcp().refs(),
        &ctx.caller.key(),
        named,
        args.get("private_mode").and_then(Value::as_bool) == Some(true),
        used,
        now,
    ) {
        Ok(b) => b,
        Err(e) => return Answer::error(e),
    };
    let opening = ref_line(&begun);
    // A private conversation has no journal of its own, not even one named
    // for its first write: it never writes one.
    let conversation = (!begun.r.private).then(|| begun.r.conversation());
    let m = match mem::read_in(ctx, conversation) {
        Ok(m) => m,
        Err(e) => return Answer::error(e),
    };
    let guest = m.who == Who::Guest;
    let Some(group) = m.group.as_ref() else {
        return Answer::ok(format!("{opening}\n\n{}", nothing_kept(&m.why_no_here())));
    };
    let guests = Slug::parse(GUEST).expect("a name");
    if guest && m.tree.root(&guests).is_none() {
        return Answer::ok(format!(
            "{opening}\n\n{}",
            nothing_kept("this account is mapped to no one")
        ));
    }
    let mut text = format!("{opening}\n\n");
    text += &context::render(
        &m.tree,
        group,
        m.here.as_ref(),
        m.channel,
        guest,
        Stamp::now(),
    );
    // What ended while nobody waited on it (`crate::mcp::outcomes`): read from
    // a bounded ring, marking nothing.
    if !guest {
        if let Some(records) = ctx.session.shared_substrate() {
            let shown = outcomes_shown(m.channel);
            let ended = outcomes::newest(&records, &ctx.caller.key(), shown);
            if let Some(block) = outcomes::block(&ended) {
                text.push('\n');
                text.push_str(&block);
            }
        }
    }
    Answer::ok(text)
}

/// How many outcomes a frame carries: one, aloud; five, written.
fn outcomes_shown(channel: Channel) -> usize {
    match context::budget(channel) {
        Budget::Voice => 1,
        Budget::Chat | Budget::Worker => 5,
    }
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
