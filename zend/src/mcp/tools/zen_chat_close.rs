//! `zen_chat_close {}`: dispose of this session's chat with Zen now
//! (`docs/zen_chat.md` §3).

use serde_json::{json, Value};

use super::{Answer, Ctx, LOADING};
use crate::mcp::chats::dispose::{dispose, tombstone_with, Disposal, When};
use crate::mcp::state::CHAT;
use crate::mcp::workers::now_ms;

pub const NAME: &str = "zen_chat_close";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Close this session's private chat with Zen and delete it now: Zen's conversation is \
            tombstoned and every message ceased, so nothing said on it is kept. An answer Zen is still giving \
            is stopped at its next tool round, and the chat is deleted as it ends. The next `zen_chat_send` \
            opens a new chat. Close a chat once you are done with it; one left alone is deleted twelve hours \
            after its last message.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": false}
    })
}

/// What closing says once the chat is deleted.
const DISPOSED: &str = "The chat is deleted: Zen's conversation is tombstoned and every message \
    ceased. The next `zen_chat_send` opens a new chat.";

/// What closing says while Zen's answer is stopped.
const ANSWERING: &str = "The chat is closing: Zen's answer is being stopped at its next tool \
    round, and the chat is deleted as it ends.";

/// What closing came to, as the tool says it.
pub fn said(disposal: &Result<Disposal, String>) -> String {
    match disposal {
        Ok(Disposal::Disposed) => DISPOSED.into(),
        Ok(Disposal::Answering) => ANSWERING.into(),
        Ok(Disposal::Gone | Disposal::NotDue) => "The chat was already deleted.".into(),
        Err(e) => format!(
            "The chat is closing, but could not be deleted yet ({e}); the next sweep deletes it, \
             within ten minutes."
        ),
    }
}

pub fn call(ctx: &Ctx, _args: &Value) -> Answer {
    let Some(id) = ctx.mcp.held(CHAT).filter(|id| !id.is_empty()) else {
        return Answer::ok("No chat is open on this session.");
    };
    let Some(records) = ctx.session.shared_substrate() else {
        return Answer::error(LOADING);
    };
    let disposal = dispose(
        &records,
        &tombstone_with(&ctx.session),
        &ctx.caller.key(),
        id,
        now_ms(),
        When::Now,
    );
    // The session lets go of the chat whatever came of it: it is closing, and
    // the next send opens a new one.
    if let Err(e) = ctx.store.hold(&ctx.mcp_id, &ctx.mcp, CHAT, "", now_ms()) {
        return Answer::error(format!(
            "{} This session still names it: {e}",
            said(&disposal)
        ));
    }
    Answer::ok(said(&disposal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closing_says_what_came_of_it() {
        assert_eq!(
            said(&Ok(Disposal::Disposed)),
            "The chat is deleted: Zen's conversation is tombstoned and every message ceased. The \
             next `zen_chat_send` opens a new chat."
        );
        assert_eq!(
            said(&Ok(Disposal::Answering)),
            "The chat is closing: Zen's answer is being stopped at its next tool round, and the \
             chat is deleted as it ends."
        );
        assert_eq!(said(&Ok(Disposal::Gone)), "The chat was already deleted.");
        assert_eq!(
            said(&Err("the model is still loading".into())),
            "The chat is closing, but could not be deleted yet (the model is still loading); the \
             next sweep deletes it, within ten minutes."
        );
    }
}
