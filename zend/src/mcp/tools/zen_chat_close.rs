//! `zen_chat_close {chat}`: dispose of a private chat with Zen now
//! (`docs/zen_chat.md` §3).

use serde_json::{json, Value};

use super::{string, Answer, Ctx, LOADING};
use crate::mcp::chats::dispose::{dispose, tombstone_with, Disposal, When};
use crate::mcp::chats::naming::checked_id;
use crate::mcp::workers::now_ms;

pub const NAME: &str = "zen_chat_close";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Close a private chat with Zen and delete it now: Zen's conversation is tombstoned and \
            every message ceased, so nothing said on it is kept. An answer Zen is still giving is stopped at \
            its next tool round, and the chat is deleted as it ends. Close a chat once you are done with it; \
            one left alone is deleted twelve hours after its last message.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chat": {"type": "string", "description": "The chat: the id `zen_chat_send` answered with."}
            },
            "required": ["chat"],
            "additionalProperties": false
        }
    })
}

/// What closing says once the chat is deleted.
const DISPOSED: &str = "The chat is deleted: Zen's conversation is tombstoned and every message \
    ceased.";

/// What closing says while Zen's answer is stopped.
const ANSWERING: &str = "The chat is closing: Zen's answer is being stopped at its next tool \
    round, and the chat is deleted as it ends.";

/// What closing the chat `id` came to, as the tool says it.
pub fn said(id: &str, disposal: &Result<Disposal, String>) -> String {
    match disposal {
        Ok(Disposal::Disposed) => DISPOSED.into(),
        Ok(Disposal::Answering) => ANSWERING.into(),
        // Named, so a mistyped id shows as the one that is not kept.
        Ok(Disposal::Gone | Disposal::NotDue) => format!(
            "No chat `{id}` is kept: it was already deleted, or no chat of yours has that id."
        ),
        Err(e) => format!(
            "The chat is closing, but could not be deleted yet ({e}); the next sweep deletes it, \
             within ten minutes."
        ),
    }
}

pub fn call(ctx: &Ctx, args: &Value) -> Answer {
    let id = match string(args, "chat").and_then(checked_id) {
        Ok(id) => id,
        Err(e) => return Answer::error(e),
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
    Answer::ok(said(id, &disposal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closing_says_what_came_of_it() {
        let id = "0123456789abcdef";
        assert_eq!(
            said(id, &Ok(Disposal::Disposed)),
            "The chat is deleted: Zen's conversation is tombstoned and every message ceased."
        );
        assert_eq!(
            said(id, &Ok(Disposal::Answering)),
            "The chat is closing: Zen's answer is being stopped at its next tool round, and the \
             chat is deleted as it ends."
        );
        assert_eq!(
            said(id, &Ok(Disposal::Gone)),
            "No chat `0123456789abcdef` is kept: it was already deleted, or no chat of yours has \
             that id."
        );
        assert_eq!(
            said(id, &Err("the model is still loading".into())),
            "The chat is closing, but could not be deleted yet (the model is still loading); the \
             next sweep deletes it, within ten minutes."
        );
    }
}
