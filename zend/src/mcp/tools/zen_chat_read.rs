//! `zen_chat_read {after?, wait_s?}`: read this session's chat with Zen, waiting
//! while Zen answers (`docs/zen_chat.md` §1). It only reads: it keeps no
//! cursor, so the caller pages on with `after`.

use serde_json::{json, Value};

use super::{wait_secs, Answer, Ctx};
use crate::mcp::chats::talk::{self, render};
use crate::mcp::state::CHAT;
use crate::mcp::workers::now_ms;

pub const NAME: &str = "zen_chat_read";

/// What a read answers on a session that holds no chat.
pub const NO_CHAT: &str = "No chat is open on this session: `zen_chat_send` opens one.";

/// What a read answers when the session's chat was disposed of.
pub const DELETED: &str = "This session's chat was deleted, with everything said on it: \
    `zen_chat_send` opens a new one.";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Read this session's private chat with Zen: the messages after message `after`, each \
            numbered — or, with no `after`, what Zen has said since your last message. While Zen is still \
            answering and has said nothing new, waits up to `wait_s` for it. Use it when `zen_chat_send` \
            answered that Zen was still answering, and pass `after` to page through the chat from the start. \
            The chat belongs to this session alone and is deleted when the session is done.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "after": {"type": "integer", "minimum": 0, "description": "The number of the last message you have read; 0 for the whole chat."},
                "wait_s": {"type": "integer", "minimum": 0, "description": "How long to wait while Zen answers (45 by default)."}
            },
            "additionalProperties": false
        }
    })
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    let Some(id) = ctx.mcp.held(CHAT).filter(|id| !id.is_empty()) else {
        return Answer::ok(NO_CHAT);
    };
    let after = args
        .get("after")
        .and_then(Value::as_u64)
        .map(|n| n as usize);
    let wait = wait_secs(ctx, args);
    let owner = ctx.caller.key();
    let read = talk::read(
        &ctx.session,
        &owner,
        id,
        after,
        wait.duration(),
        ctx.progress.as_ref(),
    )
    .await;
    let (view, from) = match read {
        Ok(r) => r,
        Err(e) => return Answer::error(e),
    };
    if view.chat.is_none() {
        return Answer::ok(DELETED);
    }
    let mut out = render(&view, from, now_ms());
    if let Some(note) = wait.note() {
        out.push_str("\n\n");
        out.push_str(&note);
    }
    Answer::ok(out)
}
