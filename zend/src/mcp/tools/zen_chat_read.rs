//! `zen_chat_read {chat, after?, wait_s?}`: read a private chat with Zen, or a
//! conversation of the web UI's by its id, waiting while Zen answers
//! (`docs/zen_chat.md` §1, §6). It only reads: it keeps no cursor, so the
//! caller pages on with `after`.

use serde_json::{json, Value};

use super::{string, wait_secs, Answer, Ctx};
use crate::mcp::chats::dialogue::{self, target, Target};
use crate::mcp::chats::talk::{self, not_open, render};
use crate::mcp::workers::now_ms;

pub const NAME: &str = "zen_chat_read";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "annotations": {"readOnlyHint": true},
        "description": "Read a private chat with Zen: the messages after message `after`, each numbered — or, \
            with no `after`, what Zen has said since your last message. While Zen is still answering and has \
            said nothing new, waits up to `wait_s` for it. Use it when `zen_chat_send` answered that Zen was \
            still answering, and pass `after` to page through the chat from the start. `chat` is the id \
            `zen_chat_send` answered with when it opened the chat — or a web UI conversation's id, as its \
            `#conv=` link names it, whose messages are read the same way: with no `after`, from the last \
            message a person wrote.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chat": {"type": "string", "description": "The chat: the id `zen_chat_send` answered with, or a web UI conversation's id."},
                "after": {"type": "integer", "minimum": 0, "description": "The number of the last message you have read; 0 for the whole chat."},
                "wait_s": {"type": "integer", "minimum": 0, "description": "How long to wait while Zen answers (45 by default)."}
            },
            "required": ["chat"],
            "additionalProperties": false
        }
    })
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    let on = match string(args, "chat").and_then(target) {
        Ok(on) => on,
        Err(e) => return Answer::error(e),
    };
    let after = args
        .get("after")
        .and_then(Value::as_u64)
        .map(|n| n as usize);
    let wait = wait_secs(ctx, args);
    let id = match on {
        Target::Chat(id) => id,
        Target::Conversation(id) => {
            return match dialogue::read(&ctx.session, &ctx.caller, &id, after, wait.duration())
                .await
            {
                Ok(mut out) => {
                    if let Some(note) = wait.note() {
                        out.push_str("\n\n");
                        out.push_str(&note);
                    }
                    Answer::ok(out)
                }
                Err(e) => Answer::error(e),
            };
        }
    };
    let id = id.as_str();
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
    // A closing chat is answered as a send answers it: no longer open.
    if view.chat.as_ref().is_none_or(|c| c.closing) {
        return Answer::ok(not_open(id));
    }
    let mut out = render(&view, id, from, now_ms());
    if let Some(note) = wait.note() {
        out.push_str("\n\n");
        out.push_str(&note);
    }
    Answer::ok(out)
}
