//! `zen_chat_send {text, effort?, wait_s?}`: send a message to Zen on this
//! session's chat, opening it on first use, and wait for the answer
//! (`docs/zen_chat.md` §1).

use serde_json::{json, Value};

use super::{string, wait_secs, Answer, Ctx};
use crate::mcp::chats::talk::{self, render};
use crate::mcp::threads::Effort;
use crate::mcp::workers::now_ms;

pub const NAME: &str = "zen_chat_send";

/// What an answer opens with when the chat the session held was disposed of.
pub const REOPENED: &str = "This session's earlier chat was deleted after it ended, so this \
    message opened a new one.";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Send a message to Zen on this session's private chat, and wait up to `wait_s` for its \
            answer. Zen is the user's own model, on hardware the user owns, with the user's repositories, \
            journals and the web to read; the chat is a conversation with it, not a task — no thread, no \
            journal entry, nothing kept. The chat opens with the first message and belongs to this session \
            alone: no other session or conversation can read it. It is deleted, with everything said on it, \
            when the session is done — by `zen_chat_close`, when the session ends, or after twelve hours with \
            nothing said. Answers with Zen's reply, or that Zen is still answering: then read the reply with \
            `zen_chat_read`. One message is answered at a time. For work in a repository that should be kept \
            and checked, delegate it with `repo_task_start` instead.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The message."},
                "effort": {"type": "string", "enum": ["balanced", "deep", "exhaustive"], "description": "How hard Zen thinks about its answer: `balanced` by default, which answers soonest; `deep` or `exhaustive` for a question that must weigh evidence."},
                "wait_s": {"type": "integer", "minimum": 0, "description": "How long to wait for the answer (45 by default)."}
            },
            "required": ["text"],
            "additionalProperties": false
        }
    })
}

/// The effort a send names, `balanced` when it names none: a chat's answer is
/// wanted soon.
fn effort_of(args: &Value) -> Result<Effort, String> {
    match args.get("effort").and_then(Value::as_str) {
        None => Ok(Effort::Balanced),
        named => Effort::of(named),
    }
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    let text = match string(args, "text") {
        Ok(t) => t,
        Err(e) => return Answer::error(e),
    };
    let effort = match effort_of(args) {
        Ok(e) => e,
        Err(e) => return Answer::error(e),
    };
    let sent = match talk::send(
        &ctx.session,
        &ctx.caller,
        &*ctx.store,
        (&ctx.mcp_id, &ctx.mcp),
        text,
        effort,
    ) {
        Ok(s) => s,
        Err(e) => return Answer::error(e),
    };
    let wait = wait_secs(ctx, args);
    let owner = ctx.caller.key();
    let read = talk::read(
        &ctx.session,
        &owner,
        &sent.chat.id,
        Some(sent.number),
        wait.duration(),
        ctx.progress.as_ref(),
    )
    .await;
    let (view, from) = match read {
        Ok(r) => r,
        Err(e) => return Answer::error(e),
    };
    let mut out = String::new();
    if sent.reopened {
        out.push_str(REOPENED);
        out.push_str("\n\n");
    }
    out.push_str(&format!("Sent as #{}. ", sent.number));
    out.push_str(&render(&view, from, now_ms()));
    if let Some(note) = wait.note() {
        out.push_str("\n\n");
        out.push_str(&note);
    }
    Answer::ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_send_thinks_balanced_unless_asked() {
        assert_eq!(effort_of(&json!({"text": "hi"})), Ok(Effort::Balanced));
        assert_eq!(
            effort_of(&json!({"text": "hi", "effort": "deep"})),
            Ok(Effort::Deep)
        );
        assert!(effort_of(&json!({"effort": "quick"})).is_err());
    }
}
