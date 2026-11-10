//! `zen_chat_send {text, chat?, effort?, wait_s?}`: send a message to Zen on a
//! private chat — a new one, or the one `chat` names — or on a conversation of
//! the web UI's that `chat` names by its id, and wait for the answer
//! (`docs/zen_chat.md` §1, §6).

use serde_json::{json, Value};

use super::{string, wait_secs, Answer, Ctx};
use crate::mcp::chats::dialogue::{self, target, Target};
use crate::mcp::chats::talk::{self, render};
use crate::mcp::threads::Effort;
use crate::mcp::workers::now_ms;

pub const NAME: &str = "zen_chat_send";

pub fn definition() -> Value {
    json!({
        "name": NAME,
        "description": "Send a message to Zen on a private chat, and wait up to `wait_s` for its answer. Zen is \
            the user's own model, on hardware the user owns, with the user's repositories, journals and the web \
            to read; a chat is a conversation with it, not a task — no thread, no journal entry, nothing kept. \
            With no `chat`, the message opens a new chat, and the answer names its id: pass that id as `chat` \
            on every later message, read and close, since nothing else ties the chat to you — a new chat for \
            each subject or conversation, never another's id. A chat is deleted, with everything said on it, \
            by `zen_chat_close`, or after twelve hours with nothing said. `chat` may instead name a \
            conversation of the web UI's by its id, as its `#conv=` link names it, when the user asks you to \
            speak in it: the message is that conversation's next turn, exactly as if typed into it, at its own \
            dials — kept, as the conversation is. Only a conversation you may open there is reached. Answers \
            with Zen's reply, or that Zen is still answering: then read the reply with `zen_chat_read`. One \
            message is answered at a time. For work in a repository that should be kept and checked, delegate \
            it with `repo_task_start` instead.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The message."},
                "chat": {"type": "string", "description": "The chat to send on: the id an earlier `zen_chat_send` answered with, or a web UI conversation's id when the user asked you to speak in it. Leave it out to open a new chat."},
                "effort": {"type": "string", "enum": ["balanced", "deep", "exhaustive"], "description": "How hard Zen thinks about its answer: on a chat `balanced` by default, which answers soonest; on a conversation, its own dial unless named. `deep` or `exhaustive` for a question that must weigh evidence."},
                "wait_s": {"type": "integer", "minimum": 0, "description": "How long to wait for the answer (45 by default)."}
            },
            "required": ["text"],
            "additionalProperties": false
        }
    })
}

/// The effort a send names, if it names one.
fn effort_of(args: &Value) -> Result<Option<Effort>, String> {
    match args.get("effort").and_then(Value::as_str) {
        None => Ok(None),
        named => Effort::of(named).map(Some),
    }
}

/// What a send names: `None` when it opens a new chat. A `chat` given as
/// anything but a chat's or a conversation's id is refused, never taken for
/// none: the message would go to a new chat and the conversation on the old
/// one be lost.
fn chat_of(args: &Value) -> Result<Option<Target>, String> {
    match args.get("chat") {
        None => Ok(None),
        Some(_) => string(args, "chat").and_then(target).map(Some),
    }
}

pub async fn call(ctx: &Ctx, args: &Value) -> Answer {
    let text = match string(args, "text") {
        Ok(t) => t,
        Err(e) => return Answer::error(e),
    };
    let (effort, on) = match (effort_of(args), chat_of(args)) {
        (Ok(e), Ok(on)) => (e, on),
        (Err(e), _) | (_, Err(e)) => return Answer::error(e),
    };
    let on = match on {
        Some(Target::Conversation(id)) => {
            let wait = wait_secs(ctx, args);
            return match dialogue::send(
                &ctx.session,
                &ctx.caller,
                &id,
                text,
                effort,
                wait.duration(),
            )
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
        Some(Target::Chat(id)) => Some(id),
        None => None,
    };
    // A chat's answer is wanted soon.
    let effort = effort.unwrap_or(Effort::Balanced);
    let sent = match talk::send(&ctx.session, &ctx.caller, on.as_deref(), text, effort) {
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
    if on.is_none() {
        out.push_str(&format!(
            "Opened chat `{}`: pass it as `chat` to go on.\n\n",
            sent.chat.id
        ));
    }
    out.push_str(&format!("Sent as #{}. ", sent.number));
    out.push_str(&render(&view, &sent.chat.id, from, now_ms()));
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
    fn a_send_names_its_effort_or_none() {
        assert_eq!(effort_of(&json!({"text": "hi"})), Ok(None));
        assert_eq!(
            effort_of(&json!({"text": "hi", "effort": "deep"})),
            Ok(Some(Effort::Deep))
        );
        assert!(effort_of(&json!({"effort": "quick"})).is_err());
    }

    #[test]
    fn a_send_names_its_chat_or_opens_one() {
        assert_eq!(chat_of(&json!({"text": "hi"})), Ok(None));
        assert_eq!(
            chat_of(&json!({"chat": "0123456789abcdef"})),
            Ok(Some(Target::Chat("0123456789abcdef".into())))
        );
        assert_eq!(
            chat_of(&json!({"chat": " 0123456789abcdef "})),
            Ok(Some(Target::Chat("0123456789abcdef".into()))),
            "trimmed as read and close trim it"
        );
        assert_eq!(
            chat_of(&json!({"chat": "425bad8d-769b-4e3c-a85b-d2e3eef0fd5c"})),
            Ok(Some(Target::Conversation(
                "425bad8d-769b-4e3c-a85b-d2e3eef0fd5c".into()
            )))
        );
        assert!(chat_of(&json!({"chat": "../session/x"})).is_err());
        assert!(
            chat_of(&json!({"chat": 1234})).is_err(),
            "never taken for none"
        );
        assert!(chat_of(&json!({"chat": null})).is_err());
        assert!(chat_of(&json!({"chat": ""})).is_err());
    }
}
