//! `zen_chat_send {text, images?, chat?, effort?, projection?, seed?,
//! reprojection?, wait_s?}`: send a message, with any pictures, to Zen on a
//! private chat — a new one, or the one `chat` names — or on a conversation of
//! the web UI's that `chat` names by its id, and wait for the answer
//! (`docs/zen_chat.md` §1, §6).

use serde_json::{json, Value};

use super::{string, wait_secs, Answer, Ctx};
use crate::mcp::chats::dialogue::{self, target, Target};
use crate::mcp::chats::pictures;
use crate::mcp::chats::talk::{self, render};
use crate::mcp::threads::Effort;
use crate::mcp::workers::now_ms;
use crate::retrieval_dials::RetrievalDials;

pub const NAME: &str = "zen_chat_send";

pub fn definition() -> Value {
    RetrievalDials::with_schema(json!({
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
                "images": {"type": "array", "items": {"type": "string"}, "maxItems": 4, "description": "Pictures for Zen to look at with the message: each the id (`<hash>.jpg`) or the link that `image_upload` answered once the user's picture was sent, or that `image_generate` or `image_edit` answered. Zen reads them and answers about them."},
                "chat": {"type": "string", "description": "The chat to send on: the id an earlier `zen_chat_send` answered with, or a web UI conversation's id when the user asked you to speak in it. Leave it out to open a new chat."},
                "effort": {"type": "string", "enum": ["balanced", "deep", "exhaustive"], "description": "How hard Zen thinks about its answer: on a chat `balanced` by default, which answers soonest; on a conversation, its own dial unless named. `deep` or `exhaustive` for a question that must weigh evidence."},
                "wait_s": {"type": "integer", "minimum": 0, "description": "How long to wait for the answer (45 by default)."}
            },
            "required": ["text"],
            "additionalProperties": false
        }
    }))
}

/// The effort a send names, if it names one.
fn effort_of(args: &Value) -> Result<Option<Effort>, String> {
    match args.get("effort").and_then(Value::as_str) {
        None => Ok(None),
        named => Effort::of(named).map(Some),
    }
}

/// The pictures a send shows, as it names them: a list of ids or links. A
/// client that sends a list argument as its JSON text, as some do for a
/// property their copy of the schema lacks, is read the same; a lone string is
/// one picture.
fn pictures_of(args: &Value) -> Result<Vec<String>, String> {
    let list = |items: &[Value]| -> Result<Vec<String>, String> {
        items
            .iter()
            .map(|i| {
                i.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| "each of `images` is a picture's id or link".to_string())
            })
            .collect()
    };
    match args.get("images") {
        None => Ok(Vec::new()),
        Some(Value::Array(items)) => list(items),
        Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Array(items)) => list(&items),
            _ if s.trim().is_empty() => Ok(Vec::new()),
            _ => Ok(vec![s.trim().to_string()]),
        },
        Some(_) => Err("`images` is a list of pictures' ids or links".into()),
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
    let (effort, on, dials) = match (
        effort_of(args),
        chat_of(args),
        RetrievalDials::of_args(args),
    ) {
        (Ok(e), Ok(on), Ok(d)) => (e, on, d),
        (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => return Answer::error(e),
    };
    // Fetched before the message goes: a picture that cannot be had sends
    // nothing.
    let fetched = match pictures_of(args) {
        Ok(named) if named.is_empty() => Vec::new(),
        Ok(named) => match pictures::fetch(&ctx.session, &ctx.caller, &named).await {
            Ok(fetched) => fetched,
            Err(e) => return Answer::error(e),
        },
        Err(e) => return Answer::error(e),
    };
    let on = match on {
        Some(Target::Conversation(id)) => {
            // The turn runs where the gateway places it: the pictures go
            // with it.
            let images: Vec<String> = fetched.iter().map(|b| pictures::data_url(b)).collect();
            let wait = wait_secs(ctx, args);
            return match dialogue::send(
                &ctx.session,
                &ctx.caller,
                &id,
                text,
                &images,
                (effort, dials),
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
    // A chat's turn begins on this member: its pictures are read here, in one
    // job.
    let images = match pictures::keep(&ctx.session, &ctx.caller, fetched).await {
        Ok(kept) => kept,
        Err(e) => return Answer::error(e),
    };
    // A chat's answer is wanted soon.
    let effort = effort.unwrap_or(Effort::Balanced);
    let sent = match talk::send(
        &ctx.session,
        &ctx.caller,
        on.as_deref(),
        text,
        images,
        (effort, dials),
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
    fn a_send_shows_the_pictures_it_names() {
        assert_eq!(pictures_of(&json!({"text": "hi"})), Ok(Vec::new()));
        assert_eq!(
            pictures_of(&json!({"text": "hi", "images": ["a1", "b2"]})),
            Ok(vec!["a1".to_string(), "b2".to_string()])
        );
        // A list sent as its JSON text, and a lone id, are read too.
        assert_eq!(
            pictures_of(&json!({"images": "[\"a1\", \"b2\"]"})),
            Ok(vec!["a1".to_string(), "b2".to_string()])
        );
        assert_eq!(
            pictures_of(&json!({"images": " a1 "})),
            Ok(vec!["a1".to_string()])
        );
        assert_eq!(pictures_of(&json!({"images": ""})), Ok(Vec::new()));
        assert!(pictures_of(&json!({"images": [1]})).is_err());
        assert!(pictures_of(&json!({"images": 7})).is_err());
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
