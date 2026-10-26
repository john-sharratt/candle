//! `bot_message_send {bot, text}`: message a bot on its handset —
//! `POST /v1/npc/:nid/message`.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;
use serde_json::{json, Value};

use super::args::string;
use super::route::read;
use super::{Answer, Tool};
use crate::api::Authored;
use crate::engine::bodies::{MessageSent, Say};
use crate::engine::post_message;

pub struct Message;

impl Tool for Message {
    const NAME: &'static str = "bot_message_send";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Message a bot on its handset — the phone it carries in its world — as you. \
                Bots message each other the same way, so it reads and answers yours as it would \
                anyone's, in its own time. Read its replies with bot_message_read.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "bot": {"type": "string", "description": "The bot's id, as bot_list gives it."},
                    "text": {"type": "string", "description": "What to say."}
                },
                "required": ["bot", "text"],
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        let (bot, text) = match (string(args, "bot"), string(args, "text")) {
            (Ok(b), Ok(t)) => (b, t),
            (Err(e), _) | (_, Err(e)) => return Answer::error(e),
        };
        let say = Say {
            text: text.to_string(),
        };
        let res = post_message(
            State(Arc::clone(s)),
            headers.clone(),
            Path(bot.to_string()),
            Ok(Json(say)),
        )
        .await;
        Answer::of(read::<MessageSent>(res).await.map(|m| render(&m)))
    }
}

/// What sending did.
pub fn render(s: &MessageSent) -> String {
    let mut out = format!(
        "Sent to {} as {}. {} unread on its handset.",
        s.to, s.from, s.waiting_for_them
    );
    if !s.can_reply {
        out.push_str(" It carries no handset, so it cannot answer.");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_says_who_got_it_and_whether_they_can_answer() {
        let s = MessageSent {
            sent: "hi".into(),
            from: "John".into(),
            to: "Paxon Vael".into(),
            waiting_for_them: 2,
            can_reply: false,
        };
        assert_eq!(
            render(&s),
            "Sent to Paxon Vael as John. 2 unread on its handset. It carries no handset, so it \
             cannot answer."
        );
    }
}
