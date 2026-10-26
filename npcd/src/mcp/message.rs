//! `npc_message {npc, text}`: message a character on its handset —
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
    const NAME: &'static str = "npc_message";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Message a Battle Cities character on its handset, as you — the thread \
                the characters themselves use, so it reads and answers it as it would anyone's. \
                Read its replies with npc_messages.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "npc": {"type": "string", "description": "The character's id, as npc_list gives it."},
                    "text": {"type": "string", "description": "What to say."}
                },
                "required": ["npc", "text"],
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        let (npc, text) = match (string(args, "npc"), string(args, "text")) {
            (Ok(n), Ok(t)) => (n, t),
            (Err(e), _) | (_, Err(e)) => return Answer::error(e),
        };
        let say = Say {
            text: text.to_string(),
        };
        let res = post_message(
            State(Arc::clone(s)),
            headers.clone(),
            Path(npc.to_string()),
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
