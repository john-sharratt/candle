//! `npc_messages {npc}`: the caller's thread with a character —
//! `GET /v1/npc/:nid/message`.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use serde_json::{json, Value};

use super::args::string;
use super::route::read;
use super::{said, Answer, Tool};
use crate::api::Authored;
use crate::engine::bodies::Messages;
use crate::engine::get_messages;

pub struct Thread;

impl Tool for Thread {
    const NAME: &'static str = "npc_messages";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Read your handset thread with a Battle Cities character, oldest first: \
                what you sent it and what it answered.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "npc": {"type": "string", "description": "The character's id, as npc_list gives it."}
                },
                "required": ["npc"],
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        let npc = match string(args, "npc") {
            Ok(n) => n,
            Err(e) => return Answer::error(e),
        };
        let res = get_messages(State(Arc::clone(s)), headers.clone(), Path(npc.to_string())).await;
        Answer::of(read::<Messages>(res).await.map(|t| render(npc, &t)))
    }
}

/// The thread, a line per message.
pub fn render(npc: &str, t: &Messages) -> String {
    if !t.in_a_world {
        return format!("Character {npc} has no body in a world, so there is no handset thread.");
    }
    let with = t.with.as_deref().unwrap_or(npc);
    if t.messages.is_empty() {
        return format!("Nothing has been said between you and {with} yet.");
    }
    format!("Your thread with {with}:\n{}", said(&t.messages))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::bodies::Said;

    #[test]
    fn a_thread_reads_oldest_first() {
        let t = Messages {
            messages: vec![
                Said {
                    from: "John".into(),
                    text: "Where are you?".into(),
                },
                Said {
                    from: "Paxon Vael".into(),
                    text: "The plant room.".into(),
                },
            ],
            in_a_world: true,
            with: Some("Paxon Vael".into()),
            as_name: Some("John".into()),
        };
        assert_eq!(
            render("1z", &t),
            "Your thread with Paxon Vael:\nJohn: Where are you?\nPaxon Vael: The plant room."
        );
        let away = Messages {
            messages: vec![],
            in_a_world: false,
            with: None,
            as_name: None,
        };
        assert_eq!(
            render("1z", &away),
            "Character 1z has no body in a world, so there is no handset thread."
        );
    }
}
