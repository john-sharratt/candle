//! `bot_channel_read {world}`: what a world's channel has carried —
//! `GET /v1/world/:wid/channel`.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use serde_json::{json, Value};

use super::args::string;
use super::route::read;
use super::{said, Answer, Tool};
use crate::api::Authored;
use crate::engine::bodies::Channel;
use crate::engine::get_channel;

pub struct Read;

impl Tool for Read {
    const NAME: &'static str = "bot_channel_read";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Read a world's channel — the group chat every bot in that world is on — \
                as you have heard it: what was said since you joined it, oldest first. A world's id is \
                in bot_list.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "world": {"type": "string", "description": "The world's id."}
                },
                "required": ["world"],
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        let world = match string(args, "world") {
            Ok(w) => w,
            Err(e) => return Answer::error(e),
        };
        let res = get_channel(
            State(Arc::clone(s)),
            headers.clone(),
            Path(world.to_string()),
        )
        .await;
        Answer::of(read::<Channel>(res).await.map(|c| render(&c)))
    }
}

/// The channel, a line per message.
pub fn render(c: &Channel) -> String {
    if c.messages.is_empty() {
        return format!(
            "Nothing has been said on {} in {} since you joined it.",
            c.channel, c.world
        );
    }
    format!("{} in {}:\n{}", c.channel, c.world, said(&c.messages))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::bodies::Said;

    #[test]
    fn a_channel_reads_oldest_first_and_says_when_it_is_quiet() {
        let mut c = Channel {
            channel: "the channel".into(),
            world: "w1".into(),
            as_name: "John".into(),
            messages: vec![],
        };
        assert_eq!(
            render(&c),
            "Nothing has been said on the channel in w1 since you joined it."
        );
        c.messages.push(Said {
            from: "Pax".into(),
            text: " Report in. ".into(),
        });
        assert_eq!(render(&c), "the channel in w1:\nPax: Report in.");
    }
}
