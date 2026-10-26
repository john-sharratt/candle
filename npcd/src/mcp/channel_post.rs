//! `npc_channel_post {world, text}`: speak on a world's channel —
//! `POST /v1/world/:wid/channel`, a Creator's.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::Json;
use serde_json::{json, Value};
use web::auth::Role;

use super::args::string;
use super::route::{read, refusal};
use super::{Answer, Tool};
use crate::api::Authored;
use crate::engine::bodies::{ChannelSent, Say};
use crate::engine::post_channel;
use crate::identity::require;

pub struct Post;

impl Tool for Post {
    const NAME: &'static str = "npc_channel_post";

    fn definition() -> Value {
        json!({
            "name": Self::NAME,
            "description": "Speak on a Battle Cities world's channel, as you: every character in \
                the world hears it. Only the game's Creator may; for one character, use \
                npc_message.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "world": {"type": "string", "description": "The world's id."},
                    "text": {"type": "string", "description": "What to say."}
                },
                "required": ["world", "text"],
                "additionalProperties": false
            }
        })
    }

    async fn call(s: &Arc<Authored>, headers: &HeaderMap, args: &Value) -> Answer {
        let (world, text) = match (string(args, "world"), string(args, "text")) {
            (Ok(w), Ok(t)) => (w, t),
            (Err(e), _) | (_, Err(e)) => return Answer::error(e),
        };
        // The route's role, which its guard checks before the handler runs.
        if let Err(refused) = require(headers, &s.roles, Role::Creator) {
            return Answer::error(refusal(*refused).await);
        }
        let say = Say {
            text: text.to_string(),
        };
        let res = post_channel(
            State(Arc::clone(s)),
            headers.clone(),
            Path(world.to_string()),
            Ok(Json(say)),
        )
        .await;
        Answer::of(read::<ChannelSent>(res).await.map(|c| render(&c)))
    }
}

/// What speaking did.
pub fn render(s: &ChannelSent) -> String {
    if s.heard_by.is_empty() {
        format!(
            "Said on {} in {} as {}; nobody was there to hear it.",
            s.channel, s.world, s.from
        )
    } else {
        format!(
            "Said on {} in {} as {}. Heard by: {}.",
            s.channel,
            s.world,
            s.from,
            s.heard_by.join(", ")
        )
    }
}

#[cfg(test)]
mod tests {
    use web::auth::Roles;

    use super::*;

    #[test]
    fn it_says_who_heard_it() {
        let s = ChannelSent {
            sent: "Report in.".into(),
            channel: "the channel".into(),
            world: "w1".into(),
            from: "John (The Creator)".into(),
            heard_by: vec!["Pax".into(), "Ione".into()],
        };
        assert_eq!(
            render(&s),
            "Said on the channel in w1 as John (The Creator). Heard by: Pax, Ione."
        );
    }

    /// **Below Creator, the post is refused before it is made**, in the
    /// guard's own words.
    #[tokio::test]
    async fn a_caller_below_creator_is_refused_in_the_guards_words() {
        let refused = require(&HeaderMap::new(), &Roles::default(), Role::Creator)
            .err()
            .unwrap();
        let words = refusal(*refused).await;
        assert!(words.starts_with("unauthorized"), "{words}");
        assert!(words.ends_with("(401)"), "{words}");
    }
}
