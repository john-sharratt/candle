//! A mind-loop post's words, decoded on the lease holder before the act is
//! sent (`docs/npc_conversations.md` §11.4).
//!
//! A `message` on a channel, a `reply` on a group, a group opened with
//! something to say and a goodbye on a group each carry what the character
//! meant. What lands on the thread is the line it says: decoded here, where
//! the character's live slot is, and carried to the world's authority in
//! `WorldOp::Act::said`. The authority holds no slot for a character leased
//! elsewhere, so it could not decode it.

use candle_conversation::projection::DecodePriority;

use crate::authority::ops::Voiced;
use crate::engine::act::Act;
use crate::engine::loopguard::would_repeat;
use crate::engine::mind::Voicing;
use crate::engine::runtime::Runtime;
use crate::engine::talk::templates::Destination;
use crate::engine::tools::CHANNEL_WAY;

/// The most a post decodes: one Discord message.
pub const POST_TOKENS: usize = 320;

/// Where `act` puts words on a group, and what it means to say there: the
/// thread as the character named it, and the intent. `None` for an act that
/// puts no words on a group.
pub fn post_of(act: &Act) -> Option<(String, String)> {
    let text = |k: &str| {
        act.args
            .get(k)
            .and_then(|v| v.as_str())
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
    };
    let on_group = match act.tool {
        "message" => text("via").as_deref() == Some(CHANNEL_WAY),
        "reply" | "sign_off" => true,
        "open_group" => true,
        _ => false,
    };
    if !on_group {
        return None;
    }
    let to = match act.tool {
        "open_group" => text("called")?,
        _ => text("to")?,
    };
    Some((to, text("intent")?))
}

/// The user turn a post's words answer.
pub fn post_prompt(tool: &str, to: &str, intent: &str) -> String {
    match tool {
        "sign_off" => format!(
            "Say goodbye on {to}, in your own words, as you would type it: {intent}"
        ),
        _ => format!("Post to {to}, in your own words, as you would type it: {intent}"),
    }
}

/// What a mind-loop act puts on a group, once its words are decoded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Post {
    /// The act puts no words on a group.
    None,
    /// The words it says.
    Words(Voiced),
    /// The words say again what the character just said there: refused
    /// before they are posted, never posted and then struck
    /// (`docs/npc_conversations.md` §11.3).
    Repeat,
}

/// What `words` would be on a thread where the character's own last lines
/// are `own_last`, newest last.
pub fn post(own_last: &[&str], words: &str) -> Post {
    match would_repeat(own_last, words) {
        true => Post::Repeat,
        false => Post::Words(Voiced::words(words)),
    }
}

impl Runtime {
    /// The words a mind-loop act puts on a group, decoded for it and checked
    /// against what the character last said there. A post whose words could
    /// not be decoded is `Err`: it is refused rather than posted as its
    /// intent.
    pub async fn voice_post(&self, npc_id: u64, act: &Act) -> anyhow::Result<Post> {
        let Some((to, intent)) = post_of(act) else {
            return Ok(Post::None);
        };
        let minds = self
            .minds
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| anyhow::anyhow!("the engine is not loaded"))?;
        let persona = self
            .persona_of_async(npc_id)
            .await
            .ok_or_else(|| anyhow::anyhow!("npc {npc_id} has no persona"))?;
        let within = self.within(npc_id);
        let prompt = post_prompt(act.tool, &to, &intent);
        let words = minds
            .voice(
                npc_id,
                &persona.as_persona(),
                &within,
                Voicing {
                    prompt: &prompt,
                    to: Destination::Channel,
                    max_tokens: POST_TOKENS,
                    band: DecodePriority::Normal,
                },
            )
            .await?;
        let own_last: Vec<String> = self
            .body_of(npc_id)
            .and_then(|(hosted, body)| {
                hosted.sim(|s| {
                    let t = s.threads.by_name_for(&body, &to)?;
                    Some(
                        t.since_joined(&body)
                            .iter()
                            .filter(|m| m.from == body)
                            .map(|m| m.text.clone())
                            .collect(),
                    )
                })
            })
            .unwrap_or_default();
        let own: Vec<&str> = own_last.iter().map(String::as_str).collect();
        Ok(post(&own, &words))
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn act(tool: &'static str, args: serde_json::Value) -> Act {
        Act {
            tool,
            args: args.as_object().unwrap().clone(),
        }
    }

    /// **Every act that puts words on a group is voiced, and nothing else.**
    #[test]
    fn only_a_post_on_a_group_is_voiced() {
        let channel = act(
            "message",
            json!({"via": "channel", "to": "the channel", "intent": "that the lift is stuck"}),
        );
        assert_eq!(
            post_of(&channel),
            Some(("the channel".into(), "that the lift is stuck".into()))
        );
        let private = act(
            "message",
            json!({"via": "private", "to": "Mira", "intent": "that it is in"}),
        );
        assert_eq!(post_of(&private), None, "a private message is a purpose");
        let opened = act(
            "open_group",
            json!({"called": "the sweep", "with": "Wren", "intent": "who is free?"}),
        );
        assert_eq!(
            post_of(&opened),
            Some(("the sweep".into(), "who is free?".into()))
        );
        assert_eq!(
            post_of(&act("open_group", json!({"called": "the sweep", "with": "Wren"}))),
            None,
            "opened with nothing to say"
        );
        assert_eq!(
            post_of(&act("tell", json!({"to": "Wren", "intent": "hi"}))),
            None
        );
        assert_eq!(
            post_prompt("sign_off", "the sweep", "I'm off shift"),
            "Say goodbye on the sweep, in your own words, as you would type it: I'm off shift"
        );
    }

    /// **A post saying again what was just said there is refused before it
    /// is sent**; one that says something new goes out as its words.
    #[test]
    fn a_repeated_post_is_caught_before_it_is_posted() {
        let own = ["The lift on level three is stuck again, I've reported it."];
        assert_eq!(
            post(&own, "The lift on level three is stuck again; I've reported it."),
            Post::Repeat
        );
        assert_eq!(
            post(&own, "Who is free to cover the night shift?"),
            Post::Words(Voiced::words("Who is free to cover the night shift?"))
        );
        assert_eq!(post(&[], "Anyone?"), Post::Words(Voiced::words("Anyone?")));
    }
}
