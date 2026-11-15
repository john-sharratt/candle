//! A hand-off: what a conversation's decision sends to somebody **other** than
//! the one it is talking with (`docs/npc_conversations.md` §6.2, §10).
//!
//! **It is not the answer, and it does not end the decision.** Offered under
//! the main loop's name, `message`, a decision answering a person chose to
//! message somebody else as its whole turn: with no `reply` among its acts the
//! words went to the journal and the person was answered with nothing. So in a
//! conversation the act has a name of its own that says who it is for, and once
//! it is chosen the decision is asked again with it struck, so the turn goes on
//! to what is said to the party.

use crate::engine::act::Act;
use crate::engine::acts::MESSAGE;
use crate::engine::tools::{Availability, Example, Plane, Tool, Within};

/// Message somebody other than the party, from a conversation.
pub const MESSAGE_SOMEONE_ELSE: Tool = Tool {
    name: "message_someone_else",
    at: &[],
    category: "Messaging",
    plane: Plane::Speech,
    availability: Availability::OnThread,
    description: "Send something to somebody OTHER than the one you are talking with — another \
                  person on your phone, or one of your channels. Say first how it goes — \
                  `private`, to one person alone, or `channel`, to one of your channels or \
                  groups — and then which one. This is NOT your answer to the one you are \
                  talking with: once it is sent you still decide what to say to them.",
    params: MESSAGE.params,
    examples: &[
        Example {
            situation: "Mira asks you whether the light ring is fixed, and Wick is the one \
                        working on it.",
            call: r#"{"via":"private","to":"Wick","intent":"Mira is asking whether the ring is fixed — where are you with it?"}"#,
            because: "Wick has the answer, not you; asking him is a hand-off, and once it is sent \
                      you still tell Mira that you have asked.",
        },
        Example {
            situation: "Somebody on your phone tells you something everyone on the channel needs \
                        to know before they act.",
            call: r#"{"via":"channel","to":"the channel","intent":"the lounge is closed until the morning"}"#,
            because: "The channel reaches everyone at once; the one you are talking with is still \
                      answered after it.",
        },
    ],
};

/// The conversation's hand-offs: acts that go to somebody else and leave the
/// decision of what to say to the party still to make.
pub const HAND_OFFS: &[&str] = &[MESSAGE_SOMEONE_ELSE.name, "reach_out"];

/// The conversation's own acts, for the catalog.
pub const HAND_OFF_ACTS: &[Tool] = &[MESSAGE_SOMEONE_ELSE];

/// `within` with `names` struck from what the conversation may take: what the
/// decision is asked again with once it has handed those off.
pub fn struck(within: &Within, names: &[&str]) -> Within {
    let mut left = within.clone();
    if let Some(allow) = left.allow.as_mut() {
        allow.retain(|a| !names.contains(&a.as_str()));
    }
    left
}

/// `act` as the world performs it: a message to somebody else is a `message`.
pub fn as_performed(act: &Act) -> Act {
    match act.tool == MESSAGE_SOMEONE_ELSE.name {
        true => Act {
            tool: MESSAGE.name,
            args: act.args.clone(),
        },
        false => act.clone(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// **A message to somebody else is performed as the world's `message`**,
    /// with every argument as given; any other act is performed as it is.
    #[test]
    fn a_message_to_someone_else_is_performed_as_a_message() {
        let args = json!({"via": "private", "to": "Wick", "intent": "check the ring"});
        let chosen = Act {
            tool: "message_someone_else",
            args: args.as_object().unwrap().clone(),
        };
        let done = as_performed(&chosen);
        assert_eq!(done.tool, "message");
        assert_eq!(done.args, chosen.args);
        let reply = Act {
            tool: "reply",
            args: json!({"to": "Mira"}).as_object().unwrap().clone(),
        };
        assert_eq!(as_performed(&reply).tool, "reply");
    }

    /// **What was handed off is struck, and nothing else.**
    #[test]
    fn a_hand_off_is_struck_from_what_is_left() {
        let within = Within {
            allow: Some(
                ["reply", "message_someone_else", "reach_out"]
                    .map(String::from)
                    .to_vec(),
            ),
            ..Within::default()
        };
        assert_eq!(
            struck(&within, &["message_someone_else"]).allow,
            Some(vec!["reply".to_string(), "reach_out".to_string()])
        );
    }
}
