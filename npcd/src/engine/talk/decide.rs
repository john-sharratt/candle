//! What a conversation's decision means (`docs/npc_conversations.md` §6.2):
//! the destination its words go to, and the decision restated to the turn
//! that says them.
//!
//! **The destination is read after the parse.** The loop cannot make one call
//! terminal, so the last of `reply`, `sign_off` or `reflect` in the list
//! decides; a list with none of them goes to the journal.

use crate::engine::act::Act;
use crate::engine::talk::hand_off::HAND_OFFS;
use crate::engine::talk::templates::Destination;

/// The hand-offs `acts` made when they say nothing to the party: none of
/// `reply`, `sign_off` or `reflect`, and at least one act to somebody else
/// (`hand_off`). `None` when the decision answered, or handed nothing off.
///
/// **A hand-off does not end the decision.** Asked again with these struck,
/// the turn goes on to what is said to the party.
pub fn handed_off_only(acts: &[Act]) -> Option<Vec<&'static str>> {
    let answers = acts
        .iter()
        .any(|a| matches!(a.tool, "reply" | "sign_off" | "reflect"));
    let handed: Vec<&'static str> = acts
        .iter()
        .map(|a| a.tool)
        .filter(|t| HAND_OFFS.contains(t))
        .collect();
    (!answers && !handed.is_empty()).then_some(handed)
}

/// What the decision is told when it is asked again after handing off: what it
/// already sent, and that the party still waits on it.
pub fn after_hand_off(party: &str, sent: &[Act]) -> String {
    let what: Vec<String> = sent
        .iter()
        .filter(|a| HAND_OFFS.contains(&a.tool))
        .map(|a| {
            let arg = |k: &str| {
                a.args
                    .get(k)
                    .and_then(|v| v.as_str())
                    .map(str::trim)
                    .unwrap_or_default()
                    .trim_end_matches('.')
                    .to_string()
            };
            format!("to {}: {}", arg("to"), arg("intent"))
        })
        .collect();
    format!(
        "You have sent {}. {party} is still waiting on you: decide what you say to them.",
        what.join("; ")
    )
}

/// Where a decision's words go.
pub fn destination(acts: &[Act]) -> Destination {
    acts.iter()
        .rev()
        .find_map(|a| match a.tool {
            "reply" => Some(Destination::Reply),
            "sign_off" => Some(Destination::Parting),
            "reflect" => Some(Destination::Journal),
            _ => None,
        })
        .unwrap_or(Destination::Journal)
}

/// The decision, restated at the end of the turn that says the words, so the
/// words cannot contradict it (§6.3).
pub fn restated(party: &str, acts: &[Act]) -> String {
    let text = |a: &Act, k: &str| {
        a.args
            .get(k)
            .and_then(|v| v.as_str())
            .map(str::trim)
            .unwrap_or_default()
            .trim_end_matches('.')
            .to_string()
    };
    let mut out = match destination(acts) {
        Destination::Reply => {
            let meant = acts
                .iter()
                .rev()
                .find(|a| a.tool == "reply")
                .map(|a| text(a, "intent"))
                .unwrap_or_default();
            format!("You mean to tell {party}: {meant}.")
        }
        Destination::Parting => {
            let meant = acts
                .iter()
                .rev()
                .find(|a| a.tool == "sign_off")
                .map(|a| text(a, "intent"))
                .unwrap_or_default();
            format!("You are saying goodbye to {party}: {meant}.")
        }
        Destination::Journal | Destination::Channel => {
            "Write down what you make of this; you are not answering.".to_string()
        }
    };
    for a in acts.iter().filter(|a| a.tool == "promise") {
        out.push_str(&format!(
            " You promised: {}, by {}.",
            text(a, "what"),
            text(a, "by")
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use std::slice;

    use serde_json::json;

    use super::*;

    fn act(tool: &'static str, args: serde_json::Value) -> Act {
        Act {
            tool,
            args: args.as_object().unwrap().clone(),
        }
    }

    #[test]
    fn the_last_reply_goodbye_or_reflection_decides_where_the_words_go() {
        let reply = act(
            "reply",
            json!({"to": "Mira", "intent": "it is at the door"}),
        );
        let bye = act("sign_off", json!({"to": "Mira", "intent": "I'm off shift"}));
        let reflect = act("reflect", json!({"feeling": "calm"}));
        let promise = act(
            "promise",
            json!({"to": "Mira", "what": "the garden report", "by": "Friday"}),
        );
        assert_eq!(destination(slice::from_ref(&reply)), Destination::Reply);
        assert_eq!(
            destination(&[reply.clone(), bye.clone()]),
            Destination::Parting
        );
        assert_eq!(destination(&[bye, reflect.clone()]), Destination::Journal);
        assert_eq!(destination(slice::from_ref(&promise)), Destination::Journal);
        assert_eq!(destination(&[]), Destination::Journal);
        assert_eq!(destination(&[promise, reply]), Destination::Reply);
    }

    /// **Handing off without an answer asks again**; an answer beside a
    /// hand-off, or no hand-off at all, does not.
    #[test]
    fn a_hand_off_alone_leaves_the_answer_to_decide() {
        let sent = act(
            "message_someone_else",
            json!({"via": "private", "to": "Wick", "intent": "check the ring."}),
        );
        let reply = act("reply", json!({"to": "Mira", "intent": "it is done"}));
        let promise = act(
            "promise",
            json!({"to": "Mira", "what": "the report", "by": "Friday"}),
        );
        assert_eq!(
            handed_off_only(slice::from_ref(&sent)),
            Some(vec!["message_someone_else"])
        );
        assert_eq!(handed_off_only(&[sent.clone(), reply]), None);
        assert_eq!(handed_off_only(slice::from_ref(&promise)), None);
        assert_eq!(
            after_hand_off("Mira", &[sent, promise]),
            "You have sent to Wick: check the ring. Mira is still waiting on you: decide what \
             you say to them."
        );
    }

    #[test]
    fn the_decision_is_restated_to_the_turn_that_says_it() {
        let acts = [
            act(
                "promise",
                json!({"to": "Mira", "what": "the garden report.", "by": "Friday"}),
            ),
            act(
                "reply",
                json!({"to": "Mira", "intent": "that it is at the door."}),
            ),
        ];
        assert_eq!(
            restated("Mira", &acts),
            "You mean to tell Mira: that it is at the door. You promised: the garden report, \
             by Friday."
        );
        assert_eq!(
            restated("Mira", &[act("reflect", json!({"feeling": "calm"}))]),
            "Write down what you make of this; you are not answering."
        );
        assert_eq!(
            restated(
                "Mira",
                &[act(
                    "sign_off",
                    json!({"to": "Mira", "intent": "time for bed"})
                )]
            ),
            "You are saying goodbye to Mira: time for bed."
        );
    }
}
