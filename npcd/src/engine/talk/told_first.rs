//! Telling the one who asked before looking (`docs/npc_conversations.md` §8).
//!
//! **What a mind tells someone it has not looked at is invented.** Asked which
//! lights were on, Zen's first act — before it had connected to anything — was
//! to tell John "I have checked the house lights and they are all off". Its
//! personality says never to state a reading it did not take, and it did
//! anyway. So while what a person asked still stands and nothing the mind did
//! has come back toward it, a private message to that person is refused: all
//! it could say is a promise they already have from the phone, or a reading
//! never taken. Once a reading comes back, it says what the reading says.

use crate::authority::ops::ActSource;
use crate::engine::act::Act;
use crate::engine::runtime::Runtime;

/// What the mind is told of a message it tried before looking.
pub const LOOK_FIRST: &str = "Not yet: you have looked at nothing toward what they asked, so \
     all you could tell them is a promise they already have, or a reading you never took. \
     Look first — connect to the terminal or call the act that answers it — and tell them \
     what it says.";

/// Whether `act` is a private message to `asker`, who is still owed an
/// answer nothing has been looked at for: if so, what refuses it.
pub fn refuses(act: &Act, asker: Option<&str>) -> Option<&'static str> {
    let asker = asker?;
    let arg = |k: &str| act.args.get(k).and_then(|v| v.as_str()).map(str::trim);
    let private = act.tool == "message" && arg("via") == Some("private");
    (private && arg("to").is_some_and(|to| to.eq_ignore_ascii_case(asker))).then_some(LOOK_FIRST)
}

impl Runtime {
    /// [`refuses`], for an act the mind itself took.
    pub(crate) fn told_before_looking(
        &self,
        npc: u64,
        act: &Act,
        source: &ActSource,
    ) -> Option<String> {
        if !matches!(source, ActSource::Mind) {
            return None;
        }
        let asker = self.scheduler.asked_and_unworked(npc, self.world_ms(npc));
        refuses(act, asker.as_deref()).map(str::to_string)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn message(via: &str, to: &str) -> Act {
        Act {
            tool: "message",
            args: json!({"via": via, "to": to, "intent": "the lights are all off"})
                .as_object()
                .unwrap()
                .clone(),
        }
    }

    /// **A private word to the one still waiting, before anything was looked
    /// at, is refused**; to anyone else, on a channel, or once a reading came
    /// back, it is not.
    #[test]
    fn telling_the_asker_before_looking_is_refused() {
        let john = "Johnathan Sharratt (The Creator)";
        assert_eq!(
            refuses(&message("private", john), Some(john)),
            Some(LOOK_FIRST)
        );
        assert_eq!(
            refuses(
                &message("private", " johnathan sharratt (the creator) "),
                Some(john)
            ),
            Some(LOOK_FIRST)
        );
        assert_eq!(refuses(&message("private", "Yen"), Some(john)), None);
        assert_eq!(refuses(&message("channel", "#general"), Some(john)), None);
        assert_eq!(refuses(&message("private", john), None), None);
        let read = Act {
            tool: "cluster_status",
            args: Default::default(),
        };
        assert_eq!(refuses(&read, Some(john)), None);
    }
}
