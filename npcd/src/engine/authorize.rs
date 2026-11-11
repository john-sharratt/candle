//! The critical-step gate (`docs/mission_templates.md` §6).
//!
//! Autonomy is the default: a drone carries a mission to the end on its own.
//! The one brake is a step flagged `critical` — an irreversible move (delete
//! data, wipe a substrate, force-push, drop a device's pairing). The engine
//! never signs a critical step off from what it sees and the carrier may not
//! tick it itself (`engine::mission`), so it holds the mission until an
//! **authorizer** approves — the Creator or Yen, reached on the phone. It holds
//! *safe*: no reply, or a refusal, leaves the step undone and the mission
//! stalled at it, never done without leave.
//!
//! This module is the pure half: who is an authorizer, whether a reply grants
//! the go-ahead, and what the compass tells a carrier stopped at a critical
//! step. A step is signed only through a conversation that asked for it
//! (`docs/npc_conversations.md` §7.2, `engine::talk::critical`).

use serde::Deserialize;

/// One who may approve a critical step: known by the public handle of their
/// account (`u_…`), never by a name — an identity provider can change a name,
/// and two people can share one (`docs/npc_conversations.md` §7.2). The name
/// is what the compass calls them.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Authorizer {
    pub account: String,
    pub name: String,
}

/// Whether the account with handle `account` is one of the world's
/// authorizers. An empty list means nobody can approve, so a critical step
/// can only be held.
pub fn is_authorizer(account: &str, authorizers: &[Authorizer]) -> bool {
    let account = account.trim();
    !account.is_empty() && authorizers.iter().any(|a| a.account.trim() == account)
}

/// Whether a reply grants the go-ahead — a forgiving read of a family member's
/// reply: an affirmative word, and no refusal. A reply that says both
/// ("yes, but wait") is treated as withheld, because a critical step is held
/// safe.
pub fn is_affirmative(text: &str) -> bool {
    let t = text.to_lowercase();
    !is_refusal(&t) && AFFIRMATIVE.iter().any(|w| has_word(&t, w))
}

/// Whether a reply withholds the go-ahead.
pub fn is_refusal(text: &str) -> bool {
    let t = text.to_lowercase();
    REFUSAL.iter().any(|w| has_word(&t, w))
}

/// Words that grant it.
const AFFIRMATIVE: &[&str] = &[
    "yes",
    "yep",
    "yeah",
    "yup",
    "ok",
    "okay",
    "sure",
    "go ahead",
    "go for it",
    "do it",
    "approved",
    "approve",
    "permitted",
    "allowed",
    "confirmed",
    "confirm",
    "granted",
];

/// Words that withhold it. Checked first, so any of these overrides an
/// affirmative in the same reply.
const REFUSAL: &[&str] = &[
    "no", "nope", "nah", "don't", "do not", "dont", "stop", "wait", "hold off", "cancel", "deny",
    "denied", "not yet", "never", "refuse", "refused",
];

/// Whether `text` holds `word`. A multi-word or punctuated token ("go ahead",
/// "don't") is matched as a substring; a plain word is matched whole, so "no"
/// does not fire on "now" and "ok" does not fire on "poke".
fn has_word(text: &str, word: &str) -> bool {
    if !word.chars().all(|c| c.is_ascii_alphanumeric()) {
        return text.contains(word);
    }
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .any(|tok| tok == word)
}

/// The compass line for a carrier stopped at a critical step: it may not do the
/// thing until an authorizer approves, so it reaches out and waits.
pub fn critical_compass(step: &str, authorizers: &[Authorizer]) -> String {
    let who = name_them(authorizers);
    let step = step.trim().trim_end_matches('.');
    format!(
        "Your mission, next: {step}. This cannot be undone, so it needs a go-ahead first: \
         `message` {who} saying plainly what you mean to do and why, and wait for the answer. \
         Do it only once they say yes; if they say no, or no one answers, leave it and report \
         that it is waiting on authorization."
    )
}

/// The authorizers named for the compass: "the Creator", "the Creator or Yen",
/// "A, B or C"; "an authorizer" when none is set.
fn name_them(authorizers: &[Authorizer]) -> String {
    let names: Vec<&str> = authorizers.iter().map(|a| a.name.as_str()).collect();
    match names.as_slice() {
        [] => "an authorizer".to_string(),
        [one] => one.to_string(),
        [most @ .., last] => format!("{} or {last}", most.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn authorizers() -> Vec<Authorizer> {
        vec![
            Authorizer {
                account: "u_creator1".into(),
                name: "the Creator".into(),
            },
            Authorizer {
                account: "u_yen00002".into(),
                name: "Yen".into(),
            },
        ]
    }

    #[test]
    fn an_authorizer_is_known_by_account_and_never_by_name() {
        let a = authorizers();
        assert!(is_authorizer("u_yen00002", &a));
        assert!(is_authorizer(" u_creator1 ", &a));
        assert!(!is_authorizer("Yen", &a), "a name is not an account");
        assert!(!is_authorizer("u_wick0003", &a));
        assert!(!is_authorizer("", &a));
        assert!(
            !is_authorizer("u_yen00002", &[]),
            "nobody approves with no list"
        );
    }

    #[test]
    fn an_authorizer_reads_from_its_file_by_account_and_name() {
        let a: Vec<Authorizer> =
            serde_yaml::from_str("- account: u_creator1\n  name: the Creator\n").unwrap();
        assert_eq!(a, authorizers()[..1]);
        assert!(serde_yaml::from_str::<Vec<Authorizer>>("- the Creator\n").is_err());
    }

    #[test]
    fn a_reply_is_read_for_a_yes_not_a_no() {
        assert!(is_affirmative("yes, go ahead"));
        assert!(is_affirmative("ok do it"));
        assert!(is_affirmative("Approved."));
        assert!(!is_affirmative("no"));
        assert!(!is_affirmative("not yet"));
        assert!(!is_affirmative("yes but wait"), "a refusal overrides");
        assert!(!is_affirmative("hmm"), "no affirmative word");
        // Whole-word matching, not substrings.
        assert!(!is_refusal("now is good"), "`no` is not in `now`");
        assert!(is_affirmative("now is good, do it"));
    }

    #[test]
    fn the_compass_names_the_authorizers() {
        assert_eq!(
            critical_compass("delete the old logs", &authorizers()),
            "Your mission, next: delete the old logs. This cannot be undone, so it needs a \
             go-ahead first: `message` the Creator or Yen saying plainly what you mean to do and \
             why, and wait for the answer. Do it only once they say yes; if they say no, or no \
             one answers, leave it and report that it is waiting on authorization."
        );
    }
}
