//! The critical-step gate (`docs/mission_templates.md` §6).
//!
//! Autonomy is the default: a drone carries a mission to the end on its own.
//! The one brake is a step flagged `critical` — an irreversible move (delete
//! data, wipe a substrate, force-push, drop a device's pairing). The engine
//! never signs a critical step off from what it sees and the carrier may not
//! tick it itself (`engine::mission`), so it holds the mission until an
//! **authorizer** approves — the Creator or Yen, reached on Discord. It holds
//! *safe*: no reply, or a refusal, leaves the step undone and the mission
//! stalled at it, never done without leave.
//!
//! This module is the pure half: who is an authorizer, whether a reply grants
//! the go-ahead, and what the compass tells a carrier stopped at a critical
//! step. The engine calls [`authorize_on_message`] when a message reaches a
//! carrier that holds one, with the world's authorizer list.

use crate::engine::mission::Mission;

/// Whether `from` is one of the world's authorizers, by the name the world
/// writes for them. Case- and space-insensitive; empty list means nobody can
/// approve, so a critical step can only be held.
pub fn is_authorizer(from: &str, authorizers: &[String]) -> bool {
    let from = from.trim();
    authorizers
        .iter()
        .any(|a| a.trim().eq_ignore_ascii_case(from))
}

/// Whether a reply grants the go-ahead — a forgiving read of a family member's
/// Discord reply: an affirmative word, and no refusal. A reply that says both
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

/// Apply an authorizer's reply to a carrier's mission. When the carrier's next
/// step is critical, `from` is an authorizer and the reply grants it, the step
/// is signed off and its text returned; otherwise the mission is left held at
/// the step and `None` is returned (a refusal, a stranger, or a step that is
/// not critical).
pub fn authorize_on_message(
    mission: &mut Mission,
    from: &str,
    text: &str,
    authorizers: &[String],
) -> Option<String> {
    let step = mission.critical_next()?.to_string();
    (is_authorizer(from, authorizers) && is_affirmative(text) && mission.authorize())
        .then_some(step)
}

/// The compass line for a carrier stopped at a critical step: it may not do the
/// thing until an authorizer approves, so it reaches out and waits.
pub fn critical_compass(step: &str, authorizers: &[String]) -> String {
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
fn name_them(authorizers: &[String]) -> String {
    match authorizers {
        [] => "an authorizer".to_string(),
        [one] => one.clone(),
        [most @ .., last] => format!("{} or {last}", most.join(", ")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::mission::{Mission, Origin, Todo};

    fn authorizers() -> Vec<String> {
        vec!["the Creator".to_string(), "Yen".to_string()]
    }

    #[test]
    fn an_authorizer_is_matched_by_name_case_insensitively() {
        let a = authorizers();
        assert!(is_authorizer("Yen", &a));
        assert!(is_authorizer("  the creator ", &a));
        assert!(!is_authorizer("Wick", &a));
        assert!(!is_authorizer("Yen", &[]), "nobody approves with no list");
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

    fn mission_with_critical() -> Mission {
        let todo = vec![
            Todo::new("check what is safe to remove"),
            Todo::critical("delete the old logs"),
            Todo::report("go back to the table and report it"),
        ];
        let mut m = Mission::new(
            "Clear the old logs.",
            todo,
            Origin::Random {
                routine: "clear".into(),
            },
        );
        // The ordinary first step is signed off, so the critical one is next.
        m.check_off(
            "check what is safe to remove",
            crate::engine::mission::StepOutcome::Achieved,
        );
        m
    }

    #[test]
    fn an_affirmative_from_an_authorizer_signs_off_the_critical_step() {
        let a = authorizers();
        let mut m = mission_with_critical();
        assert_eq!(m.critical_next(), Some("delete the old logs"));
        // A stranger's yes does nothing.
        assert_eq!(authorize_on_message(&mut m, "Wick", "yes", &a), None);
        assert_eq!(m.critical_next(), Some("delete the old logs"), "still held");
        // An authorizer's refusal does nothing.
        assert_eq!(
            authorize_on_message(&mut m, "Yen", "no, hold off", &a),
            None
        );
        assert_eq!(m.critical_next(), Some("delete the old logs"), "still held");
        // An authorizer's yes signs it off.
        assert_eq!(
            authorize_on_message(&mut m, "the Creator", "yes, go ahead", &a),
            Some("delete the old logs".to_string())
        );
        assert_eq!(m.critical_next(), None, "no longer held");
        assert_eq!(
            m.next_step().map(|t| t.text.as_str()),
            Some("go back to the table and report it"),
            "only the report remains"
        );
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
