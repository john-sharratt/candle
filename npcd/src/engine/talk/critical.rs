//! Critical steps through a conversation (`docs/npc_conversations.md` §7.2):
//! a request, then a yes.
//!
//! A conversation signs off a critical step only when both hold:
//!
//! 1. **It asked.** A purpose reached the conversation with an authorizer
//!    while the step was the mission's next, and the conversation recorded it
//!    as `talk.pending_critical`.
//! 2. **The authorizer answered yes.** Their next message is affirmative, and
//!    the step is still the next — re-checked on the authority.
//!
//! A "yes" to "want tea?" approves nothing.

use crate::engine::authorize::is_affirmative;
use crate::engine::talk::authority::Standing;
use crate::engine::talk::clock::Input;

/// The step this turn asks the party to sign off: the mission's next
/// critical step, when the turn began from a purpose and the party may sign.
pub fn asks(standing: Standing, next_critical: Option<&str>, input: Input) -> Option<String> {
    match (standing, input) {
        (Standing::Owner | Standing::Authorizer, Input::Purpose) => {
            next_critical.map(str::to_string)
        }
        _ => None,
    }
}

/// The step the party's lines sign off: the one asked for, when it is still
/// the next critical step and one of their lines says yes.
pub fn signs(
    pending: Option<&str>,
    next_critical: Option<&str>,
    standing: Standing,
    lines: &[&str],
) -> Option<String> {
    let step = pending?;
    let may = matches!(standing, Standing::Owner | Standing::Authorizer);
    (may && next_critical == Some(step) && lines.iter().any(|l| is_affirmative(l)))
        .then(|| step.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const LEASE: &str = "Sign the lease";

    #[test]
    fn a_purpose_to_an_authorizer_asks_for_the_next_critical_step() {
        assert_eq!(
            asks(Standing::Authorizer, Some(LEASE), Input::Purpose).as_deref(),
            Some(LEASE)
        );
        assert_eq!(asks(Standing::Authorizer, Some(LEASE), Input::Line), None);
        assert_eq!(asks(Standing::Nobody, Some(LEASE), Input::Purpose), None);
        assert_eq!(asks(Standing::Authorizer, None, Input::Purpose), None);
    }

    #[test]
    fn only_a_yes_to_what_was_asked_signs_it() {
        let yes = ["Yes, go ahead."];
        assert_eq!(
            signs(Some(LEASE), Some(LEASE), Standing::Authorizer, &yes).as_deref(),
            Some(LEASE)
        );
        assert_eq!(
            signs(None, Some(LEASE), Standing::Authorizer, &yes),
            None,
            "a yes to a question nobody asked"
        );
        assert_eq!(
            signs(
                Some(LEASE),
                Some("Pay the deposit"),
                Standing::Authorizer,
                &yes
            ),
            None,
            "the step moved on"
        );
        assert_eq!(
            signs(
                Some(LEASE),
                Some(LEASE),
                Standing::Authorizer,
                &["No, wait."]
            ),
            None
        );
        assert_eq!(
            signs(Some(LEASE), Some(LEASE), Standing::Nobody, &yes),
            None
        );
    }
}
