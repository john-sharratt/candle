//! Tests that the reflection line passes the committed example from the design
//! document, and is caught for paragraphs, explanation, and statements of fact.
//!
//! The worked example from the design document must pass, and each failure mode
//! measured in a live run must fail. A single sentence ending in a full stop
//! counts as one sentence, so the full stop does not register as a second. Unframed
//! lines, which once came back as the character's own conclusions, are caught, and
//! the framed version of each is accepted.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// §6 fixes the line that crosses back at one sentence. The worked example
    /// from the document itself has to pass.
    #[test]
    fn the_committed_line_from_the_design_document_passes() {
        assert_eq!(reflection_fault("I'm thinking of challenging him."), None);
        assert_eq!(
            reflection_fault("I keep wanting to go back and look at it"),
            None
        );
    }

    /// **The failure measured live: four paragraphs where a line was asked for.**
    #[test]
    fn a_reflection_that_runs_to_paragraphs_is_caught() {
        let f = reflection_fault(
            "The silence in the quiet room is heavy. I'm still sitting at my desk. My head is \
             spinning with that thought.",
        );
        assert!(f.unwrap().contains("3 sentences"));
    }

    /// §6: *"A response that starts explaining itself is a regression, and
    /// nothing else would flag it."* One sentence is not enough on its own — a
    /// single sentence can still be an argument.
    #[test]
    fn a_reflection_that_explains_itself_is_caught() {
        let f = reflection_fault(
            "I want to go back to the east ridge because I need to see if what I made holds up",
        );
        assert!(f.unwrap().contains("explains itself"));
    }

    /// A trailing full stop ends the one sentence; it does not begin a second.
    #[test]
    fn a_single_terminated_sentence_is_one_sentence() {
        assert_eq!(
            reflection_fault("I feel like I am not going to say anything about it."),
            None
        );
    }

    /// **A reflection says how it is for the character, never what is so.**
    /// The unframed lines are the live ones: read back as the character's own
    /// conclusions, each told it what it was.
    #[test]
    fn a_reflection_stated_as_fact_is_caught_and_a_framed_one_is_not() {
        for fact in [
            "I am the room itself, and the silence is simply my own breath.",
            "I am the vibration in the pipe, the quiet hum beneath the silence.",
            "I am here, and that is enough.",
            "The hum vibrates in my teeth.",
        ] {
            let f = reflection_fault(fact);
            assert!(
                f.as_deref().is_some_and(|w| w.contains("as a fact")),
                "{fact} → {f:?}"
            );
        }
        for framed in [
            "I feel like I am the room itself.",
            "It seems as if the silence belongs to the room and not to me.",
            "I find myself wanting to stop holding my breath.",
            "The hum feels like it is inside my teeth.",
            "I\u{2019}m thinking of challenging him.",
        ] {
            assert_eq!(reflection_fault(framed), None, "{framed}");
        }
    }

    /// A character's first reflection has no corpus to be unlike, and a lead-in
    /// standing over an empty list reads as a list it failed to remember.
    #[test]
    fn no_prior_axes_means_no_steer_at_all() {
        assert_eq!(steer(&[]), "");
        let s = steer(&["that the ground bears weight".to_string()]);
        assert!(s.starts_with("Make it unlike these"));
        assert!(s.contains("  - that the ground bears weight"));
    }

    /// Coverage is structural, so the space has to be a real spread rather than
    /// eight ways of saying "your job".
    #[test]
    fn the_domains_are_distinct_and_nonempty() {
        assert!(DOMAINS.len() >= 6);
        for d in DOMAINS {
            assert!(!d.trim().is_empty());
        }
        let heads: std::collections::HashSet<&str> = DOMAINS
            .iter()
            .map(|d| d.split(" — ").next().unwrap())
            .collect();
        assert_eq!(heads.len(), DOMAINS.len(), "two domains share a head");
    }
}
