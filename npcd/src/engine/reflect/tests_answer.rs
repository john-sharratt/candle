//! Tests that a reflection's answer is read from either call shape, repaired when
//! it contains raw newlines, and unwrapped when the model put it in a call.
//!
//! Each case reproduces a failure seen in a live run. An answer written in the
//! other call shape was found empty, a newline inside a JSON brief failed the whole
//! call, and a paragraph wrapped in a tool call was discarded. Plain prose is left
//! unchanged, so a bracket in ordinary text is not taken for markup.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// **An answer reads the same from either call shape.** The dialect decides
    /// whether the grammar forces an element or a JSON block, and a reader that
    /// knew only one found nothing in every answer written in the other.
    #[test]
    fn an_answer_is_read_from_either_call_shape() {
        let element = "<tool_call>\n<function=dream>\n<parameter=assumption>\n\
                       that the floor holds\n</parameter>\n<parameter=brief>\n\
                       You walk \"out\".\nThen down.\n</parameter>\n</function>\n</tool_call>";
        let json = "<tool_call>\n{\"name\": \"dream\", \"arguments\": {\"assumption\": \
                    \"that the floor holds\", \"brief\": \"You walk \\\"out\\\".\\nThen \
                    down.\"}}\n</tool_call>";
        for answer in [element, json] {
            assert_eq!(
                field(answer, "assumption").as_deref(),
                Some("that the floor holds"),
                "{answer}"
            );
            assert_eq!(
                field(answer, "brief").as_deref(),
                Some("You walk \"out\".\nThen down."),
                "{answer}"
            );
            assert_eq!(field(answer, "missing"), None, "{answer}");
        }
    }

    /// **A raw newline inside a JSON value does not cost the answer.** The
    /// string span lets one through, and a brief is prose that runs to
    /// paragraphs — unrepaired, the whole call failed to parse and a run lost
    /// its dream on every pass, reported as having no assumption at all.
    #[test]
    fn a_raw_newline_inside_a_json_value_is_read_rather_than_lost() {
        let answer = "<tool_call>\n{\"name\": \"dream\", \"arguments\": {\"assumption\": \
                      \"that the floor holds\", \"brief\": \"You walk out.\nThen down.\"}}\n\
                      </tool_call>";
        assert_eq!(
            field(answer, "assumption").as_deref(),
            Some("that the floor holds")
        );
        assert_eq!(
            field(answer, "brief").as_deref(),
            Some("You walk out.\nThen down.")
        );
    }

    /// The close is grammar scaffolding, not something the character said.
    #[test]
    fn the_injected_think_close_is_not_part_of_the_reflection() {
        assert_eq!(
            plain_prose("</think>\n\nThe room is quiet."),
            "The room is quiet."
        );
    }

    /// **Measured, not hypothetical.** Told twice that there is nothing to call,
    /// the checkpoint wrapped a perfectly good paragraph in a call anyway. The
    /// paragraph is the answer, so it is recovered rather than discarded.
    #[test]
    fn a_reflection_wrapped_in_a_tool_call_is_unwrapped() {
        let raw = "</think>\n\n<tool_call>\n<function=reflect>\n<parameter=inner_thoughts>\n\
                   The silence here is thick. I am not second-guessing it.\n</parameter>\n\
                   </function>\n</tool_call>";
        assert_eq!(
            plain_prose(raw),
            "The silence here is thick. I am not second-guessing it."
        );
    }

    /// The ordinary case must survive the salvage untouched — a reflection that
    /// merely mentions a bracket is not markup.
    #[test]
    fn plain_prose_is_left_alone() {
        let p = "I keep coming back to how fast it was. Under a minute, and I did not check.";
        assert_eq!(plain_prose(p), p);
    }

    /// A `dream` call as the answer grammar emits one.
    ///
    /// Every check below reads a brief out of one of these rather than out of
    /// prose, because prose is no longer a shape this conversation can produce:
    /// the two fields are named parameters of a call the tree forces. The three
    /// tests that used to live here — that the labels split apart, that a
    /// wrapped brief rejoins, that prose faults instead of failing — were all
    /// about a parser that no longer exists, guarding failures the grammar has
    /// made unreachable.
    pub(in super::super) fn answered(assumption: &str, brief: &str) -> Brief {
        Brief::from_call(&format!(
            "<tool_call>\n<function=dream>\n<parameter=assumption>\n{assumption}\n</parameter>\n\
             <parameter=brief>\n{brief}\n</parameter>\n</function>\n</tool_call>"
        ))
    }

    /// The grammar names both fields, so both arrive.
    #[test]
    fn a_call_splits_into_its_two_parts() {
        let b = answered(
            "The ground bears indefinite weight.",
            "You are four levels deep when the weight enters your awareness.",
        );
        assert_eq!(
            b.assumption.as_deref(),
            Some("The ground bears indefinite weight.")
        );
        assert!(b.brief.unwrap().starts_with("You are four levels deep"));
    }
}
