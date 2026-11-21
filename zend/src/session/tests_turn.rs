//! Tests how a held tool call is rendered, how the round cap closes the loop,
//! and how long a turn may think.
//!
//! A bare tool-call object is wrapped in tags so the client renders a card,
//! while a bare object that is not a registered call streams as plain text. The
//! round cap answers the last answerable round with a closing turn instead of
//! stopping the loop. A turn can think as long as any dial allows and still
//! write one whole string value, so the turn's own cap never cuts a large file.

use super::*;

#[cfg(test)]
pub(super) mod held_tail_tests {
    use super::render_held_tail;

    #[test]
    fn wraps_a_bare_tool_call_so_the_gui_renders_a_card() {
        // The 584 shape: bare object, `parameters` key — the GUI needs the wrapper.
        let held = r#"{"name":"datetime","parameters":{"timezone":"Australia/Sydney"}}"#;
        let out = render_held_tail(held).expect("held tail");
        assert!(out.starts_with("<tool_call>"), "not wrapped: {out}");
        assert!(
            out.trim_end().ends_with("</tool_call>"),
            "not closed: {out}"
        );
        assert!(out.contains("Australia/Sydney"), "lost the args: {out}");
    }

    #[test]
    fn leaves_a_non_tool_json_object_verbatim() {
        // A bare object that isn't a registered call must stream as plain text.
        let held = r#"{"answer": 42, "note": "not a tool"}"#;
        let out = render_held_tail(held).expect("held tail");
        assert!(!out.contains("<tool_call>"), "wrongly wrapped: {out}");
        assert_eq!(out, held);
    }

    #[test]
    fn empty_tail_emits_nothing() {
        assert!(render_held_tail("   \n  ").is_none());
    }
}

#[cfg(test)]
pub(super) mod round_cap_tests {
    use super::{closes_at_cap, MAX_TOOL_ROUNDS};

    /// **The round cap closes the loop rather than stopping it**: the results
    /// of the round before the cap are answered by a closing turn, which still
    /// runs under the cap, and no earlier round's are.
    #[test]
    fn the_last_answerable_round_is_closed() {
        assert!(!closes_at_cap(0));
        assert!(!closes_at_cap(MAX_TOOL_ROUNDS - 3));
        assert!(closes_at_cap(MAX_TOOL_ROUNDS - 2));
        let closing_turn = MAX_TOOL_ROUNDS - 2 + 1;
        assert!(closing_turn < MAX_TOOL_ROUNDS, "the closing turn runs");
    }
}

#[cfg(test)]
pub(super) mod turn_cap_tests {
    use super::MAX_TURN_TOKENS;
    use candle_conversation::stencil::{ThinkMode, MAX_STRING_VALUE_TOKENS};

    /// **A turn can think as long as any dial allows and still write one whole
    /// string value.** Otherwise the turn's own cap, not the value's, is what
    /// cuts a large file — which is the failure this cap exists to remove.
    #[test]
    fn a_turn_holds_the_longest_think_block_and_a_full_string_value() {
        let longest_think = [
            ThinkMode::Quick,
            ThinkMode::Balanced,
            ThinkMode::Deep,
            ThinkMode::Exhaustive,
        ]
        .into_iter()
        .map(|m| m.span_cap() as usize)
        .max()
        .unwrap();
        assert!(
            MAX_TURN_TOKENS > longest_think + MAX_STRING_VALUE_TOKENS as usize,
            "{MAX_TURN_TOKENS} cannot hold {longest_think} thinking + {MAX_STRING_VALUE_TOKENS} of value",
        );
    }
}
