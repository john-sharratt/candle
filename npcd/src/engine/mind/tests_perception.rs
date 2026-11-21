//! Tests the order, grouping and bounds of what a character reads back.
//!
//! The protocol admits no order other than answers before narration, since a
//! result answers the turn before it, so the tests pin that order along with
//! one block per call in call order. Refusals are answers too. A turn with no
//! acts carries no empty wrapper, and an answer or narration past its bound is
//! cut at a character boundary and says how much was left out, counted in
//! characters rather than bytes. Recall is counted per layer beneath the turn's
//! own, leaving out the dreams and any layer the gather passed over.

use super::*;

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    use std::slice::from_ref;

    /// **Each layer beneath the turn's own is counted, with its best score.**
    /// The turn's own layer and the dreams (logged on their own) are not, and
    /// a turn the gather passed over is not.
    #[test]
    fn what_a_turn_recalled_is_counted_by_layer() {
        let turn = |layer: &str, selected: bool, score: f32, timeline: u64| -> SelectedTurn {
            serde_json::from_value(serde_json::json!({
                "layer": layer, "group": "g", "index": 0, "role": "assistant", "tokens": 9,
                "selected": selected, "score": score, "timeline": timeline,
            }))
            .unwrap()
        };
        let turns = [
            turn("eras", true, 640.0, 1),
            turn("eras", true, 910.0, 2),
            turn("world", true, 300.0, 3),
            turn("stories", false, 990.0, 4),
            turn("interaction", true, 0.0, 5),
            turn(dreams::LAYER, true, 815.0, 6),
        ];
        let got = recalled_by_layer(&turns, "interaction");
        let recalled = |count, best, top| Recalled {
            count,
            best,
            top: Some(top),
        };
        assert_eq!(
            got.into_iter().collect::<Vec<_>>(),
            [
                ("eras".to_string(), recalled(2, 910.0, 2)),
                ("world".to_string(), recalled(1, 300.0, 3))
            ]
        );
    }

    /// **An act is answered, and the answer comes first**, then the narration.
    ///
    /// The protocol admits no other order: a result answers the turn before it,
    /// so nothing may come between a call and its response. Measured before
    /// this existed — 23 calls, 0 responses, and all 23 acts `reflect`.
    #[test]
    fn an_answer_leads_the_turn_and_the_narration_follows_it() {
        let s = compose_narrated(
            &["You moved to the sorting room.".to_string()],
            "The air near the door is noticeably fresher than by the wall.",
        );
        assert_eq!(
            s,
            "<tool_response>\nYou moved to the sorting room.\n</tool_response>\n\n\
             The air near the door is noticeably fresher than by the wall."
        );
    }

    /// One block per call, in the order the character called them — so a
    /// character reading them back can tell which answer belongs to which act.
    #[test]
    fn every_call_gets_its_own_block_in_call_order() {
        let s = compose_narrated(
            &[
                "You moved to the sorting room.".to_string(),
                "Nobody there answered to that name.".to_string(),
            ],
            "",
        );
        assert_eq!(s.matches("<tool_response>").count(), 2);
        assert_eq!(s.matches("</tool_response>").count(), 2);
        assert!(
            s.find("You moved").unwrap() < s.find("Nobody there").unwrap(),
            "answers must keep the order the acts were called in: {s}"
        );
    }

    /// A refusal is an answer too — the only signal that separates "that did
    /// not work" from "nothing happened", and the two want different next acts.
    #[test]
    fn a_refusal_is_returned_like_any_other_answer() {
        let s = compose_narrated(&["You cannot reach the vault from here.".to_string()], "");
        assert!(s.starts_with("<tool_response>\n"));
        assert!(s.contains("cannot reach the vault"));
    }

    /// With no acts behind it the turn is the narration alone — no empty wrapper
    /// for the model to read as a result that never came.
    #[test]
    fn a_turn_with_no_acts_behind_it_carries_no_wrapper() {
        let s = compose_narrated(&[], "Rain starts against the panes.");
        assert_eq!(s, "Rain starts against the panes.");
        assert!(!s.contains("tool_response"));
    }

    /// An answer with nothing to narrate still stands on its own — a character
    /// that acted during a quiet moment is told what happened.
    #[test]
    fn an_answer_alone_does_not_trail_a_blank_line() {
        let s = compose_narrated(&["You put it down.".to_string()], "");
        assert_eq!(s, "<tool_response>\nYou put it down.\n</tool_response>\n");
    }

    /// A quiet tick with nothing to narrate and nothing owed composes to
    /// nothing.
    #[test]
    fn an_empty_turn_composes_to_nothing() {
        assert_eq!(compose_narrated(&[], ""), "");
    }

    /// **No answer is handed over past its bound**, however it was made: one
    /// within it is untouched, one past it is cut at a character and says how
    /// much more there was — counted in characters, not bytes.
    #[test]
    fn an_answer_is_bounded_however_it_was_made() {
        let within = "é".repeat(ANSWER_MAX_CHARS);
        assert_eq!(bounded(&within), within);
        let past = format!("{}{}", "é".repeat(ANSWER_MAX_CHARS), "x".repeat(7));
        assert_eq!(
            bounded(&past),
            format!(
                "{}\n[cut: 7 more characters of this answer are not shown]",
                "é".repeat(ANSWER_MAX_CHARS)
            )
        );
        let turn = compose_narrated(from_ref(&past), "");
        assert!(turn.ends_with(
            "[cut: 7 more characters of this answer are not shown]\n</tool_response>\n"
        ));
        // The narration — where the compass rides — is bounded the same way,
        // and one within the bound is left exactly as it was.
        let narrated = compose_narrated(&[], &past);
        assert!(narrated.ends_with("[cut: 7 more characters of this answer are not shown]"));
        assert_eq!(compose_narrated(&[], "  Rain.  "), "  Rain.  ");
    }

    #[test]
    fn an_act_renders_distinguishably_from_perception() {
        assert_eq!(render_turn(Speaker::World, "it rains"), "it rains");
        assert_eq!(render_turn(Speaker::Npc, "waits"), "→ waits");
    }
}
