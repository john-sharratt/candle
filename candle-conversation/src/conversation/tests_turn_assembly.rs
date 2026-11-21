//! Tests how a probe's question and a conversation's framing glue are laid out,
//! for every dialect the engine can run.
//!
//! The probe's shape is written out as text, `hi [/INST] `, so the expected layout
//! can be read directly. A suppressed turn opens on its closed reasoning block with
//! the caller's seed after it, while a thinking turn or a switch family leads with
//! the seed alone. Glue markers are checked so a block-suppressing family carries
//! its block, a switch family carries its switch, and no dialect carries both.

use super::*;

#[cfg(test)]
pub(super) mod probe_turn_tests {
    use super::probe_turn_tokens;
    use candle_transformers::models::dialect::Dialect;
    use tokenizers::Tokenizer;

    /// One id a character: `h i ␠ [ / I N S T ]` are 1–10.
    const CHARS: &str = r#"{
      "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
      "normalizer": null,
      "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "."}, "behavior": "Isolated", "invert": false},
      "post_processor": null, "decoder": null,
      "model": {"type": "WordLevel",
        "vocab": {"[UNK]": 0, "h": 1, "i": 2, " ": 3, "[": 4, "/": 5, "I": 6, "N": 7, "S": 8, "T": 9, "]": 10},
        "unk_token": "[UNK]"}
    }"#;

    /// **A probe prefills the question as a thinking turn**: the user
    /// opener (empty on Llama 2), the question, `user_end` and the assistant
    /// header — `hi [/INST] ` — and nothing for a question that is empty.
    #[test]
    fn a_probe_prefills_its_question_as_a_thinking_turn() {
        let t: Tokenizer = CHARS.parse().unwrap();
        let d = Dialect::llama2();
        assert_eq!(
            probe_turn_tokens(&t, &d, "hi").unwrap(),
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 3]
        );
        assert!(probe_turn_tokens(&t, &d, "").unwrap().is_empty());
    }
}

#[cfg(test)]
pub(super) mod assistant_lead_tests {
    use super::assistant_lead;
    use candle_transformers::models::dialect::Dialect;

    /// A block-suppressing family's suppressed turn prefills its closed block,
    /// and a caller's seed continues from after it — the block is scaffolding
    /// ahead of the content, never after it.
    #[test]
    fn a_suppressed_qwen35_turn_opens_on_its_closed_block() {
        let closed = Dialect::qwen35().thinking_suppression(true).1;
        assert_eq!(assistant_lead(closed, ""), "<think>\n\n</think>\n\n");
        assert_eq!(
            assistant_lead(closed, "<tool_call>"),
            "<think>\n\n</think>\n\n<tool_call>"
        );
    }

    /// A thinking turn, or a family that suppresses with the soft switch in the
    /// user turn, adds nothing: the lead is the caller's seed alone.
    #[test]
    fn a_thinking_turn_or_a_switch_family_leads_with_the_seed_alone() {
        let on = Dialect::qwen35().thinking_suppression(false).1;
        assert_eq!(assistant_lead(on, "<tool_call>"), "<tool_call>");
        let switch = Dialect::chat_ml().thinking_suppression(true).1;
        assert_eq!(assistant_lead(switch, "<tool_call>"), "<tool_call>");
    }
}

#[cfg(test)]
pub(super) mod glue_marker_tests {
    use super::GlueMarkers;
    use candle_transformers::models::dialect::Dialect;

    /// **The panel must show whichever half of suppression the family uses.**
    ///
    /// Qwen3.5/3.8 has no `/no_think` soft switch — it suppresses by opening the
    /// assistant turn with the block already closed, prefilled into the grid. The
    /// block half was once omitted here on the reasoning that a suppressed turn
    /// "decodes its own empty block into the body", which stopped being true when
    /// suppression became structural: the panel then rendered strictly less than
    /// the turn contained, on exactly the family that depends on it.
    #[test]
    fn a_block_suppressing_dialect_carries_its_block() {
        let g = GlueMarkers::from_dialect(&Dialect::qwen35());
        assert_eq!(g.no_think_block, "<think>\n\n</think>\n\n");
        assert!(
            g.no_think.is_empty(),
            "qwen35 has no soft switch, so the switch half must be empty"
        );
    }

    /// The mirror: a family WITH the soft switch carries that and no block, so
    /// the panel never shows a prefilled block the grid does not contain.
    #[test]
    fn a_switch_suppressing_dialect_carries_its_switch() {
        let g = GlueMarkers::from_dialect(&Dialect::chat_ml());
        assert_eq!(g.no_think, "/no_think\n");
        assert!(
            g.no_think_block.is_empty(),
            "ChatML suppresses with the switch, so the block half must be empty"
        );
    }

    /// **Exactly one half is live, for every dialect the engine can run.**
    ///
    /// This is the invariant `Dialect::thinking_suppression` exists to enforce,
    /// and reading it through that function rather than off the fields is what
    /// keeps it true here. A dialect with neither (Llama-family, no reasoning
    /// markers at all) is a legitimate third case: nothing to suppress.
    #[test]
    fn never_both_halves_at_once() {
        for d in [
            Dialect::chat_ml(),
            Dialect::qwen35(),
            Dialect::llama2(),
            Dialect::llama3(),
        ] {
            let g = GlueMarkers::from_dialect(&d);
            assert!(
                g.no_think.is_empty() || g.no_think_block.is_empty(),
                "{:?}: both suppression halves are non-empty — the dialect must \
                 pick one mechanism",
                d.dialect_type()
            );
        }
    }

    /// The role markers still come straight off the dialect — the refactor to a
    /// single constructor must not have quietly changed what the panel frames
    /// turns with.
    #[test]
    fn role_markers_are_passed_through_verbatim() {
        let d = Dialect::qwen35();
        let g = GlueMarkers::from_dialect(&d);
        assert_eq!(g.system_start, d.system_start);
        assert_eq!(g.system_end, d.system_end);
        assert_eq!(g.user_start, d.user_start);
        assert_eq!(g.user_end, d.user_end);
        assert_eq!(g.assistant_start, d.assistant_start);
        assert_eq!(g.assistant_end, d.assistant_end);
    }
}
