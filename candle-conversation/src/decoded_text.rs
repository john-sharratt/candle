//! A decoded assistant half as a [`TurnText`]: what the model wrote as tags,
//! told apart from what it wrote as text.
//!
//! Decoding a turn to one string erases the distinction every reader of its
//! markup depends on. `<tool_call>` the control token and `<tool_call>` spelled
//! out in ordinary sub-word tokens decode to the same eleven characters, and
//! the second is how a model *quotes* the tag: a file read into a turn as
//! literal text (see [`crate::turn_text`]) shows the model the tag spelled out,
//! and its summary of that file writes it back the same way. A reader that
//! takes the string for markup sees a call open in the middle of a sentence.
//!
//! Measured on `code_reading` ingest: six file summaries quoted `<tool_call>`
//! from the file they described. Each was planned as a call cut off at its
//! opener and answered with a `call_cut_off` error, and the model — told its
//! summary was a broken call — went on to read unrelated files and summarise
//! those in its place.
//!
//! The decode therefore keeps the split the ids carry: every token of the
//! tokenizer's added vocabulary is markup, and every run of ordinary tokens is
//! literal. Encoding the result with [`crate::turn_text::encode_pieces`] gives
//! back the ids it was decoded from.

use std::collections::HashSet;

use tokenizers::Tokenizer;

use crate::turn_text::TurnText;

/// The ids of a tokenizer's added vocabulary — every tag it can emit as a
/// single control token. Built once per engine.
#[derive(Debug, Clone, Default)]
pub struct TagIds(HashSet<u32>);

impl TagIds {
    pub fn of(tokenizer: &Tokenizer) -> Self {
        Self(tokenizer.get_added_tokens_decoder().into_keys().collect())
    }

    fn contains(&self, id: u32) -> bool {
        self.0.contains(&id)
    }
}

/// `written` followed by `ids` decoded: `written` and every tag token as
/// markup, every run of ordinary tokens as literal. Its
/// [`text`](TurnText::text) is `written` followed by `ids` decoded whole.
///
/// Each piece is cut from the decode of every id up to its end, never decoded
/// on its own: a decoder may treat the first token of a sequence differently
/// (a SentencePiece leading space), so a run decoded alone need not be the
/// text it contributes to the whole.
pub fn decode_turn_text(
    tokenizer: &Tokenizer,
    tags: &TagIds,
    written: &str,
    ids: &[u32],
    skip_special: bool,
) -> TurnText {
    let mut out = TurnText::markup(written);
    let mut decoded = String::new();
    let mut start = 0;
    while start < ids.len() {
        let tag = tags.contains(ids[start]);
        let end = if tag {
            start + 1
        } else {
            ids[start..]
                .iter()
                .position(|&id| tags.contains(id))
                .map_or(ids.len(), |n| start + n)
        };
        let through = tokenizer
            .decode(&ids[..end], skip_special)
            .unwrap_or_default();
        let piece = through.strip_prefix(decoded.as_str()).unwrap_or_else(|| {
            panic!(
                "decoding ids[..{end}] does not extend the decode of ids[..{start}] \
                 ({decoded:?} → {through:?}) — the tokenizer's decoder rewrites text \
                 across a tag"
            )
        });
        out = if tag {
            out.then_markup(piece)
        } else {
            out.then_literal(piece)
        };
        decoded = through;
        start = end;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::turn_text::{encode_pieces, literal_tokenizer, TextPiece};

    /// One token per character, a decoder that joins them back without
    /// spaces, and three tags: `<tool_call>` and `</tool_call>` registered
    /// plainly, as a chat vocabulary registers them, and `<|im_end|>` marked
    /// special.
    const FIXTURE: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        {"id": 18, "content": "<tool_call>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
        {"id": 19, "content": "</tool_call>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
        {"id": 20, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ],
      "normalizer": null,
      "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "."}, "behavior": "Isolated", "invert": false},
      "post_processor": null,
      "decoder": {"type": "Fuse"},
      "model": {
        "type": "WordLevel",
        "vocab": {"[UNK]": 0, "<": 1, ">": 2, "/": 3, "t": 4, "o": 5, "l": 6, "_": 7,
                  "c": 8, "a": 9, "x": 10, "|": 11, "i": 12, "m": 13, "e": 14, "n": 15,
                  "d": 16, " ": 17},
        "unk_token": "[UNK]"
      }
    }"#;

    /// `<tool_call>` spelled out, one ordinary token per character.
    const SPELLED_OPEN: [u32; 11] = [1, 4, 5, 5, 6, 7, 8, 9, 6, 6, 2];

    fn fixture() -> Tokenizer {
        FIXTURE.parse().expect("fixture tokenizer")
    }

    fn piece(text: &str, literal: bool) -> TextPiece {
        TextPiece {
            text: text.to_string(),
            literal,
        }
    }

    fn decoded(written: &str, ids: &[u32], skip_special: bool) -> TurnText {
        let t = fixture();
        decode_turn_text(&t, &TagIds::of(&t), written, ids, skip_special)
    }

    /// **The live failure.** A summary quoting the tag decodes to the tag's
    /// characters, and they stay literal: nothing in the text opened a call.
    #[test]
    fn a_tag_spelled_out_is_literal() {
        let mut ids = vec![10, 17];
        ids.extend(SPELLED_OPEN);
        ids.extend([17, 10]);
        let text = decoded("", &ids, true);
        assert_eq!(text.pieces(), [piece("x <tool_call> x", true)]);
    }

    #[test]
    fn a_tag_token_is_markup_between_literal_runs() {
        let text = decoded("", &[10, 18, 10, 10, 19, 10], true);
        assert_eq!(
            text.pieces(),
            [
                piece("x", true),
                piece("<tool_call>", false),
                piece("xx", true),
                piece("</tool_call>", false),
                piece("x", true),
            ]
        );
    }

    /// The tag and its quotation side by side: one markup piece, the other
    /// literal, though the text reads `<tool_call>` twice.
    #[test]
    fn the_token_and_its_quotation_decode_apart() {
        let mut ids = vec![18];
        ids.extend(SPELLED_OPEN);
        let text = decoded("", &ids, true);
        assert_eq!(text.text(), "<tool_call><tool_call>");
        assert_eq!(
            text.pieces(),
            [piece("<tool_call>", false), piece("<tool_call>", true)]
        );
    }

    /// The written half leads, as markup — it is what the caller wrote.
    #[test]
    fn the_written_half_comes_first_as_markup() {
        let text = decoded("<tool_call>x", &[10, 19], true);
        assert_eq!(
            text.pieces(),
            [
                piece("<tool_call>x", false),
                piece("x", true),
                piece("</tool_call>", false),
            ]
        );
    }

    #[test]
    fn a_skipped_special_token_leaves_no_piece() {
        assert_eq!(decoded("", &[10, 20], true).pieces(), [piece("x", true)]);
        assert_eq!(
            decoded("", &[10, 20], false).pieces(),
            [piece("x", true), piece("<|im_end|>", false)]
        );
    }

    /// Encoding the decode gives back the ids — the pieces are exactly the
    /// split the ids carried.
    #[test]
    fn encoding_the_decode_gives_back_the_ids() {
        let mut ids = vec![10, 18, 10];
        ids.extend(SPELLED_OPEN);
        ids.extend([19, 17]);
        ids.extend(SPELLED_OPEN);
        let markup = fixture();
        let literal = literal_tokenizer(&markup);
        let text = decoded("", &ids, false);
        assert_eq!(
            Vec::from(encode_pieces(&markup, &literal, &text).unwrap()),
            ids
        );
    }

    #[test]
    fn no_ids_is_the_written_half_alone() {
        assert_eq!(decoded("ab", &[], true).pieces(), [piece("ab", false)]);
        assert_eq!(decoded("", &[], true), TurnText::default());
    }
}
