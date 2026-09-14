//! A user half as the engine takes it: pieces of text, each **markup** or
//! **literal**.
//!
//! Markup is tokenized the way the chat template expects — a chat tag written in
//! it (`<tool_response>`, `<|im_end|>`) is that control token. Literal text can
//! never become one: it is content the model reads, such as a file a tool
//! returned. A file is free to contain `<think>` or `<|im_end|>`, and read into
//! a turn as markup it closes a reasoning block or ends the turn in the model's
//! context. Measured on a GUI turn that read `zend/src/think_gate.rs`: having
//! seen the file's `<think></think>` as the tokens themselves, the model wrote
//! them back as tokens while quoting the file, closed its own reasoning
//! mid-thought, and reasoned on into its answer.
//!
//! The pieces are encoded one by one and their ids concatenated. A tokenizer
//! splits its input at every registered tag before any sub-word merging, so
//! where the pieces meet on a tag — the way callers build them, the wrapper tag
//! on one side and the content on the other — the concatenation is exactly the
//! whole string's encoding.

use tokenizers::{AddedToken, Tokenizer};

use crate::token_buffer::TokenBuffer;

/// One run of a [`TurnText`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextPiece {
    pub text: String,
    /// Encoded so that no part of it can become a control token.
    pub literal: bool,
}

/// A user half, as an ordered list of [`TextPiece`]s.
///
/// Adjacent pieces of one kind are held as one, and empty ones are dropped, so
/// two `TurnText`s are equal exactly when they would encode alike. Plain text
/// converts into a single markup piece, which is how every half that carries
/// no outside content is written.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TurnText {
    pieces: Vec<TextPiece>,
}

impl TurnText {
    /// A half of markup alone.
    pub fn markup(text: impl Into<String>) -> Self {
        Self::default().then_markup(text)
    }

    /// A half of literal text alone.
    pub fn literal(text: impl Into<String>) -> Self {
        Self::default().then_literal(text)
    }

    /// This half followed by `text` as markup.
    #[must_use]
    pub fn then_markup(mut self, text: impl Into<String>) -> Self {
        self.push(text.into(), false);
        self
    }

    /// This half followed by `text` as literal text.
    #[must_use]
    pub fn then_literal(mut self, text: impl Into<String>) -> Self {
        self.push(text.into(), true);
        self
    }

    /// `parts` in order, with `separator` between each as markup.
    pub fn join(parts: impl IntoIterator<Item = TurnText>, separator: &str) -> Self {
        let mut out = Self::default();
        for (i, part) in parts.into_iter().enumerate() {
            if i > 0 {
                out.push(separator.to_string(), false);
            }
            for piece in part.pieces {
                out.push(piece.text, piece.literal);
            }
        }
        out
    }

    pub fn pieces(&self) -> &[TextPiece] {
        &self.pieces
    }

    /// The half's text — every piece, in order.
    pub fn text(&self) -> String {
        self.pieces.iter().map(|p| p.text.as_str()).collect()
    }

    /// Whether the half holds nothing but whitespace.
    pub fn is_blank(&self) -> bool {
        self.pieces.iter().all(|p| p.text.trim().is_empty())
    }

    fn push(&mut self, text: String, literal: bool) {
        if text.is_empty() {
            return;
        }
        match self.pieces.last_mut() {
            Some(last) if last.literal == literal => last.text.push_str(&text),
            _ => self.pieces.push(TextPiece { text, literal }),
        }
    }
}

impl From<&str> for TurnText {
    fn from(text: &str) -> Self {
        Self::markup(text)
    }
}

impl From<String> for TurnText {
    fn from(text: String) -> Self {
        Self::markup(text)
    }
}

impl From<&String> for TurnText {
    fn from(text: &String) -> Self {
        Self::markup(text.as_str())
    }
}

/// `base` with every registered tag read as plain text.
///
/// The tokenizer's own switch for this (`set_encode_special_tokens`) passes
/// over only the tags marked special, and a chat vocabulary marks few: Qwen3.6
/// marks `<|im_start|>` and `<|im_end|>` but not `<think>`, `</think>`,
/// `<tool_call>` or `<tool_response>`. Re-adding every registered tag as
/// special — same content, same id — brings them all under it. Built once, at
/// engine start: it is a copy of the whole vocabulary.
pub fn literal_tokenizer(base: &Tokenizer) -> Tokenizer {
    let mut literal = base.clone();
    let tags: Vec<AddedToken> = base
        .get_added_tokens_decoder()
        .values()
        .map(|tag| {
            let mut tag = tag.clone();
            tag.special = true;
            tag
        })
        .collect();
    literal.add_special_tokens(&tags);
    literal.set_encode_special_tokens(true);
    literal
}

/// Encode `text` piece by piece: markup through `markup`, literal pieces
/// through `literal` (a [`literal_tokenizer`] of it), ids concatenated.
pub fn encode_pieces(
    markup: &Tokenizer,
    literal: &Tokenizer,
    text: &TurnText,
) -> tokenizers::Result<TokenBuffer> {
    let mut ids: Vec<u32> = Vec::new();
    for piece in &text.pieces {
        let tokenizer = if piece.literal { literal } else { markup };
        ids.extend_from_slice(tokenizer.encode(piece.text.as_str(), false)?.get_ids());
    }
    Ok(TokenBuffer::from(ids))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One token per character, plus five tags: four registered plainly, as a
    /// chat vocabulary registers `<think>` and `<tool_response>`, and one marked
    /// special, as it marks `<|im_end|>`. The tags take the ids after the
    /// vocabulary's 21, in order — the tokenizer assigns them so on load.
    const FIXTURE: &str = r#"{
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        {"id": 21, "content": "<think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
        {"id": 22, "content": "</think>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
        {"id": 23, "content": "<tool_response>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
        {"id": 24, "content": "</tool_response>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
        {"id": 25, "content": "<|im_end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
      ],
      "normalizer": null,
      "pre_tokenizer": {"type": "Split", "pattern": {"Regex": "."}, "behavior": "Isolated", "invert": false},
      "post_processor": null,
      "decoder": null,
      "model": {
        "type": "WordLevel",
        "vocab": {"[UNK]": 0, "<": 1, ">": 2, "/": 3, "|": 4, "_": 5, "t": 6, "h": 7, "i": 8,
                  "n": 9, "k": 10, "o": 11, "l": 12, "r": 13, "e": 14, "s": 15, "p": 16,
                  "m": 17, "d": 18, "x": 19, " ": 20},
        "unk_token": "[UNK]"
      }
    }"#;

    const THINK: [u32; 7] = [1, 6, 7, 8, 9, 10, 2];
    const THINK_CLOSE: [u32; 8] = [1, 3, 6, 7, 8, 9, 10, 2];
    const IM_END: [u32; 10] = [1, 4, 8, 17, 5, 14, 9, 18, 4, 2];

    fn fixture() -> Tokenizer {
        FIXTURE.parse().expect("fixture tokenizer")
    }

    fn ids(tokenizer: &Tokenizer, text: &str) -> Vec<u32> {
        tokenizer.encode(text, false).unwrap().get_ids().to_vec()
    }

    fn encoded(text: &TurnText) -> Vec<u32> {
        let markup = fixture();
        let literal = literal_tokenizer(&markup);
        Vec::from(encode_pieces(&markup, &literal, text).unwrap())
    }

    #[test]
    fn markup_carries_every_tag_as_its_token() {
        let t = fixture();
        assert_eq!(ids(&t, "x<think>x"), [19, 21, 19]);
        assert_eq!(ids(&t, "x<|im_end|>x"), [19, 25, 19]);
    }

    /// Plain and special tags alike: the tokenizer's own switch alone would
    /// have spelled out `<|im_end|>` and left `<think>` a token.
    #[test]
    fn literal_text_spells_out_every_tag() {
        let lit = literal_tokenizer(&fixture());
        assert_eq!(ids(&lit, "<think>"), THINK);
        assert_eq!(ids(&lit, "</think>"), THINK_CLOSE);
        assert_eq!(ids(&lit, "<|im_end|>"), IM_END);
    }

    #[test]
    fn a_tool_response_keeps_its_wrapper_and_spells_out_its_content() {
        let text = TurnText::markup("<tool_response>")
            .then_literal("x<think></think><|im_end|>x")
            .then_markup("</tool_response>");
        let mut expected = vec![23, 19];
        expected.extend(THINK);
        expected.extend(THINK_CLOSE);
        expected.extend(IM_END);
        expected.extend([19, 24]);
        assert_eq!(encoded(&text), expected);
    }

    /// Where the pieces meet on a tag, their concatenation is the whole
    /// string's encoding — the literal flag changes nothing for content that
    /// holds no tag.
    #[test]
    fn pieces_meeting_on_tags_encode_as_the_whole_string() {
        let text = TurnText::markup("<tool_response>")
            .then_literal("x x")
            .then_markup("</tool_response> x");
        assert_eq!(
            encoded(&text),
            ids(&fixture(), "<tool_response>x x</tool_response> x")
        );
    }

    #[test]
    fn plain_text_is_one_markup_piece() {
        assert_eq!(
            TurnText::from("hi").pieces(),
            [TextPiece {
                text: "hi".to_string(),
                literal: false,
            }]
        );
        assert_eq!(TurnText::from("x<think>").text(), "x<think>");
        assert_eq!(encoded(&TurnText::from("x<think>")), [19, 21]);
    }

    #[test]
    fn adjacent_pieces_of_a_kind_merge_and_empty_ones_drop() {
        let text = TurnText::markup("a")
            .then_markup("b")
            .then_literal("")
            .then_literal("c")
            .then_literal("d");
        assert_eq!(
            text.pieces(),
            [
                TextPiece {
                    text: "ab".to_string(),
                    literal: false,
                },
                TextPiece {
                    text: "cd".to_string(),
                    literal: true,
                },
            ]
        );
        assert_eq!(text.text(), "abcd");
    }

    #[test]
    fn joining_puts_the_separator_between_parts_as_markup() {
        let text = TurnText::join(
            [
                TurnText::literal("a"),
                TurnText::markup("b"),
                TurnText::literal("c"),
            ],
            "\n\n",
        );
        assert_eq!(text.text(), "a\n\nb\n\nc");
        let kinds: Vec<bool> = text.pieces().iter().map(|p| p.literal).collect();
        assert_eq!(kinds, [true, false, true]);
        assert_eq!(TurnText::join([], "\n"), TurnText::default());
    }

    #[test]
    fn blank_is_whitespace_in_every_piece() {
        assert!(TurnText::default().is_blank());
        assert!(TurnText::markup(" \n").then_literal("\t").is_blank());
        assert!(!TurnText::markup(" ").then_literal("x").is_blank());
    }
}
