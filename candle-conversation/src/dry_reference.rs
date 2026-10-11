//! Which tokens make up a reference — a number, an id, a path — so the DRY
//! penalty lets one be written again.
//!
//! DRY prices out the token that would continue a run the value has already
//! written. That is right for a sentence said twice and wrong for a reference:
//! an id handed from one act to the next (`npc-4025671320141624214`), a path
//! (`layers/stories/the-unmapped-feed.md`), a year or a count is only correct
//! when it repeats exactly, and pushed off its second spelling it becomes a
//! different id, a path that does not exist, the wrong year.
//!
//! The kernel reads one byte per token: [`REFERENCE`] when the token's own text
//! marks it as part of a reference, and [`JOINS`] when it carries on the word
//! before it (it does not open with whitespace). A token sits in a reference
//! when any token of the word around it is a reference token, and DRY leaves
//! such a continuation alone.

use tokenizers::Tokenizer;

/// The token's text marks a reference: a digit, a path or id separator, or a
/// `-`, `.` or `:` running on into a letter or digit (`-un`, `.md`, `:42`).
pub const REFERENCE: u8 = 1;

/// The token carries on the word before it: its text does not open with
/// whitespace.
pub const JOINS: u8 = 2;

/// Separators that only ever appear inside a reference.
const ALWAYS: &[char] = &['/', '\\', '_', '#', '@', '=', '~', '|'];

/// Separators that mark a reference when they run on into a letter or digit,
/// and are ordinary punctuation otherwise — a full stop, a dash, a colon.
const BEFORE_ALNUM: &[char] = &['-', '.', ':'];

/// The flags for one token's text.
pub fn flags_of(text: &str) -> u8 {
    let mut flags = 0;
    if text.chars().next().is_some_and(|c| !c.is_whitespace()) {
        flags |= JOINS;
    }
    let chars: Vec<char> = text.chars().collect();
    let reference = chars.iter().enumerate().any(|(i, c)| {
        c.is_ascii_digit()
            || ALWAYS.contains(c)
            || (BEFORE_ALNUM.contains(c) && chars.get(i + 1).is_some_and(|n| n.is_alphanumeric()))
    });
    if reference {
        flags |= REFERENCE;
    }
    flags
}

/// The flags for every id below `vocab_size`, as the kernel indexes them. A
/// tokenizer's added vocabulary — its control tokens — is neither: a tag is
/// structure, not part of a word.
pub fn reference_flags(tokenizer: &Tokenizer, vocab_size: usize) -> Vec<u8> {
    let tags = tokenizer.get_added_tokens_decoder();
    (0..vocab_size as u32)
        .map(|id| {
            if tags.contains_key(&id) {
                return 0;
            }
            // The raw piece keeps the leading-space marker a byte-level or
            // SentencePiece vocabulary writes (`Ġ`, `▁`), which a decode of
            // the token alone may drop.
            let opens_with_space = tokenizer
                .id_to_token(id)
                .is_some_and(|p| p.starts_with(['Ġ', '▁', 'Ċ', 'ĉ']));
            let text = tokenizer.decode(&[id], false).unwrap_or_default();
            match opens_with_space {
                true => flags_of(&text) & !JOINS,
                false => flags_of(&text),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_word_after_a_space_is_neither() {
        assert_eq!(flags_of(" catalogue"), 0);
        assert_eq!(flags_of(" I"), 0);
    }

    #[test]
    fn a_word_piece_joins_and_is_no_reference() {
        assert_eq!(flags_of("mapped"), JOINS);
    }

    #[test]
    fn a_digit_is_a_reference() {
        assert_eq!(flags_of("402"), REFERENCE | JOINS);
        assert_eq!(flags_of(" 2937"), REFERENCE);
    }

    #[test]
    fn a_path_or_id_separator_is_a_reference() {
        assert_eq!(flags_of("/"), REFERENCE | JOINS);
        assert_eq!(flags_of("_"), REFERENCE | JOINS);
    }

    /// A full stop, a dash or a colon ends a sentence or a clause as often as
    /// it sits inside a reference, so only one running on into a letter or
    /// digit counts.
    #[test]
    fn a_full_stop_ending_a_sentence_is_no_reference() {
        assert_eq!(flags_of("."), JOINS);
        assert_eq!(flags_of(":"), JOINS);
        assert_eq!(flags_of(" -"), 0);
        assert_eq!(flags_of(".md"), REFERENCE | JOINS);
        assert_eq!(flags_of("-un"), REFERENCE | JOINS);
    }

    #[test]
    fn nothing_is_neither() {
        assert_eq!(flags_of(""), 0);
    }

    /// A word piece joins, a number is a reference, a piece written with the
    /// vocabulary's leading-space marker opens a word, a control tag is
    /// neither, and the padding past the vocabulary is neither.
    #[test]
    fn the_table_covers_every_id_up_to_the_row_width() {
        let tokenizer: Tokenizer = r#"{
          "version": "1.0", "truncation": null, "padding": null,
          "added_tokens": [
            {"id": 3, "content": "<t>", "single_word": false, "lstrip": false,
             "rstrip": false, "normalized": false, "special": true}
          ],
          "normalizer": null, "pre_tokenizer": null, "post_processor": null,
          "decoder": null,
          "model": {"type": "WordLevel",
                    "vocab": {"[UNK]": 0, "word": 1, "42": 2, "<t>": 3, "Ġthe": 4},
                    "unk_token": "[UNK]"}
        }"#
        .parse()
        .expect("fixture tokenizer");
        assert_eq!(
            reference_flags(&tokenizer, 6),
            vec![JOINS, JOINS, REFERENCE | JOINS, 0, 0, 0]
        );
    }
}
