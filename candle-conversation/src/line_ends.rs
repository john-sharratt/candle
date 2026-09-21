//! The tokens that end a line — the boundary a thinking block closes on.
//!
//! See [`SamplingConfig::line_end_token_ids`](crate::SamplingConfig) for why a
//! line, not a sentence: `.` also sits inside numbers, addresses and paths.

use std::sync::Arc;

use tokenizers::Tokenizer;

/// Every token id whose decoded text ends in `\n`, ascending.
///
/// Scans the whole vocabulary, special tokens included, decoding each id on
/// its own. Probing a fixed list of spellings (`"\n"`, `".\n"`) misses a
/// byte-level vocabulary, which stores a newline as `Ċ` and has hundreds of
/// merged forms (`")\n\n"`, `"```\n"`) that end a line just as well.
pub fn line_end_token_ids(tokenizer: &Tokenizer) -> Arc<[i32]> {
    let size = tokenizer.get_vocab_size(true) as u32;
    (0..size)
        .filter(|&id| {
            tokenizer
                .decode(&[id], false)
                .is_ok_and(|text| text.ends_with('\n'))
        })
        .map(|id| id as i32)
        .collect()
}

/// Whether `token` is one of `line_ends` (sorted ascending).
pub fn ends_a_line(line_ends: &[i32], token: i32) -> bool {
    line_ends.binary_search(&token).is_ok()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use tokenizers::models::wordlevel::WordLevel;

    use super::*;

    fn tokenizer(words: &[&str]) -> Tokenizer {
        let vocab: HashMap<String, u32> = words
            .iter()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i as u32))
            .collect();
        let model = WordLevel::builder()
            .vocab(vocab.into_iter().collect())
            .unk_token("[UNK]".to_string())
            .build()
            .unwrap();
        Tokenizer::new(model)
    }

    /// **A `.` inside a number is not a line end; a newline is, in any
    /// merged form.**
    #[test]
    fn the_line_ends_are_the_tokens_that_end_in_a_newline() {
        let t = tokenizer(&["[UNK]", "169.", "254", "\n", ".\n", ")\n\n", "end.", "x"]);
        let ends = line_end_token_ids(&t);
        assert_eq!(&*ends, &[3, 4, 5]);
        assert!(ends_a_line(&ends, 4));
        assert!(!ends_a_line(&ends, 1), "`169.` does not end a line");
        assert!(!ends_a_line(&ends, 6), "a sentence end is not a line end");
        assert!(!ends_a_line(&[], 3));
    }
}
