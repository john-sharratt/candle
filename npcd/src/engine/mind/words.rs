//! What a turn of plain words may not decode: a conversation's words, a
//! channel post's voicing.
//!
//! **No reasoning, no call, and no call written out as text.** `<think>` and
//! the dialect's call marker are taken out of the vocabulary — and so is every
//! token that opens with `{`. A character's history is its own turns, and each
//! of those is a call: `{"name": …, "arguments": …}`. With the marker banned and
//! the acts hidden, Zen still answered a person with that object, the marker
//! left off (`{"name": "reply", "arguments": {"text": "I am alert and
//! steady…"}}`). Words to somebody have no use for a brace; without one, a call
//! cannot be begun.

use std::sync::OnceLock;

use super::Minds;

/// The markers a turn of words may not decode.
const MARKERS: [&str; 2] = ["<think>", "<tool_call>"];

/// The ids of the tokens among `vocab` — `(id, text)` — whose text opens with
/// a brace once leading whitespace is passed.
fn opening_a_brace(vocab: impl IntoIterator<Item = (u32, String)>) -> Vec<i32> {
    let mut ids: Vec<i32> = vocab
        .into_iter()
        .filter(|(_, text)| text.trim_start().starts_with('{'))
        .map(|(id, _)| id as i32)
        .collect();
    ids.sort_unstable();
    ids
}

impl Minds {
    /// The tokens a turn of words may not decode: the markers, and every token
    /// that opens a brace. Worked out once from the model's vocabulary.
    pub(super) fn banned_in_words(&self) -> Vec<i32> {
        static BANNED: OnceLock<Vec<i32>> = OnceLock::new();
        BANNED
            .get_or_init(|| {
                let tok = self.engine.tokenizer();
                let mut ids: Vec<i32> = MARKERS
                    .iter()
                    .filter_map(|m| tok.token_to_id(m))
                    .map(|id| id as i32)
                    .collect();
                ids.extend(opening_a_brace(
                    tok.get_vocab(true)
                        .into_values()
                        .map(|id| (id, tok.decode(&[id], false).unwrap_or_default())),
                ));
                ids
            })
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::opening_a_brace;

    /// **Every token that opens a brace is banned, with or without the space
    /// before it**, and no other: a brace inside a token is not an opening.
    #[test]
    fn the_tokens_that_open_a_brace() {
        let vocab = [
            (1, "{".to_string()),
            (2, " {".to_string()),
            (3, "{\"".to_string()),
            (4, "\n{".to_string()),
            (5, "hello".to_string()),
            (6, "a{".to_string()),
            (7, "}".to_string()),
        ];
        assert_eq!(opening_a_brace(vocab), vec![1, 2, 3, 4]);
    }
}
